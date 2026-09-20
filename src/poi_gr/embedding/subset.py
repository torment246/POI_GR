"""Derive an embedding artifact for an ordered POI-catalog subset."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class EmbeddingSubsetError(ValueError):
    """Raised when an embedding subset cannot be proven row-aligned."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise EmbeddingSubsetError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EmbeddingSubsetError(f"{name} 不是合法 JSON：{path}") from error
    if not isinstance(value, dict):
        raise EmbeddingSubsetError(f"{name} 必须是 JSON object：{path}")
    return value


def _iter_ids(path: Path, name: str) -> Iterator[tuple[bytes, str, int]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n"):
                raise EmbeddingSubsetError(
                    f"{name} 第 {line_number} 行缺少换行符，无法逐字节保留 ID 清单"
                )
            try:
                value = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise EmbeddingSubsetError(
                    f"{name} 第 {line_number} 行不是合法 JSON"
                ) from error
            if not isinstance(value, str) or not value.strip():
                raise EmbeddingSubsetError(f"{name} 第 {line_number} 行 POI ID 无效")
            yield raw_line, value, line_number


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def build_embedding_subset(
    *,
    source_dir: Path,
    target_poi_ids: Path,
    target_manifest: Path,
    output_dir: Path,
    expected_rows: int,
    project_root: Path,
    copy_chunk_rows: int = 8192,
) -> dict[str, Any]:
    """Copy exact source vectors for an ordered ID subsequence into a new artifact."""

    if expected_rows <= 0 or copy_chunk_rows <= 0:
        raise EmbeddingSubsetError("expected_rows 和 copy_chunk_rows 必须大于 0")
    source_dir = source_dir.resolve()
    target_poi_ids = target_poi_ids.resolve()
    target_manifest = target_manifest.resolve()
    output_dir = output_dir.resolve()
    source_manifest_path = source_dir / "manifest.json"
    source_ids_path = source_dir / "poi_ids.jsonl"
    source_embeddings_path = source_dir / "embeddings.npy"
    for path, name in (
        (source_ids_path, "源 POI ID 清单"),
        (source_embeddings_path, "源 Embedding"),
        (target_poi_ids, "目标 POI ID 清单"),
    ):
        if not path.is_file():
            raise EmbeddingSubsetError(f"{name} 不存在：{path}")
    source_manifest = _load_json(source_manifest_path, "源 Embedding manifest")
    catalog_manifest = _load_json(target_manifest, "目标 POI catalog manifest")
    if source_manifest.get("status") != "completed":
        raise EmbeddingSubsetError("源 Embedding manifest 状态不是 completed")
    if catalog_manifest.get("status") != "completed":
        raise EmbeddingSubsetError("目标 POI catalog manifest 状态不是 completed")
    if output_dir.exists():
        raise EmbeddingSubsetError(f"输出目录已存在，拒绝覆盖：{output_dir}")

    source_embeddings = np.load(
        source_embeddings_path, mmap_mode="r", allow_pickle=False
    )
    if source_embeddings.ndim != 2:
        raise EmbeddingSubsetError("源 Embedding 必须是二维数组")
    declared_shape = tuple(source_manifest.get("output", {}).get("shape", ()))
    declared_dtype = source_manifest.get("output", {}).get("dtype")
    if tuple(source_embeddings.shape) != declared_shape:
        raise EmbeddingSubsetError("源 Embedding shape 与 manifest 不一致")
    if str(source_embeddings.dtype) != declared_dtype:
        raise EmbeddingSubsetError("源 Embedding dtype 与 manifest 不一致")
    if int(source_manifest.get("input", {}).get("total_rows", -1)) != len(
        source_embeddings
    ):
        raise EmbeddingSubsetError("源 manifest 行数与 Embedding 不一致")

    started = time.perf_counter()
    build_dir = output_dir.with_name(f".{output_dir.name}.building-{os.getpid()}")
    if build_dir.exists():
        raise EmbeddingSubsetError(f"临时目录已存在：{build_dir}")
    build_dir.mkdir(parents=True)
    output_embeddings_path = build_dir / "embeddings.npy"
    output_ids_path = build_dir / "poi_ids.jsonl"
    output_embeddings = np.lib.format.open_memmap(
        output_embeddings_path,
        mode="w+",
        dtype=source_embeddings.dtype,
        shape=(expected_rows, int(source_embeddings.shape[1])),
    )
    selected_indices: list[int] = []
    written_rows = 0
    source_rows = 0
    target_iterator = _iter_ids(target_poi_ids, "目标 POI ID 清单")
    current_target = next(target_iterator, None)
    if current_target is None:
        raise EmbeddingSubsetError("目标 POI ID 清单不能为空")

    def flush_vectors() -> None:
        nonlocal written_rows
        if not selected_indices:
            return
        stop = written_rows + len(selected_indices)
        if stop > expected_rows:
            raise EmbeddingSubsetError("目标 POI ID 行数超过 expected_rows")
        indices = np.asarray(selected_indices, dtype=np.int64)
        # ``source_embeddings[indices]`` issues many tiny random reads against an
        # OFS-backed memmap.  The selected active rows preserve source order, so
        # read their enclosing source range once and select locally in memory.
        # This changes only the I/O pattern; copied float16 values remain exact.
        source_start = int(indices[0])
        source_stop = int(indices[-1]) + 1
        source_block = np.array(
            source_embeddings[source_start:source_stop], copy=True
        )
        output_embeddings[written_rows:stop] = source_block[
            indices - source_start
        ]
        written_rows = stop
        selected_indices.clear()

    try:
        with output_ids_path.open("wb") as output_ids:
            for source_index, (_, source_id, _) in enumerate(
                _iter_ids(source_ids_path, "源 POI ID 清单")
            ):
                source_rows += 1
                if current_target is None or source_id != current_target[1]:
                    continue
                output_ids.write(current_target[0])
                selected_indices.append(source_index)
                current_target = next(target_iterator, None)
                if len(selected_indices) >= copy_chunk_rows:
                    flush_vectors()
            flush_vectors()
            output_ids.flush()
            os.fsync(output_ids.fileno())
        output_embeddings.flush()
        del output_embeddings
        if source_rows != len(source_embeddings):
            raise EmbeddingSubsetError(
                f"源 POI ID 行数 {source_rows} 与 Embedding {len(source_embeddings)} 不一致"
            )
        if current_target is not None:
            raise EmbeddingSubsetError(
                "目标 POI ID 不是源清单的有序子序列，首个未匹配 ID："
                f"{current_target[1]}"
            )
        if written_rows != expected_rows:
            raise EmbeddingSubsetError(
                f"实际写入 {written_rows} 行，与 expected_rows {expected_rows} 不一致"
            )

        target_ids_sha256 = _sha256_file(target_poi_ids)
        output_ids_sha256 = _sha256_file(output_ids_path)
        if target_ids_sha256 != output_ids_sha256:
            raise EmbeddingSubsetError("输出 POI ID 清单未逐字节保留目标清单")
        output_vectors = np.load(
            output_embeddings_path, mmap_mode="r", allow_pickle=False
        )
        norm_sum = 0.0
        norm_min = float("inf")
        norm_max = 0.0
        for start in range(0, expected_rows, copy_chunk_rows):
            stop = min(start + copy_chunk_rows, expected_rows)
            block = np.asarray(output_vectors[start:stop], dtype=np.float32)
            if not np.isfinite(block).all():
                raise EmbeddingSubsetError(
                    f"输出 Embedding 行区间 [{start}, {stop}) 包含 NaN 或 Inf"
                )
            norms = np.linalg.norm(block, axis=1)
            norm_sum += float(norms.sum(dtype=np.float64))
            norm_min = min(norm_min, float(norms.min()))
            norm_max = max(norm_max, float(norms.max()))
        output_embeddings_sha256 = _sha256_file(output_embeddings_path)
        source_manifest_sha256 = _sha256_file(source_manifest_path)
        source_ids_sha256 = _sha256_file(source_ids_path)
        source_embeddings_sha256 = _sha256_file(source_embeddings_path)
        target_manifest_sha256 = _sha256_file(target_manifest)
        signature_payload = {
            "source_manifest_sha256": source_manifest_sha256,
            "source_ids_sha256": source_ids_sha256,
            "source_embeddings_sha256": source_embeddings_sha256,
            "target_manifest_sha256": target_manifest_sha256,
            "target_ids_sha256": target_ids_sha256,
            "expected_rows": expected_rows,
            "selection": "strict_ordered_poi_id_subsequence",
        }
        signature = hashlib.sha256(
            json.dumps(
                signature_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": "poi-embedding-subset-v1",
            "job_name": "beijing_poi_active_order14d_history10_mmbert_recall_128",
            "status": "completed",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "input": {
                "dir": _display_path(target_poi_ids.parent, project_root),
                "file_pattern": target_poi_ids.name,
                "id_field": "poi_id",
                "text_field": source_manifest.get("input", {}).get(
                    "text_field", "text"
                ),
                "total_rows": expected_rows,
                "fingerprint": target_ids_sha256,
                "sources": [
                    {
                        "name": target_poi_ids.name,
                        "rows_scanned": expected_rows,
                        "sha256_scanned": target_ids_sha256,
                    }
                ],
            },
            "model": source_manifest.get("model"),
            "derivation": {
                "method": "strict_ordered_poi_id_subsequence",
                "exact_vector_copy": True,
                "copy_strategy": "contiguous_source_range_then_memory_gather",
                "source_dir": _display_path(source_dir, project_root),
                "source_rows_scanned": source_rows,
                "source_manifest_sha256": source_manifest_sha256,
                "source_poi_ids_sha256": source_ids_sha256,
                "source_embeddings_sha256": source_embeddings_sha256,
                "target_catalog_manifest": _display_path(
                    target_manifest, project_root
                ),
                "target_catalog_manifest_sha256": target_manifest_sha256,
                "target_poi_ids_sha256": target_ids_sha256,
            },
            "output": {
                "dir": _display_path(output_dir, project_root),
                "embeddings": "embeddings.npy",
                "poi_ids": "poi_ids.jsonl",
                "shape": [expected_rows, int(source_embeddings.shape[1])],
                "dtype": str(source_embeddings.dtype),
                "embeddings_sha256": output_embeddings_sha256,
                "poi_ids_sha256": output_ids_sha256,
            },
            "metrics": {
                "rows_copied": expected_rows,
                "copy_seconds": time.perf_counter() - started,
                "l2_norm_min": norm_min,
                "l2_norm_mean": norm_sum / expected_rows,
                "l2_norm_max": norm_max,
                "vectors_all_finite": True,
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "device": "cpu_memmap_exact_copy",
            },
            "signature": signature,
        }
        manifest_path = build_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(build_dir, output_dir)
        return manifest
    except BaseException:
        try:
            del output_embeddings
        except UnboundLocalError:
            pass
        shutil.rmtree(build_dir, ignore_errors=True)
        raise

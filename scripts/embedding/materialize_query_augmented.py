#!/usr/bin/env python3
"""Stream a sparse Query-augmented POI representation into one frozen NPY."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.query_augmented_eval import fuse_embedding_chunk  # noqa: E402
from poi_gr.embedding.query_augmentation import (  # noqa: E402
    validate_query_poi_aggregates,
)
from poi_gr.embedding.query_category_residual import (  # noqa: E402
    validate_query_category_residual,
)


SCHEMA_VERSION = "query-augmented-poi-embedding-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{name} 不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} 必须是 JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _create_npy(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> int:
    """Create an NPY header and return the raw row-data offset."""

    with path.open("wb") as handle:
        np.lib.format.write_array_header_2_0(
            handle,
            {
                "descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": False,
                "shape": shape,
            },
        )
        handle.flush()
        os.fsync(handle.fileno())
        return handle.tell()


def _npy_data_offset(
    path: Path, expected_shape: tuple[int, ...], expected_dtype: np.dtype[Any]
) -> int:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise RuntimeError(f"不支持断点 NPY 版本：{version}")
        if shape != expected_shape or fortran_order or dtype != expected_dtype:
            raise RuntimeError("E4 全量 Embedding 断点 NPY 头非法")
        return handle.tell()


def _validate_completed(
    output_dir: Path, *, expected_signature: str
) -> dict[str, Any] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file() or not (output_dir / "_SUCCESS").is_file():
        return None
    manifest = _load_json(manifest_path, "E4 全量 Embedding manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("signature") != expected_signature
    ):
        raise RuntimeError("输出目录已有产物，但协议或输入签名不一致")
    output = manifest.get("output", {})
    values = np.load(output_dir / str(output.get("embeddings")), mmap_mode="r")
    if list(values.shape) != output.get("shape") or str(values.dtype) != output.get(
        "dtype"
    ):
        raise RuntimeError("已完成 E4 Embedding 的 shape/dtype 与 manifest 不一致")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按评测同一公式流式冻结全量 Query 增强 POI Embedding。"
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--aggregates-dir", type=Path, required=True)
    parser.add_argument("--category-residual-dir", type=Path, required=True)
    parser.add_argument("--category-indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--chunk-rows", type=int, default=32_768)
    parser.add_argument("--checkpoint-chunks", type=int, default=8)
    parser.add_argument(
        "--expected-base-embeddings-sha256",
        help="已由冻结评测验证的源向量 SHA256；记录进派生输入契约。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.alpha <= 1.0 or not 0.0 < args.beta <= 1.0:
        raise RuntimeError("alpha/beta 必须位于 (0,1]")
    if args.chunk_rows <= 0 or args.checkpoint_chunks <= 0:
        raise RuntimeError("chunk-rows/checkpoint-chunks 必须为正整数")

    base_dir = _resolve(args.base_dir)
    aggregates_dir = _resolve(args.aggregates_dir)
    residual_dir = _resolve(args.category_residual_dir)
    category_indices_path = _resolve(args.category_indices)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest_path = base_dir / "manifest.json"
    base_manifest = _load_json(base_manifest_path, "BGE manifest")
    if base_manifest.get("status") != "completed":
        raise RuntimeError("BGE manifest 状态不是 completed")
    validate_query_poi_aggregates(aggregates_dir)
    validate_query_category_residual(residual_dir)

    embeddings_path = base_dir / str(base_manifest.get("output", {}).get("embeddings"))
    base = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    declared_shape = tuple(base_manifest.get("output", {}).get("shape", ()))
    declared_dtype = str(base_manifest.get("output", {}).get("dtype"))
    if base.shape != declared_shape or str(base.dtype) != declared_dtype:
        raise RuntimeError("BGE Embedding shape/dtype 与 manifest 不一致")
    if base.ndim != 2 or str(base.dtype) != "float16":
        raise RuntimeError("当前冻结协议要求二维 float16 BGE Embedding")

    aggregates_manifest_path = aggregates_dir / "manifest.json"
    residual_manifest_path = residual_dir / "manifest.json"
    source_hashes = {
        "base_manifest": _sha256_file(base_manifest_path),
        "base_embeddings": args.expected_base_embeddings_sha256,
        "aggregates_manifest": _sha256_file(aggregates_manifest_path),
        "category_residual_manifest": _sha256_file(residual_manifest_path),
        "category_indices": _sha256_file(category_indices_path),
    }
    if args.expected_base_embeddings_sha256 is not None and len(
        args.expected_base_embeddings_sha256
    ) != 64:
        raise RuntimeError("expected-base-embeddings-sha256 必须是 64 位 SHA256")

    covered_rows = np.load(
        aggregates_dir / "covered_poi_rows.npy", mmap_mode="r", allow_pickle=False
    )
    query_aggregates = np.load(
        aggregates_dir / "e2_query_weighted.npy", mmap_mode="r", allow_pickle=False
    )
    category_indices = np.load(
        category_indices_path, mmap_mode="r", allow_pickle=False
    )
    category_means = np.load(
        residual_dir / "category_query_means.npy", mmap_mode="r", allow_pickle=False
    )
    if covered_rows.shape != (len(query_aggregates),):
        raise RuntimeError("covered_poi_rows 与 E2 Query 聚合行数不一致")
    if category_indices.shape != (len(base),):
        raise RuntimeError("类别行号与全量 POI 行数不一致")
    if query_aggregates.shape[1] != base.shape[1] or category_means.shape[1] != base.shape[1]:
        raise RuntimeError("BGE、E2 Query 聚合与类别均值维度不一致")

    position_by_poi_row = np.full(len(base), -1, dtype=np.int32)
    position_by_poi_row[covered_rows] = np.arange(len(covered_rows), dtype=np.int32)
    if np.count_nonzero(position_by_poi_row >= 0) != len(covered_rows):
        raise RuntimeError("covered_poi_rows 存在重复或越界")
    category_by_aggregate = np.asarray(category_indices[covered_rows], dtype=np.int32)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_hashes": source_hashes,
        "source_shape": list(base.shape),
        "source_dtype": str(base.dtype),
        "covered_pois": int(len(covered_rows)),
        "method": {
            "name": "E4-category-residual-query-fusion",
            "query_aggregate": "e2_query_weighted",
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "uncovered_poi_policy": "copy normalized BGE content embedding",
            "formula": "normalize((1-alpha)*c_i + alpha*normalize(q_i-beta*mean_category(i)))",
        },
        "output_dtype": "float16",
    }
    signature = _signature(provenance)
    completed = _validate_completed(output_dir, expected_signature=signature)
    if completed is not None:
        print(json.dumps({"status": "completed", "reused": True, "output_dir": str(output_dir)}, ensure_ascii=False))
        return 0

    embeddings_output = output_dir / "embeddings.npy"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "manifest.json"
    start_row = 0
    output_dtype = np.dtype(np.float16)
    if progress_path.is_file():
        progress = _load_json(progress_path, "E4 全量 Embedding progress")
        if progress.get("signature") != signature:
            raise RuntimeError("输出目录存在不同输入签名的断点")
        start_row = int(progress.get("next_row", -1))
        if start_row < 0 or start_row > len(base) or not embeddings_output.is_file():
            raise RuntimeError("E4 全量 Embedding 断点非法")
        data_offset = _npy_data_offset(embeddings_output, base.shape, output_dtype)
    else:
        if manifest_path.exists():
            prior_manifest = _load_json(manifest_path, "E4 全量 Embedding manifest")
            if (
                prior_manifest.get("status") != "running"
                or prior_manifest.get("signature") != signature
            ):
                raise RuntimeError("输出目录存在无有效断点的不同产物")
        data_offset = _create_npy(embeddings_output, base.shape, output_dtype)

    started = time.perf_counter()
    manifest = {
        **provenance,
        "status": "running",
        "signature": signature,
        "started_at": _utc_now(),
        "input": {
            "total_rows": int(len(base)),
            "fingerprint": signature,
            "base_dir": str(base_dir),
            "aggregates_dir": str(aggregates_dir),
            "category_residual_dir": str(residual_dir),
            "category_indices": str(category_indices_path),
        },
        "output": {
            "dir": str(output_dir),
            "embeddings": embeddings_output.name,
            "poi_ids": str(base_dir / str(base_manifest["output"]["poi_ids"])),
            "shape": list(base.shape),
            "dtype": "float16",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "chunk_rows": args.chunk_rows,
            "checkpoint_chunks": args.checkpoint_chunks,
            "storage": "project-local resumable sequential NPY",
        },
    }
    _write_json_atomic(manifest_path, manifest)

    row_bytes = base.shape[1] * output_dtype.itemsize
    expected_offset = data_offset + start_row * row_bytes
    with embeddings_output.open("r+b", buffering=16 * 1024 * 1024) as output:
        output.seek(expected_offset)
        output.truncate(expected_offset)
        chunk_count = 0
        with tqdm(
            desc="Materialize E4 embedding",
            unit="poi",
            initial=start_row,
            total=len(base),
        ) as progress_bar:
            for start in range(start_row, len(base), args.chunk_rows):
                stop = min(start + args.chunk_rows, len(base))
                fused = fuse_embedding_chunk(
                    base,
                    query_aggregates,
                    position_by_poi_row,
                    start=start,
                    stop=stop,
                    alpha=args.alpha,
                    fusion_chunk_rows=args.chunk_rows,
                    category_by_aggregate=category_by_aggregate,
                    category_query_means=category_means,
                    residual_beta=args.beta,
                )
                frozen_chunk = np.ascontiguousarray(fused, dtype=output_dtype)
                output.write(frozen_chunk.tobytes(order="C"))
                progress_bar.update(stop - start)
                chunk_count += 1
                if chunk_count % args.checkpoint_chunks == 0 or stop == len(base):
                    output.flush()
                    os.fsync(output.fileno())
                    _write_json_atomic(
                        progress_path,
                        {
                            "signature": signature,
                            "status": "running" if stop < len(base) else "completed",
                            "next_row": stop,
                            "total_rows": int(len(base)),
                            "updated_at": _utc_now(),
                        },
                    )

    frozen = np.load(embeddings_output, mmap_mode="r", allow_pickle=False)
    sample_rows = np.linspace(0, len(frozen) - 1, 4096, dtype=np.int64)
    sample = np.asarray(frozen[sample_rows], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)
    if not np.isfinite(sample).all() or np.any(norms <= 0):
        raise RuntimeError("冻结 E4 Embedding 抽样校验失败")
    manifest.update(
        {
            "status": "completed",
            "finished_at": _utc_now(),
            "validation": {
                "sample_rows": int(len(sample_rows)),
                "sample_all_finite": True,
                "sample_l2_norm_min": float(norms.min()),
                "sample_l2_norm_max": float(norms.max()),
                "sample_l2_norm_mean": float(norms.mean()),
            },
        }
    )
    manifest["runtime"]["elapsed_seconds"] = time.perf_counter() - started
    _write_json_atomic(manifest_path, manifest)
    (output_dir / "_SUCCESS").touch()
    print(
        json.dumps(
            {
                "status": "completed",
                "reused": False,
                "output_dir": str(output_dir),
                "shape": list(base.shape),
                "covered_pois": int(len(covered_rows)),
                "elapsed_seconds": manifest["runtime"]["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

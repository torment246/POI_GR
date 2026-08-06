"""Build paper-compatible GNPR identifiers from three-level Semantic IDs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from poi_gr.methods.tiger_identifier import assign_collision_codes
from poi_gr.sid_evaluation import (
    SidEvaluationError,
    compute_basic_metrics,
    load_sid_input,
)


GNPR_IDENTIFIER_SCHEMA_VERSION = "gnpr-poi-identifier-v1"
GNPR_IDENTIFIER_METRICS_SCHEMA_VERSION = "gnpr-poi-identifier-metrics-v1"
OUTPUT_FILENAMES = (
    "dedup_codes.npy",
    "gnpr_ids.npy",
    "poi_gnpr_id_mapping.parquet",
    "gnpr_id_manifest.json",
    "metrics.json",
)


class GnprIdentifierError(ValueError):
    """Raised when GNPR identifier inputs or outputs violate the contract."""


@dataclass(frozen=True)
class GnprIdentifierResult:
    """Completed GNPR identifier artifacts and their stable mapping hash."""

    manifest: dict[str, Any]
    metrics: dict[str, Any]
    mapping_sha256: str


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def compose_gnpr_ids(
    sid_codes: np.ndarray,
    collision_codes: np.ndarray,
    bucket_sizes_by_row: np.ndarray,
) -> np.ndarray:
    """Append dedup codes to colliding SIDs and use -1 for singleton rows."""

    sid_codes = np.asarray(sid_codes)
    collision_codes = np.asarray(collision_codes)
    bucket_sizes_by_row = np.asarray(bucket_sizes_by_row)
    if sid_codes.ndim != 2 or sid_codes.shape[1] != 3:
        raise GnprIdentifierError("GNPR base SID shape 必须是 [N,3]")
    if sid_codes.dtype.kind not in {"i", "u"} or np.any(sid_codes < 0):
        raise GnprIdentifierError("GNPR base SID 必须是非负整数")
    expected_shape = (sid_codes.shape[0],)
    if collision_codes.shape != expected_shape:
        raise GnprIdentifierError("collision_codes shape 必须是 [N]")
    if bucket_sizes_by_row.shape != expected_shape:
        raise GnprIdentifierError("bucket_sizes_by_row shape 必须是 [N]")
    if collision_codes.dtype.kind not in {"i", "u"}:
        raise GnprIdentifierError("collision_codes 必须使用整数 dtype")
    if np.any(collision_codes < 0) or np.any(bucket_sizes_by_row <= 0):
        raise GnprIdentifierError("碰撞编号或桶大小无效")

    gnpr_ids = np.full((sid_codes.shape[0], 4), -1, dtype=np.int32)
    gnpr_ids[:, :3] = sid_codes
    colliding = bucket_sizes_by_row > 1
    gnpr_ids[colliding, 3] = collision_codes[colliding]
    return gnpr_ids


def gnpr_id_key(tokens: Sequence[int]) -> str:
    """Serialize one three-token or deduplicated four-token GNPR identifier."""

    if len(tokens) != 4:
        raise GnprIdentifierError("GNPR identifier key 要求四列内部表示")
    base = f"{int(tokens[0])}-{int(tokens[1])}-{int(tokens[2])}"
    dedup_code = int(tokens[3])
    return base if dedup_code < 0 else f"{base}|d{dedup_code}"


def _validate_dedup_assignment(
    sid_codes: np.ndarray,
    poi_ids: Sequence[str],
    collision_codes: np.ndarray,
    bucket_sizes_by_row: np.ndarray,
) -> None:
    rows = np.flatnonzero(bucket_sizes_by_row > 1)
    ordered = sorted(
        (int(row) for row in rows),
        key=lambda row: (
            tuple(int(token) for token in sid_codes[row]),
            poi_ids[row],
        ),
    )
    previous_sid: tuple[int, ...] | None = None
    expected_code = 0
    for row in ordered:
        current_sid = tuple(int(token) for token in sid_codes[row])
        if current_sid != previous_sid:
            previous_sid = current_sid
            expected_code = 0
        if int(collision_codes[row]) != expected_code:
            raise GnprIdentifierError(
                "dedup code 未按 SID、poi_id 顺序从 0 连续分配"
            )
        expected_code += 1


def _mapping_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("poi_id", pa.string(), nullable=False),
            pa.field("s1", pa.int32(), nullable=False),
            pa.field("s2", pa.int32(), nullable=False),
            pa.field("s3", pa.int32(), nullable=False),
            pa.field("base_sid_key", pa.string(), nullable=False),
            pa.field("base_sid_bucket_size", pa.int32(), nullable=False),
            pa.field("has_semantic_collision", pa.bool_(), nullable=False),
            pa.field("dedup_code", pa.int32(), nullable=True),
            pa.field("gnpr_id_key", pa.string(), nullable=False),
        ]
    )


def _write_mapping_parquet(
    path: Path,
    poi_ids: Sequence[str],
    gnpr_ids: np.ndarray,
    bucket_sizes_by_row: np.ndarray,
    *,
    chunk_rows: int,
) -> None:
    if chunk_rows <= 0:
        raise GnprIdentifierError("chunk_rows 必须大于 0")

    schema = _mapping_schema()
    writer = pq.ParquetWriter(
        path,
        schema,
        version="2.6",
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )
    try:
        for start in range(0, len(poi_ids), chunk_rows):
            end = min(start + chunk_rows, len(poi_ids))
            ids = np.asarray(gnpr_ids[start:end], dtype=np.int32)
            sizes = np.asarray(
                bucket_sizes_by_row[start:end], dtype=np.int32
            )
            base_keys = [
                f"{int(row[0])}-{int(row[1])}-{int(row[2])}"
                for row in ids
            ]
            keys = [
                base if int(row[3]) < 0 else f"{base}|d{int(row[3])}"
                for base, row in zip(base_keys, ids, strict=True)
            ]
            dedup_codes = ids[:, 3]
            table = pa.Table.from_arrays(
                [
                    pa.array(poi_ids[start:end], type=pa.string()),
                    pa.array(ids[:, 0], type=pa.int32()),
                    pa.array(ids[:, 1], type=pa.int32()),
                    pa.array(ids[:, 2], type=pa.int32()),
                    pa.array(base_keys, type=pa.string()),
                    pa.array(sizes, type=pa.int32()),
                    pa.array(sizes > 1, type=pa.bool_()),
                    pa.array(
                        dedup_codes,
                        mask=dedup_codes < 0,
                        type=pa.int32(),
                    ),
                    pa.array(keys, type=pa.string()),
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=chunk_rows)
    finally:
        writer.close()


def _write_npy(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_output_directory(output_dir: Path) -> bool:
    created = False
    if output_dir.exists():
        if not output_dir.is_dir():
            raise GnprIdentifierError(f"输出路径不是目录：{output_dir}")
        unexpected = {
            path.name for path in output_dir.iterdir()
        } - set(OUTPUT_FILENAMES)
        if unexpected:
            raise GnprIdentifierError(
                "输出目录包含 GNPR identifier 范围外文件，不会覆盖："
                + ", ".join(sorted(unexpected))
            )
    else:
        output_dir.mkdir(parents=True)
        created = True
    return created


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start=start.resolve())


def build_gnpr_identifiers(
    sid_manifest_path: Path,
    output_dir: Path,
    *,
    chunk_rows: int = 100_000,
    progress: Callable[[str], None] | None = None,
) -> GnprIdentifierResult:
    """Build and atomically persist paper-compatible unique GNPR identifiers."""

    started = time.perf_counter()
    sid_manifest_path = sid_manifest_path.resolve()
    output_dir = output_dir.resolve()

    if progress is not None:
        progress("读取三层 SID 与 POI ID，并校验行级对齐")
    try:
        sid_input = load_sid_input(sid_manifest_path)
    except SidEvaluationError as error:
        raise GnprIdentifierError(str(error)) from error
    if sid_input.codes.shape[1] != 3:
        raise GnprIdentifierError("GNPR 论文复现要求三层 base SID")

    if progress is not None:
        progress("在碰撞桶内按 poi_id 字典序分配 dedup code")
    basic, _, _, _ = compute_basic_metrics(sid_input.codes)
    assignment = assign_collision_codes(sid_input.codes, sid_input.poi_ids)
    expected = {
        "base_sid_distinct_count": int(basic["distinct_sid_count"]),
        "collision_bucket_count": int(basic["colliding_bucket_count"]),
        "singleton_poi_count": int(basic["singleton_poi_count"]),
        "colliding_poi_count": int(basic["colliding_poi_count"]),
        "max_bucket_size": int(basic["bucket_size_max"]),
    }
    actual = {key: int(getattr(assignment, key)) for key in expected}
    if actual != expected:
        raise GnprIdentifierError(
            "dedup 分配统计与 SID 指标不一致："
            + json.dumps(
                {"expected": expected, "actual": actual},
                ensure_ascii=False,
            )
        )
    _validate_dedup_assignment(
        sid_input.codes,
        sid_input.poi_ids,
        assignment.collision_codes,
        assignment.bucket_sizes_by_row,
    )
    gnpr_ids = compose_gnpr_ids(
        sid_input.codes,
        assignment.collision_codes,
        assignment.bucket_sizes_by_row,
    )
    if not np.array_equal(gnpr_ids[:, :3], sid_input.codes):
        raise GnprIdentifierError("移除 dedup code 后无法恢复原始三层 SID")

    if progress is not None:
        progress("验证三或四 token 的 GNPR identifier 全局唯一")
    identifier_distinct_count = int(np.unique(gnpr_ids, axis=0).shape[0])
    poi_count = int(gnpr_ids.shape[0])
    if identifier_distinct_count != poi_count:
        raise GnprIdentifierError(
            f"GNPR identifier 不唯一：{identifier_distinct_count:,}/{poi_count:,}"
        )

    dedup_codes = np.asarray(gnpr_ids[:, 3], dtype=np.int32)
    created_output_dir = _validate_output_directory(output_dir)
    suffix = f".{os.getpid()}.tmp"
    temporary_paths = {
        filename: output_dir / f".{filename}{suffix}"
        for filename in OUTPUT_FILENAMES
    }
    final_paths = {
        filename: output_dir / filename for filename in OUTPUT_FILENAMES
    }
    try:
        if progress is not None:
            progress("写入 dedup 数组、GNPR identifier 和 POI 映射")
        _write_npy(temporary_paths["dedup_codes.npy"], dedup_codes)
        _write_npy(temporary_paths["gnpr_ids.npy"], gnpr_ids)
        _write_mapping_parquet(
            temporary_paths["poi_gnpr_id_mapping.parquet"],
            sid_input.poi_ids,
            gnpr_ids,
            assignment.bucket_sizes_by_row,
            chunk_rows=chunk_rows,
        )

        dedup_hash = sha256_file(temporary_paths["dedup_codes.npy"])
        ids_hash = sha256_file(temporary_paths["gnpr_ids.npy"])
        mapping_hash = sha256_file(
            temporary_paths["poi_gnpr_id_mapping.parquet"]
        )
        elapsed = time.perf_counter() - started
        metrics: dict[str, Any] = {
            "schema_version": GNPR_IDENTIFIER_METRICS_SCHEMA_VERSION,
            "status": "completed",
            "poi_count": poi_count,
            "base_sid_distinct_count": assignment.base_sid_distinct_count,
            "base_sid_distinct_ratio": basic["distinct_sid_ratio"],
            "base_sid_collision_bucket_count": assignment.collision_bucket_count,
            "base_sid_colliding_poi_count": assignment.colliding_poi_count,
            "base_sid_colliding_poi_ratio": basic["colliding_poi_ratio"],
            "base_sid_collision_excess_count": basic["collision_excess_count"],
            "singleton_without_dedup_token_count": assignment.singleton_poi_count,
            "poi_with_dedup_token_count": assignment.colliding_poi_count,
            "required_dedup_token_count": assignment.max_bucket_size,
            "max_dedup_code": assignment.max_collision_code,
            "gnpr_id_distinct_count": identifier_distinct_count,
            "gnpr_id_unique_ratio": identifier_distinct_count / poi_count,
            "assignment_order": "poi_id_lexicographic_within_base_sid",
            "build_seconds": elapsed,
        }
        manifest: dict[str, Any] = {
            "schema_version": GNPR_IDENTIFIER_SCHEMA_VERSION,
            "status": "completed",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "paper_method": "GNPR-SID",
            "source": {
                "sid_manifest": _relative_path(sid_manifest_path, output_dir),
                "sid_manifest_sha256": sha256_file(sid_manifest_path),
                "sid_codes": _relative_path(sid_input.codes_path, output_dir),
                "sid_codes_sha256": sha256_file(sid_input.codes_path),
                "codebook_sizes": list(sid_input.codebook_sizes),
            },
            "poi_ids": {
                "path": _relative_path(sid_input.poi_ids_path, output_dir),
                "rows": poi_count,
                "sha256": sha256_file(sid_input.poi_ids_path),
                "unique": True,
            },
            "dedup_assignment": {
                "paper_behavior": "append_dedup_token_only_for_colliding_base_sid",
                "rule": "zero_based_poi_id_lexicographic_within_base_sid",
                "singleton_internal_sentinel": -1,
                "singleton_has_dedup_token": False,
                "codes_reused_across_base_sid_buckets": True,
                "required_token_count": assignment.max_bucket_size,
                "maximum_code": assignment.max_collision_code,
            },
            "dedup_codes": {
                "path": "dedup_codes.npy",
                "shape": [poi_count],
                "dtype": str(dedup_codes.dtype),
                "sha256": dedup_hash,
            },
            "gnpr_ids": {
                "path": "gnpr_ids.npy",
                "shape": list(gnpr_ids.shape),
                "dtype": str(gnpr_ids.dtype),
                "internal_token_order": ["S1", "S2", "S3", "D_OR_MINUS_ONE"],
                "codebook_capacities": list(sid_input.codebook_sizes),
                "dedup_token_capacity": assignment.max_bucket_size,
                "serialized_length": {"singleton": 3, "collision": 4},
                "sha256": ids_hash,
            },
            "mapping": {
                "path": "poi_gnpr_id_mapping.parquet",
                "rows": poi_count,
                "sha256": mapping_hash,
                "poi_id_unique": True,
                "gnpr_id_key_unique": True,
            },
            "metrics": {
                "path": "metrics.json",
                "schema_version": GNPR_IDENTIFIER_METRICS_SCHEMA_VERSION,
            },
        }
        _write_json(temporary_paths["metrics.json"], metrics)
        _write_json(temporary_paths["gnpr_id_manifest.json"], manifest)
        for filename in OUTPUT_FILENAMES:
            os.replace(temporary_paths[filename], final_paths[filename])
    except Exception:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        if created_output_dir:
            try:
                output_dir.rmdir()
            except OSError:
                pass
        raise

    return GnprIdentifierResult(
        manifest=manifest,
        metrics=metrics,
        mapping_sha256=mapping_hash,
    )

"""Build fixed-length TIGER item identifiers from three-level semantic IDs."""

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

from poi_gr.sid.evaluation import (
    SidEvaluationError,
    compute_basic_metrics,
    load_sid_input,
)


TIGER_IDENTIFIER_SCHEMA_VERSION = "tiger-item-identifier-v1"
TIGER_IDENTIFIER_METRICS_SCHEMA_VERSION = "tiger-item-identifier-metrics-v1"
OUTPUT_FILENAMES = (
    "collision_codes.npy",
    "tiger_ids.npy",
    "poi_tiger_id_mapping.parquet",
    "tiger_id_manifest.json",
    "metrics.json",
)


class TigerIdentifierError(ValueError):
    """Raised when TIGER identifier inputs or outputs violate the contract."""


@dataclass(frozen=True)
class CollisionAssignment:
    """Deterministic fourth-token assignment aligned with input SID rows."""

    collision_codes: np.ndarray
    bucket_sizes_by_row: np.ndarray
    base_sid_distinct_count: int
    collision_bucket_count: int
    singleton_poi_count: int
    colliding_poi_count: int
    max_bucket_size: int
    max_collision_code: int


@dataclass(frozen=True)
class TigerIdentifierResult:
    """Completed TIGER identifier artifacts and their stable mapping hash."""

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


def assign_collision_codes(
    sid_codes: np.ndarray,
    poi_ids: Sequence[str],
) -> CollisionAssignment:
    """Assign zero-based fourth codes by POI ID within every Semantic ID bucket."""

    sid_codes = np.asarray(sid_codes)
    if sid_codes.ndim != 2 or sid_codes.shape[1] != 3:
        raise TigerIdentifierError("TIGER base SID shape 必须是 [N,3]")
    if sid_codes.dtype.kind not in {"i", "u"}:
        raise TigerIdentifierError("TIGER base SID 必须使用整数 dtype")
    if sid_codes.shape[0] == 0:
        raise TigerIdentifierError("TIGER base SID 不能为空")
    if np.any(sid_codes < 0):
        raise TigerIdentifierError("TIGER base SID 不允许包含负 Token")
    if len(poi_ids) != sid_codes.shape[0]:
        raise TigerIdentifierError("POI ID 行数与 TIGER base SID 行数不一致")
    if any(not isinstance(poi_id, str) or not poi_id.strip() for poi_id in poi_ids):
        raise TigerIdentifierError("POI ID 必须是非空字符串")
    if len(set(poi_ids)) != len(poi_ids):
        raise TigerIdentifierError("POI ID 存在重复")

    _, inverse, bucket_sizes = np.unique(
        sid_codes,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    bucket_sizes_by_row = bucket_sizes[inverse].astype(np.int32, copy=False)
    collision_codes = np.zeros(sid_codes.shape[0], dtype=np.int32)

    colliding_rows = np.flatnonzero(bucket_sizes_by_row > 1)
    ordered_rows = np.fromiter(
        sorted(
            (int(row) for row in colliding_rows),
            key=lambda row: (int(inverse[row]), poi_ids[row]),
        ),
        dtype=np.int64,
        count=len(colliding_rows),
    )
    if len(ordered_rows):
        ordered_buckets = inverse[ordered_rows]
        group_start = np.empty(len(ordered_rows), dtype=np.bool_)
        group_start[0] = True
        group_start[1:] = ordered_buckets[1:] != ordered_buckets[:-1]
        positions = np.arange(len(ordered_rows), dtype=np.int64)
        starts = np.maximum.accumulate(np.where(group_start, positions, 0))
        ordered_codes = positions - starts
        if int(ordered_codes.max()) > np.iinfo(np.int32).max:
            raise TigerIdentifierError("collision code 超出 int32 范围")
        collision_codes[ordered_rows] = ordered_codes.astype(np.int32)

    collision_bucket_count = int((bucket_sizes > 1).sum())
    singleton_poi_count = int((bucket_sizes_by_row == 1).sum())
    colliding_poi_count = int((bucket_sizes_by_row > 1).sum())
    max_bucket_size = int(bucket_sizes.max())
    max_collision_code = max_bucket_size - 1
    return CollisionAssignment(
        collision_codes=collision_codes,
        bucket_sizes_by_row=bucket_sizes_by_row,
        base_sid_distinct_count=len(bucket_sizes),
        collision_bucket_count=collision_bucket_count,
        singleton_poi_count=singleton_poi_count,
        colliding_poi_count=colliding_poi_count,
        max_bucket_size=max_bucket_size,
        max_collision_code=max_collision_code,
    )


def compose_tiger_ids(
    sid_codes: np.ndarray,
    collision_codes: np.ndarray,
) -> np.ndarray:
    """Append the always-present collision code to three-level Semantic IDs."""

    sid_codes = np.asarray(sid_codes)
    collision_codes = np.asarray(collision_codes)
    if sid_codes.ndim != 2 or sid_codes.shape[1] != 3:
        raise TigerIdentifierError("TIGER base SID shape 必须是 [N,3]")
    if collision_codes.shape != (sid_codes.shape[0],):
        raise TigerIdentifierError("collision_codes shape 必须是 [N]")
    if collision_codes.dtype.kind not in {"i", "u"}:
        raise TigerIdentifierError("collision_codes 必须使用整数 dtype")
    if np.any(collision_codes < 0):
        raise TigerIdentifierError("collision_codes 不允许包含负 Token")

    tiger_ids = np.empty((sid_codes.shape[0], 4), dtype=np.int32)
    tiger_ids[:, :3] = sid_codes
    tiger_ids[:, 3] = collision_codes
    return tiger_ids


def tiger_id_key(tokens: Sequence[int]) -> str:
    """Serialize one fixed four-token TIGER identifier."""

    if len(tokens) != 4:
        raise TigerIdentifierError("TIGER identifier key 要求四个 Token")
    return (
        f"{int(tokens[0])}-{int(tokens[1])}-{int(tokens[2])}"
        f"|c{int(tokens[3])}"
    )


def _validate_assignment_continuity(
    sid_codes: np.ndarray,
    poi_ids: Sequence[str],
    assignment: CollisionAssignment,
) -> None:
    colliding_rows = np.flatnonzero(assignment.bucket_sizes_by_row > 1)
    ordered = sorted(
        (int(row) for row in colliding_rows),
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
        if int(assignment.collision_codes[row]) != expected_code:
            raise TigerIdentifierError(
                "collision code 未按 SID、poi_id 顺序从 0 连续分配"
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
            pa.field("collision_code", pa.int32(), nullable=False),
            pa.field("tiger_id_key", pa.string(), nullable=False),
        ]
    )


def _write_mapping_parquet(
    path: Path,
    poi_ids: Sequence[str],
    sid_codes: np.ndarray,
    assignment: CollisionAssignment,
    *,
    chunk_rows: int,
) -> None:
    if chunk_rows <= 0:
        raise TigerIdentifierError("chunk_rows 必须大于 0")

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
            codes = np.asarray(sid_codes[start:end], dtype=np.int32)
            collision = assignment.collision_codes[start:end]
            base_keys = [
                f"{int(row[0])}-{int(row[1])}-{int(row[2])}" for row in codes
            ]
            tiger_keys = [
                f"{key}|c{int(code)}"
                for key, code in zip(base_keys, collision, strict=True)
            ]
            table = pa.Table.from_arrays(
                [
                    pa.array(poi_ids[start:end], type=pa.string()),
                    pa.array(codes[:, 0], type=pa.int32()),
                    pa.array(codes[:, 1], type=pa.int32()),
                    pa.array(codes[:, 2], type=pa.int32()),
                    pa.array(base_keys, type=pa.string()),
                    pa.array(
                        assignment.bucket_sizes_by_row[start:end],
                        type=pa.int32(),
                    ),
                    pa.array(
                        assignment.bucket_sizes_by_row[start:end] > 1,
                        type=pa.bool_(),
                    ),
                    pa.array(collision, type=pa.int32()),
                    pa.array(tiger_keys, type=pa.string()),
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
            raise TigerIdentifierError(f"输出路径不是目录：{output_dir}")
        unexpected = {
            path.name for path in output_dir.iterdir()
        } - set(OUTPUT_FILENAMES)
        if unexpected:
            raise TigerIdentifierError(
                "输出目录包含 TIGER identifier 范围外文件，不会覆盖："
                + ", ".join(sorted(unexpected))
            )
    else:
        output_dir.mkdir(parents=True)
        created = True
    return created


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start=start.resolve())


def build_tiger_identifiers(
    sid_manifest_path: Path,
    output_dir: Path,
    *,
    chunk_rows: int = 100_000,
    progress: Callable[[str], None] | None = None,
) -> TigerIdentifierResult:
    """Build and atomically persist fixed-length unique TIGER identifiers."""

    started = time.perf_counter()
    sid_manifest_path = sid_manifest_path.resolve()
    output_dir = output_dir.resolve()

    if progress is not None:
        progress("读取三层 SID 与 POI ID，并校验行级对齐")
    try:
        sid_input = load_sid_input(sid_manifest_path)
    except SidEvaluationError as error:
        raise TigerIdentifierError(str(error)) from error
    if sid_input.codes.shape[1] != 3:
        raise TigerIdentifierError("TIGER 论文复现要求三层 base SID")

    if progress is not None:
        progress("按完整 SID 分桶，并在桶内按 poi_id 字典序分配 collision code")
    basic, _, _, _ = compute_basic_metrics(sid_input.codes)
    assignment = assign_collision_codes(sid_input.codes, sid_input.poi_ids)
    expected = {
        "base_sid_distinct_count": int(basic["distinct_sid_count"]),
        "collision_bucket_count": int(basic["colliding_bucket_count"]),
        "singleton_poi_count": int(basic["singleton_poi_count"]),
        "colliding_poi_count": int(basic["colliding_poi_count"]),
        "max_bucket_size": int(basic["bucket_size_max"]),
    }
    actual = {
        key: int(getattr(assignment, key))
        for key in expected
    }
    if actual != expected:
        raise TigerIdentifierError(
            "collision 分配统计与 SID 指标不一致："
            + json.dumps({"expected": expected, "actual": actual}, ensure_ascii=False)
        )
    _validate_assignment_continuity(
        sid_input.codes,
        sid_input.poi_ids,
        assignment,
    )
    tiger_ids = compose_tiger_ids(
        sid_input.codes,
        assignment.collision_codes,
    )
    if not np.array_equal(tiger_ids[:, :3], sid_input.codes):
        raise TigerIdentifierError("移除 collision code 后无法恢复原始三层 SID")

    if progress is not None:
        progress("验证四层 TIGER identifier 全局唯一")
    tiger_id_distinct_count = int(np.unique(tiger_ids, axis=0).shape[0])
    poi_count = int(tiger_ids.shape[0])
    if tiger_id_distinct_count != poi_count:
        raise TigerIdentifierError(
            f"TIGER identifier 不唯一：{tiger_id_distinct_count:,}/{poi_count:,}"
        )

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
            progress("写入 collision 数组、四层 identifier 和 POI 映射")
        _write_npy(
            temporary_paths["collision_codes.npy"],
            assignment.collision_codes,
        )
        _write_npy(temporary_paths["tiger_ids.npy"], tiger_ids)
        _write_mapping_parquet(
            temporary_paths["poi_tiger_id_mapping.parquet"],
            sid_input.poi_ids,
            sid_input.codes,
            assignment,
            chunk_rows=chunk_rows,
        )

        collision_hash = sha256_file(temporary_paths["collision_codes.npy"])
        tiger_ids_hash = sha256_file(temporary_paths["tiger_ids.npy"])
        mapping_hash = sha256_file(
            temporary_paths["poi_tiger_id_mapping.parquet"]
        )
        source_sid_hash = sha256_file(sid_input.codes_path)
        poi_ids_hash = sha256_file(sid_input.poi_ids_path)
        source_manifest_hash = sha256_file(sid_manifest_path)
        elapsed = time.perf_counter() - started

        metrics: dict[str, Any] = {
            "schema_version": TIGER_IDENTIFIER_METRICS_SCHEMA_VERSION,
            "status": "completed",
            "poi_count": poi_count,
            "base_sid_distinct_count": assignment.base_sid_distinct_count,
            "base_sid_distinct_ratio": basic["distinct_sid_ratio"],
            "base_sid_collision_bucket_count": assignment.collision_bucket_count,
            "base_sid_colliding_poi_count": assignment.colliding_poi_count,
            "base_sid_colliding_poi_ratio": basic["colliding_poi_ratio"],
            "base_sid_collision_excess_count": basic["collision_excess_count"],
            "base_sid_collision_excess_ratio": basic[
                "collision_excess_ratio"
            ],
            "singleton_poi_count": assignment.singleton_poi_count,
            "max_base_sid_bucket_size": assignment.max_bucket_size,
            "required_collision_token_count": assignment.max_bucket_size,
            "max_collision_code": assignment.max_collision_code,
            "zero_collision_code_count": assignment.base_sid_distinct_count,
            "nonzero_collision_code_count": basic["collision_excess_count"],
            "tiger_id_distinct_count": tiger_id_distinct_count,
            "tiger_id_unique_ratio": tiger_id_distinct_count / poi_count,
            "assignment_order": "poi_id_lexicographic_within_base_sid",
            "build_seconds": elapsed,
        }
        manifest: dict[str, Any] = {
            "schema_version": TIGER_IDENTIFIER_SCHEMA_VERSION,
            "status": "completed",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "paper_method": "TIGER",
            "source": {
                "sid_manifest": _relative_path(sid_manifest_path, output_dir),
                "sid_manifest_sha256": source_manifest_hash,
                "sid_codes": _relative_path(sid_input.codes_path, output_dir),
                "sid_codes_sha256": source_sid_hash,
                "codebook_sizes": list(sid_input.codebook_sizes),
            },
            "poi_ids": {
                "path": _relative_path(sid_input.poi_ids_path, output_dir),
                "rows": poi_count,
                "sha256": poi_ids_hash,
                "unique": True,
            },
            "collision_assignment": {
                "rule": "zero_based_poi_id_lexicographic_within_base_sid",
                "singleton_code": 0,
                "singleton_always_has_fourth_token": True,
                "codes_reused_across_base_sid_buckets": True,
                "required_token_count": assignment.max_bucket_size,
                "maximum_code": assignment.max_collision_code,
            },
            "collision_codes": {
                "path": "collision_codes.npy",
                "shape": [poi_count],
                "dtype": str(assignment.collision_codes.dtype),
                "sha256": collision_hash,
            },
            "tiger_ids": {
                "path": "tiger_ids.npy",
                "shape": list(tiger_ids.shape),
                "dtype": str(tiger_ids.dtype),
                "token_order": ["S1", "S2", "S3", "C"],
                "token_capacities": [
                    *sid_input.codebook_sizes,
                    assignment.max_bucket_size,
                ],
                "fixed_length": 4,
                "sha256": tiger_ids_hash,
            },
            "mapping": {
                "path": "poi_tiger_id_mapping.parquet",
                "rows": poi_count,
                "sha256": mapping_hash,
                "poi_id_unique": True,
                "tiger_id_key_unique": True,
            },
            "metrics": {
                "path": "metrics.json",
                "schema_version": TIGER_IDENTIFIER_METRICS_SCHEMA_VERSION,
            },
        }
        _write_json(temporary_paths["metrics.json"], metrics)
        _write_json(
            temporary_paths["tiger_id_manifest.json"],
            manifest,
        )
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

    return TigerIdentifierResult(
        manifest=manifest,
        metrics=metrics,
        mapping_sha256=mapping_hash,
    )

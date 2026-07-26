"""Build deterministic Dedup Codes for colliding base PIDs."""

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

from .sid_evaluation import compute_basic_metrics


FINAL_PID_SCHEMA_VERSION = "dedup-pid-v1"
FINAL_PID_METRICS_SCHEMA_VERSION = "dedup-pid-metrics-v1"
EXPECTED_BASE_PID_TOKEN_ORDER = (
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "S1",
    "S2",
    "S3",
)
EXPECTED_PID001_BASELINE = {
    "poi_count": 2_337_178,
    "base_pid_distinct_count": 1_984_068,
    "base_pid_colliding_poi_count": 557_154,
    "base_pid_collision_bucket_count": 204_044,
    "singleton_poi_count": 1_780_024,
    "max_base_bucket_size": 322,
}
OUTPUT_FILENAMES = (
    "dedup_codes.npy",
    "final_pid_codes.npy",
    "poi_pid_mapping.parquet",
    "final_pid_manifest.json",
    "metrics.json",
)


class DedupPidError(ValueError):
    pass


@dataclass(frozen=True)
class BasePidInput:
    manifest_path: Path
    manifest: dict[str, Any]
    metrics_path: Path
    metrics: dict[str, Any]
    codes_path: Path
    codes: np.ndarray
    poi_ids_path: Path
    poi_ids: tuple[str, ...]
    poi_ids_sha256: str
    codes_sha256: str


@dataclass(frozen=True)
class DedupAssignment:
    dedup_codes: np.ndarray
    bucket_sizes_by_row: np.ndarray
    base_pid_distinct_count: int
    collision_bucket_count: int
    singleton_poi_count: int
    dedup_poi_count: int
    max_base_bucket_size: int
    max_dedup_code: int


@dataclass(frozen=True)
class DedupPidResult:
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    mapping_sha256: str


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 of one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise DedupPidError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DedupPidError(f"{name} JSON 解析失败：{path}") from error
    if not isinstance(value, dict):
        raise DedupPidError(f"{name} 必须是 JSON object：{path}")
    return value


def _resolve_path(value: Any, manifest_path: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DedupPidError(f"{name} 必须是非空路径")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _load_poi_ids(path: Path, expected_rows: int) -> tuple[str, ...]:
    if not path.is_file():
        raise DedupPidError(f"POI ID 文件不存在：{path}")
    poi_ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DedupPidError(f"POI ID 文件第 {line_number} 行为空")
            try:
                poi_id = json.loads(line)
            except json.JSONDecodeError as error:
                raise DedupPidError(
                    f"POI ID 文件第 {line_number} 行 JSON 解析失败"
                ) from error
            if not isinstance(poi_id, str) or not poi_id.strip():
                raise DedupPidError(
                    f"POI ID 文件第 {line_number} 行必须是非空 JSON 字符串"
                )
            if poi_id in seen:
                raise DedupPidError(f"POI ID 重复：{poi_id}")
            seen.add(poi_id)
            poi_ids.append(poi_id)
    if len(poi_ids) != expected_rows:
        raise DedupPidError(
            f"POI ID 行数 {len(poi_ids)} 与 base PID 行数 {expected_rows} 不一致"
        )
    return tuple(poi_ids)


def load_base_pid_input(pid_manifest_path: Path) -> BasePidInput:
    """Load and validate the PID-001 base array and aligned POI IDs."""

    pid_manifest_path = pid_manifest_path.resolve()
    manifest = _load_json_object(pid_manifest_path, "PID manifest")
    if manifest.get("schema_version") != "geohash-pid-v1":
        raise DedupPidError("base PID schema_version 必须是 geohash-pid-v1")
    if manifest.get("status") != "completed":
        raise DedupPidError("base PID manifest 状态不是 completed")

    pid_spec = manifest.get("pid_codes")
    if not isinstance(pid_spec, dict):
        raise DedupPidError("PID manifest 缺少 pid_codes")
    declared_shape = pid_spec.get("shape")
    if (
        not isinstance(declared_shape, list)
        or len(declared_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in declared_shape
        )
    ):
        raise DedupPidError("pid_codes.shape 必须是 [N,9]")
    if declared_shape[1] != 9:
        raise DedupPidError(
            f"base PID shape 必须是 [N,9]，manifest 实际为 {declared_shape}"
        )
    if pid_spec.get("order") != "gid_sid":
        raise DedupPidError("base PID order 必须是 gid_sid")
    if tuple(pid_spec.get("token_order", ())) != EXPECTED_BASE_PID_TOKEN_ORDER:
        raise DedupPidError("base PID token_order 必须为 G1..G6,S1..S3")
    try:
        declared_dtype = np.dtype(pid_spec.get("dtype"))
    except TypeError as error:
        raise DedupPidError("pid_codes.dtype 无效") from error
    if declared_dtype != np.dtype(np.int32):
        raise DedupPidError("base PID dtype 必须是 int32")

    codes_path = _resolve_path(
        pid_spec.get("path"), pid_manifest_path, "pid_codes.path"
    )
    if not codes_path.is_file():
        raise DedupPidError(f"base PID 文件不存在：{codes_path}")
    try:
        codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise DedupPidError(f"base PID NPY 读取失败：{codes_path}") from error
    if codes.ndim != 2 or codes.shape[1] != 9:
        raise DedupPidError(f"base PID shape 必须是 [N,9]，实际为 {codes.shape}")
    if list(codes.shape) != declared_shape:
        raise DedupPidError(
            f"base PID shape {list(codes.shape)} 与 manifest {declared_shape} 不一致"
        )
    if codes.dtype != declared_dtype:
        raise DedupPidError(
            f"base PID dtype {codes.dtype} 与 manifest {declared_dtype} 不一致"
        )
    if np.any(codes < 0):
        raise DedupPidError("base PID 不允许包含负 Token")
    if manifest.get("poi_count") != codes.shape[0]:
        raise DedupPidError(
            f"PID manifest poi_count {manifest.get('poi_count')} "
            f"与 base PID 行数 {codes.shape[0]} 不一致"
        )

    poi_spec = manifest.get("poi_ids")
    if not isinstance(poi_spec, dict):
        raise DedupPidError("PID manifest 缺少 poi_ids")
    if poi_spec.get("rows") != codes.shape[0]:
        raise DedupPidError(
            f"PID manifest poi_ids.rows {poi_spec.get('rows')} "
            f"与 base PID 行数 {codes.shape[0]} 不一致"
        )
    poi_ids_path = _resolve_path(
        poi_spec.get("path"), pid_manifest_path, "poi_ids.path"
    )
    actual_poi_hash = sha256_file(poi_ids_path)
    if poi_spec.get("sha256") != actual_poi_hash:
        raise DedupPidError(
            f"POI ID SHA256 {actual_poi_hash} 与 PID manifest 不一致"
        )
    poi_ids = _load_poi_ids(poi_ids_path, codes.shape[0])

    metrics_path = pid_manifest_path.parent / "metrics.json"
    metrics = _load_json_object(metrics_path, "PID-001 metrics")
    if metrics.get("status") != "completed":
        raise DedupPidError("PID-001 metrics 状态不是 completed")
    return BasePidInput(
        manifest_path=pid_manifest_path,
        manifest=manifest,
        metrics_path=metrics_path.resolve(),
        metrics=metrics,
        codes_path=codes_path,
        codes=codes,
        poi_ids_path=poi_ids_path,
        poi_ids=poi_ids,
        poi_ids_sha256=actual_poi_hash,
        codes_sha256=sha256_file(codes_path),
    )


def _base_metrics_from_basic(basic: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "poi_count": int(basic["poi_count"]),
        "base_pid_distinct_count": int(basic["distinct_sid_count"]),
        "base_pid_colliding_poi_count": int(basic["colliding_poi_count"]),
        "base_pid_collision_bucket_count": int(basic["colliding_bucket_count"]),
        "singleton_poi_count": int(basic["singleton_poi_count"]),
        "max_base_bucket_size": int(basic["bucket_size_max"]),
    }


def validate_base_metrics(
    basic: Mapping[str, Any],
    existing_metrics: Mapping[str, Any],
    expected_baseline: Mapping[str, int] | None,
) -> dict[str, int]:
    """Check recomputed base PID metrics against PID-001 and task constants."""

    existing = existing_metrics.get("gid6_sid_pid")
    if not isinstance(existing, dict):
        raise DedupPidError("PID-001 metrics 缺少 gid6_sid_pid")
    renamed = dict(basic)
    renamed["distinct_pid_count"] = renamed.pop("distinct_sid_count")
    renamed["distinct_pid_ratio"] = renamed.pop("distinct_sid_ratio")
    renamed["singleton_pid_count"] = renamed.pop("singleton_sid_count")
    differences = {
        key: {"recomputed": renamed.get(key), "existing": existing.get(key)}
        for key in sorted(set(renamed) | set(existing))
        if renamed.get(key) != existing.get(key)
    }
    if differences:
        raise DedupPidError(
            "base PID 重算指标与 PID-001 不一致："
            + json.dumps(dict(list(differences.items())[:5]), ensure_ascii=False)
        )

    summary = _base_metrics_from_basic(basic)
    if expected_baseline is not None:
        baseline_differences = {
            key: {"actual": summary.get(key), "expected": expected}
            for key, expected in expected_baseline.items()
            if summary.get(key) != expected
        }
        if baseline_differences:
            raise DedupPidError(
                "PID-001 关键基线不一致："
                + json.dumps(baseline_differences, ensure_ascii=False)
            )
    return summary


def assign_dedup_codes(
    base_pid_codes: np.ndarray,
    poi_ids: Sequence[str],
    *,
    dedup_capacity: int,
) -> DedupAssignment:
    """Assign zero-based codes by POI ID within every colliding base PID."""

    base_pid_codes = np.asarray(base_pid_codes)
    if base_pid_codes.ndim != 2 or base_pid_codes.shape[1] != 9:
        raise DedupPidError("base PID shape 必须是 [N,9]")
    if base_pid_codes.dtype.kind not in {"i", "u"}:
        raise DedupPidError("base PID 必须使用整数 dtype")
    if len(poi_ids) != base_pid_codes.shape[0]:
        raise DedupPidError("POI ID 行数与 base PID 行数不一致")
    if len(set(poi_ids)) != len(poi_ids):
        raise DedupPidError("POI ID 存在重复")
    if (
        isinstance(dedup_capacity, bool)
        or not isinstance(dedup_capacity, int)
        or dedup_capacity <= 0
        or dedup_capacity > np.iinfo(np.int16).max + 1
    ):
        raise DedupPidError("dedup_capacity 必须位于 [1,32768]")

    _, inverse, bucket_sizes = np.unique(
        base_pid_codes, axis=0, return_inverse=True, return_counts=True
    )
    bucket_sizes_by_row = bucket_sizes[inverse].astype(np.int32, copy=False)
    max_bucket_size = int(bucket_sizes.max())
    if max_bucket_size > dedup_capacity:
        raise DedupPidError(
            f"最大 base PID 桶为 {max_bucket_size}，超过 Dedup 容量 "
            f"{dedup_capacity}"
        )

    dedup_codes = np.full(base_pid_codes.shape[0], -1, dtype=np.int16)
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
        if int(ordered_codes.max()) >= dedup_capacity:
            raise DedupPidError("分配出的 Dedup Code 超过预留容量")
        dedup_codes[ordered_rows] = ordered_codes.astype(np.int16)

    singleton_mask = bucket_sizes_by_row == 1
    if np.any(dedup_codes[singleton_mask] != -1):
        raise DedupPidError("单例 base PID 被错误分配 Dedup Code")
    if np.any(dedup_codes[~singleton_mask] < 0):
        raise DedupPidError("碰撞 base PID 存在未分配 Dedup Code 的 POI")

    collision_bucket_count = int((bucket_sizes > 1).sum())
    singleton_poi_count = int(singleton_mask.sum())
    dedup_poi_count = int((~singleton_mask).sum())
    max_dedup_code = int(dedup_codes.max()) if dedup_poi_count else -1
    return DedupAssignment(
        dedup_codes=dedup_codes,
        bucket_sizes_by_row=bucket_sizes_by_row,
        base_pid_distinct_count=len(bucket_sizes),
        collision_bucket_count=collision_bucket_count,
        singleton_poi_count=singleton_poi_count,
        dedup_poi_count=dedup_poi_count,
        max_base_bucket_size=max_bucket_size,
        max_dedup_code=max_dedup_code,
    )


def compose_final_pid(
    base_pid_codes: np.ndarray, dedup_codes: np.ndarray
) -> np.ndarray:
    """Store variable-length final PIDs in a fixed-width matrix with -1 sentinel."""

    base_pid_codes = np.asarray(base_pid_codes)
    dedup_codes = np.asarray(dedup_codes)
    if base_pid_codes.ndim != 2 or base_pid_codes.shape[1] != 9:
        raise DedupPidError("base PID shape 必须是 [N,9]")
    if dedup_codes.shape != (base_pid_codes.shape[0],):
        raise DedupPidError("dedup_codes shape 必须是 [N]")
    if dedup_codes.dtype != np.int16:
        raise DedupPidError("dedup_codes dtype 必须是 int16")
    final_codes = np.empty((base_pid_codes.shape[0], 10), dtype=np.int32)
    final_codes[:, :9] = base_pid_codes
    final_codes[:, 9] = dedup_codes
    return final_codes


def base_pid_key(tokens: Sequence[int]) -> str:
    """Serialize one nine-token base PID without Python container formatting."""

    if len(tokens) != 9:
        raise DedupPidError("base PID key 要求九个 Token")
    gid = "-".join(str(int(value)) for value in tokens[:6])
    sid = "-".join(str(int(value)) for value in tokens[6:])
    return f"{gid}|{sid}"


def final_pid_key(tokens: Sequence[int], dedup_code: int) -> str:
    """Serialize a variable-length final PID into a stable reversible key."""

    value = base_pid_key(tokens)
    return value if dedup_code == -1 else f"{value}|d{dedup_code}"


def _mapping_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("poi_id", pa.string(), nullable=False),
            *[
                pa.field(f"g{index}", pa.int32(), nullable=False)
                for index in range(1, 7)
            ],
            *[
                pa.field(f"s{index}", pa.int32(), nullable=False)
                for index in range(1, 4)
            ],
            pa.field("base_pid_key", pa.string(), nullable=False),
            pa.field("base_pid_bucket_size", pa.int32(), nullable=False),
            pa.field("requires_dedup", pa.bool_(), nullable=False),
            pa.field("dedup_code", pa.int16(), nullable=True),
            pa.field("final_pid_length", pa.int8(), nullable=False),
            pa.field("final_pid_key", pa.string(), nullable=False),
        ]
    )


def write_mapping_parquet(
    path: Path,
    poi_ids: Sequence[str],
    base_pid_codes: np.ndarray,
    assignment: DedupAssignment,
    *,
    chunk_rows: int = 100_000,
) -> None:
    """Write the aligned POI-to-PID mapping in deterministic row groups."""

    if chunk_rows <= 0:
        raise DedupPidError("chunk_rows 必须大于 0")
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
            codes = np.asarray(base_pid_codes[start:end], dtype=np.int32)
            dedup = assignment.dedup_codes[start:end]
            keys = [base_pid_key(row) for row in codes]
            final_keys = [
                key if int(code) == -1 else f"{key}|d{int(code)}"
                for key, code in zip(keys, dedup)
            ]
            arrays: list[pa.Array] = [
                pa.array(poi_ids[start:end], type=pa.string()),
                *[
                    pa.array(codes[:, column], type=pa.int32())
                    for column in range(9)
                ],
                pa.array(keys, type=pa.string()),
                pa.array(
                    assignment.bucket_sizes_by_row[start:end], type=pa.int32()
                ),
                pa.array(dedup >= 0, type=pa.bool_()),
                pa.array(
                    [None if int(code) == -1 else int(code) for code in dedup],
                    type=pa.int16(),
                ),
                pa.array(
                    np.where(dedup >= 0, 10, 9).astype(np.int8), type=pa.int8()
                ),
                pa.array(final_keys, type=pa.string()),
            ]
            table = pa.Table.from_arrays(arrays, schema=schema)
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
            raise DedupPidError(f"输出路径不是目录：{output_dir}")
        unexpected = {
            path.name for path in output_dir.iterdir()
        } - set(OUTPUT_FILENAMES)
        if unexpected:
            raise DedupPidError(
                "输出目录包含 PID-002 范围外文件，不会覆盖："
                + ", ".join(sorted(unexpected))
            )
    else:
        output_dir.mkdir(parents=True)
        created = True
    return created


def _validate_assignment_continuity(
    base_pid_codes: np.ndarray,
    poi_ids: Sequence[str],
    assignment: DedupAssignment,
) -> None:
    colliding_rows = np.flatnonzero(assignment.bucket_sizes_by_row > 1)
    ordered = sorted(
        (int(row) for row in colliding_rows),
        key=lambda row: (
            tuple(int(value) for value in base_pid_codes[row]),
            poi_ids[row],
        ),
    )
    previous_pid: tuple[int, ...] | None = None
    expected_code = 0
    for row in ordered:
        current_pid = tuple(int(value) for value in base_pid_codes[row])
        if current_pid != previous_pid:
            previous_pid = current_pid
            expected_code = 0
        if int(assignment.dedup_codes[row]) != expected_code:
            raise DedupPidError("碰撞桶 Dedup Code 不连续或未按 poi_id 排序")
        expected_code += 1


def build_dedup_pid(
    pid_manifest_path: Path,
    output_dir: Path,
    *,
    dedup_capacity: int = 512,
    expected_baseline: Mapping[str, int] | None = EXPECTED_PID001_BASELINE,
    progress: Callable[[str], None] | None = None,
) -> DedupPidResult:
    """Build deterministic unique final PIDs and five PID-002 artifacts."""

    started = time.perf_counter()
    pid_manifest_path = pid_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if progress is not None:
        progress("读取 base PID、PID-001 metrics 与 POI ID，并校验输入契约")
    base = load_base_pid_input(pid_manifest_path)

    if progress is not None:
        progress("重算 base PID 指标并与 PID-001 关键基线逐项核对")
    basic, _, _, _ = compute_basic_metrics(base.codes)
    baseline = validate_base_metrics(
        basic, base.metrics, expected_baseline=expected_baseline
    )

    if progress is not None:
        progress("按 base PID 分桶，并在碰撞桶内按 poi_id 字典序分配 Dedup Code")
    assignment = assign_dedup_codes(
        base.codes, base.poi_ids, dedup_capacity=dedup_capacity
    )
    if (
        assignment.base_pid_distinct_count
        != baseline["base_pid_distinct_count"]
        or assignment.collision_bucket_count
        != baseline["base_pid_collision_bucket_count"]
        or assignment.singleton_poi_count != baseline["singleton_poi_count"]
        or assignment.dedup_poi_count
        != baseline["base_pid_colliding_poi_count"]
        or assignment.max_base_bucket_size != baseline["max_base_bucket_size"]
    ):
        raise DedupPidError("Dedup 分桶统计与重算 base PID 指标不一致")
    _validate_assignment_continuity(base.codes, base.poi_ids, assignment)
    final_codes = compose_final_pid(base.codes, assignment.dedup_codes)

    if not np.array_equal(final_codes[:, :9], base.codes):
        raise DedupPidError("删除最终 Dedup 列后无法恢复原始 base PID")
    singleton_mask = assignment.bucket_sizes_by_row == 1
    if np.any(final_codes[singleton_mask, 9] != -1):
        raise DedupPidError("单例 POI 的最终第十列必须为 -1")
    if progress is not None:
        progress("直接验证 final_pid_codes 全局唯一性")
    final_distinct_count = int(np.unique(final_codes, axis=0).shape[0])
    poi_count = int(final_codes.shape[0])
    if final_distinct_count != poi_count:
        raise DedupPidError(
            f"final PID 不唯一：{final_distinct_count:,}/{poi_count:,}"
        )

    created_output_dir = _validate_output_directory(output_dir)
    suffix = f".{os.getpid()}.tmp"
    temporary_paths = {
        filename: output_dir / f".{filename}{suffix}"
        for filename in OUTPUT_FILENAMES
    }
    built_at = datetime.now(timezone.utc).isoformat()
    try:
        if progress is not None:
            progress("写入 Dedup 数组与固定 schema 的 Parquet 映射")
        _write_npy(
            temporary_paths["dedup_codes.npy"], assignment.dedup_codes
        )
        _write_npy(temporary_paths["final_pid_codes.npy"], final_codes)
        write_mapping_parquet(
            temporary_paths["poi_pid_mapping.parquet"],
            base.poi_ids,
            base.codes,
            assignment,
        )

        artifact_hashes = {
            name: sha256_file(temporary_paths[name])
            for name in (
                "dedup_codes.npy",
                "final_pid_codes.npy",
                "poi_pid_mapping.parquet",
            )
        }
        metrics = {
            "schema_version": FINAL_PID_METRICS_SCHEMA_VERSION,
            "status": "completed",
            "built_at": built_at,
            "build_seconds": time.perf_counter() - started,
            "poi_count": poi_count,
            "base_pid_distinct_count": assignment.base_pid_distinct_count,
            "base_pid_colliding_poi_count": assignment.dedup_poi_count,
            "base_pid_collision_bucket_count": assignment.collision_bucket_count,
            "singleton_poi_count": assignment.singleton_poi_count,
            "dedup_poi_count": assignment.dedup_poi_count,
            "final_pid_distinct_count": final_distinct_count,
            "final_pid_unique_ratio": final_distinct_count / poi_count,
            "max_base_bucket_size": assignment.max_base_bucket_size,
            "max_dedup_code": assignment.max_dedup_code,
            "dedup_token_capacity": dedup_capacity,
            "validation": {
                "poi_ids_unique": True,
                "poi_id_rows_match_base_pid": True,
                "base_pid_metrics_match_pid001": True,
                "dedup_codes_continuous_per_collision_bucket": True,
                "singleton_sentinel_valid": True,
                "base_pid_recovery_valid": True,
                "final_pid_globally_unique": True,
            },
        }
        _write_json(temporary_paths["metrics.json"], metrics)
        artifact_hashes["metrics.json"] = sha256_file(
            temporary_paths["metrics.json"]
        )

        manifest = {
            "schema_version": FINAL_PID_SCHEMA_VERSION,
            "status": "completed",
            "built_at": built_at,
            "poi_count": poi_count,
            "poi_ids": {
                "path": str(base.poi_ids_path),
                "rows": len(base.poi_ids),
                "sha256": base.poi_ids_sha256,
                "unique": True,
            },
            "base_pid_source": {
                "manifest": str(base.manifest_path),
                "manifest_sha256": sha256_file(base.manifest_path),
                "metrics": str(base.metrics_path),
                "metrics_sha256": sha256_file(base.metrics_path),
                "pid_codes": str(base.codes_path),
                "pid_codes_sha256": base.codes_sha256,
                "shape": [int(value) for value in base.codes.shape],
                "dtype": str(base.codes.dtype),
                "order": "gid_sid",
                "token_order": list(EXPECTED_BASE_PID_TOKEN_ORDER),
            },
            "dedup_assignment": {
                "scope": "local_within_each_colliding_base_pid",
                "row_order": "poi_id_lexicographic_ascending",
                "code_start": 0,
                "codes_are_contiguous": True,
                "codes_may_repeat_across_base_pid_buckets": True,
                "singleton_sentinel": -1,
                "singleton_sentinel_is_token": False,
                "dedup_token_capacity": dedup_capacity,
                "actual_max_dedup_code": assignment.max_dedup_code,
                "incremental_update": "not_implemented",
                "initial_mapping_is_persisted": True,
            },
            "final_pid_definition": {
                "base_order": list(EXPECTED_BASE_PID_TOKEN_ORDER),
                "colliding_pid_order": [
                    *EXPECTED_BASE_PID_TOKEN_ORDER,
                    "D",
                ],
                "singleton_length": 9,
                "colliding_length": 10,
                "fixed_width_storage_shape": [poi_count, 10],
                "fixed_width_singleton_value": -1,
                "key_format": {
                    "singleton": "g1-g2-g3-g4-g5-g6|s1-s2-s3",
                    "colliding": (
                        "g1-g2-g3-g4-g5-g6|s1-s2-s3|d<dedup_code>"
                    ),
                },
            },
            "outputs": {
                "dedup_codes.npy": {
                    "shape": [poi_count],
                    "dtype": "int16",
                    "sha256": artifact_hashes["dedup_codes.npy"],
                },
                "final_pid_codes.npy": {
                    "shape": [poi_count, 10],
                    "dtype": "int32",
                    "sha256": artifact_hashes["final_pid_codes.npy"],
                },
                "poi_pid_mapping.parquet": {
                    "shape": [poi_count, len(_mapping_schema())],
                    "dtype": str(_mapping_schema()),
                    "sha256": artifact_hashes["poi_pid_mapping.parquet"],
                    "row_order": "same_as_poi_ids_and_base_pid",
                },
                "metrics.json": {
                    "shape": None,
                    "dtype": "json",
                    "sha256": artifact_hashes["metrics.json"],
                },
            },
        }
        canonical_payload = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        manifest["manifest_payload_sha256"] = hashlib.sha256(
            canonical_payload
        ).hexdigest()
        manifest["manifest_payload_hash_definition"] = (
            "sha256 of canonical JSON before adding the two "
            "manifest_payload_hash fields"
        )
        _write_json(temporary_paths["final_pid_manifest.json"], manifest)

        for filename in OUTPUT_FILENAMES:
            os.replace(temporary_paths[filename], output_dir / filename)
    except Exception:
        if created_output_dir and output_dir.is_dir() and not any(
            output_dir.iterdir()
        ):
            output_dir.rmdir()
        raise
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    return DedupPidResult(
        manifest=manifest,
        metrics=metrics,
        mapping_sha256=artifact_hashes["poi_pid_mapping.parquet"],
    )

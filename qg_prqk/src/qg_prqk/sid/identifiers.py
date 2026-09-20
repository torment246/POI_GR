"""Build deterministic unique identifiers from frozen SID mappings."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, utc_now
from qg_prqk.sid.geo import GEOHASH_ALPHABET


FINAL_IDENTIFIER_SCHEMA_VERSION = "qg-prqk-final-identifier-v1"
VARIANTS = ("a4_gid_parent", "a4_nogid")
Variant = Literal["a4_gid_parent", "a4_nogid"]
OUTPUT_FILENAMES = (
    "base_identifier_codes.npy",
    "dedup_codes.npy",
    "requires_dedup.npy",
    "poi_final_id_mapping.parquet",
    "metrics.json",
    "manifest.json",
    "_SUCCESS",
)


class FinalIdentifierError(ValueError):
    """Raised when a QG-PRQK Final-ID contract is violated."""


@dataclass(frozen=True)
class DedupAssignment:
    """Deterministic optional Dedup codes aligned to POI rows."""

    codes: np.ndarray
    bucket_sizes_by_row: np.ndarray
    base_distinct_count: int
    collision_bucket_count: int
    singleton_poi_count: int
    colliding_poi_count: int
    max_bucket_size: int


@dataclass(frozen=True)
class FinalIdentifierResult:
    """Metadata for one completed Final-ID directory."""

    output_dir: Path
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    manifest_sha256: str


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FinalIdentifierError(f"{name}不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalIdentifierError(f"{name}不是合法 JSON：{path}") from error
    if not isinstance(payload, dict):
        raise FinalIdentifierError(f"{name}必须是 JSON object：{path}")
    return payload


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


def _load_sid_mapping(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.is_file():
        raise FinalIdentifierError(f"SID mapping 不存在：{path}")
    parquet_file = pq.ParquetFile(path)
    required = ("poi_row_index", "poi_id", "s1", "s2", "s3")
    missing = set(required) - set(parquet_file.schema_arrow.names)
    if missing:
        raise FinalIdentifierError(
            "SID mapping 缺少字段：" + ", ".join(sorted(missing))
        )
    row_count = parquet_file.metadata.num_rows
    if row_count <= 0:
        raise FinalIdentifierError("SID mapping 不能为空")
    poi_ids: list[str] = []
    sid = np.empty((row_count, 3), dtype=np.int32)
    offset = 0
    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=list(required))
        data = table.to_pydict()
        count = table.num_rows
        end = offset + count
        expected_rows = np.arange(offset, end, dtype=np.int64)
        observed_rows = np.asarray(data["poi_row_index"], dtype=np.int64)
        if not np.array_equal(observed_rows, expected_rows):
            raise FinalIdentifierError("SID mapping 的 poi_row_index 必须连续且同行同序")
        group_poi_ids = data["poi_id"]
        if any(not isinstance(value, str) or not value for value in group_poi_ids):
            raise FinalIdentifierError("SID mapping 包含非法 poi_id")
        poi_ids.extend(group_poi_ids)
        sid[offset:end] = np.column_stack(
            [np.asarray(data[name], dtype=np.int32) for name in ("s1", "s2", "s3")]
        )
        offset = end
    if offset != row_count or len(set(poi_ids)) != row_count:
        raise FinalIdentifierError("SID mapping 行数不完整或 poi_id 不唯一")
    if np.any(sid < 0) or np.any(sid >= 512):
        raise FinalIdentifierError("SID code 超出冻结容量 [0,512)")
    return poi_ids, sid


def assign_optional_dedup(
    base_codes: np.ndarray,
    poi_ids: Sequence[str],
) -> DedupAssignment:
    """Assign ``D0..`` only inside colliding base-ID buckets."""

    codes = np.asarray(base_codes)
    if codes.ndim != 2 or codes.shape[0] == 0:
        raise FinalIdentifierError("base identifier 必须是非空二维数组")
    if codes.dtype.kind not in "iu" or np.any(codes < 0):
        raise FinalIdentifierError("base identifier 必须是非负整数")
    if len(poi_ids) != len(codes):
        raise FinalIdentifierError("POI ID 与 base identifier 行数不一致")
    if len(set(poi_ids)) != len(poi_ids):
        raise FinalIdentifierError("POI ID 必须唯一")

    _, inverse, bucket_sizes = np.unique(
        codes,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    bucket_sizes_by_row = bucket_sizes[inverse].astype(np.int32, copy=False)
    dedup_codes = np.full(len(codes), -1, dtype=np.int32)
    colliding_rows = np.flatnonzero(bucket_sizes_by_row > 1)
    ordered_rows = sorted(
        (int(row) for row in colliding_rows),
        key=lambda row: (int(inverse[row]), poi_ids[row]),
    )
    previous_bucket = -1
    next_code = 0
    for row in ordered_rows:
        bucket = int(inverse[row])
        if bucket != previous_bucket:
            previous_bucket = bucket
            next_code = 0
        dedup_codes[row] = next_code
        next_code += 1

    final_codes = np.column_stack((codes, dedup_codes))
    if len(np.unique(final_codes, axis=0)) != len(codes):
        raise FinalIdentifierError("追加 Dedup 后 Final ID 仍不唯一")
    return DedupAssignment(
        codes=dedup_codes,
        bucket_sizes_by_row=bucket_sizes_by_row,
        base_distinct_count=int(len(bucket_sizes)),
        collision_bucket_count=int(np.count_nonzero(bucket_sizes > 1)),
        singleton_poi_count=int(np.count_nonzero(bucket_sizes_by_row == 1)),
        colliding_poi_count=int(np.count_nonzero(bucket_sizes_by_row > 1)),
        max_bucket_size=int(bucket_sizes.max()),
    )


def identifier_content(
    base_codes: Sequence[int],
    dedup_code: int,
    *,
    variant: Variant,
) -> str:
    """Format one Final ID using ordinary atomic-token strings."""

    values = tuple(int(value) for value in base_codes)
    if variant == "a4_gid_parent":
        if len(values) != 9:
            raise FinalIdentifierError("GID-parent base identifier 必须是 GID6+SID3")
        gid, sid = values[:6], values[6:]
        if any(not 0 <= value < 32 for value in gid):
            raise FinalIdentifierError("GID code 超出 [0,32)")
        prefix = "".join(f"<G_{GEOHASH_ALPHABET[value]}>" for value in gid)
    elif variant == "a4_nogid":
        if len(values) != 3:
            raise FinalIdentifierError("NoGID base identifier 必须只有 SID3")
        sid = values
        prefix = ""
    else:
        raise FinalIdentifierError(f"未知 Final-ID variant：{variant}")
    if any(not 0 <= value < 512 for value in sid):
        raise FinalIdentifierError("SID code 超出 [0,512)")
    content = prefix + "".join(
        f"<S{level}_{value}>" for level, value in enumerate(sid, start=1)
    )
    if dedup_code >= 0:
        content += f"<D_{dedup_code}>"
    elif dedup_code != -1:
        raise FinalIdentifierError("Dedup code 只能是 -1 或非负整数")
    return content


def identifier_key(
    base_codes: Sequence[int],
    dedup_code: int,
    *,
    variant: Variant,
) -> str:
    """Serialize one Final ID for stable joins and manifests."""

    values = tuple(int(value) for value in base_codes)
    if variant == "a4_gid_parent":
        if len(values) != 9:
            raise FinalIdentifierError("GID-parent key 必须是九个 base code")
        gid = "-".join(str(value) for value in values[:6])
        sid = "-".join(str(value) for value in values[6:])
        base = f"{gid}|{sid}"
    elif variant == "a4_nogid":
        if len(values) != 3:
            raise FinalIdentifierError("NoGID key 必须是三个 SID code")
        base = "-".join(str(value) for value in values)
    else:
        raise FinalIdentifierError(f"未知 Final-ID variant：{variant}")
    return base if dedup_code == -1 else f"{base}|d{dedup_code}"


def _mapping_schema(variant: Variant) -> pa.Schema:
    fields = [
        pa.field("poi_row_index", pa.int64(), nullable=False),
        pa.field("poi_id", pa.string(), nullable=False),
    ]
    if variant == "a4_gid_parent":
        fields.extend(
            pa.field(f"g{level}", pa.int32(), nullable=False)
            for level in range(1, 7)
        )
    fields.extend(
        pa.field(f"s{level}", pa.int32(), nullable=False)
        for level in range(1, 4)
    )
    fields.extend(
        [
            pa.field("base_id_key", pa.string(), nullable=False),
            pa.field("base_id_bucket_size", pa.int32(), nullable=False),
            pa.field("requires_dedup", pa.bool_(), nullable=False),
            pa.field("dedup_code", pa.int32(), nullable=True),
            pa.field("final_id_length", pa.int32(), nullable=False),
            pa.field("final_id_key", pa.string(), nullable=False),
        ]
    )
    return pa.schema(fields)


def _write_mapping(
    path: Path,
    poi_ids: Sequence[str],
    base_codes: np.ndarray,
    assignment: DedupAssignment,
    *,
    variant: Variant,
    chunk_rows: int,
) -> None:
    if chunk_rows <= 0:
        raise FinalIdentifierError("chunk_rows 必须大于 0")
    schema = _mapping_schema(variant)
    writer = pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=True,
    )
    base_length = base_codes.shape[1]
    try:
        for start in range(0, len(poi_ids), chunk_rows):
            end = min(start + chunk_rows, len(poi_ids))
            group = np.asarray(base_codes[start:end], dtype=np.int32)
            dedup = assignment.codes[start:end]
            base_keys = [
                identifier_key(row, -1, variant=variant) for row in group
            ]
            final_keys = [
                identifier_key(row, int(code), variant=variant)
                for row, code in zip(group, dedup, strict=True)
            ]
            columns: list[pa.Array] = [
                pa.array(np.arange(start, end, dtype=np.int64), type=pa.int64()),
                pa.array(poi_ids[start:end], type=pa.string()),
            ]
            columns.extend(
                pa.array(group[:, index], type=pa.int32())
                for index in range(group.shape[1])
            )
            columns.extend(
                [
                    pa.array(base_keys, type=pa.string()),
                    pa.array(
                        assignment.bucket_sizes_by_row[start:end], type=pa.int32()
                    ),
                    pa.array(dedup >= 0, type=pa.bool_()),
                    pa.array(
                        [None if value < 0 else int(value) for value in dedup],
                        type=pa.int32(),
                    ),
                    pa.array(base_length + (dedup >= 0), type=pa.int32()),
                    pa.array(final_keys, type=pa.string()),
                ]
            )
            writer.write_table(pa.Table.from_arrays(columns, schema=schema))
    finally:
        writer.close()


def build_final_identifiers(
    *,
    variant: Variant,
    sid_mapping_path: Path,
    sid_manifest_path: Path,
    output_dir: Path,
    gid_codes_path: Path | None = None,
    gid_manifest_path: Path | None = None,
    chunk_rows: int = 100_000,
    progress: Callable[[str], None] | None = None,
) -> FinalIdentifierResult:
    """Build one immutable unique Final-ID mapping from a frozen full SID."""

    if variant not in VARIANTS:
        raise FinalIdentifierError(f"variant 必须是 {VARIANTS}")
    sid_mapping_path = sid_mapping_path.resolve()
    sid_manifest_path = sid_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FinalIdentifierError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    sid_manifest = _load_json(sid_manifest_path, "SID manifest")
    if sid_manifest.get("status") != "completed":
        raise FinalIdentifierError("SID manifest 状态不是 completed")
    if not (sid_manifest_path.parent / "_SUCCESS").is_file():
        raise FinalIdentifierError("SID 来源目录缺少 _SUCCESS")
    if progress is not None:
        progress(f"读取 {variant} 冻结 SID mapping")
    poi_ids, sid = _load_sid_mapping(sid_mapping_path)

    gid_source: dict[str, Any] | None = None
    if variant == "a4_gid_parent":
        if gid_codes_path is None or gid_manifest_path is None:
            raise FinalIdentifierError("GID-parent 必须显式提供 GID codes 与 manifest")
        resolved_gid_codes = gid_codes_path.resolve()
        resolved_gid_manifest = gid_manifest_path.resolve()
        gid_manifest = _load_json(resolved_gid_manifest, "GID manifest")
        if gid_manifest.get("status") != "completed":
            raise FinalIdentifierError("GID manifest 状态不是 completed")
        gid = np.load(resolved_gid_codes, mmap_mode="r", allow_pickle=False)
        if gid.shape != (len(poi_ids), 6) or gid.dtype.kind not in "iu":
            raise FinalIdentifierError("GID codes 必须是同行同序的 [N,6] 整数数组")
        if np.any(gid < 0) or np.any(gid >= 32):
            raise FinalIdentifierError("GID codes 超出 [0,32)")
        base_codes = np.column_stack((gid, sid)).astype(np.int32, copy=False)
        gid_source = {
            "codes_path": str(resolved_gid_codes),
            "codes_sha256": sha256_file(resolved_gid_codes),
            "manifest_path": str(resolved_gid_manifest),
            "manifest_sha256": sha256_file(resolved_gid_manifest),
        }
    else:
        if gid_codes_path is not None or gid_manifest_path is not None:
            raise FinalIdentifierError("NoGID 分支禁止读取或登记目标 POI GID")
        base_codes = sid

    if progress is not None:
        progress(f"{variant}：在完整 base identifier 内分配可选 Dedup")
    assignment = assign_optional_dedup(base_codes, poi_ids)
    dedup_capacity = assignment.max_bucket_size
    metrics = {
        "poi_count": len(poi_ids),
        "base_length": int(base_codes.shape[1]),
        "base_distinct_count": assignment.base_distinct_count,
        "base_distinct_ratio": assignment.base_distinct_count / len(poi_ids),
        "collision_excess": len(poi_ids) - assignment.base_distinct_count,
        "collision_bucket_count": assignment.collision_bucket_count,
        "singleton_poi_count": assignment.singleton_poi_count,
        "singleton_poi_ratio": assignment.singleton_poi_count / len(poi_ids),
        "colliding_poi_count": assignment.colliding_poi_count,
        "colliding_poi_ratio": assignment.colliding_poi_count / len(poi_ids),
        "max_base_bucket_size": assignment.max_bucket_size,
        "dedup_token_capacity": dedup_capacity,
        "max_dedup_code": dedup_capacity - 1,
        "final_distinct_count": len(poi_ids),
        "final_unique_ratio": 1.0,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}.building-")
    )
    try:
        np.save(
            temporary_dir / "base_identifier_codes.npy",
            np.asarray(base_codes, dtype=np.int32),
            allow_pickle=False,
        )
        np.save(
            temporary_dir / "dedup_codes.npy",
            assignment.codes,
            allow_pickle=False,
        )
        np.save(
            temporary_dir / "requires_dedup.npy",
            assignment.codes >= 0,
            allow_pickle=False,
        )
        _write_mapping(
            temporary_dir / "poi_final_id_mapping.parquet",
            poi_ids,
            base_codes,
            assignment,
            variant=variant,
            chunk_rows=chunk_rows,
        )
        _write_json(temporary_dir / "metrics.json", metrics)
        artifacts = {
            filename: {
                "bytes": (temporary_dir / filename).stat().st_size,
                "sha256": sha256_file(temporary_dir / filename),
            }
            for filename in OUTPUT_FILENAMES
            if filename not in {"manifest.json", "_SUCCESS"}
        }
        manifest = {
            "schema_version": FINAL_IDENTIFIER_SCHEMA_VERSION,
            "status": "completed",
            "built_at": utc_now(),
            "variant": variant,
            "identifier_contract": {
                "base_token_order": (
                    [f"G{level}" for level in range(1, 7)]
                    + [f"S{level}" for level in range(1, 4)]
                    if variant == "a4_gid_parent"
                    else [f"S{level}" for level in range(1, 4)]
                ),
                "dedup_position": "last_if_base_identifier_collides",
                "dedup_assignment": "poi_id_lexicographic_zero_based_within_base_bucket",
                "singletons_have_dedup": False,
                "final_unique": True,
            },
            "sid_source": {
                "mapping_path": str(sid_mapping_path),
                "mapping_sha256": sha256_file(sid_mapping_path),
                "manifest_path": str(sid_manifest_path),
                "manifest_sha256": sha256_file(sid_manifest_path),
            },
            "gid_source": gid_source,
            "metrics": metrics,
            "artifacts": artifacts,
        }
        _write_json(temporary_dir / "manifest.json", manifest)
        (temporary_dir / "_SUCCESS").touch()
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    manifest_path = output_dir / "manifest.json"
    return FinalIdentifierResult(
        output_dir=output_dir,
        manifest=manifest,
        metrics=metrics,
        manifest_sha256=sha256_file(manifest_path),
    )

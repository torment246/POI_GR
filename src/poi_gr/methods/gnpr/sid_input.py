"""Prepare bounded sparse inputs for the GNPR-SID baseline."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_USER_HASH_BUCKETS = 8192
USER_HASH_PERSON = b"gnpr-user-v1"
TIME_FEATURE_DIM = 24


class GnprSidInputError(ValueError):
    """Raised when a raw GNPR POI feature violates the SID input contract."""


@dataclass(frozen=True)
class GnprFeatureDimensions:
    """Feature-block dimensions and offsets for one GNPR input vector."""

    category_dim: int
    region_dim: int
    time_dim: int
    user_hash_dim: int

    @property
    def region_offset(self) -> int:
        return self.category_dim

    @property
    def time_offset(self) -> int:
        return self.category_dim + self.region_dim

    @property
    def user_hash_offset(self) -> int:
        return self.time_offset + self.time_dim

    @property
    def total_dim(self) -> int:
        return self.user_hash_offset + self.user_hash_dim


@dataclass(frozen=True)
class GnprSidInput:
    """One behavior-supported POI represented by bounded sparse indices."""

    poi_id: str
    category_index: int
    region_index: int
    top_visit_hours: tuple[int, ...]
    user_hash_indices: tuple[int, ...]
    interaction_count: int


def build_feature_dimensions(
    *,
    category_count: int,
    region_count: int,
    user_hash_buckets: int = DEFAULT_USER_HASH_BUCKETS,
) -> GnprFeatureDimensions:
    """Build and validate the four GNPR feature-block dimensions."""

    if category_count <= 0:
        raise GnprSidInputError("category_count 必须大于 0")
    if region_count <= 0:
        raise GnprSidInputError("region_count 必须大于 0")
    if user_hash_buckets <= 0:
        raise GnprSidInputError("user_hash_buckets 必须大于 0")
    return GnprFeatureDimensions(
        category_dim=category_count,
        region_dim=region_count,
        time_dim=TIME_FEATURE_DIM,
        user_hash_dim=user_hash_buckets,
    )


def stable_user_hash(
    passenger_id: Any,
    *,
    buckets: int = DEFAULT_USER_HASH_BUCKETS,
) -> int:
    """Map one passenger ID to a stable, process-independent hash bucket."""

    if buckets <= 0:
        raise GnprSidInputError("用户哈希桶数必须大于 0")
    text = "" if passenger_id is None else str(passenger_id).strip()
    if not text:
        raise GnprSidInputError("passenger_id 必须是非空字符串")
    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
        person=USER_HASH_PERSON,
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % buckets


def hash_top_visitors(
    passenger_ids: Sequence[Any],
    *,
    buckets: int = DEFAULT_USER_HASH_BUCKETS,
) -> tuple[int, ...]:
    """Hash a POI's visitor IDs into a deterministic deduplicated multi-hot set."""

    if isinstance(passenger_ids, (str, bytes)):
        raise GnprSidInputError("top_visitor_ids 必须是列表")
    return tuple(
        sorted({stable_user_hash(value, buckets=buckets) for value in passenger_ids})
    )


def prepare_gnpr_sid_input(
    record: Mapping[str, Any],
    *,
    dimensions: GnprFeatureDimensions,
) -> GnprSidInput | None:
    """Filter cold POIs and convert one raw GNPR row to bounded hash features."""

    try:
        interaction_count = int(record.get("interaction_count", 0) or 0)
    except (TypeError, ValueError) as error:
        raise GnprSidInputError("interaction_count 必须是整数") from error
    if interaction_count <= 0:
        return None

    poi_id = "" if record.get("poi_id") is None else str(record["poi_id"]).strip()
    if not poi_id:
        raise GnprSidInputError("poi_id 必须是非空字符串")
    try:
        category_index = int(record["category_index"])
        region_index = int(record["region_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise GnprSidInputError("category_index 和 region_index 必须是整数") from error
    if not 0 <= category_index < dimensions.category_dim:
        raise GnprSidInputError("category_index 超出类别词表范围")
    if not 0 <= region_index < dimensions.region_dim:
        raise GnprSidInputError("region_index 超出区域词表范围")

    raw_hours = record.get("top_visit_hours")
    raw_visitors = record.get("top_visitor_ids")
    if not isinstance(raw_hours, (list, tuple)):
        raise GnprSidInputError("top_visit_hours 必须是列表")
    if not isinstance(raw_visitors, (list, tuple)):
        raise GnprSidInputError("top_visitor_ids 必须是列表")
    try:
        hours = tuple(sorted({int(value) for value in raw_hours}))
    except (TypeError, ValueError) as error:
        raise GnprSidInputError("top_visit_hours 必须只包含整数") from error
    if not hours or any(value < 0 or value >= TIME_FEATURE_DIM for value in hours):
        raise GnprSidInputError("有交互 POI 的小时特征必须位于 [0, 23]")
    if not raw_visitors:
        raise GnprSidInputError("有交互 POI 必须包含访问用户")

    return GnprSidInput(
        poi_id=poi_id,
        category_index=category_index,
        region_index=region_index,
        top_visit_hours=hours,
        user_hash_indices=hash_top_visitors(
            raw_visitors,
            buckets=dimensions.user_hash_dim,
        ),
        interaction_count=interaction_count,
    )


def active_feature_indices(
    row: GnprSidInput,
    *,
    dimensions: GnprFeatureDimensions,
) -> tuple[int, ...]:
    """Return the active indices in the logical bounded multi-hot vector."""

    indices = [
        row.category_index,
        dimensions.region_offset + row.region_index,
    ]
    indices.extend(dimensions.time_offset + value for value in row.top_visit_hours)
    indices.extend(
        dimensions.user_hash_offset + value for value in row.user_hash_indices
    )
    return tuple(indices)


def load_gnpr_sid_input_manifest(
    input_dir: Path,
) -> tuple[GnprFeatureDimensions, dict[str, Any]]:
    """Validate a completed bounded-input artifact and return its dimensions."""

    manifest_path = input_dir / "manifest.json"
    success_path = input_dir / "_SUCCESS"
    parquet_dir = input_dir / "poi_sid_inputs.parquet"
    if not manifest_path.is_file() or not success_path.is_file():
        raise GnprSidInputError("GNPR SID 输入目录未完成或缺少 manifest")
    if not parquet_dir.is_dir():
        raise GnprSidInputError("GNPR SID 输入目录缺少 poi_sid_inputs.parquet")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GnprSidInputError("GNPR SID 输入 manifest 无法解析") from error
    if manifest.get("schema_version") != "gnpr-sid-input-v1":
        raise GnprSidInputError("GNPR SID 输入 schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise GnprSidInputError("GNPR SID 输入 manifest 尚未完成")
    if manifest.get("catalog_filter") != "interaction_count > 0":
        raise GnprSidInputError("GNPR SID 输入未声明无交互 POI 过滤规则")
    dimensions_payload = manifest.get("dimensions")
    if not isinstance(dimensions_payload, dict):
        raise GnprSidInputError("GNPR SID 输入 manifest 缺少 dimensions")
    dimensions = build_feature_dimensions(
        category_count=int(dimensions_payload.get("category", 0)),
        region_count=int(dimensions_payload.get("region", 0)),
        user_hash_buckets=int(dimensions_payload.get("user_hash", 0)),
    )
    if int(dimensions_payload.get("time", -1)) != dimensions.time_dim:
        raise GnprSidInputError("GNPR SID 输入时间维度必须是 24")
    if int(dimensions_payload.get("total", -1)) != dimensions.total_dim:
        raise GnprSidInputError("GNPR SID 输入总维度与各特征块不一致")
    return dimensions, manifest


def _load_prepared_row(
    record: Mapping[str, Any],
    *,
    dimensions: GnprFeatureDimensions,
) -> GnprSidInput:
    try:
        row = GnprSidInput(
            poi_id=str(record["poi_id"]).strip(),
            category_index=int(record["category_index"]),
            region_index=int(record["region_index"]),
            top_visit_hours=tuple(int(value) for value in record["top_visit_hours"]),
            user_hash_indices=tuple(
                int(value) for value in record["user_hash_indices"]
            ),
            interaction_count=int(record["interaction_count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GnprSidInputError("GNPR SID Parquet 行字段无效") from error
    if not row.poi_id or row.interaction_count <= 0:
        raise GnprSidInputError("GNPR SID Parquet 只能包含有交互的有效 POI")
    if not 0 <= row.category_index < dimensions.category_dim:
        raise GnprSidInputError("category_index 超出类别词表范围")
    if not 0 <= row.region_index < dimensions.region_dim:
        raise GnprSidInputError("region_index 超出区域词表范围")
    if (
        not row.top_visit_hours
        or tuple(sorted(set(row.top_visit_hours))) != row.top_visit_hours
        or any(value < 0 or value >= dimensions.time_dim for value in row.top_visit_hours)
    ):
        raise GnprSidInputError("top_visit_hours 必须是非空升序去重小时")
    if (
        not row.user_hash_indices
        or tuple(sorted(set(row.user_hash_indices))) != row.user_hash_indices
        or any(
            value < 0 or value >= dimensions.user_hash_dim
            for value in row.user_hash_indices
        )
    ):
        raise GnprSidInputError("user_hash_indices 必须是非空升序去重桶")
    return row


def iter_gnpr_sid_input_batches(
    input_dir: Path,
    *,
    batch_size: int,
    max_rows: int | None = None,
    shuffle_seed: int | None = None,
) -> tuple[GnprFeatureDimensions, Iterable[tuple[GnprSidInput, ...]]]:
    """Return dimensions and a re-iterable streaming Parquet batch source."""

    if batch_size <= 0:
        raise GnprSidInputError("batch_size 必须大于 0")
    if max_rows is not None and max_rows <= 0:
        raise GnprSidInputError("max_rows 必须大于 0")
    dimensions, _ = load_gnpr_sid_input_manifest(input_dir)
    files = tuple(sorted((input_dir / "poi_sid_inputs.parquet").glob("*.parquet")))
    if not files:
        raise GnprSidInputError("GNPR SID 输入没有 Parquet 分片")

    class BatchSource:
        def __iter__(self):
            import pyarrow.parquet as pq

            emitted = 0
            ordered_files = list(files)
            if shuffle_seed is not None:
                random.Random(shuffle_seed).shuffle(ordered_files)
            row_rng = random.Random(shuffle_seed)
            for path in ordered_files:
                parquet_file = pq.ParquetFile(path)
                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    remaining = None if max_rows is None else max_rows - emitted
                    if remaining is not None and remaining <= 0:
                        return
                    if remaining is not None and batch.num_rows > remaining:
                        batch = batch.slice(0, remaining)
                    rows = [
                        _load_prepared_row(record, dimensions=dimensions)
                        for record in batch.to_pylist()
                    ]
                    if shuffle_seed is not None:
                        row_rng.shuffle(rows)
                    emitted += len(rows)
                    if rows:
                        yield tuple(rows)

    return dimensions, BatchSource()

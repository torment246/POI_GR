"""Build map-search-adapted TIGER supervised fine-tuning data."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq

from poi_gr.pid.dedup import sha256_file
from poi_gr.pid.geohash import GEOHASH_ALPHABET, encode_geohash
from poi_gr.sft.data import (
    TimeSplit,
    discover_order_files,
    stable_sample_id,
)


SCHEMA_VERSION = "tiger-map-search-sft-data-v1"
SPECIAL_TOKENS_SCHEMA_VERSION = "tiger-map-search-special-tokens-v1"
TASK_NAME = "tiger_map_search_history_query_gid_poi_to_target"
BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")
OUTPUT_FILENAMES = (
    "train.jsonl",
    "valid.jsonl",
    "test.jsonl",
    "special_tokens.json",
    "manifest.json",
    "stats.json",
)
REQUIRED_TARGET_FIELDS = (
    "order_id",
    "searchid",
    "passenger_id",
    "query",
    "disp_lng",
    "disp_lat",
    "create_time",
    "source_dt",
    "poi_id",
    "history_length",
    "history_sequence",
)
REQUIRED_HISTORY_FIELDS = (
    "event_time",
    "order_id",
    "searchid",
    "query",
    "disp_lng",
    "disp_lat",
    "poi_id",
)
TIGER_MAPPING_COLUMNS = (
    "poi_id",
    "s1",
    "s2",
    "s3",
    "collision_code",
)
STRUCTURE_TOKENS = (
    "<USER_ID>",
    "</USER_ID>",
    "<HISTORY>",
    "</HISTORY>",
    "<EVENT>",
    "</EVENT>",
    "<USER_GID>",
    "</USER_GID>",
    "<QUERY>",
    "</QUERY>",
    "<POI_TIGER_ID>",
    "</POI_TIGER_ID>",
    "<CURRENT>",
    "</CURRENT>",
    "<TARGET_POI>",
    "</TARGET_POI>",
)


class TigerDataError(ValueError):
    """Raised when TIGER data violates the declared contract."""


@dataclass(frozen=True)
class HistoryWindow:
    """Closed local-date range used for historical interactions."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise TigerDataError("history_start 不能晚于 history_end")

    def contains(self, value: datetime) -> bool:
        """Return whether one normalized local timestamp is in the window."""

        return self.start <= value.date() <= self.end

    def to_manifest(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass(frozen=True)
class TigerIdLookup:
    """In-memory POI to four-level TIGER identifier lookup."""

    row_by_poi_id: dict[str, int]
    codes: np.ndarray
    poi_count: int
    token_capacities: tuple[int, int, int, int]


@dataclass(frozen=True)
class TigerDataResult:
    """Completed TIGER data artifacts."""

    manifest: dict[str, Any]
    stats: dict[str, Any]
    output_hashes: dict[str, str]


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise TigerDataError(f"{name} 不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerDataError(f"{name} JSON 解析失败：{path}") from error
    if not isinstance(payload, dict):
        raise TigerDataError(f"{name} 必须是 JSON object：{path}")
    return payload


def load_tiger_id_lookup(
    tiger_id_dir: Path,
    *,
    expected_base_codebook_sizes: Sequence[int] = (1024, 1024, 1024),
) -> tuple[TigerIdLookup, dict[str, Any], str, str]:
    """Load and verify the formal TIGER item identifier mapping."""

    if (
        len(expected_base_codebook_sizes) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in expected_base_codebook_sizes
        )
    ):
        raise TigerDataError(
            "expected_base_codebook_sizes 必须包含三个正整数"
        )
    expected_base_codebook_sizes = tuple(
        int(value) for value in expected_base_codebook_sizes
    )

    tiger_id_dir = tiger_id_dir.resolve()
    manifest_path = tiger_id_dir / "tiger_id_manifest.json"
    mapping_path = tiger_id_dir / "poi_tiger_id_mapping.parquet"
    manifest = _load_json_object(manifest_path, "TIGER identifier manifest")
    if manifest.get("schema_version") != "tiger-item-identifier-v1":
        raise TigerDataError(
            "TIGER identifier manifest schema_version 必须是 "
            "tiger-item-identifier-v1"
        )
    if manifest.get("status") != "completed":
        raise TigerDataError("TIGER identifier manifest 状态不是 completed")

    mapping_spec = manifest.get("mapping")
    if not isinstance(mapping_spec, dict):
        raise TigerDataError("TIGER identifier manifest 缺少 mapping")
    poi_count = mapping_spec.get("rows")
    if (
        isinstance(poi_count, bool)
        or not isinstance(poi_count, int)
        or poi_count <= 0
    ):
        raise TigerDataError("TIGER identifier mapping.rows 无效")
    mapping_sha256 = sha256_file(mapping_path)
    if mapping_spec.get("sha256") != mapping_sha256:
        raise TigerDataError("TIGER identifier 映射 SHA256 与 manifest 不一致")

    tiger_ids_spec = manifest.get("tiger_ids")
    if not isinstance(tiger_ids_spec, dict):
        raise TigerDataError("TIGER identifier manifest 缺少 tiger_ids")
    if tiger_ids_spec.get("fixed_length") != 4:
        raise TigerDataError("TIGER identifier 必须固定为四层")
    if tiger_ids_spec.get("token_order") != ["S1", "S2", "S3", "C"]:
        raise TigerDataError("TIGER identifier token_order 必须是 S1/S2/S3/C")
    raw_capacities = tiger_ids_spec.get("token_capacities")
    if (
        not isinstance(raw_capacities, list)
        or len(raw_capacities) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in raw_capacities
        )
    ):
        raise TigerDataError("TIGER identifier token_capacities 无效")
    token_capacities = tuple(int(value) for value in raw_capacities)
    if token_capacities[:3] != expected_base_codebook_sizes:
        raise TigerDataError(
            "TIGER identifier base SID 码本容量与预期不一致："
            f"manifest={token_capacities[:3]}，"
            f"expected={expected_base_codebook_sizes}"
        )

    parquet_file = pq.ParquetFile(mapping_path)
    if parquet_file.metadata.num_rows != poi_count:
        raise TigerDataError("TIGER identifier 映射行数与 manifest 不一致")
    missing_columns = set(TIGER_MAPPING_COLUMNS) - set(
        parquet_file.schema_arrow.names
    )
    if missing_columns:
        raise TigerDataError(
            "TIGER identifier 映射缺少字段："
            + ", ".join(sorted(missing_columns))
        )

    codes = np.empty((poi_count, 4), dtype=np.int32)
    row_by_poi_id: dict[str, int] = {}
    offset = 0
    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group_index,
            columns=list(TIGER_MAPPING_COLUMNS),
        )
        data = table.to_pydict()
        row_count = table.num_rows
        end = offset + row_count
        group_codes = np.column_stack(
            [
                np.asarray(data[column], dtype=np.int32)
                for column in ("s1", "s2", "s3", "collision_code")
            ]
        )
        codes[offset:end] = group_codes
        for local_index, poi_id in enumerate(data["poi_id"]):
            if not isinstance(poi_id, str) or not poi_id:
                raise TigerDataError("TIGER identifier 映射包含无效 POI ID")
            row = offset + local_index
            if poi_id in row_by_poi_id:
                raise TigerDataError(f"TIGER identifier 映射 POI 重复：{poi_id}")
            row_by_poi_id[poi_id] = row
        offset = end

    if offset != poi_count or len(row_by_poi_id) != poi_count:
        raise TigerDataError("TIGER identifier 映射未完整载入")
    for level, capacity in enumerate(token_capacities):
        values = codes[:, level]
        if np.any(values < 0) or np.any(values >= capacity):
            raise TigerDataError(
                f"TIGER identifier 第 {level + 1} 层超出 [0,{capacity})"
            )

    return (
        TigerIdLookup(
            row_by_poi_id=row_by_poi_id,
            codes=codes,
            poi_count=poi_count,
            token_capacities=token_capacities,
        ),
        manifest,
        mapping_sha256,
        sha256_file(manifest_path),
    )


def stable_user_bucket(passenger_id: str, bucket_count: int = 2000) -> int:
    """Hash one passenger ID into TIGER's fixed user-token vocabulary."""

    if not isinstance(passenger_id, str) or not passenger_id:
        raise TigerDataError("passenger_id 必须是非空字符串")
    if bucket_count <= 0:
        raise TigerDataError("user_bucket_count 必须大于 0")
    digest = hashlib.sha256(passenger_id.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big") % bucket_count


def user_token(passenger_id: str, bucket_count: int = 2000) -> str:
    """Return the zero-padded hashed TIGER user token."""

    width = max(4, len(str(bucket_count - 1)))
    return f"<U_{stable_user_bucket(passenger_id, bucket_count):0{width}d}>"


def tiger_id_content(codes: Sequence[int]) -> str:
    """Format one fixed four-token TIGER item identifier."""

    if len(codes) != 4:
        raise TigerDataError("TIGER item identifier 必须包含四个 Token")
    return (
        f"<S1_{int(codes[0])}>"
        f"<S2_{int(codes[1])}>"
        f"<S3_{int(codes[2])}>"
        f"<C_{int(codes[3])}>"
    )


def _tiger_id_for_poi(
    poi_id: str,
    tiger_lookup: TigerIdLookup,
) -> tuple[str, str]:
    row = tiger_lookup.row_by_poi_id.get(poi_id)
    if row is None:
        raise TigerDataError(f"POI 未匹配 TIGER identifier：{poi_id}")
    codes = tiger_lookup.codes[row]
    content = tiger_id_content(codes)
    key = (
        f"{int(codes[0])}-{int(codes[1])}-{int(codes[2])}"
        f"|c{int(codes[3])}"
    )
    return content, key


def _user_gid(longitude: float, latitude: float, length: int) -> str:
    geohash = encode_geohash(longitude, latitude, length)
    return "".join(f"<G_{character}>" for character in geohash)


def format_tiger_user_content(
    *,
    passenger_id: str,
    current_query: str,
    current_longitude: float,
    current_latitude: float,
    history: Sequence[Mapping[str, Any]],
    tiger_lookup: TigerIdLookup,
    user_bucket_count: int,
    geohash_length: int,
) -> str:
    """Linearize the confirmed map-search-adapted TIGER prompt."""

    parts = [
        f"<USER_ID>{user_token(passenger_id, user_bucket_count)}</USER_ID>",
        "<HISTORY>",
    ]
    for event in history:
        history_tiger_id, _ = _tiger_id_for_poi(
            str(event["poi_id"]),
            tiger_lookup,
        )
        parts.extend(
            (
                "<EVENT>",
                "<USER_GID>"
                + _user_gid(
                    float(event["disp_lng"]),
                    float(event["disp_lat"]),
                    geohash_length,
                )
                + "</USER_GID>",
                f"<QUERY>{event['query']}</QUERY>",
                f"<POI_TIGER_ID>{history_tiger_id}</POI_TIGER_ID>",
                "</EVENT>",
            )
        )
    parts.extend(
        (
            "</HISTORY>",
            "<CURRENT>",
            "<USER_GID>"
            + _user_gid(
                current_longitude,
                current_latitude,
                geohash_length,
            )
            + "</USER_GID>",
            f"<QUERY>{current_query}</QUERY>",
            "</CURRENT>",
        )
    )
    return "\n".join(parts)


def build_tiger_special_tokens(
    *,
    token_capacities: Sequence[int],
    user_bucket_count: int = 2000,
) -> dict[str, Any]:
    """Return the complete tokenizer inventory for the TIGER data."""

    if (
        len(token_capacities) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in token_capacities
        )
    ):
        raise TigerDataError("token_capacities 必须包含四个正整数")
    if user_bucket_count <= 0:
        raise TigerDataError("user_bucket_count 必须大于 0")

    geohash_tokens = [
        f"<G_{character}>" for character in GEOHASH_ALPHABET
    ]
    width = max(4, len(str(user_bucket_count - 1)))
    user_tokens = [
        f"<U_{value:0{width}d}>" for value in range(user_bucket_count)
    ]
    item_tokens = {
        "s1": [
            f"<S1_{value}>" for value in range(token_capacities[0])
        ],
        "s2": [
            f"<S2_{value}>" for value in range(token_capacities[1])
        ],
        "s3": [
            f"<S3_{value}>" for value in range(token_capacities[2])
        ],
        "collision": [
            f"<C_{value}>" for value in range(token_capacities[3])
        ],
    }
    tokens = [
        *STRUCTURE_TOKENS,
        *geohash_tokens,
        *user_tokens,
        *item_tokens["s1"],
        *item_tokens["s2"],
        *item_tokens["s3"],
        *item_tokens["collision"],
    ]
    if len(tokens) != len(set(tokens)):
        raise TigerDataError("TIGER 特殊 Token 表包含重复项")
    return {
        "schema_version": SPECIAL_TOKENS_SCHEMA_VERSION,
        "structure_tokens": list(STRUCTURE_TOKENS),
        "geohash_tokens": geohash_tokens,
        "user_bucket_count": user_bucket_count,
        "user_tokens": user_tokens,
        "item_token_capacities": list(token_capacities),
        "item_tokens": item_tokens,
        "additional_special_tokens": tokens,
        "token_count": len(tokens),
    }


def _parse_target_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise TigerDataError(f"{field_name} 必须是时间字符串")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError as error:
        raise TigerDataError(
            f"{field_name} 必须为 YYYY-MM-DD HH:MM:SS：{value}"
        ) from error
    if parsed.strftime("%Y-%m-%d %H:%M:%S") != value:
        raise TigerDataError(
            f"{field_name} 必须为 YYYY-MM-DD HH:MM:SS：{value}"
        )
    return parsed


def _parse_history_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise TigerDataError(f"{field_name} 必须是 ISO 时间字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise TigerDataError(f"{field_name} 不是合法 ISO 时间：{value}") from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(BEIJING_TIMEZONE).replace(tzinfo=None)
    return parsed


def _parse_source_date(value: Any) -> date:
    if not isinstance(value, str):
        raise TigerDataError("source_dt 必须是 YYYYMMDD 字符串")
    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError as error:
        raise TigerDataError(f"source_dt 必须为 YYYYMMDD：{value}") from error
    if parsed.strftime("%Y%m%d") != value:
        raise TigerDataError(f"source_dt 必须为 YYYYMMDD：{value}")
    return parsed


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TigerDataError(f"{field_name} 必须是非空字符串")
    return value


def _require_query(value: Any, field_name: str) -> str:
    query = _require_string(value, field_name)
    if not query.strip():
        raise TigerDataError(f"{field_name} 不能为空白字符串")
    return query


def _require_coordinate(
    value: Any,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise TigerDataError(
            f"{field_name} 必须位于 [{minimum}, {maximum}]"
        )
    return float(value)


def _validate_history(
    record: Mapping[str, Any],
    *,
    current_time: datetime,
    history_window: HistoryWindow,
    max_history_events: int,
    tiger_lookup: TigerIdLookup,
) -> tuple[list[dict[str, Any]], datetime | None, datetime | None]:
    history_length = record["history_length"]
    if (
        isinstance(history_length, bool)
        or not isinstance(history_length, int)
        or not 0 <= history_length <= max_history_events
    ):
        raise TigerDataError(
            f"history_length 必须位于 [0,{max_history_events}]"
        )
    raw_history = record["history_sequence"]
    if not isinstance(raw_history, list):
        raise TigerDataError("history_sequence 必须是数组")
    if len(raw_history) != history_length:
        raise TigerDataError(
            "history_length 与 history_sequence 实际长度不一致"
        )

    normalized: list[dict[str, Any]] = []
    previous_time: datetime | None = None
    first_time: datetime | None = None
    for index, raw_event in enumerate(raw_history):
        prefix = f"history_sequence[{index}]"
        if not isinstance(raw_event, dict):
            raise TigerDataError(f"{prefix} 必须是对象")
        missing = [
            field for field in REQUIRED_HISTORY_FIELDS if field not in raw_event
        ]
        if missing:
            raise TigerDataError(
                f"{prefix} 缺少字段：" + ", ".join(missing)
            )
        event_time = _parse_history_timestamp(
            raw_event["event_time"],
            f"{prefix}.event_time",
        )
        if not history_window.contains(event_time):
            raise TigerDataError(f"{prefix}.event_time 超出历史窗口")
        if event_time >= current_time:
            raise TigerDataError(f"{prefix}.event_time 未严格早于当前请求")
        if previous_time is not None and event_time < previous_time:
            raise TigerDataError("history_sequence 没有按时间正序排列")
        first_time = first_time or event_time
        previous_time = event_time

        poi_id = _require_string(
            raw_event["poi_id"],
            f"{prefix}.poi_id",
        )
        _tiger_id_for_poi(poi_id, tiger_lookup)
        normalized.append(
            {
                "event_time": event_time,
                "order_id": _require_string(
                    raw_event["order_id"],
                    f"{prefix}.order_id",
                ),
                "searchid": _require_string(
                    raw_event["searchid"],
                    f"{prefix}.searchid",
                ),
                "query": _require_query(
                    raw_event["query"],
                    f"{prefix}.query",
                ),
                "disp_lng": _require_coordinate(
                    raw_event["disp_lng"],
                    f"{prefix}.disp_lng",
                    -180.0,
                    180.0,
                ),
                "disp_lat": _require_coordinate(
                    raw_event["disp_lat"],
                    f"{prefix}.disp_lat",
                    -90.0,
                    90.0,
                ),
                "poi_id": poi_id,
            }
        )
    return normalized, first_time, previous_time


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


def build_tiger_sft_data(
    orders_dir: Path,
    tiger_id_dir: Path,
    output_dir: Path,
    split: TimeSplit,
    history_window: HistoryWindow,
    *,
    max_history_events: int = 10,
    geohash_length: int = 6,
    user_bucket_count: int = 2000,
    expected_base_codebook_sizes: Sequence[int] = (1024, 1024, 1024),
    max_retained_per_split: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> TigerDataResult:
    """Build deterministic map-search-adapted TIGER Messages JSONL."""

    if history_window.end >= split.train_start:
        raise TigerDataError(
            "历史窗口必须完整早于训练窗口，避免固定历史包含未来信息"
        )
    if max_history_events <= 0:
        raise TigerDataError("max_history_events 必须大于 0")
    if geohash_length != 6:
        raise TigerDataError("TIGER 地图检索第一版 Geohash 长度固定为 6")
    if user_bucket_count != 2000:
        raise TigerDataError("TIGER 论文口径固定使用 2000 个用户哈希 Token")
    if max_retained_per_split is not None and max_retained_per_split <= 0:
        raise TigerDataError("max_retained_per_split 必须大于 0")

    orders_dir = orders_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise TigerDataError(
            f"输出目录已存在，请更换或先确认清理：{output_dir}"
        )
    order_files = discover_order_files(orders_dir)
    (
        tiger_lookup,
        tiger_manifest,
        tiger_mapping_sha256,
        tiger_manifest_sha256,
    ) = load_tiger_id_lookup(
        tiger_id_dir,
        expected_base_codebook_sizes=expected_base_codebook_sizes,
    )
    special_tokens = build_tiger_special_tokens(
        token_capacities=tiger_lookup.token_capacities,
        user_bucket_count=user_bucket_count,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "scanned_order_count": 0,
        "blank_query_count": 0,
        "split_limit_skip_count": 0,
        "retained_sample_count": 0,
        "train_count": 0,
        "valid_count": 0,
        "test_count": 0,
        "empty_history_count": 0,
        "nonempty_history_count": 0,
        "history_event_occurrence_count": 0,
        "history_length_sum": 0,
        "history_length_histogram": {
            str(value): 0 for value in range(max_history_events + 1)
        },
        "history_event_min_time": None,
        "history_event_max_time": None,
        "daily_sample_counts": {
            date.fromordinal(ordinal).isoformat(): 0
            for ordinal in range(
                split.allowed_start.toordinal(),
                split.allowed_end.toordinal() + 1,
            )
        },
        "distinct_user_token_count": 0,
    }
    used_user_buckets: set[int] = set()
    input_files: list[dict[str, Any]] = []
    stopped_early = False

    def all_split_limits_reached() -> bool:
        return max_retained_per_split is not None and all(
            stats[f"{name}_count"] >= max_retained_per_split
            for name in ("train", "valid", "test")
        )

    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.building-",
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        handles = {
            name: (temporary_dir / f"{name}.jsonl").open(
                "w",
                encoding="utf-8",
                newline="\n",
            )
            for name in ("train", "valid", "test")
        }
        try:
            for file_index, path in enumerate(order_files, start=1):
                relative_path = path.relative_to(orders_dir).as_posix()
                digest = hashlib.sha256()
                scanned_rows = 0
                complete_file = True
                with path.open("rb") as source:
                    for line_number, raw_line in enumerate(source, start=1):
                        digest.update(raw_line)
                        scanned_rows += 1
                        stats["scanned_order_count"] += 1
                        location = f"{relative_path}:{line_number}"
                        try:
                            record = json.loads(raw_line.decode("utf-8"))
                            if not isinstance(record, dict):
                                raise TigerDataError("记录必须是 JSON 对象")
                            missing = [
                                field
                                for field in REQUIRED_TARGET_FIELDS
                                if field not in record
                            ]
                            if missing:
                                raise TigerDataError(
                                    "缺少字段：" + ", ".join(missing)
                                )
                            current_time = _parse_target_timestamp(
                                record["create_time"],
                                "create_time",
                            )
                            source_date = _parse_source_date(record["source_dt"])
                            if source_date != current_time.date():
                                raise TigerDataError(
                                    "source_dt 与 create_time 日期不一致"
                                )
                            split_name = split.split_for(current_time.date())
                            if (
                                max_retained_per_split is not None
                                and stats[f"{split_name}_count"]
                                >= max_retained_per_split
                            ):
                                stats["split_limit_skip_count"] += 1
                                continue

                            query_value = record["query"]
                            if query_value is None or (
                                isinstance(query_value, str)
                                and not query_value.strip()
                            ):
                                stats["blank_query_count"] += 1
                                continue
                            current_query = _require_query(query_value, "query")
                            current_longitude = _require_coordinate(
                                record["disp_lng"],
                                "disp_lng",
                                -180.0,
                                180.0,
                            )
                            current_latitude = _require_coordinate(
                                record["disp_lat"],
                                "disp_lat",
                                -90.0,
                                90.0,
                            )
                            order_id = _require_string(
                                record["order_id"],
                                "order_id",
                            )
                            searchid = _require_string(
                                record["searchid"],
                                "searchid",
                            )
                            passenger_id = _require_string(
                                record["passenger_id"],
                                "passenger_id",
                            )
                            target_poi_id = _require_string(
                                record["poi_id"],
                                "poi_id",
                            )
                            target_tiger_id, target_tiger_id_key = (
                                _tiger_id_for_poi(
                                    target_poi_id,
                                    tiger_lookup,
                                )
                            )
                            history, first_time, last_time = _validate_history(
                                record,
                                current_time=current_time,
                                history_window=history_window,
                                max_history_events=max_history_events,
                                tiger_lookup=tiger_lookup,
                            )
                        except (
                            TigerDataError,
                            UnicodeDecodeError,
                            json.JSONDecodeError,
                        ) as error:
                            raise TigerDataError(
                                f"{location} 数据校验失败：{error}"
                            ) from error

                        hashed_user_token = user_token(
                            passenger_id,
                            user_bucket_count,
                        )
                        used_user_buckets.add(
                            stable_user_bucket(
                                passenger_id,
                                user_bucket_count,
                            )
                        )
                        history_length = len(history)
                        sample = {
                            "sample_id": stable_sample_id(
                                relative_path,
                                line_number,
                            ),
                            "messages": [
                                {
                                    "role": "user",
                                    "content": format_tiger_user_content(
                                        passenger_id=passenger_id,
                                        current_query=current_query,
                                        current_longitude=current_longitude,
                                        current_latitude=current_latitude,
                                        history=history,
                                        tiger_lookup=tiger_lookup,
                                        user_bucket_count=user_bucket_count,
                                        geohash_length=geohash_length,
                                    ),
                                },
                                {
                                    "role": "assistant",
                                    "content": (
                                        f"<TARGET_POI>{target_tiger_id}"
                                        "</TARGET_POI>"
                                    ),
                                },
                            ],
                            "user_token": hashed_user_token,
                            "order_id": order_id,
                            "searchid": searchid,
                            "target_poi_id": target_poi_id,
                            "target_tiger_id_key": target_tiger_id_key,
                            "history_length": history_length,
                            "split": split_name,
                        }
                        handles[split_name].write(
                            json.dumps(
                                sample,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )

                        stats["retained_sample_count"] += 1
                        stats[f"{split_name}_count"] += 1
                        stats["daily_sample_counts"][
                            current_time.date().isoformat()
                        ] += 1
                        stats["history_length_sum"] += history_length
                        stats["history_event_occurrence_count"] += history_length
                        stats["history_length_histogram"][
                            str(history_length)
                        ] += 1
                        if history_length == 0:
                            stats["empty_history_count"] += 1
                        else:
                            stats["nonempty_history_count"] += 1
                            first_value = first_time.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            last_value = last_time.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            current_min = stats["history_event_min_time"]
                            current_max = stats["history_event_max_time"]
                            stats["history_event_min_time"] = (
                                first_value
                                if current_min is None
                                else min(current_min, first_value)
                            )
                            stats["history_event_max_time"] = (
                                last_value
                                if current_max is None
                                else max(current_max, last_value)
                            )

                        if all_split_limits_reached():
                            complete_file = False
                            stopped_early = True
                            break

                input_files.append(
                    {
                        "relative_path": relative_path,
                        "scanned_rows": scanned_rows,
                        "scanned_bytes_sha256": digest.hexdigest(),
                        "complete_file": complete_file,
                    }
                )
                if progress is not None:
                    progress(
                        f"TIGER SFT：{file_index}/{len(order_files)} 个分片，"
                        f"扫描 {stats['scanned_order_count']:,} 行，"
                        f"保留 {stats['retained_sample_count']:,} 行"
                    )
                if stopped_early:
                    break
        finally:
            for handle in handles.values():
                handle.close()

        if stats["retained_sample_count"] <= 0:
            raise TigerDataError("过滤空 Query 后没有保留样本")
        if (
            stats["empty_history_count"] + stats["nonempty_history_count"]
            != stats["retained_sample_count"]
        ):
            raise TigerDataError("有历史和无历史样本数不守恒")
        if (
            stats["train_count"]
            + stats["valid_count"]
            + stats["test_count"]
            != stats["retained_sample_count"]
        ):
            raise TigerDataError("Train/Valid/Test 样本数不守恒")
        if max_retained_per_split is not None and not all_split_limits_reached():
            raise TigerDataError(
                "扫描完输入后仍未达到各切分的 smoke 样本上限"
            )

        stats["history_coverage_ratio"] = (
            stats["nonempty_history_count"] / stats["retained_sample_count"]
        )
        stats["average_history_length"] = (
            stats["history_length_sum"] / stats["retained_sample_count"]
        )
        stats["distinct_user_token_count"] = len(used_user_buckets)

        _write_json(temporary_dir / "special_tokens.json", special_tokens)
        _write_json(temporary_dir / "stats.json", stats)
        outputs = {
            name: {
                "sha256": sha256_file(temporary_dir / name),
                **(
                    {"rows": stats[f"{name[:-6]}_count"]}
                    if name.endswith(".jsonl")
                    else {}
                ),
            }
            for name in OUTPUT_FILENAMES
            if name != "manifest.json"
        }
        fingerprint_payload = {
            "schema_version": SCHEMA_VERSION,
            "input_files": input_files,
            "tiger_mapping_sha256": tiger_mapping_sha256,
            "tiger_manifest_sha256": tiger_manifest_sha256,
            "time_split": split.to_manifest(),
            "history_window": history_window.to_manifest(),
            "max_history_events": max_history_events,
            "geohash_length": geohash_length,
            "user_bucket_count": user_bucket_count,
            "max_retained_per_split": max_retained_per_split,
        }
        build_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "task_name": TASK_NAME,
            "baseline_variant": "map_search_adapted_tiger",
            "strict_paper_replication": False,
            "build_fingerprint": build_fingerprint,
            "input": {
                "directory": orders_dir.name,
                "files": input_files,
                "scan_mode": (
                    "full"
                    if max_retained_per_split is None
                    else "smoke_prefix_until_each_split_limit"
                ),
                "max_retained_per_split": max_retained_per_split,
                "all_files_fully_scanned": not stopped_early,
            },
            "tiger_identifier": {
                "schema_version": tiger_manifest["schema_version"],
                "poi_count": tiger_lookup.poi_count,
                "mapping_sha256": tiger_mapping_sha256,
                "manifest_sha256": tiger_manifest_sha256,
                "token_order": ["S1", "S2", "S3", "C"],
                "token_capacities": list(tiger_lookup.token_capacities),
            },
            "time_split": split.to_manifest(),
            "history": {
                "window": history_window.to_manifest(),
                "max_events": max_history_events,
                "order": "event_time_ascending",
                "event_time_in_prompt": False,
                "empty_history_policy": "retain_empty_sequence",
            },
            "user_identifier": {
                "source_field": "passenger_id",
                "raw_value_in_prompt": False,
                "hash": "sha256_utf8_full_digest_big_endian_modulo",
                "bucket_count": user_bucket_count,
                "token_format": "<U_XXXX>",
            },
            "prompt": {
                "history_event_order": [
                    "request_geohash6_gid",
                    "raw_query",
                    "four_token_tiger_poi_id",
                ],
                "current_order": [
                    "request_geohash6_gid",
                    "raw_query",
                ],
                "target": "four_token_tiger_poi_id",
                "target_wrapper": ["<TARGET_POI>", "</TARGET_POI>"],
                "forbidden_fields": [
                    "dest_lng",
                    "dest_lat",
                    "dest_name",
                ],
            },
            "processing_rules": {
                "only_filter": "query_is_null_or_query_strip_is_empty",
                "preserve_query_text": True,
                "preserve_duplicate_target_rows": True,
                "history_must_precede_target": True,
                "history_and_current_gid_use_disp_lng_disp_lat": True,
            },
            "stats": stats,
            "outputs": outputs,
        }
        _write_json(temporary_dir / "manifest.json", manifest)
        output_hashes = {
            name: sha256_file(temporary_dir / name)
            for name in OUTPUT_FILENAMES
        }
        os.replace(temporary_dir, output_dir)

    return TigerDataResult(
        manifest=manifest,
        stats=stats,
        output_hashes=output_hashes,
    )

"""Build map-search-adapted GNPR supervised fine-tuning samples."""

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

from poi_gr.dedup_pid import sha256_file
from poi_gr.geohash_pid import GEOHASH_ALPHABET, encode_geohash
from poi_gr.sft_main_data import (
    TimeSplit,
    discover_order_files,
    stable_sample_id,
)


SCHEMA_VERSION = "gnpr-map-search-sft-data-v1"
SPECIAL_TOKENS_SCHEMA_VERSION = "gnpr-map-search-special-tokens-v1"
TASK_NAME = "gnpr_map_search_history_query_gid_poi_to_target"
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
STRUCTURE_TOKENS = (
    "<HISTORY>",
    "</HISTORY>",
    "<EVENT>",
    "</EVENT>",
    "<USER_GID>",
    "</USER_GID>",
    "<QUERY>",
    "</QUERY>",
    "<POI_GNPR_ID>",
    "</POI_GNPR_ID>",
    "<CURRENT>",
    "</CURRENT>",
    "<TARGET_POI>",
    "</TARGET_POI>",
)


class GnprDataError(ValueError):
    """Raised when GNPR SFT data violates the declared contract."""


@dataclass(frozen=True)
class HistoryWindow:
    """Closed local-date range used for historical interactions."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise GnprDataError("history_start 不能晚于 history_end")

    def contains(self, value: datetime) -> bool:
        """Return whether one normalized local timestamp is in the window."""

        return self.start <= value.date() <= self.end

    def to_manifest(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass(frozen=True)
class GnprIdLookup:
    """Compact numeric POI lookup backed by a memory-mapped ID array."""

    row_by_numeric_poi_id: dict[int, int]
    codes: np.ndarray
    poi_count: int
    token_capacities: tuple[int, int, int, int]


@dataclass(frozen=True)
class GnprDataResult:
    """Completed GNPR SFT data artifacts."""

    manifest: dict[str, Any]
    stats: dict[str, Any]
    output_hashes: dict[str, str]


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise GnprDataError(f"{name} 不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GnprDataError(f"{name} JSON 解析失败：{path}") from error
    if not isinstance(payload, dict):
        raise GnprDataError(f"{name} 必须是 JSON object：{path}")
    return payload


def _resolve_manifest_path(
    base_dir: Path,
    spec: Mapping[str, Any],
    name: str,
) -> Path:
    raw_path = spec.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise GnprDataError(f"GNPR identifier manifest 缺少 {name}.path")
    path = (base_dir / raw_path).resolve()
    if not path.is_file():
        raise GnprDataError(f"{name} 不存在：{path}")
    expected_hash = spec.get("sha256")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise GnprDataError(f"{name} SHA256 与 manifest 不一致")
    return path


def _numeric_poi_id(value: Any, field_name: str) -> int:
    if not isinstance(value, str) or not value:
        raise GnprDataError(f"{field_name} 必须是非空数字字符串")
    if not value.isdecimal():
        raise GnprDataError(f"{field_name} 必须是非空数字字符串：{value}")
    numeric = int(value)
    if numeric < 0:
        raise GnprDataError(f"{field_name} 不允许为负数")
    return numeric


def load_gnpr_id_lookup(
    gnpr_id_dir: Path,
) -> tuple[GnprIdLookup, dict[str, Any], str, str]:
    """Load and verify the formal paper-compatible GNPR identifiers."""

    gnpr_id_dir = gnpr_id_dir.resolve()
    manifest_path = gnpr_id_dir / "gnpr_id_manifest.json"
    manifest = _load_json_object(manifest_path, "GNPR identifier manifest")
    if manifest.get("schema_version") != "gnpr-poi-identifier-v1":
        raise GnprDataError(
            "GNPR identifier manifest schema_version 必须是 "
            "gnpr-poi-identifier-v1"
        )
    if manifest.get("status") != "completed":
        raise GnprDataError("GNPR identifier manifest 状态不是 completed")

    poi_spec = manifest.get("poi_ids")
    ids_spec = manifest.get("gnpr_ids")
    mapping_spec = manifest.get("mapping")
    if not isinstance(poi_spec, dict):
        raise GnprDataError("GNPR identifier manifest 缺少 poi_ids")
    if not isinstance(ids_spec, dict):
        raise GnprDataError("GNPR identifier manifest 缺少 gnpr_ids")
    if not isinstance(mapping_spec, dict):
        raise GnprDataError("GNPR identifier manifest 缺少 mapping")

    poi_count = poi_spec.get("rows")
    if (
        isinstance(poi_count, bool)
        or not isinstance(poi_count, int)
        or poi_count <= 0
        or mapping_spec.get("rows") != poi_count
    ):
        raise GnprDataError("GNPR identifier POI 行数无效或不一致")
    if poi_spec.get("unique") is not True:
        raise GnprDataError("GNPR identifier manifest 未声明 POI ID 唯一")
    if mapping_spec.get("gnpr_id_key_unique") is not True:
        raise GnprDataError("GNPR identifier manifest 未声明标识全局唯一")

    raw_capacities = ids_spec.get("codebook_capacities")
    dedup_capacity = ids_spec.get("dedup_token_capacity")
    if (
        raw_capacities != [512, 512, 512]
        or isinstance(dedup_capacity, bool)
        or not isinstance(dedup_capacity, int)
        or dedup_capacity <= 0
    ):
        raise GnprDataError("当前 GNPR baseline 要求 512×3 和正整数 Dedup 容量")
    if ids_spec.get("serialized_length") != {
        "singleton": 3,
        "collision": 4,
    }:
        raise GnprDataError("GNPR identifier 必须是条件 Dedup 的三或四 Token")
    token_capacities = (512, 512, 512, int(dedup_capacity))

    poi_ids_path = _resolve_manifest_path(
        gnpr_id_dir,
        poi_spec,
        "poi_ids",
    )
    ids_path = _resolve_manifest_path(
        gnpr_id_dir,
        ids_spec,
        "gnpr_ids",
    )
    mapping_path = _resolve_manifest_path(
        gnpr_id_dir,
        mapping_spec,
        "mapping",
    )

    codes = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    if codes.shape != (poi_count, 4) or codes.dtype != np.int32:
        raise GnprDataError("gnpr_ids.npy 必须是与 POI 对齐的 [N,4] int32")
    for level, capacity in enumerate(token_capacities[:3]):
        values = codes[:, level]
        if np.any(values < 0) or np.any(values >= capacity):
            raise GnprDataError(
                f"GNPR identifier 第 {level + 1} 层超出 [0,{capacity})"
            )
    dedup_values = codes[:, 3]
    if np.any(dedup_values < -1) or np.any(dedup_values >= dedup_capacity):
        raise GnprDataError("GNPR Dedup 层超出 [-1,dedup_capacity)")

    row_by_numeric_poi_id: dict[int, int] = {}
    with poi_ids_path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            try:
                poi_id = json.loads(line)
            except json.JSONDecodeError as error:
                raise GnprDataError(
                    f"poi_ids 第 {row + 1} 行 JSON 解析失败"
                ) from error
            numeric = _numeric_poi_id(poi_id, f"poi_ids[{row}]")
            if numeric in row_by_numeric_poi_id:
                raise GnprDataError(f"GNPR identifier POI 重复：{poi_id}")
            row_by_numeric_poi_id[numeric] = row
    if len(row_by_numeric_poi_id) != poi_count:
        raise GnprDataError("GNPR identifier POI ID 未完整载入")

    return (
        GnprIdLookup(
            row_by_numeric_poi_id=row_by_numeric_poi_id,
            codes=codes,
            poi_count=poi_count,
            token_capacities=token_capacities,
        ),
        manifest,
        sha256_file(mapping_path),
        sha256_file(manifest_path),
    )


def gnpr_id_content(codes: Sequence[int]) -> str:
    """Format one paper-style three- or four-token GNPR identifier."""

    if len(codes) != 4:
        raise GnprDataError("GNPR identifier 内部表示必须包含四列")
    values = tuple(int(value) for value in codes)
    if any(value < 0 for value in values[:3]) or values[3] < -1:
        raise GnprDataError("GNPR identifier 包含非法负值")
    content = (
        f"<a_{values[0]}>"
        f"<b_{values[1]}>"
        f"<c_{values[2]}>"
    )
    return content if values[3] == -1 else content + f"<d_{values[3]}>"


def _gnpr_id_for_poi(
    poi_id: str,
    gnpr_lookup: GnprIdLookup,
) -> tuple[str, str]:
    numeric = _numeric_poi_id(poi_id, "poi_id")
    row = gnpr_lookup.row_by_numeric_poi_id.get(numeric)
    if row is None:
        raise GnprDataError(f"POI 未匹配 GNPR identifier：{poi_id}")
    codes = gnpr_lookup.codes[row]
    content = gnpr_id_content(codes)
    key = f"{int(codes[0])}-{int(codes[1])}-{int(codes[2])}"
    if int(codes[3]) >= 0:
        key += f"|d{int(codes[3])}"
    return content, key


def _user_gid(longitude: float, latitude: float, length: int) -> str:
    geohash = encode_geohash(longitude, latitude, length)
    return "".join(f"<G_{character}>" for character in geohash)


def format_gnpr_user_content(
    *,
    current_query: str,
    current_longitude: float,
    current_latitude: float,
    history: Sequence[Mapping[str, Any]],
    gnpr_lookup: GnprIdLookup,
    geohash_length: int,
) -> str:
    """Linearize GNPR map-search input without time or user identity."""

    parts = ["<HISTORY>"]
    for event in history:
        history_identifier, _ = _gnpr_id_for_poi(
            str(event["poi_id"]),
            gnpr_lookup,
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
                f"<POI_GNPR_ID>{history_identifier}</POI_GNPR_ID>",
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


def build_gnpr_special_tokens(
    token_capacities: Sequence[int],
) -> dict[str, Any]:
    """Return the tokenizer inventory for conditional GNPR identifiers."""

    if (
        len(token_capacities) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in token_capacities
        )
    ):
        raise GnprDataError("token_capacities 必须包含四个正整数")

    geohash_tokens = [
        f"<G_{character}>" for character in GEOHASH_ALPHABET
    ]
    item_tokens = {
        "a": [f"<a_{value}>" for value in range(token_capacities[0])],
        "b": [f"<b_{value}>" for value in range(token_capacities[1])],
        "c": [f"<c_{value}>" for value in range(token_capacities[2])],
        "dedup": [
            f"<d_{value}>" for value in range(token_capacities[3])
        ],
    }
    tokens = [
        *STRUCTURE_TOKENS,
        *geohash_tokens,
        *item_tokens["a"],
        *item_tokens["b"],
        *item_tokens["c"],
        *item_tokens["dedup"],
    ]
    if len(tokens) != len(set(tokens)):
        raise GnprDataError("GNPR 特殊 Token 表包含重复项")
    return {
        "schema_version": SPECIAL_TOKENS_SCHEMA_VERSION,
        "structure_tokens": list(STRUCTURE_TOKENS),
        "geohash_tokens": geohash_tokens,
        "user_tokens": [],
        "item_token_capacities": list(token_capacities),
        "item_tokens": item_tokens,
        "conditional_dedup_token": True,
        "additional_special_tokens": tokens,
        "token_count": len(tokens),
    }


def _parse_target_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise GnprDataError(f"{field_name} 必须是时间字符串")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError as error:
        raise GnprDataError(
            f"{field_name} 必须为 YYYY-MM-DD HH:MM:SS：{value}"
        ) from error
    if parsed.strftime("%Y-%m-%d %H:%M:%S") != value:
        raise GnprDataError(
            f"{field_name} 必须为 YYYY-MM-DD HH:MM:SS：{value}"
        )
    return parsed


def _parse_history_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise GnprDataError(f"{field_name} 必须是 ISO 时间字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GnprDataError(f"{field_name} 不是合法 ISO 时间：{value}") from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(BEIJING_TIMEZONE).replace(tzinfo=None)
    return parsed


def _parse_source_date(value: Any) -> date:
    if not isinstance(value, str):
        raise GnprDataError("source_dt 必须是 YYYYMMDD 字符串")
    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError as error:
        raise GnprDataError(f"source_dt 必须为 YYYYMMDD：{value}") from error
    if parsed.strftime("%Y%m%d") != value:
        raise GnprDataError(f"source_dt 必须为 YYYYMMDD：{value}")
    return parsed


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GnprDataError(f"{field_name} 必须是非空字符串")
    return value


def _require_query(value: Any, field_name: str) -> str:
    query = _require_string(value, field_name)
    if not query.strip():
        raise GnprDataError(f"{field_name} 不能为空白字符串")
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
        raise GnprDataError(
            f"{field_name} 必须位于 [{minimum}, {maximum}]"
        )
    return float(value)


def _validate_history(
    record: Mapping[str, Any],
    *,
    current_time: datetime,
    history_window: HistoryWindow,
    max_history_events: int,
    gnpr_lookup: GnprIdLookup,
) -> tuple[list[dict[str, Any]], datetime | None, datetime | None]:
    history_length = record["history_length"]
    if (
        isinstance(history_length, bool)
        or not isinstance(history_length, int)
        or not 0 <= history_length <= max_history_events
    ):
        raise GnprDataError(
            f"history_length 必须位于 [0,{max_history_events}]"
        )
    raw_history = record["history_sequence"]
    if not isinstance(raw_history, list):
        raise GnprDataError("history_sequence 必须是数组")
    if len(raw_history) != history_length:
        raise GnprDataError(
            "history_length 与 history_sequence 实际长度不一致"
        )

    normalized: list[dict[str, Any]] = []
    previous_time: datetime | None = None
    first_time: datetime | None = None
    for index, raw_event in enumerate(raw_history):
        prefix = f"history_sequence[{index}]"
        if not isinstance(raw_event, dict):
            raise GnprDataError(f"{prefix} 必须是对象")
        missing = [
            field for field in REQUIRED_HISTORY_FIELDS if field not in raw_event
        ]
        if missing:
            raise GnprDataError(
                f"{prefix} 缺少字段：" + ", ".join(missing)
            )
        event_time = _parse_history_timestamp(
            raw_event["event_time"],
            f"{prefix}.event_time",
        )
        if not history_window.contains(event_time):
            raise GnprDataError(f"{prefix}.event_time 超出历史窗口")
        if event_time >= current_time:
            raise GnprDataError(f"{prefix}.event_time 未严格早于当前请求")
        if previous_time is not None and event_time < previous_time:
            raise GnprDataError("history_sequence 没有按时间正序排列")
        first_time = first_time or event_time
        previous_time = event_time

        poi_id = _require_string(raw_event["poi_id"], f"{prefix}.poi_id")
        _gnpr_id_for_poi(poi_id, gnpr_lookup)
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


def build_gnpr_sft_data(
    orders_dir: Path,
    gnpr_id_dir: Path,
    output_dir: Path,
    split: TimeSplit,
    history_window: HistoryWindow,
    *,
    max_history_events: int = 10,
    geohash_length: int = 6,
    max_retained_per_split: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> GnprDataResult:
    """Build deterministic GNPR Messages without time or user identity."""

    if history_window.end >= split.train_start:
        raise GnprDataError(
            "历史窗口必须完整早于训练窗口，避免固定历史包含未来信息"
        )
    if max_history_events <= 0:
        raise GnprDataError("max_history_events 必须大于 0")
    if geohash_length != 6:
        raise GnprDataError("GNPR 地图检索第一版 Geohash 长度固定为 6")
    if max_retained_per_split is not None and max_retained_per_split <= 0:
        raise GnprDataError("max_retained_per_split 必须大于 0")

    orders_dir = orders_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise GnprDataError(
            f"输出目录已存在，请更换或先确认清理：{output_dir}"
        )
    order_files = discover_order_files(orders_dir)
    (
        gnpr_lookup,
        gnpr_manifest,
        gnpr_mapping_sha256,
        gnpr_manifest_sha256,
    ) = load_gnpr_id_lookup(gnpr_id_dir)
    special_tokens = build_gnpr_special_tokens(
        gnpr_lookup.token_capacities
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
    }
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
                                raise GnprDataError("记录必须是 JSON 对象")
                            missing = [
                                field
                                for field in REQUIRED_TARGET_FIELDS
                                if field not in record
                            ]
                            if missing:
                                raise GnprDataError(
                                    "缺少字段：" + ", ".join(missing)
                                )
                            current_time = _parse_target_timestamp(
                                record["create_time"],
                                "create_time",
                            )
                            source_date = _parse_source_date(record["source_dt"])
                            if source_date != current_time.date():
                                raise GnprDataError(
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
                            target_poi_id = _require_string(
                                record["poi_id"],
                                "poi_id",
                            )
                            target_identifier, target_identifier_key = (
                                _gnpr_id_for_poi(
                                    target_poi_id,
                                    gnpr_lookup,
                                )
                            )
                            history, first_time, last_time = _validate_history(
                                record,
                                current_time=current_time,
                                history_window=history_window,
                                max_history_events=max_history_events,
                                gnpr_lookup=gnpr_lookup,
                            )
                        except (
                            GnprDataError,
                            UnicodeDecodeError,
                            json.JSONDecodeError,
                        ) as error:
                            raise GnprDataError(
                                f"{location} 数据校验失败：{error}"
                            ) from error

                        history_length = len(history)
                        sample = {
                            "sample_id": stable_sample_id(
                                relative_path,
                                line_number,
                            ),
                            "messages": [
                                {
                                    "role": "user",
                                    "content": format_gnpr_user_content(
                                        current_query=current_query,
                                        current_longitude=current_longitude,
                                        current_latitude=current_latitude,
                                        history=history,
                                        gnpr_lookup=gnpr_lookup,
                                        geohash_length=geohash_length,
                                    ),
                                },
                                {
                                    "role": "assistant",
                                    "content": (
                                        f"<TARGET_POI>{target_identifier}"
                                        "</TARGET_POI>"
                                    ),
                                },
                            ],
                            "order_id": order_id,
                            "searchid": searchid,
                            "target_poi_id": target_poi_id,
                            "target_gnpr_id_key": target_identifier_key,
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
                        f"GNPR SFT：{file_index}/{len(order_files)} 个分片，"
                        f"扫描 {stats['scanned_order_count']:,} 行，"
                        f"保留 {stats['retained_sample_count']:,} 行"
                    )
                if stopped_early:
                    break
        finally:
            for handle in handles.values():
                handle.close()

        if stats["retained_sample_count"] <= 0:
            raise GnprDataError("过滤空 Query 后没有保留样本")
        if (
            stats["empty_history_count"] + stats["nonempty_history_count"]
            != stats["retained_sample_count"]
        ):
            raise GnprDataError("有历史和无历史样本数不守恒")
        if (
            stats["train_count"]
            + stats["valid_count"]
            + stats["test_count"]
            != stats["retained_sample_count"]
        ):
            raise GnprDataError("Train/Valid/Test 样本数不守恒")
        if max_retained_per_split is not None and not all_split_limits_reached():
            raise GnprDataError(
                "扫描完输入后仍未达到各切分的 smoke 样本上限"
            )

        stats["history_coverage_ratio"] = (
            stats["nonempty_history_count"] / stats["retained_sample_count"]
        )
        stats["average_history_length"] = (
            stats["history_length_sum"] / stats["retained_sample_count"]
        )

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
            "gnpr_mapping_sha256": gnpr_mapping_sha256,
            "gnpr_manifest_sha256": gnpr_manifest_sha256,
            "time_split": split.to_manifest(),
            "history_window": history_window.to_manifest(),
            "max_history_events": max_history_events,
            "geohash_length": geohash_length,
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
            "baseline_variant": "map_search_adapted_gnpr_no_time_no_user_id",
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
            "gnpr_identifier": {
                "schema_version": gnpr_manifest["schema_version"],
                "poi_count": gnpr_lookup.poi_count,
                "mapping_sha256": gnpr_mapping_sha256,
                "manifest_sha256": gnpr_manifest_sha256,
                "token_order": ["a", "b", "c", "d_if_collision"],
                "token_capacities": list(gnpr_lookup.token_capacities),
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
                "source_field_required": False,
                "raw_value_in_prompt": False,
                "hashed_value_in_prompt": False,
            },
            "paper_adaptation": {
                "visit_time_in_prompt": False,
                "target_time_in_prompt": False,
                "multiple_cropping_enabled": False,
                "fill_in_blank_augmentation_enabled": False,
                "reason": "地图检索场景与 TIGER 保持无时间、等样本量输入",
            },
            "prompt": {
                "history_event_order": [
                    "request_geohash6_gid",
                    "raw_query",
                    "conditional_dedup_gnpr_identifier",
                ],
                "current_order": [
                    "request_geohash6_gid",
                    "raw_query",
                ],
                "target": "conditional_dedup_gnpr_identifier",
                "target_wrapper": ["<TARGET_POI>", "</TARGET_POI>"],
                "forbidden_fields": [
                    "passenger_id",
                    "event_time",
                    "create_time",
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

    return GnprDataResult(
        manifest=manifest,
        stats=stats,
        output_hashes=output_hashes,
    )

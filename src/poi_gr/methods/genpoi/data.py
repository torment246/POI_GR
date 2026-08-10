"""Build causal GenPOI supervised fine-tuning data."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from poi_gr.pid.dedup import sha256_file
from poi_gr.pid.geohash import GEOHASH_ALPHABET, encode_geohash
from poi_gr.sft.data import (
    PidLookup,
    TimeSplit,
    assistant_pid_content,
    discover_order_files,
    load_pid_lookup,
    stable_final_pid_key,
    stable_sample_id,
)


SCHEMA_VERSION = "genpoi-sft-data-v1"
TASK_NAME = "genpoi_history_query_location_poi_to_target_pid"
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
STRUCTURE_TOKENS = (
    "<HISTORY>",
    "</HISTORY>",
    "<CURRENT>",
    "</CURRENT>",
    "<QUERY>",
    "</QUERY>",
    "<USER_GID>",
    "</USER_GID>",
    "<POI_PID>",
    "</POI_PID>",
)


class GenpoiDataError(ValueError):
    """Raised when GenPOI data violates the declared contract."""


@dataclass(frozen=True)
class HistoryWindow:
    """Closed date range used for all historical interactions."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise GenpoiDataError("history_start 不能晚于 history_end")

    def contains(self, value: datetime) -> bool:
        """Return whether one event timestamp belongs to the window."""

        return self.start <= value.date() <= self.end

    def to_manifest(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass(frozen=True)
class GenpoiDataResult:
    """Formal result returned after all six artifacts are complete."""

    manifest: dict[str, Any]
    stats: dict[str, Any]
    output_hashes: dict[str, str]


def _parse_target_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise GenpoiDataError(f"{field_name} 必须是时间字符串")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError as error:
        raise GenpoiDataError(
            f"{field_name} 必须为 YYYY-MM-DD HH:MM:SS：{value}"
        ) from error
    if parsed.strftime("%Y-%m-%d %H:%M:%S") != value:
        raise GenpoiDataError(
            f"{field_name} 必须为 YYYY-MM-DD HH:MM:SS：{value}"
        )
    return parsed


def _parse_history_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise GenpoiDataError(f"{field_name} 必须是 ISO 时间字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GenpoiDataError(
            f"{field_name} 不是合法 ISO 时间：{value}"
        ) from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(BEIJING_TIMEZONE).replace(tzinfo=None)
    return parsed


def _parse_source_date(value: Any) -> date:
    if not isinstance(value, str):
        raise GenpoiDataError("source_dt 必须是 YYYYMMDD 字符串")
    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError as error:
        raise GenpoiDataError(f"source_dt 必须为 YYYYMMDD：{value}") from error
    if parsed.strftime("%Y%m%d") != value:
        raise GenpoiDataError(f"source_dt 必须为 YYYYMMDD：{value}")
    return parsed


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GenpoiDataError(f"{field_name} 必须是非空字符串")
    return value


def _require_query(value: Any, field_name: str) -> str:
    query = _require_string(value, field_name)
    if not query.strip():
        raise GenpoiDataError(f"{field_name} 不能为空白字符串")
    return query


def _require_coordinate(
    value: Any, field_name: str, minimum: float, maximum: float
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise GenpoiDataError(
            f"{field_name} 必须位于 [{minimum}, {maximum}]"
        )
    return float(value)


def _pid_for_poi(
    poi_id: str,
    pid_lookup: PidLookup,
    sid_codebook_size: int,
    pid_cache: dict[str, tuple[str, str, bool]] | None = None,
) -> tuple[str, str, bool]:
    if pid_cache is not None:
        cached = pid_cache.get(poi_id)
        if cached is not None:
            return cached
    mapping_row = pid_lookup.row_by_poi_id.get(poi_id)
    if mapping_row is None:
        raise GenpoiDataError(f"POI 未匹配正式 PID：{poi_id}")
    codes = pid_lookup.codes[mapping_row]
    sid_values = [int(value) for value in codes[6:]]
    if any(not 0 <= value < sid_codebook_size for value in sid_values):
        raise GenpoiDataError(
            f"POI SID 超出 GenPOI 码本 [0,{sid_codebook_size})：{poi_id}"
        )
    dedup_code = int(pid_lookup.dedup_codes[mapping_row])
    requires_dedup = bool(pid_lookup.requires_dedup[mapping_row])
    if requires_dedup != (dedup_code >= 0):
        raise GenpoiDataError(f"POI Dedup 状态不一致：{poi_id}")
    result = (
        assistant_pid_content(codes, dedup_code),
        stable_final_pid_key(codes, dedup_code),
        requires_dedup,
    )
    if pid_cache is not None:
        pid_cache[poi_id] = result
    return result


def _user_gid(longitude: float, latitude: float, length: int) -> str:
    geohash = encode_geohash(longitude, latitude, length)
    return "".join(f"<G_{character}>" for character in geohash)


def format_genpoi_user_content(
    *,
    current_query: str,
    current_longitude: float,
    current_latitude: float,
    history: Sequence[Mapping[str, Any]],
    pid_lookup: PidLookup,
    geohash_length: int,
    sid_codebook_size: int,
    pid_cache: dict[str, tuple[str, str, bool]] | None = None,
) -> str:
    """Linearize history and current request in the paper's field order."""

    parts = ["<HISTORY>"]
    for event in history:
        history_pid = event.get("_pid_content")
        if not isinstance(history_pid, str):
            history_pid, _, _ = _pid_for_poi(
                str(event["poi_id"]),
                pid_lookup,
                sid_codebook_size,
                pid_cache,
            )
        parts.extend(
            (
                "<USER_GID>"
                + _user_gid(
                    float(event["disp_lng"]),
                    float(event["disp_lat"]),
                    geohash_length,
                )
                + "</USER_GID>",
                f"<QUERY>{event['query']}</QUERY>",
                f"<POI_PID>{history_pid}</POI_PID>",
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


def build_genpoi_special_tokens(
    *, sid_codebook_size: int, dedup_codebook_size: int = 512
) -> dict[str, Any]:
    """Return the fixed GenPOI structure and PID token inventory."""

    if sid_codebook_size <= 0:
        raise GenpoiDataError("sid_codebook_size 必须大于 0")
    if dedup_codebook_size <= 0:
        raise GenpoiDataError("dedup_codebook_size 必须大于 0")
    geohash_tokens = [
        f"<G_{character}>" for character in GEOHASH_ALPHABET
    ]
    sid_tokens = {
        f"s{level}": [
            f"<S{level}_{value}>" for value in range(sid_codebook_size)
        ]
        for level in range(1, 4)
    }
    dedup_tokens = [
        f"<D_{value}>" for value in range(dedup_codebook_size)
    ]
    tokens = [
        *STRUCTURE_TOKENS,
        *geohash_tokens,
        *sid_tokens["s1"],
        *sid_tokens["s2"],
        *sid_tokens["s3"],
        *dedup_tokens,
    ]
    if len(tokens) != len(set(tokens)):
        raise GenpoiDataError("GenPOI 特殊 Token 表包含重复项")
    return {
        "schema_version": "genpoi-special-tokens-v1",
        "structure_tokens": list(STRUCTURE_TOKENS),
        "geohash_tokens": geohash_tokens,
        "sid_codebook_size": sid_codebook_size,
        "sid_tokens": sid_tokens,
        "dedup_codebook_size": dedup_codebook_size,
        "dedup_tokens": dedup_tokens,
        "additional_special_tokens": tokens,
        "token_count": len(tokens),
    }


def _validate_history(
    record: Mapping[str, Any],
    *,
    current_time: datetime,
    history_window: HistoryWindow,
    max_history_events: int,
    pid_lookup: PidLookup,
    sid_codebook_size: int,
    pid_cache: dict[str, tuple[str, str, bool]],
) -> tuple[list[dict[str, Any]], datetime | None, datetime | None]:
    history_length = record["history_length"]
    if (
        isinstance(history_length, bool)
        or not isinstance(history_length, int)
        or not 0 <= history_length <= max_history_events
    ):
        raise GenpoiDataError(
            f"history_length 必须位于 [0,{max_history_events}]"
        )
    raw_history = record["history_sequence"]
    if not isinstance(raw_history, list):
        raise GenpoiDataError("history_sequence 必须是数组")
    if len(raw_history) != history_length:
        raise GenpoiDataError(
            "history_length 与 history_sequence 实际长度不一致"
        )

    normalized: list[dict[str, Any]] = []
    previous_time: datetime | None = None
    first_time: datetime | None = None
    for index, raw_event in enumerate(raw_history):
        prefix = f"history_sequence[{index}]"
        if not isinstance(raw_event, dict):
            raise GenpoiDataError(f"{prefix} 必须是对象")
        missing = [
            field
            for field in REQUIRED_HISTORY_FIELDS
            if field not in raw_event
        ]
        if missing:
            raise GenpoiDataError(
                f"{prefix} 缺少字段：" + ", ".join(missing)
            )
        event_time = _parse_history_timestamp(
            raw_event["event_time"], f"{prefix}.event_time"
        )
        if not history_window.contains(event_time):
            raise GenpoiDataError(f"{prefix}.event_time 超出历史窗口")
        if event_time >= current_time:
            raise GenpoiDataError(f"{prefix}.event_time 未严格早于当前请求")
        if previous_time is not None and event_time < previous_time:
            raise GenpoiDataError("history_sequence 没有按时间正序排列")
        first_time = first_time or event_time
        previous_time = event_time

        poi_id = _require_string(
            raw_event["poi_id"], f"{prefix}.poi_id"
        )
        history_pid, _, _ = _pid_for_poi(
            poi_id,
            pid_lookup,
            sid_codebook_size,
            pid_cache,
        )
        normalized.append(
            {
                "event_time": event_time,
                "order_id": _require_string(
                    raw_event["order_id"], f"{prefix}.order_id"
                ),
                "searchid": _require_string(
                    raw_event["searchid"], f"{prefix}.searchid"
                ),
                "query": _require_query(
                    raw_event["query"], f"{prefix}.query"
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
                "_pid_content": history_pid,
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


def build_genpoi_sft_data(
    orders_dir: Path,
    pid_mapping_path: Path,
    pid_manifest_path: Path,
    output_dir: Path,
    split: TimeSplit,
    history_window: HistoryWindow,
    *,
    max_history_events: int = 10,
    geohash_length: int = 6,
    sid_codebook_size: int = 1024,
    progress: Callable[[str], None] | None = None,
) -> GenpoiDataResult:
    """Build deterministic GenPOI train/valid/test Messages JSONL."""

    if history_window.end >= split.train_start:
        raise GenpoiDataError(
            "历史窗口必须完整早于训练窗口，避免固定历史包含未来信息"
        )
    if max_history_events <= 0:
        raise GenpoiDataError("max_history_events 必须大于 0")
    if geohash_length != 6:
        raise GenpoiDataError("GenPOI 第一版 Geohash 长度固定为 6")
    if sid_codebook_size != 1024:
        raise GenpoiDataError("北京 GenPOI GeoPE SID 码本固定为 1024")

    orders_dir = orders_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise GenpoiDataError(
            f"输出目录已存在，请更换或先确认清理：{output_dir}"
        )
    order_files = discover_order_files(orders_dir)
    (
        pid_lookup,
        pid_manifest,
        pid_mapping_sha256,
        pid_manifest_sha256,
    ) = load_pid_lookup(pid_mapping_path, pid_manifest_path)
    special_tokens = build_genpoi_special_tokens(
        sid_codebook_size=sid_codebook_size
    )
    pid_cache: dict[str, tuple[str, str, bool]] = {}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "raw_order_count": 0,
        "blank_query_count": 0,
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

    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.building-",
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        split_paths = {
            name: temporary_dir / f"{name}.jsonl"
            for name in ("train", "valid", "test")
        }
        handles = {
            name: path.open("w", encoding="utf-8", newline="\n")
            for name, path in split_paths.items()
        }
        try:
            for file_index, path in enumerate(order_files, start=1):
                relative_path = path.relative_to(orders_dir).as_posix()
                digest = hashlib.sha256()
                file_rows = 0
                with path.open("rb") as source:
                    for line_number, raw_line in enumerate(source, start=1):
                        digest.update(raw_line)
                        file_rows += 1
                        stats["raw_order_count"] += 1
                        location = f"{relative_path}:{line_number}"
                        try:
                            record = json.loads(raw_line.decode("utf-8"))
                            if not isinstance(record, dict):
                                raise GenpoiDataError("记录必须是 JSON 对象")
                            missing = [
                                field
                                for field in REQUIRED_TARGET_FIELDS
                                if field not in record
                            ]
                            if missing:
                                raise GenpoiDataError(
                                    "缺少字段：" + ", ".join(missing)
                                )
                            query_value = record["query"]
                            if query_value is None or (
                                isinstance(query_value, str)
                                and not query_value.strip()
                            ):
                                stats["blank_query_count"] += 1
                                continue
                            current_query = _require_query(
                                query_value, "query"
                            )
                            current_time = _parse_target_timestamp(
                                record["create_time"], "create_time"
                            )
                            source_date = _parse_source_date(
                                record["source_dt"]
                            )
                            if source_date != current_time.date():
                                raise GenpoiDataError(
                                    "source_dt 与 create_time 日期不一致"
                                )
                            split_name = split.split_for(
                                current_time.date()
                            )
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
                            _require_string(record["order_id"], "order_id")
                            _require_string(record["searchid"], "searchid")
                            _require_string(
                                record["passenger_id"], "passenger_id"
                            )
                            target_poi_id = _require_string(
                                record["poi_id"], "poi_id"
                            )
                            (
                                target_pid,
                                target_pid_key,
                                requires_dedup,
                            ) = _pid_for_poi(
                                target_poi_id,
                                pid_lookup,
                                sid_codebook_size,
                                pid_cache,
                            )
                            history, first_time, last_time = (
                                _validate_history(
                                    record,
                                    current_time=current_time,
                                    history_window=history_window,
                                    max_history_events=max_history_events,
                                    pid_lookup=pid_lookup,
                                    sid_codebook_size=sid_codebook_size,
                                    pid_cache=pid_cache,
                                )
                            )
                        except (
                            GenpoiDataError,
                            UnicodeDecodeError,
                            json.JSONDecodeError,
                        ) as error:
                            raise GenpoiDataError(
                                f"{location} 数据校验失败：{error}"
                            ) from error

                        history_length = len(history)
                        sample = {
                            "sample_id": stable_sample_id(
                                relative_path, line_number
                            ),
                            "messages": [
                                {
                                    "role": "user",
                                    "content": format_genpoi_user_content(
                                        current_query=current_query,
                                        current_longitude=current_longitude,
                                        current_latitude=current_latitude,
                                        history=history,
                                        pid_lookup=pid_lookup,
                                        geohash_length=geohash_length,
                                        sid_codebook_size=sid_codebook_size,
                                        pid_cache=pid_cache,
                                    ),
                                },
                                {
                                    "role": "assistant",
                                    "content": target_pid,
                                },
                            ],
                            "order_id": record["order_id"],
                            "searchid": record["searchid"],
                            "target_poi_id": target_poi_id,
                            "target_pid_key": target_pid_key,
                            "requires_dedup": requires_dedup,
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
                        stats["history_event_occurrence_count"] += (
                            history_length
                        )
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

                input_files.append(
                    {
                        "relative_path": relative_path,
                        "rows": file_rows,
                        "sha256": digest.hexdigest(),
                    }
                )
                if progress is not None:
                    progress(
                        f"GenPOI SFT：{file_index}/{len(order_files)} 个分片，"
                        f"累计 {stats['raw_order_count']:,} 行"
                    )
        finally:
            for handle in handles.values():
                handle.close()

        if stats["retained_sample_count"] <= 0:
            raise GenpoiDataError("过滤空 Query 后没有保留样本")
        if (
            stats["empty_history_count"]
            + stats["nonempty_history_count"]
            != stats["retained_sample_count"]
        ):
            raise GenpoiDataError("有历史和无历史样本数不守恒")
        if (
            stats["train_count"]
            + stats["valid_count"]
            + stats["test_count"]
            != stats["retained_sample_count"]
        ):
            raise GenpoiDataError("Train/Valid/Test 样本数不守恒")
        stats["history_coverage_ratio"] = (
            stats["nonempty_history_count"]
            / stats["retained_sample_count"]
        )
        stats["average_history_length"] = (
            stats["history_length_sum"]
            / stats["retained_sample_count"]
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
            "pid_mapping_sha256": pid_mapping_sha256,
            "pid_manifest_sha256": pid_manifest_sha256,
            "time_split": split.to_manifest(),
            "history_window": history_window.to_manifest(),
            "max_history_events": max_history_events,
            "geohash_length": geohash_length,
            "sid_codebook_size": sid_codebook_size,
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
            "build_fingerprint": build_fingerprint,
            "input": {
                "directory": orders_dir.name,
                "files": input_files,
            },
            "pid_mapping": {
                "schema_version": pid_manifest["schema_version"],
                "poi_count": pid_lookup.poi_count,
                "mapping_sha256": pid_mapping_sha256,
                "manifest_sha256": pid_manifest_sha256,
            },
            "time_split": split.to_manifest(),
            "history": {
                "window": history_window.to_manifest(),
                "max_events": max_history_events,
                "order": "event_time_ascending",
                "event_time_in_prompt": False,
                "empty_history_policy": "retain_empty_sequence",
            },
            "prompt": {
                "history_event_order": [
                    "user_gid",
                    "query",
                    "poi_pid",
                ],
                "current_order": ["user_gid", "query"],
                "target": "target_poi_pid",
            },
            "identifier": {
                "geohash_length": geohash_length,
                "sid_levels": 3,
                "sid_codebook_size": sid_codebook_size,
                "optional_dedup": True,
            },
            "processing_rules": {
                "only_filter": (
                    "query_is_null_or_query_strip_is_empty"
                ),
                "preserve_query_text": True,
                "preserve_duplicate_target_rows": True,
                "history_must_precede_target": True,
            },
            "stats": stats,
            "outputs": outputs,
        }
        _write_json(temporary_dir / "manifest.json", manifest)
        output_hashes = {
            **{
                name: specification["sha256"]
                for name, specification in outputs.items()
            },
            "manifest.json": sha256_file(temporary_dir / "manifest.json"),
        }
        os.replace(temporary_dir, output_dir)

    return GenpoiDataResult(
        manifest=manifest,
        stats=stats,
        output_hashes=output_hashes,
    )

"""Build SID-free behavior data directly from raw orders and BGE row order."""

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

from poi_gr.methods.tiger_joint.catalog import PoiEmbeddingStore
from poi_gr.methods.tiger_joint.data import TigerJointDataError
from poi_gr.pid.dedup import sha256_file
from poi_gr.pid.geohash import GEOHASH_ALPHABET, encode_geohash
from poi_gr.sft.data import TimeSplit, discover_order_files, stable_sample_id


DATA_SCHEMA_VERSION = "tiger-joint-sid-free-data-v1"
SPECIAL_TOKENS_SCHEMA_VERSION = "tiger-joint-fresh-vocab-v1"
TASK_NAME = "tiger_joint_history_query_gid_poi_row_to_dynamic_sid"
BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")
DYNAMIC_HISTORY_IDENTIFIER = "<POI_SID><S1_0><S2_0><S3_0></POI_SID>"
DYNAMIC_TARGET_IDENTIFIER = "<TARGET_POI><S1_0><S2_0><S3_0></TARGET_POI>"
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
    "<POI_SID>",
    "</POI_SID>",
    "<CURRENT>",
    "</CURRENT>",
    "<TARGET_POI>",
    "</TARGET_POI>",
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


class TigerJointDatasetError(TigerJointDataError):
    """Raised when a raw order violates the SID-free data contract."""


@dataclass(frozen=True)
class HistoryWindow:
    """Closed local-date range used for historical interactions."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise TigerJointDatasetError("history_start 不能晚于 history_end")

    def contains(self, value: datetime) -> bool:
        return self.start <= value.date() <= self.end

    def to_manifest(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True)
class SidFreeDataResult:
    """Completed Train/Valid SID-free data artifacts."""

    manifest: dict[str, Any]
    stats: dict[str, Any]


def stable_user_bucket(passenger_id: str, bucket_count: int = 2000) -> int:
    """Hash one passenger identity into the fixed user-token vocabulary."""

    if not isinstance(passenger_id, str) or not passenger_id:
        raise TigerJointDatasetError("passenger_id 必须是非空字符串")
    if bucket_count <= 0:
        raise TigerJointDatasetError("user_bucket_count 必须大于 0")
    digest = hashlib.sha256(passenger_id.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big") % bucket_count


def user_token(passenger_id: str, bucket_count: int = 2000) -> str:
    """Return the zero-padded hashed user token."""

    width = max(4, len(str(bucket_count - 1)))
    return f"<U_{stable_user_bucket(passenger_id, bucket_count):0{width}d}>"


def build_joint_special_tokens(
    *,
    codebook_sizes: Sequence[int] = (1024, 1024, 1024),
    user_bucket_count: int = 2000,
) -> dict[str, Any]:
    """Build a fresh three-level vocabulary with no collision namespace."""

    if len(codebook_sizes) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in codebook_sizes
    ):
        raise TigerJointDatasetError("codebook_sizes 必须包含三个正整数")
    if user_bucket_count <= 0:
        raise TigerJointDatasetError("user_bucket_count 必须大于 0")
    geohash_tokens = [f"<G_{character}>" for character in GEOHASH_ALPHABET]
    width = max(4, len(str(user_bucket_count - 1)))
    user_tokens = [f"<U_{value:0{width}d}>" for value in range(user_bucket_count)]
    sid_tokens = {
        f"s{level}": [f"<S{level}_{value}>" for value in range(codebook_size)]
        for level, codebook_size in enumerate(codebook_sizes, start=1)
    }
    tokens = [
        *STRUCTURE_TOKENS,
        *geohash_tokens,
        *user_tokens,
        *sid_tokens["s1"],
        *sid_tokens["s2"],
        *sid_tokens["s3"],
    ]
    if len(tokens) != len(set(tokens)):
        raise TigerJointDatasetError("TIGER-Joint 特殊 Token 表包含重复项")
    if any(token.startswith("<C_") for token in tokens):
        raise AssertionError("fresh vocabulary 不得包含旧 collision Token")
    return {
        "schema_version": SPECIAL_TOKENS_SCHEMA_VERSION,
        "base_model": "Qwen3-0.6B",
        "initialization": "add_to_vanilla_qwen_with_fresh_rows",
        "structure_tokens": list(STRUCTURE_TOKENS),
        "geohash_tokens": geohash_tokens,
        "user_bucket_count": user_bucket_count,
        "user_tokens": user_tokens,
        "sid_token_capacities": list(codebook_sizes),
        "sid_tokens": sid_tokens,
        "collision_tokens": [],
        "additional_tokens": tokens,
        "token_count": len(tokens),
    }


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TigerJointDatasetError(f"{field_name} 必须是非空字符串")
    return value


def _require_query(value: Any, field_name: str) -> str:
    query = _require_string(value, field_name)
    if not query.strip():
        raise TigerJointDatasetError(f"{field_name} 不能为空白字符串")
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
        raise TigerJointDatasetError(f"{field_name} 必须位于 [{minimum}, {maximum}]")
    return float(value)


def _parse_target_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise TigerJointDatasetError(f"{field_name} 必须是时间字符串")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError as error:
        raise TigerJointDatasetError(
            f"{field_name} 必须为 YYYY-MM-DD HH:MM:SS"
        ) from error
    if parsed.strftime("%Y-%m-%d %H:%M:%S") != value:
        raise TigerJointDatasetError(f"{field_name} 必须为 YYYY-MM-DD HH:MM:SS")
    return parsed


def _parse_history_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise TigerJointDatasetError(f"{field_name} 必须是 ISO 时间字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise TigerJointDatasetError(f"{field_name} 不是合法 ISO 时间") from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(BEIJING_TIMEZONE).replace(tzinfo=None)
    return parsed


def _parse_source_date(value: Any) -> date:
    if not isinstance(value, str):
        raise TigerJointDatasetError("source_dt 必须是 YYYYMMDD 字符串")
    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError as error:
        raise TigerJointDatasetError("source_dt 必须为 YYYYMMDD") from error
    if parsed.strftime("%Y%m%d") != value:
        raise TigerJointDatasetError("source_dt 必须为 YYYYMMDD")
    return parsed


def _user_gid(longitude: float, latitude: float, length: int) -> str:
    geohash = encode_geohash(longitude, latitude, length)
    return "".join(f"<G_{character}>" for character in geohash)


def _normalize_history(
    record: Mapping[str, Any],
    *,
    current_time: datetime,
    history_window: HistoryWindow,
    max_history_events: int,
    embedding_store: PoiEmbeddingStore,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    history_length = record["history_length"]
    if (
        isinstance(history_length, bool)
        or not isinstance(history_length, int)
        or not 0 <= history_length <= max_history_events
    ):
        raise TigerJointDatasetError(
            f"history_length 必须位于 [0,{max_history_events}]"
        )
    raw_history = record["history_sequence"]
    if not isinstance(raw_history, list) or len(raw_history) != history_length:
        raise TigerJointDatasetError(
            "history_length 与 history_sequence 实际长度不一致"
        )
    normalized: list[dict[str, Any]] = []
    poi_rows: list[int] = []
    previous_time: datetime | None = None
    for index, raw_event in enumerate(raw_history):
        prefix = f"history_sequence[{index}]"
        if not isinstance(raw_event, dict):
            raise TigerJointDatasetError(f"{prefix} 必须是对象")
        missing = [field for field in REQUIRED_HISTORY_FIELDS if field not in raw_event]
        if missing:
            raise TigerJointDatasetError(f"{prefix} 缺少字段：" + ", ".join(missing))
        event_time = _parse_history_timestamp(
            raw_event["event_time"], f"{prefix}.event_time"
        )
        if not history_window.contains(event_time):
            raise TigerJointDatasetError(f"{prefix}.event_time 超出历史窗口")
        if event_time >= current_time:
            raise TigerJointDatasetError(f"{prefix}.event_time 未严格早于当前请求")
        if previous_time is not None and event_time < previous_time:
            raise TigerJointDatasetError("history_sequence 没有按时间正序排列")
        previous_time = event_time
        poi_id = _require_string(raw_event["poi_id"], f"{prefix}.poi_id")
        poi_rows.append(embedding_store.row_for_poi_id(poi_id))
        normalized.append(
            {
                "query": _require_query(raw_event["query"], f"{prefix}.query"),
                "disp_lng": _require_coordinate(
                    raw_event["disp_lng"], f"{prefix}.disp_lng", -180.0, 180.0
                ),
                "disp_lat": _require_coordinate(
                    raw_event["disp_lat"], f"{prefix}.disp_lat", -90.0, 90.0
                ),
            }
        )
    return normalized, tuple(poi_rows)


def format_joint_user_content(
    *,
    passenger_id: str,
    current_query: str,
    current_longitude: float,
    current_latitude: float,
    history: Sequence[Mapping[str, Any]],
    user_bucket_count: int = 2000,
    geohash_length: int = 6,
) -> str:
    """Linearize behavior with dynamic SID slots and no assigned SID values."""

    parts = [
        f"<USER_ID>{user_token(passenger_id, user_bucket_count)}</USER_ID>",
        "<HISTORY>",
    ]
    for event in history:
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
                DYNAMIC_HISTORY_IDENTIFIER,
                "</EVENT>",
            )
        )
    parts.extend(
        (
            "</HISTORY>",
            "<CURRENT>",
            "<USER_GID>"
            + _user_gid(current_longitude, current_latitude, geohash_length)
            + "</USER_GID>",
            f"<QUERY>{current_query}</QUERY>",
            "</CURRENT>",
        )
    )
    return "\n".join(parts)


def raw_order_to_sid_free_sample(
    record: Mapping[str, Any],
    *,
    relative_path: str,
    line_number: int,
    split: TimeSplit,
    history_window: HistoryWindow,
    embedding_store: PoiEmbeddingStore,
    max_history_events: int = 10,
    geohash_length: int = 6,
    user_bucket_count: int = 2000,
) -> dict[str, Any] | None:
    """Convert one raw Train/Valid order; return None for blank Query or Test."""

    missing = [field for field in REQUIRED_TARGET_FIELDS if field not in record]
    if missing:
        raise TigerJointDatasetError("缺少字段：" + ", ".join(missing))
    current_time = _parse_target_timestamp(record["create_time"], "create_time")
    source_date = _parse_source_date(record["source_dt"])
    if source_date != current_time.date():
        raise TigerJointDatasetError("source_dt 与 create_time 日期不一致")
    split_name = split.split_for(current_time.date())
    if split_name == "test":
        return None
    query_value = record["query"]
    if query_value is None or (
        isinstance(query_value, str) and not query_value.strip()
    ):
        return None
    current_query = _require_query(query_value, "query")
    passenger_id = _require_string(record["passenger_id"], "passenger_id")
    current_longitude = _require_coordinate(
        record["disp_lng"], "disp_lng", -180.0, 180.0
    )
    current_latitude = _require_coordinate(record["disp_lat"], "disp_lat", -90.0, 90.0)
    target_poi_id = _require_string(record["poi_id"], "poi_id")
    target_poi_row = embedding_store.row_for_poi_id(target_poi_id)
    history, history_poi_rows = _normalize_history(
        record,
        current_time=current_time,
        history_window=history_window,
        max_history_events=max_history_events,
        embedding_store=embedding_store,
    )
    user_content = format_joint_user_content(
        passenger_id=passenger_id,
        current_query=current_query,
        current_longitude=current_longitude,
        current_latitude=current_latitude,
        history=history,
        user_bucket_count=user_bucket_count,
        geohash_length=geohash_length,
    )
    if "<POI_TIGER_ID>" in user_content or "<C_" in user_content:
        raise AssertionError("SID-free 样本意外包含旧 TIGER identifier")
    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "sample_id": stable_sample_id(relative_path, line_number),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": DYNAMIC_TARGET_IDENTIFIER},
        ],
        "order_id": _require_string(record["order_id"], "order_id"),
        "searchid": _require_string(record["searchid"], "searchid"),
        "target_poi_id": target_poi_id,
        "history_poi_rows": list(history_poi_rows),
        "target_poi_row": target_poi_row,
        "history_length": len(history_poi_rows),
        "split": split_name,
    }


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


def build_sid_free_training_data(
    orders_dir: Path,
    embedding_dir: Path,
    output_dir: Path,
    split: TimeSplit,
    history_window: HistoryWindow,
    *,
    max_history_events: int = 10,
    geohash_length: int = 6,
    user_bucket_count: int = 2000,
    codebook_sizes: Sequence[int] = (1024, 1024, 1024),
    max_retained_per_split: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> SidFreeDataResult:
    """Stream raw orders into isolated Train/Valid dynamic templates."""

    if history_window.end >= split.train_start:
        raise TigerJointDatasetError("历史窗口必须完整早于训练窗口")
    if max_history_events != 10 or geohash_length != 6:
        raise TigerJointDatasetError("第一版固定 history=10、Geohash=6")
    if user_bucket_count != 2000:
        raise TigerJointDatasetError("第一版固定 2,000 个用户 Token")
    if max_retained_per_split is not None and max_retained_per_split <= 0:
        raise TigerJointDatasetError("max_retained_per_split 必须大于 0")
    orders_dir = orders_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise TigerJointDatasetError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    order_files = discover_order_files(orders_dir)
    embedding_store = PoiEmbeddingStore.from_directory(embedding_dir)
    special_tokens = build_joint_special_tokens(
        codebook_sizes=codebook_sizes,
        user_bucket_count=user_bucket_count,
    )
    stats: dict[str, Any] = {
        "scanned_order_count": 0,
        "test_record_skipped_before_behavior_formatting": 0,
        "blank_query_count": 0,
        "train_count": 0,
        "valid_count": 0,
        "history_event_count": 0,
        "history_length_histogram": {
            str(value): 0 for value in range(max_history_events + 1)
        },
    }
    input_files: list[dict[str, Any]] = []
    stopped_early = False

    def limits_reached() -> bool:
        return max_retained_per_split is not None and all(
            stats[f"{name}_count"] >= max_retained_per_split
            for name in ("train", "valid")
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.building-",
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        handles = {
            split_name: (temporary_dir / f"{split_name}.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            )
            for split_name in ("train", "valid")
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
                        try:
                            record = json.loads(raw_line)
                            if not isinstance(record, dict):
                                raise TigerJointDatasetError("记录必须是 JSON 对象")
                            raw_split = split.split_for(
                                _parse_source_date(record.get("source_dt"))
                            )
                            if raw_split == "test":
                                stats[
                                    "test_record_skipped_before_behavior_formatting"
                                ] += 1
                                continue
                            if (
                                max_retained_per_split is not None
                                and stats[f"{raw_split}_count"]
                                >= max_retained_per_split
                            ):
                                continue
                            sample = raw_order_to_sid_free_sample(
                                record,
                                relative_path=relative_path,
                                line_number=line_number,
                                split=split,
                                history_window=history_window,
                                embedding_store=embedding_store,
                                max_history_events=max_history_events,
                                geohash_length=geohash_length,
                                user_bucket_count=user_bucket_count,
                            )
                        except (
                            UnicodeDecodeError,
                            json.JSONDecodeError,
                            TigerJointDataError,
                        ) as error:
                            raise TigerJointDatasetError(
                                f"{relative_path}:{line_number} 数据校验失败：{error}"
                            ) from error
                        if sample is None:
                            stats["blank_query_count"] += 1
                            continue
                        split_name = str(sample["split"])
                        handles[split_name].write(
                            json.dumps(
                                sample,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        history_length = int(sample["history_length"])
                        stats[f"{split_name}_count"] += 1
                        stats["history_event_count"] += history_length
                        stats["history_length_histogram"][str(history_length)] += 1
                        if limits_reached():
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
                        f"TIGER-Joint SID-free：{file_index}/{len(order_files)} "
                        f"分片，Train={stats['train_count']:,}，"
                        f"Valid={stats['valid_count']:,}"
                    )
                if stopped_early:
                    break
        finally:
            for handle in handles.values():
                handle.close()

        if stats["train_count"] <= 0 or stats["valid_count"] <= 0:
            raise TigerJointDatasetError("Train/Valid 都必须至少保留一条样本")
        if max_retained_per_split is not None and not limits_reached():
            raise TigerJointDatasetError("未达到 Train/Valid smoke 样本上限")
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
            for name in (
                "train.jsonl",
                "valid.jsonl",
                "special_tokens.json",
                "stats.json",
            )
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "schema_version": DATA_SCHEMA_VERSION,
                    "input_files": input_files,
                    "embedding_signature": embedding_store.manifest_signature,
                    "poi_ids_sha256": embedding_store.poi_ids_sha256,
                    "time_split": split.to_manifest(),
                    "history_window": history_window.to_manifest(),
                    "codebook_sizes": list(codebook_sizes),
                    "max_retained_per_split": max_retained_per_split,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": DATA_SCHEMA_VERSION,
            "status": "completed",
            "task_name": TASK_NAME,
            "build_fingerprint": fingerprint,
            "isolation": {
                "source": "raw_orders_plus_bge_row_order",
                "old_tiger_jsonl_loaded": False,
                "old_sid_mapping_loaded": False,
                "old_rqvae_checkpoint_loaded": False,
                "old_qwen_tiger_checkpoint_loaded": False,
            },
            "input": {
                "orders_dir": orders_dir.name,
                "files": input_files,
                "scan_mode": (
                    "full" if max_retained_per_split is None else "prefix_smoke"
                ),
            },
            "embedding_catalog": {
                "directory": str(Path(embedding_dir).resolve()),
                "manifest_signature": embedding_store.manifest_signature,
                "poi_ids_sha256": embedding_store.poi_ids_sha256,
                "shape": list(embedding_store.shape),
                "row_source": "BGE poi_ids.jsonl",
            },
            "time_split": split.to_manifest(),
            "history": {
                "window": history_window.to_manifest(),
                "max_events": max_history_events,
            },
            "dynamic_sid": {
                "assignment_source": "current_fresh_rqvae_at_runtime",
                "codebook_sizes": list(codebook_sizes),
                "history_placeholder": DYNAMIC_HISTORY_IDENTIFIER,
                "target_placeholder": DYNAMIC_TARGET_IDENTIFIER,
                "collision_level": None,
            },
            "test_policy": {
                "jsonl_written": False,
                "available_to_training": False,
                "raw_rows_skipped_before_behavior_formatting": stats[
                    "test_record_skipped_before_behavior_formatting"
                ],
            },
            "outputs": outputs,
        }
        _write_json(temporary_dir / "manifest.json", manifest)
        os.replace(temporary_dir, output_dir)
    return SidFreeDataResult(manifest=manifest, stats=stats)

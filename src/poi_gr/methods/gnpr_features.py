"""Build GNPR-SID discrete POI features from the fixed history-10 dataset."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")
DEFAULT_PLUS_CODE_LENGTH = 6
DEFAULT_TOP_K = 10
PLUS_CODE_ALPHABET = "23456789CFGHJMPQRVWX"


class GnprFeatureError(ValueError):
    """Raised when POI or order data violates the GNPR feature contract."""


@dataclass(frozen=True)
class GnprInteraction:
    """One user-to-POI interaction used by the GNPR feature builder."""

    passenger_id: str
    poi_id: str
    event_hour: int
    source: str


@dataclass(frozen=True)
class GnprPoiFeature:
    """Sparse four-part GNPR representation for one POI."""

    poi_id: str
    category_code: str
    category_index: int
    plus_code_region: str
    region_index: int
    top_visit_hours: tuple[int, ...]
    top_visitor_ids: tuple[str, ...]
    top_visitor_indices: tuple[int, ...]
    interaction_count: int


@dataclass(frozen=True)
class GnprFeatureDataset:
    """Deterministic feature rows, vocabularies, and construction statistics."""

    rows: tuple[GnprPoiFeature, ...]
    category_vocab: Mapping[str, int]
    region_vocab: Mapping[str, int]
    user_vocab: Mapping[str, int]
    stats: Mapping[str, int]


def _required_text(value: Any, field: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise GnprFeatureError(f"{field} 必须是非空字符串")
    return text


def _parse_local_datetime(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise GnprFeatureError(f"{field} 不是有效时间：{value}") from error
    else:
        raise GnprFeatureError(f"{field} 必须是有效时间")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=BEIJING_TIMEZONE)
    return parsed.astimezone(BEIJING_TIMEZONE)


def _parse_source_date(value: Any, fallback_time: Any) -> date:
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 8 and text.isdigit():
            try:
                return datetime.strptime(text, "%Y%m%d").date()
            except ValueError:
                pass
    return _parse_local_datetime(fallback_time, "create_time").date()


def _coalesced_target_time(record: Mapping[str, Any]) -> datetime:
    value = record.get("birth_time") or record.get("create_time")
    return _parse_local_datetime(value, "birth_time/create_time")


def encode_plus_code_region(
    longitude: Any,
    latitude: Any,
    *,
    code_length: int = DEFAULT_PLUS_CODE_LENGTH,
) -> str:
    """Encode one coordinate as the GNPR spatial region identifier."""

    if code_length != DEFAULT_PLUS_CODE_LENGTH:
        raise GnprFeatureError("GNPR baseline 的 Plus Code 长度固定为 6")
    try:
        lon = float(longitude)
        lat = float(latitude)
    except (TypeError, ValueError) as error:
        raise GnprFeatureError("POI 经纬度必须是数值") from error
    if not math.isfinite(lon) or not math.isfinite(lat):
        raise GnprFeatureError("POI 经纬度必须是有限数值")
    if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
        raise GnprFeatureError("POI 经纬度超出合法范围")
    if lat == 90.0:
        lat -= 0.05
    lon = ((lon + 180.0) % 360.0) - 180.0
    latitude_value = lat + 90.0
    longitude_value = lon + 180.0
    place_value = 20.0
    characters: list[str] = []
    for _ in range(code_length // 2):
        latitude_digit = int(latitude_value / place_value)
        longitude_digit = int(longitude_value / place_value)
        characters.append(PLUS_CODE_ALPHABET[latitude_digit])
        characters.append(PLUS_CODE_ALPHABET[longitude_digit])
        latitude_value -= latitude_digit * place_value
        longitude_value -= longitude_digit * place_value
        place_value /= 20.0
    return "".join(characters) + "00+"


def _history_signature(history: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, ...], ...]:
    signature: list[tuple[str, ...]] = []
    for index, event in enumerate(history):
        if not isinstance(event, Mapping):
            raise GnprFeatureError(f"history_sequence[{index}] 必须是 object")
        event_time = _parse_local_datetime(
            event.get("event_time"),
            f"history_sequence[{index}].event_time",
        ).isoformat()
        signature.append(
            (
                _required_text(event.get("order_id"), "history.order_id"),
                _required_text(event.get("searchid"), "history.searchid"),
                _required_text(event.get("poi_id"), "history.poi_id"),
                event_time,
            )
        )
    return tuple(signature)


def extract_gnpr_interactions(
    order_records: Iterable[Mapping[str, Any]],
    *,
    train_start: date,
    train_end: date,
    max_history_events: int = 10,
) -> tuple[tuple[GnprInteraction, ...], dict[str, int]]:
    """Extract deduplicated fixed histories plus train-window target events."""

    if train_start > train_end:
        raise GnprFeatureError("train_start 不能晚于 train_end")
    if max_history_events <= 0:
        raise GnprFeatureError("max_history_events 必须大于 0")

    interactions: list[GnprInteraction] = []
    history_signature_by_user: dict[str, tuple[tuple[str, ...], ...]] = {}
    history_event_count = 0
    target_event_count = 0
    order_record_count = 0

    for record_index, record in enumerate(order_records):
        if not isinstance(record, Mapping):
            raise GnprFeatureError(f"order_records[{record_index}] 必须是 object")
        order_record_count += 1
        passenger_id = _required_text(record.get("passenger_id"), "passenger_id")
        history = record.get("history_sequence")
        if not isinstance(history, list):
            raise GnprFeatureError("history_sequence 必须是列表")
        if len(history) > max_history_events:
            raise GnprFeatureError(
                f"history_sequence 长度 {len(history)} 超过 {max_history_events}"
            )

        signature = _history_signature(history)
        previous_signature = history_signature_by_user.get(passenger_id)
        if previous_signature is None:
            history_signature_by_user[passenger_id] = signature
            seen_history_keys: set[tuple[str, ...]] = set()
            for event, event_key in zip(history, signature):
                if event_key in seen_history_keys:
                    continue
                seen_history_keys.add(event_key)
                event_time = _parse_local_datetime(
                    event.get("event_time"),
                    "history.event_time",
                )
                interactions.append(
                    GnprInteraction(
                        passenger_id=passenger_id,
                        poi_id=_required_text(event.get("poi_id"), "history.poi_id"),
                        event_hour=event_time.hour,
                        source="history",
                    )
                )
                history_event_count += 1
        elif previous_signature != signature:
            raise GnprFeatureError(
                f"同一 passenger_id 的固定 history_sequence 不一致：{passenger_id}"
            )

        target_time_value = record.get("birth_time") or record.get("create_time")
        target_date = _parse_source_date(record.get("source_dt"), target_time_value)
        if train_start <= target_date <= train_end:
            target_time = _coalesced_target_time(record)
            interactions.append(
                GnprInteraction(
                    passenger_id=passenger_id,
                    poi_id=_required_text(record.get("poi_id"), "poi_id"),
                    event_hour=target_time.hour,
                    source="target_train",
                )
            )
            target_event_count += 1

    return (
        tuple(interactions),
        {
            "order_record_count": order_record_count,
            "history_user_count": len(history_signature_by_user),
            "history_event_count": history_event_count,
            "target_train_event_count": target_event_count,
            "interaction_count": len(interactions),
        },
    )


def _top_keys(counter: Counter[Any], limit: int) -> tuple[Any, ...]:
    return tuple(
        key
        for key, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    )


def build_gnpr_feature_dataset(
    poi_records: Iterable[Mapping[str, Any]],
    interactions: Iterable[GnprInteraction],
    *,
    plus_code_length: int = DEFAULT_PLUS_CODE_LENGTH,
    top_visit_hours: int = DEFAULT_TOP_K,
    top_visitors: int = DEFAULT_TOP_K,
) -> GnprFeatureDataset:
    """Build deterministic sparse GNPR features for every catalog POI."""

    if top_visit_hours <= 0 or top_visitors <= 0:
        raise GnprFeatureError("top_visit_hours 和 top_visitors 必须大于 0")

    static_rows: dict[str, tuple[str, str]] = {}
    for index, record in enumerate(poi_records):
        if not isinstance(record, Mapping):
            raise GnprFeatureError(f"poi_records[{index}] 必须是 object")
        poi_id = _required_text(record.get("poi_id"), "poi_id")
        if poi_id in static_rows:
            raise GnprFeatureError(f"POI ID 重复：{poi_id}")
        category_code = _required_text(record.get("category_code"), "category_code")
        region = encode_plus_code_region(
            record.get("lng"),
            record.get("lat"),
            code_length=plus_code_length,
        )
        static_rows[poi_id] = (category_code, region)

    hour_counts: dict[str, Counter[int]] = defaultdict(Counter)
    user_counts: dict[str, Counter[str]] = defaultdict(Counter)
    interaction_counts: Counter[str] = Counter()
    observed_users: set[str] = set()
    unmatched_interaction_count = 0
    matched_interaction_count = 0
    for interaction in interactions:
        if interaction.poi_id not in static_rows:
            unmatched_interaction_count += 1
            continue
        if not 0 <= interaction.event_hour <= 23:
            raise GnprFeatureError("event_hour 必须位于 [0, 23]")
        hour_counts[interaction.poi_id][interaction.event_hour] += 1
        user_counts[interaction.poi_id][interaction.passenger_id] += 1
        interaction_counts[interaction.poi_id] += 1
        observed_users.add(interaction.passenger_id)
        matched_interaction_count += 1

    categories = sorted({value[0] for value in static_rows.values()})
    regions = sorted({value[1] for value in static_rows.values()})
    users = sorted(observed_users)
    category_vocab = {value: index for index, value in enumerate(categories)}
    region_vocab = {value: index for index, value in enumerate(regions)}
    user_vocab = {value: index for index, value in enumerate(users)}

    rows: list[GnprPoiFeature] = []
    behavior_poi_count = 0
    for poi_id in sorted(static_rows):
        category_code, region = static_rows[poi_id]
        top_hours = _top_keys(hour_counts[poi_id], top_visit_hours)
        top_user_ids = _top_keys(user_counts[poi_id], top_visitors)
        interaction_count = interaction_counts[poi_id]
        if interaction_count:
            behavior_poi_count += 1
        rows.append(
            GnprPoiFeature(
                poi_id=poi_id,
                category_code=category_code,
                category_index=category_vocab[category_code],
                plus_code_region=region,
                region_index=region_vocab[region],
                top_visit_hours=top_hours,
                top_visitor_ids=top_user_ids,
                top_visitor_indices=tuple(user_vocab[value] for value in top_user_ids),
                interaction_count=interaction_count,
            )
        )

    return GnprFeatureDataset(
        rows=tuple(rows),
        category_vocab=category_vocab,
        region_vocab=region_vocab,
        user_vocab=user_vocab,
        stats={
            "poi_count": len(rows),
            "category_count": len(category_vocab),
            "region_count": len(region_vocab),
            "user_count": len(user_vocab),
            "behavior_poi_count": behavior_poi_count,
            "matched_interaction_count": matched_interaction_count,
            "unmatched_interaction_count": unmatched_interaction_count,
        },
    )

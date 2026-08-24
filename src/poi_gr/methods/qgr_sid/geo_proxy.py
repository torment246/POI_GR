"""Static-POI GEO and request-GID proxy for QGR-SID M2-B."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from poi_gr.methods.qgr_sid.compiler import branch_entropy_bits
from poi_gr.methods.qgr_sid.proxy import (
    ORDER_REQUIRED_FIELDS,
    PROXY_SCHEMA_VERSION,
    QgrSidProxyError,
    StrictRelationCatalog,
    TemporalSplit,
    _compile_all_buckets,
    _git_state,
    _json_dump,
    _load_json,
    _query_arrays,
    _rank_for_event,
    _ranking_metrics,
    _ratio,
    _save_array,
    _signature,
    _split_tie_key,
    _utc_now,
    validate_proxy_output,
)
from poi_gr.methods.qgr_sid.relations import (
    DIRECT_NUMERIC_RELATION_TYPES,
    RELATION_TYPES,
    extract_query_relation_features,
)
from poi_gr.methods.tiger.identifier import sha256_file
from poi_gr.pid.geohash import (
    GEOHASH_ALPHABET,
    encode_geohash,
    geohash_to_tokens,
)


GEO_PROXY_SCHEMA_VERSION = "qgr-sid-geo-proxy-v1"
GEO_PROXY_METRICS_SCHEMA_VERSION = "qgr-sid-geo-proxy-metrics-v1"
GEO_RELATION_INDEX = len(RELATION_TYPES)
HYBRID_RELATION_TYPES = RELATION_TYPES + ("R_GEO",)
GEO_CANDIDATE_LENGTHS = (1, 2)


class QgrSidGeoProxyError(QgrSidProxyError):
    """Raised when M2-B GEO/GID inputs or outputs violate the protocol."""


@dataclass(frozen=True)
class GeoProxyResult:
    """Completed M2-B artifacts."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


@dataclass(frozen=True)
class GeoSelection:
    """One static R_GEO value selection for every compact collision POI."""

    values: np.ndarray
    lengths: np.ndarray
    matches: np.ndarray
    resolved: np.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _GeoHoldoutEvent:
    collision_index: int
    query: str
    request_gid: tuple[int, ...]


def derive_geo_candidates(
    *,
    bucket_keys: np.ndarray,
    poi_gid_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive one/two-character numeric GEO values after each bucket prefix."""

    bucket_keys = np.asarray(bucket_keys)
    poi_gid_codes = np.asarray(poi_gid_codes)
    if bucket_keys.ndim != 1:
        raise QgrSidGeoProxyError("bucket_keys 必须是一维")
    if poi_gid_codes.shape != (len(bucket_keys), 6):
        raise QgrSidGeoProxyError("poi_gid_codes 必须是 [N,6]")
    if np.any(poi_gid_codes >= 32):
        raise QgrSidGeoProxyError("POI GID Token 必须位于 [0,32)")

    order = np.argsort(bucket_keys, kind="stable")
    sorted_keys = bucket_keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]
    prefix_lengths = np.full(len(bucket_keys), 6, dtype=np.uint8)
    candidates = np.full((len(bucket_keys), 2), -1, dtype=np.int16)
    for start, stop in zip(starts, stops, strict=True):
        indices = order[start:stop]
        codes = poi_gid_codes[indices]
        equal_columns = np.all(codes == codes[0], axis=0)
        different = np.flatnonzero(~equal_columns)
        prefix = 6 if not len(different) else int(different[0])
        prefix_lengths[indices] = prefix
        if prefix < 6:
            candidates[indices, 0] = codes[:, prefix]
        if prefix < 5:
            candidates[indices, 1] = (
                32
                + codes[:, prefix].astype(np.int16) * 32
                + codes[:, prefix + 1].astype(np.int16)
            )
    if np.any(candidates > 1055):
        raise QgrSidGeoProxyError("两字符 GEO 值超出冻结范围 32—1055")
    return prefix_lengths, candidates, order, starts


def _load_m2a_arrays(
    m2a_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    validate_proxy_output(m2a_dir)
    manifest = _load_json(m2a_dir / "manifest.json", "M2-A manifest")
    metrics = _load_json(m2a_dir / "metrics.json", "M2-A metrics")
    arrays_contract = manifest.get("outputs", {}).get("arrays")
    if not isinstance(arrays_contract, dict):
        raise QgrSidGeoProxyError("M2-A manifest 缺少 arrays")
    required = (
        "collision_poi_ids",
        "base_sid_keys",
        "strict_relations",
        "conflict_masks",
        "early_order_counts",
        "early_relation_match_counts",
        "static_path_types",
        "static_path_values",
        "static_resolved",
        "query_guided_path_types",
        "query_guided_path_values",
        "query_guided_resolved",
    )
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        contract = arrays_contract.get(name)
        if not isinstance(contract, dict):
            raise QgrSidGeoProxyError(f"M2-A 缺少数组：{name}")
        arrays[name] = np.load(
            m2a_dir / str(contract["path"]), mmap_mode="r", allow_pickle=False
        )
    return arrays, manifest, metrics


def extract_collision_gid_codes(
    *,
    gid_codes_path: Path,
    gid_manifest_path: Path,
    identifier_dir: Path,
    collision_poi_ids: np.ndarray,
    batch_rows: int,
    progress: Callable[[str], None] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate full POI row alignment and compact GID6 to collision POIs."""

    gid_manifest = _load_json(gid_manifest_path, "GID manifest")
    if gid_manifest.get("schema_version") != "geohash-pid-v1":
        raise QgrSidGeoProxyError("GID manifest schema_version 不受支持")
    if gid_manifest.get("status") != "completed":
        raise QgrSidGeoProxyError("GID manifest 尚未 completed")
    contract = gid_manifest.get("gid_codes")
    if not isinstance(contract, dict):
        raise QgrSidGeoProxyError("GID manifest 缺少 gid_codes 契约")
    gid_codes = np.load(gid_codes_path, mmap_mode="r", allow_pickle=False)
    if list(gid_codes.shape) != contract.get("shape") or str(
        gid_codes.dtype
    ) != contract.get("dtype"):
        raise QgrSidGeoProxyError("GID codes shape/dtype 与 manifest 不一致")
    if gid_codes.shape[1] != 6 or np.any(gid_codes >= 32):
        raise QgrSidGeoProxyError("仅支持合法 Geohash6 codes")

    poi_ids_contract = gid_manifest.get("poi_ids")
    if not isinstance(poi_ids_contract, dict):
        raise QgrSidGeoProxyError("GID manifest 缺少 poi_ids 契约")
    gid_poi_ids_path = Path(str(poi_ids_contract.get("path", ""))).resolve()
    if sha256_file(gid_poi_ids_path) != poi_ids_contract.get("sha256"):
        raise QgrSidGeoProxyError("GID POI IDs SHA256 与 manifest 不一致")

    mapping_path = identifier_dir / "poi_tiger_id_mapping.parquet"
    parquet = pq.ParquetFile(mapping_path)
    if parquet.metadata.num_rows != len(gid_codes):
        raise QgrSidGeoProxyError("GID 与 TIGER mapping 全量行数不一致")
    output = np.empty((len(collision_poi_ids), 6), dtype=np.uint8)
    compact_row = 0
    scanned = 0
    with gid_poi_ids_path.open("r", encoding="utf-8") as poi_id_handle:
        for batch in parquet.iter_batches(
            batch_size=batch_rows,
            columns=["poi_id", "base_sid_bucket_size"],
        ):
            mapping_ids = batch.column(0).to_pylist()
            bucket_sizes = batch.column(1).to_numpy(zero_copy_only=False)
            for offset, mapping_id in enumerate(mapping_ids):
                line = poi_id_handle.readline()
                if not line:
                    raise QgrSidGeoProxyError("GID POI IDs 早于 TIGER mapping 结束")
                try:
                    gid_poi_id = json.loads(line)
                except json.JSONDecodeError as error:
                    raise QgrSidGeoProxyError("GID POI IDs JSON 解析失败") from error
                if str(gid_poi_id) != str(mapping_id):
                    raise QgrSidGeoProxyError(
                        f"GID/TIGER 全量行对齐失败：row={scanned}"
                    )
                if int(bucket_sizes[offset]) > 1:
                    if compact_row >= len(collision_poi_ids):
                        raise QgrSidGeoProxyError("实际碰撞 POI 超过 M2-A")
                    if int(mapping_id) != int(collision_poi_ids[compact_row]):
                        raise QgrSidGeoProxyError("GID/TIGER/M2-A 碰撞行对齐失败")
                    output[compact_row] = gid_codes[scanned]
                    compact_row += 1
                scanned += 1
            if progress is not None and scanned % (batch_rows * 8) == 0:
                progress(
                    f"GID 已对齐 {scanned:,}/{len(gid_codes):,} POI；"
                    f"碰撞 POI {compact_row:,}/{len(collision_poi_ids):,}"
                )
        if poi_id_handle.readline():
            raise QgrSidGeoProxyError("GID POI IDs 行数多于 TIGER mapping")
    if scanned != len(gid_codes) or compact_row != len(collision_poi_ids):
        raise QgrSidGeoProxyError("GID 紧凑抽取计数不守恒")
    return output, {
        "gid_codes": {
            "path": str(gid_codes_path),
            "sha256": sha256_file(gid_codes_path),
            "shape": list(gid_codes.shape),
            "dtype": str(gid_codes.dtype),
        },
        "gid_manifest": {
            "path": str(gid_manifest_path),
            "sha256": sha256_file(gid_manifest_path),
        },
        "gid_poi_ids": {
            "path": str(gid_poi_ids_path),
            "sha256": sha256_file(gid_poi_ids_path),
            "rows": scanned,
        },
    }


def _encode_request_gid(record: Mapping[str, Any]) -> tuple[int, ...]:
    try:
        longitude = float(record["disp_lng"])
        latitude = float(record["disp_lat"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise QgrSidGeoProxyError("订单请求坐标非法") from error
    try:
        return tuple(
            int(value)
            for value in geohash_to_tokens(
                encode_geohash(longitude, latitude, length=6)
            )
        )
    except ValueError as error:
        raise QgrSidGeoProxyError("订单请求坐标无法编码为 Geohash6") from error


def _request_geo_values(
    request_gid: Sequence[int],
    prefix: int,
) -> tuple[int, int]:
    if not 0 <= prefix <= 6 or len(request_gid) != 6:
        raise QgrSidGeoProxyError("请求 GEO prefix/GID 非法")
    one = -1 if prefix >= 6 else int(request_gid[prefix])
    two = (
        -1
        if prefix >= 5
        else 32 + int(request_gid[prefix]) * 32 + int(request_gid[prefix + 1])
    )
    return one, two


def _read_geo_order_events(
    *,
    order_dir: Path,
    sft_manifest: Mapping[str, Any],
    split: TemporalSplit,
    collision_poi_ids: np.ndarray,
    prefix_lengths: np.ndarray,
    geo_candidates: np.ndarray,
    expected_early_orders: np.ndarray,
    progress: Callable[[str], None] | None,
) -> tuple[np.ndarray, list[_GeoHoldoutEvent], dict[str, Any], list[dict[str, Any]]]:
    collision_lookup = {
        int(poi_id): index for index, poi_id in enumerate(collision_poi_ids)
    }
    early_orders = np.zeros(len(collision_poi_ids), dtype=np.int32)
    early_geo_matches = np.zeros((len(collision_poi_ids), 2), dtype=np.int32)
    holdout_events: list[_GeoHoldoutEvent] = []
    tie_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    orders_contract = sft_manifest.get("orders")
    if not isinstance(orders_contract, dict):
        raise QgrSidGeoProxyError("SFT manifest 缺少 orders")
    expected_files = {
        str(item.get("relative_path")): item
        for item in orders_contract.get("files", [])
        if isinstance(item, dict)
    }
    actual_paths = tuple(sorted(order_dir.glob("part-*.json")))
    if set(path.name for path in actual_paths) != set(expected_files):
        raise QgrSidGeoProxyError("订单文件集合与 SFT manifest 不一致")
    source_outputs: list[dict[str, Any]] = []

    def process(record: Mapping[str, Any], split_name: str) -> None:
        counts[f"{split_name}_orders"] += 1
        try:
            poi_id = int(record["poi_id"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise QgrSidGeoProxyError("订单 poi_id 非法") from error
        target = collision_lookup.get(poi_id)
        if target is None:
            return
        counts[f"{split_name}_collision_orders"] += 1
        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise QgrSidGeoProxyError("碰撞订单 Query 为空")
        request_gid = _encode_request_gid(record)
        if split_name == "holdout":
            holdout_events.append(
                _GeoHoldoutEvent(
                    collision_index=target,
                    query=query,
                    request_gid=request_gid,
                )
            )
            return
        early_orders[target] += 1
        one, two = _request_geo_values(
            request_gid, int(prefix_lengths[target])
        )
        if geo_candidates[target, 0] >= 0 and one == geo_candidates[target, 0]:
            early_geo_matches[target, 0] += 1
        if geo_candidates[target, 1] >= 0 and two == geo_candidates[target, 1]:
            early_geo_matches[target, 1] += 1

    scanned = 0
    required_fields = (*ORDER_REQUIRED_FIELDS, "disp_lng", "disp_lat")
    for path in actual_paths:
        digest = hashlib.sha256()
        file_rows = 0
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise QgrSidGeoProxyError(
                        f"订单 JSON 解析失败：{path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise QgrSidGeoProxyError("订单 JSON 行必须是对象")
                missing = [field for field in required_fields if field not in record]
                if missing:
                    raise QgrSidGeoProxyError("订单缺少字段：" + ", ".join(missing))
                create_time = str(record["create_time"])
                source_dt = str(record["source_dt"])
                if (
                    len(create_time) < 16
                    or source_dt != create_time[:10].replace("-", "")
                ):
                    raise QgrSidGeoProxyError("订单 source_dt/create_time 不一致")
                file_rows += 1
                scanned += 1
                date = create_time[:10]
                if not "2026-07-01" <= date <= "2026-07-12":
                    counts["non_train_orders"] += 1
                    continue
                minute = create_time[:16]
                if minute < split.cutoff_minute:
                    process(record, "early")
                elif minute > split.cutoff_minute:
                    process(record, "holdout")
                else:
                    tie_records.append(record)
                if progress is not None and scanned % 500_000 == 0:
                    progress(
                        f"GID 订单已扫描 {scanned:,}；early/holdout 碰撞请求 "
                        f"{counts['early_collision_orders']:,}/"
                        f"{counts['holdout_collision_orders']:,}"
                    )
        contract = expected_files[path.name]
        digest_value = digest.hexdigest()
        if file_rows != int(contract.get("rows", -1)):
            raise QgrSidGeoProxyError("订单行数与 SFT manifest 不一致")
        if digest_value != contract.get("sha256"):
            raise QgrSidGeoProxyError("订单 SHA256 与 SFT manifest 不一致")
        source_outputs.append(
            {
                "path": str(path.resolve()),
                "rows": file_rows,
                "sha256": digest_value,
            }
        )

    if len(tie_records) != split.cutoff_minute_order_count:
        raise QgrSidGeoProxyError("切分分钟订单数与 M2-A 不一致")
    tie_records.sort(key=_split_tie_key)
    for index, record in enumerate(tie_records):
        process(
            record,
            "early" if index < split.early_from_cutoff_minute else "holdout",
        )
    if counts["early_orders"] != split.early_target_count or counts[
        "holdout_orders"
    ] != split.holdout_target_count:
        raise QgrSidGeoProxyError("M2-B 时间切分计数不守恒")
    if not np.array_equal(early_orders, expected_early_orders):
        raise QgrSidGeoProxyError("M2-B early POI 请求计数与 M2-A 不一致")
    return early_geo_matches, holdout_events, dict(counts), source_outputs


def select_geo_encoding(
    *,
    geo_candidates: np.ndarray,
    early_orders: np.ndarray,
    early_geo_matches: np.ndarray,
    bucket_order: np.ndarray,
    bucket_starts: np.ndarray,
    query_guided: bool,
    p99_order_count: float,
    prior_orders: float,
) -> GeoSelection:
    """Select the shortest useful 1/2-char R_GEO encoding per base bucket."""

    rows = len(early_orders)
    if geo_candidates.shape != (rows, 2) or early_geo_matches.shape != (rows, 2):
        raise QgrSidGeoProxyError("GEO candidate/match shape 非法")
    weights = np.maximum(
        1.0,
        np.sqrt(1.0 + np.minimum(early_orders, p99_order_count)),
    )
    global_predictability = np.zeros(2, dtype=np.float64)
    for candidate_index in range(2):
        available = geo_candidates[:, candidate_index] >= 0
        support = int(early_orders[available].sum())
        matched = int(early_geo_matches[available, candidate_index].sum())
        global_predictability[candidate_index] = _ratio(matched, support)

    values = np.full(rows, -1, dtype=np.int16)
    lengths = np.zeros(rows, dtype=np.uint8)
    selected_matches = np.zeros(rows, dtype=np.int32)
    resolved = np.zeros(rows, dtype=np.bool_)
    stops = np.r_[bucket_starts[1:], len(bucket_order)]
    selected_bucket_counts = Counter()
    for start, stop in zip(bucket_starts, stops, strict=True):
        indices = bucket_order[start:stop]
        best_candidate: int | None = None
        best_score = -math.inf
        for candidate_index in range(2):
            candidate_values = geo_candidates[indices, candidate_index]
            available = candidate_values >= 0
            if len(np.unique(candidate_values[available])) < 2:
                continue
            score = branch_entropy_bits(candidate_values, weights[indices])
            if query_guided:
                support = float(early_orders[indices][available].sum())
                matched = float(
                    early_geo_matches[indices, candidate_index][available].sum()
                )
                predictability = (
                    matched
                    + prior_orders * global_predictability[candidate_index]
                ) / (support + prior_orders)
                score *= predictability
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and (best_candidate is None or candidate_index < best_candidate)
            ):
                best_candidate = candidate_index
                best_score = score
        if best_candidate is None:
            selected_bucket_counts["none"] += 1
            continue
        selected_bucket_counts[str(GEO_CANDIDATE_LENGTHS[best_candidate])] += 1
        selected_values = geo_candidates[indices, best_candidate]
        values[indices] = selected_values
        lengths[indices] = GEO_CANDIDATE_LENGTHS[best_candidate]
        selected_matches[indices] = early_geo_matches[indices, best_candidate]
        _, inverse, counts = np.unique(
            selected_values, return_inverse=True, return_counts=True
        )
        resolved[indices] = counts[inverse] == 1

    available = values >= 0
    support = int(early_orders[available].sum())
    matched = int(selected_matches[available].sum())
    metrics = {
        "query_guided": query_guided,
        "selected_bucket_count_by_length": dict(selected_bucket_counts),
        "global_candidate_predictability": {
            str(length): float(global_predictability[index])
            for index, length in enumerate(GEO_CANDIDATE_LENGTHS)
        },
        "selected_geo_poi_count": int(available.sum()),
        "selected_geo_poi_ratio": _ratio(int(available.sum()), rows),
        "selected_geo_early_support_order_count": support,
        "selected_geo_early_match_count": matched,
        "selected_geo_early_match_ratio": _ratio(matched, support),
        "geo_only_resolved_poi_count": int(resolved.sum()),
        "geo_only_resolved_poi_ratio": _ratio(int(resolved.sum()), rows),
        "geo_only_resolved_early_order_count": int(early_orders[resolved].sum()),
        "geo_only_resolved_early_order_ratio": _ratio(
            int(early_orders[resolved].sum()), int(early_orders.sum())
        ),
    }
    return GeoSelection(
        values=values,
        lengths=lengths,
        matches=selected_matches,
        resolved=resolved,
        metrics=metrics,
    )


def _rank_geo_or_hybrid(
    *,
    candidates: np.ndarray,
    target: int,
    path_types: np.ndarray,
    path_values: np.ndarray,
    early_orders: np.ndarray,
    poi_ids: np.ndarray,
    features: Any,
    request_gid: Sequence[int],
    prefix_lengths: np.ndarray,
    geo_lengths: np.ndarray,
) -> tuple[int, bool]:
    typed, numeric_values = _query_arrays(features)
    candidate_types = path_types[candidates]
    candidate_values = path_values[candidates]
    emitted = candidate_types >= 0
    lexical = emitted & (candidate_types < GEO_RELATION_INDEX)
    safe_types = np.where(lexical, candidate_types, 0)
    typed_for_path = typed[safe_types]
    typed_match = lexical & (typed_for_path == candidate_values)
    typed_mismatch = lexical & (typed_for_path >= 0) & ~typed_match
    direct_mask = np.asarray(
        [name in DIRECT_NUMERIC_RELATION_TYPES for name in RELATION_TYPES],
        dtype=np.bool_,
    )[safe_types]
    numeric_match = lexical & direct_mask & np.isin(
        candidate_values, list(numeric_values)
    )
    untyped_match = numeric_match & ~typed_match

    geo = emitted & (candidate_types == GEO_RELATION_INDEX)
    target_prefix = int(prefix_lengths[target])
    target_geo_length = int(geo_lengths[target])
    if len(np.unique(prefix_lengths[candidates])) != 1 or len(
        np.unique(geo_lengths[candidates])
    ) != 1:
        raise QgrSidGeoProxyError("同一 TIGER 桶的 GEO prefix/length 不一致")
    one, two = _request_geo_values(request_gid, target_prefix)
    request_geo_value = one if target_geo_length == 1 else two
    geo_match = geo & (candidate_values == request_geo_value)
    geo_mismatch = geo & ~geo_match

    positive = typed_match | untyped_match | geo_match
    net_typed_geo = (typed_match | geo_match).sum(axis=1) - (
        typed_mismatch | geo_mismatch
    ).sum(axis=1)
    untyped_count = untyped_match.sum(axis=1)
    path_lengths = emitted.sum(axis=1)
    complete = (path_lengths > 0) & (positive.sum(axis=1) == path_lengths)
    ranking = np.lexsort(
        (
            poi_ids[candidates],
            -early_orders[candidates].astype(np.int64),
            -complete.astype(np.int8),
            -untyped_count,
            -net_typed_geo,
        )
    )
    target_positions = np.flatnonzero(candidates[ranking] == target)
    if len(target_positions) != 1:
        raise QgrSidGeoProxyError("GEO 已知桶候选中目标 POI 不唯一")
    target_local = int(np.flatnonzero(candidates == target)[0])
    return int(target_positions[0]) + 1, bool(complete[target_local])


def _evaluate_variants(
    *,
    events: Sequence[_GeoHoldoutEvent],
    catalog: StrictRelationCatalog,
    early_orders: np.ndarray,
    bucket_order: np.ndarray,
    bucket_starts: np.ndarray,
    prefix_lengths: np.ndarray,
    variants: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    lexical_variants: Mapping[
        str, tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    inverse_order = np.empty(len(bucket_order), dtype=np.int64)
    inverse_order[bucket_order] = np.arange(len(bucket_order), dtype=np.int64)
    stops = np.r_[bucket_starts[1:], len(bucket_order)]
    group_ids = np.empty(len(bucket_order), dtype=np.int32)
    for group_id, (start, stop) in enumerate(
        zip(bucket_starts, stops, strict=True)
    ):
        group_ids[start:stop] = group_id
    popularity_rank = np.empty(len(bucket_order), dtype=np.int16)
    for start, stop in zip(bucket_starts, stops, strict=True):
        candidates = bucket_order[start:stop]
        ranking = np.lexsort(
            (catalog.poi_ids[candidates], -early_orders[candidates].astype(np.int64))
        )
        popularity_rank[candidates[ranking]] = np.arange(1, len(candidates) + 1)

    rank_lists: dict[str, list[int]] = {
        "popularity": [],
        **{name: [] for name in lexical_variants},
        **{name: [] for name in variants},
    }
    resolved_counts = {name: 0 for name in (*lexical_variants, *variants)}
    exact_counts = {name: 0 for name in (*lexical_variants, *variants)}
    for event_number, event in enumerate(events, start=1):
        target = event.collision_index
        position = int(inverse_order[target])
        group_id = int(group_ids[position])
        candidates = bucket_order[
            int(bucket_starts[group_id]) : int(stops[group_id])
        ]
        rank_lists["popularity"].append(int(popularity_rank[target]))
        features = extract_query_relation_features(event.query)
        for name, (path_types, path_values, resolved) in lexical_variants.items():
            rank, exact = _rank_for_event(
                candidates=candidates,
                target=target,
                path_types=path_types,
                path_values=path_values,
                early_orders=early_orders,
                poi_ids=catalog.poi_ids,
                features=features,
            )
            rank_lists[name].append(rank)
            exact_counts[name] += int(exact)
            resolved_counts[name] += int(resolved[target])
        for name, (
            path_types,
            path_values,
            resolved,
            geo_lengths,
        ) in variants.items():
            rank, exact = _rank_geo_or_hybrid(
                candidates=candidates,
                target=target,
                path_types=path_types,
                path_values=path_values,
                early_orders=early_orders,
                poi_ids=catalog.poi_ids,
                features=features,
                request_gid=event.request_gid,
                prefix_lengths=prefix_lengths,
                geo_lengths=geo_lengths,
            )
            rank_lists[name].append(rank)
            exact_counts[name] += int(exact)
            resolved_counts[name] += int(resolved[target])
        if progress is not None and event_number % 50_000 == 0:
            progress(f"M2-B holdout 已评测 {event_number:,}/{len(events):,}")

    ranking_metrics = {
        name: _ranking_metrics(ranks) for name, ranks in rank_lists.items()
    }
    path_metrics = {
        name: {
            "resolved_order_count": resolved_counts[name],
            "resolved_order_ratio": _ratio(resolved_counts[name], len(events)),
            "exact_path_order_count": exact_counts[name],
            "exact_path_order_ratio": _ratio(exact_counts[name], len(events)),
        }
        for name in (*lexical_variants, *variants)
    }
    return {
        "collision_holdout_order_count": len(events),
        "ranking": ranking_metrics,
        "paths": path_metrics,
    }


def _path_arrays_for_geo(selection: GeoSelection) -> tuple[np.ndarray, np.ndarray]:
    path_types = np.full((len(selection.values), 1), -1, dtype=np.int16)
    path_values = np.full_like(path_types, -1)
    available = selection.values >= 0
    path_types[available, 0] = GEO_RELATION_INDEX
    path_values[available, 0] = selection.values[available]
    return path_types, path_values


def evaluate_geo_relation_proxy(
    *,
    project_root: Path,
    order_dir: Path,
    sft_manifest_path: Path,
    identifier_dir: Path,
    m2a_dir: Path,
    gid_codes_path: Path,
    gid_manifest_path: Path,
    output_dir: Path,
    max_pairs: int = 3,
    batch_rows: int = 65_536,
    prior_orders: float = 20.0,
    progress: Callable[[str], None] | None = None,
) -> GeoProxyResult:
    """Evaluate GEO-only and lexical+GEO under the frozen M2-A split."""

    started_at = _utc_now()
    started = time.monotonic()
    paths = {
        "project_root": project_root.resolve(),
        "order_dir": order_dir.resolve(),
        "sft_manifest": sft_manifest_path.resolve(),
        "identifier_dir": identifier_dir.resolve(),
        "m2a_dir": m2a_dir.resolve(),
        "gid_codes": gid_codes_path.resolve(),
        "gid_manifest": gid_manifest_path.resolve(),
        "output_dir": output_dir.resolve(),
    }
    if max_pairs <= 0 or batch_rows <= 0 or prior_orders <= 0:
        raise QgrSidGeoProxyError("max_pairs/batch_rows/prior_orders 参数非法")
    if paths["output_dir"].exists():
        raise QgrSidGeoProxyError("输出目录已存在，拒绝覆盖")
    for key in ("project_root", "order_dir", "identifier_dir", "m2a_dir"):
        if not paths[key].is_dir():
            raise QgrSidGeoProxyError(f"目录不存在：{paths[key]}")
    for key in ("sft_manifest", "gid_codes", "gid_manifest"):
        if not paths[key].is_file():
            raise QgrSidGeoProxyError(f"文件不存在：{paths[key]}")

    arrays, m2a_manifest, m2a_metrics = _load_m2a_arrays(paths["m2a_dir"])
    if m2a_manifest.get("schema_version") != PROXY_SCHEMA_VERSION:
        raise QgrSidGeoProxyError("M2-A schema_version 不受支持")
    split = TemporalSplit(**m2a_metrics["temporal_split"])
    sft_manifest = _load_json(paths["sft_manifest"], "SFT manifest")
    if sft_manifest.get("schema_version") != "sft-main-data-v1" or sft_manifest.get(
        "status"
    ) != "completed":
        raise QgrSidGeoProxyError("SFT manifest 非 completed 正式版本")

    output_dir = paths["output_dir"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}"
    if staging_dir.exists():
        raise QgrSidGeoProxyError("临时输出目录已存在")
    staging_dir.mkdir()

    if progress is not None:
        progress("全量验证 GID6 POI 行序，并抽取 TIGER 碰撞 POI 静态 GEO")
    poi_gid_codes, gid_inputs = extract_collision_gid_codes(
        gid_codes_path=paths["gid_codes"],
        gid_manifest_path=paths["gid_manifest"],
        identifier_dir=paths["identifier_dir"],
        collision_poi_ids=arrays["collision_poi_ids"],
        batch_rows=batch_rows,
        progress=progress,
    )
    prefix_lengths, geo_candidates, bucket_order, bucket_starts = (
        derive_geo_candidates(
            bucket_keys=arrays["base_sid_keys"], poi_gid_codes=poi_gid_codes
        )
    )

    if progress is not None:
        progress("按 M2-A 同一时间边界扫描请求 GID，统计 early 匹配并冻结 holdout")
    early_geo_matches, holdout_events, order_metrics, order_sources = (
        _read_geo_order_events(
            order_dir=paths["order_dir"],
            sft_manifest=sft_manifest,
            split=split,
            collision_poi_ids=arrays["collision_poi_ids"],
            prefix_lengths=prefix_lengths,
            geo_candidates=geo_candidates,
            expected_early_orders=arrays["early_order_counts"],
            progress=progress,
        )
    )
    p99_order_count = float(
        m2a_metrics["early"]["p99_positive_poi_order_count"]
    )
    static_geo = select_geo_encoding(
        geo_candidates=geo_candidates,
        early_orders=arrays["early_order_counts"],
        early_geo_matches=early_geo_matches,
        bucket_order=bucket_order,
        bucket_starts=bucket_starts,
        query_guided=False,
        p99_order_count=p99_order_count,
        prior_orders=prior_orders,
    )
    guided_geo = select_geo_encoding(
        geo_candidates=geo_candidates,
        early_orders=arrays["early_order_counts"],
        early_geo_matches=early_geo_matches,
        bucket_order=bucket_order,
        bucket_starts=bucket_starts,
        query_guided=True,
        p99_order_count=p99_order_count,
        prior_orders=prior_orders,
    )

    catalog_base = StrictRelationCatalog(
        poi_ids=arrays["collision_poi_ids"],
        bucket_keys=arrays["base_sid_keys"],
        relations=arrays["strict_relations"],
        conflict_masks=arrays["conflict_masks"],
        input_poi_count=int(m2a_metrics["catalog"]["input_poi_count"]),
        colliding_poi_count=len(arrays["collision_poi_ids"]),
    )
    lexical_predictability = np.asarray(
        [
            m2a_metrics["early"]["global_value_aware_predictability"][name]
            for name in RELATION_TYPES
        ],
        dtype=np.float64,
    )

    def compile_hybrid(
        selection: GeoSelection, query_guided: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        extended_relations = np.column_stack(
            (arrays["strict_relations"], selection.values)
        ).astype(np.int16, copy=False)
        extended_matches = np.column_stack(
            (arrays["early_relation_match_counts"], selection.matches)
        ).astype(np.int32, copy=False)
        available = selection.values >= 0
        geo_predictability = _ratio(
            int(selection.matches[available].sum()),
            int(arrays["early_order_counts"][available].sum()),
        )
        extended_predictability = np.r_[
            lexical_predictability, geo_predictability
        ]
        catalog = StrictRelationCatalog(
            poi_ids=catalog_base.poi_ids,
            bucket_keys=catalog_base.bucket_keys,
            relations=extended_relations,
            conflict_masks=catalog_base.conflict_masks,
            input_poi_count=catalog_base.input_poi_count,
            colliding_poi_count=catalog_base.colliding_poi_count,
        )
        types, values, resolved, compile_metrics, order, starts = (
            _compile_all_buckets(
                catalog=catalog,
                early_orders=arrays["early_order_counts"],
                early_matches=extended_matches,
                global_predictability=extended_predictability,
                query_guided=query_guided,
                max_pairs=max_pairs,
                p99_order_count=p99_order_count,
                prior_orders=prior_orders,
                progress=progress,
            )
        )
        if not np.array_equal(order, bucket_order) or not np.array_equal(
            starts, bucket_starts
        ):
            raise QgrSidGeoProxyError("M2-A/M2-B 桶顺序不一致")
        return types, values, resolved, compile_metrics

    if progress is not None:
        progress("编译静态词法+GEO 变长关系树")
    static_hybrid_types, static_hybrid_values, static_hybrid_resolved, static_hybrid_metrics = compile_hybrid(
        static_geo, False
    )
    if progress is not None:
        progress("编译 Query+GID 引导的词法+GEO 变长关系树")
    guided_hybrid_types, guided_hybrid_values, guided_hybrid_resolved, guided_hybrid_metrics = compile_hybrid(
        guided_geo, True
    )

    static_geo_types, static_geo_values = _path_arrays_for_geo(static_geo)
    guided_geo_types, guided_geo_values = _path_arrays_for_geo(guided_geo)
    if progress is not None:
        progress("在冻结 holdout 上统一评测 GEO-only、纯词法和词法+GEO")
    holdout_metrics = _evaluate_variants(
        events=holdout_events,
        catalog=catalog_base,
        early_orders=arrays["early_order_counts"],
        bucket_order=bucket_order,
        bucket_starts=bucket_starts,
        prefix_lengths=prefix_lengths,
        lexical_variants={
            "static_lexical": (
                arrays["static_path_types"],
                arrays["static_path_values"],
                arrays["static_resolved"],
            ),
            "query_guided_lexical": (
                arrays["query_guided_path_types"],
                arrays["query_guided_path_values"],
                arrays["query_guided_resolved"],
            ),
        },
        variants={
            "static_geo_only": (
                static_geo_types,
                static_geo_values,
                static_geo.resolved,
                static_geo.lengths,
            ),
            "gid_guided_geo_only": (
                guided_geo_types,
                guided_geo_values,
                guided_geo.resolved,
                guided_geo.lengths,
            ),
            "static_lexical_geo": (
                static_hybrid_types,
                static_hybrid_values,
                static_hybrid_resolved,
                static_geo.lengths,
            ),
            "query_gid_lexical_geo": (
                guided_hybrid_types,
                guided_hybrid_values,
                guided_hybrid_resolved,
                guided_geo.lengths,
            ),
        },
        progress=progress,
    )
    ranking = holdout_metrics["ranking"]
    for new_name, old_name in (
        ("popularity", "popularity"),
        ("static_lexical", "static"),
        ("query_guided_lexical", "query_guided"),
    ):
        old = m2a_metrics["holdout"]["ranking"][old_name]
        if any(
            abs(float(ranking[new_name][key]) - float(old[key])) > 1e-12
            for key in ("hr_at_1", "hr_at_10", "ndcg_at_10", "mean_rank")
        ):
            raise QgrSidGeoProxyError("M2-B 未精确复现 M2-A 冻结基线")

    best_hybrid = ranking["query_gid_lexical_geo"]
    lexical = ranking["query_guided_lexical"]
    geo_only = ranking["gid_guided_geo_only"]
    deltas = {
        "query_gid_hybrid_hr_at_1_vs_query_lexical": best_hybrid["hr_at_1"]
        - lexical["hr_at_1"],
        "query_gid_hybrid_ndcg_at_10_vs_query_lexical": best_hybrid["ndcg_at_10"]
        - lexical["ndcg_at_10"],
        "query_gid_hybrid_hr_at_1_vs_gid_geo_only": best_hybrid["hr_at_1"]
        - geo_only["hr_at_1"],
        "query_gid_hybrid_ndcg_at_10_vs_gid_geo_only": best_hybrid["ndcg_at_10"]
        - geo_only["ndcg_at_10"],
    }
    prefix_distribution = Counter(int(value) for value in prefix_lengths)
    metrics: dict[str, Any] = {
        "schema_version": GEO_PROXY_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "warning": (
            "Known-gold-bucket exact-GID-branch proxy only; not downstream SFT "
            "or unconstrained generation evidence."
        ),
        "temporal_split": asdict(split),
        "geo_catalog": {
            "alphabet": GEOHASH_ALPHABET,
            "poi_geohash_length": 6,
            "one_character_value_range": [0, 31],
            "two_character_value_range": [32, 1055],
            "common_prefix_length_distribution": {
                str(key): value for key, value in sorted(prefix_distribution.items())
            },
            "no_geo_candidate_poi_count": int(
                np.all(geo_candidates < 0, axis=1).sum()
            ),
        },
        "orders": order_metrics,
        "geo_selection": {
            "static": static_geo.metrics,
            "gid_guided": guided_geo.metrics,
        },
        "compilation": {
            "static_lexical_geo": static_hybrid_metrics,
            "query_gid_lexical_geo": guided_hybrid_metrics,
        },
        "holdout": holdout_metrics,
        "deltas": deltas,
        "decision": {
            "geo_adds_signal_over_lexical": bool(
                deltas["query_gid_hybrid_hr_at_1_vs_query_lexical"] > 0
                and deltas["query_gid_hybrid_ndcg_at_10_vs_query_lexical"] > 0
            ),
            "hybrid_beats_geo_only": bool(
                deltas["query_gid_hybrid_hr_at_1_vs_gid_geo_only"] > 0
                and deltas["query_gid_hybrid_ndcg_at_10_vs_gid_geo_only"] > 0
            ),
            "request_weighted_hybrid_resolved_ratio": guided_hybrid_metrics[
                "early_order_resolved_ratio"
            ],
            "sft_offline_resolution_gate_passed": bool(
                guided_hybrid_metrics["early_order_resolved_ratio"] >= 0.60
            ),
        },
    }
    metrics["signature"] = _signature(
        {
            "schema_version": GEO_PROXY_SCHEMA_VERSION,
            "m2a_manifest_sha256": sha256_file(paths["m2a_dir"] / "manifest.json"),
            "gid_codes_sha256": gid_inputs["gid_codes"]["sha256"],
            "max_pairs": max_pairs,
            "prior_orders": prior_orders,
        }
    )

    output_arrays = {
        "collision_poi_gid_codes": poi_gid_codes,
        "geo_common_prefix_lengths": prefix_lengths,
        "geo_candidate_values": geo_candidates,
        "early_geo_match_counts": early_geo_matches,
        "static_geo_values": static_geo.values,
        "static_geo_lengths": static_geo.lengths,
        "static_geo_resolved": static_geo.resolved,
        "gid_guided_geo_values": guided_geo.values,
        "gid_guided_geo_lengths": guided_geo.lengths,
        "gid_guided_geo_resolved": guided_geo.resolved,
        "static_hybrid_path_types": static_hybrid_types,
        "static_hybrid_path_values": static_hybrid_values,
        "static_hybrid_resolved": static_hybrid_resolved,
        "query_gid_hybrid_path_types": guided_hybrid_types,
        "query_gid_hybrid_path_values": guided_hybrid_values,
        "query_gid_hybrid_resolved": guided_hybrid_resolved,
    }
    array_contracts = {
        name: _save_array(staging_dir, name, array)
        for name, array in output_arrays.items()
    }
    metrics_path = staging_dir / "metrics.json"
    _json_dump(metrics_path, metrics)
    finished_at = _utc_now()
    manifest: dict[str, Any] = {
        "schema_version": GEO_PROXY_SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "signature": metrics["signature"],
        "configuration": {
            "base_identifier": "TIGER-BGE-M3-RQ-VAE-1024x3-epoch20",
            "relation_types": list(HYBRID_RELATION_TYPES),
            "geo_relation_type": "R_GEO",
            "geo_value_encoding": "one char: c; two chars: 32 + 32*c1 + c2",
            "geo_candidate_rule": "1/2 chars after base-bucket static POI GID6 common prefix",
            "request_signal": "existing request Geohash6 exact selected branch value",
            "max_relation_pairs": max_pairs,
            "prior_orders": prior_orders,
            "evaluation_scope": "known gold TIGER collision bucket",
        },
        "inputs": {
            "m2a_manifest": {
                "path": str(paths["m2a_dir"] / "manifest.json"),
                "sha256": sha256_file(paths["m2a_dir"] / "manifest.json"),
            },
            "m2a_metrics": {
                "path": str(paths["m2a_dir"] / "metrics.json"),
                "sha256": sha256_file(paths["m2a_dir"] / "metrics.json"),
            },
            "sft_manifest": {
                "path": str(paths["sft_manifest"]),
                "sha256": sha256_file(paths["sft_manifest"]),
            },
            "tiger_identifier_manifest": {
                "path": str(paths["identifier_dir"] / "tiger_id_manifest.json"),
                "sha256": sha256_file(
                    paths["identifier_dir"] / "tiger_id_manifest.json"
                ),
            },
            **gid_inputs,
            "order_sources": order_sources,
        },
        "outputs": {
            "metrics": {
                "path": metrics_path.name,
                "bytes": metrics_path.stat().st_size,
                "sha256": sha256_file(metrics_path),
            },
            "arrays": array_contracts,
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "git": _git_state(paths["project_root"]),
        "validation": {
            "m2a_baseline_exactly_reproduced": True,
            "true_temporal_holdout_not_used_for_compilation": True,
            "gid_tiger_full_row_alignment": True,
            "early_poi_order_counts_equal_m2a": True,
            "geo_values_within_0_1055": True,
            "paths_are_static_per_poi": True,
            "max_relation_pairs_respected": bool(
                np.all((static_hybrid_types >= 0).sum(axis=1) <= max_pairs)
                and np.all((guided_hybrid_types >= 0).sum(axis=1) <= max_pairs)
            ),
        },
    }
    _json_dump(staging_dir / "manifest.json", manifest)
    (staging_dir / "_SUCCESS").touch()
    staging_dir.replace(output_dir)
    return GeoProxyResult(metrics=metrics, manifest=manifest, output_dir=output_dir)


def validate_geo_proxy_output(output_dir: Path) -> dict[str, Any]:
    """Validate all M2-B outputs without recomputing GEO/GID statistics."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "M2-B manifest")
    if manifest.get("schema_version") != GEO_PROXY_SCHEMA_VERSION:
        raise QgrSidGeoProxyError("M2-B schema_version 不受支持")
    if manifest.get("status") != "completed" or not (
        output_dir / "_SUCCESS"
    ).is_file():
        raise QgrSidGeoProxyError("M2-B 正式输出未 completed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise QgrSidGeoProxyError("M2-B manifest 缺少 outputs")
    metrics_contract = outputs.get("metrics")
    arrays = outputs.get("arrays")
    if not isinstance(metrics_contract, dict) or not isinstance(arrays, dict):
        raise QgrSidGeoProxyError("M2-B 输出契约非法")
    contracts = {"metrics": metrics_contract, **arrays}
    checked: dict[str, str] = {}
    for name, contract in contracts.items():
        if not isinstance(contract, dict):
            raise QgrSidGeoProxyError(f"M2-B 输出契约非法：{name}")
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file() or path.stat().st_size != int(
            contract.get("bytes", -1)
        ):
            raise QgrSidGeoProxyError(f"M2-B 输出文件/字节数非法：{name}")
        digest = sha256_file(path)
        if digest != contract.get("sha256"):
            raise QgrSidGeoProxyError(f"M2-B 输出 SHA256 不一致：{name}")
        if name != "metrics":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != contract.get("shape") or str(
                array.dtype
            ) != contract.get("dtype"):
                raise QgrSidGeoProxyError(f"M2-B 数组契约不一致：{name}")
        checked[name] = digest
    return {
        "status": "validated",
        "schema_version": manifest["schema_version"],
        "signature": manifest["signature"],
        "checked_output_count": len(checked),
        "checked_outputs": checked,
    }

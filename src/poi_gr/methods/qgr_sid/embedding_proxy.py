"""Early-only Query-embedding residual proxy for QGR-SID M2-C."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from poi_gr.embedding.query_shards import stable_query_shard
from poi_gr.methods.qgr_sid.proxy import (
    ORDER_REQUIRED_FIELDS,
    PROXY_SCHEMA_VERSION,
    QgrSidProxyError,
    TemporalSplit,
    _git_state,
    _json_dump,
    _load_json,
    _query_arrays,
    _rank_for_event,
    _ranking_metrics,
    _ratio,
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


EMBEDDING_PROXY_SCHEMA_VERSION = "qgr-sid-query-embedding-proxy-v1"
EMBEDDING_PROXY_METRICS_SCHEMA_VERSION = (
    "qgr-sid-query-embedding-proxy-metrics-v1"
)


class QgrSidEmbeddingProxyError(QgrSidProxyError):
    """Raised when M2-C inputs or temporal subtraction are inconsistent."""


@dataclass(frozen=True)
class EmbeddingProxyResult:
    """Completed M2-C Query residual proxy artifacts."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


@dataclass(frozen=True)
class _EmbeddingEvent:
    collision_index: int
    query: str
    query_shard: int


def _load_m2a_arrays(
    m2a_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    validate_proxy_output(m2a_dir)
    manifest = _load_json(m2a_dir / "manifest.json", "M2-A manifest")
    metrics = _load_json(m2a_dir / "metrics.json", "M2-A metrics")
    if manifest.get("schema_version") != PROXY_SCHEMA_VERSION:
        raise QgrSidEmbeddingProxyError("M2-A schema_version 不受支持")
    contracts = manifest.get("outputs", {}).get("arrays")
    if not isinstance(contracts, dict):
        raise QgrSidEmbeddingProxyError("M2-A manifest 缺少 arrays")
    names = (
        "collision_poi_ids",
        "base_sid_keys",
        "early_order_counts",
        "query_guided_path_types",
        "query_guided_path_values",
        "query_guided_resolved",
    )
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        contract = contracts.get(name)
        if not isinstance(contract, dict):
            raise QgrSidEmbeddingProxyError(f"M2-A 缺少数组：{name}")
        arrays[name] = np.load(
            m2a_dir / str(contract["path"]), mmap_mode="r", allow_pickle=False
        )
    return arrays, manifest, metrics


def _scan_holdout_counts(
    *,
    order_dir: Path,
    sft_manifest: Mapping[str, Any],
    split: TemporalSplit,
    collision_poi_ids: np.ndarray,
    query_shards: int,
    expected_early_orders: np.ndarray,
    progress: Callable[[str], None] | None,
) -> tuple[
    list[Counter[tuple[str, str]]],
    Counter[str],
    list[_EmbeddingEvent],
    list[list[int]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    collision_lookup = {
        int(poi_id): index for index, poi_id in enumerate(collision_poi_ids)
    }
    holdout_pairs = [Counter() for _ in range(query_shards)]
    holdout_pois: Counter[str] = Counter()
    events: list[_EmbeddingEvent] = []
    events_by_shard: list[list[int]] = [[] for _ in range(query_shards)]
    early_orders = np.zeros(len(collision_poi_ids), dtype=np.int32)
    tie_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    orders_contract = sft_manifest.get("orders")
    if not isinstance(orders_contract, dict):
        raise QgrSidEmbeddingProxyError("SFT manifest 缺少 orders")
    expected_files = {
        str(item.get("relative_path")): item
        for item in orders_contract.get("files", [])
        if isinstance(item, dict)
    }
    actual_paths = tuple(sorted(order_dir.glob("part-*.json")))
    if set(path.name for path in actual_paths) != set(expected_files):
        raise QgrSidEmbeddingProxyError("订单文件集合与 SFT manifest 不一致")
    sources: list[dict[str, Any]] = []

    def process(record: Mapping[str, Any], split_name: str) -> None:
        counts[f"{split_name}_orders"] += 1
        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise QgrSidEmbeddingProxyError("Train Query 为空")
        poi_id = str(record.get("poi_id", ""))
        try:
            numeric_poi_id = int(poi_id)
        except (TypeError, ValueError, OverflowError) as error:
            raise QgrSidEmbeddingProxyError("订单 poi_id 非法") from error
        target = collision_lookup.get(numeric_poi_id)
        if split_name == "early":
            if target is not None:
                early_orders[target] += 1
                counts["early_collision_orders"] += 1
            return
        shard = stable_query_shard(query, query_shards)
        holdout_pairs[shard][(query, poi_id)] += 1
        holdout_pois[poi_id] += 1
        if target is not None:
            counts["holdout_collision_orders"] += 1
            event_index = len(events)
            events.append(
                _EmbeddingEvent(
                    collision_index=target,
                    query=query,
                    query_shard=shard,
                )
            )
            events_by_shard[shard].append(event_index)

    scanned = 0
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
                    raise QgrSidEmbeddingProxyError(
                        f"订单 JSON 解析失败：{path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise QgrSidEmbeddingProxyError("订单 JSON 行必须是对象")
                missing = [field for field in ORDER_REQUIRED_FIELDS if field not in record]
                if missing:
                    raise QgrSidEmbeddingProxyError(
                        "订单缺少字段：" + ", ".join(missing)
                    )
                create_time = str(record["create_time"])
                source_dt = str(record["source_dt"])
                if (
                    len(create_time) < 16
                    or source_dt != create_time[:10].replace("-", "")
                ):
                    raise QgrSidEmbeddingProxyError(
                        "订单 source_dt/create_time 不一致"
                    )
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
                        f"Embedding 时间扣除已扫描 {scanned:,}；"
                        f"holdout pair {sum(len(x) for x in holdout_pairs):,}"
                    )
        contract = expected_files[path.name]
        digest_value = digest.hexdigest()
        if file_rows != int(contract.get("rows", -1)):
            raise QgrSidEmbeddingProxyError("订单分片行数与 manifest 不一致")
        if digest_value != contract.get("sha256"):
            raise QgrSidEmbeddingProxyError("订单分片 SHA256 与 manifest 不一致")
        sources.append(
            {
                "path": str(path.resolve()),
                "rows": file_rows,
                "sha256": digest_value,
            }
        )
    if len(tie_records) != split.cutoff_minute_order_count:
        raise QgrSidEmbeddingProxyError("切分分钟订单数与 M2-A 不一致")
    tie_records.sort(key=_split_tie_key)
    for index, record in enumerate(tie_records):
        process(
            record,
            "early" if index < split.early_from_cutoff_minute else "holdout",
        )
    if counts["early_orders"] != split.early_target_count or counts[
        "holdout_orders"
    ] != split.holdout_target_count:
        raise QgrSidEmbeddingProxyError("M2-C 时间切分计数不守恒")
    if not np.array_equal(early_orders, expected_early_orders):
        raise QgrSidEmbeddingProxyError("M2-C early POI 请求计数与 M2-A 不一致")
    return (
        holdout_pairs,
        holdout_pois,
        events,
        events_by_shard,
        dict(counts),
        sources,
    )


def _contract_files(
    stats_manifest: Mapping[str, Any], name: str
) -> tuple[str, list[dict[str, Any]]]:
    contract = stats_manifest.get("outputs", {}).get(name)
    if not isinstance(contract, dict):
        raise QgrSidEmbeddingProxyError(f"Query stats 缺少 {name}")
    directory = contract.get("dir")
    files = contract.get("files")
    if not isinstance(directory, str) or not isinstance(files, list):
        raise QgrSidEmbeddingProxyError(f"Query stats {name} 契约非法")
    return directory, [item for item in files if isinstance(item, dict)]


def _read_verified_table(
    path: Path,
    contract: Mapping[str, Any],
    *,
    columns: Sequence[str],
) -> Any:
    if path.stat().st_size != int(contract.get("size_bytes", -1)):
        raise QgrSidEmbeddingProxyError(f"Parquet 字节数不一致：{path}")
    if sha256_file(path) != contract.get("sha256"):
        raise QgrSidEmbeddingProxyError(f"Parquet SHA256 不一致：{path}")
    table = pq.read_table(path, columns=list(columns))
    if table.num_rows != int(contract.get("rows", -1)):
        raise QgrSidEmbeddingProxyError(f"Parquet 行数不一致：{path}")
    return table


def _early_covered_poi_count(
    *,
    query_stats_dir: Path,
    stats_manifest: Mapping[str, Any],
    holdout_pois: Mapping[str, int],
) -> tuple[int, dict[str, int]]:
    directory, files = _contract_files(stats_manifest, "poi_stats")
    early_covered = 0
    full_orders = 0
    early_orders = 0
    seen_holdout: set[str] = set()
    for contract in files:
        path = query_stats_dir / directory / str(contract["file"])
        table = _read_verified_table(
            path,
            contract,
            columns=["target_poi_id", "train_order_count"],
        )
        for poi_id, count in zip(
            table.column(0).to_pylist(),
            table.column(1).to_numpy(zero_copy_only=False),
            strict=True,
        ):
            full = int(count)
            held = int(holdout_pois.get(poi_id, 0))
            if held > full:
                raise QgrSidEmbeddingProxyError("POI holdout count 超过 full Train")
            if held:
                seen_holdout.add(poi_id)
            early = full - held
            full_orders += full
            early_orders += early
            early_covered += int(early > 0)
    if len(seen_holdout) != len(holdout_pois):
        raise QgrSidEmbeddingProxyError("部分 holdout POI 不在 full Train 统计")
    return early_covered, {
        "full_train_order_count": full_orders,
        "early_train_order_count": early_orders,
        "holdout_order_count": full_orders - early_orders,
    }


def build_early_query_centers(
    *,
    query_stats_dir: Path,
    stats_manifest: Mapping[str, Any],
    query_embeddings: np.ndarray,
    collision_poi_ids: np.ndarray,
    active_collision_indices: np.ndarray,
    holdout_pairs: list[Counter[tuple[str, str]]],
    events: Sequence[_EmbeddingEvent],
    events_by_shard: Sequence[Sequence[int]],
    early_covered_pois: int,
    progress: Callable[[str], None] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Subtract holdout counts and build early-only E2 POI Query centers."""

    query_catalog_dir, catalog_files = _contract_files(
        stats_manifest, "query_catalog"
    )
    query_poi_dir, pair_files = _contract_files(stats_manifest, "query_poi")
    if len(catalog_files) != len(pair_files) or len(pair_files) != len(holdout_pairs):
        raise QgrSidEmbeddingProxyError("Query 分片数量不一致")
    collision_lookup = {
        int(poi_id): index for index, poi_id in enumerate(collision_poi_ids)
    }
    position_by_collision = np.full(len(collision_poi_ids), -1, dtype=np.int32)
    position_by_collision[active_collision_indices] = np.arange(
        len(active_collision_indices), dtype=np.int32
    )
    embedding_dim = query_embeddings.shape[1]
    accumulator = np.zeros(
        (len(active_collision_indices), embedding_dim), dtype=np.float32
    )
    weight_sum = np.zeros(len(active_collision_indices), dtype=np.float64)
    event_query_ids = np.full(len(events), -1, dtype=np.int64)
    total_full_orders = 0
    total_early_orders = 0
    total_early_pairs = 0
    early_unique_queries = 0

    for shard, (catalog_contract, pair_contract) in enumerate(
        zip(catalog_files, pair_files, strict=True)
    ):
        if int(catalog_contract.get("partition_id", -1)) != shard or int(
            pair_contract.get("partition_id", -1)
        ) != shard:
            raise QgrSidEmbeddingProxyError("Query 分片 partition_id 不连续")
        catalog_path = (
            query_stats_dir / query_catalog_dir / str(catalog_contract["file"])
        )
        catalog = _read_verified_table(
            catalog_path,
            catalog_contract,
            columns=["query_id", "query"],
        )
        catalog_query_ids = catalog.column(0).to_numpy(zero_copy_only=False)
        catalog_queries = catalog.column(1).to_pylist()
        query_to_id = dict(zip(catalog_queries, catalog_query_ids, strict=True))
        if len(query_to_id) != len(catalog_queries):
            raise QgrSidEmbeddingProxyError("Query catalog Query 不唯一")
        for event_index in events_by_shard[shard]:
            query_id = query_to_id.get(events[event_index].query)
            if query_id is None:
                raise QgrSidEmbeddingProxyError("holdout Query 不在 full catalog")
            event_query_ids[event_index] = int(query_id)

        held_by_id: dict[tuple[int, str], int] = {}
        for (query, poi_id), count in holdout_pairs[shard].items():
            query_id = query_to_id.get(query)
            if query_id is None:
                raise QgrSidEmbeddingProxyError("holdout pair Query 不在 full catalog")
            held_by_id[(int(query_id), poi_id)] = int(count)
        pair_path = query_stats_dir / query_poi_dir / str(pair_contract["file"])
        pairs = _read_verified_table(
            pair_path,
            pair_contract,
            columns=["query_id", "target_poi_id", "train_order_count"],
        )
        query_ids = pairs.column(0).to_numpy(zero_copy_only=False)
        poi_ids = pairs.column(1).to_pylist()
        full_counts = pairs.column(2).to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        held_counts = np.fromiter(
            (
                held_by_id.get((int(query_id), poi_id), 0)
                for query_id, poi_id in zip(query_ids, poi_ids, strict=True)
            ),
            dtype=np.int64,
            count=len(query_ids),
        )
        if np.any(held_counts > full_counts):
            raise QgrSidEmbeddingProxyError("holdout pair count 超过 full Train")
        early_counts = full_counts - held_counts
        found_held = int(held_counts.sum())
        expected_held = sum(holdout_pairs[shard].values())
        if found_held != expected_held:
            raise QgrSidEmbeddingProxyError("holdout pair 未在 full Query-POI 中守恒")
        positive = early_counts > 0
        positive_query_ids, early_df = np.unique(
            query_ids[positive], return_counts=True
        )
        df_by_query = dict(
            zip(positive_query_ids.tolist(), early_df.tolist(), strict=True)
        )
        early_unique_queries += len(positive_query_ids)
        total_early_pairs += int(positive.sum())
        total_full_orders += int(full_counts.sum())
        total_early_orders += int(early_counts.sum())

        positions = np.fromiter(
            (
                position_by_collision[collision_lookup.get(int(poi_id), -1)]
                if int(poi_id) in collision_lookup
                else -1
                for poi_id in poi_ids
            ),
            dtype=np.int32,
            count=len(poi_ids),
        )
        selected = positive & (positions >= 0)
        if np.any(selected):
            selected_query_ids = query_ids[selected].astype(np.int64, copy=False)
            selected_positions = positions[selected]
            selected_counts = early_counts[selected]
            selected_df = np.fromiter(
                (df_by_query[int(query_id)] for query_id in selected_query_ids),
                dtype=np.float64,
                count=len(selected_query_ids),
            )
            weights = np.log1p(selected_counts.astype(np.float64)) * np.log(
                (early_covered_pois + 1.0) / (selected_df + 1.0)
            )
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise QgrSidEmbeddingProxyError("early E2 权重非正或非有限")
            vectors = np.asarray(
                query_embeddings[selected_query_ids], dtype=np.float32
            )
            order = np.argsort(selected_positions, kind="stable")
            sorted_positions = selected_positions[order]
            starts = np.r_[
                0,
                np.flatnonzero(
                    sorted_positions[1:] != sorted_positions[:-1]
                )
                + 1,
            ]
            unique_positions = sorted_positions[starts]
            sorted_weights = weights[order]
            accumulator[unique_positions] += np.add.reduceat(
                vectors[order] * sorted_weights[:, None], starts, axis=0
            )
            weight_sum[unique_positions] += np.add.reduceat(sorted_weights, starts)
        holdout_pairs[shard].clear()
        if progress is not None and (shard + 1) % 16 == 0:
            progress(
                f"early Query center 已聚合 {shard + 1}/{len(pair_files)} 分片"
            )

    if np.any(event_query_ids < 0):
        raise QgrSidEmbeddingProxyError("部分 holdout event 未映射 query_id")
    if np.any(weight_sum <= 0):
        raise QgrSidEmbeddingProxyError("active collision POI 缺少 early Query center")
    accumulator /= weight_sum[:, None].astype(np.float32)
    norms = np.linalg.norm(accumulator, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise QgrSidEmbeddingProxyError("early Query center 范数非法")
    accumulator /= norms[:, None]
    return (
        accumulator,
        weight_sum,
        event_query_ids,
        {
            "full_train_order_count_from_query_pairs": total_full_orders,
            "early_train_order_count_from_query_pairs": total_early_orders,
            "holdout_order_count_subtracted": total_full_orders
            - total_early_orders,
            "early_unique_query_count": early_unique_queries,
            "early_unique_query_poi_pair_count": total_early_pairs,
            "early_center_collision_poi_count": len(active_collision_indices),
        },
    )


def _lexical_key_arrays(
    *,
    candidates: np.ndarray,
    path_types: np.ndarray,
    path_values: np.ndarray,
    features: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    typed, numeric_values = _query_arrays(features)
    candidate_types = path_types[candidates]
    candidate_values = path_values[candidates]
    emitted = candidate_types >= 0
    safe_types = np.where(emitted, candidate_types, 0)
    typed_for_path = typed[safe_types]
    typed_match = emitted & (typed_for_path == candidate_values)
    typed_mismatch = emitted & (typed_for_path >= 0) & ~typed_match
    direct_mask = np.asarray(
        [name in DIRECT_NUMERIC_RELATION_TYPES for name in RELATION_TYPES],
        dtype=np.bool_,
    )[safe_types]
    numeric_match = emitted & direct_mask & np.isin(
        candidate_values, list(numeric_values)
    )
    untyped_match = numeric_match & ~typed_match
    net_typed = typed_match.sum(axis=1) - typed_mismatch.sum(axis=1)
    untyped_count = untyped_match.sum(axis=1)
    path_lengths = emitted.sum(axis=1)
    complete = (path_lengths > 0) & (
        (typed_match | untyped_match).sum(axis=1) == path_lengths
    )
    return net_typed, untyped_count, complete


def rank_embedding_candidates(
    *,
    candidates: np.ndarray,
    target: int,
    similarities: np.ndarray,
    early_orders: np.ndarray,
    poi_ids: np.ndarray,
    lexical_keys: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> int:
    """Rank one known bucket by embedding, optionally after lexical evidence."""

    if similarities.shape != (len(candidates),):
        raise QgrSidEmbeddingProxyError("similarities shape 非法")
    if lexical_keys is None:
        ranking = np.lexsort(
            (
                poi_ids[candidates],
                -early_orders[candidates].astype(np.int64),
                -similarities,
            )
        )
    else:
        net_typed, untyped_count, complete = lexical_keys
        ranking = np.lexsort(
            (
                poi_ids[candidates],
                -early_orders[candidates].astype(np.int64),
                -similarities,
                -complete.astype(np.int8),
                -untyped_count,
                -net_typed,
            )
        )
    positions = np.flatnonzero(candidates[ranking] == target)
    if len(positions) != 1:
        raise QgrSidEmbeddingProxyError("Embedding 已知桶中目标 POI 不唯一")
    return int(positions[0]) + 1


def evaluate_embedding_proxy(
    *,
    events: Sequence[_EmbeddingEvent],
    event_query_ids: np.ndarray,
    query_embeddings: np.ndarray,
    collision_poi_ids: np.ndarray,
    bucket_keys: np.ndarray,
    early_orders: np.ndarray,
    active_collision_indices: np.ndarray,
    early_query_centers: np.ndarray,
    path_types: np.ndarray,
    path_values: np.ndarray,
    expected_m2a_ranking: Mapping[str, Any],
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate raw/residual Query centers and lexical-first residual fusion."""

    order = np.argsort(bucket_keys, kind="stable")
    sorted_keys = bucket_keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    group_by_position = np.empty(len(order), dtype=np.int32)
    for group_id, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        group_by_position[start:stop] = group_id
    position_by_collision = np.full(len(collision_poi_ids), -1, dtype=np.int32)
    position_by_collision[active_collision_indices] = np.arange(
        len(active_collision_indices), dtype=np.int32
    )
    event_groups: list[list[int]] = [[] for _ in range(len(starts))]
    for event_index, event in enumerate(events):
        group_id = int(group_by_position[int(inverse[event.collision_index])])
        event_groups[group_id].append(event_index)

    popularity_rank = np.empty(len(order), dtype=np.int16)
    for start, stop in zip(starts, stops, strict=True):
        candidates = order[start:stop]
        ranking = np.lexsort(
            (
                collision_poi_ids[candidates],
                -early_orders[candidates].astype(np.int64),
            )
        )
        popularity_rank[candidates[ranking]] = np.arange(1, len(candidates) + 1)
    ranks = {
        "popularity": [],
        "query_guided_lexical": [],
        "query_center": [],
        "query_residual": [],
        "lexical_first_query_residual": [],
    }
    coverage = Counter()
    cases: list[dict[str, Any]] = []

    processed = 0
    for group_id, event_indices in enumerate(event_groups):
        if not event_indices:
            continue
        start, stop = int(starts[group_id]), int(stops[group_id])
        candidates = order[start:stop]
        center_positions = position_by_collision[candidates]
        valid = center_positions >= 0
        candidate_centers = np.zeros(
            (len(candidates), early_query_centers.shape[1]), dtype=np.float32
        )
        candidate_centers[valid] = early_query_centers[center_positions[valid]]
        if np.any(valid):
            bucket_center = candidate_centers[valid].mean(axis=0)
            bucket_norm = float(np.linalg.norm(bucket_center))
            if bucket_norm > 0:
                bucket_center /= bucket_norm
        else:
            bucket_center = np.zeros(
                early_query_centers.shape[1], dtype=np.float32
            )
        residual_centers = candidate_centers - bucket_center[None, :]
        residual_norms = np.linalg.norm(residual_centers, axis=1)
        residual_valid = valid & (residual_norms > 1e-12)
        residual_centers[residual_valid] /= residual_norms[residual_valid, None]

        for event_index in event_indices:
            event = events[event_index]
            target = event.collision_index
            query = np.asarray(
                query_embeddings[int(event_query_ids[event_index])],
                dtype=np.float32,
            )
            query_norm = float(np.linalg.norm(query))
            if query_norm <= 0 or not np.isfinite(query_norm):
                raise QgrSidEmbeddingProxyError("holdout Query embedding 范数非法")
            query /= query_norm
            center_similarity = candidate_centers @ query
            center_similarity[~valid] = -2.0
            query_residual = query - bucket_center
            query_residual_norm = float(np.linalg.norm(query_residual))
            if query_residual_norm > 1e-12:
                query_residual /= query_residual_norm
                residual_similarity = residual_centers @ query_residual
                residual_similarity[~residual_valid] = -2.0
            else:
                residual_similarity = np.full(len(candidates), -2.0)

            features = extract_query_relation_features(event.query)
            lexical_keys = _lexical_key_arrays(
                candidates=candidates,
                path_types=path_types,
                path_values=path_values,
                features=features,
            )
            lexical_rank, _ = _rank_for_event(
                candidates=candidates,
                target=target,
                path_types=path_types,
                path_values=path_values,
                early_orders=early_orders,
                poi_ids=collision_poi_ids,
                features=features,
            )
            center_rank = rank_embedding_candidates(
                candidates=candidates,
                target=target,
                similarities=center_similarity,
                early_orders=early_orders,
                poi_ids=collision_poi_ids,
            )
            residual_rank = rank_embedding_candidates(
                candidates=candidates,
                target=target,
                similarities=residual_similarity,
                early_orders=early_orders,
                poi_ids=collision_poi_ids,
            )
            fused_rank = rank_embedding_candidates(
                candidates=candidates,
                target=target,
                similarities=residual_similarity,
                early_orders=early_orders,
                poi_ids=collision_poi_ids,
                lexical_keys=lexical_keys,
            )
            popularity = int(popularity_rank[target])
            ranks["popularity"].append(popularity)
            ranks["query_guided_lexical"].append(lexical_rank)
            ranks["query_center"].append(center_rank)
            ranks["query_residual"].append(residual_rank)
            ranks["lexical_first_query_residual"].append(fused_rank)
            target_position = int(np.flatnonzero(candidates == target)[0])
            coverage["target_has_early_center"] += int(valid[target_position])
            coverage["target_has_nonzero_residual"] += int(
                residual_valid[target_position]
            )
            if fused_rank < lexical_rank and len(cases) < 20:
                cases.append(
                    {
                        "case_kind": "residual_improved_over_lexical",
                        "query": event.query,
                        "target_poi_id": str(int(collision_poi_ids[target])),
                        "bucket_size": len(candidates),
                        "popularity_rank": popularity,
                        "lexical_rank": lexical_rank,
                        "query_residual_rank": residual_rank,
                        "fused_rank": fused_rank,
                        "target_has_early_center": bool(valid[target_position]),
                    }
                )
            processed += 1
            if progress is not None and processed % 50_000 == 0:
                progress(f"M2-C holdout 已评测 {processed:,}/{len(events):,}")

    ranking = {name: _ranking_metrics(values) for name, values in ranks.items()}
    for new_name, old_name in (
        ("popularity", "popularity"),
        ("query_guided_lexical", "query_guided"),
    ):
        expected = expected_m2a_ranking[old_name]
        actual = ranking[new_name]
        if any(
            abs(float(actual[key]) - float(expected[key])) > 1e-12
            for key in ("hr_at_1", "hr_at_10", "ndcg_at_10", "mean_rank")
        ):
            raise QgrSidEmbeddingProxyError("M2-C 未精确复现 M2-A 基线")
    return (
        {
            "collision_holdout_order_count": len(events),
            "ranking": ranking,
            "coverage": {
                key: {
                    "order_count": int(value),
                    "order_ratio": _ratio(value, len(events)),
                }
                for key, value in coverage.items()
            },
        },
        cases,
    )


def run_query_embedding_proxy(
    *,
    project_root: Path,
    order_dir: Path,
    sft_manifest_path: Path,
    m2a_dir: Path,
    query_stats_dir: Path,
    query_embeddings_path: Path,
    query_embeddings_manifest_path: Path,
    output_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> EmbeddingProxyResult:
    """Run M2-C with full-Train aggregates exactly subtracting holdout counts."""

    started_at = _utc_now()
    started = time.monotonic()
    project_root = project_root.resolve()
    order_dir = order_dir.resolve()
    sft_manifest_path = sft_manifest_path.resolve()
    m2a_dir = m2a_dir.resolve()
    query_stats_dir = query_stats_dir.resolve()
    query_embeddings_path = query_embeddings_path.resolve()
    query_embeddings_manifest_path = query_embeddings_manifest_path.resolve()
    output_dir = output_dir.resolve()
    for path, name in (
        (project_root, "project_root"),
        (order_dir, "order_dir"),
        (m2a_dir, "m2a_dir"),
        (query_stats_dir, "query_stats_dir"),
    ):
        if not path.is_dir():
            raise QgrSidEmbeddingProxyError(f"目录不存在：{name}={path}")
    for path in (
        sft_manifest_path,
        query_embeddings_path,
        query_embeddings_manifest_path,
    ):
        if not path.is_file():
            raise QgrSidEmbeddingProxyError(f"输入文件不存在：{path}")
    if output_dir.exists():
        raise QgrSidEmbeddingProxyError("输出目录已存在，拒绝覆盖")

    arrays, m2a_manifest, m2a_metrics = _load_m2a_arrays(m2a_dir)
    split = TemporalSplit(**m2a_metrics["temporal_split"])
    sft_manifest = _load_json(sft_manifest_path, "SFT manifest")
    if sft_manifest.get("schema_version") != "sft-main-data-v1" or sft_manifest.get(
        "status"
    ) != "completed":
        raise QgrSidEmbeddingProxyError("SFT manifest 非正式 completed")
    stats_manifest_path = query_stats_dir / "manifest.json"
    stats_manifest = _load_json(stats_manifest_path, "Query stats manifest")
    if stats_manifest.get("schema_version") != "train-query-stats-v1" or stats_manifest.get(
        "status"
    ) != "completed":
        raise QgrSidEmbeddingProxyError("Query stats manifest 非正式 completed")
    embedding_manifest = _load_json(
        query_embeddings_manifest_path, "Query embedding manifest"
    )
    if embedding_manifest.get("schema_version") != "train-query-embeddings-v1" or embedding_manifest.get(
        "status"
    ) != "completed":
        raise QgrSidEmbeddingProxyError("Query embedding manifest 非正式 completed")
    embedding_contract = embedding_manifest.get("output")
    if not isinstance(embedding_contract, dict):
        raise QgrSidEmbeddingProxyError("Query embedding manifest 缺少 output")
    if sha256_file(query_embeddings_path) != embedding_contract.get("sha256"):
        raise QgrSidEmbeddingProxyError("Query embedding SHA256 与 manifest 不一致")
    query_embeddings = np.load(
        query_embeddings_path, mmap_mode="r", allow_pickle=False
    )
    if list(query_embeddings.shape) != embedding_contract.get("shape") or str(
        query_embeddings.dtype
    ) != embedding_contract.get("dtype"):
        raise QgrSidEmbeddingProxyError("Query embedding shape/dtype 不一致")
    query_shards = int(stats_manifest["config"]["query_shards"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}"
    if staging_dir.exists():
        raise QgrSidEmbeddingProxyError("临时输出目录已存在")
    staging_dir.mkdir()

    if progress is not None:
        progress("扫描原始订单，冻结同一 holdout 并按 Query 分片统计扣除量")
    (
        holdout_pairs,
        holdout_pois,
        events,
        events_by_shard,
        order_metrics,
        order_sources,
    ) = _scan_holdout_counts(
        order_dir=order_dir,
        sft_manifest=sft_manifest,
        split=split,
        collision_poi_ids=arrays["collision_poi_ids"],
        query_shards=query_shards,
        expected_early_orders=arrays["early_order_counts"],
        progress=progress,
    )
    if progress is not None:
        progress("验证 POI full Train 统计并精确扣除 holdout，计算 early 覆盖")
    early_covered_pois, poi_subtraction = _early_covered_poi_count(
        query_stats_dir=query_stats_dir,
        stats_manifest=stats_manifest,
        holdout_pois=holdout_pois,
    )
    if poi_subtraction["early_train_order_count"] != split.early_target_count:
        raise QgrSidEmbeddingProxyError("POI 统计扣除后的 early 订单数不守恒")
    active_collision_indices = np.flatnonzero(arrays["early_order_counts"] > 0)
    if progress is not None:
        progress(
            f"构建 {len(active_collision_indices):,} 个碰撞 POI 的 early-only E2 Query center"
        )
    (
        early_query_centers,
        early_weight_sums,
        event_query_ids,
        center_metrics,
    ) = build_early_query_centers(
        query_stats_dir=query_stats_dir,
        stats_manifest=stats_manifest,
        query_embeddings=query_embeddings,
        collision_poi_ids=arrays["collision_poi_ids"],
        active_collision_indices=active_collision_indices,
        holdout_pairs=holdout_pairs,
        events=events,
        events_by_shard=events_by_shard,
        early_covered_pois=early_covered_pois,
        progress=progress,
    )
    if center_metrics["early_train_order_count_from_query_pairs"] != split.early_target_count:
        raise QgrSidEmbeddingProxyError("Query-POI 扣除后的 early 订单数不守恒")
    if progress is not None:
        progress("评测 Query center、桶内残差和词法优先残差融合")
    holdout_metrics, cases = evaluate_embedding_proxy(
        events=events,
        event_query_ids=event_query_ids,
        query_embeddings=query_embeddings,
        collision_poi_ids=arrays["collision_poi_ids"],
        bucket_keys=arrays["base_sid_keys"],
        early_orders=arrays["early_order_counts"],
        active_collision_indices=active_collision_indices,
        early_query_centers=early_query_centers,
        path_types=arrays["query_guided_path_types"],
        path_values=arrays["query_guided_path_values"],
        expected_m2a_ranking=m2a_metrics["holdout"]["ranking"],
        progress=progress,
    )
    ranking = holdout_metrics["ranking"]
    lexical = ranking["query_guided_lexical"]
    fused = ranking["lexical_first_query_residual"]
    residual = ranking["query_residual"]
    deltas = {
        "fused_hr_at_1_vs_lexical": fused["hr_at_1"] - lexical["hr_at_1"],
        "fused_ndcg_at_10_vs_lexical": fused["ndcg_at_10"]
        - lexical["ndcg_at_10"],
        "residual_hr_at_1_vs_lexical": residual["hr_at_1"]
        - lexical["hr_at_1"],
        "residual_ndcg_at_10_vs_lexical": residual["ndcg_at_10"]
        - lexical["ndcg_at_10"],
    }
    metrics: dict[str, Any] = {
        "schema_version": EMBEDDING_PROXY_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "warning": (
            "Known-gold-bucket pretrained-embedding branch proxy only; it is "
            "not a discrete residual assignment or downstream SFT result."
        ),
        "temporal_split": asdict(split),
        "orders": order_metrics,
        "temporal_subtraction": {
            "early_covered_poi_count": early_covered_pois,
            **poi_subtraction,
            **center_metrics,
        },
        "holdout": holdout_metrics,
        "deltas": deltas,
        "decision": {
            "query_residual_positive_over_lexical": bool(
                deltas["fused_hr_at_1_vs_lexical"] > 0
                and deltas["fused_ndcg_at_10_vs_lexical"] > 0
            ),
            "next_step_rule": (
                "Only positive lexical-first residual signal justifies learning "
                "static global residual prototypes and Hungarian assignment."
            ),
        },
    }
    metrics["signature"] = _signature(
        {
            "schema_version": EMBEDDING_PROXY_SCHEMA_VERSION,
            "m2a_manifest_sha256": sha256_file(m2a_dir / "manifest.json"),
            "query_stats_manifest_sha256": sha256_file(stats_manifest_path),
            "query_embeddings_sha256": embedding_contract["sha256"],
            "split": asdict(split),
        }
    )

    center_indices_path = staging_dir / "early_center_collision_indices.npy"
    np.save(center_indices_path, active_collision_indices.astype(np.int32))
    centers_path = staging_dir / "early_query_centers.npy"
    np.save(centers_path, early_query_centers.astype(np.float16))
    weights_path = staging_dir / "early_query_center_weight_sums.npy"
    np.save(weights_path, early_weight_sums)
    cases_path = staging_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path = staging_dir / "metrics.json"
    _json_dump(metrics_path, metrics)

    def contract(path: Path, array: np.ndarray | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if array is not None:
            value.update({"shape": list(array.shape), "dtype": str(array.dtype)})
        return value

    saved_indices = np.load(center_indices_path, mmap_mode="r")
    saved_centers = np.load(centers_path, mmap_mode="r")
    saved_weights = np.load(weights_path, mmap_mode="r")
    manifest: dict[str, Any] = {
        "schema_version": EMBEDDING_PROXY_SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "signature": metrics["signature"],
        "configuration": {
            "aggregation": (
                "early-only E2: log1p(pair_count) * "
                "log((early_covered_pois+1)/(early_query_df+1))"
            ),
            "holdout_subtraction": "full Train query stats minus exact frozen holdout counts",
            "query_residual": "normalize(query - current TIGER bucket mean center)",
            "poi_residual": "normalize(early_poi_query_center - current TIGER bucket mean center)",
            "fusion": "lexical evidence lexicographic first; residual cosine only breaks lexical ties",
            "evaluation_scope": "known gold TIGER collision bucket",
        },
        "inputs": {
            "m2a_manifest": {
                "path": str(m2a_dir / "manifest.json"),
                "sha256": sha256_file(m2a_dir / "manifest.json"),
            },
            "sft_manifest": {
                "path": str(sft_manifest_path),
                "sha256": sha256_file(sft_manifest_path),
            },
            "query_stats_manifest": {
                "path": str(stats_manifest_path),
                "sha256": sha256_file(stats_manifest_path),
            },
            "query_embeddings_manifest": {
                "path": str(query_embeddings_manifest_path),
                "sha256": sha256_file(query_embeddings_manifest_path),
            },
            "query_embeddings": {
                "path": str(query_embeddings_path),
                "sha256": embedding_contract["sha256"],
                "shape": list(query_embeddings.shape),
                "dtype": str(query_embeddings.dtype),
            },
            "order_sources": order_sources,
        },
        "outputs": {
            "metrics": contract(metrics_path),
            "cases": {**contract(cases_path), "rows": len(cases)},
            "arrays": {
                "early_center_collision_indices": contract(
                    center_indices_path, saved_indices
                ),
                "early_query_centers": contract(centers_path, saved_centers),
                "early_query_center_weight_sums": contract(
                    weights_path, saved_weights
                ),
            },
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "git": _git_state(project_root),
        "validation": {
            "m2a_baseline_exactly_reproduced": True,
            "full_train_counts_minus_holdout_equal_early": True,
            "holdout_labels_not_used_in_centers": True,
            "pretrained_query_embeddings_do_not_depend_on_target_labels": True,
            "early_poi_order_counts_equal_m2a": True,
        },
    }
    _json_dump(staging_dir / "manifest.json", manifest)
    (staging_dir / "_SUCCESS").touch()
    staging_dir.replace(output_dir)
    return EmbeddingProxyResult(metrics=metrics, manifest=manifest, output_dir=output_dir)


def validate_embedding_proxy_output(output_dir: Path) -> dict[str, Any]:
    """Validate immutable M2-C metrics, cases, and center arrays."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "M2-C manifest")
    if manifest.get("schema_version") != EMBEDDING_PROXY_SCHEMA_VERSION:
        raise QgrSidEmbeddingProxyError("M2-C schema_version 不受支持")
    if manifest.get("status") != "completed" or not (
        output_dir / "_SUCCESS"
    ).is_file():
        raise QgrSidEmbeddingProxyError("M2-C 输出未 completed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise QgrSidEmbeddingProxyError("M2-C manifest 缺少 outputs")
    contracts: dict[str, Any] = {}
    for name in ("metrics", "cases"):
        if not isinstance(outputs.get(name), dict):
            raise QgrSidEmbeddingProxyError(f"M2-C 缺少 {name}")
        contracts[name] = outputs[name]
    arrays = outputs.get("arrays")
    if not isinstance(arrays, dict):
        raise QgrSidEmbeddingProxyError("M2-C 缺少 arrays")
    contracts.update(arrays)
    checked: dict[str, str] = {}
    for name, item in contracts.items():
        path = output_dir / str(item.get("path", ""))
        if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
            raise QgrSidEmbeddingProxyError(f"M2-C 输出文件/字节数非法：{name}")
        digest = sha256_file(path)
        if digest != item.get("sha256"):
            raise QgrSidEmbeddingProxyError(f"M2-C 输出 SHA256 不一致：{name}")
        if name in arrays:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != item.get("shape") or str(
                array.dtype
            ) != item.get("dtype"):
                raise QgrSidEmbeddingProxyError(f"M2-C 数组契约不一致：{name}")
        checked[name] = digest
    return {
        "status": "validated",
        "schema_version": manifest["schema_version"],
        "signature": manifest["signature"],
        "checked_output_count": len(checked),
        "checked_outputs": checked,
    }

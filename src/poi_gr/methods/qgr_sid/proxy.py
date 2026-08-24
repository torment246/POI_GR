"""Train-time lexical proxy evaluation for QGR-SID relation paths."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from poi_gr.methods.qgr_sid.compiler import compile_relation_bucket
from poi_gr.methods.qgr_sid.relations import (
    DIRECT_NUMERIC_RELATION_TYPES,
    RELATION_SCHEMA_VERSION,
    RELATION_TYPES,
    QueryRelationFeatures,
    extract_numeric_relations,
    extract_query_relation_features,
)
from poi_gr.methods.tiger.identifier import sha256_file


PROXY_SCHEMA_VERSION = "qgr-sid-lexical-proxy-v1"
PROXY_METRICS_SCHEMA_VERSION = "qgr-sid-lexical-proxy-metrics-v1"
CATALOG_SCHEMA_VERSION = "qgr-sid-strict-relation-catalog-v1"
ORDER_REQUIRED_FIELDS = (
    "order_id",
    "searchid",
    "query",
    "create_time",
    "source_dt",
    "poi_id",
)
_TRAIN_MINUTE_PATTERN = re.compile(
    br'"create_time"\s*:\s*"(2026-07-(?:0[1-9]|1[0-2]) [0-9]{2}:[0-9]{2}):[0-9]{2}"'
)


class QgrSidProxyError(ValueError):
    """Raised when M2 proxy inputs or outputs violate the frozen protocol."""


@dataclass(frozen=True)
class TemporalSplit:
    """Exact event-count time split with a deterministic cutoff-minute tie rule."""

    train_order_count: int
    early_target_count: int
    holdout_target_count: int
    cutoff_minute: str
    early_before_cutoff_minute: int
    cutoff_minute_order_count: int
    early_from_cutoff_minute: int


@dataclass(frozen=True)
class StrictRelationCatalog:
    """Collision-only numeric relations aligned to compact POI indices."""

    poi_ids: np.ndarray
    bucket_keys: np.ndarray
    relations: np.ndarray
    conflict_masks: np.ndarray
    input_poi_count: int
    colliding_poi_count: int


@dataclass(frozen=True)
class ProxyResult:
    """Completed M2-A lexical proxy artifacts."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


@dataclass(frozen=True)
class _HoldoutEvent:
    collision_index: int
    query: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QgrSidProxyError(f"无法读取{name}：{path}：{error}") from error
    if not isinstance(payload, dict):
        raise QgrSidProxyError(f"{name}必须是 JSON 对象：{path}")
    return payload


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def _signature(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _bucket_key(s1: int, s2: int, s3: int) -> int:
    if not (0 <= s1 < 2**21 and 0 <= s2 < 2**21 and 0 <= s3 < 2**21):
        raise QgrSidProxyError("基础 SID Token 超出审计打包范围")
    return (s1 << 42) | (s2 << 21) | s3


def _iter_raw_pois(poi_dir: Path) -> Iterator[dict[str, Any]]:
    paths = tuple(sorted(poi_dir.glob("part-*.json")))
    if not paths:
        raise QgrSidProxyError(f"POI 目录中没有 part-*.json：{poi_dir}")
    for path in paths:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise QgrSidProxyError(
                        f"POI JSON 解析失败：{path}:{line_number}"
                    ) from error
                if not isinstance(payload, dict):
                    raise QgrSidProxyError(
                        f"POI JSON 行必须是对象：{path}:{line_number}"
                    )
                yield payload


def build_strict_relation_catalog(
    *,
    poi_dir: Path,
    identifier_dir: Path,
    value_max: int,
    batch_rows: int,
    progress: Callable[[str], None] | None = None,
) -> StrictRelationCatalog:
    """Rebuild conflict-filtered M1 relations for all TIGER collision POIs."""

    manifest = _load_json(
        identifier_dir / "tiger_id_manifest.json", "TIGER identifier manifest"
    )
    tiger_metrics = _load_json(identifier_dir / "metrics.json", "TIGER metrics")
    mapping_path = identifier_dir / "poi_tiger_id_mapping.parquet"
    mapping_contract = manifest.get("mapping")
    if not isinstance(mapping_contract, dict):
        raise QgrSidProxyError("TIGER manifest 缺少 mapping 契约")
    if sha256_file(mapping_path) != mapping_contract.get("sha256"):
        raise QgrSidProxyError("TIGER mapping SHA256 与 manifest 不一致")
    poi_count = int(mapping_contract.get("rows", 0))
    colliding_count = int(tiger_metrics["base_sid_colliding_poi_count"])
    if poi_count <= 0 or colliding_count <= 0:
        raise QgrSidProxyError("TIGER POI/碰撞计数非法")

    poi_ids = np.empty(colliding_count, dtype=np.int64)
    bucket_keys = np.empty(colliding_count, dtype=np.int64)
    relations = np.full(
        (colliding_count, len(RELATION_TYPES)), -1, dtype=np.int16
    )
    conflict_masks = np.zeros(colliding_count, dtype=np.uint32)
    relation_index = {name: index for index, name in enumerate(RELATION_TYPES)}
    raw_iterator = _iter_raw_pois(poi_dir)
    parquet = pq.ParquetFile(mapping_path)
    compact_row = 0
    scanned = 0
    columns = ["poi_id", "s1", "s2", "s3", "base_sid_bucket_size"]
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        values = {
            name: batch.column(index).to_pylist()
            for index, name in enumerate(columns)
        }
        for offset in range(batch.num_rows):
            try:
                poi = next(raw_iterator)
            except StopIteration as error:
                raise QgrSidProxyError("POI 原始表早于 TIGER mapping 结束") from error
            mapping_poi_id = str(values["poi_id"][offset])
            if mapping_poi_id != str(poi.get("poi_id", "")):
                raise QgrSidProxyError(
                    f"POI/TIGER 行对齐失败：row={scanned}, poi_id={mapping_poi_id}"
                )
            scanned += 1
            if int(values["base_sid_bucket_size"][offset]) <= 1:
                continue
            if compact_row >= colliding_count:
                raise QgrSidProxyError("实际碰撞 POI 超过 TIGER metrics")
            try:
                poi_ids[compact_row] = int(mapping_poi_id)
            except (TypeError, ValueError, OverflowError) as error:
                raise QgrSidProxyError("POI ID 不能表示为 int64") from error
            bucket_keys[compact_row] = _bucket_key(
                int(values["s1"][offset]),
                int(values["s2"][offset]),
                int(values["s3"][offset]),
            )
            extraction = extract_numeric_relations(poi)
            conflict_types = {
                conflict.relation_type for conflict in extraction.conflicts
            }
            mask = 0
            for conflict_type in conflict_types:
                mask |= 1 << relation_index[conflict_type]
            conflict_masks[compact_row] = mask
            for relation in extraction.relations:
                index = relation_index[relation.relation_type]
                if relation.relation_type in conflict_types:
                    continue
                if 0 <= relation.numeric_value <= value_max:
                    relations[compact_row, index] = relation.numeric_value
            compact_row += 1
        if progress is not None:
            progress(
                f"关系目录已扫描 {scanned:,}/{poi_count:,} POI；"
                f"碰撞 POI {compact_row:,}/{colliding_count:,}"
            )
    if scanned != poi_count or compact_row != colliding_count:
        raise QgrSidProxyError(
            f"关系目录计数不守恒：POI {scanned}/{poi_count}，"
            f"碰撞 {compact_row}/{colliding_count}"
        )
    try:
        extra = next(raw_iterator)
    except StopIteration:
        extra = None
    if extra is not None:
        raise QgrSidProxyError("POI 原始表行数多于 TIGER mapping")
    if len(np.unique(poi_ids)) != colliding_count:
        raise QgrSidProxyError("碰撞 POI ID 不唯一")
    return StrictRelationCatalog(
        poi_ids=poi_ids,
        bucket_keys=bucket_keys,
        relations=relations,
        conflict_masks=conflict_masks,
        input_poi_count=poi_count,
        colliding_poi_count=colliding_count,
    )


def profile_temporal_split(
    order_dir: Path,
    *,
    early_ratio: float,
) -> TemporalSplit:
    """Find the minute containing the exact early/holdout event-count boundary."""

    if not 0.5 <= early_ratio < 1.0:
        raise QgrSidProxyError("early_ratio 必须位于 [0.5,1.0)")
    minute_counts: Counter[str] = Counter()
    for path in sorted(order_dir.glob("part-*.json")):
        with path.open("rb") as handle:
            for line in handle:
                match = _TRAIN_MINUTE_PATTERN.search(line)
                if match:
                    minute_counts[match.group(1).decode("ascii")] += 1
    train_count = sum(minute_counts.values())
    if train_count <= 0:
        raise QgrSidProxyError("原始订单中没有 Train 时间范围数据")
    early_target = int(math.floor(train_count * early_ratio + 0.5))
    cumulative = 0
    cutoff_minute = ""
    minute_count = 0
    for minute in sorted(minute_counts):
        count = minute_counts[minute]
        if cumulative + count >= early_target:
            cutoff_minute = minute
            minute_count = count
            break
        cumulative += count
    if not cutoff_minute:
        raise QgrSidProxyError("无法解析 Train 时间切分点")
    return TemporalSplit(
        train_order_count=train_count,
        early_target_count=early_target,
        holdout_target_count=train_count - early_target,
        cutoff_minute=cutoff_minute,
        early_before_cutoff_minute=cumulative,
        cutoff_minute_order_count=minute_count,
        early_from_cutoff_minute=early_target - cumulative,
    )


def _split_tie_key(record: Mapping[str, Any]) -> bytes:
    payload = "\x1f".join(
        str(record.get(field, ""))
        for field in ("create_time", "order_id", "searchid")
    ).encode("utf-8")
    return hashlib.blake2b(
        payload,
        digest_size=16,
        person=b"qgr-split-v1",
    ).digest()


def _query_arrays(features: QueryRelationFeatures) -> tuple[np.ndarray, set[int]]:
    typed = np.full(len(RELATION_TYPES), -1, dtype=np.int16)
    relation_index = {name: index for index, name in enumerate(RELATION_TYPES)}
    for relation_type, value in features.typed_relations:
        if 0 <= value <= np.iinfo(np.int16).max:
            typed[relation_index[relation_type]] = value
    return typed, set(features.numeric_values)


def _relation_matches(
    target_relations: np.ndarray,
    features: QueryRelationFeatures,
) -> tuple[np.ndarray, np.ndarray]:
    typed, numeric_values = _query_arrays(features)
    available = target_relations >= 0
    typed_match = available & (typed == target_relations)
    value_match = typed_match.copy()
    for index, relation_type in enumerate(RELATION_TYPES):
        if (
            available[index]
            and relation_type in DIRECT_NUMERIC_RELATION_TYPES
            and int(target_relations[index]) in numeric_values
        ):
            value_match[index] = True
    return typed_match, value_match


def _read_order_events(
    *,
    order_dir: Path,
    sft_manifest: Mapping[str, Any],
    split: TemporalSplit,
    catalog: StrictRelationCatalog,
    progress: Callable[[str], None] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[_HoldoutEvent],
    dict[str, Any],
    list[dict[str, Any]],
]:
    collision_lookup = {
        int(poi_id): index for index, poi_id in enumerate(catalog.poi_ids)
    }
    early_orders = np.zeros(catalog.colliding_poi_count, dtype=np.int32)
    early_matches = np.zeros_like(catalog.relations, dtype=np.int32)
    holdout_events: list[_HoldoutEvent] = []
    tie_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    early_relation_support = np.zeros(len(RELATION_TYPES), dtype=np.int64)
    early_relation_typed_match = np.zeros(len(RELATION_TYPES), dtype=np.int64)
    early_relation_value_match = np.zeros(len(RELATION_TYPES), dtype=np.int64)

    orders_contract = sft_manifest.get("orders")
    if not isinstance(orders_contract, dict):
        raise QgrSidProxyError("SFT manifest 缺少 orders 契约")
    expected_files = {
        str(item.get("relative_path")): item
        for item in orders_contract.get("files", [])
        if isinstance(item, dict)
    }
    actual_paths = tuple(sorted(order_dir.glob("part-*.json")))
    if set(path.name for path in actual_paths) != set(expected_files):
        raise QgrSidProxyError("原始订单文件集合与 SFT manifest 不一致")
    source_outputs: list[dict[str, Any]] = []

    def process(record: Mapping[str, Any], split_name: str) -> None:
        counts[f"{split_name}_orders"] += 1
        try:
            poi_id = int(record["poi_id"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise QgrSidProxyError("订单 poi_id 非法") from error
        collision_index = collision_lookup.get(poi_id)
        if collision_index is None:
            return
        counts[f"{split_name}_collision_orders"] += 1
        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise QgrSidProxyError("Train 订单 Query 为空")
        if split_name == "holdout":
            holdout_events.append(
                _HoldoutEvent(collision_index=collision_index, query=query)
            )
            return
        early_orders[collision_index] += 1
        features = extract_query_relation_features(query)
        typed_match, value_match = _relation_matches(
            catalog.relations[collision_index], features
        )
        early_matches[collision_index] += value_match.astype(np.int32)
        available = catalog.relations[collision_index] >= 0
        early_relation_support[:] += available
        early_relation_typed_match[:] += typed_match
        early_relation_value_match[:] += value_match

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
                    raise QgrSidProxyError(
                        f"订单 JSON 解析失败：{path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise QgrSidProxyError("订单 JSON 行必须是对象")
                missing = [field for field in ORDER_REQUIRED_FIELDS if field not in record]
                if missing:
                    raise QgrSidProxyError("订单缺少字段：" + ", ".join(missing))
                create_time = str(record["create_time"])
                source_dt = str(record["source_dt"])
                if len(create_time) < 16 or source_dt != create_time[:10].replace("-", ""):
                    raise QgrSidProxyError("订单 source_dt/create_time 不一致")
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
                        f"订单已扫描 {scanned:,}；早期碰撞请求 "
                        f"{counts['early_collision_orders']:,}；"
                        f"末段碰撞请求 {counts['holdout_collision_orders']:,}"
                    )
        contract = expected_files[path.name]
        digest_value = digest.hexdigest()
        if file_rows != int(contract.get("rows", -1)):
            raise QgrSidProxyError(f"订单分片行数与 manifest 不一致：{path.name}")
        if digest_value != contract.get("sha256"):
            raise QgrSidProxyError(f"订单分片 SHA256 与 manifest 不一致：{path.name}")
        source_outputs.append(
            {
                "path": str(path.resolve()),
                "rows": file_rows,
                "sha256": digest_value,
            }
        )

    if len(tie_records) != split.cutoff_minute_order_count:
        raise QgrSidProxyError("切分分钟订单数与预扫描不一致")
    tie_records.sort(key=_split_tie_key)
    for index, record in enumerate(tie_records):
        process(
            record,
            "early" if index < split.early_from_cutoff_minute else "holdout",
        )
    if counts["early_orders"] != split.early_target_count:
        raise QgrSidProxyError("早期 90% 订单数不守恒")
    if counts["holdout_orders"] != split.holdout_target_count:
        raise QgrSidProxyError("末段 10% 订单数不守恒")
    per_relation = {
        relation_type: {
            "early_support_order_count": int(early_relation_support[index]),
            "early_typed_match_count": int(early_relation_typed_match[index]),
            "early_typed_match_ratio": _ratio(
                early_relation_typed_match[index], early_relation_support[index]
            ),
            "early_value_aware_match_count": int(
                early_relation_value_match[index]
            ),
            "early_value_aware_match_ratio": _ratio(
                early_relation_value_match[index], early_relation_support[index]
            ),
        }
        for index, relation_type in enumerate(RELATION_TYPES)
    }
    return (
        early_orders,
        early_matches,
        holdout_events,
        {**dict(counts), "per_relation": per_relation},
        source_outputs,
    )


def _compile_all_buckets(
    *,
    catalog: StrictRelationCatalog,
    early_orders: np.ndarray,
    early_matches: np.ndarray,
    global_predictability: np.ndarray,
    query_guided: bool,
    max_pairs: int,
    p99_order_count: float,
    prior_orders: float,
    progress: Callable[[str], None] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    order = np.argsort(catalog.bucket_keys, kind="stable")
    sorted_keys = catalog.bucket_keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]
    path_types = np.full(
        (catalog.colliding_poi_count, max_pairs), -1, dtype=np.int16
    )
    path_values = np.full_like(path_types, -1)
    resolved = np.zeros(catalog.colliding_poi_count, dtype=np.bool_)
    decision_count = 0
    for bucket_number, (start, stop) in enumerate(
        zip(starts, stops, strict=True), start=1
    ):
        indices = order[start:stop]
        compilation = compile_relation_bucket(
            catalog.relations[indices],
            early_orders[indices],
            early_matches[indices],
            global_predictability,
            query_guided=query_guided,
            max_pairs=max_pairs,
            p99_order_count=p99_order_count,
            prior_orders=prior_orders,
        )
        path_types[indices] = compilation.path_types
        path_values[indices] = compilation.path_values
        resolved[indices] = compilation.relation_only_resolved
        decision_count += compilation.decision_count
        if progress is not None and bucket_number % 50_000 == 0:
            progress(
                f"{'Query 引导' if query_guided else '静态'}关系树已编译 "
                f"{bucket_number:,}/{len(starts):,} 桶"
            )
    lengths = (path_types >= 0).sum(axis=1)
    metrics = {
        "collision_bucket_count": len(starts),
        "decision_node_count": decision_count,
        "relation_only_resolved_poi_count": int(resolved.sum()),
        "relation_only_resolved_poi_ratio": _ratio(
            int(resolved.sum()), catalog.colliding_poi_count
        ),
        "early_order_resolved_count": int(early_orders[resolved].sum()),
        "early_order_resolved_ratio": _ratio(
            int(early_orders[resolved].sum()), int(early_orders.sum())
        ),
        "path_length_distribution": {
            str(length): int((lengths == length).sum())
            for length in range(max_pairs + 1)
        },
    }
    return path_types, path_values, resolved, metrics, order, starts


def _rank_for_event(
    *,
    candidates: np.ndarray,
    target: int,
    path_types: np.ndarray,
    path_values: np.ndarray,
    early_orders: np.ndarray,
    poi_ids: np.ndarray,
    features: QueryRelationFeatures,
) -> tuple[int, bool]:
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
    ranking = np.lexsort(
        (
            poi_ids[candidates],
            -early_orders[candidates].astype(np.int64),
            -complete.astype(np.int8),
            -untyped_count,
            -net_typed,
        )
    )
    target_positions = np.flatnonzero(candidates[ranking] == target)
    if len(target_positions) != 1:
        raise QgrSidProxyError("已知桶候选中目标 POI 不唯一")
    target_local = int(np.flatnonzero(candidates == target)[0])
    return int(target_positions[0]) + 1, bool(complete[target_local])


def _ranking_metrics(ranks: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(ranks, dtype=np.int32)
    if not len(values):
        return {
            "order_count": 0,
            **{f"hr_at_{k}": 0.0 for k in (1, 3, 5, 10)},
            "ndcg_at_10": 0.0,
            "mean_rank": 0.0,
        }
    return {
        "order_count": len(values),
        **{f"hr_at_{k}": float(np.mean(values <= k)) for k in (1, 3, 5, 10)},
        "ndcg_at_10": float(
            np.mean(
                np.where(
                    values <= 10,
                    1.0 / np.log2(values.astype(np.float64) + 1.0),
                    0.0,
                )
            )
        ),
        "mean_rank": float(values.mean()),
    }


def _evaluate_holdout(
    *,
    holdout_events: Sequence[_HoldoutEvent],
    catalog: StrictRelationCatalog,
    early_orders: np.ndarray,
    bucket_order: np.ndarray,
    bucket_starts: np.ndarray,
    static_types: np.ndarray,
    static_values: np.ndarray,
    static_resolved: np.ndarray,
    guided_types: np.ndarray,
    guided_values: np.ndarray,
    guided_resolved: np.ndarray,
    examples_per_kind: int,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inverse_order = np.empty(len(bucket_order), dtype=np.int64)
    inverse_order[bucket_order] = np.arange(len(bucket_order), dtype=np.int64)
    group_ids = np.empty(len(bucket_order), dtype=np.int32)
    stops = np.r_[bucket_starts[1:], len(bucket_order)]
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

    ranks = {"popularity": [], "static": [], "query_guided": []}
    exact_paths = {"static": 0, "query_guided": 0}
    resolved_orders = {"static": 0, "query_guided": 0}
    relation_support = np.zeros(len(RELATION_TYPES), dtype=np.int64)
    relation_typed_match = np.zeros(len(RELATION_TYPES), dtype=np.int64)
    relation_value_match = np.zeros(len(RELATION_TYPES), dtype=np.int64)
    cases: list[dict[str, Any]] = []
    case_counts: Counter[str] = Counter()

    for event_number, event in enumerate(holdout_events, start=1):
        target = event.collision_index
        position = int(inverse_order[target])
        group_id = int(group_ids[position])
        start = int(bucket_starts[group_id])
        stop = int(stops[group_id])
        candidates = bucket_order[start:stop]
        features = extract_query_relation_features(event.query)
        typed_match, value_match = _relation_matches(
            catalog.relations[target], features
        )
        available = catalog.relations[target] >= 0
        relation_support += available
        relation_typed_match += typed_match
        relation_value_match += value_match

        popularity = int(popularity_rank[target])
        static_rank, static_exact = _rank_for_event(
            candidates=candidates,
            target=target,
            path_types=static_types,
            path_values=static_values,
            early_orders=early_orders,
            poi_ids=catalog.poi_ids,
            features=features,
        )
        guided_rank, guided_exact = _rank_for_event(
            candidates=candidates,
            target=target,
            path_types=guided_types,
            path_values=guided_values,
            early_orders=early_orders,
            poi_ids=catalog.poi_ids,
            features=features,
        )
        ranks["popularity"].append(popularity)
        ranks["static"].append(static_rank)
        ranks["query_guided"].append(guided_rank)
        exact_paths["static"] += int(static_exact)
        exact_paths["query_guided"] += int(guided_exact)
        resolved_orders["static"] += int(static_resolved[target])
        resolved_orders["query_guided"] += int(guided_resolved[target])

        for name, rank, path_types, path_values in (
            ("static", static_rank, static_types, static_values),
            ("query_guided", guided_rank, guided_types, guided_values),
        ):
            kind = (
                f"{name}_improved"
                if rank < popularity
                else f"{name}_worsened"
                if rank > popularity
                else ""
            )
            if kind and case_counts[kind] < examples_per_kind:
                emitted = path_types[target] >= 0
                cases.append(
                    {
                        "case_kind": kind,
                        "query": event.query,
                        "target_poi_id": str(int(catalog.poi_ids[target])),
                        "bucket_size": len(candidates),
                        "popularity_rank": popularity,
                        "relation_rank": rank,
                        "target_path": [
                            {
                                "relation_type": RELATION_TYPES[int(type_index)],
                                "numeric_value": int(value),
                            }
                            for type_index, value in zip(
                                path_types[target][emitted],
                                path_values[target][emitted],
                                strict=True,
                            )
                        ],
                        "query_typed_relations": [
                            list(item) for item in features.typed_relations
                        ],
                        "query_numeric_values": list(features.numeric_values),
                    }
                )
                case_counts[kind] += 1
        if progress is not None and event_number % 50_000 == 0:
            progress(
                f"末段代理已评测 {event_number:,}/{len(holdout_events):,} 碰撞请求"
            )

    per_relation = {
        relation_type: {
            "holdout_support_order_count": int(relation_support[index]),
            "holdout_typed_match_count": int(relation_typed_match[index]),
            "holdout_typed_match_ratio": _ratio(
                relation_typed_match[index], relation_support[index]
            ),
            "holdout_value_aware_match_count": int(relation_value_match[index]),
            "holdout_value_aware_match_ratio": _ratio(
                relation_value_match[index], relation_support[index]
            ),
        }
        for index, relation_type in enumerate(RELATION_TYPES)
    }
    metrics = {
        "collision_holdout_order_count": len(holdout_events),
        "ranking": {name: _ranking_metrics(value) for name, value in ranks.items()},
        "static": {
            "relation_only_resolved_order_count": resolved_orders["static"],
            "relation_only_resolved_order_ratio": _ratio(
                resolved_orders["static"], len(holdout_events)
            ),
            "exact_lexical_path_order_count": exact_paths["static"],
            "exact_lexical_path_order_ratio": _ratio(
                exact_paths["static"], len(holdout_events)
            ),
        },
        "query_guided": {
            "relation_only_resolved_order_count": resolved_orders["query_guided"],
            "relation_only_resolved_order_ratio": _ratio(
                resolved_orders["query_guided"], len(holdout_events)
            ),
            "exact_lexical_path_order_count": exact_paths["query_guided"],
            "exact_lexical_path_order_ratio": _ratio(
                exact_paths["query_guided"], len(holdout_events)
            ),
        },
        "per_relation": per_relation,
    }
    return metrics, cases


def _array_contract(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _save_array(
    staging_dir: Path,
    name: str,
    array: np.ndarray,
) -> dict[str, Any]:
    path = staging_dir / f"{name}.npy"
    np.save(path, array, allow_pickle=False)
    return _array_contract(path, array)


def _validate_input_manifest(
    path: Path,
    *,
    name: str,
    schema_version: str,
) -> dict[str, Any]:
    payload = _load_json(path, name)
    if payload.get("schema_version") != schema_version:
        raise QgrSidProxyError(f"{name} schema_version 不受支持")
    if payload.get("status") != "completed":
        raise QgrSidProxyError(f"{name} 尚未 completed")
    return payload


def evaluate_lexical_relation_proxy(
    *,
    project_root: Path,
    poi_dir: Path,
    order_dir: Path,
    sft_manifest_path: Path,
    identifier_dir: Path,
    m1_manifest_path: Path,
    output_dir: Path,
    early_ratio: float = 0.9,
    value_max: int = 1055,
    max_pairs: int = 3,
    batch_rows: int = 65_536,
    prior_orders: float = 20.0,
    examples_per_kind: int = 20,
    progress: Callable[[str], None] | None = None,
) -> ProxyResult:
    """Run the frozen M2-A lexical-only proxy on a true Train 90/10 split."""

    started_at = _utc_now()
    started = time.monotonic()
    project_root = project_root.resolve()
    poi_dir = poi_dir.resolve()
    order_dir = order_dir.resolve()
    sft_manifest_path = sft_manifest_path.resolve()
    identifier_dir = identifier_dir.resolve()
    m1_manifest_path = m1_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if value_max < 0 or value_max > np.iinfo(np.int16).max:
        raise QgrSidProxyError("value_max 必须位于 int16 非负范围")
    if max_pairs <= 0:
        raise QgrSidProxyError("max_pairs 必须大于 0")
    if batch_rows <= 0:
        raise QgrSidProxyError("batch_rows 必须大于 0")
    if prior_orders <= 0:
        raise QgrSidProxyError("prior_orders 必须大于 0")
    if examples_per_kind < 0:
        raise QgrSidProxyError("examples_per_kind 不能为负数")
    for path, name in (
        (project_root, "项目根目录"),
        (poi_dir, "POI 目录"),
        (order_dir, "订单目录"),
        (identifier_dir, "TIGER identifier 目录"),
    ):
        if not path.is_dir():
            raise QgrSidProxyError(f"{name}不存在：{path}")
    if output_dir.exists():
        raise QgrSidProxyError(f"输出目录已存在，拒绝覆盖：{output_dir}")

    sft_manifest = _validate_input_manifest(
        sft_manifest_path,
        name="SFT data manifest",
        schema_version="sft-main-data-v1",
    )
    _validate_input_manifest(
        m1_manifest_path,
        name="QGR-SID M1 manifest",
        schema_version="qgr-sid-numeric-relation-audit-v1",
    )
    tiger_manifest_path = identifier_dir / "tiger_id_manifest.json"
    tiger_metrics_path = identifier_dir / "metrics.json"
    tiger_manifest = _validate_input_manifest(
        tiger_manifest_path,
        name="TIGER identifier manifest",
        schema_version="tiger-item-identifier-v1",
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}"
    if staging_dir.exists():
        raise QgrSidProxyError(f"临时输出目录已存在：{staging_dir}")
    staging_dir.mkdir()

    if progress is not None:
        progress("第一遍扫描原始 Train 时间分布，冻结真实 90/10 边界")
    split = profile_temporal_split(order_dir, early_ratio=early_ratio)
    outputs_contract = sft_manifest.get("outputs")
    if not isinstance(outputs_contract, dict):
        raise QgrSidProxyError("SFT manifest 缺少 outputs")
    train_contract = outputs_contract.get("train.jsonl")
    if not isinstance(train_contract, dict):
        raise QgrSidProxyError("SFT manifest 缺少 train.jsonl 契约")
    if split.train_order_count != int(train_contract.get("rows", -1)):
        raise QgrSidProxyError("原始 Train 订单数与冻结 SFT Train 不一致")

    if progress is not None:
        progress("重建 M1 严格无冲突关系目录，并与 TIGER mapping 全量对齐")
    catalog = build_strict_relation_catalog(
        poi_dir=poi_dir,
        identifier_dir=identifier_dir,
        value_max=value_max,
        batch_rows=batch_rows,
        progress=progress,
    )
    if progress is not None:
        progress("第二遍扫描订单，严格分离 early 90% 与 holdout 10%")
    (
        early_orders,
        early_matches,
        holdout_events,
        order_metrics,
        order_sources,
    ) = _read_order_events(
        order_dir=order_dir,
        sft_manifest=sft_manifest,
        split=split,
        catalog=catalog,
        progress=progress,
    )
    positive_orders = early_orders[early_orders > 0]
    p99_order_count = (
        0.0
        if not len(positive_orders)
        else float(np.percentile(positive_orders, 99))
    )
    global_predictability = np.asarray(
        [
            order_metrics["per_relation"][relation_type][
                "early_value_aware_match_ratio"
            ]
            for relation_type in RELATION_TYPES
        ],
        dtype=np.float64,
    )

    if progress is not None:
        progress("编译不使用 Query 可预测性的静态变长关系树")
    (
        static_types,
        static_values,
        static_resolved,
        static_metrics,
        bucket_order,
        bucket_starts,
    ) = _compile_all_buckets(
        catalog=catalog,
        early_orders=early_orders,
        early_matches=early_matches,
        global_predictability=global_predictability,
        query_guided=False,
        max_pairs=max_pairs,
        p99_order_count=p99_order_count,
        prior_orders=prior_orders,
        progress=progress,
    )
    if progress is not None:
        progress("编译乘以 early Query 可预测性的 Query 引导变长关系树")
    (
        guided_types,
        guided_values,
        guided_resolved,
        guided_metrics,
        guided_order,
        guided_starts,
    ) = _compile_all_buckets(
        catalog=catalog,
        early_orders=early_orders,
        early_matches=early_matches,
        global_predictability=global_predictability,
        query_guided=True,
        max_pairs=max_pairs,
        p99_order_count=p99_order_count,
        prior_orders=prior_orders,
        progress=progress,
    )
    if not np.array_equal(bucket_order, guided_order) or not np.array_equal(
        bucket_starts, guided_starts
    ):
        raise QgrSidProxyError("静态树与 Query 引导树的桶顺序不一致")

    if progress is not None:
        progress("仅在冻结 holdout 10% 上评测已知 gold bucket 的代理排序")
    holdout_metrics, cases = _evaluate_holdout(
        holdout_events=holdout_events,
        catalog=catalog,
        early_orders=early_orders,
        bucket_order=bucket_order,
        bucket_starts=bucket_starts,
        static_types=static_types,
        static_values=static_values,
        static_resolved=static_resolved,
        guided_types=guided_types,
        guided_values=guided_values,
        guided_resolved=guided_resolved,
        examples_per_kind=examples_per_kind,
        progress=progress,
    )

    ranking = holdout_metrics["ranking"]
    popularity = ranking["popularity"]
    static_ranking = ranking["static"]
    guided_ranking = ranking["query_guided"]
    deltas = {
        "static_hr_at_1_vs_popularity": static_ranking["hr_at_1"]
        - popularity["hr_at_1"],
        "static_ndcg_at_10_vs_popularity": static_ranking["ndcg_at_10"]
        - popularity["ndcg_at_10"],
        "query_guided_hr_at_1_vs_popularity": guided_ranking["hr_at_1"]
        - popularity["hr_at_1"],
        "query_guided_ndcg_at_10_vs_popularity": guided_ranking["ndcg_at_10"]
        - popularity["ndcg_at_10"],
        "query_guided_hr_at_1_vs_static": guided_ranking["hr_at_1"]
        - static_ranking["hr_at_1"],
        "query_guided_ndcg_at_10_vs_static": guided_ranking["ndcg_at_10"]
        - static_ranking["ndcg_at_10"],
    }
    relation_any = np.any(catalog.relations >= 0, axis=1)
    conflicted = catalog.conflict_masks != 0
    catalog_per_relation = {
        relation_type: {
            "strict_poi_count": int((catalog.relations[:, index] >= 0).sum()),
            "conflict_poi_count": int(
                ((catalog.conflict_masks & (1 << index)) != 0).sum()
            ),
        }
        for index, relation_type in enumerate(RELATION_TYPES)
    }
    metrics: dict[str, Any] = {
        "schema_version": PROXY_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "method": "TIGER lexical numeric relation M2-A proxy",
        "warning": (
            "Known-gold-bucket lexical proxy only; it is neither unconstrained "
            "generation nor downstream SFT evidence."
        ),
        "temporal_split": asdict(split),
        "catalog": {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "input_poi_count": catalog.input_poi_count,
            "collision_poi_count": catalog.colliding_poi_count,
            "collision_bucket_count": int(len(bucket_starts)),
            "strict_relation_poi_count": int(relation_any.sum()),
            "strict_relation_poi_ratio": _ratio(
                int(relation_any.sum()), catalog.colliding_poi_count
            ),
            "no_strict_relation_poi_count": int((~relation_any).sum()),
            "conflicted_poi_count": int(conflicted.sum()),
            "conflicted_poi_ratio": _ratio(
                int(conflicted.sum()), catalog.colliding_poi_count
            ),
            "per_relation": catalog_per_relation,
        },
        "early": {
            **order_metrics,
            "positive_collision_poi_count": int((early_orders > 0).sum()),
            "p99_positive_poi_order_count": p99_order_count,
            "global_value_aware_predictability": {
                relation_type: float(global_predictability[index])
                for index, relation_type in enumerate(RELATION_TYPES)
            },
        },
        "compilation": {
            "static": static_metrics,
            "query_guided": guided_metrics,
            "different_path_poi_count": int(
                np.any(
                    (static_types != guided_types)
                    | (static_values != guided_values),
                    axis=1,
                ).sum()
            ),
        },
        "holdout": holdout_metrics,
        "deltas": deltas,
        "decision": {
            "lexical_proxy_positive_vs_popularity": bool(
                deltas["query_guided_hr_at_1_vs_popularity"] > 0
                and deltas["query_guided_ndcg_at_10_vs_popularity"] > 0
            ),
            "query_guidance_positive_vs_static": bool(
                deltas["query_guided_hr_at_1_vs_static"] > 0
                and deltas["query_guided_ndcg_at_10_vs_static"] > 0
            ),
            "next_step_rule": (
                "Only positive lexical signal justifies adding GEO/GID in M2-B; "
                "SFT remains blocked until the complete offline gate passes."
            ),
        },
    }
    metrics["signature"] = _signature(
        {
            "schema_version": PROXY_SCHEMA_VERSION,
            "relation_schema_version": RELATION_SCHEMA_VERSION,
            "sft_manifest_sha256": sha256_file(sft_manifest_path),
            "m1_manifest_sha256": sha256_file(m1_manifest_path),
            "tiger_manifest_sha256": sha256_file(tiger_manifest_path),
            "configuration": {
                "early_ratio": early_ratio,
                "value_max": value_max,
                "max_pairs": max_pairs,
                "prior_orders": prior_orders,
            },
        }
    )

    arrays = {
        "collision_poi_ids": catalog.poi_ids,
        "base_sid_keys": catalog.bucket_keys,
        "strict_relations": catalog.relations,
        "conflict_masks": catalog.conflict_masks,
        "early_order_counts": early_orders,
        "early_relation_match_counts": early_matches,
        "static_path_types": static_types,
        "static_path_values": static_values,
        "static_resolved": static_resolved,
        "query_guided_path_types": guided_types,
        "query_guided_path_values": guided_values,
        "query_guided_resolved": guided_resolved,
    }
    array_contracts = {
        name: _save_array(staging_dir, name, array)
        for name, array in arrays.items()
    }
    metrics_path = staging_dir / "metrics.json"
    _json_dump(metrics_path, metrics)
    cases_path = staging_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n"
            )

    mapping_contract = tiger_manifest.get("mapping")
    if not isinstance(mapping_contract, dict):
        raise QgrSidProxyError("TIGER manifest 缺少 mapping 契约")
    mapping_path = identifier_dir / "poi_tiger_id_mapping.parquet"
    finished_at = _utc_now()
    manifest: dict[str, Any] = {
        "schema_version": PROXY_SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "signature": metrics["signature"],
        "configuration": {
            "early_ratio": early_ratio,
            "split_rule": (
                "chronological event-count 90/10; cutoff-minute events sorted "
                "by BLAKE2b(create_time,order_id,searchid)"
            ),
            "value_token_range": [0, value_max],
            "max_relation_pairs": max_pairs,
            "query_usage": (
                "early-only relation-value predictability and holdout lexical "
                "ranking; never changes a POI identifier per request"
            ),
            "missing_branch_consumes_pair": False,
            "compiler_weight": "sqrt(1 + min(early_poi_orders, early_positive_p99))",
            "query_predictability_prior_orders": prior_orders,
            "evaluation_scope": "known gold TIGER collision bucket",
        },
        "inputs": {
            "poi_dir": str(poi_dir),
            "order_dir": str(order_dir),
            "order_sources": order_sources,
            "sft_manifest": {
                "path": str(sft_manifest_path),
                "sha256": sha256_file(sft_manifest_path),
            },
            "m1_manifest": {
                "path": str(m1_manifest_path),
                "sha256": sha256_file(m1_manifest_path),
            },
            "tiger_identifier_manifest": {
                "path": str(tiger_manifest_path),
                "sha256": sha256_file(tiger_manifest_path),
            },
            "tiger_identifier_metrics": {
                "path": str(tiger_metrics_path),
                "sha256": sha256_file(tiger_metrics_path),
            },
            "tiger_mapping": {
                "path": str(mapping_path),
                "rows": int(mapping_contract.get("rows", 0)),
                "sha256": sha256_file(mapping_path),
            },
        },
        "outputs": {
            "metrics": {
                "path": metrics_path.name,
                "bytes": metrics_path.stat().st_size,
                "sha256": sha256_file(metrics_path),
            },
            "cases": {
                "path": cases_path.name,
                "bytes": cases_path.stat().st_size,
                "sha256": sha256_file(cases_path),
                "rows": len(cases),
            },
            "arrays": array_contracts,
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "git": _git_state(project_root),
        "validation": {
            "true_temporal_holdout_not_used_for_compilation": True,
            "cutoff_ties_deterministic": True,
            "raw_order_files_match_sft_manifest": True,
            "poi_mapping_row_alignment": True,
            "collision_poi_count_conserved": True,
            "collision_bucket_count_conserved": True,
            "paths_are_pure_numeric_type_value_pairs": True,
            "max_relation_pairs_respected": bool(
                np.all((static_types >= 0).sum(axis=1) <= max_pairs)
                and np.all((guided_types >= 0).sum(axis=1) <= max_pairs)
            ),
        },
    }
    _json_dump(staging_dir / "manifest.json", manifest)
    (staging_dir / "_SUCCESS").touch()
    staging_dir.replace(output_dir)
    return ProxyResult(metrics=metrics, manifest=manifest, output_dir=output_dir)


def validate_proxy_output(output_dir: Path) -> dict[str, Any]:
    """Validate all immutable M2-A artifacts without re-running the proxy."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "QGR-SID proxy manifest")
    if manifest.get("schema_version") != PROXY_SCHEMA_VERSION:
        raise QgrSidProxyError("QGR-SID proxy schema_version 不受支持")
    if manifest.get("status") != "completed":
        raise QgrSidProxyError("仅支持验证 completed 正式输出")
    if not (output_dir / "_SUCCESS").is_file():
        raise QgrSidProxyError("QGR-SID proxy 缺少 _SUCCESS")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise QgrSidProxyError("QGR-SID proxy manifest 缺少 outputs")
    contracts: dict[str, Any] = {}
    for name in ("metrics", "cases"):
        contract = outputs.get(name)
        if not isinstance(contract, dict):
            raise QgrSidProxyError(f"QGR-SID proxy 缺少 {name} 契约")
        contracts[name] = contract
    arrays = outputs.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise QgrSidProxyError("QGR-SID proxy 缺少 arrays 契约")
    contracts.update({f"arrays.{name}": value for name, value in arrays.items()})

    checked: dict[str, str] = {}
    for name, contract in contracts.items():
        if not isinstance(contract, dict):
            raise QgrSidProxyError(f"输出契约非法：{name}")
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file():
            raise QgrSidProxyError(f"输出不存在：{path}")
        digest = sha256_file(path)
        if digest != contract.get("sha256"):
            raise QgrSidProxyError(f"输出 SHA256 不一致：{name}")
        if int(contract.get("bytes", -1)) != path.stat().st_size:
            raise QgrSidProxyError(f"输出字节数不一致：{name}")
        if name.startswith("arrays."):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != contract.get("shape"):
                raise QgrSidProxyError(f"数组 shape 不一致：{name}")
            if str(array.dtype) != contract.get("dtype"):
                raise QgrSidProxyError(f"数组 dtype 不一致：{name}")
        checked[name] = digest
    return {
        "status": "validated",
        "schema_version": manifest["schema_version"],
        "signature": manifest["signature"],
        "checked_output_count": len(checked),
        "checked_outputs": checked,
    }

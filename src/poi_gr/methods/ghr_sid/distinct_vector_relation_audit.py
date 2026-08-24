"""Relation audit after collapsing exact-vector duplicates in EXP-09 residuals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from poi_gr.methods.ghr_sid.entity_structure import (
    _file_contract,
    _git_state,
    _json_dump,
    _load_json,
    _load_structure_arrays,
    _ratio,
    _signature,
    _stream_target_pois,
    _utc_now,
)
from poi_gr.methods.ghr_sid.fine_relations import (
    RawFineRelations,
    build_fine_relation_matrix,
    extract_raw_fine_relations,
)
from poi_gr.methods.ghr_sid.g6_entity_structure import (
    _load_proxy_arrays,
    _matrix_sha256,
    reconstruct_post_g6_groups,
    validate_g6_entity_structure_output,
)
from poi_gr.methods.ghr_sid.minimum_description import (
    CollisionGrouping,
    MinimumDescriptionCompilation,
    compile_minimum_descriptions,
)
from poi_gr.methods.ghr_sid.residual import normalize_static_text
from poi_gr.methods.ghr_sid.residual_vector_audit import (
    GID_LENGTHS,
    STABLE_FIELDS,
    collision_partition_stats,
    validate_residual_vector_audit_output,
)
from poi_gr.methods.ghr_sid.unified_relations import (
    SEMANTIC_TIERS,
    UNIFIED_RELATION_TYPES,
    UnifiedRelationMatrix,
    build_unified_relation_matrix,
    decode_unified_relation_value,
)
from poi_gr.methods.tiger.identifier import sha256_file


DISTINCT_VECTOR_RELATION_SCHEMA_VERSION = (
    "ghr-sid-distinct-vector-relation-audit-v1"
)
DISTINCT_VECTOR_RELATION_METRICS_SCHEMA_VERSION = (
    "ghr-sid-distinct-vector-relation-audit-metrics-v1"
)
EXPECTED_VECTOR_CLASS_COUNT = 11_754
EXPECTED_GROUP_COUNT = 5_650
EXPECTED_COLLISION_EXCESS = 6_104
EXPECTED_CANDIDATE_MATRIX_SHA256 = (
    "51bcad09ced597ac8699d6085dbcb38591c3cf81e9acdaadbd2a23e609e907d7"
)
DISTANCE_THRESHOLDS_METERS = (10, 25, 50, 100, 200, 500, 1_000)


class DistinctVectorRelationAuditError(ValueError):
    """Raised when a distinct-vector relation audit violates its contract."""


@dataclass(frozen=True)
class VectorClassCollapse:
    """One deterministic representative for each exact-vector class."""

    representative_rows: np.ndarray
    original_group_ids: np.ndarray
    grouping: CollisionGrouping
    class_sizes: np.ndarray


@dataclass(frozen=True)
class DistinctVectorRelationAuditResult:
    """Completed relation-focused audit artifacts."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


def collapse_exact_vector_classes(
    group_ids: Sequence[int], embedding_keys: Sequence[Hashable]
) -> VectorClassCollapse:
    """Collapse exact vectors and keep only groups with multiple vector classes."""

    if len(group_ids) != len(embedding_keys):
        raise DistinctVectorRelationAuditError(
            "group_ids 与 embedding_keys 行数不一致"
        )
    first_by_class: dict[tuple[int, Hashable], int] = {}
    class_counts: Counter[tuple[int, Hashable]] = Counter()
    classes_by_group: Counter[int] = Counter()
    for row, (group_id, embedding_key) in enumerate(
        zip(group_ids, embedding_keys, strict=True)
    ):
        key = (int(group_id), embedding_key)
        if key not in first_by_class:
            first_by_class[key] = row
            classes_by_group[int(group_id)] += 1
        class_counts[key] += 1
    eligible_groups = {
        group_id for group_id, count in classes_by_group.items() if count > 1
    }
    ordered_keys = sorted(
        (key for key in first_by_class if key[0] in eligible_groups),
        key=lambda key: first_by_class[key],
    )
    representative_rows = np.asarray(
        [first_by_class[key] for key in ordered_keys], dtype=np.int64
    )
    original_group_ids = np.asarray(
        [key[0] for key in ordered_keys], dtype=np.int64
    )
    class_sizes = np.asarray(
        [class_counts[key] for key in ordered_keys], dtype=np.int32
    )
    unique_groups = sorted(eligible_groups)
    dense_by_original = {
        original: dense for dense, original in enumerate(unique_groups)
    }
    dense_group_ids = np.asarray(
        [dense_by_original[int(value)] for value in original_group_ids],
        dtype=np.int32,
    )
    group_sizes = np.bincount(
        dense_group_ids, minlength=len(unique_groups)
    ).astype(np.int32)
    member_order = np.argsort(dense_group_ids, kind="stable").astype(np.int64)
    group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(group_sizes, dtype=np.int64)]
    )
    grouping = CollisionGrouping(
        collision_rows=representative_rows,
        group_ids=dense_group_ids,
        group_sizes=group_sizes,
        member_order=member_order,
        group_offsets=group_offsets,
    )
    return VectorClassCollapse(
        representative_rows=representative_rows,
        original_group_ids=original_group_ids,
        grouping=grouping,
        class_sizes=class_sizes,
    )


def _path_keys(
    compilation: MinimumDescriptionCompilation,
) -> list[tuple[tuple[int, int], ...]]:
    return [
        tuple(
            (int(relation_type), int(value))
            for relation_type, value in zip(types, values, strict=True)
            if relation_type >= 0
        )
        for types, values in zip(
            compilation.path_types, compilation.path_values, strict=True
        )
    ]


def _stable_key(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(normalize_static_text(record.get(field, "")) for field in fields)


def _haversine_meters(
    first_lng: float, first_lat: float, second_lng: float, second_lat: float
) -> float:
    radius = 6_371_008.8
    lat1 = math.radians(first_lat)
    lat2 = math.radians(second_lat)
    delta_lat = lat2 - lat1
    delta_lng = math.radians(second_lng - first_lng)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lng / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(value)))


def summarize_pair_distances(
    clusters: Sequence[Sequence[int]],
    longitudes: Sequence[float],
    latitudes: Sequence[float],
) -> dict[str, Any]:
    """Summarize all unordered POI pairs inside fixed collision clusters."""

    distances: list[float] = []
    group_maxima: list[float] = []
    for members in clusters:
        current: list[float] = []
        for offset, first in enumerate(members):
            for second in members[offset + 1 :]:
                distance = _haversine_meters(
                    float(longitudes[first]),
                    float(latitudes[first]),
                    float(longitudes[second]),
                    float(latitudes[second]),
                )
                current.append(distance)
                distances.append(distance)
        if current:
            group_maxima.append(max(current))
    values = np.asarray(distances, dtype=np.float64)
    maxima = np.asarray(group_maxima, dtype=np.float64)

    def percentiles(array: np.ndarray) -> dict[str, float]:
        if not len(array):
            return {key: 0.0 for key in ("p50", "p90", "p95", "p99", "max")}
        return {
            "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(array.max()),
        }

    cumulative = {
        f"le_{threshold}m": int(np.count_nonzero(values <= threshold))
        for threshold in DISTANCE_THRESHOLDS_METERS
    }
    cumulative["gt_1000m"] = int(np.count_nonzero(values > 1_000))
    return {
        "cluster_count": len(group_maxima),
        "pair_count": len(distances),
        "same_coordinate_pair_count": int(np.count_nonzero(values == 0)),
        "same_coordinate_pair_ratio": float(np.mean(values == 0))
        if len(values)
        else 0.0,
        "pair_distance_meters": percentiles(values),
        "cluster_max_span_meters": percentiles(maxima),
        "pair_cumulative_counts": cumulative,
        "pair_cumulative_ratios": {
            key: _ratio(count, len(distances)) for key, count in cumulative.items()
        },
    }


def _collision_clusters(
    group_ids: Sequence[int], keys: Sequence[Hashable]
) -> list[list[int]]:
    members: dict[tuple[int, Hashable], list[int]] = defaultdict(list)
    for row, (group_id, key) in enumerate(zip(group_ids, keys, strict=True)):
        members[(int(group_id), key)].append(row)
    return [rows for rows in members.values() if len(rows) > 1]


def _classify_cluster(
    members: Sequence[int],
    full_relation_keys: Sequence[Hashable],
    display_address_keys: Sequence[Hashable],
    stable_keys: Sequence[Hashable],
) -> str:
    size = len(members)
    relation_unique = len({full_relation_keys[row] for row in members})
    display_address_unique = len(
        {display_address_keys[row] for row in members}
    )
    stable_unique = len({stable_keys[row] for row in members})
    if relation_unique == size:
        return "current_candidates_differ_but_path_not_unique"
    if stable_unique == 1:
        return "all_stable_text_identical_relation_impossible"
    if display_address_unique > relation_unique:
        return "name_address_entity_extraction_gap"
    if stable_unique > relation_unique:
        return "other_stable_field_relation_gap"
    return "current_relation_signature_collision"


def _path_payload(
    row: int,
    compilation: MinimumDescriptionCompilation,
    relation_matrix: UnifiedRelationMatrix,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for relation_index, value in zip(
        compilation.path_types[row], compilation.path_values[row], strict=True
    ):
        if relation_index < 0:
            continue
        relation_type = UNIFIED_RELATION_TYPES[int(relation_index)]
        output.append(
            {
                "type": relation_type,
                "value": int(value),
                "decoded": decode_unified_relation_value(
                    relation_type, int(value), relation_matrix.vocabularies
                ),
            }
        )
    return output


def _relation_signature(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest()


def _rebuild_relation_matrix(
    *,
    poi_dir: Path,
    exp09_dir: Path,
    progress: Callable[[str], None] | None,
) -> tuple[
    CollisionGrouping,
    list[dict[str, Any]],
    UnifiedRelationMatrix,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    exp09_manifest = _load_json(exp09_dir / "manifest.json", "EXP-09 manifest")
    structure_path = Path(
        str(exp09_manifest["inputs"]["structure_manifest"]["path"])
    )
    proxy_path = Path(str(exp09_manifest["inputs"]["proxy_manifest"]["path"]))
    arrays, structure_manifest, _ = _load_structure_arrays(structure_path.parent)
    grouping = reconstruct_post_g6_groups(
        base_sid_keys=arrays["base_sid_keys"],
        gid8_codes=arrays["collision_gid8_codes"],
        resolution_stage=arrays["resolution_stage"],
    )
    proxy_poi_ids, proxy_base, strict_all, proxy_manifest = _load_proxy_arrays(
        proxy_path.parent
    )
    if not np.array_equal(proxy_poi_ids, arrays["collision_poi_ids"]):
        raise DistinctVectorRelationAuditError("proxy 与结构 POI 行序不一致")
    if not np.array_equal(proxy_base, arrays["base_sid_keys"]):
        raise DistinctVectorRelationAuditError("proxy 与结构 base SID 不一致")
    target_poi_ids = np.asarray(arrays["collision_poi_ids"])[
        grouping.collision_rows
    ]
    expected_sources = structure_manifest.get("inputs", {}).get("raw_sources")
    if not isinstance(expected_sources, list):
        raise DistinctVectorRelationAuditError("结构 manifest 缺少 raw_sources")
    if progress is not None:
        progress("重建 EXP-09 的 80 类统一实体关系候选")
    records, _, _ = _stream_target_pois(
        poi_dir=poi_dir,
        target_poi_ids=target_poi_ids,
        expected_sources=expected_sources,
        progress=progress,
    )
    raw_relations: list[RawFineRelations] = [
        extract_raw_fine_relations(record) for record in records
    ]
    fine_matrix = build_fine_relation_matrix(
        raw_relations, entity_min_support=2, entity_vocab_size=32_767
    )
    relation_matrix = build_unified_relation_matrix(
        strict_relations=np.asarray(
            strict_all[grouping.collision_rows], dtype=np.int64
        ),
        fine_relations=fine_matrix,
    )
    digest = _matrix_sha256(relation_matrix.values)
    frozen_digest = exp09_manifest["inputs"]["candidate_matrix"]["sha256"]
    if digest != frozen_digest or digest != EXPECTED_CANDIDATE_MATRIX_SHA256:
        raise DistinctVectorRelationAuditError("重建关系候选矩阵 SHA256 不一致")
    return (
        grouping,
        records,
        relation_matrix,
        target_poi_ids,
        exp09_manifest,
        proxy_manifest,
    )


def audit_distinct_vector_relations(
    *,
    project_root: Path,
    poi_dir: Path,
    exp09_dir: Path,
    exp10_dir: Path,
    output_dir: Path,
    experiment_id: str,
    cases_per_type: int = 8,
    pois_per_case: int = 12,
    progress: Callable[[str], None] | None = None,
) -> DistinctVectorRelationAuditResult:
    """Audit relation-solvable collisions after exact-vector deduplication."""

    started_at = _utc_now()
    started = time.monotonic()
    project_root = project_root.resolve()
    poi_dir = poi_dir.resolve()
    exp09_dir = exp09_dir.resolve()
    exp10_dir = exp10_dir.resolve()
    output_dir = output_dir.resolve()
    for path, name in (
        (project_root, "项目根目录"),
        (poi_dir, "POI 目录"),
        (exp09_dir, "EXP-09 目录"),
        (exp10_dir, "EXP-10 目录"),
    ):
        if not path.is_dir():
            raise DistinctVectorRelationAuditError(f"{name}不存在：{path}")
    if output_dir.exists():
        raise DistinctVectorRelationAuditError(f"输出目录已存在：{output_dir}")
    validate_g6_entity_structure_output(exp09_dir)
    validate_residual_vector_audit_output(exp10_dir)

    residual_path = exp10_dir / "residual_vector_pois.parquet"
    residual_records = pq.read_table(residual_path).to_pylist()
    residual_group_ids = np.asarray(
        [record["g6_group_id"] for record in residual_records], dtype=np.int64
    )
    embedding_keys = [record["embedding_sha256"] for record in residual_records]
    collapse = collapse_exact_vector_classes(residual_group_ids, embedding_keys)
    frozen_counts = {
        "vector_class_count": len(collapse.representative_rows),
        "group_count": len(collapse.grouping.group_sizes),
        "collision_excess": int(np.sum(collapse.grouping.group_sizes - 1)),
        "max_group_size": int(collapse.grouping.group_sizes.max(initial=0)),
    }
    expected_counts = {
        "vector_class_count": EXPECTED_VECTOR_CLASS_COUNT,
        "group_count": EXPECTED_GROUP_COUNT,
        "collision_excess": EXPECTED_COLLISION_EXCESS,
    }
    if {key: frozen_counts[key] for key in expected_counts} != expected_counts:
        raise DistinctVectorRelationAuditError(
            f"向量类冻结计数变化：{frozen_counts}"
        )

    (
        post_g6_grouping,
        _,
        relation_matrix,
        target_poi_ids,
        exp09_manifest,
        proxy_manifest,
    ) = _rebuild_relation_matrix(
        poi_dir=poi_dir, exp09_dir=exp09_dir, progress=progress
    )
    exp09_resolved = np.load(
        exp09_dir / "relation_resolved.npy", mmap_mode="r", allow_pickle=False
    ).astype(bool)
    unresolved_positions = np.flatnonzero(~exp09_resolved)
    if len(unresolved_positions) != len(residual_records):
        raise DistinctVectorRelationAuditError("EXP-09/10 残留行数不一致")
    residual_poi_ids = np.asarray(
        [int(record["poi_id"]) for record in residual_records], dtype=np.int64
    )
    if not np.array_equal(target_poi_ids[unresolved_positions], residual_poi_ids):
        raise DistinctVectorRelationAuditError("EXP-09/10 残留 POI 行序不一致")
    if not np.array_equal(
        post_g6_grouping.group_ids[unresolved_positions], residual_group_ids
    ):
        raise DistinctVectorRelationAuditError("EXP-09/10 残留分组不一致")

    representative_records = [
        residual_records[int(row)] for row in collapse.representative_rows
    ]
    candidate_rows = relation_matrix.values[unresolved_positions][
        collapse.representative_rows
    ]
    if progress is not None:
        progress(
            "完全相同向量折叠为 11,754 个代表，重新编译桶内最短关系描述"
        )
    compilation = compile_minimum_descriptions(
        grouping=collapse.grouping,
        relation_values=candidate_rows,
        semantic_tiers=SEMANTIC_TIERS,
        max_pairs=3,
        max_tokens=6,
        progress=progress,
    )
    path_keys = _path_keys(compilation)
    full_relation_keys = [tuple(row.tolist()) for row in candidate_rows]
    display_keys = [
        _stable_key(record, ("displayname",))
        for record in representative_records
    ]
    display_address_keys = [
        _stable_key(record, ("displayname", "address"))
        for record in representative_records
    ]
    stable_keys = [
        _stable_key(record, STABLE_FIELDS) for record in representative_records
    ]
    gid_keys = {
        length: [str(record["gid12"])[:length] for record in representative_records]
        for length in GID_LENGTHS
    }
    coordinate_keys = [
        (float(record["lng"]), float(record["lat"]))
        for record in representative_records
    ]
    group_ids = collapse.grouping.group_ids
    partitions: dict[str, dict[str, int]] = {
        "baseline_distinct_vectors": collision_partition_stats(
            group_ids, [0] * len(group_ids)
        ),
        "compiled_relation_path": collision_partition_stats(group_ids, path_keys),
        "full_80_relation_signature": collision_partition_stats(
            group_ids, full_relation_keys
        ),
        "displayname": collision_partition_stats(group_ids, display_keys),
        "displayname_address": collision_partition_stats(
            group_ids, display_address_keys
        ),
        "all_stable_fields": collision_partition_stats(group_ids, stable_keys),
        "exact_coordinate": collision_partition_stats(group_ids, coordinate_keys),
    }
    for length in GID_LENGTHS:
        partitions[f"gid{length}"] = collision_partition_stats(
            group_ids, gid_keys[length]
        )
    for stats in partitions.values():
        stats["excess_reduction"] = EXPECTED_COLLISION_EXCESS - int(
            stats["collision_excess"]
        )
        stats["excess_reduction_ratio"] = _ratio(
            stats["excess_reduction"], EXPECTED_COLLISION_EXCESS
        )

    baseline_clusters = _collision_clusters(group_ids, [0] * len(group_ids))
    unresolved_clusters = _collision_clusters(group_ids, path_keys)
    stable_irreducible_clusters = _collision_clusters(group_ids, stable_keys)
    longitudes = [float(record["lng"]) for record in representative_records]
    latitudes = [float(record["lat"]) for record in representative_records]
    distance = {
        "baseline_distinct_vector_groups": summarize_pair_distances(
            baseline_clusters, longitudes, latitudes
        ),
        "after_current_relation_compiler": summarize_pair_distances(
            unresolved_clusters, longitudes, latitudes
        ),
        "all_stable_text_identical": summarize_pair_distances(
            stable_irreducible_clusters, longitudes, latitudes
        ),
    }

    cluster_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    category_excess: Counter[str] = Counter()
    category_pois: Counter[str] = Counter()
    for cluster_id, members in enumerate(unresolved_clusters):
        category = _classify_cluster(
            members, full_relation_keys, display_address_keys, stable_keys
        )
        category_counts[category] += 1
        category_excess[category] += len(members) - 1
        category_pois[category] += len(members)
        cluster_distance = summarize_pair_distances(
            [members], longitudes, latitudes
        )
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "g6_group_id": int(collapse.original_group_ids[members[0]]),
                "collision_size": len(members),
                "collision_excess": len(members) - 1,
                "diagnostic_type": category,
                "full_relation_unique_count": len(
                    {full_relation_keys[row] for row in members}
                ),
                "display_address_unique_count": len(
                    {display_address_keys[row] for row in members}
                ),
                "stable_signature_unique_count": len(
                    {stable_keys[row] for row in members}
                ),
                "coordinate_unique_count": len(
                    {coordinate_keys[row] for row in members}
                ),
                "max_span_meters": cluster_distance["pair_distance_meters"]["max"],
                "member_rows": list(members),
            }
        )
    diagnostic_distribution = {
        category: {
            "collision_cluster_count": category_counts[category],
            "collision_poi_count": category_pois[category],
            "collision_excess": category_excess[category],
            "baseline_excess_ratio": _ratio(
                category_excess[category], EXPECTED_COLLISION_EXCESS
            ),
        }
        for category in sorted(category_counts)
    }

    selected_cases: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cluster_rows:
        by_category[str(row["diagnostic_type"])].append(row)
    for category in sorted(by_category):
        chosen = sorted(
            by_category[category],
            key=lambda row: (
                -int(row["collision_size"]),
                -float(row["max_span_meters"]),
                int(row["g6_group_id"]),
            ),
        )[:cases_per_type]
        for cluster in chosen:
            pois = []
            for row in cluster["member_rows"][:pois_per_case]:
                record = representative_records[row]
                pois.append(
                    {
                        "poi_id": record["poi_id"],
                        "displayname": record["displayname"],
                        "address": record["address"],
                        "alias": record["alias"],
                        "category": record["category"],
                        "category_code": record["category_code"],
                        "lng": record["lng"],
                        "lat": record["lat"],
                        "gid12": record["gid12"],
                        "embedding_sha256": record["embedding_sha256"],
                        "exact_vector_class_size": int(collapse.class_sizes[row]),
                        "compiled_path": _path_payload(
                            row, compilation, relation_matrix
                        ),
                        "present_relations": [
                            {
                                "type": UNIFIED_RELATION_TYPES[index],
                                "decoded": decode_unified_relation_value(
                                    UNIFIED_RELATION_TYPES[index],
                                    int(value),
                                    relation_matrix.vocabularies,
                                ),
                            }
                            for index, value in enumerate(candidate_rows[row])
                            if value >= 0
                        ],
                    }
                )
            selected_cases.append(
                {
                    **{key: value for key, value in cluster.items() if key != "member_rows"},
                    "pois": pois,
                }
            )

    representative_rows = []
    unresolved_members = {
        row: cluster["diagnostic_type"]
        for cluster in cluster_rows
        for row in cluster["member_rows"]
    }
    for row, record in enumerate(representative_records):
        representative_rows.append(
            {
                "dense_group_id": int(group_ids[row]),
                "g6_group_id": int(collapse.original_group_ids[row]),
                "poi_id": record["poi_id"],
                "embedding_sha256": record["embedding_sha256"],
                "exact_vector_class_size": int(collapse.class_sizes[row]),
                "displayname": record["displayname"],
                "address": record["address"],
                "alias": record["alias"],
                "category": record["category"],
                "category_code": record["category_code"],
                "lng": record["lng"],
                "lat": record["lat"],
                "gid12": record["gid12"],
                "compiled_relation_path": json.dumps(
                    _path_payload(row, compilation, relation_matrix),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "compiled_path_unique": bool(compilation.resolved[row]),
                "residual_diagnostic_type": unresolved_members.get(row, "resolved"),
                "full_relation_signature_sha256": _relation_signature(
                    candidate_rows[row]
                ),
            }
        )

    compiled_excess = int(partitions["compiled_relation_path"]["collision_excess"])
    stable_excess = int(partitions["all_stable_fields"]["collision_excess"])
    metrics = {
        "schema_version": DISTINCT_VECTOR_RELATION_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "experiment_id": experiment_id,
        "method": "GHR-SID distinct-vector relation solvability audit",
        "scope": "EXP-09 residual after collapsing exact BGE-vector duplicates",
        "query_usage": "none",
        "dedup_contract": {
            "exact_vector_duplicates_ignored_here": True,
            "representative_selection": "first frozen EXP-10 residual row",
            "embedding_digest_is_diagnostic_not_sid_token": True,
            "later_exact_vector_duplicates_use_stable_dedup_id": True,
        },
        "baseline": frozen_counts,
        "partitions": partitions,
        "compiler": compilation.metrics,
        "unresolved_cluster_diagnostics": diagnostic_distribution,
        "distance_diagnostics": distance,
        "decision": {
            "current_relations_resolved_excess": EXPECTED_COLLISION_EXCESS
            - compiled_excess,
            "current_relations_resolved_ratio": _ratio(
                EXPECTED_COLLISION_EXCESS - compiled_excess,
                EXPECTED_COLLISION_EXCESS,
            ),
            "remaining_excess_after_current_relations": compiled_excess,
            "stable_text_theoretical_remaining_excess": stable_excess,
            "potential_excess_for_finer_relation_extraction": max(
                0, compiled_excess - stable_excess
            ),
            "relation_impossible_excess_at_stable_text_boundary": stable_excess,
            "geography_is_diagnostic_only_after_g6": True,
            "sft_started": False,
        },
        "interpretation_contract": {
            "full_80_relation_signature_is_partition_lower_bound_not_emittable_path": True,
            "all_stable_fields_is_relation_information_upper_bound": True,
            "distance_uses_all_unordered_pairs_with_haversine": True,
            "query_or_order_consumed": False,
            "poi_id_used_as_relation_feature": False,
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
    }

    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    representative_path = staging_dir / "vector_class_representatives.parquet"
    cluster_path = staging_dir / "unresolved_relation_clusters.parquet"
    cases_path = staging_dir / "cases.jsonl"
    metrics_path = staging_dir / "metrics.json"
    pq.write_table(
        pa.Table.from_pylist(representative_rows), representative_path, compression="zstd"
    )
    cluster_output = [
        {key: value for key, value in row.items() if key != "member_rows"}
        for row in cluster_rows
    ]
    pq.write_table(pa.Table.from_pylist(cluster_output), cluster_path, compression="zstd")
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in selected_cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    _json_dump(metrics_path, metrics)
    outputs = {
        "vector_class_representatives": _file_contract(
            representative_path, rows=len(representative_rows)
        ),
        "unresolved_relation_clusters": _file_contract(
            cluster_path, rows=len(cluster_output)
        ),
        "cases": _file_contract(cases_path, rows=len(selected_cases)),
        "metrics": _file_contract(metrics_path),
    }
    manifest = {
        "schema_version": DISTINCT_VECTOR_RELATION_SCHEMA_VERSION,
        "status": "completed",
        "experiment_id": experiment_id,
        "inputs": {
            "exp09_manifest": {
                "path": str((exp09_dir / "manifest.json").resolve()),
                "sha256": sha256_file(exp09_dir / "manifest.json"),
            },
            "exp10_manifest": {
                "path": str((exp10_dir / "manifest.json").resolve()),
                "sha256": sha256_file(exp10_dir / "manifest.json"),
            },
            "structure_manifest": exp09_manifest["inputs"]["structure_manifest"],
            "proxy_manifest": {
                **exp09_manifest["inputs"]["proxy_manifest"],
                "schema_version": proxy_manifest.get("schema_version"),
            },
            "candidate_matrix": {
                "shape": list(relation_matrix.values.shape),
                "dtype": str(relation_matrix.values.dtype),
                "sha256": EXPECTED_CANDIDATE_MATRIX_SHA256,
                "saved": False,
            },
        },
        "configuration": {
            "query_or_order_used": False,
            "exact_vector_classes_collapsed": True,
            "max_relation_pairs": 3,
            "max_relation_tokens": 6,
            "relation_types": list(UNIFIED_RELATION_TYPES),
        },
        "outputs": outputs,
        "validation": {
            "exp09_and_exp10_row_order_match": True,
            "candidate_matrix_matches_exp09": True,
            "frozen_vector_class_counts_match": True,
            "query_or_order_consumed": False,
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "git": _git_state(project_root),
        "started_at": started_at,
        "finished_at": metrics["finished_at"],
    }
    manifest["signature"] = _signature(
        {key: value for key, value in manifest.items() if key != "signature"}
    )
    _json_dump(staging_dir / "manifest.json", manifest)
    (staging_dir / "_SUCCESS").touch()
    staging_dir.replace(output_dir)
    validate_distinct_vector_relation_audit_output(output_dir)
    return DistinctVectorRelationAuditResult(metrics, manifest, output_dir)


def validate_distinct_vector_relation_audit_output(
    output_dir: Path,
) -> dict[str, Any]:
    """Validate every managed distinct-vector relation audit output."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "关系审计 manifest")
    if manifest.get("schema_version") != DISTINCT_VECTOR_RELATION_SCHEMA_VERSION:
        raise DistinctVectorRelationAuditError("关系审计 schema_version 不一致")
    if not (output_dir / "_SUCCESS").is_file():
        raise DistinctVectorRelationAuditError("关系审计缺少 _SUCCESS")
    expected_signature = _signature(
        {key: value for key, value in manifest.items() if key != "signature"}
    )
    if expected_signature != manifest.get("signature"):
        raise DistinctVectorRelationAuditError("关系审计 manifest signature 不一致")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise DistinctVectorRelationAuditError("关系审计 manifest 缺少 outputs")
    for name, contract in outputs.items():
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file() or path.stat().st_size != contract.get("bytes"):
            raise DistinctVectorRelationAuditError(f"输出缺失或大小不一致：{name}")
        if sha256_file(path) != contract.get("sha256"):
            raise DistinctVectorRelationAuditError(f"输出 SHA256 不一致：{name}")
        if path.suffix == ".parquet" and pq.read_metadata(path).num_rows != contract.get(
            "rows"
        ):
            raise DistinctVectorRelationAuditError(f"Parquet 行数不一致：{name}")
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                if sum(1 for _ in handle) != contract.get("rows"):
                    raise DistinctVectorRelationAuditError(f"JSONL 行数不一致：{name}")
    metrics = _load_json(output_dir / "metrics.json", "关系审计 metrics")
    if metrics.get("schema_version") != DISTINCT_VECTOR_RELATION_METRICS_SCHEMA_VERSION:
        raise DistinctVectorRelationAuditError("关系审计 metrics schema 不一致")
    return {
        "status": "validated",
        "experiment_id": manifest.get("experiment_id"),
        "validated_output_count": len(outputs),
        "signature": manifest.get("signature"),
    }

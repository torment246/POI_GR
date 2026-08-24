"""Formal TIGER3 -> G6 -> strong relation tree -> leaf Dedup experiment."""

from __future__ import annotations

import json
import os
import platform
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from poi_gr.methods.ghr_sid.entity_structure import (
    GhrSidEntityStructureError,
    _file_contract,
    _git_state,
    _json_dump,
    _load_json,
    _load_structure_arrays,
    _ratio,
    _save_array,
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
    EXPECTED_POST_G6_EXCESS,
    EXPECTED_POST_G6_GROUP_COUNT,
    EXPECTED_POST_G6_MAX_GROUP_SIZE,
    EXPECTED_POST_G6_POI_COUNT,
    _load_proxy_arrays,
    _matrix_sha256,
    reconstruct_post_g6_groups,
)
from poi_gr.methods.ghr_sid.minimum_description import relation_token_cost
from poi_gr.methods.ghr_sid.relation_tree import (
    DEDUP_RELATION_TYPE,
    DEDUP_RELATION_TYPE_ID,
    RELATION_TREE_SCHEMA_VERSION,
    TREE_RELATION_TYPES,
    RelationTreeCompilation,
    build_relation_trie,
    build_tree_relation_values,
    compile_relation_trees,
    decoded_path,
)
from poi_gr.methods.ghr_sid.unified_relations import (
    UNIFIED_RELATION_SCHEMA_VERSION,
    UNIFIED_RELATION_TYPES,
    build_unified_relation_matrix,
)
from poi_gr.methods.qgr_sid.relations import (
    RELATION_TYPES,
    extract_numeric_relations,
)
from poi_gr.methods.tiger.identifier import sha256_file
from poi_gr.pid.geohash import tokens_to_geohash


G6_RELATION_TREE_SCHEMA_VERSION = "ghr-sid-g6-relation-tree-experiment-v1"
G6_RELATION_TREE_METRICS_SCHEMA_VERSION = (
    "ghr-sid-g6-relation-tree-metrics-v1"
)
EXPECTED_UNIFIED_MATRIX_SHA256 = (
    "51bcad09ced597ac8699d6085dbcb38591c3cf81e9acdaadbd2a23e609e907d7"
)
FOCUS_GROUP_ID = 33_182
FOCUS_GROUP_NAME = "冠城大通百旺府"


@dataclass(frozen=True)
class G6RelationTreeResult:
    """Completed Beijing relation-tree artifacts."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


def _decode_base_sid_key(value: int) -> list[int]:
    mask = (1 << 21) - 1
    return [value >> 42, (value >> 21) & mask, value & mask]


def _distribution(values: np.ndarray) -> dict[str, int]:
    return {
        str(int(value)): int(count)
        for value, count in sorted(Counter(values.tolist()).items())
    }


def build_display_address_strict_relations(
    records: Sequence[Mapping[str, Any]],
    *,
    value_max: int = np.iinfo(np.int16).max,
) -> np.ndarray:
    """Rebuild strict numeric roles from display name and address, never alias."""

    relation_index = {
        relation_type: index
        for index, relation_type in enumerate(RELATION_TYPES)
    }
    values = np.full(
        (len(records), len(RELATION_TYPES)), -1, dtype=np.int64
    )
    for row, record in enumerate(records):
        static_record = dict(record)
        static_record["alias"] = ""
        extraction = extract_numeric_relations(static_record)
        conflicts = {
            conflict.relation_type for conflict in extraction.conflicts
        }
        for relation in extraction.relations:
            if relation.relation_type in conflicts:
                continue
            if 0 <= relation.numeric_value <= value_max:
                values[row, relation_index[relation.relation_type]] = (
                    relation.numeric_value
                )
    return values


def _path_key(
    row: int, compilation: RelationTreeCompilation
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            int(compilation.path_types[row, depth]),
            int(compilation.path_values[row, depth]),
        )
        for depth in range(int(compilation.path_lengths[row]))
    )


def _semantic_leaf_sizes(
    group_ids: np.ndarray, compilation: RelationTreeCompilation
) -> tuple[dict[int, int], dict[int, int]]:
    leaves: Counter[tuple[int, tuple[tuple[int, int], ...]]] = Counter(
        (int(group_ids[row]), _path_key(row, compilation))
        for row in range(len(group_ids))
    )
    max_leaf: dict[int, int] = defaultdict(int)
    excess: dict[int, int] = defaultdict(int)
    for (group_id, _), count in leaves.items():
        max_leaf[group_id] = max(max_leaf[group_id], count)
        excess[group_id] += max(0, count - 1)
    return dict(max_leaf), dict(excess)


def _readable_sid(
    *,
    tiger3: Sequence[int],
    geo_branch: Sequence[int],
    path: Sequence[Mapping[str, Any]],
    dedup_code: int,
) -> str:
    pieces = [
        "T(" + ",".join(str(int(value)) for value in tiger3) + ")"
    ]
    if geo_branch:
        pieces.append(
            "G(" + ",".join(str(int(value)) for value in geo_branch) + ")"
        )
    pieces.extend(
        f"R{int(item['type_id'])}({int(item['value'])})" for item in path
    )
    if dedup_code >= 0:
        pieces.append(f"D({dedup_code})")
    pieces.append("EOS")
    return "/".join(pieces)


def _select_case_groups(
    *,
    grouping_group_sizes: np.ndarray,
    group_ids: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    compilation: RelationTreeCompilation,
    max_cases: int,
) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    seen: set[int] = set()

    def add(kind: str, group_id: int) -> None:
        if 0 <= group_id < len(grouping_group_sizes) and group_id not in seen:
            seen.add(group_id)
            selected.append((kind, group_id))

    add("focus_crown_city", FOCUS_GROUP_ID)
    for row, record in enumerate(records):
        if "C12B" in str(record.get("displayname", "")):
            add("green_c12b", int(group_ids[row]))
            break
    for keyword, kind in (
        ("绿茵花园别墅西区", "green_garden"),
        ("合景香悦四季33号院", "compound_zone_building"),
    ):
        keyword_groups: Counter[int] = Counter()
        for row, record in enumerate(records):
            if keyword in str(record.get("displayname", "")):
                keyword_groups[int(group_ids[row])] += 1
        if keyword_groups:
            group_id = min(
                keyword_groups,
                key=lambda item: (
                    -keyword_groups[item],
                    -int(grouping_group_sizes[item]),
                    item,
                ),
            )
            add(kind, group_id)

    for group_id in np.argsort(-grouping_group_sizes, kind="stable")[:8]:
        add("large_input_bucket", int(group_id))
        if len(selected) >= max_cases:
            return selected[:max_cases]

    max_leaf, excess = _semantic_leaf_sizes(group_ids, compilation)
    for group_id, _ in sorted(
        excess.items(), key=lambda item: (-item[1], -max_leaf[item[0]], item[0])
    ):
        add("large_dedup_residual", group_id)
        if len(selected) >= max_cases:
            return selected[:max_cases]

    max_lengths: dict[int, int] = defaultdict(int)
    for row, group_id in enumerate(group_ids):
        max_lengths[int(group_id)] = max(
            max_lengths[int(group_id)], int(compilation.path_lengths[row])
        )
    for group_id, _ in sorted(
        max_lengths.items(), key=lambda item: (-item[1], item[0])
    ):
        add("deep_relation_path", group_id)
        if len(selected) >= max_cases:
            break
    return selected[:max_cases]


def _case_payloads(
    *,
    grouping: Any,
    arrays: Mapping[str, np.ndarray],
    records: Sequence[Mapping[str, Any]],
    compilation: RelationTreeCompilation,
    vocabularies: Mapping[str, Sequence[str]],
    max_cases: int,
    max_pois_per_case: int,
) -> list[dict[str, Any]]:
    selected = _select_case_groups(
        grouping_group_sizes=grouping.group_sizes,
        group_ids=grouping.group_ids,
        records=records,
        compilation=compilation,
        max_cases=max_cases,
    )
    cases: list[dict[str, Any]] = []
    for kind, group_id in selected:
        start = int(grouping.group_offsets[group_id])
        end = int(grouping.group_offsets[group_id + 1])
        members = grouping.member_order[start:end]
        first_collision_row = int(grouping.collision_rows[int(members[0])])
        base_key = int(arrays["base_sid_keys"][first_collision_row])
        tiger3 = _decode_base_sid_key(base_key)
        gid6 = np.asarray(
            arrays["collision_gid8_codes"][first_collision_row, :6]
        ).astype(int).tolist()
        geo_length = int(arrays["coarse_geo_lengths"][first_collision_row])
        geo_branch = np.asarray(
            arrays["coarse_geo_codes"][first_collision_row, :geo_length]
        ).astype(int).tolist()
        for row in members:
            collision_row = int(grouping.collision_rows[int(row)])
            if not np.array_equal(
                arrays["collision_gid8_codes"][collision_row, :6], gid6
            ):
                raise GhrSidEntityStructureError("案例桶 GID6 不一致")
            row_geo_length = int(arrays["coarse_geo_lengths"][collision_row])
            row_geo_branch = np.asarray(
                arrays["coarse_geo_codes"][collision_row, :row_geo_length]
            ).astype(int).tolist()
            if row_geo_branch != geo_branch:
                raise GhrSidEntityStructureError("案例桶 G6 分支路径不一致")

        poi_payloads: list[dict[str, Any]] = []
        for raw_row in members[:max_pois_per_case]:
            row = int(raw_row)
            record = records[row]
            path = decoded_path(row, compilation, vocabularies)
            dedup_code = int(compilation.dedup_codes[row])
            numeric_relations = [
                [int(item["type_id"]), int(item["value"])] for item in path
            ]
            poi_payloads.append(
                {
                    "poi_id": str(record.get("poi_id", "")),
                    "displayname": record.get("displayname", ""),
                    "address": record.get("address", ""),
                    "alias": record.get("alias", ""),
                    "category_code": record.get("category_code", ""),
                    "longitude": record.get("lng"),
                    "latitude": record.get("lat"),
                    "relations": path,
                    "dedup_code": None if dedup_code < 0 else dedup_code,
                    "sid_numeric": {
                        "tiger3": tiger3,
                        "geo_branch": geo_branch,
                        "relations": numeric_relations,
                        "dedup": None if dedup_code < 0 else dedup_code,
                    },
                    "sid": _readable_sid(
                        tiger3=tiger3,
                        geo_branch=geo_branch,
                        path=path,
                        dedup_code=dedup_code,
                    ),
                }
            )
        semantic_excess = sum(
            1 for row in members if compilation.dedup_codes[int(row)] > 0
        )
        cases.append(
            {
                "kind": kind,
                "group_id": group_id,
                "group_size": int(grouping.group_sizes[group_id]),
                "semantic_excess_before_dedup": semantic_excess,
                "prefix": {
                    "base_sid_key": base_key,
                    "tiger3": tiger3,
                    "gid6_codes": gid6,
                    "gid6": tokens_to_geohash(gid6),
                    "emitted_geo_branch": geo_branch,
                },
                "relation_tree": build_relation_trie(
                    members, compilation, vocabularies
                ),
                "pois_truncated": len(members) > max_pois_per_case,
                "pois": poi_payloads,
            }
        )
    return cases


def _focus_gate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    case = next(
        (item for item in cases if item.get("kind") == "focus_crown_city"),
        None,
    )
    if not isinstance(case, Mapping):
        return {"passed": False, "reason": "focus case missing"}
    pois = case.get("pois", [])
    target = next(
        (
            poi
            for poi in pois
            if poi.get("displayname") == "冠城大通百旺府4区-20号楼"
        ),
        None,
    )
    if not isinstance(target, Mapping):
        return {"passed": False, "reason": "4区-20号楼 missing"}
    relations = target.get("relations", [])
    decoded = {
        str(item.get("type")): item.get("decoded")
        for item in relations
        if isinstance(item, Mapping)
    }
    passed = decoded.get("R_ZONE_NUM") == 4 and decoded.get("R_BUILDING") == 20
    return {
        "passed": passed,
        "group_id": int(case["group_id"]),
        "group_size": int(case["group_size"]),
        "poi_id": target.get("poi_id"),
        "sid": target.get("sid"),
        "relations": relations,
    }


def _c12b_gate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    case = next(
        (item for item in cases if item.get("kind") == "green_c12b"), None
    )
    if not isinstance(case, Mapping):
        return {"passed": False, "reason": "C12B case missing"}
    target = next(
        (
            poi
            for poi in case.get("pois", [])
            if "C12B" in str(poi.get("displayname", ""))
        ),
        None,
    )
    if not isinstance(target, Mapping):
        return {"passed": False, "reason": "C12B POI missing"}
    relations = target.get("relations", [])
    building = [
        item
        for item in relations
        if isinstance(item, Mapping) and item.get("type") == "R_BUILDING"
    ]
    passed = len(building) == 1 and building[0].get("decoded") == "C12B"
    return {
        "passed": passed,
        "group_id": int(case["group_id"]),
        "group_size": int(case["group_size"]),
        "poi_id": target.get("poi_id"),
        "sid": target.get("sid"),
        "relations": relations,
    }


def evaluate_g6_relation_tree(
    *,
    project_root: Path,
    poi_dir: Path,
    structure_dir: Path,
    proxy_dir: Path,
    output_dir: Path,
    experiment_id: str,
    entity_min_support: int = 2,
    entity_vocab_size: int = 32_767,
    max_cases: int = 12,
    max_pois_per_case: int = 300,
    enforce_frozen_counts: bool = True,
    progress: Callable[[str], None] | None = None,
) -> G6RelationTreeResult:
    """Run the full Beijing relation-tree structure experiment."""

    started_at = _utc_now()
    started = time.monotonic()
    project_root = project_root.resolve()
    poi_dir = poi_dir.resolve()
    structure_dir = structure_dir.resolve()
    proxy_dir = proxy_dir.resolve()
    output_dir = output_dir.resolve()
    if not re.fullmatch(r"EXP-\d{8}-\d{2}", experiment_id):
        raise GhrSidEntityStructureError("experiment_id 必须形如 EXP-YYYYMMDD-NN")
    if output_dir.exists():
        raise GhrSidEntityStructureError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    for path, name in (
        (project_root, "项目根目录"),
        (poi_dir, "POI 目录"),
        (structure_dir, "G6 冻结结构目录"),
        (proxy_dir, "严格关系代理目录"),
    ):
        if not path.is_dir():
            raise GhrSidEntityStructureError(f"{name}不存在：{path}")

    arrays, structure_manifest, structure_metrics = _load_structure_arrays(
        structure_dir
    )
    grouping = reconstruct_post_g6_groups(
        base_sid_keys=arrays["base_sid_keys"],
        gid8_codes=arrays["collision_gid8_codes"],
        resolution_stage=arrays["resolution_stage"],
    )
    frozen_counts = {
        "poi_count": len(grouping.collision_rows),
        "group_count": len(grouping.group_sizes),
        "collision_excess": int(np.sum(grouping.group_sizes - 1)),
        "max_group_size": int(grouping.group_sizes.max(initial=0)),
    }
    expected_counts = {
        "poi_count": EXPECTED_POST_G6_POI_COUNT,
        "group_count": EXPECTED_POST_G6_GROUP_COUNT,
        "collision_excess": EXPECTED_POST_G6_EXCESS,
        "max_group_size": EXPECTED_POST_G6_MAX_GROUP_SIZE,
    }
    if enforce_frozen_counts and frozen_counts != expected_counts:
        raise GhrSidEntityStructureError(
            f"post-G6 冻结计数变化：实际 {frozen_counts}，期望 {expected_counts}"
        )

    proxy_poi_ids, proxy_base, strict_all, proxy_manifest = _load_proxy_arrays(
        proxy_dir
    )
    if not np.array_equal(proxy_poi_ids, arrays["collision_poi_ids"]):
        raise GhrSidEntityStructureError("QGR strict relation 与 G6 POI 行序不一致")
    if not np.array_equal(proxy_base, arrays["base_sid_keys"]):
        raise GhrSidEntityStructureError("QGR strict relation 与 G6 base SID 不一致")
    target_poi_ids = np.asarray(arrays["collision_poi_ids"])[
        grouping.collision_rows
    ]
    expected_sources = structure_manifest.get("inputs", {}).get("raw_sources")
    if not isinstance(expected_sources, list):
        raise GhrSidEntityStructureError("G6 manifest 缺少 raw_sources")
    records, raw_sources, full_poi_count = _stream_target_pois(
        poi_dir=poi_dir,
        target_poi_ids=target_poi_ids,
        expected_sources=expected_sources,
        progress=progress,
    )

    if progress is not None:
        progress("抽取强实体角色并建立桶共享关系树，不读取 Query 或订单")
    raw_relations: list[RawFineRelations] = [
        extract_raw_fine_relations(record) for record in records
    ]
    fine_matrix = build_fine_relation_matrix(
        raw_relations,
        entity_min_support=entity_min_support,
        entity_vocab_size=entity_vocab_size,
    )
    frozen_strict = np.asarray(
        strict_all[grouping.collision_rows], dtype=np.int64
    )
    frozen_relation_matrix = build_unified_relation_matrix(
        strict_relations=frozen_strict, fine_relations=fine_matrix
    )
    if progress is not None:
        progress("校验 EX9 统一候选矩阵指纹")
    unified_matrix_sha256 = _matrix_sha256(frozen_relation_matrix.values)
    frozen_matrix_shape = list(frozen_relation_matrix.values.shape)
    frozen_matrix_dtype = str(frozen_relation_matrix.values.dtype)
    if enforce_frozen_counts and unified_matrix_sha256 != EXPECTED_UNIFIED_MATRIX_SHA256:
        raise GhrSidEntityStructureError(
            "统一候选矩阵已偏离冻结 EX9："
            f"{unified_matrix_sha256} != {EXPECTED_UNIFIED_MATRIX_SHA256}"
        )
    del frozen_relation_matrix, frozen_strict
    if progress is not None:
        progress("从名称与地址重建高置信关系，排除 alias-only 路径")
    display_address_strict = build_display_address_strict_relations(records)
    relation_matrix = build_unified_relation_matrix(
        strict_relations=display_address_strict, fine_relations=fine_matrix
    )
    display_address_matrix_sha256 = _matrix_sha256(relation_matrix.values)
    tree_values = build_tree_relation_values(
        relation_matrix, raw_relations=raw_relations
    )
    tree_matrix_sha256 = _matrix_sha256(tree_values)
    compilation = compile_relation_trees(
        grouping=grouping,
        relation_values=tree_values,
        poi_ids=target_poi_ids,
        progress=progress,
    )

    relation_token_lengths = np.zeros(len(compilation.path_lengths), dtype=np.int64)
    for row in range(len(relation_token_lengths)):
        relation_token_lengths[row] = sum(
            relation_token_cost(int(value))
            for value in compilation.path_values[
                row, : int(compilation.path_lengths[row])
            ]
        )
    dedup_token_lengths = np.asarray(
        [
            0 if code < 0 else relation_token_cost(int(code))
            for code in compilation.dedup_codes
        ],
        dtype=np.int64,
    )
    all_suffix_tokens = np.asarray(arrays["coarse_geo_lengths"]).astype(
        np.int64, copy=True
    )
    all_suffix_tokens[grouping.collision_rows] += (
        relation_token_lengths + dedup_token_lengths
    )
    all_semantic_atoms = np.asarray(arrays["coarse_geo_lengths"]).astype(
        np.int64, copy=True
    )
    all_semantic_atoms[grouping.collision_rows] += (
        compilation.path_lengths.astype(np.int64)
        + (compilation.dedup_codes >= 0).astype(np.int64)
    )

    original_excess = int(
        structure_metrics["main"]["catalog"]["collision_excess_before"]
    )
    semantic_excess = int(compilation.metrics["semantic_collision_excess"])
    cases = _case_payloads(
        grouping=grouping,
        arrays=arrays,
        records=records,
        compilation=compilation,
        vocabularies=relation_matrix.vocabularies,
        max_cases=max_cases,
        max_pois_per_case=max_pois_per_case,
    )
    focus_gate = _focus_gate(cases)
    c12b_gate = _c12b_gate(cases)
    final_unique = int(compilation.metrics["final_collision_excess"]) == 0
    metrics: dict[str, Any] = {
        "schema_version": G6_RELATION_TREE_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "experiment_id": experiment_id,
        "method": "GHR-SID",
        "scope": "北京市冻结 TIGER 碰撞 POI",
        "query_usage": "none",
        "main_protocol": "TIGER3->G6->strong_relation_tree->leaf_Dedup->EOS",
        "configuration": {
            "relation_tree_schema": RELATION_TREE_SCHEMA_VERSION,
            "relation_types": list(TREE_RELATION_TYPES),
            "relation_type_ids": {
                relation_type: index
                for index, relation_type in enumerate(TREE_RELATION_TYPES)
            },
            "dedup_relation_type": DEDUP_RELATION_TYPE,
            "dedup_relation_type_id": DEDUP_RELATION_TYPE_ID,
            "selection": "bucket-varying strong roles plus present ancestors",
            "missing_relation_serialization": "skip; no fake missing token",
            "dedup_assignment": "only residual semantic leaves; POI ID ascending",
            "excluded_weak_relations": [
                "category",
                "layer",
                "road/full parent/full child lexical values",
                "entity character positions",
                "fine GID after G6",
            ],
            "query_or_order_used": False,
            "relation_source_fields": ["displayname", "address"],
            "alias_used_as_relation_source": False,
            "poi_id_used_before_semantic_exhaustion": False,
            "candidate_matrix_saved": False,
        },
        "post_g6_input": frozen_counts,
        "compiler": compilation.metrics,
        "main": {
            "g6_resolved_poi_count": int(
                len(arrays["base_sid_keys"]) - len(grouping.collision_rows)
            ),
            "tree_semantic_resolved_poi_count": int(
                compilation.metrics["semantic_resolved_poi_count"]
            ),
            "semantic_collision_excess_before_dedup": semantic_excess,
            "semantic_excess_reduction_ratio_before_dedup": _ratio(
                original_excess - semantic_excess, original_excess
            ),
            "dedup_poi_count": int(compilation.metrics["dedup_poi_count"]),
            "dedup_leaf_count": int(compilation.metrics["dedup_leaf_count"]),
            "max_dedup_code": int(compilation.metrics["max_dedup_code"]),
            "final_collision_excess": 0,
            "full_distinct_identifier_count": full_poi_count,
            "full_distinct_identifier_ratio": 1.0,
            "mean_relation_pair_count_on_post_g6": float(
                compilation.path_lengths.mean()
            ),
            "max_relation_pair_count": int(
                compilation.path_lengths.max(initial=0)
            ),
            "mean_relation_token_count_on_post_g6": float(
                relation_token_lengths.mean()
            ),
            "max_relation_token_count": int(
                relation_token_lengths.max(initial=0)
            ),
            "mean_suffix_token_count_on_all_collision_pois": float(
                all_suffix_tokens.mean()
            ),
            "max_suffix_token_count": int(all_suffix_tokens.max(initial=0)),
            "mean_semantic_atom_count_on_all_collision_pois": float(
                all_semantic_atoms.mean()
            ),
            "max_full_target_tokens_including_tiger3_and_eos": int(
                3 + all_suffix_tokens.max(initial=0) + 1
            ),
            "relation_pair_distribution": _distribution(
                compilation.path_lengths
            ),
            "relation_token_distribution": _distribution(
                relation_token_lengths
            ),
            "dedup_token_distribution": _distribution(dedup_token_lengths),
            "relation_emitted_pair_count_by_type": compilation.metrics[
                "emitted_pair_count_by_type"
            ],
        },
        "focus_crown_city_gate": focus_gate,
        "green_c12b_gate": c12b_gate,
        "decision": {
            "structure_compiled": True,
            "all_catalog_identifiers_unique_after_leaf_dedup": final_unique,
            "ready_for_sft": False,
            "next_step": "先人工审计关系树案例和长度，再决定是否生成训练 mapping",
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
    }
    if (
        not focus_gate.get("passed")
        or not c12b_gate.get("passed")
        or not final_unique
    ):
        raise GhrSidEntityStructureError(
            "关系树核心门禁失败："
            f"focus={focus_gate}, c12b={c12b_gate}, unique={final_unique}"
        )

    if progress is not None:
        progress("写入关系树、叶子 Dedup、完整案例与数值 SID 受管产物")
    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging_dir.exists():
        raise GhrSidEntityStructureError(f"暂存目录已存在：{staging_dir}")
    staging_dir.mkdir(parents=True)
    outputs: dict[str, Any] = {"arrays": {}}
    output_arrays = {
        "post_g6_collision_rows": grouping.collision_rows,
        "post_g6_group_ids": grouping.group_ids,
        "post_g6_poi_ids": target_poi_ids,
        "relation_path_types": compilation.path_types,
        "relation_path_values": compilation.path_values,
        "relation_path_lengths": compilation.path_lengths,
        "semantic_resolved": compilation.semantic_resolved.astype(np.uint8),
        "dedup_codes": compilation.dedup_codes,
    }
    for name, array in output_arrays.items():
        outputs["arrays"][name] = _save_array(staging_dir, name, array)

    vocabulary_path = staging_dir / "vocabularies.json"
    _json_dump(
        vocabulary_path,
        {
            "schema_version": UNIFIED_RELATION_SCHEMA_VERSION,
            "tree_relation_schema_version": RELATION_TREE_SCHEMA_VERSION,
            "relation_types": list(TREE_RELATION_TYPES),
            "relation_type_ids": {
                relation_type: index
                for index, relation_type in enumerate(TREE_RELATION_TYPES)
            },
            "dedup": {
                "type": DEDUP_RELATION_TYPE,
                "type_id": DEDUP_RELATION_TYPE_ID,
            },
            "source_unified_relation_types": list(UNIFIED_RELATION_TYPES),
            "vocabularies": {
                name: list(values)
                for name, values in relation_matrix.vocabularies.items()
            },
        },
    )
    outputs["vocabularies"] = _file_contract(vocabulary_path)
    cases_path = staging_dir / "relation_tree_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n"
            )
    outputs["cases"] = _file_contract(cases_path, len(cases))
    metrics_path = staging_dir / "metrics.json"
    _json_dump(metrics_path, metrics)
    outputs["metrics"] = _file_contract(metrics_path)
    success_path = staging_dir / "_SUCCESS"
    success_path.touch()
    outputs["success"] = _file_contract(success_path)

    signature_payload = {
        "experiment_id": experiment_id,
        "configuration": metrics["configuration"],
        "inputs": {
            "structure_manifest_sha256": sha256_file(
                structure_dir / "manifest.json"
            ),
            "proxy_manifest_sha256": sha256_file(proxy_dir / "manifest.json"),
        },
        "unified_candidate_matrix_sha256": unified_matrix_sha256,
        "display_address_matrix_sha256": display_address_matrix_sha256,
        "tree_candidate_matrix_sha256": tree_matrix_sha256,
    }
    manifest: dict[str, Any] = {
        "schema_version": G6_RELATION_TREE_SCHEMA_VERSION,
        "status": "completed",
        "experiment_id": experiment_id,
        "started_at": started_at,
        "finished_at": metrics["finished_at"],
        "signature": _signature(signature_payload),
        "configuration": metrics["configuration"],
        "git": _git_state(project_root),
        "inputs": {
            "structure_manifest": {
                "path": str((structure_dir / "manifest.json").resolve()),
                "sha256": signature_payload["inputs"][
                    "structure_manifest_sha256"
                ],
                "schema_version": structure_manifest.get("schema_version"),
            },
            "proxy_manifest": {
                "path": str((proxy_dir / "manifest.json").resolve()),
                "sha256": signature_payload["inputs"]["proxy_manifest_sha256"],
                "schema_version": proxy_manifest.get("schema_version"),
                "consumed_arrays": [
                    "collision_poi_ids",
                    "base_sid_keys",
                    "strict_relations",
                ],
                "query_or_order_arrays_consumed": False,
            },
            "raw_sources": raw_sources,
            "unified_candidate_matrix": {
                "shape": frozen_matrix_shape,
                "dtype": frozen_matrix_dtype,
                "sha256": unified_matrix_sha256,
                "matches_frozen_exp09": (
                    unified_matrix_sha256 == EXPECTED_UNIFIED_MATRIX_SHA256
                ),
                "saved": False,
            },
            "display_address_unified_matrix": {
                "shape": list(relation_matrix.values.shape),
                "dtype": str(relation_matrix.values.dtype),
                "sha256": display_address_matrix_sha256,
                "source_fields": ["displayname", "address"],
                "alias_excluded": True,
                "saved": False,
            },
            "tree_candidate_matrix": {
                "shape": list(tree_values.shape),
                "dtype": str(tree_values.dtype),
                "sha256": tree_matrix_sha256,
                "saved": False,
            },
        },
        "outputs": outputs,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "validation": {
            "query_inputs_consumed": False,
            "order_inputs_consumed": False,
            "poi_id_used_before_semantic_exhaustion": False,
            "poi_id_used_only_for_leaf_dedup": True,
            "post_g6_counts_match": frozen_counts == expected_counts,
            "proxy_and_g6_row_order_match": True,
            "raw_sources_match_structure_manifest": True,
            "unified_candidates_match_frozen_exp09": (
                unified_matrix_sha256 == EXPECTED_UNIFIED_MATRIX_SHA256
            ),
            "focus_crown_city_zone_building_path": bool(
                focus_gate.get("passed")
            ),
            "green_c12b_is_building_relation": bool(
                c12b_gate.get("passed")
            ),
            "all_final_identifiers_unique": final_unique,
        },
    }
    _json_dump(staging_dir / "manifest.json", manifest)
    staging_dir.rename(output_dir)
    validate_g6_relation_tree_output(output_dir)
    return G6RelationTreeResult(
        metrics=metrics, manifest=manifest, output_dir=output_dir
    )


def validate_g6_relation_tree_output(output_dir: Path) -> dict[str, Any]:
    """Validate all managed relation-tree artifacts."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "关系树 manifest")
    if manifest.get("schema_version") != G6_RELATION_TREE_SCHEMA_VERSION:
        raise GhrSidEntityStructureError("关系树 manifest schema 不受支持")
    if manifest.get("status") != "completed":
        raise GhrSidEntityStructureError("关系树 manifest 尚未 completed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise GhrSidEntityStructureError("关系树 manifest 缺少 outputs")
    arrays = outputs.get("arrays")
    if not isinstance(arrays, dict):
        raise GhrSidEntityStructureError("关系树 manifest 缺少 arrays")
    validated = 0
    for name, contract in arrays.items():
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file() or sha256_file(path) != contract.get("sha256"):
            raise GhrSidEntityStructureError(f"数组缺失或 SHA 不一致：{name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != contract.get("shape") or str(
            array.dtype
        ) != contract.get("dtype"):
            raise GhrSidEntityStructureError(f"数组 shape/dtype 不一致：{name}")
        validated += 1
    for name in ("vocabularies", "cases", "metrics", "success"):
        contract = outputs.get(name)
        if not isinstance(contract, dict):
            raise GhrSidEntityStructureError(f"缺少输出契约：{name}")
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file() or sha256_file(path) != contract.get("sha256"):
            raise GhrSidEntityStructureError(f"输出缺失或 SHA 不一致：{name}")
        expected_rows = contract.get("rows")
        if expected_rows is not None:
            with path.open("r", encoding="utf-8") as handle:
                if sum(1 for _ in handle) != expected_rows:
                    raise GhrSidEntityStructureError(f"输出行数不一致：{name}")
        validated += 1
    metrics = _load_json(output_dir / "metrics.json", "关系树 metrics")
    if metrics.get("schema_version") != G6_RELATION_TREE_METRICS_SCHEMA_VERSION:
        raise GhrSidEntityStructureError("关系树 metrics schema 不受支持")
    if metrics.get("main", {}).get("final_collision_excess") != 0:
        raise GhrSidEntityStructureError("关系树最终 SID 仍存在碰撞")
    return {
        "status": "validated",
        "validated_output_count": validated,
        "experiment_id": manifest.get("experiment_id"),
        "signature": manifest.get("signature"),
        "main_protocol": metrics.get("main_protocol"),
    }

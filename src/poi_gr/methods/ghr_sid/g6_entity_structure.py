"""Formal G6 -> minimum distinguishing entity relation experiment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
from collections import Counter
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
    _static_text_upper_bound,
    _stream_target_pois,
    _utc_now,
)
from poi_gr.methods.ghr_sid.fine_relations import (
    RawFineRelations,
    build_fine_relation_matrix,
    extract_raw_fine_relations,
)
from poi_gr.methods.ghr_sid.minimum_description import (
    CollisionGrouping,
    MinimumDescriptionCompilation,
    compile_minimum_descriptions,
    relation_token_cost,
)
from poi_gr.methods.ghr_sid.structure import STAGE_COARSE_GEO
from poi_gr.methods.ghr_sid.unified_relations import (
    SEMANTIC_TIERS,
    UNIFIED_RELATION_INDEX,
    UNIFIED_RELATION_SCHEMA_VERSION,
    UNIFIED_RELATION_TYPES,
    UnifiedRelationMatrix,
    build_unified_relation_matrix,
    decode_unified_relation_value,
)
from poi_gr.methods.qgr_sid.proxy import validate_proxy_output
from poi_gr.methods.tiger.identifier import sha256_file


G6_ENTITY_SCHEMA_VERSION = "ghr-sid-g6-minimum-entity-v1"
G6_ENTITY_METRICS_SCHEMA_VERSION = "ghr-sid-g6-minimum-entity-metrics-v1"
EXPECTED_POST_G6_POI_COUNT = 595_173
EXPECTED_POST_G6_GROUP_COUNT = 192_330
EXPECTED_POST_G6_EXCESS = 402_843
EXPECTED_POST_G6_MAX_GROUP_SIZE = 277


@dataclass(frozen=True)
class G6EntityStructureResult:
    """Completed G6-to-entity structural experiment artifacts."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


def reconstruct_post_g6_groups(
    *,
    base_sid_keys: np.ndarray,
    gid8_codes: np.ndarray,
    resolution_stage: np.ndarray,
) -> CollisionGrouping:
    """Rebuild all groups still colliding after the frozen G6 branch."""

    base = np.asarray(base_sid_keys)
    gid8 = np.asarray(gid8_codes)
    stage = np.asarray(resolution_stage)
    rows = len(base)
    if base.ndim != 1 or gid8.shape != (rows, 8) or stage.shape != (rows,):
        raise GhrSidEntityStructureError("G6 grouping 输入 shape 不一致")
    collision_rows = np.flatnonzero(stage != STAGE_COARSE_GEO).astype(
        np.int64, copy=False
    )
    gid6 = np.ascontiguousarray(gid8[collision_rows, :6])
    gid6_bytes = gid6.view(np.dtype((np.void, 6))).reshape(-1)
    keys = np.empty(
        len(collision_rows),
        dtype=[("base", "<i8"), ("gid6", gid6_bytes.dtype)],
    )
    keys["base"] = base[collision_rows]
    keys["gid6"] = gid6_bytes
    order = np.argsort(keys, order=("base", "gid6"), kind="stable")
    sorted_keys = keys[order]
    boundaries = np.ones(len(order), dtype=np.bool_)
    boundaries[1:] = (sorted_keys["base"][1:] != sorted_keys["base"][:-1]) | (
        sorted_keys["gid6"][1:] != sorted_keys["gid6"][:-1]
    )
    starts = np.flatnonzero(boundaries)
    ends = np.append(starts[1:], len(order))
    group_sizes = (ends - starts).astype(np.int32, copy=False)
    if np.any(group_sizes < 2):
        raise GhrSidEntityStructureError("post-G6 输入出现单例组")
    group_ids = np.empty(len(collision_rows), dtype=np.int32)
    for group_id, (start, end) in enumerate(zip(starts, ends, strict=True)):
        group_ids[order[start:end]] = group_id
    member_order = np.argsort(group_ids, kind="stable").astype(np.int64)
    group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(group_sizes, dtype=np.int64)]
    )
    return CollisionGrouping(
        collision_rows=collision_rows,
        group_ids=group_ids,
        group_sizes=group_sizes,
        member_order=member_order,
        group_offsets=group_offsets,
    )


def _load_proxy_arrays(
    proxy_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    validate_proxy_output(proxy_dir)
    manifest = _load_json(proxy_dir / "manifest.json", "QGR proxy manifest")
    arrays = manifest.get("outputs", {}).get("arrays")
    if not isinstance(arrays, dict):
        raise GhrSidEntityStructureError("QGR proxy manifest 缺少 arrays")

    def load(name: str) -> np.ndarray:
        contract = arrays.get(name)
        if not isinstance(contract, dict):
            raise GhrSidEntityStructureError(f"QGR proxy 缺少数组：{name}")
        return np.load(
            proxy_dir / str(contract.get("path", "")),
            mmap_mode="r",
            allow_pickle=False,
        )

    return (
        load("collision_poi_ids"),
        load("base_sid_keys"),
        load("strict_relations"),
        manifest,
    )


def _path_token_lengths(compilation: MinimumDescriptionCompilation) -> np.ndarray:
    lengths = np.zeros(len(compilation.path_types), dtype=np.int64)
    for row in range(len(lengths)):
        lengths[row] = sum(
            relation_token_cost(int(value))
            for relation, value in zip(
                compilation.path_types[row],
                compilation.path_values[row],
                strict=True,
            )
            if relation >= 0
        )
    return lengths


def _distribution(values: np.ndarray) -> dict[str, int]:
    return {
        str(int(value)): int(count)
        for value, count in sorted(Counter(values.tolist()).items())
    }


def _path_payload(
    row: int,
    compilation: MinimumDescriptionCompilation,
    vocabularies: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for relation_index, value in zip(
        compilation.path_types[row],
        compilation.path_values[row],
        strict=True,
    ):
        if relation_index < 0:
            continue
        relation_type = UNIFIED_RELATION_TYPES[int(relation_index)]
        payload.append(
            {
                "type": relation_type,
                "value": int(value),
                "decoded": decode_unified_relation_value(
                    relation_type, int(value), vocabularies
                ),
                "tokens": relation_token_cost(int(value)),
            }
        )
    return payload


def _case_payloads(
    *,
    grouping: CollisionGrouping,
    records: Sequence[Mapping[str, Any]],
    compilation: MinimumDescriptionCompilation,
    relation_matrix: UnifiedRelationMatrix,
    examples_per_kind: int,
    pois_per_case: int,
) -> list[dict[str, Any]]:
    if examples_per_kind <= 0:
        return []
    selected: list[tuple[str, int]] = []
    seen_groups: set[int] = set()

    def add(kind: str, row: int) -> None:
        group_id = int(grouping.group_ids[row])
        if group_id in seen_groups:
            return
        seen_groups.add(group_id)
        selected.append((kind, group_id))

    for row, record in enumerate(records):
        displayname = str(record.get("displayname", ""))
        if "绿茵花园别墅西区" in displayname and "C12B" in displayname:
            add("green_garden", row)
            break

    building = UNIFIED_RELATION_INDEX["R_BUILDING"]
    building_code_rows = np.flatnonzero(
        (relation_matrix.values[:, building] >= 0)
        & (relation_matrix.values[:, building] % 2 == 1)
        & np.any(compilation.path_types == building, axis=1)
    )
    for row in building_code_rows[:examples_per_kind]:
        add("building_code", int(row))

    pair_lengths = np.sum(compilation.path_types >= 0, axis=1)
    for row in np.argsort(-pair_lengths, kind="stable")[: examples_per_kind * 4]:
        if pair_lengths[row] <= 0:
            break
        add("long_path", int(row))
        if sum(kind == "long_path" for kind, _ in selected) >= examples_per_kind:
            break

    unresolved_group_sizes: list[tuple[int, int]] = []
    for group_id in range(len(grouping.group_sizes)):
        start = int(grouping.group_offsets[group_id])
        end = int(grouping.group_offsets[group_id + 1])
        members = grouping.member_order[start:end]
        if np.any(~compilation.resolved[members]):
            unresolved_group_sizes.append((len(members), group_id))
    for _, group_id in sorted(unresolved_group_sizes, reverse=True)[
        :examples_per_kind
    ]:
        if group_id not in seen_groups:
            seen_groups.add(group_id)
            selected.append(("unresolved", group_id))

    cases: list[dict[str, Any]] = []
    for kind, group_id in selected:
        start = int(grouping.group_offsets[group_id])
        end = int(grouping.group_offsets[group_id + 1])
        members = grouping.member_order[start:end]
        poi_rows = []
        for row in members[:pois_per_case]:
            record = records[int(row)]
            poi_rows.append(
                {
                    "poi_id": record["poi_id"],
                    "displayname": record["displayname"],
                    "address": record["address"],
                    "category_code": record["category_code"],
                    "resolved": bool(compilation.resolved[row]),
                    "path": _path_payload(
                        int(row), compilation, relation_matrix.vocabularies
                    ),
                }
            )
        cases.append(
            {
                "kind": kind,
                "group_id": group_id,
                "group_size": int(grouping.group_sizes[group_id]),
                "pois": poi_rows,
            }
        )
    return cases


def _matrix_sha256(values: np.ndarray, chunk_rows: int = 16_384) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(values), chunk_rows):
        chunk = np.ascontiguousarray(values[start : start + chunk_rows])
        digest.update(chunk.view(np.uint8))
    return digest.hexdigest()


def evaluate_g6_entity_structure(
    *,
    project_root: Path,
    poi_dir: Path,
    structure_dir: Path,
    proxy_dir: Path,
    output_dir: Path,
    experiment_id: str,
    entity_min_support: int = 2,
    entity_vocab_size: int = 32_767,
    max_relation_pairs: int = 3,
    max_relation_tokens: int = 6,
    examples_per_kind: int = 20,
    pois_per_case: int = 12,
    enforce_frozen_counts: bool = True,
    progress: Callable[[str], None] | None = None,
) -> G6EntityStructureResult:
    """Run the formal direct-G6 minimum entity description experiment."""

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
        progress("抽取统一实体角色候选，不读取 Query 或订单")
    raw_relations: list[RawFineRelations] = [
        extract_raw_fine_relations(record) for record in records
    ]
    fine_matrix = build_fine_relation_matrix(
        raw_relations,
        entity_min_support=entity_min_support,
        entity_vocab_size=entity_vocab_size,
    )
    strict = np.asarray(strict_all[grouping.collision_rows], dtype=np.int64)
    relation_matrix = build_unified_relation_matrix(
        strict_relations=strict, fine_relations=fine_matrix
    )
    if progress is not None:
        progress(
            f"编译桶内最短可区分描述：最多 {max_relation_pairs} 对、"
            f"{max_relation_tokens} 个关系 Token"
        )
    compilation = compile_minimum_descriptions(
        grouping=grouping,
        relation_values=relation_matrix.values,
        semantic_tiers=SEMANTIC_TIERS,
        max_pairs=max_relation_pairs,
        max_tokens=max_relation_tokens,
        progress=progress,
    )

    relation_pair_lengths = np.sum(compilation.path_types >= 0, axis=1)
    relation_token_lengths = _path_token_lengths(compilation)
    all_suffix_tokens = np.asarray(arrays["coarse_geo_lengths"]).astype(
        np.int64, copy=True
    )
    all_suffix_tokens[grouping.collision_rows] += relation_token_lengths
    all_semantic_atoms = np.asarray(arrays["coarse_geo_lengths"]).astype(
        np.int64, copy=True
    )
    all_semantic_atoms[grouping.collision_rows] += relation_pair_lengths

    original_excess = int(
        structure_metrics["main"]["catalog"]["collision_excess_before"]
    )
    remaining_excess = int(compilation.metrics["collision_excess"])
    reduction_ratio = _ratio(original_excess - remaining_excess, original_excess)
    relation_emissions = {
        relation_type: int(
            np.count_nonzero(compilation.path_types == relation_index)
        )
        for relation_index, relation_type in enumerate(UNIFIED_RELATION_TYPES)
    }
    building_index = UNIFIED_RELATION_INDEX["R_BUILDING"]
    building_values = relation_matrix.values[:, building_index]
    building_code_candidates = (building_values >= 0) & (building_values % 2 == 1)
    building_code_selected = np.any(
        (compilation.path_types == building_index)
        & (compilation.path_values % 2 == 1),
        axis=1,
    )
    static_upper_bounds = {
        "displayname": _static_text_upper_bound(
            grouping, records, ("displayname",)
        ),
        "displayname_address": _static_text_upper_bound(
            grouping, records, ("displayname", "address")
        ),
        "all_stable_text": _static_text_upper_bound(
            grouping,
            records,
            (
                "displayname",
                "address",
                "alias",
                "category",
                "category_code",
                "layer",
            ),
        ),
    }
    cases = _case_payloads(
        grouping=grouping,
        records=records,
        compilation=compilation,
        relation_matrix=relation_matrix,
        examples_per_kind=examples_per_kind,
        pois_per_case=pois_per_case,
    )
    green_case = next((case for case in cases if case["kind"] == "green_garden"), None)
    green_c12b = None
    if green_case is not None:
        green_c12b = next(
            (
                poi
                for poi in green_case["pois"]
                if "C12B" in str(poi["displayname"])
            ),
            None,
        )
    green_gate = bool(
        green_c12b
        and len(green_c12b["path"]) == 1
        and green_c12b["path"][0]["type"] == "R_BUILDING"
        and green_c12b["path"][0]["decoded"] == "C12B"
    )
    structure_gate_passed = (
        reduction_ratio >= 0.95
        and int(relation_token_lengths.max(initial=0)) <= max_relation_tokens
        and green_gate
    )
    metrics = {
        "schema_version": G6_ENTITY_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "experiment_id": experiment_id,
        "method": "GHR-SID",
        "scope": "北京市冻结 TIGER 碰撞 POI",
        "query_usage": "none",
        "main_protocol": "TIGER3->G6->minimum_entity_description->EOS",
        "configuration": {
            "independent_r3_stage": False,
            "fine_geography_after_g6": False,
            "max_relation_pairs": max_relation_pairs,
            "max_relation_tokens": max_relation_tokens,
            "relation_types": list(UNIFIED_RELATION_TYPES),
            "semantic_tiers": {
                relation_type: int(SEMANTIC_TIERS[index])
                for index, relation_type in enumerate(UNIFIED_RELATION_TYPES)
            },
            "selection": "exact minimum distinguishing description",
            "poi_id_or_bucket_order_used": False,
            "query_or_order_used": False,
            "identical_entities_keep_collision": True,
            "candidate_matrix_saved": False,
        },
        "post_g6_input": frozen_counts,
        "relation_features": relation_matrix.metrics,
        "compiler": compilation.metrics,
        "main": {
            "g6_resolved_poi_count": int(
                len(arrays["base_sid_keys"]) - len(grouping.collision_rows)
            ),
            "entity_resolved_poi_count": int(
                compilation.metrics["resolved_poi_count"]
            ),
            "unresolved_poi_count": int(
                compilation.metrics["collision_poi_count"]
            ),
            "unresolved_group_count": int(
                compilation.metrics["collision_group_count"]
            ),
            "max_unresolved_group_size": int(
                compilation.metrics["max_collision_group_size"]
            ),
            "collision_excess_after": remaining_excess,
            "collision_excess_reduction_ratio": reduction_ratio,
            "full_distinct_identifier_count": full_poi_count - remaining_excess,
            "full_distinct_identifier_ratio": _ratio(
                full_poi_count - remaining_excess, full_poi_count
            ),
            "relation_pair_distribution": _distribution(relation_pair_lengths),
            "relation_token_distribution": _distribution(relation_token_lengths),
            "mean_relation_pair_count_on_post_g6": float(
                relation_pair_lengths.mean()
            ),
            "mean_relation_token_count_on_post_g6": float(
                relation_token_lengths.mean()
            ),
            "max_relation_pair_count": int(
                relation_pair_lengths.max(initial=0)
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
            "building_code_candidate_poi_count": int(
                np.count_nonzero(building_code_candidates)
            ),
            "building_code_selected_poi_count": int(
                np.count_nonzero(building_code_selected)
            ),
            "relation_emitted_pair_count_by_type": relation_emissions,
        },
        "static_text_upper_bounds_after_g6": static_upper_bounds,
        "green_garden_c12b": green_c12b,
        "decision": {
            "pre_registered_global_excess_reduction_gate": 0.95,
            "pre_registered_max_relation_tokens": max_relation_tokens,
            "green_c12b_single_building_relation_gate": green_gate,
            "structure_gate_passed": structure_gate_passed,
            "ready_for_final_mapping": False,
            "ready_for_sft": False,
            "next_step": "先审计全量路径与残留，不直接启动 SFT",
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
    }

    if progress is not None:
        progress("计算统一实体候选矩阵指纹并写入受管产物")
    candidate_matrix_sha256 = _matrix_sha256(relation_matrix.values)
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
        "relation_resolved": compilation.resolved.astype(np.uint8),
    }
    for name, array in output_arrays.items():
        outputs["arrays"][name] = _save_array(staging_dir, name, array)
    vocab_path = staging_dir / "vocabularies.json"
    _json_dump(
        vocab_path,
        {
            "schema_version": UNIFIED_RELATION_SCHEMA_VERSION,
            "relation_types": list(UNIFIED_RELATION_TYPES),
            "vocabularies": {
                name: list(values)
                for name, values in relation_matrix.vocabularies.items()
            },
        },
    )
    outputs["vocabularies"] = _file_contract(vocab_path)
    cases_path = staging_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
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
        "candidate_matrix_sha256": candidate_matrix_sha256,
    }
    manifest = {
        "schema_version": G6_ENTITY_SCHEMA_VERSION,
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
            "candidate_matrix": {
                "shape": list(relation_matrix.values.shape),
                "dtype": str(relation_matrix.values.dtype),
                "sha256": candidate_matrix_sha256,
                "saved": False,
                "rebuild": "raw POI + strict_relations + frozen code",
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
            "poi_id_used_as_feature": False,
            "bucket_order_used_as_feature": False,
            "post_g6_counts_match": frozen_counts == expected_counts,
            "proxy_and_g6_row_order_match": True,
            "raw_sources_match_structure_manifest": True,
            "all_target_pois_found_exactly_once": True,
            "green_c12b_is_single_building_relation": green_gate,
        },
    }
    _json_dump(staging_dir / "manifest.json", manifest)
    staging_dir.rename(output_dir)
    validate_g6_entity_structure_output(output_dir)
    return G6EntityStructureResult(
        metrics=metrics, manifest=manifest, output_dir=output_dir
    )


def validate_g6_entity_structure_output(output_dir: Path) -> dict[str, Any]:
    """Validate every managed direct-G6 entity artifact."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "G6 entity manifest")
    if manifest.get("schema_version") != G6_ENTITY_SCHEMA_VERSION:
        raise GhrSidEntityStructureError("G6 entity manifest schema 不受支持")
    if manifest.get("status") != "completed":
        raise GhrSidEntityStructureError("G6 entity manifest 尚未 completed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise GhrSidEntityStructureError("G6 entity manifest 缺少 outputs")
    validated = 0
    arrays = outputs.get("arrays")
    if not isinstance(arrays, dict):
        raise GhrSidEntityStructureError("G6 entity manifest 缺少 arrays")
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
    metrics = _load_json(output_dir / "metrics.json", "G6 entity metrics")
    if metrics.get("schema_version") != G6_ENTITY_METRICS_SCHEMA_VERSION:
        raise GhrSidEntityStructureError("G6 entity metrics schema 不受支持")
    return {
        "status": "validated",
        "validated_output_count": validated,
        "experiment_id": manifest.get("experiment_id"),
        "signature": manifest.get("signature"),
        "main_protocol": metrics.get("main_protocol"),
    }

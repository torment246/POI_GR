"""G6 -> coarse R3 -> fine entity relation structural experiment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np

from poi_gr.methods.ghr_sid.fine_relations import (
    BASE_FINE_RELATION_COUNT,
    CATEGORY_RELATION_COUNT,
    FINE_RELATION_SCHEMA_VERSION,
    FINE_RELATION_TYPES,
    NUMERIC_RELATION_COUNT,
    ROAD_RELATION_COUNT,
    RawFineRelations,
    build_fine_relation_matrix,
    decode_fine_relation_value,
    extract_raw_fine_relations,
    normalize_entity_label,
)
from poi_gr.methods.ghr_sid.structure import validate_structure_output
from poi_gr.methods.qgr_sid.compiler import compile_relation_bucket
from poi_gr.methods.tiger.identifier import sha256_file
from poi_gr.pid.geohash import tokens_to_geohash


ENTITY_STRUCTURE_SCHEMA_VERSION = "ghr-sid-entity-structure-v1"
ENTITY_STRUCTURE_METRICS_SCHEMA_VERSION = "ghr-sid-entity-structure-metrics-v1"
EXPECTED_POST_R3_POI_COUNT = 276_828
EXPECTED_POST_R3_GROUP_COUNT = 119_271
EXPECTED_POST_R3_EXCESS = 157_557
EXPECTED_POST_R3_MAX_GROUP_SIZE = 97
ABLATION_COLUMNS = {
    "typed_numeric": NUMERIC_RELATION_COUNT,
    "typed_numeric_category": NUMERIC_RELATION_COUNT + CATEGORY_RELATION_COUNT,
    "typed_numeric_category_road": (
        NUMERIC_RELATION_COUNT + CATEGORY_RELATION_COUNT + ROAD_RELATION_COUNT
    ),
    "full_entity_relation": BASE_FINE_RELATION_COUNT,
    "full_entity_relation_positional_chars": len(FINE_RELATION_TYPES),
}


class GhrSidEntityStructureError(ValueError):
    """Raised when entity-relation structure inputs violate the contract."""


@dataclass(frozen=True)
class PostR3Grouping:
    """Rows unresolved after frozen G6 and coarse R3, in collision-row order."""

    collision_rows: np.ndarray
    group_ids: np.ndarray
    group_sizes: np.ndarray
    member_order: np.ndarray
    group_offsets: np.ndarray


@dataclass(frozen=True)
class FinePathCompilation:
    """Minimal greedy fine-relation paths in post-R3 row order."""

    path_types: np.ndarray
    path_values: np.ndarray
    resolved: np.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class EntityStructureResult:
    """Completed fine entity relation structural experiment."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


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
        raise GhrSidEntityStructureError(
            f"无法读取{name}：{path}：{error}"
        ) from error
    if not isinstance(payload, dict):
        raise GhrSidEntityStructureError(f"{name}必须是 JSON object：{path}")
    return payload


def _signature(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_state(project_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}


def _load_array(
    directory: Path,
    contract: Mapping[str, Any],
    name: str,
) -> np.ndarray:
    path = directory / str(contract.get("path", ""))
    if not path.is_file() or sha256_file(path) != contract.get("sha256"):
        raise GhrSidEntityStructureError(f"输入数组缺失或 SHA 不一致：{name}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != contract.get("shape") or str(
        array.dtype
    ) != contract.get("dtype"):
        raise GhrSidEntityStructureError(f"输入数组 shape/dtype 不一致：{name}")
    return array


def _load_structure_arrays(
    structure_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    validate_structure_output(structure_dir)
    manifest = _load_json(structure_dir / "manifest.json", "结构 manifest")
    metrics = _load_json(structure_dir / "metrics.json", "结构 metrics")
    contracts = manifest.get("outputs", {}).get("arrays")
    if not isinstance(contracts, dict):
        raise GhrSidEntityStructureError("结构 manifest 缺少 arrays")
    names = (
        "collision_poi_ids",
        "base_sid_keys",
        "collision_gid8_codes",
        "root_prefix_lengths",
        "coarse_geo_codes",
        "coarse_geo_lengths",
        "relation_path_types",
        "relation_path_values",
        "resolution_stage",
    )
    arrays = {
        name: _load_array(structure_dir, contracts[name], name) for name in names
    }
    return arrays, manifest, metrics


def reconstruct_post_r3_groups(
    *,
    base_sid_keys: np.ndarray,
    root_prefix_lengths: np.ndarray,
    coarse_geo_codes: np.ndarray,
    coarse_geo_lengths: np.ndarray,
    coarse_relation_types: np.ndarray,
    coarse_relation_values: np.ndarray,
    resolution_stage: np.ndarray,
) -> PostR3Grouping:
    """Reconstruct frozen groups entering G8, before any fine geography."""

    base_sid_keys = np.asarray(base_sid_keys)
    rows = len(base_sid_keys)
    one_dimensional = (root_prefix_lengths, coarse_geo_lengths, resolution_stage)
    if base_sid_keys.ndim != 1 or any(np.asarray(value).shape != (rows,) for value in one_dimensional):
        raise GhrSidEntityStructureError("post-R3 一维数组 shape 不一致")
    if coarse_geo_codes.shape[0] != rows:
        raise GhrSidEntityStructureError("coarse_geo_codes 行数不一致")
    if coarse_relation_types.shape != coarse_relation_values.shape or (
        coarse_relation_types.shape[0] != rows
    ):
        raise GhrSidEntityStructureError("coarse relation 数组 shape 不一致")
    collision_rows = np.flatnonzero(
        (np.asarray(resolution_stage) == 0) | (np.asarray(resolution_stage) == 3)
    ).astype(np.int64, copy=False)
    feature_matrix = np.ascontiguousarray(
        np.column_stack(
            [
                np.asarray(root_prefix_lengths)[collision_rows].astype(np.int16),
                np.asarray(coarse_geo_lengths)[collision_rows].astype(np.int16),
                np.asarray(coarse_geo_codes)[collision_rows].astype(np.int16),
                np.asarray(coarse_relation_types)[collision_rows].astype(np.int16),
                np.asarray(coarse_relation_values)[collision_rows].astype(np.int16),
            ]
        )
    )
    feature_bytes = feature_matrix.view(
        np.dtype((np.void, feature_matrix.dtype.itemsize * feature_matrix.shape[1]))
    ).reshape(-1)
    keys = np.empty(
        len(collision_rows),
        dtype=[("base_sid_key", "<i8"), ("features", feature_bytes.dtype)],
    )
    keys["base_sid_key"] = np.asarray(base_sid_keys)[collision_rows]
    keys["features"] = feature_bytes
    order = np.argsort(
        keys, order=("base_sid_key", "features"), kind="stable"
    )
    sorted_keys = keys[order]
    boundaries = np.ones(len(order), dtype=bool)
    boundaries[1:] = (
        sorted_keys["base_sid_key"][1:] != sorted_keys["base_sid_key"][:-1]
    ) | (sorted_keys["features"][1:] != sorted_keys["features"][:-1])
    starts = np.flatnonzero(boundaries)
    ends = np.append(starts[1:], len(order))
    group_sizes = (ends - starts).astype(np.int32, copy=False)
    if np.any(group_sizes < 2):
        raise GhrSidEntityStructureError("post-R3 输入出现单例组")
    group_ids = np.empty(len(collision_rows), dtype=np.int32)
    for group_id, (start, end) in enumerate(zip(starts, ends, strict=True)):
        group_ids[order[start:end]] = group_id
    member_order = np.argsort(group_ids, kind="stable").astype(np.int64)
    group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(group_sizes, dtype=np.int64)]
    )
    return PostR3Grouping(
        collision_rows=collision_rows,
        group_ids=group_ids,
        group_sizes=group_sizes,
        member_order=member_order,
        group_offsets=group_offsets,
    )


def _group_members(grouping: PostR3Grouping, group_id: int) -> np.ndarray:
    start = int(grouping.group_offsets[group_id])
    end = int(grouping.group_offsets[group_id + 1])
    return grouping.member_order[start:end]


def _raw_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _stream_target_pois(
    *,
    poi_dir: Path,
    target_poi_ids: np.ndarray,
    expected_sources: Sequence[Mapping[str, Any]],
    progress: Callable[[str], None] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    source_paths = tuple(sorted(poi_dir.glob("part-*.json")))
    expected_by_path = {
        str(Path(str(item["path"])).resolve()): item for item in expected_sources
    }
    if {str(path.resolve()) for path in source_paths} != set(expected_by_path):
        raise GhrSidEntityStructureError("POI 原始分片集合与冻结结构不一致")
    target_by_id = {
        int(poi_id): position for position, poi_id in enumerate(target_poi_ids)
    }
    if len(target_by_id) != len(target_poi_ids):
        raise GhrSidEntityStructureError("post-R3 POI ID 存在重复")
    records: list[dict[str, Any] | None] = [None] * len(target_poi_ids)
    completed_sources: list[dict[str, Any]] = []
    scanned = 0
    found = 0
    for source_path in source_paths:
        digest = hashlib.sha256()
        source_rows = 0
        with source_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                digest.update(line)
                if not line.strip():
                    raise GhrSidEntityStructureError(
                        f"POI 数据存在空行：{source_path.name}:{line_number}"
                    )
                try:
                    raw = json.loads(line)
                    poi_id = int(str(raw["poi_id"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise GhrSidEntityStructureError(
                        f"POI JSON/ID 非法：{source_path.name}:{line_number}"
                    ) from error
                position = target_by_id.get(poi_id)
                if position is not None:
                    if records[position] is not None:
                        raise GhrSidEntityStructureError(f"POI 重复出现：{poi_id}")
                    records[position] = {
                        "poi_id": str(poi_id),
                        "displayname": _raw_text(raw.get("displayname")),
                        "address": _raw_text(raw.get("address")),
                        "alias": _raw_text(raw.get("alias")),
                        "category": _raw_text(raw.get("category")),
                        "category_code": _raw_text(raw.get("category_code")),
                        "layer": _raw_text(raw.get("layer")),
                    }
                    found += 1
                scanned += 1
                source_rows += 1
        completed = {
            "path": str(source_path.resolve()),
            "bytes": source_path.stat().st_size,
            "rows": source_rows,
            "sha256": digest.hexdigest(),
        }
        expected = expected_by_path[completed["path"]]
        for key in ("bytes", "rows", "sha256"):
            if completed[key] != expected.get(key):
                raise GhrSidEntityStructureError(
                    f"POI 分片 {source_path.name} 的 {key} 与冻结结构不一致"
                )
        completed_sources.append(completed)
        if progress is not None:
            progress(
                f"扫描原始 POI {scanned:,} 行；细关系目标 {found:,}/"
                f"{len(target_poi_ids):,}"
            )
    if found != len(target_poi_ids) or any(record is None for record in records):
        raise GhrSidEntityStructureError(
            f"post-R3 POI 回查不完整：{found} != {len(target_poi_ids)}"
        )
    return [record for record in records if record is not None], completed_sources, scanned


def _path_keys(path_types: np.ndarray, path_values: np.ndarray) -> list[Hashable]:
    return [
        tuple(
            (int(relation_type), int(value))
            for relation_type, value in zip(types, values, strict=True)
            if relation_type >= 0
        )
        for types, values in zip(path_types, path_values, strict=True)
    ]


def _collision_stats(
    group_ids: np.ndarray,
    keys: Sequence[Hashable],
) -> tuple[dict[str, int], np.ndarray]:
    counts = Counter(
        (int(group_id), key)
        for group_id, key in zip(group_ids, keys, strict=True)
    )
    collision_sizes = [count for count in counts.values() if count > 1]
    resolved = np.asarray(
        [counts[(int(group_id), key)] == 1 for group_id, key in zip(group_ids, keys, strict=True)],
        dtype=bool,
    )
    return (
        {
            "collision_excess": int(sum(count - 1 for count in collision_sizes)),
            "collision_poi_count": int(sum(collision_sizes)),
            "collision_group_count": len(collision_sizes),
            "max_collision_group_size": max(collision_sizes, default=1),
            "distinct_count": len(counts),
        },
        resolved,
    )


def compile_fine_relation_paths(
    *,
    grouping: PostR3Grouping,
    relation_values: np.ndarray,
    relation_count: int,
    allow_presence_split: bool = False,
) -> FinePathCompilation:
    """Greedily compile shortest available entity paths with early EOS."""

    relation_values = np.asarray(relation_values)
    if relation_values.shape[0] != len(grouping.collision_rows):
        raise GhrSidEntityStructureError("fine relation 行数与 post-R3 行数不一致")
    if not 1 <= relation_count <= relation_values.shape[1]:
        raise GhrSidEntityStructureError("relation_count 非法")
    path_types = np.full(
        (len(grouping.collision_rows), relation_count), -1, dtype=np.int16
    )
    path_values = np.full(
        (len(grouping.collision_rows), relation_count),
        -1,
        dtype=relation_values.dtype,
    )
    decisions = 0
    for group_id in range(len(grouping.group_sizes)):
        members = _group_members(grouping, group_id)
        rows = len(members)
        selected = relation_values[members, :relation_count]
        zeros = np.zeros(rows, dtype=np.int32)
        compilation = compile_relation_bucket(
            selected,
            zeros,
            np.zeros_like(selected, dtype=np.int32),
            np.zeros(relation_count, dtype=np.float64),
            query_guided=False,
            max_pairs=relation_count,
            p99_order_count=0.0,
            prior_orders=1.0,
            allow_presence_split=allow_presence_split,
        )
        path_types[members] = compilation.path_types
        path_values[members] = compilation.path_values
        decisions += compilation.decision_count
    keys = _path_keys(path_types, path_values)
    stats, resolved = _collision_stats(grouping.group_ids, keys)
    lengths = np.sum(path_types >= 0, axis=1)
    stats.update(
        {
            "relation_count": relation_count,
            "decision_node_count": decisions,
            "resolved_poi_count": int(np.count_nonzero(resolved)),
            "resolved_poi_ratio": _ratio(np.count_nonzero(resolved), len(resolved)),
            "mean_path_pair_count": float(lengths.mean()),
            "max_path_pair_count": int(lengths.max(initial=0)),
            "path_pair_count_distribution": {
                str(int(value)): int(count)
                for value, count in sorted(Counter(lengths.tolist()).items())
            },
        }
    )
    return FinePathCompilation(
        path_types=path_types,
        path_values=path_values,
        resolved=resolved,
        metrics=stats,
    )


def _distribution(values: np.ndarray) -> dict[str, int]:
    return {
        str(int(value)): int(count)
        for value, count in sorted(Counter(values.tolist()).items())
    }


def _serialized_relation_lengths(types: np.ndarray, values: np.ndarray) -> np.ndarray:
    active = types >= 0
    remaining = np.maximum(values, 0).astype(np.int64, copy=True)
    value_digits = np.ones_like(remaining, dtype=np.int64)
    while np.any(remaining >= 1024):
        needs_digit = remaining >= 1024
        value_digits[needs_digit] += 1
        remaining[needs_digit] //= 1024
    return np.sum(active * (1 + value_digits), axis=1, dtype=np.int64)


def _static_text_upper_bound(
    grouping: PostR3Grouping,
    records: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, int]:
    keys = [
        tuple(
            normalize_entity_label(record[field])
            for field in fields
        )
        for record in records
    ]
    stats, _ = _collision_stats(grouping.group_ids, keys)
    return stats


def _case_payloads(
    *,
    grouping: PostR3Grouping,
    records: Sequence[Mapping[str, Any]],
    collision_poi_ids: np.ndarray,
    gid8_codes: np.ndarray,
    compilation: FinePathCompilation,
    vocabularies: Mapping[str, Sequence[str]],
    examples_per_stage: int,
    pois_per_case: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    path_keys = _path_keys(compilation.path_types, compilation.path_values)
    path_counts = Counter(
        (int(group_id), key)
        for group_id, key in zip(grouping.group_ids, path_keys, strict=True)
    )

    resolved_rows = np.flatnonzero(compilation.resolved)[:examples_per_stage]
    for row in resolved_rows:
        collision_row = int(grouping.collision_rows[row])
        fine_path = []
        for relation_index, value in zip(
            compilation.path_types[row], compilation.path_values[row], strict=True
        ):
            if relation_index < 0:
                continue
            relation_type = FINE_RELATION_TYPES[int(relation_index)]
            fine_path.append(
                {
                    "type": relation_type,
                    "value": int(value),
                    "decoded": decode_fine_relation_value(
                        relation_type, int(value), vocabularies
                    ),
                }
            )
        cases.append(
            {
                "stage": "fine_entity_resolved",
                "group_id": int(grouping.group_ids[row]),
                "poi_id": str(int(collision_poi_ids[collision_row])),
                "displayname": records[row]["displayname"],
                "address": records[row]["address"],
                "gid6": tokens_to_geohash(gid8_codes[collision_row, :6]),
                "fine_relation_path": fine_path,
            }
        )

    unresolved_clusters: list[tuple[int, int, Hashable, np.ndarray]] = []
    for group_id in range(len(grouping.group_sizes)):
        members = _group_members(grouping, group_id)
        by_path: dict[Hashable, list[int]] = {}
        for member in members:
            key = path_keys[int(member)]
            if path_counts[(group_id, key)] > 1:
                by_path.setdefault(key, []).append(int(member))
        for key, cluster in by_path.items():
            unresolved_clusters.append(
                (len(cluster), group_id, key, np.asarray(cluster, dtype=np.int64))
            )
    unresolved_clusters.sort(key=lambda item: (-item[0], item[1], repr(item[2])))
    for cluster_size, group_id, key, members in unresolved_clusters[
        :examples_per_stage
    ]:
        first = int(members[0])
        collision_row = int(grouping.collision_rows[first])
        fine_path = [
            {
                "type": FINE_RELATION_TYPES[int(relation_index)],
                "value": int(value),
                "decoded": decode_fine_relation_value(
                    FINE_RELATION_TYPES[int(relation_index)],
                    int(value),
                    vocabularies,
                ),
            }
            for relation_index, value in key
        ]
        pois = [
            {
                "poi_id": records[int(member)]["poi_id"],
                "displayname": records[int(member)]["displayname"],
                "address": records[int(member)]["address"],
                "alias": records[int(member)]["alias"],
                "category": records[int(member)]["category"],
            }
            for member in members[:pois_per_case]
        ]
        cases.append(
            {
                "stage": "unresolved",
                "group_id": group_id,
                "cluster_size": cluster_size,
                "shown_poi_count": len(pois),
                "gid6": tokens_to_geohash(gid8_codes[collision_row, :6]),
                "fine_relation_path": fine_path,
                "pois": pois,
            }
        )
    return cases


def _array_contract(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _save_array(directory: Path, name: str, array: np.ndarray) -> dict[str, Any]:
    path = directory / f"{name}.npy"
    np.save(path, array, allow_pickle=False)
    return _array_contract(path, array)


def _file_contract(path: Path, rows: int | None = None) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        contract["rows"] = rows
    return contract


def evaluate_entity_relation_structure(
    *,
    project_root: Path,
    poi_dir: Path,
    structure_dir: Path,
    output_dir: Path,
    experiment_id: str,
    entity_min_support: int = 2,
    entity_vocab_size: int = 32_767,
    examples_per_stage: int = 20,
    pois_per_case: int = 8,
    enforce_frozen_counts: bool = True,
    progress: Callable[[str], None] | None = None,
) -> EntityStructureResult:
    """Run the query-free G6 -> R3 -> fine entity relation experiment."""

    started_at = _utc_now()
    started = time.monotonic()
    project_root = project_root.resolve()
    poi_dir = poi_dir.resolve()
    structure_dir = structure_dir.resolve()
    output_dir = output_dir.resolve()
    if not re.fullmatch(r"EXP-\d{8}-\d{2}", experiment_id):
        raise GhrSidEntityStructureError("experiment_id 必须形如 EXP-YYYYMMDD-NN")
    if examples_per_stage < 0 or pois_per_case <= 0:
        raise GhrSidEntityStructureError("案例参数非法")
    for path, name in (
        (project_root, "项目根目录"),
        (poi_dir, "POI 目录"),
        (structure_dir, "结构实验目录"),
    ):
        if not path.is_dir():
            raise GhrSidEntityStructureError(f"{name}不存在：{path}")
    if output_dir.exists():
        raise GhrSidEntityStructureError(f"输出目录已存在，拒绝覆盖：{output_dir}")

    arrays, structure_manifest, structure_metrics = _load_structure_arrays(
        structure_dir
    )
    grouping = reconstruct_post_r3_groups(
        base_sid_keys=arrays["base_sid_keys"],
        root_prefix_lengths=arrays["root_prefix_lengths"],
        coarse_geo_codes=arrays["coarse_geo_codes"],
        coarse_geo_lengths=arrays["coarse_geo_lengths"],
        coarse_relation_types=arrays["relation_path_types"],
        coarse_relation_values=arrays["relation_path_values"],
        resolution_stage=arrays["resolution_stage"],
    )
    frozen_counts = {
        "poi_count": len(grouping.collision_rows),
        "group_count": len(grouping.group_sizes),
        "collision_excess": int(np.sum(grouping.group_sizes - 1)),
        "max_group_size": int(grouping.group_sizes.max(initial=0)),
    }
    expected_counts = {
        "poi_count": EXPECTED_POST_R3_POI_COUNT,
        "group_count": EXPECTED_POST_R3_GROUP_COUNT,
        "collision_excess": EXPECTED_POST_R3_EXCESS,
        "max_group_size": EXPECTED_POST_R3_MAX_GROUP_SIZE,
    }
    if enforce_frozen_counts and frozen_counts != expected_counts:
        raise GhrSidEntityStructureError(
            f"post-R3 冻结计数变化：实际 {frozen_counts}，期望 {expected_counts}"
        )
    target_poi_ids = np.asarray(arrays["collision_poi_ids"])[
        grouping.collision_rows
    ]
    expected_sources = structure_manifest.get("inputs", {}).get("raw_sources")
    if not isinstance(expected_sources, list):
        raise GhrSidEntityStructureError("结构 manifest 缺少 raw_sources")
    records, raw_sources, full_poi_count = _stream_target_pois(
        poi_dir=poi_dir,
        target_poi_ids=target_poi_ids,
        expected_sources=expected_sources,
        progress=progress,
    )
    if progress is not None:
        progress("抽取细粒度数字、类别、道路、父场所和子实体关系")
    raw_relations: list[RawFineRelations] = [
        extract_raw_fine_relations(record) for record in records
    ]
    relation_matrix = build_fine_relation_matrix(
        raw_relations,
        entity_min_support=entity_min_support,
        entity_vocab_size=entity_vocab_size,
    )
    compilations: dict[str, FinePathCompilation] = {}
    for name, relation_count in ABLATION_COLUMNS.items():
        if progress is not None:
            progress(f"编译细关系消融 {name}：{relation_count} 类候选")
        compilations[name] = compile_fine_relation_paths(
            grouping=grouping,
            relation_values=relation_matrix.values,
            relation_count=relation_count,
            allow_presence_split=True,
        )
    main = compilations["full_entity_relation_positional_chars"]

    coarse_geo_lengths = np.asarray(arrays["coarse_geo_lengths"]).astype(
        np.int64, copy=False
    )
    coarse_relation_types = np.asarray(arrays["relation_path_types"])
    coarse_relation_values = np.asarray(arrays["relation_path_values"])
    coarse_relation_lengths = np.sum(coarse_relation_types >= 0, axis=1)
    fine_relation_lengths = np.sum(main.path_types >= 0, axis=1)
    semantic_atom_lengths = coarse_geo_lengths + coarse_relation_lengths
    semantic_atom_lengths = semantic_atom_lengths.astype(np.int64, copy=True)
    semantic_atom_lengths[grouping.collision_rows] += fine_relation_lengths
    serialized_lengths = coarse_geo_lengths + _serialized_relation_lengths(
        coarse_relation_types, coarse_relation_values
    )
    serialized_lengths = serialized_lengths.astype(np.int64, copy=True)
    serialized_lengths[grouping.collision_rows] += _serialized_relation_lengths(
        main.path_types, main.path_values
    )
    original_excess = int(
        structure_metrics["main"]["catalog"]["collision_excess_before"]
    )
    remaining_excess = int(main.metrics["collision_excess"])
    reduction_ratio = _ratio(original_excess - remaining_excess, original_excess)
    relation_emissions = {
        relation_type: int(np.count_nonzero(main.path_types == relation_index))
        for relation_index, relation_type in enumerate(FINE_RELATION_TYPES)
    }
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
    gate_passed = reduction_ratio >= 0.95 and float(
        semantic_atom_lengths.mean()
    ) <= 3.0
    metrics = {
        "schema_version": ENTITY_STRUCTURE_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "experiment_id": experiment_id,
        "method": "GHR-SID",
        "scope": "北京市冻结 TIGER 碰撞 POI",
        "query_usage": "none",
        "main_protocol": "G6->coarse_R3->fine_entity_relations->EOS",
        "configuration": {
            "fine_geography_after_g6": False,
            "coarse_relation_pairs": 3,
            "fine_relation_types": list(FINE_RELATION_TYPES),
            "fine_relation_max_pairs": len(FINE_RELATION_TYPES),
            "fine_path_early_stop_when_unique": True,
            "missing_relation_presence_is_branch": True,
            "entity_min_support": entity_min_support,
            "entity_vocab_size": entity_vocab_size,
            "entity_value_scope": "global_post_R3_not_bucket_local",
            "entity_value_assignment": "support_desc_then_lexical_order",
            "poi_id_or_bucket_order_used": False,
            "numeric_value_serialization": "base1024_one_or_two_digits",
        },
        "post_r3_input": frozen_counts,
        "fine_relation_features": relation_matrix.metrics,
        "ablations": {
            name: compilation.metrics
            for name, compilation in compilations.items()
        },
        "main": {
            "fine_relation_resolved_poi_count": main.metrics["resolved_poi_count"],
            "fine_relation_resolved_poi_ratio_of_post_r3": main.metrics[
                "resolved_poi_ratio"
            ],
            "unresolved_poi_count": main.metrics["collision_poi_count"],
            "unresolved_group_count": main.metrics["collision_group_count"],
            "max_unresolved_group_size": main.metrics[
                "max_collision_group_size"
            ],
            "collision_excess_after": remaining_excess,
            "collision_excess_reduction_ratio": reduction_ratio,
            "full_distinct_identifier_count": full_poi_count - remaining_excess,
            "full_distinct_identifier_ratio": _ratio(
                full_poi_count - remaining_excess, full_poi_count
            ),
            "fine_relation_pair_length_distribution": _distribution(
                fine_relation_lengths
            ),
            "mean_fine_relation_pair_count_on_post_r3": float(
                fine_relation_lengths.mean()
            ),
            "mean_semantic_atom_length_on_all_collision_pois": float(
                semantic_atom_lengths.mean()
            ),
            "max_semantic_atom_length": int(semantic_atom_lengths.max()),
            "mean_serialized_token_length_on_all_collision_pois": float(
                serialized_lengths.mean()
            ),
            "max_serialized_token_length": int(serialized_lengths.max()),
            "fine_relation_emitted_pair_count_by_type": relation_emissions,
        },
        "static_text_upper_bound_after_r3": static_upper_bounds[
            "all_stable_text"
        ],
        "static_text_upper_bounds_after_r3": static_upper_bounds,
        "decision": {
            "pre_registered_excess_reduction_gate": 0.95,
            "pre_registered_mean_semantic_atom_gate": 3.0,
            "structure_gate_passed": gate_passed,
            "ready_for_final_mapping": False,
            "ready_for_sft": False,
            "mapping_blocker": (
                "必须先确定完全同静态实体的 canonical/equivalence 口径"
            ),
            "sft_blocker": (
                "静态结构通过也不能替代请求加权 Token 支持与生成评测"
            ),
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
    }

    cases = _case_payloads(
        grouping=grouping,
        records=records,
        collision_poi_ids=arrays["collision_poi_ids"],
        gid8_codes=arrays["collision_gid8_codes"],
        compilation=main,
        vocabularies=relation_matrix.vocabularies,
        examples_per_stage=examples_per_stage,
        pois_per_case=pois_per_case,
    )
    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging_dir.exists():
        raise GhrSidEntityStructureError(f"暂存目录已存在：{staging_dir}")
    staging_dir.mkdir(parents=True)
    outputs: dict[str, Any] = {"arrays": {}}
    output_arrays = {
        "post_r3_collision_rows": grouping.collision_rows,
        "post_r3_group_ids": grouping.group_ids,
        "post_r3_poi_ids": target_poi_ids,
        "fine_relation_values": relation_matrix.values,
        "fine_path_types": main.path_types,
        "fine_path_values": main.path_values,
        "fine_resolved": main.resolved.astype(np.uint8),
    }
    for name, array in output_arrays.items():
        outputs["arrays"][name] = _save_array(staging_dir, name, array)
    vocab_path = staging_dir / "vocabularies.json"
    _json_dump(
        vocab_path,
        {
            "schema_version": FINE_RELATION_SCHEMA_VERSION,
            "relation_types": list(FINE_RELATION_TYPES),
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

    signature_payload = {
        "experiment_id": experiment_id,
        "configuration": metrics["configuration"],
        "inputs": {
            "structure_manifest_sha256": sha256_file(
                structure_dir / "manifest.json"
            )
        },
    }
    manifest = {
        "schema_version": ENTITY_STRUCTURE_SCHEMA_VERSION,
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
            "raw_sources": raw_sources,
        },
        "outputs": outputs,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "validation": {
            "query_inputs_consumed": False,
            "poi_id_used_as_feature": False,
            "bucket_order_used_as_feature": False,
            "post_r3_counts_match": frozen_counts == expected_counts,
            "raw_sources_match_structure_manifest": True,
            "all_target_pois_found_exactly_once": True,
            "entity_vocab_excludes_singletons": entity_min_support >= 2,
        },
    }
    _json_dump(staging_dir / "manifest.json", manifest)
    staging_dir.rename(output_dir)
    validated = validate_entity_structure_output(output_dir)
    if validated["validated_output_count"] != len(output_arrays) + 3:
        raise GhrSidEntityStructureError("完成后的输出验证计数不一致")
    return EntityStructureResult(
        metrics=metrics,
        manifest=manifest,
        output_dir=output_dir,
    )


def validate_entity_structure_output(output_dir: Path) -> dict[str, Any]:
    """Validate every managed fine entity relation artifact."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "实体关系 manifest")
    if manifest.get("schema_version") != ENTITY_STRUCTURE_SCHEMA_VERSION:
        raise GhrSidEntityStructureError("实体关系 manifest schema 不受支持")
    if manifest.get("status") != "completed":
        raise GhrSidEntityStructureError("实体关系 manifest 尚未 completed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise GhrSidEntityStructureError("实体关系 manifest 缺少 outputs")
    validated = 0
    arrays = outputs.get("arrays")
    if not isinstance(arrays, dict):
        raise GhrSidEntityStructureError("实体关系 manifest 缺少 arrays")
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
    for name in ("vocabularies", "cases", "metrics"):
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
    metrics = _load_json(output_dir / "metrics.json", "实体关系 metrics")
    if metrics.get("schema_version") != ENTITY_STRUCTURE_METRICS_SCHEMA_VERSION:
        raise GhrSidEntityStructureError("实体关系 metrics schema 不受支持")
    return {
        "status": "validated",
        "validated_output_count": validated,
        "experiment_id": manifest.get("experiment_id"),
        "signature": manifest.get("signature"),
        "main_protocol": metrics.get("main_protocol"),
    }

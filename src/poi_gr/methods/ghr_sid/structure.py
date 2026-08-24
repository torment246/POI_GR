"""Query-free structural experiment for geo-first relational collision SIDs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from poi_gr.methods.qgr_sid.compiler import compile_relation_bucket
from poi_gr.methods.qgr_sid.proxy import validate_proxy_output
from poi_gr.methods.qgr_sid.relations import RELATION_TYPES
from poi_gr.methods.tiger.identifier import sha256_file
from poi_gr.pid.geohash import (
    GEOHASH_ALPHABET,
    encode_geohash_tokens,
    tokens_to_geohash,
)


STRUCTURE_SCHEMA_VERSION = "ghr-sid-structure-v1"
STRUCTURE_METRICS_SCHEMA_VERSION = "ghr-sid-structure-metrics-v1"
STAGE_UNRESOLVED = 0
STAGE_COARSE_GEO = 1
STAGE_RELATION = 2
STAGE_FINE_GEO = 3
STAGE_NAMES = {
    STAGE_UNRESOLVED: "unresolved",
    STAGE_COARSE_GEO: "coarse_geo",
    STAGE_RELATION: "relation",
    STAGE_FINE_GEO: "fine_geo",
}


class GhrSidStructureError(ValueError):
    """Raised when GHR-SID structural inputs or outputs violate the contract."""


@dataclass(frozen=True)
class StructureCompilation:
    """One query-free geo-relation structure compiled in collision-row order."""

    root_prefix_lengths: np.ndarray
    coarse_geo_codes: np.ndarray
    coarse_geo_lengths: np.ndarray
    relation_path_types: np.ndarray
    relation_path_values: np.ndarray
    fine_geo_codes: np.ndarray
    fine_geo_lengths: np.ndarray
    resolution_stage: np.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class StructureExperimentResult:
    """Completed GHR-SID structural experiment artifacts."""

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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GhrSidStructureError(f"无法读取{name}：{path}：{error}") from error
    if not isinstance(value, dict):
        raise GhrSidStructureError(f"{name}必须是 JSON object：{path}")
    return value


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


def _common_prefix_length(codes: np.ndarray, stop: int) -> int:
    if len(codes) == 0:
        return 0
    equal = np.all(codes[:, :stop] == codes[0, :stop], axis=0)
    different = np.flatnonzero(~equal)
    return stop if not len(different) else int(different[0])


def _append_geo_codes(
    output: np.ndarray,
    lengths: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
) -> None:
    positions = lengths[indices].astype(np.int64, copy=False)
    if np.any(positions >= output.shape[1]):
        raise GhrSidStructureError("GEO 路径长度超过预分配上限")
    output[indices, positions] = values.astype(np.int16, copy=False)
    lengths[indices] += 1


def _split_geo_tree(
    *,
    indices: np.ndarray,
    gid_codes: np.ndarray,
    start_depth: int,
    stop_depth: int,
    output_codes: np.ndarray,
    output_lengths: np.ndarray,
    resolution_stage: np.ndarray,
    stage: int,
) -> list[np.ndarray]:
    """Split by contiguous Geohash characters and return unresolved leaves."""

    unresolved: list[np.ndarray] = []

    def recurse(rows: np.ndarray, depth: int) -> None:
        if len(rows) <= 1:
            resolution_stage[rows] = stage
            return
        if depth >= stop_depth:
            unresolved.append(rows)
            return
        values = gid_codes[rows, depth]
        _append_geo_codes(output_codes, output_lengths, rows, values)
        for value in np.unique(values):
            branch = rows[values == value]
            if len(branch) == 1:
                resolution_stage[branch] = stage
            else:
                recurse(branch, depth + 1)

    recurse(indices, start_depth)
    return unresolved


def _relation_paths(
    relation_types: np.ndarray,
    relation_values: np.ndarray,
    indices: np.ndarray,
) -> dict[tuple[tuple[int, int], ...], list[int]]:
    groups: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
    for row in indices:
        path = tuple(
            (int(relation_type), int(value))
            for relation_type, value in zip(
                relation_types[row], relation_values[row], strict=True
            )
            if relation_type >= 0
        )
        groups[path].append(int(row))
    return groups


def compile_geo_relation_structure(
    *,
    base_sid_keys: np.ndarray,
    gid_codes: np.ndarray,
    strict_relations: np.ndarray,
    coarse_precision: int,
    fine_precision: int,
    max_relation_pairs: int = 3,
    full_poi_count: int | None = None,
) -> StructureCompilation:
    """Compile GEO-coarse -> relation -> GEO-fine paths without Query data."""

    base_sid_keys = np.asarray(base_sid_keys)
    gid_codes = np.asarray(gid_codes)
    strict_relations = np.asarray(strict_relations)
    if base_sid_keys.ndim != 1 or len(base_sid_keys) < 2:
        raise GhrSidStructureError("base_sid_keys 必须是一维碰撞 POI 数组")
    rows = len(base_sid_keys)
    if gid_codes.shape != (rows, fine_precision):
        raise GhrSidStructureError(
            f"gid_codes 必须是 [{rows},{fine_precision}]"
        )
    if strict_relations.shape != (rows, len(RELATION_TYPES)):
        raise GhrSidStructureError(
            f"strict_relations 必须是 [{rows},{len(RELATION_TYPES)}]"
        )
    if not 1 <= coarse_precision < fine_precision:
        raise GhrSidStructureError("要求 1 <= coarse_precision < fine_precision")
    if max_relation_pairs <= 0:
        raise GhrSidStructureError("max_relation_pairs 必须大于 0")
    if np.any(gid_codes >= len(GEOHASH_ALPHABET)):
        raise GhrSidStructureError("gid_codes 包含非法 Base32 token")
    if full_poi_count is not None and full_poi_count < rows:
        raise GhrSidStructureError("full_poi_count 不能小于碰撞 POI 数")

    root_prefix_lengths = np.zeros(rows, dtype=np.uint8)
    coarse_geo_codes = np.full((rows, coarse_precision), -1, dtype=np.int16)
    coarse_geo_lengths = np.zeros(rows, dtype=np.uint8)
    relation_path_types = np.full(
        (rows, max_relation_pairs), -1, dtype=np.int16
    )
    relation_path_values = np.full(
        (rows, max_relation_pairs), -1, dtype=np.int16
    )
    fine_geo_codes = np.full((rows, fine_precision), -1, dtype=np.int16)
    fine_geo_lengths = np.zeros(rows, dtype=np.uint8)
    resolution_stage = np.zeros(rows, dtype=np.uint8)

    order = np.argsort(base_sid_keys, kind="stable")
    sorted_keys = base_sid_keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]
    relation_decisions = 0
    residual_group_sizes: list[int] = []
    fully_resolved_buckets = 0
    bucket_root_prefixes: Counter[int] = Counter()

    for start, stop in zip(starts, stops, strict=True):
        bucket = order[start:stop]
        if len(bucket) < 2:
            raise GhrSidStructureError("输入包含非碰撞基础 SID 桶")
        root_prefix = _common_prefix_length(gid_codes[bucket], fine_precision)
        root_prefix_lengths[bucket] = root_prefix
        bucket_root_prefixes[root_prefix] += 1
        coarse_start = min(root_prefix, coarse_precision)
        coarse_leaves = _split_geo_tree(
            indices=bucket,
            gid_codes=gid_codes,
            start_depth=coarse_start,
            stop_depth=coarse_precision,
            output_codes=coarse_geo_codes,
            output_lengths=coarse_geo_lengths,
            resolution_stage=resolution_stage,
            stage=STAGE_COARSE_GEO,
        )

        for leaf in coarse_leaves:
            zeros = np.zeros(len(leaf), dtype=np.int32)
            compilation = compile_relation_bucket(
                strict_relations[leaf],
                zeros,
                np.zeros_like(strict_relations[leaf], dtype=np.int32),
                np.zeros(len(RELATION_TYPES), dtype=np.float64),
                query_guided=False,
                max_pairs=max_relation_pairs,
                p99_order_count=0.0,
                prior_orders=1.0,
            )
            relation_decisions += compilation.decision_count
            relation_path_types[leaf] = compilation.path_types
            relation_path_values[leaf] = compilation.path_values
            relation_groups = _relation_paths(
                relation_path_types, relation_path_values, leaf
            )
            for group_rows in relation_groups.values():
                group = np.asarray(group_rows, dtype=np.int64)
                if len(group) == 1:
                    resolution_stage[group] = STAGE_RELATION
                    continue
                fine_root = max(
                    coarse_precision,
                    _common_prefix_length(gid_codes[group], fine_precision),
                )
                fine_leaves = _split_geo_tree(
                    indices=group,
                    gid_codes=gid_codes,
                    start_depth=fine_root,
                    stop_depth=fine_precision,
                    output_codes=fine_geo_codes,
                    output_lengths=fine_geo_lengths,
                    resolution_stage=resolution_stage,
                    stage=STAGE_FINE_GEO,
                )
                residual_group_sizes.extend(len(item) for item in fine_leaves)
        if np.all(resolution_stage[bucket] != STAGE_UNRESOLVED):
            fully_resolved_buckets += 1

    stage_counts = Counter(int(value) for value in resolution_stage)
    unresolved_pois = stage_counts[STAGE_UNRESOLVED]
    unresolved_groups = len(residual_group_sizes)
    collision_bucket_count = len(starts)
    resolved_pois = rows - unresolved_pois
    final_collision_distinct = resolved_pois + unresolved_groups
    collision_excess_before = rows - collision_bucket_count
    collision_excess_after = unresolved_pois - unresolved_groups
    relation_pair_counts = np.sum(relation_path_types >= 0, axis=1)
    geo_character_counts = coarse_geo_lengths.astype(np.int64) + fine_geo_lengths
    semantic_atom_counts = geo_character_counts + relation_pair_counts
    serialized_token_counts = geo_character_counts + 2 * relation_pair_counts
    full_count = rows if full_poi_count is None else int(full_poi_count)
    non_collision_count = full_count - rows
    final_full_distinct = non_collision_count + final_collision_distinct

    relation_emissions: Counter[str] = Counter()
    for relation_index, relation_type in enumerate(RELATION_TYPES):
        relation_emissions[relation_type] = int(
            np.count_nonzero(relation_path_types == relation_index)
        )

    def distribution(values: np.ndarray) -> dict[str, int]:
        return {
            str(int(value)): int(count)
            for value, count in sorted(Counter(values.tolist()).items())
        }

    metrics = {
        "configuration": {
            "city_scope": "北京市",
            "coarse_geohash_precision": coarse_precision,
            "fine_geohash_precision": fine_precision,
            "max_relation_pairs": max_relation_pairs,
            "stage_order": ["coarse_geo", "relation", "fine_geo"],
            "query_usage": "none",
            "relation_weighting": "POI equal weight static information gain",
            "missing_relation_branch_consumes_pair": False,
        },
        "catalog": {
            "full_poi_count": full_count,
            "collision_poi_count": rows,
            "collision_bucket_count": collision_bucket_count,
            "collision_excess_before": collision_excess_before,
        },
        "resolution": {
            "coarse_geo_resolved_poi_count": stage_counts[STAGE_COARSE_GEO],
            "coarse_geo_resolved_poi_ratio": _ratio(
                stage_counts[STAGE_COARSE_GEO], rows
            ),
            "relation_resolved_poi_count": stage_counts[STAGE_RELATION],
            "relation_resolved_poi_ratio": _ratio(
                stage_counts[STAGE_RELATION], rows
            ),
            "fine_geo_resolved_poi_count": stage_counts[STAGE_FINE_GEO],
            "fine_geo_resolved_poi_ratio": _ratio(
                stage_counts[STAGE_FINE_GEO], rows
            ),
            "semantic_resolved_poi_count": resolved_pois,
            "semantic_resolved_poi_ratio": _ratio(resolved_pois, rows),
            "unresolved_poi_count": unresolved_pois,
            "unresolved_poi_ratio": _ratio(unresolved_pois, rows),
            "unresolved_group_count": unresolved_groups,
            "max_unresolved_group_size": max(residual_group_sizes, default=0),
            "fully_resolved_collision_bucket_count": fully_resolved_buckets,
            "fully_resolved_collision_bucket_ratio": _ratio(
                fully_resolved_buckets, collision_bucket_count
            ),
        },
        "identifier": {
            "collision_distinct_identifier_count": final_collision_distinct,
            "collision_excess_after": collision_excess_after,
            "collision_excess_reduction_ratio": _ratio(
                collision_excess_before - collision_excess_after,
                collision_excess_before,
            ),
            "full_distinct_identifier_count": final_full_distinct,
            "full_distinct_identifier_ratio": _ratio(
                final_full_distinct, full_count
            ),
            "coarse_geo_length_distribution": distribution(coarse_geo_lengths),
            "relation_pair_length_distribution": distribution(
                relation_pair_counts
            ),
            "fine_geo_length_distribution": distribution(fine_geo_lengths),
            "semantic_atom_length_distribution": distribution(
                semantic_atom_counts
            ),
            "serialized_token_length_without_markers_distribution": distribution(
                serialized_token_counts
            ),
            "mean_semantic_atom_length": float(semantic_atom_counts.mean()),
            "mean_serialized_token_length_without_markers": float(
                serialized_token_counts.mean()
            ),
            "max_semantic_atom_length": int(semantic_atom_counts.max()),
            "max_serialized_token_length_without_markers": int(
                serialized_token_counts.max()
            ),
        },
        "geo": {
            "alphabet": GEOHASH_ALPHABET,
            "bucket_common_prefix_length_distribution": {
                str(key): value for key, value in sorted(bucket_root_prefixes.items())
            },
        },
        "relations": {
            "decision_node_count": relation_decisions,
            "emitted_pair_count": int(relation_pair_counts.sum()),
            "emitted_pair_count_by_type": dict(relation_emissions),
        },
    }
    return StructureCompilation(
        root_prefix_lengths=root_prefix_lengths,
        coarse_geo_codes=coarse_geo_codes,
        coarse_geo_lengths=coarse_geo_lengths,
        relation_path_types=relation_path_types,
        relation_path_values=relation_path_values,
        fine_geo_codes=fine_geo_codes,
        fine_geo_lengths=fine_geo_lengths,
        resolution_stage=resolution_stage,
        metrics=metrics,
    )


def _load_m2a_arrays(
    m2a_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    validate_proxy_output(m2a_dir)
    manifest = _load_json(m2a_dir / "manifest.json", "M2-A manifest")
    contracts = manifest.get("outputs", {}).get("arrays")
    if not isinstance(contracts, dict):
        raise GhrSidStructureError("M2-A manifest 缺少 arrays")
    arrays: dict[str, np.ndarray] = {}
    for name in ("collision_poi_ids", "base_sid_keys", "strict_relations"):
        contract = contracts.get(name)
        if not isinstance(contract, dict):
            raise GhrSidStructureError(f"M2-A 缺少数组：{name}")
        arrays[name] = np.load(
            m2a_dir / str(contract["path"]), mmap_mode="r", allow_pickle=False
        )
    return arrays, manifest


def _extract_collision_gid_codes(
    *,
    poi_dir: Path,
    gid_codes_path: Path,
    gid_manifest_path: Path,
    collision_poi_ids: np.ndarray,
    fine_precision: int,
    batch_rows: int,
    progress: Callable[[str], None] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stream raw POIs, validate frozen GID6, and extract collision GID-fine."""

    manifest = _load_json(gid_manifest_path, "GID manifest")
    if manifest.get("schema_version") != "geohash-pid-v1" or manifest.get(
        "status"
    ) != "completed":
        raise GhrSidStructureError("GID manifest 状态或 schema 不受支持")
    gid_contract = manifest.get("gid_codes")
    poi_ids_contract = manifest.get("poi_ids")
    if not isinstance(gid_contract, dict) or not isinstance(
        poi_ids_contract, dict
    ):
        raise GhrSidStructureError("GID manifest 缺少 gid_codes/poi_ids 契约")
    gid_codes = np.load(gid_codes_path, mmap_mode="r", allow_pickle=False)
    if gid_codes.ndim != 2 or gid_codes.shape[1] != 6:
        raise GhrSidStructureError("冻结 GID 必须是 [N,6]")
    if list(gid_codes.shape) != gid_contract.get("shape") or str(
        gid_codes.dtype
    ) != gid_contract.get("dtype"):
        raise GhrSidStructureError("GID codes shape/dtype 与 manifest 不一致")
    actual_gid_sha256 = sha256_file(gid_codes_path)
    expected_gid_sha256 = gid_contract.get("sha256")
    if expected_gid_sha256 and actual_gid_sha256 != expected_gid_sha256:
        raise GhrSidStructureError("GID codes SHA256 与 manifest 不一致")
    poi_ids_path = Path(str(poi_ids_contract.get("path", ""))).resolve()
    if sha256_file(poi_ids_path) != poi_ids_contract.get("sha256"):
        raise GhrSidStructureError("GID POI IDs SHA256 与 manifest 不一致")

    source_paths = tuple(sorted(poi_dir.glob("part-*.json")))
    if not source_paths:
        raise GhrSidStructureError(f"POI 目录没有 part-*.json：{poi_dir}")
    output = np.empty((len(collision_poi_ids), fine_precision), dtype=np.uint8)
    compact_row = 0
    scanned = 0
    prefix_mismatch_count = 0
    city_counts: Counter[str] = Counter()
    coordinate_minmax = [math.inf, -math.inf, math.inf, -math.inf]
    completed_sources: list[dict[str, Any]] = []
    batch_ids: list[int] = []
    batch_longitudes: list[float] = []
    batch_latitudes: list[float] = []

    def flush_batch() -> None:
        nonlocal compact_row, scanned, prefix_mismatch_count
        if not batch_ids:
            return
        encoded = encode_geohash_tokens(
            np.asarray(batch_longitudes, dtype=np.float64),
            np.asarray(batch_latitudes, dtype=np.float64),
            length=fine_precision,
            chunk_rows=batch_rows,
        )
        stop = scanned + len(batch_ids)
        mismatch = np.any(encoded[:, :6] != gid_codes[scanned:stop], axis=1)
        prefix_mismatch_count += int(np.count_nonzero(mismatch))
        for offset, poi_id in enumerate(batch_ids):
            if compact_row < len(collision_poi_ids) and poi_id == int(
                collision_poi_ids[compact_row]
            ):
                output[compact_row] = encoded[offset]
                compact_row += 1
        scanned = stop
        batch_ids.clear()
        batch_longitudes.clear()
        batch_latitudes.clear()

    with poi_ids_path.open("r", encoding="utf-8") as poi_id_handle:
        for source_path in source_paths:
            digest = hashlib.sha256()
            source_rows = 0
            with source_path.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    digest.update(line)
                    if not line.strip():
                        raise GhrSidStructureError(
                            f"POI 数据存在空行：{source_path.name}:{line_number}"
                        )
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise GhrSidStructureError(
                            f"POI JSON 解析失败：{source_path.name}:{line_number}"
                        ) from error
                    id_line = poi_id_handle.readline()
                    if not id_line:
                        raise GhrSidStructureError("POI IDs 早于原始 POI 结束")
                    try:
                        aligned_poi_id = str(json.loads(id_line))
                    except json.JSONDecodeError as error:
                        raise GhrSidStructureError("POI IDs JSON 解析失败") from error
                    raw_poi_id = str(record.get("poi_id", ""))
                    if raw_poi_id != aligned_poi_id:
                        raise GhrSidStructureError(
                            f"POI/GID 行映射错位：full_row={scanned + len(batch_ids)}"
                        )
                    try:
                        longitude = float(record["lng"])
                        latitude = float(record["lat"])
                        poi_id = int(raw_poi_id)
                    except (KeyError, TypeError, ValueError, OverflowError) as error:
                        raise GhrSidStructureError("POI 坐标或 poi_id 非法") from error
                    if not (
                        math.isfinite(longitude)
                        and math.isfinite(latitude)
                        and -180.0 <= longitude <= 180.0
                        and -90.0 <= latitude <= 90.0
                    ):
                        raise GhrSidStructureError("POI 坐标非 finite 或越界")
                    coordinate_minmax[0] = min(coordinate_minmax[0], longitude)
                    coordinate_minmax[1] = max(coordinate_minmax[1], longitude)
                    coordinate_minmax[2] = min(coordinate_minmax[2], latitude)
                    coordinate_minmax[3] = max(coordinate_minmax[3], latitude)
                    city_counts[str(record.get("city", ""))] += 1
                    batch_ids.append(poi_id)
                    batch_longitudes.append(longitude)
                    batch_latitudes.append(latitude)
                    source_rows += 1
                    if len(batch_ids) >= batch_rows:
                        flush_batch()
                        if progress is not None and scanned % (batch_rows * 8) == 0:
                            progress(
                                f"POI/GID 对齐 {scanned:,}/{len(gid_codes):,}；"
                                f"碰撞 GID{fine_precision} {compact_row:,}/"
                                f"{len(collision_poi_ids):,}"
                            )
            completed_sources.append(
                {
                    "path": str(source_path.resolve()),
                    "bytes": source_path.stat().st_size,
                    "rows": source_rows,
                    "sha256": digest.hexdigest(),
                }
            )
        flush_batch()
        if poi_id_handle.readline():
            raise GhrSidStructureError("POI IDs 行数多于原始 POI")

    if scanned != len(gid_codes):
        raise GhrSidStructureError(
            f"全量 POI 行数不守恒：{scanned} != {len(gid_codes)}"
        )
    if compact_row != len(collision_poi_ids):
        raise GhrSidStructureError(
            f"碰撞 POI 行数不守恒：{compact_row} != {len(collision_poi_ids)}"
        )
    if prefix_mismatch_count:
        raise GhrSidStructureError(
            f"新编码 GID 前六位与冻结产物有 {prefix_mismatch_count} 行不一致"
        )
    return output, {
        "gid_codes": {
            "path": str(gid_codes_path.resolve()),
            "sha256": actual_gid_sha256,
            "source_manifest_declares_sha256": bool(expected_gid_sha256),
            "shape": list(gid_codes.shape),
            "dtype": str(gid_codes.dtype),
        },
        "gid_manifest": {
            "path": str(gid_manifest_path.resolve()),
            "sha256": sha256_file(gid_manifest_path),
        },
        "gid_poi_ids": {
            "path": str(poi_ids_path),
            "sha256": poi_ids_contract.get("sha256"),
            "rows": scanned,
        },
        "raw_sources": completed_sources,
        "validation": {
            "full_row_count": scanned,
            "collision_row_count": compact_row,
            "gid6_prefix_mismatch_count": prefix_mismatch_count,
            "city_counts": dict(city_counts),
            "longitude_min": coordinate_minmax[0],
            "longitude_max": coordinate_minmax[1],
            "latitude_min": coordinate_minmax[2],
            "latitude_max": coordinate_minmax[3],
        },
    }


def _case_payloads(
    *,
    collision_poi_ids: np.ndarray,
    base_sid_keys: np.ndarray,
    gid_codes: np.ndarray,
    compilation: StructureCompilation,
    examples_per_stage: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for stage, stage_name in STAGE_NAMES.items():
        rows = np.flatnonzero(compilation.resolution_stage == stage)[
            :examples_per_stage
        ]
        for row in rows:
            relation_path = [
                {
                    "type": RELATION_TYPES[int(relation_type)],
                    "value": int(value),
                }
                for relation_type, value in zip(
                    compilation.relation_path_types[row],
                    compilation.relation_path_values[row],
                    strict=True,
                )
                if relation_type >= 0
            ]
            coarse_length = int(compilation.coarse_geo_lengths[row])
            fine_length = int(compilation.fine_geo_lengths[row])
            cases.append(
                {
                    "stage": stage_name,
                    "poi_id": str(int(collision_poi_ids[row])),
                    "base_sid_key": int(base_sid_keys[row]),
                    "gid": tokens_to_geohash(gid_codes[row]),
                    "bucket_common_prefix_length": int(
                        compilation.root_prefix_lengths[row]
                    ),
                    "coarse_geo_codes": [
                        int(value)
                        for value in compilation.coarse_geo_codes[
                            row, :coarse_length
                        ]
                    ],
                    "relations": relation_path,
                    "fine_geo_codes": [
                        int(value)
                        for value in compilation.fine_geo_codes[row, :fine_length]
                    ],
                }
            )
    return cases


def evaluate_structure_experiment(
    *,
    project_root: Path,
    poi_dir: Path,
    m2a_dir: Path,
    gid_codes_path: Path,
    gid_manifest_path: Path,
    output_dir: Path,
    coarse_precisions: Sequence[int] = (5, 6, 7),
    main_coarse_precision: int = 6,
    fine_precision: int = 8,
    max_relation_pairs: int = 3,
    batch_rows: int = 65_536,
    examples_per_stage: int = 20,
    progress: Callable[[str], None] | None = None,
) -> StructureExperimentResult:
    """Run the Beijing query-free GHR-SID structural experiment."""

    started_at = _utc_now()
    started = time.monotonic()
    project_root = project_root.resolve()
    poi_dir = poi_dir.resolve()
    m2a_dir = m2a_dir.resolve()
    gid_codes_path = gid_codes_path.resolve()
    gid_manifest_path = gid_manifest_path.resolve()
    output_dir = output_dir.resolve()
    precisions = tuple(sorted(set(int(value) for value in coarse_precisions)))
    if not precisions or main_coarse_precision not in precisions:
        raise GhrSidStructureError("主 coarse precision 必须包含在对照集合中")
    if any(value <= 0 or value >= fine_precision for value in precisions):
        raise GhrSidStructureError("所有 coarse precision 必须小于 fine precision")
    if batch_rows <= 0 or examples_per_stage < 0:
        raise GhrSidStructureError("batch_rows/examples_per_stage 非法")
    for path, name in (
        (project_root, "项目根目录"),
        (poi_dir, "POI 目录"),
        (m2a_dir, "M2-A 目录"),
    ):
        if not path.is_dir():
            raise GhrSidStructureError(f"{name}不存在：{path}")
    if output_dir.exists():
        raise GhrSidStructureError(f"输出目录已存在，拒绝覆盖：{output_dir}")

    arrays, m2a_manifest = _load_m2a_arrays(m2a_dir)
    collision_poi_ids = np.asarray(arrays["collision_poi_ids"])
    base_sid_keys = np.asarray(arrays["base_sid_keys"])
    strict_relations = np.asarray(arrays["strict_relations"])
    collision_gid_codes, geo_inputs = _extract_collision_gid_codes(
        poi_dir=poi_dir,
        gid_codes_path=gid_codes_path,
        gid_manifest_path=gid_manifest_path,
        collision_poi_ids=collision_poi_ids,
        fine_precision=fine_precision,
        batch_rows=batch_rows,
        progress=progress,
    )
    full_poi_count = int(geo_inputs["validation"]["full_row_count"])

    compilations: dict[int, StructureCompilation] = {}
    for precision in precisions:
        if progress is not None:
            progress(
                f"编译 G{precision} → Relation{max_relation_pairs} → "
                f"G{fine_precision} 静态结构"
            )
        compilations[precision] = compile_geo_relation_structure(
            base_sid_keys=base_sid_keys,
            gid_codes=collision_gid_codes,
            strict_relations=strict_relations,
            coarse_precision=precision,
            fine_precision=fine_precision,
            max_relation_pairs=max_relation_pairs,
            full_poi_count=full_poi_count,
        )
    main = compilations[main_coarse_precision]

    metrics = {
        "schema_version": STRUCTURE_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "method": "GHR-SID",
        "scope": "北京市全量 TIGER 碰撞 POI",
        "query_usage": "none",
        "main_protocol": f"G{main_coarse_precision}->R{max_relation_pairs}->G{fine_precision}",
        "main": main.metrics,
        "sensitivity": {
            f"G{precision}->R{max_relation_pairs}->G{fine_precision}": compilation.metrics
            for precision, compilation in compilations.items()
        },
        "decision": {
            "main_protocol_pre_registered": True,
            "main_coarse_precision": main_coarse_precision,
            "sensitivity_is_not_model_selection": True,
            "ready_for_final_identifier": False,
            "ready_for_sft": False,
            "next_gate": (
                "先核验结构覆盖、残留碰撞和数值序列化；本实验不以静态唯一率"
                "直接宣称下游提升"
            ),
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
    }

    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging_dir.exists():
        raise GhrSidStructureError(f"暂存目录已存在：{staging_dir}")
    staging_dir.mkdir(parents=True)
    outputs: dict[str, Any] = {"arrays": {}}
    output_arrays = {
        "collision_poi_ids": collision_poi_ids,
        "base_sid_keys": base_sid_keys,
        f"collision_gid{fine_precision}_codes": collision_gid_codes,
        "root_prefix_lengths": main.root_prefix_lengths,
        "coarse_geo_codes": main.coarse_geo_codes,
        "coarse_geo_lengths": main.coarse_geo_lengths,
        "relation_path_types": main.relation_path_types,
        "relation_path_values": main.relation_path_values,
        "fine_geo_codes": main.fine_geo_codes,
        "fine_geo_lengths": main.fine_geo_lengths,
        "resolution_stage": main.resolution_stage,
    }
    for name, array in output_arrays.items():
        outputs["arrays"][name] = _save_array(staging_dir, name, array)

    cases = _case_payloads(
        collision_poi_ids=collision_poi_ids,
        base_sid_keys=base_sid_keys,
        gid_codes=collision_gid_codes,
        compilation=main,
        examples_per_stage=examples_per_stage,
    )
    cases_path = staging_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    outputs["cases"] = {
        "path": cases_path.name,
        "rows": len(cases),
        "bytes": cases_path.stat().st_size,
        "sha256": sha256_file(cases_path),
    }
    metrics_path = staging_dir / "metrics.json"
    _json_dump(metrics_path, metrics)
    outputs["metrics"] = {
        "path": metrics_path.name,
        "bytes": metrics_path.stat().st_size,
        "sha256": sha256_file(metrics_path),
    }

    signature_payload = {
        "configuration": {
            "city_scope": "北京市",
            "coarse_precisions": list(precisions),
            "main_coarse_precision": main_coarse_precision,
            "fine_precision": fine_precision,
            "max_relation_pairs": max_relation_pairs,
            "query_usage": "none",
        },
        "inputs": {
            "m2a_manifest_sha256": sha256_file(m2a_dir / "manifest.json"),
            "gid_manifest_sha256": sha256_file(gid_manifest_path),
            "gid_codes_sha256": geo_inputs["gid_codes"]["sha256"],
        },
    }
    manifest = {
        "schema_version": STRUCTURE_SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at,
        "finished_at": metrics["finished_at"],
        "signature": _signature(signature_payload),
        "configuration": signature_payload["configuration"],
        "git": _git_state(project_root),
        "inputs": {
            "m2a_manifest": {
                "path": str((m2a_dir / "manifest.json").resolve()),
                "sha256": signature_payload["inputs"]["m2a_manifest_sha256"],
                "schema_version": m2a_manifest.get("schema_version"),
                "consumed_arrays": [
                    "collision_poi_ids",
                    "base_sid_keys",
                    "strict_relations",
                ],
                "query_or_order_arrays_consumed": False,
            },
            **geo_inputs,
        },
        "outputs": outputs,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "validation": {
            "query_inputs_consumed": False,
            "gid6_prefix_reproduced": True,
            "collision_row_count_conserved": True,
            "main_arrays_share_collision_row_order": True,
        },
    }
    _json_dump(staging_dir / "manifest.json", manifest)
    staging_dir.rename(output_dir)
    validated = validate_structure_output(output_dir)
    if validated["validated_output_count"] != len(output_arrays) + 2:
        raise GhrSidStructureError("完成后的输出验证计数不一致")
    return StructureExperimentResult(
        metrics=metrics, manifest=manifest, output_dir=output_dir
    )


def validate_structure_output(output_dir: Path) -> dict[str, Any]:
    """Validate every managed GHR-SID structural output."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "GHR-SID manifest")
    if manifest.get("schema_version") != STRUCTURE_SCHEMA_VERSION:
        raise GhrSidStructureError("GHR-SID manifest schema_version 不受支持")
    if manifest.get("status") != "completed":
        raise GhrSidStructureError("GHR-SID manifest 尚未 completed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise GhrSidStructureError("GHR-SID manifest 缺少 outputs")
    validated = 0
    arrays = outputs.get("arrays")
    if not isinstance(arrays, dict):
        raise GhrSidStructureError("GHR-SID manifest 缺少 arrays")
    for name, contract in arrays.items():
        if not isinstance(contract, dict):
            raise GhrSidStructureError(f"数组契约非法：{name}")
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file() or sha256_file(path) != contract.get("sha256"):
            raise GhrSidStructureError(f"数组缺失或 SHA 不一致：{name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != contract.get("shape") or str(
            array.dtype
        ) != contract.get("dtype"):
            raise GhrSidStructureError(f"数组 shape/dtype 不一致：{name}")
        if path.stat().st_size != contract.get("bytes"):
            raise GhrSidStructureError(f"数组字节数不一致：{name}")
        validated += 1
    for name in ("cases", "metrics"):
        contract = outputs.get(name)
        if not isinstance(contract, dict):
            raise GhrSidStructureError(f"缺少输出契约：{name}")
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file() or sha256_file(path) != contract.get("sha256"):
            raise GhrSidStructureError(f"输出缺失或 SHA 不一致：{name}")
        if path.stat().st_size != contract.get("bytes"):
            raise GhrSidStructureError(f"输出字节数不一致：{name}")
        validated += 1
    metrics = _load_json(output_dir / "metrics.json", "GHR-SID metrics")
    if metrics.get("schema_version") != STRUCTURE_METRICS_SCHEMA_VERSION:
        raise GhrSidStructureError("GHR-SID metrics schema_version 不受支持")
    return {
        "status": "validated",
        "validated_output_count": validated,
        "signature": manifest.get("signature"),
        "main_protocol": metrics.get("main_protocol"),
    }

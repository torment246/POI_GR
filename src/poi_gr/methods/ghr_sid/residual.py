"""Static residual audit for query-free GHR-SID collision groups."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from poi_gr.methods.ghr_sid.structure import validate_structure_output
from poi_gr.methods.qgr_sid.proxy import validate_proxy_output
from poi_gr.methods.qgr_sid.relations import RELATION_TYPES
from poi_gr.methods.tiger.identifier import sha256_file
from poi_gr.pid.geohash import encode_geohash_tokens, tokens_to_geohash


RESIDUAL_SCHEMA_VERSION = "ghr-sid-residual-audit-v1"
RESIDUAL_METRICS_SCHEMA_VERSION = "ghr-sid-residual-audit-metrics-v1"
EXPECTED_RESIDUAL_POI_COUNT = 75_427
EXPECTED_RESIDUAL_GROUP_COUNT = 34_714
EXPECTED_RESIDUAL_EXCESS = 40_713
EXPECTED_MAX_GROUP_SIZE = 62
RAW_TEXT_FIELDS = (
    "displayname",
    "address",
    "alias",
    "text",
    "category",
    "category_code",
    "area",
    "city",
    "layer",
)
FEATURE_ORDER = (
    "gid9",
    "gid10",
    "exact_coordinate",
    "displayname",
    "address",
    "alias",
    "category",
    "scope",
    "text",
)


class GhrSidResidualError(ValueError):
    """Raised when residual audit inputs or outputs violate the contract."""


@dataclass(frozen=True)
class ResidualGrouping:
    """Feature-irreducible rows in original collision-row order."""

    collision_rows: np.ndarray
    group_ids: np.ndarray
    group_sizes: np.ndarray
    member_order: np.ndarray
    group_offsets: np.ndarray


@dataclass(frozen=True)
class ResidualAuditResult:
    """Completed residual audit artifacts."""

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
        raise GhrSidResidualError(f"无法读取{name}：{path}：{error}") from error
    if not isinstance(payload, dict):
        raise GhrSidResidualError(f"{name}必须是 JSON object：{path}")
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


def normalize_static_text(value: Any) -> str:
    """Normalize static text for identity comparison, not SID serialization."""

    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if character.isalnum())


def _raw_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _bounded_text(value: Any, limit: int = 500) -> str:
    text = _raw_text(value)
    return text if len(text) <= limit else text[:limit] + "…"


def find_feature_irreducible_groups(
    *,
    base_sid_keys: np.ndarray,
    gid8_codes: np.ndarray,
    strict_relations: np.ndarray,
) -> ResidualGrouping:
    """Find groups identical on base SID, GID8 and every strict relation."""

    base_sid_keys = np.asarray(base_sid_keys)
    gid8_codes = np.asarray(gid8_codes)
    strict_relations = np.asarray(strict_relations)
    rows = len(base_sid_keys)
    if base_sid_keys.ndim != 1:
        raise GhrSidResidualError("base_sid_keys 必须是一维")
    if gid8_codes.shape != (rows, 8):
        raise GhrSidResidualError("gid8_codes 必须是 [N,8]")
    if strict_relations.shape != (rows, len(RELATION_TYPES)):
        raise GhrSidResidualError("strict_relations shape 不符合关系契约")
    if rows == 0:
        empty = np.empty(0, dtype=np.int64)
        return ResidualGrouping(empty, empty, empty, empty, np.asarray([0]))

    feature_matrix = np.ascontiguousarray(
        np.concatenate(
            [
                gid8_codes.astype(np.int16, copy=False),
                strict_relations.astype(np.int16, copy=False),
            ],
            axis=1,
        )
    )
    feature_bytes = feature_matrix.view(
        np.dtype((np.void, feature_matrix.dtype.itemsize * feature_matrix.shape[1]))
    ).reshape(-1)
    sort_keys = np.empty(
        rows,
        dtype=[("base_sid_key", "<i8"), ("features", feature_bytes.dtype)],
    )
    sort_keys["base_sid_key"] = base_sid_keys.astype(np.int64, copy=False)
    sort_keys["features"] = feature_bytes
    order = np.argsort(
        sort_keys,
        order=("base_sid_key", "features"),
        kind="stable",
    )
    sorted_keys = sort_keys[order]
    boundaries = np.ones(rows, dtype=bool)
    boundaries[1:] = (
        sorted_keys["base_sid_key"][1:] != sorted_keys["base_sid_key"][:-1]
    ) | (sorted_keys["features"][1:] != sorted_keys["features"][:-1])
    starts = np.flatnonzero(boundaries)
    ends = np.append(starts[1:], rows)
    sizes = ends - starts
    retained = np.flatnonzero(sizes > 1)

    group_by_collision_row = np.full(rows, -1, dtype=np.int32)
    group_sizes = sizes[retained].astype(np.int32, copy=False)
    for group_id, retained_index in enumerate(retained):
        start = int(starts[retained_index])
        end = int(ends[retained_index])
        group_by_collision_row[order[start:end]] = group_id
    collision_rows = np.flatnonzero(group_by_collision_row >= 0).astype(
        np.int64, copy=False
    )
    group_ids = group_by_collision_row[collision_rows]
    member_order = np.argsort(group_ids, kind="stable").astype(np.int64)
    group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(group_sizes, dtype=np.int64)]
    )
    if not np.array_equal(group_ids[member_order], np.repeat(
        np.arange(len(group_sizes), dtype=np.int32), group_sizes
    )):
        raise GhrSidResidualError("残留分组成员索引不守恒")
    return ResidualGrouping(
        collision_rows=collision_rows,
        group_ids=group_ids,
        group_sizes=group_sizes,
        member_order=member_order,
        group_offsets=group_offsets,
    )


def collision_partition_stats(
    group_ids: Sequence[int],
    feature_keys: Sequence[Hashable],
) -> dict[str, int]:
    """Measure collisions after splitting original groups by feature keys."""

    if len(group_ids) != len(feature_keys):
        raise GhrSidResidualError("group_ids 与 feature_keys 行数不一致")
    counts = Counter(
        (int(group_id), feature)
        for group_id, feature in zip(group_ids, feature_keys, strict=True)
    )
    collision_sizes = [count for count in counts.values() if count > 1]
    return {
        "collision_excess": int(sum(count - 1 for count in collision_sizes)),
        "collision_poi_count": int(sum(collision_sizes)),
        "collision_group_count": len(collision_sizes),
        "max_collision_group_size": max(collision_sizes, default=1),
        "distinct_count": len(counts),
    }


def _load_array(
    directory: Path,
    contract: Mapping[str, Any],
    name: str,
) -> np.ndarray:
    path = directory / str(contract.get("path", ""))
    if not path.is_file():
        raise GhrSidResidualError(f"输入数组不存在：{name}：{path}")
    if sha256_file(path) != contract.get("sha256"):
        raise GhrSidResidualError(f"输入数组 SHA256 不一致：{name}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != contract.get("shape") or str(
        array.dtype
    ) != contract.get("dtype"):
        raise GhrSidResidualError(f"输入数组 shape/dtype 不一致：{name}")
    return array


def _load_inputs(
    structure_dir: Path,
    m2a_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    validate_structure_output(structure_dir)
    validate_proxy_output(m2a_dir)
    structure_manifest = _load_json(
        structure_dir / "manifest.json", "GHR-SID structure manifest"
    )
    m2a_manifest = _load_json(m2a_dir / "manifest.json", "M2-A manifest")
    declared_m2a = structure_manifest.get("inputs", {}).get("m2a_manifest", {})
    if declared_m2a.get("sha256") != sha256_file(m2a_dir / "manifest.json"):
        raise GhrSidResidualError("结构实验与 M2-A manifest 不一致")
    structure_contracts = structure_manifest.get("outputs", {}).get("arrays")
    m2a_contracts = m2a_manifest.get("outputs", {}).get("arrays")
    if not isinstance(structure_contracts, dict) or not isinstance(
        m2a_contracts, dict
    ):
        raise GhrSidResidualError("输入 manifest 缺少数组契约")
    arrays = {
        "collision_poi_ids": _load_array(
            structure_dir,
            structure_contracts["collision_poi_ids"],
            "collision_poi_ids",
        ),
        "base_sid_keys": _load_array(
            structure_dir,
            structure_contracts["base_sid_keys"],
            "base_sid_keys",
        ),
        "gid8_codes": _load_array(
            structure_dir,
            structure_contracts["collision_gid8_codes"],
            "collision_gid8_codes",
        ),
        "resolution_stage": _load_array(
            structure_dir,
            structure_contracts["resolution_stage"],
            "resolution_stage",
        ),
        "strict_relations": _load_array(
            m2a_dir,
            m2a_contracts["strict_relations"],
            "strict_relations",
        ),
    }
    for name in ("collision_poi_ids", "base_sid_keys"):
        m2a_array = _load_array(m2a_dir, m2a_contracts[name], f"M2-A {name}")
        if arrays[name].shape != m2a_array.shape or not np.array_equal(
            arrays[name], m2a_array
        ):
            raise GhrSidResidualError(f"结构实验与 M2-A {name} 行映射不一致")
    return arrays, structure_manifest, m2a_manifest


def _stream_residual_records(
    *,
    poi_dir: Path,
    residual_poi_ids: np.ndarray,
    expected_sources: Sequence[Mapping[str, Any]],
    progress: Callable[[str], None] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    source_paths = tuple(sorted(poi_dir.glob("part-*.json")))
    expected_by_path = {
        str(Path(str(item["path"])).resolve()): item for item in expected_sources
    }
    if {str(path.resolve()) for path in source_paths} != set(expected_by_path):
        raise GhrSidResidualError("POI 原始分片集合与结构实验不一致")
    target_by_id = {
        int(poi_id): position for position, poi_id in enumerate(residual_poi_ids)
    }
    if len(target_by_id) != len(residual_poi_ids):
        raise GhrSidResidualError("残留 POI ID 存在重复")
    records: list[dict[str, Any] | None] = [None] * len(residual_poi_ids)
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
                    raise GhrSidResidualError(
                        f"POI 数据存在空行：{source_path.name}:{line_number}"
                    )
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise GhrSidResidualError(
                        f"POI JSON 解析失败：{source_path.name}:{line_number}"
                    ) from error
                if not isinstance(raw, dict):
                    raise GhrSidResidualError("POI JSON 行必须是 object")
                try:
                    poi_id = int(str(raw["poi_id"]))
                except (KeyError, TypeError, ValueError, OverflowError) as error:
                    raise GhrSidResidualError("POI ID 非法") from error
                position = target_by_id.get(poi_id)
                if position is not None:
                    if records[position] is not None:
                        raise GhrSidResidualError(f"残留 POI 重复出现：{poi_id}")
                    try:
                        longitude = float(raw["lng"])
                        latitude = float(raw["lat"])
                    except (KeyError, TypeError, ValueError, OverflowError) as error:
                        raise GhrSidResidualError("残留 POI 坐标非法") from error
                    if not (
                        math.isfinite(longitude)
                        and math.isfinite(latitude)
                        and -180 <= longitude <= 180
                        and -90 <= latitude <= 90
                    ):
                        raise GhrSidResidualError("残留 POI 坐标非 finite 或越界")
                    record = {
                        "poi_id": str(poi_id),
                        "lng": longitude,
                        "lat": latitude,
                    }
                    for field in RAW_TEXT_FIELDS:
                        record[field] = _raw_text(raw.get(field))
                    records[position] = record
                    found += 1
                source_rows += 1
                scanned += 1
        completed = {
            "path": str(source_path.resolve()),
            "bytes": source_path.stat().st_size,
            "rows": source_rows,
            "sha256": digest.hexdigest(),
        }
        expected = expected_by_path[completed["path"]]
        for key in ("bytes", "rows", "sha256"):
            if completed[key] != expected.get(key):
                raise GhrSidResidualError(
                    f"POI 源分片 {source_path.name} 的 {key} 与结构实验不一致"
                )
        completed_sources.append(completed)
        if progress is not None:
            progress(
                f"扫描原始 POI {scanned:,} 行；已回查残留 {found:,}/"
                f"{len(residual_poi_ids):,}"
            )
    if found != len(residual_poi_ids) or any(record is None for record in records):
        raise GhrSidResidualError(
            f"残留 POI 回查不完整：{found} != {len(residual_poi_ids)}"
        )
    return [record for record in records if record is not None], completed_sources, scanned


def _group_members(grouping: ResidualGrouping, group_id: int) -> np.ndarray:
    start = int(grouping.group_offsets[group_id])
    end = int(grouping.group_offsets[group_id + 1])
    return grouping.member_order[start:end]


def _feature_values(
    records: Sequence[Mapping[str, Any]],
    gid10_codes: np.ndarray,
) -> dict[str, list[Hashable]]:
    normalized = {
        field: [normalize_static_text(record[field]) for record in records]
        for field in RAW_TEXT_FIELDS
    }
    return {
        "gid9": [int(row[8]) for row in gid10_codes],
        "gid10": [(int(row[8]), int(row[9])) for row in gid10_codes],
        "exact_coordinate": [
            (float(record["lng"]), float(record["lat"])) for record in records
        ],
        "displayname": normalized["displayname"],
        "address": normalized["address"],
        "alias": normalized["alias"],
        "category": list(
            zip(normalized["category"], normalized["category_code"], strict=True)
        ),
        "scope": list(
            zip(
                normalized["area"],
                normalized["city"],
                normalized["layer"],
                strict=True,
            )
        ),
        "text": normalized["text"],
    }


def _stage_metrics(
    grouping: ResidualGrouping,
    features: Mapping[str, Sequence[Hashable]],
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[Hashable, ...]]]:
    baseline_excess = int(np.sum(grouping.group_sizes - 1))
    baseline = {
        "collision_excess": baseline_excess,
        "collision_poi_count": len(grouping.collision_rows),
        "collision_group_count": len(grouping.group_sizes),
        "max_collision_group_size": int(grouping.group_sizes.max(initial=1)),
        "distinct_count": len(grouping.group_sizes),
        "incremental_excess_reduction": 0,
        "cumulative_excess_reduction": 0,
        "cumulative_excess_reduction_ratio": 0.0,
    }
    cumulative: dict[str, Any] = {"gid8_all_relations": baseline}
    active_keys: list[tuple[Hashable, ...]] = [tuple() for _ in grouping.group_ids]
    previous_excess = baseline_excess
    for name in FEATURE_ORDER:
        active_keys = [
            key + (value,)
            for key, value in zip(active_keys, features[name], strict=True)
        ]
        stats = collision_partition_stats(grouping.group_ids, active_keys)
        stats["incremental_excess_reduction"] = (
            previous_excess - stats["collision_excess"]
        )
        stats["cumulative_excess_reduction"] = (
            baseline_excess - stats["collision_excess"]
        )
        stats["cumulative_excess_reduction_ratio"] = _ratio(
            stats["cumulative_excess_reduction"], baseline_excess
        )
        cumulative[name] = stats
        previous_excess = stats["collision_excess"]

    independent: dict[str, Any] = {}
    for name in FEATURE_ORDER:
        stats = collision_partition_stats(grouping.group_ids, features[name])
        stats["excess_reduction"] = baseline_excess - stats["collision_excess"]
        stats["excess_reduction_ratio"] = _ratio(
            stats["excess_reduction"], baseline_excess
        )
        independent[name] = stats
    return cumulative, independent, active_keys


def _diagnose_groups(
    *,
    grouping: ResidualGrouping,
    records: Sequence[Mapping[str, Any]],
    features: Mapping[str, Sequence[Hashable]],
    all_static_keys: Sequence[tuple[Hashable, ...]],
    base_sid_keys: np.ndarray,
    gid10_codes: np.ndarray,
    strict_relations: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_rows: list[dict[str, Any]] = []
    first_cue_groups: Counter[str] = Counter()
    first_cue_pois: Counter[str] = Counter()
    static_outcomes: Counter[str] = Counter()
    static_outcome_pois: Counter[str] = Counter()
    for group_id, group_size in enumerate(grouping.group_sizes):
        members = _group_members(grouping, group_id)
        unique_counts = {
            name: len({features[name][int(row)] for row in members})
            for name in FEATURE_ORDER
        }
        first_cue = next(
            (name for name in FEATURE_ORDER if unique_counts[name] > 1),
            "stable_metadata_identical",
        )
        final_counts = Counter(all_static_keys[int(row)] for row in members)
        final_collision_sizes = [count for count in final_counts.values() if count > 1]
        final_excess = sum(count - 1 for count in final_collision_sizes)
        initial_excess = int(group_size) - 1
        if final_excess == 0:
            static_outcome = "fully_resolved"
        elif final_excess == initial_excess:
            static_outcome = "unchanged"
        else:
            static_outcome = "partially_resolved"
        first_cue_groups[first_cue] += 1
        first_cue_pois[first_cue] += int(group_size)
        static_outcomes[static_outcome] += 1
        static_outcome_pois[static_outcome] += int(group_size)
        longitudes = [float(records[int(row)]["lng"]) for row in members]
        latitudes = [float(records[int(row)]["lat"]) for row in members]
        mean_latitude = sum(latitudes) / len(latitudes)
        longitude_span_m = (
            (max(longitudes) - min(longitudes))
            * 111_320.0
            * math.cos(math.radians(mean_latitude))
        )
        latitude_span_m = (max(latitudes) - min(latitudes)) * 111_320.0
        first_member = int(members[0])
        collision_row = int(grouping.collision_rows[first_member])
        names = list(
            dict.fromkeys(_bounded_text(records[int(row)]["displayname"]) for row in members)
        )[:8]
        addresses = list(
            dict.fromkeys(_bounded_text(records[int(row)]["address"]) for row in members)
        )[:8]
        relation_values = {
            relation_type: int(value)
            for relation_type, value in zip(
                RELATION_TYPES,
                strict_relations[collision_row],
                strict=True,
            )
            if int(value) >= 0
        }
        group_rows.append(
            {
                "group_id": group_id,
                "group_size": int(group_size),
                "base_sid_key": int(base_sid_keys[collision_row]),
                "gid8": tokens_to_geohash(gid10_codes[first_member, :8]),
                "relation_values_json": json.dumps(
                    relation_values, ensure_ascii=False, sort_keys=True
                ),
                "first_distinguishing_cue": first_cue,
                "static_outcome": static_outcome,
                "final_collision_excess": final_excess,
                "final_collision_poi_count": int(sum(final_collision_sizes)),
                "final_collision_group_count": len(final_collision_sizes),
                "max_final_collision_group_size": max(final_collision_sizes, default=1),
                "longitude_span_m": longitude_span_m,
                "latitude_span_m": latitude_span_m,
                "representative_names_json": json.dumps(names, ensure_ascii=False),
                "representative_addresses_json": json.dumps(
                    addresses, ensure_ascii=False
                ),
                **{f"unique_{name}_count": unique_counts[name] for name in FEATURE_ORDER},
            }
        )
    metrics = {
        "first_distinguishing_cue": {
            name: {
                "group_count": first_cue_groups[name],
                "poi_count": first_cue_pois[name],
            }
            for name in (*FEATURE_ORDER, "stable_metadata_identical")
        },
        "all_static_outcome": {
            name: {
                "group_count": static_outcomes[name],
                "poi_count": static_outcome_pois[name],
            }
            for name in ("fully_resolved", "partially_resolved", "unchanged")
        },
    }
    return group_rows, metrics


def _poi_rows(
    *,
    grouping: ResidualGrouping,
    records: Sequence[Mapping[str, Any]],
    base_sid_keys: np.ndarray,
    gid10_codes: np.ndarray,
    strict_relations: np.ndarray,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for residual_row, record in enumerate(records):
        collision_row = int(grouping.collision_rows[residual_row])
        row = {
            "group_id": int(grouping.group_ids[residual_row]),
            "group_size": int(grouping.group_sizes[grouping.group_ids[residual_row]]),
            "collision_row": collision_row,
            "poi_id": record["poi_id"],
            "base_sid_key": int(base_sid_keys[collision_row]),
            "gid8": tokens_to_geohash(gid10_codes[residual_row, :8]),
            "gid9": tokens_to_geohash(gid10_codes[residual_row, :9]),
            "gid10": tokens_to_geohash(gid10_codes[residual_row]),
            "lng": float(record["lng"]),
            "lat": float(record["lat"]),
        }
        for field in RAW_TEXT_FIELDS:
            row[field] = record[field]
            row[f"normalized_{field}"] = normalize_static_text(record[field])
        for relation_index, relation_type in enumerate(RELATION_TYPES):
            row[relation_type] = int(strict_relations[collision_row, relation_index])
        output.append(row)
    return output


def _case_payloads(
    *,
    grouping: ResidualGrouping,
    records: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    all_static_keys: Sequence[tuple[Hashable, ...]],
    gid10_codes: np.ndarray,
    strict_relations: np.ndarray,
    cases_per_type: int,
    pois_per_case: int,
) -> list[dict[str, Any]]:
    selected: list[tuple[str, int, np.ndarray]] = []
    for cue in (*FEATURE_ORDER, "stable_metadata_identical"):
        candidates = [
            row for row in group_rows if row["first_distinguishing_cue"] == cue
        ]
        candidates.sort(key=lambda row: (-int(row["group_size"]), int(row["group_id"])))
        for row in candidates[:cases_per_type]:
            group_id = int(row["group_id"])
            selected.append((f"first_cue:{cue}", group_id, _group_members(grouping, group_id)))

    duplicate_candidates: list[tuple[int, int, tuple[Hashable, ...], np.ndarray]] = []
    for group_id in range(len(grouping.group_sizes)):
        members = _group_members(grouping, group_id)
        by_signature: dict[tuple[Hashable, ...], list[int]] = {}
        for member in members:
            by_signature.setdefault(all_static_keys[int(member)], []).append(int(member))
        for signature, duplicate_members in by_signature.items():
            if len(duplicate_members) > 1:
                duplicate_candidates.append(
                    (
                        len(duplicate_members),
                        group_id,
                        signature,
                        np.asarray(duplicate_members, dtype=np.int64),
                    )
                )
    duplicate_candidates.sort(key=lambda item: (-item[0], item[1], repr(item[2])))
    for _, group_id, _, members in duplicate_candidates[:cases_per_type]:
        selected.append(("remaining_static_duplicate", group_id, members))

    cases: list[dict[str, Any]] = []
    for case_type, group_id, members in selected:
        first_member = int(members[0])
        collision_row = int(grouping.collision_rows[first_member])
        relation_values = {
            relation_type: int(value)
            for relation_type, value in zip(
                RELATION_TYPES,
                strict_relations[collision_row],
                strict=True,
            )
            if int(value) >= 0
        }
        poi_payloads = []
        for member in members[:pois_per_case]:
            record = records[int(member)]
            poi_payloads.append(
                {
                    "poi_id": record["poi_id"],
                    "displayname": _bounded_text(record["displayname"]),
                    "address": _bounded_text(record["address"]),
                    "alias": _bounded_text(record["alias"]),
                    "category": _bounded_text(record["category"]),
                    "category_code": _bounded_text(record["category_code"]),
                    "area": _bounded_text(record["area"]),
                    "city": _bounded_text(record["city"]),
                    "layer": _bounded_text(record["layer"]),
                    "lng": record["lng"],
                    "lat": record["lat"],
                    "gid8": tokens_to_geohash(gid10_codes[int(member), :8]),
                    "gid9": tokens_to_geohash(gid10_codes[int(member), :9]),
                    "gid10": tokens_to_geohash(gid10_codes[int(member)]),
                }
            )
        cases.append(
            {
                "case_type": case_type,
                "group_id": group_id,
                "group_size": int(grouping.group_sizes[group_id]),
                "selected_cluster_size": len(members),
                "shown_poi_count": len(poi_payloads),
                "relation_values": relation_values,
                "pois": poi_payloads,
            }
        )
    return cases


def _write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    chunk_rows: int,
) -> None:
    if not rows:
        raise GhrSidResidualError(f"拒绝写出空 Parquet：{path.name}")
    schema = pa.Table.from_pylist(list(rows[:1])).schema
    writer = pq.ParquetWriter(
        path,
        schema,
        version="2.6",
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_statistics=True,
    )
    try:
        for start in range(0, len(rows), chunk_rows):
            table = pa.Table.from_pylist(
                list(rows[start : start + chunk_rows]), schema=schema
            )
            writer.write_table(table)
    finally:
        writer.close()


def _file_contract(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        contract["rows"] = rows
    return contract


def audit_residual_pois(
    *,
    project_root: Path,
    poi_dir: Path,
    structure_dir: Path,
    m2a_dir: Path,
    output_dir: Path,
    experiment_id: str,
    batch_rows: int = 65_536,
    cases_per_type: int = 10,
    pois_per_case: int = 8,
    enforce_frozen_counts: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ResidualAuditResult:
    """Audit why GID8 plus all strict relations cannot split residual POIs."""

    started_at = _utc_now()
    started = time.monotonic()
    project_root = project_root.resolve()
    poi_dir = poi_dir.resolve()
    structure_dir = structure_dir.resolve()
    m2a_dir = m2a_dir.resolve()
    output_dir = output_dir.resolve()
    if batch_rows <= 0 or cases_per_type < 0 or pois_per_case <= 0:
        raise GhrSidResidualError("batch_rows/cases 参数非法")
    if not re.fullmatch(r"EXP-\d{8}-\d{2}", experiment_id):
        raise GhrSidResidualError("experiment_id 必须形如 EXP-YYYYMMDD-NN")
    for path, name in (
        (project_root, "项目根目录"),
        (poi_dir, "POI 目录"),
        (structure_dir, "结构实验目录"),
        (m2a_dir, "M2-A 目录"),
    ):
        if not path.is_dir():
            raise GhrSidResidualError(f"{name}不存在：{path}")
    if output_dir.exists():
        raise GhrSidResidualError(f"输出目录已存在，拒绝覆盖：{output_dir}")

    arrays, structure_manifest, m2a_manifest = _load_inputs(
        structure_dir, m2a_dir
    )
    if progress is not None:
        progress("定位 base SID + GID8 + 全 17 类关系的同签名残留组")
    grouping = find_feature_irreducible_groups(
        base_sid_keys=arrays["base_sid_keys"],
        gid8_codes=arrays["gid8_codes"],
        strict_relations=arrays["strict_relations"],
    )
    residual_excess = int(np.sum(grouping.group_sizes - 1))
    frozen_counts = {
        "poi_count": len(grouping.collision_rows),
        "group_count": len(grouping.group_sizes),
        "collision_excess": residual_excess,
        "max_group_size": int(grouping.group_sizes.max(initial=1)),
    }
    expected_counts = {
        "poi_count": EXPECTED_RESIDUAL_POI_COUNT,
        "group_count": EXPECTED_RESIDUAL_GROUP_COUNT,
        "collision_excess": EXPECTED_RESIDUAL_EXCESS,
        "max_group_size": EXPECTED_MAX_GROUP_SIZE,
    }
    if enforce_frozen_counts and frozen_counts != expected_counts:
        raise GhrSidResidualError(
            f"冻结残留计数变化：实际 {frozen_counts}，期望 {expected_counts}"
        )
    current_unresolved = np.asarray(arrays["resolution_stage"]) == 0
    if not np.all(current_unresolved[grouping.collision_rows]):
        raise GhrSidResidualError("全关系下界残留不是当前编译器残留的子集")

    residual_poi_ids = np.asarray(arrays["collision_poi_ids"])[
        grouping.collision_rows
    ]
    expected_sources = structure_manifest.get("inputs", {}).get("raw_sources")
    if not isinstance(expected_sources, list):
        raise GhrSidResidualError("结构 manifest 缺少 raw_sources")
    records, raw_sources, full_poi_count = _stream_residual_records(
        poi_dir=poi_dir,
        residual_poi_ids=residual_poi_ids,
        expected_sources=expected_sources,
        progress=progress,
    )
    longitudes = np.asarray([record["lng"] for record in records], dtype=np.float64)
    latitudes = np.asarray([record["lat"] for record in records], dtype=np.float64)
    gid10_codes = encode_geohash_tokens(
        longitudes,
        latitudes,
        length=10,
        chunk_rows=batch_rows,
    )
    frozen_gid8 = np.asarray(arrays["gid8_codes"])[grouping.collision_rows]
    gid8_mismatch_count = int(
        np.count_nonzero(np.any(gid10_codes[:, :8] != frozen_gid8, axis=1))
    )
    if gid8_mismatch_count:
        raise GhrSidResidualError(
            f"回查坐标与冻结 GID8 有 {gid8_mismatch_count} 行不一致"
        )

    features = _feature_values(records, gid10_codes)
    cumulative, independent, all_static_keys = _stage_metrics(grouping, features)
    group_rows, diagnostic_metrics = _diagnose_groups(
        grouping=grouping,
        records=records,
        features=features,
        all_static_keys=all_static_keys,
        base_sid_keys=arrays["base_sid_keys"],
        gid10_codes=gid10_codes,
        strict_relations=arrays["strict_relations"],
    )
    poi_rows = _poi_rows(
        grouping=grouping,
        records=records,
        base_sid_keys=arrays["base_sid_keys"],
        gid10_codes=gid10_codes,
        strict_relations=arrays["strict_relations"],
    )
    cases = _case_payloads(
        grouping=grouping,
        records=records,
        group_rows=group_rows,
        all_static_keys=all_static_keys,
        gid10_codes=gid10_codes,
        strict_relations=arrays["strict_relations"],
        cases_per_type=cases_per_type,
        pois_per_case=pois_per_case,
    )
    field_completeness = {
        field: {
            "non_empty_count": sum(
                bool(normalize_static_text(record[field])) for record in records
            ),
            "non_empty_ratio": _ratio(
                sum(bool(normalize_static_text(record[field])) for record in records),
                len(records),
            ),
        }
        for field in RAW_TEXT_FIELDS
    }
    current_metrics = _load_json(structure_dir / "metrics.json", "结构 metrics")
    current_resolution = current_metrics.get("main", {}).get("resolution", {})
    current_identifier = current_metrics.get("main", {}).get("identifier", {})
    all_static_final = cumulative[FEATURE_ORDER[-1]]
    metrics = {
        "schema_version": RESIDUAL_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "experiment_id": experiment_id,
        "method": "GHR-SID residual audit",
        "scope": "北京市冻结 POI 目录的 TIGER 碰撞桶",
        "query_usage": "none",
        "residual_definition": "base_sid + GID8 + all_17_strict_relations",
        "current_compiler_context": {
            "protocol": current_metrics.get("main_protocol"),
            "unresolved_poi_count": current_resolution.get("unresolved_poi_count"),
            "unresolved_group_count": current_resolution.get("unresolved_group_count"),
            "collision_excess_after": current_identifier.get("collision_excess_after"),
            "feature_irreducible_is_subset": True,
            "compiler_only_unresolved_poi_count": int(
                np.count_nonzero(current_unresolved) - len(grouping.collision_rows)
            ),
            "compiler_gap_collision_excess": int(
                current_identifier.get("collision_excess_after", 0) - residual_excess
            ),
        },
        "feature_irreducible": frozen_counts,
        "cumulative_cascade": cumulative,
        "independent_feature_split": independent,
        "group_diagnostics": diagnostic_metrics,
        "field_completeness": field_completeness,
        "all_static_final": {
            **all_static_final,
            "full_catalog_distinct_count": full_poi_count
            - all_static_final["collision_excess"],
            "full_catalog_distinct_ratio": _ratio(
                full_poi_count - all_static_final["collision_excess"], full_poi_count
            ),
        },
        "interpretation_contract": {
            "query_or_order_consumed": False,
            "click_score_consumed": False,
            "source_dt_consumed": False,
            "normalization": "NFKC + casefold + retain alphanumeric characters",
            "cascade_order_is_diagnostic_not_final_sid_order": True,
            "static_identity_does_not_prove_query_predictability": True,
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
    }

    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging_dir.exists():
        raise GhrSidResidualError(f"暂存目录已存在：{staging_dir}")
    staging_dir.mkdir(parents=True)
    poi_path = staging_dir / "residual_pois.parquet"
    group_path = staging_dir / "residual_groups.parquet"
    cases_path = staging_dir / "cases.jsonl"
    metrics_path = staging_dir / "metrics.json"
    _write_parquet(poi_path, poi_rows, chunk_rows=batch_rows)
    _write_parquet(group_path, group_rows, chunk_rows=batch_rows)
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    _json_dump(metrics_path, metrics)
    outputs = {
        "residual_pois": _file_contract(poi_path, rows=len(poi_rows)),
        "residual_groups": _file_contract(group_path, rows=len(group_rows)),
        "cases": _file_contract(cases_path, rows=len(cases)),
        "metrics": _file_contract(metrics_path),
    }
    signature_payload = {
        "experiment_id": experiment_id,
        "configuration": {
            "residual_definition": metrics["residual_definition"],
            "geohash_audit_precision": 10,
            "feature_order": list(FEATURE_ORDER),
            "query_usage": "none",
        },
        "inputs": {
            "structure_manifest_sha256": sha256_file(
                structure_dir / "manifest.json"
            ),
            "m2a_manifest_sha256": sha256_file(m2a_dir / "manifest.json"),
        },
    }
    manifest = {
        "schema_version": RESIDUAL_SCHEMA_VERSION,
        "status": "completed",
        "experiment_id": experiment_id,
        "started_at": started_at,
        "finished_at": metrics["finished_at"],
        "signature": _signature(signature_payload),
        "configuration": signature_payload["configuration"],
        "git": _git_state(project_root),
        "inputs": {
            "structure_manifest": {
                "path": str((structure_dir / "manifest.json").resolve()),
                "sha256": signature_payload["inputs"]["structure_manifest_sha256"],
                "schema_version": structure_manifest.get("schema_version"),
            },
            "m2a_manifest": {
                "path": str((m2a_dir / "manifest.json").resolve()),
                "sha256": signature_payload["inputs"]["m2a_manifest_sha256"],
                "schema_version": m2a_manifest.get("schema_version"),
            },
            "raw_sources": raw_sources,
        },
        "outputs": outputs,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "validation": {
            "query_inputs_consumed": False,
            "frozen_counts_match": frozen_counts == expected_counts,
            "raw_sources_match_structure_manifest": True,
            "gid8_prefix_reproduced": True,
            "all_residual_pois_found_exactly_once": True,
        },
    }
    _json_dump(staging_dir / "manifest.json", manifest)
    staging_dir.rename(output_dir)
    validated = validate_residual_output(output_dir)
    if validated["validated_output_count"] != len(outputs):
        raise GhrSidResidualError("完成后的输出验证计数不一致")
    return ResidualAuditResult(metrics=metrics, manifest=manifest, output_dir=output_dir)


def validate_residual_output(output_dir: Path) -> dict[str, Any]:
    """Validate every managed residual audit artifact."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "残留审计 manifest")
    if manifest.get("schema_version") != RESIDUAL_SCHEMA_VERSION:
        raise GhrSidResidualError("残留审计 manifest schema_version 不受支持")
    if manifest.get("status") != "completed":
        raise GhrSidResidualError("残留审计 manifest 尚未 completed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise GhrSidResidualError("残留审计 manifest 缺少 outputs")
    validated = 0
    for name, contract in outputs.items():
        if not isinstance(contract, dict):
            raise GhrSidResidualError(f"输出契约非法：{name}")
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file() or sha256_file(path) != contract.get("sha256"):
            raise GhrSidResidualError(f"输出缺失或 SHA 不一致：{name}")
        if path.stat().st_size != contract.get("bytes"):
            raise GhrSidResidualError(f"输出字节数不一致：{name}")
        expected_rows = contract.get("rows")
        if expected_rows is not None:
            if path.suffix == ".parquet":
                actual_rows = pq.ParquetFile(path).metadata.num_rows
            else:
                with path.open("r", encoding="utf-8") as handle:
                    actual_rows = sum(1 for _ in handle)
            if actual_rows != expected_rows:
                raise GhrSidResidualError(f"输出行数不一致：{name}")
        validated += 1
    metrics = _load_json(output_dir / "metrics.json", "残留审计 metrics")
    if metrics.get("schema_version") != RESIDUAL_METRICS_SCHEMA_VERSION:
        raise GhrSidResidualError("残留审计 metrics schema_version 不受支持")
    return {
        "status": "validated",
        "validated_output_count": validated,
        "experiment_id": manifest.get("experiment_id"),
        "signature": manifest.get("signature"),
    }

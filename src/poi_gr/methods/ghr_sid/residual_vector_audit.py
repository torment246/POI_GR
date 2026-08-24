"""Exact BGE-vector and fine-GID audit for EXP-09 residual collisions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
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
    _ratio,
    _signature,
    _utc_now,
)
from poi_gr.methods.ghr_sid.g6_entity_structure import (
    validate_g6_entity_structure_output,
)
from poi_gr.methods.ghr_sid.residual import normalize_static_text
from poi_gr.methods.ghr_sid.structure import validate_structure_output
from poi_gr.methods.tiger.identifier import sha256_file
from poi_gr.pid.geohash import encode_geohash_tokens, tokens_to_geohash


VECTOR_AUDIT_SCHEMA_VERSION = "ghr-sid-residual-vector-gid-audit-v1"
VECTOR_AUDIT_METRICS_SCHEMA_VERSION = (
    "ghr-sid-residual-vector-gid-audit-metrics-v1"
)
EXPECTED_RESIDUAL_POI_COUNT = 59_601
EXPECTED_RESIDUAL_GROUP_COUNT = 27_132
EXPECTED_RESIDUAL_EXCESS = 32_469
EXPECTED_MAX_GROUP_SIZE = 62
GID_LENGTHS = tuple(range(7, 13))
STABLE_FIELDS = (
    "displayname",
    "address",
    "alias",
    "category",
    "category_code",
    "area",
    "city",
    "layer",
    "text",
)


class ResidualVectorAuditError(ValueError):
    """Raised when vector/GID audit inputs violate the frozen contract."""


@dataclass(frozen=True)
class ResidualVectorAuditResult:
    """Completed exact-vector and fine-geography audit."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


def collision_partition_stats(
    group_ids: Sequence[int], feature_keys: Sequence[Hashable]
) -> dict[str, int]:
    """Measure collisions after splitting fixed residual groups by keys."""

    if len(group_ids) != len(feature_keys):
        raise ResidualVectorAuditError("group_ids 与 feature_keys 行数不一致")
    counts = Counter(
        (int(group_id), key)
        for group_id, key in zip(group_ids, feature_keys, strict=True)
    )
    collisions = [count for count in counts.values() if count > 1]
    return {
        "collision_excess": int(sum(count - 1 for count in collisions)),
        "collision_poi_count": int(sum(collisions)),
        "collision_group_count": len(collisions),
        "max_collision_group_size": max(collisions, default=1),
        "distinct_count": len(counts),
    }


def _with_reduction(
    stats: Mapping[str, int], *, baseline_excess: int, previous_excess: int
) -> dict[str, int | float]:
    excess = int(stats["collision_excess"])
    return {
        **stats,
        "incremental_excess_reduction": previous_excess - excess,
        "cumulative_excess_reduction": baseline_excess - excess,
        "cumulative_excess_reduction_ratio": _ratio(
            baseline_excess - excess, baseline_excess
        ),
    }


def build_partition_cascade(
    *,
    group_ids: np.ndarray,
    embedding_keys: Sequence[Hashable],
    gid_keys: Mapping[int, Sequence[Hashable]],
    coordinate_keys: Sequence[Hashable],
) -> dict[str, dict[str, dict[str, int | float]]]:
    """Build GID-only and exact-vector-then-GID collision cascades."""

    baseline = collision_partition_stats(group_ids, [0] * len(group_ids))
    baseline_excess = int(baseline["collision_excess"])
    gid_only: dict[str, dict[str, int | float]] = {
        "gid6": _with_reduction(
            baseline,
            baseline_excess=baseline_excess,
            previous_excess=baseline_excess,
        )
    }
    previous = baseline_excess
    for length in GID_LENGTHS:
        stats = collision_partition_stats(group_ids, gid_keys[length])
        gid_only[f"gid{length}"] = _with_reduction(
            stats,
            baseline_excess=baseline_excess,
            previous_excess=previous,
        )
        previous = int(stats["collision_excess"])
    coordinate_stats = collision_partition_stats(group_ids, coordinate_keys)
    gid_only["exact_coordinate"] = _with_reduction(
        coordinate_stats,
        baseline_excess=baseline_excess,
        previous_excess=previous,
    )

    vector_stats = collision_partition_stats(group_ids, embedding_keys)
    vector_then_gid: dict[str, dict[str, int | float]] = {
        "exact_embedding": _with_reduction(
            vector_stats,
            baseline_excess=baseline_excess,
            previous_excess=baseline_excess,
        )
    }
    previous = int(vector_stats["collision_excess"])
    for length in GID_LENGTHS:
        keys = list(zip(embedding_keys, gid_keys[length], strict=True))
        stats = collision_partition_stats(group_ids, keys)
        vector_then_gid[f"exact_embedding_gid{length}"] = _with_reduction(
            stats,
            baseline_excess=baseline_excess,
            previous_excess=previous,
        )
        previous = int(stats["collision_excess"])
    keys = list(zip(embedding_keys, coordinate_keys, strict=True))
    stats = collision_partition_stats(group_ids, keys)
    vector_then_gid["exact_embedding_coordinate"] = _with_reduction(
        stats,
        baseline_excess=baseline_excess,
        previous_excess=previous,
    )
    return {
        "gid_only": gid_only,
        "exact_embedding_then_geography": vector_then_gid,
    }


def minimum_gid_labels_for_duplicate_vectors(
    *,
    group_ids: Sequence[int],
    embedding_keys: Sequence[Hashable],
    gid_keys: Mapping[int, Sequence[Hashable]],
    coordinate_keys: Sequence[Hashable],
) -> tuple[list[str], dict[str, int]]:
    """Find the first geographic precision that isolates each duplicate vector."""

    vector_counts = Counter(zip(group_ids, embedding_keys, strict=True))
    gid_counts = {
        length: Counter(
            zip(group_ids, embedding_keys, gid_keys[length], strict=True)
        )
        for length in GID_LENGTHS
    }
    coordinate_counts = Counter(
        zip(group_ids, embedding_keys, coordinate_keys, strict=True)
    )
    labels: list[str] = []
    for row, (group_id, embedding_key) in enumerate(
        zip(group_ids, embedding_keys, strict=True)
    ):
        if vector_counts[(group_id, embedding_key)] == 1:
            labels.append("not_duplicate_vector")
            continue
        label = ""
        for length in GID_LENGTHS:
            key = (group_id, embedding_key, gid_keys[length][row])
            if gid_counts[length][key] == 1:
                label = f"gid{length}"
                break
        if not label:
            coordinate_key = (group_id, embedding_key, coordinate_keys[row])
            label = (
                "exact_coordinate"
                if coordinate_counts[coordinate_key] == 1
                else "same_coordinate_unresolved"
            )
        labels.append(label)
    return labels, dict(sorted(Counter(labels).items()))


def _load_array(
    directory: Path, contract: Mapping[str, Any], name: str
) -> np.ndarray:
    path = directory / str(contract.get("path", ""))
    if not path.is_file() or sha256_file(path) != contract.get("sha256"):
        raise ResidualVectorAuditError(f"输入数组缺失或 SHA256 不一致：{name}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != contract.get("shape") or str(
        array.dtype
    ) != contract.get("dtype"):
        raise ResidualVectorAuditError(f"输入数组 shape/dtype 不一致：{name}")
    return array


def _load_residual_inputs(
    experiment_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    validate_g6_entity_structure_output(experiment_dir)
    manifest = _load_json(experiment_dir / "manifest.json", "EXP-09 manifest")
    contracts = manifest.get("outputs", {}).get("arrays")
    if not isinstance(contracts, dict):
        raise ResidualVectorAuditError("EXP-09 manifest 缺少数组契约")
    arrays = {
        name: _load_array(experiment_dir, contracts[name], name)
        for name in (
            "post_g6_collision_rows",
            "post_g6_group_ids",
            "post_g6_poi_ids",
            "relation_path_types",
            "relation_resolved",
        )
    }
    unresolved = np.asarray(arrays["relation_resolved"]) == 0
    if np.any(np.asarray(arrays["relation_path_types"])[unresolved] >= 0):
        raise ResidualVectorAuditError("EXP-09 残留 POI 出现非空关系路径")
    group_ids = np.asarray(arrays["post_g6_group_ids"])[unresolved]
    group_counts = Counter(int(value) for value in group_ids)
    counts = {
        "poi_count": int(np.count_nonzero(unresolved)),
        "group_count": len(group_counts),
        "collision_excess": int(sum(value - 1 for value in group_counts.values())),
        "max_group_size": max(group_counts.values(), default=1),
    }
    expected = {
        "poi_count": EXPECTED_RESIDUAL_POI_COUNT,
        "group_count": EXPECTED_RESIDUAL_GROUP_COUNT,
        "collision_excess": EXPECTED_RESIDUAL_EXCESS,
        "max_group_size": EXPECTED_MAX_GROUP_SIZE,
    }
    if counts != expected:
        raise ResidualVectorAuditError(
            f"EXP-09 残留计数变化：实际 {counts}，期望 {expected}"
        )
    arrays["unresolved"] = unresolved
    return arrays, manifest


def _raw_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _stream_records(
    *,
    poi_dir: Path,
    target_poi_ids: np.ndarray,
    expected_sources: Sequence[Mapping[str, Any]],
    embedding_poi_ids_path: Path,
    progress: Callable[[str], None] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    target_by_id = {
        int(poi_id): position for position, poi_id in enumerate(target_poi_ids)
    }
    if len(target_by_id) != len(target_poi_ids):
        raise ResidualVectorAuditError("残留 POI ID 不唯一")
    source_paths = tuple(sorted(poi_dir.glob("part-*.json")))
    expected_by_path = {
        str(Path(str(item["path"])).resolve()): item for item in expected_sources
    }
    if {str(path.resolve()) for path in source_paths} != set(expected_by_path):
        raise ResidualVectorAuditError("原始 POI 分片集合与 EXP-09 不一致")
    records: list[dict[str, Any] | None] = [None] * len(target_poi_ids)
    completed_sources: list[dict[str, Any]] = []
    scanned = 0
    found = 0
    with embedding_poi_ids_path.open("rb") as id_handle:
        for source_path in source_paths:
            digest = hashlib.sha256()
            source_rows = 0
            with source_path.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    digest.update(line)
                    try:
                        raw = json.loads(line)
                        poi_id = int(str(raw["poi_id"]))
                        embedded_id_line = id_handle.readline()
                        embedded_id = int(str(json.loads(embedded_id_line)))
                    except (
                        json.JSONDecodeError,
                        KeyError,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ) as error:
                        raise ResidualVectorAuditError(
                            f"POI JSON/ID 非法：{source_path.name}:{line_number}"
                        ) from error
                    if embedded_id != poi_id:
                        raise ResidualVectorAuditError(
                            f"Embedding 行序错位：row={scanned}"
                        )
                    position = target_by_id.get(poi_id)
                    if position is not None:
                        if records[position] is not None:
                            raise ResidualVectorAuditError(f"POI 重复：{poi_id}")
                        try:
                            longitude = float(raw["lng"])
                            latitude = float(raw["lat"])
                        except (KeyError, TypeError, ValueError) as error:
                            raise ResidualVectorAuditError("POI 坐标非法") from error
                        if not (
                            math.isfinite(longitude)
                            and math.isfinite(latitude)
                            and -180 <= longitude <= 180
                            and -90 <= latitude <= 90
                        ):
                            raise ResidualVectorAuditError("POI 坐标越界")
                        record = {
                            "poi_id": str(poi_id),
                            "embedding_row": scanned,
                            "lng": longitude,
                            "lat": latitude,
                        }
                        for field in STABLE_FIELDS:
                            record[field] = _raw_text(raw.get(field))
                        records[position] = record
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
                    raise ResidualVectorAuditError(
                        f"POI 分片 {source_path.name} 的 {key} 与 EXP-09 不一致"
                    )
            completed_sources.append(completed)
            if progress is not None:
                progress(
                    f"扫描原始 POI {scanned:,} 行；回查残留 "
                    f"{found:,}/{len(target_poi_ids):,}"
                )
        if id_handle.readline():
            raise ResidualVectorAuditError("Embedding POI ID 文件存在额外行")
    if found != len(target_poi_ids) or any(record is None for record in records):
        raise ResidualVectorAuditError("残留 POI 回查不完整")
    return [record for record in records if record is not None], completed_sources, scanned


def _embedding_digests(
    embeddings: np.ndarray,
    embedding_rows: np.ndarray,
    *,
    progress: Callable[[str], None] | None,
) -> list[bytes]:
    digests: list[bytes] = []
    first_by_digest: dict[bytes, int] = {}
    for position, embedding_row in enumerate(embedding_rows):
        vector = np.asarray(embeddings[int(embedding_row)])
        digest = hashlib.sha256(vector.tobytes()).digest()
        first = first_by_digest.get(digest)
        if first is not None and not np.array_equal(
            vector, np.asarray(embeddings[first])
        ):
            raise ResidualVectorAuditError("Embedding SHA256 碰撞")
        first_by_digest.setdefault(digest, int(embedding_row))
        digests.append(digest)
        if progress is not None and (position + 1) % 20_000 == 0:
            progress(f"计算精确向量指纹 {position + 1:,}/{len(embedding_rows):,}")
    return digests


def _gid_keys(gid12_codes: np.ndarray) -> dict[int, list[bytes]]:
    return {
        length: [row[:length].tobytes() for row in gid12_codes]
        for length in GID_LENGTHS
    }


def _stable_signatures(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, ...]]:
    return [
        tuple(normalize_static_text(record[field]) for field in STABLE_FIELDS)
        for record in records
    ]


def _group_rows(
    *,
    group_ids: np.ndarray,
    embedding_keys: Sequence[bytes],
    gid_keys: Mapping[int, Sequence[bytes]],
    coordinate_keys: Sequence[Hashable],
    raw_text_keys: Sequence[str],
    stable_keys: Sequence[Hashable],
) -> list[dict[str, Any]]:
    members_by_group: dict[int, list[int]] = {}
    for row, group_id in enumerate(group_ids):
        members_by_group.setdefault(int(group_id), []).append(row)
    rows: list[dict[str, Any]] = []
    for group_id, members in members_by_group.items():
        size = len(members)
        embedding_unique = len({embedding_keys[row] for row in members})
        coordinate_unique = len({coordinate_keys[row] for row in members})
        raw_text_unique = len({raw_text_keys[row] for row in members})
        stable_unique = len({stable_keys[row] for row in members})
        minimum: str | None = None
        for length in GID_LENGTHS:
            if len(
                {
                    (embedding_keys[row], gid_keys[length][row])
                    for row in members
                }
            ) == size:
                minimum = f"gid{length}"
                break
        if minimum is None and len(
            {(embedding_keys[row], coordinate_keys[row]) for row in members}
        ) == size:
            minimum = "exact_coordinate"
        if embedding_unique == size:
            category = "all_vectors_distinct_same_rq_sid"
        elif embedding_unique == 1:
            if minimum and minimum.startswith("gid"):
                category = f"all_same_vector_resolved_by_{minimum}"
            elif minimum == "exact_coordinate":
                category = "all_same_vector_resolved_by_exact_coordinate"
            elif coordinate_unique == 1:
                category = "all_same_vector_same_coordinate"
            else:
                category = "all_same_vector_partially_geographic"
        elif minimum and minimum.startswith("gid"):
            category = f"mixed_vectors_resolved_by_{minimum}"
        elif minimum == "exact_coordinate":
            category = "mixed_vectors_resolved_by_exact_coordinate"
        else:
            category = "mixed_vectors_exact_duplicates_remain"
        rows.append(
            {
                "g6_group_id": group_id,
                "group_size": size,
                "diagnostic_type": category,
                "embedding_unique_count": embedding_unique,
                "embedding_collision_excess": size - embedding_unique,
                "all_embeddings_identical": embedding_unique == 1,
                "all_embeddings_distinct": embedding_unique == size,
                "coordinate_unique_count": coordinate_unique,
                "all_coordinates_identical": coordinate_unique == 1,
                "raw_embedding_text_unique_count": raw_text_unique,
                "stable_signature_unique_count": stable_unique,
                "minimum_geo_after_embedding_for_full_group": minimum or "unresolved",
                **{
                    f"gid{length}_unique_count": len(
                        {gid_keys[length][row] for row in members}
                    )
                    for length in GID_LENGTHS
                },
            }
        )
    return rows


def _case_payloads(
    *,
    group_rows: Sequence[Mapping[str, Any]],
    group_ids: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    embedding_keys: Sequence[bytes],
    gid12_codes: np.ndarray,
    minimum_labels: Sequence[str],
    cases_per_type: int,
    pois_per_case: int,
) -> list[dict[str, Any]]:
    members_by_group: dict[int, list[int]] = {}
    for row, group_id in enumerate(group_ids):
        members_by_group.setdefault(int(group_id), []).append(row)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in group_rows:
        grouped.setdefault(str(row["diagnostic_type"]), []).append(row)
    cases: list[dict[str, Any]] = []
    for diagnostic_type in sorted(grouped):
        selected = sorted(
            grouped[diagnostic_type],
            key=lambda item: (-int(item["group_size"]), int(item["g6_group_id"])),
        )[:cases_per_type]
        for group in selected:
            group_id = int(group["g6_group_id"])
            members = members_by_group[group_id]
            pois = []
            for row in members[:pois_per_case]:
                record = records[row]
                pois.append(
                    {
                        "poi_id": record["poi_id"],
                        "displayname": record["displayname"],
                        "address": record["address"],
                        "alias": record["alias"],
                        "lng": record["lng"],
                        "lat": record["lat"],
                        "gid12": tokens_to_geohash(gid12_codes[row]),
                        "embedding_sha256": embedding_keys[row].hex(),
                        "minimum_geo_for_duplicate_vector": minimum_labels[row],
                    }
                )
            cases.append({**group, "shown_pois": pois})
    return cases


def audit_residual_vectors(
    *,
    project_root: Path,
    poi_dir: Path,
    experiment_dir: Path,
    embedding_dir: Path,
    output_dir: Path,
    experiment_id: str,
    cases_per_type: int = 5,
    pois_per_case: int = 12,
    progress: Callable[[str], None] | None = None,
) -> ResidualVectorAuditResult:
    """Audit exact BGE vectors and GID7-12 in EXP-09 residual groups."""

    started_at = _utc_now()
    started = time.monotonic()
    project_root = project_root.resolve()
    poi_dir = poi_dir.resolve()
    experiment_dir = experiment_dir.resolve()
    embedding_dir = embedding_dir.resolve()
    output_dir = output_dir.resolve()
    for path, name in (
        (project_root, "项目根目录"),
        (poi_dir, "POI 目录"),
        (experiment_dir, "EXP-09 目录"),
        (embedding_dir, "Embedding 目录"),
    ):
        if not path.is_dir():
            raise ResidualVectorAuditError(f"{name}不存在：{path}")
    if output_dir.exists():
        raise ResidualVectorAuditError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    if cases_per_type < 0 or pois_per_case <= 0:
        raise ResidualVectorAuditError("案例参数非法")

    arrays, experiment_manifest = _load_residual_inputs(experiment_dir)
    unresolved = arrays["unresolved"]
    group_ids = np.asarray(arrays["post_g6_group_ids"])[unresolved]
    poi_ids = np.asarray(arrays["post_g6_poi_ids"])[unresolved]
    collision_rows = np.asarray(arrays["post_g6_collision_rows"])[unresolved]

    embedding_manifest = _load_json(
        embedding_dir / "manifest.json", "BGE embedding manifest"
    )
    embedding_output = embedding_manifest.get("output", {})
    embeddings_path = embedding_dir / str(embedding_output.get("embeddings", ""))
    embedding_poi_ids_path = embedding_dir / str(
        embedding_output.get("poi_ids", "")
    )
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if list(embeddings.shape) != embedding_output.get("shape") or str(
        embeddings.dtype
    ) != embedding_output.get("dtype"):
        raise ResidualVectorAuditError("BGE embedding shape/dtype 与 manifest 不一致")
    expected_sources = experiment_manifest.get("inputs", {}).get("raw_sources")
    if not isinstance(expected_sources, list):
        raise ResidualVectorAuditError("EXP-09 manifest 缺少 raw_sources")
    if progress is not None:
        progress("回查 EXP-09 残留 POI，并逐行验证 BGE Embedding 行序")
    records, raw_sources, full_poi_count = _stream_records(
        poi_dir=poi_dir,
        target_poi_ids=poi_ids,
        expected_sources=expected_sources,
        embedding_poi_ids_path=embedding_poi_ids_path,
        progress=progress,
    )
    if full_poi_count != len(embeddings):
        raise ResidualVectorAuditError("POI 行数与 BGE Embedding 行数不一致")

    embedding_rows = np.asarray(
        [record["embedding_row"] for record in records], dtype=np.int64
    )
    if progress is not None:
        progress("计算原始 BGE-M3 float16×1024 的逐字节 SHA256")
    embedding_keys = _embedding_digests(
        embeddings, embedding_rows, progress=progress
    )
    longitudes = np.asarray([record["lng"] for record in records])
    latitudes = np.asarray([record["lat"] for record in records])
    gid12_codes = encode_geohash_tokens(longitudes, latitudes, length=12)

    structure_input = experiment_manifest.get("inputs", {}).get(
        "structure_manifest", {}
    )
    structure_manifest_path = Path(str(structure_input.get("path", "")))
    structure_dir = structure_manifest_path.parent
    if sha256_file(structure_manifest_path) != structure_input.get("sha256"):
        raise ResidualVectorAuditError("EXP-09 冻结结构 manifest SHA256 不一致")
    validate_structure_output(structure_dir)
    structure_manifest = _load_json(structure_manifest_path, "结构 manifest")
    structure_contracts = structure_manifest.get("outputs", {}).get("arrays", {})
    frozen_gid8 = _load_array(
        structure_dir,
        structure_contracts["collision_gid8_codes"],
        "collision_gid8_codes",
    )
    if np.any(gid12_codes[:, :8] != np.asarray(frozen_gid8)[collision_rows]):
        raise ResidualVectorAuditError("重算 GID8 与冻结结构不一致")

    gid_keys = _gid_keys(gid12_codes)
    coordinate_keys = [
        (float(record["lng"]), float(record["lat"])) for record in records
    ]
    raw_text_keys = [str(record["text"]) for record in records]
    stable_keys = _stable_signatures(records)
    cascades = build_partition_cascade(
        group_ids=group_ids,
        embedding_keys=embedding_keys,
        gid_keys=gid_keys,
        coordinate_keys=coordinate_keys,
    )
    baseline_excess = EXPECTED_RESIDUAL_EXCESS
    vector_stats = collision_partition_stats(group_ids, embedding_keys)
    vector_text_stats = collision_partition_stats(
        group_ids, list(zip(embedding_keys, raw_text_keys, strict=True))
    )
    vector_stable_stats = collision_partition_stats(
        group_ids, list(zip(embedding_keys, stable_keys, strict=True))
    )
    vector_stable_coord_stats = collision_partition_stats(
        group_ids,
        list(
            zip(embedding_keys, stable_keys, coordinate_keys, strict=True)
        ),
    )
    minimum_labels, minimum_distribution = (
        minimum_gid_labels_for_duplicate_vectors(
            group_ids=group_ids,
            embedding_keys=embedding_keys,
            gid_keys=gid_keys,
            coordinate_keys=coordinate_keys,
        )
    )
    vector_peer_counts = Counter(zip(group_ids, embedding_keys, strict=True))
    vector_coordinate_counts = Counter(
        zip(group_ids, embedding_keys, coordinate_keys, strict=True)
    )
    group_rows = _group_rows(
        group_ids=group_ids,
        embedding_keys=embedding_keys,
        gid_keys=gid_keys,
        coordinate_keys=coordinate_keys,
        raw_text_keys=raw_text_keys,
        stable_keys=stable_keys,
    )
    diagnostic_distribution = dict(
        sorted(Counter(row["diagnostic_type"] for row in group_rows).items())
    )
    cases = _case_payloads(
        group_rows=group_rows,
        group_ids=group_ids,
        records=records,
        embedding_keys=embedding_keys,
        gid12_codes=gid12_codes,
        minimum_labels=minimum_labels,
        cases_per_type=cases_per_type,
        pois_per_case=pois_per_case,
    )

    poi_rows = []
    for row, record in enumerate(records):
        poi_rows.append(
            {
                "g6_group_id": int(group_ids[row]),
                "poi_id": record["poi_id"],
                "embedding_row": int(record["embedding_row"]),
                "embedding_sha256": embedding_keys[row].hex(),
                "same_embedding_peer_count": int(
                    vector_peer_counts[(int(group_ids[row]), embedding_keys[row])]
                ),
                "same_embedding_coordinate_peer_count": int(
                    vector_coordinate_counts[
                        (
                            int(group_ids[row]),
                            embedding_keys[row],
                            coordinate_keys[row],
                        )
                    ]
                ),
                "minimum_geo_for_duplicate_vector": minimum_labels[row],
                "gid6": tokens_to_geohash(gid12_codes[row, :6]),
                "gid12": tokens_to_geohash(gid12_codes[row]),
                "lng": record["lng"],
                "lat": record["lat"],
                **{field: record[field] for field in STABLE_FIELDS},
            }
        )

    exact_vector_excess = int(vector_stats["collision_excess"])
    final_vector_geo = cascades["exact_embedding_then_geography"][
        "exact_embedding_coordinate"
    ]
    metrics = {
        "schema_version": VECTOR_AUDIT_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "experiment_id": experiment_id,
        "method": "GHR-SID EXP-09 residual exact-vector/fine-GID audit",
        "query_usage": "none",
        "baseline": {
            "poi_count": EXPECTED_RESIDUAL_POI_COUNT,
            "group_count": EXPECTED_RESIDUAL_GROUP_COUNT,
            "collision_excess": baseline_excess,
            "max_group_size": EXPECTED_MAX_GROUP_SIZE,
        },
        "embedding_partition": {
            **vector_stats,
            "different_exact_vectors_excess": baseline_excess
            - exact_vector_excess,
            "different_exact_vectors_excess_ratio": _ratio(
                baseline_excess - exact_vector_excess, baseline_excess
            ),
            "byte_identity_contract": (
                "SHA256 over raw float16×1024 bytes; repeated digests are "
                "verified by np.array_equal"
            ),
        },
        "cascades": cascades,
        "duplicate_vector_poi_minimum_geography": minimum_distribution,
        "exact_vector_metadata_diagnostics": {
            "exact_embedding": vector_stats,
            "exact_embedding_plus_raw_embedding_text": vector_text_stats,
            "exact_embedding_plus_all_stable_fields": vector_stable_stats,
            "exact_embedding_plus_all_stable_fields_plus_coordinate": (
                vector_stable_coord_stats
            ),
        },
        "group_diagnostic_distribution": diagnostic_distribution,
        "decision": {
            "exact_vector_excess": exact_vector_excess,
            "exact_vector_excess_resolved_by_gid7_to_gid12_or_coordinate": (
                exact_vector_excess
                - int(final_vector_geo["collision_excess"])
            ),
            "exact_vector_and_coordinate_excess": int(
                final_vector_geo["collision_excess"]
            ),
            "finer_gid_cannot_resolve_same_coordinate": True,
            "ready_to_change_identifier": False,
            "next_step": "先审计案例，再决定仅对同向量异地残留追加最短细 GID",
        },
        "interpretation_contract": {
            "rq_sid_equal_is_not_treated_as_vector_equal": True,
            "embedding_is_original_bge_m3_not_e4": True,
            "gid_lengths_audited": list(GID_LENGTHS),
            "query_or_order_consumed": False,
            "poi_id_used_as_feature": False,
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
    }

    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        poi_path = staging_dir / "residual_vector_pois.parquet"
        group_path = staging_dir / "residual_vector_groups.parquet"
        pq.write_table(pa.Table.from_pylist(poi_rows), poi_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(group_rows), group_path, compression="zstd")
        cases_path = staging_dir / "cases.jsonl"
        with cases_path.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
        metrics_path = staging_dir / "metrics.json"
        _json_dump(metrics_path, metrics)
        manifest = {
            "schema_version": VECTOR_AUDIT_SCHEMA_VERSION,
            "status": "completed",
            "experiment_id": experiment_id,
            "inputs": {
                "exp09_manifest": {
                    "path": str((experiment_dir / "manifest.json").resolve()),
                    "sha256": sha256_file(experiment_dir / "manifest.json"),
                },
                "structure_manifest": {
                    "path": str(structure_manifest_path.resolve()),
                    "sha256": sha256_file(structure_manifest_path),
                },
                "embedding_manifest": {
                    "path": str((embedding_dir / "manifest.json").resolve()),
                    "sha256": sha256_file(embedding_dir / "manifest.json"),
                    "signature": embedding_manifest.get("signature"),
                },
                "embedding_file": {
                    "path": str(embeddings_path.resolve()),
                    "bytes": embeddings_path.stat().st_size,
                    "shape": list(embeddings.shape),
                    "dtype": str(embeddings.dtype),
                },
                "embedding_poi_ids": {
                    "path": str(embedding_poi_ids_path.resolve()),
                    "bytes": embedding_poi_ids_path.stat().st_size,
                    "sha256": sha256_file(embedding_poi_ids_path),
                },
                "raw_sources": raw_sources,
            },
            "configuration": {
                "vector_equality": "raw float16×1024 byte equality",
                "gid_lengths": list(GID_LENGTHS),
                "query_or_order_used": False,
                "poi_id_used_as_feature": False,
            },
            "outputs": {
                "residual_vector_pois": _file_contract(
                    poi_path, rows=len(poi_rows)
                ),
                "residual_vector_groups": _file_contract(
                    group_path, rows=len(group_rows)
                ),
                "cases": _file_contract(cases_path, rows=len(cases)),
                "metrics": _file_contract(metrics_path),
            },
            "validation": {
                "all_target_pois_found_exactly_once": True,
                "embedding_and_raw_poi_order_match": True,
                "recomputed_gid8_matches_frozen": True,
                "repeated_embedding_hashes_verified_exact": True,
                "query_or_order_consumed": False,
            },
            "runtime": {
                "elapsed_seconds": time.monotonic() - started,
                "numpy": np.__version__,
            },
            "git": _git_state(project_root),
            "started_at": started_at,
            "finished_at": _utc_now(),
        }
        manifest["signature"] = _signature(
            {key: value for key, value in manifest.items() if key != "signature"}
        )
        _json_dump(staging_dir / "manifest.json", manifest)
        (staging_dir / "_SUCCESS").touch()
        staging_dir.replace(output_dir)
    except Exception:
        raise
    return ResidualVectorAuditResult(metrics, manifest, output_dir)


def validate_residual_vector_audit_output(output_dir: Path) -> dict[str, Any]:
    """Validate every managed EXP-10 output without changing it."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "向量审计 manifest")
    if manifest.get("schema_version") != VECTOR_AUDIT_SCHEMA_VERSION:
        raise ResidualVectorAuditError("向量审计 schema_version 不一致")
    if not (output_dir / "_SUCCESS").is_file():
        raise ResidualVectorAuditError("向量审计缺少 _SUCCESS")
    expected_signature = _signature(
        {key: value for key, value in manifest.items() if key != "signature"}
    )
    if expected_signature != manifest.get("signature"):
        raise ResidualVectorAuditError("向量审计 manifest signature 不一致")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ResidualVectorAuditError("向量审计 manifest 缺少 outputs")
    for name, contract in outputs.items():
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file():
            raise ResidualVectorAuditError(f"输出不存在：{name}")
        if path.stat().st_size != contract.get("bytes"):
            raise ResidualVectorAuditError(f"输出字节数不一致：{name}")
        if sha256_file(path) != contract.get("sha256"):
            raise ResidualVectorAuditError(f"输出 SHA256 不一致：{name}")
        if path.suffix == ".parquet" and pq.read_metadata(path).num_rows != contract.get(
            "rows"
        ):
            raise ResidualVectorAuditError(f"Parquet 行数不一致：{name}")
    metrics = _load_json(output_dir / "metrics.json", "向量审计 metrics")
    if metrics.get("status") != "completed":
        raise ResidualVectorAuditError("向量审计 metrics 未完成")
    return {
        "status": "validated",
        "experiment_id": manifest.get("experiment_id"),
        "validated_output_count": len(outputs),
        "signature": manifest.get("signature"),
    }

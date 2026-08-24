"""Deterministic geographic-query Validation subset construction."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..methods.genpoi.proximity import (
    GenPoiProximityError,
    extract_proximity_example,
)
from ..pid.trie import sha256_file
from .evaluation import atomic_write_json


GEO_QUERY_SUBSET_SCHEMA_VERSION = "geo-query-validation-subset-v1"
DEFAULT_MIN_QUERY_LENGTH = 6
DEFAULT_SUBSET_SIZE = 10_000

GEOGRAPHIC_RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nearby", re.compile(r"附近|周边|旁边|边上|周围|就近|最近|邻近")),
    (
        "distance",
        re.compile(
            r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千两]+)\s*"
            r"(?:米|公里|千米|km)"
        ),
    ),
    (
        "direction",
        re.compile(
            r"(?:东北|东南|西北|西南|东|南|西|北)(?:门|侧|边|面|口)"
            r"|(?:往|向)(?:东|南|西|北)"
        ),
    ),
    ("opposite_or_intersection", re.compile(r"对面|隔壁|路口|交叉口|十字路口")),
    (
        "inside_or_entrance",
        re.compile(r"里面|外面|内部|园内|校内|院内|楼内|馆内|站内|站外|门口"),
    ),
    (
        "transit_exit",
        re.compile(
            r"(?:地铁|轨道交通).{0,8}(?:站|口)"
            r"|(?:[a-h]|[一二三四五六七八九十])口"
        ),
    ),
)


class GeoQuerySubsetError(ValueError):
    """Raised when geographic-query subset inputs violate the contract."""


@dataclass(frozen=True)
class GeoQueryCandidate:
    """One eligible source row and its model-independent selection metadata."""

    row_index: int
    sample_id: str
    order_id: str
    searchid: str
    target_poi_id: str
    selection_hash: str
    query_length: int
    relation_types: tuple[str, ...]
    geo_prefix_length: int
    raw_line: bytes


@dataclass(frozen=True)
class GeoQuerySubset:
    """Frozen geographic-query subset artifact."""

    data_path: Path
    manifest_path: Path
    row_count: int
    sha256: str
    manifest: dict[str, Any]


def normalize_query(query: str) -> str:
    """Normalize text and remove whitespace before complexity checks."""

    normalized = unicodedata.normalize("NFKC", query).lower()
    return "".join(normalized.split())


def geographic_relation_types(query: str) -> tuple[str, ...]:
    """Return deterministic explicit geographic-relation categories."""

    normalized = normalize_query(query)
    return tuple(
        name
        for name, pattern in GEOGRAPHIC_RELATION_PATTERNS
        if pattern.search(normalized)
    )


def stable_selection_hash(
    order_id: str,
    searchid: str,
    target_poi_id: str,
) -> str:
    """Hash immutable business keys for result-independent sampling."""

    value = f"{order_id}\0{searchid}\0{target_poi_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _required_text(record: Mapping[str, Any], field: str, source: str) -> str:
    value = record.get(field)
    if value is None or not str(value).strip():
        raise GeoQuerySubsetError(f"{source} 缺少有效 {field}")
    return str(value)


def _counter_json(counter: Counter[Any], keys: Sequence[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in keys}


def _validate_existing_subset(
    data_path: Path,
    manifest_path: Path,
    expected_manifest: Mapping[str, Any],
) -> GeoQuerySubset:
    if not data_path.is_file() or not manifest_path.is_file():
        raise GeoQuerySubsetError("专项集 JSONL 与 manifest 必须同时存在")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GeoQuerySubsetError("专项集 manifest JSON 非法") from error
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise GeoQuerySubsetError("已有专项集 manifest 与当前输入或规则不一致")
    output_sha256 = sha256_file(data_path)
    if manifest.get("output_sha256") != output_sha256:
        raise GeoQuerySubsetError("已有专项集 SHA256 与 manifest 不一致")
    with data_path.open("rb") as handle:
        rows = sum(1 for _ in handle)
    if rows != manifest.get("output_rows") or rows != expected_manifest["subset_size"]:
        raise GeoQuerySubsetError("已有专项集行数与 manifest 不一致")
    return GeoQuerySubset(
        data_path=data_path,
        manifest_path=manifest_path,
        row_count=rows,
        sha256=output_sha256,
        manifest=manifest,
    )


def build_geo_query_validation_subset(
    valid_file: Path,
    output_dir: Path,
    *,
    source_rows: int,
    source_sha256: str,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    min_query_length: int = DEFAULT_MIN_QUERY_LENGTH,
) -> GeoQuerySubset:
    """Freeze a stable sample of complex queries with explicit geo relations."""

    if subset_size <= 0 or subset_size > source_rows:
        raise GeoQuerySubsetError("专项集大小必须位于 (0, source_rows]")
    if min_query_length <= 0:
        raise GeoQuerySubsetError("最小 Query 长度必须为正整数")
    valid_file = valid_file.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"geo_query_validation_subset_{subset_size}.jsonl"
    manifest_path = output_dir / f"geo_query_validation_subset_{subset_size}_manifest.json"
    rule_patterns = {
        name: pattern.pattern for name, pattern in GEOGRAPHIC_RELATION_PATTERNS
    }
    expected_manifest = {
        "schema_version": GEO_QUERY_SUBSET_SCHEMA_VERSION,
        "status": "completed",
        "split": "valid",
        "source_file": str(valid_file),
        "source_rows": source_rows,
        "source_sha256": source_sha256,
        "subset_size": subset_size,
        "query_source": "messages[role=user]/CURRENT/QUERY",
        "normalization": "NFKC_lower_remove_whitespace",
        "minimum_normalized_query_length": min_query_length,
        "geographic_relation_patterns": rule_patterns,
        "eligibility_rule": "query_length>=minimum AND any_geographic_relation_pattern",
        "selection_method": "lowest_sha256(order_id\\0searchid\\0target_poi_id)",
        "selection_uses_model_outputs": False,
        "selection_uses_target_geo_prefix_length": False,
        "output_order": "source_row_ascending",
    }
    if data_path.exists() or manifest_path.exists():
        return _validate_existing_subset(data_path, manifest_path, expected_manifest)

    candidates: list[GeoQueryCandidate] = []
    actual_source_rows = 0
    source_digest = hashlib.sha256()
    with valid_file.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            source_digest.update(raw_line)
            actual_source_rows += 1
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise GeoQuerySubsetError(
                    f"Validation 第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(record, Mapping):
                raise GeoQuerySubsetError("Validation 每行必须是 object")
            try:
                example = extract_proximity_example(record, expected_split="valid")
            except GenPoiProximityError as error:
                raise GeoQuerySubsetError(
                    f"Validation 第 {line_number} 行地理字段非法：{error}"
                ) from error
            normalized = normalize_query(example.query)
            relation_types = geographic_relation_types(normalized)
            if len(normalized) < min_query_length or not relation_types:
                continue
            source = f"Validation 第 {line_number} 行"
            order_id = _required_text(record, "order_id", source)
            searchid = _required_text(record, "searchid", source)
            target_poi_id = _required_text(record, "target_poi_id", source)
            candidates.append(
                GeoQueryCandidate(
                    row_index=line_number - 1,
                    sample_id=example.sample_id,
                    order_id=order_id,
                    searchid=searchid,
                    target_poi_id=target_poi_id,
                    selection_hash=stable_selection_hash(
                        order_id,
                        searchid,
                        target_poi_id,
                    ),
                    query_length=len(normalized),
                    relation_types=relation_types,
                    geo_prefix_length=example.label,
                    raw_line=raw_line,
                )
            )
    if actual_source_rows != source_rows:
        raise GeoQuerySubsetError(
            f"Validation 实际行数 {actual_source_rows:,} != manifest {source_rows:,}"
        )
    if source_digest.hexdigest() != source_sha256:
        raise GeoQuerySubsetError("Validation SHA256 与 manifest 不一致")
    if len(candidates) < subset_size:
        raise GeoQuerySubsetError(
            f"符合规则的 Query 仅 {len(candidates):,} 条，不足 {subset_size:,}"
        )

    selected = heapq.nsmallest(
        subset_size,
        candidates,
        key=lambda item: (item.selection_hash, item.row_index),
    )
    if len({item.sample_id for item in selected}) != subset_size:
        raise GeoQuerySubsetError("选中样本的 sample_id 存在重复")
    business_keys = [(item.order_id, item.searchid) for item in selected]
    if len(set(business_keys)) != subset_size:
        raise GeoQuerySubsetError("选中样本的 order_id + searchid 存在重复")
    selected.sort(key=lambda item: item.row_index)

    temporary = data_path.with_name(f".{data_path.name}.tmp")
    output_digest = hashlib.sha256()
    with temporary.open("wb") as destination:
        for item in selected:
            destination.write(item.raw_line)
            output_digest.update(item.raw_line)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, data_path)

    candidate_relations: Counter[str] = Counter()
    candidate_prefixes: Counter[int] = Counter()
    selected_relations: Counter[str] = Counter()
    selected_prefixes: Counter[int] = Counter()
    candidate_lengths: Counter[int] = Counter()
    selected_lengths: Counter[int] = Counter()
    for item in candidates:
        candidate_relations.update(item.relation_types)
        candidate_prefixes[item.geo_prefix_length] += 1
        candidate_lengths[item.query_length] += 1
    for item in selected:
        selected_relations.update(item.relation_types)
        selected_prefixes[item.geo_prefix_length] += 1
        selected_lengths[item.query_length] += 1
    key_bytes = "".join(
        f"{item.order_id}\t{item.searchid}\n" for item in selected
    ).encode("utf-8")
    sample_id_bytes = "".join(f"{item.sample_id}\n" for item in selected).encode(
        "utf-8"
    )
    relation_names = [name for name, _ in GEOGRAPHIC_RELATION_PATTERNS]
    all_lengths = sorted(set(candidate_lengths) | set(selected_lengths))
    manifest = {
        **expected_manifest,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "date": "2026-07-13",
        "eligible_rows": len(candidates),
        "sampling_rate": subset_size / len(candidates),
        "candidate_relation_counts": _counter_json(
            candidate_relations,
            relation_names,
        ),
        "selected_relation_counts": _counter_json(
            selected_relations,
            relation_names,
        ),
        "candidate_geo_prefix_length_counts": _counter_json(
            candidate_prefixes,
            range(7),
        ),
        "selected_geo_prefix_length_counts": _counter_json(
            selected_prefixes,
            range(7),
        ),
        "candidate_query_length_counts": _counter_json(
            candidate_lengths,
            all_lengths,
        ),
        "selected_query_length_counts": _counter_json(
            selected_lengths,
            all_lengths,
        ),
        "business_keys_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "selected_sample_ids_sha256": hashlib.sha256(sample_id_bytes).hexdigest(),
        "output_file": str(data_path),
        "output_rows": subset_size,
        "output_sha256": output_digest.hexdigest(),
    }
    atomic_write_json(manifest_path, manifest)
    return GeoQuerySubset(
        data_path=data_path,
        manifest_path=manifest_path,
        row_count=subset_size,
        sha256=output_digest.hexdigest(),
        manifest=manifest,
    )

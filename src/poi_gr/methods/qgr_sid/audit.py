"""Streaming coverage audit for QGR-SID numeric collision relations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pyarrow.parquet as pq

from poi_gr.methods.qgr_sid.relations import (
    RELATION_SCHEMA_VERSION,
    RELATION_TYPES,
    ExtractionResult,
    extract_numeric_relations,
)
from poi_gr.methods.tiger.identifier import sha256_file


AUDIT_SCHEMA_VERSION = "qgr-sid-numeric-relation-audit-v1"
AUDIT_METRICS_SCHEMA_VERSION = "qgr-sid-numeric-relation-audit-metrics-v1"
REQUIRED_MAPPING_COLUMNS = (
    "poi_id",
    "s1",
    "s2",
    "s3",
    "base_sid_bucket_size",
)
OUTPUT_FILENAMES = ("metrics.json", "relation_examples.jsonl", "manifest.json")


class QgrSidAuditError(ValueError):
    """Raised when QGR-SID audit inputs or outputs violate the contract."""


@dataclass(frozen=True)
class AuditResult:
    """Completed audit metrics and artifact manifest."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


@dataclass(frozen=True)
class _QuerySupport:
    train_order_count: np.ndarray
    unique_query_count: np.ndarray
    p99_train_order_count: float
    manifest_path: Path
    manifest_sha256: str
    input_files: dict[str, dict[str, Any]]


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
        raise QgrSidAuditError(f"无法读取{name}：{path}：{error}") from error
    if not isinstance(payload, dict):
        raise QgrSidAuditError(f"{name}必须是 JSON 对象：{path}")
    return payload


def _verify_npy_contract(
    path: Path,
    contract: Mapping[str, Any],
    *,
    name: str,
) -> np.ndarray:
    if not path.is_file():
        raise QgrSidAuditError(f"{name}不存在：{path}")
    expected_sha = contract.get("sha256")
    if expected_sha and sha256_file(path) != expected_sha:
        raise QgrSidAuditError(f"{name} SHA256 与 manifest 不一致")
    array = np.load(path, mmap_mode="r")
    expected_shape = tuple(contract.get("shape", ()))
    if expected_shape and array.shape != expected_shape:
        raise QgrSidAuditError(
            f"{name} shape 不一致：实际 {array.shape}，期望 {expected_shape}"
        )
    expected_dtype = contract.get("dtype")
    if expected_dtype and str(array.dtype) != expected_dtype:
        raise QgrSidAuditError(
            f"{name} dtype 不一致：实际 {array.dtype}，期望 {expected_dtype}"
        )
    return array


def _load_query_support(query_aggregate_dir: Path, poi_count: int) -> _QuerySupport:
    manifest_path = query_aggregate_dir / "manifest.json"
    manifest = _load_json(manifest_path, "Query 聚合 manifest")
    if manifest.get("status") != "completed":
        raise QgrSidAuditError("Query 聚合 manifest 尚未 completed")
    if manifest.get("schema_version") != "query-poi-aggregates-v1":
        raise QgrSidAuditError("Query 聚合 schema_version 不受支持")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise QgrSidAuditError("Query 聚合 manifest 缺少 outputs")

    arrays: dict[str, np.ndarray] = {}
    input_files: dict[str, dict[str, Any]] = {}
    for key in ("covered_poi_rows", "train_order_count", "unique_query_count"):
        contract = outputs.get(key)
        if not isinstance(contract, dict) or not isinstance(contract.get("file"), str):
            raise QgrSidAuditError(f"Query 聚合 manifest 缺少 {key} 契约")
        path = query_aggregate_dir / contract["file"]
        arrays[key] = _verify_npy_contract(path, contract, name=key)
        input_files[key] = {
            "path": str(path.resolve()),
            "sha256": contract.get("sha256"),
            "shape": list(arrays[key].shape),
            "dtype": str(arrays[key].dtype),
        }

    covered_rows = np.asarray(arrays["covered_poi_rows"], dtype=np.int64)
    order_count = np.asarray(arrays["train_order_count"], dtype=np.int64)
    unique_count = np.asarray(arrays["unique_query_count"], dtype=np.int64)
    if not (len(covered_rows) == len(order_count) == len(unique_count)):
        raise QgrSidAuditError("Query 聚合三个索引数组行数不一致")
    if len(covered_rows) and (
        int(covered_rows.min()) < 0 or int(covered_rows.max()) >= poi_count
    ):
        raise QgrSidAuditError("covered_poi_rows 超出 POI catalog 行号范围")
    if len(np.unique(covered_rows)) != len(covered_rows):
        raise QgrSidAuditError("covered_poi_rows 存在重复")
    if np.any(order_count <= 0) or np.any(unique_count <= 0):
        raise QgrSidAuditError("Query 覆盖行的统计必须为正数")

    full_order_count = np.zeros(poi_count, dtype=np.int64)
    full_unique_count = np.zeros(poi_count, dtype=np.int32)
    full_order_count[covered_rows] = order_count
    full_unique_count[covered_rows] = unique_count
    p99 = float(np.percentile(order_count, 99)) if len(order_count) else 0.0
    return _QuerySupport(
        train_order_count=full_order_count,
        unique_query_count=full_unique_count,
        p99_train_order_count=p99,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        input_files=input_files,
    )


def _iter_raw_pois(
    poi_dir: Path,
    completed_sources: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    source_paths = tuple(sorted(poi_dir.glob("part-*.json")))
    if not source_paths:
        raise QgrSidAuditError(f"POI 目录中没有 part-*.json：{poi_dir}")
    for path in source_paths:
        digest = hashlib.sha256()
        rows = 0
        try:
            with path.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    digest.update(line)
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise QgrSidAuditError(
                            f"POI JSON 解析失败：{path}:{line_number}：{error}"
                        ) from error
                    if not isinstance(record, dict):
                        raise QgrSidAuditError(
                            f"POI JSON 行必须是对象：{path}:{line_number}"
                        )
                    rows += 1
                    yield record
        except OSError as error:
            raise QgrSidAuditError(f"无法读取 POI 源文件 {path}：{error}") from error
        completed_sources.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "rows": rows,
                "sha256": digest.hexdigest(),
            }
        )


def _bucket_key(s1: int, s2: int, s3: int) -> int:
    if not (0 <= s1 < 2**21 and 0 <= s2 < 2**21 and 0 <= s3 < 2**21):
        raise QgrSidAuditError("基础 SID Token 超出审计打包范围")
    return (s1 << 42) | (s2 << 21) | s3


def _bounded_text(value: Any, limit: int = 300) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _example_payload(
    *,
    poi: Mapping[str, Any],
    base_sid: tuple[int, int, int],
    bucket_size: int,
    result: ExtractionResult,
) -> dict[str, Any]:
    return {
        "poi_id": str(poi.get("poi_id", "")),
        "base_sid": list(base_sid),
        "base_sid_bucket_size": bucket_size,
        "displayname": _bounded_text(poi.get("displayname")),
        "address": _bounded_text(poi.get("address")),
        "alias": _bounded_text(poi.get("alias")),
        "relations": [asdict(item) for item in result.relations],
        "conflicts": [asdict(item) for item in result.conflicts],
    }


def _audit_signature(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def audit_numeric_relations(
    *,
    poi_dir: Path,
    identifier_dir: Path,
    output_dir: Path,
    query_aggregate_dir: Path | None = None,
    value_max: int = 1055,
    batch_rows: int = 65_536,
    examples_per_relation: int = 20,
    max_rows: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> AuditResult:
    """Audit lexical numeric relation coverage on every TIGER collision POI."""

    started_at = _utc_now()
    started = time.monotonic()
    poi_dir = poi_dir.resolve()
    identifier_dir = identifier_dir.resolve()
    output_dir = output_dir.resolve()
    query_aggregate_dir = (
        None if query_aggregate_dir is None else query_aggregate_dir.resolve()
    )
    if value_max < 0:
        raise QgrSidAuditError("value_max 不能为负数")
    if batch_rows <= 0:
        raise QgrSidAuditError("batch_rows 必须大于 0")
    if examples_per_relation < 0:
        raise QgrSidAuditError("examples_per_relation 不能为负数")
    if max_rows is not None and max_rows <= 0:
        raise QgrSidAuditError("max_rows 必须大于 0")
    if not poi_dir.is_dir():
        raise QgrSidAuditError(f"POI 目录不存在：{poi_dir}")
    if not identifier_dir.is_dir():
        raise QgrSidAuditError(f"TIGER identifier 目录不存在：{identifier_dir}")
    if output_dir.exists():
        raise QgrSidAuditError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}"
    if staging_dir.exists():
        raise QgrSidAuditError(f"临时输出目录已存在：{staging_dir}")
    staging_dir.mkdir()

    manifest_path = identifier_dir / "tiger_id_manifest.json"
    tiger_metrics_path = identifier_dir / "metrics.json"
    mapping_path = identifier_dir / "poi_tiger_id_mapping.parquet"
    tiger_manifest = _load_json(manifest_path, "TIGER identifier manifest")
    tiger_metrics = _load_json(tiger_metrics_path, "TIGER identifier metrics")
    if tiger_manifest.get("schema_version") != "tiger-item-identifier-v1":
        raise QgrSidAuditError("TIGER identifier schema_version 不受支持")
    if tiger_manifest.get("status") != "completed":
        raise QgrSidAuditError("TIGER identifier 尚未 completed")
    mapping_contract = tiger_manifest.get("mapping")
    if not isinstance(mapping_contract, dict):
        raise QgrSidAuditError("TIGER manifest 缺少 mapping 契约")
    expected_rows = int(mapping_contract.get("rows", 0))
    if expected_rows <= 0:
        raise QgrSidAuditError("TIGER mapping rows 必须为正数")
    if not mapping_path.is_file():
        raise QgrSidAuditError(f"TIGER mapping 不存在：{mapping_path}")
    mapping_sha256 = sha256_file(mapping_path)
    if mapping_sha256 != mapping_contract.get("sha256"):
        raise QgrSidAuditError("TIGER mapping SHA256 与 manifest 不一致")

    parquet = pq.ParquetFile(mapping_path)
    if parquet.metadata.num_rows != expected_rows:
        raise QgrSidAuditError("TIGER mapping Parquet 行数与 manifest 不一致")
    missing_columns = sorted(
        set(REQUIRED_MAPPING_COLUMNS) - set(parquet.schema_arrow.names)
    )
    if missing_columns:
        raise QgrSidAuditError(
            "TIGER mapping 缺少列：" + ", ".join(missing_columns)
        )

    query_support = (
        None
        if query_aggregate_dir is None
        else _load_query_support(query_aggregate_dir, expected_rows)
    )
    completed_sources: list[dict[str, Any]] = []
    raw_iterator = _iter_raw_pois(poi_dir, completed_sources)

    scanned_rows = 0
    colliding_pois = 0
    stable_pois = 0
    in_vocab_pois = 0
    conflicted_pois = 0
    query_supported_collision_pois = 0
    query_supported_in_vocab_pois = 0
    collision_train_orders = 0
    in_vocab_train_orders = 0
    collision_sqrt_weight = 0.0
    in_vocab_sqrt_weight = 0.0
    evidence_count = 0
    collision_buckets: set[int] = set()
    stable_buckets: set[int] = set()
    in_vocab_buckets: set[int] = set()
    stable_count_distribution: Counter[int] = Counter()
    in_vocab_count_distribution: Counter[int] = Counter()
    per_relation: dict[str, Counter[str]] = {
        relation_type: Counter() for relation_type in RELATION_TYPES
    }
    relation_value_counts: dict[str, Counter[int]] = {
        relation_type: Counter() for relation_type in RELATION_TYPES
    }
    relation_sources: dict[str, Counter[str]] = {
        relation_type: Counter() for relation_type in RELATION_TYPES
    }
    examples: dict[str, list[dict[str, Any]]] = {
        relation_type: [] for relation_type in RELATION_TYPES
    }
    out_of_vocab_examples: dict[str, list[dict[str, Any]]] = {
        relation_type: [] for relation_type in RELATION_TYPES
    }
    conflict_examples: dict[str, list[dict[str, Any]]] = {
        relation_type: [] for relation_type in RELATION_TYPES
    }
    category_metrics: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    stop = False
    for batch in parquet.iter_batches(
        batch_size=batch_rows,
        columns=list(REQUIRED_MAPPING_COLUMNS),
    ):
        columns = {
            name: batch.column(index).to_pylist()
            for index, name in enumerate(REQUIRED_MAPPING_COLUMNS)
        }
        for offset in range(batch.num_rows):
            if max_rows is not None and scanned_rows >= max_rows:
                stop = True
                break
            try:
                poi = next(raw_iterator)
            except StopIteration as error:
                raise QgrSidAuditError(
                    f"POI 原始表在第 {scanned_rows} 行提前结束"
                ) from error
            mapping_poi_id = str(columns["poi_id"][offset])
            raw_poi_id = str(poi.get("poi_id", ""))
            if mapping_poi_id != raw_poi_id:
                raise QgrSidAuditError(
                    "POI 与 TIGER mapping 行对齐失败："
                    f"row={scanned_rows}, raw={raw_poi_id!r}, mapping={mapping_poi_id!r}"
                )
            bucket_size = int(columns["base_sid_bucket_size"][offset])
            if bucket_size <= 0:
                raise QgrSidAuditError("base_sid_bucket_size 必须为正数")
            base_sid = (
                int(columns["s1"][offset]),
                int(columns["s2"][offset]),
                int(columns["s3"][offset]),
            )
            row_index = scanned_rows
            scanned_rows += 1
            if bucket_size == 1:
                continue

            colliding_pois += 1
            key = _bucket_key(*base_sid)
            collision_buckets.add(key)
            result = extract_numeric_relations(poi)
            evidence_count += result.evidence_count
            stable = result.relations
            in_vocab = tuple(
                item for item in stable if 0 <= item.numeric_value <= value_max
            )
            stable_count_distribution[len(stable)] += 1
            in_vocab_count_distribution[len(in_vocab)] += 1
            if stable:
                stable_pois += 1
                stable_buckets.add(key)
            if in_vocab:
                in_vocab_pois += 1
                in_vocab_buckets.add(key)
            if result.conflicts:
                conflicted_pois += 1

            order_count = 0
            sqrt_weight = 0.0
            if query_support is not None:
                order_count = int(query_support.train_order_count[row_index])
                if order_count > 0:
                    query_supported_collision_pois += 1
                    clipped = min(order_count, query_support.p99_train_order_count)
                    sqrt_weight = math.sqrt(1.0 + clipped)
                    collision_train_orders += order_count
                    collision_sqrt_weight += sqrt_weight
                    if in_vocab:
                        query_supported_in_vocab_pois += 1
                        in_vocab_train_orders += order_count
                        in_vocab_sqrt_weight += sqrt_weight

            conflict_types = {item.relation_type for item in result.conflicts}
            for relation_type in conflict_types:
                per_relation[relation_type]["conflicted_poi_count"] += 1
                if len(conflict_examples[relation_type]) < examples_per_relation:
                    conflict_examples[relation_type].append(
                        _example_payload(
                            poi=poi,
                            base_sid=base_sid,
                            bucket_size=bucket_size,
                            result=result,
                        )
                    )
            for relation in stable:
                relation_type = relation.relation_type
                stats = per_relation[relation_type]
                stats["stable_poi_count"] += 1
                relation_value_counts[relation_type][relation.numeric_value] += 1
                relation_sources[relation_type][relation.source_field] += 1
                if not 0 <= relation.numeric_value <= value_max:
                    stats["out_of_vocab_poi_count"] += 1
                    if (
                        len(out_of_vocab_examples[relation_type])
                        < examples_per_relation
                    ):
                        out_of_vocab_examples[relation_type].append(
                            _example_payload(
                                poi=poi,
                                base_sid=base_sid,
                                bucket_size=bucket_size,
                                result=result,
                            )
                        )
                else:
                    stats["in_vocab_poi_count"] += 1
                    if order_count > 0:
                        stats["in_vocab_train_order_count"] += order_count
                if len(examples[relation_type]) < examples_per_relation:
                    examples[relation_type].append(
                        _example_payload(
                            poi=poi,
                            base_sid=base_sid,
                            bucket_size=bucket_size,
                            result=result,
                        )
                    )

            category_key = (
                str(poi.get("category_code", "")),
                str(poi.get("category", "")),
            )
            category = category_metrics[category_key]
            category["colliding_poi_count"] += 1
            category["stable_poi_count"] += int(bool(stable))
            category["in_vocab_poi_count"] += int(bool(in_vocab))
            category["conflicted_poi_count"] += int(bool(result.conflicts))
            category["train_order_count"] += order_count
            category["in_vocab_train_order_count"] += order_count * int(bool(in_vocab))
        if progress is not None:
            progress(
                f"已扫描 {scanned_rows:,} / "
                f"{min(expected_rows, max_rows or expected_rows):,} 行；"
                f"碰撞 POI {colliding_pois:,}；词表内关系覆盖 {in_vocab_pois:,}"
            )
        if stop:
            break

    is_partial = max_rows is not None and scanned_rows < expected_rows
    if not is_partial:
        if scanned_rows != expected_rows:
            raise QgrSidAuditError(
                f"扫描行数不守恒：实际 {scanned_rows}，期望 {expected_rows}"
            )
        try:
            extra = next(raw_iterator)
        except StopIteration:
            extra = None
        if extra is not None:
            raise QgrSidAuditError("POI 原始表行数多于 TIGER mapping")
        expected_colliding = int(tiger_metrics["base_sid_colliding_poi_count"])
        expected_buckets = int(tiger_metrics["base_sid_collision_bucket_count"])
        if colliding_pois != expected_colliding:
            raise QgrSidAuditError(
                f"碰撞 POI 数不守恒：实际 {colliding_pois}，期望 {expected_colliding}"
            )
        if len(collision_buckets) != expected_buckets:
            raise QgrSidAuditError(
                "碰撞桶数不守恒："
                f"实际 {len(collision_buckets)}，期望 {expected_buckets}"
            )

    per_relation_payload: dict[str, Any] = {}
    for relation_type in RELATION_TYPES:
        stats = per_relation[relation_type]
        stable_count = stats["stable_poi_count"]
        value_counts = relation_value_counts[relation_type]
        out_of_vocab_value_counts = {
            value: count
            for value, count in value_counts.items()
            if not 0 <= value <= value_max
        }
        per_relation_payload[relation_type] = {
            "stable_poi_count": stable_count,
            "stable_poi_ratio": _ratio(stable_count, colliding_pois),
            "in_vocab_poi_count": stats["in_vocab_poi_count"],
            "out_of_vocab_poi_count": stats["out_of_vocab_poi_count"],
            "conflicted_poi_count": stats["conflicted_poi_count"],
            "unique_numeric_value_count": len(value_counts),
            "min_numeric_value": (
                min(value_counts) if value_counts else None
            ),
            "max_numeric_value": (
                max(value_counts) if value_counts else None
            ),
            "top_numeric_values": [
                {"numeric_value": value, "poi_count": count}
                for value, count in sorted(
                    value_counts.items(), key=lambda item: (-item[1], item[0])
                )[:20]
            ],
            "top_out_of_vocab_values": [
                {"numeric_value": value, "poi_count": count}
                for value, count in sorted(
                    out_of_vocab_value_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:20]
            ],
            "selected_source_counts": dict(
                sorted(relation_sources[relation_type].items())
            ),
            "in_vocab_train_order_count": stats["in_vocab_train_order_count"],
        }

    category_payload = []
    for (category_code, category_name), stats in sorted(
        category_metrics.items(),
        key=lambda item: (-item[1]["colliding_poi_count"], item[0]),
    ):
        total = stats["colliding_poi_count"]
        category_payload.append(
            {
                "category_code": category_code,
                "category": category_name,
                "colliding_poi_count": total,
                "stable_poi_count": stats["stable_poi_count"],
                "in_vocab_poi_count": stats["in_vocab_poi_count"],
                "in_vocab_poi_ratio": _ratio(stats["in_vocab_poi_count"], total),
                "conflicted_poi_count": stats["conflicted_poi_count"],
                "train_order_count": stats["train_order_count"],
                "in_vocab_train_order_count": stats["in_vocab_train_order_count"],
                "in_vocab_train_order_ratio": _ratio(
                    stats["in_vocab_train_order_count"],
                    stats["train_order_count"],
                ),
            }
        )

    metrics: dict[str, Any] = {
        "schema_version": AUDIT_METRICS_SCHEMA_VERSION,
        "status": "partial" if is_partial else "completed",
        "relation_schema_version": RELATION_SCHEMA_VERSION,
        "value_token_min": 0,
        "value_token_max": value_max,
        "scanned_poi_count": scanned_rows,
        "expected_full_poi_count": expected_rows,
        "colliding_poi_count": colliding_pois,
        "collision_bucket_count": len(collision_buckets),
        "stable_lexical_relation_poi_count": stable_pois,
        "stable_lexical_relation_poi_ratio": _ratio(stable_pois, colliding_pois),
        "in_vocab_relation_poi_count": in_vocab_pois,
        "in_vocab_relation_poi_ratio": _ratio(in_vocab_pois, colliding_pois),
        "conflicted_relation_poi_count": conflicted_pois,
        "conflicted_relation_poi_ratio": _ratio(conflicted_pois, colliding_pois),
        "collision_bucket_with_stable_relation_count": len(stable_buckets),
        "collision_bucket_with_in_vocab_relation_count": len(in_vocab_buckets),
        "collision_bucket_with_in_vocab_relation_ratio": _ratio(
            len(in_vocab_buckets), len(collision_buckets)
        ),
        "numeric_evidence_count": evidence_count,
        "stable_relation_count_per_poi": {
            str(key): value for key, value in sorted(stable_count_distribution.items())
        },
        "in_vocab_relation_count_per_poi": {
            str(key): value for key, value in sorted(in_vocab_count_distribution.items())
        },
        "per_relation": per_relation_payload,
        "query_weighted_coverage": (
            None
            if query_support is None
            else {
                "query_supported_collision_poi_count": query_supported_collision_pois,
                "query_supported_in_vocab_relation_poi_count": query_supported_in_vocab_pois,
                "query_supported_in_vocab_relation_poi_ratio": _ratio(
                    query_supported_in_vocab_pois,
                    query_supported_collision_pois,
                ),
                "collision_train_order_count": collision_train_orders,
                "in_vocab_relation_train_order_count": in_vocab_train_orders,
                "in_vocab_relation_train_order_ratio": _ratio(
                    in_vocab_train_orders, collision_train_orders
                ),
                "sqrt_p99_collision_weight": collision_sqrt_weight,
                "sqrt_p99_in_vocab_relation_weight": in_vocab_sqrt_weight,
                "sqrt_p99_in_vocab_relation_weight_ratio": _ratio(
                    in_vocab_sqrt_weight, collision_sqrt_weight
                ),
                "train_order_count_p99_clip": query_support.p99_train_order_count,
                "weight_rule": "sqrt(1 + min(train_order_count, P99))",
            }
        ),
        "category_breakdown": category_payload,
    }
    metrics_path = staging_dir / "metrics.json"
    _json_dump(metrics_path, metrics)

    examples_path = staging_dir / "relation_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for relation_type in RELATION_TYPES:
            for example_kind, selected_examples in (
                ("stable", examples[relation_type]),
                ("out_of_vocab", out_of_vocab_examples[relation_type]),
                ("conflict", conflict_examples[relation_type]),
            ):
                for example in selected_examples:
                    payload = {
                        "relation_type": relation_type,
                        "example_kind": example_kind,
                        **example,
                    }
                    handle.write(
                        json.dumps(payload, ensure_ascii=False, sort_keys=True)
                    )
                    handle.write("\n")

    input_signature_payload = {
        "tiger_mapping_sha256": mapping_sha256,
        "raw_sources": completed_sources,
        "query_manifest_sha256": (
            None if query_support is None else query_support.manifest_sha256
        ),
        "value_max": value_max,
        "max_rows": max_rows,
        "relation_schema_version": RELATION_SCHEMA_VERSION,
    }
    finished_at = _utc_now()
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "partial" if is_partial else "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "signature": _audit_signature(input_signature_payload),
        "configuration": {
            "value_token_min": 0,
            "value_token_max": value_max,
            "batch_rows": batch_rows,
            "examples_per_relation": examples_per_relation,
            "max_rows": max_rows,
            "scope": "TIGER collision POIs only",
            "query_usage": "Train-only aggregate weights for audit; never changes POI SID",
        },
        "inputs": {
            "poi_dir": str(poi_dir),
            "raw_sources": completed_sources,
            "tiger_identifier_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "tiger_identifier_metrics": {
                "path": str(tiger_metrics_path),
                "sha256": sha256_file(tiger_metrics_path),
            },
            "tiger_mapping": {
                "path": str(mapping_path),
                "sha256": mapping_sha256,
                "rows": expected_rows,
            },
            "query_aggregates": (
                None
                if query_support is None
                else {
                    "manifest": str(query_support.manifest_path),
                    "manifest_sha256": query_support.manifest_sha256,
                    "files": query_support.input_files,
                }
            ),
        },
        "outputs": {
            "metrics": {
                "path": "metrics.json",
                "sha256": sha256_file(metrics_path),
            },
            "relation_examples": {
                "path": "relation_examples.jsonl",
                "sha256": sha256_file(examples_path),
            },
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "validation": {
            "poi_mapping_row_alignment": True,
            "full_row_count_conserved": not is_partial,
            "collision_poi_count_conserved": not is_partial,
            "collision_bucket_count_conserved": not is_partial,
            "output_is_partial_smoke": is_partial,
        },
    }
    manifest_output_path = staging_dir / "manifest.json"
    _json_dump(manifest_output_path, manifest)
    (staging_dir / "_SUCCESS").touch()
    staging_dir.replace(output_dir)
    return AuditResult(metrics=metrics, manifest=manifest, output_dir=output_dir)


def validate_audit_output(output_dir: Path) -> dict[str, Any]:
    """Validate a completed audit directory without re-running extraction."""

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = _load_json(manifest_path, "QGR-SID audit manifest")
    if manifest.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise QgrSidAuditError("QGR-SID audit schema_version 不受支持")
    if manifest.get("status") != "completed":
        raise QgrSidAuditError("仅支持验证 completed 正式审计输出")
    if not (output_dir / "_SUCCESS").is_file():
        raise QgrSidAuditError("QGR-SID audit 缺少 _SUCCESS")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise QgrSidAuditError("QGR-SID audit manifest 缺少 outputs")
    checked: dict[str, str] = {}
    for key in ("metrics", "relation_examples"):
        contract = outputs.get(key)
        if not isinstance(contract, dict):
            raise QgrSidAuditError(f"QGR-SID audit manifest 缺少 {key}")
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file():
            raise QgrSidAuditError(f"QGR-SID audit 输出不存在：{path}")
        digest = sha256_file(path)
        if digest != contract.get("sha256"):
            raise QgrSidAuditError(f"QGR-SID audit 输出 SHA256 不一致：{key}")
        checked[key] = digest
    return {
        "status": "validated",
        "schema_version": manifest["schema_version"],
        "signature": manifest["signature"],
        "checked_outputs": checked,
    }

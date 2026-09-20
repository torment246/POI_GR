"""Offline evaluation of lightweight POI ranking over generated SID buckets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import resource
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from poi_gr.methods.tiger.eval import TigerIdIndex, load_tiger_id_index
from poi_gr.methods.tiger.identifier import sha256_file


SCHEMA_VERSION = "bucket-rerank-eval-v1"
METRICS_SCHEMA_VERSION = "bucket-rerank-metrics-v1"
METRIC_KS = (1, 3, 5, 10)
RRF_K = 60
RANKING_VARIANTS = (
    "bucket_then_popularity",
    "bucket_then_lexical",
    "bucket_then_semantic",
    "global_popularity",
    "global_lexical",
    "global_semantic",
    "global_rrf_content",
    "global_rrf_all",
)
LEGACY_RANKING_VARIANTS = (
    "bucket_then_popularity",
    "bucket_then_lexical",
    "bucket_then_bge",
    "global_popularity",
    "global_lexical",
    "global_bge",
    "global_rrf_content",
    "global_rrf_all",
)
SUPPORTED_RANKING_VARIANTS = frozenset(RANKING_VARIANTS + LEGACY_RANKING_VARIANTS)
OUTPUT_FILENAMES = (
    "metrics.json",
    "rankings.npz",
    "comparison_cases.jsonl",
    "manifest.json",
    "_SUCCESS",
)
_ALIAS_SPLIT = re.compile(r"[|｜,，;；/、]+")


class BucketRerankError(ValueError):
    """Raised when bucket-rerank inputs violate the frozen protocol."""


@dataclass(frozen=True)
class PoiMetadata:
    """Text and coordinates needed by the first resolver diagnostic."""

    row: int
    poi_id: str
    displayname: str
    address: str
    alias: str
    category: str
    lng: float | None
    lat: float | None


@dataclass(frozen=True)
class ExpandedCandidate:
    """One POI expanded from a generated three-token semantic bucket."""

    row: int
    bucket_order: int
    first_beam_rank: int
    bucket_score: float


@dataclass(frozen=True)
class EvaluationResult:
    """Completed bucket-rerank evaluation artifacts."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BucketRerankError(f"无法读取{name}：{path}") from error
    if not isinstance(payload, dict):
        raise BucketRerankError(f"{name}必须是 JSON object：{path}")
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


def normalize_text(value: Any) -> str:
    """Normalize text for deterministic character-level POI matching."""

    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return "".join(character.lower() for character in text if character.isalnum())


def _text_similarity(query: str, text: str) -> float:
    if not query or not text:
        return 0.0
    if query == text:
        return 1.0
    if query in text:
        return 0.9 + 0.1 * len(query) / len(text)
    if text in query:
        return 0.85 + 0.15 * len(text) / len(query)
    return float(SequenceMatcher(None, query, text, autojunk=False).ratio())


def lexical_relevance(query: str, metadata: PoiMetadata) -> float:
    """Score name/alias first, with address and category as weaker evidence."""

    normalized_query = normalize_text(query)
    if not normalized_query:
        return 0.0
    entity_values = [normalize_text(metadata.displayname)]
    entity_values.extend(
        normalize_text(value)
        for value in _ALIAS_SPLIT.split(metadata.alias)
        if value.strip()
    )
    entity_score = max(
        (_text_similarity(normalized_query, value) for value in entity_values),
        default=0.0,
    )
    address_score = 0.8 * _text_similarity(
        normalized_query, normalize_text(metadata.address)
    )
    category_score = 0.6 * _text_similarity(
        normalized_query, normalize_text(metadata.category)
    )
    return max(entity_score, address_score, category_score)


def canonical_key(metadata: PoiMetadata) -> tuple[Any, ...] | None:
    """Return the strict duplicate signature used for canonical evaluation."""

    name = normalize_text(metadata.displayname)
    if not name or metadata.lng is None or metadata.lat is None:
        return None
    aliases = tuple(
        sorted(
            {
                normalized
                for value in _ALIAS_SPLIT.split(metadata.alias)
                if (normalized := normalize_text(value))
            }
        )
    )
    return (
        name,
        normalize_text(metadata.address),
        aliases,
        normalize_text(metadata.category),
        float(metadata.lng).hex(),
        float(metadata.lat).hex(),
    )


def expand_trace_buckets(
    trace: Mapping[str, Any], index: TigerIdIndex
) -> tuple[ExpandedCandidate, ...]:
    """Expand every unique valid trace bucket into distinct corpus POI rows."""

    buckets = trace.get("unique_expandable_buckets")
    if not isinstance(buckets, list):
        raise BucketRerankError("候选轨迹缺少 unique_expandable_buckets")
    expanded: list[ExpandedCandidate] = []
    seen_rows: set[int] = set()
    previous_beam_rank = 0
    for bucket_order, value in enumerate(buckets, start=1):
        if not isinstance(value, Mapping):
            raise BucketRerankError("唯一 Bucket 必须是 JSON object")
        codes = value.get("codes")
        if not isinstance(codes, list) or len(codes) != 3:
            raise BucketRerankError("唯一 Bucket codes 必须包含三层")
        first_beam_rank = int(value.get("first_beam_rank", 0))
        if first_beam_rank <= previous_beam_rank:
            raise BucketRerankError("唯一 Bucket 必须按首次 Beam rank 严格递增")
        previous_beam_rank = first_beam_rank
        start, stop = index.bucket_bounds(codes)
        stored_size = int(value.get("bucket_size", -1))
        if stop - start != stored_size or stored_size <= 0:
            raise BucketRerankError("候选轨迹 Bucket size 与冻结目录不一致")
        score = float(value.get("first_sequence_score"))
        if not math.isfinite(score):
            raise BucketRerankError("候选 Bucket 分数必须有限")
        for row_value in index.sorted_rows[start:stop]:
            row = int(row_value)
            if row in seen_rows:
                raise BucketRerankError("不同语义 Bucket 展开出重复 POI 行")
            seen_rows.add(row)
            expanded.append(
                ExpandedCandidate(
                    row=row,
                    bucket_order=bucket_order,
                    first_beam_rank=first_beam_rank,
                    bucket_score=score,
                )
            )
    return tuple(expanded)


def _ordinal_ranks(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    order = np.lexsort((rows, -values))
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int32)
    return ranks


def rank_candidates(
    candidates: Sequence[ExpandedCandidate],
    *,
    lexical_scores: np.ndarray,
    semantic_scores: np.ndarray,
    popularity: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Return candidate indices ordered by one predeclared resolver variant."""

    if variant not in SUPPORTED_RANKING_VARIANTS:
        raise BucketRerankError(f"未知排序变体：{variant}")
    normalized_variant = {
        "bucket_then_bge": "bucket_then_semantic",
        "global_bge": "global_semantic",
    }.get(variant, variant)
    count = len(candidates)
    for name, values in (
        ("lexical_scores", lexical_scores),
        ("semantic_scores", semantic_scores),
        ("popularity", popularity),
    ):
        if values.shape != (count,):
            raise BucketRerankError(f"{name} shape 必须为 [{count}]")
        if not np.isfinite(values).all():
            raise BucketRerankError(f"{name} 存在 NaN/Inf")
    if not count:
        return np.empty(0, dtype=np.int64)

    rows = np.fromiter((item.row for item in candidates), dtype=np.int64, count=count)
    bucket_order = np.fromiter(
        (item.bucket_order for item in candidates), dtype=np.int32, count=count
    )
    if normalized_variant == "bucket_then_popularity":
        return np.lexsort((rows, -popularity, bucket_order))
    if normalized_variant == "bucket_then_lexical":
        return np.lexsort((rows, -popularity, -lexical_scores, bucket_order))
    if normalized_variant == "bucket_then_semantic":
        return np.lexsort((rows, -popularity, -semantic_scores, bucket_order))
    if normalized_variant == "global_popularity":
        return np.lexsort((rows, bucket_order, -popularity))
    if normalized_variant == "global_lexical":
        return np.lexsort((rows, -popularity, bucket_order, -lexical_scores))
    if normalized_variant == "global_semantic":
        return np.lexsort((rows, -popularity, bucket_order, -semantic_scores))

    lexical_rank = _ordinal_ranks(lexical_scores, rows)
    semantic_rank = _ordinal_ranks(semantic_scores, rows)
    rrf = 1.0 / (RRF_K + bucket_order)
    rrf += 1.0 / (RRF_K + lexical_rank)
    rrf += 1.0 / (RRF_K + semantic_rank)
    if normalized_variant == "global_rrf_all":
        popularity_rank = _ordinal_ranks(popularity, rows)
        rrf += 1.0 / (RRF_K + popularity_rank)
    return np.lexsort((rows, -popularity, -rrf))


def _rank_of_row(target_row: int, ordered_rows: Sequence[int]) -> int:
    for rank, row in enumerate(ordered_rows, start=1):
        if int(row) == target_row:
            return rank
    return 0


def _rank_of_canonical(
    target_row: int,
    ordered_rows: Sequence[int],
    metadata: Mapping[int, PoiMetadata],
) -> int:
    target_key = canonical_key(metadata[target_row])
    if target_key is None:
        return _rank_of_row(target_row, ordered_rows)
    for rank, row_value in enumerate(ordered_rows, start=1):
        row = int(row_value)
        if row == target_row or canonical_key(metadata[row]) == target_key:
            return rank
    return 0


def _metrics_from_ranks(ranks: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    selected = ranks if mask is None else ranks[mask]
    count = int(len(selected))
    metrics: dict[str, Any] = {"samples": count}
    for k in METRIC_KS:
        hits = (selected > 0) & (selected <= k)
        metrics[f"hr@{k}"] = 0.0 if not count else float(hits.mean())
        discounts = np.zeros(count, dtype=np.float64)
        discounts[hits] = 1.0 / np.log2(selected[hits] + 1)
        metrics[f"ndcg@{k}"] = 0.0 if not count else float(discounts.mean())
    reciprocal = np.zeros(count, dtype=np.float64)
    reciprocal_hits = (selected > 0) & (selected <= 10)
    reciprocal[reciprocal_hits] = 1.0 / selected[reciprocal_hits]
    metrics["mrr@10"] = 0.0 if not count else float(reciprocal.mean())
    return metrics


def _paired_metrics(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in (1, 10):
        candidate_hit = (candidate > 0) & (candidate <= k)
        baseline_hit = (baseline > 0) & (baseline <= k)
        result[f"hr@{k}"] = {
            "wins": int(np.sum(candidate_hit & ~baseline_hit)),
            "losses": int(np.sum(~candidate_hit & baseline_hit)),
            "ties": int(np.sum(candidate_hit == baseline_hit)),
            "net_hits": int(candidate_hit.sum() - baseline_hit.sum()),
            "delta": float(candidate_hit.mean() - baseline_hit.mean()),
        }
    return result


def _nearest_rank(values: np.ndarray, quantile: float) -> int:
    if not len(values):
        return 0
    return int(np.percentile(values, quantile * 100, method="higher"))


def _load_popularity(query_aggregate_dir: Path, poi_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = query_aggregate_dir / "manifest.json"
    manifest = _load_json(manifest_path, "Query 聚合 manifest")
    if manifest.get("status") != "completed" or manifest.get("schema_version") != "query-poi-aggregates-v1":
        raise BucketRerankError("Query 聚合 manifest 状态或版本不兼容")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise BucketRerankError("Query 聚合 manifest 缺少 outputs")
    arrays: dict[str, np.ndarray] = {}
    contracts: dict[str, Any] = {}
    for name in ("covered_poi_rows", "train_order_count"):
        contract = outputs.get(name)
        if not isinstance(contract, Mapping):
            raise BucketRerankError(f"Query 聚合缺少 {name}")
        path = query_aggregate_dir / str(contract.get("file", ""))
        if not path.is_file() or sha256_file(path) != contract.get("sha256"):
            raise BucketRerankError(f"Query 聚合 {name} 文件或 SHA256 不一致")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != tuple(contract.get("shape", ())) or str(array.dtype) != contract.get("dtype"):
            raise BucketRerankError(f"Query 聚合 {name} shape/dtype 不一致")
        arrays[name] = array
        contracts[name] = {
            "path": str(path.resolve()),
            "sha256": contract.get("sha256"),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    covered = np.asarray(arrays["covered_poi_rows"], dtype=np.int64)
    counts = np.asarray(arrays["train_order_count"], dtype=np.int64)
    if len(covered) != len(counts) or len(np.unique(covered)) != len(covered):
        raise BucketRerankError("Query 聚合覆盖行与订单计数不守恒")
    if len(covered) and (int(covered.min()) < 0 or int(covered.max()) >= poi_count):
        raise BucketRerankError("Query 聚合 POI 行号越界")
    if np.any(counts <= 0):
        raise BucketRerankError("Train 订单计数必须为正数")
    popularity = np.zeros(poi_count, dtype=np.float64)
    popularity[covered] = np.log1p(counts)
    return popularity, {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "covered_pois": int(len(covered)),
        "arrays": contracts,
        "transform": "log1p(train_order_count), uncovered POI=0",
    }


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_selected_metadata(
    *,
    poi_dir: Path,
    selected_rows: set[int],
    index: TigerIdIndex,
) -> tuple[dict[int, PoiMetadata], dict[str, Any]]:
    paths = tuple(sorted(poi_dir.glob("part-*.json")))
    if not paths:
        raise BucketRerankError(f"POI 目录中没有 part-*.json：{poi_dir}")
    mask = np.zeros(index.row_count, dtype=np.bool_)
    selected = np.fromiter(sorted(selected_rows), dtype=np.int64)
    if len(selected) and (int(selected.min()) < 0 or int(selected.max()) >= index.row_count):
        raise BucketRerankError("待加载 POI 行号越界")
    mask[selected] = True
    metadata: dict[int, PoiMetadata] = {}
    row = 0
    sources: list[dict[str, Any]] = []
    for path in paths:
        source_rows = 0
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise BucketRerankError(f"POI 源文件存在空行：{path}:{line_number}")
                if row >= index.row_count:
                    raise BucketRerankError("POI 原始数据长于 TIGER 目录")
                if mask[row]:
                    try:
                        value = json.loads(raw_line)
                    except json.JSONDecodeError as error:
                        raise BucketRerankError(f"POI JSON 非法：{path}:{line_number}") from error
                    if not isinstance(value, Mapping):
                        raise BucketRerankError(f"POI JSON 行不是 object：{path}:{line_number}")
                    poi_id = str(value.get("poi_id", ""))
                    if not poi_id or poi_id != index.poi_id(row):
                        raise BucketRerankError(f"POI/TIGER 行对齐失败：row={row}")
                    metadata[row] = PoiMetadata(
                        row=row,
                        poi_id=poi_id,
                        displayname=str(value.get("displayname") or ""),
                        address=str(value.get("address") or ""),
                        alias=str(value.get("alias") or ""),
                        category=str(value.get("category") or ""),
                        lng=_optional_float(value.get("lng")),
                        lat=_optional_float(value.get("lat")),
                    )
                row += 1
                source_rows += 1
        sources.append(
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "rows": source_rows}
        )
    if row != index.row_count:
        raise BucketRerankError(f"POI 原始行数 {row:,} != TIGER 目录 {index.row_count:,}")
    if set(metadata) != selected_rows:
        raise BucketRerankError("未完整加载候选与目标 POI 元数据")
    return metadata, {
        "directory": str(poi_dir.resolve()),
        "catalog_rows": row,
        "selected_rows": len(metadata),
        "sources": sources,
        "alignment": "selected poi_id rows exactly match TIGER mapping",
    }


def _validate_embedding_inputs(
    *,
    query_embeddings_path: Path,
    query_run_manifest_path: Path,
    poi_embeddings_path: Path,
    poi_embedding_manifest_path: Path,
    expected_rows: int,
    expected_pois: int,
    semantic_label: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    query_manifest = _load_json(query_run_manifest_path, "BGE Query run manifest")
    if query_manifest.get("status") != "completed" or query_manifest.get("model") != "bge_m3":
        raise BucketRerankError("BGE Query run manifest 状态或模型不兼容")
    expected_query_hash = query_manifest.get("outputs", {}).get("query_embeddings_sha256")
    if not query_embeddings_path.is_file() or sha256_file(query_embeddings_path) != expected_query_hash:
        raise BucketRerankError("BGE Query embedding 文件或 SHA256 不一致")
    query_embeddings = np.load(query_embeddings_path, mmap_mode="r", allow_pickle=False)
    if query_embeddings.shape[0] < expected_rows or query_embeddings.ndim != 2 or query_embeddings.dtype != np.float16:
        raise BucketRerankError("BGE Query embedding shape/dtype 不兼容")

    poi_manifest = _load_json(poi_embedding_manifest_path, "POI 向量 manifest")
    expected_shape = tuple(poi_manifest.get("output", {}).get("shape", ()))
    if poi_manifest.get("status") != "completed" or expected_shape[0:1] != (expected_pois,):
        raise BucketRerankError("POI 向量 manifest 状态或行数不兼容")
    if not poi_embeddings_path.is_file():
        raise BucketRerankError(f"POI 向量不存在：{poi_embeddings_path}")
    poi_embeddings = np.load(poi_embeddings_path, mmap_mode="r", allow_pickle=False)
    if poi_embeddings.shape != expected_shape or poi_embeddings.dtype != np.float16:
        raise BucketRerankError("POI 向量 shape/dtype 与 manifest 不一致")
    if query_embeddings.shape[1] != poi_embeddings.shape[1]:
        raise BucketRerankError("Query/POI 向量维度不一致")
    return query_embeddings, poi_embeddings, {
        "query_embeddings": str(query_embeddings_path.resolve()),
        "query_embeddings_sha256": expected_query_hash,
        "query_run_manifest": str(query_run_manifest_path.resolve()),
        "query_run_manifest_sha256": sha256_file(query_run_manifest_path),
        "poi_embeddings": str(poi_embeddings_path.resolve()),
        "poi_embedding_manifest": str(poi_embedding_manifest_path.resolve()),
        "poi_embedding_manifest_sha256": sha256_file(poi_embedding_manifest_path),
        "query_shape": list(query_embeddings.shape),
        "poi_shape": list(poi_embeddings.shape),
        "dtype": "float16",
        "semantic_label": semantic_label,
        "score": (
            "float32 dot product over frozen L2-normalized BGE-M3 query and "
            f"{semantic_label} POI vectors"
        ),
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BucketRerankError(f"JSONL 非法：{path}:{line_number}") from error
            if not isinstance(value, dict):
                raise BucketRerankError(f"JSONL 行不是 object：{path}:{line_number}")
            yield value


def _source_exact_ranks(traces: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.zeros(len(traces), dtype=np.int32)
    for index, trace in enumerate(traces):
        rank = trace.get("exact_target_rank")
        if rank is not None:
            values[index] = int(rank)
    return values


def _comparison_case(
    *,
    row_index: int,
    query: str,
    target_row: int,
    baseline_rank: int,
    candidate_rank: int,
    top_rows: Sequence[int],
    metadata: Mapping[int, PoiMetadata],
) -> dict[str, Any]:
    target = metadata[target_row]
    return {
        "row_index": row_index,
        "query": query,
        "target": {
            "poi_id": target.poi_id,
            "displayname": target.displayname,
            "address": target.address,
        },
        "original_exact_rank": baseline_rank or None,
        "global_rrf_content_rank": candidate_rank or None,
        "global_rrf_content_top10": [
            {
                "poi_id": metadata[int(candidate_row)].poi_id,
                "displayname": metadata[int(candidate_row)].displayname,
                "address": metadata[int(candidate_row)].address,
            }
            for candidate_row in top_rows
            if int(candidate_row) >= 0
        ],
    }


def evaluate_bucket_rerank(
    *,
    project_root: Path,
    candidate_trace_path: Path,
    candidate_trace_manifest_path: Path,
    source_result_path: Path,
    eval_data_path: Path,
    query_mapping_path: Path,
    query_embeddings_path: Path,
    query_run_manifest_path: Path,
    poi_embeddings_path: Path,
    poi_embedding_manifest_path: Path,
    identifier_dir: Path,
    poi_dir: Path,
    query_aggregate_dir: Path,
    output_dir: Path,
    max_rows: int | None = None,
    examples_per_kind: int = 10,
    generator_label: str = "TIGER epoch 3 unconstrained Beam=10 trace",
    semantic_label: str = "BGE-M3",
) -> EvaluationResult:
    """Evaluate deterministic no-training resolvers on one frozen SID trace."""

    if max_rows is not None and max_rows <= 0:
        raise BucketRerankError("max_rows 必须大于 0")
    if examples_per_kind < 0:
        raise BucketRerankError("examples_per_kind 不能为负数")
    if not generator_label.strip() or not semantic_label.strip():
        raise BucketRerankError("generator_label 和 semantic_label 不能为空")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise BucketRerankError(f"输出目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "git": _git_state(project_root),
        "config": {
            "ranking_variants": list(RANKING_VARIANTS),
            "rrf_k": RRF_K,
            "top_k": 10,
            "max_rows": max_rows,
            "lexical": "NFKC lowercase alnum; name/alias first, address×0.8, category×0.6",
            "geography_used": False,
            "training_used": False,
            "generator_label": generator_label,
            "semantic_label": semantic_label,
        },
    }
    _atomic_json(manifest_path, manifest)

    try:
        trace_manifest = _load_json(candidate_trace_manifest_path, "候选轨迹 manifest")
        if trace_manifest.get("status") != "completed" or trace_manifest.get("schema_version") != "tiger-candidate-trace-v3":
            raise BucketRerankError("候选轨迹 manifest 状态或版本不兼容")
        if sha256_file(candidate_trace_path) != trace_manifest.get("output_sha256"):
            raise BucketRerankError("候选轨迹 SHA256 与 manifest 不一致")
        source_result = _load_json(source_result_path, "TIGER 来源结果")
        index, identifier_contract = load_tiger_id_index(identifier_dir)

        trace_iterator = _iter_jsonl(candidate_trace_path)
        eval_iterator = _iter_jsonl(eval_data_path)
        mapping_iterator = _iter_jsonl(query_mapping_path)
        traces: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        expanded_rows: list[tuple[ExpandedCandidate, ...]] = []
        selected_rows: set[int] = set()
        for row_index, (trace, eval_row, query_map) in enumerate(
            zip(trace_iterator, eval_iterator, mapping_iterator, strict=True)
        ):
            if max_rows is not None and row_index >= max_rows:
                break
            for name, left, right in (
                ("order_id", trace.get("order_id"), eval_row.get("order_id")),
                ("order_id/query_mapping", trace.get("order_id"), query_map.get("order_id")),
                ("target_poi_id", trace.get("target_poi_id"), eval_row.get("poi_id")),
                ("target_poi_id/query_mapping", trace.get("target_poi_id"), query_map.get("target_poi_id")),
            ):
                if str(left) != str(right):
                    raise BucketRerankError(f"固定评测行 {row_index} 的 {name} 不一致")
            if int(trace.get("row_index", -1)) != row_index or int(query_map.get("row_index", -1)) != row_index:
                raise BucketRerankError(f"固定评测 row_index 不连续：{row_index}")
            target_row = int(trace.get("target_poi_row", -1))
            if target_row < 0 or index.poi_id(target_row) != str(trace.get("target_poi_id")):
                raise BucketRerankError(f"固定评测目标 POI 行不一致：{row_index}")
            expanded = expand_trace_buckets(trace, index)
            target_in_pool = any(item.row == target_row for item in expanded)
            if target_in_pool != (trace.get("unique_bucket_target_rank") is not None):
                raise BucketRerankError(f"目标 Bucket 展开命中状态不一致：{row_index}")
            traces.append(trace)
            eval_rows.append(eval_row)
            expanded_rows.append(expanded)
            selected_rows.add(target_row)
            selected_rows.update(item.row for item in expanded)
        if not traces:
            raise BucketRerankError("固定评测没有可用行")
        expected_full_rows = int(trace_manifest.get("rows", 0))
        if max_rows is None and len(traces) != expected_full_rows:
            raise BucketRerankError("候选轨迹完整行数不一致")

        query_embeddings, poi_embeddings, embedding_contract = _validate_embedding_inputs(
            query_embeddings_path=query_embeddings_path,
            query_run_manifest_path=query_run_manifest_path,
            poi_embeddings_path=poi_embeddings_path,
            poi_embedding_manifest_path=poi_embedding_manifest_path,
            expected_rows=len(traces),
            expected_pois=index.row_count,
            semantic_label=semantic_label,
        )
        popularity, popularity_contract = _load_popularity(query_aggregate_dir, index.row_count)
        metadata, metadata_contract = _load_selected_metadata(
            poi_dir=poi_dir,
            selected_rows=selected_rows,
            index=index,
        )

        row_count = len(traces)
        baseline_exact = _source_exact_ranks(traces)
        baseline_canonical = np.zeros(row_count, dtype=np.int32)
        exact_ranks = {
            variant: np.zeros(row_count, dtype=np.int32) for variant in RANKING_VARIANTS
        }
        canonical_ranks = {
            variant: np.zeros(row_count, dtype=np.int32) for variant in RANKING_VARIANTS
        }
        top10_rows = {
            variant: np.full((row_count, 10), -1, dtype=np.int32)
            for variant in RANKING_VARIANTS
        }
        candidate_counts = np.zeros(row_count, dtype=np.int32)
        target_bucket_sizes = np.zeros(row_count, dtype=np.int32)
        target_bucket_ranks = np.zeros(row_count, dtype=np.int32)
        target_in_pool = np.zeros(row_count, dtype=np.bool_)

        for row_index, (trace, eval_row, candidates) in enumerate(
            zip(traces, eval_rows, expanded_rows, strict=True)
        ):
            target_row = int(trace["target_poi_row"])
            target_bucket_sizes[row_index] = int(trace["target_bucket_size"])
            target_bucket_rank = trace.get("unique_bucket_target_rank")
            if target_bucket_rank is not None:
                target_bucket_ranks[row_index] = int(target_bucket_rank)
            candidate_counts[row_index] = len(candidates)
            target_in_pool[row_index] = any(item.row == target_row for item in candidates)

            baseline_rows = [
                int(value["poi_row"])
                for value in trace.get("candidates", [])
                if value.get("poi_row") is not None
            ]
            baseline_canonical[row_index] = _rank_of_canonical(
                target_row, baseline_rows, metadata
            )
            if not candidates:
                continue
            candidate_row_values = np.fromiter(
                (item.row for item in candidates), dtype=np.int64, count=len(candidates)
            )
            query_vector = np.asarray(query_embeddings[row_index], dtype=np.float32)
            candidate_vectors = np.asarray(
                poi_embeddings[candidate_row_values], dtype=np.float32
            )
            semantic_scores = candidate_vectors @ query_vector
            lexical_scores = np.asarray(
                [
                    lexical_relevance(str(eval_row.get("query", "")), metadata[item.row])
                    for item in candidates
                ],
                dtype=np.float64,
            )
            popularity_scores = popularity[candidate_row_values]
            for variant in RANKING_VARIANTS:
                order = rank_candidates(
                    candidates,
                    lexical_scores=lexical_scores,
                    semantic_scores=semantic_scores,
                    popularity=popularity_scores,
                    variant=variant,
                )
                ordered_rows = candidate_row_values[order]
                exact_ranks[variant][row_index] = _rank_of_row(target_row, ordered_rows)
                canonical_ranks[variant][row_index] = _rank_of_canonical(
                    target_row, ordered_rows, metadata
                )
                retained = ordered_rows[:10]
                top10_rows[variant][row_index, : len(retained)] = retained.astype(
                    np.int32, copy=False
                )

        collision_mask = target_bucket_sizes > 1
        singleton_mask = ~collision_mask
        target_bucket_first_mask = target_bucket_ranks == 1
        collision_bucket_first_mask = collision_mask & target_bucket_first_mask
        rankings: dict[str, Any] = {
            "original_c_token": {
                "exact": _metrics_from_ranks(baseline_exact),
                "canonical": _metrics_from_ranks(baseline_canonical),
                "subsets": {
                    "target_collision_bucket": _metrics_from_ranks(
                        baseline_exact, collision_mask
                    ),
                    "target_singleton_bucket": _metrics_from_ranks(
                        baseline_exact, singleton_mask
                    ),
                    "target_bucket_first": _metrics_from_ranks(
                        baseline_exact, target_bucket_first_mask
                    ),
                    "target_collision_bucket_first": _metrics_from_ranks(
                        baseline_exact, collision_bucket_first_mask
                    ),
                },
            }
        }
        for variant in RANKING_VARIANTS:
            rankings[variant] = {
                "exact": _metrics_from_ranks(exact_ranks[variant]),
                "canonical": _metrics_from_ranks(canonical_ranks[variant]),
                "conditional_target_in_pool": _metrics_from_ranks(
                    exact_ranks[variant], target_in_pool
                ),
                "subsets": {
                    "target_collision_bucket": _metrics_from_ranks(
                        exact_ranks[variant], collision_mask
                    ),
                    "target_singleton_bucket": _metrics_from_ranks(
                        exact_ranks[variant], singleton_mask
                    ),
                    "target_bucket_first": _metrics_from_ranks(
                        exact_ranks[variant], target_bucket_first_mask
                    ),
                    "target_collision_bucket_first": _metrics_from_ranks(
                        exact_ranks[variant], collision_bucket_first_mask
                    ),
                },
                "paired_vs_original_exact": _paired_metrics(
                    exact_ranks[variant], baseline_exact
                ),
            }

        source_metrics = source_result.get("metrics")
        if not isinstance(source_metrics, Mapping):
            raise BucketRerankError("来源结果缺少 metrics")
        reproduced = rankings["original_c_token"]["exact"]
        if max_rows is None:
            for k in METRIC_KS:
                if not math.isclose(
                    reproduced[f"hr@{k}"],
                    float(source_metrics[f"hr@{k}"]),
                    abs_tol=1e-12,
                ):
                    raise BucketRerankError(f"原始完整 ID HR@{k} 未能从轨迹精确复现")
            if not math.isclose(
                reproduced["ndcg@10"],
                float(source_metrics["ndcg@10"]),
                abs_tol=1e-12,
            ):
                raise BucketRerankError("原始完整 ID NDCG@10 未能从轨迹精确复现")

        metrics = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "status": "completed",
            "protocol": {
                "generator": generator_label,
                "semantic_signal": semantic_label,
                "candidate_pool": "deduplicated expandable [S1,S2,S3] buckets",
                "final_output": "global or bucket-preserving top-10 POI rows",
                "uses_test": False,
                "uses_geo": False,
                "uses_training": False,
            },
            "candidate_pool": {
                "samples": row_count,
                "target_recall": float(target_in_pool.mean()),
                "zero_candidate_samples": int(np.sum(candidate_counts == 0)),
                "candidate_count_mean": float(candidate_counts.mean()),
                "candidate_count_p50": _nearest_rank(candidate_counts, 0.50),
                "candidate_count_p90": _nearest_rank(candidate_counts, 0.90),
                "candidate_count_p95": _nearest_rank(candidate_counts, 0.95),
                "candidate_count_p99": _nearest_rank(candidate_counts, 0.99),
                "candidate_count_max": int(candidate_counts.max()),
                "selected_catalog_rows": len(selected_rows),
                "target_collision_bucket_samples": int(collision_mask.sum()),
                "target_singleton_bucket_samples": int(singleton_mask.sum()),
                "target_bucket_first_samples": int(target_bucket_first_mask.sum()),
                "target_collision_bucket_first_samples": int(
                    collision_bucket_first_mask.sum()
                ),
            },
            "rankings": rankings,
        }
        metrics_path = output_dir / "metrics.json"
        _atomic_json(metrics_path, metrics)

        arrays: dict[str, np.ndarray] = {
            "original_exact_ranks": baseline_exact,
            "original_canonical_ranks": baseline_canonical,
            "candidate_counts": candidate_counts,
            "target_bucket_sizes": target_bucket_sizes,
            "target_bucket_ranks": target_bucket_ranks,
            "target_in_pool": target_in_pool,
        }
        for variant in RANKING_VARIANTS:
            arrays[f"{variant}__exact_ranks"] = exact_ranks[variant]
            arrays[f"{variant}__canonical_ranks"] = canonical_ranks[variant]
            arrays[f"{variant}__top10_rows"] = top10_rows[variant]
        rankings_path = output_dir / "rankings.npz"
        _atomic_npz(rankings_path, arrays)

        primary = "global_rrf_content"
        primary_ranks = exact_ranks[primary]
        wins = np.flatnonzero((primary_ranks == 1) & (baseline_exact != 1))
        losses = np.flatnonzero((primary_ranks != 1) & (baseline_exact == 1))
        cases_path = output_dir / "comparison_cases.jsonl"
        with cases_path.open("w", encoding="utf-8") as handle:
            for kind, indices in (("top1_win", wins), ("top1_loss", losses)):
                for row_index_value in indices[:examples_per_kind]:
                    row_index = int(row_index_value)
                    case = _comparison_case(
                        row_index=row_index,
                        query=str(eval_rows[row_index].get("query", "")),
                        target_row=int(traces[row_index]["target_poi_row"]),
                        baseline_rank=int(baseline_exact[row_index]),
                        candidate_rank=int(primary_ranks[row_index]),
                        top_rows=top10_rows[primary][row_index],
                        metadata=metadata,
                    )
                    case["kind"] = kind
                    handle.write(json.dumps(case, ensure_ascii=False) + "\n")

        success_path = output_dir / "_SUCCESS"
        success_path.touch()
        elapsed = time.perf_counter() - started
        manifest.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "inputs": {
                    "candidate_trace": str(candidate_trace_path.resolve()),
                    "candidate_trace_sha256": trace_manifest.get("output_sha256"),
                    "candidate_trace_manifest": str(candidate_trace_manifest_path.resolve()),
                    "candidate_trace_manifest_sha256": sha256_file(candidate_trace_manifest_path),
                    "source_result": str(source_result_path.resolve()),
                    "source_result_sha256": sha256_file(source_result_path),
                    "eval_data": str(eval_data_path.resolve()),
                    "eval_data_sha256": sha256_file(eval_data_path),
                    "query_mapping": str(query_mapping_path.resolve()),
                    "query_mapping_sha256": sha256_file(query_mapping_path),
                    "identifier": identifier_contract,
                    "embeddings": embedding_contract,
                    "popularity": popularity_contract,
                    "poi_metadata": metadata_contract,
                },
                "rows": row_count,
                "outputs": {
                    "metrics": {
                        "path": str(metrics_path.resolve()),
                        "sha256": sha256_file(metrics_path),
                    },
                    "rankings": {
                        "path": str(rankings_path.resolve()),
                        "sha256": sha256_file(rankings_path),
                    },
                    "comparison_cases": {
                        "path": str(cases_path.resolve()),
                        "sha256": sha256_file(cases_path),
                    },
                    "success": str(success_path.resolve()),
                },
                "runtime": {
                    "elapsed_seconds": elapsed,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    "device": "CPU; frozen embeddings loaded by mmap",
                },
            }
        )
        _atomic_json(manifest_path, manifest)
        return EvaluationResult(metrics=metrics, manifest=manifest, output_dir=output_dir)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _atomic_json(manifest_path, manifest)
        raise


def validate_bucket_rerank_output(output_dir: Path) -> dict[str, Any]:
    """Validate terminal output files, hashes and array shapes."""

    manifest = _load_json(output_dir / "manifest.json", "桶内重排 manifest")
    metrics = _load_json(output_dir / "metrics.json", "桶内重排 metrics")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "completed":
        raise BucketRerankError("桶内重排 manifest 状态或版本不兼容")
    if metrics.get("schema_version") != METRICS_SCHEMA_VERSION or metrics.get("status") != "completed":
        raise BucketRerankError("桶内重排 metrics 状态或版本不兼容")
    rows = int(manifest.get("rows", 0))
    if rows <= 0 or int(metrics.get("candidate_pool", {}).get("samples", 0)) != rows:
        raise BucketRerankError("桶内重排行数不一致")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise BucketRerankError("桶内重排 manifest 缺少 outputs")
    for name in ("metrics", "rankings", "comparison_cases"):
        contract = outputs.get(name)
        if not isinstance(contract, Mapping):
            raise BucketRerankError(f"桶内重排缺少输出契约：{name}")
        path = Path(str(contract.get("path", "")))
        if not path.is_file() or sha256_file(path) != contract.get("sha256"):
            raise BucketRerankError(f"桶内重排输出文件或 SHA256 不一致：{name}")
    success_path = Path(str(outputs.get("success", "")))
    if not success_path.is_file():
        raise BucketRerankError("桶内重排缺少 _SUCCESS")
    arrays = np.load(Path(str(outputs["rankings"]["path"])), allow_pickle=False)
    if arrays["original_exact_ranks"].shape != (rows,):
        raise BucketRerankError("original_exact_ranks shape 不一致")
    configured_variants = manifest.get("config", {}).get("ranking_variants")
    if (
        not isinstance(configured_variants, list)
        or not configured_variants
        or any(value not in SUPPORTED_RANKING_VARIANTS for value in configured_variants)
    ):
        raise BucketRerankError("桶内重排 ranking_variants 契约不兼容")
    for variant in configured_variants:
        if arrays[f"{variant}__exact_ranks"].shape != (rows,):
            raise BucketRerankError(f"{variant} exact rank shape 不一致")
        if arrays[f"{variant}__top10_rows"].shape != (rows, 10):
            raise BucketRerankError(f"{variant} top10 shape 不一致")
    return {
        "status": "passed",
        "rows": rows,
        "ranking_variants": configured_variants,
        "metrics_sha256": outputs["metrics"]["sha256"],
        "rankings_sha256": outputs["rankings"]["sha256"],
    }


# Backward-compatible entry point for the first TIGER-only experiment.
evaluate_tiger_bucket_rerank = evaluate_bucket_rerank

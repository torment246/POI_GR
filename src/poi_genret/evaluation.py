"""Ranking metrics for POI retrieval experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _top_k(items: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        return []
    return list(items[:k])


def recall_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    """Compute Recall@K for one query."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = len(relevant_set.intersection(_top_k(ranked, k)))
    return hits / len(relevant_set)


def mrr_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    """Compute reciprocal rank at K for one query."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, poi_id in enumerate(_top_k(ranked, k), start=1):
        if poi_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance: Mapping[str, float] | Iterable[str], ranked: Sequence[str], k: int) -> float:
    """Compute NDCG@K for one query.

    ``relevance`` may be a mapping of ``poi_id -> graded label`` or an iterable
    of relevant POI IDs, in which case binary relevance is assumed.
    """
    if isinstance(relevance, Mapping):
        rel_by_poi = {str(poi_id): float(label) for poi_id, label in relevance.items() if float(label) > 0}
    else:
        rel_by_poi = {str(poi_id): 1.0 for poi_id in relevance}

    if not rel_by_poi or k <= 0:
        return 0.0

    def gain(label: float, rank: int) -> float:
        return (2.0**label - 1.0) / math.log2(rank + 1)

    dcg = 0.0
    for rank, poi_id in enumerate(_top_k(ranked, k), start=1):
        dcg += gain(rel_by_poi.get(str(poi_id), 0.0), rank)

    ideal_labels = sorted(rel_by_poi.values(), reverse=True)[:k]
    idcg = sum(gain(label, rank) for rank, label in enumerate(ideal_labels, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(
    qrels: Mapping[str, Mapping[str, float]],
    run: Mapping[str, Sequence[str]],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Evaluate a run over all queries in qrels."""
    results: dict[str, float] = {}
    query_ids = list(qrels.keys())
    if not query_ids:
        for k in ks:
            results[f"recall@{k}"] = 0.0
            results[f"mrr@{k}"] = 0.0
            results[f"ndcg@{k}"] = 0.0
        return results

    for k in ks:
        recall_total = 0.0
        mrr_total = 0.0
        ndcg_total = 0.0
        for query_id in query_ids:
            labels = qrels[query_id]
            ranked = run.get(query_id, [])
            relevant = [poi_id for poi_id, label in labels.items() if label > 0]
            recall_total += recall_at_k(relevant, ranked, k)
            mrr_total += mrr_at_k(relevant, ranked, k)
            ndcg_total += ndcg_at_k(labels, ranked, k)
        denom = len(query_ids)
        results[f"recall@{k}"] = recall_total / denom
        results[f"mrr@{k}"] = mrr_total / denom
        results[f"ndcg@{k}"] = ndcg_total / denom
    return results


def rows_to_qrels(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        qrels[str(row["query_id"])][str(row["poi_id"])] = float(row.get("label", 1))
    return dict(qrels)


def rows_to_run(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        query_id = str(row["query_id"])
        poi_id = str(row["poi_id"])
        if row.get("rank") not in (None, ""):
            sort_key = float(row["rank"])
        elif row.get("score") not in (None, ""):
            sort_key = -float(row["score"])
        else:
            sort_key = float(idx)
        grouped[query_id].append((sort_key, idx, poi_id))
    return {
        query_id: [poi_id for _, _, poi_id in sorted(items)]
        for query_id, items in grouped.items()
    }


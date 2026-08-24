"""Greedy, bounded relation-path compiler used by the QGR-SID M2 proxy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


MISSING_RELATION_VALUE = -1
MISSING_RELATION_TYPE = -1


class RelationCompilerError(ValueError):
    """Raised when relation compiler inputs violate the frozen M2 contract."""


@dataclass(frozen=True)
class BucketCompilation:
    """Variable relation paths for one base-SID collision bucket."""

    path_types: np.ndarray
    path_values: np.ndarray
    relation_only_resolved: np.ndarray
    decision_count: int


def branch_entropy_bits(values: np.ndarray, weights: np.ndarray) -> float:
    """Return weighted entropy of one deterministic branch assignment."""

    values = np.asarray(values)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or not len(values):
        raise RelationCompilerError("values/weights 必须是一维等长非空数组")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RelationCompilerError("weights 必须是有限正数")
    _, inverse = np.unique(values, return_inverse=True)
    branch_weights = np.bincount(inverse, weights=weights)
    probabilities = branch_weights / branch_weights.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _candidate_score(
    *,
    relation_values: np.ndarray,
    weights: np.ndarray,
    early_orders: np.ndarray,
    early_matches: np.ndarray,
    global_predictability: float,
    prior_orders: float,
    query_guided: bool,
    allow_presence_split: bool,
) -> float | None:
    available = relation_values >= 0
    distinct_values = np.unique(relation_values[available])
    if allow_presence_split:
        if not np.any(available) or len(np.unique(relation_values)) < 2:
            return None
    elif len(distinct_values) < 2:
        return None
    entropy = branch_entropy_bits(relation_values, weights)
    available_weight_ratio = float(weights[available].sum() / weights.sum())
    static_gain = entropy * available_weight_ratio
    if not query_guided:
        return static_gain

    support = float(early_orders[available].sum())
    matched = float(early_matches[available].sum())
    predictability = (
        matched + prior_orders * global_predictability
    ) / (support + prior_orders)
    return static_gain * predictability


def compile_relation_bucket(
    relations: np.ndarray,
    early_order_counts: np.ndarray,
    early_match_counts: np.ndarray,
    global_predictability: Sequence[float],
    *,
    query_guided: bool,
    max_pairs: int = 3,
    p99_order_count: float = 179.0,
    prior_orders: float = 20.0,
    allow_presence_split: bool = False,
) -> BucketCompilation:
    """Compile deterministic variable paths for one collision bucket.

    A chosen relation is emitted only by POIs that contain it. Its missing branch
    continues with another relation without consuming a relation-pair budget.
    Relations are never reused on one branch and paths never exceed ``max_pairs``.
    """

    relations = np.asarray(relations)
    early_order_counts = np.asarray(early_order_counts)
    early_match_counts = np.asarray(early_match_counts)
    global_predictability = np.asarray(global_predictability, dtype=np.float64)
    if relations.ndim != 2 or relations.shape[0] < 2:
        raise RelationCompilerError("碰撞桶 relations 必须是 [N,R] 且 N>=2")
    rows, relation_count = relations.shape
    if early_order_counts.shape != (rows,):
        raise RelationCompilerError("early_order_counts shape 必须是 [N]")
    if early_match_counts.shape != relations.shape:
        raise RelationCompilerError("early_match_counts shape 必须与 relations 相同")
    if global_predictability.shape != (relation_count,):
        raise RelationCompilerError("global_predictability shape 必须是 [R]")
    if max_pairs <= 0:
        raise RelationCompilerError("max_pairs 必须大于 0")
    if p99_order_count < 0 or prior_orders <= 0:
        raise RelationCompilerError("P99 与 prior_orders 参数非法")
    if np.any(early_order_counts < 0) or np.any(early_match_counts < 0):
        raise RelationCompilerError("早期请求统计不能为负数")
    if np.any(early_match_counts > early_order_counts[:, None]):
        raise RelationCompilerError("关系命中数不能超过 POI 早期订单数")
    if not np.isfinite(global_predictability).all() or np.any(
        (global_predictability < 0) | (global_predictability > 1)
    ):
        raise RelationCompilerError("global_predictability 必须位于 [0,1]")

    weights = np.maximum(
        1.0,
        np.sqrt(1.0 + np.minimum(early_order_counts, p99_order_count)),
    )
    path_types = np.full(
        (rows, max_pairs), MISSING_RELATION_TYPE, dtype=np.int16
    )
    path_values = np.full(
        (rows, max_pairs), MISSING_RELATION_VALUE, dtype=relations.dtype
    )
    decisions = 0

    def recurse(indices: np.ndarray, remaining: tuple[int, ...], depth: int) -> None:
        nonlocal decisions
        if len(indices) <= 1 or depth >= max_pairs or not remaining:
            return
        best_relation: int | None = None
        best_score = -math.inf
        for relation_index in remaining:
            score = _candidate_score(
                relation_values=relations[indices, relation_index],
                weights=weights[indices],
                early_orders=early_order_counts[indices],
                early_matches=early_match_counts[indices, relation_index],
                global_predictability=float(
                    global_predictability[relation_index]
                ),
                prior_orders=prior_orders,
                query_guided=query_guided,
                allow_presence_split=allow_presence_split,
            )
            if score is None:
                continue
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and (best_relation is None or relation_index < best_relation)
            ):
                best_relation = relation_index
                best_score = score
        if best_relation is None:
            return

        decisions += 1
        remaining_next = tuple(
            relation_index
            for relation_index in remaining
            if relation_index != best_relation
        )
        selected_values = relations[indices, best_relation]
        missing_indices = indices[selected_values < 0]
        if len(missing_indices) > 1:
            recurse(missing_indices, remaining_next, depth)

        for value in np.unique(selected_values[selected_values >= 0]):
            branch = indices[selected_values == value]
            path_types[branch, depth] = best_relation
            path_values[branch, depth] = value
            if len(branch) > 1:
                recurse(branch, remaining_next, depth + 1)

    active_relations = tuple(
        int(index)
        for index in np.flatnonzero(np.any(relations >= 0, axis=0))
    )
    recurse(np.arange(rows, dtype=np.int64), active_relations, 0)
    serialized = [
        tuple(
            (int(relation_type), int(value))
            for relation_type, value in zip(
                path_types[row], path_values[row], strict=True
            )
            if relation_type >= 0
        )
        for row in range(rows)
    ]
    path_counts: dict[tuple[tuple[int, int], ...], int] = {}
    for path in serialized:
        path_counts[path] = path_counts.get(path, 0) + 1
    resolved = np.asarray(
        [bool(path) and path_counts[path] == 1 for path in serialized],
        dtype=np.bool_,
    )
    return BucketCompilation(
        path_types=path_types,
        path_values=path_values,
        relation_only_resolved=resolved,
        decision_count=decisions,
    )

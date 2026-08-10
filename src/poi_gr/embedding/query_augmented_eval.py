"""Exact retrieval helpers for sparse Query-augmented POI embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np


class QueryAugmentedEvalError(RuntimeError):
    """Raised when sparse fusion or retrieval metrics are invalid."""


def category_residual_query_rows(
    query_rows: np.ndarray,
    category_indices: np.ndarray,
    category_query_means: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    """Remove a bounded category-common direction and L2 normalize rows."""

    rows = np.asarray(query_rows, dtype=np.float32)
    categories = np.asarray(category_indices)
    means = np.asarray(category_query_means, dtype=np.float32)
    if rows.ndim != 2 or means.ndim != 2 or rows.shape[1] != means.shape[1]:
        raise QueryAugmentedEvalError("E4 Query 行或类别中心 shape 非法")
    if categories.shape != (len(rows),) or not np.issubdtype(
        categories.dtype, np.integer
    ):
        raise QueryAugmentedEvalError("E4 类别行号 shape/dtype 非法")
    if np.any(categories < 0) or np.any(categories >= len(means)):
        raise QueryAugmentedEvalError("E4 类别行号越界")
    if not np.isfinite(beta) or not 0.0 < beta <= 1.0:
        raise QueryAugmentedEvalError("E4 beta 必须位于 (0,1]")
    residual = rows - beta * means[categories]
    norms = np.linalg.norm(residual, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise QueryAugmentedEvalError("E4 残差存在非有限值或零范数")
    return np.ascontiguousarray(residual / norms[:, None], dtype=np.float32)


def adaptive_alpha_from_effective_count(
    effective_query_count: np.ndarray,
    *,
    alpha_min: float,
    alpha_max: float,
    tau: float,
    gamma: float,
) -> np.ndarray:
    """Map Train-only Query heterogeneity to a bounded fusion coefficient."""

    values = np.asarray(effective_query_count, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 1):
        raise QueryAugmentedEvalError("有效 Query 数必须是一维有限正数")
    if not 0.0 <= alpha_min < alpha_max <= 1.0:
        raise QueryAugmentedEvalError("alpha_min/alpha_max 边界非法")
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(gamma) or gamma <= 0:
        raise QueryAugmentedEvalError("tau/gamma 必须是有限正数")
    alpha = alpha_min + (alpha_max - alpha_min) / (
        1.0 + np.power(values / tau, gamma)
    )
    if (
        not np.isfinite(alpha).all()
        or np.any(alpha < alpha_min)
        or np.any(alpha > alpha_max)
    ):
        raise QueryAugmentedEvalError("自适应 alpha 计算结果非法")
    return alpha.astype(np.float32)


def fuse_embedding_chunk(
    base_embeddings: np.ndarray,
    query_aggregates: np.ndarray,
    position_by_poi_row: np.ndarray,
    *,
    start: int,
    stop: int,
    alpha: float | np.ndarray,
    fusion_chunk_rows: int = 8_192,
    category_by_aggregate: np.ndarray | None = None,
    category_query_means: np.ndarray | None = None,
    residual_beta: float | None = None,
) -> np.ndarray:
    """Materialize one contiguous base-row range with sparse normalized fusion."""

    if np.isscalar(alpha):
        scalar_alpha = float(alpha)
        if not 0.0 < scalar_alpha <= 1.0:
            raise QueryAugmentedEvalError("alpha 必须位于 (0, 1]")
        alpha_by_aggregate = None
    else:
        alpha_by_aggregate = np.asarray(alpha, dtype=np.float32)
        if (
            alpha_by_aggregate.shape != (len(query_aggregates),)
            or not np.isfinite(alpha_by_aggregate).all()
            or np.any(alpha_by_aggregate <= 0)
            or np.any(alpha_by_aggregate > 1)
        ):
            raise QueryAugmentedEvalError("逐 POI alpha shape 或取值非法")
        scalar_alpha = None
    if start < 0 or stop <= start or stop > len(base_embeddings):
        raise QueryAugmentedEvalError("融合行区间非法")
    if len(position_by_poi_row) != len(base_embeddings):
        raise QueryAugmentedEvalError("POI 聚合位置映射行数不一致")
    if fusion_chunk_rows <= 0:
        raise QueryAugmentedEvalError("fusion_chunk_rows 必须大于 0")
    residual_enabled = residual_beta is not None
    if residual_enabled:
        if category_by_aggregate is None or category_query_means is None:
            raise QueryAugmentedEvalError("E4 残差缺少类别行号或类别中心")
        if np.asarray(category_by_aggregate).shape != (len(query_aggregates),):
            raise QueryAugmentedEvalError("E4 聚合类别行号 shape 非法")
    elif category_by_aggregate is not None or category_query_means is not None:
        raise QueryAugmentedEvalError("E4 类别输入与 residual_beta 必须同时提供")
    values = np.asarray(base_embeddings[start:stop], dtype=np.float32).copy()
    positions = position_by_poi_row[start:stop]
    for local_start in range(0, len(values), fusion_chunk_rows):
        local_stop = min(local_start + fusion_chunk_rows, len(values))
        local_positions = positions[local_start:local_stop]
        covered = local_positions >= 0
        if not np.any(covered):
            continue
        aggregate_positions = local_positions[covered]
        if np.any(aggregate_positions >= len(query_aggregates)):
            raise QueryAugmentedEvalError("POI 聚合位置映射越界")
        base_rows = values[local_start:local_stop][covered]
        aggregate_rows = np.asarray(
            query_aggregates[aggregate_positions], dtype=np.float32
        )
        if residual_enabled:
            aggregate_rows = category_residual_query_rows(
                aggregate_rows,
                np.asarray(category_by_aggregate)[aggregate_positions],
                np.asarray(category_query_means),
                beta=float(residual_beta),
            )
        if alpha_by_aggregate is None:
            alpha_values: float | np.ndarray = scalar_alpha
        else:
            alpha_values = alpha_by_aggregate[aggregate_positions, None]
        fused = (1.0 - alpha_values) * base_rows + alpha_values * aggregate_rows
        norms = np.linalg.norm(fused, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise QueryAugmentedEvalError("融合向量存在非有限值或零范数")
        values[local_start:local_stop][covered] = fused / norms[:, None]
    if not np.isfinite(values).all():
        raise QueryAugmentedEvalError("融合输出存在 NaN/Inf")
    return np.ascontiguousarray(values, dtype=np.float32)


def target_ranks_from_topk(
    topk_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Return one-based target ranks and -1 for misses."""

    if topk_indices.ndim != 2 or target_indices.shape != (len(topk_indices),):
        raise QueryAugmentedEvalError("Top-K 或目标行号 shape 非法")
    matches = topk_indices == target_indices[:, None]
    matched = matches.any(axis=1)
    ranks = np.full(len(target_indices), -1, dtype=np.int16)
    ranks[matched] = matches[matched].argmax(axis=1).astype(np.int16) + 1
    return ranks


def rank_metrics(target_ranks: np.ndarray) -> dict[str, Any]:
    """Compute exact single-target retrieval metrics through rank 20."""

    if target_ranks.ndim != 1:
        raise QueryAugmentedEvalError("target_ranks 必须是一维数组")
    samples = len(target_ranks)
    if samples == 0:
        return {
            "samples": 0,
            "hit_at_1": None,
            "hit_at_3": None,
            "hit_at_5": None,
            "hit_at_10": None,
            "hit_at_20": None,
            "mrr_at_10": None,
            "ndcg_at_10": None,
        }
    valid_at_10 = (target_ranks >= 1) & (target_ranks <= 10)
    reciprocal = np.where(valid_at_10, 1.0 / target_ranks.clip(min=1), 0.0)
    ndcg = np.where(
        valid_at_10,
        1.0 / np.log2(target_ranks.clip(min=1).astype(np.float64) + 1.0),
        0.0,
    )
    return {
        "samples": samples,
        "hit_at_1": float(np.mean(target_ranks == 1)),
        "hit_at_3": float(np.mean((target_ranks >= 1) & (target_ranks <= 3))),
        "hit_at_5": float(np.mean((target_ranks >= 1) & (target_ranks <= 5))),
        "hit_at_10": float(valid_at_10.mean()),
        "hit_at_20": float(
            np.mean((target_ranks >= 1) & (target_ranks <= 20))
        ),
        "mrr_at_10": float(reciprocal.mean()),
        "ndcg_at_10": float(ndcg.mean()),
    }


def metrics_by_masks(
    target_ranks: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Compute the same metrics for deterministic diagnostic masks."""

    result: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        if mask.dtype != np.bool_ or mask.shape != target_ranks.shape:
            raise QueryAugmentedEvalError(f"诊断 mask {name} shape/dtype 非法")
        result[name] = rank_metrics(target_ranks[mask])
    return result


def paired_rank_comparison(
    baseline_ranks: np.ndarray,
    current_ranks: np.ndarray,
    *,
    top_k: int,
) -> dict[str, int]:
    """Compare aligned ranks, treating misses as top_k + 1."""

    if baseline_ranks.shape != current_ranks.shape:
        raise QueryAugmentedEvalError("成对比较 rank shape 不一致")
    baseline = np.where(baseline_ranks > 0, baseline_ranks, top_k + 1)
    current = np.where(current_ranks > 0, current_ranks, top_k + 1)
    return {
        "wins": int(np.sum(current < baseline)),
        "ties": int(np.sum(current == baseline)),
        "losses": int(np.sum(current > baseline)),
    }

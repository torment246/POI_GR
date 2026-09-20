"""Graph-aligned dual-view PRQ-KMeans refinement for S1 and S2."""

from __future__ import annotations

import gc
import json
import os
import resource
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.base_quantizer import (
    EPSILON,
    _distribution_metrics,
    _hard_centroid_update,
    _normalize_torch,
    _soft_centroid_update,
    projection_residual,
)
from qg_prqk.sid.relational_config import RelationalCodebookConfig
from qg_prqk.sid.relational_data import RelationalCodebookDataError, RelationalCodebookInputs, load_relational_codebook_inputs


SCHEMA_VERSION = "qg-prqk-p6-dual-view-category-prqk-v1"
DEFAULT_CHUNK_ROWS = 8192


@dataclass(frozen=True)
class LevelGraph:
    """One layer's complete weighted bipartite graph in local row indices."""

    poi_rows: np.ndarray
    query_rows: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class CategoryCostLookup:
    """Compact row-to-category cost lookup without an N-POI dense matrix."""

    row_indices: np.ndarray
    cost_table: np.ndarray

    def costs_for_rows(self, start: int, stop: int) -> np.ndarray:
        return np.ascontiguousarray(self.cost_table[self.row_indices[start:stop]])

    def selected_costs(self, assignments: np.ndarray) -> np.ndarray:
        labels = np.asarray(assignments, dtype=np.int64)
        if labels.shape != self.row_indices.shape:
            raise RelationalCodebookDataError("P6 category cost/assignment 行未对齐")
        return self.cost_table[self.row_indices, labels]


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RelationalCodebookDataError("P6-CAT 需要 PyTorch") from error
    return torch


def _event(stage: str, **values: Any) -> None:
    print(json.dumps({"time": utc_now(), "stage": stage, **values}, ensure_ascii=False), flush=True)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise RelationalCodebookDataError(f"输出已存在且禁止覆盖：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_tensor_npy(path: Path, values, *, dtype: np.dtype, chunk_rows: int) -> None:
    if path.exists():
        raise RelationalCodebookDataError(f"输出已存在且禁止覆盖：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        target_dtype = np.dtype(dtype)
        with temporary.open("wb") as handle:
            np.lib.format.write_array_header_2_0(
                handle,
                {
                    "descr": np.lib.format.dtype_to_descr(target_dtype),
                    "fortran_order": False,
                    "shape": tuple(values.shape),
                },
            )
            for start in range(0, len(values), chunk_rows):
                stop = min(start + chunk_rows, len(values))
                block = np.ascontiguousarray(
                    values[start:stop].detach().float().cpu().numpy(),
                    dtype=target_dtype,
                )
                handle.write(block.tobytes(order="C"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        result.update(shape=list(values.shape), dtype=str(values.dtype))
    return result


def output_directory(config: RelationalCodebookConfig, gate: str) -> Path:
    available_gates = set(config.p6_gates) - {"selection", "seed"}
    if gate not in available_gates or gate not in ("sample", "medium", "full"):
        raise RelationalCodebookDataError(
            f"P6 配置 {config.protocol_id} 未开放 {gate}；"
            f"可用规模为 {sorted(available_gates)}"
        )
    settings = config.p6_gates[gate]
    path = config.p6_output_dir / (
        f"{gate}_{int(settings['query_rows']):06d}q_"
        f"{int(settings['expected_closed_poi_rows']):06d}p"
    )
    if not path.resolve().is_relative_to(config.p6_output_dir.resolve()):
        raise RelationalCodebookDataError("P6 输出越出冻结 namespace")
    return path


def next_status_for_gate(gate: str) -> str:
    """Return the mandatory manual review stop after one P6 scale."""
    statuses = {
        "sample": "HOLD_FOR_P6_SAMPLE_REVIEW",
        "medium": "HOLD_FOR_P6_MEDIUM_REVIEW",
        "full": "HOLD_FOR_P6_FULL_REVIEW",
    }
    try:
        return statuses[gate]
    except KeyError as error:
        raise RelationalCodebookDataError(f"未知 P6 规模：{gate}") from error


def warmup_weights(
    iteration: int,
    *,
    query_target: float,
    graph_target: float,
    category_target: float,
    zero_weight_iters: int = 2,
    ramp_end_iter: int = 6,
) -> tuple[float, float, float]:
    """Return the frozen 50%-then-linear warm-up weights."""
    if iteration <= 0 or zero_weight_iters < 0 or ramp_end_iter <= zero_weight_iters:
        raise RelationalCodebookDataError("P6 warm-up 参数非法")
    if iteration <= zero_weight_iters:
        return 0.5 * query_target, 0.5 * graph_target, 0.0
    fraction = min(1.0, (iteration - zero_weight_iters) / (ramp_end_iter - zero_weight_iters))
    query_factor = 0.5 + 0.5 * fraction
    return query_factor * query_target, query_factor * graph_target, fraction * category_target


def objective_increase_streak(
    previous_streak: int,
    relative_improvement: float | None,
    *,
    iteration: int,
    warmup_end_iteration: int,
    tolerance: float,
) -> int:
    """Count comparable objective increases only after all weights are frozen."""
    if iteration <= warmup_end_iteration or relative_improvement is None:
        return 0
    return previous_streak + 1 if relative_improvement < -tolerance else 0


def build_category_cost_lookup(
    assignments: np.ndarray,
    categories: np.ndarray,
    codebook_size: int,
    *,
    smoothing: float,
    parent_assignments: np.ndarray | None = None,
    fine_to_coarse: np.ndarray | None = None,
) -> CategoryCostLookup:
    """Build the exact S1/S2 costs as a compact category lookup table."""
    labels = np.asarray(assignments, dtype=np.int64)
    values = np.asarray(categories, dtype=np.int64)
    if labels.shape != values.shape or labels.ndim != 1 or not len(labels):
        raise RelationalCodebookDataError("P6 category assignment/category 行未对齐")
    if np.any(labels < 0) or np.any(labels >= codebook_size) or smoothing <= 0:
        raise RelationalCodebookDataError("P6 category assignment 或 smoothing 非法")
    category_count = int(values.max()) + 1
    if np.any(values < 0):
        raise RelationalCodebookDataError("P6 category index 不得为负")

    if parent_assignments is None:
        global_counts = np.bincount(values, minlength=category_count).astype(np.float64)
        global_prior = global_counts / global_counts.sum()
        counts = np.zeros((codebook_size, category_count), dtype=np.float64)
        np.add.at(counts, (labels, values), 1.0)
        totals = np.bincount(labels, minlength=codebook_size).astype(np.float64)
        probabilities = (counts + smoothing * global_prior[None, :]) / (
            totals[:, None] + smoothing
        )
        denominator = np.log(max(category_count, 2))
        return CategoryCostLookup(
            row_indices=values,
            cost_table=np.ascontiguousarray(
                -np.log(np.maximum(probabilities.T, EPSILON)) / denominator,
                dtype=np.float32,
            ),
        )

    parents = np.asarray(parent_assignments, dtype=np.int64)
    if parents.shape != labels.shape or np.any(parents < 0) or np.any(parents >= codebook_size):
        raise RelationalCodebookDataError("P6 S2 parent assignment 非法")
    if fine_to_coarse is None or len(fine_to_coarse) < category_count:
        raise RelationalCodebookDataError("P6 S2 缺少 fine_to_coarse")
    fine_per_coarse = np.bincount(
        np.asarray(fine_to_coarse[:category_count], dtype=np.int64)
    )
    unique_pairs, inverse = np.unique(
        parents * category_count + values, return_inverse=True
    )
    pair_counts = np.zeros((len(unique_pairs), codebook_size), dtype=np.float64)
    np.add.at(pair_counts, (inverse, labels), 1.0)
    parent_totals = np.bincount(parents, minlength=codebook_size).astype(np.float64)
    path_totals = np.zeros((codebook_size, codebook_size), dtype=np.float64)
    np.add.at(path_totals, (parents, labels), 1.0)
    pair_parents = unique_pairs // category_count
    pair_categories = unique_pairs % category_count
    pair_totals = np.bincount(inverse, minlength=len(unique_pairs)).astype(np.float64)
    prior = pair_totals / parent_totals[pair_parents]
    pair_counts += smoothing * prior[:, None]
    pair_counts /= path_totals[pair_parents] + smoothing
    pair_coarse = np.asarray(fine_to_coarse, dtype=np.int64)[pair_categories]
    denominators = np.log(np.maximum(fine_per_coarse[pair_coarse], 2))
    np.maximum(pair_counts, EPSILON, out=pair_counts)
    np.log(pair_counts, out=pair_counts)
    pair_counts *= -1.0
    pair_counts /= denominators[:, None]
    pair_costs = np.ascontiguousarray(
        pair_counts,
        dtype=np.float32,
    )
    return CategoryCostLookup(
        row_indices=np.asarray(inverse, dtype=np.int64),
        cost_table=pair_costs,
    )


def build_category_costs(
    assignments: np.ndarray,
    categories: np.ndarray,
    codebook_size: int,
    *,
    smoothing: float,
    parent_assignments: np.ndarray | None = None,
    fine_to_coarse: np.ndarray | None = None,
) -> np.ndarray:
    """Materialize category costs for diagnostics and small compatibility tests."""
    lookup = build_category_cost_lookup(
        assignments,
        categories,
        codebook_size,
        smoothing=smoothing,
        parent_assignments=parent_assignments,
        fine_to_coarse=fine_to_coarse,
    )
    return lookup.costs_for_rows(0, len(lookup.row_indices))


def _level_graph(inputs: RelationalCodebookInputs, level: int, active_query_rows: np.ndarray) -> LevelGraph:
    edges = inputs.selection.edges
    selected = edges.filter(pc.equal(edges["layer"], level))
    global_poi = selected["poi_row_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    poi_local = np.searchsorted(inputs.selection.selected_poi_rows, global_poi)
    if not np.array_equal(inputs.selection.selected_poi_rows[poi_local], global_poi):
        raise RelationalCodebookDataError("P6 edge target 未进入 POI 闭包")
    global_query = selected["node_row"].to_numpy(zero_copy_only=False).astype(np.int64)
    query_local = np.searchsorted(active_query_rows, global_query)
    if not np.array_equal(active_query_rows[query_local], global_query):
        raise RelationalCodebookDataError("P6 edge query 未进入当前层 Query 集")
    weights = selected["edge_weight"].to_numpy(zero_copy_only=False).astype(np.float32)
    return LevelGraph(poi_rows=poi_local, query_rows=query_local, weights=weights)


def _graph_penalty(
    entity_count: int,
    edge_entities: np.ndarray,
    edge_neighbor_labels: np.ndarray,
    edge_weights: np.ndarray,
    codebook_size: int,
    *,
    device,
):
    torch = _torch()
    agreement = torch.zeros(
        (entity_count, codebook_size), dtype=torch.float32, device=device
    )
    entities = torch.from_numpy(np.asarray(edge_entities, dtype=np.int64)).to(device)
    labels = torch.from_numpy(np.asarray(edge_neighbor_labels, dtype=np.int64)).to(device)
    weights = torch.from_numpy(np.asarray(edge_weights, dtype=np.float32)).to(device)
    agreement.index_put_((entities, labels), weights, accumulate=True)
    totals = torch.zeros(entity_count, dtype=torch.float32, device=device)
    totals.index_add_(0, entities, weights)
    return totals[:, None] - agreement


def _assign_poi(
    values,
    centroids,
    graph: LevelGraph,
    query_assignments,
    category_costs: CategoryCostLookup,
    *,
    graph_weight: float,
    category_weight: float,
    chunk_rows: int,
):
    torch = _torch()
    neighbor = query_assignments.detach().cpu().numpy()[graph.query_rows]
    graph_cost = _graph_penalty(
        len(values), graph.poi_rows, neighbor, graph.weights, len(centroids), device=values.device
    )
    labels = torch.empty(len(values), dtype=torch.int64, device=values.device)
    content_selected = torch.empty(len(values), dtype=torch.float32, device=values.device)
    normalized = _normalize_torch(centroids)
    for start in range(0, len(values), chunk_rows):
        stop = min(start + chunk_rows, len(values))
        content = 1.0 - values[start:stop] @ normalized.T
        category = torch.from_numpy(category_costs.costs_for_rows(start, stop)).to(
            values.device
        )
        objective = (
            content
            + graph_weight * graph_cost[start:stop]
            + category_weight * category
        )
        chosen = torch.argmin(objective, dim=1)
        labels[start:stop] = chosen
        content_selected[start:stop] = content.gather(1, chosen[:, None]).squeeze(1)
    return labels, content_selected


def _assign_query(
    values,
    centroids,
    graph: LevelGraph,
    poi_assignments,
    *,
    query_weight: float,
    graph_weight: float,
    chunk_rows: int,
):
    torch = _torch()
    neighbor = poi_assignments.detach().cpu().numpy()[graph.poi_rows]
    graph_cost = _graph_penalty(
        len(values), graph.query_rows, neighbor, graph.weights, len(centroids), device=values.device
    )
    labels = torch.empty(len(values), dtype=torch.int64, device=values.device)
    distances = torch.empty(len(values), dtype=torch.float32, device=values.device)
    normalized = _normalize_torch(centroids)
    for start in range(0, len(values), chunk_rows):
        stop = min(start + chunk_rows, len(values))
        content = 1.0 - values[start:stop] @ normalized.T
        objective = query_weight * content + graph_weight * graph_cost[start:stop]
        chosen = torch.argmin(objective, dim=1)
        labels[start:stop] = chosen
        distances[start:stop] = content.gather(1, chosen[:, None]).squeeze(1)
    return labels, distances


def _shrink_query_centroids(values, assignments, poi_centroids, tau: float):
    torch = _torch()
    sums = torch.zeros_like(poi_centroids, dtype=torch.float32)
    sums.index_add_(0, assignments, values)
    counts = torch.bincount(assignments, minlength=len(poi_centroids))
    centroids = (sums + tau * poi_centroids) / (counts.float() + tau)[:, None]
    return _normalize_torch(centroids), counts


def _soft_shrink_query_centroids(values, centroids, poi_centroids, *, topk: int, beta: float, tau: float):
    torch = _torch()
    similarities = values @ _normalize_torch(centroids).T
    scores, indices = torch.topk(similarities, k=topk, dim=1, largest=True, sorted=True)
    weights = torch.softmax(beta * scores, dim=1)
    sums = torch.zeros_like(centroids, dtype=torch.float32)
    support = torch.zeros(len(centroids), dtype=torch.float32, device=values.device)
    for neighbor in range(topk):
        labels = indices[:, neighbor]
        weight = weights[:, neighbor]
        sums.index_add_(0, labels, values * weight[:, None])
        support.index_add_(0, labels, weight)
    result = (sums + tau * poi_centroids) / (support + tau)[:, None]
    return _normalize_torch(result)


def _graph_metrics(graph: LevelGraph, poi_assignments: np.ndarray, query_assignments: np.ndarray) -> dict[str, Any]:
    agreement = poi_assignments[graph.poi_rows] == query_assignments[graph.query_rows]
    total = float(graph.weights.sum())
    agreed = float(graph.weights[agreement].sum())
    return {
        "edge_rows": int(len(graph.weights)),
        "edge_weight_sum": total,
        "weighted_agreement": agreed / total if total else 0.0,
        "weighted_disagreement": 1.0 - agreed / total if total else 0.0,
        "unweighted_agreement": float(np.mean(agreement)),
    }


def _category_metrics(assignments: np.ndarray, categories: np.ndarray, codebook_size: int) -> dict[str, Any]:
    labels = np.asarray(assignments, dtype=np.int64)
    values = np.asarray(categories, dtype=np.int64)
    counts: dict[tuple[int, int], int] = {}
    cluster_sizes = np.bincount(labels, minlength=codebook_size)
    for label, category in zip(labels, values, strict=True):
        key = (int(label), int(category))
        counts[key] = counts.get(key, 0) + 1
    majority = np.zeros(codebook_size, dtype=np.int64)
    for (label, _), count in counts.items():
        majority[label] = max(majority[label], count)
    purity = float(majority.sum() / len(labels))
    entropy_sum = 0.0
    for label in np.flatnonzero(cluster_sizes):
        local = np.array([count for (code, _), count in counts.items() if code == label], dtype=np.float64)
        probabilities = local / local.sum()
        entropy_sum += float(cluster_sizes[label]) * float(-np.sum(probabilities * np.log(probabilities)))
    return {"purity": purity, "conditional_entropy": entropy_sum / len(labels)}


def fit_dual_view_level(
    poi_values,
    query_values,
    initial_poi_centroids,
    initial_poi_assignments: np.ndarray,
    graph: LevelGraph,
    categories: np.ndarray,
    *,
    level: int,
    codebook_size: int,
    settings: Mapping[str, Any],
    prqk: Mapping[str, Any],
    warmup: Mapping[str, Any],
    parent_assignments: np.ndarray | None = None,
    fine_to_coarse: np.ndarray | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
):
    """Fit one coupled P6 layer from the frozen P5 A0 endpoint."""
    torch = _torch()
    poi_centroids = _normalize_torch(initial_poi_centroids.float())
    poi_assignments = torch.from_numpy(np.asarray(initial_poi_assignments, dtype=np.int64)).to(poi_values.device)
    query_centroids = poi_centroids.clone()
    initial_query_weight, initial_graph_weight, _ = warmup_weights(
        1,
        query_target=float(settings["query_distortion_weight"]),
        graph_target=float(settings["graph_alignment_weight"]),
        category_target=float(settings["category_weight"]),
        zero_weight_iters=int(warmup["zero_weight_iters"]),
        ramp_end_iter=int(warmup["linear_ramp_end_iter"]),
    )
    query_assignments, initial_query_distances = _assign_query(
        query_values,
        query_centroids,
        graph,
        poi_assignments,
        query_weight=initial_query_weight,
        graph_weight=initial_graph_weight,
        chunk_rows=chunk_rows,
    )
    query_centroids, _ = _shrink_query_centroids(
        query_values, query_assignments, poi_centroids, float(settings["query_centroid_shrinkage"])
    )
    trace: list[dict[str, Any]] = []
    previous_objective: float | None = None
    stable_rounds = 0
    increase_streak = 0
    converged = False
    for iteration in range(1, int(prqk["max_iter"]) + 1):
        query_weight, graph_weight, category_weight = warmup_weights(
            iteration,
            query_target=float(settings["query_distortion_weight"]),
            graph_target=float(settings["graph_alignment_weight"]),
            category_target=float(settings["category_weight"]),
            zero_weight_iters=int(warmup["zero_weight_iters"]),
            ramp_end_iter=int(warmup["linear_ramp_end_iter"]),
        )
        old_poi = poi_assignments
        old_query = query_assignments
        category_costs = build_category_cost_lookup(
            old_poi.detach().cpu().numpy(),
            categories,
            codebook_size,
            smoothing=float(settings["category_smoothing"]),
            parent_assignments=parent_assignments,
            fine_to_coarse=fine_to_coarse,
        )
        poi_assignments, poi_distances = _assign_poi(
            poi_values,
            poi_centroids,
            graph,
            query_assignments,
            category_costs,
            graph_weight=graph_weight,
            category_weight=category_weight,
            chunk_rows=chunk_rows,
        )
        query_assignments, query_distances = _assign_query(
            query_values,
            query_centroids,
            graph,
            poi_assignments,
            query_weight=query_weight,
            graph_weight=graph_weight,
            chunk_rows=chunk_rows,
        )
        poi_scores = 1.0 - poi_distances
        poi_centroids, poi_counts, empty = _hard_centroid_update(
            poi_values, poi_assignments, poi_scores, codebook_size
        )
        query_centroids, query_counts = _shrink_query_centroids(
            query_values,
            query_assignments,
            poi_centroids,
            float(settings["query_centroid_shrinkage"]),
        )
        poi_change = float((poi_assignments != old_poi).float().mean().item())
        query_change = float((query_assignments != old_query).float().mean().item())
        poi_np = poi_assignments.detach().cpu().numpy()
        query_np = query_assignments.detach().cpu().numpy()
        graph_result = _graph_metrics(graph, poi_np, query_np)
        selected_category = category_costs.selected_costs(poi_np)
        objective_components = {
            "poi_content_sum": float(poi_distances.sum().item()),
            "query_content_sum": float(query_distances.sum().item()),
            "graph_disagreement_weight_sum": graph_result["weighted_disagreement"] * graph_result["edge_weight_sum"],
            "category_cost_sum": float(selected_category.sum()),
        }
        objective = (
            objective_components["poi_content_sum"]
            + query_weight * objective_components["query_content_sum"]
            + graph_weight * objective_components["graph_disagreement_weight_sum"]
            + category_weight * objective_components["category_cost_sum"]
        )
        relative = None if previous_objective is None else (
            previous_objective - objective
        ) / max(abs(previous_objective), EPSILON)
        trace.append(
            {
                "stage": "hard_alternating",
                "iteration": iteration,
                "objective": objective,
                "relative_objective_improvement": relative,
                "poi_assignment_change": poi_change,
                "query_assignment_change": query_change,
                "weights": {"query": query_weight, "graph": graph_weight, "category": category_weight},
                "components": objective_components,
                "active_poi_codes": int(torch.count_nonzero(poi_counts).item()),
                "active_query_codes": int(torch.count_nonzero(query_counts).item()),
                "empty_poi_codes_reinitialized": empty,
                "weighted_graph_disagreement": graph_result["weighted_disagreement"],
            }
        )
        if relative is not None:
            increase_streak = objective_increase_streak(
                increase_streak,
                relative,
                iteration=iteration,
                warmup_end_iteration=int(warmup["linear_ramp_end_iter"]),
                tolerance=float(prqk["objective_rel_tol"]),
            )
            if increase_streak >= 3:
                raise RelationalCodebookDataError(f"P6 S{level} 复合 objective 连续 3 轮明显升高")
            if (
                iteration >= int(prqk["min_iter"])
                and 0.0 <= relative < float(prqk["objective_rel_tol"])
                and poi_change < float(prqk["poi_assignment_change_tol"])
                and query_change < float(prqk["query_assignment_change_tol"])
            ):
                stable_rounds += 1
            else:
                stable_rounds = 0
        previous_objective = objective
        if iteration >= int(prqk["min_iter"]) and stable_rounds >= int(prqk["patience"]):
            converged = True
            break

    refinement = prqk["topk_refinement"]
    if refinement["enabled"]:
        for iteration in range(1, int(refinement["max_iter"]) + 1):
            old_poi = poi_assignments
            old_query = query_assignments
            poi_centroids = _soft_centroid_update(
                poi_values,
                poi_centroids,
                topk=int(refinement["topk"]),
                beta=float(refinement["beta"]),
                chunk_rows=chunk_rows,
            )
            query_centroids = _soft_shrink_query_centroids(
                query_values,
                query_centroids,
                poi_centroids,
                topk=int(refinement["topk"]),
                beta=float(refinement["beta"]),
                tau=float(settings["query_centroid_shrinkage"]),
            )
            category_costs = build_category_cost_lookup(
                old_poi.detach().cpu().numpy(),
                categories,
                codebook_size,
                smoothing=float(settings["category_smoothing"]),
                parent_assignments=parent_assignments,
                fine_to_coarse=fine_to_coarse,
            )
            poi_assignments, poi_distances = _assign_poi(
                poi_values,
                poi_centroids,
                graph,
                query_assignments,
                category_costs,
                graph_weight=float(settings["graph_alignment_weight"]),
                category_weight=float(settings["category_weight"]),
                chunk_rows=chunk_rows,
            )
            query_assignments, query_distances = _assign_query(
                query_values,
                query_centroids,
                graph,
                poi_assignments,
                query_weight=float(settings["query_distortion_weight"]),
                graph_weight=float(settings["graph_alignment_weight"]),
                chunk_rows=chunk_rows,
            )
            graph_result = _graph_metrics(
                graph,
                poi_assignments.detach().cpu().numpy(),
                query_assignments.detach().cpu().numpy(),
            )
            trace.append(
                {
                    "stage": "topk_soft_centroid_refinement",
                    "iteration": iteration,
                    "poi_content_mean": float(poi_distances.mean().item()),
                    "query_content_mean": float(query_distances.mean().item()),
                    "poi_assignment_change": float((poi_assignments != old_poi).float().mean().item()),
                    "query_assignment_change": float((query_assignments != old_query).float().mean().item()),
                    "weighted_graph_disagreement": graph_result["weighted_disagreement"],
                    "topk": int(refinement["topk"]),
                    "beta": float(refinement["beta"]),
                }
            )

    final_poi, final_poi_distances = _assign_poi(
        poi_values,
        poi_centroids,
        graph,
        query_assignments,
        build_category_cost_lookup(
            poi_assignments.detach().cpu().numpy(),
            categories,
            codebook_size,
            smoothing=float(settings["category_smoothing"]),
            parent_assignments=parent_assignments,
            fine_to_coarse=fine_to_coarse,
        ),
        graph_weight=float(settings["graph_alignment_weight"]),
        category_weight=float(settings["category_weight"]),
        chunk_rows=chunk_rows,
    )
    final_query, final_query_distances = _assign_query(
        query_values,
        query_centroids,
        graph,
        final_poi,
        query_weight=float(settings["query_distortion_weight"]),
        graph_weight=float(settings["graph_alignment_weight"]),
        chunk_rows=chunk_rows,
    )
    return poi_centroids, query_centroids, final_poi, final_query, {
        "hard_converged": converged,
        "hard_iterations": sum(row["stage"] == "hard_alternating" for row in trace),
        "soft_refinement_iterations": sum(row["stage"] == "topk_soft_centroid_refinement" for row in trace),
        "initial_query_content_mean": float(initial_query_distances.mean().item()),
        "final_poi_content_mean": float(final_poi_distances.mean().item()),
        "final_query_content_mean": float(final_query_distances.mean().item()),
        "trace": trace,
    }


def _to_device_normalized_float32(values: np.ndarray, *, device, chunk_rows: int):
    torch = _torch()
    result = torch.empty(values.shape, dtype=torch.float32, device=device)
    for start in range(0, len(values), chunk_rows):
        stop = min(start + chunk_rows, len(values))
        block = np.ascontiguousarray(values[start:stop], dtype=np.float32)
        result[start:stop] = _normalize_torch(torch.from_numpy(block).to(device))
    return result


def _transform_query(
    values: np.ndarray,
    global_mean: np.ndarray,
    *,
    device,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
):
    torch = _torch()
    result = torch.empty(values.shape, dtype=torch.float32, device=device)
    mean = torch.from_numpy(np.asarray(global_mean, dtype=np.float32)).to(device)
    mean_norm_sq = torch.dot(mean, mean).clamp_min(EPSILON)
    for start in range(0, len(values), chunk_rows):
        stop = min(start + chunk_rows, len(values))
        block = np.ascontiguousarray(values[start:stop], dtype=np.float32)
        tensor = _normalize_torch(torch.from_numpy(block).to(device))
        coefficient = (tensor @ mean) / mean_norm_sq
        result[start:stop] = _normalize_torch(
            tensor - coefficient[:, None] * mean
        )
    return result


def _code_files() -> dict[str, str]:
    directory = Path(__file__).parent
    files = {
        "sid/base_quantizer.py": directory / "base_quantizer.py",
        "sid/relational_config.py": directory / "relational_config.py",
        "sid/relational_data.py": directory / "relational_data.py",
        "sid/relational_quantizer.py": directory / "relational_quantizer.py",
        "commands/relational_codebook.py": (
            directory.parent / "commands/relational_codebook.py"
        ),
    }
    return {name: sha256_file(path) for name, path in files.items()}


def _contract(config: RelationalCodebookConfig, inputs: RelationalCodebookInputs, gate: str, chunk_rows: int) -> dict[str, Any]:
    gate_config = dict(config.p6_gates[gate])
    contract = {
        "schema_version": SCHEMA_VERSION,
        "config_signature": config.signature(),
        "gate": gate,
        "selection": {
            "algorithm": config.p6_gates["selection"],
            "seed": config.p6_gates["seed"],
            **gate_config,
            "actual_closed_poi_rows": len(inputs.selection.selected_poi_rows),
            "actual_s1_s2_edge_rows": len(inputs.selection.edges),
            "added_graph_target_rows": inputs.selection.added_graph_target_rows,
        },
        "query_rows": len(inputs.selection.selected_query_rows),
        "embedding_dim": inputs.embedding_dim,
        "codebook_sizes": list(config.codebook_sizes),
        "layers": [1, 2],
        "query_view_policy": config.base.query_view_policy,
        "algorithm": {
            "prqk": dict(config.prqk),
            "s1": config.base.resolved_payload()["s1"],
            "s2": config.base.resolved_payload()["s2"],
            "warmup": config.base.resolved_payload()["warmup"],
            "query_centroid_prior": "paired_poi_centroid",
            "initial_query_centroid": "paired_poi_centroid",
            "alternating_order": ["poi_assignment", "query_assignment", "poi_centroid", "query_centroid_shrinkage"],
            "topk_refinement": "independent_view_soft_centroids_then_coupled_hard_assignment",
        },
        "source_hashes": dict(inputs.source_hashes),
        "chunk_rows": chunk_rows,
        "code": _code_files(),
        "business_validation_read": False,
        "business_test_read": False,
    }
    if gate == "full" and "p6_sample_manifest" in config.frozen_inputs:
        contract["previous_gate"] = {
            "gate": "sample",
            "manifest_path": config.frozen_inputs["p6_sample_manifest"]["path"],
            "manifest_sha256": config.frozen_inputs["p6_sample_manifest"]["sha256"],
            "gate_sequence_override": {
                "authorization": config.authorization["id"],
                "skipped_gates": ["medium"],
                "scope": "P6_CAT_S1_S2_ONLY",
            },
        }
    return contract


def _prefix_metrics(assignments: np.ndarray) -> dict[str, Any]:
    _, counts = np.unique(assignments, axis=0, return_counts=True)
    return {
        "distinct_prefix": int(len(counts)),
        "distinct_prefix_ratio": float(len(counts) / len(assignments)),
        "collision_excess": int(np.sum(counts - 1)),
        "bucket_p50": float(np.quantile(counts, 0.5)),
        "bucket_p99": float(np.quantile(counts, 0.99)),
        "bucket_max": int(counts.max()),
    }


def _level_names(level: int) -> tuple[str, ...]:
    return (
        f"poi_codebook_s{level}.npy",
        f"query_codebook_s{level}.npy",
        f"poi_assignments_s{level}.npy",
        f"query_assignments_s{level}.npy",
        f"poi_residual_after_s{level}.npy",
        f"query_residual_after_s{level}.npy",
        f"level_s{level}_metrics.json",
    )


def build_relational_codebook(
    config: RelationalCodebookConfig,
    inputs: RelationalCodebookInputs,
    *,
    gate: str,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Build one immutable P6 scale and stop before any downstream stage."""
    started = time.perf_counter()
    torch = _torch()
    if device != "cuda" or not torch.cuda.is_available():
        raise RelationalCodebookDataError("真实 P6 要求 CUDA；不得静默回退 CPU")
    directory = output_directory(config, gate)
    if (directory / "_SUCCESS").is_file():
        raise RelationalCodebookDataError(f"P6 输出已完成且 overwrite=false：{directory}")
    if directory.exists():
        raise RelationalCodebookDataError(f"P6 未完成输出目录已存在，禁止隐式覆盖：{directory}")
    directory.mkdir(parents=True)
    contract = _contract(config, inputs, gate, chunk_rows)
    write_json_atomic(directory / "config_resolved.json", config.resolved_payload())
    _atomic_npy(directory / "selected_poi_rows.npy", inputs.selection.selected_poi_rows)
    _atomic_npy(directory / "selected_query_rows.npy", inputs.selection.selected_query_rows)
    pq.write_table(inputs.selection.nodes, directory / "query_nodes.parquet", compression="zstd")
    pq.write_table(inputs.selection.edges, directory / "query_poi_edges_s1_s2.parquet", compression="zstd")

    target = torch.device(device)
    poi_values = _to_device_normalized_float32(
        inputs.poi_residual_s0, device=target, chunk_rows=chunk_rows
    )
    query_values = _transform_query(
        inputs.query_embeddings,
        inputs.global_mean,
        device=target,
        chunk_rows=chunk_rows,
    )
    _atomic_tensor_npy(
        directory / "poi_residual_s0.npy",
        poi_values,
        dtype=np.float16,
        chunk_rows=chunk_rows,
    )
    _atomic_tensor_npy(
        directory / "query_residual_s0.npy",
        query_values,
        dtype=np.float16,
        chunk_rows=chunk_rows,
    )
    depths = inputs.selection.nodes["supervision_depth"].to_numpy(zero_copy_only=False)
    poi_level_assignments: list[np.ndarray] = []
    query_level_assignments: list[np.ndarray] = []
    level_metrics: list[dict[str, Any]] = []
    s1_poi_assignments: np.ndarray | None = None

    for level in (1, 2):
        active_global_query_rows = inputs.selection.selected_query_rows[depths >= level]
        active_mask = depths >= level
        active_query_values = query_values[torch.from_numpy(active_mask).to(target)]
        graph = _level_graph(inputs, level, active_global_query_rows)
        initial_u = torch.from_numpy(inputs.initial_poi_codebooks[level - 1]).to(target)
        initial_s = inputs.initial_poi_assignments[level - 1]
        category = inputs.coarse_category_indices if level == 1 else inputs.fine_category_indices
        _event(
            "p6_level_started",
            gate=gate,
            level=level,
            poi_rows=len(poi_values),
            query_rows=len(active_query_values),
            edge_rows=len(graph.weights),
        )
        u, v, poi_assignment, query_assignment, fit = fit_dual_view_level(
            poi_values,
            active_query_values,
            initial_u,
            initial_s,
            graph,
            category,
            level=level,
            codebook_size=config.codebook_sizes[level - 1],
            settings=config.base.resolved_payload()[f"s{level}"],
            prqk=config.prqk,
            warmup=config.base.resolved_payload()["warmup"],
            parent_assignments=s1_poi_assignments if level == 2 else None,
            fine_to_coarse=inputs.fine_to_coarse if level == 2 else None,
            chunk_rows=chunk_rows,
        )
        poi_np = poi_assignment.cpu().numpy().astype(np.int32)
        active_query_np = query_assignment.cpu().numpy().astype(np.int32)
        query_np = np.full(len(depths), -1, dtype=np.int32)
        query_np[active_mask] = active_query_np
        poi_values, poi_residual_metrics = projection_residual(poi_values, u, poi_assignment)
        active_next, query_residual_metrics = projection_residual(active_query_values, v, query_assignment)
        query_next = torch.zeros_like(query_values)
        query_next[torch.from_numpy(active_mask).to(target)] = active_next
        query_values = query_next
        graph_result = _graph_metrics(graph, poi_np, active_query_np)
        level_result = {
            "level": level,
            "fit": fit,
            "poi_assignment": _distribution_metrics(poi_np, config.codebook_sizes[level - 1]),
            "query_assignment": _distribution_metrics(active_query_np, config.codebook_sizes[level - 1]),
            "graph": graph_result,
            "category": _category_metrics(poi_np, category, config.codebook_sizes[level - 1]),
            "reference_poi_assignment_change": float(np.mean(poi_np != initial_s)),
            "poi_residual": poi_residual_metrics,
            "query_residual": query_residual_metrics,
        }
        artifacts = {
            f"poi_codebook_s{level}.npy": u.cpu().numpy().astype(np.float32),
            f"query_codebook_s{level}.npy": v.cpu().numpy().astype(np.float32),
            f"poi_assignments_s{level}.npy": poi_np,
            f"query_assignments_s{level}.npy": query_np,
        }
        for name, values in artifacts.items():
            _atomic_npy(directory / name, values)
        _atomic_tensor_npy(
            directory / f"poi_residual_after_s{level}.npy",
            poi_values,
            dtype=np.float16,
            chunk_rows=chunk_rows,
        )
        _atomic_tensor_npy(
            directory / f"query_residual_after_s{level}.npy",
            query_values,
            dtype=np.float16,
            chunk_rows=chunk_rows,
        )
        write_json_atomic(directory / f"level_s{level}_metrics.json", level_result)
        poi_level_assignments.append(poi_np)
        query_level_assignments.append(query_np)
        level_metrics.append(level_result)
        if level == 1:
            s1_poi_assignments = poi_np
        _event(
            "p6_level_completed",
            gate=gate,
            level=level,
            hard_iterations=fit["hard_iterations"],
            weighted_graph_agreement=graph_result["weighted_agreement"],
            category_purity=level_result["category"]["purity"],
        )
        del u, v, poi_assignment, query_assignment, active_query_values, active_next
        gc.collect()
        torch.cuda.empty_cache()

    poi_prefixes = np.column_stack(poi_level_assignments).astype(np.int32)
    query_prefixes = np.column_stack(query_level_assignments).astype(np.int32)
    _atomic_npy(directory / "poi_assignments_s1_s2.npy", poi_prefixes)
    _atomic_npy(directory / "query_assignments_s1_s2.npy", query_prefixes)
    metrics = {
        "levels": level_metrics,
        "poi_prefix": _prefix_metrics(poi_prefixes),
        "structural_gate": {
            "both_levels_completed": True,
            "all_poi_codes_active_each_level": all(
                item["poi_assignment"]["active_codes"] == config.codebook_sizes[item["level"] - 1]
                for item in level_metrics
            ),
            "graph_closed": True,
            "hard_category_encoding_used": False,
            "geo_used": False,
            "ready_for_manual_review": True,
            "sample_role": "CODE_AND_DATA_SMOKE_ONLY" if gate == "sample" else None,
            "formal_metric_basis": gate == "full",
        },
    }
    write_json_atomic(directory / "metrics.json", metrics)
    artifact_names: list[str] = [
        "config_resolved.json",
        "selected_poi_rows.npy",
        "selected_query_rows.npy",
        "query_nodes.parquet",
        "query_poi_edges_s1_s2.parquet",
        "poi_residual_s0.npy",
        "query_residual_s0.npy",
        "poi_assignments_s1_s2.npy",
        "query_assignments_s1_s2.npy",
        "metrics.json",
    ]
    for level in (1, 2):
        artifact_names.extend(_level_names(level))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "phase": f"P6-CAT-{gate.upper()}",
        "role": "S1_S2_QUERY_CATEGORY_DUAL_VIEW_GATE",
        "built_at": utc_now(),
        "contract": contract,
        "metrics": metrics,
        "source_access": {
            "p4_train_query_graph_read": True,
            "p5_poi_only_initialization_read": True,
            "active_poi_category_read": True,
            "business_validation_read": False,
            "business_test_read": False,
            "geo_read": False,
        },
        "artifacts": {name: _artifact(directory / name) for name in artifact_names},
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        },
        "next_status": next_status_for_gate(gate),
    }
    write_json_atomic(directory / "manifest.json", manifest)
    write_json_atomic(directory / "_SUCCESS", {"manifest_sha256": sha256_file(directory / "manifest.json")})
    return validate_relational_codebook(config, gate=gate)


def _expected_artifacts() -> Iterable[str]:
    yield from (
        "config_resolved.json",
        "selected_poi_rows.npy",
        "selected_query_rows.npy",
        "query_nodes.parquet",
        "query_poi_edges_s1_s2.parquet",
        "poi_residual_s0.npy",
        "query_residual_s0.npy",
        "poi_assignments_s1_s2.npy",
        "query_assignments_s1_s2.npy",
        "metrics.json",
    )
    for level in (1, 2):
        yield from _level_names(level)


def validate_relational_codebook(config: RelationalCodebookConfig, *, gate: str) -> dict[str, Any]:
    """Validate the immutable P6 manifest, graph closure, hashes, and shapes."""
    directory = output_directory(config, gate)
    manifest_path = directory / "manifest.json"
    marker = json.loads((directory / "_SUCCESS").read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != manifest_sha:
        raise RelationalCodebookDataError("P6 manifest 与 _SUCCESS 哈希不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != f"P6-CAT-{gate.upper()}"
        or manifest.get("role") != "S1_S2_QUERY_CATEGORY_DUAL_VIEW_GATE"
        or manifest.get("contract", {}).get("config_signature") != config.signature()
    ):
        raise RelationalCodebookDataError("P6 manifest schema/状态/配置签名不匹配")
    if set(manifest.get("artifacts", {})) != set(_expected_artifacts()):
        raise RelationalCodebookDataError("P6 artifact 清单不完整")
    for name, entry in manifest["artifacts"].items():
        path = directory / name
        if entry.get("file") != name or sha256_file(path) != entry.get("sha256"):
            raise RelationalCodebookDataError(f"P6 artifact 缺失或哈希错误：{name}")
    settings = config.p6_gates[gate]
    poi_rows = int(settings["expected_closed_poi_rows"])
    query_rows = int(settings["query_rows"])
    dimension = int(manifest["contract"]["embedding_dim"])
    selected_poi = np.load(directory / "selected_poi_rows.npy", allow_pickle=False)
    selected_query = np.load(directory / "selected_query_rows.npy", allow_pickle=False)
    if (
        selected_poi.shape != (poi_rows,)
        or selected_poi.dtype != np.int64
        or selected_query.shape != (query_rows,)
        or selected_query.dtype != np.int64
        or not np.array_equal(selected_query, np.arange(query_rows, dtype=np.int64))
    ):
        raise RelationalCodebookDataError("P6 选样 artifact shape/dtype/Query 前缀错误")
    if gate == "full" and not np.array_equal(
        selected_poi, np.arange(poi_rows, dtype=np.int64)
    ):
        raise RelationalCodebookDataError("P6 full POI 必须使用 active POI identity 行序")
    nodes = pq.read_table(directory / "query_nodes.parquet")
    edges = pq.read_table(directory / "query_poi_edges_s1_s2.parquet")
    if len(nodes) != query_rows or len(edges) != int(settings["expected_s1_s2_edge_rows"]):
        raise RelationalCodebookDataError("P6 图 artifact 行数错误")
    depths = nodes["supervision_depth"].to_numpy(zero_copy_only=False)
    for level in (1, 2):
        active_queries = int(np.count_nonzero(depths >= level))
        expected = {
            f"poi_codebook_s{level}.npy": ((512, dimension), np.float32),
            f"query_codebook_s{level}.npy": ((512, dimension), np.float32),
            f"poi_assignments_s{level}.npy": ((poi_rows,), np.int32),
            f"query_assignments_s{level}.npy": ((query_rows,), np.int32),
            f"poi_residual_after_s{level}.npy": ((poi_rows, dimension), np.float16),
            f"query_residual_after_s{level}.npy": ((query_rows, dimension), np.float16),
        }
        for name, (shape, dtype) in expected.items():
            values = np.load(directory / name, mmap_mode="r", allow_pickle=False)
            if values.shape != shape or values.dtype != np.dtype(dtype) or not np.isfinite(values).all():
                raise RelationalCodebookDataError(f"P6 {name} shape/dtype/finite 错误")
        query_assignment = np.load(directory / f"query_assignments_s{level}.npy", allow_pickle=False)
        if (
            np.count_nonzero(query_assignment >= 0) != active_queries
            or np.any(query_assignment[depths < level] != -1)
            or np.any(query_assignment[depths >= level] < 0)
            or np.any(query_assignment >= 512)
        ):
            raise RelationalCodebookDataError(f"P6 S{level} Query depth mask/assignment 错误")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if manifest.get("metrics") != metrics:
        raise RelationalCodebookDataError("P6 manifest 与 metrics.json 不一致")
    if manifest.get("next_status") != next_status_for_gate(gate):
        raise RelationalCodebookDataError("P6 next_status 错误")
    return {
        "status": "p6_validated",
        "phase": manifest["phase"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "poi_rows": poi_rows,
        "query_rows": query_rows,
        "edge_rows": len(edges),
        "metrics": metrics,
        "next_status": manifest["next_status"],
    }


def load_and_build_p6(
    config: RelationalCodebookConfig,
    *,
    gate: str,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Load the confirmed inputs and build one P6 gate."""
    return build_relational_codebook(
        config,
        load_relational_codebook_inputs(config, gate=gate),
        gate=gate,
        device=device,
        chunk_rows=chunk_rows,
    )

"""Shared-assignment dual-view primitives for Query-Predictable RQ-KMeans.

The content view is observed for every POI.  The Query view is sparse and must
therefore be excluded, rather than replaced by a zero vector, for uncovered
POIs.  These primitives intentionally do not depend on the experiment runner so
the missing-view and residual contracts can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from poi_gr.sid.evaluation import (
    compute_basic_metrics,
    compute_layer_metrics,
    compute_prefix_metrics,
)


class QueryPredictableRQKMeansError(ValueError):
    """Raised when a dual-view quantization contract is invalid."""


@dataclass(frozen=True)
class DualViewAssignment:
    """One shared assignment and its two objective components."""

    labels: np.ndarray
    content_squared_distance: np.ndarray
    query_squared_distance: np.ndarray


def _matrix(name: str, value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise QueryPredictableRQKMeansError(
            f"{name} must be a finite two-dimensional matrix"
        )
    return matrix


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(values * values, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * values @ centers.T
    )
    # Round-off can produce tiny negative values after the matrix expansion.
    return np.maximum(distances, 0.0, out=distances)


def assign_dual_view_numpy(
    content: np.ndarray,
    content_centers: np.ndarray,
    *,
    query: np.ndarray | None = None,
    query_centers: np.ndarray | None = None,
    query_covered: np.ndarray | None = None,
    query_weight: float = 0.0,
) -> DualViewAssignment:
    """Assign one chunk without treating a missing Query view as a zero vector."""

    content_rows = _matrix("content", content)
    content_codebook = _matrix("content_centers", content_centers)
    if content_rows.shape[1] != content_codebook.shape[1]:
        raise QueryPredictableRQKMeansError(
            "content and content_centers dimensions differ"
        )
    if not np.isfinite(query_weight) or query_weight < 0:
        raise QueryPredictableRQKMeansError(
            "query_weight must be finite and non-negative"
        )

    content_distances = _squared_distances(content_rows, content_codebook)
    objective = content_distances.copy()
    query_distances = np.zeros_like(content_distances)

    if query_weight > 0:
        if query is None or query_centers is None or query_covered is None:
            raise QueryPredictableRQKMeansError(
                "positive query_weight requires Query rows, centers, and coverage"
            )
        query_rows = _matrix("query", query)
        query_codebook = _matrix("query_centers", query_centers)
        covered = np.asarray(query_covered)
        if covered.dtype != np.bool_ or covered.shape != (len(content_rows),):
            raise QueryPredictableRQKMeansError(
                "query_covered must be a boolean vector aligned to content"
            )
        if query_rows.shape != content_rows.shape:
            raise QueryPredictableRQKMeansError(
                "query must be dense and aligned for this chunk-level primitive"
            )
        if query_codebook.shape != content_codebook.shape:
            raise QueryPredictableRQKMeansError(
                "content and Query codebooks must have the same shape"
            )
        if np.any(covered):
            covered_query_distances = _squared_distances(
                query_rows[covered], query_codebook
            )
            query_distances[covered] = covered_query_distances
            objective[covered] += query_weight * covered_query_distances

    labels = np.argmin(objective, axis=1).astype(np.int32, copy=False)
    row_indices = np.arange(len(labels))
    return DualViewAssignment(
        labels=labels,
        content_squared_distance=content_distances[row_indices, labels],
        query_squared_distance=query_distances[row_indices, labels],
    )


def shrink_query_centers_numpy(
    query: np.ndarray,
    labels: np.ndarray,
    query_covered: np.ndarray,
    cluster_count: int,
    *,
    support_tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate frozen Query prototypes with empirical-Bayes support shrinkage."""

    rows = _matrix("query", query)
    assignments = np.asarray(labels)
    covered = np.asarray(query_covered)
    if cluster_count <= 0:
        raise QueryPredictableRQKMeansError("cluster_count must be positive")
    if not np.isfinite(support_tau) or support_tau <= 0:
        raise QueryPredictableRQKMeansError("support_tau must be positive")
    if assignments.shape != (len(rows),) or not np.issubdtype(
        assignments.dtype, np.integer
    ):
        raise QueryPredictableRQKMeansError("labels are not aligned integers")
    if covered.dtype != np.bool_ or covered.shape != (len(rows),):
        raise QueryPredictableRQKMeansError("query_covered is not aligned")
    if not np.any(covered):
        raise QueryPredictableRQKMeansError("at least one Query row is required")
    if np.any(assignments < 0) or np.any(assignments >= cluster_count):
        raise QueryPredictableRQKMeansError("labels are out of range")

    counts = np.bincount(
        assignments[covered], minlength=cluster_count
    ).astype(np.int64)
    sums = np.zeros((cluster_count, rows.shape[1]), dtype=np.float64)
    np.add.at(sums, assignments[covered], rows[covered])
    global_mean = np.mean(rows[covered], axis=0, dtype=np.float64)
    raw_centers = np.broadcast_to(global_mean, sums.shape).copy()
    supported = counts > 0
    raw_centers[supported] = sums[supported] / counts[supported, None]
    reliability = counts.astype(np.float64) / (
        counts.astype(np.float64) + float(support_tau)
    )
    centers = (
        reliability[:, None] * raw_centers
        + (1.0 - reliability[:, None]) * global_mean[None, :]
    )
    return (
        np.ascontiguousarray(centers, dtype=np.float32),
        counts,
        np.ascontiguousarray(reliability, dtype=np.float32),
    )


def route_covered_topk_numpy(
    content: np.ndarray,
    content_centers: np.ndarray,
    query: np.ndarray,
    query_centers: np.ndarray,
    query_covered: np.ndarray,
    reference_labels: np.ndarray,
    *,
    query_weight: float,
    content_top_t: int,
) -> DualViewAssignment:
    """Route only Query-covered rows inside frozen content Top-T candidates."""

    content_rows = _matrix("content", content)
    content_codebook = _matrix("content_centers", content_centers)
    query_rows = _matrix("query", query)
    query_codebook = _matrix("query_centers", query_centers)
    covered = np.asarray(query_covered)
    reference = np.asarray(reference_labels)
    if content_rows.shape != query_rows.shape:
        raise QueryPredictableRQKMeansError("content and query shapes differ")
    if content_codebook.shape != query_codebook.shape:
        raise QueryPredictableRQKMeansError("content and Query codebooks differ")
    if content_rows.shape[1] != content_codebook.shape[1]:
        raise QueryPredictableRQKMeansError("row and codebook dimensions differ")
    if covered.dtype != np.bool_ or covered.shape != (len(content_rows),):
        raise QueryPredictableRQKMeansError("query_covered is not aligned")
    if reference.shape != (len(content_rows),) or not np.issubdtype(
        reference.dtype, np.integer
    ):
        raise QueryPredictableRQKMeansError("reference_labels are not aligned integers")
    if np.any(reference < 0) or np.any(reference >= len(content_codebook)):
        raise QueryPredictableRQKMeansError("reference_labels are out of range")
    if not np.isfinite(query_weight) or query_weight < 0:
        raise QueryPredictableRQKMeansError("query_weight must be non-negative")
    if not 1 <= content_top_t <= len(content_codebook):
        raise QueryPredictableRQKMeansError("content_top_t is out of range")

    content_distances = _squared_distances(content_rows, content_codebook)
    query_distances = _squared_distances(query_rows, query_codebook)
    labels = reference.astype(np.int32, copy=True)
    selected_content = content_distances[np.arange(len(labels)), labels].copy()
    selected_query = np.zeros(len(labels), dtype=np.float32)
    selected_query[covered] = query_distances[
        np.flatnonzero(covered), labels[covered]
    ]
    for row in np.flatnonzero(covered):
        candidates = np.argpartition(
            content_distances[row], content_top_t - 1
        )[:content_top_t]
        if reference[row] not in candidates:
            candidates = np.append(candidates, reference[row])
        objective = content_distances[row, candidates] + float(query_weight) * (
            query_distances[row, candidates]
        )
        label = int(candidates[int(np.argmin(objective))])
        labels[row] = label
        selected_content[row] = content_distances[row, label]
        selected_query[row] = query_distances[row, label]
    return DualViewAssignment(
        labels=labels,
        content_squared_distance=np.asarray(selected_content, dtype=np.float32),
        query_squared_distance=np.asarray(selected_query, dtype=np.float32),
    )


def update_dual_view_centers(
    content: np.ndarray,
    query: np.ndarray,
    query_covered: np.ndarray,
    labels: np.ndarray,
    previous_content_centers: np.ndarray,
    previous_query_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update content centers from all rows and Query centers from covered rows."""

    content_rows = _matrix("content", content)
    query_rows = _matrix("query", query)
    content_centers = _matrix("previous_content_centers", previous_content_centers)
    query_centers = _matrix("previous_query_centers", previous_query_centers)
    covered = np.asarray(query_covered)
    assignments = np.asarray(labels)
    if query_rows.shape != content_rows.shape:
        raise QueryPredictableRQKMeansError("content and query shapes differ")
    if content_centers.shape != query_centers.shape:
        raise QueryPredictableRQKMeansError("codebook shapes differ")
    if content_rows.shape[1] != content_centers.shape[1]:
        raise QueryPredictableRQKMeansError("row and codebook dimensions differ")
    if covered.dtype != np.bool_ or covered.shape != (len(content_rows),):
        raise QueryPredictableRQKMeansError("query_covered is not aligned")
    if assignments.shape != (len(content_rows),) or not np.issubdtype(
        assignments.dtype, np.integer
    ):
        raise QueryPredictableRQKMeansError("labels are not aligned integers")
    cluster_count = len(content_centers)
    if np.any(assignments < 0) or np.any(assignments >= cluster_count):
        raise QueryPredictableRQKMeansError("labels are out of range")

    content_counts = np.bincount(assignments, minlength=cluster_count).astype(
        np.int64
    )
    query_counts = np.bincount(
        assignments[covered], minlength=cluster_count
    ).astype(np.int64)
    next_content = content_centers.copy()
    next_query = query_centers.copy()
    content_sums = np.zeros_like(next_content, dtype=np.float64)
    query_sums = np.zeros_like(next_query, dtype=np.float64)
    np.add.at(content_sums, assignments, content_rows)
    np.add.at(query_sums, assignments[covered], query_rows[covered])
    content_nonempty = content_counts > 0
    query_nonempty = query_counts > 0
    next_content[content_nonempty] = (
        content_sums[content_nonempty]
        / content_counts[content_nonempty, None]
    ).astype(np.float32)
    next_query[query_nonempty] = (
        query_sums[query_nonempty] / query_counts[query_nonempty, None]
    ).astype(np.float32)
    return next_content, next_query, content_counts, query_counts


def subtract_assigned_centers(
    residual: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return the next residual without mutating the caller's array."""

    rows = _matrix("residual", residual)
    codebook = _matrix("centers", centers)
    assignments = np.asarray(labels)
    if rows.shape[1] != codebook.shape[1] or assignments.shape != (len(rows),):
        raise QueryPredictableRQKMeansError("residual update shapes differ")
    if not np.issubdtype(assignments.dtype, np.integer):
        raise QueryPredictableRQKMeansError("labels must be integers")
    if np.any(assignments < 0) or np.any(assignments >= len(codebook)):
        raise QueryPredictableRQKMeansError("labels are out of range")
    return np.ascontiguousarray(rows - codebook[assignments], dtype=np.float32)


def build_category_residual_query_rows(
    query_rows: np.ndarray,
    category_ids: np.ndarray,
    category_means: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    """Build the frozen E4 Query residual view for covered POIs."""

    queries = _matrix("query_rows", query_rows)
    means = _matrix("category_means", category_means)
    categories = np.asarray(category_ids)
    if categories.shape != (len(queries),) or not np.issubdtype(
        categories.dtype, np.integer
    ):
        raise QueryPredictableRQKMeansError("category_ids are not aligned integers")
    if np.any(categories < 0) or np.any(categories >= len(means)):
        raise QueryPredictableRQKMeansError("category_ids are out of range")
    if queries.shape[1] != means.shape[1]:
        raise QueryPredictableRQKMeansError("Query and category dimensions differ")
    if not np.isfinite(beta) or not 0 < beta <= 1:
        raise QueryPredictableRQKMeansError("beta must be in (0, 1]")
    residual = queries - float(beta) * means[categories]
    norms = np.linalg.norm(residual, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise QueryPredictableRQKMeansError("category residual has an invalid norm")
    return np.ascontiguousarray(residual / norms[:, None], dtype=np.float32)


def recover_query_view_from_fused(
    content_rows: np.ndarray,
    fused_rows: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Recover the unit Query view from a normalized convex E4 fusion.

    For unit vectors ``c`` and ``q`` and ``e = normalize((1-a)c + aq)``,
    the positive scale of the pre-normalized vector is recovered analytically.
    This avoids repeatedly reading the physically fragmented Query aggregate.
    """

    content = _matrix("content_rows", content_rows).copy()
    fused = _matrix("fused_rows", fused_rows).copy()
    if content.shape != fused.shape:
        raise QueryPredictableRQKMeansError("content and fused shapes differ")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise QueryPredictableRQKMeansError("alpha must be in (0, 1)")
    content_norms = np.linalg.norm(content, axis=1)
    fused_norms = np.linalg.norm(fused, axis=1)
    if np.any(content_norms <= 0) or np.any(fused_norms <= 0):
        raise QueryPredictableRQKMeansError("content or fused row has zero norm")
    content /= content_norms[:, None]
    fused /= fused_norms[:, None]
    content_weight = 1.0 - float(alpha)
    cosine = np.sum(content * fused, axis=1)
    discriminant = alpha * alpha - content_weight * content_weight * (
        1.0 - cosine * cosine
    )
    # Float16 E4 storage can push a theoretically non-negative value a few ulps
    # below zero.  A material violation indicates an incompatible fusion input.
    if np.any(discriminant < -1e-3):
        raise QueryPredictableRQKMeansError(
            "fused rows are incompatible with the declared E4 alpha"
        )
    scale = content_weight * cosine + np.sqrt(np.maximum(discriminant, 0.0))
    query = (scale[:, None] * fused - content_weight * content) / float(alpha)
    query_norms = np.linalg.norm(query, axis=1)
    if not np.isfinite(query_norms).all() or np.any(query_norms <= 0):
        raise QueryPredictableRQKMeansError("recovered Query view has invalid norm")
    return np.ascontiguousarray(query / query_norms[:, None], dtype=np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()


def _save_npy(path: Path, values: np.ndarray) -> str:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)
    return _sha256_file(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise QueryPredictableRQKMeansError("PyTorch is required for GPU screening") from error
    if not torch.cuda.is_available():
        raise QueryPredictableRQKMeansError("the GPU screen requires a visible CUDA device")
    return torch


def _torch_distances(values: Any, centers: Any) -> Any:
    torch = _torch()
    distances = (
        torch.sum(values * values, dim=1, keepdim=True)
        + torch.sum(centers * centers, dim=1).unsqueeze(0)
        - 2.0 * values @ centers.T
    )
    return torch.clamp_min_(distances, 0.0)


def _gpu_assign(
    content: np.ndarray,
    query: np.ndarray,
    query_covered: np.ndarray,
    content_centers: np.ndarray,
    query_centers: np.ndarray,
    *,
    query_weight: float,
    chunk_rows: int,
    content_top_t: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    torch = _torch()
    device = torch.device("cuda:0")
    content_codebook = torch.as_tensor(
        np.asarray(content_centers, dtype=np.float32), device=device
    )
    query_codebook = torch.as_tensor(
        np.asarray(query_centers, dtype=np.float32), device=device
    )
    labels = np.empty(len(content), dtype=np.int32)
    content_distance_sum = 0.0
    query_distance_sum = 0.0
    covered_count = 0
    with torch.inference_mode():
        for start in range(0, len(content), chunk_rows):
            stop = min(start + chunk_rows, len(content))
            content_chunk = torch.as_tensor(
                np.asarray(content[start:stop], dtype=np.float32), device=device
            )
            content_distances = _torch_distances(content_chunk, content_codebook)
            if content_top_t is not None and content_top_t < len(content_centers):
                candidate_content_distances, candidate_indices = torch.topk(
                    content_distances,
                    k=content_top_t,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                objective = candidate_content_distances
            else:
                candidate_content_distances = content_distances
                candidate_indices = None
                objective = content_distances
            covered = np.asarray(query_covered[start:stop], dtype=np.bool_)
            query_distances = None
            covered_tensor = None
            if query_weight > 0 and np.any(covered):
                query_chunk = torch.as_tensor(
                    np.asarray(query[start:stop], dtype=np.float32), device=device
                )
                query_distances = _torch_distances(query_chunk, query_codebook)
                candidate_query_distances = (
                    query_distances
                    if candidate_indices is None
                    else torch.gather(query_distances, 1, candidate_indices)
                )
                covered_tensor = torch.as_tensor(covered, device=device)
                objective = objective.clone()
                objective[covered_tensor] += float(query_weight) * (
                    candidate_query_distances[covered_tensor]
                )
            local_labels = torch.argmin(objective, dim=1)
            rows = torch.arange(stop - start, device=device)
            chunk_labels = (
                local_labels
                if candidate_indices is None
                else candidate_indices[rows, local_labels]
            )
            labels[start:stop] = chunk_labels.cpu().numpy().astype(
                np.int32, copy=False
            )
            content_distance_sum += float(
                candidate_content_distances[rows, local_labels].sum().item()
            )
            if query_distances is not None and covered_tensor is not None:
                query_distance_sum += float(
                    query_distances[covered_tensor, chunk_labels[covered_tensor]]
                    .sum()
                    .item()
                )
                covered_count += int(np.count_nonzero(covered))
    return labels, {
        "mean_content_squared_distance": content_distance_sum / len(content),
        "mean_query_squared_distance_covered": (
            query_distance_sum / covered_count if covered_count else 0.0
        ),
    }


def _gpu_centers_from_labels(
    values: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    previous_centers: np.ndarray,
    *,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch = _torch()
    device = torch.device("cuda:0")
    cluster_count, dimension = previous_centers.shape
    sums = torch.zeros((cluster_count, dimension), dtype=torch.float32, device=device)
    counts = torch.zeros(cluster_count, dtype=torch.int64, device=device)
    with torch.inference_mode():
        for start in range(0, len(values), chunk_rows):
            stop = min(start + chunk_rows, len(values))
            keep = np.asarray(selected[start:stop], dtype=np.bool_)
            if not np.any(keep):
                continue
            chunk_values = torch.as_tensor(
                np.asarray(values[start:stop][keep], dtype=np.float32), device=device
            )
            chunk_labels = torch.as_tensor(
                np.asarray(labels[start:stop][keep], dtype=np.int64), device=device
            )
            sums.index_add_(0, chunk_labels, chunk_values)
            counts += torch.bincount(chunk_labels, minlength=cluster_count)
    counts_numpy = counts.cpu().numpy()
    centers = np.asarray(previous_centers, dtype=np.float32).copy()
    nonempty = counts_numpy > 0
    nonempty_tensor = counts > 0
    if np.any(nonempty):
        means = (
            sums[nonempty_tensor] / counts[nonempty_tensor, None]
        ).cpu().numpy()
        centers[nonempty] = means
    if np.any(~nonempty) and np.any(nonempty):
        # A global covered-view mean is a neutral deterministic fallback.  It is
        # retained only until a previously unsupported token receives coverage.
        global_mean = (
            sums[nonempty_tensor].sum(dim=0) / counts[nonempty_tensor].sum()
        ).cpu().numpy()
        centers[~nonempty] = global_mean
    return centers, counts_numpy


def _shrink_query_centers(
    centers: np.ndarray,
    counts: np.ndarray,
    *,
    support_tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(support_tau) or support_tau <= 0:
        raise QueryPredictableRQKMeansError("support_tau must be positive")
    query_centers = _matrix("query_centers", centers)
    support = np.asarray(counts, dtype=np.int64)
    if support.shape != (len(query_centers),) or np.any(support < 0):
        raise QueryPredictableRQKMeansError("query center counts are invalid")
    if support.sum() <= 0:
        raise QueryPredictableRQKMeansError("at least one Query row is required")
    global_mean = np.average(
        query_centers[support > 0],
        axis=0,
        weights=support[support > 0],
    )
    reliability = support.astype(np.float64) / (
        support.astype(np.float64) + float(support_tau)
    )
    shrunk = (
        reliability[:, None] * query_centers
        + (1.0 - reliability[:, None]) * global_mean[None, :]
    )
    return (
        np.ascontiguousarray(shrunk, dtype=np.float32),
        np.ascontiguousarray(reliability, dtype=np.float32),
    )


def _gpu_route_covered_topk(
    content: np.ndarray,
    query: np.ndarray,
    query_covered: np.ndarray,
    content_centers: np.ndarray,
    query_centers: np.ndarray,
    reference_labels: np.ndarray,
    *,
    query_weight: float,
    chunk_rows: int,
    content_top_t: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply one frozen-codebook route while preserving every uncovered label."""

    if query_weight < 0 or not np.isfinite(query_weight):
        raise QueryPredictableRQKMeansError("query_weight must be non-negative")
    if chunk_rows <= 0:
        raise QueryPredictableRQKMeansError("chunk_rows must be positive")
    if not 1 <= content_top_t <= len(content_centers):
        raise QueryPredictableRQKMeansError("content_top_t is out of range")
    covered_all = np.asarray(query_covered, dtype=np.bool_)
    reference = np.asarray(reference_labels, dtype=np.int32)
    if covered_all.shape != (len(content),) or reference.shape != (len(content),):
        raise QueryPredictableRQKMeansError("routing vectors are not aligned")

    torch = _torch()
    device = torch.device("cuda:0")
    content_codebook = torch.as_tensor(
        np.asarray(content_centers, dtype=np.float32), device=device
    )
    query_codebook = torch.as_tensor(
        np.asarray(query_centers, dtype=np.float32), device=device
    )
    labels = reference.copy()
    baseline_content_sum = 0.0
    routed_content_sum = 0.0
    baseline_query_sum = 0.0
    routed_query_sum = 0.0
    covered_count = 0
    reference_outside_top_t = 0
    with torch.inference_mode():
        for start in range(0, len(content), chunk_rows):
            stop = min(start + chunk_rows, len(content))
            covered = covered_all[start:stop]
            if not np.any(covered):
                continue
            local_rows = np.flatnonzero(covered)
            content_chunk = torch.as_tensor(
                np.asarray(content[start:stop][covered], dtype=np.float32),
                device=device,
            )
            query_chunk = torch.as_tensor(
                np.asarray(query[start:stop][covered], dtype=np.float32),
                device=device,
            )
            reference_chunk = torch.as_tensor(
                reference[start:stop][covered].astype(np.int64, copy=False),
                device=device,
            )
            content_distances = _torch_distances(content_chunk, content_codebook)
            query_distances = _torch_distances(query_chunk, query_codebook)
            candidate_content, candidates = torch.topk(
                content_distances,
                k=content_top_t,
                dim=1,
                largest=False,
                sorted=True,
            )
            reference_is_candidate = torch.any(
                candidates == reference_chunk[:, None], dim=1
            )
            reference_outside_top_t += int(
                torch.count_nonzero(~reference_is_candidate).item()
            )
            candidates = torch.cat((candidates, reference_chunk[:, None]), dim=1)
            candidate_content = torch.cat(
                (
                    candidate_content,
                    content_distances.gather(1, reference_chunk[:, None]),
                ),
                dim=1,
            )
            candidate_query = query_distances.gather(1, candidates)
            objective = candidate_content + float(query_weight) * candidate_query
            local_choice = torch.argmin(objective, dim=1)
            rows = torch.arange(len(local_rows), device=device)
            routed = candidates[rows, local_choice]
            labels[start + local_rows] = routed.cpu().numpy().astype(
                np.int32, copy=False
            )
            baseline_content_sum += float(
                content_distances[rows, reference_chunk].sum().item()
            )
            routed_content_sum += float(
                content_distances[rows, routed].sum().item()
            )
            baseline_query_sum += float(
                query_distances[rows, reference_chunk].sum().item()
            )
            routed_query_sum += float(query_distances[rows, routed].sum().item())
            covered_count += len(local_rows)
    if covered_count == 0:
        raise QueryPredictableRQKMeansError("routing sample has no Query-covered rows")
    return labels, {
        "covered_rows": covered_count,
        "baseline_mean_content_squared_distance_covered": (
            baseline_content_sum / covered_count
        ),
        "routed_mean_content_squared_distance_covered": (
            routed_content_sum / covered_count
        ),
        "baseline_mean_query_squared_distance_covered": (
            baseline_query_sum / covered_count
        ),
        "routed_mean_query_squared_distance_covered": (
            routed_query_sum / covered_count
        ),
        "reference_outside_content_top_t_count": reference_outside_top_t,
        "reference_outside_content_top_t_ratio": (
            reference_outside_top_t / covered_count
        ),
    }


def fit_dual_view_gpu(
    content: np.ndarray,
    query: np.ndarray,
    query_covered: np.ndarray,
    initial_content_centers: np.ndarray,
    initial_labels: np.ndarray,
    *,
    query_weight: float,
    iterations: int,
    chunk_rows: int,
    content_top_t: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run bounded-memory missing-view Lloyd iterations on one residual level."""

    if iterations <= 0 or chunk_rows <= 0:
        raise QueryPredictableRQKMeansError("iterations and chunk_rows must be positive")
    if content_top_t is not None and not 1 <= content_top_t <= len(initial_content_centers):
        raise QueryPredictableRQKMeansError("content_top_t is out of range")
    content_centers = np.asarray(initial_content_centers, dtype=np.float32).copy()
    labels = np.asarray(initial_labels, dtype=np.int32).copy()
    query_centers, initial_query_counts = _gpu_centers_from_labels(
        query,
        labels,
        np.asarray(query_covered, dtype=np.bool_),
        np.zeros_like(content_centers),
        chunk_rows=chunk_rows,
    )
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        started = time.perf_counter()
        next_labels, distances = _gpu_assign(
            content,
            query,
            query_covered,
            content_centers,
            query_centers,
            query_weight=query_weight,
            chunk_rows=chunk_rows,
            content_top_t=content_top_t,
        )
        next_content_centers, content_counts = _gpu_centers_from_labels(
            content,
            next_labels,
            np.ones(len(content), dtype=np.bool_),
            content_centers,
            chunk_rows=chunk_rows,
        )
        next_query_centers, query_counts = _gpu_centers_from_labels(
            query,
            next_labels,
            np.asarray(query_covered, dtype=np.bool_),
            query_centers,
            chunk_rows=chunk_rows,
        )
        changed = int(np.count_nonzero(next_labels != labels))
        history.append(
            {
                "iteration": iteration + 1,
                "changed_assignments": changed,
                "changed_ratio": changed / len(labels),
                "content_dead_codes": int(np.count_nonzero(content_counts == 0)),
                "query_unsupported_codes": int(np.count_nonzero(query_counts == 0)),
                "initial_query_unsupported_codes": int(
                    np.count_nonzero(initial_query_counts == 0)
                ),
                **distances,
                "seconds": time.perf_counter() - started,
            }
        )
        labels = next_labels
        content_centers = next_content_centers
        query_centers = next_query_centers
        if changed == 0:
            break
    return content_centers, query_centers, labels, history


@dataclass(frozen=True)
class QueryPredictableScreenConfig:
    content_embeddings: Path
    fused_embeddings: Path
    query_aggregates: Path
    covered_poi_rows: Path
    category_indices: Path
    category_means: Path
    reference_dir: Path
    sample_cache_dir: Path
    output_dir: Path
    codebook_sizes: tuple[int, ...]
    query_weights: tuple[float, ...]
    beta: float
    fusion_alpha: float
    iterations: int
    chunk_rows: int
    scan_chunk_rows: int
    content_top_t: int | None
    expected_rows: int
    expected_dimension: int
    expected_sample_indices_sha256: str
    routing_mode: str = "lloyd"
    query_support_tau: float = 20.0
    validation_query_embeddings: Path | None = None
    validation_query_mapping: Path | None = None
    poi_ids: Path | None = None
    expected_validation_rows: int | None = None
    expected_validation_query_sha256: str | None = None
    expected_validation_mapping_sha256: str | None = None
    expected_poi_ids_sha256: str | None = None


def _load_codebooks(config: QueryPredictableScreenConfig) -> list[np.ndarray]:
    codebooks: list[np.ndarray] = []
    for level, size in enumerate(config.codebook_sizes, start=1):
        path = config.reference_dir / f"codebook_level_{level}.npy"
        values = np.load(path, allow_pickle=False)
        if values.shape != (size, config.expected_dimension) or values.dtype != np.float32:
            raise QueryPredictableRQKMeansError(f"invalid reference codebook L{level}")
        codebooks.append(values)
    return codebooks


def _precompute_frozen_query_codebooks(
    query_view: np.ndarray,
    query_covered: np.ndarray,
    reference_codes: np.ndarray,
    codebook_sizes: tuple[int, ...],
    work_dir: Path,
    *,
    chunk_rows: int,
    support_tau: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], Path]:
    """Fit every Query prototype once on the untouched reference SID path."""

    residual_path = work_dir / "reference_query_residual.npy"
    residual = np.lib.format.open_memmap(
        residual_path,
        mode="w+",
        dtype=np.float32,
        shape=query_view.shape,
    )
    for start in range(0, len(query_view), chunk_rows):
        stop = min(start + chunk_rows, len(query_view))
        residual[start:stop] = np.asarray(query_view[start:stop], dtype=np.float32)
    residual.flush()
    codebooks: list[np.ndarray] = []
    counts_by_level: list[np.ndarray] = []
    reliability_by_level: list[np.ndarray] = []
    for level, codebook_size in enumerate(codebook_sizes):
        labels = np.asarray(reference_codes[:, level], dtype=np.int32)
        centers, counts = _gpu_centers_from_labels(
            residual,
            labels,
            np.asarray(query_covered, dtype=np.bool_),
            np.zeros((codebook_size, query_view.shape[1]), dtype=np.float32),
            chunk_rows=chunk_rows,
        )
        centers, reliability = _shrink_query_centers(
            centers,
            counts,
            support_tau=support_tau,
        )
        codebooks.append(centers)
        counts_by_level.append(counts)
        reliability_by_level.append(reliability)
        for start in range(0, len(residual), chunk_rows):
            stop = min(start + chunk_rows, len(residual))
            covered = np.asarray(query_covered[start:stop], dtype=np.bool_)
            if not np.any(covered):
                continue
            block = np.asarray(residual[start:stop])
            block[covered] -= centers[labels[start:stop][covered]]
            residual[start:stop] = block
        residual.flush()
    del residual
    return codebooks, counts_by_level, reliability_by_level, residual_path


def _gpu_query_token_metrics(
    query_residual: np.ndarray,
    query_centers: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    *,
    chunk_rows: int,
) -> dict[str, float | int]:
    torch = _torch()
    device = torch.device("cuda:0")
    centers = torch.as_tensor(
        np.asarray(query_centers, dtype=np.float32), device=device
    )
    assignments = np.asarray(labels, dtype=np.int64)
    keep_all = np.asarray(selected, dtype=np.bool_)
    selected_rows = int(np.count_nonzero(keep_all))
    if selected_rows == 0:
        return {
            "rows": 0,
            "top1": 0.0,
            "top5": 0.0,
            "top10": 0.0,
            "mrr": 0.0,
            "mean_gold_squared_distance": 0.0,
        }
    hits1 = 0
    hits5 = 0
    hits10 = 0
    reciprocal_rank_sum = 0.0
    gold_distance_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(query_residual), chunk_rows):
            stop = min(start + chunk_rows, len(query_residual))
            keep = keep_all[start:stop]
            if not np.any(keep):
                continue
            values = torch.as_tensor(
                np.asarray(query_residual[start:stop][keep], dtype=np.float32),
                device=device,
            )
            gold = torch.as_tensor(
                assignments[start:stop][keep], device=device
            )
            distances = _torch_distances(values, centers)
            rows = torch.arange(len(values), device=device)
            gold_distances = distances[rows, gold]
            ranks = torch.count_nonzero(
                distances < gold_distances[:, None], dim=1
            ) + 1
            hits1 += int(torch.count_nonzero(ranks <= 1).item())
            hits5 += int(torch.count_nonzero(ranks <= 5).item())
            hits10 += int(torch.count_nonzero(ranks <= 10).item())
            reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
            gold_distance_sum += float(gold_distances.sum().item())
    return {
        "rows": selected_rows,
        "top1": hits1 / selected_rows,
        "top5": hits5 / selected_rows,
        "top10": hits10 / selected_rows,
        "mrr": reciprocal_rank_sum / selected_rows,
        "mean_gold_squared_distance": gold_distance_sum / selected_rows,
    }


def _gpu_content_labels(
    content_residual: np.ndarray,
    content_centers: np.ndarray,
    *,
    chunk_rows: int,
) -> np.ndarray:
    torch = _torch()
    device = torch.device("cuda:0")
    centers = torch.as_tensor(
        np.asarray(content_centers, dtype=np.float32), device=device
    )
    labels = np.empty(len(content_residual), dtype=np.int32)
    with torch.inference_mode():
        for start in range(0, len(content_residual), chunk_rows):
            stop = min(start + chunk_rows, len(content_residual))
            values = torch.as_tensor(
                np.asarray(content_residual[start:stop], dtype=np.float32),
                device=device,
            )
            labels[start:stop] = (
                torch.argmin(_torch_distances(values, centers), dim=1)
                .cpu()
                .numpy()
                .astype(np.int32, copy=False)
            )
    return labels


def _read_selected_embedding_rows(
    embeddings: np.ndarray,
    rows: np.ndarray,
    *,
    chunk_rows: int,
) -> np.ndarray:
    selected = np.asarray(rows, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(embeddings)):
        raise QueryPredictableRQKMeansError("selected embedding rows are invalid")
    output = np.empty((len(selected), embeddings.shape[1]), dtype=np.float32)
    for start in range(0, len(selected), chunk_rows):
        stop = min(start + chunk_rows, len(selected))
        output[start:stop] = np.asarray(
            embeddings[selected[start:stop]], dtype=np.float32
        )
    return output


def _sample_structure_metrics(
    codes: np.ndarray,
    category_ids: np.ndarray,
    codebook_sizes: tuple[int, ...],
) -> dict[str, Any]:
    basic, _, inverse, bucket_sizes = compute_basic_metrics(codes)
    return {
        "basic": basic,
        "layers": compute_layer_metrics(codes, codebook_sizes),
        "prefixes": compute_prefix_metrics(
            codes,
            np.asarray(category_ids, dtype=np.int32),
            full_inverse=inverse,
            full_bucket_sizes=bucket_sizes,
        ),
    }


def _load_validation_target_rows(
    config: QueryPredictableScreenConfig,
) -> tuple[np.ndarray, np.ndarray]:
    required = (
        config.validation_query_embeddings,
        config.validation_query_mapping,
        config.poi_ids,
        config.expected_validation_rows,
        config.expected_validation_query_sha256,
        config.expected_validation_mapping_sha256,
        config.expected_poi_ids_sha256,
    )
    if any(value is None for value in required):
        raise QueryPredictableRQKMeansError(
            "content-preserving routing requires the frozen Validation contract"
        )
    query_path = config.validation_query_embeddings
    mapping_path = config.validation_query_mapping
    poi_ids_path = config.poi_ids
    assert query_path is not None and mapping_path is not None and poi_ids_path is not None
    if _sha256_file(query_path) != config.expected_validation_query_sha256:
        raise QueryPredictableRQKMeansError("Validation Query embedding hash differs")
    if _sha256_file(mapping_path) != config.expected_validation_mapping_sha256:
        raise QueryPredictableRQKMeansError("Validation Query mapping hash differs")
    if _sha256_file(poi_ids_path) != config.expected_poi_ids_sha256:
        raise QueryPredictableRQKMeansError("POI ID catalog hash differs")
    queries = np.load(query_path, mmap_mode="r", allow_pickle=False)
    expected_rows = int(config.expected_validation_rows)
    if queries.shape != (expected_rows, config.expected_dimension):
        raise QueryPredictableRQKMeansError("Validation Query embedding shape differs")

    target_ids: list[str] = []
    with mapping_path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            record = json.loads(line)
            if record.get("row_index") != row:
                raise QueryPredictableRQKMeansError(
                    "Validation Query mapping row index differs"
                )
            target = record.get("target_poi_id")
            if not isinstance(target, str) or not target:
                raise QueryPredictableRQKMeansError(
                    "Validation Query mapping has an invalid target POI"
                )
            target_ids.append(target)
    if len(target_ids) != expected_rows:
        raise QueryPredictableRQKMeansError("Validation Query mapping row count differs")

    wanted = set(target_ids)
    row_by_id: dict[str, int] = {}
    poi_rows = 0
    with poi_ids_path.open("r", encoding="utf-8") as handle:
        for poi_rows, line in enumerate(handle, start=1):
            poi_id = json.loads(line)
            if poi_id in wanted:
                if poi_id in row_by_id:
                    raise QueryPredictableRQKMeansError(
                        f"duplicate target POI ID: {poi_id}"
                    )
                row_by_id[poi_id] = poi_rows - 1
    if poi_rows != config.expected_rows:
        raise QueryPredictableRQKMeansError("POI ID catalog row count differs")
    if len(row_by_id) != len(wanted):
        raise QueryPredictableRQKMeansError("Validation target POI is missing")
    target_rows = np.fromiter(
        (row_by_id[poi_id] for poi_id in target_ids),
        dtype=np.int64,
        count=expected_rows,
    )
    return queries, target_rows


def _evaluate_validation_query_predictability(
    config: QueryPredictableScreenConfig,
    content_codebooks: list[np.ndarray],
    query_codebooks: list[np.ndarray],
    content_embeddings: np.ndarray,
    covered_rows: np.ndarray,
    category_indices: np.ndarray,
    category_means: np.ndarray,
) -> dict[str, Any]:
    queries, target_rows = _load_validation_target_rows(config)
    unique_target_rows, inverse = np.unique(target_rows, return_inverse=True)
    target_content_unique = _read_selected_embedding_rows(
        content_embeddings,
        unique_target_rows,
        chunk_rows=config.scan_chunk_rows,
    )
    target_content = np.ascontiguousarray(target_content_unique[inverse])
    fused_embeddings = np.load(
        config.fused_embeddings, mmap_mode="r", allow_pickle=False
    )
    if fused_embeddings.shape != content_embeddings.shape:
        raise QueryPredictableRQKMeansError("fused embedding shape differs")
    position_by_poi = np.full(config.expected_rows, -1, dtype=np.int32)
    position_by_poi[covered_rows] = np.arange(len(covered_rows), dtype=np.int32)
    target_covered = position_by_poi[target_rows] >= 0
    target_query_view = np.zeros_like(target_content)
    if np.any(target_covered):
        covered_unique_rows, covered_inverse = np.unique(
            target_rows[target_covered], return_inverse=True
        )
        fused_unique = _read_selected_embedding_rows(
            fused_embeddings,
            covered_unique_rows,
            chunk_rows=config.scan_chunk_rows,
        )
        content_unique = _read_selected_embedding_rows(
            content_embeddings,
            covered_unique_rows,
            chunk_rows=config.scan_chunk_rows,
        )
        recovered_unique = recover_query_view_from_fused(
            content_unique,
            fused_unique,
            alpha=config.fusion_alpha,
        )
        target_query_view[target_covered] = recovered_unique[covered_inverse]

    validation_query_view = build_category_residual_query_rows(
        np.asarray(queries, dtype=np.float32),
        np.asarray(category_indices[target_rows], dtype=np.int32),
        category_means,
        beta=config.beta,
    )
    baseline_content_residual = target_content.copy()
    routed_content_residual = target_content.copy()
    routed_poi_query_residual = target_query_view.copy()
    baseline_query_residual = validation_query_view.copy()
    routed_query_residual = validation_query_view.copy()
    levels: list[dict[str, Any]] = []
    baseline_codes = np.empty(
        (len(queries), len(content_codebooks)), dtype=np.int32
    )
    routed_codes = np.empty_like(baseline_codes)
    selected = np.ones(len(queries), dtype=np.bool_)
    for level, (content_centers, query_centers, query_weight) in enumerate(
        zip(content_codebooks, query_codebooks, config.query_weights)
    ):
        baseline_labels = _gpu_content_labels(
            baseline_content_residual,
            content_centers,
            chunk_rows=config.chunk_rows,
        )
        if level == 0:
            routed_labels = baseline_labels.copy()
        else:
            assert config.content_top_t is not None
            routed_labels, _ = _gpu_route_covered_topk(
                routed_content_residual,
                routed_poi_query_residual,
                target_covered,
                content_centers,
                query_centers,
                baseline_labels,
                query_weight=query_weight,
                chunk_rows=config.chunk_rows,
                content_top_t=config.content_top_t,
            )
        baseline_codes[:, level] = baseline_labels
        routed_codes[:, level] = routed_labels
        baseline_metrics = _gpu_query_token_metrics(
            baseline_query_residual,
            query_centers,
            baseline_labels,
            selected,
            chunk_rows=config.chunk_rows,
        )
        routed_metrics = _gpu_query_token_metrics(
            routed_query_residual,
            query_centers,
            routed_labels,
            selected,
            chunk_rows=config.chunk_rows,
        )
        levels.append(
            {
                "level": level + 1,
                "baseline": baseline_metrics,
                "routed": routed_metrics,
                "target_token_change_count": int(
                    np.count_nonzero(baseline_labels != routed_labels)
                ),
                "target_token_change_ratio": float(
                    np.mean(baseline_labels != routed_labels)
                ),
            }
        )
        baseline_content_residual -= content_centers[baseline_labels]
        routed_content_residual -= content_centers[routed_labels]
        if np.any(target_covered):
            routed_poi_query_residual[target_covered] -= query_centers[
                routed_labels[target_covered]
            ]
        baseline_query_residual -= query_centers[baseline_labels]
        routed_query_residual -= query_centers[routed_labels]
    _, first_positions = np.unique(target_rows, return_index=True)
    first_positions.sort()
    for target_row in np.unique(target_rows):
        duplicate_rows = np.flatnonzero(target_rows == target_row)
        if len(duplicate_rows) > 1 and not np.all(
            routed_codes[duplicate_rows] == routed_codes[duplicate_rows[0]]
        ):
            raise QueryPredictableRQKMeansError(
                "the same Validation target POI received inconsistent routed SID"
            )
    unique_categories = np.asarray(
        category_indices[target_rows[first_positions]], dtype=np.int32
    )
    reference_structure = _sample_structure_metrics(
        baseline_codes[first_positions], unique_categories, config.codebook_sizes
    )
    routed_structure = _sample_structure_metrics(
        routed_codes[first_positions], unique_categories, config.codebook_sizes
    )
    return {
        "protocol": "fixed_validation_query_teacher_forcing_on_all_targets",
        "validation_rows": int(len(queries)),
        "unique_target_pois": int(len(unique_target_rows)),
        "train_query_covered_validation_rows": int(np.count_nonzero(target_covered)),
        "train_query_covered_validation_ratio": float(np.mean(target_covered)),
        "uncovered_full_sid_change_count": int(
            np.count_nonzero(
                np.any(
                    routed_codes[~target_covered] != baseline_codes[~target_covered],
                    axis=1,
                )
            )
        ),
        "reference_codes_sha256": _save_npy(
            config.output_dir / "validation_reference_codes.npy", baseline_codes
        ),
        "routed_codes_sha256": _save_npy(
            config.output_dir / "validation_routed_codes.npy", routed_codes
        ),
        "target_rows_sha256": _save_npy(
            config.output_dir / "validation_target_rows.npy", target_rows
        ),
        "unique_target_structure": {
            "reference": reference_structure,
            "routed": routed_structure,
        },
        "levels": levels,
    }


def _prepare_sample_views(
    config: QueryPredictableScreenConfig,
    sample_indices: np.ndarray,
    content_embeddings: np.ndarray,
    query_aggregates: np.ndarray,
    covered_rows: np.ndarray,
    category_indices: np.ndarray,
    category_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    cache_manifest_path = config.sample_cache_dir / "manifest.json"
    cache_success_path = config.sample_cache_dir / "_SUCCESS"
    if cache_success_path.is_file() and cache_manifest_path.is_file():
        with cache_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = {
            "sample_indices_sha256": config.expected_sample_indices_sha256,
            "sample_rows": len(sample_indices),
            "dimension": config.expected_dimension,
            "fusion_alpha": config.fusion_alpha,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise QueryPredictableRQKMeansError("sample cache manifest differs")
        cached_content = np.load(
            config.sample_cache_dir / "content.npy", mmap_mode="r", allow_pickle=False
        )
        cached_query = np.load(
            config.sample_cache_dir / "query.npy", mmap_mode="r", allow_pickle=False
        )
        cached_covered = np.load(
            config.sample_cache_dir / "query_covered.npy", allow_pickle=False
        )
        expected_shape = (len(sample_indices), config.expected_dimension)
        if cached_content.shape != expected_shape or cached_query.shape != expected_shape:
            raise QueryPredictableRQKMeansError("sample cache view shape differs")
        if cached_covered.shape != (len(sample_indices),) or cached_covered.dtype != np.bool_:
            raise QueryPredictableRQKMeansError("sample cache coverage differs")
        work_dir = config.output_dir / ".work"
        work_dir.mkdir()
        content_residual = np.array(cached_content, dtype=np.float32, copy=True)
        query_residual = np.array(cached_query, dtype=np.float32, copy=True)
        print(
            json.dumps(
                {
                    "qd_rq_stage": "reuse_sample_cache",
                    "cache_dir": str(config.sample_cache_dir),
                    "sample_rows": len(sample_indices),
                }
            ),
            flush=True,
        )
        return (
            content_residual,
            query_residual,
            cached_covered,
            work_dir,
        )

    if config.sample_cache_dir.exists():
        raise QueryPredictableRQKMeansError("incomplete sample cache already exists")
    config.sample_cache_dir.mkdir(parents=True)
    work_dir = config.output_dir / ".work"
    work_dir.mkdir()
    content_residual = np.empty(
        (len(sample_indices), config.expected_dimension), dtype=np.float32
    )
    query_residual = np.zeros_like(content_residual)
    position_by_poi = np.full(config.expected_rows, -1, dtype=np.int32)
    position_by_poi[covered_rows] = np.arange(len(covered_rows), dtype=np.int32)
    sample_positions = position_by_poi[sample_indices]
    sample_covered = sample_positions >= 0
    written_query_rows = int(np.count_nonzero(sample_covered))
    # The sample rows are sorted.  On this server, reading the selected rows in
    # bounded blocks is materially faster than scanning the complete 4.8 GiB
    # matrix, while preserving a bounded in-memory footprint.
    print(
        json.dumps(
            {
                "qd_rq_stage": "prepare_content_view",
                "source_rows": len(sample_indices),
                "sample_rows": len(sample_indices),
                "scan_chunk_rows": config.scan_chunk_rows,
            }
        ),
        flush=True,
    )
    for sample_start in range(0, len(sample_indices), config.scan_chunk_rows):
        sample_stop = min(sample_start + config.scan_chunk_rows, len(sample_indices))
        content_residual[sample_start:sample_stop] = np.asarray(
            content_embeddings[sample_indices[sample_start:sample_stop]],
            dtype=np.float32,
        )
    print(
        json.dumps(
            {
                "qd_rq_stage": "prepare_fused_view",
                "source_rows": written_query_rows,
                "covered_sample_rows": int(np.count_nonzero(sample_covered)),
                "scan_chunk_rows": config.scan_chunk_rows,
            }
        ),
        flush=True,
    )
    fused_embeddings = np.load(config.fused_embeddings, mmap_mode="r", allow_pickle=False)
    if fused_embeddings.shape != content_embeddings.shape:
        raise QueryPredictableRQKMeansError("fused embedding shape differs")
    covered_poi_rows = sample_indices[sample_covered]
    fused_covered = np.empty(
        (written_query_rows, config.expected_dimension), dtype=np.float32
    )
    for sample_start in range(0, written_query_rows, config.scan_chunk_rows):
        sample_stop = min(sample_start + config.scan_chunk_rows, written_query_rows)
        fused_covered[sample_start:sample_stop] = np.asarray(
            fused_embeddings[covered_poi_rows[sample_start:sample_stop]],
            dtype=np.float32,
        )
    if np.any(sample_covered):
        query_residual[sample_covered] = recover_query_view_from_fused(
            np.asarray(content_residual[sample_covered], dtype=np.float32),
            fused_covered,
            alpha=config.fusion_alpha,
        )
    del fused_covered
    cache_outputs = {}
    for name, values in (
        ("content.npy", np.asarray(content_residual)),
        ("query.npy", np.asarray(query_residual)),
        ("query_covered.npy", sample_covered),
    ):
        cache_outputs[name] = _save_npy(config.sample_cache_dir / name, values)
    _write_json(
        cache_manifest_path,
        {
            "schema_version": "query-predictable-sample-cache-v1",
            "sample_indices_sha256": config.expected_sample_indices_sha256,
            "sample_rows": len(sample_indices),
            "dimension": config.expected_dimension,
            "fusion_alpha": config.fusion_alpha,
            "covered_sample_rows": written_query_rows,
            "query_source": "analytic recovery from frozen E4 fusion",
            "outputs_sha256": cache_outputs,
        },
    )
    cache_success_path.write_text("completed\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "qd_rq_stage": "prepare_views_completed",
                "sample_rows": len(sample_indices),
                "covered_sample_rows": written_query_rows,
            }
        ),
        flush=True,
    )
    return content_residual, query_residual, sample_covered, work_dir


def _run_content_preserving_screen(
    config: QueryPredictableScreenConfig,
    sample_indices: np.ndarray,
    reference_codes: np.ndarray,
    initial_codebooks: list[np.ndarray],
    content_residual: np.ndarray,
    query_residual: np.ndarray,
    query_covered: np.ndarray,
    work_dir: Path,
    category_indices: np.ndarray,
    category_means: np.ndarray,
    content_embeddings: np.ndarray,
    covered_rows: np.ndarray,
    started: float,
) -> dict[str, Any]:
    if config.content_top_t is None:
        raise QueryPredictableRQKMeansError(
            "content-preserving routing requires content_top_t"
        )
    if config.iterations != 1:
        raise QueryPredictableRQKMeansError(
            "content-preserving routing is a single frozen-prototype pass"
        )
    frozen_query_codebooks, query_counts, query_reliability, reference_query_path = (
        _precompute_frozen_query_codebooks(
            query_residual,
            query_covered,
            reference_codes,
            config.codebook_sizes,
            work_dir,
            chunk_rows=config.chunk_rows,
            support_tau=config.query_support_tau,
        )
    )
    sample_codes = np.asarray(reference_codes, dtype=np.int32).copy()
    levels: list[dict[str, Any]] = []
    try:
        for level, (content_centers, query_centers, query_weight) in enumerate(
            zip(initial_codebooks, frozen_query_codebooks, config.query_weights)
        ):
            level_started = time.perf_counter()
            reference_labels = np.asarray(reference_codes[:, level], dtype=np.int32)
            if level == 0:
                labels = reference_labels.copy()
                routing = {
                    "covered_rows": int(np.count_nonzero(query_covered)),
                    "baseline_mean_content_squared_distance_covered": 0.0,
                    "routed_mean_content_squared_distance_covered": 0.0,
                    "baseline_mean_query_squared_distance_covered": 0.0,
                    "routed_mean_query_squared_distance_covered": 0.0,
                    "reference_outside_content_top_t_count": 0,
                    "reference_outside_content_top_t_ratio": 0.0,
                }
            else:
                labels, routing = _gpu_route_covered_topk(
                    content_residual,
                    query_residual,
                    query_covered,
                    content_centers,
                    query_centers,
                    reference_labels,
                    query_weight=query_weight,
                    chunk_rows=config.chunk_rows,
                    content_top_t=config.content_top_t,
                )
            if np.any(labels[~query_covered] != reference_labels[~query_covered]):
                raise QueryPredictableRQKMeansError(
                    f"uncovered POI changed at level {level + 1}"
                )
            sample_codes[:, level] = labels
            counts = query_counts[level]
            reliability = query_reliability[level]
            changed = labels != reference_labels
            covered_changed = changed & query_covered
            levels.append(
                {
                    "level": level + 1,
                    "query_weight": query_weight,
                    "content_codebook_frozen": True,
                    "query_codebook_frozen_after_reference_fit": True,
                    "sample_used_codes": int(np.unique(labels).size),
                    "sample_query_supported_codes": int(np.count_nonzero(counts)),
                    "query_support_count_min_nonzero": int(counts[counts > 0].min()),
                    "query_support_count_median_nonzero": float(
                        np.median(counts[counts > 0])
                    ),
                    "query_support_count_max": int(counts.max()),
                    "query_reliability_mean": float(np.mean(reliability)),
                    "assignment_change_from_reference": int(np.count_nonzero(changed)),
                    "assignment_change_ratio_from_reference": float(np.mean(changed)),
                    "covered_assignment_change_count": int(
                        np.count_nonzero(covered_changed)
                    ),
                    "covered_assignment_change_ratio": float(
                        np.mean(changed[query_covered])
                    ),
                    "uncovered_assignment_change_count": int(
                        np.count_nonzero(changed & ~query_covered)
                    ),
                    "query_codebook_sha256": _save_npy(
                        config.output_dir / f"query_codebook_level_{level + 1}.npy",
                        query_centers,
                    ),
                    "routing": routing,
                    "seconds": time.perf_counter() - level_started,
                }
            )
            print(json.dumps({"qd_rq_level": levels[-1]}), flush=True)
            for start in range(0, len(sample_indices), config.chunk_rows):
                stop = min(start + config.chunk_rows, len(sample_indices))
                block_labels = labels[start:stop]
                content_residual[start:stop] -= content_centers[block_labels]
                covered = np.asarray(query_covered[start:stop], dtype=np.bool_)
                if np.any(covered):
                    block = np.asarray(query_residual[start:stop])
                    block[covered] -= query_centers[block_labels[covered]]
                    query_residual[start:stop] = block
            content_residual.flush()
            query_residual.flush()

        if not np.array_equal(
            sample_codes[~query_covered], np.asarray(reference_codes[~query_covered])
        ):
            raise QueryPredictableRQKMeansError(
                "uncovered POI SID changed under content-preserving routing"
            )
        sample_category_ids = np.asarray(
            category_indices[sample_indices], dtype=np.int32
        )
        reference_structure = _sample_structure_metrics(
            np.asarray(reference_codes), sample_category_ids, config.codebook_sizes
        )
        routed_structure = _sample_structure_metrics(
            sample_codes, sample_category_ids, config.codebook_sizes
        )
        validation = _evaluate_validation_query_predictability(
            config,
            initial_codebooks,
            frozen_query_codebooks,
            content_embeddings,
            covered_rows,
            category_indices,
            category_means,
        )
        sample_codes_sha = _save_npy(
            config.output_dir / "sample_codes.npy", sample_codes
        )
        metrics = {
            "schema_version": "query-predictable-content-preserving-screen-v1",
            "status": "completed",
            "method": "R3_content_preserving_query_guided_fine_routing",
            "routing_contract": {
                "content_codebooks": "frozen reference RQ-KMeans",
                "query_codebooks": "one fit on reference assignment, then frozen",
                "uncovered_poi": "all SID levels exactly frozen",
                "covered_poi": "S2/S3 route within content Top-T plus reference token",
            },
            "codebook_sizes": list(config.codebook_sizes),
            "query_weights": list(config.query_weights),
            "content_top_t": config.content_top_t,
            "query_support_tau": config.query_support_tau,
            "beta": config.beta,
            "sample_rows": int(len(sample_indices)),
            "query_covered_sample_rows": int(np.count_nonzero(query_covered)),
            "query_covered_sample_ratio": float(np.mean(query_covered)),
            "sample_indices_sha256": _sha256_indices(sample_indices),
            "sample_codes_sha256": sample_codes_sha,
            "uncovered_full_sid_change_count": int(
                np.count_nonzero(
                    np.any(
                        sample_codes[~query_covered]
                        != np.asarray(reference_codes[~query_covered]),
                        axis=1,
                    )
                )
            ),
            "levels": levels,
            "sample_structure": {
                "reference": reference_structure,
                "routed": routed_structure,
            },
            "validation_query_predictability": validation,
            "runtime": {
                "seconds": time.perf_counter() - started,
                "chunk_rows": config.chunk_rows,
                "work_storage": str(work_dir),
                "gpu_peak_allocated_mib": float(
                    _torch().cuda.max_memory_allocated() / (1024 * 1024)
                ),
                "gpu_peak_reserved_mib": float(
                    _torch().cuda.max_memory_reserved() / (1024 * 1024)
                ),
            },
        }
        _write_json(config.output_dir / "screen_metrics.json", metrics)
        (config.output_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")
        return metrics
    finally:
        for path in (
            work_dir / "content_residual.npy",
            work_dir / "query_residual.npy",
            reference_query_path,
        ):
            path.unlink(missing_ok=True)
        work_dir.rmdir()


def run_query_predictable_screen(
    config: QueryPredictableScreenConfig,
) -> dict[str, Any]:
    """Run an R3 sample screen initialized from a frozen content RQ-KMeans."""

    if config.routing_mode not in {"lloyd", "content_preserving"}:
        raise QueryPredictableRQKMeansError("unsupported routing_mode")
    if len(config.codebook_sizes) != len(config.query_weights):
        raise QueryPredictableRQKMeansError("codebook_sizes and query_weights differ")
    if config.query_weights[0] != 0 or any(
        right <= left
        for left, right in zip(config.query_weights, config.query_weights[1:])
    ):
        raise QueryPredictableRQKMeansError(
            "R3 requires query_weights[0]=0 and strictly increasing later weights"
        )
    if (config.output_dir / "_SUCCESS").is_file():
        with (config.output_dir / "screen_metrics.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if config.output_dir.exists():
        raise QueryPredictableRQKMeansError(
            f"incomplete output already exists: {config.output_dir}"
        )
    config.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    content_embeddings = np.load(
        config.content_embeddings, mmap_mode="r", allow_pickle=False
    )
    query_aggregates = np.load(
        config.query_aggregates, mmap_mode="r", allow_pickle=False
    )
    covered_rows = np.load(
        config.covered_poi_rows, mmap_mode="r", allow_pickle=False
    )
    category_indices = np.load(
        config.category_indices, mmap_mode="r", allow_pickle=False
    )
    category_means = np.load(config.category_means, allow_pickle=False)
    sample_indices = np.load(
        config.reference_dir / "sample_indices.npy", allow_pickle=False
    )
    reference_codes = np.load(
        config.reference_dir / "sample_codes.npy", mmap_mode="r", allow_pickle=False
    )
    if content_embeddings.shape != (config.expected_rows, config.expected_dimension):
        raise QueryPredictableRQKMeansError("content embedding shape differs")
    if query_aggregates.shape != (len(covered_rows), config.expected_dimension):
        raise QueryPredictableRQKMeansError("Query aggregate shape differs")
    if category_indices.shape != (config.expected_rows,):
        raise QueryPredictableRQKMeansError("category index shape differs")
    if category_means.ndim != 2 or category_means.shape[1] != config.expected_dimension:
        raise QueryPredictableRQKMeansError("category mean shape differs")
    if _sha256_indices(sample_indices) != config.expected_sample_indices_sha256:
        raise QueryPredictableRQKMeansError("sample index hash differs")
    if reference_codes.shape != (len(sample_indices), len(config.codebook_sizes)):
        raise QueryPredictableRQKMeansError("reference sample code shape differs")
    if np.any(covered_rows < 0) or np.any(covered_rows >= config.expected_rows):
        raise QueryPredictableRQKMeansError("covered POI rows are out of range")
    if len(np.unique(covered_rows)) != len(covered_rows):
        raise QueryPredictableRQKMeansError("covered POI rows are not unique")

    initial_codebooks = _load_codebooks(config)
    content_residual, query_residual, query_covered, work_dir = _prepare_sample_views(
        config,
        sample_indices,
        content_embeddings,
        query_aggregates,
        covered_rows,
        category_indices,
        category_means,
    )
    if config.routing_mode == "content_preserving":
        return _run_content_preserving_screen(
            config,
            sample_indices,
            reference_codes,
            initial_codebooks,
            content_residual,
            query_residual,
            query_covered,
            work_dir,
            category_indices,
            category_means,
            content_embeddings,
            covered_rows,
            started,
        )
    content_codebooks: list[np.ndarray] = []
    query_codebooks: list[np.ndarray] = []
    sample_codes = np.empty_like(reference_codes)
    levels: list[dict[str, Any]] = []
    try:
        for level, (initial_content_centers, query_weight) in enumerate(
            zip(initial_codebooks, config.query_weights)
        ):
            level_started = time.perf_counter()
            print(
                json.dumps(
                    {
                        "qd_rq_stage": "fit_level",
                        "level": level + 1,
                        "query_weight": query_weight,
                    }
                ),
                flush=True,
            )
            if level == 0:
                labels = np.asarray(reference_codes[:, 0], dtype=np.int32)
                content_centers = np.asarray(initial_content_centers, dtype=np.float32)
                query_centers, query_counts = _gpu_centers_from_labels(
                    query_residual,
                    labels,
                    query_covered,
                    np.zeros_like(content_centers),
                    chunk_rows=config.chunk_rows,
                )
                history: list[dict[str, Any]] = []
            else:
                if level == 1:
                    initial_labels = np.asarray(reference_codes[:, level], dtype=np.int32)
                else:
                    initial_query_centers, _ = _gpu_centers_from_labels(
                        query_residual,
                        np.asarray(reference_codes[:, level], dtype=np.int32),
                        query_covered,
                        np.zeros_like(initial_content_centers),
                        chunk_rows=config.chunk_rows,
                    )
                    initial_labels, _ = _gpu_assign(
                        content_residual,
                        query_residual,
                        query_covered,
                        initial_content_centers,
                        initial_query_centers,
                        query_weight=0.0,
                        chunk_rows=config.chunk_rows,
                        content_top_t=config.content_top_t,
                    )
                content_centers, query_centers, labels, history = fit_dual_view_gpu(
                    content_residual,
                    query_residual,
                    query_covered,
                    initial_content_centers,
                    initial_labels,
                    query_weight=query_weight,
                    iterations=config.iterations,
                    chunk_rows=config.chunk_rows,
                    content_top_t=config.content_top_t,
                )
                query_counts = np.bincount(
                    labels[query_covered], minlength=config.codebook_sizes[level]
                )
            sample_codes[:, level] = labels
            content_codebooks.append(content_centers)
            query_codebooks.append(query_centers)
            content_sha = _save_npy(
                config.output_dir / f"content_codebook_level_{level + 1}.npy",
                content_centers,
            )
            query_sha = _save_npy(
                config.output_dir / f"query_codebook_level_{level + 1}.npy",
                query_centers,
            )
            levels.append(
                {
                    "level": level + 1,
                    "query_weight": query_weight,
                    "sample_used_codes": int(np.unique(labels).size),
                    "sample_query_supported_codes": int(np.count_nonzero(query_counts)),
                    "assignment_change_from_reference": int(
                        np.count_nonzero(labels != np.asarray(reference_codes[:, level]))
                    ),
                    "assignment_change_ratio_from_reference": float(
                        np.mean(labels != np.asarray(reference_codes[:, level]))
                    ),
                    "content_codebook_sha256": content_sha,
                    "query_codebook_sha256": query_sha,
                    "iterations": history,
                    "seconds": time.perf_counter() - level_started,
                }
            )
            print(json.dumps({"qd_rq_level": levels[-1]}), flush=True)
            for start in range(0, len(sample_indices), config.chunk_rows):
                stop = min(start + config.chunk_rows, len(sample_indices))
                block_labels = labels[start:stop]
                content_residual[start:stop] -= content_centers[block_labels]
                covered = query_covered[start:stop]
                if np.any(covered):
                    block = np.asarray(query_residual[start:stop])
                    block[covered] -= query_centers[block_labels[covered]]
                    query_residual[start:stop] = block
            content_residual.flush()
            query_residual.flush()
    finally:
        del content_residual, query_residual

    sample_codes_sha = _save_npy(config.output_dir / "sample_codes.npy", sample_codes)
    metrics = {
        "schema_version": "query-predictable-rqkmeans-screen-v1",
        "status": "completed",
        "method": "R3_layerwise_dual_view_shared_assignment",
        "codebook_sizes": list(config.codebook_sizes),
        "query_weights": list(config.query_weights),
        "content_top_t": config.content_top_t,
        "beta": config.beta,
        "sample_rows": int(len(sample_indices)),
        "query_covered_sample_rows": int(np.count_nonzero(query_covered)),
        "query_covered_sample_ratio": float(np.mean(query_covered)),
        "sample_indices_sha256": _sha256_indices(sample_indices),
        "sample_codes_sha256": sample_codes_sha,
        "levels": levels,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "chunk_rows": config.chunk_rows,
            "work_storage": str(work_dir),
            "gpu_peak_allocated_mib": float(
                _torch().cuda.max_memory_allocated() / (1024 * 1024)
            ),
            "gpu_peak_reserved_mib": float(
                _torch().cuda.max_memory_reserved() / (1024 * 1024)
            ),
        },
    }
    _write_json(config.output_dir / "screen_metrics.json", metrics)
    (config.output_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")
    for path in (work_dir / "content_residual.npy", work_dir / "query_residual.npy"):
        path.unlink(missing_ok=True)
    work_dir.rmdir()
    return metrics

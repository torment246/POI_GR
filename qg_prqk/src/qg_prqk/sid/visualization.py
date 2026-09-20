"""Read-only SID diagnostics for POI-only, GID-parent, and no-GID variants."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.evaluation_data import SidEvaluationDataError, _require_npy
from qg_prqk.sid.visualization_config import METHOD_ORDER, SIDVisualizationConfig


SCHEMA_VERSION = "qg-prqk-sid-visualization-v1"
LEVEL_NAMES = ("S1", "S2", "S3")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def _event(stage: str, **values: Any) -> None:
    print(
        json.dumps({"time": utc_now(), "stage": stage, **values}, ensure_ascii=False),
        flush=True,
    )


def _code_files(config: SIDVisualizationConfig) -> dict[str, str]:
    source_dir = Path(__file__).parent
    return {
        "sid/visualization_config.py": sha256_file(
            source_dir / "visualization_config.py"
        ),
        "sid/visualization.py": sha256_file(source_dir / "visualization.py"),
        "commands/visualize_sid.py": sha256_file(
            source_dir.parent / "commands/visualize_sid.py"
        ),
        "scripts/qg_prqk.py": sha256_file(
            config.project_root / "qg_prqk/scripts/qg_prqk.py"
        ),
    }


def _source_hashes(config: SIDVisualizationConfig) -> dict[str, str]:
    return {
        name: sha256_file(Path(entry["path"]))
        for name, entry in config.frozen_manifests.items()
    }


def inspect_sid_visualization_inputs(config: SIDVisualizationConfig) -> dict[str, Any]:
    """Validate frozen array and metadata headers without producing outputs."""
    rows = config.rows["poi"]
    dimension = config.rows["embedding_dim"]
    codebook_size = config.rows["codebook_size"]
    source_hashes = _source_hashes(config)
    expected_hashes = {
        name: str(entry["sha256"])
        for name, entry in config.frozen_manifests.items()
    }
    if source_hashes != expected_hashes:
        raise SidEvaluationDataError("SID 可视化上游 manifest 已变化")

    _require_npy(config.poi_embedding_path, (rows, dimension), np.float16)
    shared_residual_paths: dict[str, list[str]] = {}
    for method in config.methods:
        sid = _require_npy(method.sid_path, (rows, 3), np.int32)
        if np.any(sid < 0) or np.any(sid >= codebook_size):
            raise SidEvaluationDataError(f"{method.name} SID token 超出 [0, {codebook_size})")
        for level, path in enumerate(method.codebook_paths):
            _require_npy(path, (codebook_size, dimension), np.float32)
            _require_npy(
                method.residual_input_paths[level],
                (rows, dimension),
                np.float16,
            )
            shared_residual_paths.setdefault(
                str(method.residual_input_paths[level]), []
            ).append(f"{method.name}.{LEVEL_NAMES[level]}")

    metadata = pq.ParquetFile(config.poi_metadata_path)
    required_columns = {
        "poi_row_index",
        "poi_id",
        "displayname",
        "category",
        "category_code",
        "fine_category_index",
    }
    if metadata.metadata.num_rows != rows or not required_columns.issubset(
        metadata.schema_arrow.names
    ):
        raise SidEvaluationDataError("SID 可视化 POI metadata 行数或字段合同不匹配")
    row_index = pq.read_table(
        config.poi_metadata_path, columns=["poi_row_index"]
    )["poi_row_index"].to_numpy(zero_copy_only=False)
    if not np.array_equal(row_index, np.arange(rows, dtype=np.int64)):
        raise SidEvaluationDataError("SID 可视化 metadata 未与 active POI 行空间对齐")

    return {
        "status": "sid_visualization_input_headers_validated",
        "poi_rows": rows,
        "embedding_dim": dimension,
        "codebook_size": codebook_size,
        "methods": list(METHOD_ORDER),
        "source_hashes": source_hashes,
        "shared_residual_inputs": shared_residual_paths,
        "raw_business_order_read": False,
        "business_validation_read": False,
        "business_test_read": False,
        "sid_rewritten": False,
        "output_written": False,
    }


def code_usage_metrics(tokens: np.ndarray, codebook_size: int) -> dict[str, Any]:
    """Summarize occupancy over every code slot, including unused codes."""
    counts = np.bincount(np.asarray(tokens, dtype=np.int64), minlength=codebook_size)
    total = int(counts.sum())
    probabilities = counts.astype(np.float64) / total
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    sorted_counts = np.sort(counts.astype(np.float64))
    ranks = np.arange(1, codebook_size + 1, dtype=np.float64)
    gini = float(
        2.0 * np.sum(ranks * sorted_counts) / (codebook_size * total)
        - (codebook_size + 1.0) / codebook_size
    )
    return {
        "active_codes": int(np.count_nonzero(counts)),
        "active_ratio": float(np.count_nonzero(counts) / codebook_size),
        "effective_codes": float(1.0 / np.sum(probabilities**2)),
        "normalized_entropy": float(entropy / math.log(codebook_size)),
        "gini": gini,
        "count_min": int(counts.min()),
        "count_p50": float(np.percentile(counts, 50)),
        "count_p90": float(np.percentile(counts, 90)),
        "count_p99": float(np.percentile(counts, 99)),
        "count_max": int(counts.max()),
    }


def transition_matrix(
    sid: np.ndarray, source_level: int, target_level: int, codebook_size: int
) -> np.ndarray:
    """Count token transitions between two SID positions."""
    source = np.asarray(sid[:, source_level], dtype=np.int64)
    target = np.asarray(sid[:, target_level], dtype=np.int64)
    flat = source * codebook_size + target
    return np.bincount(flat, minlength=codebook_size**2).reshape(
        codebook_size, codebook_size
    )


def _packed_prefix(sid: np.ndarray, length: int, codebook_size: int) -> np.ndarray:
    if length < 1 or length > sid.shape[1]:
        raise ValueError("prefix length 必须在 [1, SID levels] 内")
    packed = np.zeros(len(sid), dtype=np.int64)
    for level in range(length):
        packed = packed * codebook_size + np.asarray(sid[:, level], dtype=np.int64)
    return packed


def _draw_weighted_index(rng: np.random.Generator, weights: np.ndarray) -> int:
    cumulative = np.cumsum(weights, dtype=np.float64)
    return int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))


def _draw_cumulative_index(
    rng: np.random.Generator, cumulative_weights: np.ndarray
) -> int:
    return int(
        np.searchsorted(
            cumulative_weights,
            rng.random() * cumulative_weights[-1],
            side="right",
        )
    )


def _sample_cross_child_pairs(
    parent: np.ndarray,
    child: np.ndarray,
    pair_count: int,
    rng: np.random.Generator,
    codebook_size: int,
) -> np.ndarray:
    combined = parent.astype(np.int64) * codebook_size + child.astype(np.int64)
    row_order = np.argsort(combined, kind="stable")
    ordered = combined[row_order]
    _, child_starts, child_counts = np.unique(
        ordered, return_index=True, return_counts=True
    )
    child_parents = parent[row_order[child_starts]]
    _, parent_starts, children_per_parent = np.unique(
        child_parents, return_index=True, return_counts=True
    )
    parent_totals = np.add.reduceat(child_counts, parent_starts).astype(np.float64)
    squared = np.add.reduceat(child_counts.astype(np.float64) ** 2, parent_starts)
    cross_pair_counts = (parent_totals**2 - squared) / 2.0
    valid_parent = np.flatnonzero(cross_pair_counts > 0)
    if not len(valid_parent):
        raise SidEvaluationDataError("不存在指定 exact-prefix 的 POI pair")
    parent_weights = cross_pair_counts[valid_parent]
    parent_cumulative = np.cumsum(parent_weights, dtype=np.float64)

    pairs = np.empty((pair_count, 2), dtype=np.int64)
    for pair_index in range(pair_count):
        parent_index = int(
            valid_parent[_draw_cumulative_index(rng, parent_cumulative)]
        )
        child_begin = int(parent_starts[parent_index])
        child_end = child_begin + int(children_per_parent[parent_index])
        local_counts = child_counts[child_begin:child_end].astype(np.float64)
        total = float(local_counts.sum())
        first_local = _draw_weighted_index(
            rng, local_counts * (total - local_counts)
        )
        second_weights = local_counts.copy()
        second_weights[first_local] = 0.0
        second_local = _draw_weighted_index(rng, second_weights)
        first_group = child_begin + first_local
        second_group = child_begin + second_local
        first_offset = int(rng.integers(int(child_counts[first_group])))
        second_offset = int(rng.integers(int(child_counts[second_group])))
        pairs[pair_index, 0] = row_order[int(child_starts[first_group]) + first_offset]
        pairs[pair_index, 1] = row_order[int(child_starts[second_group]) + second_offset]
    return pairs


def sample_exact_prefix_pairs(
    sid: np.ndarray,
    prefix_length: int,
    pair_count: int,
    seed: int,
    codebook_size: int = 512,
) -> np.ndarray:
    """Sample POI pairs uniformly from pairs with an exact shared-prefix length."""
    if prefix_length < 0 or prefix_length > sid.shape[1]:
        raise ValueError("prefix_length 超出 SID 层数")
    if pair_count <= 0:
        raise ValueError("pair_count 必须为正数")
    sid_array = np.asarray(sid, dtype=np.int32)
    rng = np.random.default_rng(seed)
    if prefix_length < sid_array.shape[1]:
        parent = (
            np.zeros(len(sid_array), dtype=np.int64)
            if prefix_length == 0
            else _packed_prefix(sid_array, prefix_length, codebook_size)
        )
        pairs = _sample_cross_child_pairs(
            parent,
            sid_array[:, prefix_length],
            pair_count,
            rng,
            codebook_size,
        )
    else:
        prefix = _packed_prefix(sid_array, prefix_length, codebook_size)
        row_order = np.argsort(prefix, kind="stable")
        _, starts, counts = np.unique(
            prefix[row_order], return_index=True, return_counts=True
        )
        valid = np.flatnonzero(counts >= 2)
        if not len(valid):
            raise SidEvaluationDataError("不存在共享完整 SID 的 POI pair")
        weights = counts[valid].astype(np.float64)
        weights = weights * (weights - 1.0) / 2.0
        cumulative_weights = np.cumsum(weights, dtype=np.float64)
        pairs = np.empty((pair_count, 2), dtype=np.int64)
        for pair_index in range(pair_count):
            group_index = int(
                valid[_draw_cumulative_index(rng, cumulative_weights)]
            )
            count = int(counts[group_index])
            first = int(rng.integers(count))
            second = int(rng.integers(count - 1))
            if second >= first:
                second += 1
            start = int(starts[group_index])
            pairs[pair_index] = (row_order[start + first], row_order[start + second])

    left = sid_array[pairs[:, 0]]
    right = sid_array[pairs[:, 1]]
    shared = np.sum(np.cumprod(left == right, axis=1), axis=1)
    if np.any(shared != prefix_length) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise SidEvaluationDataError("exact-prefix pair 抽样合同未满足")
    return pairs


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def _pair_semantics(
    embedding: np.ndarray,
    coarse_category: np.ndarray,
    fine_category: np.ndarray,
    pairs: np.ndarray,
    chunk_rows: int = 4096,
) -> dict[str, Any]:
    cosine = np.empty(len(pairs), dtype=np.float32)
    unique_rows, inverse = np.unique(pairs.reshape(-1), return_inverse=True)
    cached_embedding = np.asarray(embedding[unique_rows], dtype=np.float32)
    cached_pairs = inverse.reshape(-1, 2)
    for start in range(0, len(pairs), chunk_rows):
        stop = min(start + chunk_rows, len(pairs))
        left = cached_embedding[cached_pairs[start:stop, 0]]
        right = cached_embedding[cached_pairs[start:stop, 1]]
        numerator = np.sum(left * right, axis=1)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        cosine[start:stop] = numerator / np.clip(denominator, 1.0e-12, None)
    return {
        "embedding_cosine": _summary(cosine),
        "coarse_category_agreement": float(
            np.mean(coarse_category[pairs[:, 0]] == coarse_category[pairs[:, 1]])
        ),
        "fine_category_agreement": float(
            np.mean(fine_category[pairs[:, 0]] == fine_category[pairs[:, 1]])
        ),
    }


def _prefix_group_metrics(
    sid: np.ndarray, prefix_length: int, codebook_size: int
) -> dict[str, Any]:
    counts = np.unique(
        _packed_prefix(sid, prefix_length, codebook_size), return_counts=True
    )[1]
    return {
        "distinct_prefixes": int(len(counts)),
        "singleton_prefixes": int(np.count_nonzero(counts == 1)),
        "singleton_ratio": float(np.mean(counts == 1)),
        "size_p50": float(np.percentile(counts, 50)),
        "size_p90": float(np.percentile(counts, 90)),
        "size_p99": float(np.percentile(counts, 99)),
        "size_max": int(counts.max()),
    }


def _representative_rows(
    sid: np.ndarray,
    level: int,
    codebook_size: int,
    code_count: int,
    rows_per_code: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = np.asarray(sid[:, level], dtype=np.int32)
    counts = np.bincount(tokens, minlength=codebook_size)
    top_codes = np.argsort(-counts, kind="stable")[:code_count].astype(np.int32)
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    for rank, code in enumerate(top_codes):
        candidates = np.flatnonzero(tokens == code)
        take = min(rows_per_code, len(candidates))
        selected = np.sort(rng.choice(candidates, size=take, replace=False))
        rows.append(selected.astype(np.int64))
        ranks.append(np.full(take, rank, dtype=np.int32))
    return np.concatenate(rows), np.concatenate(ranks), top_codes


def _joint_projection(
    config: SIDVisualizationConfig,
    level: int,
    sample_rows: Mapping[str, np.ndarray],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    feature_blocks: list[np.ndarray] = []
    point_lookup: dict[tuple[str, int], int] = {}
    source_rows: dict[str, list[int]] = {}
    source_paths: dict[str, Path] = {}
    for method in config.methods:
        source_key = str(method.residual_input_paths[level])
        source_paths[source_key] = method.residual_input_paths[level]
        source_rows.setdefault(source_key, []).extend(
            int(value) for value in sample_rows[method.name]
        )
    for source_key in source_rows:
        source_rows[source_key] = sorted(set(source_rows[source_key]))
        residual = np.load(source_paths[source_key], mmap_mode="r", allow_pickle=False)
        rows = np.asarray(source_rows[source_key], dtype=np.int64)
        start = sum(len(block) for block in feature_blocks)
        feature_blocks.append(np.asarray(residual[rows], dtype=np.float32))
        for offset, row in enumerate(rows):
            point_lookup[(source_key, int(row))] = start + offset

    codebook_slices: dict[str, slice] = {}
    for method in config.methods:
        start = sum(len(block) for block in feature_blocks)
        book = np.asarray(
            np.load(method.codebook_paths[level], mmap_mode="r", allow_pickle=False),
            dtype=np.float32,
        )
        feature_blocks.append(book)
        codebook_slices[method.name] = slice(start, start + len(book))
    features = np.concatenate(feature_blocks, axis=0)
    component_count = min(
        int(config.projection["pca_components"]), features.shape[0] - 1, features.shape[1]
    )
    with threadpool_limits(limits=int(config.projection["cpu_threads"])):
        pca = PCA(
            n_components=component_count,
            svd_solver="randomized",
            random_state=seed,
        )
        reduced = pca.fit_transform(features)
        coordinates = TSNE(
            n_components=2,
            perplexity=float(config.projection["tsne_perplexity"]),
            learning_rate=config.projection["tsne_learning_rate"],
            max_iter=int(config.projection["tsne_max_iter"]),
            metric=str(config.projection["tsne_metric"]),
            init="pca",
            random_state=seed,
            n_jobs=1,
        ).fit_transform(reduced)

    residual_coordinates: dict[str, np.ndarray] = {}
    codebook_coordinates: dict[str, np.ndarray] = {}
    for method in config.methods:
        source_key = str(method.residual_input_paths[level])
        indices = np.asarray(
            [point_lookup[(source_key, int(row))] for row in sample_rows[method.name]],
            dtype=np.int64,
        )
        residual_coordinates[method.name] = coordinates[indices]
        codebook_coordinates[method.name] = coordinates[codebook_slices[method.name]]
    projection_metrics = {
        "points": int(len(features)),
        "unique_residual_points": int(
            sum(len(rows) for rows in source_rows.values())
        ),
        "codebook_points": int(sum(config.rows["codebook_size"] for _ in config.methods)),
        "pca_components": component_count,
        "pca_explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
        "cpu_threads": int(config.projection["cpu_threads"]),
    }
    return residual_coordinates, codebook_coordinates, projection_metrics


def _plot_code_distribution(
    output_path: Path,
    config: SIDVisualizationConfig,
    methods_sid: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    transition_values: dict[tuple[str, int], np.ndarray] = {}
    maxima = [0.0, 0.0]
    for method in config.methods:
        sid = methods_sid[method.name]
        for row, levels in enumerate(((0, 1), (1, 2)), start=1):
            matrix = np.log1p(
                transition_matrix(sid, levels[0], levels[1], config.rows["codebook_size"])
            )
            transition_values[(method.name, row)] = matrix
            maxima[row - 1] = max(maxima[row - 1], float(matrix.max()))

    for column, method in enumerate(config.methods):
        sid = methods_sid[method.name]
        axis = axes[0, column]
        for level, color in enumerate(COLORS[:3]):
            counts = np.bincount(
                np.asarray(sid[:, level], dtype=np.int64),
                minlength=config.rows["codebook_size"],
            )
            axis.plot(
                np.arange(1, len(counts) + 1),
                np.sort(counts)[::-1],
                color=color,
                linewidth=1.4,
                label=LEVEL_NAMES[level],
            )
        axis.set_yscale("log")
        axis.set_xlabel("Code rank")
        axis.set_ylabel("POI count (log)")
        axis.set_title(method.display_name)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=3, fontsize=8)
        usage = metrics[method.name]["code_usage"]
        axis.text(
            0.02,
            0.04,
            "\n".join(
                f"{LEVEL_NAMES[level]}: active={item['active_codes']}, "
                f"ESS={item['effective_codes']:.1f}, Gini={item['gini']:.3f}"
                for level, item in enumerate(usage)
            ),
            transform=axis.transAxes,
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        for row, label in ((1, "S1 → S2"), (2, "S2 → S3")):
            matrix = transition_values[(method.name, row)]
            axes[row, column].imshow(
                matrix.T,
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=maxima[row - 1],
            )
            axes[row, column].set_xlabel(label.split(" → ")[0] + " code")
            axes[row, column].set_ylabel(label.split(" → ")[1] + " code")
            axes[row, column].set_title(f"{label} transitions, log(1+count)")
    figure.suptitle("SID code usage and transition structure", fontsize=16)
    figure.savefig(output_path, dpi=int(config.figures["dpi"]), format="png")
    plt.close(figure)


def _plot_residual_semantics(
    output_path: Path,
    config: SIDVisualizationConfig,
    residual_coordinates: Mapping[int, Mapping[str, np.ndarray]],
    codebook_coordinates: Mapping[int, Mapping[str, np.ndarray]],
    sample_ranks: Mapping[tuple[str, int], np.ndarray],
    top_codes: Mapping[tuple[str, int], np.ndarray],
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    cmap = ListedColormap(COLORS)
    for level in range(3):
        all_points = np.concatenate(
            [
                *residual_coordinates[level].values(),
                *codebook_coordinates[level].values(),
            ],
            axis=0,
        )
        padding = np.maximum(np.ptp(all_points, axis=0) * 0.04, 1.0e-6)
        limits = (
            (float(all_points[:, 0].min() - padding[0]), float(all_points[:, 0].max() + padding[0])),
            (float(all_points[:, 1].min() - padding[1]), float(all_points[:, 1].max() + padding[1])),
        )
        for column, method in enumerate(config.methods):
            axis = axes[level, column]
            points = residual_coordinates[level][method.name]
            ranks = sample_ranks[(method.name, level)]
            book = codebook_coordinates[level][method.name]
            axis.scatter(
                book[:, 0],
                book[:, 1],
                s=9,
                marker="x",
                linewidths=0.5,
                color="#777777",
                alpha=0.45,
                label="All centroids",
            )
            axis.scatter(
                points[:, 0],
                points[:, 1],
                c=ranks,
                cmap=cmap,
                vmin=-0.5,
                vmax=4.5,
                s=8,
                alpha=0.55,
                linewidths=0,
            )
            selected_codes = top_codes[(method.name, level)]
            selected_book = book[selected_codes]
            for rank, (code, coordinate) in enumerate(zip(selected_codes, selected_book, strict=True)):
                axis.scatter(
                    coordinate[0],
                    coordinate[1],
                    marker="*",
                    s=115,
                    color=COLORS[rank],
                    edgecolor="black",
                    linewidth=0.45,
                    label=f"rank {rank + 1}: code {int(code)}",
                    zorder=5,
                )
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{method.display_name} · {LEVEL_NAMES[level]}")
            if column == 2:
                axis.legend(loc="best", fontsize=6.5, framealpha=0.85)
    figure.suptitle(
        "Residual-space t-SNE: POIs assigned to the five most-used codes",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=int(config.figures["dpi"]), format="png")
    plt.close(figure)


def _plot_prefix_semantics(
    output_path: Path,
    config: SIDVisualizationConfig,
    metrics: Mapping[str, Any],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    method_colors = ("#0072B2", "#D55E00", "#009E73")
    prefix_lengths = np.arange(4)
    for color, method in zip(method_colors, config.methods, strict=True):
        semantics = metrics[method.name]["exact_prefix_semantics"]
        cosine = np.asarray(
            [semantics[str(level)]["embedding_cosine"]["mean"] for level in prefix_lengths]
        )
        axes[0].plot(
            prefix_lengths,
            cosine,
            marker="o",
            color=color,
            label=method.display_name,
        )
        axes[1].plot(
            prefix_lengths,
            [semantics[str(level)]["coarse_category_agreement"] for level in prefix_lengths],
            marker="o",
            color=color,
            linestyle="-",
            label=f"{method.display_name} / coarse",
        )
        axes[1].plot(
            prefix_lengths,
            [semantics[str(level)]["fine_category_agreement"] for level in prefix_lengths],
            marker="s",
            color=color,
            linestyle="--",
            label=f"{method.display_name} / fine",
        )
        groups = metrics[method.name]["prefix_groups"]
        levels = np.arange(1, 4)
        axes[2].plot(
            levels,
            [groups[str(level)]["size_p50"] for level in levels],
            marker="o",
            color=color,
            linestyle="--",
            label=f"{method.display_name} / p50",
        )
        axes[2].plot(
            levels,
            [groups[str(level)]["size_p90"] for level in levels],
            marker="o",
            color=color,
            label=f"{method.display_name} / p90",
        )
    axes[0].set_title("Original BGE cosine")
    axes[0].set_ylabel("Mean cosine")
    axes[1].set_title("Category agreement")
    axes[1].set_ylabel("Agreement rate")
    axes[2].set_title("Prefix bucket size")
    axes[2].set_ylabel("POIs per prefix (log)")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Exact shared prefix length" if axis is not axes[2] else "Prefix length")
        axis.set_xticks(prefix_lengths if axis is not axes[2] else np.arange(1, 4))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=6.4)
    figure.suptitle("SID prefix semantics and hierarchy", fontsize=15)
    figure.savefig(output_path, dpi=int(config.figures["dpi"]), format="png")
    plt.close(figure)


def _semantic_examples(
    config: SIDVisualizationConfig,
    methods_sid: Mapping[str, np.ndarray],
    fine_category: np.ndarray,
) -> dict[str, Any]:
    table = pq.read_table(
        config.poi_metadata_path,
        columns=["poi_id", "displayname", "category", "category_code"],
    )
    poi_id = table["poi_id"].to_pylist()
    displayname = table["displayname"].to_pylist()
    category = table["category"].to_pylist()
    category_code = table["category_code"].to_pylist()
    fine_labels: dict[int, tuple[str, str]] = {}
    for index, fine_index in enumerate(fine_category):
        fine_labels.setdefault(int(fine_index), (category[index], category_code[index]))

    result: dict[str, Any] = {}
    for method in config.methods:
        sid = methods_sid[method.name]
        method_result: dict[str, Any] = {}
        for prefix_length in range(1, 4):
            keys = _packed_prefix(sid, prefix_length, config.rows["codebook_size"])
            unique, counts = np.unique(keys, return_counts=True)
            top = np.argsort(-counts, kind="stable")[: int(config.sampling["top_prefixes_per_level"])]
            examples: list[dict[str, Any]] = []
            for group_index in top:
                key = int(unique[group_index])
                rows = np.flatnonzero(keys == key)
                fine_counts = Counter(int(value) for value in fine_category[rows])
                prefix = [int(value) for value in sid[int(rows[0]), :prefix_length]]
                examples.append(
                    {
                        "prefix": prefix,
                        "poi_count": int(len(rows)),
                        "top_fine_categories": [
                            {
                                "fine_category_index": fine_index,
                                "category": fine_labels[fine_index][0],
                                "category_code": fine_labels[fine_index][1],
                                "poi_count": count,
                            }
                            for fine_index, count in fine_counts.most_common(3)
                        ],
                        "poi_examples": [
                            {
                                "poi_row_index": int(row),
                                "poi_id": poi_id[int(row)],
                                "displayname": displayname[int(row)],
                                "category": category[int(row)],
                            }
                            for row in rows[: int(config.sampling["semantic_examples_per_prefix"])]
                        ],
                    }
                )
            method_result[str(prefix_length)] = examples
        result[method.name] = method_result
    return result


def _artifact(path: Path) -> dict[str, Any]:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _expected_artifacts(config: SIDVisualizationConfig) -> tuple[str, ...]:
    return (
        "metrics.json",
        "semantic_examples.json",
        "sampling.npz",
        str(config.figures["code_distribution"]),
        str(config.figures["residual_semantic_map"]),
        str(config.figures["prefix_semantics"]),
    )


def build_sid_visualization(config: SIDVisualizationConfig) -> dict[str, Any]:
    """Generate the frozen, side-by-side full SID visualization package."""
    started = time.perf_counter()
    checks = inspect_sid_visualization_inputs(config)
    if config.output_dir.exists():
        raise SidEvaluationDataError(f"SID 可视化输出已存在且 overwrite=false：{config.output_dir}")
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = config.output_dir.with_name(
        f".{config.output_dir.name}.{uuid.uuid4().hex}.writing"
    )
    staging.mkdir()
    try:
        methods_sid = {
            method.name: np.load(method.sid_path, mmap_mode="r", allow_pickle=False)
            for method in config.methods
        }
        embedding = np.load(config.poi_embedding_path, mmap_mode="r", allow_pickle=False)
        category_table = pq.read_table(
            config.poi_metadata_path,
            columns=["category", "fine_category_index"],
        )
        category_labels = category_table["category"].to_pylist()
        coarse_labels = [value.split(":", maxsplit=1)[0] for value in category_labels]
        coarse_lookup = {value: index for index, value in enumerate(sorted(set(coarse_labels)))}
        coarse_category = np.fromiter(
            (coarse_lookup[value] for value in coarse_labels),
            dtype=np.int32,
            count=len(coarse_labels),
        )
        fine_category = category_table["fine_category_index"].to_numpy(
            zero_copy_only=False
        ).astype(np.int32, copy=False)

        metrics: dict[str, Any] = {}
        sampling_artifacts: dict[str, np.ndarray] = {}
        sample_rows: dict[tuple[str, int], np.ndarray] = {}
        sample_ranks: dict[tuple[str, int], np.ndarray] = {}
        top_codes: dict[tuple[str, int], np.ndarray] = {}
        for method_index, method in enumerate(config.methods):
            sid = methods_sid[method.name]
            method_metrics: dict[str, Any] = {
                "display_name": method.display_name,
                "code_usage": [
                    code_usage_metrics(sid[:, level], config.rows["codebook_size"])
                    for level in range(3)
                ],
                "prefix_groups": {
                    str(level): _prefix_group_metrics(
                        sid, level, config.rows["codebook_size"]
                    )
                    for level in range(1, 4)
                },
                "exact_prefix_semantics": {},
            }
            for prefix_length in range(4):
                seed = int(config.sampling["seed"]) + method_index * 100 + prefix_length
                pairs = sample_exact_prefix_pairs(
                    sid,
                    prefix_length,
                    int(config.sampling["prefix_pairs_per_level"]),
                    seed,
                    config.rows["codebook_size"],
                )
                sampling_artifacts[
                    f"{method.name.lower()}_exact_prefix_{prefix_length}_pairs"
                ] = pairs
                method_metrics["exact_prefix_semantics"][str(prefix_length)] = (
                    _pair_semantics(
                        embedding, coarse_category, fine_category, pairs
                    )
                )
            for level in range(3):
                seed = int(config.sampling["seed"]) + method_index * 100 + 20 + level
                rows, ranks, codes = _representative_rows(
                    sid,
                    level,
                    config.rows["codebook_size"],
                    int(config.sampling["representative_codes_per_level"]),
                    int(config.sampling["rows_per_representative_code"]),
                    seed,
                )
                sample_rows[(method.name, level)] = rows
                sample_ranks[(method.name, level)] = ranks
                top_codes[(method.name, level)] = codes
                sampling_artifacts[
                    f"{method.name.lower()}_{LEVEL_NAMES[level].lower()}_rows"
                ] = rows
                sampling_artifacts[
                    f"{method.name.lower()}_{LEVEL_NAMES[level].lower()}_ranks"
                ] = ranks
                sampling_artifacts[
                    f"{method.name.lower()}_{LEVEL_NAMES[level].lower()}_top_codes"
                ] = codes
            metrics[method.name] = method_metrics
            _event("sid_visualization_method_metrics", method=method.name)

        residual_coordinates: dict[int, dict[str, np.ndarray]] = {}
        codebook_coordinates: dict[int, dict[str, np.ndarray]] = {}
        projection_metrics: dict[str, Any] = {}
        for level in range(3):
            residual, codebook, projection = _joint_projection(
                config,
                level,
                {method.name: sample_rows[(method.name, level)] for method in config.methods},
                int(config.sampling["seed"]) + 1000 + level,
            )
            residual_coordinates[level] = residual
            codebook_coordinates[level] = codebook
            projection_metrics[LEVEL_NAMES[level]] = projection
            _event("sid_visualization_projection", level=LEVEL_NAMES[level], **projection)

        metrics_payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": "sid_visualization_a0_a4_nogid_full_v1",
            "pair_sampling": {
                "definition": "uniform over POI pairs with exact shared SID prefix length",
                "pairs_per_method_and_length": int(
                    config.sampling["prefix_pairs_per_level"]
                ),
                "seed": int(config.sampling["seed"]),
                "semantic_embedding": "original active-POI BGE embedding",
            },
            "projection": {
                "definition": "joint PCA+t-SNE in each level's matched residual space",
                "level_metrics": projection_metrics,
            },
            "methods": metrics,
        }
        write_json_atomic(staging / "metrics.json", metrics_payload)
        write_json_atomic(
            staging / "semantic_examples.json",
            {
                "schema_version": SCHEMA_VERSION,
                "selection": "three largest prefix buckets per method and level",
                "methods": _semantic_examples(config, methods_sid, fine_category),
            },
        )
        np.savez_compressed(staging / "sampling.npz", **sampling_artifacts)
        _plot_code_distribution(
            staging / str(config.figures["code_distribution"]),
            config,
            methods_sid,
            metrics,
        )
        _plot_residual_semantics(
            staging / str(config.figures["residual_semantic_map"]),
            config,
            residual_coordinates,
            codebook_coordinates,
            sample_ranks,
            top_codes,
        )
        _plot_prefix_semantics(
            staging / str(config.figures["prefix_semantics"]), config, metrics
        )

        artifacts = {
            name: _artifact(staging / name) for name in _expected_artifacts(config)
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "phase": "SID-VISUALIZATION-A0-A4-GID-A4-NOGID-FULL",
            "built_at": utc_now(),
            "contract": {
                "config_path": str(config.source_path),
                "config_sha256": config.source_sha256,
                "config_signature": config.signature(),
                "poi_rows": config.rows["poi"],
                "methods": list(METHOD_ORDER),
                "final_sid_positions": ["s1", "s2", "s3"],
                "source_manifest_hashes": checks["source_hashes"],
                "code": _code_files(config),
            },
            "artifacts": artifacts,
            "source_access": {
                "frozen_train_artifacts_read": True,
                "raw_business_order_read": False,
                "business_validation_read": False,
                "business_test_read": False,
                "sid_rewritten": False,
                "downstream_started": False,
            },
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "device": "cpu",
                "seed": int(config.sampling["seed"]),
            },
            "next_status": "HOLD_FOR_NOGID_S3_REVIEW",
        }
        write_json_atomic(staging / "manifest.json", manifest)
        write_json_atomic(
            staging / "_SUCCESS",
            {"manifest_sha256": sha256_file(staging / "manifest.json")},
        )
        _validate_output_dir(config, staging, require_current_code=True)
        os.replace(staging, config.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _event(
        "sid_visualization_completed",
        output_dir=str(config.output_dir),
        elapsed_seconds=time.perf_counter() - started,
    )
    return validate_sid_visualization(config)


def _validate_output_dir(
    config: SIDVisualizationConfig,
    output_dir: Path,
    *,
    require_current_code: bool,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    marker = json.loads((output_dir / "_SUCCESS").read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != manifest_sha:
        raise SidEvaluationDataError("SID 可视化 manifest 与 _SUCCESS 不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("contract", {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase")
        != "SID-VISUALIZATION-A0-A4-GID-A4-NOGID-FULL"
        or contract.get("config_sha256") != config.source_sha256
        or contract.get("config_signature") != config.signature()
        or contract.get("methods") != list(METHOD_ORDER)
        or manifest.get("next_status") != "HOLD_FOR_NOGID_S3_REVIEW"
    ):
        raise SidEvaluationDataError("SID 可视化 manifest 状态或合同不匹配")
    if require_current_code and contract.get("code") != _code_files(config):
        raise SidEvaluationDataError("SID 可视化运行源码已变化")
    if contract.get("source_manifest_hashes") != _source_hashes(config):
        raise SidEvaluationDataError("SID 可视化 frozen source 已变化")
    expected = set(_expected_artifacts(config))
    if set(manifest.get("artifacts", {})) != expected:
        raise SidEvaluationDataError("SID 可视化 artifact 清单不完整")
    for name, entry in manifest["artifacts"].items():
        path = output_dir / name
        if (
            entry.get("file") != name
            or entry.get("bytes") != path.stat().st_size
            or entry.get("sha256") != sha256_file(path)
        ):
            raise SidEvaluationDataError(f"SID 可视化 artifact 缺失或哈希错误：{name}")
    for figure_name in (
        config.figures["code_distribution"],
        config.figures["residual_semantic_map"],
        config.figures["prefix_semantics"],
    ):
        path = output_dir / str(figure_name)
        with path.open("rb") as handle:
            signature = handle.read(8)
        if path.stat().st_size < 10_000 or signature != b"\x89PNG\r\n\x1a\n":
            raise SidEvaluationDataError(f"SID 可视化 PNG 无效：{figure_name}")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    if list(metrics.get("methods", {})) != list(METHOD_ORDER):
        raise SidEvaluationDataError("SID 可视化 metrics 方法顺序不匹配")
    if manifest.get("source_access", {}).get("downstream_started") is not False:
        raise SidEvaluationDataError("SID 可视化越过了只读边界")
    return {
        "status": "sid_visualization_validated",
        "output_dir": str(output_dir),
        "manifest_sha256": manifest_sha,
        "artifacts": sorted(expected),
        "methods": list(METHOD_ORDER),
        "next_status": manifest["next_status"],
    }


def validate_sid_visualization(config: SIDVisualizationConfig) -> dict[str, Any]:
    """Validate output hashes, source bindings, plots, and the stop boundary."""
    inspect_sid_visualization_inputs(config)
    return _validate_output_dir(config, config.output_dir, require_current_code=True)

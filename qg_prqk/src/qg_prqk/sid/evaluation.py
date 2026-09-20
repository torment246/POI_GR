"""A0/A4 static evaluation, prefix probes, and SID publication."""

from __future__ import annotations

import json
import math
import os
import resource
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.evaluation_config import SidEvaluationConfig
from qg_prqk.sid.evaluation_data import SidEvaluationDataError, SidEvaluationInputs, SidMethodArtifacts, load_sid_evaluation_inputs


SCHEMA_VERSION = "qg-prqk-p8-static-evaluation-v1"
EPSILON = 1.0e-12
POI_SID_SCHEMA = pa.schema(
    [
        pa.field("poi_row_index", pa.int64(), nullable=False),
        pa.field("poi_id", pa.string(), nullable=False),
        pa.field("s1", pa.int32(), nullable=False),
        pa.field("s2", pa.int32(), nullable=False),
        pa.field("s3", pa.int32(), nullable=False),
        pa.field("bucket_id", pa.int64(), nullable=False),
    ]
)
BUCKET_SCHEMA = pa.schema(
    [
        pa.field("bucket_id", pa.int64(), nullable=False),
        pa.field("s1", pa.int32(), nullable=False),
        pa.field("s2", pa.int32(), nullable=False),
        pa.field("s3", pa.int32(), nullable=False),
        pa.field("bucket_size", pa.int64(), nullable=False),
        pa.field("first_sorted_offset", pa.int64(), nullable=False),
        pa.field("is_collision", pa.bool_(), nullable=False),
    ]
)
CATEGORY_S1_SCHEMA = pa.schema(
    [
        pa.field("s1", pa.int32(), nullable=False),
        pa.field("coarse_category_index", pa.int32(), nullable=False),
        pa.field("poi_count", pa.int64(), nullable=False),
        pa.field("share_within_s1", pa.float64(), nullable=False),
    ]
)
CATEGORY_S1_S2_SCHEMA = pa.schema(
    [
        pa.field("s1", pa.int32(), nullable=False),
        pa.field("s2", pa.int32(), nullable=False),
        pa.field("fine_category_index", pa.int32(), nullable=False),
        pa.field("poi_count", pa.int64(), nullable=False),
        pa.field("share_within_s1_s2", pa.float64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class CandidateIndex:
    """Sorted parent keys and slices of allowed child token IDs."""

    parents: np.ndarray
    offsets: np.ndarray
    codes: np.ndarray


@dataclass(frozen=True)
class EdgeArrays:
    """One layer of graph edges mapped into local Query and POI row space."""

    query_rows: np.ndarray
    poi_rows: np.ndarray
    depths: np.ndarray
    weights: np.ndarray


def _event(stage: str, **values: Any) -> None:
    print(json.dumps({"time": utc_now(), "stage": stage, **values}, ensure_ascii=False), flush=True)


def output_directory(config: SidEvaluationConfig, gate: str) -> Path:
    if gate not in ("sample", "full"):
        raise SidEvaluationDataError("P8 只开放 sample/full")
    settings = config.gates[gate]
    path = config.output_dir / (
        f"{gate}_{int(settings['query_rows']):06d}q_{int(settings['poi_rows']):06d}p"
    )
    if not path.resolve().is_relative_to(config.output_dir.resolve()):
        raise SidEvaluationDataError("P8 输出越出冻结 namespace")
    return path


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise SidEvaluationDataError(f"P8 输出已存在：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, target: Path) -> None:
    if target.exists():
        raise SidEvaluationDataError(f"P8 输出已存在：{target}")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.writing")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    if path.exists():
        raise SidEvaluationDataError(f"P8 输出已存在：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        pq.write_table(table, temporary, compression="zstd", row_group_size=65_536)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise SidEvaluationDataError(f"P8 输出已存在：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        temporary.write_text(value, encoding="utf-8")
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
    elif path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        result.update(rows=parquet.metadata.num_rows, schema=str(parquet.schema_arrow))
    return result


def _code_files() -> dict[str, str]:
    directory = Path(__file__).parent
    files = {
        "sid/evaluation_config.py": directory / "evaluation_config.py",
        "sid/evaluation_data.py": directory / "evaluation_data.py",
        "sid/evaluation.py": directory / "evaluation.py",
        "commands/evaluate_sid.py": directory.parent / "commands/evaluate_sid.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def _distribution(values: np.ndarray, size: int = 512) -> dict[str, Any]:
    labels = np.asarray(values, dtype=np.int64)
    counts = np.bincount(labels, minlength=size).astype(np.float64)
    if len(labels) == 0 or np.any(labels < 0) or np.any(labels >= size):
        raise SidEvaluationDataError("P8 token assignment 非法")
    active = counts[counts > 0]
    ordered = np.sort(counts)
    positions = np.arange(1, size + 1, dtype=np.float64)
    gini = float(
        np.sum((2.0 * positions - size - 1.0) * ordered) / (size * counts.sum())
    )
    probabilities = active / active.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return {
        "active_codes": int(len(active)),
        "utilization": float(len(active) / size),
        "kish_ess": float(counts.sum() ** 2 / np.sum(counts * counts)),
        "kish_ess_ratio": float(counts.sum() ** 2 / np.sum(counts * counts) / size),
        "gini": gini,
        "entropy_nats": entropy,
        "normalized_entropy": float(entropy / math.log(size)),
        "cluster_size": _summary(active),
    }


def _summary(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return {"count": 0}
    return {
        "count": int(len(data)),
        "mean": float(np.mean(data)),
        "min": float(np.min(data)),
        "p50": float(np.quantile(data, 0.50)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(np.max(data)),
    }


def _path_metrics(values: np.ndarray) -> dict[str, Any]:
    paths = np.asarray(values, dtype=np.int64)
    _, counts = np.unique(paths, axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / len(paths)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    theoretical = 512 ** paths.shape[1]
    return {
        "levels": int(paths.shape[1]),
        "theoretical_paths": theoretical,
        "distinct": int(len(counts)),
        "distinct_ratio_of_pois": float(len(counts) / len(paths)),
        "utilization_of_theoretical_paths": float(len(counts) / theoretical),
        "path_entropy_nats": entropy,
        "path_entropy_normalized_by_capacity": float(entropy / math.log(theoretical)),
        "collision_excess": int(np.sum(counts - 1)),
        "bucket": _summary(counts),
    }


def partition_metrics(partitions: np.ndarray, categories: np.ndarray) -> dict[str, Any]:
    """Return purity and geometric-mean normalized mutual information."""
    keys = np.asarray(partitions, dtype=np.int64)
    values = np.asarray(categories, dtype=np.int64)
    if keys.ndim == 1:
        keys = keys[:, None]
    if keys.ndim != 2 or len(keys) != len(values) or not len(values):
        raise SidEvaluationDataError("P8 partition/category 未对齐")
    _, partition_ids = np.unique(keys, axis=0, return_inverse=True)
    _, category_ids = np.unique(values, return_inverse=True)
    partition_count = int(partition_ids.max()) + 1
    category_count = int(category_ids.max()) + 1
    pair = partition_ids.astype(np.int64) * category_count + category_ids
    unique_pair, pair_counts = np.unique(pair, return_counts=True)
    pair_partitions = unique_pair // category_count
    pair_categories = unique_pair % category_count
    partition_sizes = np.bincount(partition_ids, minlength=partition_count).astype(np.float64)
    category_sizes = np.bincount(category_ids, minlength=category_count).astype(np.float64)
    majority = np.zeros(partition_count, dtype=np.int64)
    np.maximum.at(majority, pair_partitions, pair_counts)
    total = float(len(values))
    p_pair = pair_counts.astype(np.float64) / total
    mutual_information = float(
        np.sum(
            p_pair
            * np.log(
                p_pair
                / (
                    (partition_sizes[pair_partitions] / total)
                    * (category_sizes[pair_categories] / total)
                )
            )
        )
    )
    p_partition = partition_sizes[partition_sizes > 0] / total
    p_category = category_sizes[category_sizes > 0] / total
    partition_entropy = float(-np.sum(p_partition * np.log(p_partition)))
    category_entropy = float(-np.sum(p_category * np.log(p_category)))
    denominator = math.sqrt(partition_entropy * category_entropy)
    nmi = mutual_information / denominator if denominator > EPSILON else 0.0
    conditional_entropy = category_entropy - mutual_information
    return {
        "partition_count": partition_count,
        "category_count": category_count,
        "purity": float(majority.sum() / total),
        "mutual_information_nats": mutual_information,
        "nmi_geometric": float(nmi),
        "category_given_partition_entropy_nats": float(max(0.0, conditional_entropy)),
    }


def _pack_gid(gid6: np.ndarray) -> np.ndarray:
    result = np.zeros(len(gid6), dtype=np.int64)
    for column in range(gid6.shape[1]):
        result = (result << 5) | np.asarray(gid6[:, column], dtype=np.int64)
    return result


def _local_gid_metrics(gid6: np.ndarray, sid: np.ndarray) -> dict[str, Any]:
    gids = _pack_gid(gid6)
    _, gid_counts = np.unique(gids, return_counts=True)
    gid_pairs = int(np.sum(gid_counts * (gid_counts - 1) // 2))
    composite = np.column_stack((gids, np.asarray(sid, dtype=np.int64)))
    _, collision_counts = np.unique(composite, axis=0, return_counts=True)
    collision_pairs = int(np.sum(collision_counts * (collision_counts - 1) // 2))
    _, exact_counts = np.unique(composite, axis=0, return_counts=True)
    probabilities = exact_counts.astype(np.float64) / len(sid)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return {
        "definition": "exact_pairwise_separation_within_same_gid6",
        "gid6_groups": int(len(gid_counts)),
        "same_gid6_unordered_pairs": gid_pairs,
        "same_gid6_same_sid_pairs": collision_pairs,
        "pair_collision_rate": float(collision_pairs / gid_pairs) if gid_pairs else 0.0,
        "local_entity_separation_rate": float(1.0 - collision_pairs / gid_pairs)
        if gid_pairs
        else 1.0,
        "gid6_sid_bucket": {
            "distinct": int(len(exact_counts)),
            "distinct_ratio_of_pois": float(len(exact_counts) / len(sid)),
            "collision_excess": int(np.sum(exact_counts - 1)),
            "path_entropy_nats": entropy,
            "bucket": _summary(exact_counts),
            "capacity_normalization": None,
            "capacity_note": "GID6 uses observed geohash cells, not a fourth 512-way SID layer",
        },
    }


def _codes_per_category(assignments: np.ndarray, categories: np.ndarray) -> dict[str, Any]:
    keys = np.asarray(assignments, dtype=np.int64)
    if keys.ndim == 1:
        keys = keys[:, None]
    _, path_ids = np.unique(keys, axis=0, return_inverse=True)
    pairs = np.unique(
        np.column_stack((np.asarray(categories, dtype=np.int64), path_ids)), axis=0
    )
    _, counts = np.unique(pairs[:, 0], return_counts=True)
    return _summary(counts)


def method_static_metrics(
    method: SidMethodArtifacts,
    *,
    coarse_categories: np.ndarray,
    fine_categories: np.ndarray,
    gid6: np.ndarray,
) -> dict[str, Any]:
    """Compute capacity, category, bucket, residual, and local-separation metrics."""
    sid = np.asarray(method.poi_sid)
    paired_codebooks = []
    for poi_book, query_book in zip(
        method.poi_codebooks, method.query_codebooks, strict=True
    ):
        poi = np.asarray(poi_book, dtype=np.float64)
        query = np.asarray(query_book, dtype=np.float64)
        poi /= np.linalg.norm(poi, axis=1, keepdims=True).clip(min=EPSILON)
        query /= np.linalg.norm(query, axis=1, keepdims=True).clip(min=EPSILON)
        paired_codebooks.append(_summary(np.sum(poi * query, axis=1)))
    return {
        "paired_poi_query_codebook_cosine": paired_codebooks,
        "codebook_levels": [
            _distribution(sid[:, level]) for level in range(3)
        ],
        "paths": {
            "s1": _path_metrics(sid[:, :1]),
            "s1_s2": _path_metrics(sid[:, :2]),
            "s1_s2_s3": _path_metrics(sid),
        },
        "category": {
            "coarse_vs_s1": partition_metrics(sid[:, 0], coarse_categories),
            "fine_vs_s1_s2": partition_metrics(sid[:, :2], fine_categories),
            "s1_codes_per_coarse_category": _codes_per_category(
                sid[:, 0], coarse_categories
            ),
            "s1_s2_paths_per_fine_category": _codes_per_category(
                sid[:, :2], fine_categories
            ),
        },
        "residual_energy": [dict(item) for item in method.residual_metrics],
        "same_gid6_local_separation": _local_gid_metrics(gid6, sid),
    }


def _candidate_index(parent_keys: np.ndarray, child_codes: np.ndarray) -> CandidateIndex:
    pairs = np.unique(
        np.asarray(parent_keys, dtype=np.int64) * 512
        + np.asarray(child_codes, dtype=np.int64)
    )
    parents = pairs // 512
    codes = (pairs % 512).astype(np.int32)
    unique_parents, first, counts = np.unique(
        parents, return_index=True, return_counts=True
    )
    offsets = np.append(first, first[-1] + counts[-1]).astype(np.int64)
    return CandidateIndex(unique_parents, offsets, codes)


def _s3_parent_keys(gid6: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    return (_pack_gid(gid6) << 18) | (np.asarray(s1, dtype=np.int64) << 9) | np.asarray(
        s2, dtype=np.int64
    )


def _normalize_torch(values):
    import torch

    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(1.0e-12)


def _projection_residual(values, centroids, assignments):
    selected = centroids[assignments]
    denominator = (selected * selected).sum(dim=1).clamp_min(1.0e-12)
    coefficient = (values * selected).sum(dim=1) / denominator
    return _normalize_torch(values - coefficient[:, None] * selected)


def _predict_all(
    values: np.ndarray,
    rows: np.ndarray,
    codebook: np.ndarray,
    *,
    device,
    chunk_rows: int,
) -> np.ndarray:
    import torch

    centroids = _normalize_torch(
        torch.from_numpy(np.ascontiguousarray(codebook, dtype=np.float32)).to(device)
    )
    predictions = np.empty(len(rows), dtype=np.int32)
    for start in range(0, len(rows), chunk_rows):
        stop = min(start + chunk_rows, len(rows))
        block = torch.from_numpy(
            np.ascontiguousarray(values[rows[start:stop]], dtype=np.float32)
        ).to(device)
        predictions[start:stop] = (
            torch.argmax(_normalize_torch(block) @ centroids.T, dim=1)
            .cpu()
            .numpy()
            .astype(np.int32)
        )
    return predictions


def _candidate_mask(index: CandidateIndex, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(parents, dtype=np.int64)
    mask = np.zeros((len(keys), 512), dtype=np.bool_)
    positions = np.searchsorted(index.parents, keys)
    valid = positions < len(index.parents)
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= index.parents[positions[valid_indices]] == keys[valid_indices]
    for position in np.unique(positions[valid]):
        rows = np.flatnonzero(valid & (positions == position))
        start, stop = index.offsets[position : position + 2]
        mask[np.ix_(rows, index.codes[start:stop])] = True
    return mask, valid


def _candidate_sizes(index: CandidateIndex, parents: np.ndarray) -> np.ndarray:
    keys = np.asarray(parents, dtype=np.int64)
    positions = np.searchsorted(index.parents, keys)
    result = np.zeros(len(keys), dtype=np.int32)
    valid = positions < len(index.parents)
    rows = np.flatnonzero(valid)
    valid[rows] &= index.parents[positions[rows]] == keys[rows]
    positions = positions[valid]
    result[valid] = (index.offsets[positions + 1] - index.offsets[positions]).astype(
        np.int32
    )
    return result


def _predict_conditioned(
    values: np.ndarray,
    query_rows: np.ndarray,
    prefix_tokens: np.ndarray,
    codebooks: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate_index: CandidateIndex,
    candidate_parents: np.ndarray,
    *,
    predicted_level: int,
    device,
    chunk_rows: int,
) -> np.ndarray:
    import torch

    torch_codebooks = tuple(
        _normalize_torch(
            torch.from_numpy(np.ascontiguousarray(book, dtype=np.float32)).to(device)
        )
        for book in codebooks
    )
    predictions = np.full(len(query_rows), -1, dtype=np.int32)
    for start in range(0, len(query_rows), chunk_rows):
        stop = min(start + chunk_rows, len(query_rows))
        block = _normalize_torch(
            torch.from_numpy(
                np.ascontiguousarray(values[query_rows[start:stop]], dtype=np.float32)
            ).to(device)
        )
        prefix = np.asarray(prefix_tokens[start:stop], dtype=np.int64)
        if prefix.ndim == 1:
            prefix = prefix[:, None]
        valid_prefix = np.all(prefix >= 0, axis=1)
        for level in range(predicted_level):
            safe = np.maximum(prefix[:, level], 0)
            assignment = torch.from_numpy(safe).to(device)
            block = _projection_residual(block, torch_codebooks[level], assignment)
        mask, valid_candidates = _candidate_mask(
            candidate_index, np.asarray(candidate_parents[start:stop], dtype=np.int64)
        )
        valid = valid_prefix & valid_candidates
        if np.any(valid):
            scores = block @ torch_codebooks[predicted_level].T
            scores.masked_fill_(~torch.from_numpy(mask).to(device), float("-inf"))
            chosen = torch.argmax(scores, dim=1).cpu().numpy().astype(np.int32)
            chosen[~valid] = -1
            predictions[start:stop] = chosen
    return predictions


def _map_edges(inputs: SidEvaluationInputs, table: pa.Table, level: int) -> EdgeArrays:
    selected = table.filter(pc.equal(table["layer"], pa.scalar(level, type=pa.int8())))
    global_poi = selected["poi_row_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    poi_rows = np.searchsorted(inputs.selected_poi_rows, global_poi)
    if np.any(poi_rows >= len(inputs.selected_poi_rows)) or not np.array_equal(
        inputs.selected_poi_rows[poi_rows], global_poi
    ):
        raise SidEvaluationDataError(f"P8 S{level} edge target 未进入 POI 闭包")
    node_rows = inputs.query_nodes["node_row"].to_numpy(zero_copy_only=False).astype(np.int64)
    global_query = selected["node_row"].to_numpy(zero_copy_only=False).astype(np.int64)
    query_rows = np.searchsorted(node_rows, global_query)
    if np.any(query_rows >= len(node_rows)) or not np.array_equal(node_rows[query_rows], global_query):
        raise SidEvaluationDataError(f"P8 S{level} edge Query 未进入节点表")
    depths = inputs.query_nodes["supervision_depth"].to_numpy(zero_copy_only=False)[query_rows]
    return EdgeArrays(
        query_rows=query_rows,
        poi_rows=poi_rows,
        depths=np.asarray(depths, dtype=np.int8),
        weights=selected["edge_weight"].to_numpy(zero_copy_only=False).astype(np.float64),
    )


def _agreement(hit: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    values = np.asarray(hit, dtype=np.bool_)
    mass = np.asarray(weights, dtype=np.float64)
    total = float(mass.sum())
    return {
        "edge_rows": int(len(values)),
        "edge_weight_sum": total,
        "weighted_accuracy": float(mass[values].sum() / total) if total else 0.0,
        "unweighted_accuracy": float(np.mean(values)) if len(values) else 0.0,
    }


def _by_depth(hit: np.ndarray, edge: EdgeArrays) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for depth, label in ((1, "D1"), (2, "D2"), (3, "D3")):
        mask = edge.depths == depth
        if np.any(mask):
            result[label] = _agreement(hit[mask], edge.weights[mask])
    return result


def _alignment_metrics(query_sid: np.ndarray, poi_sid: np.ndarray, edges: Mapping[int, EdgeArrays]) -> dict[str, Any]:
    levels: dict[str, Any] = {}
    cumulative: dict[str, Any] = {}
    for level, edge in edges.items():
        token_hit = query_sid[edge.query_rows, level - 1] == poi_sid[edge.poi_rows, level - 1]
        prefix_hit = np.all(
            query_sid[edge.query_rows, :level] == poi_sid[edge.poi_rows, :level], axis=1
        )
        levels[f"s{level}"] = {"overall": _agreement(token_hit, edge.weights), "by_depth": _by_depth(token_hit, edge)}
        cumulative[f"s1_to_s{level}"] = {"overall": _agreement(prefix_hit, edge.weights), "by_depth": _by_depth(prefix_hit, edge)}
    return {"token": levels, "cumulative_prefix": cumulative}


def prefix_probe(
    inputs: SidEvaluationInputs,
    method: SidMethodArtifacts,
    *,
    method_name: str,
    s3_parent_mode: str = "gid6_s1_s2",
    device,
    chunk_rows: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run content-only nearest-code probes with correct-prefix candidate scopes."""
    sid = np.asarray(method.poi_sid, dtype=np.int32)
    codebooks = method.query_codebooks
    depths = inputs.query_nodes["supervision_depth"].to_numpy(zero_copy_only=False).astype(np.int8)
    edges = {
        1: _map_edges(inputs, inputs.edges_s1_s2, 1),
        2: _map_edges(inputs, inputs.edges_s1_s2, 2),
        3: _map_edges(inputs, inputs.edges_s3, 3),
    }
    all_query_rows = np.arange(len(depths), dtype=np.int64)
    _event("p8_probe_s1_started", method=method_name, rows=len(all_query_rows))
    free_sid = np.full((len(depths), 3), -1, dtype=np.int32)
    free_sid[:, 0] = _predict_all(
        inputs.query_residual_s0,
        all_query_rows,
        codebooks[0],
        device=device,
        chunk_rows=chunk_rows,
    )

    s2_index = _candidate_index(sid[:, 0], sid[:, 1])
    active_s2 = np.flatnonzero(depths >= 2).astype(np.int64)
    free_sid[active_s2, 1] = _predict_conditioned(
        inputs.query_residual_s0,
        active_s2,
        free_sid[active_s2, :1],
        codebooks,
        s2_index,
        free_sid[active_s2, 0].astype(np.int64),
        predicted_level=1,
        device=device,
        chunk_rows=chunk_rows,
    )
    edge2 = edges[2]
    target2 = sid[edge2.poi_rows]
    teacher_s2 = _predict_conditioned(
        inputs.query_residual_s0,
        edge2.query_rows,
        target2[:, :1],
        codebooks,
        s2_index,
        target2[:, 0].astype(np.int64),
        predicted_level=1,
        device=device,
        chunk_rows=chunk_rows,
    )
    _event("p8_probe_s2_completed", method=method_name, teacher_edges=len(edge2.query_rows))

    if s3_parent_mode == "gid6_s1_s2":
        parent = _s3_parent_keys(inputs.gid6, sid[:, 0], sid[:, 1])
    elif s3_parent_mode == "s1_s2":
        parent = sid[:, 0].astype(np.int64) * 512 + sid[:, 1].astype(np.int64)
    else:
        raise SidEvaluationDataError(f"P8 未知 S3 parent mode：{s3_parent_mode}")
    s3_index = _candidate_index(parent, sid[:, 2])
    edge3 = edges[3]
    target3 = sid[edge3.poi_rows]
    if s3_parent_mode == "gid6_s1_s2":
        edge_gid = _pack_gid(inputs.gid6[edge3.poi_rows])
        teacher_parent = (edge_gid << 18) | (target3[:, 0].astype(np.int64) << 9) | target3[:, 1]
    else:
        teacher_parent = target3[:, 0].astype(np.int64) * 512 + target3[:, 1].astype(np.int64)
    teacher_s3 = _predict_conditioned(
        inputs.query_residual_s0,
        edge3.query_rows,
        target3[:, :2],
        codebooks,
        s3_index,
        teacher_parent,
        predicted_level=2,
        device=device,
        chunk_rows=chunk_rows,
    )
    free_prefix = free_sid[edge3.query_rows, :2]
    if s3_parent_mode == "gid6_s1_s2":
        free_parent = (edge_gid << 18) | (free_prefix[:, 0].astype(np.int64) << 9) | free_prefix[:, 1]
    else:
        free_parent = free_prefix[:, 0].astype(np.int64) * 512 + free_prefix[:, 1].astype(np.int64)
    free_s3 = _predict_conditioned(
        inputs.query_residual_s0,
        edge3.query_rows,
        free_prefix,
        codebooks,
        s3_index,
        free_parent,
        predicted_level=2,
        device=device,
        chunk_rows=chunk_rows,
    )
    if len(np.unique(edge3.query_rows)) != len(edge3.query_rows):
        raise SidEvaluationDataError("P8 D3 Prefix Probe 要求每个 Exact Query 恰好一条 S3 edge")
    free_sid[edge3.query_rows, 2] = free_s3
    _event("p8_probe_s3_completed", method=method_name, teacher_edges=len(edge3.query_rows))

    edge1 = edges[1]
    target1 = sid[edge1.poi_rows]
    s1_hit = free_sid[edge1.query_rows, 0] == target1[:, 0]
    s2_hit = teacher_s2 == target2[:, 1]
    s3_hit = teacher_s3 == target3[:, 2]
    free_alignment = _alignment_metrics(free_sid, sid, edges)
    s3_metric_name = (
        "query_correct_gid6_s1_s2_to_s3"
        if s3_parent_mode == "gid6_s1_s2"
        else "query_correct_s1_s2_to_s3"
    )
    payload = {
        "method": method_name,
        "semantics": {
            "query_to_s1": "content-only cosine over all 512 codes",
            "query_correct_s1_to_s2": "correct S1 projection residual; candidates observed under correct S1",
            s3_metric_name: (
                "correct S1/S2 projection residual; candidates observed under correct GID6/S1/S2"
                if s3_parent_mode == "gid6_s1_s2"
                else "correct S1/S2 projection residual; candidates observed under correct S1/S2 only"
            ),
            "cumulative": (
                "autoregressive semantic S1/S2 with correct GID6 supplied at S3"
                if s3_parent_mode == "gid6_s1_s2"
                else "autoregressive S1/S2/S3 without GID input"
            ),
            "s3_parent_mode": s3_parent_mode,
            "multi_target_aggregation": "frozen complete edges; edge_weight and unweighted",
        },
        "teacher_forced": {
            "query_to_s1": {"overall": _agreement(s1_hit, edge1.weights), "by_depth": _by_depth(s1_hit, edge1)},
            "query_correct_s1_to_s2": {"overall": _agreement(s2_hit, edge2.weights), "by_depth": _by_depth(s2_hit, edge2)},
            s3_metric_name: {"overall": _agreement(s3_hit, edge3.weights), "by_depth": _by_depth(s3_hit, edge3)},
        },
        "free_running": free_alignment,
        "prediction_coverage": {
            "s1": float(np.mean(free_sid[:, 0] >= 0)),
            "s2_applicable": float(np.mean(free_sid[active_s2, 1] >= 0)),
            "s3_applicable": float(np.mean(free_sid[edge3.query_rows, 2] >= 0)),
        },
        "candidate_set_size": {
            "s2_teacher_forced": _summary(_candidate_sizes(s2_index, target2[:, 0])),
            "s2_free_running": _summary(
                _candidate_sizes(s2_index, free_sid[active_s2, 0])
            ),
            "s3_teacher_forced": _summary(
                _candidate_sizes(s3_index, teacher_parent)
            ),
            "s3_free_running": _summary(_candidate_sizes(s3_index, free_parent)),
            "s3_teacher_forced_singleton_ratio": float(
                np.mean(_candidate_sizes(s3_index, teacher_parent) == 1)
            ),
        },
    }
    return payload, free_sid


def _category_distribution_s1(sid: np.ndarray, categories: np.ndarray) -> pa.Table:
    labels = np.asarray(sid[:, 0], dtype=np.int64)
    values = np.asarray(categories, dtype=np.int64)
    category_count = int(values.max()) + 1
    pair, counts = np.unique(labels * category_count + values, return_counts=True)
    s1, category = pair // category_count, pair % category_count
    totals = np.bincount(labels, minlength=512)
    rows = [
        {
            "s1": int(token),
            "coarse_category_index": int(cat),
            "poi_count": int(count),
            "share_within_s1": float(count / totals[token]),
        }
        for token, cat, count in zip(s1, category, counts, strict=True)
    ]
    return pa.Table.from_pylist(rows, schema=CATEGORY_S1_SCHEMA)


def _category_distribution_s1_s2(sid: np.ndarray, categories: np.ndarray) -> pa.Table:
    paths = np.asarray(sid[:, 0], dtype=np.int64) * 512 + sid[:, 1]
    values = np.asarray(categories, dtype=np.int64)
    category_count = int(values.max()) + 1
    pair, counts = np.unique(paths * category_count + values, return_counts=True)
    prefix, category = pair // category_count, pair % category_count
    totals = np.bincount(paths, minlength=512 * 512)
    rows = [
        {
            "s1": int(path // 512),
            "s2": int(path % 512),
            "fine_category_index": int(cat),
            "poi_count": int(count),
            "share_within_s1_s2": float(count / totals[path]),
        }
        for path, cat, count in zip(prefix, category, counts, strict=True)
    ]
    return pa.Table.from_pylist(rows, schema=CATEGORY_S1_S2_SCHEMA)


def _poi_sid_table(inputs: SidEvaluationInputs) -> pa.Table:
    sid = np.asarray(inputs.a4.poi_sid, dtype=np.int32)
    _, bucket_ids = np.unique(sid, axis=0, return_inverse=True)
    return pa.Table.from_arrays(
        [
            pa.array(inputs.selected_poi_rows, type=pa.int64()),
            inputs.poi_ids,
            pa.array(sid[:, 0], type=pa.int32()),
            pa.array(sid[:, 1], type=pa.int32()),
            pa.array(sid[:, 2], type=pa.int32()),
            pa.array(bucket_ids.astype(np.int64)),
        ],
        schema=POI_SID_SCHEMA,
    )


def _bucket_table(sid: np.ndarray) -> pa.Table:
    paths, counts = np.unique(np.asarray(sid, dtype=np.int32), axis=0, return_counts=True)
    offsets = np.cumsum(np.concatenate(([0], counts[:-1]))).astype(np.int64)
    return pa.Table.from_arrays(
        [
            pa.array(np.arange(len(paths), dtype=np.int64)),
            pa.array(paths[:, 0], type=pa.int32()),
            pa.array(paths[:, 1], type=pa.int32()),
            pa.array(paths[:, 2], type=pa.int32()),
            pa.array(counts.astype(np.int64)),
            pa.array(offsets),
            pa.array(counts > 1),
        ],
        schema=BUCKET_SCHEMA,
    )


def _comparison(a0: Mapping[str, Any], a4: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "distinct_sid": {
            "a0": a0["paths"]["s1_s2_s3"]["distinct"],
            "a4": a4["paths"]["s1_s2_s3"]["distinct"],
            "delta": a4["paths"]["s1_s2_s3"]["distinct"] - a0["paths"]["s1_s2_s3"]["distinct"],
        },
        "coarse_s1_purity": {
            "a0": a0["category"]["coarse_vs_s1"]["purity"],
            "a4": a4["category"]["coarse_vs_s1"]["purity"],
            "delta": a4["category"]["coarse_vs_s1"]["purity"] - a0["category"]["coarse_vs_s1"]["purity"],
        },
        "fine_s1_s2_purity": {
            "a0": a0["category"]["fine_vs_s1_s2"]["purity"],
            "a4": a4["category"]["fine_vs_s1_s2"]["purity"],
            "delta": a4["category"]["fine_vs_s1_s2"]["purity"] - a0["category"]["fine_vs_s1_s2"]["purity"],
        },
        "gid6_local_separation": {
            "a0": a0["same_gid6_local_separation"]["local_entity_separation_rate"],
            "a4": a4["same_gid6_local_separation"]["local_entity_separation_rate"],
            "delta": a4["same_gid6_local_separation"]["local_entity_separation_rate"] - a0["same_gid6_local_separation"]["local_entity_separation_rate"],
        },
    }


def _review_assessment(
    static: Mapping[str, Any], probe: Mapping[str, Any]
) -> dict[str, Any]:
    teacher = probe["methods"]
    probe_deltas = {
        key: (
            teacher["A4"]["teacher_forced"][key]["overall"]["weighted_accuracy"]
            - teacher["A0"]["teacher_forced"][key]["overall"]["weighted_accuracy"]
        )
        for key in (
            "query_to_s1",
            "query_correct_s1_to_s2",
            "query_correct_gid6_s1_s2_to_s3",
        )
    }
    cumulative_delta = (
        teacher["A4"]["free_running"]["cumulative_prefix"]["s1_to_s3"]["overall"]["weighted_accuracy"]
        - teacher["A0"]["free_running"]["cumulative_prefix"]["s1_to_s3"]["overall"]["weighted_accuracy"]
    )
    structure = static["comparison"]
    structure_improved = (
        structure["distinct_sid"]["delta"] > 0
        and structure["coarse_s1_purity"]["delta"] > 0
        and structure["fine_s1_s2_purity"]["delta"] > 0
        and structure["gid6_local_separation"]["delta"] >= 0
    )
    predictability_degraded = any(value < 0 for value in probe_deltas.values()) or cumulative_delta < 0
    return {
        "outcome": "REVIEW_REQUIRED" if predictability_degraded else "COMPLETED",
        "structure_improved_all_headlines": structure_improved,
        "prefix_probe_weighted_deltas": probe_deltas,
        "cumulative_s1_s3_weighted_delta": cumulative_delta,
        "query_predictability_degraded": predictability_degraded,
        "automatic_parameter_change": False,
        "automatic_downstream_launch": False,
        "next_status": "HOLD_FOR_REVIEW",
    }


def _data_quality(inputs: SidEvaluationInputs, *, gate: str) -> dict[str, Any]:
    p25 = inputs.p2_5_metrics
    nodes = inputs.query_nodes
    depths = nodes["supervision_depth"].to_numpy(zero_copy_only=False)
    gate_counts = {f"D{depth}": int(np.count_nonzero(depths == depth)) for depth in (1, 2, 3)}
    return {
        "full_train_query_depth": dict(p25["queries"]),
        "evaluation_gate_depth_counts": gate_counts,
        "poi_coverage": p25["poi_coverage"],
        "category_mapping": p25["category_mapping"],
        "concentration_and_entropy_global": p25["distributions"],
        "concentration_and_entropy_by_depth": _p2_5_depth_distributions(inputs)
        if gate == "full"
        else None,
        "manual_sample_source": {
            "source": "frozen_p2_5_deterministic_samples",
            "rows_per_depth": {name: len(rows) for name, rows in inputs.p2_5_samples.items()},
            "artifact": "data_quality_samples.json",
        },
        "gate": gate,
        "formal_metric_basis": gate == "full",
    }


def _p2_5_depth_distributions(inputs: SidEvaluationInputs) -> dict[str, Any]:
    base = inputs.source_paths["p2_5_manifest"].parent
    depth_files = sorted((base / "query_category_depth.parquet").glob("*.parquet"))
    stats_files = sorted((base / "query_category_stats.parquet").glob("*.parquet"))
    if not depth_files or len(depth_files) != len(stats_files):
        raise SidEvaluationDataError("P8 P2.5 depth/stats 分片不完整")
    columns = (
        "fine_concentration",
        "fine_normalized_entropy",
        "coarse_concentration",
        "coarse_normalized_entropy",
    )
    accumulated: dict[str, dict[str, list[np.ndarray]]] = {
        f"D{depth}": {name: [] for name in columns} for depth in range(4)
    }
    for depth_file, stats_file in zip(depth_files, stats_files, strict=True):
        depth_table = pq.read_table(
            depth_file, columns=["query_id", "supervision_depth"]
        )
        stats_table = pq.read_table(stats_file, columns=["query_id", *columns])
        if not np.array_equal(
            depth_table["query_id"].to_numpy(zero_copy_only=False),
            stats_table["query_id"].to_numpy(zero_copy_only=False),
        ):
            raise SidEvaluationDataError("P8 P2.5 depth/stats Query 行序不一致")
        depths = depth_table["supervision_depth"].to_numpy(zero_copy_only=False)
        for depth in range(4):
            mask = depths == depth
            for name in columns:
                accumulated[f"D{depth}"][name].append(
                    stats_table[name].to_numpy(zero_copy_only=False)[mask]
                )
    return {
        label: {
            name: _summary(np.concatenate(parts)) for name, parts in metrics.items()
        }
        for label, metrics in accumulated.items()
    }


def _report(
    gate: str,
    static: Mapping[str, Any],
    probe: Mapping[str, Any],
    review: Mapping[str, Any],
) -> str:
    comparison = static["comparison"]
    lines = [
        f"# P8-CAT {gate} A0/A4 静态评测",
        "",
        "本报告只比较内部初始化 A0 与完整方法 A4；不包含外部基线、Final PID、Dedup、Tokenizer、Trie 或 SFT。",
        "",
        "## 核心结果",
        "",
        "| 指标 | A0 | A4 | 差值 |",
        "|---|---:|---:|---:|",
        f"| Distinct SID | {comparison['distinct_sid']['a0']:,} | {comparison['distinct_sid']['a4']:,} | {comparison['distinct_sid']['delta']:+,} |",
        f"| coarse purity(S1) | {comparison['coarse_s1_purity']['a0']:.6f} | {comparison['coarse_s1_purity']['a4']:.6f} | {comparison['coarse_s1_purity']['delta']:+.6f} |",
        f"| fine purity(S1+S2) | {comparison['fine_s1_s2_purity']['a0']:.6f} | {comparison['fine_s1_s2_purity']['a4']:.6f} | {comparison['fine_s1_s2_purity']['delta']:+.6f} |",
        f"| GID6 局部分离率 | {comparison['gid6_local_separation']['a0']:.6f} | {comparison['gid6_local_separation']['a4']:.6f} | {comparison['gid6_local_separation']['delta']:+.6f} |",
        "",
        "## Prefix Probe",
        "",
        "| 指标 | A0 | A4 |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("query_to_s1", "Query → S1"),
        ("query_correct_s1_to_s2", "Query + correct S1 → S2"),
        ("query_correct_gid6_s1_s2_to_s3", "Query + correct GID6/S1/S2 → S3"),
    ):
        left = probe["methods"]["A0"]["teacher_forced"][key]["overall"]["weighted_accuracy"]
        right = probe["methods"]["A4"]["teacher_forced"][key]["overall"]["weighted_accuracy"]
        lines.append(f"| {label}（weighted） | {left:.6f} | {right:.6f} |")
    lines.extend(
        [
            "",
            "Prefix Probe 是冻结 Query view 上的 content-only nearest-code 探针；S2/S3 使用正确前缀重算 projection residual，并把候选限制到该正确前缀在对应 POI SID 中实际出现的 child code。完整分 D1/D2/D3、free-running 前缀及未加权结果见 `prefix_probe_metrics.json`。",
            "",
        "## 停止边界",
        "",
            f"评测结论为 `{review['outcome']}`，`NEXT_ACTION: HOLD_FOR_REVIEW`。本阶段不会自动进入 Final PID、Dedup、Tokenizer、Trie、Qwen SFT、外部基线或其他消融。",
            "",
        ]
    )
    return "\n".join(lines)


def _copy_publication_arrays(inputs: SidEvaluationInputs, output: Path) -> None:
    p6_dir = inputs.source_paths["p6_manifest"].parent
    p7_dir = inputs.source_paths["p7_manifest"].parent
    sources = {
        "poi_codebook_s1.npy": p6_dir / "poi_codebook_s1.npy",
        "poi_codebook_s2.npy": p6_dir / "poi_codebook_s2.npy",
        "poi_codebook_s3.npy": p7_dir / "poi_codebook_s3.npy",
        "query_codebook_s1.npy": p6_dir / "query_codebook_s1.npy",
        "query_codebook_s2.npy": p6_dir / "query_codebook_s2.npy",
        "query_codebook_s3.npy": p7_dir / "query_codebook_s3.npy",
        "query_assignments_s1_s2_s3.npy": p7_dir / "query_sid_s1_s2_s3.npy",
    }
    for name, source in sources.items():
        _atomic_copy(source, output / name)
    _atomic_npy(output / "poi_assignments_s1_s2_s3.npy", np.asarray(inputs.a4.poi_sid))


def _expected_artifacts() -> Iterable[str]:
    yield from (
        "config_resolved.json",
        "poi_codebook_s1.npy",
        "poi_codebook_s2.npy",
        "poi_codebook_s3.npy",
        "query_codebook_s1.npy",
        "query_codebook_s2.npy",
        "query_codebook_s3.npy",
        "poi_assignments_s1_s2_s3.npy",
        "query_assignments_s1_s2_s3.npy",
        "poi_sid.parquet",
        "sid_bucket_index.parquet",
        "category_distribution_s1.parquet",
        "category_distribution_s1_s2.parquet",
        "residual_diagnostics.json",
        "static_metrics.json",
        "prefix_probe_metrics.json",
        "data_quality_samples.json",
        "report.md",
    )


def build_sid_evaluation(
    config: SidEvaluationConfig,
    *,
    gate: str,
    device: str = "cuda",
    chunk_rows: int = 8192,
) -> dict[str, Any]:
    """Build one immutable P8 sample/full comparison and A4 SID publication."""
    started = time.perf_counter()
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise SidEvaluationDataError("P8 Prefix Probe 要求 CUDA；不得静默回退 CPU")
    if chunk_rows <= 0:
        raise SidEvaluationDataError("P8 chunk_rows 必须为正整数")
    output = output_directory(config, gate)
    if output.exists():
        raise SidEvaluationDataError(f"P8 overwrite=false：{output}")
    inputs = load_sid_evaluation_inputs(config, gate=gate)
    output.mkdir(parents=True)
    write_json_atomic(output / "config_resolved.json", config.resolved_payload())
    target = torch.device(device)
    torch.cuda.reset_peak_memory_stats(target)
    _event("p8_static_metrics_started", gate=gate)
    a0_static = method_static_metrics(
        inputs.a0,
        coarse_categories=inputs.coarse_category_indices,
        fine_categories=inputs.fine_category_indices,
        gid6=inputs.gid6,
    )
    a4_static = method_static_metrics(
        inputs.a4,
        coarse_categories=inputs.coarse_category_indices,
        fine_categories=inputs.fine_category_indices,
        gid6=inputs.gid6,
    )
    edges = {
        1: _map_edges(inputs, inputs.edges_s1_s2, 1),
        2: _map_edges(inputs, inputs.edges_s1_s2, 2),
        3: _map_edges(inputs, inputs.edges_s3, 3),
    }
    a4_static["query_poi_training_assignment_alignment"] = _alignment_metrics(
        np.asarray(inputs.a4.query_sid), np.asarray(inputs.a4.poi_sid), edges
    )
    static_metrics = {
        "schema_version": SCHEMA_VERSION,
        "gate": gate,
        "formal_metric_basis": gate == "full",
        "data_quality": _data_quality(inputs, gate=gate),
        "methods": {"A0": a0_static, "A4": a4_static},
        "comparison": _comparison(a0_static, a4_static),
        "scope": {
            "only_a0_a4": True,
            "a0_is_internal_initialization": True,
            "external_baseline_run": False,
        },
    }
    write_json_atomic(output / "static_metrics.json", static_metrics)
    write_json_atomic(
        output / "residual_diagnostics.json",
        {
            "A0": a0_static["residual_energy"],
            "A4": a4_static["residual_energy"],
            "source_only": True,
        },
    )
    write_json_atomic(output / "data_quality_samples.json", dict(inputs.p2_5_samples))

    probes: dict[str, Any] = {}
    for name, method in (("A0", inputs.a0), ("A4", inputs.a4)):
        probes[name], _ = prefix_probe(
            inputs,
            method,
            method_name=name,
            device=target,
            chunk_rows=chunk_rows,
        )
        torch.cuda.empty_cache()
    prefix_metrics = {
        "schema_version": SCHEMA_VERSION,
        "gate": gate,
        "formal_metric_basis": gate == "full",
        "contract": dict(config.prefix_probe),
        "methods": probes,
    }
    write_json_atomic(output / "prefix_probe_metrics.json", prefix_metrics)

    _copy_publication_arrays(inputs, output)
    _atomic_parquet(output / "poi_sid.parquet", _poi_sid_table(inputs))
    _atomic_parquet(
        output / "sid_bucket_index.parquet", _bucket_table(np.asarray(inputs.a4.poi_sid))
    )
    _atomic_parquet(
        output / "category_distribution_s1.parquet",
        _category_distribution_s1(np.asarray(inputs.a4.poi_sid), inputs.coarse_category_indices),
    )
    _atomic_parquet(
        output / "category_distribution_s1_s2.parquet",
        _category_distribution_s1_s2(np.asarray(inputs.a4.poi_sid), inputs.fine_category_indices),
    )
    review = _review_assessment(static_metrics, prefix_metrics)
    _atomic_text(output / "report.md", _report(gate, static_metrics, prefix_metrics, review))

    artifacts = {name: _artifact(output / name) for name in _expected_artifacts()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "phase": f"P8-CAT-{gate.upper()}",
        "role": "A0_A4_STATIC_EVALUATION_AND_A4_SID_CANDIDATE",
        "built_at": utc_now(),
        "contract": {
            "config_signature": config.signature(),
            "config_sha256": config.source_sha256,
            "gate": gate,
            "poi_rows": len(inputs.selected_poi_rows),
            "query_rows": len(inputs.query_nodes),
            "d3_query_rows": int(np.count_nonzero(inputs.query_nodes["supervision_depth"].to_numpy() == 3)),
            "codebook_sizes": [512, 512, 512],
            "methods": list(config.comparison["methods"]),
            "prefix_probe": dict(config.prefix_probe),
            "publication": dict(config.publication),
            "source_paths": {name: str(path) for name, path in inputs.source_paths.items()},
            "source_hashes": dict(inputs.source_hashes),
            "code": _code_files(),
        },
        "headline_metrics": {
            "comparison": static_metrics["comparison"],
            "prefix_probe_weighted": {
                name: {
                    key: values["overall"]["weighted_accuracy"]
                    for key, values in payload["teacher_forced"].items()
                }
                for name, payload in probes.items()
            },
        },
        "evaluation_outcome": review,
        "artifacts": artifacts,
        "source_access": {
            "qg_frozen_train_artifacts_read": True,
            "raw_business_order_read": False,
            "business_validation_read": False,
            "business_test_read": False,
            "external_baseline_run": False,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
            "chunk_rows": chunk_rows,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(target) / (1024 * 1024),
        },
        "gate_evidence": {
            "sample_role": "CODE_DATA_GPU_ARTIFACT_SMOKE_ONLY" if gate == "sample" else None,
            "formal_metric_basis": gate == "full",
            "sid_force_uniqueness_used": False,
            "downstream_started": False,
            "manual_review_required": True,
            "evaluation_outcome": review["outcome"],
        },
        "next_status": "HOLD_FOR_REVIEW",
    }
    write_json_atomic(output / "manifest.json", manifest)
    validate_sid_evaluation(config, gate=gate, require_success_marker=False)
    write_json_atomic(output / "_SUCCESS", {"manifest_sha256": sha256_file(output / "manifest.json")})
    _event(
        "p8_completed",
        gate=gate,
        a0_distinct_sid=static_metrics["comparison"]["distinct_sid"]["a0"],
        a4_distinct_sid=static_metrics["comparison"]["distinct_sid"]["a4"],
        next_status="HOLD_FOR_REVIEW",
    )
    return validate_sid_evaluation(config, gate=gate)


def validate_sid_evaluation(
    config: SidEvaluationConfig,
    *,
    gate: str,
    require_success_marker: bool = True,
) -> dict[str, Any]:
    """Independently verify P8 sources, artifacts, publication, and stop boundary."""
    output = output_directory(config, gate)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != f"P8-CAT-{gate.upper()}"
        or manifest.get("next_status") != "HOLD_FOR_REVIEW"
        or manifest.get("contract", {}).get("config_signature") != config.signature()
        or manifest.get("contract", {}).get("config_sha256") != config.source_sha256
    ):
        raise SidEvaluationDataError("P8 manifest schema/配置/停止状态不匹配")
    inspected = load_sid_evaluation_inputs(config, gate=gate)
    if manifest["contract"]["source_hashes"] != dict(inspected.source_hashes):
        raise SidEvaluationDataError("P8 manifest 冻结来源哈希不匹配")
    if manifest["contract"]["code"] != _code_files():
        raise SidEvaluationDataError("P8 当前源码与构建源码哈希不一致")
    for name in _expected_artifacts():
        path = output / name
        if name not in manifest["artifacts"] or _artifact(path) != manifest["artifacts"][name]:
            raise SidEvaluationDataError(f"P8 artifact 哈希/shape/schema 不匹配：{name}")
    assignments = np.load(
        output / "poi_assignments_s1_s2_s3.npy", mmap_mode="r", allow_pickle=False
    )
    query_assignments = np.load(
        output / "query_assignments_s1_s2_s3.npy", mmap_mode="r", allow_pickle=False
    )
    if not np.array_equal(assignments, inspected.a4.poi_sid) or not np.array_equal(
        query_assignments, inspected.a4.query_sid
    ):
        raise SidEvaluationDataError("P8 发布 assignment 未逐值冻结 A4")
    poi_sid = pq.read_table(output / "poi_sid.parquet")
    _, expected_bucket_ids = np.unique(assignments, axis=0, return_inverse=True)
    if (
        not poi_sid.schema.equals(POI_SID_SCHEMA)
        or len(poi_sid) != len(assignments)
        or not np.array_equal(
            np.column_stack(
                [poi_sid[name].to_numpy(zero_copy_only=False) for name in ("s1", "s2", "s3")]
            ),
            assignments,
        )
        or not np.array_equal(
            poi_sid["bucket_id"].to_numpy(zero_copy_only=False),
            expected_bucket_ids,
        )
    ):
        raise SidEvaluationDataError("P8 poi_sid.parquet 未逐值对应 A4 assignment")
    buckets = pq.read_table(output / "sid_bucket_index.parquet")
    if not buckets.schema.equals(BUCKET_SCHEMA) or pc.sum(buckets["bucket_size"]).as_py() != len(assignments):
        raise SidEvaluationDataError("P8 bucket index schema/计数不守恒")
    static = json.loads((output / "static_metrics.json").read_text(encoding="utf-8"))
    probe = json.loads((output / "prefix_probe_metrics.json").read_text(encoding="utf-8"))
    if (
        static.get("scope", {}).get("only_a0_a4") is not True
        or static.get("scope", {}).get("external_baseline_run") is not False
        or probe.get("contract") != dict(config.prefix_probe)
        or any(
            manifest.get("source_access", {}).get(name) is not False
            for name in ("raw_business_order_read", "business_validation_read", "business_test_read", "external_baseline_run")
        )
        or manifest.get("gate_evidence", {}).get("downstream_started") is not False
        or manifest.get("evaluation_outcome", {}).get("next_status") != "HOLD_FOR_REVIEW"
        or manifest.get("evaluation_outcome", {}).get("outcome")
        not in ("COMPLETED", "REVIEW_REQUIRED")
    ):
        raise SidEvaluationDataError("P8 比较范围、数据读取或下游停止合同不匹配")
    manifest_sha = sha256_file(manifest_path)
    if require_success_marker:
        marker = json.loads((output / "_SUCCESS").read_text(encoding="utf-8"))
        if marker.get("manifest_sha256") != manifest_sha:
            raise SidEvaluationDataError("P8 _SUCCESS 与 manifest SHA256 不一致")
    return {
        "status": "p8_evaluation_validated",
        "phase": manifest["phase"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "poi_rows": len(assignments),
        "query_rows": len(query_assignments),
        "headline_metrics": manifest["headline_metrics"],
        "next_status": "HOLD_FOR_REVIEW",
    }

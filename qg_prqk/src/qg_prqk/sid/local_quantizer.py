"""Geo-aware S3 refinement over a frozen S1/S2 endpoint."""

from __future__ import annotations

import gc
import json
import os
import resource
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.base_quantizer import EPSILON, _distribution_metrics, _hard_centroid_update, _normalize_torch, _soft_centroid_update, projection_residual
from qg_prqk.sid.relational_quantizer import LevelGraph, _assign_query, _graph_metrics, _graph_penalty, _shrink_query_centroids, _soft_shrink_query_centroids, _to_device_normalized_float32, objective_increase_streak, warmup_weights
from qg_prqk.sid.local_config import LocalCodebookConfig
from qg_prqk.sid.local_data import HARD_EDGE_SCHEMA, LocalCodebookDataError, LocalCodebookInputs, P7_POI_METADATA_SCHEMA, load_local_codebook_inputs
from qg_prqk.sid.geo import GEO_FEATURE_DIM, pack_gid


SCHEMA_VERSION = "qg-prqk-p7-geo-aware-s3-v1"
DEFAULT_CHUNK_ROWS = 8192


def _torch():
    try:
        import torch
    except ImportError as error:
        raise LocalCodebookDataError("P7-CAT 需要 PyTorch") from error
    return torch


def _event(stage: str, **values: Any) -> None:
    print(json.dumps({"time": utc_now(), "stage": stage, **values}, ensure_ascii=False), flush=True)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise LocalCodebookDataError(f"输出已存在且禁止覆盖：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_tensor_npy(path: Path, values, *, dtype: np.dtype, chunk_rows: int) -> None:
    if path.exists():
        raise LocalCodebookDataError(f"输出已存在且禁止覆盖：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        target_dtype = np.dtype(dtype)
        with temporary.open("wb") as handle:
            np.lib.format.write_array_header_2_0(
                handle,
                {"descr": np.lib.format.dtype_to_descr(target_dtype), "fortran_order": False, "shape": tuple(values.shape)},
            )
            for start in range(0, len(values), chunk_rows):
                stop = min(start + chunk_rows, len(values))
                block = np.ascontiguousarray(values[start:stop].detach().float().cpu().numpy(), dtype=target_dtype)
                handle.write(block.tobytes(order="C"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        result.update(shape=list(values.shape), dtype=str(values.dtype))
    elif path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        result.update(rows=parquet.metadata.num_rows)
    return result


def output_directory(config: LocalCodebookConfig, gate: str) -> Path:
    if gate not in ("sample", "full") or gate not in config.gates:
        raise LocalCodebookDataError("P7 只开放 sample/full")
    settings = config.gates[gate]
    result = config.output_dir / f"{gate}_{int(settings['d3_query_rows']):06d}q_{int(settings['poi_rows']):06d}p"
    if not result.resolve().is_relative_to(config.output_dir.resolve()):
        raise LocalCodebookDataError("P7 输出越出冻结 namespace")
    return result


def next_status_for_gate(gate: str) -> str:
    statuses = {"sample": "HOLD_FOR_P7_SAMPLE_REVIEW", "full": "HOLD_FOR_P7_FULL_REVIEW"}
    try:
        return statuses[gate]
    except KeyError as error:
        raise LocalCodebookDataError(f"未知 P7 gate：{gate}") from error


def _s3_graph(inputs: LocalCodebookInputs) -> LevelGraph:
    global_poi = inputs.s3_edges["poi_row_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    poi_local = np.searchsorted(inputs.selected_poi_rows, global_poi)
    if not np.array_equal(inputs.selected_poi_rows[poi_local], global_poi):
        raise LocalCodebookDataError("P7 S3 target 未进入 P6 POI 行空间")
    global_query = inputs.s3_edges["node_row"].to_numpy(zero_copy_only=False).astype(np.int64)
    query_local = np.searchsorted(inputs.d3_query_rows, global_query)
    if not np.array_equal(inputs.d3_query_rows[query_local], global_query):
        raise LocalCodebookDataError("P7 S3 edge Query 未进入 D3 行空间")
    weights = inputs.s3_edges["edge_weight"].to_numpy(zero_copy_only=False).astype(np.float32)
    return LevelGraph(poi_rows=poi_local, query_rows=query_local, weights=weights)


def _geo_centroids(geo_values, assignments, codebook_size: int):
    torch = _torch()
    nonzero = torch.linalg.vector_norm(geo_values, dim=1) > EPSILON
    sums = torch.zeros((codebook_size, geo_values.shape[1]), dtype=torch.float32, device=geo_values.device)
    counts = torch.zeros(codebook_size, dtype=torch.float32, device=geo_values.device)
    if bool(nonzero.any()):
        sums.index_add_(0, assignments[nonzero], geo_values[nonzero])
        counts.index_add_(0, assignments[nonzero], torch.ones(int(nonzero.sum().item()), device=geo_values.device))
    valid = counts > 0
    centroids = torch.zeros_like(sums)
    if bool(valid.any()):
        centroids[valid] = _normalize_torch(sums[valid] / counts[valid, None])
    return centroids, valid, counts


def _hard_collision_penalty(inputs: LocalCodebookInputs, poi_assignments, codebook_size: int):
    torch = _torch()
    graph = inputs.hard_graph
    penalty = torch.zeros((len(poi_assignments), codebook_size), dtype=torch.float32, device=poi_assignments.device)
    totals = torch.zeros(len(poi_assignments), dtype=torch.float32, device=poi_assignments.device)
    if len(graph.weights):
        sources = torch.from_numpy(graph.poi_rows).to(poi_assignments.device)
        neighbors = torch.from_numpy(graph.neighbor_rows).to(poi_assignments.device)
        weights = torch.from_numpy(graph.weights).to(poi_assignments.device)
        labels = poi_assignments[neighbors]
        penalty.index_put_((sources, labels), weights, accumulate=True)
        totals.index_add_(0, sources, weights)
        penalty /= totals.clamp_min(EPSILON)[:, None]
    return penalty, totals


def _assign_poi_s3(
    inputs: LocalCodebookInputs,
    values,
    centroids,
    query_graph: LevelGraph,
    query_assignments,
    geo_values,
    geo_centroids,
    geo_valid,
    old_poi_assignments,
    *,
    graph_weight: float,
    geo_weight: float,
    hard_weight: float,
    candidate_codes: int,
    chunk_rows: int,
):
    torch = _torch()
    neighbor = query_assignments.detach().cpu().numpy()[query_graph.query_rows]
    graph_cost = _graph_penalty(
        len(values), query_graph.poi_rows, neighbor, query_graph.weights, len(centroids), device=values.device
    )
    hard_cost, _ = _hard_collision_penalty(inputs, old_poi_assignments, len(centroids))
    labels = torch.empty(len(values), dtype=torch.int64, device=values.device)
    selected = {
        "content": torch.empty(len(values), dtype=torch.float32, device=values.device),
        "graph": torch.empty(len(values), dtype=torch.float32, device=values.device),
        "geo": torch.empty(len(values), dtype=torch.float32, device=values.device),
        "hard": torch.empty(len(values), dtype=torch.float32, device=values.device),
    }
    normalized = _normalize_torch(centroids)
    for start in range(0, len(values), chunk_rows):
        stop = min(start + chunk_rows, len(values))
        content = 1.0 - values[start:stop] @ normalized.T
        candidate = torch.topk(content, k=candidate_codes, dim=1, largest=False, sorted=True).indices
        content_local = content.gather(1, candidate)
        graph_local = graph_cost[start:stop].gather(1, candidate)
        hard_local = hard_cost[start:stop].gather(1, candidate)
        geo_cosine = geo_values[start:stop] @ geo_centroids.T
        geo_cost = 1.0 - geo_cosine
        geo_cost[:, ~geo_valid] = 1.0
        singleton = torch.linalg.vector_norm(geo_values[start:stop], dim=1) <= EPSILON
        geo_cost[singleton] = 0.0
        geo_local = geo_cost.gather(1, candidate)
        objective = content_local + graph_weight * graph_local + geo_weight * geo_local + hard_weight * hard_local
        local_choice = torch.argmin(objective, dim=1)
        chosen = candidate.gather(1, local_choice[:, None]).squeeze(1)
        labels[start:stop] = chosen
        for name, matrix in (("content", content_local), ("graph", graph_local), ("geo", geo_local), ("hard", hard_local)):
            selected[name][start:stop] = matrix.gather(1, local_choice[:, None]).squeeze(1)
    del graph_cost, hard_cost
    return labels, selected


def _hard_metrics(inputs: LocalCodebookInputs, assignments: np.ndarray) -> dict[str, Any]:
    graph = inputs.hard_graph
    if not len(graph.weights):
        return {"directed_edge_rows": 0, "weight_sum": 0.0, "collision_rate": 0.0, "weighted_collision_rate": 0.0}
    collisions = assignments[graph.poi_rows] == assignments[graph.neighbor_rows]
    weight_sum = float(graph.weights.sum())
    return {
        "directed_edge_rows": int(len(graph.weights)),
        "weight_sum": weight_sum,
        "collision_rate": float(np.mean(collisions)),
        "weighted_collision_rate": float(graph.weights[collisions].sum() / weight_sum),
    }


def _s3_warmup(iteration: int, settings: Mapping[str, Any], warmup: Mapping[str, Any]) -> tuple[float, float, float, float]:
    query, graph, structure_fraction = warmup_weights(
        iteration,
        query_target=float(settings["query_distortion_weight"]),
        graph_target=float(settings["graph_alignment_weight"]),
        category_target=1.0,
        zero_weight_iters=int(warmup["zero_weight_iters"]),
        ramp_end_iter=int(warmup["linear_ramp_end_iter"]),
    )
    return query, graph, structure_fraction * float(settings["geo_weight"]), structure_fraction * float(settings["hard_collision_weight"])


def fit_s3(
    inputs: LocalCodebookInputs,
    poi_values,
    query_values,
    initial_centroids,
    *,
    settings: Mapping[str, Any],
    prqk: Mapping[str, Any],
    warmup: Mapping[str, Any],
    local_refinement: Mapping[str, Any],
    chunk_rows: int,
):
    """Fit the single P7 layer with content, exact Query, Geo, and hard-edge views."""
    torch = _torch()
    codebook_size = len(initial_centroids)
    query_graph = _s3_graph(inputs)
    poi_centroids = _normalize_torch(initial_centroids.float())
    poi_assignments = torch.from_numpy(np.asarray(inputs.initial_poi_assignments_s3, dtype=np.int64)).to(poi_values.device)
    query_centroids = poi_centroids.clone()
    query_weight, graph_weight, geo_weight, hard_weight = _s3_warmup(1, settings, warmup)
    query_assignments, initial_query_distances = _assign_query(
        query_values, query_centroids, query_graph, poi_assignments,
        query_weight=query_weight, graph_weight=graph_weight, chunk_rows=chunk_rows,
    )
    query_centroids, _ = _shrink_query_centroids(
        query_values, query_assignments, poi_centroids, float(settings["query_centroid_shrinkage"])
    )
    geo_values = torch.from_numpy(
        np.asarray(inputs.geo_features, dtype=np.float32)
    ).to(poi_values.device)
    geo_centroids, geo_valid, _ = _geo_centroids(
        geo_values, poi_assignments, codebook_size
    )
    trace: list[dict[str, Any]] = []
    previous_objective: float | None = None
    stable_rounds = 0
    increase_streak = 0
    converged = False
    candidate_codes = int(local_refinement["candidate_codes"])
    for iteration in range(1, int(prqk["max_iter"]) + 1):
        query_weight, graph_weight, geo_weight, hard_weight = _s3_warmup(iteration, settings, warmup)
        old_poi = poi_assignments
        old_query = query_assignments
        poi_assignments, components = _assign_poi_s3(
            inputs, poi_values, poi_centroids, query_graph, query_assignments,
            geo_values, geo_centroids, geo_valid, old_poi,
            graph_weight=graph_weight, geo_weight=geo_weight, hard_weight=hard_weight,
            candidate_codes=candidate_codes, chunk_rows=chunk_rows,
        )
        query_assignments, query_distances = _assign_query(
            query_values, query_centroids, query_graph, poi_assignments,
            query_weight=query_weight, graph_weight=graph_weight, chunk_rows=chunk_rows,
        )
        poi_centroids, poi_counts, empty = _hard_centroid_update(
            poi_values, poi_assignments, 1.0 - components["content"], codebook_size
        )
        query_centroids, query_counts = _shrink_query_centroids(
            query_values, query_assignments, poi_centroids, float(settings["query_centroid_shrinkage"])
        )
        geo_centroids, geo_valid, geo_counts = _geo_centroids(geo_values, poi_assignments, codebook_size)
        poi_change = float((poi_assignments != old_poi).float().mean().item())
        query_change = float((query_assignments != old_query).float().mean().item())
        graph_result = _graph_metrics(
            query_graph, poi_assignments.detach().cpu().numpy(), query_assignments.detach().cpu().numpy()
        )
        sums = {name: float(value.sum().item()) for name, value in components.items()}
        objective = (
            sums["content"] + query_weight * float(query_distances.sum().item())
            + graph_weight * sums["graph"] + geo_weight * sums["geo"] + hard_weight * sums["hard"]
        )
        relative = None if previous_objective is None else (previous_objective - objective) / max(abs(previous_objective), EPSILON)
        trace.append(
            {
                "stage": "hard_alternating",
                "iteration": iteration,
                "objective": objective,
                "relative_objective_improvement": relative,
                "poi_assignment_change": poi_change,
                "query_assignment_change": query_change,
                "weights": {"query": query_weight, "graph": graph_weight, "geo": geo_weight, "hard_collision": hard_weight},
                "components": {**sums, "query_content": float(query_distances.sum().item())},
                "active_poi_codes": int(torch.count_nonzero(poi_counts).item()),
                "active_query_codes": int(torch.count_nonzero(query_counts).item()),
                "active_geo_codes": int(torch.count_nonzero(geo_counts).item()),
                "empty_poi_codes_reinitialized": empty,
                "weighted_graph_disagreement": graph_result["weighted_disagreement"],
            }
        )
        if relative is not None:
            increase_streak = objective_increase_streak(
                increase_streak, relative, iteration=iteration,
                warmup_end_iteration=int(warmup["linear_ramp_end_iter"]),
                tolerance=float(prqk["objective_rel_tol"]),
            )
            if increase_streak >= 3:
                raise LocalCodebookDataError("P7 S3 复合 objective 连续 3 轮明显升高")
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
                poi_values, poi_centroids, topk=int(refinement["topk"]),
                beta=float(refinement["beta"]), chunk_rows=chunk_rows,
            )
            query_centroids = _soft_shrink_query_centroids(
                query_values, query_centroids, poi_centroids, topk=int(refinement["topk"]),
                beta=float(refinement["beta"]), tau=float(settings["query_centroid_shrinkage"]),
            )
            geo_centroids, geo_valid, _ = _geo_centroids(geo_values, old_poi, codebook_size)
            poi_assignments, components = _assign_poi_s3(
                inputs, poi_values, poi_centroids, query_graph, query_assignments,
                geo_values, geo_centroids, geo_valid, old_poi,
                graph_weight=float(settings["graph_alignment_weight"]),
                geo_weight=float(settings["geo_weight"]), hard_weight=float(settings["hard_collision_weight"]),
                candidate_codes=candidate_codes, chunk_rows=chunk_rows,
            )
            query_assignments, query_distances = _assign_query(
                query_values, query_centroids, query_graph, poi_assignments,
                query_weight=float(settings["query_distortion_weight"]),
                graph_weight=float(settings["graph_alignment_weight"]), chunk_rows=chunk_rows,
            )
            trace.append(
                {
                    "stage": "topk_soft_centroid_refinement",
                    "iteration": iteration,
                    "poi_content_mean": float(components["content"].mean().item()),
                    "query_content_mean": float(query_distances.mean().item()),
                    "poi_assignment_change": float((poi_assignments != old_poi).float().mean().item()),
                    "query_assignment_change": float((query_assignments != old_query).float().mean().item()),
                    "topk": int(refinement["topk"]),
                    "beta": float(refinement["beta"]),
                }
            )

    for sweep in range(1, int(local_refinement["sweeps"]) + 1):
        old_poi = poi_assignments
        geo_centroids, geo_valid, _ = _geo_centroids(geo_values, old_poi, codebook_size)
        poi_assignments, components = _assign_poi_s3(
            inputs, poi_values, poi_centroids, query_graph, query_assignments,
            geo_values, geo_centroids, geo_valid, old_poi,
            graph_weight=float(settings["graph_alignment_weight"]),
            geo_weight=float(settings["geo_weight"]), hard_weight=float(settings["hard_collision_weight"]),
            candidate_codes=candidate_codes, chunk_rows=chunk_rows,
        )
        query_assignments, query_distances = _assign_query(
            query_values, query_centroids, query_graph, poi_assignments,
            query_weight=float(settings["query_distortion_weight"]),
            graph_weight=float(settings["graph_alignment_weight"]), chunk_rows=chunk_rows,
        )
        trace.append(
            {
                "stage": "bounded_local_refinement",
                "iteration": sweep,
                "candidate_codes": candidate_codes,
                "poi_assignment_change": float((poi_assignments != old_poi).float().mean().item()),
                "hard_weight": float(settings["hard_collision_weight"]),
            }
        )

    geo_centroids, geo_valid, _ = _geo_centroids(geo_values, poi_assignments, codebook_size)
    final_poi, final_components = _assign_poi_s3(
        inputs, poi_values, poi_centroids, query_graph, query_assignments,
        geo_values, geo_centroids, geo_valid, poi_assignments,
        graph_weight=float(settings["graph_alignment_weight"]),
        geo_weight=float(settings["geo_weight"]), hard_weight=float(settings["hard_collision_weight"]),
        candidate_codes=candidate_codes, chunk_rows=chunk_rows,
    )
    final_query, final_query_distances = _assign_query(
        query_values, query_centroids, query_graph, final_poi,
        query_weight=float(settings["query_distortion_weight"]),
        graph_weight=float(settings["graph_alignment_weight"]), chunk_rows=chunk_rows,
    )
    return poi_centroids, query_centroids, geo_centroids, final_poi, final_query, {
        "hard_converged": converged,
        "hard_iterations": sum(row["stage"] == "hard_alternating" for row in trace),
        "soft_refinement_iterations": sum(row["stage"] == "topk_soft_centroid_refinement" for row in trace),
        "local_refinement_sweeps": sum(row["stage"] == "bounded_local_refinement" for row in trace),
        "initial_query_content_mean": float(initial_query_distances.mean().item()),
        "final_poi_content_mean": float(final_components["content"].mean().item()),
        "final_query_content_mean": float(final_query_distances.mean().item()),
        "final_geo_distortion_mean": float(final_components["geo"].mean().item()),
        "final_hard_collision_cost_mean": float(final_components["hard"].mean().item()),
        "trace": trace,
    }


def _prefix_metrics(values: np.ndarray) -> dict[str, Any]:
    _, counts = np.unique(values, axis=0, return_counts=True)
    return {
        "distinct": int(len(counts)),
        "distinct_ratio": float(len(counts) / len(values)),
        "collision_excess": int(np.sum(counts - 1)),
        "bucket_mean": float(np.mean(counts)),
        "bucket_p50": float(np.quantile(counts, 0.50)),
        "bucket_p90": float(np.quantile(counts, 0.90)),
        "bucket_p95": float(np.quantile(counts, 0.95)),
        "bucket_p99": float(np.quantile(counts, 0.99)),
        "bucket_max": int(counts.max()),
    }


def _proxy_subset_metrics(inputs: LocalCodebookInputs, assignments: np.ndarray) -> dict[str, Any]:
    categories = inputs.poi_metadata["category"].to_pylist()
    result: dict[str, Any] = {}
    for name, keyword in (("mall", "商场"), ("hospital", "医院"), ("station", "车站")):
        mask = np.fromiter((keyword in value for value in categories), dtype=np.bool_, count=len(categories))
        edge_mask = mask[inputs.hard_graph.poi_rows] & mask[inputs.hard_graph.neighbor_rows]
        if np.any(edge_mask):
            collisions = assignments[inputs.hard_graph.poi_rows[edge_mask]] == assignments[inputs.hard_graph.neighbor_rows[edge_mask]]
            result[name] = {"definition": f"category_path_contains_{keyword}", "poi_rows": int(mask.sum()), "hard_edge_rows": int(edge_mask.sum()), "collision_rate": float(np.mean(collisions))}
        else:
            result[name] = {"definition": f"category_path_contains_{keyword}", "poi_rows": int(mask.sum()), "hard_edge_rows": 0, "collision_rate": None}
    return result


def _code_files() -> dict[str, str]:
    directory = Path(__file__).parent
    files = {
        "sid/base_quantizer.py": directory / "base_quantizer.py",
        "sid/relational_quantizer.py": directory / "relational_quantizer.py",
        "sid/local_config.py": directory / "local_config.py",
        "sid/geo.py": directory / "geo.py",
        "sid/local_data.py": directory / "local_data.py",
        "sid/local_quantizer.py": directory / "local_quantizer.py",
        "commands/local_codebook.py": directory.parent / "commands/local_codebook.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def _contract(config: LocalCodebookConfig, inputs: LocalCodebookInputs, gate: str, chunk_rows: int) -> dict[str, Any]:
    settings = config.gates[gate]
    return {
        "schema_version": SCHEMA_VERSION,
        "config_signature": config.signature(),
        "gate": gate,
        "poi_rows": len(inputs.selected_poi_rows),
        "p6_query_rows": len(inputs.p6_query_nodes),
        "d3_query_rows": len(inputs.d3_query_rows),
        "s3_edge_rows": len(inputs.s3_edges),
        "embedding_dim": inputs.embedding_dim,
        "codebook_size": 512,
        "parent_key": ["gid6", "s1", "s2"],
        "query_depth": "D3_ONLY",
        "algorithm": {
            "prqk": dict(config.base.prqk),
            "s3": dict(config.base.base.resolved_payload()["s3"]),
            "warmup": dict(config.base.base.resolved_payload()["warmup"]),
            "hard_graph": dict(config.hard_graph),
            "geo": dict(config.geo),
            "local_refinement": dict(config.local_refinement),
            "initialization": "P5_A0_S3_CODEBOOK_AND_ASSIGNMENT",
            "upstream_endpoint": "FROZEN_P6_S1_S2",
        },
        "source_hashes": dict(inputs.source_hashes),
        "gate_source": dict(settings["p6_manifest"]),
        "gate_sequence": config.authorization["gate_sequence"],
        "chunk_rows": chunk_rows,
        "code": _code_files(),
        "business_validation_query_read": False,
        "business_test_query_read": False,
    }


def build_local_codebook(
    config: LocalCodebookConfig,
    inputs: LocalCodebookInputs,
    *,
    gate: str,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Build one immutable P7 S3 gate and stop before P8."""
    started = time.perf_counter()
    torch = _torch()
    if device != "cuda" or not torch.cuda.is_available():
        raise LocalCodebookDataError("真实 P7 要求 CUDA；不得静默回退 CPU")
    directory = output_directory(config, gate)
    if (directory / "_SUCCESS").is_file():
        raise LocalCodebookDataError(f"P7 输出已完成且 overwrite=false：{directory}")
    if directory.exists():
        raise LocalCodebookDataError(f"P7 未完成输出目录已存在，禁止隐式覆盖：{directory}")
    directory.mkdir(parents=True)
    write_json_atomic(directory / "config_resolved.json", config.resolved_payload())
    write_json_atomic(directory / "geo_standardization.json", dict(inputs.geo_standardization))
    _atomic_npy(directory / "selected_poi_rows.npy", inputs.selected_poi_rows)
    _atomic_npy(directory / "selected_d3_query_rows.npy", inputs.d3_query_rows)
    _atomic_npy(directory / "poi_gid6.npy", inputs.gid6_codes)
    _atomic_npy(directory / "poi_parent_groups.npy", inputs.parent_groups)
    _atomic_npy(directory / "poi_geo_features.npy", inputs.geo_features)
    _atomic_npy(directory / "poi_singleton_parent_mask.npy", inputs.singleton_mask)
    pq.write_table(inputs.poi_metadata, directory / "poi_geo_metadata.parquet", compression="zstd")
    pq.write_table(inputs.s3_edges, directory / "query_poi_edges_s3.parquet", compression="zstd")
    pq.write_table(inputs.false_negative_mask, directory / "false_negative_mask_s3.parquet", compression="zstd")
    pq.write_table(inputs.hard_edges, directory / "hard_entity_edges.parquet", compression="zstd")

    target = torch.device(device)
    poi_values = _to_device_normalized_float32(inputs.poi_residual_after_s2, device=target, chunk_rows=chunk_rows)
    query_values = _to_device_normalized_float32(inputs.query_residual_after_s2, device=target, chunk_rows=chunk_rows)
    _event("p7_s3_started", gate=gate, poi_rows=len(poi_values), d3_query_rows=len(query_values), hard_edge_rows=len(inputs.hard_edges))
    u, v, geo_centroids, poi_assignment, query_assignment, fit = fit_s3(
        inputs, poi_values, query_values, torch.from_numpy(inputs.initial_poi_codebook_s3).to(target),
        settings=config.base.base.resolved_payload()["s3"], prqk=config.base.prqk,
        warmup=config.base.base.resolved_payload()["warmup"], local_refinement=config.local_refinement,
        chunk_rows=chunk_rows,
    )
    poi_np = poi_assignment.detach().cpu().numpy().astype(np.int32)
    d3_query_np = query_assignment.detach().cpu().numpy().astype(np.int32)
    full_query_s3 = np.full(len(inputs.p6_query_nodes), -1, dtype=np.int32)
    full_query_s3[inputs.d3_query_rows] = d3_query_np
    poi_sid = np.column_stack((np.asarray(inputs.p6_poi_assignments_s1_s2), poi_np)).astype(np.int32)
    query_sid = np.column_stack((np.asarray(inputs.p6_query_assignments_s1_s2), full_query_s3)).astype(np.int32)
    poi_after_s3, poi_residual_metrics = projection_residual(poi_values, u, poi_assignment)
    query_after_s3, query_residual_metrics = projection_residual(query_values, v, query_assignment)
    _atomic_npy(directory / "poi_codebook_s3.npy", u.detach().cpu().numpy().astype(np.float32))
    _atomic_npy(directory / "query_codebook_s3.npy", v.detach().cpu().numpy().astype(np.float32))
    _atomic_npy(directory / "geo_codebook_s3.npy", geo_centroids.detach().cpu().numpy().astype(np.float32))
    _atomic_npy(directory / "poi_assignments_s3.npy", poi_np)
    _atomic_npy(directory / "query_assignments_s3.npy", full_query_s3)
    _atomic_npy(directory / "poi_sid_s1_s2_s3.npy", poi_sid)
    _atomic_npy(directory / "query_sid_s1_s2_s3.npy", query_sid)
    _atomic_tensor_npy(directory / "poi_residual_after_s3.npy", poi_after_s3, dtype=np.float16, chunk_rows=chunk_rows)
    _atomic_tensor_npy(directory / "query_residual_after_s3_d3.npy", query_after_s3, dtype=np.float16, chunk_rows=chunk_rows)

    graph = _s3_graph(inputs)
    graph_result = _graph_metrics(graph, poi_np, d3_query_np)
    hard_result = _hard_metrics(inputs, poi_np)
    bucket_metrics = {"sid3": _prefix_metrics(poi_sid)}
    for precision in (4, 5, 6):
        gid = pack_gid(inputs.gid6_codes, precision=precision).astype(np.int64)
        bucket_metrics[f"gid{precision}_sid3"] = _prefix_metrics(np.column_stack((gid, poi_sid)))
    parent_counts = np.bincount(inputs.parent_groups)
    metrics = {
        "fit": fit,
        "poi_assignment": _distribution_metrics(poi_np, 512),
        "query_assignment": _distribution_metrics(d3_query_np, 512),
        "conditional_s3_accuracy": graph_result,
        "hard_entity_collision": hard_result,
        "semantic_distortion": {"poi": fit["final_poi_content_mean"], "query": fit["final_query_content_mean"]},
        "geo_distortion": fit["final_geo_distortion_mean"],
        "poi_residual": poi_residual_metrics,
        "query_residual": query_residual_metrics,
        "parent_groups": {
            "count": int(len(parent_counts)), "singleton_groups": int(np.count_nonzero(parent_counts == 1)),
            "singleton_poi_rows": int(inputs.singleton_mask.sum()), "max_size": int(parent_counts.max()),
        },
        "bucket_metrics": bucket_metrics,
        "category_path_proxy_subsets": _proxy_subset_metrics(inputs, poi_np),
        "structural_gate": {
            "s1_s2_frozen": True, "d3_only": True, "geo_used_only_in_s3": True,
            "category_classification_cost_used_in_s3": False,
            "false_negative_pairs_excluded": True, "all_poi_codes_active": int(np.unique(poi_np).size) == 512,
            "sample_role": "CODE_DATA_GPU_ARTIFACT_SMOKE_ONLY" if gate == "sample" else None,
            "formal_metric_basis": gate == "full", "ready_for_manual_review": True,
        },
    }
    write_json_atomic(directory / "metrics.json", metrics)
    artifact_names = [
        "config_resolved.json", "geo_standardization.json", "selected_poi_rows.npy", "selected_d3_query_rows.npy",
        "poi_gid6.npy", "poi_parent_groups.npy", "poi_geo_features.npy", "poi_singleton_parent_mask.npy",
        "poi_geo_metadata.parquet", "query_poi_edges_s3.parquet", "false_negative_mask_s3.parquet",
        "hard_entity_edges.parquet", "poi_codebook_s3.npy", "query_codebook_s3.npy", "geo_codebook_s3.npy",
        "poi_assignments_s3.npy", "query_assignments_s3.npy", "poi_sid_s1_s2_s3.npy", "query_sid_s1_s2_s3.npy",
        "poi_residual_after_s3.npy", "query_residual_after_s3_d3.npy", "metrics.json",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "phase": f"P7-CAT-{gate.upper()}",
        "role": "S3_D3_QUERY_LOCAL_GEO_HARD_ENTITY_GATE", "built_at": utc_now(),
        "contract": _contract(config, inputs, gate, chunk_rows), "metrics": metrics,
        "source_access": {
            "p6_frozen_s1_s2_read": True, "p5_s3_initialization_read": True,
            "p4_train_s3_graph_read": True, "active_poi_metadata_read": True,
            "business_validation_query_read": False, "business_test_query_read": False,
        },
        "artifacts": {name: _artifact(directory / name) for name in artifact_names},
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started, "device": device,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        },
        "next_status": next_status_for_gate(gate),
    }
    write_json_atomic(directory / "manifest.json", manifest)
    validate_local_codebook(config, gate=gate, require_success_marker=False)
    write_json_atomic(directory / "_SUCCESS", {"manifest_sha256": sha256_file(directory / "manifest.json")})
    _event("p7_s3_completed", gate=gate, hard_iterations=fit["hard_iterations"], conditional_s3_accuracy=graph_result["weighted_agreement"], distinct_sid=bucket_metrics["sid3"]["distinct"])
    del poi_values, query_values, poi_after_s3, query_after_s3, u, v, geo_centroids, poi_assignment, query_assignment
    gc.collect()
    torch.cuda.empty_cache()
    return validate_local_codebook(config, gate=gate)


def _expected_artifacts() -> Iterable[str]:
    yield from (
        "config_resolved.json", "geo_standardization.json", "selected_poi_rows.npy", "selected_d3_query_rows.npy",
        "poi_gid6.npy", "poi_parent_groups.npy", "poi_geo_features.npy", "poi_singleton_parent_mask.npy",
        "poi_geo_metadata.parquet", "query_poi_edges_s3.parquet", "false_negative_mask_s3.parquet",
        "hard_entity_edges.parquet", "poi_codebook_s3.npy", "query_codebook_s3.npy", "geo_codebook_s3.npy",
        "poi_assignments_s3.npy", "query_assignments_s3.npy", "poi_sid_s1_s2_s3.npy", "query_sid_s1_s2_s3.npy",
        "poi_residual_after_s3.npy", "query_residual_after_s3_d3.npy", "metrics.json",
    )


def validate_local_codebook(
    config: LocalCodebookConfig,
    *,
    gate: str,
    require_success_marker: bool = True,
) -> dict[str, Any]:
    """Independently validate P7 hashes, shapes, masks, and frozen prefixes."""
    directory = output_directory(config, gate)
    manifest_path = directory / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if require_success_marker:
        marker = json.loads((directory / "_SUCCESS").read_text(encoding="utf-8"))
        if marker.get("manifest_sha256") != manifest_sha:
            raise LocalCodebookDataError("P7 manifest 与 _SUCCESS 哈希不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != f"P7-CAT-{gate.upper()}"
        or manifest.get("role") != "S3_D3_QUERY_LOCAL_GEO_HARD_ENTITY_GATE"
        or manifest.get("contract", {}).get("config_signature") != config.signature()
        or manifest.get("next_status") != next_status_for_gate(gate)
    ):
        raise LocalCodebookDataError("P7 manifest schema/状态/配置/停止点不匹配")
    if set(manifest.get("artifacts", {})) != set(_expected_artifacts()):
        raise LocalCodebookDataError("P7 artifact 清单不完整")
    for name, entry in manifest["artifacts"].items():
        path = directory / name
        if entry.get("file") != name or sha256_file(path) != entry.get("sha256"):
            raise LocalCodebookDataError(f"P7 artifact 缺失或哈希错误：{name}")
    settings = config.gates[gate]
    poi_rows = int(settings["poi_rows"])
    query_rows = int(settings["p6_query_rows"])
    d3_rows = int(settings["d3_query_rows"])
    dimension = int(manifest["contract"]["embedding_dim"])
    expected_arrays = {
        "selected_poi_rows.npy": ((poi_rows,), np.int64), "selected_d3_query_rows.npy": ((d3_rows,), np.int64),
        "poi_gid6.npy": ((poi_rows, 6), np.uint8), "poi_parent_groups.npy": ((poi_rows,), np.int64),
        "poi_geo_features.npy": ((poi_rows, GEO_FEATURE_DIM), np.float32), "poi_singleton_parent_mask.npy": ((poi_rows,), np.bool_),
        "poi_codebook_s3.npy": ((512, dimension), np.float32), "query_codebook_s3.npy": ((512, dimension), np.float32),
        "geo_codebook_s3.npy": ((512, GEO_FEATURE_DIM), np.float32), "poi_assignments_s3.npy": ((poi_rows,), np.int32),
        "query_assignments_s3.npy": ((query_rows,), np.int32), "poi_sid_s1_s2_s3.npy": ((poi_rows, 3), np.int32),
        "query_sid_s1_s2_s3.npy": ((query_rows, 3), np.int32), "poi_residual_after_s3.npy": ((poi_rows, dimension), np.float16),
        "query_residual_after_s3_d3.npy": ((d3_rows, dimension), np.float16),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in expected_arrays.items():
        values = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        if values.shape != shape or values.dtype != np.dtype(dtype) or not np.isfinite(values).all():
            raise LocalCodebookDataError(f"P7 {name} shape/dtype/finite 错误")
        arrays[name] = values
    assignments = arrays["poi_assignments_s3.npy"]
    query_assignments = arrays["query_assignments_s3.npy"]
    d3_query_rows = arrays["selected_d3_query_rows.npy"]
    if np.any(assignments < 0) or np.any(assignments >= 512):
        raise LocalCodebookDataError("P7 POI S3 assignment 超界")
    d3_mask = np.zeros(query_rows, dtype=bool)
    d3_mask[d3_query_rows] = True
    if np.any(query_assignments[d3_mask] < 0) or np.any(query_assignments[d3_mask] >= 512) or np.any(query_assignments[~d3_mask] != -1):
        raise LocalCodebookDataError("P7 Query S3 D3 mask/assignment 错误")
    p6_manifest_path = _manifest_for_gate(config, gate)
    p6_dir = p6_manifest_path.parent
    p6_prefix = np.load(p6_dir / "poi_assignments_s1_s2.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(arrays["poi_sid_s1_s2_s3.npy"][:, :2], p6_prefix) or not np.array_equal(arrays["poi_sid_s1_s2_s3.npy"][:, 2], assignments):
        raise LocalCodebookDataError("P7 POI SID 未逐值冻结 P6 S1/S2")
    metadata_file = pq.ParquetFile(directory / "poi_geo_metadata.parquet")
    hard_file = pq.ParquetFile(directory / "hard_entity_edges.parquet")
    edges_file = pq.ParquetFile(directory / "query_poi_edges_s3.parquet")
    if metadata_file.metadata.num_rows != poi_rows or not metadata_file.schema_arrow.equals(P7_POI_METADATA_SCHEMA):
        raise LocalCodebookDataError("P7 POI Geo metadata schema/行数错误")
    if not hard_file.schema_arrow.equals(HARD_EDGE_SCHEMA) or edges_file.metadata.num_rows != d3_rows:
        raise LocalCodebookDataError("P7 hard/S3 edge schema 或行数错误")
    hard = hard_file.read()
    if len(hard):
        if pc.any(pc.equal(hard["poi_row_index"], hard["neighbor_poi_row_index"])).as_py():
            raise LocalCodebookDataError("P7 hard edge 含 self-loop")
        if pc.min(hard["composite_similarity"]).as_py() + 1.0e-6 < 0.60:
            raise LocalCodebookDataError("P7 hard edge 低于冻结阈值")
        counts = np.unique(hard["poi_row_index"].to_numpy(zero_copy_only=False), return_counts=True)[1]
        if int(counts.max()) > 20:
            raise LocalCodebookDataError("P7 hard edge 超过每 POI Top-20")
    singleton = arrays["poi_singleton_parent_mask.npy"]
    geo = arrays["poi_geo_features.npy"]
    if np.any(geo[singleton] != 0.0):
        raise LocalCodebookDataError("P7 singleton Geo feature 必须为 0")
    nonzero_norms = np.linalg.norm(geo[~singleton], axis=1)
    if len(nonzero_norms) and np.any((nonzero_norms > 1.0e-6) & (np.abs(nonzero_norms - 1.0) > 2.0e-5)):
        raise LocalCodebookDataError("P7 非零 Geo feature 必须 L2 normalize")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if manifest.get("metrics") != metrics:
        raise LocalCodebookDataError("P7 manifest 与 metrics.json 不一致")
    return {
        "status": "p7_validated", "phase": manifest["phase"], "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha, "poi_rows": poi_rows, "d3_query_rows": d3_rows,
        "hard_edge_rows": hard_file.metadata.num_rows, "metrics": metrics, "next_status": manifest["next_status"],
    }


def _manifest_for_gate(config: LocalCodebookConfig, gate: str) -> Path:
    return (config.project_root / str(config.gates[gate]["p6_manifest"]["path"])).resolve()


def load_and_build_local_codebook(config: LocalCodebookConfig, *, gate: str, device: str = "cuda", chunk_rows: int = DEFAULT_CHUNK_ROWS) -> dict[str, Any]:
    """Load the confirmed P7 inputs and build one S3 gate."""
    return build_local_codebook(config, load_local_codebook_inputs(config, gate=gate), gate=gate, device=device, chunk_rows=chunk_rows)

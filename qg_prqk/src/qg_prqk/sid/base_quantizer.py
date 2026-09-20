"""Resumable POI-only PRQ-KMeans base-codebook training."""

from __future__ import annotations

import copy
import gc
import json
import os
import resource
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.pipeline_config import DownstreamConfig
from qg_prqk.sid.base_data import (
    PREVIOUS_GATE,
    BaseCodebookDataError,
    BaseCodebookInputs,
    BaseCodebookAlgorithmConfig,
    compute_global_mean,
    gate_next_status,
    load_base_codebook_algorithm,
    output_directory,
    selected_rows_sha256,
)


SCHEMA_VERSION = "qg-prqk-p5-poi-prqk-a0-v1"
DEFAULT_CHUNK_ROWS = 8192
EPSILON = 1.0e-12


def _event(stage: str, **values: Any) -> None:
    print(
        json.dumps({"time": utc_now(), "stage": stage, **values}, ensure_ascii=False),
        flush=True,
    )


def _torch():
    try:
        import torch
    except ImportError as error:
        raise BaseCodebookDataError("P5-CAT 需要 PyTorch") from error
    return torch


def _normalize_torch(values):
    torch = _torch()
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    return torch.where(norms > EPSILON, values / norms.clamp_min(EPSILON), values)


def _code_files() -> dict[str, str]:
    directory = Path(__file__).parent
    files = {
        "sid/base_config.py": directory / "base_config.py",
        "sid/base_data.py": directory / "base_data.py",
        "sid/base_quantizer.py": directory / "base_quantizer.py",
        "commands/base_codebook.py": directory.parent / "commands/base_codebook.py",
    }
    result = {}
    for name, path in files.items():
        if not path.is_file():
            raise BaseCodebookDataError(f"基础码本源码不完整：{path}")
        result[name] = sha256_file(path)
    return result


def _contract(
    config: DownstreamConfig,
    settings: BaseCodebookAlgorithmConfig,
    inputs: BaseCodebookInputs,
    gate: str,
    previous_gate: Mapping[str, Any] | None,
    chunk_rows: int,
) -> dict[str, Any]:
    contract = {
        "schema_version": SCHEMA_VERSION,
        "config_signature": config.signature(),
        "gate": gate,
        "poi_rows": len(inputs.selected_rows),
        "active_poi_rows": inputs.total_rows,
        "embedding_dim": inputs.embedding_dim,
        "codebook_sizes": list(config.codebook_sizes),
        "selection": {
            "algorithm": (
                "identity_active_poi_row_order"
                if gate == "full"
                else "sorted_prefix_of_numpy_pcg64_seeded_permutation"
            ),
            "seed": settings.seed,
            "selected_rows_sha256": selected_rows_sha256(inputs.selected_rows),
            "full_uses_identity_order": gate == "full",
        },
        "algorithm": asdict(settings),
        "global_component": "mean_of_l2_normalized_full_active_poi_catalog",
        "residual_artifact_dtype": "float16",
        "fit_accumulation_dtype": "float32",
        "assignment_dtype": "int32",
        "chunk_rows": chunk_rows,
        "previous_gate": dict(previous_gate) if previous_gate else None,
        "source_hashes": dict(inputs.source_hashes),
        "code": _code_files(),
    }
    if getattr(config, "protocol_id", None) is not None:
        contract["p5_protocol"] = {
            "protocol_id": config.protocol_id,
            "config_path": str(config.source_path),
            "config_sha256": config.source_sha256,
            "output_subdir": config.output_subdir,
            "selection_evidence": copy.deepcopy(dict(config.selection_evidence)),
        }
    return contract


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise BaseCodebookDataError(f"输出已存在且禁止覆盖：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_tensor_npy(path: Path, values, *, dtype: np.dtype, chunk_rows: int) -> None:
    if path.exists():
        raise BaseCodebookDataError(f"输出已存在且禁止覆盖：{path}")
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


def _preserve_uncommitted(path: Path) -> None:
    if path.exists():
        path.rename(path.with_name(f"{path.name}.unconfirmed.{uuid.uuid4().hex}"))


def _artifact(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        entry.update(shape=list(values.shape), dtype=str(values.dtype))
    return entry


def _load_selected_transformed(
    inputs: BaseCodebookInputs,
    global_mean: np.ndarray,
    *,
    device: str,
    chunk_rows: int,
):
    torch = _torch()
    if device == "cuda" and not torch.cuda.is_available():
        raise BaseCodebookDataError("正式 P5 要求 CUDA；不得静默回退 CPU")
    target = torch.device(device)
    values = torch.empty(
        (len(inputs.selected_rows), inputs.embedding_dim),
        dtype=torch.float32,
        device=target,
    )
    mean = torch.from_numpy(np.asarray(global_mean, dtype=np.float32)).to(target)
    mean_norm_sq = torch.dot(mean, mean).clamp_min(EPSILON)
    for start in range(0, len(inputs.selected_rows), chunk_rows):
        stop = min(start + chunk_rows, len(inputs.selected_rows))
        block = np.asarray(
            inputs.embeddings[inputs.selected_rows[start:stop]], dtype=np.float32
        )
        tensor = torch.from_numpy(block).to(target)
        tensor = _normalize_torch(tensor)
        coefficients = (tensor @ mean) / mean_norm_sq
        values[start:stop] = _normalize_torch(tensor - coefficients[:, None] * mean)
    return values


def _hard_assign(values, centroids, *, chunk_rows: int):
    torch = _torch()
    assignments = torch.empty(len(values), dtype=torch.int64, device=values.device)
    scores = torch.empty(len(values), dtype=torch.float32, device=values.device)
    normalized_centroids = _normalize_torch(centroids)
    for start in range(0, len(values), chunk_rows):
        stop = min(start + chunk_rows, len(values))
        similarities = values[start:stop] @ normalized_centroids.T
        block_scores, block_assignments = torch.max(similarities, dim=1)
        assignments[start:stop] = block_assignments
        scores[start:stop] = block_scores.clamp(-1.0, 1.0)
    return assignments, scores


def _kmeans_plus_plus(values, codebook_size: int, *, seed: int, chunk_rows: int):
    torch = _torch()
    norms = torch.linalg.vector_norm(values, dim=1)
    valid = torch.nonzero(norms > EPSILON, as_tuple=False).flatten()
    if len(valid) < codebook_size:
        raise BaseCodebookDataError("非零 residual 数少于码本容量，无法执行 kmeans++")
    generator = torch.Generator(device=values.device)
    generator.manual_seed(seed)
    first_position = int(
        torch.randint(
            len(valid), (1,), generator=generator, device=values.device
        ).item()
    )
    first = int(valid[first_position].item())
    selected = torch.zeros(len(values), dtype=torch.bool, device=values.device)
    selected[first] = True
    centroids = torch.empty(
        (codebook_size, values.shape[1]), dtype=torch.float32, device=values.device
    )
    centroids[0] = values[first]
    min_distance = torch.full(
        (len(values),), float("inf"), dtype=torch.float32, device=values.device
    )
    for code in range(1, codebook_size):
        center = centroids[code - 1]
        for start in range(0, len(values), chunk_rows):
            stop = min(start + chunk_rows, len(values))
            similarity = (values[start:stop] @ center).clamp(-1.0, 1.0)
            min_distance[start:stop] = torch.minimum(
                min_distance[start:stop], (1.0 - similarity).clamp_min(0.0)
            )
        weights = min_distance.clone()
        weights[selected] = 0.0
        weights[norms <= EPSILON] = 0.0
        if not bool(torch.isfinite(weights).all()) or float(weights.sum().item()) <= 0:
            candidates = torch.nonzero((~selected) & (norms > EPSILON)).flatten()
            chosen = int(candidates[0].item())
        else:
            chosen = int(torch.multinomial(weights, 1, generator=generator).item())
        selected[chosen] = True
        centroids[code] = values[chosen]
    return _normalize_torch(centroids)


def _farthest_rows(scores, count: int):
    torch = _torch()
    # Stable ascending sort makes equal-score fallback prefer the smaller POI row.
    return torch.argsort(scores, stable=True)[:count]


def _hard_centroid_update(values, assignments, scores, codebook_size: int):
    torch = _torch()
    sums = torch.zeros(
        (codebook_size, values.shape[1]), dtype=torch.float32, device=values.device
    )
    sums.index_add_(0, assignments, values)
    assigned_counts = torch.bincount(assignments, minlength=codebook_size)
    empty = torch.nonzero(assigned_counts == 0, as_tuple=False).flatten()
    update_counts = assigned_counts.clone()
    if len(empty):
        rows = _farthest_rows(scores, len(empty))
        sums[empty] = values[rows]
        update_counts[empty] = 1
    centroids = sums / update_counts.to(torch.float32).clamp_min(1.0)[:, None]
    centroids = _normalize_torch(centroids)
    if bool((torch.linalg.vector_norm(centroids, dim=1) <= EPSILON).any()):
        raise BaseCodebookDataError("hard centroid update 产生零质心")
    return centroids, assigned_counts, int(len(empty))


def _soft_centroid_update(
    values,
    centroids,
    *,
    topk: int,
    beta: float,
    chunk_rows: int,
):
    torch = _torch()
    codebook_size = len(centroids)
    sums = torch.zeros_like(centroids, dtype=torch.float32)
    weights_by_code = torch.zeros(
        codebook_size, dtype=torch.float32, device=values.device
    )
    centroids = _normalize_torch(centroids)
    for start in range(0, len(values), chunk_rows):
        stop = min(start + chunk_rows, len(values))
        block = values[start:stop]
        nonzero = torch.linalg.vector_norm(block, dim=1) > EPSILON
        if not bool(nonzero.any()):
            continue
        active = block[nonzero]
        similarities = active @ centroids.T
        top_scores, top_indices = torch.topk(
            similarities, k=topk, dim=1, largest=True, sorted=True
        )
        weights = torch.softmax(beta * top_scores, dim=1)
        for neighbor in range(topk):
            indices = top_indices[:, neighbor]
            neighbor_weights = weights[:, neighbor]
            sums.index_add_(0, indices, active * neighbor_weights[:, None])
            weights_by_code.index_add_(0, indices, neighbor_weights)
    empty = torch.nonzero(weights_by_code <= EPSILON, as_tuple=False).flatten()
    if len(empty):
        _, hard_scores = _hard_assign(values, centroids, chunk_rows=chunk_rows)
        rows = _farthest_rows(hard_scores, len(empty))
        sums[empty] = values[rows]
        weights_by_code[empty] = 1.0
    refined = sums / weights_by_code.clamp_min(EPSILON)[:, None]
    refined = _normalize_torch(refined)
    if bool((torch.linalg.vector_norm(refined, dim=1) <= EPSILON).any()):
        raise BaseCodebookDataError("soft centroid refinement 产生零质心")
    return refined


def fit_prqk_level(
    values,
    codebook_size: int,
    settings: BaseCodebookAlgorithmConfig,
    *,
    level: int,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
):
    """Fit one spherical codebook, then apply frozen Top-k soft refinement."""
    torch = _torch()
    if values.ndim != 2 or len(values) < codebook_size:
        raise BaseCodebookDataError("PRQK 输入行数必须不小于码本容量")
    values = _normalize_torch(values.float())
    centroids = _kmeans_plus_plus(
        values,
        codebook_size,
        seed=settings.seed + 1009 * level,
        chunk_rows=chunk_rows,
    )
    trace: list[dict[str, Any]] = []
    previous_objective: float | None = None
    previous_assignments = None
    stable_rounds = 0
    increase_streak = 0
    converged = False
    for iteration in range(1, settings.max_iter + 1):
        assignments, scores = _hard_assign(values, centroids, chunk_rows=chunk_rows)
        objective = float((1.0 - scores).mean().item())
        change = (
            1.0
            if previous_assignments is None
            else float((assignments != previous_assignments).float().mean().item())
        )
        relative_improvement = (
            None
            if previous_objective is None
            else (previous_objective - objective)
            / max(abs(previous_objective), EPSILON)
        )
        new_centroids, counts, empty_count = _hard_centroid_update(
            values, assignments, scores, codebook_size
        )
        trace.append(
            {
                "stage": "hard_spherical_kmeans",
                "iteration": iteration,
                "objective": objective,
                "relative_objective_improvement": relative_improvement,
                "assignment_change": change,
                "active_codes": int(torch.count_nonzero(counts).item()),
                "empty_codes_reinitialized": empty_count,
            }
        )
        if relative_improvement is not None:
            if relative_improvement < -settings.objective_rel_tol:
                increase_streak += 1
            else:
                increase_streak = 0
            if increase_streak >= 3:
                raise BaseCodebookDataError("hard spherical objective 连续 3 轮明显升高")
            if (
                iteration >= settings.min_iter
                and 0.0 <= relative_improvement < settings.objective_rel_tol
                and change < settings.assignment_change_tol
            ):
                stable_rounds += 1
            else:
                stable_rounds = 0
        centroids = new_centroids
        previous_objective = objective
        previous_assignments = assignments
        if iteration >= settings.min_iter and stable_rounds >= settings.patience:
            converged = True
            break
    if settings.topk_refinement.enabled:
        for iteration in range(1, settings.topk_refinement.max_iter + 1):
            centroids = _soft_centroid_update(
                values,
                centroids,
                topk=settings.topk_refinement.topk,
                beta=settings.topk_refinement.beta,
                chunk_rows=chunk_rows,
            )
            assignments, scores = _hard_assign(values, centroids, chunk_rows=chunk_rows)
            objective = float((1.0 - scores).mean().item())
            change = float((assignments != previous_assignments).float().mean().item())
            trace.append(
                {
                    "stage": "topk_soft_centroid_refinement",
                    "iteration": iteration,
                    "objective": objective,
                    "assignment_change": change,
                    "topk": settings.topk_refinement.topk,
                    "beta": settings.topk_refinement.beta,
                }
            )
            previous_assignments = assignments
    assignments, scores = _hard_assign(values, centroids, chunk_rows=chunk_rows)
    return (
        centroids,
        assignments,
        scores,
        {
            "hard_converged": converged,
            "hard_iterations": sum(
                row["stage"] == "hard_spherical_kmeans" for row in trace
            ),
            "soft_refinement_iterations": sum(
                row["stage"] == "topk_soft_centroid_refinement" for row in trace
            ),
            "final_objective": float((1.0 - scores).mean().item()),
            "trace": trace,
        },
    )


def projection_residual(values, centroids, assignments):
    """Remove each row's selected-centroid projection and L2-normalize."""
    torch = _torch()
    selected = centroids[assignments]
    denominators = torch.sum(selected * selected, dim=1).clamp_min(EPSILON)
    coefficients = torch.sum(values * selected, dim=1) / denominators
    raw = values - coefficients[:, None] * selected
    raw_norms = torch.linalg.vector_norm(raw, dim=1)
    residual = _normalize_torch(raw)
    nonzero = raw_norms > EPSILON
    orthogonality = (
        float(
            torch.abs(torch.sum(residual[nonzero] * selected[nonzero], dim=1))
            .max()
            .item()
        )
        if bool(nonzero.any())
        else 0.0
    )
    return residual, {
        "retained_energy_mean": float(torch.mean(raw_norms * raw_norms).item()),
        "retained_energy_p50": float(torch.quantile(raw_norms * raw_norms, 0.5).item()),
        "zero_residual_rows": int(torch.count_nonzero(~nonzero).item()),
        "orthogonality_abs_dot_max": orthogonality,
    }


def _gini(counts: np.ndarray) -> float:
    values = np.sort(np.asarray(counts, dtype=np.float64))
    if not len(values) or values.sum() == 0:
        return 0.0
    indices = np.arange(1, len(values) + 1, dtype=np.float64)
    return float(
        (2 * np.sum(indices * values) / (len(values) * values.sum()))
        - (len(values) + 1) / len(values)
    )


def _distribution_metrics(
    assignments: np.ndarray, codebook_size: int
) -> dict[str, Any]:
    counts = np.bincount(assignments.astype(np.int64), minlength=codebook_size)
    probabilities = counts / max(int(counts.sum()), 1)
    denominator = float(np.sum(probabilities * probabilities))
    return {
        "active_codes": int(np.count_nonzero(counts)),
        "utilization": float(np.count_nonzero(counts) / codebook_size),
        "kish_ess": float(1.0 / denominator) if denominator else 0.0,
        "gini": _gini(counts),
        "cluster_size": {
            "min": int(counts.min()),
            "p50": float(np.quantile(counts, 0.50)),
            "p90": float(np.quantile(counts, 0.90)),
            "p99": float(np.quantile(counts, 0.99)),
            "max": int(counts.max()),
        },
    }


def sid_metrics(assignments: np.ndarray, codebook_size: int) -> dict[str, Any]:
    if assignments.ndim != 2 or assignments.shape[1] != 3:
        raise BaseCodebookDataError("POI assignment 必须是 [N,3]")
    _, counts = np.unique(assignments, axis=0, return_counts=True)
    collision_mask = counts > 1
    return {
        "levels": [
            _distribution_metrics(assignments[:, level], codebook_size)
            for level in range(3)
        ],
        "distinct_sid": int(len(counts)),
        "distinct_sid_ratio": float(len(counts) / len(assignments)),
        "collision_poi_rows": int(counts[collision_mask].sum()),
        "collision_excess": int(np.sum(counts - 1)),
        "bucket_size": {
            "mean": float(np.mean(counts)),
            "p50": float(np.quantile(counts, 0.50)),
            "p90": float(np.quantile(counts, 0.90)),
            "p95": float(np.quantile(counts, 0.95)),
            "p99": float(np.quantile(counts, 0.99)),
            "max": int(counts.max()),
        },
    }


def _check_previous_gate(
    config: DownstreamConfig,
    gate: str,
    path: Path | None,
    expected_sha256: str | None,
    *,
    direct_full_from_sample_user_authorized: bool = False,
) -> dict[str, Any] | None:
    if direct_full_from_sample_user_authorized and gate != "full":
        raise BaseCodebookDataError("direct-full 用户授权只允许用于 full gate")
    expected_gate = (
        "sample"
        if direct_full_from_sample_user_authorized
        else PREVIOUS_GATE.get(gate)
    )
    if expected_gate is None:
        if path is not None or expected_sha256 is not None:
            raise BaseCodebookDataError("sample gate 不接受 previous manifest")
        return None
    if path is None or expected_sha256 is None:
        raise BaseCodebookDataError(f"{gate} 必须显式绑定已完成的 {expected_gate} manifest")
    path = path.resolve()
    if sha256_file(path) != expected_sha256:
        raise BaseCodebookDataError("previous gate manifest SHA256 不匹配")
    marker_path = path.parent / "_SUCCESS"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("manifest_sha256") != expected_sha256:
        raise BaseCodebookDataError("previous gate 成功标记未锁定给定 manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != f"P5-CAT-{expected_gate.upper()}"
        or manifest.get("contract", {}).get("config_signature") != config.signature()
        or manifest.get("next_status") != gate_next_status(config, expected_gate)
    ):
        raise BaseCodebookDataError("previous P5 gate 状态或配置不匹配")
    result: dict[str, Any] = {
        "gate": expected_gate,
        "manifest_path": str(path),
        "manifest_sha256": expected_sha256,
    }
    if direct_full_from_sample_user_authorized:
        result["gate_sequence_override"] = {
            "authorization": "USER_CONFIRMED_20260909_DIRECT_P5_FULL_FROM_SAMPLE",
            "scope": "P5_A0_ONLY",
            "skipped_gates": ["medium100k", "medium500k"],
        }
    return result


def _level_files(level: int) -> tuple[str, str, str, str]:
    name = f"s{level}"
    return (
        f"poi_codebook_{name}.npy",
        f"poi_assignments_{name}.npy",
        f"poi_residual_after_{name}.npy",
        f"metrics_{name}.json",
    )


def _load_progress(directory: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    path = directory / "progress.json"
    if not path.is_file():
        return {"contract": dict(contract), "levels": []}
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("contract") != contract or not isinstance(
        progress.get("levels"), list
    ):
        raise BaseCodebookDataError("P5 progress 与当前配置/输入/源码不一致")
    for expected_level, entry in enumerate(progress["levels"], start=1):
        if entry.get("level") != expected_level:
            raise BaseCodebookDataError("P5 progress 层级不连续")
        for artifact in entry.get("artifacts", {}).values():
            path = directory / artifact["file"]
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise BaseCodebookDataError("P5 已提交层级 artifact 缺失或损坏")
    return progress


def _write_progress(directory: Path, progress: Mapping[str, Any]) -> None:
    write_json_atomic(directory / "progress.json", progress, overwrite=True)


def _validate_base_artifacts(
    directory: Path,
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    expected_mean: np.ndarray,
) -> None:
    selected = np.load(directory / "selected_poi_rows.npy", allow_pickle=False)
    mean = np.load(directory / "global_mean.npy", allow_pickle=False)
    if (
        selected.dtype != np.int64
        or not np.array_equal(selected, inputs.selected_rows)
        or selected_rows_sha256(selected)
        != inputs.source_hashes["selected_rows_sha256"]
        or mean.dtype != np.float32
        or not np.array_equal(mean, expected_mean)
        or json.loads((directory / "config_resolved.json").read_text(encoding="utf-8"))
        != config.resolved_payload()
    ):
        raise BaseCodebookDataError("P5 基础 artifact 与冻结输入不一致")


def build_base_codebook(
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    *,
    gate: str,
    previous_manifest: Path | None = None,
    previous_manifest_sha256: str | None = None,
    direct_full_from_sample_user_authorized: bool = False,
    resume: bool = False,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Build one immutable P5 engineering gate and stop before P6."""
    started = time.perf_counter()
    settings = load_base_codebook_algorithm(config)
    previous = _check_previous_gate(
        config,
        gate,
        previous_manifest,
        previous_manifest_sha256,
        direct_full_from_sample_user_authorized=(
            direct_full_from_sample_user_authorized
        ),
    )
    directory = output_directory(config, gate, len(inputs.selected_rows))
    if (directory / "_SUCCESS").is_file():
        if not resume:
            raise BaseCodebookDataError(f"P5 输出已完成且 overwrite=false：{directory}")
        return validate_base_codebook(
            config, inputs, gate=gate, device=device, chunk_rows=chunk_rows
        )
    if directory.exists() and not resume:
        raise BaseCodebookDataError(f"P5 输出目录已存在；必须显式 --resume：{directory}")
    directory.mkdir(parents=True, exist_ok=True)
    _event("p5_global_mean_started", active_poi_rows=inputs.total_rows)
    global_mean, global_metrics = compute_global_mean(
        inputs.embeddings, chunk_rows=chunk_rows
    )
    contract = _contract(config, settings, inputs, gate, previous, chunk_rows)
    progress = _load_progress(directory, contract)
    if progress["levels"]:
        _validate_base_artifacts(directory, config, inputs, global_mean)
    else:
        for path in (
            directory / "selected_poi_rows.npy",
            directory / "global_mean.npy",
            directory / "config_resolved.json",
        ):
            _preserve_uncommitted(path)
        _atomic_npy(directory / "selected_poi_rows.npy", inputs.selected_rows)
        _atomic_npy(directory / "global_mean.npy", global_mean)
        write_json_atomic(directory / "config_resolved.json", config.resolved_payload())
        _write_progress(directory, progress)
    _event("p5_transform_started", gate=gate, poi_rows=len(inputs.selected_rows))
    values = _load_selected_transformed(
        inputs, global_mean, device=device, chunk_rows=chunk_rows
    )
    if not progress["levels"]:
        path = directory / "poi_residual_s0.npy"
        _preserve_uncommitted(path)
        _atomic_tensor_npy(path, values, dtype=np.float16, chunk_rows=chunk_rows)
    level_metrics: list[dict[str, Any]] = []
    completed = len(progress["levels"])
    for level in range(1, completed + 1):
        codebook_file, assignment_file, _, metrics_file = _level_files(level)
        centroids = (
            _torch()
            .from_numpy(np.load(directory / codebook_file, allow_pickle=False))
            .to(values.device)
        )
        assignments = (
            _torch()
            .from_numpy(
                np.load(directory / assignment_file, allow_pickle=False).astype(
                    np.int64
                )
            )
            .to(values.device)
        )
        level_metrics.append(
            json.loads((directory / metrics_file).read_text(encoding="utf-8"))
        )
        values, _ = projection_residual(values, centroids, assignments)
    for level in range(completed + 1, 4):
        _event("p5_level_started", gate=gate, level=level)
        centroids, assignments, scores, fit_metrics = fit_prqk_level(
            values,
            config.codebook_sizes[level - 1],
            settings,
            level=level,
            chunk_rows=chunk_rows,
        )
        values, residual_metrics = projection_residual(values, centroids, assignments)
        assignments_np = assignments.detach().cpu().numpy().astype(np.int32)
        metrics = {
            "level": level,
            "fit": fit_metrics,
            "assignment": _distribution_metrics(
                assignments_np, config.codebook_sizes[level - 1]
            ),
            "hard_cosine_similarity_mean": float(scores.mean().item()),
            "residual": residual_metrics,
        }
        codebook_file, assignment_file, residual_file, metrics_file = _level_files(
            level
        )
        for path in (
            directory / codebook_file,
            directory / assignment_file,
            directory / residual_file,
            directory / metrics_file,
        ):
            _preserve_uncommitted(path)
        _atomic_npy(
            directory / codebook_file,
            centroids.detach().float().cpu().numpy().astype(np.float32),
        )
        _atomic_npy(directory / assignment_file, assignments_np)
        _atomic_tensor_npy(
            directory / residual_file,
            values,
            dtype=np.float16,
            chunk_rows=chunk_rows,
        )
        write_json_atomic(directory / metrics_file, metrics)
        level_entry = {
            "level": level,
            "artifacts": {
                name: _artifact(directory / name)
                for name in (
                    codebook_file,
                    assignment_file,
                    residual_file,
                    metrics_file,
                )
            },
        }
        progress["levels"].append(level_entry)
        _write_progress(directory, progress)
        level_metrics.append(metrics)
        _event(
            "p5_level_completed",
            gate=gate,
            level=level,
            objective=fit_metrics["final_objective"],
            active_codes=metrics["assignment"]["active_codes"],
        )
        del centroids, assignments, scores, assignments_np
        gc.collect()
    assignments = np.column_stack(
        [
            np.load(
                directory / _level_files(level)[1], mmap_mode="r", allow_pickle=False
            )
            for level in range(1, 4)
        ]
    ).astype(np.int32)
    codebooks = np.stack(
        [
            np.load(directory / _level_files(level)[0], allow_pickle=False)
            for level in range(1, 4)
        ]
    ).astype(np.float32)
    for path in (
        directory / "poi_assignments_s1_s2_s3.npy",
        directory / "poi_codebooks_s1_s2_s3.npy",
        directory / "metrics.json",
    ):
        _preserve_uncommitted(path)
    _atomic_npy(directory / "poi_assignments_s1_s2_s3.npy", assignments)
    _atomic_npy(directory / "poi_codebooks_s1_s2_s3.npy", codebooks)
    metrics = {
        "global_component": global_metrics,
        "level_metrics": level_metrics,
        "sid": sid_metrics(assignments, config.codebook_sizes[0]),
        "structural_gate": {
            "all_three_levels_completed": True,
            "assignments_in_range": True,
            "finite_codebooks_and_residuals": True,
            "query_category_geo_used": False,
        },
    }
    write_json_atomic(directory / "metrics.json", metrics)
    artifact_names: list[str] = [
        "selected_poi_rows.npy",
        "global_mean.npy",
        "config_resolved.json",
        "poi_residual_s0.npy",
        "poi_assignments_s1_s2_s3.npy",
        "poi_codebooks_s1_s2_s3.npy",
        "metrics.json",
        "progress.json",
    ]
    for level in range(1, 4):
        artifact_names.extend(_level_files(level))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "phase": f"P5-CAT-{gate.upper()}",
        "built_at": utc_now(),
        "contract": contract,
        "metrics": metrics,
        "source_access": {
            "active_poi_embedding_values_read": True,
            "raw_train_read": False,
            "business_validation_read": False,
            "business_test_read": False,
            "query_graph_read": False,
            "category_values_read": False,
            "geo_values_read": False,
        },
        "role": "A0_POI_ONLY_PRQK_INITIALIZATION",
        "artifacts": {name: _artifact(directory / name) for name in artifact_names},
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "resumed_levels": completed,
        },
        "next_status": gate_next_status(config, gate),
    }
    _preserve_uncommitted(directory / "manifest.json")
    write_json_atomic(directory / "manifest.json", manifest)
    write_json_atomic(
        directory / "_SUCCESS",
        {"manifest_sha256": sha256_file(directory / "manifest.json")},
    )
    del values
    gc.collect()
    return validate_base_codebook(
        config, inputs, gate=gate, device=device, chunk_rows=chunk_rows
    )


def _iter_expected_artifacts() -> Iterable[str]:
    yield from (
        "selected_poi_rows.npy",
        "global_mean.npy",
        "config_resolved.json",
        "poi_residual_s0.npy",
        "poi_assignments_s1_s2_s3.npy",
        "poi_codebooks_s1_s2_s3.npy",
        "metrics.json",
        "progress.json",
    )
    for level in range(1, 4):
        yield from _level_files(level)


def validate_base_codebook(
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    *,
    gate: str,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Independently verify sources, hashes, hard assignments, and residual recursion."""
    settings = load_base_codebook_algorithm(config)
    directory = output_directory(config, gate, len(inputs.selected_rows))
    manifest_path = directory / "manifest.json"
    marker = json.loads((directory / "_SUCCESS").read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != manifest_sha:
        raise BaseCodebookDataError("P5 manifest 与成功标记 SHA256 不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = manifest.get("contract", {}).get("previous_gate")
    if gate in PREVIOUS_GATE:
        if not isinstance(previous, Mapping):
            raise BaseCodebookDataError("P5 非 sample manifest 缺少 previous gate")
        override = previous.get("gate_sequence_override")
        direct_full_from_sample_user_authorized = override == {
            "authorization": "USER_CONFIRMED_20260909_DIRECT_P5_FULL_FROM_SAMPLE",
            "scope": "P5_A0_ONLY",
            "skipped_gates": ["medium100k", "medium500k"],
        }
        previous = _check_previous_gate(
            config,
            gate,
            Path(previous["manifest_path"]),
            previous["manifest_sha256"],
            direct_full_from_sample_user_authorized=(
                direct_full_from_sample_user_authorized
            ),
        )
    elif previous is not None:
        raise BaseCodebookDataError("P5 sample manifest 不得声明 previous gate")
    contract = _contract(config, settings, inputs, gate, previous, chunk_rows)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != f"P5-CAT-{gate.upper()}"
        or manifest.get("role") != "A0_POI_ONLY_PRQK_INITIALIZATION"
        or manifest.get("contract") != contract
        or manifest.get("next_status") != gate_next_status(config, gate)
    ):
        raise BaseCodebookDataError("P5 manifest schema/状态/合同不匹配")
    expected_names = set(_iter_expected_artifacts())
    if set(manifest.get("artifacts", {})) != expected_names:
        raise BaseCodebookDataError("P5 artifact 清单不完整")
    for name, entry in manifest["artifacts"].items():
        path = directory / name
        if entry.get("file") != name or sha256_file(path) != entry.get("sha256"):
            raise BaseCodebookDataError(f"P5 artifact 缺失或 SHA256 错误：{name}")
    global_mean, _ = compute_global_mean(inputs.embeddings, chunk_rows=chunk_rows)
    _validate_base_artifacts(directory, config, inputs, global_mean)
    values = _load_selected_transformed(
        inputs, global_mean, device=device, chunk_rows=chunk_rows
    )
    residual_s0 = np.load(
        directory / "poi_residual_s0.npy", mmap_mode="r", allow_pickle=False
    )
    if residual_s0.shape != tuple(values.shape) or residual_s0.dtype != np.float16:
        raise BaseCodebookDataError("P5 S0 residual shape/dtype 错误")
    for start in range(0, len(values), chunk_rows):
        stop = min(start + chunk_rows, len(values))
        expected = values[start:stop].detach().float().cpu().numpy().astype(np.float16)
        if not np.array_equal(expected, residual_s0[start:stop]):
            raise BaseCodebookDataError("P5 S0 residual 未按 global component removal 构建")
    for level in range(1, 4):
        codebook_file, assignment_file, residual_file, _ = _level_files(level)
        codebook_np = np.load(directory / codebook_file, allow_pickle=False)
        assignment_np = np.load(directory / assignment_file, allow_pickle=False)
        residual_np = np.load(
            directory / residual_file, mmap_mode="r", allow_pickle=False
        )
        if (
            codebook_np.shape
            != (config.codebook_sizes[level - 1], inputs.embedding_dim)
            or codebook_np.dtype != np.float32
            or assignment_np.shape != (len(inputs.selected_rows),)
            or assignment_np.dtype != np.int32
            or residual_np.shape != (len(inputs.selected_rows), inputs.embedding_dim)
            or residual_np.dtype != np.float16
            or not np.isfinite(codebook_np).all()
            or not np.isfinite(residual_np).all()
            or int(assignment_np.min()) < 0
            or int(assignment_np.max()) >= config.codebook_sizes[level - 1]
        ):
            raise BaseCodebookDataError(f"P5 S{level} codebook/assignment/residual 合同错误")
        torch = _torch()
        centroids = torch.from_numpy(codebook_np).to(values.device)
        expected_assignments, _ = _hard_assign(values, centroids, chunk_rows=chunk_rows)
        actual_assignments = torch.from_numpy(assignment_np.astype(np.int64)).to(
            values.device
        )
        if not bool(torch.equal(expected_assignments, actual_assignments)):
            raise BaseCodebookDataError(f"P5 S{level} assignment 不是冻结 codebook 的精确 argmax")
        values, diagnostics = projection_residual(values, centroids, actual_assignments)
        for start in range(0, len(values), chunk_rows):
            stop = min(start + chunk_rows, len(values))
            expected = (
                values[start:stop].detach().float().cpu().numpy().astype(np.float16)
            )
            if not np.array_equal(expected, residual_np[start:stop]):
                raise BaseCodebookDataError(f"P5 S{level} residual 未按 projection 递推")
        recorded = json.loads(
            (directory / _level_files(level)[3]).read_text(encoding="utf-8")
        )["residual"]
        for key in (
            "zero_residual_rows",
            "orthogonality_abs_dot_max",
            "retained_energy_mean",
            "retained_energy_p50",
        ):
            if not np.isclose(recorded[key], diagnostics[key], rtol=1e-6, atol=1e-7):
                raise BaseCodebookDataError(f"P5 S{level} residual metric 独立复算不一致：{key}")
    assignments = np.load(
        directory / "poi_assignments_s1_s2_s3.npy", allow_pickle=False
    )
    combined_codebooks = np.load(
        directory / "poi_codebooks_s1_s2_s3.npy", allow_pickle=False
    )
    if assignments.dtype != np.int32 or assignments.shape != (
        len(inputs.selected_rows),
        3,
    ):
        raise BaseCodebookDataError("P5 combined assignments shape/dtype 错误")
    if combined_codebooks.shape != (3, 512, inputs.embedding_dim):
        raise BaseCodebookDataError("P5 combined codebooks shape 错误")
    for level in range(3):
        if not np.array_equal(
            assignments[:, level],
            np.load(directory / _level_files(level + 1)[1], allow_pickle=False),
        ) or not np.array_equal(
            combined_codebooks[level],
            np.load(directory / _level_files(level + 1)[0], allow_pickle=False),
        ):
            raise BaseCodebookDataError("P5 combined artifact 与逐层 artifact 不一致")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if metrics["sid"] != sid_metrics(assignments, 512):
        raise BaseCodebookDataError("P5 SID metrics 独立复算不一致")
    if manifest.get("metrics") != metrics:
        raise BaseCodebookDataError("P5 manifest metrics 与 metrics.json 不一致")
    source_access = manifest.get("source_access", {})
    if any(
        source_access.get(name) is not False
        for name in (
            "raw_train_read",
            "business_validation_read",
            "business_test_read",
            "query_graph_read",
            "category_values_read",
            "geo_values_read",
        )
    ):
        raise BaseCodebookDataError("P5 A0 非法读取 Query/Category/Geo/Validation/Test")
    del values
    gc.collect()
    return {
        "status": "validated",
        "phase": f"P5-CAT-{gate.upper()}",
        "output_dir": str(directory),
        "manifest_sha256": manifest_sha,
        "metrics": metrics,
        "next_status": gate_next_status(config, gate),
    }

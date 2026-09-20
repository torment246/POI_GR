"""Paired hard-fit/Top-k diagnostics for the base codebook."""

from __future__ import annotations

import gc
import json
import os
import resource
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.pipeline_config import DownstreamConfig
from qg_prqk.sid.base_data import BaseCodebookDataError, BaseCodebookInputs, BaseCodebookAlgorithmConfig, compute_global_mean
from qg_prqk.sid.base_diagnostics import (
    SCHEMA_VERSION as ENDPOINT_DIAGNOSTIC_SCHEMA_VERSION,
    _artifact_relative,
    _validate_reference,
    diagnostic_output_directory as endpoint_diagnostic_output_directory,
    diagnostic_settings,
)
from qg_prqk.sid.base_quantizer import (
    DEFAULT_CHUNK_ROWS,
    _atomic_npy,
    _distribution_metrics,
    _hard_assign,
    _load_selected_transformed,
    _soft_centroid_update,
    _torch,
    fit_prqk_level,
    projection_residual,
    sid_metrics,
)


SCHEMA_VERSION = "qg-prqk-p5-medium100k-hard60-topk5-diagnostic-v1"
DIAGNOSTIC_NAME = "medium_100000_hard60_topk5_v1"
TOPK_BRANCH = "hard60_topk5"
LOCAL_HARD_BRANCH = "hard60_pre_topk_local"
ROLE = "P5_A0_CONTROLLED_HARD60_TOPK5_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT"
NEXT_STATUS = "HOLD_FOR_P5_MEDIUM100K_PARAMETER_REVIEW"


def diagnostic_output_directory(config: DownstreamConfig) -> Path:
    """Return the isolated output directory for the paired Top-k diagnostic."""
    path = config.output_dir / "poi_prqk_a0" / "diagnostics" / DIAGNOSTIC_NAME
    if not path.resolve().is_relative_to(config.output_dir.resolve()):
        raise BaseCodebookDataError("P5 hard60+Top-k5 诊断输出越出 QG 下游目录")
    return path


def hard60_topk5_settings(canonical: BaseCodebookAlgorithmConfig) -> BaseCodebookAlgorithmConfig:
    """Change only the hard ceiling while preserving canonical Top-k settings."""
    refinement = canonical.topk_refinement
    if (
        not refinement.enabled
        or refinement.topk != 5
        or refinement.beta != 15
        or refinement.max_iter != 5
    ):
        raise BaseCodebookDataError("canonical Top-k5 设置与已审核合同不匹配")
    return replace(
        canonical,
        max_iter=60,
        topk_refinement=replace(refinement, enabled=True),
    )


def _code_files() -> dict[str, str]:
    directory = Path(__file__).parent
    files = {
        "sid/base_data.py": directory / "base_data.py",
        "sid/base_quantizer.py": directory / "base_quantizer.py",
        "sid/base_diagnostics.py": directory / "base_diagnostics.py",
        "sid/base_topk_diagnostics.py": directory / "base_topk_diagnostics.py",
        "commands/diagnose_base_topk.py": (
            directory.parent / "commands/diagnose_base_topk.py"
        ),
    }
    return {name: sha256_file(path) for name, path in files.items()}


def _validate_endpoint_diagnostic(
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    canonical: BaseCodebookAlgorithmConfig,
    path: Path,
    expected_sha256: str,
    *,
    base_reference_sha256: str,
    verify_artifacts: bool,
) -> dict[str, Any]:
    path = path.resolve()
    expected_path = (
        endpoint_diagnostic_output_directory(config) / "manifest.json"
    ).resolve()
    if path != expected_path or not path.is_file():
        raise BaseCodebookDataError("Top-k 补充诊断必须绑定已冻结 hard endpoint manifest")
    if sha256_file(path) != expected_sha256:
        raise BaseCodebookDataError("hard endpoint diagnostic manifest SHA256 不匹配")
    marker = json.loads((path.parent / "_SUCCESS").read_text(encoding="utf-8"))
    if marker.get("manifest_sha256") != expected_sha256:
        raise BaseCodebookDataError("hard endpoint diagnostic 成功标记未锁定 manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    contract = manifest.get("contract", {})
    expected_settings = asdict(diagnostic_settings(canonical, max_iter=60))
    if (
        manifest.get("schema_version") != ENDPOINT_DIAGNOSTIC_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != "P5-CAT-MEDIUM100K-DIAGNOSTIC"
        or manifest.get("role")
        != "P5_A0_CONTROLLED_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT"
        or contract.get("config_signature") != config.signature()
        or contract.get("poi_rows") != 100_000
        or contract.get("selected_rows_sha256")
        != inputs.source_hashes["selected_rows_sha256"]
        or contract.get("source_hashes") != dict(inputs.source_hashes)
        or contract.get("reference_manifest_sha256") != base_reference_sha256
        or contract.get("branches", {}).get("hard60_topk_off") != expected_settings
    ):
        raise BaseCodebookDataError("hard endpoint diagnostic 合同不匹配")
    artifacts = manifest.get("artifacts", {})
    required = {
        "hard60_topk_off/metrics.json",
        "hard60_topk_off/poi_assignments_s1_s2_s3.npy",
        "hard60_topk_off/poi_codebooks_s1_s2_s3.npy",
    }
    if not required.issubset(artifacts):
        raise BaseCodebookDataError("hard endpoint diagnostic 缺少 Top-k 对照所需 artifact")
    if verify_artifacts:
        for name, entry in artifacts.items():
            artifact_path = path.parent / name
            if (
                entry.get("file") != name
                or not artifact_path.is_file()
                or sha256_file(artifact_path) != entry.get("sha256")
            ):
                raise BaseCodebookDataError(f"hard endpoint diagnostic artifact 损坏：{name}")
    metrics_path = path.parent / "hard60_topk_off" / "metrics.json"
    return {
        "manifest_path": str(path),
        "manifest_sha256": expected_sha256,
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "assignments_path": str(
            path.parent / "hard60_topk_off" / "poi_assignments_s1_s2_s3.npy"
        ),
        "assignments_sha256": artifacts["hard60_topk_off/poi_assignments_s1_s2_s3.npy"][
            "sha256"
        ],
    }


def _hard_trace(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row for row in metrics["trace"] if row["stage"] == "hard_spherical_kmeans"]


def refine_topk_from_hard_endpoint(
    values,
    hard_centroids,
    hard_assignments,
    hard_fit: Mapping[str, Any],
    settings: BaseCodebookAlgorithmConfig,
    *,
    chunk_rows: int,
):
    """Apply Top-k refinement to one saved hard endpoint without refitting it."""
    centroids = hard_centroids.clone()
    previous_assignments = hard_assignments
    trace = list(hard_fit["trace"])
    for iteration in range(1, settings.topk_refinement.max_iter + 1):
        centroids = _soft_centroid_update(
            values,
            centroids,
            topk=settings.topk_refinement.topk,
            beta=settings.topk_refinement.beta,
            chunk_rows=chunk_rows,
        )
        assignments, scores = _hard_assign(values, centroids, chunk_rows=chunk_rows)
        trace.append(
            {
                "stage": "topk_soft_centroid_refinement",
                "iteration": iteration,
                "objective": float((1.0 - scores).mean().item()),
                "assignment_change": float(
                    (assignments != previous_assignments).float().mean().item()
                ),
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
            "hard_converged": bool(hard_fit["hard_converged"]),
            "hard_iterations": int(hard_fit["hard_iterations"]),
            "soft_refinement_iterations": settings.topk_refinement.max_iter,
            "final_objective": float((1.0 - scores).mean().item()),
            "trace": trace,
        },
    )


def _run_paired_topk_branch(
    initial_values,
    config: DownstreamConfig,
    canonical: BaseCodebookAlgorithmConfig,
    directory: Path,
    *,
    chunk_rows: int,
) -> dict[str, Any]:
    topk_settings = hard60_topk5_settings(canonical)
    local_hard_settings = diagnostic_settings(canonical, max_iter=60)
    values = initial_values
    topk_codebooks: list[np.ndarray] = []
    topk_assignments_by_level: list[np.ndarray] = []
    topk_level_metrics: list[dict[str, Any]] = []
    local_codebooks: list[np.ndarray] = []
    local_assignments_by_level: list[np.ndarray] = []
    local_level_metrics: list[dict[str, Any]] = []
    for level, codebook_size in enumerate(config.codebook_sizes, start=1):
        (
            local_centroids,
            local_assignments,
            local_scores,
            local_fit,
        ) = fit_prqk_level(
            values,
            codebook_size,
            local_hard_settings,
            level=level,
            chunk_rows=chunk_rows,
        )
        topk_centroids, topk_assignments, topk_scores, topk_fit = (
            refine_topk_from_hard_endpoint(
                values,
                local_centroids,
                local_assignments,
                local_fit,
                topk_settings,
                chunk_rows=chunk_rows,
            )
        )
        if _hard_trace(local_fit) != _hard_trace(topk_fit):
            raise BaseCodebookDataError(f"P5 hard60+Top-k5 S{level} 的配对 hard trace 不一致")
        values, residual_metrics = projection_residual(
            values, topk_centroids, topk_assignments
        )
        local_assignments_np = local_assignments.detach().cpu().numpy().astype(np.int32)
        topk_assignments_np = topk_assignments.detach().cpu().numpy().astype(np.int32)
        local_codebooks.append(
            local_centroids.detach().float().cpu().numpy().astype(np.float32)
        )
        topk_codebooks.append(
            topk_centroids.detach().float().cpu().numpy().astype(np.float32)
        )
        local_assignments_by_level.append(local_assignments_np)
        topk_assignments_by_level.append(topk_assignments_np)
        local_level_metrics.append(
            {
                "level": level,
                "fit": local_fit,
                "assignment": _distribution_metrics(
                    local_assignments_np, codebook_size
                ),
                "hard_cosine_similarity_mean": float(local_scores.mean().item()),
            }
        )
        topk_level_metrics.append(
            {
                "level": level,
                "fit": topk_fit,
                "assignment": _distribution_metrics(topk_assignments_np, codebook_size),
                "hard_cosine_similarity_mean": float(topk_scores.mean().item()),
                "residual": residual_metrics,
            }
        )
        del (
            local_centroids,
            local_assignments,
            local_scores,
            topk_centroids,
            topk_assignments,
            topk_scores,
            local_assignments_np,
            topk_assignments_np,
        )
        gc.collect()
    local_assignments_all = np.column_stack(local_assignments_by_level).astype(np.int32)
    topk_assignments_all = np.column_stack(topk_assignments_by_level).astype(np.int32)
    local_codebooks_all = np.stack(local_codebooks).astype(np.float32)
    topk_codebooks_all = np.stack(topk_codebooks).astype(np.float32)
    local_metrics = {
        "role": (
            "PER_LEVEL_HARD60_ENDPOINT_ON_TOPK_BRANCH_INPUTS_"
            "NOT_A_COHERENT_ALL_OFF_SID_PATH"
        ),
        "settings": asdict(local_hard_settings),
        "level_metrics": local_level_metrics,
    }
    topk_metrics = {
        "settings": asdict(topk_settings),
        "level_metrics": topk_level_metrics,
        "sid": sid_metrics(topk_assignments_all, config.codebook_sizes[0]),
    }
    local_directory = directory / LOCAL_HARD_BRANCH
    topk_directory = directory / TOPK_BRANCH
    local_directory.mkdir(parents=True, exist_ok=False)
    topk_directory.mkdir(parents=True, exist_ok=False)
    _atomic_npy(
        local_directory / "poi_assignments_s1_s2_s3.npy",
        local_assignments_all,
    )
    _atomic_npy(
        local_directory / "poi_codebooks_s1_s2_s3.npy",
        local_codebooks_all,
    )
    write_json_atomic(local_directory / "metrics.json", local_metrics)
    _atomic_npy(
        topk_directory / "poi_assignments_s1_s2_s3.npy",
        topk_assignments_all,
    )
    _atomic_npy(
        topk_directory / "poi_codebooks_s1_s2_s3.npy",
        topk_codebooks_all,
    )
    write_json_atomic(topk_directory / "metrics.json", topk_metrics)
    del (
        values,
        local_assignments_all,
        topk_assignments_all,
        local_codebooks_all,
        topk_codebooks_all,
        local_codebooks,
        topk_codebooks,
        local_assignments_by_level,
        topk_assignments_by_level,
    )
    gc.collect()
    return {
        "local_hard_metrics": local_metrics,
        "topk_metrics": topk_metrics,
    }


def compare_hard60_topk5(
    reference_metrics: Mapping[str, Any],
    reference_assignments: np.ndarray,
    all_off_metrics: Mapping[str, Any],
    all_off_assignments: np.ndarray,
    local_hard_metrics: Mapping[str, Any],
    local_hard_assignments: np.ndarray,
    topk_metrics: Mapping[str, Any],
    topk_assignments: np.ndarray,
) -> dict[str, Any]:
    """Compare Top-k against paired per-level and coherent all-off endpoints."""
    levels: list[dict[str, Any]] = []
    for index in range(3):
        reference_level = reference_metrics["level_metrics"][index]
        all_off_level = all_off_metrics["level_metrics"][index]
        local_level = local_hard_metrics["level_metrics"][index]
        topk_level = topk_metrics["level_metrics"][index]
        local_objective = float(local_level["fit"]["final_objective"])
        topk_objective = float(topk_level["fit"]["final_objective"])
        all_off_objective = float(all_off_level["fit"]["final_objective"])
        levels.append(
            {
                "level": index + 1,
                "original_hard30_topk5_final_objective": float(
                    reference_level["fit"]["final_objective"]
                ),
                "all_off_hard60_final_objective": all_off_objective,
                "paired_pre_topk_hard60_final_objective": local_objective,
                "hard60_topk5_final_objective": topk_objective,
                "topk5_minus_paired_pre_topk": topk_objective - local_objective,
                "topk5_relative_change_vs_paired_pre_topk": (
                    (topk_objective - local_objective) / local_objective
                ),
                "topk5_minus_all_off_pipeline": topk_objective - all_off_objective,
                "paired_topk_assignment_change": float(
                    np.mean(
                        local_hard_assignments[:, index] != topk_assignments[:, index]
                    )
                ),
                "all_off_vs_topk_assignment_change": float(
                    np.mean(all_off_assignments[:, index] != topk_assignments[:, index])
                ),
                "hard_converged": bool(topk_level["fit"]["hard_converged"]),
                "hard_iterations": int(topk_level["fit"]["hard_iterations"]),
                "soft_refinement_iterations": int(
                    topk_level["fit"]["soft_refinement_iterations"]
                ),
                "hard_trace_matches_paired_topk_off": (
                    _hard_trace(local_level["fit"]) == _hard_trace(topk_level["fit"])
                ),
                "paired_pre_topk_distribution": local_level["assignment"],
                "topk5_distribution": topk_level["assignment"],
            }
        )
    return {
        "comparison_scope": (
            "Per-level paired_pre_topk versus Top-k5 is a direct comparison on "
            "the same residual input. all_off versus Top-k5 is direct only for "
            "S1; S2/S3 are complete-pipeline comparisons."
        ),
        "levels": levels,
        "sid": {
            "original_hard30_topk5": reference_metrics["sid"],
            "hard60_topk_off": all_off_metrics["sid"],
            "hard60_topk5": topk_metrics["sid"],
            "original_vs_hard60_topk5_path_change": float(
                np.mean(np.any(reference_assignments != topk_assignments, axis=1))
            ),
            "hard60_off_vs_topk5_path_change": float(
                np.mean(np.any(all_off_assignments != topk_assignments, axis=1))
            ),
        },
        "evidence": {
            "hard60_all_levels_converged": all(row["hard_converged"] for row in levels),
            "hard_trace_matches_all_paired_runs": all(
                row["hard_trace_matches_paired_topk_off"] for row in levels
            ),
            "topk5_worse_than_paired_pre_topk_all_levels": all(
                row["topk5_minus_paired_pre_topk"] > 0 for row in levels
            ),
            "topk5_improves_kish_ess_all_levels": all(
                row["topk5_distribution"]["kish_ess"]
                > row["paired_pre_topk_distribution"]["kish_ess"]
                for row in levels
            ),
            "topk5_reduces_gini_all_levels": all(
                row["topk5_distribution"]["gini"]
                < row["paired_pre_topk_distribution"]["gini"]
                for row in levels
            ),
            "topk5_increases_distinct_sid_vs_hard60_off": (
                topk_metrics["sid"]["distinct_sid"]
                > all_off_metrics["sid"]["distinct_sid"]
            ),
            "canonical_change_applied": False,
        },
    }


def _contract(
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    canonical: BaseCodebookAlgorithmConfig,
    reference: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    chunk_rows: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_signature": config.signature(),
        "poi_rows": len(inputs.selected_rows),
        "active_poi_rows": inputs.total_rows,
        "embedding_dim": inputs.embedding_dim,
        "selected_rows_sha256": inputs.source_hashes["selected_rows_sha256"],
        "source_hashes": dict(inputs.source_hashes),
        "reference_manifest_path": reference["manifest_path"],
        "reference_manifest_sha256": reference["manifest_sha256"],
        "reference_assignments_sha256": reference["assignments_sha256"],
        "endpoint_diagnostic_manifest_path": endpoint["manifest_path"],
        "endpoint_diagnostic_manifest_sha256": endpoint["manifest_sha256"],
        "endpoint_hard60_assignments_sha256": endpoint["assignments_sha256"],
        "canonical_settings": asdict(canonical),
        "hard60_pre_topk_local_settings": asdict(
            diagnostic_settings(canonical, max_iter=60)
        ),
        "hard60_topk5_settings": asdict(hard60_topk5_settings(canonical)),
        "chunk_rows": chunk_rows,
        "code": _code_files(),
    }


def _artifact_paths(directory: Path) -> list[Path]:
    paths = [directory / "comparison.json", directory / "config_resolved.json"]
    for branch in (LOCAL_HARD_BRANCH, TOPK_BRANCH):
        branch_directory = directory / branch
        paths.extend(
            [
                branch_directory / "poi_assignments_s1_s2_s3.npy",
                branch_directory / "poi_codebooks_s1_s2_s3.npy",
                branch_directory / "metrics.json",
            ]
        )
    return paths


def build_base_topk_diagnostic(
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    canonical: BaseCodebookAlgorithmConfig,
    *,
    reference_manifest: Path,
    reference_manifest_sha256: str,
    endpoint_diagnostic_manifest: Path,
    endpoint_diagnostic_manifest_sha256: str,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Build the approved paired hard60+Top-k5 diagnostic atomically."""
    if len(inputs.selected_rows) != 100_000:
        raise BaseCodebookDataError("P5 hard60+Top-k5 诊断必须使用 100,000 行")
    final_directory = diagnostic_output_directory(config)
    if final_directory.exists():
        raise BaseCodebookDataError(
            f"P5 hard60+Top-k5 诊断输出已存在且禁止覆盖：{final_directory}"
        )
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    work_directory = final_directory.with_name(
        f".{final_directory.name}.{uuid.uuid4().hex}.writing"
    )
    work_directory.mkdir()
    started = time.perf_counter()
    reference = _validate_reference(
        config,
        inputs,
        canonical,
        reference_manifest,
        reference_manifest_sha256,
        verify_artifacts=True,
    )
    endpoint = _validate_endpoint_diagnostic(
        config,
        inputs,
        canonical,
        endpoint_diagnostic_manifest,
        endpoint_diagnostic_manifest_sha256,
        base_reference_sha256=reference_manifest_sha256,
        verify_artifacts=True,
    )
    global_mean, global_metrics = compute_global_mean(
        inputs.embeddings, chunk_rows=chunk_rows
    )
    initial_values = _load_selected_transformed(
        inputs, global_mean, device=device, chunk_rows=chunk_rows
    )
    paired = _run_paired_topk_branch(
        initial_values,
        config,
        canonical,
        work_directory,
        chunk_rows=chunk_rows,
    )
    reference_assignments = np.load(reference["assignments_path"], allow_pickle=False)
    all_off_assignments = np.load(endpoint["assignments_path"], allow_pickle=False)
    local_hard_assignments = np.load(
        work_directory / LOCAL_HARD_BRANCH / "poi_assignments_s1_s2_s3.npy",
        allow_pickle=False,
    )
    topk_assignments = np.load(
        work_directory / TOPK_BRANCH / "poi_assignments_s1_s2_s3.npy",
        allow_pickle=False,
    )
    comparison = compare_hard60_topk5(
        reference["metrics"],
        reference_assignments,
        endpoint["metrics"],
        all_off_assignments,
        paired["local_hard_metrics"],
        local_hard_assignments,
        paired["topk_metrics"],
        topk_assignments,
    )
    write_json_atomic(work_directory / "comparison.json", comparison)
    write_json_atomic(
        work_directory / "config_resolved.json", config.resolved_payload()
    )
    contract = _contract(config, inputs, canonical, reference, endpoint, chunk_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "phase": "P5-CAT-MEDIUM100K-HARD60-TOPK5-DIAGNOSTIC",
        "built_at": utc_now(),
        "contract": contract,
        "global_component": global_metrics,
        "comparison": comparison,
        "source_access": {
            "active_poi_embedding_values_read": True,
            "reference_p5_medium100k_read": True,
            "endpoint_diagnostic_read": True,
            "raw_train_read": False,
            "business_validation_read": False,
            "business_test_read": False,
            "query_graph_read": False,
            "category_values_read": False,
            "geo_values_read": False,
        },
        "role": ROLE,
        "artifacts": {
            str(path.relative_to(work_directory)): _artifact_relative(
                work_directory, path
            )
            for path in _artifact_paths(work_directory)
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
            "peak_rss_mib": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        },
        "next_status": NEXT_STATUS,
    }
    write_json_atomic(work_directory / "manifest.json", manifest)
    write_json_atomic(
        work_directory / "_SUCCESS",
        {"manifest_sha256": sha256_file(work_directory / "manifest.json")},
    )
    os.replace(work_directory, final_directory)
    del initial_values
    gc.collect()
    return validate_base_topk_diagnostic(
        config,
        inputs,
        canonical,
        reference_manifest=reference_manifest,
        reference_manifest_sha256=reference_manifest_sha256,
        endpoint_diagnostic_manifest=endpoint_diagnostic_manifest,
        endpoint_diagnostic_manifest_sha256=(endpoint_diagnostic_manifest_sha256),
        device=device,
        chunk_rows=chunk_rows,
    )


def validate_base_topk_diagnostic(
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    canonical: BaseCodebookAlgorithmConfig,
    *,
    reference_manifest: Path,
    reference_manifest_sha256: str,
    endpoint_diagnostic_manifest: Path,
    endpoint_diagnostic_manifest_sha256: str,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Independently validate paired endpoints, residuals, and comparison."""
    directory = diagnostic_output_directory(config)
    reference = _validate_reference(
        config,
        inputs,
        canonical,
        reference_manifest,
        reference_manifest_sha256,
        verify_artifacts=True,
    )
    endpoint = _validate_endpoint_diagnostic(
        config,
        inputs,
        canonical,
        endpoint_diagnostic_manifest,
        endpoint_diagnostic_manifest_sha256,
        base_reference_sha256=reference_manifest_sha256,
        verify_artifacts=True,
    )
    manifest_path = directory / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    marker = json.loads((directory / "_SUCCESS").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = _contract(config, inputs, canonical, reference, endpoint, chunk_rows)
    if (
        marker.get("manifest_sha256") != manifest_sha
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != "P5-CAT-MEDIUM100K-HARD60-TOPK5-DIAGNOSTIC"
        or manifest.get("role") != ROLE
        or manifest.get("contract") != contract
        or manifest.get("next_status") != NEXT_STATUS
    ):
        raise BaseCodebookDataError("P5 hard60+Top-k5 诊断 manifest/状态/合同不匹配")
    expected_paths = _artifact_paths(directory)
    expected_names = {str(path.relative_to(directory)) for path in expected_paths}
    if set(manifest.get("artifacts", {})) != expected_names:
        raise BaseCodebookDataError("P5 hard60+Top-k5 诊断 artifact 清单不完整")
    for name, entry in manifest["artifacts"].items():
        path = directory / name
        if (
            entry.get("file") != name
            or not path.is_file()
            or sha256_file(path) != entry.get("sha256")
        ):
            raise BaseCodebookDataError(f"P5 hard60+Top-k5 诊断 artifact 损坏：{name}")
    if (
        json.loads((directory / "config_resolved.json").read_text(encoding="utf-8"))
        != config.resolved_payload()
    ):
        raise BaseCodebookDataError("P5 hard60+Top-k5 resolved config 不匹配")
    local_directory = directory / LOCAL_HARD_BRANCH
    topk_directory = directory / TOPK_BRANCH
    local_metrics = json.loads(
        (local_directory / "metrics.json").read_text(encoding="utf-8")
    )
    topk_metrics = json.loads(
        (topk_directory / "metrics.json").read_text(encoding="utf-8")
    )
    local_assignments = np.load(
        local_directory / "poi_assignments_s1_s2_s3.npy",
        allow_pickle=False,
    )
    topk_assignments = np.load(
        topk_directory / "poi_assignments_s1_s2_s3.npy",
        allow_pickle=False,
    )
    local_codebooks = np.load(
        local_directory / "poi_codebooks_s1_s2_s3.npy",
        allow_pickle=False,
    )
    topk_codebooks = np.load(
        topk_directory / "poi_codebooks_s1_s2_s3.npy",
        allow_pickle=False,
    )
    for name, assignments, codebooks in (
        (LOCAL_HARD_BRANCH, local_assignments, local_codebooks),
        (TOPK_BRANCH, topk_assignments, topk_codebooks),
    ):
        if (
            assignments.shape != (100_000, 3)
            or assignments.dtype != np.int32
            or codebooks.shape != (3, 512, inputs.embedding_dim)
            or codebooks.dtype != np.float32
        ):
            raise BaseCodebookDataError(f"P5 hard60+Top-k5 {name} shape/dtype 错误")
    if (
        local_metrics.get("settings") != contract["hard60_pre_topk_local_settings"]
        or topk_metrics.get("settings") != contract["hard60_topk5_settings"]
    ):
        raise BaseCodebookDataError("P5 hard60+Top-k5 settings 错误")
    global_mean, _ = compute_global_mean(inputs.embeddings, chunk_rows=chunk_rows)
    values = _load_selected_transformed(
        inputs, global_mean, device=device, chunk_rows=chunk_rows
    )
    torch = _torch()
    for index, codebook_size in enumerate(config.codebook_sizes):
        local_centroids = torch.from_numpy(local_codebooks[index]).to(values.device)
        topk_centroids = torch.from_numpy(topk_codebooks[index]).to(values.device)
        local_expected, local_scores = _hard_assign(
            values, local_centroids, chunk_rows=chunk_rows
        )
        topk_expected, topk_scores = _hard_assign(
            values, topk_centroids, chunk_rows=chunk_rows
        )
        local_actual = torch.from_numpy(
            local_assignments[:, index].astype(np.int64)
        ).to(values.device)
        topk_actual = torch.from_numpy(topk_assignments[:, index].astype(np.int64)).to(
            values.device
        )
        if not bool(torch.equal(local_expected, local_actual)):
            raise BaseCodebookDataError(f"P5 hard60 pre-Top-k S{index + 1} assignment 错误")
        if not bool(torch.equal(topk_expected, topk_actual)):
            raise BaseCodebookDataError(f"P5 hard60+Top-k5 S{index + 1} assignment 错误")
        local_level = local_metrics["level_metrics"][index]
        topk_level = topk_metrics["level_metrics"][index]
        local_objective = float((1.0 - local_scores).mean().item())
        topk_objective = float((1.0 - topk_scores).mean().item())
        if not np.isclose(
            local_objective,
            local_level["fit"]["final_objective"],
            rtol=1.0e-7,
            atol=1.0e-8,
        ):
            raise BaseCodebookDataError(f"P5 hard60 pre-Top-k S{index + 1} objective 错误")
        if not np.isclose(
            topk_objective,
            topk_level["fit"]["final_objective"],
            rtol=1.0e-7,
            atol=1.0e-8,
        ):
            raise BaseCodebookDataError(f"P5 hard60+Top-k5 S{index + 1} objective 错误")
        if (
            local_level["assignment"]
            != _distribution_metrics(local_assignments[:, index], codebook_size)
            or topk_level["assignment"]
            != _distribution_metrics(topk_assignments[:, index], codebook_size)
            or _hard_trace(local_level["fit"]) != _hard_trace(topk_level["fit"])
        ):
            raise BaseCodebookDataError(f"P5 hard60+Top-k5 S{index + 1} 配对指标错误")
        values, residual = projection_residual(values, topk_centroids, topk_actual)
        for key, actual in residual.items():
            if not np.isclose(
                actual,
                topk_level["residual"][key],
                rtol=1.0e-6,
                atol=1.0e-7,
            ):
                raise BaseCodebookDataError(
                    f"P5 hard60+Top-k5 S{index + 1} residual metric 错误：{key}"
                )
    if topk_metrics["sid"] != sid_metrics(topk_assignments, 512):
        raise BaseCodebookDataError("P5 hard60+Top-k5 SID metrics 错误")
    reference_assignments = np.load(reference["assignments_path"], allow_pickle=False)
    all_off_assignments = np.load(endpoint["assignments_path"], allow_pickle=False)
    comparison = compare_hard60_topk5(
        reference["metrics"],
        reference_assignments,
        endpoint["metrics"],
        all_off_assignments,
        local_metrics,
        local_assignments,
        topk_metrics,
        topk_assignments,
    )
    if (
        comparison
        != json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
        or manifest.get("comparison") != comparison
    ):
        raise BaseCodebookDataError("P5 hard60+Top-k5 comparison 独立复算不一致")
    forbidden_reads = (
        "raw_train_read",
        "business_validation_read",
        "business_test_read",
        "query_graph_read",
        "category_values_read",
        "geo_values_read",
    )
    if any(
        manifest.get("source_access", {}).get(key) is not False
        for key in forbidden_reads
    ):
        raise BaseCodebookDataError(
            "P5 hard60+Top-k5 非法读取 Query/Category/Geo/Validation/Test"
        )
    del values
    gc.collect()
    return {
        "status": "validated",
        "phase": "P5-CAT-MEDIUM100K-HARD60-TOPK5-DIAGNOSTIC",
        "output_dir": str(directory),
        "manifest_sha256": manifest_sha,
        "comparison": comparison,
        "next_status": NEXT_STATUS,
    }

"""Controlled hard-endpoint diagnostics for the base codebook."""

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
from qg_prqk.sid.base_quantizer import (
    DEFAULT_CHUNK_ROWS,
    SCHEMA_VERSION as P5_SCHEMA_VERSION,
    _artifact,
    _atomic_npy,
    _distribution_metrics,
    _hard_assign,
    _load_selected_transformed,
    _torch,
    fit_prqk_level,
    projection_residual,
    sid_metrics,
)


SCHEMA_VERSION = "qg-prqk-p5-medium100k-hard-diagnostic-v1"
DIAGNOSTIC_NAME = "medium_100000_hard_endpoint_v1"
BRANCH_MAX_ITER = {
    "hard30_topk_off": 30,
    "hard60_topk_off": 60,
}


def diagnostic_output_directory(config: DownstreamConfig) -> Path:
    """Return the isolated directory for the approved 100k diagnostic."""
    path = config.output_dir / "poi_prqk_a0" / "diagnostics" / DIAGNOSTIC_NAME
    if not path.resolve().is_relative_to(config.output_dir.resolve()):
        raise BaseCodebookDataError("P5 诊断输出越出 QG 下游目录")
    return path


def diagnostic_settings(
    canonical: BaseCodebookAlgorithmConfig, *, max_iter: int
) -> BaseCodebookAlgorithmConfig:
    """Disable Top-k and change only the diagnostic hard-iteration ceiling."""
    if max_iter not in BRANCH_MAX_ITER.values():
        raise BaseCodebookDataError("P5 hard 诊断只允许 max_iter=30 或 60")
    return replace(
        canonical,
        max_iter=max_iter,
        topk_refinement=replace(canonical.topk_refinement, enabled=False),
    )


def _code_files() -> dict[str, str]:
    directory = Path(__file__).parent
    files = {
        "sid/base_data.py": directory / "base_data.py",
        "sid/base_quantizer.py": directory / "base_quantizer.py",
        "sid/base_diagnostics.py": directory / "base_diagnostics.py",
        "commands/diagnose_base_codebook.py": (
            directory.parent / "commands/diagnose_base_codebook.py"
        ),
    }
    return {name: sha256_file(path) for name, path in files.items()}


def _artifact_relative(root: Path, path: Path) -> dict[str, Any]:
    entry = _artifact(path)
    entry["file"] = str(path.relative_to(root))
    return entry


def _validate_reference(
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    canonical: BaseCodebookAlgorithmConfig,
    path: Path,
    expected_sha256: str,
    *,
    verify_artifacts: bool,
) -> dict[str, Any]:
    path = path.resolve()
    expected_path = (
        config.output_dir / "poi_prqk_a0" / "medium_100000" / "manifest.json"
    ).resolve()
    if path != expected_path or not path.is_file():
        raise BaseCodebookDataError("诊断必须绑定 canonical P5 100k manifest")
    if sha256_file(path) != expected_sha256:
        raise BaseCodebookDataError("P5 100k reference manifest SHA256 不匹配")
    marker = json.loads((path.parent / "_SUCCESS").read_text(encoding="utf-8"))
    if marker.get("manifest_sha256") != expected_sha256:
        raise BaseCodebookDataError("P5 100k reference 成功标记未锁定 manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    contract = manifest.get("contract", {})
    if (
        manifest.get("schema_version") != P5_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != "P5-CAT-MEDIUM100K"
        or manifest.get("role") != "A0_POI_ONLY_PRQK_INITIALIZATION"
        or contract.get("gate") != "medium100k"
        or contract.get("poi_rows") != 100_000
        or contract.get("config_signature") != config.signature()
        or contract.get("algorithm") != asdict(canonical)
        or contract.get("selection", {}).get("selected_rows_sha256")
        != inputs.source_hashes["selected_rows_sha256"]
        or contract.get("source_hashes") != dict(inputs.source_hashes)
    ):
        raise BaseCodebookDataError("P5 100k reference 合同不匹配")
    forbidden_reads = (
        "raw_train_read",
        "business_validation_read",
        "business_test_read",
        "query_graph_read",
        "category_values_read",
        "geo_values_read",
    )
    if any(manifest.get("source_access", {}).get(key) is not False for key in forbidden_reads):
        raise BaseCodebookDataError("P5 100k reference 非法读取了诊断禁用数据")
    artifacts = manifest.get("artifacts", {})
    required = {"metrics.json", "poi_assignments_s1_s2_s3.npy"}
    if not required.issubset(artifacts):
        raise BaseCodebookDataError("P5 100k reference 缺少诊断所需 artifact")
    if verify_artifacts:
        for name, entry in artifacts.items():
            artifact_path = path.parent / name
            if (
                entry.get("file") != name
                or not artifact_path.is_file()
                or sha256_file(artifact_path) != entry.get("sha256")
            ):
                raise BaseCodebookDataError(f"P5 100k reference artifact 损坏：{name}")
    metrics = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
    if metrics != manifest.get("metrics"):
        raise BaseCodebookDataError("P5 100k reference metrics 与 manifest 不一致")
    return {
        "manifest_path": str(path),
        "manifest_sha256": expected_sha256,
        "metrics": metrics,
        "assignments_path": str(path.parent / "poi_assignments_s1_s2_s3.npy"),
        "assignments_sha256": artifacts["poi_assignments_s1_s2_s3.npy"]["sha256"],
    }


def _run_branch(
    initial_values,
    config: DownstreamConfig,
    settings: BaseCodebookAlgorithmConfig,
    directory: Path,
    *,
    chunk_rows: int,
) -> dict[str, Any]:
    values = initial_values
    codebooks = []
    assignments_by_level = []
    level_metrics = []
    for level, codebook_size in enumerate(config.codebook_sizes, start=1):
        centroids, assignments, scores, fit_metrics = fit_prqk_level(
            values,
            codebook_size,
            settings,
            level=level,
            chunk_rows=chunk_rows,
        )
        values, residual_metrics = projection_residual(values, centroids, assignments)
        assignments_np = assignments.detach().cpu().numpy().astype(np.int32)
        codebooks.append(centroids.detach().float().cpu().numpy().astype(np.float32))
        assignments_by_level.append(assignments_np)
        level_metrics.append(
            {
                "level": level,
                "fit": fit_metrics,
                "assignment": _distribution_metrics(assignments_np, codebook_size),
                "hard_cosine_similarity_mean": float(scores.mean().item()),
                "residual": residual_metrics,
            }
        )
        del centroids, assignments, scores, assignments_np
        gc.collect()
    combined_assignments = np.column_stack(assignments_by_level).astype(np.int32)
    combined_codebooks = np.stack(codebooks).astype(np.float32)
    metrics = {
        "settings": asdict(settings),
        "level_metrics": level_metrics,
        "sid": sid_metrics(combined_assignments, config.codebook_sizes[0]),
    }
    directory.mkdir(parents=True, exist_ok=False)
    _atomic_npy(directory / "poi_assignments_s1_s2_s3.npy", combined_assignments)
    _atomic_npy(directory / "poi_codebooks_s1_s2_s3.npy", combined_codebooks)
    write_json_atomic(directory / "metrics.json", metrics)
    del values, combined_assignments, combined_codebooks, codebooks, assignments_by_level
    gc.collect()
    return metrics


def _last_hard_trace(level_metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = [
        row
        for row in level_metrics["fit"]["trace"]
        if row["stage"] == "hard_spherical_kmeans"
    ]
    if not rows:
        raise BaseCodebookDataError("P5 metrics 缺少 hard trace")
    return rows[-1]


def compare_diagnostic_branches(
    reference_metrics: Mapping[str, Any],
    reference_assignments: np.ndarray,
    hard30_metrics: Mapping[str, Any],
    hard30_assignments: np.ndarray,
    hard60_metrics: Mapping[str, Any],
    hard60_assignments: np.ndarray,
) -> dict[str, Any]:
    """Compare frozen Top-k output with two deterministic hard-only branches."""
    levels = []
    for index in range(3):
        reference_level = reference_metrics["level_metrics"][index]
        hard30_level = hard30_metrics["level_metrics"][index]
        hard60_level = hard60_metrics["level_metrics"][index]
        reference_last_hard = _last_hard_trace(reference_level)
        reference_final = float(reference_level["fit"]["final_objective"])
        hard30_final = float(hard30_level["fit"]["final_objective"])
        hard60_final = float(hard60_level["fit"]["final_objective"])
        levels.append(
            {
                "level": index + 1,
                "reference_hard30_trace_objective_before_final_update": float(
                    reference_last_hard["objective"]
                ),
                "reference_topk5_final_objective": reference_final,
                "reference_topk5_minus_hard30_trace": (
                    reference_final - float(reference_last_hard["objective"])
                ),
                "hard30_topk_off_final_objective": hard30_final,
                "hard60_topk_off_final_objective": hard60_final,
                "topk5_minus_hard30_off": reference_final - hard30_final,
                "hard60_minus_hard30": hard60_final - hard30_final,
                "hard30_converged": bool(hard30_level["fit"]["hard_converged"]),
                "hard60_converged": bool(hard60_level["fit"]["hard_converged"]),
                "hard30_iterations": int(hard30_level["fit"]["hard_iterations"]),
                "hard60_iterations": int(hard60_level["fit"]["hard_iterations"]),
                "reference_vs_hard30_assignment_change": float(
                    np.mean(reference_assignments[:, index] != hard30_assignments[:, index])
                ),
                "hard30_vs_hard60_assignment_change": float(
                    np.mean(hard30_assignments[:, index] != hard60_assignments[:, index])
                ),
            }
        )
    hard30_all_converged = all(row["hard30_converged"] for row in levels)
    hard60_all_converged = all(row["hard60_converged"] for row in levels)
    topk_worse_all = all(row["topk5_minus_hard30_off"] > 0 for row in levels)
    topk_worse_in_reference_trace_all = all(
        row["reference_topk5_minus_hard30_trace"] > 0 for row in levels
    )
    if hard30_all_converged:
        max_iter_evidence = "MAX_ITER_30_SUFFICIENT"
    elif hard60_all_converged:
        max_iter_evidence = "MAX_ITER_30_INSUFFICIENT_60_REACHES_FROZEN_STOP_RULE"
    else:
        max_iter_evidence = "MAX_ITER_60_STILL_DOES_NOT_REACH_FROZEN_STOP_RULE"
    return {
        "comparison_scope": (
            "S1 is directly paired; S2/S3 are complete-pipeline branch comparisons "
            "because preceding projection residuals differ"
        ),
        "levels": levels,
        "sid": {
            "reference_topk5": reference_metrics["sid"],
            "hard30_topk_off": hard30_metrics["sid"],
            "hard60_topk_off": hard60_metrics["sid"],
            "reference_vs_hard30_path_change": float(
                np.mean(np.any(reference_assignments != hard30_assignments, axis=1))
            ),
            "hard30_vs_hard60_path_change": float(
                np.mean(np.any(hard30_assignments != hard60_assignments, axis=1))
            ),
        },
        "evidence": {
            "topk5_worse_than_hard30_off_all_levels": topk_worse_all,
            "topk5_worse_than_its_own_hard30_trace_all_levels": (
                topk_worse_in_reference_trace_all
            ),
            "hard30_all_levels_converged": hard30_all_converged,
            "hard60_all_levels_converged": hard60_all_converged,
            "max_iter_assessment": max_iter_evidence,
            "canonical_change_applied": False,
        },
    }


def _contract(
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    canonical: BaseCodebookAlgorithmConfig,
    reference: Mapping[str, Any],
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
        "canonical_settings": asdict(canonical),
        "branches": {
            name: asdict(diagnostic_settings(canonical, max_iter=max_iter))
            for name, max_iter in BRANCH_MAX_ITER.items()
        },
        "chunk_rows": chunk_rows,
        "code": _code_files(),
    }


def _branch_artifact_paths(directory: Path, branch: str) -> list[Path]:
    branch_dir = directory / branch
    return [
        branch_dir / "poi_assignments_s1_s2_s3.npy",
        branch_dir / "poi_codebooks_s1_s2_s3.npy",
        branch_dir / "metrics.json",
    ]


def build_base_codebook_diagnostic(
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    canonical: BaseCodebookAlgorithmConfig,
    *,
    reference_manifest: Path,
    reference_manifest_sha256: str,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Build the approved hard30/hard60 Top-k-off diagnostic atomically."""
    if len(inputs.selected_rows) != 100_000:
        raise BaseCodebookDataError("P5 hard 诊断必须使用 medium100k 的 100,000 行")
    final_directory = diagnostic_output_directory(config)
    if final_directory.exists():
        raise BaseCodebookDataError(f"P5 诊断输出已存在且禁止覆盖：{final_directory}")
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
    global_mean, global_metrics = compute_global_mean(
        inputs.embeddings, chunk_rows=chunk_rows
    )
    initial_values = _load_selected_transformed(
        inputs, global_mean, device=device, chunk_rows=chunk_rows
    )
    branch_metrics = {}
    for name, max_iter in BRANCH_MAX_ITER.items():
        print(
            json.dumps(
                {"time": utc_now(), "stage": "p5_diagnostic_branch_started", "branch": name},
                ensure_ascii=False,
            ),
            flush=True,
        )
        branch_metrics[name] = _run_branch(
            initial_values,
            config,
            diagnostic_settings(canonical, max_iter=max_iter),
            work_directory / name,
            chunk_rows=chunk_rows,
        )
    reference_assignments = np.load(
        reference["assignments_path"], allow_pickle=False
    )
    hard30_assignments = np.load(
        work_directory / "hard30_topk_off" / "poi_assignments_s1_s2_s3.npy",
        allow_pickle=False,
    )
    hard60_assignments = np.load(
        work_directory / "hard60_topk_off" / "poi_assignments_s1_s2_s3.npy",
        allow_pickle=False,
    )
    comparison = compare_diagnostic_branches(
        reference["metrics"],
        reference_assignments,
        branch_metrics["hard30_topk_off"],
        hard30_assignments,
        branch_metrics["hard60_topk_off"],
        hard60_assignments,
    )
    write_json_atomic(work_directory / "comparison.json", comparison)
    write_json_atomic(work_directory / "config_resolved.json", config.resolved_payload())
    contract = _contract(config, inputs, canonical, reference, chunk_rows)
    artifact_paths = [work_directory / "comparison.json", work_directory / "config_resolved.json"]
    for branch in BRANCH_MAX_ITER:
        artifact_paths.extend(_branch_artifact_paths(work_directory, branch))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "phase": "P5-CAT-MEDIUM100K-DIAGNOSTIC",
        "built_at": utc_now(),
        "contract": contract,
        "global_component": global_metrics,
        "comparison": comparison,
        "source_access": {
            "active_poi_embedding_values_read": True,
            "reference_p5_medium100k_read": True,
            "raw_train_read": False,
            "business_validation_read": False,
            "business_test_read": False,
            "query_graph_read": False,
            "category_values_read": False,
            "geo_values_read": False,
        },
        "role": "P5_A0_CONTROLLED_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT",
        "artifacts": {
            str(path.relative_to(work_directory)): _artifact_relative(work_directory, path)
            for path in artifact_paths
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "next_status": "HOLD_FOR_P5_MEDIUM100K_DIAGNOSTIC_REVIEW",
    }
    write_json_atomic(work_directory / "manifest.json", manifest)
    write_json_atomic(
        work_directory / "_SUCCESS",
        {"manifest_sha256": sha256_file(work_directory / "manifest.json")},
    )
    os.replace(work_directory, final_directory)
    del initial_values
    gc.collect()
    return validate_base_codebook_diagnostic(
        config,
        inputs,
        canonical,
        reference_manifest=reference_manifest,
        reference_manifest_sha256=reference_manifest_sha256,
        device=device,
        chunk_rows=chunk_rows,
    )


def validate_base_codebook_diagnostic(
    config: DownstreamConfig,
    inputs: BaseCodebookInputs,
    canonical: BaseCodebookAlgorithmConfig,
    *,
    reference_manifest: Path,
    reference_manifest_sha256: str,
    device: str = "cuda",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Independently validate diagnostic sources, endpoints, and comparison."""
    directory = diagnostic_output_directory(config)
    reference = _validate_reference(
        config,
        inputs,
        canonical,
        reference_manifest,
        reference_manifest_sha256,
        verify_artifacts=True,
    )
    manifest_path = directory / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    marker = json.loads((directory / "_SUCCESS").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = _contract(config, inputs, canonical, reference, chunk_rows)
    if (
        marker.get("manifest_sha256") != manifest_sha
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != "P5-CAT-MEDIUM100K-DIAGNOSTIC"
        or manifest.get("role") != "P5_A0_CONTROLLED_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT"
        or manifest.get("contract") != contract
        or manifest.get("next_status") != "HOLD_FOR_P5_MEDIUM100K_DIAGNOSTIC_REVIEW"
    ):
        raise BaseCodebookDataError("P5 hard 诊断 manifest/状态/合同不匹配")
    expected_paths = [directory / "comparison.json", directory / "config_resolved.json"]
    for branch in BRANCH_MAX_ITER:
        expected_paths.extend(_branch_artifact_paths(directory, branch))
    expected_names = {str(path.relative_to(directory)) for path in expected_paths}
    if set(manifest.get("artifacts", {})) != expected_names:
        raise BaseCodebookDataError("P5 hard 诊断 artifact 清单不完整")
    for name, entry in manifest["artifacts"].items():
        path = directory / name
        if (
            entry.get("file") != name
            or not path.is_file()
            or sha256_file(path) != entry.get("sha256")
        ):
            raise BaseCodebookDataError(f"P5 hard 诊断 artifact 损坏：{name}")
    if json.loads((directory / "config_resolved.json").read_text(encoding="utf-8")) != config.resolved_payload():
        raise BaseCodebookDataError("P5 hard 诊断 resolved config 不匹配")
    global_mean, _ = compute_global_mean(inputs.embeddings, chunk_rows=chunk_rows)
    initial_values = _load_selected_transformed(
        inputs, global_mean, device=device, chunk_rows=chunk_rows
    )
    validated_metrics = {}
    for branch in BRANCH_MAX_ITER:
        branch_dir = directory / branch
        assignments_np = np.load(
            branch_dir / "poi_assignments_s1_s2_s3.npy", allow_pickle=False
        )
        codebooks_np = np.load(
            branch_dir / "poi_codebooks_s1_s2_s3.npy", allow_pickle=False
        )
        metrics = json.loads((branch_dir / "metrics.json").read_text(encoding="utf-8"))
        if (
            assignments_np.shape != (100_000, 3)
            or assignments_np.dtype != np.int32
            or codebooks_np.shape != (3, 512, inputs.embedding_dim)
            or codebooks_np.dtype != np.float32
            or metrics.get("settings") != contract["branches"][branch]
        ):
            raise BaseCodebookDataError(f"P5 hard 诊断 {branch} shape/dtype/settings 错误")
        values = initial_values
        for index in range(3):
            torch = _torch()
            centroids = torch.from_numpy(codebooks_np[index]).to(values.device)
            expected_assignments, scores = _hard_assign(
                values, centroids, chunk_rows=chunk_rows
            )
            actual_assignments = torch.from_numpy(
                assignments_np[:, index].astype(np.int64)
            ).to(values.device)
            if not bool(torch.equal(expected_assignments, actual_assignments)):
                raise BaseCodebookDataError(f"P5 hard 诊断 {branch} S{index + 1} assignment 错误")
            level_metrics = metrics["level_metrics"][index]
            objective = float((1.0 - scores).mean().item())
            if not np.isclose(
                objective,
                level_metrics["fit"]["final_objective"],
                rtol=1.0e-7,
                atol=1.0e-8,
            ):
                raise BaseCodebookDataError(f"P5 hard 诊断 {branch} S{index + 1} objective 错误")
            values, residual = projection_residual(values, centroids, actual_assignments)
            for key, actual in residual.items():
                expected = level_metrics["residual"][key]
                if not np.isclose(actual, expected, rtol=1.0e-6, atol=1.0e-7):
                    raise BaseCodebookDataError(
                        f"P5 hard 诊断 {branch} S{index + 1} residual metric 错误：{key}"
                    )
        if metrics["sid"] != sid_metrics(assignments_np, 512):
            raise BaseCodebookDataError(f"P5 hard 诊断 {branch} SID metrics 错误")
        validated_metrics[branch] = metrics
    reference_assignments = np.load(reference["assignments_path"], allow_pickle=False)
    comparison = compare_diagnostic_branches(
        reference["metrics"],
        reference_assignments,
        validated_metrics["hard30_topk_off"],
        np.load(
            directory / "hard30_topk_off" / "poi_assignments_s1_s2_s3.npy",
            allow_pickle=False,
        ),
        validated_metrics["hard60_topk_off"],
        np.load(
            directory / "hard60_topk_off" / "poi_assignments_s1_s2_s3.npy",
            allow_pickle=False,
        ),
    )
    if (
        comparison != json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
        or manifest.get("comparison") != comparison
    ):
        raise BaseCodebookDataError("P5 hard 诊断 comparison 独立复算不一致")
    forbidden_reads = (
        "raw_train_read",
        "business_validation_read",
        "business_test_read",
        "query_graph_read",
        "category_values_read",
        "geo_values_read",
    )
    if any(manifest.get("source_access", {}).get(key) is not False for key in forbidden_reads):
        raise BaseCodebookDataError("P5 hard 诊断非法读取 Query/Category/Geo/Validation/Test")
    del initial_values
    gc.collect()
    return {
        "status": "validated",
        "phase": "P5-CAT-MEDIUM100K-DIAGNOSTIC",
        "output_dir": str(directory),
        "manifest_sha256": manifest_sha,
        "comparison": comparison,
        "next_status": "HOLD_FOR_P5_MEDIUM100K_DIAGNOSTIC_REVIEW",
    }

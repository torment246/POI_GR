"""Full-scale S2-only diagnosis from the frozen S1 endpoint."""

from __future__ import annotations

import copy
import gc
import json
import os
import resource
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.base_quantizer import (
    _distribution_metrics,
    _hard_assign,
    _normalize_torch,
    projection_residual,
)
from qg_prqk.sid.relational_attribution import NEAR_ZERO_QUERY_PRIOR, query_prior_metrics
from qg_prqk.sid.relational_config import RelationalCodebookConfig
from qg_prqk.sid.relational_data import RelationalCodebookDataError, RelationalCodebookInputs, load_relational_codebook_inputs
from qg_prqk.sid.relational_evaluation import evaluation_directory, partition_category_metrics
from qg_prqk.sid.relational_quantizer import (
    _atomic_npy,
    _graph_metrics,
    _level_graph,
    _prefix_metrics,
    _to_device_normalized_float32,
    _transform_query,
    fit_dual_view_level,
    output_directory,
    validate_relational_codebook,
)


SCHEMA_VERSION = "qg-prqk-p6-full-s2-diagnostic-v1"
DIAGNOSTIC_NAME = "full_342879q_716245p_s2_attribution_v1"
BRANCH_ORDER = (
    "canonical",
    "topk_off",
    "graph_off",
    "near_zero_query_prior",
    "graph_off_near_zero_query_prior",
    "category_off",
)


def diagnostic_output_directory(config: RelationalCodebookConfig) -> Path:
    """Return the immutable, non-canonical full S2 diagnostic directory."""
    path = config.p6_output_dir / "diagnostics" / DIAGNOSTIC_NAME
    if not path.resolve().is_relative_to(config.p6_output_dir.resolve()):
        raise RelationalCodebookDataError("P6 S2 diagnostic 输出越出 P6 namespace")
    return path


def build_s2_diagnostic_branches(
    s2: Mapping[str, Any], prqk: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Build fixed one-factor and graph/prior-factorial S2 branches."""

    def branch(description: str) -> dict[str, Any]:
        return {
            "description": description,
            "s2": copy.deepcopy(dict(s2)),
            "prqk": copy.deepcopy(dict(prqk)),
        }

    branches = {
        "canonical": branch("从冻结 P6-S1 endpoint 完整复跑 canonical S2"),
        "topk_off": branch("只关闭 S2 Top-k5 refinement"),
        "graph_off": branch("只把 S2 graph alignment weight 置零"),
        "near_zero_query_prior": branch(
            "只把 S2 Query 质心 POI 先验 tau 从 32 降到 1e-6"
        ),
        "graph_off_near_zero_query_prior": branch(
            "同时关闭 S2 graph weight 并把 Query 质心 POI 先验降到近零"
        ),
        "category_off": branch("只把 S2 category weight 置零"),
    }
    branches["topk_off"]["prqk"]["topk_refinement"]["enabled"] = False
    branches["graph_off"]["s2"]["graph_alignment_weight"] = 0.0
    branches["near_zero_query_prior"]["s2"][
        "query_centroid_shrinkage"
    ] = NEAR_ZERO_QUERY_PRIOR
    branches["graph_off_near_zero_query_prior"]["s2"][
        "graph_alignment_weight"
    ] = 0.0
    branches["graph_off_near_zero_query_prior"]["s2"][
        "query_centroid_shrinkage"
    ] = NEAR_ZERO_QUERY_PRIOR
    branches["category_off"]["s2"]["category_weight"] = 0.0
    return {name: branches[name] for name in BRANCH_ORDER}


def diagnostic_branches(config: RelationalCodebookConfig) -> dict[str, dict[str, Any]]:
    resolved = config.base.resolved_payload()
    return build_s2_diagnostic_branches(resolved["s2"], config.prqk)


def validate_frozen_full_contracts(
    config: RelationalCodebookConfig, *, deep_p6: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind full P6/evaluation manifests, with at most one optional deep P6 scan."""
    canonical_dir = output_directory(config, "full")
    canonical_manifest_path = canonical_dir / "manifest.json"
    canonical_marker = json.loads(
        (canonical_dir / "_SUCCESS").read_text(encoding="utf-8")
    )
    canonical_manifest_sha = sha256_file(canonical_manifest_path)
    if canonical_marker.get("manifest_sha256") != canonical_manifest_sha:
        raise RelationalCodebookDataError("P6 full manifest 与 _SUCCESS 不一致")
    canonical_manifest = json.loads(
        canonical_manifest_path.read_text(encoding="utf-8")
    )
    if (
        canonical_manifest.get("schema_version")
        != "qg-prqk-p6-dual-view-category-prqk-v1"
        or canonical_manifest.get("status") != "completed"
        or canonical_manifest.get("phase") != "P6-CAT-FULL"
        or canonical_manifest.get("role") != "S1_S2_QUERY_CATEGORY_DUAL_VIEW_GATE"
        or canonical_manifest.get("contract", {}).get("config_signature")
        != config.signature()
    ):
        raise RelationalCodebookDataError("P6 full manifest schema/状态/配置签名不匹配")
    canonical = (
        validate_relational_codebook(config, gate="full")
        if deep_p6
        else {
            "manifest": str(canonical_manifest_path),
            "manifest_sha256": canonical_manifest_sha,
            "metrics": canonical_manifest["metrics"],
        }
    )

    baseline_dir = evaluation_directory(config, "full")
    baseline_manifest_path = baseline_dir / "manifest.json"
    baseline_comparison_path = baseline_dir / "comparison.json"
    baseline_marker = json.loads(
        (baseline_dir / "_SUCCESS").read_text(encoding="utf-8")
    )
    baseline_manifest_sha = sha256_file(baseline_manifest_path)
    if baseline_marker.get("manifest_sha256") != baseline_manifest_sha:
        raise RelationalCodebookDataError("P6 full evaluation manifest 与 _SUCCESS 不一致")
    baseline_manifest = json.loads(
        baseline_manifest_path.read_text(encoding="utf-8")
    )
    baseline_comparison = json.loads(
        baseline_comparison_path.read_text(encoding="utf-8")
    )
    baseline_contract = baseline_comparison.get("contract", {})
    if (
        baseline_manifest.get("schema_version")
        != "qg-prqk-p6-full-evaluation-v1"
        or baseline_manifest.get("status") != "completed"
        or baseline_manifest.get("phase") != "P6-CAT-FULL-EVALUATION"
        or baseline_manifest.get("comparison", {}).get("sha256")
        != sha256_file(baseline_comparison_path)
        or baseline_manifest.get("contract") != baseline_contract
        or baseline_manifest.get("gate_evidence")
        != baseline_comparison.get("gate_evidence")
        or baseline_contract.get("config_signature") != config.signature()
        or baseline_contract.get("p6_manifest_sha256") != canonical_manifest_sha
        or baseline_contract.get("p5_full_manifest_sha256")
        != config.frozen_inputs["p5_full_manifest"]["sha256"]
    ):
        raise RelationalCodebookDataError("P6 full evaluation schema/来源/结果合同不匹配")
    if any(
        baseline_comparison.get("source_access", {}).get(name) is not False
        for name in ("business_validation_read", "business_test_read", "geo_read")
    ):
        raise RelationalCodebookDataError("P6 full evaluation 禁止数据读取标志错误")
    baseline = {
        "manifest": str(baseline_manifest_path),
        "manifest_sha256": baseline_manifest_sha,
        "comparison_sha256": sha256_file(baseline_comparison_path),
        "comparison": baseline_comparison,
    }
    return canonical, baseline


def additive_s2_decomposition(
    *,
    p5_path: float,
    p5_s2_on_p6_s1: float,
    p6_s2_content_optimal: float,
    p6_s2_coupled: float,
) -> dict[str, Any]:
    """Split the P5-to-P6 S2 delta into three exactly additive effects."""
    points = {
        "a_p5_s1_residual_p5_s2_content": float(p5_path),
        "b_p6_s1_residual_p5_s2_content": float(p5_s2_on_p6_s1),
        "c_p6_s1_residual_p6_s2_content_optimal": float(
            p6_s2_content_optimal
        ),
        "d_p6_s1_residual_p6_s2_coupled_assignment": float(p6_s2_coupled),
    }
    effects = {
        "upstream_s1_residual_shift_b_minus_a": float(
            p5_s2_on_p6_s1 - p5_path
        ),
        "s2_query_centroid_adaptation_c_minus_b": float(
            p6_s2_content_optimal - p5_s2_on_p6_s1
        ),
        "s2_coupled_assignment_tax_d_minus_c": float(
            p6_s2_coupled - p6_s2_content_optimal
        ),
        "total_d_minus_a": float(p6_s2_coupled - p5_path),
    }
    effects["component_sum"] = float(
        effects["upstream_s1_residual_shift_b_minus_a"]
        + effects["s2_query_centroid_adaptation_c_minus_b"]
        + effects["s2_coupled_assignment_tax_d_minus_c"]
    )
    return {"points": points, "effects": effects}


def s2_factor_contrasts(
    branches: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute canonical-minus-ablation S2 effects and graph/prior interaction."""
    if tuple(branches) != BRANCH_ORDER:
        raise RelationalCodebookDataError("P6 S2 diagnostic 分支集合或顺序不匹配")

    pairs = {
        "topk5_at_canonical": ("canonical", "topk_off"),
        "graph_at_tau32_topk5": ("canonical", "graph_off"),
        "query_prior_tau32_at_graph_topk5": (
            "canonical",
            "near_zero_query_prior",
        ),
        "category_at_canonical": ("canonical", "category_off"),
    }
    result: dict[str, Any] = {}
    for name, (enabled_name, disabled_name) in pairs.items():
        enabled = branches[enabled_name]
        disabled = branches[disabled_name]
        result[name] = {
            "with_factor": enabled_name,
            "without_factor": disabled_name,
            "query_distortion_effect": enabled["query_cosine_distortion"]
            - disabled["query_cosine_distortion"],
            "graph_agreement_effect": enabled["weighted_graph_agreement"]
            - disabled["weighted_graph_agreement"],
            "fine_path_purity_effect": enabled["fine_path_purity"]
            - disabled["fine_path_purity"],
            "distinct_prefix_effect": enabled["distinct_prefix"]
            - disabled["distinct_prefix"],
        }
    canonical = branches["canonical"]
    graph_off = branches["graph_off"]
    prior_off = branches["near_zero_query_prior"]
    both_off = branches["graph_off_near_zero_query_prior"]
    result["graph_x_query_prior_interaction"] = {
        "query_distortion_interaction": canonical["query_cosine_distortion"]
        - graph_off["query_cosine_distortion"]
        - prior_off["query_cosine_distortion"]
        + both_off["query_cosine_distortion"],
        "graph_agreement_interaction": canonical["weighted_graph_agreement"]
        - graph_off["weighted_graph_agreement"]
        - prior_off["weighted_graph_agreement"]
        + both_off["weighted_graph_agreement"],
    }
    return result


def _content_distortion(values, centroids, assignments) -> float:
    normalized = _normalize_torch(centroids)[assignments]
    return float((1.0 - (values * normalized).sum(dim=1)).mean().item())


def _reconstruct_frozen_s1(
    inputs: RelationalCodebookInputs,
    canonical_dir: Path,
    *,
    device,
    chunk_rows: int,
) -> dict[str, Any]:
    import torch

    poi_s0 = _to_device_normalized_float32(
        inputs.poi_residual_s0, device=device, chunk_rows=chunk_rows
    )
    query_s0 = _transform_query(
        inputs.query_embeddings,
        inputs.global_mean,
        device=device,
        chunk_rows=chunk_rows,
    )
    p6_poi_s1 = torch.from_numpy(
        np.load(canonical_dir / "poi_assignments_s1.npy", allow_pickle=False).astype(
            np.int64
        )
    ).to(device)
    p6_query_s1_np = np.load(
        canonical_dir / "query_assignments_s1.npy", allow_pickle=False
    )
    if np.any(p6_query_s1_np < 0):
        raise RelationalCodebookDataError("P6 full S1 Query assignment 不得包含 inactive 行")
    p6_query_s1 = torch.from_numpy(p6_query_s1_np.astype(np.int64)).to(device)
    p6_poi_codebook_s1 = torch.from_numpy(
        np.load(canonical_dir / "poi_codebook_s1.npy", allow_pickle=False)
    ).to(device)
    p6_query_codebook_s1 = torch.from_numpy(
        np.load(canonical_dir / "query_codebook_s1.npy", allow_pickle=False)
    ).to(device)
    poi_after_s1, poi_residual_metrics = projection_residual(
        poi_s0, p6_poi_codebook_s1, p6_poi_s1
    )
    query_after_s1, query_residual_metrics = projection_residual(
        query_s0, p6_query_codebook_s1, p6_query_s1
    )
    return {
        "poi_s0": poi_s0,
        "query_s0": query_s0,
        "poi_after_s1": poi_after_s1,
        "query_after_s1": query_after_s1,
        "poi_assignment_s1_np": p6_poi_s1.detach().cpu().numpy().astype(np.int32),
        "query_assignment_s1_np": p6_query_s1_np.astype(np.int32, copy=False),
        "poi_residual_metrics": poi_residual_metrics,
        "query_residual_metrics": query_residual_metrics,
    }


def _compute_static_decomposition(
    inputs: RelationalCodebookInputs,
    frozen: Mapping[str, Any],
    canonical_dir: Path,
    *,
    active_mask: np.ndarray,
    device,
    chunk_rows: int,
) -> dict[str, Any]:
    import torch

    active_tensor = torch.from_numpy(active_mask).to(device)
    p5_s1 = torch.from_numpy(inputs.initial_poi_codebooks[0]).to(device)
    p5_s2 = torch.from_numpy(inputs.initial_poi_codebooks[1]).to(device)
    p5_s1_assignment, _ = _hard_assign(
        frozen["query_s0"], p5_s1, chunk_rows=chunk_rows
    )
    p5_after_s1, _ = projection_residual(
        frozen["query_s0"], p5_s1, p5_s1_assignment
    )
    p5_active = p5_after_s1[active_tensor]
    p6_active = frozen["query_after_s1"][active_tensor]
    _, p5_path_scores = _hard_assign(p5_active, p5_s2, chunk_rows=chunk_rows)
    _, cross_scores = _hard_assign(p6_active, p5_s2, chunk_rows=chunk_rows)

    p6_s2_codebook = torch.from_numpy(
        np.load(canonical_dir / "query_codebook_s2.npy", allow_pickle=False)
    ).to(device)
    _, p6_optimal_scores = _hard_assign(
        p6_active, p6_s2_codebook, chunk_rows=chunk_rows
    )
    p6_query_s2_np = np.load(
        canonical_dir / "query_assignments_s2.npy", allow_pickle=False
    )
    if np.any(p6_query_s2_np[~active_mask] != -1):
        raise RelationalCodebookDataError("P6 full S2 inactive Query assignment 必须为 -1")
    p6_query_s2 = torch.from_numpy(
        p6_query_s2_np[active_mask].astype(np.int64)
    ).to(device)
    p6_coupled = _content_distortion(p6_active, p6_s2_codebook, p6_query_s2)
    decomposition = additive_s2_decomposition(
        p5_path=float((1.0 - p5_path_scores).mean().item()),
        p5_s2_on_p6_s1=float((1.0 - cross_scores).mean().item()),
        p6_s2_content_optimal=float((1.0 - p6_optimal_scores).mean().item()),
        p6_s2_coupled=p6_coupled,
    )
    residual_cosine = (p5_active * p6_active).sum(dim=1)
    decomposition["s1_residual_geometry"] = {
        "p5_vs_p6_cosine_mean": float(residual_cosine.mean().item()),
        "p5_vs_p6_cosine_p10": float(torch.quantile(residual_cosine, 0.1).item()),
        "p5_vs_p6_cosine_p50": float(torch.quantile(residual_cosine, 0.5).item()),
        "p5_vs_p6_cosine_p90": float(torch.quantile(residual_cosine, 0.9).item()),
        "p5_vs_p6_s1_assignment_change_rate": float(
            (
                p5_s1_assignment.detach().cpu().numpy()
                != frozen["query_assignment_s1_np"]
            ).mean()
        ),
    }
    return decomposition


def _run_s2_branch(
    name: str,
    branch: Mapping[str, Any],
    inputs: RelationalCodebookInputs,
    frozen: Mapping[str, Any],
    graph,
    active_mask: np.ndarray,
    output: Path,
    *,
    device,
    chunk_rows: int,
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    active_query_values = frozen["query_after_s1"][
        torch.from_numpy(active_mask).to(device)
    ]
    initial_codebook = torch.from_numpy(inputs.initial_poi_codebooks[1]).to(device)
    settings = branch["s2"]
    u, v, poi_assignment, query_assignment, fit = fit_dual_view_level(
        frozen["poi_after_s1"],
        active_query_values,
        initial_codebook,
        inputs.initial_poi_assignments[1],
        graph,
        inputs.fine_category_indices,
        level=2,
        codebook_size=512,
        settings=settings,
        prqk=branch["prqk"],
        warmup={"zero_weight_iters": 2, "linear_ramp_end_iter": 6},
        parent_assignments=frozen["poi_assignment_s1_np"],
        fine_to_coarse=inputs.fine_to_coarse,
        chunk_rows=chunk_rows,
    )
    poi_np = poi_assignment.detach().cpu().numpy().astype(np.int32)
    active_query_np = query_assignment.detach().cpu().numpy().astype(np.int32)
    query_np = np.full(len(active_mask), -1, dtype=np.int32)
    query_np[active_mask] = active_query_np
    hard_rows = [
        row for row in fit["trace"] if row["stage"] == "hard_alternating"
    ]
    hard_distortion = hard_rows[-1]["components"]["query_content_sum"] / len(
        active_query_np
    )
    poi_prefix = np.column_stack(
        [frozen["poi_assignment_s1_np"], poi_np]
    ).astype(np.int32)
    path_category = partition_category_metrics(
        poi_prefix, inputs.fine_category_indices
    )
    prefix = _prefix_metrics(poi_prefix)
    branch_dir = output / "branches" / name
    branch_dir.mkdir(parents=True)
    _atomic_npy(branch_dir / "poi_codebook_s2.npy", u.cpu().numpy().astype(np.float32))
    _atomic_npy(
        branch_dir / "query_codebook_s2.npy", v.cpu().numpy().astype(np.float32)
    )
    _atomic_npy(branch_dir / "poi_assignments_s2.npy", poi_np)
    _atomic_npy(branch_dir / "query_assignments_s2.npy", query_np)
    metrics = {
        "status": "completed",
        "branch": name,
        "description": branch["description"],
        "settings": {"s2": branch["s2"], "prqk": branch["prqk"]},
        "fit": fit,
        "hard_endpoint_query_content_mean": hard_distortion,
        "topk_query_distortion_delta": fit["final_query_content_mean"]
        - hard_distortion,
        "poi_assignment": _distribution_metrics(poi_np, 512),
        "query_assignment": _distribution_metrics(active_query_np, 512),
        "query_prior": query_prior_metrics(
            active_query_np,
            codebook_size=512,
            tau=float(settings["query_centroid_shrinkage"]),
        ),
        "graph": _graph_metrics(graph, poi_np, active_query_np),
        "fine_path_category": path_category,
        "poi_prefix": prefix,
        "reference_poi_assignment_change": float(
            np.mean(poi_np != inputs.initial_poi_assignments[1])
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(branch_dir / "metrics.json", metrics)
    del u, v, poi_assignment, query_assignment, active_query_values
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def _compact_branch(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_cosine_distortion": metrics["fit"]["final_query_content_mean"],
        "hard_endpoint_query_cosine_distortion": metrics[
            "hard_endpoint_query_content_mean"
        ],
        "topk_query_distortion_delta": metrics["topk_query_distortion_delta"],
        "poi_cosine_distortion": metrics["fit"]["final_poi_content_mean"],
        "weighted_graph_agreement": metrics["graph"]["weighted_agreement"],
        "fine_path_purity": metrics["fine_path_category"]["purity"],
        "fine_path_conditional_entropy": metrics["fine_path_category"][
            "conditional_entropy"
        ],
        "poi_active_codes": metrics["poi_assignment"]["active_codes"],
        "query_active_codes": metrics["query_assignment"]["active_codes"],
        "poi_prior_mass_query_weighted_mean": metrics["query_prior"][
            "poi_prior_mass_query_weighted_mean"
        ],
        "distinct_prefix": metrics["poi_prefix"]["distinct_prefix"],
        "collision_excess": metrics["poi_prefix"]["collision_excess"],
        "bucket_p99": metrics["poi_prefix"]["bucket_p99"],
        "bucket_max": metrics["poi_prefix"]["bucket_max"],
        "hard_iterations": metrics["fit"]["hard_iterations"],
        "hard_converged": metrics["fit"]["hard_converged"],
        "soft_refinement_iterations": metrics["fit"][
            "soft_refinement_iterations"
        ],
        "runtime_seconds": metrics["runtime_seconds"],
    }


def _canonical_parity(
    canonical: Mapping[str, Any], canonical_dir: Path, output: Path
) -> dict[str, Any]:
    branch_dir = output / "branches" / "canonical"
    reference_metrics = json.loads(
        (canonical_dir / "level_s2_metrics.json").read_text(encoding="utf-8")
    )
    reference_poi = np.load(
        canonical_dir / "poi_assignments_s2.npy", allow_pickle=False
    )
    reference_query = np.load(
        canonical_dir / "query_assignments_s2.npy", allow_pickle=False
    )
    actual_poi = np.load(branch_dir / "poi_assignments_s2.npy", allow_pickle=False)
    actual_query = np.load(
        branch_dir / "query_assignments_s2.npy", allow_pickle=False
    )
    valid = reference_query >= 0
    poi_match = float(np.mean(reference_poi == actual_poi))
    query_match = float(np.mean(reference_query[valid] == actual_query[valid]))
    distortion_delta = abs(
        canonical["fit"]["final_query_content_mean"]
        - reference_metrics["fit"]["final_query_content_mean"]
    )
    graph_delta = abs(
        canonical["graph"]["weighted_agreement"]
        - reference_metrics["graph"]["weighted_agreement"]
    )
    passed = (
        poi_match >= 0.999
        and query_match >= 0.999
        and distortion_delta <= 1.0e-5
        and graph_delta <= 1.0e-9
    )
    return {
        "passed": passed,
        "poi_assignment_match_rate": poi_match,
        "query_assignment_match_rate": query_match,
        "query_distortion_abs_delta": distortion_delta,
        "graph_agreement_abs_delta": graph_delta,
    }


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "file": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        entry.update(shape=list(values.shape), dtype=str(values.dtype))
    return entry


def _code_hashes() -> dict[str, str]:
    package = Path(__file__).parent
    root = package.parent
    paths = [
        package / "base_quantizer.py",
        package / "relational_config.py",
        package / "relational_data.py",
        package / "relational_quantizer.py",
        package / "relational_evaluation.py",
        package / "relational_attribution.py",
        package / "relational_s2_diagnostics.py",
        root / "commands" / "diagnose_relational_s2.py",
    ]
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def build_relational_s2_diagnostic(
    config: RelationalCodebookConfig,
    *,
    device: str = "cuda",
    chunk_rows: int = 8192,
) -> dict[str, Any]:
    """Run immutable full S2-only ablations from the frozen P6 S1 endpoint."""
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise RelationalCodebookDataError("P6 full S2 diagnostic 要求 CUDA；不得静默回退 CPU")
    canonical_validation, baseline_validation = validate_frozen_full_contracts(
        config, deep_p6=True
    )
    inputs = load_relational_codebook_inputs(config, gate="full")
    canonical_dir = output_directory(config, "full")
    final = diagnostic_output_directory(config)
    if final.exists():
        raise RelationalCodebookDataError(f"P6 full S2 diagnostic overwrite=false：{final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(f".{final.name}.{uuid.uuid4().hex}.building")
    staging.mkdir()
    started = time.perf_counter()
    target = torch.device(device)
    torch.manual_seed(int(config.prqk["seed"]))
    torch.cuda.manual_seed_all(int(config.prqk["seed"]))
    torch.cuda.reset_peak_memory_stats()
    try:
        frozen = _reconstruct_frozen_s1(
            inputs,
            canonical_dir,
            device=target,
            chunk_rows=chunk_rows,
        )
        depths = inputs.selection.nodes["supervision_depth"].to_numpy(
            zero_copy_only=False
        )
        active_mask = depths >= 2
        active_rows = inputs.selection.selected_query_rows[active_mask]
        graph = _level_graph(inputs, 2, active_rows)
        decomposition = _compute_static_decomposition(
            inputs,
            frozen,
            canonical_dir,
            active_mask=active_mask,
            device=target,
            chunk_rows=chunk_rows,
        )
        frozen.pop("poi_s0")
        frozen.pop("query_s0")
        gc.collect()
        torch.cuda.empty_cache()
        branch_metrics: dict[str, Any] = {}
        for name, branch in diagnostic_branches(config).items():
            print(
                json.dumps(
                    {
                        "time": utc_now(),
                        "stage": "p6_full_s2_diagnostic_branch_started",
                        "branch": name,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            branch_metrics[name] = _run_s2_branch(
                name,
                branch,
                inputs,
                frozen,
                graph,
                active_mask,
                staging,
                device=target,
                chunk_rows=chunk_rows,
            )
            print(
                json.dumps(
                    {
                        "time": utc_now(),
                        "stage": "p6_full_s2_diagnostic_branch_completed",
                        "branch": name,
                        "runtime_seconds": branch_metrics[name]["runtime_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        compact = {name: _compact_branch(branch_metrics[name]) for name in BRANCH_ORDER}
        parity = _canonical_parity(branch_metrics["canonical"], canonical_dir, staging)
        if not parity["passed"]:
            raise RelationalCodebookDataError("P6 full S2 canonical 复跑未通过正式 checkpoint parity")
        formal_levels = baseline_validation["comparison"]["levels"]
        decomposition["formal_metric_parity"] = {
            "p5_a0_s2_abs_delta": abs(
                decomposition["points"]["a_p5_s1_residual_p5_s2_content"]
                - formal_levels[1]["query_cosine_distortion"]["p5_a0"]
            ),
            "p6_s2_abs_delta": abs(
                decomposition["points"][
                    "d_p6_s1_residual_p6_s2_coupled_assignment"
                ]
                - formal_levels[1]["query_cosine_distortion"]["p6"]
            ),
        }
        comparison = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "phase": "P6-CAT-FULL-S2-DIAGNOSTIC",
            "built_at": utc_now(),
            "contract": {
                "config_signature": config.signature(),
                "canonical_p6_manifest_sha256": canonical_validation[
                    "manifest_sha256"
                ],
                "canonical_evaluation_manifest_sha256": baseline_validation[
                    "manifest_sha256"
                ],
                "scale": "full",
                "poi_rows": len(inputs.selection.selected_poi_rows),
                "query_rows": len(inputs.selection.selected_query_rows),
                "active_s2_query_rows": int(active_mask.sum()),
                "s2_edge_rows": len(graph.weights),
                "frozen_s1_endpoint": True,
                "s1_optimization_rerun": False,
                "same_selected_poi_rows": True,
                "same_selected_query_rows": True,
                "same_complete_s2_edges": True,
                "branch_order": list(BRANCH_ORDER),
                "near_zero_query_prior": NEAR_ZERO_QUERY_PRIOR,
            },
            "formal_p5_p6_s2": formal_levels[1],
            "static_additive_decomposition": decomposition,
            "canonical_parity": parity,
            "branches": compact,
            "factor_contrasts": s2_factor_contrasts(compact),
            "source_access": {
                "business_validation_read": False,
                "business_test_read": False,
                "geo_read": False,
            },
            "next_status": "HOLD_FOR_P6_S2_DIAGNOSTIC_REVIEW",
        }
        write_json_atomic(staging / "comparison.json", comparison)
        artifacts = {
            path.relative_to(staging).as_posix(): _artifact(path, staging)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "phase": comparison["phase"],
            "role": "CONTROLLED_S2_FULL_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT",
            "built_at": utc_now(),
            "contract": comparison["contract"],
            "source_hashes": dict(inputs.source_hashes),
            "code": _code_hashes(),
            "artifacts": artifacts,
            "source_access": comparison["source_access"],
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "device": device,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated()
                / (1024 * 1024),
            },
            "next_status": comparison["next_status"],
        }
        write_json_atomic(staging / "manifest.json", manifest)
        write_json_atomic(
            staging / "_SUCCESS",
            {"manifest_sha256": sha256_file(staging / "manifest.json")},
        )
        os.replace(staging, final)
    except Exception:
        if staging.exists():
            failed = final.with_name(f"{final.name}.failed_{uuid.uuid4().hex[:8]}")
            os.replace(staging, failed)
        raise
    return validate_relational_s2_diagnostic(config, device=device)


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise RelationalCodebookDataError(f"P6 full S2 diagnostic 非有限指标：{label}")


def validate_relational_s2_diagnostic(
    config: RelationalCodebookConfig, *, device: str = "cuda"
) -> dict[str, Any]:
    """Validate S2 diagnostic provenance, artifacts, and recomputed metrics."""
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise RelationalCodebookDataError("P6 full S2 diagnostic validator 要求 CUDA")
    canonical_validation, baseline_validation = validate_frozen_full_contracts(
        config, deep_p6=True
    )
    inputs = load_relational_codebook_inputs(config, gate="full")
    directory = diagnostic_output_directory(config)
    canonical_dir = output_directory(config, "full")
    manifest_path = directory / "manifest.json"
    marker = json.loads((directory / "_SUCCESS").read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != manifest_sha:
        raise RelationalCodebookDataError("P6 full S2 diagnostic manifest 与 _SUCCESS 不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    contract = comparison.get("contract", {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != "P6-CAT-FULL-S2-DIAGNOSTIC"
        or manifest.get("role")
        != "CONTROLLED_S2_FULL_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT"
        or manifest.get("contract") != contract
        or contract.get("config_signature") != config.signature()
        or contract.get("canonical_p6_manifest_sha256")
        != canonical_validation["manifest_sha256"]
        or contract.get("canonical_evaluation_manifest_sha256")
        != baseline_validation["manifest_sha256"]
        or contract.get("branch_order") != list(BRANCH_ORDER)
        or contract.get("frozen_s1_endpoint") is not True
        or contract.get("s1_optimization_rerun") is not False
        or comparison.get("next_status")
        != "HOLD_FOR_P6_S2_DIAGNOSTIC_REVIEW"
    ):
        raise RelationalCodebookDataError("P6 full S2 diagnostic schema/来源/状态合同不匹配")
    expected_artifacts = {"comparison.json"}
    for name in BRANCH_ORDER:
        expected_artifacts.update(
            {
                f"branches/{name}/metrics.json",
                f"branches/{name}/poi_codebook_s2.npy",
                f"branches/{name}/query_codebook_s2.npy",
                f"branches/{name}/poi_assignments_s2.npy",
                f"branches/{name}/query_assignments_s2.npy",
            }
        )
    if set(manifest.get("artifacts", {})) != expected_artifacts:
        raise RelationalCodebookDataError("P6 full S2 diagnostic artifact 集合不完整")
    for relative, entry in manifest["artifacts"].items():
        path = directory / relative
        if entry.get("file") != relative or sha256_file(path) != entry.get("sha256"):
            raise RelationalCodebookDataError(f"P6 full S2 diagnostic artifact 损坏：{relative}")

    target = torch.device(device)
    frozen = _reconstruct_frozen_s1(
        inputs, canonical_dir, device=target, chunk_rows=8192
    )
    depths = inputs.selection.nodes["supervision_depth"].to_numpy(
        zero_copy_only=False
    )
    active_mask = depths >= 2
    active_tensor = torch.from_numpy(active_mask).to(target)
    active_query_values = frozen["query_after_s1"][active_tensor]
    graph = _level_graph(
        inputs, 2, inputs.selection.selected_query_rows[active_mask]
    )
    decomposition = _compute_static_decomposition(
        inputs,
        frozen,
        canonical_dir,
        active_mask=active_mask,
        device=target,
        chunk_rows=8192,
    )
    frozen.pop("poi_s0")
    frozen.pop("query_s0")
    gc.collect()
    torch.cuda.empty_cache()
    compact: dict[str, Any] = {}
    for name in BRANCH_ORDER:
        branch_dir = directory / "branches" / name
        recorded = json.loads((branch_dir / "metrics.json").read_text(encoding="utf-8"))
        poi_np = np.load(branch_dir / "poi_assignments_s2.npy", allow_pickle=False)
        query_np = np.load(branch_dir / "query_assignments_s2.npy", allow_pickle=False)
        u_np = np.load(branch_dir / "poi_codebook_s2.npy", allow_pickle=False)
        v_np = np.load(branch_dir / "query_codebook_s2.npy", allow_pickle=False)
        if (
            poi_np.shape != (len(inputs.selection.selected_poi_rows),)
            or poi_np.dtype != np.int32
            or query_np.shape != (len(inputs.selection.selected_query_rows),)
            or query_np.dtype != np.int32
            or u_np.shape != (512, inputs.embedding_dim)
            or u_np.dtype != np.float32
            or v_np.shape != (512, inputs.embedding_dim)
            or v_np.dtype != np.float32
            or np.any(poi_np < 0)
            or np.any(poi_np >= 512)
            or np.any(query_np[active_mask] < 0)
            or np.any(query_np[active_mask] >= 512)
            or np.any(query_np[~active_mask] != -1)
            or not np.isfinite(u_np).all()
            or not np.isfinite(v_np).all()
        ):
            raise RelationalCodebookDataError(f"P6 full S2 diagnostic {name} artifact shape/range 错误")
        poi_assignment = torch.from_numpy(poi_np.astype(np.int64)).to(target)
        query_assignment = torch.from_numpy(
            query_np[active_mask].astype(np.int64)
        ).to(target)
        u = torch.from_numpy(u_np).to(target)
        v = torch.from_numpy(v_np).to(target)
        poi_distortion = _content_distortion(
            frozen["poi_after_s1"], u, poi_assignment
        )
        query_distortion = _content_distortion(
            active_query_values, v, query_assignment
        )
        if not np.isclose(
            poi_distortion,
            recorded["fit"]["final_poi_content_mean"],
            rtol=1.0e-5,
            atol=1.0e-6,
        ) or not np.isclose(
            query_distortion,
            recorded["fit"]["final_query_content_mean"],
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise RelationalCodebookDataError(f"P6 full S2 diagnostic {name} distortion 不一致")
        graph_metrics = _graph_metrics(graph, poi_np, query_np[active_mask])
        if not np.isclose(
            graph_metrics["weighted_agreement"],
            recorded["graph"]["weighted_agreement"],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RelationalCodebookDataError(f"P6 full S2 diagnostic {name} 图指标不一致")
        prefix = np.column_stack([frozen["poi_assignment_s1_np"], poi_np])
        if (
            _prefix_metrics(prefix) != recorded["poi_prefix"]
            or partition_category_metrics(prefix, inputs.fine_category_indices)
            != recorded["fine_path_category"]
        ):
            raise RelationalCodebookDataError(f"P6 full S2 diagnostic {name} path 指标不一致")
        _assert_finite(recorded, f"branches.{name}")
        compact[name] = _compact_branch(recorded)
    parity = _canonical_parity(
        json.loads(
            (directory / "branches/canonical/metrics.json").read_text(
                encoding="utf-8"
            )
        ),
        canonical_dir,
        directory,
    )
    recorded_decomposition = comparison["static_additive_decomposition"]
    if (
        compact != comparison.get("branches")
        or s2_factor_contrasts(compact) != comparison.get("factor_contrasts")
        or not parity["passed"]
        or parity != comparison.get("canonical_parity")
        or decomposition["points"] != recorded_decomposition.get("points")
        or decomposition["effects"] != recorded_decomposition.get("effects")
        or decomposition["s1_residual_geometry"]
        != recorded_decomposition.get("s1_residual_geometry")
    ):
        raise RelationalCodebookDataError("P6 full S2 diagnostic comparison/parity 不一致")
    if any(
        comparison.get("source_access", {}).get(key) is not False
        for key in ("business_validation_read", "business_test_read", "geo_read")
    ):
        raise RelationalCodebookDataError("P6 full S2 diagnostic 禁止数据读取标志错误")
    return {
        "status": "p6_full_s2_diagnostic_validated",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "comparison_sha256": sha256_file(directory / "comparison.json"),
        "comparison": comparison,
    }

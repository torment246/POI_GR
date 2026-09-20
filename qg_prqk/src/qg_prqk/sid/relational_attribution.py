"""Controlled attribution for relational S1/S2 refinement."""

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
from qg_prqk.sid.base_quantizer import _distribution_metrics, _normalize_torch, projection_residual
from qg_prqk.sid.relational_config import RelationalCodebookConfig
from qg_prqk.sid.relational_data import RelationalCodebookDataError, RelationalCodebookInputs, load_relational_codebook_inputs
from qg_prqk.sid.relational_evaluation import validate_relational_evaluation
from qg_prqk.sid.relational_quantizer import (
    _atomic_npy,
    _category_metrics,
    _graph_metrics,
    _level_graph,
    _prefix_metrics,
    _transform_query,
    fit_dual_view_level,
    output_directory,
    validate_relational_codebook,
)


SCHEMA_VERSION = "qg-prqk-p6-sample-attribution-v1"
DIAGNOSTIC_NAME = "sample_001000q_011459p_attribution_v1"
NEAR_ZERO_QUERY_PRIOR = 1.0e-6
BRANCH_ORDER = (
    "canonical",
    "topk_off",
    "graph_off",
    "near_zero_query_prior",
    "graph_off_near_zero_query_prior",
    "category_off",
)


def diagnostic_output_directory(config: RelationalCodebookConfig) -> Path:
    """Return the isolated, non-canonical P6 attribution directory."""
    path = config.p6_output_dir / "diagnostics" / DIAGNOSTIC_NAME
    if not path.resolve().is_relative_to(config.p6_output_dir.resolve()):
        raise RelationalCodebookDataError("P6 attribution 输出越出 P6 namespace")
    return path


def build_diagnostic_branches(
    s1: Mapping[str, Any],
    s2: Mapping[str, Any],
    prqk: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build fixed one-factor and graph/prior-factorial diagnostic branches."""

    def branch(description: str) -> dict[str, Any]:
        return {
            "description": description,
            "s1": copy.deepcopy(dict(s1)),
            "s2": copy.deepcopy(dict(s2)),
            "prqk": copy.deepcopy(dict(prqk)),
        }

    branches = {
        "canonical": branch("完整复跑 canonical，验证诊断入口与正式 P6 一致"),
        "topk_off": branch("只关闭 Top-k5，定位软质心 refinement 的边际影响"),
        "graph_off": branch("只把 S1/S2 graph alignment weight 置零"),
        "near_zero_query_prior": branch(
            "只把 Query 质心 POI 先验 tau 从 32 降到 1e-6；空码仍回退 POI 质心"
        ),
        "graph_off_near_zero_query_prior": branch(
            "同时关闭 graph weight 并把 Query 质心 POI 先验降到近零"
        ),
        "category_off": branch("只把 S1/S2 category weight 置零"),
    }
    branches["topk_off"]["prqk"]["topk_refinement"]["enabled"] = False
    for level in ("s1", "s2"):
        branches["graph_off"][level]["graph_alignment_weight"] = 0.0
        branches["near_zero_query_prior"][level][
            "query_centroid_shrinkage"
        ] = NEAR_ZERO_QUERY_PRIOR
        branches["graph_off_near_zero_query_prior"][level][
            "graph_alignment_weight"
        ] = 0.0
        branches["graph_off_near_zero_query_prior"][level][
            "query_centroid_shrinkage"
        ] = NEAR_ZERO_QUERY_PRIOR
        branches["category_off"][level]["category_weight"] = 0.0
    return {name: branches[name] for name in BRANCH_ORDER}


def diagnostic_branches(config: RelationalCodebookConfig) -> dict[str, dict[str, Any]]:
    resolved = config.base.resolved_payload()
    return build_diagnostic_branches(
        resolved["s1"], resolved["s2"], config.prqk
    )


def query_prior_metrics(
    assignments: np.ndarray, *, codebook_size: int, tau: float
) -> dict[str, Any]:
    """Measure the effective POI-prior mass in occupied Query centroids."""
    labels = np.asarray(assignments, dtype=np.int64)
    if labels.ndim != 1 or not len(labels) or tau < 0:
        raise RelationalCodebookDataError("P6 attribution Query assignment/tau 非法")
    counts = np.bincount(labels, minlength=codebook_size).astype(np.float64)
    active = counts[counts > 0]
    if len(active) == 0:
        raise RelationalCodebookDataError("P6 attribution Query assignment 不得为空")
    prior_mass = tau / (active + tau)
    query_weights = active / active.sum()
    return {
        "tau": float(tau),
        "active_codes": int(len(active)),
        "queries_per_active_code_mean": float(active.mean()),
        "queries_per_active_code_p50": float(np.quantile(active, 0.5)),
        "poi_prior_mass_code_p50": float(np.quantile(prior_mass, 0.5)),
        "poi_prior_mass_query_weighted_mean": float(
            np.sum(query_weights * prior_mass)
        ),
        "query_data_mass_query_weighted_mean": float(
            np.sum(query_weights * (1.0 - prior_mass))
        ),
    }


def _compact_branch(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "levels": [
            {
                "level": item["level"],
                "query_cosine_distortion": item["fit"][
                    "final_query_content_mean"
                ],
                "hard_endpoint_query_cosine_distortion": item[
                    "hard_endpoint_query_content_mean"
                ],
                "topk_query_distortion_delta": item[
                    "topk_query_distortion_delta"
                ],
                "poi_cosine_distortion": item["fit"]["final_poi_content_mean"],
                "weighted_graph_agreement": item["graph"]["weighted_agreement"],
                "category_purity": item["category"]["purity"],
                "poi_active_codes": item["poi_assignment"]["active_codes"],
                "query_active_codes": item["query_assignment"]["active_codes"],
                "poi_prior_mass_query_weighted_mean": item["query_prior"][
                    "poi_prior_mass_query_weighted_mean"
                ],
                "hard_iterations": item["fit"]["hard_iterations"],
                "hard_converged": item["fit"]["hard_converged"],
                "soft_refinement_iterations": item["fit"][
                    "soft_refinement_iterations"
                ],
            }
            for item in metrics["levels"]
        ],
        "poi_prefix": metrics["poi_prefix"],
        "runtime_seconds": metrics["runtime_seconds"],
    }


def factor_contrasts(branches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compute signed canonical-minus-ablation effects for each factor."""
    required = set(BRANCH_ORDER)
    if set(branches) != required:
        raise RelationalCodebookDataError("P6 attribution 分支集合不完整")

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
    for name, (with_factor, without_factor) in pairs.items():
        levels = []
        for index in range(2):
            enabled = branches[with_factor]["levels"][index]
            disabled = branches[without_factor]["levels"][index]
            levels.append(
                {
                    "level": index + 1,
                    "query_distortion_effect": enabled[
                        "query_cosine_distortion"
                    ]
                    - disabled["query_cosine_distortion"],
                    "graph_agreement_effect": enabled[
                        "weighted_graph_agreement"
                    ]
                    - disabled["weighted_graph_agreement"],
                    "category_purity_effect": enabled["category_purity"]
                    - disabled["category_purity"],
                }
            )
        result[name] = {
            "with_factor": with_factor,
            "without_factor": without_factor,
            "levels": levels,
        }

    interaction = []
    for index in range(2):
        canonical = branches["canonical"]["levels"][index]
        graph_off = branches["graph_off"]["levels"][index]
        prior_off = branches["near_zero_query_prior"]["levels"][index]
        both_off = branches["graph_off_near_zero_query_prior"]["levels"][index]
        interaction.append(
            {
                "level": index + 1,
                "query_distortion_interaction": canonical[
                    "query_cosine_distortion"
                ]
                - graph_off["query_cosine_distortion"]
                - prior_off["query_cosine_distortion"]
                + both_off["query_cosine_distortion"],
                "graph_agreement_interaction": canonical[
                    "weighted_graph_agreement"
                ]
                - graph_off["weighted_graph_agreement"]
                - prior_off["weighted_graph_agreement"]
                + both_off["weighted_graph_agreement"],
            }
        )
    result["graph_x_query_prior_interaction"] = {"levels": interaction}
    return result


def _run_branch(
    name: str,
    branch: Mapping[str, Any],
    inputs: RelationalCodebookInputs,
    output: Path,
    *,
    device: str,
    chunk_rows: int,
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    target = torch.device(device)
    poi_values = _normalize_torch(
        torch.from_numpy(np.asarray(inputs.poi_residual_s0, dtype=np.float32)).to(
            target
        )
    )
    query_values = _transform_query(
        inputs.query_embeddings, inputs.global_mean, device=target
    )
    depths = inputs.selection.nodes["supervision_depth"].to_numpy(
        zero_copy_only=False
    )
    poi_level_assignments: list[np.ndarray] = []
    query_level_assignments: list[np.ndarray] = []
    level_metrics: list[dict[str, Any]] = []
    s1_poi_assignments: np.ndarray | None = None
    branch_dir = output / "branches" / name
    branch_dir.mkdir(parents=True)

    for level in (1, 2):
        active_mask = depths >= level
        active_global_query_rows = inputs.selection.selected_query_rows[active_mask]
        active_query_values = query_values[torch.from_numpy(active_mask).to(target)]
        graph = _level_graph(inputs, level, active_global_query_rows)
        settings = branch[f"s{level}"]
        u, v, poi_assignment, query_assignment, fit = fit_dual_view_level(
            poi_values,
            active_query_values,
            torch.from_numpy(inputs.initial_poi_codebooks[level - 1]).to(target),
            inputs.initial_poi_assignments[level - 1],
            graph,
            inputs.coarse_category_indices
            if level == 1
            else inputs.fine_category_indices,
            level=level,
            codebook_size=512,
            settings=settings,
            prqk=branch["prqk"],
            warmup={"zero_weight_iters": 2, "linear_ramp_end_iter": 6},
            parent_assignments=s1_poi_assignments if level == 2 else None,
            fine_to_coarse=inputs.fine_to_coarse if level == 2 else None,
            chunk_rows=chunk_rows,
        )
        poi_np = poi_assignment.cpu().numpy().astype(np.int32)
        active_query_np = query_assignment.cpu().numpy().astype(np.int32)
        query_np = np.full(len(depths), -1, dtype=np.int32)
        query_np[active_mask] = active_query_np
        hard_rows = [
            row for row in fit["trace"] if row["stage"] == "hard_alternating"
        ]
        hard_query_distortion = hard_rows[-1]["components"][
            "query_content_sum"
        ] / len(active_query_np)
        poi_values, poi_residual = projection_residual(
            poi_values, u, poi_assignment
        )
        active_next, query_residual = projection_residual(
            active_query_values, v, query_assignment
        )
        query_next = torch.zeros_like(query_values)
        query_next[torch.from_numpy(active_mask).to(target)] = active_next
        query_values = query_next
        category = (
            inputs.coarse_category_indices
            if level == 1
            else inputs.fine_category_indices
        )
        level_result = {
            "level": level,
            "fit": fit,
            "hard_endpoint_query_content_mean": hard_query_distortion,
            "topk_query_distortion_delta": fit["final_query_content_mean"]
            - hard_query_distortion,
            "poi_assignment": _distribution_metrics(poi_np, 512),
            "query_assignment": _distribution_metrics(active_query_np, 512),
            "query_prior": query_prior_metrics(
                active_query_np,
                codebook_size=512,
                tau=float(settings["query_centroid_shrinkage"]),
            ),
            "graph": _graph_metrics(graph, poi_np, active_query_np),
            "category": _category_metrics(poi_np, category, 512),
            "reference_poi_assignment_change": float(
                np.mean(poi_np != inputs.initial_poi_assignments[level - 1])
            ),
            "poi_residual": poi_residual,
            "query_residual": query_residual,
        }
        _atomic_npy(branch_dir / f"poi_codebook_s{level}.npy", u.cpu().numpy().astype(np.float32))
        _atomic_npy(branch_dir / f"query_codebook_s{level}.npy", v.cpu().numpy().astype(np.float32))
        poi_level_assignments.append(poi_np)
        query_level_assignments.append(query_np)
        level_metrics.append(level_result)
        if level == 1:
            s1_poi_assignments = poi_np
        del u, v, poi_assignment, query_assignment, active_query_values, active_next
        gc.collect()
        torch.cuda.empty_cache()

    poi_prefix = np.column_stack(poi_level_assignments).astype(np.int32)
    query_prefix = np.column_stack(query_level_assignments).astype(np.int32)
    _atomic_npy(branch_dir / "poi_assignments_s1_s2.npy", poi_prefix)
    _atomic_npy(branch_dir / "query_assignments_s1_s2.npy", query_prefix)
    metrics = {
        "status": "completed",
        "branch": name,
        "description": branch["description"],
        "settings": {
            "s1": branch["s1"],
            "s2": branch["s2"],
            "prqk": branch["prqk"],
        },
        "levels": level_metrics,
        "poi_prefix": _prefix_metrics(poi_prefix),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(branch_dir / "metrics.json", metrics)
    return metrics


def _canonical_parity(
    config: RelationalCodebookConfig, canonical: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    reference_dir = output_directory(config, "sample")
    reference_metrics = json.loads(
        (reference_dir / "metrics.json").read_text(encoding="utf-8")
    )
    branch_dir = output / "branches" / "canonical"
    levels = []
    passed = True
    for level in (1, 2):
        reference_poi = np.load(
            reference_dir / f"poi_assignments_s{level}.npy", allow_pickle=False
        )
        reference_query = np.load(
            reference_dir / f"query_assignments_s{level}.npy", allow_pickle=False
        )
        actual_poi = np.load(
            branch_dir / "poi_assignments_s1_s2.npy", allow_pickle=False
        )[:, level - 1]
        actual_query = np.load(
            branch_dir / "query_assignments_s1_s2.npy", allow_pickle=False
        )[:, level - 1]
        valid_query = reference_query >= 0
        poi_match = float(np.mean(reference_poi == actual_poi))
        query_match = float(
            np.mean(reference_query[valid_query] == actual_query[valid_query])
        )
        distortion_delta = abs(
            canonical["levels"][level - 1]["fit"][
                "final_query_content_mean"
            ]
            - reference_metrics["levels"][level - 1]["fit"][
                "final_query_content_mean"
            ]
        )
        graph_delta = abs(
            canonical["levels"][level - 1]["graph"]["weighted_agreement"]
            - reference_metrics["levels"][level - 1]["graph"][
                "weighted_agreement"
            ]
        )
        level_passed = (
            poi_match >= 0.999
            and query_match >= 0.999
            and distortion_delta <= 1.0e-5
            and graph_delta <= 1.0e-9
        )
        passed = passed and level_passed
        levels.append(
            {
                "level": level,
                "poi_assignment_match_rate": poi_match,
                "query_assignment_match_rate": query_match,
                "query_distortion_abs_delta": distortion_delta,
                "graph_agreement_abs_delta": graph_delta,
                "passed": level_passed,
            }
        )
    return {"passed": passed, "levels": levels}


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
        root / "commands" / "diagnose_relational_codebook.py",
    ]
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def build_relational_attribution(
    config: RelationalCodebookConfig,
    *,
    device: str = "cuda",
    chunk_rows: int = 8192,
) -> dict[str, Any]:
    """Run fixed P6 sample ablations and publish an immutable diagnostic."""
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise RelationalCodebookDataError("P6 attribution 要求 CUDA；不得静默回退 CPU")
    canonical = validate_relational_codebook(config, gate="sample")
    baseline = validate_relational_evaluation(config, gate="sample")
    inputs = load_relational_codebook_inputs(config, gate="sample")
    final = diagnostic_output_directory(config)
    if final.exists():
        raise RelationalCodebookDataError(f"P6 attribution overwrite=false：{final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(f".{final.name}.{uuid.uuid4().hex}.building")
    staging.mkdir()
    started = time.perf_counter()
    branches = diagnostic_branches(config)
    branch_metrics: dict[str, Any] = {}
    torch.manual_seed(int(config.prqk["seed"]))
    torch.cuda.manual_seed_all(int(config.prqk["seed"]))
    torch.cuda.reset_peak_memory_stats()
    try:
        for name, branch in branches.items():
            print(
                json.dumps(
                    {"time": utc_now(), "stage": "p6_attribution_branch_started", "branch": name},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            branch_metrics[name] = _run_branch(
                name,
                branch,
                inputs,
                staging,
                device=device,
                chunk_rows=chunk_rows,
            )
            print(
                json.dumps(
                    {
                        "time": utc_now(),
                        "stage": "p6_attribution_branch_completed",
                        "branch": name,
                        "runtime_seconds": branch_metrics[name]["runtime_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        compact = {
            name: _compact_branch(branch_metrics[name]) for name in BRANCH_ORDER
        }
        parity = _canonical_parity(config, branch_metrics["canonical"], staging)
        if not parity["passed"]:
            raise RelationalCodebookDataError("P6 attribution canonical 复跑未通过正式 sample parity")
        comparison = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "phase": "P6-CAT-SAMPLE-ATTRIBUTION",
            "built_at": utc_now(),
            "contract": {
                "config_signature": config.signature(),
                "canonical_p6_manifest_sha256": canonical["manifest_sha256"],
                "canonical_evaluation_manifest_sha256": baseline[
                    "manifest_sha256"
                ],
                "same_selected_poi_rows": True,
                "same_selected_query_rows": True,
                "same_complete_s1_s2_edges": True,
                "branch_order": list(BRANCH_ORDER),
                "near_zero_query_prior": NEAR_ZERO_QUERY_PRIOR,
            },
            "p5_a0": baseline["comparison"]["levels"],
            "canonical_parity": parity,
            "branches": compact,
            "factor_contrasts": factor_contrasts(compact),
            "source_access": {
                "business_validation_read": False,
                "business_test_read": False,
                "geo_read": False,
            },
            "next_status": "HOLD_FOR_P6_SAMPLE_ATTRIBUTION_REVIEW",
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
            "phase": "P6-CAT-SAMPLE-ATTRIBUTION",
            "role": "CONTROLLED_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT",
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
    return validate_relational_attribution(config, device=device)


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise RelationalCodebookDataError(f"P6 attribution 非有限指标：{label}")


def validate_relational_attribution(
    config: RelationalCodebookConfig, *, device: str = "cuda"
) -> dict[str, Any]:
    """Validate attribution provenance, artifacts, shapes, and key metrics."""
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise RelationalCodebookDataError("P6 attribution validator 要求 CUDA")
    canonical = validate_relational_codebook(config, gate="sample")
    baseline = validate_relational_evaluation(config, gate="sample")
    inputs = load_relational_codebook_inputs(config, gate="sample")
    directory = diagnostic_output_directory(config)
    manifest_path = directory / "manifest.json"
    marker = json.loads((directory / "_SUCCESS").read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != manifest_sha:
        raise RelationalCodebookDataError("P6 attribution manifest 与 _SUCCESS 不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    contract = comparison.get("contract", {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != "P6-CAT-SAMPLE-ATTRIBUTION"
        or manifest.get("role") != "CONTROLLED_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT"
        or manifest.get("contract") != contract
        or contract.get("config_signature") != config.signature()
        or contract.get("canonical_p6_manifest_sha256")
        != canonical["manifest_sha256"]
        or contract.get("canonical_evaluation_manifest_sha256")
        != baseline["manifest_sha256"]
        or contract.get("branch_order") != list(BRANCH_ORDER)
        or comparison.get("next_status")
        != "HOLD_FOR_P6_SAMPLE_ATTRIBUTION_REVIEW"
    ):
        raise RelationalCodebookDataError("P6 attribution schema/来源/状态合同不匹配")
    expected_artifacts = {"comparison.json"}
    for name in BRANCH_ORDER:
        expected_artifacts.add(f"branches/{name}/metrics.json")
        expected_artifacts.add(f"branches/{name}/poi_assignments_s1_s2.npy")
        expected_artifacts.add(f"branches/{name}/query_assignments_s1_s2.npy")
        for level in (1, 2):
            expected_artifacts.add(f"branches/{name}/poi_codebook_s{level}.npy")
            expected_artifacts.add(f"branches/{name}/query_codebook_s{level}.npy")
    if set(manifest.get("artifacts", {})) != expected_artifacts:
        raise RelationalCodebookDataError("P6 attribution artifact 集合不完整")
    for relative, entry in manifest["artifacts"].items():
        path = directory / relative
        if entry.get("file") != relative or sha256_file(path) != entry.get("sha256"):
            raise RelationalCodebookDataError(f"P6 attribution artifact 损坏：{relative}")

    target = torch.device(device)
    depths = inputs.selection.nodes["supervision_depth"].to_numpy(
        zero_copy_only=False
    )
    for name in BRANCH_ORDER:
        branch_dir = directory / "branches" / name
        recorded = json.loads((branch_dir / "metrics.json").read_text(encoding="utf-8"))
        poi_prefix = np.load(
            branch_dir / "poi_assignments_s1_s2.npy", allow_pickle=False
        )
        query_prefix = np.load(
            branch_dir / "query_assignments_s1_s2.npy", allow_pickle=False
        )
        if (
            poi_prefix.shape != (len(inputs.selection.selected_poi_rows), 2)
            or poi_prefix.dtype != np.int32
            or query_prefix.shape != (len(inputs.selection.selected_query_rows), 2)
            or query_prefix.dtype != np.int32
            or np.any(poi_prefix < 0)
            or np.any(poi_prefix >= 512)
        ):
            raise RelationalCodebookDataError(f"P6 attribution {name} assignment shape/range 错误")
        poi_values = _normalize_torch(
            torch.from_numpy(
                np.asarray(inputs.poi_residual_s0, dtype=np.float32)
            ).to(target)
        )
        query_values = _transform_query(
            inputs.query_embeddings, inputs.global_mean, device=target
        )
        for level in (1, 2):
            active = depths >= level
            if (
                np.any(query_prefix[active, level - 1] < 0)
                or np.any(query_prefix[active, level - 1] >= 512)
                or np.any(query_prefix[~active, level - 1] != -1)
            ):
                raise RelationalCodebookDataError(f"P6 attribution {name} Query depth mask 错误")
            u_np = np.load(
                branch_dir / f"poi_codebook_s{level}.npy", allow_pickle=False
            )
            v_np = np.load(
                branch_dir / f"query_codebook_s{level}.npy", allow_pickle=False
            )
            if (
                u_np.shape != (512, inputs.embedding_dim)
                or u_np.dtype != np.float32
                or v_np.shape != (512, inputs.embedding_dim)
                or v_np.dtype != np.float32
                or not np.isfinite(u_np).all()
                or not np.isfinite(v_np).all()
            ):
                raise RelationalCodebookDataError(f"P6 attribution {name} codebook 错误")
            poi_assignment = torch.from_numpy(
                poi_prefix[:, level - 1].astype(np.int64)
            ).to(target)
            active_query_np = query_prefix[active, level - 1]
            query_assignment = torch.from_numpy(
                active_query_np.astype(np.int64)
            ).to(target)
            u = torch.from_numpy(u_np).to(target)
            v = torch.from_numpy(v_np).to(target)
            active_query = query_values[torch.from_numpy(active).to(target)]
            poi_distortion = float(
                (
                    1.0
                    - torch.sum(poi_values * _normalize_torch(u)[poi_assignment], dim=1)
                )
                .mean()
                .item()
            )
            query_distortion = float(
                (
                    1.0
                    - torch.sum(
                        active_query * _normalize_torch(v)[query_assignment], dim=1
                    )
                )
                .mean()
                .item()
            )
            level_record = recorded["levels"][level - 1]
            if not np.isclose(
                poi_distortion,
                level_record["fit"]["final_poi_content_mean"],
                rtol=1.0e-5,
                atol=1.0e-6,
            ) or not np.isclose(
                query_distortion,
                level_record["fit"]["final_query_content_mean"],
                rtol=1.0e-5,
                atol=1.0e-6,
            ):
                raise RelationalCodebookDataError(f"P6 attribution {name} S{level} distortion 不一致")
            graph = _level_graph(
                inputs, level, inputs.selection.selected_query_rows[active]
            )
            graph_metrics = _graph_metrics(
                graph, poi_prefix[:, level - 1], active_query_np
            )
            if not np.isclose(
                graph_metrics["weighted_agreement"],
                level_record["graph"]["weighted_agreement"],
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RelationalCodebookDataError(f"P6 attribution {name} S{level} 图指标不一致")
            poi_values, _ = projection_residual(
                poi_values, u, poi_assignment
            )
            active_next, _ = projection_residual(
                active_query, v, query_assignment
            )
            query_next = torch.zeros_like(query_values)
            query_next[torch.from_numpy(active).to(target)] = active_next
            query_values = query_next
        if _prefix_metrics(poi_prefix) != recorded["poi_prefix"]:
            raise RelationalCodebookDataError(f"P6 attribution {name} prefix 指标不一致")
        _assert_finite(recorded, f"branches.{name}")
    if comparison.get("factor_contrasts") != factor_contrasts(
        comparison["branches"]
    ) or not comparison.get("canonical_parity", {}).get("passed"):
        raise RelationalCodebookDataError("P6 attribution contrast/parity 不一致")
    if any(
        comparison.get("source_access", {}).get(key) is not False
        for key in ("business_validation_read", "business_test_read", "geo_read")
    ):
        raise RelationalCodebookDataError("P6 attribution 禁止数据读取标志错误")
    return {
        "status": "p6_sample_attribution_validated",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "comparison_sha256": sha256_file(directory / "comparison.json"),
        "comparison": comparison,
    }

"""Compare relational S1/S2 refinement against the frozen POI-only base."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.base_quantizer import _hard_assign, projection_residual
from qg_prqk.sid.relational_config import RelationalCodebookConfig
from qg_prqk.sid.relational_data import RelationalCodebookDataError, load_relational_codebook_inputs
from qg_prqk.sid.relational_quantizer import (
    _graph_metrics,
    _level_graph,
    _prefix_metrics,
    _to_device_normalized_float32,
    output_directory,
    validate_relational_codebook,
)


SCHEMA_VERSION = "qg-prqk-p6-sample-gate-evaluation-v1"
FULL_SCHEMA_VERSION = "qg-prqk-p6-full-evaluation-v1"


def _evaluation_schema(gate: str) -> str:
    return SCHEMA_VERSION if gate == "sample" else FULL_SCHEMA_VERSION


def _evaluation_phase(gate: str) -> str:
    return f"P6-CAT-{gate.upper()}-EVALUATION"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RelationalCodebookDataError(f"P6 evaluation 输出已存在：{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.writing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def partition_category_metrics(keys: np.ndarray, categories: np.ndarray) -> dict[str, Any]:
    """Measure category purity and entropy for a token or prefix partition."""
    partitions = np.asarray(keys)
    values = np.asarray(categories, dtype=np.int64)
    if partitions.ndim == 1:
        partitions = partitions[:, None]
    if partitions.ndim != 2 or len(partitions) != len(values) or not len(values):
        raise RelationalCodebookDataError("P6 evaluation partition/category 未对齐")
    groups: dict[tuple[int, ...], dict[int, int]] = {}
    for key, category in zip(partitions, values, strict=True):
        counts = groups.setdefault(tuple(int(item) for item in key), {})
        counts[int(category)] = counts.get(int(category), 0) + 1
    majority = 0
    entropy_sum = 0.0
    for counts in groups.values():
        local = np.asarray(list(counts.values()), dtype=np.float64)
        majority += int(local.max())
        probabilities = local / local.sum()
        entropy_sum += float(local.sum()) * float(
            -np.sum(probabilities * np.log(probabilities))
        )
    return {
        "partitions": len(groups),
        "purity": majority / len(values),
        "conditional_entropy": entropy_sum / len(values),
    }


def _comparison(current: float, reference: float) -> dict[str, float]:
    return {
        "p5_a0": reference,
        "p6": current,
        "delta": current - reference,
    }


def evaluation_directory(config: RelationalCodebookConfig, gate: str) -> Path:
    training = output_directory(config, gate)
    return training.parent / "evaluations" / f"{training.name}_vs_p5_a0_v1"


def evaluate_relational_codebook(
    config: RelationalCodebookConfig, *, gate: str, device: str = "cuda"
) -> dict[str, Any]:
    """Evaluate P6 and P5 A0 on the exact same closed POI/Query graph."""
    started = time.perf_counter()
    if gate not in (set(config.p6_gates) - {"selection", "seed"}):
        raise RelationalCodebookDataError(f"配置 {config.protocol_id} 未开放 P6 {gate} 评估")
    validated = validate_relational_codebook(config, gate=gate)
    inputs = load_relational_codebook_inputs(config, gate=gate)
    directory = output_directory(config, gate)
    output = evaluation_directory(config, gate)
    if output.exists():
        raise RelationalCodebookDataError(f"P6 evaluation overwrite=false：{output}")
    output.mkdir(parents=True)

    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise RelationalCodebookDataError("P6 sample 对照评估要求 CUDA；不得静默回退 CPU")
    target = torch.device(device)
    query_residual = _to_device_normalized_float32(
        np.load(
            directory / "query_residual_s0.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        device=target,
        chunk_rows=8192,
    )
    depths = inputs.selection.nodes["supervision_depth"].to_numpy(zero_copy_only=False)
    p5_poi = np.column_stack(inputs.initial_poi_assignments).astype(np.int32)
    p6_poi = np.load(directory / "poi_assignments_s1_s2.npy", allow_pickle=False)
    p6_query = np.load(directory / "query_assignments_s1_s2.npy", allow_pickle=False)
    p5_query = np.full_like(p6_query, -1)
    reference_graph: list[dict[str, Any]] = []
    reference_query_distortion: list[float] = []

    for level in (1, 2):
        active = depths >= level
        active_global = inputs.selection.selected_query_rows[active]
        active_values = query_residual[torch.from_numpy(active).to(target)]
        codebook = torch.from_numpy(inputs.initial_poi_codebooks[level - 1]).to(target)
        assignments, scores = _hard_assign(active_values, codebook, chunk_rows=8192)
        assignment_np = assignments.cpu().numpy().astype(np.int32)
        p5_query[active, level - 1] = assignment_np
        graph = _level_graph(inputs, level, active_global)
        reference_graph.append(
            _graph_metrics(graph, p5_poi[:, level - 1], assignment_np)
        )
        reference_query_distortion.append(float((1.0 - scores).mean().item()))
        residual, _ = projection_residual(active_values, codebook, assignments)
        next_query = torch.zeros_like(query_residual)
        next_query[torch.from_numpy(active).to(target)] = residual
        query_residual = next_query

    current_metrics = validated["metrics"]
    p5_s1_category = partition_category_metrics(
        p5_poi[:, 0], inputs.coarse_category_indices
    )
    p6_s1_category = partition_category_metrics(
        p6_poi[:, 0], inputs.coarse_category_indices
    )
    p5_s2_category = partition_category_metrics(
        p5_poi[:, :2], inputs.fine_category_indices
    )
    p6_s2_category = partition_category_metrics(
        p6_poi[:, :2], inputs.fine_category_indices
    )
    levels = []
    for level in (1, 2):
        current = current_metrics["levels"][level - 1]
        levels.append(
            {
                "level": level,
                "query_cosine_distortion": _comparison(
                    current["fit"]["final_query_content_mean"],
                    reference_query_distortion[level - 1],
                ),
                "weighted_graph_agreement": _comparison(
                    current["graph"]["weighted_agreement"],
                    reference_graph[level - 1]["weighted_agreement"],
                ),
                "poi_active_codes": _comparison(
                    float(current["poi_assignment"]["active_codes"]),
                    float(np.unique(p5_poi[:, level - 1]).size),
                ),
            }
        )
    payload = {
        "schema_version": _evaluation_schema(gate),
        "status": "completed",
        "phase": _evaluation_phase(gate),
        "built_at": utc_now(),
        "contract": {
            "config_signature": config.signature(),
            "p6_manifest": validated["manifest"],
            "p6_manifest_sha256": validated["manifest_sha256"],
            "p5_full_manifest_sha256": config.frozen_inputs["p5_full_manifest"]["sha256"],
            "same_selected_poi_rows": True,
            "same_selected_query_rows": True,
            "same_complete_s1_s2_edges": True,
            "p5_query_probe": "same_query_view_projection_residual_with_p5_poi_codebooks",
        },
        "levels": levels,
        "category": {
            "s1_coarse_token": {
                "p5_a0": p5_s1_category,
                "p6": p6_s1_category,
                "purity_delta": p6_s1_category["purity"] - p5_s1_category["purity"],
            },
            "s2_fine_s1_s2_path": {
                "p5_a0": p5_s2_category,
                "p6": p6_s2_category,
                "purity_delta": p6_s2_category["purity"] - p5_s2_category["purity"],
            },
        },
        "poi_prefix": {
            "p5_a0": _prefix_metrics(p5_poi),
            "p6": _prefix_metrics(p6_poi),
        },
        "gate_evidence": {
            "query_distortion_improved_both_levels": all(
                item["query_cosine_distortion"]["delta"] < 0 for item in levels
            ),
            "graph_agreement_improved_both_levels": all(
                item["weighted_graph_agreement"]["delta"] >= 0 for item in levels
            ),
            "category_purity_improved_both_levels": (
                p6_s1_category["purity"] >= p5_s1_category["purity"]
                and p6_s2_category["purity"] >= p5_s2_category["purity"]
            ),
            "all_poi_codes_active": all(
                item["poi_active_codes"]["p6"] == 512 for item in levels
            ),
            "sample_is_code_smoke_only": gate == "sample",
            "formal_metric_basis": gate == "full",
            "automatic_downstream_launch": False,
            "next_status": f"HOLD_FOR_P6_{gate.upper()}_REVIEW",
        },
        "source_access": {
            "business_validation_read": False,
            "business_test_read": False,
            "geo_read": False,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": device,
        },
    }
    _atomic_json(output / "comparison.json", payload)
    manifest = {
        "schema_version": _evaluation_schema(gate),
        "status": "completed",
        "phase": payload["phase"],
        "comparison": {
            "file": "comparison.json",
            "sha256": sha256_file(output / "comparison.json"),
        },
        "contract": payload["contract"],
        "gate_evidence": payload["gate_evidence"],
    }
    write_json_atomic(output / "manifest.json", manifest)
    write_json_atomic(
        output / "_SUCCESS", {"manifest_sha256": sha256_file(output / "manifest.json")}
    )
    return validate_relational_evaluation(config, gate=gate)["comparison"]


def validate_relational_evaluation(
    config: RelationalCodebookConfig, *, gate: str
) -> dict[str, Any]:
    """Independently verify the P6 Gate comparison and its frozen sources."""
    validated = validate_relational_codebook(config, gate=gate)
    output = evaluation_directory(config, gate)
    manifest_path = output / "manifest.json"
    comparison_path = output / "comparison.json"
    marker = json.loads((output / "_SUCCESS").read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    if marker.get("manifest_sha256") != manifest_sha:
        raise RelationalCodebookDataError("P6 evaluation manifest 与 _SUCCESS 哈希不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    contract = comparison.get("contract", {})
    if (
        manifest.get("schema_version") != _evaluation_schema(gate)
        or manifest.get("status") != "completed"
        or manifest.get("phase") != _evaluation_phase(gate)
        or manifest.get("comparison", {}).get("sha256")
        != sha256_file(comparison_path)
        or manifest.get("contract") != contract
        or manifest.get("gate_evidence") != comparison.get("gate_evidence")
        or contract.get("config_signature") != config.signature()
        or contract.get("p6_manifest_sha256") != validated["manifest_sha256"]
        or contract.get("p5_full_manifest_sha256")
        != config.frozen_inputs["p5_full_manifest"]["sha256"]
    ):
        raise RelationalCodebookDataError("P6 evaluation schema/来源/结果合同不匹配")
    if any(
        comparison.get("source_access", {}).get(name) is not False
        for name in ("business_validation_read", "business_test_read", "geo_read")
    ):
        raise RelationalCodebookDataError("P6 evaluation 禁止数据读取标志错误")
    return {
        "status": "p6_gate_evaluation_validated",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "comparison_sha256": sha256_file(comparison_path),
        "comparison": comparison,
    }

"""Validate and summarize the two frozen final-epoch TIGER evaluations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .config import BeamRiskConfig
from .errors import BeamRiskError
from .io import atomic_write_json, atomic_write_text, read_json


EVALUATION_SCHEMA_VERSION = "beamrisk-final-evaluation-v1"
METRIC_NAMES = ("hr@1", "hr@3", "hr@5", "hr@10", "ndcg@10", "valid_id_rate")
TIGER_FIXED10K_BASELINE = {
    "hr@1": 0.5187,
    "hr@3": 0.7600,
    "hr@5": 0.8226,
    "hr@10": 0.8716,
    "ndcg@10": 0.704190,
    "valid_id_rate": 0.74105,
}


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BeamRiskError(f"评测指标 {name} 缺失或不是数值")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise BeamRiskError(f"评测指标 {name} 越界：{result}")
    return result


def _load_eval_result(
    path: Path,
    *,
    checkpoint: Path,
    legal_path_constraint: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(path, name="TIGER fixed-10k evaluation")
    if payload.get("status") != "completed":
        raise BeamRiskError(f"评测尚未完成：{path}")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise BeamRiskError("最终评测必须且只能包含 epoch-3 一个结果")
    result = results[0]
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise BeamRiskError("epoch-3 评测结果状态无效")
    run_config = result.get("config")
    metrics = result.get("metrics")
    if not isinstance(run_config, dict) or not isinstance(metrics, dict):
        raise BeamRiskError("epoch-3 评测缺少 config/metrics")
    result_checkpoint = Path(str(run_config.get("checkpoint", ""))).resolve()
    if result_checkpoint != checkpoint.resolve():
        raise BeamRiskError(
            f"评测 checkpoint {result_checkpoint} != 最终模型 {checkpoint.resolve()}"
        )
    if not math.isclose(float(run_config.get("epoch", -1)), 3.0, abs_tol=1e-4):
        raise BeamRiskError("最终评测 checkpoint 不是 epoch 3")
    if run_config.get("num_beams") != 10 or run_config.get("rows") != 10_000:
        raise BeamRiskError("最终评测必须固定为 10,000 条、Beam=10")
    if run_config.get("legal_path_constraint") is not legal_path_constraint:
        raise BeamRiskError("评测约束模式与输出目录不一致")
    if metrics.get("sample_count") != 10_000:
        raise BeamRiskError("最终评测 sample_count 不是 10,000")
    normalized_metrics = {
        name: _finite_metric(metrics, name) for name in METRIC_NAMES
    }
    if legal_path_constraint and not math.isclose(
        normalized_metrics["valid_id_rate"], 1.0, abs_tol=1e-12
    ):
        raise BeamRiskError("合法路径约束评测的 valid_id_rate 必须为 1")
    subset = payload.get("evaluation_subset")
    if not isinstance(subset, dict) or subset.get("output_rows") != 10_000:
        raise BeamRiskError("评测没有使用冻结的 10,000 条 Validation 子集")
    output_sha256 = subset.get("output_sha256")
    if not isinstance(output_sha256, str) or len(output_sha256) != 64:
        raise BeamRiskError("评测子集缺少有效 SHA256")
    compact: dict[str, Any] = {
        "result_file": str(path.resolve()),
        "run_dir": result.get("run_dir"),
        "metrics": normalized_metrics,
        "performance": result.get("performance"),
        "evaluation_protocol": payload.get("evaluation_protocol"),
    }
    bucket_metrics = result.get("bucket_metrics")
    if isinstance(bucket_metrics, dict):
        compact["bucket_metrics"] = bucket_metrics
    return compact, subset


def _checkpoint_eval_loss(checkpoint: Path, expected_step: int) -> float:
    state = read_json(checkpoint / "trainer_state.json", name="Trainer state")
    if state.get("global_step") != expected_step:
        raise BeamRiskError(
            f"{checkpoint.name} global_step != {expected_step}"
        )
    for row in reversed(state.get("log_history", [])):
        if (
            isinstance(row, Mapping)
            and row.get("step") == expected_step
            and isinstance(row.get("eval_loss"), (int, float))
        ):
            return float(row["eval_loss"])
    raise BeamRiskError(f"{checkpoint.name} 缺少 epoch-end eval_loss")


def _training_summary(
    config: BeamRiskConfig,
    *,
    final_checkpoint: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in (1, 2, 3):
        manifest = read_json(
            config.paths.sft_output_dir / "stages" / f"epoch_{epoch}.json",
            name=f"Epoch {epoch} stage manifest",
        )
        if manifest.get("status") != "completed" or manifest.get("target_epoch") != epoch:
            raise BeamRiskError(f"Epoch {epoch} stage manifest 无效")
        checkpoint = Path(str(manifest.get("checkpoint", ""))).resolve()
        expected_step = config.training.expected_optimizer_steps_per_epoch * epoch
        if checkpoint.name != f"checkpoint-{expected_step}" or not checkpoint.is_dir():
            raise BeamRiskError(f"Epoch {epoch} checkpoint 路径无效：{checkpoint}")
        rows.append(
            {
                "epoch": epoch,
                "global_step": expected_step,
                "checkpoint": str(checkpoint),
                "validation_loss": _checkpoint_eval_loss(checkpoint, expected_step),
                "risk_enabled": bool(manifest.get("risk_enabled")),
                "beamrisk_summary": manifest.get("beamrisk_summary"),
            }
        )
    if Path(rows[-1]["checkpoint"]).resolve() != final_checkpoint.resolve():
        raise BeamRiskError("训练阶段 manifest 的 epoch-3 checkpoint 与评测不一致")
    return rows


def _percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# BeamRisk-SFT 最终训练与固定 10k 评测",
        "",
        "## 三轮训练",
        "",
        "| Epoch | Global step | Validation loss | BeamRisk | 风险 pair 展示次数 |",
        "|---:|---:|---:|---|---:|",
    ]
    for row in payload["training"]:
        summary = row.get("beamrisk_summary")
        global_pairs = summary.get("global_pairs", 0) if isinstance(summary, Mapping) else 0
        lines.append(
            f"| {row['epoch']} | {row['global_step']} | "
            f"{row['validation_loss']:.6f} | "
            f"{'开启' if row['risk_enabled'] else '关闭'} | {global_pairs:,} |"
        )
    lines.extend(
        [
            "",
            "## 固定 10k 生成式检索",
            "",
            "| 模式 | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | Valid ID Rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, metrics in (
        ("TIGER 基线（无约束）", payload["tiger_baseline"]),
        ("BeamRisk-SFT 无约束", payload["unconstrained"]["metrics"]),
        ("BeamRisk-SFT 合法路径约束", payload["legal_path_constrained"]["metrics"]),
    ):
        lines.append(
            f"| {label} | {_percentage(metrics['hr@1'])} | "
            f"{_percentage(metrics['hr@3'])} | {_percentage(metrics['hr@5'])} | "
            f"{_percentage(metrics['hr@10'])} | {_percentage(metrics['ndcg@10'])} | "
            f"{_percentage(metrics['valid_id_rate'])} |"
        )
    delta = payload["unconstrained_minus_tiger"]
    lines.extend(
        [
            "",
            "无约束主口径相对 TIGER："
            f"HR@1 {delta['hr@1'] * 100:+.2f}pp，"
            f"HR@10 {delta['hr@10'] * 100:+.2f}pp，"
            f"NDCG@10 {delta['ndcg@10'] * 100:+.2f}pp。",
            "",
            f"固定子集 SHA256：`{payload['fixed_validation']['output_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_final_evaluation(
    config: BeamRiskConfig,
    *,
    final_checkpoint: Path,
    unconstrained_result_path: Path,
    constrained_result_path: Path,
) -> dict[str, Any]:
    final_checkpoint = final_checkpoint.resolve()
    expected_step = config.training.expected_optimizer_steps_per_epoch * 3
    if final_checkpoint.name != f"checkpoint-{expected_step}":
        raise BeamRiskError("最终 checkpoint 名称与三轮 step 不一致")
    unconstrained, unconstrained_subset = _load_eval_result(
        unconstrained_result_path,
        checkpoint=final_checkpoint,
        legal_path_constraint=False,
    )
    constrained, constrained_subset = _load_eval_result(
        constrained_result_path,
        checkpoint=final_checkpoint,
        legal_path_constraint=True,
    )
    identity_fields = (
        "output_rows",
        "output_sha256",
        "business_keys_sha256",
        "reference_sha256",
        "source_sha256",
    )
    if any(
        unconstrained_subset.get(name) != constrained_subset.get(name)
        for name in identity_fields
    ):
        raise BeamRiskError("无约束与合法路径约束没有使用同一固定 10k")
    training = _training_summary(config, final_checkpoint=final_checkpoint)
    unconstrained_metrics = unconstrained["metrics"]
    constrained_metrics = constrained["metrics"]
    payload: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "completed",
        "final_checkpoint": str(final_checkpoint),
        "training": training,
        "fixed_validation": {
            name: unconstrained_subset.get(name) for name in identity_fields
        },
        "unconstrained": unconstrained,
        "legal_path_constrained": constrained,
        "legal_minus_unconstrained": {
            name: constrained_metrics[name] - unconstrained_metrics[name]
            for name in METRIC_NAMES
        },
        "tiger_baseline": dict(TIGER_FIXED10K_BASELINE),
        "unconstrained_minus_tiger": {
            name: unconstrained_metrics[name] - TIGER_FIXED10K_BASELINE[name]
            for name in METRIC_NAMES
        },
    }
    output_dir = config.paths.eval_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "final_eval_summary.json", payload)
    markdown = _render_markdown(payload)
    atomic_write_text(output_dir / "final_eval_summary.md", markdown)
    payload["markdown"] = markdown
    return payload

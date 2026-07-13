#!/usr/bin/env python3
"""Run the final CAU-RQ-VAE P1 training, selection, export, and reports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rqvae import CAURQVAE  # noqa: E402
from rqvae_preprocess import input_numeric_summary, load_preprocess  # noqa: E402
from sid_eval import (  # noqa: E402
    SID_MAPPING_COLUMNS,
    add_analysis_labels,
    build_mode_metrics,
    build_quality_report,
    build_sid_mapping,
    ensure_parent_dir,
    read_parquet,
    validate_mapping,
    validate_meta_alignment,
    write_parquet,
)
from train_utils import cpu_thread_summary, load_yaml, resolve_device, save_checkpoint, set_random_seed, write_train_log  # noqa: E402


def load_script_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import script module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAU_TRAIN = load_script_module(ROOT / "scripts" / "08_train_cau_rqvae.py", "cau_train_reuse")
SID_EXPORT = load_script_module(ROOT / "scripts" / "05_export_and_eval_sid.py", "sid_export_reuse")


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rqvae_cau.yaml")
    parser.add_argument("--output-dir", default="outputs/experiments/cau_rqvae/final_p1")
    parser.add_argument("--lambda-tag", type=float, default=0.025)
    parser.add_argument("--lambda-unique", type=float, default=0.10)
    parser.add_argument("--unique-margin", type=float, default=0.70)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--max-total-epochs", type=int, default=80)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sid-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="final_p1")
    parser.add_argument("--max-report-groups", type=int, default=1000)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "NA"
    return f"{f:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No rows."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                text = format_float(value)
            else:
                text = str(value)
            cells.append(text.replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def guard_absent(paths: list[Path], label: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing {label}: {existing}")


def merge_config(base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    cfg.setdefault("rqvae", {})
    cfg.setdefault("cau", {})
    cfg["rqvae"]["epochs"] = int(args.max_total_epochs)
    cfg["rqvae"]["early_stop_patience"] = int(args.early_stop_patience)
    cfg["rqvae"]["device"] = args.device
    if args.batch_size is not None:
        cfg["rqvae"]["batch_size"] = int(args.batch_size)
    cfg["cau"]["lambda_tag"] = float(args.lambda_tag)
    cfg["cau"]["lambda_unique"] = float(args.lambda_unique)
    cfg["cau"]["unique_margin"] = float(args.unique_margin)
    cfg["final_training"] = {
        "run_name": args.run_name,
        "warmup_epochs": int(args.warmup_epochs),
        "max_total_epochs": int(args.max_total_epochs),
        "early_stop_patience": int(args.early_stop_patience),
        "evaluation_interval": int(args.evaluation_interval),
        "sid_batch_size": int(args.sid_batch_size),
        "checkpoint_selection": {
            "protection": [
                "prefix1_semantic_purity >= baseline - 0.005",
                "prefix3_semantic_purity >= baseline - 0.005",
                "reconstruction_cosine >= baseline - 0.03",
                "unique_pid_rate >= baseline - 0.01",
                "min_codebook_usage_rate >= 0.95",
                "no NaN / Inf",
            ],
            "lexicographic_order": [
                "qrels_sid_collision_rate ascending",
                "unique_sid_rate descending",
                "prefix1_semantic_purity descending",
                "category_heldout_macro_f1 descending",
            ],
        },
    }
    return cfg


def write_command(path: Path, args: argparse.Namespace) -> None:
    path.write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")


def nvidia_smi_summary() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=20)
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"nvidia-smi failed: {exc}"
    if result.returncode != 0:
        return result.stderr.strip() or f"nvidia-smi exited {result.returncode}"
    return result.stdout.strip()


def write_environment(path: Path, device: torch.device, started_at: str) -> None:
    lines = [
        f"started_at: {started_at}",
        f"python: {platform.python_version()}",
        f"python_executable: {sys.executable}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"torch_cuda_available: {torch.cuda.is_available()}",
        f"torch_cuda_device_count: {torch.cuda.device_count()}",
        f"selected_device: {device}",
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"cpu_threads: {cpu_thread_summary()}",
        "",
        "nvidia_smi:",
        nvidia_smi_summary(),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_checksums(path: Path) -> None:
    files = [
        "src/rqvae.py",
        "src/cau_labels.py",
        "scripts/07_build_cau_labels.py",
        "scripts/08_train_cau_rqvae.py",
        "scripts/09_train_cau_final.py",
        "scripts/05_export_and_eval_sid.py",
        "configs/rqvae_cau.yaml",
    ]
    lines = []
    for rel in files:
        p = ROOT / rel
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_baseline_metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    row = df.loc[df["run"] == "semantic_baseline"]
    if row.empty:
        raise ValueError(f"Missing semantic_baseline row in {path}")
    r = row.iloc[0]
    keys = [
        "prefix1_semantic_purity",
        "prefix2_semantic_purity",
        "prefix3_semantic_purity",
        "unique_sid_rate",
        "unique_pid_rate",
        "qrels_sid_collision_rate",
        "qrels_pid_collision_rate",
        "reconstruction_cosine",
        "max_pois_per_sid",
        "sid_collision_group_count",
    ]
    return {key: float(r[key]) for key in keys}


def set_rqvae_trainable(model: CAURQVAE, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("tag_classifier."):
            param.requires_grad = True
        else:
            param.requires_grad = bool(trainable)


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters for optimizer")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def finite_numeric_values(row: dict[str, Any]) -> bool:
    for value in row.values():
        if isinstance(value, (int, np.integer)):
            continue
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)):
                return False
    return True


def flatten_sid_metrics(
    metrics: dict[str, Any],
    epoch: int,
    checkpoint_path: Path,
    train_row: dict[str, Any],
) -> dict[str, Any]:
    prefix = metrics["prefix_stats"]
    sid = metrics["sid_stats"]
    pid = metrics["pid_stats"]
    qrels = metrics["qrels_stats"]
    candidates = metrics["candidates_stats"]
    codebook = metrics["codebook_usage"]
    out: dict[str, Any] = {
        "epoch": int(epoch),
        "checkpoint": str(checkpoint_path),
        "unique_sid_count": sid["unique_sid_count"],
        "unique_sid_rate": sid["unique_sid_rate"],
        "sid_collision_group_count": sid["sid_collision_group_count"],
        "sid_collision_poi_count": sid["sid_collision_poi_count"],
        "max_pois_per_sid": sid["max_pois_per_sid"],
        "unique_pid_count": pid["unique_pid_gid6_sid_count"],
        "unique_pid_rate": pid["unique_pid_gid6_sid_rate"],
        "pid_collision_group_count": pid["pid_collision_group_count"],
        "pid_collision_poi_count": pid["pid_collision_poi_count"],
        "max_pois_per_pid": pid["max_pois_per_pid"],
        "dedup_pid_unique_rate": pid["pid_gid6_sid_dedup_unique_rate"],
        "qrels_sid_collision_count": qrels.get("qrels_targets_in_sid_collision_count", 0),
        "qrels_sid_collision_rate": qrels.get("qrels_targets_in_sid_collision_rate", 0.0),
        "qrels_pid_collision_count": qrels.get("qrels_targets_in_pid_collision_count", 0),
        "qrels_pid_collision_rate": qrels.get("qrels_targets_in_pid_collision_rate", 0.0),
        "candidates_pid_collision_rate": candidates.get("queries_with_pid_collision_rate", 0.0),
        "candidates_sid_collision_rate": candidates.get("queries_with_sid_collision_rate", 0.0),
        "reconstruction_cosine": train_row.get("val_recon_cosine", np.nan),
        "category_heldout_accuracy": train_row.get("tag_heldout_accuracy", np.nan),
        "category_heldout_macro_f1": train_row.get("tag_heldout_macro_f1", np.nan),
        "val_total_loss": train_row.get("val_total_loss", np.nan),
        "train_weighted_tag_to_base_ratio": train_row.get("train_weighted_tag_to_base_ratio", np.nan),
        "train_weighted_unique_to_base_ratio": train_row.get("train_weighted_unique_to_base_ratio", np.nan),
    }
    for key in ("prefix1", "prefix2", "prefix3"):
        short = key.replace("prefix", "prefix")
        out[f"{short}_semantic_purity"] = prefix[key]["weighted_semantic_purity"]
        out[f"{short}_city_purity"] = prefix[key]["weighted_city_purity"]
        out[f"{short}_geohash5_purity"] = prefix[key]["weighted_geohash5_purity"]
    usage_rates = []
    for usage in codebook:
        level = int(usage["level"])
        out[f"codebook{level}_usage_rate"] = usage["used_code_rate"]
        out[f"codebook{level}_entropy"] = usage["entropy"]
        out[f"codebook{level}_perplexity"] = usage["perplexity"]
        out[f"codebook{level}_used"] = usage["used_code_count"]
        usage_rates.append(float(usage["used_code_rate"]))
    out["min_codebook_usage_rate"] = min(usage_rates) if usage_rates else 0.0
    return out


def passes_protection(row: dict[str, Any], baseline: dict[str, float]) -> tuple[bool, list[str]]:
    checks = [
        (
            "prefix1_semantic_purity",
            float(row["prefix1_semantic_purity"]) >= float(baseline["prefix1_semantic_purity"]) - 0.005,
        ),
        (
            "prefix3_semantic_purity",
            float(row["prefix3_semantic_purity"]) >= float(baseline["prefix3_semantic_purity"]) - 0.005,
        ),
        (
            "reconstruction_cosine",
            float(row["reconstruction_cosine"]) >= float(baseline["reconstruction_cosine"]) - 0.03,
        ),
        (
            "unique_pid_rate",
            float(row["unique_pid_rate"]) >= float(baseline["unique_pid_rate"]) - 0.01,
        ),
        ("min_codebook_usage_rate", float(row["min_codebook_usage_rate"]) >= 0.95),
        ("finite_numeric_values", finite_numeric_values(row)),
    ]
    failed = [name for name, ok in checks if not ok]
    return not failed, failed


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["qrels_sid_collision_rate"]),
        -float(row["unique_sid_rate"]),
        -float(row["prefix1_semantic_purity"]),
        -float(row["category_heldout_macro_f1"]),
    )


@torch.no_grad()
def run_sid_eval(
    model: CAURQVAE,
    rqvae_input: np.ndarray,
    embedding_meta: pd.DataFrame,
    geo_meta: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    checkpoint_path: Path,
    report_dir: Path,
    mode: str,
    rqvae_input_path: Path,
    max_report_groups: int,
    train_row: dict[str, Any],
    epoch: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_dir.mkdir(parents=True, exist_ok=True)
    indices = SID_EXPORT.encode_sid_indices(model, rqvae_input, batch_size, device)
    mapping = build_sid_mapping(
        embedding_meta=embedding_meta,
        geo_meta=geo_meta,
        indices=indices,
        codebook_size=model.codebook_size,
    )
    validate_mapping(mapping, model.codebook_size)
    metrics = build_mode_metrics(
        mode=mode,
        mapping=mapping,
        indices=indices,
        codebook_size=model.codebook_size,
        checkpoint_path=checkpoint_path,
        rqvae_input_path=rqvae_input_path,
        mapping_path=report_dir / f"poi_sid_mapping_{mode}.parquet",
        indices_path=report_dir / f"poi_sid_indices_{mode}.npy",
        report_dir=report_dir,
        qrels_path=ROOT / "data/processed/mobilitybench/qrels.csv",
        candidates_path=ROOT / "data/processed/mobilitybench/candidates.csv",
        max_report_groups=max_report_groups,
    )
    (report_dir / "sid_quality.md").write_text(build_quality_report(metrics), encoding="utf-8")
    flat = flatten_sid_metrics(metrics, epoch=epoch, checkpoint_path=checkpoint_path, train_row=train_row)
    return flat, metrics


def save_training_checkpoint(
    path: Path,
    model: CAURQVAE,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    vocab: dict[str, Any],
    init_info: dict[str, Any],
    data_cfg: dict[str, Any],
    input_dim: int,
    seed: int,
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
    train_log_path: Path,
    phase: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload_extra = {
        "architecture": "cau_rqvae",
        "config": config,
        "cau_vocab": vocab,
        "init_info": init_info,
        "phase": phase,
    }
    if extra:
        payload_extra.update(extra)
    save_checkpoint(
        path,
        model,
        optimizer,
        mode="cau",
        input_dim=input_dim,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_log_path=train_log_path,
        preprocess_path=Path(data_cfg["preprocess_semantic_pkl"]),
        rqvae_input_path=Path(data_cfg["rqvae_input_semantic_npy"]),
        random_seed=seed,
        epoch=epoch,
        extra=payload_extra,
    )


def export_formal_cau(
    checkpoint_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    sid_batch_size: int,
    max_report_groups: int,
    train_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    formal_indices = ROOT / "data/sid/poi_sid_indices_cau.npy"
    formal_mapping = ROOT / "data/sid/poi_sid_mapping_cau.parquet"
    formal_report = ROOT / "reports/sid_quality_cau.md"
    final_indices_copy = output_dir / "poi_sid_indices_cau.npy"
    final_mapping_copy = output_dir / "poi_sid_mapping_cau.parquet"
    guard_absent([formal_indices, formal_mapping, formal_report, final_indices_copy, final_mapping_copy], "formal CAU export")

    model, checkpoint = SID_EXPORT.load_checkpoint_model(checkpoint_path, device)
    rqvae_input, embedding_meta, geo_meta = SID_EXPORT.load_aligned_inputs(
        ROOT / "data/embeddings/poi_embedding_meta.parquet",
        ROOT / "data/geo/poi_geo_meta.parquet",
        ROOT / config["data"]["rqvae_input_semantic_npy"],
    )
    indices = SID_EXPORT.encode_sid_indices(model, rqvae_input, sid_batch_size, device)
    mapping = build_sid_mapping(embedding_meta=embedding_meta, geo_meta=geo_meta, indices=indices, codebook_size=model.codebook_size)
    validate_mapping(mapping, model.codebook_size)

    ensure_parent_dir(formal_indices)
    np.save(formal_indices, indices.astype(np.int64))
    write_parquet(mapping[SID_MAPPING_COLUMNS], formal_mapping)
    shutil.copy2(formal_indices, final_indices_copy)
    shutil.copy2(formal_mapping, final_mapping_copy)

    metrics = build_mode_metrics(
        mode="cau",
        mapping=mapping,
        indices=indices,
        codebook_size=model.codebook_size,
        checkpoint_path=checkpoint_path,
        rqvae_input_path=ROOT / config["data"]["rqvae_input_semantic_npy"],
        mapping_path=formal_mapping,
        indices_path=formal_indices,
        report_dir=ROOT / "reports",
        qrels_path=ROOT / "data/processed/mobilitybench/qrels.csv",
        candidates_path=ROOT / "data/processed/mobilitybench/candidates.csv",
        max_report_groups=max_report_groups,
    )
    formal_report.write_text(build_quality_report(metrics), encoding="utf-8")
    flat = flatten_sid_metrics(metrics, epoch=int(checkpoint.get("epoch", -1)), checkpoint_path=checkpoint_path, train_row=train_row)
    return flat, metrics


def append_final_metrics(pilot_metrics_path: Path, final_flat: dict[str, Any], best_train_row: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(pilot_metrics_path, encoding="utf-8-sig")
    baseline = df.loc[df["run"] == "semantic_baseline"].iloc[0]
    final: dict[str, Any] = {col: "" for col in df.columns}
    final.update(
        {
            "run": "CAU_Final_P1",
            "lambda_tag": 0.025,
            "lambda_unique": 0.10,
            "checkpoint": "outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt",
            "sid_report": "reports/sid_quality_cau.md",
            "unique_sid_rate": final_flat["unique_sid_rate"],
            "unique_pid_rate": final_flat["unique_pid_rate"],
            "sid_collision_group_count": final_flat["sid_collision_group_count"],
            "max_pois_per_sid": final_flat["max_pois_per_sid"],
            "prefix1_semantic_purity": final_flat["prefix1_semantic_purity"],
            "prefix2_semantic_purity": final_flat["prefix2_semantic_purity"],
            "prefix3_semantic_purity": final_flat["prefix3_semantic_purity"],
            "qrels_sid_collision_count": final_flat["qrels_sid_collision_count"],
            "qrels_sid_collision_rate": final_flat["qrels_sid_collision_rate"],
            "qrels_pid_collision_count": final_flat["qrels_pid_collision_count"],
            "qrels_pid_collision_rate": final_flat["qrels_pid_collision_rate"],
            "codebook0_usage_rate": final_flat["codebook0_usage_rate"],
            "codebook1_usage_rate": final_flat["codebook1_usage_rate"],
            "codebook2_usage_rate": final_flat["codebook2_usage_rate"],
            "best_epoch": final_flat["epoch"],
            "best_val_loss": final_flat["val_total_loss"],
            "reconstruction_cosine": final_flat["reconstruction_cosine"],
            "category_heldout_macro_f1": final_flat["category_heldout_macro_f1"],
            "category_heldout_accuracy": final_flat["category_heldout_accuracy"],
            "train_tag_base_ratio": best_train_row.get("train_weighted_tag_to_base_ratio", ""),
            "train_unique_base_ratio": best_train_row.get("train_weighted_unique_to_base_ratio", ""),
            "valid_collision_pairs_per_batch": best_train_row.get("train_mean_valid_collision_pairs_per_batch", ""),
            "zero_pair_batch_rate": best_train_row.get("train_zero_pair_batch_rate", ""),
            "active_margin_pair_rate": best_train_row.get("train_active_margin_pair_rate", ""),
            "collision_pair_mean_cosine": best_train_row.get("train_collision_pair_mean_cosine", ""),
            "val_online_unique_sid_rate": best_train_row.get("val_online_unique_sid_rate", ""),
            "delta_prefix1_semantic_purity": final_flat["prefix1_semantic_purity"] - float(baseline["prefix1_semantic_purity"]),
            "delta_prefix2_semantic_purity": final_flat["prefix2_semantic_purity"] - float(baseline["prefix2_semantic_purity"]),
            "delta_prefix3_semantic_purity": final_flat["prefix3_semantic_purity"] - float(baseline["prefix3_semantic_purity"]),
            "delta_unique_sid_rate": final_flat["unique_sid_rate"] - float(baseline["unique_sid_rate"]),
            "delta_sid_collision_group_count": final_flat["sid_collision_group_count"] - float(baseline["sid_collision_group_count"]),
            "delta_max_pois_per_sid": final_flat["max_pois_per_sid"] - float(baseline["max_pois_per_sid"]),
            "delta_qrels_sid_collision_rate": final_flat["qrels_sid_collision_rate"] - float(baseline["qrels_sid_collision_rate"]),
            "delta_unique_pid_rate": final_flat["unique_pid_rate"] - float(baseline["unique_pid_rate"]),
            "delta_qrels_pid_collision_rate": final_flat["qrels_pid_collision_rate"] - float(baseline["qrels_pid_collision_rate"]),
            "delta_reconstruction_cosine": final_flat["reconstruction_cosine"] - float(baseline["reconstruction_cosine"]),
            "min_codebook_usage_rate": final_flat["min_codebook_usage_rate"],
            "passes_protection": True,
        }
    )
    final["primary_score"] = (
        (float(baseline["qrels_sid_collision_rate"]) - final_flat["qrels_sid_collision_rate"]) * 5.0
        + (final_flat["unique_sid_rate"] - float(baseline["unique_sid_rate"])) * 5.0
        + (final_flat["prefix1_semantic_purity"] - float(baseline["prefix1_semantic_purity"]))
    )
    out = pd.concat([df, pd.DataFrame([final])], ignore_index=True)
    return out


def write_compare_reports(metrics_df: pd.DataFrame, final_flat: dict[str, Any], output_dir: Path) -> None:
    metrics_path = ROOT / "reports/cau_rqvae_ppt_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    selected_cols = [
        "run",
        "prefix1_semantic_purity",
        "prefix2_semantic_purity",
        "prefix3_semantic_purity",
        "unique_sid_rate",
        "sid_collision_group_count",
        "max_pois_per_sid",
        "qrels_sid_collision_rate",
        "unique_pid_rate",
        "qrels_pid_collision_rate",
        "reconstruction_cosine",
        "category_heldout_macro_f1",
    ]
    table_rows = []
    for _, row in metrics_df[selected_cols].iterrows():
        table_rows.append(row.to_dict())
    compare_lines = [
        "# CAU Final SID Quality Compare",
        "",
        "Compared with the same SID export/eval logic used by Gate 0 and pilots.",
        "",
        markdown_table(table_rows, selected_cols),
        "",
        "## Formal CAU Output",
        "",
        "- checkpoint: `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt`",
        "- SID indices: `data/sid/poi_sid_indices_cau.npy`",
        "- SID mapping: `data/sid/poi_sid_mapping_cau.parquet`",
        "- SID report: `reports/sid_quality_cau.md`",
        "- experiment copy: `outputs/experiments/cau_rqvae/final_p1/poi_sid_indices_cau.npy`, `outputs/experiments/cau_rqvae/final_p1/poi_sid_mapping_cau.parquet`",
        "",
        "## Final Notes",
        "",
        "- P1 remains the promoted configuration; P2 is kept as a weight ablation only.",
        "- Final checkpoint selection used protected SID metrics before validation loss.",
        f"- Final qrels SID collision rate: {format_float(final_flat['qrels_sid_collision_rate'])}",
        f"- Final unique SID rate: {format_float(final_flat['unique_sid_rate'])}",
        "",
    ]
    (ROOT / "reports/sid_quality_cau_compare.md").write_text("\n".join(compare_lines), encoding="utf-8")


def write_checkpoint_selection_report(
    output_dir: Path,
    baseline: dict[str, float],
    sid_rows: list[dict[str, Any]],
    best_sid_row: dict[str, Any] | None,
) -> None:
    columns = [
        "epoch",
        "prefix1_semantic_purity",
        "prefix3_semantic_purity",
        "unique_sid_rate",
        "qrels_sid_collision_rate",
        "unique_pid_rate",
        "reconstruction_cosine",
        "min_codebook_usage_rate",
        "category_heldout_macro_f1",
        "passes_protection",
    ]
    lines = [
        "# CAU-RQ-VAE Checkpoint Selection",
        "",
        "## Protection Baseline",
        "",
    ]
    for key, value in baseline.items():
        lines.append(f"- {key}: {format_float(value, 8)}")
    lines.extend(["", "## SID Evaluation History", "", markdown_table(sid_rows, columns), ""])
    if best_sid_row is None:
        lines.extend(
            [
                "## Decision",
                "",
                "No evaluated checkpoint satisfied all protection conditions. Formal CAU SID export was not performed.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Decision",
                "",
                f"- selected_epoch: {best_sid_row['epoch']}",
                "- selected_checkpoint: `outputs/experiments/cau_rqvae/final_p1/best_sid_metrics.pt`",
                f"- qrels_sid_collision_rate: {format_float(best_sid_row['qrels_sid_collision_rate'])}",
                f"- unique_sid_rate: {format_float(best_sid_row['unique_sid_rate'])}",
                f"- prefix1_semantic_purity: {format_float(best_sid_row['prefix1_semantic_purity'])}",
                f"- category_heldout_macro_f1: {format_float(best_sid_row['category_heldout_macro_f1'])}",
                "",
            ]
        )
    (ROOT / "reports/cau_rqvae_checkpoint_selection.md").write_text("\n".join(lines), encoding="utf-8")


def group_summary(df: pd.DataFrame, poi_ids: set[str], key_col: str) -> dict[str, Any]:
    sub = df[df["poi_id"].astype(str).isin(poi_ids)].copy()
    if sub.empty:
        return {
            "dominant_key": "NA",
            "dominant_size": 0,
            "split_count": 0,
            "top_category": "NA",
            "purity": 0.0,
        }
    key_counts = sub[key_col].value_counts(dropna=False)
    dominant_key = str(key_counts.index[0])
    dominant = sub[sub[key_col] == key_counts.index[0]]
    cat_counts = dominant["semantic_label"].value_counts(dropna=False)
    return {
        "dominant_key": dominant_key,
        "dominant_size": int(len(dominant)),
        "split_count": int(sub[key_col].nunique()),
        "top_category": str(cat_counts.index[0]) if len(cat_counts) else "NA",
        "purity": float(cat_counts.iloc[0] / max(1, len(dominant))) if len(cat_counts) else 0.0,
    }


def select_cluster_cases(baseline: pd.DataFrame, final: pd.DataFrame) -> list[dict[str, Any]]:
    base = add_analysis_labels(baseline)
    fin = add_analysis_labels(final)
    cases: list[dict[str, Any]] = []

    def add_case(case_type: str, group: pd.DataFrame, key_col: str, expected: str) -> None:
        if group.empty:
            return
        poi_ids = set(group["poi_id"].astype(str))
        base_profile = group_summary(base, poi_ids, key_col)
        final_profile = group_summary(fin, poi_ids, "sid_str")
        reps = group[["poi_id", "name", "address", "semantic_label", "city"]].head(5).copy()
        rep_text = " | ".join(
            f"{r.poi_id}:{r.name}({r.semantic_label})" for r in reps.itertuples(index=False)
        )
        improved = final_profile["dominant_size"] < base_profile["dominant_size"]
        cases.append(
            {
                "case": case_type,
                "baseline_key": base_profile["dominant_key"],
                "final_dominant_sid": final_profile["dominant_key"],
                "baseline_cluster_size": base_profile["dominant_size"],
                "final_max_same_poi_bucket": final_profile["dominant_size"],
                "final_split_count": final_profile["split_count"],
                "baseline_top_category": base_profile["top_category"],
                "baseline_purity": base_profile["purity"],
                "final_top_category": final_profile["top_category"],
                "final_purity": final_profile["purity"],
                "representative_pois": rep_text,
                "improvement": "yes" if improved else "no",
                "trade_off": expected,
            }
        )

    prefix3 = base.groupby("sid_str", sort=False)
    good = (
        prefix3.filter(lambda g: 20 <= len(g) <= 120 and g["semantic_label"].value_counts(normalize=True).iloc[0] >= 0.9)
        .groupby("sid_str", sort=False)
    )
    if len(good):
        key = max(good.groups, key=lambda k: len(good.get_group(k)))
        add_case("good_category_cluster", good.get_group(key), "sid_str", "check whether good cluster remains coherent")

    prefix1 = base.groupby("sid_prefix1", sort=False)
    mixed_candidates = []
    for key, group in prefix1:
        if len(group) >= 500:
            purity = group["semantic_label"].value_counts(normalize=True).iloc[0]
            mixed_candidates.append((purity, len(group), key, group))
    if mixed_candidates:
        mixed_candidates.sort(key=lambda x: (x[0], -x[1]))
        add_case("severely_mixed_prefix1_cluster", mixed_candidates[0][3], "sid_prefix1", "still hard if same POIs remain split across many final SIDs")

    largest_key, largest_group = max(prefix3, key=lambda kv: len(kv[1]))
    add_case("largest_full_sid_collision", largest_group, "sid_str", "target is lower max bucket size")

    gov_mask = base["semantic_label"].eq("政府机构") | base["name"].astype(str).str.contains("政府|街道办|派出所|法院|政务", regex=True, na=False)
    gov_groups = []
    for key, group in base[gov_mask | base["sid_prefix1"].isin(base.loc[gov_mask, "sid_prefix1"])].groupby("sid_prefix1", sort=False):
        if len(group) >= 50 and gov_mask.loc[group.index].sum() >= 5:
            purity = group["semantic_label"].value_counts(normalize=True).iloc[0]
            gov_groups.append((purity, -len(group), key, group))
    if gov_groups:
        gov_groups.sort(key=lambda x: (x[0], x[1]))
        add_case("government_admin_mixed_case", gov_groups[0][3], "sid_prefix1", "administrative names can remain mixed with services/locations")

    brand_mask = base["name"].astype(str).str.contains("瑞幸|蜜雪冰城|星巴克|肯德基|麦当劳|中国石化|中石化|中石油", regex=True, na=False)
    brand_groups = []
    for key, group in base[brand_mask | base["sid_prefix2"].isin(base.loc[brand_mask, "sid_prefix2"])].groupby("sid_prefix2", sort=False):
        if len(group) >= 20 and brand_mask.loc[group.index].sum() >= 3:
            brand_groups.append((len(group), key, group))
    if brand_groups:
        brand_groups.sort(key=lambda x: -x[0])
        add_case("brand_or_chain_case", brand_groups[0][2], "sid_prefix2", "brand chains may improve collision but can preserve category ambiguity")

    return cases[:5]


def write_cluster_examples(output_dir: Path) -> None:
    baseline = pd.read_parquet(ROOT / "data/sid/poi_sid_mapping_semantic.parquet")
    final = pd.read_parquet(ROOT / "data/sid/poi_sid_mapping_cau.parquet")
    cases = select_cluster_cases(baseline, final)
    columns = [
        "case",
        "baseline_key",
        "final_dominant_sid",
        "baseline_cluster_size",
        "final_max_same_poi_bucket",
        "final_split_count",
        "baseline_top_category",
        "baseline_purity",
        "final_top_category",
        "final_purity",
        "improvement",
        "trade_off",
    ]
    lines = [
        "# CAU-RQ-VAE Cluster Examples",
        "",
        "The cases use baseline clusters and inspect the same POI sets under CAU Final P1. They are diagnostic examples, not cherry-picked success-only evidence.",
        "",
        markdown_table(cases, columns),
        "",
        "## Representative POIs",
        "",
    ]
    for case in cases:
        lines.append(f"### {case['case']}")
        lines.append(f"- representative_pois: {case['representative_pois']}")
        lines.append(f"- improvement: {case['improvement']}")
        lines.append(f"- trade_off: {case['trade_off']}")
        lines.append("")
    (ROOT / "reports/cau_rqvae_cluster_examples.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    output_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    best_sid_row: dict[str, Any] | None,
) -> None:
    lines = [
        "# CAU-RQ-VAE Final P1",
        "",
        "Final P1 run promoted from the pilot comparison.",
        "",
        "## Configuration",
        "",
        f"- lambda_tag: {config['cau']['lambda_tag']}",
        f"- lambda_unique: {config['cau']['lambda_unique']}",
        f"- unique_margin: {config['cau']['unique_margin']}",
        f"- seed: {config.get('seed', 42)}",
        f"- warmup_epochs: {config['final_training']['warmup_epochs']}",
        f"- max_total_epochs: {config['final_training']['max_total_epochs']}",
        f"- early_stop_patience: {config['final_training']['early_stop_patience']}",
        f"- evaluation_interval: {config['final_training']['evaluation_interval']}",
        f"- scheduler: none",
        "",
        "## Artifacts",
        "",
        "- `config.yaml`",
        "- `command.txt`",
        "- `environment.txt`",
        "- `train_log.csv`",
        "- `sid_eval_history.csv`",
        "- `metrics.json`",
        "- `best_val_loss.pt`",
        "- `best_sid_metrics.pt`",
        "- `last.pt`",
        "- `stdout.log`",
        "- `code_checksums.txt`",
        "",
    ]
    if best_sid_row is None:
        lines.extend(
            [
                "## Decision",
                "",
                "No checkpoint passed protection conditions. Formal CAU SID export was skipped.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Selected Checkpoint",
                "",
                f"- best_sid_epoch: {best_sid_row['epoch']}",
                "- best_sid_checkpoint: `best_sid_metrics.pt`",
                f"- qrels_sid_collision_rate: {format_float(best_sid_row['qrels_sid_collision_rate'])}",
                f"- unique_sid_rate: {format_float(best_sid_row['unique_sid_rate'])}",
                f"- prefix1_semantic_purity: {format_float(best_sid_row['prefix1_semantic_purity'])}",
                f"- category_heldout_macro_f1: {format_float(best_sid_row['category_heldout_macro_f1'])}",
                "",
            ]
        )
    lines.extend(["## Runtime Summary", "", f"```json\n{json.dumps(json_safe(summary), ensure_ascii=False, indent=2)}\n```", ""])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_experiment_report(summary: dict[str, Any], metrics_df: pd.DataFrame | None) -> None:
    lines = [
        "# CAU-RQ-VAE Experiment Report",
        "",
        "## Final P1 Status",
        "",
        f"- status: {summary['status']}",
        f"- output_dir: `{summary['output_dir']}`",
        f"- command: `{summary['command']}`",
        f"- device: `{summary['device']}`",
        f"- runtime_seconds: {format_float(summary['runtime_seconds'], 2)}",
        f"- best_val_loss_epoch: {summary.get('best_val_epoch')}",
        f"- best_sid_epoch: {summary.get('best_sid_epoch')}",
        f"- best_checkpoint: `{summary.get('best_sid_checkpoint', '')}`",
        "",
        "## Formal Result",
        "",
    ]
    final = summary.get("formal_metrics")
    if final:
        for key in [
            "prefix1_semantic_purity",
            "prefix2_semantic_purity",
            "prefix3_semantic_purity",
            "unique_sid_rate",
            "sid_collision_group_count",
            "max_pois_per_sid",
            "qrels_sid_collision_rate",
            "unique_pid_rate",
            "qrels_pid_collision_rate",
            "reconstruction_cosine",
            "category_heldout_macro_f1",
        ]:
            lines.append(f"- {key}: {format_float(final.get(key))}")
    else:
        lines.append("- Formal export was skipped because no checkpoint passed protection conditions.")
    if metrics_df is not None:
        selected_cols = [
            "run",
            "prefix1_semantic_purity",
            "prefix3_semantic_purity",
            "unique_sid_rate",
            "qrels_sid_collision_rate",
            "unique_pid_rate",
            "reconstruction_cosine",
            "category_heldout_macro_f1",
        ]
        lines.extend(["", "## Baseline/Pilot/Final Compare", "", markdown_table(metrics_df[selected_cols].to_dict("records"), selected_cols)])
    lines.extend(
        [
            "",
            "## Conclusion Wording",
            "",
            "Use the final measured values. Do not claim Prefix1 significant improvement, complete collision resolution, hierarchical category semantics, or full HiD-VAE/CQ-SID reproduction unless later evidence supports it.",
            "",
        ]
    )
    (ROOT / "reports/cau_rqvae_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    critical = [
        output_dir / "best_val_loss.pt",
        output_dir / "best_sid_metrics.pt",
        output_dir / "last.pt",
        output_dir / "metrics.json",
    ]
    guard_absent(critical, "final_p1 training artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = (output_dir / "stdout.log").open("w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    started = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    command = " ".join([sys.executable, *sys.argv])

    base_config = load_yaml(ROOT / args.config)
    config = merge_config(base_config, args)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_command(output_dir / "command.txt", args)
    write_checksums(output_dir / "code_checksums.txt")

    seed = int(config.get("seed", 42))
    set_random_seed(seed)
    device = resolve_device(str(config["rqvae"].get("device", "auto")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but torch.cuda.is_available() is False")
    write_environment(output_dir / "environment.txt", device, started_at)
    print(f"device: {device}")
    print(f"cuda_available: {torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    print(f"runtime: {cpu_thread_summary()}")

    data_cfg = config["data"]
    rqvae_cfg = config["rqvae"]
    cau_cfg = config["cau"]
    rqvae_input_path = ROOT / data_cfg["rqvae_input_semantic_npy"]
    x = np.load(rqvae_input_path, mmap_mode="r")
    n_rows = int(x.shape[0])
    input_summary = input_numeric_summary(np.asarray(x, dtype=np.float32))
    if input_summary["has_nan"] or input_summary["has_inf"]:
        raise ValueError(f"Invalid RQ-VAE input values: {input_summary}")

    embedding_meta = read_parquet(ROOT / "data/embeddings/poi_embedding_meta.parquet")
    geo_meta = read_parquet(ROOT / "data/geo/poi_geo_meta.parquet")
    validate_meta_alignment(embedding_meta, geo_meta, x)

    preprocess = load_preprocess(ROOT / data_cfg["preprocess_semantic_pkl"])
    train_indices = np.asarray(preprocess["train_indices"], dtype=np.int64)
    val_indices = np.asarray(preprocess["val_indices"], dtype=np.int64)
    labels = pd.read_parquet(ROOT / data_cfg["labels_parquet"])
    if len(labels) != n_rows:
        raise ValueError(f"Label rows ({len(labels)}) != input rows ({n_rows})")
    vocab = json.loads((ROOT / data_cfg["vocab_json"]).read_text(encoding="utf-8"))
    label_ids = labels["coarse_label_id"].to_numpy(dtype=np.int64)
    tag_train_mask = labels["is_tag_train"].to_numpy(dtype=bool)
    tag_heldout_indices = np.flatnonzero(labels["is_tag_heldout"].to_numpy(dtype=bool))
    train_index_set = set(int(v) for v in train_indices)
    collision_groups = CAU_TRAIN.build_collision_groups(ROOT / data_cfg["baseline_sid_mapping"], train_index_set, n_rows)

    print(f"rows={n_rows} train_rows={len(train_indices)} val_rows={len(val_indices)}")
    print(f"tag_train_rows={int(tag_train_mask.sum())} tag_heldout_rows={len(tag_heldout_indices)}")
    print(f"collision_group_count={len(collision_groups)}")

    model = CAURQVAE(
        input_dim=int(x.shape[1]),
        encoder_hidden_dims=[int(v) for v in rqvae_cfg["encoder_hidden_dims"]],
        latent_dim=int(rqvae_cfg["latent_dim"]),
        num_codebooks=int(rqvae_cfg["num_codebooks"]),
        codebook_size=int(rqvae_cfg["codebook_size"]),
        num_labels=len(vocab["vocab"]),
        commitment_beta=float(rqvae_cfg["commitment_beta"]),
        codebook_loss_weight=float(rqvae_cfg["codebook_loss_weight"]),
        dropout=float(rqvae_cfg.get("dropout", 0.05)),
        classifier_dropout=float(cau_cfg["classifier_dropout"]),
    ).to(device)
    init_info = CAU_TRAIN.load_init_checkpoint(model, ROOT / data_cfg["init_checkpoint"], device)

    lr = float(rqvae_cfg["lr"])
    weight_decay = float(rqvae_cfg["weight_decay"])
    batch_size = int(rqvae_cfg["batch_size"])
    set_rqvae_trainable(model, trainable=False)
    optimizer = make_optimizer(model, lr, weight_decay)
    phase = "warmup"

    baseline = load_baseline_metrics(ROOT / "reports/cau_pilot_metrics.csv")
    log_rows: list[dict[str, Any]] = []
    sid_rows: list[dict[str, Any]] = []
    best_sid_row: dict[str, Any] | None = None
    best_sid_key: tuple[float, float, float, float] | None = None
    best_val_loss = math.inf
    best_val_epoch = 0
    joint_epochs_without_improvement = 0
    rng = np.random.default_rng(seed)
    train_log_path = output_dir / "train_log.csv"
    sid_history_path = output_dir / "sid_eval_history.csv"
    best_train_row_for_sid: dict[str, Any] | None = None

    for epoch in range(1, int(args.max_total_epochs) + 1):
        if epoch == args.warmup_epochs + 1:
            phase = "joint"
            set_rqvae_trainable(model, trainable=True)
            optimizer = make_optimizer(model, lr, weight_decay)
            print("entering joint fine-tuning; optimizer reset with same AdamW lr/weight_decay")

        train_metrics = CAU_TRAIN.train_one_epoch(
            model=model,
            x=x,
            train_indices=train_indices,
            label_ids=label_ids,
            tag_train_mask=tag_train_mask,
            optimizer=optimizer,
            device=device,
            rng=rng,
            collision_groups=collision_groups,
            batch_size=batch_size,
            collision_sample_rate=float(cau_cfg["collision_sample_rate"]),
            max_pairs_per_group=int(cau_cfg["max_pairs_per_group_per_batch"]),
            lambda_tag=float(cau_cfg["lambda_tag"]),
            lambda_unique=float(cau_cfg["lambda_unique"]),
            unique_margin=float(cau_cfg["unique_margin"]),
            label_smoothing=float(cau_cfg["label_smoothing"]),
            grad_clip_norm=float(rqvae_cfg.get("grad_clip_norm", 0.0)),
            epoch=epoch,
        )
        val_metrics = CAU_TRAIN.evaluate_reconstruction(
            model,
            x,
            val_indices,
            device,
            batch_size,
            int(rqvae_cfg["num_codebooks"]),
            int(rqvae_cfg["codebook_size"]),
        )
        tag_metrics = CAU_TRAIN.evaluate_heldout_tags(
            model,
            x,
            tag_heldout_indices,
            label_ids,
            device,
            batch_size,
            len(vocab["vocab"]),
            float(cau_cfg["label_smoothing"]),
        )
        row = {
            "epoch": epoch,
            "phase": phase,
            **train_metrics,
            **val_metrics,
            **tag_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": float(time.time() - started),
        }
        log_rows.append(row)
        write_train_log(log_rows, train_log_path)

        improved = float(row["val_total_loss"]) < best_val_loss
        if improved:
            best_val_loss = float(row["val_total_loss"])
            best_val_epoch = epoch
            if phase == "joint":
                joint_epochs_without_improvement = 0
            save_training_checkpoint(
                output_dir / "best_val_loss.pt",
                model,
                optimizer,
                config,
                vocab,
                init_info,
                data_cfg,
                int(x.shape[1]),
                seed,
                epoch,
                best_val_epoch,
                best_val_loss,
                train_log_path,
                phase,
                extra={"selection_role": "best_val_loss", "active_rows": n_rows},
            )
        elif phase == "joint":
            joint_epochs_without_improvement += 1

        save_training_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            config,
            vocab,
            init_info,
            data_cfg,
            int(x.shape[1]),
            seed,
            epoch,
            best_val_epoch,
            best_val_loss,
            train_log_path,
            phase,
            extra={"selection_role": "last", "active_rows": n_rows},
        )

        stop_now = phase == "joint" and joint_epochs_without_improvement >= int(args.early_stop_patience)
        should_eval = epoch % int(args.evaluation_interval) == 0 or stop_now or epoch == int(args.max_total_epochs)
        if should_eval:
            print(f"running SID eval for epoch {epoch}")
            flat, _ = run_sid_eval(
                model=model,
                rqvae_input=x,
                embedding_meta=embedding_meta,
                geo_meta=geo_meta,
                device=device,
                batch_size=int(args.sid_batch_size),
                checkpoint_path=output_dir / "last.pt",
                report_dir=output_dir / "sid_eval" / f"epoch_{epoch:03d}",
                mode=f"cau_epoch_{epoch:03d}",
                rqvae_input_path=rqvae_input_path,
                max_report_groups=int(args.max_report_groups),
                train_row=row,
                epoch=epoch,
            )
            ok, failed = passes_protection(flat, baseline)
            flat["passes_protection"] = bool(ok)
            flat["failed_protection_checks"] = ",".join(failed)
            sid_rows.append(flat)
            pd.DataFrame(sid_rows).to_csv(sid_history_path, index=False, encoding="utf-8-sig")
            if ok:
                key = selection_key(flat)
                if best_sid_key is None or key < best_sid_key:
                    best_sid_key = key
                    best_sid_row = dict(flat)
                    best_train_row_for_sid = dict(row)
                    save_training_checkpoint(
                        output_dir / "best_sid_metrics.pt",
                        model,
                        optimizer,
                        config,
                        vocab,
                        init_info,
                        data_cfg,
                        int(x.shape[1]),
                        seed,
                        epoch,
                        best_val_epoch,
                        best_val_loss,
                        train_log_path,
                        phase,
                        extra={
                            "selection_role": "best_sid_metrics",
                            "active_rows": n_rows,
                            "best_sid_metrics": flat,
                            "baseline_protection_metrics": baseline,
                        },
                    )
                    print(f"new best_sid_metrics epoch={epoch} qrels_sid_collision={flat['qrels_sid_collision_rate']:.6f}")

        print(
            f"epoch={epoch} phase={phase} train_total={row['train_total_loss']:.6f} "
            f"base={row['train_base_total_loss']:.6f} tag_ratio={row['train_weighted_tag_to_base_ratio']:.4f} "
            f"unique_ratio={row['train_weighted_unique_to_base_ratio']:.4f} "
            f"val={row['val_total_loss']:.6f} recon_cos={row['val_recon_cosine']:.6f} "
            f"heldout_f1={row['tag_heldout_macro_f1']:.4f}"
        )
        if stop_now:
            print(f"early stopping after {joint_epochs_without_improvement} joint epochs without val improvement")
            break

    if not log_rows:
        raise RuntimeError("No training rows were logged")

    write_checkpoint_selection_report(output_dir, baseline, sid_rows, best_sid_row)
    formal_flat: dict[str, Any] | None = None
    metrics_df: pd.DataFrame | None = None
    if best_sid_row is not None and best_train_row_for_sid is not None:
        print("best_sid_metrics passed protection; exporting formal CAU SID")
        formal_flat, _ = export_formal_cau(
            checkpoint_path=output_dir / "best_sid_metrics.pt",
            output_dir=output_dir,
            config=config,
            device=device,
            sid_batch_size=int(args.sid_batch_size),
            max_report_groups=int(args.max_report_groups),
            train_row=best_train_row_for_sid,
        )
        metrics_df = append_final_metrics(ROOT / "reports/cau_pilot_metrics.csv", formal_flat, best_train_row_for_sid)
        write_compare_reports(metrics_df, formal_flat, output_dir)
        write_cluster_examples(output_dir)
    else:
        print("no checkpoint passed protection; formal CAU SID export skipped")

    shutil.copy2(train_log_path, ROOT / "reports/cau_rqvae_training_curves.csv")
    runtime_seconds = float(time.time() - started)
    summary = {
        "status": "success" if formal_flat is not None else "failed_no_protected_checkpoint",
        "output_dir": str(output_dir.relative_to(ROOT)),
        "command": command,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "runtime_seconds": runtime_seconds,
        "device": str(device),
        "actual_gpu": torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else "none",
        "best_val_epoch": best_val_epoch,
        "best_val_loss": best_val_loss,
        "best_val_checkpoint": str((output_dir / "best_val_loss.pt").relative_to(ROOT)),
        "best_sid_epoch": None if best_sid_row is None else int(best_sid_row["epoch"]),
        "best_sid_checkpoint": "" if best_sid_row is None else str((output_dir / "best_sid_metrics.pt").relative_to(ROOT)),
        "baseline_protection_metrics": baseline,
        "best_sid_metrics": best_sid_row,
        "formal_metrics": formal_flat,
        "train_epochs_completed": int(log_rows[-1]["epoch"]),
        "early_stop_patience": int(args.early_stop_patience),
        "evaluation_interval": int(args.evaluation_interval),
    }
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, config, summary, best_sid_row)
    write_experiment_report(summary, metrics_df)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

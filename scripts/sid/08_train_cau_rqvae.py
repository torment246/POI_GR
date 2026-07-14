#!/usr/bin/env python3
"""Train CAU-RQ-VAE smoke and pilot runs without touching baseline artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rqvae import CAURQVAE  # noqa: E402
from rqvae_preprocess import input_numeric_summary, load_preprocess  # noqa: E402
from train_utils import cpu_thread_summary, load_yaml, resolve_device, save_checkpoint, set_random_seed, write_train_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rqvae_cau.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--debug-max-rows", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lambda-tag", type=float, default=None)
    parser.add_argument("--lambda-unique", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).strip().split())


def normalized_signature(row: pd.Series) -> tuple[str, str, int, int]:
    name = re.sub(r"\s+", "", clean_text(row.get("name", ""))).casefold()
    address = re.sub(r"\s+", "", clean_text(row.get("address", ""))).casefold()
    lat = row.get("lat", np.nan)
    lon = row.get("lon", np.nan)
    lat_key = -10**12 if pd.isna(lat) else int(round(float(lat) * 1_000_000))
    lon_key = -10**12 if pd.isna(lon) else int(round(float(lon) * 1_000_000))
    return name, address, lat_key, lon_key


def build_collision_groups(
    mapping_path: Path,
    train_index_set: set[int],
    n_rows: int,
    min_group_size: int = 2,
) -> list[np.ndarray]:
    mapping = pd.read_parquet(mapping_path)
    if len(mapping) < n_rows:
        raise ValueError(f"mapping rows ({len(mapping)}) < active rows ({n_rows})")
    mapping = mapping.iloc[:n_rows].copy()
    required = {"row_id", "sid_str", "name", "address", "lat", "lon"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"baseline SID mapping missing required columns: {missing}")

    groups: list[np.ndarray] = []
    for _, group in mapping.groupby("sid_str", sort=False):
        row_indices = [int(idx) for idx in group.index.to_numpy() if int(idx) in train_index_set]
        if len(row_indices) < min_group_size:
            continue
        sub = mapping.iloc[row_indices].copy()
        sub["_duplicate_signature"] = sub.apply(normalized_signature, axis=1)
        sub = sub.drop_duplicates("_duplicate_signature", keep="first")
        indices = sub.index.to_numpy(dtype=np.int64)
        if len(indices) >= min_group_size:
            groups.append(indices)
    return groups


def load_vocab(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_init_checkpoint(model: CAURQVAE, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    disallowed_missing = [key for key in missing if not key.startswith("tag_classifier.")]
    if disallowed_missing or unexpected:
        raise ValueError(f"Unexpected init load mismatch: missing={missing}, unexpected={unexpected}")
    return {
        "init_checkpoint": str(checkpoint_path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "init_best_epoch": checkpoint.get("best_epoch"),
        "init_best_val_loss": checkpoint.get("best_val_loss"),
    }


def make_batch(
    base_chunk: np.ndarray,
    rng: np.random.Generator,
    collision_groups: list[np.ndarray],
    pair_budget: int,
    max_pairs_per_group: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    batch: list[int] = [int(v) for v in base_chunk]
    pairs: list[tuple[int, int]] = []
    remaining = int(pair_budget)
    attempts = 0
    while remaining > 0 and collision_groups and attempts < pair_budget * 20:
        attempts += 1
        group = collision_groups[int(rng.integers(0, len(collision_groups)))]
        if len(group) < 2:
            continue
        group_pairs = min(max_pairs_per_group, remaining)
        for _ in range(group_pairs):
            a, b = rng.choice(group, size=2, replace=False)
            pos_a = len(batch)
            batch.append(int(a))
            pos_b = len(batch)
            batch.append(int(b))
            pairs.append((pos_a, pos_b))
            remaining -= 1
            if remaining <= 0:
                break
    return np.asarray(batch, dtype=np.int64), pairs


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> float:
    scores: list[float] = []
    for label_id in range(num_labels):
        tp = int(((y_true == label_id) & (y_pred == label_id)).sum())
        fp = int(((y_true != label_id) & (y_pred == label_id)).sum())
        fn = int(((y_true == label_id) & (y_pred != label_id)).sum())
        support = tp + fn
        if support == 0:
            continue
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if support == 0 else tp / support
        score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def uniqueness_loss(
    h: torch.Tensor,
    indices: torch.Tensor,
    pairs: list[tuple[int, int]],
    margin: float,
) -> tuple[torch.Tensor, int, int, float]:
    if not pairs:
        return h.new_tensor(0.0), 0, 0, 0.0
    losses: list[torch.Tensor] = []
    active = 0
    valid = 0
    cosine_sum = 0.0
    denom = max(1e-8, (1.0 - float(margin)) ** 2)
    h_norm = F.normalize(h, dim=1, eps=1e-8)
    for pos_a, pos_b in pairs:
        if pos_a >= indices.shape[0] or pos_b >= indices.shape[0]:
            continue
        valid += 1
        cos = torch.sum(h_norm[pos_a] * h_norm[pos_b])
        cosine_sum += float(cos.detach().item())
        if bool(torch.equal(indices[pos_a], indices[pos_b])):
            active += 1
            losses.append(torch.relu(cos - margin).pow(2) / denom)
    if not losses:
        return h.new_tensor(0.0), valid, 0, cosine_sum
    return torch.stack(losses).mean(), valid, active, cosine_sum


def train_one_epoch(
    model: CAURQVAE,
    x: np.ndarray,
    train_indices: np.ndarray,
    label_ids: np.ndarray,
    tag_train_mask: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: np.random.Generator,
    collision_groups: list[np.ndarray],
    batch_size: int,
    collision_sample_rate: float,
    max_pairs_per_group: int,
    lambda_tag: float,
    lambda_unique: float,
    unique_margin: float,
    label_smoothing: float,
    grad_clip_norm: float,
    epoch: int,
) -> dict[str, float]:
    model.train()
    pair_budget = int(round(batch_size * collision_sample_rate / 2.0))
    pair_budget = max(0, pair_budget)
    base_count = max(1, batch_size - pair_budget * 2)
    perm = rng.permutation(train_indices)

    sums = {
        "total_loss": 0.0,
        "base_total_loss": 0.0,
        "recon_loss": 0.0,
        "commitment_loss": 0.0,
        "codebook_loss": 0.0,
        "tag_loss_weighted_batch": 0.0,
        "unique_loss_weighted_batch": 0.0,
        "tag_loss_by_label": 0.0,
        "unique_loss_by_pair": 0.0,
    }
    seen = 0
    tag_seen = 0
    tag_correct = 0
    pair_candidates = 0
    active_pairs = 0
    pair_cosine_sum = 0.0
    batch_count = 0
    zero_pair_batches = 0
    progress = tqdm(range(0, len(perm), base_count), desc=f"cau train epoch {epoch}", leave=False)
    for start in progress:
        base_chunk = perm[start : start + base_count]
        batch_indices, pairs = make_batch(base_chunk, rng, collision_groups, pair_budget, max_pairs_per_group)
        batch_x = torch.from_numpy(np.asarray(x[batch_indices], dtype=np.float32).copy()).to(device)
        batch_label_ids = torch.from_numpy(label_ids[batch_indices].astype(np.int64, copy=True)).to(device)
        batch_tag_mask = torch.from_numpy(tag_train_mask[batch_indices].astype(bool, copy=True)).to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_x)
        tag_loss = batch_x.new_tensor(0.0)
        if bool(batch_tag_mask.any()):
            tag_loss = F.cross_entropy(
                outputs["tag_logits"][batch_tag_mask],
                batch_label_ids[batch_tag_mask],
                label_smoothing=label_smoothing,
            )
            tag_pred = torch.argmax(outputs["tag_logits"][batch_tag_mask], dim=1)
            tag_correct += int((tag_pred == batch_label_ids[batch_tag_mask]).sum().item())
        unique, pair_count, active_count, pair_cos_sum = uniqueness_loss(
            outputs["h"],
            outputs["indices"],
            pairs,
            unique_margin,
        )
        total_loss = outputs["total_loss"] + lambda_tag * tag_loss + lambda_unique * unique
        total_loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_n = int(batch_x.shape[0])
        seen += batch_n
        tag_n = int(batch_tag_mask.sum().item())
        tag_seen += tag_n
        pair_candidates += int(pair_count)
        active_pairs += int(active_count)
        pair_cosine_sum += float(pair_cos_sum)
        batch_count += 1
        if int(pair_count) == 0:
            zero_pair_batches += 1
        sums["total_loss"] += float(total_loss.detach().item()) * batch_n
        sums["base_total_loss"] += float(outputs["total_loss"].detach().item()) * batch_n
        sums["recon_loss"] += float(outputs["recon_loss"].detach().item()) * batch_n
        sums["commitment_loss"] += float(outputs["commitment_loss"].detach().item()) * batch_n
        sums["codebook_loss"] += float(outputs["codebook_loss"].detach().item()) * batch_n
        sums["tag_loss_weighted_batch"] += float((lambda_tag * tag_loss).detach().item()) * batch_n
        sums["unique_loss_weighted_batch"] += float((lambda_unique * unique).detach().item()) * batch_n
        if tag_n:
            sums["tag_loss_by_label"] += float(tag_loss.detach().item()) * tag_n
        if active_count:
            sums["unique_loss_by_pair"] += float(unique.detach().item()) * active_count

    denom = max(1, seen)
    metrics = {
        "train_total_loss": sums["total_loss"] / denom,
        "train_base_total_loss": sums["base_total_loss"] / denom,
        "train_recon_loss": sums["recon_loss"] / denom,
        "train_commitment_loss": sums["commitment_loss"] / denom,
        "train_codebook_loss": sums["codebook_loss"] / denom,
        "train_tag_loss": sums["tag_loss_by_label"] / max(1, tag_seen),
        "train_weighted_tag_loss": sums["tag_loss_weighted_batch"] / denom,
        "train_unique_loss": sums["unique_loss_by_pair"] / max(1, active_pairs),
        "train_weighted_unique_loss": sums["unique_loss_weighted_batch"] / denom,
        "train_tag_examples": float(tag_seen),
        "train_tag_accuracy": float(tag_correct / max(1, tag_seen)),
        "train_unique_candidate_pairs": float(pair_candidates),
        "train_valid_collision_pair_count": float(pair_candidates),
        "train_mean_valid_collision_pairs_per_batch": float(pair_candidates / max(1, batch_count)),
        "train_zero_pair_batch_rate": float(zero_pair_batches / max(1, batch_count)),
        "train_unique_active_pairs": float(active_pairs),
        "train_active_margin_pair_count": float(active_pairs),
        "train_active_margin_pair_rate": float(active_pairs / max(1, pair_candidates)),
        "train_collision_pair_mean_cosine": float(pair_cosine_sum / max(1, pair_candidates)),
        "train_collision_sample_rate": float((2 * pair_candidates) / max(1, seen)),
    }
    metrics["train_weighted_tag_to_base_ratio"] = metrics["train_weighted_tag_loss"] / max(1e-8, metrics["train_base_total_loss"])
    metrics["train_weighted_unique_to_base_ratio"] = metrics["train_weighted_unique_loss"] / max(1e-8, metrics["train_base_total_loss"])
    return metrics


@torch.no_grad()
def evaluate_reconstruction(
    model: CAURQVAE,
    x: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_codebooks: int,
    codebook_size: int,
) -> dict[str, float]:
    model.eval()
    sums = {"total": 0.0, "recon": 0.0, "commitment": 0.0, "codebook": 0.0, "cos": 0.0}
    used: list[set[int]] = [set() for _ in range(num_codebooks)]
    unique_sids: set[tuple[int, ...]] = set()
    seen = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_x = torch.from_numpy(np.asarray(x[batch_indices], dtype=np.float32).copy()).to(device)
        outputs = model(batch_x)
        batch_n = int(batch_x.shape[0])
        seen += batch_n
        sums["total"] += float(outputs["total_loss"].item()) * batch_n
        sums["recon"] += float(outputs["recon_loss"].item()) * batch_n
        sums["commitment"] += float(outputs["commitment_loss"].item()) * batch_n
        sums["codebook"] += float(outputs["codebook_loss"].item()) * batch_n
        sums["cos"] += float(F.cosine_similarity(outputs["x_hat"], batch_x, dim=1, eps=1e-8).sum().item())
        batch_codes = outputs["indices"].detach().cpu().numpy()
        for level in range(num_codebooks):
            used[level].update(int(v) for v in np.unique(batch_codes[:, level]))
        unique_sids.update(tuple(int(v) for v in row) for row in batch_codes)
    denom = max(1, seen)
    metrics = {
        "val_total_loss": sums["total"] / denom,
        "val_recon_loss": sums["recon"] / denom,
        "val_commitment_loss": sums["commitment"] / denom,
        "val_codebook_loss": sums["codebook"] / denom,
        "val_recon_cosine": sums["cos"] / denom,
    }
    for level in range(num_codebooks):
        count = len(used[level])
        metrics[f"codebook{level}_used"] = float(count)
        metrics[f"codebook{level}_usage_rate"] = float(count / codebook_size)
    metrics["val_online_unique_sid_rate"] = float(len(unique_sids) / max(1, seen))
    return metrics


@torch.no_grad()
def evaluate_heldout_tags(
    model: CAURQVAE,
    x: np.ndarray,
    heldout_indices: np.ndarray,
    label_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_labels: int,
    label_smoothing: float,
) -> dict[str, float]:
    model.eval()
    if len(heldout_indices) == 0:
        return {"tag_heldout_loss": 0.0, "tag_heldout_accuracy": 0.0, "tag_heldout_macro_f1": 0.0}
    losses = 0.0
    seen = 0
    pred_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    for start in range(0, len(heldout_indices), batch_size):
        batch_indices = heldout_indices[start : start + batch_size]
        batch_x = torch.from_numpy(np.asarray(x[batch_indices], dtype=np.float32).copy()).to(device)
        y = torch.from_numpy(label_ids[batch_indices].astype(np.int64, copy=True)).to(device)
        outputs = model(batch_x)
        loss = F.cross_entropy(outputs["tag_logits"], y, label_smoothing=label_smoothing)
        pred = torch.argmax(outputs["tag_logits"], dim=1)
        batch_n = int(batch_x.shape[0])
        seen += batch_n
        losses += float(loss.item()) * batch_n
        pred_parts.append(pred.cpu().numpy())
        true_parts.append(y.cpu().numpy())
    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    return {
        "tag_heldout_loss": losses / max(1, seen),
        "tag_heldout_accuracy": float((y_true == y_pred).mean()),
        "tag_heldout_macro_f1": macro_f1(y_true, y_pred, num_labels),
    }


def build_report(
    args: argparse.Namespace,
    output_dir: Path,
    config: dict[str, Any],
    init_info: dict[str, Any],
    input_summary: dict[str, Any],
    collision_group_count: int,
    log_rows: list[dict[str, Any]],
    best_epoch: int,
    best_val_loss: float,
    loss_scale_status: str,
    loss_scale_notes: list[str],
) -> str:
    final = log_rows[-1]
    lines = [
        "# CAU-RQ-VAE Training Report",
        "",
        f"- run_name: {args.run_name or output_dir.name}",
        f"- output_dir: `{output_dir}`",
        f"- config: `{args.config}`",
        f"- debug_max_rows: {args.debug_max_rows}",
        f"- init_checkpoint: `{init_info['init_checkpoint']}`",
        f"- init_best_epoch: {init_info.get('init_best_epoch')}",
        f"- init_best_val_loss: {init_info.get('init_best_val_loss')}",
        f"- input_has_nan: {input_summary['has_nan']}",
        f"- input_has_inf: {input_summary['has_inf']}",
        f"- collision_group_count: {collision_group_count}",
        f"- best_epoch: {best_epoch}",
        f"- best_val_loss: {best_val_loss:.8f}",
        f"- final_val_recon_cosine: {float(final['val_recon_cosine']):.8f}",
        f"- loss_scale_status: `{loss_scale_status}`",
        "",
        "## CAU Weights",
        "",
        f"- lambda_tag: {config['cau']['lambda_tag']}",
        f"- lambda_unique: {config['cau']['lambda_unique']}",
        f"- unique_margin: {config['cau']['unique_margin']}",
        f"- label_smoothing: {config['cau']['label_smoothing']}",
        f"- classifier_dropout: {config['cau']['classifier_dropout']}",
        "",
        "## Final Metrics",
        "",
    ]
    for key in [
        "train_total_loss",
        "train_base_total_loss",
        "train_tag_loss",
        "train_weighted_tag_loss",
        "train_weighted_tag_to_base_ratio",
        "train_unique_loss",
        "train_weighted_unique_loss",
        "train_weighted_unique_to_base_ratio",
        "train_valid_collision_pair_count",
        "train_mean_valid_collision_pairs_per_batch",
        "train_zero_pair_batch_rate",
        "train_active_margin_pair_count",
        "train_active_margin_pair_rate",
        "train_collision_pair_mean_cosine",
        "train_tag_accuracy",
        "train_unique_candidate_pairs",
        "train_unique_active_pairs",
        "val_total_loss",
        "val_recon_cosine",
        "val_online_unique_sid_rate",
        "tag_heldout_loss",
        "tag_heldout_accuracy",
        "tag_heldout_macro_f1",
    ]:
        if key in final:
            value = final[key]
            lines.append(f"- {key}: {float(value):.8f}")
    lines.extend(["", "## Loss Scale Notes", ""])
    lines.extend(f"- {note}" for note in loss_scale_notes)
    lines.extend(["", "## Artifacts", ""])
    for name in ["checkpoint.pt", "last.pt", "train_log.csv", "report.md", "config_snapshot.json"]:
        lines.append(f"- `{output_dir / name}`")
    lines.append("")
    return "\n".join(lines)


def merged_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = json.loads(json.dumps(config))
    if args.epochs is not None:
        out["rqvae"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        out["rqvae"]["batch_size"] = int(args.batch_size)
    if args.device is not None:
        out["rqvae"]["device"] = args.device
    if args.lambda_tag is not None:
        out["cau"]["lambda_tag"] = float(args.lambda_tag)
    if args.lambda_unique is not None:
        out["cau"]["lambda_unique"] = float(args.lambda_unique)
    return out


def loss_scale_check(config: dict[str, Any], first_row: dict[str, Any]) -> tuple[str, list[str]]:
    tag_range = config["cau"].get("tag_loss_ratio_range", [0.05, 0.20])
    unique_range = config["cau"].get("unique_loss_ratio_range", [0.05, 0.15])
    tag_ratio = float(first_row.get("train_weighted_tag_to_base_ratio", 0.0))
    unique_ratio = float(first_row.get("train_weighted_unique_to_base_ratio", 0.0))
    notes = [
        f"first_epoch_weighted_tag_to_base_ratio={tag_ratio:.6f}, target=[{tag_range[0]}, {tag_range[1]}]",
        f"first_epoch_weighted_unique_to_base_ratio={unique_ratio:.6f}, target=[{unique_range[0]}, {unique_range[1]}]",
    ]
    ok = float(tag_range[0]) <= tag_ratio <= float(tag_range[1]) and float(unique_range[0]) <= unique_ratio <= float(unique_range[1])
    if ok:
        notes.append("loss scale is within the requested initial target range.")
        return "OK", notes
    notes.append("loss scale is outside the requested initial target range; do not proceed to a longer run without review.")
    return "WARN", notes


def main() -> None:
    args = parse_args()
    config = merged_config(load_yaml(Path(args.config)), args)
    seed = int(config.get("seed", 42))
    set_random_seed(seed)
    rng = np.random.default_rng(seed)

    data_cfg = config["data"]
    rqvae_cfg = config["rqvae"]
    cau_cfg = config["cau"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    x = np.load(data_cfg["rqvae_input_semantic_npy"], mmap_mode="r")
    n_rows = int(x.shape[0])
    if args.debug_max_rows is not None:
        n_rows = min(n_rows, int(args.debug_max_rows))
    active_row_mask = np.arange(n_rows, dtype=np.int64)
    input_summary = input_numeric_summary(np.asarray(x[:n_rows], dtype=np.float32))
    if input_summary["has_nan"] or input_summary["has_inf"]:
        raise ValueError(f"Invalid RQ-VAE input values: {input_summary}")

    preprocess = load_preprocess(Path(data_cfg["preprocess_semantic_pkl"]))
    train_indices = np.asarray(preprocess["train_indices"], dtype=np.int64)
    val_indices = np.asarray(preprocess["val_indices"], dtype=np.int64)
    train_indices = train_indices[train_indices < n_rows]
    val_indices = val_indices[val_indices < n_rows]
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError(f"Empty train/val split after debug filter: train={len(train_indices)}, val={len(val_indices)}")

    labels = pd.read_parquet(data_cfg["labels_parquet"]).iloc[:n_rows].copy()
    vocab = load_vocab(Path(data_cfg["vocab_json"]))
    label_ids = labels["coarse_label_id"].to_numpy(dtype=np.int64)
    tag_train_mask = labels["is_tag_train"].to_numpy(dtype=bool)
    tag_heldout_indices = np.flatnonzero(labels["is_tag_heldout"].to_numpy(dtype=bool))
    if len(labels) != n_rows:
        raise ValueError(f"Label rows ({len(labels)}) != active rows ({n_rows})")

    train_index_set = set(int(v) for v in train_indices)
    collision_groups = build_collision_groups(Path(data_cfg["baseline_sid_mapping"]), train_index_set, n_rows)

    device = resolve_device(str(rqvae_cfg.get("device", "auto")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    print(f"device: {device}")
    print(f"runtime: {cpu_thread_summary()}")
    print(f"active_rows: {n_rows} train_rows: {len(train_indices)} val_rows: {len(val_indices)}")
    print(f"tag_train_rows: {int(tag_train_mask.sum())} tag_heldout_rows: {len(tag_heldout_indices)}")
    print(f"collision_group_count: {len(collision_groups)}")

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
    init_info = load_init_checkpoint(model, Path(data_cfg["init_checkpoint"]), device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(rqvae_cfg["lr"]),
        weight_decay=float(rqvae_cfg["weight_decay"]),
    )
    epochs = int(rqvae_cfg["epochs"])
    batch_size = int(rqvae_cfg["batch_size"])
    best_val_loss = math.inf
    best_epoch = 0
    log_rows: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    patience = int(rqvae_cfg.get("early_stop_patience", epochs + 1))

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
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
        val_metrics = evaluate_reconstruction(
            model,
            x,
            val_indices,
            device,
            batch_size,
            int(rqvae_cfg["num_codebooks"]),
            int(rqvae_cfg["codebook_size"]),
        )
        tag_metrics = evaluate_heldout_tags(
            model,
            x,
            tag_heldout_indices,
            label_ids,
            device,
            batch_size,
            len(vocab["vocab"]),
            float(cau_cfg["label_smoothing"]),
        )
        row = {"epoch": epoch, **train_metrics, **val_metrics, **tag_metrics, "lr": float(optimizer.param_groups[0]["lr"])}
        log_rows.append(row)
        write_train_log(log_rows, output_dir / "train_log.csv")

        improved = float(row["val_total_loss"]) < best_val_loss
        if improved:
            best_val_loss = float(row["val_total_loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "checkpoint.pt",
                model,
                optimizer,
                mode="cau",
                input_dim=int(x.shape[1]),
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                train_log_path=output_dir / "train_log.csv",
                preprocess_path=Path(data_cfg["preprocess_semantic_pkl"]),
                rqvae_input_path=Path(data_cfg["rqvae_input_semantic_npy"]),
                random_seed=seed,
                epoch=epoch,
                extra={
                    "architecture": "cau_rqvae",
                    "config": config,
                    "cau_vocab": vocab,
                    "init_info": init_info,
                    "active_rows": n_rows,
                    "debug_max_rows": args.debug_max_rows,
                },
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            mode="cau",
            input_dim=int(x.shape[1]),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            train_log_path=output_dir / "train_log.csv",
            preprocess_path=Path(data_cfg["preprocess_semantic_pkl"]),
            rqvae_input_path=Path(data_cfg["rqvae_input_semantic_npy"]),
            random_seed=seed,
            epoch=epoch,
            extra={
                "architecture": "cau_rqvae",
                "config": config,
                "cau_vocab": vocab,
                "init_info": init_info,
                "active_rows": n_rows,
                "debug_max_rows": args.debug_max_rows,
            },
        )

        print(
            f"epoch={epoch} train_total={row['train_total_loss']:.6f} "
            f"base={row['train_base_total_loss']:.6f} tag_w={row['train_weighted_tag_loss']:.6f} "
            f"uniq_w={row['train_weighted_unique_loss']:.6f} "
            f"tag_ratio={row['train_weighted_tag_to_base_ratio']:.4f} "
            f"uniq_ratio={row['train_weighted_unique_to_base_ratio']:.4f} "
            f"val={row['val_total_loss']:.6f} heldout_f1={row['tag_heldout_macro_f1']:.4f}"
        )
        if epochs_without_improvement >= patience:
            break

    if not log_rows:
        raise RuntimeError("No training rows logged")
    loss_scale_status, loss_scale_notes = loss_scale_check(config, log_rows[0])
    report = build_report(
        args,
        output_dir,
        config,
        init_info,
        input_summary,
        len(collision_groups),
        log_rows,
        best_epoch,
        best_val_loss,
        loss_scale_status,
        loss_scale_notes,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "best_epoch": best_epoch, "best_val_loss": best_val_loss, "loss_scale_status": loss_scale_status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

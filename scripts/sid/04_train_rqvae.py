#!/usr/bin/env python3
"""Train semantic or geo-fused RQ-VAE models for MobilityBench POIs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required. Please install torch: pip install torch") from exc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rqvae import RQVAE  # noqa: E402
from rqvae_preprocess import input_numeric_summary, output_path_for_mode, prepare_rqvae_input  # noqa: E402
from train_utils import (  # noqa: E402
    append_debug_suffix,
    cpu_thread_summary,
    evaluate,
    load_yaml,
    make_loader,
    resolve_device,
    save_checkpoint,
    set_random_seed,
    train_one_epoch,
    write_train_log,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rqvae_train.yaml", help="Training YAML config.")
    parser.add_argument("--mode", choices=["semantic", "geo_fused"], required=True, help="RQ-VAE training mode.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total target epoch.")
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=None,
        help="Do not early stop before this global epoch, useful when resuming to a required milestone.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--debug-max-rows", type=int, default=None, help="Use first N rows and write *_debug outputs.")
    parser.add_argument("--force-rebuild-input", action="store_true", help="Rebuild PCA/scaler and RQ-VAE input npy.")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint.")
    parser.add_argument(
        "--resume-from",
        default="last",
        help="Checkpoint to resume from: last, best, or an explicit checkpoint path.",
    )
    return parser.parse_args()


def mode_input_config_key(mode: str) -> str:
    return "rqvae_input_semantic_npy" if mode == "semantic" else "rqvae_input_geo_fused_npy"


def output_paths(mode: str, debug_max_rows: int | None) -> dict[str, Path]:
    suffix = f"{mode}_debug" if debug_max_rows else mode
    return {
        "checkpoint": Path(f"outputs/rqvae/rqvae_{suffix}.pt"),
        "last_checkpoint": Path(f"outputs/rqvae/rqvae_{suffix}_last.pt"),
        "train_log": Path(f"outputs/rqvae/train_log_{suffix}.csv"),
        "report": Path(f"outputs/rqvae/rqvae_train_report_{suffix}.md"),
        "preprocess": Path(f"outputs/rqvae/preprocess_{suffix}.pkl"),
    }


def merged_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rqvae_cfg = dict(config.get("rqvae", {}))
    if args.epochs is not None:
        rqvae_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        rqvae_cfg["batch_size"] = int(args.batch_size)
    if args.device is not None:
        rqvae_cfg["device"] = args.device
    out = dict(config)
    out["rqvae"] = rqvae_cfg
    return out


def validate_config(config: dict[str, Any]) -> None:
    rqvae_cfg = config["rqvae"]
    if rqvae_cfg.get("recon_loss", "mse") != "mse":
        raise ValueError("Only recon_loss=mse is supported in this stage")
    if int(rqvae_cfg["epochs"]) <= 0:
        raise ValueError(f"epochs must be positive, got {rqvae_cfg['epochs']}")
    if int(rqvae_cfg["batch_size"]) <= 0:
        raise ValueError(f"batch_size must be positive, got {rqvae_cfg['batch_size']}")


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def resolve_resume_checkpoint(resume_from: str, paths: dict[str, Path]) -> Path:
    if resume_from == "last":
        return paths["last_checkpoint"]
    if resume_from == "best":
        return paths["checkpoint"]
    return Path(resume_from)


def load_existing_train_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to resume existing train logs. Please install pandas.") from exc
    return pd.read_csv(path).to_dict("records")


def load_resume_state(
    checkpoint_path: Path,
    model: RQVAE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str,
    input_dim: int,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("mode") != mode:
        raise ValueError(f"Checkpoint mode mismatch: expected {mode}, got {checkpoint.get('mode')}")
    if int(checkpoint.get("input_dim", -1)) != int(input_dim):
        raise ValueError(f"Checkpoint input_dim mismatch: expected {input_dim}, got {checkpoint.get('input_dim')}")
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def final_codebook_usage(row: dict[str, Any], num_codebooks: int, codebook_size: int) -> list[str]:
    lines = []
    for level in range(num_codebooks):
        used = int(row.get(f"codebook{level}_used", 0))
        rate = float(row.get(f"codebook{level}_usage_rate", 0.0))
        dead = codebook_size - used
        lines.append(f"- codebook{level}_used / rate: {used} / {rate:.4f}")
        lines.append(f"- codebook{level}_dead_code_count: {dead}")
    return lines


def diagnostics(
    mode: str,
    log_rows: list[dict[str, Any]],
    num_codebooks: int,
    semantic_log_path: Path,
) -> list[str]:
    if not log_rows:
        return ["- no training log rows"]
    first = log_rows[0]
    final = log_rows[-1]
    notes: list[str] = []
    for level in range(num_codebooks):
        rate = float(final.get(f"codebook{level}_usage_rate", 0.0))
        if rate < 0.5:
            notes.append(f"- codebook{level} usage rate is below 0.5; possible codebook collapse or insufficient training.")
    if float(final.get("val_recon_cosine", 0.0)) < 0.7:
        notes.append("- val_recon_cosine is below 0.7; reconstruction quality may be weak.")
    if float(final.get("train_total_loss", 0.0)) < float(first.get("train_total_loss", 0.0)) and float(
        final.get("val_total_loss", 0.0)
    ) > float(first.get("val_total_loss", 0.0)):
        notes.append("- train loss decreased while val loss increased; possible overfitting.")
    if mode == "geo_fused" and semantic_log_path.exists():
        try:
            import pandas as pd

            semantic_log = pd.read_csv(semantic_log_path)
            semantic_best = float(semantic_log["val_total_loss"].min())
            geo_best = min(float(row["val_total_loss"]) for row in log_rows)
            if geo_best > semantic_best * 1.5:
                notes.append("- geo_fused best loss is much higher than semantic; check geo_alpha or input feature scale.")
        except Exception as exc:  # pragma: no cover - diagnostic only
            notes.append(f"- could not compare geo_fused with semantic log: {exc}")
    if not notes:
        notes.append("- no obvious training pathology detected.")
    return notes


def build_report(
    mode: str,
    input_rows: int,
    input_dim: int,
    train_rows: int,
    val_rows: int,
    text_pca_dim: int,
    geo_alpha: float | None,
    checkpoint_path: Path,
    preprocess_path: Path,
    rqvae_input_path: Path,
    input_summary: dict[str, Any],
    best_epoch: int,
    best_val_loss: float,
    log_rows: list[dict[str, Any]],
    early_stopped: bool,
    num_codebooks: int,
    codebook_size: int,
    semantic_log_path: Path,
    resumed_from_checkpoint: Path | None = None,
    resumed_from_epoch: int | None = None,
    min_epochs: int | None = None,
) -> str:
    final = log_rows[-1]
    lines = [
        "# RQ-VAE Training Report",
        "",
        "## Basic Information",
        f"- mode: {mode}",
        f"- input_rows: {input_rows}",
        f"- input_dim: {input_dim}",
        f"- train_rows: {train_rows}",
        f"- val_rows: {val_rows}",
        f"- text_pca_dim: {text_pca_dim}",
    ]
    if mode == "geo_fused":
        lines.append(f"- geo_alpha: {geo_alpha}")
    if resumed_from_checkpoint is not None:
        lines.append(f"- resumed_from_checkpoint: {resumed_from_checkpoint}")
        lines.append(f"- resumed_from_epoch: {resumed_from_epoch}")
    if min_epochs is not None:
        lines.append(f"- min_epochs_before_early_stop: {min_epochs}")
    lines.extend(
        [
            f"- checkpoint_path: {checkpoint_path}",
            f"- preprocess_path: {preprocess_path}",
            f"- rqvae_input_path: {rqvae_input_path}",
            "",
            "## Input Numeric Checks",
            f"- has_nan: {input_summary['has_nan']}",
            f"- has_inf: {input_summary['has_inf']}",
            f"- input_mean_abs_max: {input_summary['input_mean_abs_max']:.8f}",
            f"- input_std_min: {input_summary['input_std_min']:.8f}",
            f"- input_std_max: {input_summary['input_std_max']:.8f}",
            "",
            "## Training Result",
            f"- best_epoch: {best_epoch}",
            f"- best_val_loss: {best_val_loss:.8f}",
            f"- final_train_loss: {float(final['train_total_loss']):.8f}",
            f"- final_val_loss: {float(final['val_total_loss']):.8f}",
            f"- final_val_recon_cosine: {float(final['val_recon_cosine']):.8f}",
            f"- early_stopped: {early_stopped}",
            "",
            "## Codebook Usage",
            *final_codebook_usage(final, num_codebooks, codebook_size),
            "",
            "## Diagnostics",
            *diagnostics(mode, log_rows, num_codebooks, semantic_log_path),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = merged_config(load_yaml(config_path), args)
    validate_config(config)

    seed = int(config.get("seed", 42))
    set_random_seed(seed)
    data_cfg = config["data"]
    prep_cfg = config["preprocess"]
    rqvae_cfg = config["rqvae"]
    logging_cfg = config.get("logging", {})
    paths = output_paths(args.mode, args.debug_max_rows)

    rqvae_input_path = output_path_for_mode(
        data_cfg[mode_input_config_key(args.mode)],
        args.mode,
        args.debug_max_rows,
    )
    prep_result = prepare_rqvae_input(
        mode=args.mode,
        text_embeddings_path=Path(data_cfg["text_embeddings_npy"]),
        embedding_meta_path=Path(data_cfg["embedding_meta_parquet"]),
        geo_features_path=Path(data_cfg["geo_features_npy"]),
        geo_meta_path=Path(data_cfg["geo_meta_parquet"]),
        output_input_path=rqvae_input_path,
        preprocess_path=paths["preprocess"],
        text_pca_dim=int(prep_cfg["text_pca_dim"]),
        geo_alpha=float(prep_cfg["geo_alpha"]),
        val_ratio=float(prep_cfg["val_ratio"]),
        seed=seed,
        pca_random_state=int(prep_cfg.get("pca_random_state", seed)),
        debug_max_rows=args.debug_max_rows,
        force_rebuild=args.force_rebuild_input,
    )

    x = np.load(prep_result.input_path, mmap_mode="r")
    input_summary = input_numeric_summary(x)
    if input_summary["has_nan"] or input_summary["has_inf"]:
        raise ValueError(f"RQ-VAE input has invalid values: {input_summary}")

    device = resolve_device(str(rqvae_cfg.get("device", "auto")))
    print(f"device: {device}")
    print(f"runtime: {cpu_thread_summary()}")
    num_workers = int(rqvae_cfg.get("num_workers", 0))
    if num_workers > 0 and device.type == "cpu":
        print(f"num_workers: requested {num_workers}, using 0 on CPU to avoid multiprocessing restrictions")
        num_workers = 0
    train_loader = make_loader(
        x,
        prep_result.train_indices,
        batch_size=int(rqvae_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = make_loader(
        x,
        prep_result.val_indices,
        batch_size=int(rqvae_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    input_dim = int(x.shape[1])
    model = RQVAE(
        input_dim=input_dim,
        encoder_hidden_dims=[int(v) for v in rqvae_cfg["encoder_hidden_dims"]],
        latent_dim=int(rqvae_cfg["latent_dim"]),
        num_codebooks=int(rqvae_cfg["num_codebooks"]),
        codebook_size=int(rqvae_cfg["codebook_size"]),
        commitment_beta=float(rqvae_cfg["commitment_beta"]),
        codebook_loss_weight=float(rqvae_cfg["codebook_loss_weight"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(rqvae_cfg["lr"]),
        weight_decay=float(rqvae_cfg["weight_decay"]),
    )

    epochs = int(rqvae_cfg["epochs"])
    patience = int(rqvae_cfg["early_stop_patience"])
    save_every_epochs = int(logging_cfg.get("save_every_epochs", 5))
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    log_rows: list[dict[str, Any]] = []
    num_codebooks = int(rqvae_cfg["num_codebooks"])
    codebook_size = int(rqvae_cfg["codebook_size"])
    start_epoch = 1
    resumed_from_checkpoint: Path | None = None
    resumed_from_epoch: int | None = None

    if args.resume:
        resumed_from_checkpoint = resolve_resume_checkpoint(str(args.resume_from), paths)
        checkpoint = load_resume_state(resumed_from_checkpoint, model, optimizer, device, args.mode, input_dim)
        resumed_from_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = resumed_from_epoch + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        log_rows = load_existing_train_log(paths["train_log"])
        if log_rows:
            log_rows = [row for row in log_rows if int(row.get("epoch", 0)) <= resumed_from_epoch]
            epochs_without_improvement = max(0, resumed_from_epoch - best_epoch)
        print(
            f"resume_from: {resumed_from_checkpoint} "
            f"start_epoch={start_epoch} best_epoch={best_epoch} best_val_loss={best_val_loss:.8f}"
        )

    min_epochs = args.min_epochs
    if min_epochs is not None and int(min_epochs) > epochs:
        raise ValueError(f"--min-epochs ({min_epochs}) cannot be greater than --epochs/config epochs ({epochs})")
    if args.resume and start_epoch > epochs:
        raise ValueError(f"Checkpoint already reached epoch {resumed_from_epoch}; target epochs is {epochs}")

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=float(rqvae_cfg.get("grad_clip_norm", 0.0)),
            epoch=epoch,
            log_every_steps=int(logging_cfg.get("log_every_steps", 50)),
        )
        val_metrics = evaluate(model, val_loader, device, num_codebooks, codebook_size)
        row: dict[str, Any] = {"epoch": epoch, **train_metrics, **val_metrics, "lr": current_lr(optimizer)}
        log_rows.append(row)
        write_train_log(log_rows, paths["train_log"])

        improved = float(row["val_total_loss"]) < best_val_loss
        if improved:
            best_val_loss = float(row["val_total_loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                paths["checkpoint"],
                model,
                optimizer,
                mode=args.mode,
                input_dim=input_dim,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                train_log_path=paths["train_log"],
                preprocess_path=paths["preprocess"],
                rqvae_input_path=prep_result.input_path,
                random_seed=seed,
                epoch=epoch,
                extra={"config": config, "preprocess_summary": prep_result.summary},
            )
        else:
            epochs_without_improvement += 1

        if epoch % max(1, save_every_epochs) == 0 or epoch == epochs:
            save_checkpoint(
                paths["last_checkpoint"],
                model,
                optimizer,
                mode=args.mode,
                input_dim=input_dim,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                train_log_path=paths["train_log"],
                preprocess_path=paths["preprocess"],
                rqvae_input_path=prep_result.input_path,
                random_seed=seed,
                epoch=epoch,
                extra={"config": config, "preprocess_summary": prep_result.summary},
            )

        print(
            f"epoch={epoch} train_loss={row['train_total_loss']:.6f} "
            f"val_loss={row['val_total_loss']:.6f} val_cos={row['val_recon_cosine']:.6f} "
            + " ".join(
                f"cb{level}={int(row[f'codebook{level}_used'])}/{codebook_size}"
                for level in range(num_codebooks)
            )
        )
        can_early_stop = min_epochs is None or epoch >= int(min_epochs)
        if can_early_stop and epochs_without_improvement >= patience:
            early_stopped = True
            break

    if best_epoch == 0:
        raise RuntimeError("Training did not produce a best checkpoint")
    final = log_rows[-1]
    save_checkpoint(
        paths["last_checkpoint"],
        model,
        optimizer,
        mode=args.mode,
        input_dim=input_dim,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_log_path=paths["train_log"],
        preprocess_path=paths["preprocess"],
        rqvae_input_path=prep_result.input_path,
        random_seed=seed,
        epoch=int(final["epoch"]),
        extra={"config": config, "preprocess_summary": prep_result.summary},
    )
    report = build_report(
        mode=args.mode,
        input_rows=int(x.shape[0]),
        input_dim=input_dim,
        train_rows=len(prep_result.train_indices),
        val_rows=len(prep_result.val_indices),
        text_pca_dim=prep_result.text_pca_dim,
        geo_alpha=prep_result.geo_alpha,
        checkpoint_path=paths["checkpoint"],
        preprocess_path=paths["preprocess"],
        rqvae_input_path=prep_result.input_path,
        input_summary=input_summary,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        log_rows=log_rows,
        early_stopped=early_stopped,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        semantic_log_path=append_debug_suffix(Path("outputs/rqvae/train_log_semantic.csv"), args.debug_max_rows),
        resumed_from_checkpoint=resumed_from_checkpoint,
        resumed_from_epoch=resumed_from_epoch,
        min_epochs=min_epochs,
    )
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(report, encoding="utf-8")

    codebook_usage = {
        f"codebook{level}": {
            "used": int(final[f"codebook{level}_used"]),
            "usage_rate": float(final[f"codebook{level}_usage_rate"]),
        }
        for level in range(num_codebooks)
    }
    print(f"mode: {args.mode}")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_loss: {best_val_loss:.8f}")
    print(f"final_val_recon_cosine: {float(final['val_recon_cosine']):.8f}")
    print(f"codebook_usage: {codebook_usage}")
    print(f"checkpoint_path: {paths['checkpoint']}")
    print("next_step: export SID and evaluate SID prefix purity")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        message = str(exc)
        if "torch" in message.lower():
            raise SystemExit("PyTorch is required. Please install torch: pip install torch") from exc
        if "sklearn" in message.lower() or "scikit" in message.lower():
            raise SystemExit("scikit-learn is required. Please install scikit-learn: pip install scikit-learn") from exc
        raise

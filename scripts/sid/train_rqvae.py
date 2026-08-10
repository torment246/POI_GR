#!/usr/bin/env python3
"""Train one experiment from the shared Vanilla RQ-VAE matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sid.training import (
    RQVAETrainingError,
    load_training_config,
    run_training,
)


def _codebook_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("码本大小必须是逗号分隔整数") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("码本大小必须是逗号分隔正整数")
    return sizes


def _checkpoint_epochs(value: str) -> tuple[int, ...]:
    try:
        epochs = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint epoch 必须是逗号分隔整数") from error
    if not epochs or any(epoch <= 0 for epoch in epochs):
        raise argparse.ArgumentTypeError("checkpoint epoch 必须是逗号分隔正整数")
    return epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "训练 Vanilla RQ-VAE；从 mmap Embedding 读取固定划分，执行逐层 KMeans "
            "初始化、固定 epoch checkpoint 保存和完整训练状态恢复。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sid/rqvae_beijing.yaml"),
        help="共享训练配置；相对路径相对于仓库根目录。",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="配置 experiments 中的实验 ID。",
    )
    parser.add_argument("--output-dir", type=Path, help="覆盖实验输出目录。")
    parser.add_argument("--device", help="覆盖训练设备，例如 cuda 或 cpu。")
    parser.add_argument("--batch-size", type=int, help="覆盖训练 batch size。")
    parser.add_argument("--max-epochs", type=int, help="覆盖最大 epoch。")
    parser.add_argument(
        "--checkpoint-epochs",
        type=_checkpoint_epochs,
        help="覆盖固定 checkpoint epoch，例如 1,2；仅用于 smoke test。",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="仅使用 Embedding 前 N 行，用于 smoke test；正式实验不设置。",
    )
    parser.add_argument(
        "--codebook-sizes",
        type=_codebook_sizes,
        help="覆盖码本，例如 16,16,16；仅用于 smoke test。",
    )
    parser.add_argument(
        "--kmeans-sample-size",
        type=int,
        help="覆盖 KMeans 初始化样本数。",
    )
    parser.add_argument(
        "--kmeans-backend",
        choices=("faiss_gpu", "faiss_cpu", "sklearn"),
        help="覆盖 KMeans backend。",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否从 last_checkpoint.pt 恢复。",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否显示终端进度条。",
    )
    return parser.parse_args()


def _resolve(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    assert config_path is not None
    try:
        config = load_training_config(
            config_path,
            PROJECT_ROOT,
            args.experiment,
            output_dir=_resolve(args.output_dir),
            device=args.device,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            checkpoint_epochs=args.checkpoint_epochs,
            max_rows=args.max_rows,
            codebook_sizes=args.codebook_sizes,
            kmeans_sample_size=args.kmeans_sample_size,
            kmeans_backend=args.kmeans_backend,
            resume=args.resume,
            show_progress=args.progress,
        )
        resolved = run_training(config, project_root=PROJECT_ROOT)
    except (RQVAETrainingError, OSError, ValueError) as error:
        print(f"RQ-VAE 训练失败：{error}", file=sys.stderr)
        return 2
    result = resolved["training_result"]
    print(
        json.dumps(
            {
                "status": resolved["status"],
                "experiment_id": config.experiment_id,
                "output_dir": str(config.output_dir),
                "actual_batch_size": result["actual_batch_size"],
                "last_epoch": result["last_epoch"],
                "fixed_checkpoint_epochs": result["fixed_checkpoint_epochs"],
                "stop_reason": result["stop_reason"],
                "training_seconds": result["training_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train the Beijing-adapted GNPR-SID RQ-VAE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.gnpr_rqvae_training import (  # noqa: E402
    GnprTrainingError,
    load_gnpr_training_config,
    run_gnpr_training,
)


def _integer_tuple(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是逗号分隔整数") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("必须是逗号分隔正整数")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="流式训练 GNPR-SID RQ-VAE，并导出三层 SID 与碰撞指标。"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/rqvae_gnpr_sid.yaml"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--checkpoint-epochs", type=_integer_tuple)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-rows", type=int, help="仅用于 smoke。")
    parser.add_argument("--codebook-sizes", type=_integer_tuple, help="仅用于 smoke。")
    parser.add_argument("--kmeans-sample-size", type=int)
    parser.add_argument(
        "--validation-basis-points",
        type=int,
        help="覆盖验证集万分比；正式配置为 100，smoke 可提高。",
    )
    parser.add_argument("--kmeans-backend", choices=("faiss_gpu", "faiss_cpu", "sklearn"))
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def _path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        config = load_gnpr_training_config(
            _path(args.config),
            PROJECT_ROOT,
            args.experiment,
            output_dir=_path(args.output_dir),
            device=args.device,
            max_epochs=args.max_epochs,
            checkpoint_epochs=args.checkpoint_epochs,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
            codebook_sizes=args.codebook_sizes,
            kmeans_sample_size=args.kmeans_sample_size,
            kmeans_backend=args.kmeans_backend,
            resume=args.resume,
            validation_basis_points=args.validation_basis_points,
        )
        result = run_gnpr_training(config)
    except (GnprTrainingError, OSError, RuntimeError, ValueError) as error:
        print(f"GNPR RQ-VAE 训练失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

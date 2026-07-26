#!/usr/bin/env python3
"""Export SID codes from a completed RQ-VAE run and evaluate them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.rqvae_training import RQVAETrainingError, export_checkpoint_sid
from poi_gr.sid_evaluation import SidEvaluationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从固定 epoch checkpoint 全量导出 sid_codes.npy 和 sid_manifest.json，"
            "随后在指定评估目录写入统一 SID metrics.json 与 collision_cases.jsonl。"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="实验目录内人工选定的 checkpoint 文件名。",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="已完成训练的实验目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="评估输出目录；必须位于实验目录内，默认直接使用实验目录。",
    )
    parser.add_argument("--device", help="覆盖导出设备，例如 cuda 或 cpu。")
    parser.add_argument("--batch-size", type=int, help="覆盖导出 batch size。")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    run_dir = _resolve(args.run_dir)
    output_dir = _resolve(args.output_dir) if args.output_dir else run_dir
    try:
        manifest, metrics = export_checkpoint_sid(
            run_dir,
            args.checkpoint,
            output_dir=output_dir,
            device_name=args.device,
            batch_size=args.batch_size,
        )
    except (RQVAETrainingError, SidEvaluationError, OSError, ValueError) as error:
        print(f"RQ-VAE SID 导出失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "experiment_id": manifest["experiment_id"],
                "sid_shape": manifest["sid_codes"]["shape"],
                "distinct_sid_ratio": metrics["basic"]["distinct_sid_ratio"],
                "collision_excess_ratio": metrics["basic"][
                    "collision_excess_ratio"
                ],
                "colliding_poi_ratio": metrics["basic"]["colliding_poi_ratio"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

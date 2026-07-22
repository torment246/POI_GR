#!/usr/bin/env python3
"""Train RQ-VAE and export hierarchical Semantic IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.rqvae import (
    apply_rqvae_overrides,
    load_rqvae_config,
    run_rqvae_job,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按 MiniOneRec 流程训练 POI RQ-VAE，选择碰撞率最低的 checkpoint，"
            "并导出三层 SID。"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rqvae_qwen3_embedding_0.6b.yaml"),
        help="YAML 配置路径；相对路径相对于仓库根目录。",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="只使用前 N 条向量，用于 smoke test；默认使用全量。",
    )
    parser.add_argument("--output-dir", type=Path, help="覆盖配置中的输出目录。")
    parser.add_argument("--device", help="覆盖训练设备，例如 cuda、cuda:0 或 cpu。")
    parser.add_argument("--batch-size", type=int, help="覆盖训练 batch size。")
    parser.add_argument("--epochs", type=int, help="覆盖训练 epoch 数。")
    parser.add_argument("--num-workers", type=int, help="覆盖 DataLoader worker 数。")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否从匹配的 checkpoint_last.pt 续训。",
    )
    return parser.parse_args()


def resolve_from_root(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config_path = resolve_from_root(args.config)
    assert config_path is not None
    config = load_rqvae_config(config_path, PROJECT_ROOT)
    config = apply_rqvae_overrides(
        config,
        output_dir=resolve_from_root(args.output_dir),
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        resume=args.resume,
    )
    manifest = run_rqvae_job(
        config,
        project_root=PROJECT_ROOT,
        max_rows=args.max_rows,
    )
    result = manifest["result"]
    evaluation = result["evaluation"]
    summary = {
        "status": manifest["status"],
        "best_loss_epoch": result["best_loss_epoch"],
        "best_collision_epoch": result["best_collision_epoch"],
        "selected_checkpoint": result["selected_checkpoint"],
        "rows": evaluation["rows"],
        "reconstruction_mse": evaluation["reconstruction_mse"],
        "cosine_similarity": evaluation["cosine_similarity"],
        "raw_collision_rate": evaluation[
            "nearest_assignment_collisions"
        ]["collision_rate"],
        "final_collision_rate": evaluation["final_collisions"]["collision_rate"],
        "final_unique_sid_rate": evaluation[
            "final_collisions"
        ]["unique_sid_rate"],
        "output_dir": manifest["config"]["output"]["dir"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

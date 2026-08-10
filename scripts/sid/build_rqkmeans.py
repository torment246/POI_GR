#!/usr/bin/env python3
"""Build aligned Semantic IDs with streaming residual K-Means."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sid.rqkmeans import (
    RQKMeansError,
    load_rqkmeans_config,
    run_rqkmeans,
)


def _codebook_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("码本必须是逗号分隔整数") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("码本必须是逗号分隔正整数")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在原始 POI Embedding 上训练逐层 Residual K-Means，并流式导出全量 "
            "SID、重构指标和统一 SID 评测结果。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sid/rqkmeans_bge_m3_1024x3.yaml"),
        help="实验配置；相对路径相对于仓库根目录。",
    )
    parser.add_argument("--output-dir", type=Path, help="覆盖配置中的输出目录。")
    parser.add_argument(
        "--implementation",
        choices=("sequential_kmeans", "faiss_residual_quantizer"),
        help="覆盖 RQ-KMeans 实现。",
    )
    parser.add_argument(
        "--backend",
        choices=("faiss_gpu", "faiss_cpu", "sklearn"),
        help="覆盖聚类/分配 backend。",
    )
    parser.add_argument("--sample-size", type=int, help="覆盖聚类训练样本数。")
    parser.add_argument("--iterations", type=int, help="覆盖每层 K-Means 最大迭代数。")
    parser.add_argument("--nredo", type=int, help="覆盖不同初始化重复次数。")
    parser.add_argument(
        "--max-beam-size",
        type=int,
        help="覆盖 Faiss ResidualQuantizer 训练与编码 beam。",
    )
    parser.add_argument(
        "--codebook-sizes",
        type=_codebook_sizes,
        help="覆盖各层码本，例如 1024,1024,1024。",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否恢复已完成的逐层码本。",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否显示进度条。",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验输入、划分和冻结采样，不训练码本。",
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="训练码本并只在冻结 Validation POI 子集筛选，不导出全量 SID。",
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
        config = load_rqkmeans_config(
            config_path,
            PROJECT_ROOT,
            output_dir=_resolve(args.output_dir),
            resume=args.resume,
            show_progress=args.progress,
            implementation=args.implementation,
            backend=args.backend,
            sample_size=args.sample_size,
            iterations=args.iterations,
            nredo=args.nredo,
            max_beam_size=args.max_beam_size,
            codebook_sizes=args.codebook_sizes,
        )
        result = run_rqkmeans(
            config,
            project_root=PROJECT_ROOT,
            validate_only=args.validate_only,
            screen_only=args.screen_only,
        )
    except (RQKMeansError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"RQ-KMeans 构建失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "experiment_id": config.experiment_id,
                "output_dir": str(config.output_dir),
                "split": result.get("split"),
                "outputs": result.get("outputs"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

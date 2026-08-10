#!/usr/bin/env python3
"""Build GenPOI GeoPE vectors from existing BGE-M3 POI embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.genpoi.geope import (
    GenpoiGeoPEError,
    build_genpoi_geope_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按照 GenPOI 论文，用 POI 坐标 K-Means 得到地理锚点，"
            "再按方位角分段旋转已有 BGE-M3 向量。"
        )
    )
    parser.add_argument(
        "--source-embedding-dir",
        type=Path,
        required=True,
        help="已有 BGE Embedding 目录。",
    )
    parser.add_argument(
        "--poi-data-dir",
        type=Path,
        required=True,
        help="与 Embedding 行顺序一致的 POI JSONL 分片目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="不存在的 GeoPE 输出目录。",
    )
    parser.add_argument(
        "--reference-count",
        type=int,
        default=32,
        help="GeoPE 地理锚点数量；北京数据评估默认使用 32。",
    )
    parser.add_argument(
        "--reference-points",
        type=Path,
        help=(
            "复用已有 reference_points.npy；用于固定锚点的单变量对照，"
            "不再重新拟合 K-Means。"
        ),
    )
    parser.add_argument(
        "--coordinate-projection",
        choices=(
            "local_equirectangular_km",
            "raw_longitude_latitude_degrees",
        ),
        default="local_equirectangular_km",
        help="地理锚点 K-Means 的坐标系；北京数据默认使用局部公里投影。",
    )
    parser.add_argument(
        "--reference-fit-sample-size",
        type=int,
        default=200000,
        help="拟合地理锚点的固定随机 POI 数；不超过实际构建行数。",
    )
    parser.add_argument("--seed", type=int, default=42, help="K-Means 随机种子。")
    parser.add_argument(
        "--kmeans-n-init",
        type=int,
        default=3,
        help="K-Means 独立初始化次数。",
    )
    parser.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=100,
        help="每次 K-Means 最大迭代数。",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=8192,
        help="GeoPE 分块旋转行数。",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="GeoPE 旋转设备，例如 cpu 或 cuda；锚点拟合仍在 CPU。",
    )
    parser.add_argument(
        "--metric-sample-rows",
        type=int,
        default=32768,
        help="只用前 N 行计算旋转范数指标；不影响全量 GeoPE 输出。",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="输出向量 dtype。",
    )
    parser.add_argument(
        "--embedding-preprocessing",
        choices=("none", "global_mean_center_l2"),
        default="none",
        help=(
            "GeoPE 旋转前的向量预处理；global_mean_center_l2 表示先减去"
            "构建范围全局均值，再逐行 L2 归一化。"
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="只处理前 N 行，用于 smoke；全量构建时不要传。",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        result = build_genpoi_geope_embeddings(
            _resolve(args.source_embedding_dir),
            _resolve(args.poi_data_dir),
            _resolve(args.output_dir),
            reference_count=args.reference_count,
            seed=args.seed,
            kmeans_n_init=args.kmeans_n_init,
            kmeans_max_iter=args.kmeans_max_iter,
            coordinate_projection=args.coordinate_projection,
            reference_fit_sample_size=args.reference_fit_sample_size,
            chunk_rows=args.chunk_rows,
            rotation_device=args.device,
            metric_sample_rows=args.metric_sample_rows,
            output_dtype=args.output_dtype,
            embedding_preprocessing=args.embedding_preprocessing,
            reference_points_path=(
                _resolve(args.reference_points)
                if args.reference_points is not None
                else None
            ),
            max_rows=args.max_rows,
            progress=lambda message: print(
                message,
                file=sys.stderr,
                flush=True,
            ),
        )
    except (GenpoiGeoPEError, OSError, RuntimeError, ValueError) as error:
        print(f"GenPOI GeoPE 构建失败：{error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "completed",
                "scope": result.manifest["input"]["scope"],
                "rows": result.manifest["input"]["total_rows"],
                "shape": result.manifest["output"]["shape"],
                "reference_count": result.manifest["geope"][
                    "reference_count"
                ],
                "reference_point_method": result.manifest["geope"][
                    "reference_point_method"
                ],
                "embedding_preprocessing": result.manifest["geope"][
                    "embedding_preprocessing"
                ]["mode"],
                "mean_absolute_norm_delta": result.manifest["metrics"][
                    "mean_absolute_norm_delta_float32"
                ],
                "mean_vector_l2_delta": result.manifest["metrics"][
                    "mean_vector_l2_delta_float32"
                ],
                "manifest_sha256": result.output_hashes["manifest.json"],
                "output_dir": str(_resolve(args.output_dir).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

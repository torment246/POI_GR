#!/usr/bin/env python3
"""Build sparse E1/E2 Train Query aggregates for covered POIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.query_augmentation import (  # noqa: E402
    load_query_augmentation_config,
    override_query_augmentation_output,
    run_query_poi_aggregation,
    validate_query_poi_aggregates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按 POI 哈希分片流式聚合 Train Query 向量，生成 E1 等权均值与 "
            "E2 count/DF 加权的稀疏 POI 表征。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/embedding/embedding_query_augmentation_bge_m3_v1.yaml"),
        help="E1/E2 聚合配置；相对路径相对于仓库根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="覆盖输出目录；--max-partitions smoke 必须使用独立目录。",
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        help="只处理前 N 个 POI 哈希分片，用于 smoke。",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只复检已完成产物的 shape、SHA256、有限值和行号唯一性。",
    )
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条。")
    return parser.parse_args()


def _resolve(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    assert config_path is not None
    config = load_query_augmentation_config(config_path, PROJECT_ROOT)
    output_dir = _resolve(args.output_dir)
    if args.max_partitions is not None and output_dir is None:
        raise SystemExit("使用 --max-partitions 时必须提供独立 --output-dir")
    config = override_query_augmentation_output(config, output_dir)
    if args.validate_only:
        if args.max_partitions is not None:
            raise SystemExit("--validate-only 不能与 --max-partitions 同时使用")
        result = validate_query_poi_aggregates(
            config.output_dir,
            expected_stats_manifest_sha256=config.expected_stats_manifest_sha256,
        )
    else:
        result = run_query_poi_aggregation(
            config,
            project_root=PROJECT_ROOT,
            max_partitions=args.max_partitions,
            show_progress=not args.no_progress,
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "covered_pois": result.covered_pois,
                "embedding_dim": result.embedding_dim,
                "partitions": result.partitions,
                "reused": result.reused,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

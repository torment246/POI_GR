#!/usr/bin/env python3
"""Build Train-only Query frequency, DF, pair, and POI coverage statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.query_stats import (  # noqa: E402
    build_train_query_stats,
    validate_train_query_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "逐个处理 Train Query 哈希分片，构建唯一 Query 目录、"
            "Query-POI 频次/DF 和 POI 覆盖统计。"
        ),
    )
    parser.add_argument(
        "--query-shards-dir",
        type=Path,
        default=Path("outputs/embeddings/query_augmented_bge_m3/train_query_shards_v1"),
        help="已完成并通过核验的 Train Query 哈希分片目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embeddings/query_augmented_bge_m3/train_query_stats_v1"),
        help="Query 统计输出目录。",
    )
    parser.add_argument(
        "--poi-partitions",
        type=int,
        default=256,
        help="POI 覆盖统计的稳定哈希分区数。",
    )
    parser.add_argument(
        "--poi-buffer-rows",
        type=int,
        default=2_048,
        help="每个 POI 分区在写入 Parquet 前最多缓存的 Query-POI 行数。",
    )
    parser.add_argument(
        "--read-batch-rows",
        type=int,
        default=65_536,
        help="逐批读取 Parquet 的最大行数。",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只核验已有统计产物及全部文件 SHA256。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭进度条。",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    if args.validate_only:
        result = validate_train_query_stats(output_dir)
    else:
        result = build_train_query_stats(
            _resolve(args.query_shards_dir),
            output_dir,
            poi_partitions=args.poi_partitions,
            poi_buffer_rows=args.poi_buffer_rows,
            read_batch_rows=args.read_batch_rows,
            show_progress=not args.no_progress,
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "source_rows": result.source_rows,
                "unique_queries": result.unique_queries,
                "unique_query_poi_pairs": result.unique_query_poi_pairs,
                "covered_pois": result.covered_pois,
                "reused": result.reused,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

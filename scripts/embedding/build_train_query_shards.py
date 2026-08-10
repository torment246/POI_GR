#!/usr/bin/env python3
"""Build deterministic Query-to-POI Parquet shards from canonical Train data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.query_shards import (  # noqa: E402
    build_train_query_shards,
    validate_query_shards,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "流式提取固定 Train JSONL 的原始 Query 与 target_poi_id，"
            "按稳定 Query 哈希写入 Parquet 分片。"
        ),
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=Path("data/sft/beijing_order_main_v1"),
        help="包含 train.jsonl 和 manifest.json 的固定 SFT 数据目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/embeddings/query_augmented_bge_m3/train_query_shards_v1"
        ),
        help="Query 哈希分片输出目录。",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=256,
        help="稳定 Query 哈希分片数；正式产物固定为 256。",
    )
    parser.add_argument(
        "--buffer-rows-per-shard",
        type=int,
        default=4_096,
        help="每个分片累计多少行后写入一个 Parquet Row Group。",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="不读取 Train，只完整核验已经生成的分片和 SHA256。",
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
        result = validate_query_shards(output_dir)
    else:
        result = build_train_query_shards(
            _resolve(args.sft_dir),
            output_dir,
            num_shards=args.num_shards,
            buffer_rows_per_shard=args.buffer_rows_per_shard,
            show_progress=not args.no_progress,
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "total_rows": result.total_rows,
                "num_shards": result.num_shards,
                "reused": result.reused,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

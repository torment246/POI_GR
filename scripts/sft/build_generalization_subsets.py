#!/usr/bin/env python3
"""Freeze Validation-only generalization subsets for generative retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sft.evaluation import validate_split_manifest  # noqa: E402
from poi_gr.sft.generalization_subset import (  # noqa: E402
    DEFAULT_LONG_TAIL_MAX_FREQUENCY,
    DEFAULT_SUBSET_SIZE,
    GeneralizationSubsetError,
    build_generalization_validation_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "只使用完整 Validation 和 Train-only Query 统计，冻结 Query 迁移、"
            "新 Query、长尾目标和冷目标四个互斥专项集，并登记既有随机/地理集。"
        )
    )
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--train-query-shards-dir", type=Path, required=True)
    parser.add_argument("--random-subset-file", type=Path, required=True)
    parser.add_argument("--random-subset-manifest", type=Path, required=True)
    parser.add_argument("--geo-subset-file", type=Path, required=True)
    parser.add_argument("--geo-subset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument(
        "--long-tail-max-frequency",
        type=int,
        default=DEFAULT_LONG_TAIL_MAX_FREQUENCY,
        help="目标 POI 在 Train 中被视为长尾的最大正样本行数，默认 5。",
    )
    parser.add_argument(
        "--skip-data-hash",
        action="store_true",
        help="仅用于技术调试；跳过 Validation 源文件哈希复核。",
    )
    parser.add_argument(
        "--skip-train-shard-hashes",
        action="store_true",
        help="仅用于技术调试；跳过 256 个 Train Query Parquet 的逐文件哈希复核。",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        valid_file = resolve(args.valid_file)
        _, source_rows, source_sha256 = validate_split_manifest(
            valid_file,
            split="valid",
            verify_hash=not args.skip_data_hash,
        )
        suite = build_generalization_validation_suite(
            valid_file,
            resolve(args.train_query_shards_dir),
            resolve(args.random_subset_file),
            resolve(args.random_subset_manifest),
            resolve(args.geo_subset_file),
            resolve(args.geo_subset_manifest),
            resolve(args.output_dir),
            source_rows=source_rows,
            source_sha256=source_sha256,
            subset_size=args.subset_size,
            long_tail_max_frequency=args.long_tail_max_frequency,
            verify_train_shard_hashes=not args.skip_train_shard_hashes,
        )
        print(json.dumps(suite.manifest, ensure_ascii=False, indent=2))
        return 0
    except (GeneralizationSubsetError, OSError, ValueError, KeyError) as error:
        print(f"构建失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze a complex geographic-query subset from full Validation."""

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
from poi_gr.sft.geo_query_subset import (  # noqa: E402
    DEFAULT_MIN_QUERY_LENGTH,
    DEFAULT_SUBSET_SIZE,
    GeoQuerySubsetError,
    build_geo_query_validation_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从完整 Validation 中筛选显式地理关系且长度达标的复杂 Query，"
            "按业务主键稳定哈希冻结专项评测集。"
        )
    )
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument(
        "--min-query-length",
        type=int,
        default=DEFAULT_MIN_QUERY_LENGTH,
        help="NFKC 归一化并去除空白后的最小字符数。",
    )
    parser.add_argument(
        "--skip-data-hash",
        action="store_true",
        help="仅用于技术调试；跳过 SFT manifest 的源文件哈希复核。",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        valid_file = resolve(args.valid_file)
        output_dir = resolve(args.output_dir)
        _, source_rows, source_sha256 = validate_split_manifest(
            valid_file,
            split="valid",
            verify_hash=not args.skip_data_hash,
        )
        subset = build_geo_query_validation_subset(
            valid_file,
            output_dir,
            source_rows=source_rows,
            source_sha256=source_sha256,
            subset_size=args.subset_size,
            min_query_length=args.min_query_length,
        )
        print(json.dumps(subset.manifest, ensure_ascii=False, indent=2))
        return 0
    except (GeoQuerySubsetError, OSError, ValueError, KeyError) as error:
        print(f"构建失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Render GenPOI-inspired, matched-category SID comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from qg_prqk.sid.category_region_visualization import (
    build_category_region_visualization,
    validate_category_region_visualization,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="三版 SID 的共同五类 t-SNE 与全库类别/区域前缀图；只读分析"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="类别/区域可视化 YAML 配置"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="核验来源与行序，不生成输出"
    )
    mode.add_argument(
        "--validate-only", action="store_true", help="复核已有图表、数据和全部来源哈希"
    )
    args = parser.parse_args(argv)
    try:
        result = (
            validate_category_region_visualization(args.config)
            if args.validate_only
            else build_category_region_visualization(args.config, dry_run=args.dry_run)
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"可视化失败：{error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

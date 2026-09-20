"""Build or validate the three-way SID visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from qg_prqk.sid.visualization import (
    build_sid_visualization,
    inspect_sid_visualization_inputs,
    validate_sid_visualization,
)
from qg_prqk.sid.visualization_config import load_sid_visualization_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="并排可视化 A0、原始 A4 和 NoGID A4 的三位 SID"
    )
    parser.add_argument("--config", type=Path, required=True, help="冻结可视化配置")
    parser.add_argument(
        "--dry-run", action="store_true", help="只校验输入合同，不生成图片"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="只校验已有输出和来源哈希"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run and args.validate_only:
        raise SystemExit("--dry-run 与 --validate-only 不能同时使用")
    config = load_sid_visualization_config(args.config)
    if args.dry_run:
        result = inspect_sid_visualization_inputs(config)
    elif args.validate_only:
        result = validate_sid_visualization(config)
    else:
        result = build_sid_visualization(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

"""Evaluate and publish the canonical static SID comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "在冻结 Train-only 图上比较 POI-only 与 Query+Category+Geo SID，"
            "运行 Prefix Probe 并发布 A4 三层 SID 候选；结束后固定 HOLD_FOR_REVIEW。"
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="已确认的 SID 评测 YAML")
    parser.add_argument("--gate", choices=("sample", "full"), default="sample")
    parser.add_argument("--chunk-rows", type=int, default=8192, help="GPU Probe 分块行数")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="只核验冻结 manifest、NPY 与 Parquet header，不运行 Probe、不写输出",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="独立复核已完成 SID 评测的来源、artifact、逐值 assignment 和停止边界",
    )
    args = parser.parse_args(argv)
    try:
        from qg_prqk.sid.evaluation_config import load_sid_evaluation_config

        config = load_sid_evaluation_config(args.config)
        if args.validate_only:
            from qg_prqk.sid.evaluation import validate_sid_evaluation

            result = validate_sid_evaluation(config, gate=args.gate)
        elif args.dry_run:
            from qg_prqk.sid.evaluation_data import inspect_sid_evaluation_inputs

            result = inspect_sid_evaluation_inputs(config, gate=args.gate)
        else:
            from qg_prqk.sid.evaluation import build_sid_evaluation

            result = build_sid_evaluation(
                config, gate=args.gate, chunk_rows=args.chunk_rows
            )
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"SID 静态评测失败：{error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

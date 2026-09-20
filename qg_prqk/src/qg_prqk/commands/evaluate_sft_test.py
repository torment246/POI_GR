"""Expose full final-day Test evaluation separately from frozen Validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qg_prqk.sft.full_test_data import DEFAULT_CONFIG, load_test_config
from qg_prqk.sft.evaluation_data import load_json, resolve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="双卡 6000D：两个 QG epoch-3 模型评测最后一天全量 606,682 条 Test。")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG), help="冻结 Test 配置")
    parser.add_argument("--dry-run", "--preflight-only", action="store_true", help="全量 Test 哈希/同行对齐及前 100 条 Token 预检；不启动 GPU")
    parser.add_argument("--smoke-limit", type=int, help="每版只生成前 1–100 条，输出与正式 Test 隔离")
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
        parser.error("--smoke-limit 必须在 1–100 之间")
    if args.plan and args.dry_run:
        parser.error("worker 不能使用 --dry-run")
    try:
        from qg_prqk.sft.full_test_evaluation import evaluate_test, run_test_suite
        if args.plan:
            evaluate_test(load_json(args.plan), smoke_limit=args.smoke_limit)
            return 0
        return run_test_suite(load_test_config(resolve(args.config)), dry_run=args.dry_run, smoke_limit=args.smoke_limit)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"QG 全量 Test 评测失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

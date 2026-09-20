"""Expose the paired SFT evaluation suite and its isolated single-GPU worker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qg_prqk.sft.evaluation_data import DEFAULT_CONFIG, SUBSETS, load_config, load_json, resolve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QG 双 SFT 的 epoch-3 固定 10k + 四类泛化评测；一台双卡 6000D 共享调度。")
    parser.add_argument("--variant", choices=("both", "a4_gid_parent", "a4_nogid"), help="both 在同一双卡队列评测两个分支，也可只选一个")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG), help="冻结评测协议")
    parser.add_argument("--dry-run", action="store_true", help="准备/核验五组对齐缓存、checkpoint、词表和样例，不启动 GPU 推理")
    parser.add_argument("--smoke-limit", type=int, help="每组只运行 1–100 条 GPU smoke，输出与正式结果隔离")
    parser.add_argument("--worker-subset", choices=SUBSETS, help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
        parser.error("--smoke-limit 必须在 1–100 之间")
    if args.worker_subset and (not args.plan or args.dry_run or args.variant):
        parser.error("worker 必须指定 --plan，且不能同时指定 --variant/--dry-run")
    if not args.worker_subset and (not args.variant or args.plan):
        parser.error("套件必须指定 --variant，不能指定 --plan")
    try:
        if args.worker_subset:
            from qg_prqk.sft.evaluation import evaluate_subset
            evaluate_subset(load_json(args.plan), args.worker_subset, smoke_limit=args.smoke_limit)
            return 0
        from qg_prqk.sft.evaluation_suite import run_suite
        return run_suite(load_config(resolve(args.config)), args.variant, dry_run=args.dry_run, smoke_limit=args.smoke_limit)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"QG SFT 评测失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

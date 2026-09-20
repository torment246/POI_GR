"""Train one SFT target variant through the validated LLaMA-Factory wrapper."""

from __future__ import annotations

import argparse
import sys

from qg_prqk.sft.training import SftTrainingError, run_sft_variant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="启动一个已冻结的 QG-PRQK SFT 变体。"
    )
    parser.add_argument(
        "--variant",
        choices=("a4_gid_parent", "a4_nogid"),
        required=True,
        help="训练目标：GID-parent Final ID 或纯三层 SID Final ID。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只核验并打印解析配置")
    args = parser.parse_args(argv)
    try:
        return run_sft_variant(args.variant, dry_run=args.dry_run)
    except (SftTrainingError, OSError, ValueError) as error:
        print(f"QG-PRQK SFT 启动失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

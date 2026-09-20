"""Build or validate the GID-parent local S3 codebook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "冻结关系化 S1/S2，只用 D3 Query residual、GID6 局部 Geo "
            "和 same-fine-category 困难图构建 S3；sample 通过后才运行 full。"
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="已确认的局部 S3 YAML")
    parser.add_argument("--gate", choices=("sample", "full"), default="sample")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="只核验冻结 manifest/NPY/Parquet header，不读 catalog 数值、不写输出",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="独立复核已完成局部 S3 目录的 manifest、哈希、shape 和冻结前缀",
    )
    args = parser.parse_args(argv)
    try:
        from qg_prqk.sid.local_config import load_local_codebook_config

        config = load_local_codebook_config(args.config)
        if args.validate_only:
            from qg_prqk.sid.local_quantizer import validate_local_codebook

            result = validate_local_codebook(config, gate=args.gate)
        elif args.dry_run:
            from qg_prqk.sid.local_data import inspect_local_codebook_inputs

            result = inspect_local_codebook_inputs(config, gate=args.gate)
            result["next_status"] = f"READY_TO_RUN_P7_{args.gate.upper()}"
        else:
            from qg_prqk.sid.local_quantizer import load_and_build_local_codebook

            result = load_and_build_local_codebook(config, gate=args.gate)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"局部 S3 构建失败：{error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build or validate the S1/S2-parent no-GID S3 codebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from qg_prqk.sid.nogid_config import load_nogid_codebook_config
from qg_prqk.sid.nogid_data import inspect_nogid_codebook_inputs
from qg_prqk.sid.nogid_quantizer import load_and_build_nogid_codebook, validate_nogid_codebook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于冻结 S1/S2 构建不含 GID 的三位 S1/S2/S3 SID")
    parser.add_argument("--config", type=Path, required=True, help="no-GID S3 YAML 配置")
    parser.add_argument("--gate", choices=("sample", "full"), required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true", help="只核对冻结输入 header，不读目录值或写输出")
    parser.add_argument("--validate-only", action="store_true", help="只读复核已完成输出")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_nogid_codebook_config(args.config)
    if args.dry_run and args.validate_only:
        raise SystemExit("--dry-run 与 --validate-only 不能同时使用")
    if args.dry_run:
        result = inspect_nogid_codebook_inputs(config, gate=args.gate)
    elif args.validate_only:
        result = validate_nogid_codebook(config, gate=args.gate)
    else:
        result = load_and_build_nogid_codebook(config, gate=args.gate, device=args.device, chunk_rows=args.chunk_rows)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

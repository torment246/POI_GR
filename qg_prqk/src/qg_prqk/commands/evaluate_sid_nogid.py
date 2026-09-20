"""Evaluate the full three-token no-GID comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from qg_prqk.sid.nogid_evaluation_config import load_nogid_evaluation_config
from qg_prqk.sid.nogid_evaluation import build_nogid_evaluation, inspect_nogid_evaluation_inputs, validate_nogid_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较 A0、原 GID-parent A4 与 S1/S2-parent NoGID A4 的三位 SID")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run and args.validate_only:
        raise SystemExit("--dry-run 与 --validate-only 不能同时使用")
    config = load_nogid_evaluation_config(args.config)
    if args.dry_run:
        result = inspect_nogid_evaluation_inputs(config)
    elif args.validate_only:
        result = validate_nogid_evaluation(config)
    else:
        result = build_nogid_evaluation(config, device=args.device, chunk_rows=args.chunk_rows)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

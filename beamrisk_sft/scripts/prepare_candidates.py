#!/usr/bin/env python3
"""Freeze or validate the 500k Train-only mining candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from beamrisk_sft.candidate_pool import build_candidate_pool, validate_candidate_pool
from beamrisk_sft.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    result = (
        validate_candidate_pool(config, verify_hashes=args.verify_hashes)
        if args.validate_only
        else build_candidate_pool(config, reuse=args.reuse)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

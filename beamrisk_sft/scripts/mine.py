#!/usr/bin/env python3
"""Mine one epoch's Beam-risk pairs on four independent GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from beamrisk_sft.config import load_config
from beamrisk_sft.mining import run_distributed_mining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-epoch", type=int, choices=(1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    result = run_distributed_mining(
        config,
        reference_epoch=args.reference_epoch,
        checkpoint=args.checkpoint,
        resume=args.resume,
    )
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

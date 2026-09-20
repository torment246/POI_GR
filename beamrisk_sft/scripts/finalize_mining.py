#!/usr/bin/env python3
"""Finalize already-complete distributed mining parts without GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from beamrisk_sft.config import load_config
from beamrisk_sft.mining import finalize_mined_parts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-epoch", type=int, choices=(1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = finalize_mined_parts(
        load_config(args.config),
        reference_epoch=args.reference_epoch,
        checkpoint=args.checkpoint.resolve(),
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

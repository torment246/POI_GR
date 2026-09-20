#!/usr/bin/env python3
"""Run exactly one BeamRisk-SFT epoch boundary under torchrun."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from beamrisk_sft.config import load_config
from beamrisk_sft.workflow import destroy_distributed, run_training_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-epoch", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    try:
        checkpoint = run_training_stage(
            config,
            target_epoch=args.target_epoch,
            resume_checkpoint=args.resume_checkpoint,
        )
        if int(__import__("os").environ.get("RANK", "0")) == 0:
            print(f"BeamRisk-SFT epoch {args.target_epoch} checkpoint: {checkpoint}")
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

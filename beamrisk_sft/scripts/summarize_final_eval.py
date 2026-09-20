#!/usr/bin/env python3
"""Validate both final evaluations and print one compact comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from beamrisk_sft.config import load_config
from beamrisk_sft.evaluation import summarize_final_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--unconstrained-result", type=Path, required=True)
    parser.add_argument("--constrained-result", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_final_evaluation(
        load_config(args.config),
        final_checkpoint=args.checkpoint,
        unconstrained_result_path=args.unconstrained_result,
        constrained_result_path=args.constrained_result,
    )
    print(result["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

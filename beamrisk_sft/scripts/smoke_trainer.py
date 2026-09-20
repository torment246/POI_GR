#!/usr/bin/env python3
"""Exercise the real BeamRisk Trainer/optimizer/DDP contract on a tiny Qwen3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from beamrisk_sft.smoke import run_trainer_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    result = run_trainer_smoke(
        args.work_dir,
        expected_world_size=args.expected_world_size,
        device=args.device,
    )
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


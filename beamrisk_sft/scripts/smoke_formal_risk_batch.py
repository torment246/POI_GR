#!/usr/bin/env python3
"""Backpropagate the exact first formal risk batch on a tiny full-vocab Qwen3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from beamrisk_sft.config import load_config
from beamrisk_sft.smoke import run_formal_risk_batch_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-epoch", type=int, choices=(1, 2), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    result = run_formal_risk_batch_smoke(
        load_config(args.config),
        reference_epoch=args.reference_epoch,
        work_dir=args.work_dir,
        device=args.device,
    )
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


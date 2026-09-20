#!/usr/bin/env python3
"""Validate finalized risk pairs without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from beamrisk_sft.config import load_config
from beamrisk_sft.io import atomic_write_json
from beamrisk_sft.validation import validate_risk_pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-epoch", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = validate_risk_pairs(
        config,
        reference_epoch=args.reference_epoch,
    )
    report_path = config.mining_dir(args.reference_epoch) / "validation_report.json"
    atomic_write_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

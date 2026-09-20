#!/usr/bin/env python3
"""Recover stage commit records after a checkpoint was fully saved."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from beamrisk_sft.checkpoints import recover_stage_records
from beamrisk_sft.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = recover_stage_records(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Thin A0-GID entry point; leave frozen A4 command fingerprints unchanged."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qg_prqk.sft.a0_gid_pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Thin entry point for migration analysis; existing frozen visualization CLIs stay unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qg_prqk.sid.migration_analysis import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

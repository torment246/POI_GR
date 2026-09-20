#!/usr/bin/env python3
"""Repository-local entry point for the QG-PRQK command suite."""

from __future__ import annotations

import sys
from pathlib import Path


QG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QG_ROOT / "src"))

from qg_prqk.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Refresh the user's 0917 slides using active-catalog TIGER SID evidence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qg_prqk.sid.tiger_presentation import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())

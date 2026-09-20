#!/usr/bin/env python3
"""Render frozen S3 local-entity changes without retraining."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from qg_prqk.sid.local_visualization import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())

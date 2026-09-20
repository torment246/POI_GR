#!/usr/bin/env python3
"""Evaluate the trained POI-only PRQ-KMeans GID control on the frozen full Test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qg_prqk.sft.a0_full_test import main

if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python
"""Prepare Active GenPOI data on the platform, then launch four-GPU SFT."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from poi_gr.methods.genpoi.active_sft import main

if __name__ == "__main__":
    raise SystemExit(main())

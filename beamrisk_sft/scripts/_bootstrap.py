"""Make repository-local packages importable without installation."""

from __future__ import annotations

import os
import sys
from pathlib import Path


BEAMRISK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BEAMRISK_ROOT.parent
matplotlib_cache = BEAMRISK_ROOT / "outputs" / "cache" / "matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
for path in (
    BEAMRISK_ROOT / "src",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "third_party" / "LLaMA-Factory" / "src",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

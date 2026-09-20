from __future__ import annotations

import sys
from pathlib import Path


BEAMRISK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BEAMRISK_ROOT.parent
for path in (
    BEAMRISK_ROOT / "src",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "third_party" / "LLaMA-Factory" / "src",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

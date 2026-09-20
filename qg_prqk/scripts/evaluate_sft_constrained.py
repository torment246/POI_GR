#!/usr/bin/env python3
"""QG 双卡约束解码薄入口；不改已冻结的无约束 CLI。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qg_prqk.sft.constrained_evaluation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

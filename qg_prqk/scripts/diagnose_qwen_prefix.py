#!/usr/bin/env python3
"""Entry point for paired Qwen prefix diagnostics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qg_prqk.sft.prefix_diagnostic import main

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Qwen 逐层诊断失败：{error}", file=sys.stderr)
        raise SystemExit(2)

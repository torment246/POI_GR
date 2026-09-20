#!/usr/bin/env python3
"""Build active GenPOI SFT by replacing identifiers in frozen TIGER samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from poi_gr.methods.genpoi.aligned_data import build_aligned_sft_data


def main() -> int:
    parser = argparse.ArgumentParser(description="逐条替换 TIGER SID，保留用户哈希、Query、GID、历史及切分。")
    for name in ("source-dir", "tiger-id-dir", "pid-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-rows-per-split", type=int, help="仅供 smoke，不可用于正式训练。")
    parser.add_argument("--sid-codebook-size", type=int, choices=(512, 1024), default=512)
    args = parser.parse_args()
    try:
        values = vars(args)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = value if value.is_absolute() else ROOT / value
        result = build_aligned_sft_data(**values, progress=lambda s: print(s, flush=True))
    except (ValueError, OSError, KeyError) as error:
        print(f"GenPOI 对齐数据构建失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "stats": result["stats"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

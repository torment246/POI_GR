#!/usr/bin/env python3
"""Build isolated TIGER-Joint Train/Valid data from raw orders and BGE rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger_joint import (  # noqa: E402
    HistoryWindow,
    TigerJointDatasetError,
    build_sid_free_training_data,
)
from poi_gr.sft.data import TimeSplit, parse_iso_date  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从原始行为 POI ID 和冻结 BGE poi_ids.jsonl 构建 TIGER-Joint "
            "Train/Valid 动态槽位数据；不读取旧 TIGER SID 或 checkpoint。"
        )
    )
    parser.add_argument(
        "--orders-dir",
        type=Path,
        default=Path(
            "data/beijing_order_clean_20260701_20260714_"
            "history10_20260401_20260630_json"
        ),
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("outputs/embeddings/beijing_poi_bge_m3"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start", default="2026-07-01")
    parser.add_argument("--train-end", default="2026-07-12")
    parser.add_argument("--valid-date", default="2026-07-13")
    parser.add_argument("--test-date", default="2026-07-14")
    parser.add_argument("--history-start", default="2026-04-01")
    parser.add_argument("--history-end", default="2026-06-30")
    parser.add_argument(
        "--max-retained-per-split",
        type=int,
        default=None,
        help="仅用于 smoke；分别限制 Train/Valid 保留数。",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    try:
        result = build_sid_free_training_data(
            resolve(args.orders_dir),
            resolve(args.embedding_dir),
            resolve(args.output_dir),
            TimeSplit(
                train_start=parse_iso_date(args.train_start, "train-start"),
                train_end=parse_iso_date(args.train_end, "train-end"),
                valid_date=parse_iso_date(args.valid_date, "valid-date"),
                test_date=parse_iso_date(args.test_date, "test-date"),
            ),
            HistoryWindow(
                start=parse_iso_date(args.history_start, "history-start"),
                end=parse_iso_date(args.history_end, "history-end"),
            ),
            max_retained_per_split=args.max_retained_per_split,
            progress=lambda message: print(message, flush=True),
        )
    except (OSError, ValueError, TigerJointDatasetError) as error:
        print(f"TIGER-Joint SID-free 数据构建失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(resolve(args.output_dir)),
                "build_fingerprint": result.manifest["build_fingerprint"],
                "train_rows": result.stats["train_count"],
                "valid_rows": result.stats["valid_count"],
                "old_sid_mapping_loaded": False,
                "test_jsonl_written": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

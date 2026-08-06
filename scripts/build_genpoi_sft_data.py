#!/usr/bin/env python3
"""Build the causal GenPOI SFT dataset from 003 history orders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.genpoi_data import (
    GenpoiDataError,
    HistoryWindow,
    build_genpoi_sft_data,
)
from poi_gr.sft_main_data import TimeSplit, parse_iso_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将 003 的历史 Query、历史位置和历史 POI 线性化为 GenPOI "
            "Messages 数据，目标为正式 GenPOI PID。"
        )
    )
    parser.add_argument(
        "--orders-dir",
        type=Path,
        required=True,
        help="003 最终产物目录，包含 part-* JSONL 分片。",
    )
    parser.add_argument(
        "--pid-mapping",
        type=Path,
        required=True,
        help="GenPOI GeoPE PID 的 poi_pid_mapping.parquet。",
    )
    parser.add_argument(
        "--pid-manifest",
        type=Path,
        required=True,
        help="GenPOI GeoPE PID 的 final_pid_manifest.json。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="不存在的输出目录；脚本成功后生成六个正式文件。",
    )
    parser.add_argument("--train-start", required=True, help="训练开始日期。")
    parser.add_argument("--train-end", required=True, help="训练结束日期。")
    parser.add_argument("--valid-date", required=True, help="验证集日期。")
    parser.add_argument("--test-date", required=True, help="测试集日期。")
    parser.add_argument(
        "--history-start",
        required=True,
        help="固定历史窗口开始日期。",
    )
    parser.add_argument(
        "--history-end",
        required=True,
        help="固定历史窗口结束日期。",
    )
    parser.add_argument(
        "--max-history-events",
        type=int,
        default=10,
        help="每个样本最多使用的历史事件数，第一版固定为 10。",
    )
    parser.add_argument(
        "--geohash-length",
        type=int,
        default=6,
        help="用户与 POI GID 长度，GenPOI 第一版固定为 6。",
    )
    parser.add_argument(
        "--sid-codebook-size",
        type=int,
        default=1024,
        help="每层 SID 码本大小，北京 GeoPE 复现固定为 1024。",
    )
    return parser.parse_args()


def _from_project_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        split = TimeSplit(
            train_start=parse_iso_date(args.train_start, "train-start"),
            train_end=parse_iso_date(args.train_end, "train-end"),
            valid_date=parse_iso_date(args.valid_date, "valid-date"),
            test_date=parse_iso_date(args.test_date, "test-date"),
        )
        history_window = HistoryWindow(
            start=parse_iso_date(args.history_start, "history-start"),
            end=parse_iso_date(args.history_end, "history-end"),
        )
        result = build_genpoi_sft_data(
            _from_project_root(args.orders_dir),
            _from_project_root(args.pid_mapping),
            _from_project_root(args.pid_manifest),
            _from_project_root(args.output_dir),
            split,
            history_window,
            max_history_events=args.max_history_events,
            geohash_length=args.geohash_length,
            sid_codebook_size=args.sid_codebook_size,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (GenpoiDataError, OSError, ValueError) as error:
        print(f"GenPOI SFT 数据构建失败：{error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "completed",
                "retained_sample_count": result.stats[
                    "retained_sample_count"
                ],
                "train_count": result.stats["train_count"],
                "valid_count": result.stats["valid_count"],
                "test_count": result.stats["test_count"],
                "history_coverage_ratio": result.stats[
                    "history_coverage_ratio"
                ],
                "average_history_length": result.stats[
                    "average_history_length"
                ],
                "manifest_sha256": result.output_hashes["manifest.json"],
                "output_dir": str(
                    _from_project_root(args.output_dir).resolve()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

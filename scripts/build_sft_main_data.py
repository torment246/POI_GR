#!/usr/bin/env python3
"""Build the deterministic order main-task SFT dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sft_main_data import (
    SftDataValidationError,
    SftMainDataError,
    TimeSplit,
    build_sft_main_data,
    parse_iso_date,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将订单原始 Query 与用户 Geohash6 构造成 Messages SFT 输入，"
            "目标为正式唯一 Final PID；只过滤空 Query。"
        )
    )
    parser.add_argument(
        "--orders-dir",
        type=Path,
        required=True,
        help="包含排序后 part-* JSONL 分片的订单目录。",
    )
    parser.add_argument(
        "--pid-mapping",
        type=Path,
        required=True,
        help="PID-002 的 poi_pid_mapping.parquet。",
    )
    parser.add_argument(
        "--pid-manifest",
        type=Path,
        required=True,
        help="PID-002 的 final_pid_manifest.json。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录，只保留本任务规定的六个文件。",
    )
    parser.add_argument("--train-start", required=True, help="训练开始日期。")
    parser.add_argument("--train-end", required=True, help="训练结束日期。")
    parser.add_argument("--valid-date", required=True, help="验证集日期。")
    parser.add_argument("--test-date", required=True, help="测试集日期。")
    parser.add_argument(
        "--geohash-length",
        type=int,
        default=6,
        help="用户位置 Geohash 长度；本任务固定为 6。",
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
        result = build_sft_main_data(
            _from_project_root(args.orders_dir),
            _from_project_root(args.pid_mapping),
            _from_project_root(args.pid_manifest),
            _from_project_root(args.output_dir),
            split,
            geohash_length=args.geohash_length,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except SftDataValidationError as error:
        print(
            "SFT 数据构建失败："
            + json.dumps(error.summary(), ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        return 2
    except (SftMainDataError, OSError, ValueError) as error:
        print(f"SFT 数据构建失败：{error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "completed",
                "raw_order_count": result.stats["raw_order_count"],
                "blank_query_count": result.stats["blank_query_count"],
                "retained_sample_count": result.stats[
                    "retained_sample_count"
                ],
                "train_count": result.stats["train_count"],
                "valid_count": result.stats["valid_count"],
                "test_count": result.stats["test_count"],
                "pid_match_ratio": result.stats["pid_match_ratio"],
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

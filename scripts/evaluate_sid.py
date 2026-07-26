#!/usr/bin/env python3
"""Evaluate SID collision, codebook usage, prefixes, purity, and Cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sid_evaluation import SidEvaluationError, run_sid_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "评估任意层数的 SID：输入校验、完整 SID 碰撞、逐层码本利用率、"
            "前缀桶、完整 category_code 纯度和热点碰撞 Case。"
        ),
        epilog=(
            "manifest schema_version 必须为 sid-input-v1，并包含 "
            "sid_codes={path, shape:[N,L], dtype}、poi_ids={path} 和 "
            "codebook_sizes。相对文件路径相对于 manifest 所在目录。"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="SID 输入 manifest JSON；相对路径相对于仓库根目录。",
    )
    parser.add_argument(
        "--poi-data",
        type=Path,
        required=True,
        help="POI JSONL 文件或包含 part-* 分片的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录，只写 metrics.json 和 collision_cases.jsonl。",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=20,
        help="最多输出的碰撞桶数，默认 20。",
    )
    parser.add_argument(
        "--max-pois-per-case",
        type=int,
        default=20,
        help="每个碰撞桶最多输出的 POI 数，默认 20。",
    )
    return parser.parse_args()


def _resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _resolve_from_root(args.output_dir)
    try:
        metrics, cases = run_sid_evaluation(
            _resolve_from_root(args.manifest),
            _resolve_from_root(args.poi_data),
            output_dir,
            max_cases=args.max_cases,
            max_pois_per_case=args.max_pois_per_case,
        )
    except SidEvaluationError as error:
        print(f"SID 评估失败：{error}", file=sys.stderr)
        return 2

    summary = {
        "status": "completed",
        "poi_count": metrics["basic"]["poi_count"],
        "distinct_sid_count": metrics["basic"]["distinct_sid_count"],
        "collision_excess_ratio": metrics["basic"]["collision_excess_ratio"],
        "colliding_poi_ratio": metrics["basic"]["colliding_poi_ratio"],
        "prefix_bucket_counts": [
            prefix["prefix_bucket_count"] for prefix in metrics["prefixes"]
        ],
        "collision_case_count": len(cases),
        "outputs": [
            str(output_dir.resolve() / "metrics.json"),
            str(output_dir.resolve() / "collision_cases.jsonl"),
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

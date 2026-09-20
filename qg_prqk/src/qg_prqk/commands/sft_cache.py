"""Validate SFT token lengths and build reusable caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from qg_prqk.sft.preflight import SftPreflightError, prepare_sft_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="全量扫描 QG-PRQK Train/Valid，并在零截断后构建 1024 Token Cache。"
    )
    parser.add_argument("--variant", choices=("a4_gid_parent", "a4_nogid"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--valid-dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--preprocessing-batch-size", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stats, cache = prepare_sft_cache(
            variant=args.variant,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            train_dataset=args.train_dataset,
            valid_dataset=args.valid_dataset,
            batch_size=args.batch_size,
            workers=args.workers,
            preprocessing_batch_size=args.preprocessing_batch_size,
        )
    except (SftPreflightError, OSError, ValueError) as error:
        print(f"QG-PRQK Token 预检/缓存失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": cache["status"],
                "variant": args.variant,
                "train_valid_rows": stats["row_count"],
                "cutoff_len": stats["cutoff_len"],
                "over_cutoff_count": stats["over_cutoff_count"],
                "target_truncated_count": stats["target_truncated_count"],
                "packed_rows": cache["packed_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0

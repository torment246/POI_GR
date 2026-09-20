"""Build paired history-aware SFT datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from qg_prqk.sid.identifiers import FinalIdentifierError
from qg_prqk.sft.data import SftDataError, build_paired_sft_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从冻结 history10 Messages 一次扫描构造 A4 GID-parent 与 NoGID "
            "两套逐行对齐的 QG-PRQK SFT 数据。"
        )
    )
    parser.add_argument("--source-sft-dir", type=Path, required=True)
    parser.add_argument("--tiger-identifier-dir", type=Path, required=True)
    parser.add_argument("--gid-final-id-dir", type=Path, required=True)
    parser.add_argument("--nogid-final-id-dir", type=Path, required=True)
    parser.add_argument("--gid-output-dir", type=Path, required=True)
    parser.add_argument("--nogid-output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        help="仅用于 smoke；正式全量构建不要传。",
    )
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_paired_sft_data(
            source_sft_dir=args.source_sft_dir,
            tiger_identifier_dir=args.tiger_identifier_dir,
            final_identifier_dirs={
                "a4_gid_parent": args.gid_final_id_dir,
                "a4_nogid": args.nogid_final_id_dir,
            },
            output_dirs={
                "a4_gid_parent": args.gid_output_dir,
                "a4_nogid": args.nogid_output_dir,
            },
            max_rows_per_split=args.max_rows_per_split,
            progress_every=args.progress_every,
            progress=lambda value: print(value, file=sys.stderr, flush=True),
        )
    except (SftDataError, FinalIdentifierError, OSError, ValueError) as error:
        print(f"QG-PRQK SFT 数据构建失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "scan_mode": result.manifests["a4_gid_parent"]["scan_mode"],
                "train_count": result.stats["train_count"],
                "valid_count": result.stats["valid_count"],
                "test_count": result.stats["test_count"],
                "gid_manifest_sha256": result.output_hashes[
                    "a4_gid_parent"
                ]["manifest.json"],
                "nogid_manifest_sha256": result.output_hashes[
                    "a4_nogid"
                ]["manifest.json"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0

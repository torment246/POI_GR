"""Build unique final identifiers for one SID variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from qg_prqk.sid.identifiers import FinalIdentifierError, VARIANTS, build_final_identifiers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从冻结的 A4 SID 构建唯一 Final ID；GID-parent 输出 "
            "GID6+SID3+[D]，NoGID 输出 SID3+[D]，Dedup 始终位于最后。"
        )
    )
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--sid-mapping", type=Path, required=True)
    parser.add_argument("--sid-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gid-codes", type=Path)
    parser.add_argument("--gid-manifest", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_final_identifiers(
            variant=args.variant,
            sid_mapping_path=args.sid_mapping,
            sid_manifest_path=args.sid_manifest,
            output_dir=args.output_dir,
            gid_codes_path=args.gid_codes,
            gid_manifest_path=args.gid_manifest,
            chunk_rows=args.chunk_rows,
            progress=lambda value: print(value, file=sys.stderr, flush=True),
        )
    except (FinalIdentifierError, OSError, ValueError) as error:
        print(f"QG-PRQK Final ID 构建失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "variant": args.variant,
                "poi_count": result.metrics["poi_count"],
                "base_distinct_ratio": result.metrics["base_distinct_ratio"],
                "dedup_token_capacity": result.metrics["dedup_token_capacity"],
                "final_unique_ratio": result.metrics["final_unique_ratio"],
                "manifest_sha256": result.manifest_sha256,
                "output_dir": str(result.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0

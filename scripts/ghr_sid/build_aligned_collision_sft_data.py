#!/usr/bin/env python3
"""Build paired SFT Messages for the fixed five-layer GHR-SID."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.aligned_collision_sft_data import (  # noqa: E402
    AlignedCollisionSftDataError,
    build_aligned_collision_sft_data,
)
from poi_gr.methods.ghr_sid.sft_data import GhrSftDataError  # noqa: E402
from poi_gr.methods.tiger.data import TigerDataError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "逐行复用冻结 TIGER SFT，只把历史和目标 POI identifier 替换为"
            "固定五层 [S1,S2,S3,R1,R2] GHR-SID。"
        )
    )
    parser.add_argument("--tiger-sft-dir", type=Path, required=True)
    parser.add_argument("--tiger-id-dir", type=Path, required=True)
    parser.add_argument("--aligned-id-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="仅用于 smoke；正式全量构建不要传。",
    )
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        result = build_aligned_collision_sft_data(
            source_tiger_sft_dir=_rooted(args.tiger_sft_dir),
            tiger_identifier_dir=_rooted(args.tiger_id_dir),
            aligned_identifier_dir=_rooted(args.aligned_id_dir),
            output_dir=_rooted(args.output_dir),
            max_rows_per_split=args.max_rows_per_split,
            progress_every=args.progress_every,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (
        AlignedCollisionSftDataError,
        GhrSftDataError,
        TigerDataError,
        OSError,
        ValueError,
    ) as error:
        print(f"aligned-collision SFT 数据构建失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result.manifest["status"],
                "scan_mode": result.manifest["input"]["scan_mode"],
                "row_counts": result.manifest["alignment"]["row_counts"],
                "token_count": json.loads(
                    (_rooted(args.output_dir) / "special_tokens.json").read_text(
                        encoding="utf-8"
                    )
                )["token_count"],
                "manifest_sha256": result.output_hashes["manifest.json"],
                "output_dir": str(_rooted(args.output_dir).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Derive EXP-09 and EXP-14 SFT Messages from the frozen TIGER rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.identifier import GhrIdentifierError  # noqa: E402
from poi_gr.methods.ghr_sid.sft_data import (  # noqa: E402
    GhrSftCandidate,
    GhrSftDataError,
    build_paired_ghr_sft_data,
)
from poi_gr.methods.tiger.data import TigerDataError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将冻结 TIGER SFT 逐行只替换为 EXP-09/EXP-14 的变长唯一 GHR-SID，"
            "从构造上保证切分、Query、请求 GID 和历史窗口一致。"
        )
    )
    parser.add_argument("--tiger-sft-dir", type=Path, required=True)
    parser.add_argument("--tiger-id-dir", type=Path, required=True)
    parser.add_argument("--exp09-id-dir", type=Path, required=True)
    parser.add_argument("--exp09-output-dir", type=Path, required=True)
    parser.add_argument("--exp14-id-dir", type=Path, required=True)
    parser.add_argument("--exp14-output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-rows-per-split", type=int, default=None, help="仅用于 smoke；正式构建不传。",
    )
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    candidates = (
        GhrSftCandidate(
            variant="exp09",
            identifier_dir=_rooted(args.exp09_id_dir),
            output_dir=_rooted(args.exp09_output_dir),
        ),
        GhrSftCandidate(
            variant="exp14",
            identifier_dir=_rooted(args.exp14_id_dir),
            output_dir=_rooted(args.exp14_output_dir),
        ),
    )
    try:
        result = build_paired_ghr_sft_data(
            source_tiger_sft_dir=_rooted(args.tiger_sft_dir),
            tiger_identifier_dir=_rooted(args.tiger_id_dir),
            candidates=candidates,
            max_rows_per_split=args.max_rows_per_split,
            progress_every=args.progress_every,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (
        GhrSftDataError,
        GhrIdentifierError,
        TigerDataError,
        OSError,
        ValueError,
    ) as error:
        print(f"GHR SFT 数据构建失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                variant: {
                    "status": manifest["status"],
                    "scan_mode": manifest["input"]["scan_mode"],
                    "row_counts": manifest["alignment"]["row_counts"],
                    "token_count": json.loads(
                        (
                            candidates[index].output_dir / "special_tokens.json"
                        ).read_text(encoding="utf-8")
                    )["token_count"],
                    "manifest_sha256": result.output_hashes[variant]["manifest.json"],
                    "output_dir": str(candidates[index].output_dir.resolve()),
                }
                for index, (variant, manifest) in enumerate(result.manifests.items())
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

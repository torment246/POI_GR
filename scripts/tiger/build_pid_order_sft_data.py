#!/usr/bin/env python3
"""Build paired GID-first and SID-first TIGER PID SFT datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger.pid_order_data import (  # noqa: E402
    TigerPidOrderDataError,
    build_tiger_pid_order_sft_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从冻结的 TIGER history10 SFT 数据和 TIGER-SID+Geohash6 "
            "唯一映射，一次扫描同时生成 GID+SID+D 与 SID+GID+D 两版数据。"
        )
    )
    parser.add_argument("--source-sft-dir", type=Path, required=True)
    parser.add_argument("--tiger-id-dir", type=Path, required=True)
    parser.add_argument("--pid-mapping", type=Path, required=True)
    parser.add_argument("--pid-manifest", type=Path, required=True)
    parser.add_argument("--gid-sid-output-dir", type=Path, required=True)
    parser.add_argument("--sid-gid-output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="仅用于 smoke；每个切分只转换前 N 行。正式构建不要传。",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        result = build_tiger_pid_order_sft_data(
            _resolve(args.source_sft_dir),
            _resolve(args.tiger_id_dir),
            _resolve(args.pid_mapping),
            _resolve(args.pid_manifest),
            {
                "gid_sid": _resolve(args.gid_sid_output_dir),
                "sid_gid": _resolve(args.sid_gid_output_dir),
            },
            max_rows_per_split=args.max_rows_per_split,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (TigerPidOrderDataError, OSError, ValueError) as error:
        print(f"TIGER PID 顺序消融数据构建失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "scan_mode": result.manifests["gid_sid"]["scan_mode"],
                "train_count": result.stats["train_count"],
                "valid_count": result.stats["valid_count"],
                "test_count": result.stats["test_count"],
                "requires_dedup_sample_count": result.stats[
                    "requires_dedup_sample_count"
                ],
                "gid_sid_manifest_sha256": result.output_hashes[
                    "gid_sid"
                ]["manifest.json"],
                "sid_gid_manifest_sha256": result.output_hashes[
                    "sid_gid"
                ]["manifest.json"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

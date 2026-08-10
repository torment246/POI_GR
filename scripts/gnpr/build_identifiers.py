#!/usr/bin/env python3
"""Append deterministic GNPR dedup tokens to colliding Semantic IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.gnpr.identifier import (
    GnprIdentifierError,
    build_gnpr_identifiers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 GNPR 三层 RQ-VAE Semantic ID，仅对碰撞桶内 POI 按 "
            "poi_id 字典序追加从 0 开始的 dedup code；单例保持三 token，"
            "输出论文兼容且全局唯一的 GNPR identifier。"
        )
    )
    parser.add_argument(
        "--sid-manifest",
        type=Path,
        required=True,
        help="三层 SID 的 sid_manifest.json；相对路径相对于仓库根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="GNPR identifier 五个正式产物的输出目录。",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=100_000,
        help="Parquet 映射表单批行数，默认 100000。",
    )
    return parser.parse_args()


def _from_project_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        result = build_gnpr_identifiers(
            _from_project_root(args.sid_manifest),
            _from_project_root(args.output_dir),
            chunk_rows=args.chunk_rows,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (GnprIdentifierError, OSError, ValueError) as error:
        print(f"GNPR identifier 构建失败：{error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "completed",
                "poi_count": result.metrics["poi_count"],
                "base_sid_distinct_ratio": result.metrics[
                    "base_sid_distinct_ratio"
                ],
                "poi_with_dedup_token_count": result.metrics[
                    "poi_with_dedup_token_count"
                ],
                "required_dedup_token_count": result.metrics[
                    "required_dedup_token_count"
                ],
                "gnpr_id_unique_ratio": result.metrics[
                    "gnpr_id_unique_ratio"
                ],
                "mapping_sha256": result.mapping_sha256,
                "output_dir": str(_from_project_root(args.output_dir).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare full-catalog content-geographic GNPR-SID metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.gnpr.content_geo import (  # noqa: E402
    GnprContentGeoError,
    prepare_content_geo_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按 BGE-M3 行序对齐全量 POI 的 category_code 和 PlusCode6 索引；"
            "只保存紧凑索引，训练时再按 batch 融合，不写全量稠密向量。"
        )
    )
    parser.add_argument(
        "--embedding-dir", type=Path, required=True, help="BGE-M3 输出目录。"
    )
    parser.add_argument(
        "--feature-dir", type=Path, required=True, help="004 全量 GNPR 特征目录。"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="不存在的输出目录。")
    parser.add_argument("--expected-rows", type=int, help="可选的 BGE 全量行数门禁。")
    parser.add_argument("--max-rows", type=int, help="仅用于合成 smoke 的输出行数上限。")
    parser.add_argument("--allow-feature-superset", action="store_true",
                        help="按向量 POI 子集复用全库静态特征；保留原词表，要求全部匹配且无重复。")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    try:
        manifest = prepare_content_geo_metadata(
            embedding_dir=_resolve(args.embedding_dir),
            feature_dir=_resolve(args.feature_dir),
            output_dir=_resolve(args.output_dir),
            expected_rows=args.expected_rows,
            max_rows=args.max_rows,
            allow_feature_superset=args.allow_feature_superset,
        )
    except (GnprContentGeoError, OSError, RuntimeError, ValueError) as error:
        print(f"GNPR content-geo 输入构建失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

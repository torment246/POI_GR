#!/usr/bin/env python3
"""Build frozen E4 category-common residual Query aggregates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.query_category_residual import (  # noqa: E402
    build_query_category_residual,
    load_query_category_residual_config,
    validate_query_category_residual,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="流式构建或校验 E4 类别共性 Query 残差聚合。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/embedding/embedding_query_category_residual_bge_m3_v1.yaml"),
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (
        args.config
        if args.config.is_absolute()
        else PROJECT_ROOT / args.config
    )
    config = load_query_category_residual_config(config_path, PROJECT_ROOT)
    if args.validate_only:
        result = validate_query_category_residual(
            config.output_dir,
            expected_aggregates_manifest_sha256=(
                config.expected_aggregates_manifest_sha256
            ),
            expected_category_indices_sha256=(
                config.expected_category_indices_sha256
            ),
        )
    else:
        result = build_query_category_residual(
            config, project_root=PROJECT_ROOT, show_progress=True
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "covered_pois": result.covered_pois,
                "candidates": result.candidates,
                "reused": result.reused,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build active-POI BGE and category assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qg_prqk.data.active_poi import (
    ActivePoiAssetsError,
    build_active_poi_assets,
    load_active_poi_assets_config,
    validate_active_poi_assets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 active_poi 行序切片冻结 BGE 和类别索引，不重新编码 POI。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="独立复核已有 active 资产，不写入或覆盖文件。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_active_poi_assets_config(args.config)
        manifest = (
            validate_active_poi_assets(config)
            if args.validate_only
            else build_active_poi_assets(config)
        )
    except (ActivePoiAssetsError, OSError, ValueError) as error:
        print(f"QG-PRQK active POI assets failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "validated" if args.validate_only else "completed",
                "output_dir": str(config.paths.output_dir),
                "active_rows": manifest["row_order"]["active_rows"],
                "artifacts": manifest["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0

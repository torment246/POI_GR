#!/usr/bin/env python3
"""Build the active Beijing POI catalog used by current targets and histories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.data import ActiveCatalogConfig, build_active_poi_catalog


DEFAULT_ORDERS_DIR = Path(
    "data/beijing_order_clean_20260701_20260714_"
    "history10_20260401_20260630_json"
)
DEFAULT_POI_DIR = Path("data/beijing_poi_clean_20260715_json")
DEFAULT_OUTPUT_DIR = Path(
    "data/beijing_poi_active_order14d_history10_20260715_json"
)
DEFAULT_TEMP_DIR = Path("outputs/tmp/active_poi_catalog_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从两周全部目标 POI 与已保留 history_sequence POI 的并集中，"
            "按原始行序筛选北京全量 POI 主表。"
        )
    )
    parser.add_argument("--orders-dir", type=Path, default=DEFAULT_ORDERS_DIR)
    parser.add_argument("--poi-dir", type=Path, default=DEFAULT_POI_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    parser.add_argument("--order-pattern", default="part-*.json")
    parser.add_argument("--poi-pattern", default="part-*.json")
    parser.add_argument("--progress-interval-rows", type=int, default=1_000_000)
    parser.add_argument("--expected-order-rows", type=int, default=8_790_513)
    parser.add_argument(
        "--expected-history-occurrences", type=int, default=43_208_167
    )
    parser.add_argument("--expected-current-unique", type=int, default=520_333)
    parser.add_argument("--expected-history-unique", type=int, default=607_456)
    parser.add_argument("--expected-union-unique", type=int, default=716_245)
    parser.add_argument("--expected-poi-rows", type=int, default=2_337_178)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    manifest = build_active_poi_catalog(
        ActiveCatalogConfig(
            orders_dir=_resolve(args.orders_dir),
            poi_dir=_resolve(args.poi_dir),
            output_dir=_resolve(args.output_dir),
            temp_dir=_resolve(args.temp_dir),
            order_pattern=args.order_pattern,
            poi_pattern=args.poi_pattern,
            expected_order_rows=args.expected_order_rows,
            expected_history_occurrences=args.expected_history_occurrences,
            expected_current_unique=args.expected_current_unique,
            expected_history_unique=args.expected_history_unique,
            expected_union_unique=args.expected_union_unique,
            expected_poi_rows=args.expected_poi_rows,
            progress_interval_rows=args.progress_interval_rows,
        )
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "rows": manifest["output"]["rows"],
                "output_dir": manifest["output"]["dir"],
                "poi_ids_sha256": manifest["output"]["poi_ids_sha256"],
                "elapsed_seconds": manifest["runtime"]["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

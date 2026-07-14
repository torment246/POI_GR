#!/usr/bin/env python3
"""Generate PPT-ready SID quality figures for MobilityBench POI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sid_visualization import configure_matplotlib, generate_all_figures


DEFAULTS = {
    "semantic_report": "outputs/reports/sid_quality_semantic.md",
    "geo_fused_report": "outputs/reports/sid_quality_geo_fused.md",
    "compare_report": "outputs/reports/sid_quality_compare.md",
    "semantic_mapping": "data/sid/poi_sid_mapping_semantic.parquet",
    "geo_fused_mapping": "data/sid/poi_sid_mapping_geo_fused.parquet",
    "out_dir": "outputs/figures/sid_quality",
}

OPTIONAL_INPUTS = [
    "outputs/reports/sid_collision_groups_semantic.csv",
    "outputs/reports/sid_collision_groups_geo_fused.csv",
    "outputs/reports/sid_prefix1_summary_semantic.csv",
    "outputs/reports/sid_prefix1_summary_geo_fused.csv",
    "outputs/reports/sid_prefix2_summary_semantic.csv",
    "outputs/reports/sid_prefix2_summary_geo_fused.csv",
    "outputs/reports/sid_prefix3_summary_semantic.csv",
    "outputs/reports/sid_prefix3_summary_geo_fused.csv",
    "outputs/reports/sid_manual_cluster_samples_semantic.csv",
    "outputs/reports/sid_manual_cluster_samples_geo_fused.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-report", default=DEFAULTS["semantic_report"])
    parser.add_argument("--geo-fused-report", default=DEFAULTS["geo_fused_report"])
    parser.add_argument("--compare-report", default=DEFAULTS["compare_report"])
    parser.add_argument("--semantic-mapping", default=DEFAULTS["semantic_mapping"])
    parser.add_argument("--geo-fused-mapping", default=DEFAULTS["geo_fused_mapping"])
    parser.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    return parser.parse_args()


def require_file(path: str | Path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def check_inputs(args: argparse.Namespace) -> None:
    for path in [
        args.semantic_report,
        args.geo_fused_report,
        args.compare_report,
        args.semantic_mapping,
        args.geo_fused_mapping,
    ]:
        require_file(path)

    missing_optional = [path for path in OPTIONAL_INPUTS if not Path(path).exists()]
    if missing_optional:
        print("Optional CSV inputs missing; figures will use markdown/parquet metrics where needed:")
        for path in missing_optional:
            print(f"- {path}")


def main() -> None:
    args = parse_args()
    check_inputs(args)

    font_name, font_path = configure_matplotlib()
    if font_path:
        print(f"Using Chinese font: {font_name} ({font_path})")
    else:
        print(f"Using fallback font: {font_name}. If Chinese text renders incorrectly, install Noto Sans CJK SC or SimHei.")

    figures = generate_all_figures(
        out_dir=args.out_dir,
        semantic_report=args.semantic_report,
        compare_report=args.compare_report,
        semantic_mapping=args.semantic_mapping,
    )

    print("\nGenerated SID visualization figures:")
    for idx, fig in enumerate(figures, start=1):
        print(f"- fig{idx}: {fig['png']} | {fig['svg']}")


if __name__ == "__main__":
    main()

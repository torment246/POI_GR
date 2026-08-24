#!/usr/bin/env python3
"""Audit names, addresses and geography of GHR-SID residual POIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.residual import (  # noqa: E402
    GhrSidResidualError,
    audit_residual_pois,
    validate_residual_output,
)


DEFAULT_POI_DIR = "data/beijing_poi_clean_20260715_json"
DEFAULT_STRUCTURE_DIR = (
    "outputs/ghr_sid/EXP-20260814-02_beijing_geo_relation_structure_v1"
)
DEFAULT_M2A_DIR = (
    "outputs/qgr_sid/EXP-20260813-07_tiger_lexical_relation_proxy_v1"
)
DEFAULT_EXPERIMENT_ID = "EXP-20260814-03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "回查 GID8+全17类静态关系仍同签名的 POI，量化 GID9/GID10、"
            "精确坐标、名称、地址和其他静态字段的区分贡献。"
        )
    )
    parser.add_argument("--poi-dir", type=Path, default=Path(DEFAULT_POI_DIR))
    parser.add_argument(
        "--structure-dir", type=Path, default=Path(DEFAULT_STRUCTURE_DIR)
    )
    parser.add_argument("--m2a-dir", type=Path, default=Path(DEFAULT_M2A_DIR))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--cases-per-type", type=int, default=10)
    parser.add_argument("--pois-per-case", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _rooted(args.output_dir)
    try:
        if args.validate_only:
            payload = validate_residual_output(output_dir)
        else:
            result = audit_residual_pois(
                project_root=PROJECT_ROOT,
                poi_dir=_rooted(args.poi_dir),
                structure_dir=_rooted(args.structure_dir),
                m2a_dir=_rooted(args.m2a_dir),
                output_dir=output_dir,
                experiment_id=args.experiment_id,
                batch_rows=args.batch_rows,
                cases_per_type=args.cases_per_type,
                pois_per_case=args.pois_per_case,
                progress=lambda message: print(
                    message, file=sys.stderr, flush=True
                ),
            )
            final_metrics = result.metrics["all_static_final"]
            payload = {
                "status": result.metrics["status"],
                "experiment_id": result.metrics["experiment_id"],
                "feature_irreducible": result.metrics["feature_irreducible"],
                "all_static_final": final_metrics,
                "output_dir": str(result.output_dir),
            }
    except (GhrSidResidualError, OSError, ValueError) as error:
        print(f"GHR-SID 残留审计失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

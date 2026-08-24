#!/usr/bin/env python3
"""Run TIGER3 -> G6 -> strong relation tree -> leaf Dedup evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.entity_structure import (  # noqa: E402
    GhrSidEntityStructureError,
)
from poi_gr.methods.ghr_sid.g6_relation_tree import (  # noqa: E402
    evaluate_g6_relation_tree,
    validate_g6_relation_tree_output,
)


DEFAULT_POI_DIR = "data/beijing_poi_clean_20260715_json"
DEFAULT_STRUCTURE_DIR = (
    "outputs/ghr_sid/EXP-20260814-02_beijing_geo_relation_structure_v1"
)
DEFAULT_PROXY_DIR = (
    "outputs/qgr_sid/EXP-20260813-07_tiger_lexical_relation_proxy_v1"
)
DEFAULT_EXPERIMENT_ID = "EXP-20260814-12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在冻结 TIGER3+G6 桶中按强实体角色建立共享关系树，并且只在"
            "最终语义叶子追加稳定 Dedup。"
        )
    )
    parser.add_argument("--poi-dir", type=Path, default=Path(DEFAULT_POI_DIR))
    parser.add_argument(
        "--structure-dir", type=Path, default=Path(DEFAULT_STRUCTURE_DIR)
    )
    parser.add_argument(
        "--proxy-dir", type=Path, default=Path(DEFAULT_PROXY_DIR)
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--entity-min-support", type=int, default=2)
    parser.add_argument("--entity-vocab-size", type=int, default=32_767)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--max-pois-per-case", type=int, default=300)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _rooted(args.output_dir)
    try:
        if args.validate_only:
            payload = validate_g6_relation_tree_output(output_dir)
        else:
            result = evaluate_g6_relation_tree(
                project_root=PROJECT_ROOT,
                poi_dir=_rooted(args.poi_dir),
                structure_dir=_rooted(args.structure_dir),
                proxy_dir=_rooted(args.proxy_dir),
                output_dir=output_dir,
                experiment_id=args.experiment_id,
                entity_min_support=args.entity_min_support,
                entity_vocab_size=args.entity_vocab_size,
                max_cases=args.max_cases,
                max_pois_per_case=args.max_pois_per_case,
                progress=lambda message: print(
                    message, file=sys.stderr, flush=True
                ),
            )
            payload = {
                "status": result.metrics["status"],
                "experiment_id": result.metrics["experiment_id"],
                "main_protocol": result.metrics["main_protocol"],
                "main": result.metrics["main"],
                "focus_crown_city_gate": result.metrics[
                    "focus_crown_city_gate"
                ],
                "green_c12b_gate": result.metrics["green_c12b_gate"],
                "decision": result.metrics["decision"],
                "output_dir": str(result.output_dir),
            }
    except (GhrSidEntityStructureError, OSError, ValueError) as error:
        print(f"GHR-SID 关系树实验失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run TIGER3 -> G6 -> minimum entity description structural evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.g6_entity_structure import (  # noqa: E402
    evaluate_g6_entity_structure,
    validate_g6_entity_structure_output,
)
from poi_gr.methods.ghr_sid.entity_structure import (  # noqa: E402
    GhrSidEntityStructureError,
)


DEFAULT_POI_DIR = "data/beijing_poi_clean_20260715_json"
DEFAULT_STRUCTURE_DIR = (
    "outputs/ghr_sid/EXP-20260814-02_beijing_geo_relation_structure_v1"
)
DEFAULT_PROXY_DIR = (
    "outputs/qgr_sid/EXP-20260813-07_tiger_lexical_relation_proxy_v1"
)
DEFAULT_EXPERIMENT_ID = "EXP-20260814-08"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "删除独立 R3 阶段，在冻结 TIGER+G6 碰撞桶内统一比较实体角色，"
            "精确求解受 Token 预算约束的最短可区分关系描述。"
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
    parser.add_argument("--max-relation-pairs", type=int, default=3)
    parser.add_argument("--max-relation-tokens", type=int, default=6)
    parser.add_argument("--examples-per-kind", type=int, default=20)
    parser.add_argument("--pois-per-case", type=int, default=12)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _rooted(args.output_dir)
    try:
        if args.validate_only:
            payload = validate_g6_entity_structure_output(output_dir)
        else:
            result = evaluate_g6_entity_structure(
                project_root=PROJECT_ROOT,
                poi_dir=_rooted(args.poi_dir),
                structure_dir=_rooted(args.structure_dir),
                proxy_dir=_rooted(args.proxy_dir),
                output_dir=output_dir,
                experiment_id=args.experiment_id,
                entity_min_support=args.entity_min_support,
                entity_vocab_size=args.entity_vocab_size,
                max_relation_pairs=args.max_relation_pairs,
                max_relation_tokens=args.max_relation_tokens,
                examples_per_kind=args.examples_per_kind,
                pois_per_case=args.pois_per_case,
                progress=lambda message: print(
                    message, file=sys.stderr, flush=True
                ),
            )
            payload = {
                "status": result.metrics["status"],
                "experiment_id": result.metrics["experiment_id"],
                "main_protocol": result.metrics["main_protocol"],
                "main": result.metrics["main"],
                "decision": result.metrics["decision"],
                "output_dir": str(result.output_dir),
            }
    except (GhrSidEntityStructureError, OSError, ValueError) as error:
        print(f"GHR-SID G6 实体关系实验失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

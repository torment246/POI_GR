#!/usr/bin/env python3
"""Audit relation-solvable collisions after exact-vector deduplication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.distinct_vector_relation_audit import (  # noqa: E402
    DistinctVectorRelationAuditError,
    audit_distinct_vector_relations,
    validate_distinct_vector_relation_audit_output,
)


DEFAULT_POI_DIR = "data/beijing_poi_clean_20260715_json"
DEFAULT_EXP09_DIR = "outputs/ghr_sid/EXP-20260814-09_beijing_g6_minimum_entity_v1"
DEFAULT_EXP10_DIR = "outputs/ghr_sid/EXP-20260814-10_residual_vector_gid_audit_v1"
DEFAULT_OUTPUT_DIR = (
    "outputs/ghr_sid/EXP-20260814-11_distinct_vector_relation_audit_v1"
)
DEFAULT_EXPERIMENT_ID = "EXP-20260814-11"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="折叠完全相同 BGE 向量后，审计实体关系可解决的残留碰撞。"
    )
    parser.add_argument("--poi-dir", type=Path, default=Path(DEFAULT_POI_DIR))
    parser.add_argument("--exp09-dir", type=Path, default=Path(DEFAULT_EXP09_DIR))
    parser.add_argument("--exp10-dir", type=Path, default=Path(DEFAULT_EXP10_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--cases-per-type", type=int, default=8)
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
            payload = validate_distinct_vector_relation_audit_output(output_dir)
        else:
            result = audit_distinct_vector_relations(
                project_root=PROJECT_ROOT,
                poi_dir=_rooted(args.poi_dir),
                exp09_dir=_rooted(args.exp09_dir),
                exp10_dir=_rooted(args.exp10_dir),
                output_dir=output_dir,
                experiment_id=args.experiment_id,
                cases_per_type=args.cases_per_type,
                pois_per_case=args.pois_per_case,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            payload = {
                "status": result.metrics["status"],
                "experiment_id": result.metrics["experiment_id"],
                "baseline": result.metrics["baseline"],
                "decision": result.metrics["decision"],
                "output_dir": str(result.output_dir),
            }
    except (DistinctVectorRelationAuditError, OSError, ValueError) as error:
        print(f"GHR-SID 去重后关系审计失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

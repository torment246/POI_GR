#!/usr/bin/env python3
"""Run the Beijing query-free GHR-SID structural experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.structure import (  # noqa: E402
    GhrSidStructureError,
    evaluate_structure_experiment,
    validate_structure_output,
)


DEFAULT_POI_DIR = "data/beijing_poi_clean_20260715_json"
DEFAULT_M2A_DIR = (
    "outputs/qgr_sid/EXP-20260813-07_tiger_lexical_relation_proxy_v1"
)
DEFAULT_GID_DIR = "outputs/pid/BJ-RQVAE-1024x3-e20-G6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在北京 TIGER 碰撞桶上运行不读取 Query/订单的 GHR-SID 结构实验："
            "粗粒度 Geohash → 静态数字关系树 → 细粒度 Geohash 兜底。"
        )
    )
    parser.add_argument("--poi-dir", type=Path, default=Path(DEFAULT_POI_DIR))
    parser.add_argument("--m2a-dir", type=Path, default=Path(DEFAULT_M2A_DIR))
    parser.add_argument(
        "--gid-codes",
        type=Path,
        default=Path(DEFAULT_GID_DIR) / "gid_codes.npy",
    )
    parser.add_argument(
        "--gid-manifest",
        type=Path,
        default=Path(DEFAULT_GID_DIR) / "pid_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--coarse-precisions",
        type=int,
        nargs="+",
        default=[5, 6, 7],
    )
    parser.add_argument("--main-coarse-precision", type=int, default=6)
    parser.add_argument("--fine-precision", type=int, default=8)
    parser.add_argument("--max-relation-pairs", type=int, default=3)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--examples-per-stage", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _rooted(args.output_dir)
    try:
        if args.validate_only:
            payload = validate_structure_output(output_dir)
        else:
            result = evaluate_structure_experiment(
                project_root=PROJECT_ROOT,
                poi_dir=_rooted(args.poi_dir),
                m2a_dir=_rooted(args.m2a_dir),
                gid_codes_path=_rooted(args.gid_codes),
                gid_manifest_path=_rooted(args.gid_manifest),
                output_dir=output_dir,
                coarse_precisions=args.coarse_precisions,
                main_coarse_precision=args.main_coarse_precision,
                fine_precision=args.fine_precision,
                max_relation_pairs=args.max_relation_pairs,
                batch_rows=args.batch_rows,
                examples_per_stage=args.examples_per_stage,
                progress=lambda message: print(
                    message, file=sys.stderr, flush=True
                ),
            )
            resolution = result.metrics["main"]["resolution"]
            identifier = result.metrics["main"]["identifier"]
            payload = {
                "status": result.metrics["status"],
                "main_protocol": result.metrics["main_protocol"],
                "query_usage": result.metrics["query_usage"],
                "semantic_resolved_poi_ratio": resolution[
                    "semantic_resolved_poi_ratio"
                ],
                "unresolved_poi_count": resolution["unresolved_poi_count"],
                "full_distinct_identifier_ratio": identifier[
                    "full_distinct_identifier_ratio"
                ],
                "output_dir": str(result.output_dir),
            }
    except (GhrSidStructureError, OSError, ValueError) as error:
        print(f"GHR-SID 结构实验失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

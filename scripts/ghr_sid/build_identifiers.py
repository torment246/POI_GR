#!/usr/bin/env python3
"""Build full-catalog unique EXP-09 or EXP-14 GHR identifiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.identifier import (  # noqa: E402
    GhrIdentifierError,
    SUPPORTED_VARIANTS,
    build_ghr_identifiers,
    validate_ghr_identifier_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将冻结 TIGER3、G6 和 EXP-09/14 关系路径组合为全目录唯一的变长 "
            "GHR-SID；只在最终语义叶子追加局部 D。"
        )
    )
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS)
    parser.add_argument(
        "--tiger-id-dir",
        type=Path,
        default=Path(
            "outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/"
            "tiger_ids/epoch_20"
        ),
    )
    parser.add_argument(
        "--structure-dir",
        type=Path,
        default=Path(
            "outputs/ghr_sid/EXP-20260814-02_beijing_geo_relation_structure_v1"
        ),
    )
    parser.add_argument("--relation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        if args.validate_only:
            result = validate_ghr_identifier_output(resolve(args.output_dir))
        else:
            if args.variant is None or args.relation_dir is None:
                raise GhrIdentifierError(
                    "正式构建必须同时提供 --variant 和 --relation-dir"
                )
            built = build_ghr_identifiers(
                project_root=PROJECT_ROOT,
                variant=args.variant,
                tiger_id_dir=resolve(args.tiger_id_dir),
                structure_dir=resolve(args.structure_dir),
                relation_dir=resolve(args.relation_dir),
                output_dir=resolve(args.output_dir),
                chunk_rows=args.chunk_rows,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            result = {
                "status": "completed",
                "variant": args.variant,
                "poi_count": built.metrics["poi_count"],
                "dedup_poi_count": built.metrics["dedup_poi_count"],
                "max_full_target_tokens_including_eos": built.metrics[
                    "max_full_target_tokens_including_eos"
                ],
                "output_dir": str(built.output_dir),
            }
    except (GhrIdentifierError, OSError, ValueError) as error:
        print(f"GHR identifier 构建失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

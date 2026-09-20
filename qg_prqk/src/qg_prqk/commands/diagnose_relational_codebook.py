"""Run the controlled relational-codebook attribution diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "在同一关系码本 sample 上隔离 graph、Query 质心先验、Top-k5 和 category；"
            "只产出诊断，不修改 canonical checkpoint。"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="校验冻结来源并输出固定诊断分支"
    )
    mode.add_argument(
        "--validate-only", action="store_true", help="独立复核已完成诊断"
    )
    args = parser.parse_args(argv)
    try:
        from qg_prqk.sid.relational_config import load_relational_codebook_config

        config = load_relational_codebook_config(args.config)
        if args.validate_only:
            from qg_prqk.sid.relational_attribution import validate_relational_attribution

            result = validate_relational_attribution(config)
        elif args.dry_run:
            from qg_prqk.sid.relational_attribution import (
                BRANCH_ORDER,
                diagnostic_branches,
                diagnostic_output_directory,
            )
            from qg_prqk.sid.relational_evaluation import validate_relational_evaluation
            from qg_prqk.sid.relational_quantizer import validate_relational_codebook

            canonical = validate_relational_codebook(config, gate="sample")
            baseline = validate_relational_evaluation(config, gate="sample")
            result = {
                "status": "p6_sample_attribution_dry_run_passed",
                "config_signature": config.signature(),
                "canonical_p6_manifest_sha256": canonical["manifest_sha256"],
                "canonical_evaluation_manifest_sha256": baseline[
                    "manifest_sha256"
                ],
                "branches": {
                    name: diagnostic_branches(config)[name]
                    for name in BRANCH_ORDER
                },
                "output_dir": str(diagnostic_output_directory(config)),
                "business_validation_read": False,
                "business_test_read": False,
                "geo_read": False,
                "output_written": False,
            }
        else:
            from qg_prqk.sid.relational_attribution import build_relational_attribution

            result = build_relational_attribution(config)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"关系码本 sample attribution 失败：{error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

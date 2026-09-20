"""Run the full-scale S2-only controlled diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "冻结正式关系码本 S1 endpoint，在 full active POI/Train-only Query 上只重跑 "
            "S2 受控分支；不修改 canonical checkpoint。"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="校验冻结来源并输出固定 S2 分支"
    )
    mode.add_argument(
        "--validate-only", action="store_true", help="独立复核已完成 S2 诊断"
    )
    args = parser.parse_args(argv)
    try:
        from qg_prqk.sid.relational_config import load_relational_codebook_config

        config = load_relational_codebook_config(args.config)
        if args.validate_only:
            from qg_prqk.sid.relational_s2_diagnostics import validate_relational_s2_diagnostic

            result = validate_relational_s2_diagnostic(config)
        elif args.dry_run:
            from qg_prqk.sid.relational_s2_diagnostics import (
                BRANCH_ORDER,
                diagnostic_branches,
                diagnostic_output_directory,
                validate_frozen_full_contracts,
            )

            canonical, baseline = validate_frozen_full_contracts(
                config, deep_p6=False
            )
            branches = diagnostic_branches(config)
            result = {
                "status": "p6_full_s2_diagnostic_dry_run_passed",
                "config_signature": config.signature(),
                "canonical_p6_manifest_sha256": canonical["manifest_sha256"],
                "canonical_evaluation_manifest_sha256": baseline[
                    "manifest_sha256"
                ],
                "branches": {name: branches[name] for name in BRANCH_ORDER},
                "frozen_s1_endpoint": True,
                "s1_optimization_rerun": False,
                "output_dir": str(diagnostic_output_directory(config)),
                "business_validation_read": False,
                "business_test_read": False,
                "geo_read": False,
                "output_written": False,
            }
        else:
            from qg_prqk.sid.relational_s2_diagnostics import build_relational_s2_diagnostic

            result = build_relational_s2_diagnostic(config)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"关系码本 full S2 diagnostic 失败：{error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build or validate the relational S1/S2 codebook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "在固定图闭包上构建类别锚定、Query–POI 双质心的 S1/S2；"
            "每个规模完成后停止，不自动进入其他规模或局部 S3。"
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="已确认的关系码本 YAML")
    parser.add_argument("--gate", choices=("sample", "medium", "full"), default="sample")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="读取冻结 header/图并验证闭包规模，不读取 GPU、不写输出",
    )
    mode.add_argument(
        "--validate-only", action="store_true", help="独立复核已完成目录的 manifest、哈希和 shape"
    )
    args = parser.parse_args(argv)
    try:
        from qg_prqk.sid.relational_config import load_relational_codebook_config

        config = load_relational_codebook_config(args.config)
        if args.validate_only:
            from qg_prqk.sid.relational_quantizer import validate_relational_codebook

            result = validate_relational_codebook(config, gate=args.gate)
        else:
            from qg_prqk.sid.relational_data import load_relational_codebook_inputs
            from qg_prqk.sid.relational_quantizer import next_status_for_gate

            inputs = load_relational_codebook_inputs(config, gate=args.gate)
            if args.dry_run:
                result = {
                    "status": "dry_run_passed",
                    "phase": f"P6-CAT-{args.gate.upper()}",
                    "config_signature": config.signature(),
                    "base_poi_rows": inputs.selection.base_poi_rows,
                    "closed_poi_rows": len(inputs.selection.selected_poi_rows),
                    "added_graph_target_rows": inputs.selection.added_graph_target_rows,
                    "query_rows": len(inputs.selection.selected_query_rows),
                    "s1_s2_edge_rows": len(inputs.selection.edges),
                    "codebook_sizes": list(config.codebook_sizes),
                    "hard_max_iter": config.prqk["max_iter"],
                    "business_validation_read": False,
                    "business_test_read": False,
                    "output_written": False,
                    "next_status": f"READY_TO_RUN_P6_{args.gate.upper()}",
                    "completion_stop": next_status_for_gate(args.gate),
                }
            else:
                from qg_prqk.sid.relational_quantizer import build_relational_codebook

                result = build_relational_codebook(config, inputs, gate=args.gate)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"关系码本构建失败：{error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

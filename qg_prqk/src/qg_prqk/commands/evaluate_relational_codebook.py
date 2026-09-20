"""Compare relational S1/S2 against the POI-only base."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="在相同 POI、Query 和图上比较基础码本与关系化 S1/S2。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate", choices=("sample", "medium", "full"), default="sample")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="独立复核已完成 Gate evaluation，不重新计算对照",
    )
    args = parser.parse_args(argv)
    try:
        from qg_prqk.sid.relational_config import load_relational_codebook_config
        from qg_prqk.sid.relational_evaluation import evaluate_relational_codebook, validate_relational_evaluation

        config = load_relational_codebook_config(args.config)
        result = (
            validate_relational_evaluation(config, gate=args.gate)
            if args.validate_only
            else evaluate_relational_codebook(config, gate=args.gate)
        )
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"关系码本评估失败：{error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

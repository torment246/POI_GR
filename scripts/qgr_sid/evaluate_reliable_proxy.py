#!/usr/bin/env python3
"""Run the single pre-registered QGR-SID M2-D reliable lexical proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.reliable_proxy import (  # noqa: E402
    QgrSidReliableProxyError,
    run_reliable_lexical_proxy,
    validate_reliable_proxy_output,
)


DEFAULT_ORDER_DIR = "data/beijing_order_clean_20260701_20260714_json"
DEFAULT_SFT_MANIFEST = "data/sft/beijing_order_main_v1/manifest.json"
DEFAULT_M2A_DIR = (
    "outputs/qgr_sid/EXP-20260813-07_tiger_lexical_relation_proxy_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "只运行一套冻结门槛：early 支持不少于 10,000 请求且关系值"
            "匹配率不低于 15%；未解决叶子保留 TIGER collision fallback。"
        )
    )
    parser.add_argument("--order-dir", type=Path, default=Path(DEFAULT_ORDER_DIR))
    parser.add_argument(
        "--sft-manifest", type=Path, default=Path(DEFAULT_SFT_MANIFEST)
    )
    parser.add_argument("--m2a-dir", type=Path, default=Path(DEFAULT_M2A_DIR))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-support-orders", type=int, default=10_000)
    parser.add_argument("--min-value-match-ratio", type=float, default=0.15)
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--prior-orders", type=float, default=20.0)
    parser.add_argument("--examples-per-kind", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _rooted(args.output_dir)
    try:
        if args.validate_only:
            payload = validate_reliable_proxy_output(output_dir)
        else:
            result = run_reliable_lexical_proxy(
                project_root=PROJECT_ROOT,
                order_dir=_rooted(args.order_dir),
                sft_manifest_path=_rooted(args.sft_manifest),
                m2a_dir=_rooted(args.m2a_dir),
                output_dir=output_dir,
                min_support_orders=args.min_support_orders,
                min_value_match_ratio=args.min_value_match_ratio,
                max_pairs=args.max_pairs,
                prior_orders=args.prior_orders,
                examples_per_kind=args.examples_per_kind,
                progress=lambda message: print(
                    message, file=sys.stderr, flush=True
                ),
            )
            ranking = result.metrics["holdout"]["ranking"]
            payload = {
                "status": result.metrics["status"],
                "selected_relation_types": result.metrics["gate"][
                    "selected_relation_types"
                ],
                "full_lexical_hr_at_1": ranking[
                    "full_query_guided_lexical"
                ]["hr_at_1"],
                "reliable_lexical_hr_at_1": ranking["reliable_lexical"][
                    "hr_at_1"
                ],
                "reliable_exact_path_given_present_ratio": result.metrics[
                    "holdout"
                ]["reliable_lexical"]["exact_path_given_present_ratio"],
                "decision": result.metrics["decision"],
                "output_dir": str(result.output_dir),
            }
    except (QgrSidReliableProxyError, OSError, ValueError) as error:
        print(f"QGR-SID M2-D 可靠词法代理失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

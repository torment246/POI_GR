#!/usr/bin/env python3
"""Run the frozen QGR-SID M2-A true-temporal lexical relation proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.proxy import (  # noqa: E402
    QgrSidProxyError,
    evaluate_lexical_relation_proxy,
    validate_proxy_output,
)


DEFAULT_POI_DIR = "data/beijing_poi_clean_20260715_json"
DEFAULT_ORDER_DIR = "data/beijing_order_clean_20260701_20260714_json"
DEFAULT_SFT_MANIFEST = "data/sft/beijing_order_main_v1/manifest.json"
DEFAULT_IDENTIFIER_DIR = (
    "outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/"
    "tiger_ids/epoch_20"
)
DEFAULT_M1_MANIFEST = (
    "outputs/qgr_sid/EXP-20260813-05_tiger_numeric_relation_audit_v1/"
    "manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用原始 Train 订单的真实时间前 90% 编译静态/Query 引导纯数字"
            "变长关系树，只在后 10% 的已知 TIGER gold bucket 上评测代理排序。"
        )
    )
    parser.add_argument("--poi-dir", type=Path, default=Path(DEFAULT_POI_DIR))
    parser.add_argument("--order-dir", type=Path, default=Path(DEFAULT_ORDER_DIR))
    parser.add_argument(
        "--sft-manifest", type=Path, default=Path(DEFAULT_SFT_MANIFEST)
    )
    parser.add_argument(
        "--identifier-dir", type=Path, default=Path(DEFAULT_IDENTIFIER_DIR)
    )
    parser.add_argument(
        "--m1-manifest", type=Path, default=Path(DEFAULT_M1_MANIFEST)
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--early-ratio", type=float, default=0.9)
    parser.add_argument("--value-max", type=int, default=1055)
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--prior-orders", type=float, default=20.0)
    parser.add_argument("--examples-per-kind", type=int, default=20)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验 --output-dir 下正式产物的字节数、shape、dtype 和 SHA256。",
    )
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _rooted(args.output_dir)
    try:
        if args.validate_only:
            payload = validate_proxy_output(output_dir)
        else:
            result = evaluate_lexical_relation_proxy(
                project_root=PROJECT_ROOT,
                poi_dir=_rooted(args.poi_dir),
                order_dir=_rooted(args.order_dir),
                sft_manifest_path=_rooted(args.sft_manifest),
                identifier_dir=_rooted(args.identifier_dir),
                m1_manifest_path=_rooted(args.m1_manifest),
                output_dir=output_dir,
                early_ratio=args.early_ratio,
                value_max=args.value_max,
                max_pairs=args.max_pairs,
                batch_rows=args.batch_rows,
                prior_orders=args.prior_orders,
                examples_per_kind=args.examples_per_kind,
                progress=lambda message: print(
                    message, file=sys.stderr, flush=True
                ),
            )
            holdout = result.metrics["holdout"]["ranking"]
            payload = {
                "status": result.metrics["status"],
                "collision_holdout_order_count": result.metrics["holdout"][
                    "collision_holdout_order_count"
                ],
                "popularity_hr_at_1": holdout["popularity"]["hr_at_1"],
                "static_hr_at_1": holdout["static"]["hr_at_1"],
                "query_guided_hr_at_1": holdout["query_guided"]["hr_at_1"],
                "decision": result.metrics["decision"],
                "output_dir": str(result.output_dir),
            }
    except (QgrSidProxyError, OSError, ValueError) as error:
        print(f"QGR-SID M2-A 代理实验失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

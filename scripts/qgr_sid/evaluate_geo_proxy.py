#!/usr/bin/env python3
"""Run the frozen QGR-SID M2-B GEO/GID proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.geo_proxy import (  # noqa: E402
    QgrSidGeoProxyError,
    evaluate_geo_relation_proxy,
    validate_geo_proxy_output,
)


DEFAULT_ORDER_DIR = "data/beijing_order_clean_20260701_20260714_json"
DEFAULT_SFT_MANIFEST = "data/sft/beijing_order_main_v1/manifest.json"
DEFAULT_IDENTIFIER_DIR = (
    "outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/"
    "tiger_ids/epoch_20"
)
DEFAULT_M2A_DIR = (
    "outputs/qgr_sid/EXP-20260813-07_tiger_lexical_relation_proxy_v1"
)
DEFAULT_GID_DIR = "outputs/pid/BJ-RQVAE-1024x3-e20-G6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "复用 M2-A 的真实 Train 90/10 切分，构建静态 POI 最短 GEO "
            "数字关系，并用已有请求 GID 对比 GEO-only、纯词法和组合树。"
        )
    )
    parser.add_argument("--order-dir", type=Path, default=Path(DEFAULT_ORDER_DIR))
    parser.add_argument(
        "--sft-manifest", type=Path, default=Path(DEFAULT_SFT_MANIFEST)
    )
    parser.add_argument(
        "--identifier-dir", type=Path, default=Path(DEFAULT_IDENTIFIER_DIR)
    )
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
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--prior-orders", type=float, default=20.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _rooted(args.output_dir)
    try:
        if args.validate_only:
            payload = validate_geo_proxy_output(output_dir)
        else:
            result = evaluate_geo_relation_proxy(
                project_root=PROJECT_ROOT,
                order_dir=_rooted(args.order_dir),
                sft_manifest_path=_rooted(args.sft_manifest),
                identifier_dir=_rooted(args.identifier_dir),
                m2a_dir=_rooted(args.m2a_dir),
                gid_codes_path=_rooted(args.gid_codes),
                gid_manifest_path=_rooted(args.gid_manifest),
                output_dir=output_dir,
                max_pairs=args.max_pairs,
                batch_rows=args.batch_rows,
                prior_orders=args.prior_orders,
                progress=lambda message: print(
                    message, file=sys.stderr, flush=True
                ),
            )
            ranking = result.metrics["holdout"]["ranking"]
            payload = {
                "status": result.metrics["status"],
                "collision_holdout_order_count": result.metrics["holdout"][
                    "collision_holdout_order_count"
                ],
                "query_guided_lexical_hr_at_1": ranking[
                    "query_guided_lexical"
                ]["hr_at_1"],
                "gid_guided_geo_only_hr_at_1": ranking[
                    "gid_guided_geo_only"
                ]["hr_at_1"],
                "query_gid_lexical_geo_hr_at_1": ranking[
                    "query_gid_lexical_geo"
                ]["hr_at_1"],
                "decision": result.metrics["decision"],
                "output_dir": str(result.output_dir),
            }
    except (QgrSidGeoProxyError, OSError, ValueError) as error:
        print(f"QGR-SID M2-B GEO/GID 代理失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

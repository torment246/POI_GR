#!/usr/bin/env python3
"""Run the frozen QGR-SID M2-C early-only Query residual proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.qgr_sid.embedding_proxy import (  # noqa: E402
    QgrSidEmbeddingProxyError,
    run_query_embedding_proxy,
    validate_embedding_proxy_output,
)


DEFAULT_ORDER_DIR = "data/beijing_order_clean_20260701_20260714_json"
DEFAULT_SFT_MANIFEST = "data/sft/beijing_order_main_v1/manifest.json"
DEFAULT_M2A_DIR = (
    "outputs/qgr_sid/EXP-20260813-07_tiger_lexical_relation_proxy_v1"
)
DEFAULT_STATS_DIR = (
    "outputs/embeddings/query_augmented_bge_m3/train_query_stats_v1"
)
DEFAULT_EMBEDDING_DIR = (
    "outputs/embeddings/query_augmented_bge_m3/train_query_embeddings_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 full Train Query 统计中精确扣除冻结 holdout，构建 early-only "
            "POI Query center，并评测桶内 Query residual 与词法优先融合。"
        )
    )
    parser.add_argument("--order-dir", type=Path, default=Path(DEFAULT_ORDER_DIR))
    parser.add_argument(
        "--sft-manifest", type=Path, default=Path(DEFAULT_SFT_MANIFEST)
    )
    parser.add_argument("--m2a-dir", type=Path, default=Path(DEFAULT_M2A_DIR))
    parser.add_argument(
        "--query-stats-dir", type=Path, default=Path(DEFAULT_STATS_DIR)
    )
    parser.add_argument(
        "--query-embeddings",
        type=Path,
        default=Path(DEFAULT_EMBEDDING_DIR) / "embeddings.npy",
    )
    parser.add_argument(
        "--query-embeddings-manifest",
        type=Path,
        default=Path(DEFAULT_EMBEDDING_DIR) / "manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _rooted(args.output_dir)
    try:
        if args.validate_only:
            payload = validate_embedding_proxy_output(output_dir)
        else:
            result = run_query_embedding_proxy(
                project_root=PROJECT_ROOT,
                order_dir=_rooted(args.order_dir),
                sft_manifest_path=_rooted(args.sft_manifest),
                m2a_dir=_rooted(args.m2a_dir),
                query_stats_dir=_rooted(args.query_stats_dir),
                query_embeddings_path=_rooted(args.query_embeddings),
                query_embeddings_manifest_path=_rooted(
                    args.query_embeddings_manifest
                ),
                output_dir=output_dir,
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
                "query_residual_hr_at_1": ranking["query_residual"][
                    "hr_at_1"
                ],
                "lexical_first_query_residual_hr_at_1": ranking[
                    "lexical_first_query_residual"
                ]["hr_at_1"],
                "decision": result.metrics["decision"],
                "output_dir": str(result.output_dir),
            }
    except (QgrSidEmbeddingProxyError, OSError, ValueError) as error:
        print(f"QGR-SID M2-C Query embedding 代理失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

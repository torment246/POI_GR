"""Select and retrain the exact-query adapter on full Train data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qg_prqk.adapters.retrieval import QueryAdapterRetrievalError
from qg_prqk.adapters.data import QueryAdapterDataError, select_d3_queries
from qg_prqk.adapters.selection import AdapterSelectionError, run_adapter_selection, validate_adapter_selection
from qg_prqk.adapters.selection_config import AdapterSelectionConfigError, load_adapter_selection_config
from qg_prqk.adapters.gate import QueryAdapterGateError, _validate_frozen_inputs, validate_query_adapter_gate
from qg_prqk.adapters.cache import QueryEmbeddingCacheError, build_d3_query_cache
from qg_prqk.data.query_embeddings import QueryEmbeddingError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "运行全量 Query Adapter 选择：固定 20,000 条 Train-only internal "
            "holdout，完成 FULL-SELECT 三轮冻结 active POI 闭集精确评测，并按最佳 epoch "
            "从相同初始状态训练 FULL-FINAL。"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--prepare-query-cache-only",
        action="store_true",
        help="仅构建/复核可分块恢复的全量 D3 BGE cache，不开始 Adapter 训练。",
    )
    parser.add_argument(
        "--recover-query-prefix-from",
        type=Path,
        help="只读旧中断目录，恢复日志确认且通过逐行校验的 Query 前缀；须配合 --prepare-query-cache-only。",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只复核已有产物、数据隔离、checkpoint 角色、shape 与 SHA256。",
    )
    parser.add_argument(
        "--resume-incomplete",
        action="store_true",
        help=(
            "仅恢复经过校验的未完成全量 Adapter 输出；复用已冻结的 Query/ANN "
            "预计算，不覆盖任何已有产物。"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_adapter_selection_config(args.config)
        if args.recover_query_prefix_from and not args.prepare_query_cache_only:
            raise AdapterSelectionError("恢复旧编码前缀须配合 --prepare-query-cache-only")
        if args.prepare_query_cache_only:
            if args.validate_only or args.resume_incomplete:
                raise AdapterSelectionError("prepare-query-cache-only 不能与其他阶段模式混用")
            validate_query_adapter_gate(config.gate, gate="medium")
            _validate_frozen_inputs(config.base)
            queries, total = select_d3_queries(
                config.base.query_depth_dir / "query_category_stats.parquet",
                limit=config.d3_rows,
                seed=config.holdout_seed,
            )
            if total != config.d3_rows or len(queries) != total:
                raise AdapterSelectionError("D3 Query 总数与冻结配置不匹配")
            path, manifest = build_d3_query_cache(
                config,
                queries,
                source_dir=args.recover_query_prefix_from,
            )
            print(
                json.dumps(
                    {
                        "status": "query_cache_completed",
                        "rows": manifest["rows"],
                        "embeddings": str(path),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.validate_only and args.resume_incomplete:
            raise AdapterSelectionError("--validate-only 与 --resume-incomplete 不能同时使用")
        manifest = (
            validate_adapter_selection(config)
            if args.validate_only
            else run_adapter_selection(
                config,
                resume_incomplete=args.resume_incomplete,
            )
        )
    except (
        QueryAdapterRetrievalError,
        QueryAdapterDataError,
        AdapterSelectionConfigError,
        AdapterSelectionError,
        QueryAdapterGateError,
        QueryEmbeddingCacheError,
        QueryEmbeddingError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"QG-PRQK 全量 Query Adapter 失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "validated" if args.validate_only else "completed",
                "phase": "P3A-FULL",
                "select_train_rows": manifest["partition"]["select_train_rows"],
                "internal_holdout_rows": manifest["partition"]["internal_holdout_rows"],
                "best_epoch": manifest["select"]["best_epoch"],
                "final_checkpoint": str(
                    config.output_dir / "final/query_adapter_exact_final.pt"
                ),
                "next_status": "HOLD_FOR_P3A_FULL_REVIEW",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

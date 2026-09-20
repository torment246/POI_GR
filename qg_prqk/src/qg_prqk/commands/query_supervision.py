"""Build category-aware query supervision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from qg_prqk.data.query_supervision_config import CategoryConfigError, load_category_config
from qg_prqk.data.query_supervision import (
    CategoryDepthBuildResult,
    CategoryDepthError,
    build_category_depth,
    preview_category_depth,
    run_category_depth_gate,
    set_progress_enabled,
    validate_category_depth,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "构建类别感知的 Query 粒度、层掩码和归一 Query–POI 边。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="QG v2.1-CAT canonical YAML 配置。",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验冻结输入并预览有限 Query，不写产物。",
    )
    action.add_argument(
        "--gate",
        choices=("sample", "medium"),
        help="运行并记录 full 前必须通过的 1k/50k Query Gate。",
    )
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="独立复核已有 canonical 全量产物，不重建。",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        help="dry-run 的 Query 数；默认使用 sample gate 数。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="全量产物已存在时执行完整校验并复用。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭进度条，适合测试和日志环境。",
    )
    return parser


def _result_payload(result: CategoryDepthBuildResult) -> dict[str, Any]:
    return {
        "status": "completed",
        "output_dir": str(result.output_dir),
        "manifest": str(result.manifest_path),
        "category_rows": result.category_rows,
        "unique_queries": result.unique_queries,
        "layer_edges": result.layer_edges,
        "depth_counts": dict(result.depth_counts),
        "reused": result.reused,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.query_limit is not None and not args.dry_run:
        parser.error("--query-limit 只能与 --dry-run 一起使用")
    if args.query_limit is not None and args.query_limit <= 0:
        parser.error("--query-limit 必须大于 0")
    if args.resume and (args.dry_run or args.gate or args.validate_only):
        parser.error("--resume 只用于默认 full 动作")
    try:
        config = load_category_config(args.config)
        set_progress_enabled(not args.no_progress)
        if args.dry_run:
            payload: Any = preview_category_depth(
                config,
                query_limit=args.query_limit or config.runtime.sample_query_limit,
            )
        elif args.gate:
            payload = run_category_depth_gate(config, level=args.gate)
        elif args.validate_only:
            payload = _result_payload(
                validate_category_depth(config.query_depth_output_dir, config)
            )
        else:
            payload = _result_payload(
                build_category_depth(config, resume=args.resume)
            )
    except (CategoryConfigError, CategoryDepthError, OSError, ValueError) as error:
        print(f"QG-PRQK Query 监督构建失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

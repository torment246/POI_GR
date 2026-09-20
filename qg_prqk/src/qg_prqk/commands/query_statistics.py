"""Build or validate Train-only query statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from qg_prqk.config import QGPRQKConfig, QGPRQKConfigError, load_config
from qg_prqk.data.contracts import QGPRQKDataContractError, validate_train_record
from qg_prqk.data.query_normalization import normalize_query
from qg_prqk.data.query_statistics import (
    QueryStatsError,
    build_query_stats,
    false_negative_targets,
    summarize_query,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="构建 QG-PRQK Train-only 归一化 Query–POI 统计。"
    )
    parser.add_argument("--config", type=Path, required=True, help="QG v1.1 YAML 配置。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="覆盖实验根目录；必须位于 qg_prqk/outputs/ 下。",
    )
    parser.add_argument("--seed", type=int, help="覆盖随机种子。")
    parser.add_argument(
        "--limit",
        "--sample-size",
        dest="limit",
        type=int,
        help="只读取 Train 前 N 行；仅用于 smoke，不是正式全量产物。",
    )
    parser.add_argument("--resume", action="store_true", help="校验并复用完整已有产物。")
    parser.add_argument("--overwrite", action="store_true", help="删除并重建同一 Query 统计输出。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读有限 Train 样例并预览统计，不写入任何产物。",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="关闭进度条，适合测试和日志环境。"
    )
    return parser


def _load_json_line(line: str, source: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise QueryStatsError(f"{source} JSON 非法") from error
    if not isinstance(value, dict):
        raise QueryStatsError(f"{source} 必须是 JSON object")
    return value


def preview(config: QGPRQKConfig, limit: int) -> dict[str, Any]:
    """Read a bounded Train prefix and exercise the production formulas."""

    if limit <= 0:
        raise QueryStatsError("dry-run limit 必须大于 0")
    train_path = config.paths.sft_data_dir / "train.jsonl"
    pairs: dict[str, Counter[str]] = defaultdict(Counter)
    raws: dict[str, Counter[str]] = defaultdict(Counter)
    rows = 0
    with train_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if rows >= limit:
                break
            source = f"{train_path}:{line_number}"
            record = _load_json_line(line, source)
            raw_query, poi_id = validate_train_record(
                record, config.data_contracts, source
            )
            normalized_query = normalize_query(raw_query)
            pairs[normalized_query][poi_id] += 1
            raws[normalized_query][raw_query] += 1
            rows += 1
    if rows != limit:
        raise QueryStatsError(f"Train 样例不足：期望 {limit}，实际 {rows}")
    summaries = [
        summarize_query(query, pairs[query], raws[query], config.query_stats)
        for query in sorted(pairs)
    ]
    return {
        "status": "dry_run_passed",
        "writes": False,
        "train_rows": rows,
        "unique_normalized_queries": len(summaries),
        "retained_queries": sum(item.is_high_confidence for item in summaries),
        "false_negative_pairs": sum(
            len(false_negative_targets(pairs[query], config.query_stats))
            for query in pairs
        ),
        "normalization_examples": [
            {
                "raw": summary.representative_raw_query,
                "normalized": summary.normalized_query,
            }
            for summary in summaries[:5]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须大于 0")
    try:
        config = load_config(
            args.config,
            output_dir=args.output_dir,
            seed=args.seed,
            resume=True if args.resume else None,
            overwrite=True if args.overwrite else None,
        )
        if args.no_progress:
            config = replace(config, runtime=replace(config.runtime, show_progress=False))
        if args.dry_run:
            result: Any = preview(config, args.limit or config.runtime.sample_limit)
        else:
            result = build_query_stats(
                config,
                output_dir=config.paths.output_dir / "query_stats",
                limit=args.limit,
            )
            result = {
                "status": "completed",
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "source_rows": result.source_rows,
                "unique_queries": result.unique_queries,
                "unique_query_poi_pairs": result.unique_query_poi_pairs,
                "retained_queries": result.retained_queries,
                "false_negative_pairs": result.false_negative_pairs,
                "reused": result.reused,
            }
    except (
        QGPRQKConfigError,
        QGPRQKDataContractError,
        QueryStatsError,
        OSError,
        ValueError,
    ) as error:
        print(f"QG-PRQK Query 统计失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

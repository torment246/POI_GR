"""Build or validate the query graph and depth-specific views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="构建 sample/medium/full Query 层级图与分层 view；不训练、不聚类。"
    )
    parser.add_argument("--config", type=Path, required=True, help="512×3 下游 YAML")
    parser.add_argument(
        "--gate",
        choices=("sample", "medium", "full"),
        default="sample",
        help="sample 最多 1000，medium 最多 50000，full 固定 342879",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="按 Query 监督的 query_id 顺序取前 N 个 D1/D2/D3 节点，默认取 gate 上限",
    )
    parser.add_argument(
        "--reuse-sample", type=Path, help="medium 必须复用的已完成 sample manifest.json"
    )
    parser.add_argument(
        "--reuse-sample-sha256",
        help="冻结 sample manifest 的 SHA256，不从现存文件自动推断",
    )
    parser.add_argument(
        "--reuse-medium", type=Path, help="full 必须复用的已完成 medium manifest.json"
    )
    parser.add_argument(
        "--reuse-medium-sha256",
        help="冻结 medium manifest 的 SHA256，不从现存文件自动推断",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="仅核验数据契约，不写文件、不加载 BGE/Adapter",
    )
    mode.add_argument(
        "--resume", action="store_true", help="严格验证并复用已提交块或已完成产物"
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="只读独立核验图、来源、全部缓存块和分层 view 路由",
    )
    args = parser.parse_args(argv)
    limit = (
        args.limit
        if args.limit is not None
        else {"sample": 1000, "medium": 50000, "full": 342879}[args.gate]
    )
    if (args.reuse_sample is None) != (args.reuse_sample_sha256 is None):
        parser.error("--reuse-sample 与 --reuse-sample-sha256 必须同时提供")
    if (args.reuse_medium is None) != (args.reuse_medium_sha256 is None):
        parser.error("--reuse-medium 与 --reuse-medium-sha256 必须同时提供")
    if (
        args.gate == "medium"
        and not args.validate_only
        and not args.dry_run
        and args.reuse_sample is None
    ):
        parser.error("medium 构建/恢复必须显式指定 sample manifest 及 SHA256")
    if (
        args.gate == "full"
        and not args.validate_only
        and not args.dry_run
        and args.reuse_medium is None
    ):
        parser.error("full 构建/恢复必须显式指定 medium manifest 及 SHA256")
    try:
        from qg_prqk.sid.pipeline_config import load_downstream_config
        from qg_prqk.data.query_graph_data import graph_metrics, prepare_query_graph_inputs

        config = load_downstream_config(args.config)
        print(
            json.dumps(
                {"stage": "p4_preparing_inputs", "gate": args.gate, "limit": limit}
            ),
            flush=True,
        )
        inputs = prepare_query_graph_inputs(config, limit=limit, gate=args.gate)
        if args.dry_run:
            result = {
                "status": "dry_run_passed",
                "metrics": graph_metrics(inputs, bounded_memory=args.gate == "full"),
                "data_files_declared": len(inputs.source_files),
                "output_written": False,
                "d3_vector_chunk_contents_checked": False,
                "business_validation_read": False,
                "business_test_read": False,
            }
        else:
            from qg_prqk.data.query_graph import build_query_graph, validate_query_graph

            if args.validate_only:
                result = validate_query_graph(config, inputs, gate=args.gate)
            else:
                result = build_query_graph(
                    config,
                    inputs,
                    resume=args.resume,
                    gate=args.gate,
                    reuse_sample=args.reuse_sample,
                    reuse_sample_sha256=args.reuse_sample_sha256,
                    reuse_medium=args.reuse_medium,
                    reuse_medium_sha256=args.reuse_medium_sha256,
                )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Query 图构建失败：{error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

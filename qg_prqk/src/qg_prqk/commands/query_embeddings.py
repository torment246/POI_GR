"""Build or validate normalized query embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from qg_prqk.config import QGPRQKConfigError, load_config
from qg_prqk.data.query_embeddings import QueryEmbeddingError, build_query_embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="构建 QG-PRQK 归一化 Query BGE cache。"
    )
    parser.add_argument("--config", type=Path, required=True, help="QG v1.1 YAML 配置。")
    parser.add_argument(
        "--query-stats-dir", type=Path, required=True, help="Query 统计产物目录。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="实验根目录；实际 cache 写入其 query_embeddings/ 子目录。",
    )
    parser.add_argument("--seed", type=int, help="覆盖随机种子。")
    parser.add_argument("--limit", "--sample-size", dest="limit", type=int)
    parser.add_argument("--device", help="覆盖 Query encoder device。")
    parser.add_argument("--batch-size", type=int, help="覆盖编码 batch size。")
    parser.add_argument("--encode-buffer-size", type=int, help="覆盖编码缓冲行数。")
    parser.add_argument("--resume", action="store_true", help="复用完成产物或断点。")
    parser.add_argument("--overwrite", action="store_true", help="重建同一 Query cache。")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条。")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(
            args.config,
            output_dir=args.output_dir,
            seed=args.seed,
            resume=True if args.resume else None,
            overwrite=True if args.overwrite else None,
        )
        embedding = replace(
            config.query_embedding,
            device=args.device or config.query_embedding.device,
            batch_size=args.batch_size or config.query_embedding.batch_size,
            encode_buffer_size=(
                args.encode_buffer_size or config.query_embedding.encode_buffer_size
            ),
        )
        if embedding.batch_size <= 0 or embedding.encode_buffer_size < embedding.batch_size:
            raise QueryEmbeddingError("覆盖后的 batch/buffer 配置非法")
        config = replace(
            config,
            query_embedding=embedding,
            runtime=replace(
                config.runtime,
                show_progress=False if args.no_progress else config.runtime.show_progress,
            ),
        )
        result = build_query_embeddings(
            config,
            query_stats_dir=args.query_stats_dir,
            output_dir=config.paths.output_dir / "query_embeddings",
            limit=args.limit,
        )
    except (QGPRQKConfigError, QueryEmbeddingError, OSError, ValueError) as error:
        print(f"QG-PRQK Query cache failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "rows": result.rows,
                "embedding_dim": result.embedding_dim,
                "reused": result.reused,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

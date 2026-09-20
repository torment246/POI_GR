#!/usr/bin/env python3
"""Build an exact active-catalog subset of a completed embedding artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.subset import EmbeddingSubsetError, build_embedding_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按目标 poi_ids.jsonl 的有序子序列，从已完成的全库 Embedding "
            "逐向量精确复制 active 子集。"
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-poi-ids", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--copy-chunk-rows", type=int, default=8192)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        manifest = build_embedding_subset(
            source_dir=resolve(args.source_dir),
            target_poi_ids=resolve(args.target_poi_ids),
            target_manifest=resolve(args.target_manifest),
            output_dir=resolve(args.output_dir),
            expected_rows=args.expected_rows,
            project_root=PROJECT_ROOT,
            copy_chunk_rows=args.copy_chunk_rows,
        )
    except (EmbeddingSubsetError, OSError, ValueError) as error:
        print(f"Embedding 子集构建失败：{error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "shape": manifest["output"]["shape"],
                "dtype": manifest["output"]["dtype"],
                "embeddings_sha256": manifest["output"]["embeddings_sha256"],
                "poi_ids_sha256": manifest["output"]["poi_ids_sha256"],
                "output_dir": manifest["output"]["dir"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

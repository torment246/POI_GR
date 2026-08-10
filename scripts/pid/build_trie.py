#!/usr/bin/env python3
"""Build the compact Final PID Trie used by constrained generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.pid.trie import PidTrieError, build_pid_trie  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 PID-002 的唯一 Final PID，构建不含 Python 节点对象的紧凑 "
            "CSR Trie；单例路径在 S3 后终止，碰撞路径只允许真实 D Token。"
        )
    )
    parser.add_argument(
        "--pid-mapping",
        type=Path,
        required=True,
        help="PID-002 的 poi_pid_mapping.parquet。",
    )
    parser.add_argument(
        "--pid-manifest",
        type=Path,
        required=True,
        help="PID-002 的 final_pid_manifest.json。",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="已扩展 3,620 个 POI Token 的 tokenizer 目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Trie 输出目录。",
    )
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="仅用于技术调试；跳过大型输入文件 SHA256 复核。",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        result = build_pid_trie(
            _resolve(args.pid_mapping),
            _resolve(args.pid_manifest),
            _resolve(args.tokenizer),
            _resolve(args.output_dir),
            verify_hashes=not args.skip_input_hashes,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (PidTrieError, OSError, ValueError) as error:
        print(f"Final PID Trie 构建失败：{error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "completed",
                "node_count": result.manifest["node_count"],
                "edge_count": result.manifest["edge_count"],
                "leaf_count": result.manifest["leaf_count"],
                "file_size_bytes": result.manifest["file_size_bytes"],
                "load_memory_bytes": result.manifest["load_memory_bytes"],
                "build_seconds": result.manifest["build_seconds"],
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

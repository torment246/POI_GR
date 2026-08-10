#!/usr/bin/env python3
"""Encode the frozen unique Train Query catalog with BGE-M3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.query_encoding import (  # noqa: E402
    apply_query_encoding_overrides,
    load_train_query_encoding_config,
    run_train_query_encoding,
    validate_train_query_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按冻结的连续 query_id 顺序编码唯一 Train Query，"
            "以可恢复 float16 NPY 保存 BGE-M3 向量。"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/embedding/embedding_train_query_bge_m3_v1.yaml"),
        help="Train Query 编码配置；相对路径相对于仓库根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="覆盖输出目录；运行 --max-rows smoke 时必须显式提供。",
    )
    parser.add_argument("--device", help="覆盖设备，例如 cuda、cuda:0 或 cpu。")
    parser.add_argument("--batch-size", type=int, help="覆盖模型 batch size。")
    parser.add_argument(
        "--encode-buffer-size",
        type=int,
        help="覆盖每个可恢复写入缓冲区的 Query 数。",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="只编码前 N 个唯一 Query，用于独立 smoke 输出。",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只核验已完成产物、输入指纹和 embeddings.npy SHA256。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭进度条。",
    )
    return parser.parse_args()


def _resolve(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    assert config_path is not None
    config = load_train_query_encoding_config(config_path, PROJECT_ROOT)
    output_override = _resolve(args.output_dir)
    if args.max_rows is not None and output_override is None:
        raise SystemExit("使用 --max-rows 时必须同时提供独立 --output-dir")
    config = apply_query_encoding_overrides(
        config,
        output_dir=output_override,
        device=args.device,
        batch_size=args.batch_size,
        encode_buffer_size=args.encode_buffer_size,
    )
    if args.validate_only:
        if args.max_rows is not None:
            raise SystemExit("--validate-only 不能与 --max-rows 同时使用")
        result = validate_train_query_embeddings(
            config.output.dir,
            expected_stats_manifest_sha256=(config.expected_stats_manifest_sha256),
        )
    else:
        result = run_train_query_encoding(
            config,
            project_root=PROJECT_ROOT,
            max_rows=args.max_rows,
            show_progress=not args.no_progress,
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "embeddings": str(result.embeddings_path),
                "rows": result.rows,
                "embedding_dim": result.embedding_dim,
                "reused": result.reused,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

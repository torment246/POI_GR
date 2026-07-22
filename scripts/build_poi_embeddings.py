#!/usr/bin/env python3
"""Build POI text embeddings from a YAML job configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding import apply_overrides, load_job_config, run_embedding_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="流式读取 JSONL 的 text 字段，构建与 poi_id 严格对齐的 NPY 向量。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/embedding_qwen3_0.6b.yaml"),
        help="YAML 配置路径；相对路径相对于仓库根目录。",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="只处理前 N 行，用于 smoke test；默认处理全部数据。",
    )
    parser.add_argument("--model-path", type=Path, help="覆盖配置中的本地模型目录。")
    parser.add_argument("--output-dir", type=Path, help="覆盖配置中的输出目录。")
    parser.add_argument("--device", help="覆盖设备，例如 cuda、cuda:0 或 cpu。")
    parser.add_argument("--batch-size", type=int, help="覆盖推理 batch size。")
    parser.add_argument(
        "--encode-buffer-size",
        type=int,
        help="覆盖单次交给编码器的文本数量；编码器在缓冲区内按长度组批。",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否允许从匹配的 progress.json 继续。",
    )
    return parser.parse_args()


def resolve_from_root(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    config_path = resolve_from_root(args.config)
    assert config_path is not None
    config = load_job_config(config_path, PROJECT_ROOT)
    config = apply_overrides(
        config,
        model_path=resolve_from_root(args.model_path),
        output_dir=resolve_from_root(args.output_dir),
        device=args.device,
        batch_size=args.batch_size,
        encode_buffer_size=args.encode_buffer_size,
        resume=args.resume,
    )
    manifest = run_embedding_job(
        config,
        project_root=PROJECT_ROOT,
        max_rows=args.max_rows,
    )
    summary = {
        "status": manifest["status"],
        "rows": manifest["input"]["total_rows"],
        "shape": manifest["output"]["shape"],
        "dtype": manifest["output"]["dtype"],
        "output_dir": manifest["output"]["dir"],
        "rows_per_second": manifest["metrics"]["rows_per_second"],
    }
    if "cuda_peak_memory_allocated_gib" in manifest["metrics"]:
        summary["cuda_peak_memory_allocated_gib"] = manifest["metrics"][
            "cuda_peak_memory_allocated_gib"
        ]
        summary["cuda_peak_memory_reserved_gib"] = manifest["metrics"][
            "cuda_peak_memory_reserved_gib"
        ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

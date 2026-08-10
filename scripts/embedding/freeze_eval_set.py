#!/usr/bin/env python3
"""Freeze the shared Validation set for embedding retrieval experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.eval_set import freeze_embedding_eval_set  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="冻结无训练泄漏的 10,000 条 Embedding Validation 评测集。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/embedding/embedding_eval_set_v1.yaml"),
        help="冻结协议配置；相对路径相对于仓库根目录。",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"配置不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("配置根节点必须是 mapping")
    return value


def main() -> int:
    args = parse_args()
    config = load_config(resolve(args.config))
    source = config.get("source")
    output = config.get("output")
    expected = config.get("expected")
    if not isinstance(source, dict) or not isinstance(output, dict):
        raise ValueError("配置必须包含 source/output mapping")
    if not isinstance(expected, dict):
        raise ValueError("配置必须包含 expected mapping")

    frozen = freeze_embedding_eval_set(
        resolve(source["sft_dir"]),
        resolve(source["reference_subset"]),
        resolve(output["dir"]),
        expected_rows=int(expected["rows"]),
        expected_train_rows=int(expected["train_rows"]),
        expected_valid_rows=int(expected["valid_rows"]),
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "data": str(frozen.data_path),
                "manifest": str(frozen.manifest_path),
                "rows": frozen.row_count,
                "sha256": frozen.sha256,
                "leakage_check": frozen.manifest["leakage_check"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

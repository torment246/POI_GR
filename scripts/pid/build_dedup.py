#!/usr/bin/env python3
"""Build deterministic unique final PIDs from PID-001 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.pid.dedup import DedupPidError, build_dedup_pid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 PID-001 的九层 base PID，在每个碰撞桶内按 poi_id "
            "字典序分配局部连续 Dedup Code，并保存唯一 POI-PID 映射。"
        )
    )
    parser.add_argument(
        "--pid-manifest",
        type=Path,
        required=True,
        help="PID-001 输出的 pid_manifest.json。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录，只保留 PID-002 规定的五个文件。",
    )
    parser.add_argument(
        "--dedup-capacity",
        type=int,
        default=512,
        help="预留 Dedup Token 容量，默认 512。",
    )
    return parser.parse_args()


def _from_project_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        result = build_dedup_pid(
            _from_project_root(args.pid_manifest),
            _from_project_root(args.output_dir),
            dedup_capacity=args.dedup_capacity,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (DedupPidError, OSError, ValueError) as error:
        print(f"Dedup PID 构建失败：{error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "completed",
                "poi_count": result.metrics["poi_count"],
                "singleton_poi_count": result.metrics["singleton_poi_count"],
                "dedup_poi_count": result.metrics["dedup_poi_count"],
                "max_dedup_code": result.metrics["max_dedup_code"],
                "final_pid_unique_ratio": result.metrics[
                    "final_pid_unique_ratio"
                ],
                "mapping_sha256": result.mapping_sha256,
                "output_dir": str(_from_project_root(args.output_dir).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

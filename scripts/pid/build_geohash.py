#!/usr/bin/env python3
"""Build and evaluate a Geohash6-prefixed Semantic PID."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.pid.geohash import (
    GeohashPidError,
    build_geohash_pid,
    resolve_poi_data_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "为三层 Semantic SID 构建标准 Geohash6 GID 与 "
            "[G1..G6,S1..S3] PID，并完成全量碰撞评估。"
        )
    )
    parser.add_argument(
        "--sid-manifest",
        type=Path,
        required=True,
        help="三层码本、epoch 20 的 sid_manifest.json。",
    )
    parser.add_argument(
        "--geohash-length",
        type=int,
        default=6,
        help="Geohash 长度；PID-001 固定为 6。",
    )
    parser.add_argument(
        "--order",
        choices=("gid_sid",),
        default="gid_sid",
        help="PID Token 顺序；PID-001 固定为 gid_sid。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录，只保留 PID-001 规定的六个文件。",
    )
    parser.add_argument(
        "--poi-data",
        type=Path,
        help=(
            "可选 POI JSONL 文件或 part-* 目录；缺省时依次从现有 SID "
            "metrics 和 RQ-VAE resolved_config 解析。"
        ),
    )
    parser.add_argument("--sid-codebook-size", type=int, choices=(512, 1024), default=1024,
                        help="每层 SID 容量；旧全库默认 1024，active 对照可指定 512。")
    return parser.parse_args()


def _from_project_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    sid_manifest_path = _from_project_root(args.sid_manifest)
    output_dir = _from_project_root(args.output_dir)
    try:
        poi_data_path, poi_data_source = resolve_poi_data_path(
            sid_manifest_path,
            PROJECT_ROOT,
            args.poi_data,
        )
        print(
            f"POI 数据路径来源：{poi_data_source}；路径：{poi_data_path}",
            file=sys.stderr,
        )
        result = build_geohash_pid(
            sid_manifest_path,
            poi_data_path,
            output_dir,
            geohash_length=args.geohash_length,
            order=args.order,
            sid_codebook_size=args.sid_codebook_size,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (GeohashPidError, OSError) as error:
        print(f"PID 构建失败：{error}", file=sys.stderr)
        return 2

    sid = result.metrics["sid_only"]
    pid = result.metrics["gid6_sid_pid"]
    resolution = result.metrics["collision_resolution"]
    print(
        json.dumps(
            {
                "status": "completed",
                "poi_count": result.manifest["poi_count"],
                "sid_distinct_ratio": sid["distinct_sid_ratio"],
                "pid_distinct_ratio": pid["distinct_pid_ratio"],
                "sid_colliding_poi_resolution_ratio": resolution[
                    "sid_colliding_poi_resolution_ratio"
                ],
                "residual_colliding_poi_count": resolution[
                    "residual_colliding_poi_count"
                ],
                "maximum_residual_pid_bucket_size": resolution[
                    "maximum_residual_pid_bucket_size"
                ],
                "residual_case_count": len(result.residual_cases),
                "output_dir": str(output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

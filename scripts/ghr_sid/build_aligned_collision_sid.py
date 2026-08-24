#!/usr/bin/env python3
"""Build the fixed two-token globally aligned TIGER collision suffix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.ghr_sid.aligned_collision_quantizer import (  # noqa: E402
    AlignedCollisionQuantizerError,
    build_aligned_collision_identifiers,
    load_aligned_collision_config,
    validate_aligned_collision_output,
)


DEFAULT_CONFIG = "configs/sid/ghr_tiger_aligned_collision_32x32.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "冻结 TIGER 三层 SID，只对其碰撞桶构造两层 32×32 全局共享后缀："
            "使用 TIGER latent residual 与多尺度地理编码拟合共享锚点，再在每个桶内"
            "执行唯一最小代价分配。"
        )
    )
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="不重新构建，仅核验已完成产物及 SHA256。",
    )
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        config = load_aligned_collision_config(
            _rooted(args.config), PROJECT_ROOT
        )
        if args.validate_only:
            payload = validate_aligned_collision_output(config.output_dir)
        else:
            result = build_aligned_collision_identifiers(
                project_root=PROJECT_ROOT,
                config=config,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            payload = {
                "status": result.metrics["status"],
                "experiment_id": result.metrics["experiment_id"],
                "base_tiger": result.metrics["base_tiger"],
                "suffix": result.metrics["suffix"],
                "identifier": result.metrics["identifier"],
                "output_dir": str(result.output_dir),
            }
    except (AlignedCollisionQuantizerError, OSError, ValueError) as error:
        print(f"GHR-SID 全局对齐碰撞后缀构建失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

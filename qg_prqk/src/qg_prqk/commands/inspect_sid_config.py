"""Inspect the resolved Semantic-ID pipeline configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qg_prqk.sid.pipeline_config import DownstreamConfigError, load_downstream_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="只读检查 512×3 下游配置与冻结上游配置哈希；不读取业务数据、不启动训练。",
    )
    parser.add_argument("--config", type=Path, required=True, help="512×3 下游 YAML 配置")
    parser.add_argument("--resolved", action="store_true", help="显示继承后的完整配置")
    args = parser.parse_args(argv)
    try:
        config = load_downstream_config(args.config)
    except (DownstreamConfigError, OSError) as error:
        parser.error(str(error))
    result = {
        "status": "config_validated",
        "artifact_contents_checked": False,
        "downstream_started": False,
        "signature": config.signature(),
        "codebook_sizes": list(config.codebook_sizes),
        "poi_embedding_dim": config.resolved_payload()["frozen_inputs"]["poi_embedding_dim"],
        "output_dir": str(config.output_dir),
        "query_view_policy": config.query_view_policy,
        "upstream_artifacts": config.upstream_artifacts,
    }
    if args.resolved:
        result["resolved_config"] = config.resolved_payload()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect relational/local codebook input headers without loading payloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from qg_prqk.sid.relational_config import load_relational_codebook_config
from qg_prqk.sid.relational_data import inspect_relational_input_headers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "只读核验关系/局部码本配置及 Query 图、基础码本、类别数据的 header；"
            "不读取业务 Validation/Test、不写输出、不启动聚类。"
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="关系码本 canonical YAML")
    args = parser.parse_args(argv)
    try:
        result = inspect_relational_input_headers(
            load_relational_codebook_config(args.config)
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"关系码本输入核验失败：{error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

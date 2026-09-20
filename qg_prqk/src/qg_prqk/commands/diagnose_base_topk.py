"""Run the paired hard-fit/Top-k base-codebook diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "基础码本 100k 补充诊断：在收敛 hard60 后运行 Top-k5，"
            "与冻结 hard60 Top-k-off 配对比较；不修改 canonical 配置。"
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="512×3 下游 YAML")
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        required=True,
        help="已完成基础码本 medium100k manifest.json",
    )
    parser.add_argument(
        "--reference-manifest-sha256",
        required=True,
        help="外部冻结的基础码本 medium100k manifest SHA256",
    )
    parser.add_argument(
        "--endpoint-diagnostic-manifest",
        type=Path,
        required=True,
        help="已完成 hard30/60 Top-k-off 诊断 manifest.json",
    )
    parser.add_argument(
        "--endpoint-diagnostic-manifest-sha256",
        required=True,
        help="外部冻结的 hard endpoint 诊断 manifest SHA256",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对两个 reference 合同；不读 embedding 值、不写输出",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="独立复核来源、配对端点、assignment、residual 和 comparison",
    )
    args = parser.parse_args(argv)
    try:
        from qg_prqk.sid.pipeline_config import load_downstream_config
        from qg_prqk.sid.base_data import load_base_codebook_inputs, load_base_codebook_algorithm
        from qg_prqk.sid.base_diagnostics import _validate_reference
        from qg_prqk.sid.base_topk_diagnostics import (
            _validate_endpoint_diagnostic,
            build_base_topk_diagnostic,
            diagnostic_output_directory,
            hard60_topk5_settings,
            validate_base_topk_diagnostic,
        )

        config = load_downstream_config(args.config)
        inputs = load_base_codebook_inputs(
            config,
            gate="medium100k",
            verify_embedding_hash=not args.dry_run,
        )
        canonical = load_base_codebook_algorithm(config)
        if args.dry_run:
            reference = _validate_reference(
                config,
                inputs,
                canonical,
                args.reference_manifest,
                args.reference_manifest_sha256,
                verify_artifacts=False,
            )
            endpoint = _validate_endpoint_diagnostic(
                config,
                inputs,
                canonical,
                args.endpoint_diagnostic_manifest,
                args.endpoint_diagnostic_manifest_sha256,
                base_reference_sha256=args.reference_manifest_sha256,
                verify_artifacts=False,
            )
            result = {
                "status": "dry_run_passed",
                "phase": ("P5-CAT-MEDIUM100K-HARD60-TOPK5-DIAGNOSTIC"),
                "selected_poi_rows": len(inputs.selected_rows),
                "selected_rows_sha256": (inputs.source_hashes["selected_rows_sha256"]),
                "reference_manifest_sha256": reference["manifest_sha256"],
                "endpoint_diagnostic_manifest_sha256": endpoint["manifest_sha256"],
                "canonical_settings_unchanged": True,
                "diagnostic_settings": asdict(hard60_topk5_settings(canonical)),
                "output_dir": str(diagnostic_output_directory(config)),
                "embedding_values_read": False,
                "output_written": False,
            }
        elif args.validate_only:
            result = validate_base_topk_diagnostic(
                config,
                inputs,
                canonical,
                reference_manifest=args.reference_manifest,
                reference_manifest_sha256=(args.reference_manifest_sha256),
                endpoint_diagnostic_manifest=(args.endpoint_diagnostic_manifest),
                endpoint_diagnostic_manifest_sha256=(
                    args.endpoint_diagnostic_manifest_sha256
                ),
            )
        else:
            result = build_base_topk_diagnostic(
                config,
                inputs,
                canonical,
                reference_manifest=args.reference_manifest,
                reference_manifest_sha256=(args.reference_manifest_sha256),
                endpoint_diagnostic_manifest=(args.endpoint_diagnostic_manifest),
                endpoint_diagnostic_manifest_sha256=(
                    args.endpoint_diagnostic_manifest_sha256
                ),
            )
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"基础码本 100k hard60+Top-k5 诊断失败：{error}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

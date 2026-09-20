"""Build or validate the POI-only base codebook."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "在 active POI BGE 上构建 POI-only PRQ-KMeans 512×3 基础码本；"
            "不读取 Query、类别、Geo 或业务 Validation/Test。"
        )
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="已确认的基础码本 YAML"
    )
    parser.add_argument(
        "--gate",
        choices=("sample", "medium100k", "medium500k", "full"),
        default="sample",
        help="依次执行 sample、100k、500k、全部 716245 POI",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        help="sample 行数，默认 10000；只允许 512～10000",
    )
    parser.add_argument(
        "--previous-gate-manifest",
        type=Path,
        help="非 sample 构建必须显式绑定的上一 Gate manifest.json",
    )
    parser.add_argument(
        "--previous-gate-manifest-sha256",
        help="上一 Gate manifest 的冻结 SHA256，不从现存文件自动推断",
    )
    parser.add_argument(
        "--direct-full-from-sample-user-authorized",
        action="store_true",
        help=(
            "仅用于用户明确授权跳过 100k/500k 时，从已验收 sample 直接构建 full；"
            "仍必须显式绑定 sample manifest 及 SHA256"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查配置、manifest、shape 与确定性选样；不读向量值、不写输出",
    )
    mode.add_argument(
        "--resume", action="store_true", help="校验并恢复已提交层级或复核完成目录"
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="独立复核输入、全部 artifact、精确 assignment 与 projection residual",
    )
    args = parser.parse_args(argv)
    if (args.previous_gate_manifest is None) != (
        args.previous_gate_manifest_sha256 is None
    ):
        parser.error(
            "--previous-gate-manifest 与 --previous-gate-manifest-sha256 必须同时提供"
        )
    if args.direct_full_from_sample_user_authorized and args.gate != "full":
        parser.error("--direct-full-from-sample-user-authorized 只允许 --gate full")
    if args.direct_full_from_sample_user_authorized and (
        args.dry_run or args.validate_only
    ):
        parser.error("direct-full 授权只在正式构建/恢复时提供；独立校验从 manifest 复核")
    if (
        args.gate != "sample"
        and not args.dry_run
        and not args.validate_only
        and args.previous_gate_manifest is None
    ):
        parser.error("非 sample 构建/恢复必须显式绑定上一 Gate manifest 及 SHA256")
    try:
        from qg_prqk.sid.base_config import load_base_codebook_config, validate_selection_evidence
        from qg_prqk.sid.base_data import gate_next_status, load_base_codebook_inputs, load_base_codebook_algorithm

        config = load_base_codebook_config(args.config)
        validate_selection_evidence(config)
        inputs = load_base_codebook_inputs(
            config,
            gate=args.gate,
            sample_rows=args.sample_rows,
            verify_embedding_hash=not args.dry_run,
        )
        if args.dry_run:
            result = {
                "status": "dry_run_passed",
                "phase": f"P5-CAT-{args.gate.upper()}",
                "active_poi_rows": inputs.total_rows,
                "selected_poi_rows": len(inputs.selected_rows),
                "selected_rows_sha256": inputs.source_hashes["selected_rows_sha256"],
                "embedding_shape": list(inputs.embeddings.shape),
                "codebook_sizes": list(config.codebook_sizes),
                "protocol_id": config.protocol_id,
                "output_dir": str(config.p5_output_dir),
                "algorithm": asdict(load_base_codebook_algorithm(config)),
                "embedding_values_read": False,
                "output_written": False,
                "next_status": gate_next_status(config, args.gate),
            }
        else:
            from qg_prqk.sid.base_quantizer import build_base_codebook, validate_base_codebook

            if args.validate_only:
                result = validate_base_codebook(config, inputs, gate=args.gate)
            else:
                result = build_base_codebook(
                    config,
                    inputs,
                    gate=args.gate,
                    previous_manifest=args.previous_gate_manifest,
                    previous_manifest_sha256=args.previous_gate_manifest_sha256,
                    direct_full_from_sample_user_authorized=(
                        args.direct_full_from_sample_user_authorized
                    ),
                    resume=args.resume,
                )
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"基础码本构建失败：{error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

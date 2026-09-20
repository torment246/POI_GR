#!/usr/bin/env python3
"""Evaluate no-training POI resolvers over frozen SID Beam-10 buckets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.bucket_rerank.evaluation import (  # noqa: E402
    BucketRerankError,
    evaluate_bucket_rerank,
    validate_bucket_rerank_output,
)


TRACE_ROOT = (
    "outputs/eval/tiger_bucket_diagnostics_fixed10k_v3/tiger_e3/runs/"
    "valid_checkpoint-16713_beam10_bucketdiag_subset10000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "展开冻结 Beam=10 的三层语义 Bucket，比较 Train-only 热度、词法、"
            "冻结语义向量和无调参 RRF 排序，输出最终 POI Top-10。"
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-trace",
        type=Path,
        default=Path(TRACE_ROOT) / "candidate_trace.jsonl",
    )
    parser.add_argument(
        "--candidate-trace-manifest",
        type=Path,
        default=Path(TRACE_ROOT) / "candidate_trace_manifest.json",
    )
    parser.add_argument(
        "--source-result", type=Path, default=Path(TRACE_ROOT) / "result.json"
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=Path("data/eval/embedding_retrieval/sft_validation_10k_v1/eval.jsonl"),
    )
    parser.add_argument(
        "--query-mapping",
        type=Path,
        default=Path(
            "outputs/evaluation/sft_validation_10k_v1/query_embeddings/"
            "bge_m3_no_instruction_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--query-embeddings",
        type=Path,
        default=Path(
            "outputs/evaluation/sft_validation_10k_v1/query_embeddings/"
            "bge_m3_no_instruction.npy"
        ),
    )
    parser.add_argument(
        "--query-run-manifest",
        type=Path,
        default=Path(
            "outputs/evaluation/sft_validation_10k_v1/bge_m3_no_instruction/"
            "run_manifest.json"
        ),
    )
    parser.add_argument(
        "--poi-embeddings",
        type=Path,
        default=Path("outputs/embeddings/beijing_poi_bge_m3/embeddings.npy"),
    )
    parser.add_argument(
        "--poi-embedding-manifest",
        type=Path,
        default=Path("outputs/embeddings/beijing_poi_bge_m3/manifest.json"),
    )
    parser.add_argument(
        "--identifier-dir",
        type=Path,
        default=Path(
            "outputs/sid/tiger/bge_m3/TIGER-BGE-M3-1024x3/"
            "tiger_ids/epoch_20"
        ),
    )
    parser.add_argument(
        "--poi-dir", type=Path, default=Path("data/beijing_poi_clean_20260715_json")
    )
    parser.add_argument(
        "--query-aggregate-dir",
        type=Path,
        default=Path(
            "outputs/embeddings/query_augmented_bge_m3/query_poi_aggregates_v1"
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="仅用于 smoke；正式固定评测不设置。",
    )
    parser.add_argument("--examples-per-kind", type=int, default=10)
    parser.add_argument(
        "--generator-label",
        default="TIGER epoch 3 unconstrained Beam=10 trace",
        help="写入产物协议的候选生成器标签。",
    )
    parser.add_argument(
        "--semantic-label",
        default="BGE-M3",
        help="写入产物协议的 POI 连续语义向量标签。",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只核验 --output-dir 内终态产物的 SHA256、行数和 shape。",
    )
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    output_dir = _rooted(args.output_dir)
    try:
        if args.validate_only:
            payload = validate_bucket_rerank_output(output_dir)
        else:
            result = evaluate_bucket_rerank(
                project_root=PROJECT_ROOT,
                candidate_trace_path=_rooted(args.candidate_trace),
                candidate_trace_manifest_path=_rooted(args.candidate_trace_manifest),
                source_result_path=_rooted(args.source_result),
                eval_data_path=_rooted(args.eval_data),
                query_mapping_path=_rooted(args.query_mapping),
                query_embeddings_path=_rooted(args.query_embeddings),
                query_run_manifest_path=_rooted(args.query_run_manifest),
                poi_embeddings_path=_rooted(args.poi_embeddings),
                poi_embedding_manifest_path=_rooted(args.poi_embedding_manifest),
                identifier_dir=_rooted(args.identifier_dir),
                poi_dir=_rooted(args.poi_dir),
                query_aggregate_dir=_rooted(args.query_aggregate_dir),
                output_dir=output_dir,
                max_rows=args.max_rows,
                examples_per_kind=args.examples_per_kind,
                generator_label=args.generator_label,
                semantic_label=args.semantic_label,
            )
            rankings = result.metrics["rankings"]
            payload = {
                "status": result.metrics["status"],
                "samples": result.metrics["candidate_pool"]["samples"],
                "candidate_target_recall": result.metrics["candidate_pool"][
                    "target_recall"
                ],
                "original_exact": rankings["original_c_token"]["exact"],
                "global_rrf_content_exact": rankings["global_rrf_content"]["exact"],
                "output_dir": str(result.output_dir),
            }
    except (BucketRerankError, OSError, ValueError) as error:
        print(f"桶内重排评测失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

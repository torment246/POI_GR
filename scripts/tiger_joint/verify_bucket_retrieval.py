#!/usr/bin/env python3
"""Independently recompute a persisted TIGER-Joint Bucket-HR evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger_joint.evaluation import (  # noqa: E402
    JointBucketIndex,
    JointSidCandidateParser,
    TigerJointEvaluationError,
    empty_bucket_metrics,
    finalize_bucket_metrics,
    update_bucket_metrics,
)
from poi_gr.methods.tiger_joint.preparation import (  # noqa: E402
    build_sid_token_layout,
    load_dynamic_token_ids,
)
from poi_gr.pid.trie import sha256_file  # noqa: E402
from poi_gr.sft.evaluation import load_lf_tokenizer_and_template  # noqa: E402


CODEBOOK_SIZES = (1024, 1024, 1024)


class TigerJointVerificationError(TigerJointEvaluationError):
    """Raised when any persisted candidate or metric is inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="逐行重解析原始 Beam token，并独立复算 TIGER-Joint Bucket-HR。"
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise TigerJointVerificationError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerJointVerificationError(f"{name} JSON 非法") from error
    if not isinstance(value, dict):
        raise TigerJointVerificationError(f"{name} 必须是 object")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_candidate(candidate: Any, parser: JointSidCandidateParser) -> Any:
    if not isinstance(candidate, dict):
        raise TigerJointVerificationError("候选轨迹 candidate 不是 object")
    sequence = candidate.get("sequence_token_ids")
    score = candidate.get("sequence_score")
    if not isinstance(sequence, list) or not isinstance(score, (int, float)):
        raise TigerJointVerificationError("候选 token/score 无效")
    reparsed = parser.parse(sequence, float(score))
    stored_bucket = candidate.get("bucket")
    expected_bucket = list(reparsed.bucket) if reparsed.bucket is not None else None
    if (
        stored_bucket != expected_bucket
        or candidate.get("prefix_error") != reparsed.prefix_error
        or candidate.get("target_close_valid") != reparsed.target_close_valid
    ):
        raise TigerJointVerificationError("候选 raw token 重解析结果不一致")
    return reparsed


def verify(eval_dir: Path) -> dict[str, Any]:
    eval_dir = eval_dir.resolve()
    result_path = eval_dir / "result.json"
    sid_manifest_path = eval_dir / "sid_manifest.json"
    trace_manifest_path = eval_dir / "candidate_trace_manifest.json"
    result = load_object(result_path, "result")
    sid_manifest = load_object(sid_manifest_path, "SID manifest")
    trace_manifest = load_object(trace_manifest_path, "candidate trace manifest")
    if result.get("status") != "completed" or trace_manifest.get("status") != "completed":
        raise TigerJointVerificationError("正式结果或候选轨迹未完成")
    checkpoint_value = result.get("inputs", {}).get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise TigerJointVerificationError("result 缺少 checkpoint")
    checkpoint = Path(checkpoint_value).resolve()
    tokenizer, _ = load_lf_tokenizer_and_template(
        checkpoint,
        project_root=PROJECT_ROOT,
    )
    token_ids = load_dynamic_token_ids(tokenizer)
    layout = build_sid_token_layout(tokenizer, CODEBOOK_SIZES)
    parser = JointSidCandidateParser(
        token_layout=layout,
        target_open_token_id=token_ids.target_open,
        target_close_token_id=token_ids.target_close,
    )
    codes_spec = sid_manifest.get("sid_codes")
    if not isinstance(codes_spec, dict):
        raise TigerJointVerificationError("SID manifest 缺少 codes")
    codes_path = Path(str(codes_spec.get("path")))
    if not codes_path.is_absolute():
        codes_path = eval_dir / codes_path
    expected_codes_hash = codes_spec.get("sha256")
    if (
        not isinstance(expected_codes_hash, str)
        or sha256_file(codes_path) != expected_codes_hash
    ):
        raise TigerJointVerificationError("SID codes 哈希不一致")
    sid_codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
    if sid_codes.shape != (2_337_178, 3) or sid_codes.dtype != np.int32:
        raise TigerJointVerificationError("SID codes shape/dtype 不一致")
    index = JointBucketIndex.from_codes(sid_codes, CODEBOOK_SIZES)

    parts = trace_manifest.get("parts")
    if not isinstance(parts, list) or len(parts) != 20:
        raise TigerJointVerificationError("候选轨迹必须恰好包含 20 个分片")
    expected_ordered_hash = hashlib.sha256(
        "".join(str(part.get("sha256")) for part in parts).encode("ascii")
    ).hexdigest()
    if expected_ordered_hash != trace_manifest.get("ordered_parts_sha256"):
        raise TigerJointVerificationError("候选分片有序哈希不一致")

    metrics = empty_bucket_metrics()
    expected_row_index = 0
    for part_index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise TigerJointVerificationError("候选轨迹分片声明无效")
        part_path = eval_dir / str(part.get("path"))
        if sha256_file(part_path) != part.get("sha256"):
            raise TigerJointVerificationError("候选轨迹分片 SHA256 不一致")
        part_rows = 0
        with part_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError as error:
                    raise TigerJointVerificationError("候选轨迹 JSONL 非法") from error
                if not isinstance(trace, dict):
                    raise TigerJointVerificationError("候选轨迹行不是 object")
                if trace.get("row_index") != expected_row_index:
                    raise TigerJointVerificationError("候选轨迹 row_index 不连续")
                target_row = trace.get("target_poi_row")
                target_bucket = trace.get("target_bucket")
                if (
                    isinstance(target_row, bool)
                    or not isinstance(target_row, int)
                    or target_row < 0
                    or target_row >= len(sid_codes)
                    or not isinstance(target_bucket, list)
                    or target_bucket
                    != [int(value) for value in sid_codes[target_row]]
                    or trace.get("target_bucket_size")
                    != index.bucket_size(target_bucket)
                ):
                    raise TigerJointVerificationError("目标 POI 行、SID 或桶大小不一致")
                raw_candidates = trace.get("candidates")
                if not isinstance(raw_candidates, list) or len(raw_candidates) != 10:
                    raise TigerJointVerificationError("每行必须恰好保存 10 个候选")
                if [candidate.get("beam_rank") for candidate in raw_candidates] != list(
                    range(1, 11)
                ):
                    raise TigerJointVerificationError("候选 Beam rank 不连续")
                candidates = [
                    normalize_candidate(candidate, parser)
                    for candidate in raw_candidates
                ]
                for stored, reparsed in zip(
                    raw_candidates, candidates, strict=True
                ):
                    expected_size = (
                        index.bucket_size(reparsed.bucket)
                        if reparsed.bucket is not None
                        else 0
                    )
                    if stored.get("bucket_size") != expected_size:
                        raise TigerJointVerificationError("候选桶大小不一致")
                ranking = update_bucket_metrics(
                    metrics,
                    target_codes=target_bucket,
                    candidates=candidates,
                    index=index,
                )
                expected_unique = [
                    {
                        "codes": list(bucket),
                        "first_beam_rank": beam_rank,
                        "bucket_size": bucket_size,
                    }
                    for bucket, beam_rank, bucket_size in zip(
                        ranking.unique_buckets,
                        ranking.unique_bucket_first_beam_ranks,
                        ranking.unique_bucket_sizes,
                        strict=True,
                    )
                ]
                if (
                    trace.get("raw_slot_bucket_target_rank")
                    != ranking.raw_slot_target_rank
                    or trace.get("unique_bucket_target_rank")
                    != ranking.unique_target_rank
                    or trace.get("unique_expandable_buckets") != expected_unique
                ):
                    raise TigerJointVerificationError("候选桶排序或目标 rank 不一致")
                expected_row_index += 1
                part_rows += 1
        if (
            part.get("row_start") != part_index * 500
            or part.get("rows") != part_rows
            or part_rows != 500
        ):
            raise TigerJointVerificationError("候选分片行范围不一致")
    if expected_row_index != 10_000 or trace_manifest.get("rows") != 10_000:
        raise TigerJointVerificationError("候选轨迹总行数不是 10,000")
    legal_path_constraint = result.get("scope", {}).get("legal_path_constraint")
    if not isinstance(legal_path_constraint, bool):
        raise TigerJointVerificationError("result 缺少合法路径约束布尔标记")
    recomputed = finalize_bucket_metrics(
        metrics,
        legal_path_constraint=legal_path_constraint,
    )
    if recomputed != result.get("bucket_metrics"):
        raise TigerJointVerificationError("独立复算 Bucket 指标与 result 不一致")
    verification = {
        "schema_version": "tiger-joint-bucket-eval-verification-v1",
        "status": "passed",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "sid_manifest": str(sid_manifest_path),
        "sid_manifest_sha256": sha256_file(sid_manifest_path),
        "sid_codes_sha256": expected_codes_hash,
        "candidate_trace_manifest": str(trace_manifest_path),
        "candidate_trace_manifest_sha256": sha256_file(trace_manifest_path),
        "verified_rows": expected_row_index,
        "verified_candidates": expected_row_index * 10,
        "verified_parts": len(parts),
        "checks": {
            "raw_token_reparsed": True,
            "beam_ranks_contiguous": True,
            "target_rows_match_sid_codes": True,
            "catalog_bucket_sizes_match": True,
            "unique_bucket_ranking_match": True,
            "part_hashes_match": True,
            "bucket_metrics_exact_match": True,
            "decoding_constraint_mode_match": True,
        },
        "bucket_metrics": recomputed,
    }
    atomic_json(eval_dir / "verification.json", verification)
    return verification


def main() -> int:
    args = parse_args()
    try:
        verification = verify(resolve(args.eval_dir))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": verification["status"],
                "verified_rows": verification["verified_rows"],
                "verified_candidates": verification["verified_candidates"],
                "bucket_metrics": verification["bucket_metrics"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

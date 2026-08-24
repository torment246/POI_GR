#!/usr/bin/env python3
"""Evaluate TIGER-style models with frozen identifier lookup."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sft.evaluation import (  # noqa: E402
    build_reference_aligned_validation_subset,
    load_generation_model,
    load_lf_tokenizer_and_template,
    validate_split_manifest,
)
from poi_gr.methods.tiger.eval import (  # noqa: E402
    TigerEvalError,
    TigerIdIndex,
    TigerLegalPathConstraint,
    TigerTokenIds,
    atomic_json,
    canonical_sha256,
    empty_bucket_metrics,
    empty_metrics,
    encode_tiger_record,
    finalize_bucket_metrics,
    finalize_metrics,
    load_tiger_id_index,
    load_tiger_token_ids,
    merge_bucket_metrics,
    merge_metrics,
    parse_generated_candidate,
    update_bucket_metrics,
    update_metrics,
)
from poi_gr.pid.trie import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "默认按 TIGER 论文使用普通 Beam Search 生成四层 identifier；也可显式启用"
            "仅用于诊断的冻结语料库合法路径约束。"
        )
    )
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--reference-validation-subset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--expected-checkpoint-steps", type=int, nargs="+", required=True
    )
    parser.add_argument(
        "--expected-checkpoint-epochs", type=float, nargs="+", default=(1, 2, 3)
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--token-mapping-filename",
        default="tiger_token_mapping.json",
        help="Tokenizer 目录内的 Token 映射文件；RQ-KMeans 扩词表使用 poi_token_mapping.json。",
    )
    parser.add_argument("--identifier-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument(
        "--per-device-eval-batch-size", type=int, choices=(32, 16, 8, 4, 2), default=16
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--skip-data-hash", action="store_true")
    parser.add_argument(
        "--legal-path-constraint",
        action="store_true",
        help="诊断模式：Beam 每一步只允许沿冻结全库 identifier 的真实 Prefix 扩展。",
    )
    parser.add_argument(
        "--final-checkpoint-only",
        action="store_true",
        help="专项诊断只评测一个已冻结的最终 checkpoint；默认仍要求三轮。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只核验固定子集、Tokenizer、identifier、checkpoint 和 Prompt，不加载模型。",
    )
    parser.add_argument(
        "--bucket-diagnostics",
        action="store_true",
        help=(
            "保存全部逐样本 Beam 候选，并统计去掉 Collision Token 后前三层"
            " [S1,S2,S3] 的 Bucket-HR。"
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise TigerEvalError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerEvalError(f"{name} JSON 非法") from error
    if not isinstance(value, dict):
        raise TigerEvalError(f"{name} 必须是 object")
    return value


def validate_checkpoints(
    checkpoints: Sequence[Path],
    *,
    expected_steps: Sequence[int],
    expected_epochs: Sequence[float],
    tokenizer_path: Path,
    token_mapping_filename: str,
) -> list[dict[str, Any]]:
    if not checkpoints or len(checkpoints) != len(expected_steps):
        raise TigerEvalError("Checkpoint 数量必须与 expected steps 一致且非空")
    if len(expected_epochs) != len(expected_steps):
        raise TigerEvalError("expected epochs 数量必须与 expected steps 一致")
    mapping_payload = load_json(
        tokenizer_path / token_mapping_filename,
        "TIGER Token mapping",
    )
    token_mapping = mapping_payload.get("tokens")
    if not isinstance(token_mapping, dict):
        raise TigerEvalError("TIGER Token mapping 缺少 tokens")
    expected_vocab = mapping_payload.get("new_vocab_size")
    results: list[dict[str, Any]] = []
    for checkpoint, expected_step, expected_epoch in zip(
        checkpoints,
        expected_steps,
        expected_epochs,
    ):
        checkpoint = checkpoint.resolve()
        if checkpoint.name != f"checkpoint-{expected_step}":
            raise TigerEvalError(f"Checkpoint 顺序或步数错误：{checkpoint.name}")
        required = (
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "added_tokens.json",
            "trainer_state.json",
        )
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise TigerEvalError(f"{checkpoint.name} 缺少文件：{', '.join(missing)}")
        state = load_json(checkpoint / "trainer_state.json", "trainer_state")
        actual_epoch = float(state.get("epoch", -1))
        if abs(actual_epoch - expected_epoch) > 0.001:
            raise TigerEvalError(
                f"{checkpoint.name} epoch {actual_epoch} != {expected_epoch}"
            )
        config = load_json(checkpoint / "config.json", "checkpoint config")
        if config.get("vocab_size") != expected_vocab:
            raise TigerEvalError(f"{checkpoint.name} vocab_size 不一致")
        added = load_json(checkpoint / "added_tokens.json", "added tokens")
        for token, token_id in token_mapping.items():
            if added.get(token) != token_id:
                raise TigerEvalError(f"{checkpoint.name} Token ID 不一致：{token}")
        eval_loss = next(
            (
                float(item["eval_loss"])
                for item in reversed(state.get("log_history", []))
                if item.get("step") == expected_step and "eval_loss" in item
            ),
            None,
        )
        if eval_loss is None:
            raise TigerEvalError(f"{checkpoint.name} 缺少完整 Validation Loss")
        results.append(
            {
                "path": str(checkpoint),
                "name": checkpoint.name,
                "step": expected_step,
                "epoch": actual_epoch,
                "validation_loss": eval_loss,
                "model_sha256": sha256_file(checkpoint / "model.safetensors"),
                "tokenizer_json_sha256": sha256_file(checkpoint / "tokenizer.json"),
            }
        )
    if len({item["tokenizer_json_sha256"] for item in results}) != 1:
        raise TigerEvalError("三个 TIGER checkpoint 的 tokenizer 不一致")
    return results


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise TigerEvalError(
                    f"评测子集第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(value, dict):
                raise TigerEvalError(f"评测子集第 {line_number} 行不是 object")
            records.append(value)
    return records


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically persist one bounded candidate-trace part."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def trace_part_path(run_dir: Path, start: int, stop: int) -> Path:
    return run_dir / "candidate_trace_parts" / f"part-{start:08d}-{stop:08d}.jsonl"


def validate_candidate_trace_row(
    value: Mapping[str, Any],
    *,
    candidate_count_per_row: int,
) -> None:
    """Validate ranks and unique buckets from one persisted trace row."""

    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != candidate_count_per_row:
        raise TigerEvalError("候选轨迹候选数量不一致")
    if [item.get("beam_rank") for item in candidates] != list(
        range(1, candidate_count_per_row + 1)
    ):
        raise TigerEvalError("候选轨迹 Beam rank 不连续")
    target_poi_id = value.get("target_poi_id")
    target_bucket_value = value.get("target_base_bucket")
    if not isinstance(target_poi_id, str) or not isinstance(target_bucket_value, list):
        raise TigerEvalError("候选轨迹缺少目标 POI 或基础桶")
    target_bucket = tuple(int(code) for code in target_bucket_value)
    exact_rank = next(
        (
            int(candidate["beam_rank"])
            for candidate in candidates
            if candidate.get("poi_id") == target_poi_id
        ),
        None,
    )
    raw_rank: int | None = None
    unique_rank: int | None = None
    seen: set[tuple[int, int, int]] = set()
    expected_unique: list[tuple[list[int], int, float, int]] = []
    for candidate in candidates:
        bucket_value = candidate.get("base_bucket")
        bucket_size = candidate.get("base_bucket_size")
        if not isinstance(bucket_value, list) or not isinstance(bucket_size, int):
            if bucket_value is not None or bucket_size != 0:
                raise TigerEvalError("候选轨迹基础桶字段不一致")
            continue
        bucket = tuple(int(code) for code in bucket_value)
        if len(bucket) != 3 or bucket_size < 0:
            raise TigerEvalError("候选轨迹基础桶 shape/size 非法")
        if bucket_size == 0:
            continue
        beam_rank = int(candidate["beam_rank"])
        if bucket == target_bucket and raw_rank is None:
            raw_rank = beam_rank
        if bucket in seen:
            continue
        seen.add(bucket)
        expected_unique.append(
            (
                list(bucket),
                beam_rank,
                float(candidate["sequence_score"]),
                bucket_size,
            )
        )
        if bucket == target_bucket and unique_rank is None:
            unique_rank = len(seen)
    actual_unique = value.get("unique_expandable_buckets")
    if not isinstance(actual_unique, list):
        raise TigerEvalError("候选轨迹缺少唯一可展开桶")
    normalized_actual = [
        (
            item.get("codes"),
            item.get("first_beam_rank"),
            float(item.get("first_sequence_score")),
            item.get("bucket_size"),
        )
        for item in actual_unique
    ]
    if normalized_actual != expected_unique:
        raise TigerEvalError("候选轨迹唯一可展开桶与候选明细不一致")
    if (
        value.get("exact_target_rank"),
        value.get("raw_slot_bucket_target_rank"),
        value.get("unique_bucket_target_rank"),
    ) != (exact_rank, raw_rank, unique_rank):
        raise TigerEvalError("候选轨迹 target rank 与候选明细不一致")


def finalize_candidate_trace(
    run_dir: Path,
    *,
    target_rows: int,
    chunk_size: int,
    candidate_count_per_row: int,
) -> dict[str, Any]:
    """Validate trace parts and freeze one ordered JSONL artifact."""

    output_path = run_dir / "candidate_trace.jsonl"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    part_manifests: list[dict[str, Any]] = []
    aggregate_rows = 0
    with temporary.open("wb") as destination:
        for start in range(0, target_rows, chunk_size):
            stop = min(start + chunk_size, target_rows)
            part_path = trace_part_path(run_dir, start, stop)
            if not part_path.is_file():
                raise TigerEvalError(f"候选轨迹分片不存在：{part_path}")
            part_rows = 0
            with part_path.open("rb") as source:
                for expected_row_index, raw_line in enumerate(source, start=start):
                    try:
                        value = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        raise TigerEvalError(
                            f"候选轨迹分片 JSON 非法：{part_path}"
                        ) from error
                    if not isinstance(value, Mapping):
                        raise TigerEvalError(
                            f"候选轨迹分片每行必须是 object：{part_path}"
                        )
                    if value.get("row_index") != expected_row_index:
                        raise TigerEvalError(f"候选轨迹 row_index 不连续：{part_path}")
                    try:
                        validate_candidate_trace_row(
                            value,
                            candidate_count_per_row=candidate_count_per_row,
                        )
                    except (TigerEvalError, TypeError, ValueError) as error:
                        raise TigerEvalError(
                            f"候选轨迹逐行一致性校验失败：{part_path}"
                        ) from error
                    destination.write(raw_line)
                    part_rows += 1
            expected_rows = stop - start
            if part_rows != expected_rows:
                raise TigerEvalError(
                    f"候选轨迹分片行数 {part_rows} != {expected_rows}：{part_path}"
                )
            aggregate_rows += part_rows
            part_manifests.append(
                {
                    "path": str(part_path),
                    "start": start,
                    "stop": stop,
                    "rows": part_rows,
                    "sha256": sha256_file(part_path),
                }
            )
        destination.flush()
        os.fsync(destination.fileno())
    if aggregate_rows != target_rows:
        raise TigerEvalError(f"候选轨迹总行数 {aggregate_rows:,} != {target_rows:,}")
    os.replace(temporary, output_path)
    manifest = {
        "schema_version": "tiger-candidate-trace-v3",
        "status": "completed",
        "rows": aggregate_rows,
        "candidate_count_per_row": candidate_count_per_row,
        "output_file": str(output_path),
        "output_sha256": sha256_file(output_path),
        "parts": part_manifests,
    }
    manifest_path = run_dir / "candidate_trace_manifest.json"
    atomic_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_file": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def pad_prompts(
    prompts: Sequence[Sequence[int]], pad_token_id: int, device: Any
) -> tuple[Any, Any]:
    import torch

    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full(
        (len(prompts), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompts):
        values = torch.tensor(prompt, dtype=torch.long, device=device)
        input_ids[row, -len(prompt) :] = values
        attention_mask[row, -len(prompt) :] = 1
    return input_ids, attention_mask


def evaluate_chunk(
    records: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    template: Any,
    tokens: TigerTokenIds,
    index: TigerIdIndex,
    checkpoint_name: str,
    batch_size: int,
    cutoff_len: int,
    num_beams: int,
    existing_error_count: int,
    legal_path_constraint: bool,
    bucket_diagnostics: bool,
    row_offset: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
    float,
    int,
]:
    import torch

    examples = [
        encode_tiger_record(
            record,
            tokenizer=tokenizer,
            template=template,
            cutoff_len=cutoff_len,
            token_capacities=index.token_capacities,
        )
        for record in records
    ]
    target_rows: list[int] = []
    for example in examples:
        row = index.lookup(example.target_codes)
        if row < 0 or index.poi_id(row) != example.target_poi_id:
            raise TigerEvalError(f"Target identifier 映射不一致：{example.sample_id}")
        target_rows.append(row)

    metrics = empty_metrics()
    bucket_metrics = empty_bucket_metrics() if bucket_diagnostics else None
    errors: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    legal_path_cache: dict[tuple[int, ...], list[int]] = {}
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch_target_rows = target_rows[start : start + batch_size]
        input_ids, attention_mask = pad_prompts(
            [example.prompt_ids for example in batch_examples],
            tokenizer.pad_token_id,
            model.device,
        )
        prompt_width = int(input_ids.shape[1])
        generation_kwargs: dict[str, Any] = {}
        if legal_path_constraint:
            generation_kwargs["prefix_allowed_tokens_fn"] = TigerLegalPathConstraint(
                index=index,
                tokens=tokens,
                prompt_width=prompt_width,
                next_token_cache=legal_path_cache,
            )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=tokens.expected_sequence_length,
                length_penalty=1.0,
                early_stopping=True,
                renormalize_logits=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )
        raw_sequences = generated.sequences[:, prompt_width:].detach().cpu().tolist()
        if generated.sequences_scores is None:
            raise TigerEvalError("Beam Search 缺少 sequences_scores")
        raw_scores = generated.sequences_scores.float().detach().cpu().tolist()
        expected = len(batch_examples) * num_beams
        if len(raw_sequences) != expected or len(raw_scores) != expected:
            raise TigerEvalError("Beam Search 返回候选数量不一致")
        for batch_index, (example, target_row) in enumerate(
            zip(batch_examples, batch_target_rows)
        ):
            begin = batch_index * num_beams
            end = begin + num_beams
            ranked = sorted(
                zip(raw_sequences[begin:end], raw_scores[begin:end]),
                key=lambda item: (
                    -float(item[1]),
                    tuple(int(value) for value in item[0]),
                ),
            )
            candidates = [
                parse_generated_candidate(
                    sequence,
                    score,
                    tokens=tokens,
                    index=index,
                )
                for sequence, score in ranked
            ]
            target_rank = update_metrics(
                metrics,
                target_row=target_row,
                candidates=candidates,
            )
            bucket_ranking = None
            if bucket_metrics is not None:
                bucket_ranking = update_bucket_metrics(
                    bucket_metrics,
                    target_codes=example.target_codes,
                    candidates=candidates,
                    index=index,
                )
                record = records[start + batch_index]
                order_id = record.get("order_id")
                searchid = record.get("searchid")
                if not isinstance(order_id, str) or not order_id:
                    raise TigerEvalError(
                        f"Bucket 轨迹样本缺少 order_id：{example.sample_id}"
                    )
                if not isinstance(searchid, str) or not searchid:
                    raise TigerEvalError(
                        f"Bucket 轨迹样本缺少 searchid：{example.sample_id}"
                    )
                candidate_rows: list[dict[str, Any]] = []
                for beam_rank, ((sequence, _), candidate) in enumerate(
                    zip(ranked, candidates, strict=True),
                    start=1,
                ):
                    base_bucket = (
                        list(candidate.base_bucket)
                        if candidate.base_bucket is not None
                        else None
                    )
                    candidate_rows.append(
                        {
                            "beam_rank": beam_rank,
                            "sequence_score": candidate.score,
                            "sequence_token_ids": [int(value) for value in sequence],
                            "codes": (
                                list(candidate.codes)
                                if candidate.codes is not None
                                else None
                            ),
                            "base_bucket": base_bucket,
                            "base_bucket_size": (
                                index.bucket_size(candidate.base_bucket)
                                if candidate.base_bucket is not None
                                else 0
                            ),
                            "poi_row": candidate.poi_row,
                            "poi_id": (
                                index.poi_id(candidate.poi_row)
                                if candidate.poi_row is not None
                                else None
                            ),
                            "error": candidate.error,
                        }
                    )
                unique_bucket_rows = []
                for bucket, first_rank, bucket_size in zip(
                    bucket_ranking.unique_buckets,
                    bucket_ranking.unique_bucket_first_beam_ranks,
                    bucket_ranking.unique_bucket_sizes,
                    strict=True,
                ):
                    unique_bucket_rows.append(
                        {
                            "codes": list(bucket),
                            "first_beam_rank": first_rank,
                            "first_sequence_score": candidates[first_rank - 1].score,
                            "bucket_size": bucket_size,
                        }
                    )
                traces.append(
                    {
                        "row_index": row_offset + start + batch_index,
                        "sample_id": example.sample_id,
                        "order_id": order_id,
                        "searchid": searchid,
                        "checkpoint": checkpoint_name,
                        "target_poi_id": example.target_poi_id,
                        "target_poi_row": target_row,
                        "target_codes": list(example.target_codes),
                        "target_base_bucket": list(example.target_codes[:3]),
                        "target_bucket_size": index.bucket_size(
                            example.target_codes[:3]
                        ),
                        "exact_target_rank": target_rank,
                        "raw_slot_bucket_target_rank": (
                            bucket_ranking.raw_slot_target_rank
                        ),
                        "unique_bucket_target_rank": (
                            bucket_ranking.unique_target_rank
                        ),
                        "unique_expandable_buckets": unique_bucket_rows,
                        "candidates": candidate_rows,
                    }
                )
            if target_rank is None and existing_error_count + len(errors) < 100:
                errors.append(
                    {
                        "sample_id": example.sample_id,
                        "target_poi_id": example.target_poi_id,
                        "target_codes": list(example.target_codes),
                        "candidate_poi_ids": [
                            index.poi_id(candidate.poi_row)
                            if candidate.poi_row is not None
                            else None
                            for candidate in candidates
                        ],
                        "candidate_codes": [
                            list(candidate.codes)
                            if candidate.codes is not None
                            else None
                            for candidate in candidates
                        ],
                        "candidate_errors": [
                            candidate.error for candidate in candidates
                        ],
                        "checkpoint": checkpoint_name,
                    }
                )
        del generated, input_ids, attention_mask
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(model.device))
    return metrics, bucket_metrics, errors, traces, elapsed, peak_memory


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message and "cuda" in message


def evaluate_checkpoint(
    *,
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    subset_path: Path,
    subset_sha256: str,
    tokenizer: Any,
    template: Any,
    tokens: TigerTokenIds,
    token_metadata: Mapping[str, Any],
    index: TigerIdIndex,
    index_metadata: Mapping[str, Any],
    output_dir: Path,
    num_beams: int,
    initial_batch_size: int,
    chunk_size: int,
    cutoff_len: int,
    smoke_limit: int | None,
    legal_path_constraint: bool,
    bucket_diagnostics: bool,
) -> dict[str, Any]:
    import torch

    checkpoint = Path(metadata["path"])
    target_rows = smoke_limit or len(records)
    config = {
        "schema_version": "tiger-generative-eval-v1",
        "decoding": (
            "frozen_corpus_legal_path_constrained_beam_search"
            if legal_path_constraint
            else "unconstrained_beam_search_then_frozen_id_lookup"
        ),
        "is_paper_protocol": not legal_path_constraint,
        "invalid_ids_keep_original_rank": not legal_path_constraint,
        "legal_path_constraint": legal_path_constraint,
        "bucket_diagnostics": bucket_diagnostics,
        "candidate_trace_schema": (
            "tiger-candidate-trace-v3" if bucket_diagnostics else None
        ),
        "split": "valid",
        "data_file": str(subset_path),
        "data_sha256": subset_sha256,
        "rows": target_rows,
        "full_subset_rows": len(records),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": metadata["model_sha256"],
        "epoch": metadata["epoch"],
        "validation_loss": metadata["validation_loss"],
        "tokenizer": token_metadata,
        "identifier": index_metadata,
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
        "max_new_tokens": tokens.expected_sequence_length,
        "cutoff_len": cutoff_len,
        "chunk_size": chunk_size,
        "initial_batch_size": initial_batch_size,
        "smoke_limit": smoke_limit,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "method_code_sha256": sha256_file(
            PROJECT_ROOT / "src" / "poi_gr" / "methods" / "tiger" / "eval.py"
        ),
    }
    run_name = f"valid_{checkpoint.name}_beam{num_beams}"
    if legal_path_constraint:
        run_name += "_legalpath"
    if bucket_diagnostics:
        run_name += "_bucketdiag"
    if smoke_limit is not None:
        run_name += f"_smoke{smoke_limit}"
    else:
        run_name += f"_subset{len(records)}"
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    signature = canonical_sha256(config)
    if progress_path.is_file():
        progress = load_json(progress_path, "TIGER progress")
        if progress.get("signature") != signature:
            raise TigerEvalError(f"已有运行配置变化：{run_dir}")
        if bucket_diagnostics and not isinstance(
            progress.get("bucket_metrics"), Mapping
        ):
            raise TigerEvalError(f"已有运行缺少 Bucket 指标：{run_dir}")
    else:
        progress = {
            "schema_version": "tiger-eval-progress-v1",
            "status": "running",
            "signature": signature,
            "config": config,
            "next_line": 0,
            "metrics": empty_metrics(),
            "bucket_metrics": (empty_bucket_metrics() if bucket_diagnostics else None),
            "inference_seconds": 0.0,
            "peak_memory_bytes": 0,
            "actual_batch_size": initial_batch_size,
            "error_cases": [],
        }
        atomic_json(progress_path, progress)

    model = load_generation_model(
        checkpoint,
        expected_vocab_size=len(tokenizer),
    )
    allowed = [value for value in (32, 16, 8, 4, 2) if value <= initial_batch_size]
    batch_size = min(int(progress["actual_batch_size"]), allowed[0])
    while int(progress["next_line"]) < target_rows:
        start = int(progress["next_line"])
        stop = min(start + chunk_size, target_rows)
        while True:
            try:
                (
                    chunk_metrics,
                    chunk_bucket_metrics,
                    errors,
                    traces,
                    seconds,
                    peak_memory,
                ) = evaluate_chunk(
                    records[start:stop],
                    model=model,
                    tokenizer=tokenizer,
                    template=template,
                    tokens=tokens,
                    index=index,
                    checkpoint_name=checkpoint.name,
                    batch_size=batch_size,
                    cutoff_len=cutoff_len,
                    num_beams=num_beams,
                    existing_error_count=len(progress["error_cases"]),
                    legal_path_constraint=legal_path_constraint,
                    bucket_diagnostics=bucket_diagnostics,
                    row_offset=start,
                )
                break
            except RuntimeError as error:
                if not is_cuda_oom(error):
                    raise
                torch.cuda.empty_cache()
                smaller = [value for value in allowed if value < batch_size]
                if not smaller:
                    raise TigerEvalError("TIGER batch_size=2 仍然 OOM") from error
                batch_size = smaller[0]
                print(
                    f"CUDA OOM，当前分片改用 batch_size={batch_size}", file=sys.stderr
                )
        if bucket_diagnostics:
            if chunk_bucket_metrics is None:
                raise TigerEvalError("Bucket 诊断分片缺少 Bucket 指标")
            if len(traces) != stop - start:
                raise TigerEvalError("Bucket 候选轨迹分片行数不一致")
            atomic_jsonl(trace_part_path(run_dir, start, stop), traces)
            merge_bucket_metrics(progress["bucket_metrics"], chunk_bucket_metrics)
        merge_metrics(progress["metrics"], chunk_metrics)
        progress["next_line"] = stop
        progress["inference_seconds"] += seconds
        progress["peak_memory_bytes"] = max(
            int(progress["peak_memory_bytes"]), peak_memory
        )
        progress["actual_batch_size"] = batch_size
        progress["error_cases"].extend(errors)
        atomic_json(progress_path, progress)
        print(
            f"[{run_name}] {stop:,}/{target_rows:,} ({stop / target_rows:.2%})",
            file=sys.stderr,
            flush=True,
        )
    candidate_trace = (
        finalize_candidate_trace(
            run_dir,
            target_rows=target_rows,
            chunk_size=chunk_size,
            candidate_count_per_row=num_beams,
        )
        if bucket_diagnostics
        else None
    )
    result = {
        "schema_version": "tiger-eval-result-v1",
        "status": "completed",
        "config": config,
        "metrics": finalize_metrics(progress["metrics"]),
        "performance": {
            "inference_seconds": progress["inference_seconds"],
            "samples_per_second": target_rows / progress["inference_seconds"],
            "peak_memory_bytes": progress["peak_memory_bytes"],
            "actual_batch_size": progress["actual_batch_size"],
        },
        "run_dir": str(run_dir),
    }
    if bucket_diagnostics:
        result["bucket_metrics"] = finalize_bucket_metrics(progress["bucket_metrics"])
        result["candidate_trace"] = candidate_trace
    if legal_path_constraint and result["metrics"]["valid_id_rate"] != 1.0:
        raise TigerEvalError("合法路径约束运行仍生成了无法映射的候选")
    progress["status"] = "completed"
    atomic_json(progress_path, progress)
    atomic_json(run_dir / "result.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_results_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        metrics = result["metrics"]
        row = {
            "checkpoint": Path(result["config"]["checkpoint"]).name,
            "epoch": result["config"]["epoch"],
            "validation_loss": result["config"]["validation_loss"],
            "hr@1": metrics["hr@1"],
            "hr@3": metrics["hr@3"],
            "hr@5": metrics["hr@5"],
            "hr@10": metrics["hr@10"],
            "ndcg@1": metrics["ndcg@1"],
            "ndcg@3": metrics["ndcg@3"],
            "ndcg@5": metrics["ndcg@5"],
            "ndcg@10": metrics["ndcg@10"],
            "invalid_id_rate": metrics["invalid_id_rate"],
            "samples_per_second": result["performance"]["samples_per_second"],
        }
        bucket_metrics = result.get("bucket_metrics")
        if isinstance(bucket_metrics, Mapping):
            for name in ("raw_slot_bucket_hr", "unique_bucket_hr"):
                for k in (1, 3, 5, 10):
                    row[f"{name}@{k}"] = bucket_metrics[f"{name}@{k}"]
        rows.append(row)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    try:
        expected_checkpoint_count = 1 if args.final_checkpoint_only else 3
        if len(args.checkpoints) != expected_checkpoint_count:
            raise TigerEvalError(
                "--final-checkpoint-only 需要 1 个 checkpoint；默认正式对比需要 3 个"
            )
        if args.num_beams != 10:
            raise TigerEvalError("固定一万条 checkpoint 对比的 Beam 必须为 10")
        if args.chunk_size <= 0:
            raise TigerEvalError("--chunk-size 必须为正整数")
        if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
            raise TigerEvalError("--smoke-limit 必须位于 [1, 100]")
        valid_file = resolve(args.valid_file)
        reference = resolve(args.reference_validation_subset)
        checkpoints = [resolve(path) for path in args.checkpoints]
        tokenizer_path = resolve(args.tokenizer)
        identifier_dir = resolve(args.identifier_dir)
        output_dir = resolve(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _, source_rows, source_hash = validate_split_manifest(
            valid_file,
            split="valid",
            verify_hash=not args.skip_data_hash,
        )
        subset = build_reference_aligned_validation_subset(
            valid_file,
            reference,
            output_dir,
            source_rows=source_rows,
            source_sha256=source_hash,
        )
        if subset.row_count != 10_000:
            raise TigerEvalError("固定 Validation 子集必须为 10,000 条")
        records = load_records(subset.data_path)
        tokenizer, template = load_lf_tokenizer_and_template(
            tokenizer_path,
            project_root=PROJECT_ROOT,
        )
        index, index_metadata = load_tiger_id_index(identifier_dir)
        tokens, token_metadata = load_tiger_token_ids(
            tokenizer_path,
            tokenizer,
            token_capacities=index.token_capacities,
            mapping_filename=args.token_mapping_filename,
        )
        metadata = validate_checkpoints(
            checkpoints,
            expected_steps=args.expected_checkpoint_steps,
            expected_epochs=args.expected_checkpoint_epochs,
            tokenizer_path=tokenizer_path,
            token_mapping_filename=args.token_mapping_filename,
        )

        prompt_checks = []
        for record in records[:100]:
            example = encode_tiger_record(
                record,
                tokenizer=tokenizer,
                template=template,
                cutoff_len=args.cutoff_len,
                token_capacities=index.token_capacities,
            )
            row = index.lookup(example.target_codes)
            if row < 0 or index.poi_id(row) != example.target_poi_id:
                raise TigerEvalError(f"Prompt 样本目标映射失败：{example.sample_id}")
            prompt_checks.append(
                {
                    "sample_id": example.sample_id,
                    "prompt_tokens": len(example.prompt_ids),
                    "target_codes": list(example.target_codes),
                }
            )
        atomic_json(
            output_dir / "prompt_validation.json",
            {
                "status": "passed",
                "rows": len(prompt_checks),
                "data_sha256": subset.sha256,
                "checks": prompt_checks,
            },
        )
        if args.preflight_only:
            payload = {
                "status": "preflight_passed",
                "evaluation_subset": subset.manifest,
                "tokenizer": token_metadata,
                "identifier": index_metadata,
                "checkpoints": metadata,
                "prompt_validation_rows": len(prompt_checks),
            }
            atomic_json(output_dir / "preflight.json", payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        results = [
            evaluate_checkpoint(
                metadata=item,
                records=records,
                subset_path=subset.data_path,
                subset_sha256=subset.sha256,
                tokenizer=tokenizer,
                template=template,
                tokens=tokens,
                token_metadata=token_metadata,
                index=index,
                index_metadata=index_metadata,
                output_dir=output_dir,
                num_beams=args.num_beams,
                initial_batch_size=args.per_device_eval_batch_size,
                chunk_size=min(args.chunk_size, args.smoke_limit or args.chunk_size),
                cutoff_len=args.cutoff_len,
                smoke_limit=args.smoke_limit,
                legal_path_constraint=args.legal_path_constraint,
                bucket_diagnostics=args.bucket_diagnostics,
            )
            for item in metadata
        ]
        if args.smoke_limit is not None:
            print(
                json.dumps(
                    {"status": "smoke_completed", "results": results},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        best = max(
            results,
            key=lambda item: (
                float(item["metrics"]["ndcg@10"]),
                float(item["metrics"]["hr@10"]),
                float(item["metrics"]["hr@1"]),
                -float(item["config"]["validation_loss"]),
            ),
        )
        payload = {
            "schema_version": "tiger-valid-checkpoint-results-v1",
            "status": "completed",
            "evaluation_protocol": {
                "decoding": (
                    "beam search constrained to frozen-corpus legal identifier paths"
                    if args.legal_path_constraint
                    else "unconstrained beam search"
                ),
                "beam_size": 10,
                "identifier_lookup": "frozen four-code TIGER ID to POI table",
                "invalid_id_handling": (
                    "not applicable; all generated paths must map to corpus POIs"
                    if args.legal_path_constraint
                    else "kept at original beam rank and counted as miss"
                ),
                "legal_path_constraint": args.legal_path_constraint,
                "paper_protocol": not args.legal_path_constraint,
                "geographic_pruning": False,
                "bucket_diagnostics": args.bucket_diagnostics,
            },
            "evaluation_subset": subset.manifest,
            "results": results,
            "best_checkpoint": Path(best["config"]["checkpoint"]).name,
        }
        atomic_json(output_dir / "valid_checkpoint_results.json", payload)
        write_results_csv(output_dir / "valid_checkpoint_results.csv", results)
        with (output_dir / "validation_error_cases.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            progress = load_json(
                Path(best["run_dir"]) / "progress.json", "best progress"
            )
            for item in progress["error_cases"]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except (TigerEvalError, OSError, ValueError, KeyError) as error:
        print(f"TIGER 评测失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

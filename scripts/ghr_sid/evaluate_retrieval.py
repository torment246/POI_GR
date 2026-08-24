#!/usr/bin/env python3
"""Evaluate variable-length GHR-SID checkpoints on the frozen Validation 10k."""

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

from poi_gr.methods.ghr_sid.eval import (  # noqa: E402
    GhrEvalError,
    GhrIdIndex,
    GhrLegalPathConstraint,
    GhrPrefixIndex,
    GhrTokenIds,
    build_ghr_compact_trie,
    encode_ghr_record,
    load_ghr_id_index,
    load_ghr_token_ids,
    parse_generated_base_bucket,
    parse_generated_candidate,
)
from poi_gr.methods.tiger.eval import (  # noqa: E402
    TigerCandidate,
    atomic_json,
    canonical_sha256,
    empty_bucket_metrics,
    empty_metrics,
    finalize_bucket_metrics,
    finalize_metrics,
    merge_bucket_metrics,
    merge_metrics,
    update_bucket_metrics,
    update_metrics,
)
from poi_gr.pid.trie import sha256_file  # noqa: E402
from poi_gr.pid.trie import CompactPidTrie  # noqa: E402
from poi_gr.sft.evaluation import (  # noqa: E402
    build_reference_aligned_validation_subset,
    load_generation_model,
    load_lf_tokenizer_and_template,
    validate_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "默认按TIGER主口径对变长GHR-SID执行无约束Beam=10评测；也可显式"
            "启用仅用于诊断的冻结语料库合法路径约束。"
        )
    )
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--reference-validation-subset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-checkpoint-steps", type=int, nargs="+", required=True)
    parser.add_argument(
        "--expected-checkpoint-epochs", type=float, nargs="+", default=(1, 2, 3)
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--token-mapping-filename", default="poi_token_mapping.json"
    )
    parser.add_argument("--identifier-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        choices=(32, 16, 8, 4, 2),
        default=16,
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--skip-data-hash", action="store_true")
    parser.add_argument(
        "--legal-path-constraint",
        action="store_true",
        help="诊断模式：Beam每一步只允许沿冻结全库GHR identifier的真实Prefix扩展。",
    )
    parser.add_argument(
        "--final-checkpoint-only",
        action="store_true",
        help="专项诊断只评测一个已冻结的最终checkpoint；默认仍要求三轮。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只核验固定子集、Tokenizer、identifier、checkpoint和Prompt。",
    )
    parser.add_argument(
        "--bucket-diagnostics",
        action="store_true",
        help="同时统计自由生成 Beam 中前三层唯一可展开 Bucket-HR。",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise GhrEvalError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GhrEvalError(f"{name} JSON 非法") from error
    if not isinstance(value, dict):
        raise GhrEvalError(f"{name} 必须是 object")
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
        raise GhrEvalError("Checkpoint数量必须与expected steps一致且非空")
    if len(expected_epochs) != len(expected_steps):
        raise GhrEvalError("expected epochs数量必须与expected steps一致")
    mapping = load_json(
        tokenizer_path / token_mapping_filename, "GHR Token mapping"
    )
    token_mapping = mapping.get("tokens")
    if not isinstance(token_mapping, dict):
        raise GhrEvalError("GHR Token mapping缺少tokens")
    expected_vocab = mapping.get("new_vocab_size")
    results: list[dict[str, Any]] = []
    for checkpoint, expected_step, expected_epoch in zip(
        checkpoints, expected_steps, expected_epochs
    ):
        checkpoint = checkpoint.resolve()
        if checkpoint.name != f"checkpoint-{expected_step}":
            raise GhrEvalError(f"Checkpoint顺序或步数错误：{checkpoint.name}")
        required = (
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "added_tokens.json",
            "trainer_state.json",
        )
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise GhrEvalError(f"{checkpoint.name}缺少文件：{', '.join(missing)}")
        state = load_json(checkpoint / "trainer_state.json", "trainer_state")
        actual_epoch = float(state.get("epoch", -1))
        if abs(actual_epoch - expected_epoch) > 0.001:
            raise GhrEvalError(
                f"{checkpoint.name} epoch {actual_epoch} != {expected_epoch}"
            )
        config = load_json(checkpoint / "config.json", "checkpoint config")
        if config.get("vocab_size") != expected_vocab:
            raise GhrEvalError(f"{checkpoint.name} vocab_size不一致")
        added = load_json(checkpoint / "added_tokens.json", "added tokens")
        for token, token_id in token_mapping.items():
            if added.get(token) != token_id:
                raise GhrEvalError(f"{checkpoint.name} Token ID不一致：{token}")
        eval_loss = next(
            (
                float(item["eval_loss"])
                for item in reversed(state.get("log_history", []))
                if item.get("step") == expected_step and "eval_loss" in item
            ),
            None,
        )
        if eval_loss is None:
            raise GhrEvalError(f"{checkpoint.name}缺少完整Validation Loss")
        results.append(
            {
                "path": str(checkpoint),
                "name": checkpoint.name,
                "step": expected_step,
                "epoch": actual_epoch,
                "validation_loss": eval_loss,
                "model_sha256": sha256_file(checkpoint / "model.safetensors"),
                "tokenizer_json_sha256": sha256_file(
                    checkpoint / "tokenizer.json"
                ),
            }
        )
    if len({item["tokenizer_json_sha256"] for item in results}) != 1:
        raise GhrEvalError("GHR checkpoint的Tokenizer不一致")
    return results


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise GhrEvalError(
                    f"评测子集第{line_number}行JSON非法"
                ) from error
            if not isinstance(value, dict):
                raise GhrEvalError(f"评测子集第{line_number}行不是object")
            records.append(value)
    return records


def pad_prompts(
    prompts: Sequence[Sequence[int]], pad_token_id: int, device: Any
) -> tuple[Any, Any]:
    import torch

    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full(
        (len(prompts), width), pad_token_id, dtype=torch.long, device=device
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
    tokens: GhrTokenIds,
    index: GhrIdIndex,
    checkpoint_name: str,
    batch_size: int,
    cutoff_len: int,
    num_beams: int,
    existing_error_count: int,
    legal_path_constraint: bool,
    legal_path_trie: CompactPidTrie | None,
    bucket_diagnostics: bool,
    prefix_index: GhrPrefixIndex | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], float, int]:
    import torch

    examples = [
        encode_ghr_record(
            record,
            tokenizer=tokenizer,
            template=template,
            tokens=tokens,
            cutoff_len=cutoff_len,
        )
        for record in records
    ]
    target_rows: list[int] = []
    for example in examples:
        row = index.lookup(example.target_codes)
        if row < 0 or index.poi_id(row) != example.target_poi_id:
            raise GhrEvalError(f"Target identifier映射不一致：{example.sample_id}")
        target_rows.append(row)

    metrics = empty_metrics()
    bucket_metrics = empty_bucket_metrics() if bucket_diagnostics else None
    errors: list[dict[str, Any]] = []
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
            if legal_path_trie is None:
                raise GhrEvalError("合法路径约束缺少GHR Trie")
            generation_kwargs["prefix_allowed_tokens_fn"] = GhrLegalPathConstraint(
                trie=legal_path_trie,
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
                max_new_tokens=tokens.maximum_sequence_length,
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
            raise GhrEvalError("Beam Search缺少sequences_scores")
        raw_scores = generated.sequences_scores.float().detach().cpu().tolist()
        expected = len(batch_examples) * num_beams
        if len(raw_sequences) != expected or len(raw_scores) != expected:
            raise GhrEvalError("Beam Search返回候选数量不一致")
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
                    sequence, score, tokens=tokens, index=index
                )
                for sequence, score in ranked
            ]
            if bucket_metrics is not None:
                if prefix_index is None:
                    raise GhrEvalError("Bucket 诊断缺少 GHR prefix index")
                bucket_candidates = [
                    TigerCandidate(
                        codes=None,
                        poi_row=candidate.poi_row,
                        score=candidate.score,
                        error=candidate.error,
                        base_bucket=parse_generated_base_bucket(
                            sequence,
                            tokens=tokens,
                        ),
                    )
                    for (sequence, _score), candidate in zip(ranked, candidates)
                ]
                update_bucket_metrics(
                    bucket_metrics,
                    target_codes=(*example.target_codes[:3], 0),
                    candidates=bucket_candidates,
                    index=prefix_index,  # type: ignore[arg-type]
                )
            target_rank = update_metrics(
                metrics, target_row=target_row, candidates=candidates
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
    return metrics, bucket_metrics, errors, elapsed, peak_memory


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
    tokens: GhrTokenIds,
    token_metadata: Mapping[str, Any],
    index: GhrIdIndex,
    index_metadata: Mapping[str, Any],
    output_dir: Path,
    num_beams: int,
    initial_batch_size: int,
    chunk_size: int,
    cutoff_len: int,
    smoke_limit: int | None,
    legal_path_constraint: bool,
    legal_path_trie: CompactPidTrie | None,
    legal_path_trie_metadata: Mapping[str, Any] | None,
    bucket_diagnostics: bool,
    prefix_index: GhrPrefixIndex | None,
) -> dict[str, Any]:
    import torch

    checkpoint = Path(metadata["path"])
    target_rows = smoke_limit or len(records)
    config = {
        "schema_version": "ghr-generative-eval-v1",
        "decoding": (
            "frozen_corpus_legal_path_constrained_beam_search"
            if legal_path_constraint
            else "unconstrained_beam_search_then_frozen_id_lookup"
        ),
        "comparison_protocol": (
            "fixed Validation 10k legal-path diagnostic"
            if legal_path_constraint
            else "TIGER fixed Validation 10k main protocol"
        ),
        "is_paper_protocol": not legal_path_constraint,
        "invalid_ids_keep_original_rank": not legal_path_constraint,
        "legal_path_constraint": legal_path_constraint,
        "bucket_diagnostics": bucket_diagnostics,
        "legal_path_trie": legal_path_trie_metadata,
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
        "max_new_tokens": tokens.maximum_sequence_length,
        "cutoff_len": cutoff_len,
        "chunk_size": chunk_size,
        "initial_batch_size": initial_batch_size,
        "smoke_limit": smoke_limit,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "method_code_sha256": sha256_file(
            PROJECT_ROOT / "src" / "poi_gr" / "methods" / "ghr_sid" / "eval.py"
        ),
    }
    run_name = f"valid_{checkpoint.name}_beam{num_beams}"
    if legal_path_constraint:
        run_name += "_legalpath"
    run_name += f"_smoke{smoke_limit}" if smoke_limit is not None else f"_subset{len(records)}"
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    signature = canonical_sha256(config)
    if progress_path.is_file():
        progress = load_json(progress_path, "GHR progress")
        if progress.get("signature") != signature:
            raise GhrEvalError(f"已有运行配置变化：{run_dir}")
    else:
        progress = {
            "schema_version": "ghr-eval-progress-v1",
            "status": "running",
            "signature": signature,
            "config": config,
            "next_line": 0,
            "metrics": empty_metrics(),
            "bucket_metrics": (
                empty_bucket_metrics() if bucket_diagnostics else None
            ),
            "inference_seconds": 0.0,
            "peak_memory_bytes": 0,
            "actual_batch_size": initial_batch_size,
            "error_cases": [],
        }
        atomic_json(progress_path, progress)

    model = load_generation_model(checkpoint, expected_vocab_size=len(tokenizer))
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
                    legal_path_trie=legal_path_trie,
                    bucket_diagnostics=bucket_diagnostics,
                    prefix_index=prefix_index,
                )
                break
            except RuntimeError as error:
                if not is_cuda_oom(error):
                    raise
                torch.cuda.empty_cache()
                smaller = [value for value in allowed if value < batch_size]
                if not smaller:
                    raise GhrEvalError("GHR batch_size=2仍然OOM") from error
                batch_size = smaller[0]
                print(
                    f"CUDA OOM，当前分片改用batch_size={batch_size}",
                    file=sys.stderr,
                )
        merge_metrics(progress["metrics"], chunk_metrics)
        if bucket_diagnostics:
            if chunk_bucket_metrics is None or progress["bucket_metrics"] is None:
                raise GhrEvalError("Bucket 诊断分片指标缺失")
            merge_bucket_metrics(progress["bucket_metrics"], chunk_bucket_metrics)
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
    result = {
        "schema_version": "ghr-eval-result-v1",
        "status": "completed",
        "config": config,
        "metrics": finalize_metrics(progress["metrics"]),
        "bucket_metrics": (
            finalize_bucket_metrics(progress["bucket_metrics"])
            if bucket_diagnostics
            else None
        ),
        "performance": {
            "inference_seconds": progress["inference_seconds"],
            "samples_per_second": target_rows / progress["inference_seconds"],
            "peak_memory_bytes": progress["peak_memory_bytes"],
            "actual_batch_size": progress["actual_batch_size"],
        },
        "run_dir": str(run_dir),
    }
    if legal_path_constraint and result["metrics"]["valid_id_rate"] != 1.0:
        raise GhrEvalError("合法路径约束运行仍生成了无法映射的候选")
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
        rows.append(
            {
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
                **(
                    {
                        f"unique_bucket_hr@{k}": result["bucket_metrics"][
                            f"unique_bucket_hr@{k}"
                        ]
                        for k in (1, 3, 5, 10)
                    }
                    if result.get("bucket_metrics") is not None
                    else {}
                ),
            }
        )
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
            raise GhrEvalError(
                "--final-checkpoint-only需要1个checkpoint；默认正式对比需要3个"
            )
        if args.num_beams != 10:
            raise GhrEvalError("固定10k对比的Beam必须为10")
        if args.chunk_size <= 0:
            raise GhrEvalError("--chunk-size必须为正整数")
        if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
            raise GhrEvalError("--smoke-limit必须位于[1,100]")

        valid_file = resolve(args.valid_file)
        reference = resolve(args.reference_validation_subset)
        checkpoints = [resolve(path) for path in args.checkpoints]
        tokenizer_path = resolve(args.tokenizer)
        identifier_dir = resolve(args.identifier_dir)
        output_dir = resolve(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _, source_rows, source_sha256 = validate_split_manifest(
            valid_file,
            split="valid",
            verify_hash=not args.skip_data_hash,
        )
        subset = build_reference_aligned_validation_subset(
            valid_file,
            reference,
            output_dir,
            source_rows=source_rows,
            source_sha256=source_sha256,
        )
        if subset.row_count != 10_000:
            raise GhrEvalError("固定Validation子集必须为10,000条")
        records = load_records(subset.data_path)
        tokenizer, template = load_lf_tokenizer_and_template(
            tokenizer_path, project_root=PROJECT_ROOT
        )
        index, index_metadata = load_ghr_id_index(identifier_dir)
        prefix_index = GhrPrefixIndex(index.codes) if args.bucket_diagnostics else None
        tokens, token_metadata = load_ghr_token_ids(
            tokenizer_path,
            tokenizer,
            identifier_dir=identifier_dir,
            token_source_path=valid_file.parent / "special_tokens.json",
            mapping_filename=args.token_mapping_filename,
        )
        if index.maximum_identifier_length != tokens.maximum_identifier_length:
            raise GhrEvalError("GHR index与Tokenizer最大identifier长度不一致")
        checkpoint_metadata = validate_checkpoints(
            checkpoints,
            expected_steps=args.expected_checkpoint_steps,
            expected_epochs=args.expected_checkpoint_epochs,
            tokenizer_path=tokenizer_path,
            token_mapping_filename=args.token_mapping_filename,
        )
        legal_path_trie: CompactPidTrie | None = None
        legal_path_trie_metadata: dict[str, Any] | None = None
        if args.legal_path_constraint:
            trie_started = time.perf_counter()
            legal_path_trie = build_ghr_compact_trie(
                index,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            legal_path_trie_metadata = {
                "representation": "in_memory_compact_csr",
                "node_count": legal_path_trie.node_count,
                "edge_count": legal_path_trie.edge_count,
                "leaf_count": legal_path_trie.leaf_count,
                "memory_bytes": legal_path_trie.memory_bytes,
                "build_seconds": time.perf_counter() - trie_started,
            }

        target_mismatch = 0
        for record in records:
            example = encode_ghr_record(
                record,
                tokenizer=tokenizer,
                template=template,
                tokens=tokens,
                cutoff_len=args.cutoff_len,
            )
            row = index.lookup(example.target_codes)
            target_mismatch += int(
                row < 0 or index.poi_id(row) != example.target_poi_id
            )
        if target_mismatch:
            raise GhrEvalError(f"固定10k目标映射错位：{target_mismatch}")
        preflight = {
            "schema_version": "ghr-eval-preflight-v1",
            "status": "completed",
            "evaluation_subset": subset.manifest,
            "tokenizer": token_metadata,
            "identifier": index_metadata,
            "checkpoints": checkpoint_metadata,
            "target_poi_mismatch_count": target_mismatch,
            "protocol": {
                "num_beams": 10,
                "decoding": (
                    "beam search constrained to frozen-corpus legal GHR paths"
                    if args.legal_path_constraint
                    else "unconstrained beam search"
                ),
                "invalid_id_handling": (
                    "not applicable; all generated paths must map to corpus POIs"
                    if args.legal_path_constraint
                    else "kept at original beam rank and counted as miss"
                ),
                "legal_path_constraint": args.legal_path_constraint,
                "bucket_diagnostics": args.bucket_diagnostics,
                "paper_protocol": not args.legal_path_constraint,
                "geographic_pruning": False,
            },
            "legal_path_trie": legal_path_trie_metadata,
        }
        atomic_json(output_dir / "preflight.json", preflight)
        if args.preflight_only:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 0

        results: list[dict[str, Any]] = []
        for metadata in checkpoint_metadata:
            results.append(
                evaluate_checkpoint(
                    metadata=metadata,
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
                    chunk_size=args.chunk_size,
                    cutoff_len=args.cutoff_len,
                    smoke_limit=args.smoke_limit,
                    legal_path_constraint=args.legal_path_constraint,
                    legal_path_trie=legal_path_trie,
                    legal_path_trie_metadata=legal_path_trie_metadata,
                    bucket_diagnostics=args.bucket_diagnostics,
                    prefix_index=prefix_index,
                )
            )
        if args.smoke_limit is not None:
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0

        best = max(
            results,
            key=lambda item: (
                item["metrics"]["ndcg@10"],
                item["metrics"]["hr@1"],
                -item["performance"]["inference_seconds"],
            ),
        )
        summary = {
            "schema_version": "ghr-checkpoint-eval-summary-v1",
            "status": "completed",
            "evaluation_subset": subset.manifest,
            "evaluation_protocol": preflight["protocol"],
            "results": results,
            "best_checkpoint": Path(best["config"]["checkpoint"]).name,
        }
        atomic_json(output_dir / "valid_checkpoint_results.json", summary)
        write_results_csv(output_dir / "valid_checkpoint_results.csv", results)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (GhrEvalError, OSError, ValueError) as error:
        print(f"GHR检索评测失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

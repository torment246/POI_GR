#!/usr/bin/env python3
"""Evaluate TIGER with unconstrained beam search and frozen ID lookup."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.generative_eval import (  # noqa: E402
    build_reference_aligned_validation_subset,
    load_generation_model,
    load_lf_tokenizer_and_template,
    validate_split_manifest,
)
from poi_gr.methods.tiger_eval import (  # noqa: E402
    TigerEvalError,
    TigerIdIndex,
    TigerTokenIds,
    atomic_json,
    canonical_sha256,
    empty_metrics,
    encode_tiger_record,
    finalize_metrics,
    load_tiger_id_index,
    load_tiger_token_ids,
    merge_metrics,
    parse_generated_candidate,
    update_metrics,
)
from poi_gr.pid_trie import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按 TIGER 论文使用普通 Beam Search 生成四层 identifier；无 Trie、无地理剪枝，"
            "无效 ID 保留在原始候选排名中。"
        )
    )
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--reference-validation-subset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=3, required=True)
    parser.add_argument("--expected-checkpoint-steps", type=int, nargs=3, required=True)
    parser.add_argument("--expected-checkpoint-epochs", type=float, nargs=3, default=(1, 2, 3))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--identifier-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--per-device-eval-batch-size", type=int, choices=(32, 16, 8, 4, 2), default=16)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--skip-data-hash", action="store_true")
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
) -> list[dict[str, Any]]:
    mapping_payload = load_json(
        tokenizer_path / "tiger_token_mapping.json",
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
                raise TigerEvalError(f"评测子集第 {line_number} 行 JSON 非法") from error
            if not isinstance(value, dict):
                raise TigerEvalError(f"评测子集第 {line_number} 行不是 object")
            records.append(value)
    return records


def pad_prompts(prompts: Sequence[Sequence[int]], pad_token_id: int, device: Any) -> tuple[Any, Any]:
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
) -> tuple[dict[str, Any], list[dict[str, Any]], float, int]:
    import torch

    examples = [
        encode_tiger_record(
            record,
            tokenizer=tokenizer,
            template=template,
            cutoff_len=cutoff_len,
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
    errors: list[dict[str, Any]] = []
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
                key=lambda item: (-float(item[1]), tuple(int(value) for value in item[0])),
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
                            list(candidate.codes) if candidate.codes is not None else None
                            for candidate in candidates
                        ],
                        "candidate_errors": [candidate.error for candidate in candidates],
                        "checkpoint": checkpoint_name,
                    }
                )
        del generated, input_ids, attention_mask
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(model.device))
    return metrics, errors, elapsed, peak_memory


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
) -> dict[str, Any]:
    import torch

    checkpoint = Path(metadata["path"])
    target_rows = smoke_limit or len(records)
    config = {
        "schema_version": "tiger-generative-eval-v1",
        "paper_decoding": "unconstrained_beam_search_then_frozen_id_lookup",
        "invalid_ids_keep_original_rank": True,
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
            PROJECT_ROOT / "src" / "poi_gr" / "methods" / "tiger_eval.py"
        ),
    }
    run_name = f"valid_{checkpoint.name}_beam{num_beams}"
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
    else:
        progress = {
            "schema_version": "tiger-eval-progress-v1",
            "status": "running",
            "signature": signature,
            "config": config,
            "next_line": 0,
            "metrics": empty_metrics(),
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
    allowed = [
        value for value in (32, 16, 8, 4, 2) if value <= initial_batch_size
    ]
    batch_size = min(int(progress["actual_batch_size"]), allowed[0])
    while int(progress["next_line"]) < target_rows:
        start = int(progress["next_line"])
        stop = min(start + chunk_size, target_rows)
        while True:
            try:
                chunk_metrics, errors, seconds, peak_memory = evaluate_chunk(
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
                print(f"CUDA OOM，当前分片改用 batch_size={batch_size}", file=sys.stderr)
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
        tokens, token_metadata = load_tiger_token_ids(tokenizer_path, tokenizer)
        index, index_metadata = load_tiger_id_index(identifier_dir)
        metadata = validate_checkpoints(
            checkpoints,
            expected_steps=args.expected_checkpoint_steps,
            expected_epochs=args.expected_checkpoint_epochs,
            tokenizer_path=tokenizer_path,
        )

        prompt_checks = []
        for record in records[:100]:
            example = encode_tiger_record(
                record,
                tokenizer=tokenizer,
                template=template,
                cutoff_len=args.cutoff_len,
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
            )
            for item in metadata
        ]
        if args.smoke_limit is not None:
            print(json.dumps({"status": "smoke_completed", "results": results}, ensure_ascii=False, indent=2))
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
            "paper_protocol": {
                "decoding": "unconstrained beam search",
                "beam_size": 10,
                "identifier_lookup": "frozen four-code TIGER ID to POI table",
                "invalid_id_handling": "kept at original beam rank and counted as miss",
                "trie": False,
                "geographic_pruning": False,
            },
            "evaluation_subset": subset.manifest,
            "results": results,
            "best_checkpoint": Path(best["config"]["checkpoint"]).name,
        }
        atomic_json(output_dir / "valid_checkpoint_results.json", payload)
        write_results_csv(output_dir / "valid_checkpoint_results.csv", results)
        with (output_dir / "validation_error_cases.jsonl").open("w", encoding="utf-8") as handle:
            progress = load_json(Path(best["run_dir"]) / "progress.json", "best progress")
            for item in progress["error_cases"]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except (TigerEvalError, OSError, ValueError, KeyError) as error:
        print(f"TIGER 评测失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

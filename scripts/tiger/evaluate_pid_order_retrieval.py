#!/usr/bin/env python3
"""Evaluate paired GID-first/SID-first variable-length PID SFT checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger.eval import (  # noqa: E402
    TigerCandidate,
    TigerEvalError,
    atomic_json,
    canonical_sha256,
    empty_metrics,
    finalize_metrics,
    merge_metrics,
    update_metrics,
)
from poi_gr.pid.trie import (  # noqa: E402
    CompactPidTrie,
    PidTokenIds,
    load_pid_token_ids,
    sha256_file,
)
from poi_gr.sft.evaluation import (  # noqa: E402
    build_reference_aligned_validation_subset,
    encode_prompt_like_training,
    load_aligned_poi_ids,
    load_generation_model,
    load_lf_tokenizer_and_template,
    validate_checkpoints,
)


PID_ORDERS = ("gid_sid", "sid_gid")
DECODE_MODES = ("unconstrained", "constrained")


@dataclass(frozen=True)
class PidOrderTokens:
    target_open: int
    target_close: int
    pid: PidTokenIds

    @property
    def eos(self) -> int:
        return self.pid.eos

    @property
    def max_new_tokens(self) -> int:
        # <TARGET_POI> + nine base tokens + optional D + close + EOS.
        return 13


@dataclass(frozen=True)
class PidOrderExample:
    sample_id: str
    target_poi_id: str
    target_content: str
    ordered_pid_tokens: tuple[int, ...]
    canonical_pid_tokens: tuple[int, ...]
    prompt_ids: tuple[int, ...]
    requires_dedup: bool


class PidOrderLegalPathConstraint:
    """Add wrappers around one compact internal-PID Trie during generation."""

    def __init__(
        self,
        *,
        trie: CompactPidTrie,
        tokens: PidOrderTokens,
        prompt_width: int,
    ) -> None:
        if prompt_width <= 0:
            raise TigerEvalError("prompt_width 必须为正整数")
        self.trie = trie
        self.tokens = tokens
        self.prompt_width = prompt_width

    def __call__(self, _batch_id: int, input_ids: Any) -> list[int]:
        generated = input_ids[self.prompt_width :]
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        values = [int(value) for value in generated]
        if not values:
            return [self.tokens.target_open]
        if values[0] != self.tokens.target_open:
            raise TigerEvalError("合法路径 PID 必须以 <TARGET_POI> 开始")
        tail = values[1:]
        if tail and tail[-1] == self.tokens.eos:
            return [self.tokens.eos]
        if self.tokens.target_close in tail:
            if tail[-1] != self.tokens.target_close:
                raise TigerEvalError("</TARGET_POI> 后出现了非 EOS Token")
            internal = tail[:-1]
            if self.trie.lookup((*internal, self.tokens.eos)) < 0:
                raise TigerEvalError("闭合的 PID 不在全库 Trie")
            return [self.tokens.eos]
        node = self.trie.traverse(tail)
        if node < 0:
            raise TigerEvalError("生成序列离开 PID Trie")
        children = self.trie.children(node).astype(int).tolist()
        if children == [self.tokens.eos]:
            return [self.tokens.target_close]
        if self.tokens.eos in children:
            raise TigerEvalError("PID Trie 叶子与分支混合")
        if not children:
            raise TigerEvalError("非叶节点没有合法子节点")
        return children


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="固定同一 Validation 10k，评测 GID-first/SID-first epoch 3。"
    )
    parser.add_argument("--pid-order", choices=PID_ORDERS, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--reference-validation-subset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=8967)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--trie-dir", type=Path, required=True)
    parser.add_argument("--pid-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decode-modes", choices=DECODE_MODES, nargs="+", default=DECODE_MODES
    )
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        choices=(64, 32, 16, 8, 4, 2),
        default=64,
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--preflight-only", action="store_true")
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
        raise TigerEvalError(f"{name} JSON 非法：{path}") from error
    if not isinstance(value, dict):
        raise TigerEvalError(f"{name} 必须是 JSON object")
    return value


def validate_pid_order_split(
    valid_file: Path,
    *,
    pid_order: str,
    verify_hash: bool,
) -> tuple[int, str, dict[str, Any]]:
    manifest = load_json(valid_file.parent / "manifest.json", "SFT manifest")
    if manifest.get("pid_order") != pid_order:
        raise TigerEvalError("SFT manifest 的 pid_order 与命令不一致")
    spec = manifest.get("outputs", {}).get("valid.jsonl")
    if not isinstance(spec, Mapping) or spec.get("rows") != 597_421:
        raise TigerEvalError("Validation 必须为冻结的 597,421 条")
    expected = (valid_file.parent / "valid.jsonl").resolve()
    if valid_file.resolve() != expected:
        raise TigerEvalError("Validation 文件路径与 manifest 不一致")
    digest = spec.get("sha256")
    if not isinstance(digest, str):
        raise TigerEvalError("Validation manifest 缺少 SHA256")
    if verify_hash and sha256_file(valid_file) != digest:
        raise TigerEvalError("Validation SHA256 与 manifest 不一致")
    return 597_421, digest, manifest


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise TigerEvalError(f"固定子集第 {line_number} 行 JSON 非法") from error
            if not isinstance(value, dict):
                raise TigerEvalError("固定子集每行必须是 object")
            records.append(value)
    return records


def load_order_tokens(tokenizer_path: Path, tokenizer: Any) -> tuple[PidOrderTokens, dict[str, Any]]:
    pid_tokens, metadata = load_pid_token_ids(tokenizer_path)
    mapping = load_json(tokenizer_path / "poi_token_mapping.json", "Token mapping")
    token_map = mapping.get("tokens")
    if not isinstance(token_map, Mapping):
        raise TigerEvalError("Token mapping 缺少 tokens")

    def atomic_id(token: str) -> int:
        expected = token_map.get(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if not isinstance(expected, int) or encoded != [expected]:
            raise TigerEvalError(f"结构 Token 不是稳定原子 Token：{token}")
        return expected

    values = PidOrderTokens(
        target_open=atomic_id("<TARGET_POI>"),
        target_close=atomic_id("</TARGET_POI>"),
        pid=pid_tokens,
    )
    return values, {
        **metadata,
        "target_open_id": values.target_open,
        "target_close_id": values.target_close,
    }


def _in_range(value: int, values: Sequence[int]) -> bool:
    return int(values[0]) <= int(value) <= int(values[-1])


def ordered_pid_is_valid(
    values: Sequence[int],
    *,
    pid_order: str,
    tokens: PidTokenIds,
) -> bool:
    if len(values) not in (9, 10):
        return False
    gid = tokens.gid
    sid = tokens.sid
    if pid_order == "gid_sid":
        valid = all(_in_range(values[index], gid) for index in range(6)) and all(
            _in_range(values[index + 6], sid[index]) for index in range(3)
        )
    elif pid_order == "sid_gid":
        valid = all(_in_range(values[index], sid[index]) for index in range(3)) and all(
            _in_range(values[index + 3], gid) for index in range(6)
        )
    else:
        raise TigerEvalError(f"不支持的 PID 顺序：{pid_order}")
    if not valid:
        return False
    return len(values) == 9 or _in_range(values[9], tokens.dedup)


def canonicalize_pid(values: Sequence[int], *, pid_order: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in values)
    if len(values) not in (9, 10):
        raise TigerEvalError("PID 长度必须为 9 或 10")
    if pid_order == "gid_sid":
        return values
    if pid_order == "sid_gid":
        return (*values[3:9], *values[:3], *values[9:])
    raise TigerEvalError(f"不支持的 PID 顺序：{pid_order}")


def encode_record(
    record: Mapping[str, Any],
    *,
    pid_order: str,
    tokenizer: Any,
    template: Any,
    tokens: PidOrderTokens,
    cutoff_len: int,
) -> PidOrderExample:
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise TigerEvalError("评测 Messages 必须严格为 user + assistant")
    if record.get("split") != "valid" or record.get("pid_order") != pid_order:
        raise TigerEvalError("评测样本 split/pid_order 不一致")
    user_content = messages[0].get("content")
    target_content = messages[1].get("content")
    if not isinstance(user_content, str) or not isinstance(target_content, str):
        raise TigerEvalError("Messages content 必须是字符串")
    requires_dedup = record.get("requires_dedup")
    if not isinstance(requires_dedup, bool):
        raise TigerEvalError("requires_dedup 必须为 bool")
    prompt_ids, target_ids = encode_prompt_like_training(
        tokenizer=tokenizer,
        template=template,
        user_content=user_content,
        target_content=target_content,
        cutoff_len=cutoff_len,
    )
    expected_length = 12 if requires_dedup else 11
    if len(target_ids) != expected_length:
        raise TigerEvalError(
            f"带包装目标长度应为 {expected_length}，实际为 {len(target_ids)}"
        )
    if target_ids[0] != tokens.target_open or target_ids[-1] != tokens.target_close:
        raise TigerEvalError("目标缺少原子 <TARGET_POI> 包装")
    internal = tuple(int(value) for value in target_ids[1:-1])
    if not ordered_pid_is_valid(internal, pid_order=pid_order, tokens=tokens.pid):
        raise TigerEvalError("目标 PID 的层级 Token 或顺序非法")
    sample_id = record.get("sample_id")
    target_poi_id = record.get("target_poi_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise TigerEvalError("sample_id 非法")
    if not isinstance(target_poi_id, str) or not target_poi_id:
        raise TigerEvalError("target_poi_id 非法")
    return PidOrderExample(
        sample_id=sample_id,
        target_poi_id=target_poi_id,
        target_content=target_content,
        ordered_pid_tokens=internal,
        canonical_pid_tokens=canonicalize_pid(internal, pid_order=pid_order),
        prompt_ids=tuple(int(value) for value in prompt_ids),
        requires_dedup=requires_dedup,
    )


def parse_candidate(
    sequence: Sequence[int],
    score: float,
    *,
    pid_order: str,
    tokens: PidOrderTokens,
    trie: CompactPidTrie,
) -> TigerCandidate:
    values = [int(value) for value in sequence]
    try:
        eos_index = values.index(tokens.eos)
    except ValueError:
        return TigerCandidate(None, None, float(score), "missing_eos")
    path = values[: eos_index + 1]
    if (
        len(path) not in (12, 13)
        or path[0] != tokens.target_open
        or path[-2] != tokens.target_close
    ):
        return TigerCandidate(None, None, float(score), "invalid_wrapper_or_length")
    internal = tuple(path[1:-2])
    if not ordered_pid_is_valid(internal, pid_order=pid_order, tokens=tokens.pid):
        return TigerCandidate(None, None, float(score), "invalid_pid_structure")
    poi_row = trie.lookup((*internal, tokens.eos))
    if poi_row < 0:
        return TigerCandidate(None, None, float(score), "corpus_miss")
    canonical = canonicalize_pid(internal, pid_order=pid_order)
    return TigerCandidate(canonical, poi_row, float(score), None)


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
        values = torch.as_tensor(prompt, dtype=torch.long, device=device)
        input_ids[row, -len(prompt) :] = values
        attention_mask[row, -len(prompt) :] = 1
    return input_ids, attention_mask


def evaluate_chunk(
    records: Sequence[Mapping[str, Any]],
    *,
    pid_order: str,
    decode_mode: str,
    model: Any,
    tokenizer: Any,
    template: Any,
    tokens: PidOrderTokens,
    trie: CompactPidTrie,
    poi_ids: Sequence[str],
    checkpoint_name: str,
    batch_size: int,
    cutoff_len: int,
    num_beams: int,
    existing_error_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], float, int]:
    import torch

    examples = [
        encode_record(
            record,
            pid_order=pid_order,
            tokenizer=tokenizer,
            template=template,
            tokens=tokens,
            cutoff_len=cutoff_len,
        )
        for record in records
    ]
    target_rows: list[int] = []
    for example in examples:
        row = trie.lookup((*example.ordered_pid_tokens, tokens.eos))
        if row < 0 or poi_ids[row] != example.target_poi_id:
            raise TigerEvalError(f"目标 PID 映射不一致：{example.sample_id}")
        target_rows.append(row)

    metrics = empty_metrics()
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()
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
        if decode_mode == "constrained":
            generation_kwargs["prefix_allowed_tokens_fn"] = PidOrderLegalPathConstraint(
                trie=trie,
                tokens=tokens,
                prompt_width=prompt_width,
            )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=tokens.max_new_tokens,
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
            zip(batch_examples, batch_target_rows, strict=True)
        ):
            begin = batch_index * num_beams
            end = begin + num_beams
            ranked = sorted(
                zip(raw_sequences[begin:end], raw_scores[begin:end], strict=True),
                key=lambda item: (-float(item[1]), tuple(int(v) for v in item[0])),
            )
            candidates = [
                parse_candidate(
                    sequence,
                    score,
                    pid_order=pid_order,
                    tokens=tokens,
                    trie=trie,
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
                        "target_pid": example.target_content,
                        "candidate_poi_ids": [
                            poi_ids[candidate.poi_row]
                            if candidate.poi_row is not None
                            else None
                            for candidate in candidates
                        ],
                        "candidate_errors": [candidate.error for candidate in candidates],
                        "checkpoint": checkpoint_name,
                        "pid_order": pid_order,
                        "decode_mode": decode_mode,
                    }
                )
        del generated, input_ids, attention_mask
    return metrics, errors, time.perf_counter() - started, int(
        torch.cuda.max_memory_allocated(model.device)
    )


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message and "cuda" in message


def evaluate_mode(
    *,
    pid_order: str,
    decode_mode: str,
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    subset_path: Path,
    subset_sha256: str,
    subset_manifest_sha256: str,
    model: Any,
    tokenizer: Any,
    template: Any,
    tokens: PidOrderTokens,
    token_metadata: Mapping[str, Any],
    trie: CompactPidTrie,
    trie_dir: Path,
    poi_ids: Sequence[str],
    output_dir: Path,
    num_beams: int,
    initial_batch_size: int,
    chunk_size: int,
    cutoff_len: int,
    smoke_limit: int | None,
) -> dict[str, Any]:
    import torch

    target_rows = smoke_limit or len(records)
    checkpoint = Path(metadata["path"])
    trie_manifest = load_json(trie_dir / "trie_manifest.json", "Trie manifest")
    config = {
        "schema_version": "tiger-pid-order-eval-v1",
        "pid_order": pid_order,
        "decode_mode": decode_mode,
        "decoding": (
            "unconstrained_beam_search_then_frozen_pid_lookup"
            if decode_mode == "unconstrained"
            else "frozen_corpus_legal_path_constrained_beam_search"
        ),
        "is_paper_protocol": decode_mode == "unconstrained",
        "invalid_ids_keep_original_rank": True,
        "data_file": str(subset_path),
        "data_sha256": subset_sha256,
        "subset_manifest_sha256": subset_manifest_sha256,
        "rows": target_rows,
        "full_subset_rows": len(records),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": metadata["model_sha256"],
        "epoch": metadata["epoch"],
        "validation_loss": metadata["validation_loss"],
        "tokenizer": dict(token_metadata),
        "trie_dir": str(trie_dir),
        "trie_manifest_sha256": sha256_file(trie_dir / "trie_manifest.json"),
        "trie_load_memory_bytes": trie_manifest.get("load_memory_bytes"),
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
        "max_new_tokens": tokens.max_new_tokens,
        "cutoff_len": cutoff_len,
        "chunk_size": chunk_size,
        "initial_batch_size": initial_batch_size,
        "smoke_limit": smoke_limit,
        "evaluator_sha256": sha256_file(Path(__file__)),
    }
    run_name = f"{decode_mode}_epoch3_beam{num_beams}"
    run_name += f"_smoke{smoke_limit}" if smoke_limit is not None else "_fixed10k"
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    signature = canonical_sha256(config)
    progress: dict[str, Any] | None = None
    if progress_path.is_file():
        progress = load_json(progress_path, "PID order progress")
        if progress.get("signature") != signature:
            empty_stale_run = (
                progress.get("status") == "running"
                and progress.get("next_line") == 0
                and progress.get("metrics", {}).get("sample_count") == 0
            )
            if not empty_stale_run:
                raise TigerEvalError(f"已有运行配置变化：{run_dir}")
            progress = None
    if progress is None:
        progress = {
            "schema_version": "tiger-pid-order-progress-v1",
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

    allowed = [value for value in (64, 32, 16, 8, 4, 2) if value <= initial_batch_size]
    batch_size = min(int(progress["actual_batch_size"]), allowed[0])
    while int(progress["next_line"]) < target_rows:
        start = int(progress["next_line"])
        stop = min(start + chunk_size, target_rows)
        while True:
            try:
                chunk_metrics, errors, seconds, peak_memory = evaluate_chunk(
                    records[start:stop],
                    pid_order=pid_order,
                    decode_mode=decode_mode,
                    model=model,
                    tokenizer=tokenizer,
                    template=template,
                    tokens=tokens,
                    trie=trie,
                    poi_ids=poi_ids,
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
                    raise TigerEvalError("batch_size=2 仍然 CUDA OOM") from error
                batch_size = smaller[0]
                print(f"CUDA OOM，改用 batch_size={batch_size}", file=sys.stderr)
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
            f"[{pid_order}/{decode_mode}] {stop:,}/{target_rows:,} "
            f"({stop / target_rows:.2%})",
            file=sys.stderr,
            flush=True,
        )

    metrics = finalize_metrics(progress["metrics"])
    if decode_mode == "constrained" and metrics["valid_id_rate"] != 1.0:
        raise TigerEvalError("合法路径约束仍生成了无法映射的 PID")
    seconds = float(progress["inference_seconds"])
    result = {
        "schema_version": "tiger-pid-order-result-v1",
        "status": "completed",
        "config": config,
        "metrics": metrics,
        "performance": {
            "inference_seconds": seconds,
            "samples_per_second": target_rows / seconds if seconds else None,
            "peak_memory_bytes": progress["peak_memory_bytes"],
            "actual_batch_size": progress["actual_batch_size"],
        },
        "error_cases": progress["error_cases"],
        "run_dir": str(run_dir),
    }
    progress["status"] = "completed"
    atomic_json(progress_path, progress)
    atomic_json(run_dir / "result.json", result)
    return result


def write_summary(output_dir: Path, results: Sequence[Mapping[str, Any]]) -> None:
    modes = [str(result["config"]["decode_mode"]) for result in results]
    suffix = "" if set(modes) == set(DECODE_MODES) else f"_{modes[0]}"
    payload = {
        "schema_version": "tiger-pid-order-mode-comparison-v1",
        "status": "completed",
        "results": list(results),
    }
    atomic_json(output_dir / f"results{suffix}.json", payload)
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "pid_order": result["config"]["pid_order"],
                "decode_mode": result["config"]["decode_mode"],
                "epoch": result["config"]["epoch"],
                "validation_loss": result["config"]["validation_loss"],
                **{f"hr@{k}": metrics[f"hr@{k}"] for k in (1, 3, 5, 10)},
                **{f"ndcg@{k}": metrics[f"ndcg@{k}"] for k in (1, 3, 5, 10)},
                "valid_id_rate": metrics["valid_id_rate"],
                "samples_per_second": result["performance"]["samples_per_second"],
            }
        )
    temporary = output_dir / f".results{suffix}.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_dir / f"results{suffix}.csv")


def main() -> int:
    args = parse_args()
    try:
        if args.num_beams != 10:
            raise TigerEvalError("固定 10k 评测必须使用 Beam=10")
        if args.chunk_size <= 0:
            raise TigerEvalError("chunk-size 必须为正整数")
        if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
            raise TigerEvalError("smoke-limit 必须位于 [1,100]")
        valid_file = resolve(args.valid_file)
        reference = resolve(args.reference_validation_subset)
        checkpoint = resolve(args.checkpoint)
        tokenizer_path = resolve(args.tokenizer)
        trie_dir = resolve(args.trie_dir)
        mapping_path = resolve(args.pid_mapping)
        output_dir = resolve(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        source_rows, source_hash, _ = validate_pid_order_split(
            valid_file,
            pid_order=args.pid_order,
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
            raise TigerEvalError("参考对齐 Validation 子集必须恰好为 10,000 条")
        records = load_records(subset.data_path)
        tokenizer, template = load_lf_tokenizer_and_template(
            tokenizer_path,
            project_root=PROJECT_ROOT,
        )
        tokens, token_metadata = load_order_tokens(tokenizer_path, tokenizer)
        trie_manifest = load_json(trie_dir / "trie_manifest.json", "Trie manifest")
        if trie_manifest.get("path_definition", {}).get("pid_order") != args.pid_order:
            raise TigerEvalError("Trie 的 pid_order 与评测数据不一致")
        trie = CompactPidTrie.load(trie_dir, mmap=True)
        if trie.leaf_count != 2_337_178:
            raise TigerEvalError("全库 Trie 叶子数必须为 2,337,178")
        poi_ids = load_aligned_poi_ids(mapping_path, trie.leaf_count)
        metadata = validate_checkpoints(
            [checkpoint],
            tokenizer_path=tokenizer_path,
            expected_steps=[args.expected_step],
            expected_epochs=[3.0],
        )[0]

        prompt_lengths = []
        target_in_trie = 0
        for record in records[:100]:
            example = encode_record(
                record,
                pid_order=args.pid_order,
                tokenizer=tokenizer,
                template=template,
                tokens=tokens,
                cutoff_len=args.cutoff_len,
            )
            prompt_lengths.append(len(example.prompt_ids))
            row = trie.lookup((*example.ordered_pid_tokens, tokens.eos))
            target_in_trie += int(row >= 0 and poi_ids[row] == example.target_poi_id)
        if target_in_trie != 100:
            raise TigerEvalError("100 条 preflight 目标没有全部对齐全库 Trie")
        preflight = {
            "schema_version": "tiger-pid-order-preflight-v1",
            "status": "passed",
            "pid_order": args.pid_order,
            "checkpoint": metadata,
            "fixed_subset_rows": subset.row_count,
            "fixed_subset_sha256": subset.sha256,
            "fixed_subset_manifest_sha256": sha256_file(subset.manifest_path),
            "business_keys_sha256": subset.manifest["business_keys_sha256"],
            "target_poi_mismatch_count": subset.manifest["target_poi_mismatch_count"],
            "validated_prompt_count": 100,
            "prompt_token_length_min": min(prompt_lengths),
            "prompt_token_length_max": max(prompt_lengths),
            "target_in_trie_count": target_in_trie,
            "trie_manifest_sha256": sha256_file(trie_dir / "trie_manifest.json"),
            "tokenizer": token_metadata,
        }
        atomic_json(output_dir / "preflight.json", preflight)
        if args.preflight_only:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 0

        model = load_generation_model(checkpoint, expected_vocab_size=len(tokenizer))
        results = []
        for decode_mode in dict.fromkeys(args.decode_modes):
            results.append(
                evaluate_mode(
                    pid_order=args.pid_order,
                    decode_mode=decode_mode,
                    metadata=metadata,
                    records=records,
                    subset_path=subset.data_path,
                    subset_sha256=subset.sha256,
                    subset_manifest_sha256=sha256_file(subset.manifest_path),
                    model=model,
                    tokenizer=tokenizer,
                    template=template,
                    tokens=tokens,
                    token_metadata=token_metadata,
                    trie=trie,
                    trie_dir=trie_dir,
                    poi_ids=poi_ids,
                    output_dir=output_dir,
                    num_beams=args.num_beams,
                    initial_batch_size=args.per_device_eval_batch_size,
                    chunk_size=args.chunk_size,
                    cutoff_len=args.cutoff_len,
                    smoke_limit=args.smoke_limit,
                )
            )
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
        write_summary(output_dir, results)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "results": [
                        {
                            "pid_order": result["config"]["pid_order"],
                            "decode_mode": result["config"]["decode_mode"],
                            "metrics": result["metrics"],
                            "performance": result["performance"],
                            "run_dir": result["run_dir"],
                        }
                        for result in results
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (TigerEvalError, OSError, ValueError) as error:
        print(f"PID 顺序评测失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

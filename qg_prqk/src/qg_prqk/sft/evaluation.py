"""Generate and score QG Final IDs using the existing Beam=10 protocol."""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.sft.evaluation_data import SftEvaluationError, load_json, resolve, signature, read_records, require_hash
from qg_prqk.sft.data import load_final_identifier_lookup
from qg_prqk.sid.geo import GEOHASH_ALPHABET
from qg_prqk.sft.training import project_root
from qg_prqk.sft.vocabulary import _hash_named_files, TOKENIZER_FILES

METRIC_KS = (1, 3, 5, 10)

@dataclass(frozen=True)
class Candidate:
    codes: tuple[int, ...] | None
    poi_row: int | None
    score: float
    error: str | None



def empty_metrics() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "hit_sums": {str(k): 0 for k in METRIC_KS},
        "ndcg_sums": {str(k): 0.0 for k in METRIC_KS},
        "candidate_count": 0,
        "valid_candidate_count": 0,
        "invalid_error_counts": {},
        "samples_with_invalid_candidate": 0,
        "target_rank_histogram": {},
    }


def update_metrics(
    metrics: dict[str, Any],
    *,
    target_row: int,
    candidates: Sequence[Candidate],
) -> int | None:
    """Update Recall/NDCG while invalid beams retain their original ranks."""

    target_rank = next(
        (
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if candidate.poi_row == target_row
        ),
        None,
    )
    metrics["sample_count"] += 1
    metrics["candidate_count"] += len(candidates)
    invalid = [candidate for candidate in candidates if candidate.error is not None]
    metrics["valid_candidate_count"] += len(candidates) - len(invalid)
    metrics["samples_with_invalid_candidate"] += int(bool(invalid))
    for candidate in invalid:
        counts = metrics["invalid_error_counts"]
        counts[candidate.error] = counts.get(candidate.error, 0) + 1
    rank_key = str(target_rank) if target_rank is not None else "miss"
    histogram = metrics["target_rank_histogram"]
    histogram[rank_key] = histogram.get(rank_key, 0) + 1
    for k in METRIC_KS:
        hit = int(target_rank is not None and target_rank <= k)
        metrics["hit_sums"][str(k)] += hit
        if hit:
            metrics["ndcg_sums"][str(k)] += 1.0 / math.log2(target_rank + 1)
    return target_rank


def merge_metrics(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key in (
        "sample_count",
        "candidate_count",
        "valid_candidate_count",
        "samples_with_invalid_candidate",
    ):
        target[key] += int(source[key])
    for name in (
        "hit_sums",
        "ndcg_sums",
        "invalid_error_counts",
        "target_rank_histogram",
    ):
        for key, value in source[name].items():
            target[name][key] = target[name].get(key, 0) + value


def finalize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    samples = int(metrics["sample_count"])
    candidates = int(metrics["candidate_count"])
    if samples <= 0 or candidates <= 0:
        raise SftEvaluationError("QG SFT 评测指标不能为空")
    result: dict[str, Any] = {
        "sample_count": samples,
        "invalid_id_rate": 1 - int(metrics["valid_candidate_count"]) / candidates,
        "valid_id_rate": int(metrics["valid_candidate_count"]) / candidates,
        "samples_with_invalid_id_rate": int(metrics["samples_with_invalid_candidate"])
        / samples,
        "invalid_error_counts": metrics["invalid_error_counts"],
        "target_rank_histogram": metrics["target_rank_histogram"],
    }
    for k in METRIC_KS:
        result[f"hr@{k}"] = metrics["hit_sums"][str(k)] / samples
        result[f"ndcg@{k}"] = metrics["ndcg_sums"][str(k)] / samples
    result["mrr@10"] = sum(int(count) / int(rank) for rank, count in
                           metrics["target_rank_histogram"].items()
                           if rank != "miss" and int(rank) <= 10) / samples
    return result


def load_lf_tokenizer_and_template(
    tokenizer_path: Path,
    *,
    project_root: Path,
) -> tuple[Any, Any]:
    """Load the exact local LLaMA-Factory 0.9.4 qwen3_nothink template."""

    source = project_root / "third_party" / "LLaMA-Factory" / "src"
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        from llamafactory.data import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments, ModelArguments
        from llamafactory.model.patcher import patch_tokenizer
        from transformers.models.qwen2.tokenization_qwen2_fast import (
            Qwen2TokenizerFast,
        )
    except ImportError as error:
        raise SftEvaluationError("无法导入本地 LLaMA-Factory") from error

    tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SftEvaluationError("Tokenizer config JSON 读取失败") from error
    if tokenizer_config.get("tokenizer_class") != "Qwen2Tokenizer":
        raise SftEvaluationError("本评测入口只接受已冻结的 Qwen2 fast tokenizer")
    model_args = ModelArguments(
        model_name_or_path=str(tokenizer_path.resolve()),
        use_fast_tokenizer=True,
        trust_remote_code=False,
    )
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
    )
    patch_tokenizer(tokenizer, model_args)
    data_args = DataArguments(
        template="qwen3_nothink",
        train_on_prompt=False,
        cutoff_len=1024,
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, template


def encode_prompt_like_training(
    *,
    tokenizer: Any,
    template: Any,
    user_content: str,
    target_content: str,
    cutoff_len: int,
) -> tuple[list[int], list[int]]:
    """Return the exact retained training source IDs and unformatted PID IDs."""

    try:
        from llamafactory.data.processor.processor_utils import infer_seqlen
    except ImportError as error:
        raise SftEvaluationError("无法导入 LLaMA-Factory infer_seqlen") from error
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target_content},
    ]
    source_ids, formatted_target_ids = template.encode_oneturn(
        tokenizer,
        messages,
        system=None,
        tools=None,
    )
    source_len, target_len = infer_seqlen(
        len(source_ids),
        len(formatted_target_ids),
        cutoff_len,
    )
    if target_len != len(formatted_target_ids):
        raise SftEvaluationError("cutoff_len 会截断 Assistant Final PID")
    if cutoff_len >= 1024 and source_len != len(source_ids):
        raise SftEvaluationError(
            "1024 Token 新协议禁止截断 Source；请先只删除最早历史事件"
        )
    target_pid_ids = tokenizer.encode(target_content, add_special_tokens=False)
    return source_ids[:source_len], target_pid_ids


def load_generation_model(checkpoint: Path, *, expected_vocab_size: int) -> Any:
    """Load one full-finetuned checkpoint on a single CUDA device."""

    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    if not torch.cuda.is_available():
        raise SftEvaluationError("完整生成评测需要 CUDA GPU，当前环境不可用")
    try:
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SftEvaluationError("Checkpoint config JSON 读取失败") from error
    if config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise SftEvaluationError("本评测入口只接受 Qwen3ForCausalLM checkpoint")
    model = Qwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(torch.device("cuda:0"))
    model.eval()
    if model.config.vocab_size != expected_vocab_size:
        raise SftEvaluationError(
            "Checkpoint 模型词表大小与评测 tokenizer 不一致："
            f"{model.config.vocab_size} != {expected_vocab_size}"
        )
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model


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


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message and "cuda" in message


def validate_checkpoint(config: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """Bind epoch 3 weights, the training tokenizer and the frozen common vocabulary."""
    spec = config["variants"][variant]
    checkpoint = resolve(spec["checkpoint"])
    state = load_json(checkpoint / "trainer_state.json")
    model_config = load_json(checkpoint / "config.json")
    vocab_dir = resolve(config["tokenizer"])
    mapping = load_json(vocab_dir / "qg_prqk_token_mapping.json")
    if _hash_named_files(vocab_dir, TOKENIZER_FILES) != mapping["extended_tokenizer_sha256"]:
        raise SftEvaluationError("共同 tokenizer 文件组哈希与扩词表 manifest 不一致")
    if state.get("global_step") != spec["expected_step"] or state.get("epoch") != 3.0 or state.get("max_steps") != spec["expected_step"]:
        raise SftEvaluationError("checkpoint 不是已完成 epoch 3 的指定末步；禁止自动选模")
    if model_config.get("architectures") != ["Qwen3ForCausalLM"] or model_config.get("vocab_size") != mapping["new_vocab_size"]:
        raise SftEvaluationError("checkpoint 模型结构或词表大小不匹配")
    # LLaMA-Factory reserializes JSON; compare content, not whitespace bytes.
    if load_json(checkpoint / "tokenizer.json") != load_json(vocab_dir / "tokenizer.json"):
        raise SftEvaluationError("训练 tokenizer 的编码内容与冻结共同词表不同")
    losses = [row["eval_loss"] for row in state.get("log_history", [])
              if row.get("epoch") == 3.0 and "eval_loss" in row]
    if len(losses) != 1 or not math.isfinite(losses[0]):
        raise SftEvaluationError("缺少 epoch 3 完整 Validation loss")
    names = ("model.safetensors", "config.json", "trainer_state.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "generation_config.json")
    fingerprints = {}
    for name in names:
        path = checkpoint / name
        if not path.is_file():
            raise SftEvaluationError(f"缺少 checkpoint 文件：{path}")
        stat = path.stat()
        fingerprints[name] = dict(sha256=sha256_file(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return dict(path=str(checkpoint), epoch=3.0, global_step=spec["expected_step"],
                validation_loss=losses[0], files=fingerprints,
                token_mapping_sha256=sha256_file(vocab_dir / "qg_prqk_token_mapping.json"),
                vocab_size=mapping["new_vocab_size"])


@dataclass(frozen=True)
class FinalIdIndex:
    row_by_tokens: dict[tuple[int, ...], int]
    tokens_by_poi: dict[str, tuple[int, ...]]
    target_open: int
    target_close: int
    eos: int
    base_width: int
    poi_count: int


def build_final_id_index(config: Mapping[str, Any], variant: str, tokenizer: Any) -> FinalIdIndex:
    """Build an exact full-corpus lookup; do not prune or expand generated beams."""
    spec = config["variants"][variant]
    mapping = load_json(resolve(config["tokenizer"]) / "qg_prqk_token_mapping.json")
    token_map = mapping["tokens"]
    for token, value in token_map.items():
        if tokenizer.encode(token, add_special_tokens=False) != [value]:
            raise SftEvaluationError(f"Token 不再原子或 ID 变化：{token}")
    lookup = load_final_identifier_lookup(resolve(spec["identifier_dir"]), variant)
    if lookup.manifest_sha256 != spec["identifier_manifest_sha256"] or len(lookup.poi_ids) != 716245:
        raise SftEvaluationError("Final ID 不是冻结的 716,245 active POI")
    base = lookup.base_codes
    columns = []
    if variant == "a4_gid_parent":
        gid_ids = np.asarray([token_map[f"<G_{letter}>"] for letter in GEOHASH_ALPHABET])
        columns.extend(gid_ids[base[:, position]] for position in range(6))
    offset = 6 if variant == "a4_gid_parent" else 0
    for level in range(3):
        ids = np.asarray([token_map[f"<S{level + 1}_{code}>"] for code in range(512)])
        columns.append(ids[base[:, offset + level]])
    token_rows = np.column_stack(columns)
    row_by_tokens = {}
    tokens_by_poi = {}
    for row, (poi, values) in enumerate(zip(lookup.poi_ids, token_rows.tolist(), strict=True)):
        tokens = tuple(values)
        dedup = int(lookup.dedup_codes[row])
        if dedup >= 0:
            tokens += (token_map[f"<D_{dedup}>"],)
        if tokens in row_by_tokens:
            raise SftEvaluationError("完整 Final ID 不唯一，不能做精确 POI 评测")
        row_by_tokens[tokens] = row
        tokens_by_poi[poi] = tokens
    return FinalIdIndex(row_by_tokens, tokens_by_poi, token_map["<TARGET_POI>"],
                        token_map["</TARGET_POI>"], tokenizer.eos_token_id,
                        9 if variant == "a4_gid_parent" else 3, len(row_by_tokens))


def encode_record(record: Mapping[str, Any], *, variant: str, tokenizer: Any,
                  template: Any, index: FinalIdIndex) -> tuple[list[int], int]:
    messages = record.get("messages")
    if (record.get("split"), record.get("identifier_variant")) != ("valid", variant):
        raise SftEvaluationError("评测记录 split/variant 不一致")
    if not isinstance(messages, list) or len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        raise SftEvaluationError("Messages 必须为 user + assistant")
    expected = index.tokens_by_poi.get(record.get("target_poi_id"))
    if expected is None:
        raise SftEvaluationError("评测目标不在完整 active POI 库，禁止丢行")
    if not isinstance(record.get("requires_dedup"), bool) or record["requires_dedup"] != (len(expected) == index.base_width + 1):
        raise SftEvaluationError("末位 Dedup 与训练标签契约不一致")
    source, target = encode_prompt_like_training(
        tokenizer=tokenizer, template=template, user_content=messages[0]["content"],
        target_content=messages[1]["content"], cutoff_len=1024,
    )
    if tuple(target) != (index.target_open, *expected, index.target_close):
        raise SftEvaluationError("目标序列不是该 POI 的精确 Final ID")
    return source, index.row_by_tokens[expected]


def parse_candidate(sequence: Sequence[int], score: float, index: FinalIdIndex) -> Candidate:
    values = tuple(int(token) for token in sequence)
    if not math.isfinite(score):
        return Candidate(None, None, score, "nonfinite_score")
    if index.eos not in values:
        return Candidate(None, None, score, "missing_eos")
    path = values[:values.index(index.eos)]
    if len(path) not in (index.base_width + 2, index.base_width + 3) or path[0] != index.target_open or path[-1] != index.target_close:
        return Candidate(None, None, score, "invalid_wrapper_or_length")
    internal = path[1:-1]
    row = index.row_by_tokens.get(internal)
    if row is None:
        return Candidate(None, None, score, "corpus_miss")
    return Candidate(internal, row, score, None)


def evaluate_chunk(records: Sequence[Mapping[str, Any]], *, variant: str, model: Any,
                   tokenizer: Any, template: Any, index: FinalIdIndex,
                   batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    import torch

    examples = [encode_record(r, variant=variant, tokenizer=tokenizer, template=template, index=index)
                for r in records]
    metrics = empty_metrics()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        input_ids, attention_mask = pad_prompts([x[0] for x in batch], tokenizer.pad_token_id, model.device)
        width = input_ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, do_sample=False,
                num_beams=10, num_return_sequences=10, max_new_tokens=max_new_tokens,
                length_penalty=1.0, early_stopping=True, renormalize_logits=True,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=index.eos,
            )
        sequences = generated.sequences[:, width:].detach().cpu().tolist()
        if generated.sequences_scores is None:
            raise SftEvaluationError("Beam Search 缺少 sequences_scores")
        scores = generated.sequences_scores.float().detach().cpu().tolist()
        if len(sequences) != len(batch) * 10 or len(scores) != len(sequences):
            raise SftEvaluationError("Beam Search 候选数不完整")
        for row, (_, target_row) in enumerate(batch):
            ranked = sorted(zip(sequences[row * 10:(row + 1) * 10], scores[row * 10:(row + 1) * 10], strict=True),
                            key=lambda item: (-float(item[1]), tuple(item[0])))
            # Invalid and repeated candidates retain their original beam positions.
            candidates = [parse_candidate(sequence, float(score), index) for sequence, score in ranked]
            update_metrics(metrics, target_row=target_row, candidates=candidates)
        del generated, input_ids, attention_mask
    return metrics


def source_fingerprints() -> dict[str, str]:
    root = Path(__file__).parent
    return {str(path.relative_to(resolve("qg_prqk"))): sha256_file(path) for path in
            (root / "evaluation.py", root / "evaluation_data.py", root / "evaluation_suite.py",
             root / "data.py", root.parent / "sid/identifiers.py", root.parent / "artifacts.py")}


def verify_worker_inputs(plan: Mapping[str, Any]) -> None:
    checkpoint = Path(plan["checkpoint"]["path"])
    for name, expected in plan["checkpoint"]["files"].items():
        stat = (checkpoint / name).stat()
        if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise SftEvaluationError("checkpoint 在父进程 SHA256 核验后被修改")
    if source_fingerprints() != plan["source_sha256"]:
        raise SftEvaluationError("评测源码在准备后变化，禁止混用结果")


def evaluate_subset(plan: Mapping[str, Any], subset: str, *, smoke_limit: int | None = None) -> dict[str, Any]:
    import torch

    verify_worker_inputs(plan)
    config, variant = plan["config"], plan["variant"]
    spec = config["variants"][variant]
    data_spec = plan["data"]["outputs"][subset]
    data_path = Path(data_spec["file"])
    require_hash(data_path, data_spec["sha256"])
    records = read_records(data_path)
    if len(records) != config["subset_size"]:
        raise SftEvaluationError("评测子集不足 10k；禁止静默继续")
    if smoke_limit is not None:
        records = records[:smoke_limit]
    directory = resolve(config["output_root"]) / variant / ("smoke" if smoke_limit else "results") / subset
    run_config = dict(plan_signature=signature(plan), subset=subset, data=data_spec,
                      variant=variant, checkpoint=plan["checkpoint"], epoch=3.0,
                      decoding="unconstrained_beam_search", num_beams=10, cutoff_len=1024,
                      max_new_tokens=spec["max_new_tokens"], invalid_ids_keep_original_rank=True,
                      smoke_limit=smoke_limit, source_sha256=plan["source_sha256"])
    run_signature = signature(run_config)
    result_path, progress_path = directory / "result.json", directory / "progress.json"
    if result_path.is_file():
        result = load_json(result_path)
        if result.get("signature") != run_signature or result.get("status") != "completed" or result["metrics"]["sample_count"] != len(records):
            raise SftEvaluationError("已有结果协议不一致，拒绝覆盖")
        print(f"[{variant}/{subset}] 已完成 {len(records)} 条，跳过", flush=True)
        return result
    progress = dict(signature=run_signature, next_line=0, metrics=empty_metrics(),
                    inference_seconds=0.0, actual_batch_size=config["decoding"]["batch_size"],
                    peak_memory_bytes=0)
    if progress_path.exists():
        progress = load_json(progress_path)
        if progress.get("signature") != run_signature or progress.get("next_line") != progress.get("metrics", {}).get("sample_count"):
            raise SftEvaluationError("断点协议或累计样本数不一致")
        if not 0 <= progress["next_line"] <= len(records):
            raise SftEvaluationError("断点行数越界")
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer, template = load_lf_tokenizer_and_template(Path(plan["checkpoint"]["path"]), project_root=project_root())
    index = build_final_id_index(config, variant, tokenizer)
    model = load_generation_model(Path(plan["checkpoint"]["path"]), expected_vocab_size=len(tokenizer))
    batch_size = progress["actual_batch_size"]
    torch.cuda.reset_peak_memory_stats(model.device)
    while progress["next_line"] < len(records):
        start = progress["next_line"]
        stop = min(start + config["decoding"]["chunk_size"], len(records))
        started = time.perf_counter()
        try:
            metrics = evaluate_chunk(records[start:stop], variant=variant, model=model,
                                     tokenizer=tokenizer, template=template, index=index,
                                     batch_size=batch_size, max_new_tokens=spec["max_new_tokens"])
        except RuntimeError as error:
            if not is_cuda_oom(error) or batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"[{variant}/{subset}] CUDA OOM，batch 降为 {batch_size}，重试当前未入账 chunk", flush=True)
            # Release the failed generation traceback before retrying the chunk.
            error.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
            continue
        merge_metrics(progress["metrics"], metrics)
        progress.update(next_line=stop, actual_batch_size=batch_size,
                        inference_seconds=progress["inference_seconds"] + time.perf_counter() - started,
                        peak_memory_bytes=max(progress["peak_memory_bytes"], torch.cuda.max_memory_allocated(model.device)))
        write_json_atomic(progress_path, progress, overwrite=True)
        print(f"[{variant}/{subset}] {stop}/{len(records)}，batch={batch_size}", flush=True)
    result = dict(schema_version="qg-prqk-sft-evaluation-result-v1", status="completed",
                  signature=run_signature, config=run_config, metrics=finalize_metrics(progress["metrics"]),
                  performance={k: progress[k] for k in ("inference_seconds", "actual_batch_size", "peak_memory_bytes")})
    write_json_atomic(result_path, result)
    return result

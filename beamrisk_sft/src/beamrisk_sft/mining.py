"""Distributed Train-only Beam trajectory mining and risk-pair finalization."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist

from .candidate_pool import validate_candidate_pool
from .catalog import build_tiger_key_to_poi, load_poi_signatures, strictly_equivalent
from .config import BeamRiskConfig
from .errors import BeamRiskError
from .io import atomic_write_json, implementation_sha256, read_json, sha256_file
from .schema import RISK_PAIR_SCHEMA_VERSION, RiskPair
from .scoring import score_completion_paths
from .tracing import RiskDecision, classify_beam_risk, record_live_beams


MINING_SCHEMA_VERSION = "beamrisk-mining-v1"
MINING_STATES = frozenset(
    ("pre_sid_prune", "miss@10", "hit@10_not@1", "hit@1", "miss_after_sid")
)
TIGER_TARGET_PATTERN = re.compile(
    r"^<TARGET_POI>"
    r"(<S1_(\d+)>)"
    r"(<S2_(\d+)>)"
    r"(<S3_(\d+)>)"
    r"(<C_(\d+)>)"
    r"</TARGET_POI>$"
)
TOKEN_PATTERNS = (
    re.compile(r"^<S1_(\d+)>$"),
    re.compile(r"^<S2_(\d+)>$"),
    re.compile(r"^<S3_(\d+)>$"),
    re.compile(r"^<C_(\d+)>$"),
)


@dataclass(frozen=True)
class EncodedCandidate:
    sample_id: str
    order_id: str
    searchid: str
    target_poi_id: str
    target_tiger_id_key: str
    prompt_token_ids: tuple[int, ...]
    generation_target_ids: tuple[int, ...]
    sid_positions: tuple[int, int, int, int]


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(expected_world_size: int) -> DistributedContext:
    try:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise BeamRiskError("torchrun RANK/LOCAL_RANK/WORLD_SIZE 必须是整数") from error
    if world_size != expected_world_size:
        raise BeamRiskError(
            f"风险挖掘要求 {expected_world_size} 进程，实际 {world_size}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise BeamRiskError("风险挖掘要求四个进程各绑定一张可见 CUDA GPU")
    if not 0 <= rank < world_size or not 0 <= local_rank < torch.cuda.device_count():
        raise BeamRiskError("torchrun rank 超出范围")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)


def load_tokenizer_and_template(config: BeamRiskConfig) -> tuple[Any, Any]:
    # Reuse the repository's frozen qwen3_nothink loader; this is read-only.
    import sys

    repository_src = config.root / "src"
    if str(repository_src) not in sys.path:
        sys.path.insert(0, str(repository_src))
    from poi_gr.sft.evaluation import load_lf_tokenizer_and_template

    tokenizer, template = load_lf_tokenizer_and_template(
        config.paths.model,
        project_root=config.root,
    )
    tokenizer.padding_side = "left"
    return tokenizer, template


def _single_token_id(tokenizer: Any, token: str) -> int:
    values = tokenizer.encode(token, add_special_tokens=False)
    if len(values) != 1 or tokenizer.convert_ids_to_tokens(values[0]) != token:
        raise BeamRiskError(f"TIGER Token 不是冻结的单 Token：{token}")
    return int(values[0])


def encode_candidate_record(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    cutoff_len: int,
) -> EncodedCandidate:
    try:
        from llamafactory.data.processor.processor_utils import infer_seqlen
    except ImportError as error:
        raise BeamRiskError("无法导入 LLaMA-Factory infer_seqlen") from error
    if record.get("split") != "train":
        raise BeamRiskError("风险挖掘只允许 Train")
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(messages[0], Mapping)
        or not isinstance(messages[1], Mapping)
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
        or not isinstance(messages[0].get("content"), str)
        or not isinstance(messages[1].get("content"), str)
    ):
        raise BeamRiskError("候选 messages 必须严格为 user+assistant")
    target_content = messages[1]["content"]
    match = TIGER_TARGET_PATTERN.fullmatch(target_content)
    if match is None:
        raise BeamRiskError(f"目标不是四 Token TIGER SID：{target_content}")
    code_values = tuple(int(match.group(index)) for index in (2, 4, 6, 8))
    target_key = f"{code_values[0]}-{code_values[1]}-{code_values[2]}|c{code_values[3]}"
    if record.get("target_tiger_id_key") != target_key:
        raise BeamRiskError("target_tiger_id_key 与 Assistant 目标不一致")
    source_ids, formatted_target_ids = template.encode_oneturn(
        tokenizer,
        [dict(messages[0]), dict(messages[1])],
        system=None,
        tools=None,
    )
    source_len, target_len = infer_seqlen(
        len(source_ids), len(formatted_target_ids), cutoff_len
    )
    if target_len != len(formatted_target_ids):
        raise BeamRiskError("cutoff_len 截断了 TIGER Assistant target")
    prompt_ids = tuple(int(token) for token in source_ids[:source_len])
    formatted = [int(token) for token in formatted_target_ids]
    try:
        eos_position = formatted.index(int(tokenizer.eos_token_id))
    except ValueError as error:
        raise BeamRiskError("格式化 Assistant target 缺少 EOS") from error
    generation_target = tuple(formatted[: eos_position + 1])
    sid_ids = tuple(
        _single_token_id(tokenizer, match.group(index))
        for index in (1, 3, 5, 7)
    )
    positions: tuple[int, ...] | None = None
    for start in range(0, len(generation_target) - len(sid_ids) + 1):
        if generation_target[start : start + len(sid_ids)] == sid_ids:
            if positions is not None:
                raise BeamRiskError("格式化 target 中 SID 子序列不唯一")
            positions = tuple(range(start, start + 4))
    if positions is None:
        raise BeamRiskError("格式化 target 中找不到 S1/S2/S3/C")
    strings: dict[str, str] = {}
    for name in ("sample_id", "order_id", "searchid", "target_poi_id"):
        value = record.get(name)
        if not isinstance(value, str) or not value:
            raise BeamRiskError(f"候选缺少 {name}")
        strings[name] = value
    return EncodedCandidate(
        **strings,
        target_tiger_id_key=target_key,
        prompt_token_ids=prompt_ids,
        generation_target_ids=generation_target,
        sid_positions=(positions[0], positions[1], positions[2], positions[3]),
    )


def pad_prompts(
    prompts: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not prompts or any(not prompt for prompt in prompts):
        raise BeamRiskError("Beam generation Prompt batch 不能为空")
    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full(
        (len(prompts), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, prompt in enumerate(prompts):
        input_ids[index, width - len(prompt) :] = torch.tensor(
            prompt, dtype=torch.long, device=device
        )
        attention_mask[index, width - len(prompt) :] = 1
    return input_ids, attention_mask


def _candidate_tiger_key(
    token_ids: Sequence[int],
    *,
    tokenizer: Any,
    gold_static_layout: Sequence[int],
    sid_positions: Sequence[int],
) -> str | None:
    if len(token_ids) != len(gold_static_layout):
        return None
    sid_set = set(sid_positions)
    for position, (actual, expected) in enumerate(zip(token_ids, gold_static_layout)):
        if position not in sid_set and int(actual) != int(expected):
            return None
    values: list[int] = []
    for pattern, position in zip(TOKEN_PATTERNS, sid_positions):
        token = tokenizer.convert_ids_to_tokens(int(token_ids[position]))
        match = pattern.fullmatch(str(token))
        if match is None:
            return None
        values.append(int(match.group(1)))
    if any(value >= 1024 for value in values[:3]) or values[3] >= 306:
        return None
    return f"{values[0]}-{values[1]}-{values[2]}|c{values[3]}"


def _partial_structure_valid(
    token_ids: Sequence[int],
    *,
    tokenizer: Any,
    gold_static_layout: Sequence[int],
    sid_positions: Sequence[int],
) -> bool:
    sid_patterns = {position: TOKEN_PATTERNS[index] for index, position in enumerate(sid_positions)}
    for position, actual in enumerate(token_ids):
        pattern = sid_patterns.get(position)
        if pattern is not None:
            if pattern.fullmatch(str(tokenizer.convert_ids_to_tokens(int(actual)))) is None:
                return False
        elif position < len(gold_static_layout) and int(actual) != int(gold_static_layout[position]):
            return False
    return True


def mine_encoded_batch(
    candidates: Sequence[EncodedCandidate],
    *,
    model: Any,
    tokenizer: Any,
    beam_size: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    if not candidates:
        return [], [], Counter()
    reference_target = candidates[0].generation_target_ids
    reference_positions = candidates[0].sid_positions
    reference_sid_set = set(reference_positions)
    for item in candidates[1:]:
        if (
            len(item.generation_target_ids) != len(reference_target)
            or item.sid_positions != reference_positions
            or any(
                item.generation_target_ids[position] != reference_target[position]
                for position in range(len(reference_target))
                if position not in reference_sid_set
            )
        ):
            raise BeamRiskError("同一 mining batch 的格式化 target layout 必须一致")
    device = next(model.parameters()).device
    input_ids, attention_mask = pad_prompts(
        [item.prompt_token_ids for item in candidates],
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    prompt_width = int(input_ids.shape[1])
    with torch.inference_mode(), record_live_beams(
        model,
        prompt_width=prompt_width,
        expected_beams=beam_size,
    ) as recorder:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            max_new_tokens=max_new_tokens,
            length_penalty=1.0,
            early_stopping=True,
            renormalize_logits=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    suffixes = generated.sequences[:, prompt_width:].detach().cpu().tolist()
    if len(suffixes) != len(candidates) * beam_size:
        raise BeamRiskError("generate final sequence 数量与 Beam 配置不一致")
    decisions: list[RiskDecision] = []
    states: Counter[str] = Counter()
    for index, candidate in enumerate(candidates):
        final_sequences = suffixes[index * beam_size : (index + 1) * beam_size]
        decision = classify_beam_risk(
            gold_token_ids=candidate.generation_target_ids,
            sid_positions=candidate.sid_positions,
            live_steps=recorder.steps,
            sample_index=index,
            final_sequences=final_sequences,
        )
        decisions.append(decision)
        states[decision.state] += 1

    state_records = [
        {
            "schema_version": MINING_SCHEMA_VERSION,
            "sample_id": candidate.sample_id,
            "order_id": candidate.order_id,
            "searchid": candidate.searchid,
            "target_poi_id": candidate.target_poi_id,
            "state": decision.state,
            "risk_type": decision.risk_type,
            "first_prune_depth": decision.first_prune_depth,
            "gold_rank": decision.gold_rank,
        }
        for candidate, decision in zip(candidates, decisions)
    ]

    selected_indices = [
        index for index, decision in enumerate(decisions) if decision.risk_type is not None
    ]
    if not selected_indices:
        return [], state_records, states
    prompts = [candidates[index].prompt_token_ids for index in selected_indices]
    positives = [decisions[index].positive_token_ids for index in selected_indices]
    negatives = [decisions[index].negative_token_ids for index in selected_indices]
    if any(path is None for path in positives) or any(path is None for path in negatives):
        raise BeamRiskError("可训练 decision 缺少正负路径")
    with torch.inference_mode():
        positive_scores = score_completion_paths(
            model,
            prompts,
            [tuple(path or ()) for path in positives],
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        ).sums.detach().float().cpu().tolist()
        negative_scores = score_completion_paths(
            model,
            prompts,
            [tuple(path or ()) for path in negatives],
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        ).sums.detach().float().cpu().tolist()

    records: list[dict[str, Any]] = []
    for selected_position, index in enumerate(selected_indices):
        candidate = candidates[index]
        decision = decisions[index]
        positive = tuple(decision.positive_token_ids or ())
        negative = tuple(decision.negative_token_ids or ())
        negative_key: str | None = None
        if decision.risk_type == "final_rank":
            negative_key = _candidate_tiger_key(
                negative,
                tokenizer=tokenizer,
                gold_static_layout=candidate.generation_target_ids,
                sid_positions=candidate.sid_positions,
            )
            structure_valid = negative_key is not None
        else:
            structure_valid = _partial_structure_valid(
                negative,
                tokenizer=tokenizer,
                gold_static_layout=candidate.generation_target_ids,
                sid_positions=candidate.sid_positions,
            )
        positive_score = float(positive_scores[selected_position])
        negative_score = float(negative_scores[selected_position])
        records.append(
            {
                "schema_version": RISK_PAIR_SCHEMA_VERSION,
                "sample_id": candidate.sample_id,
                "order_id": candidate.order_id,
                "searchid": candidate.searchid,
                "target_poi_id": candidate.target_poi_id,
                "risk_type": decision.risk_type,
                "first_prune_depth": decision.first_prune_depth,
                "gold_rank_at_depth": decision.gold_rank,
                "boundary_rank": decision.boundary_rank,
                "reference_positive_score": positive_score,
                "reference_negative_score": negative_score,
                "reference_margin": negative_score - positive_score,
                "prompt_token_ids": list(candidate.prompt_token_ids),
                "positive_token_ids": list(positive),
                "negative_token_ids": list(negative),
                "negative_structure_valid": bool(structure_valid),
                "negative_catalog_expandable": False,
                "negative_poi_id": None,
                "negative_tiger_id_key": negative_key,
                "strict_duplicate_filtered": False,
            }
        )
    return records, state_records, states


def load_generation_model(checkpoint: Path, *, local_rank: int, vocab_size: int) -> Any:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    model = Qwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(torch.device("cuda", local_rank))
    model.eval()
    if int(model.config.vocab_size) != vocab_size:
        raise BeamRiskError(
            f"Checkpoint vocab {model.config.vocab_size} != tokenizer {vocab_size}"
        )
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model


def _checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    state = read_json(checkpoint / "trainer_state.json", name="Trainer state")
    model_files = sorted(checkpoint.glob("*.safetensors"))
    if not model_files:
        raise BeamRiskError(f"Checkpoint 缺少 safetensors：{checkpoint}")
    digest = hashlib.sha256()
    for path in model_files:
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(sha256_file(path).encode("ascii"))
    return {
        "path": str(checkpoint.resolve()),
        "epoch": state.get("epoch"),
        "global_step": state.get("global_step"),
        "model_files_sha256": digest.hexdigest(),
    }


def mine_rank_part(
    config: BeamRiskConfig,
    *,
    reference_epoch: int,
    checkpoint: Path,
    context: DistributedContext,
    resume: bool,
) -> dict[str, Any]:
    candidate_manifest = validate_candidate_pool(config)
    candidate_part_meta = candidate_manifest["parts"][context.rank]
    candidate_path = config.paths.candidate_pool_dir / str(candidate_part_meta["path"])
    if sha256_file(candidate_path) != candidate_part_meta["sha256"]:
        raise BeamRiskError(f"当前 rank 的 Candidate part SHA256 变化：{candidate_path}")
    output_dir = config.mining_dir(reference_epoch)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    output_path = parts_dir / f"risk_pairs_rank_{context.rank:03d}.jsonl"
    states_path = parts_dir / f"states_rank_{context.rank:03d}.jsonl"
    manifest_path = parts_dir / f"manifest_rank_{context.rank:03d}.json"
    checkpoint_identity = _checkpoint_identity(checkpoint)
    expected_epoch = float(reference_epoch)
    if not math.isclose(float(checkpoint_identity.get("epoch", -1)), expected_epoch, abs_tol=1e-4):
        raise BeamRiskError("风险挖掘 checkpoint epoch 与阶段不一致")
    if (
        resume
        and manifest_path.is_file()
        and output_path.is_file()
        and states_path.is_file()
    ):
        prior = read_json(manifest_path, name="Rank mining manifest")
        if (
            prior.get("status") == "completed"
            and prior.get("reference_checkpoint") == checkpoint_identity
            and prior.get("candidate_part_sha256") == candidate_part_meta["sha256"]
            and prior.get("output_sha256") == sha256_file(output_path)
            and prior.get("states_sha256") == sha256_file(states_path)
        ):
            return prior
    if output_path.exists() or states_path.exists() or manifest_path.exists():
        raise BeamRiskError(
            f"Rank mining 输出已存在；使用 --resume 或换目录：{output_path}"
        )

    tokenizer, template = load_tokenizer_and_template(config)
    model = load_generation_model(
        checkpoint,
        local_rank=context.local_rank,
        vocab_size=len(tokenizer),
    )
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    states_temporary = states_path.with_name(f".{states_path.name}.tmp")
    digest = hashlib.sha256()
    states_digest = hashlib.sha256()
    state_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    prune_counts: Counter[str] = Counter()
    scanned = 0
    written = 0
    states_written = 0
    batch: list[EncodedCandidate] = []

    def commit_batch(stream: Any, states_stream: Any) -> bool:
        nonlocal states_written, written
        if not batch:
            return False
        records, state_records, states = mine_encoded_batch(
            batch,
            model=model,
            tokenizer=tokenizer,
            beam_size=config.mining.beam_size,
            max_new_tokens=config.mining.max_new_tokens,
        )
        state_counts.update(states)
        for state_record in state_records:
            state_payload = (
                json.dumps(
                    state_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            states_stream.write(state_payload)
            states_digest.update(state_payload)
            states_written += 1
        for record in records:
            if written >= config.mining.per_rank_pair_cap:
                break
            payload = (
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            stream.write(payload)
            digest.update(payload)
            written += 1
            risk_counts[str(record["risk_type"])] += 1
            if record["first_prune_depth"] is not None:
                prune_counts[str(record["first_prune_depth"])] += 1
        batch.clear()
        return written >= config.mining.per_rank_pair_cap

    try:
        with (
            candidate_path.open("r", encoding="utf-8") as source,
            temporary.open("wb") as output,
            states_temporary.open("wb") as states_output,
        ):
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise BeamRiskError(
                        f"Candidate part 第 {line_number} 行 JSON 无效"
                    ) from error
                batch.append(
                    encode_candidate_record(
                        record,
                        tokenizer=tokenizer,
                        template=template,
                        cutoff_len=config.training.cutoff_len,
                    )
                )
                scanned += 1
                if len(batch) == config.mining.batch_size and commit_batch(
                    output, states_output
                ):
                    break
                if scanned % 2_000 == 0:
                    print(
                        f"[mine rank={context.rank}] scanned={scanned:,} pairs={written:,}",
                        flush=True,
                    )
            if batch and written < config.mining.per_rank_pair_cap:
                commit_batch(output, states_output)
            if states_written != scanned:
                raise BeamRiskError(
                    f"Rank state 行数 {states_written} != 扫描样本 {scanned}"
                )
            output.flush()
            os.fsync(output.fileno())
            states_output.flush()
            os.fsync(states_output.fileno())
        temporary.replace(output_path)
        states_temporary.replace(states_path)
    finally:
        temporary.unlink(missing_ok=True)
        states_temporary.unlink(missing_ok=True)
        torch.cuda.empty_cache()
    manifest = {
        "schema_version": MINING_SCHEMA_VERSION,
        "status": "completed",
        "reference_epoch": reference_epoch,
        "reference_checkpoint": checkpoint_identity,
        "rank": context.rank,
        "world_size": context.world_size,
        "implementation_sha256": implementation_sha256(config.root),
        "candidate_part": str(candidate_path),
        "candidate_part_sha256": candidate_part_meta["sha256"],
        "candidate_rows_available": candidate_part_meta["rows"],
        "candidate_rows_scanned": scanned,
        "pairs": written,
        "state_counts": dict(sorted(state_counts.items())),
        "risk_type_counts": dict(sorted(risk_counts.items())),
        "first_prune_depth_counts": dict(sorted(prune_counts.items())),
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": digest.hexdigest(),
        "states_output": str(states_path),
        "states_rows": states_written,
        "states_bytes": states_path.stat().st_size,
        "states_sha256": states_digest.hexdigest(),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _iter_part_offsets(paths: Sequence[Path]) -> Iterable[tuple[int, int, dict[str, Any]]]:
    for part_index, path in enumerate(paths):
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise BeamRiskError(f"Mined part JSON 无效：{path}@{offset}") from error
                yield part_index, offset, value


def _load_state_map(
    paths: Sequence[Path],
    *,
    expected_rows: int,
) -> tuple[dict[str, str], Counter[str]]:
    result: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise BeamRiskError(
                        f"Mining state JSON 无效：{path}:{line_number}"
                    ) from error
                if value.get("schema_version") != MINING_SCHEMA_VERSION:
                    raise BeamRiskError("Mining state schema_version 无效")
                sample_id = value.get("sample_id")
                state = value.get("state")
                if not isinstance(sample_id, str) or len(sample_id) != 64:
                    raise BeamRiskError("Mining state sample_id 无效")
                if state not in MINING_STATES:
                    raise BeamRiskError(f"Mining state 无效：{state}")
                if sample_id in result:
                    raise BeamRiskError(f"Mining state sample_id 重复：{sample_id}")
                result[sample_id] = str(state)
                counts[str(state)] += 1
    if len(result) != expected_rows:
        raise BeamRiskError(
            f"Mining state 行数 {len(result)} != rank manifest {expected_rows}"
        )
    return result, counts


def _write_state_diagnostics(
    config: BeamRiskConfig,
    *,
    reference_epoch: int,
    output_dir: Path,
    current_states: Mapping[str, str],
    current_counts: Mapping[str, int],
) -> Path:
    common = {
        "schema_version": MINING_SCHEMA_VERSION,
        "status": "completed",
        "reference_epoch": reference_epoch,
        "rows": len(current_states),
        "state_counts": dict(sorted(current_counts.items())),
    }
    if reference_epoch == 1:
        path = output_dir / "state_transition_input.json"
        atomic_write_json(path, common)
        return path

    previous_dir = config.mining_dir(1)
    previous_manifests = [
        read_json(
            previous_dir / "parts" / f"manifest_rank_{rank:03d}.json",
            name="Epoch-1 rank mining manifest",
        )
        for rank in range(config.training.world_size)
    ]
    previous_paths = [Path(str(item.get("states_output", ""))) for item in previous_manifests]
    previous_rows = sum(int(item.get("states_rows", 0)) for item in previous_manifests)
    for path, item in zip(previous_paths, previous_manifests):
        if not path.is_file() or sha256_file(path) != item.get("states_sha256"):
            raise BeamRiskError(f"Epoch-1 mining state 缺失或 SHA 变化：{path}")
    previous_states, previous_counts = _load_state_map(
        previous_paths,
        expected_rows=previous_rows,
    )
    previous_ids = set(previous_states)
    current_ids = set(current_states)
    shared = previous_ids & current_ids
    transitions: Counter[str] = Counter(
        f"{previous_states[sample_id]}->{current_states[sample_id]}"
        for sample_id in shared
    )
    payload = {
        **common,
        "previous_reference_epoch": 1,
        "previous_rows": len(previous_states),
        "previous_state_counts": dict(sorted(previous_counts.items())),
        "shared_rows": len(shared),
        "previous_only_rows": len(previous_ids - current_ids),
        "current_only_rows": len(current_ids - previous_ids),
        "transition_counts": dict(sorted(transitions.items())),
    }
    path = output_dir / "state_transition_from_epoch_1.json"
    atomic_write_json(path, payload)
    return path


def finalize_mined_parts(
    config: BeamRiskConfig,
    *,
    reference_epoch: int,
    checkpoint: Path,
    resume: bool,
) -> dict[str, Any]:
    output_dir = config.mining_dir(reference_epoch)
    final_manifest_path = output_dir / "manifest.json"
    final_jsonl = output_dir / "risk_pairs.jsonl"
    dataset_path = output_dir / "dataset"
    checkpoint_identity = _checkpoint_identity(checkpoint)
    if resume and final_manifest_path.is_file() and final_jsonl.is_file() and dataset_path.is_dir():
        prior = read_json(final_manifest_path, name="Mining manifest")
        diagnostics_path = Path(str(prior.get("state_diagnostics", "")))
        if (
            prior.get("status") == "completed"
            and prior.get("reference_checkpoint") == checkpoint_identity
            and prior.get("pairs", 0) >= config.mining.minimum_pair_count
            and prior.get("risk_pairs_sha256") == sha256_file(final_jsonl)
            and diagnostics_path.is_file()
        ):
            return prior
    part_manifests = [
        read_json(
            output_dir / "parts" / f"manifest_rank_{rank:03d}.json",
            name="Rank mining manifest",
        )
        for rank in range(config.training.world_size)
    ]
    for part in part_manifests:
        if part.get("status") != "completed" or part.get("reference_checkpoint") != checkpoint_identity:
            raise BeamRiskError("Rank mining manifest 与当前 checkpoint 不一致")
    part_paths = [Path(str(part["output"])) for part in part_manifests]
    for path, part in zip(part_paths, part_manifests):
        if not path.is_file() or sha256_file(path) != part.get("output_sha256"):
            raise BeamRiskError(f"Rank mining part 缺失或 SHA 变化：{path}")
    state_paths = [Path(str(part.get("states_output", ""))) for part in part_manifests]
    expected_state_rows = sum(int(part.get("states_rows", 0)) for part in part_manifests)
    for path, part in zip(state_paths, part_manifests):
        if not path.is_file() or sha256_file(path) != part.get("states_sha256"):
            raise BeamRiskError(f"Rank mining state 缺失或 SHA 变化：{path}")
    current_states, current_state_counts = _load_state_map(
        state_paths,
        expected_rows=expected_state_rows,
    )
    state_diagnostics_path = _write_state_diagnostics(
        config,
        reference_epoch=reference_epoch,
        output_dir=output_dir,
        current_states=current_states,
        current_counts=current_state_counts,
    )

    negative_keys: set[str] = set()
    negative_key_counts: Counter[str] = Counter()
    target_ids_by_negative_key: dict[str, set[str]] = {}
    raw_rows = 0
    for _, _, value in _iter_part_offsets(part_paths):
        raw_rows += 1
        if value.get("risk_type") == "final_rank" and value.get("negative_structure_valid"):
            key = value.get("negative_tiger_id_key")
            if not isinstance(key, str) or not key:
                raise BeamRiskError("结构合法 final_rank pair 缺少 negative_tiger_id_key")
            negative_keys.add(key)
            negative_key_counts[key] += 1
            target_ids_by_negative_key.setdefault(key, set()).add(
                str(value.get("target_poi_id"))
            )
    # Position-wise token validity does not imply that the generated S1-S3/C
    # combination exists in the catalog.  Such paths are genuine errors under
    # unconstrained decoding and remain useful negatives; only mapped paths
    # participate in the strict duplicate-POI filter.
    key_to_poi = build_tiger_key_to_poi(
        config.paths.tiger_mapping,
        negative_keys,
        require_all=False,
    )
    missing_negative_keys = negative_keys - set(key_to_poi)
    target_ids_for_rank_pairs = {
        target_poi_id
        for key in key_to_poi
        for target_poi_id in target_ids_by_negative_key[key]
    }
    signature_ids = target_ids_for_rank_pairs | set(key_to_poi.values())
    signatures = load_poi_signatures(config.paths.poi_catalog_dir, signature_ids)
    missing_digest = hashlib.sha256()
    for key in sorted(missing_negative_keys):
        missing_digest.update(key.encode("utf-8"))
        missing_digest.update(b"\n")
    catalog_diagnostics_path = output_dir / "catalog_path_diagnostics.json"
    structurally_valid_final_rank = sum(negative_key_counts.values())
    catalog_unexpandable_final_rank = sum(
        negative_key_counts[key] for key in missing_negative_keys
    )
    atomic_write_json(
        catalog_diagnostics_path,
        {
            "schema_version": MINING_SCHEMA_VERSION,
            "status": "completed",
            "reference_epoch": reference_epoch,
            "structurally_valid_final_rank_pairs": structurally_valid_final_rank,
            "candidate_path_keys": len(negative_keys),
            "catalog_expandable_path_keys": len(key_to_poi),
            "catalog_unexpandable_path_keys": len(missing_negative_keys),
            "catalog_expandable_final_rank_pairs": (
                structurally_valid_final_rank - catalog_unexpandable_final_rank
            ),
            "catalog_unexpandable_final_rank_pairs": catalog_unexpandable_final_rank,
            "catalog_unexpandable_rate": (
                catalog_unexpandable_final_rank / structurally_valid_final_rank
                if structurally_valid_final_rank
                else 0.0
            ),
            "catalog_unexpandable_keys_sha256": missing_digest.hexdigest(),
            "catalog_unexpandable_key_examples": sorted(missing_negative_keys)[:100],
        },
    )

    # Keep the smallest sample SHA values without retaining large prompt arrays.
    selected_heap: list[tuple[int, int, int, str]] = []
    eligible_rows = 0
    duplicate_filtered = 0
    structurally_invalid_final_rank_negatives = 0
    catalog_unexpandable_final_rank_negatives = 0
    catalog_expandable_final_rank_negatives = 0
    risk_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    selected_final_rank_catalog_counts: Counter[str] = Counter()
    seen_samples: set[str] = set()
    for part_index, offset, value in _iter_part_offsets(part_paths):
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or len(sample_id) != 64:
            raise BeamRiskError("Mined pair sample_id 无效")
        if sample_id in seen_samples:
            raise BeamRiskError(f"Mined pair sample_id 重复：{sample_id}")
        seen_samples.add(sample_id)
        duplicate = False
        if value.get("risk_type") == "final_rank":
            if value.get("negative_structure_valid"):
                negative_poi_id = key_to_poi.get(
                    str(value["negative_tiger_id_key"])
                )
                if negative_poi_id is None:
                    catalog_unexpandable_final_rank_negatives += 1
                else:
                    catalog_expandable_final_rank_negatives += 1
                    duplicate = strictly_equivalent(
                        str(value["target_poi_id"]), negative_poi_id, signatures
                    )
            else:
                structurally_invalid_final_rank_negatives += 1
        if duplicate:
            duplicate_filtered += 1
            continue
        eligible_rows += 1
        priority = int(sample_id, 16)
        item = (-priority, part_index, offset, sample_id)
        if len(selected_heap) < config.mining.target_pair_count:
            heapq.heappush(selected_heap, item)
        elif priority < -selected_heap[0][0]:
            heapq.heapreplace(selected_heap, item)
    selected_locations = {(item[1], item[2]) for item in selected_heap}
    final_count = len(selected_locations)
    if final_count < config.mining.minimum_pair_count:
        raise BeamRiskError(
            f"严格过滤后风险 pair 只有 {final_count:,}，低于门禁 "
            f"{config.mining.minimum_pair_count:,}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = final_jsonl.with_name(f".{final_jsonl.name}.tmp")
    digest = hashlib.sha256()
    written = 0
    try:
        with temporary.open("wb") as output:
            for part_index, offset, value in _iter_part_offsets(part_paths):
                if (part_index, offset) not in selected_locations:
                    continue
                negative_key = value.get("negative_tiger_id_key")
                negative_poi_id = (
                    key_to_poi.get(str(negative_key))
                    if value.get("negative_structure_valid") and negative_key is not None
                    else None
                )
                value["negative_poi_id"] = negative_poi_id
                value["negative_catalog_expandable"] = negative_poi_id is not None
                value["strict_duplicate_filtered"] = False
                value["reference_checkpoint"] = checkpoint_identity
                RiskPair.from_mapping(value)
                payload = (
                    json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                output.write(payload)
                digest.update(payload)
                written += 1
                risk_counts[str(value["risk_type"])] += 1
                if value["risk_type"] == "final_rank":
                    if not value["negative_structure_valid"]:
                        selected_final_rank_catalog_counts["structure_invalid"] += 1
                    elif negative_poi_id is None:
                        selected_final_rank_catalog_counts["catalog_unexpandable"] += 1
                    else:
                        selected_final_rank_catalog_counts["catalog_expandable"] += 1
                if value.get("first_prune_depth") is not None:
                    depth_counts[str(value["first_prune_depth"])] += 1
            output.flush()
            os.fsync(output.fileno())
        if written != final_count:
            raise BeamRiskError(f"最终风险 pair 写入 {written} != 选择 {final_count}")
        temporary.replace(final_jsonl)
    finally:
        temporary.unlink(missing_ok=True)

    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    temporary_dataset = dataset_path.with_name(f".{dataset_path.name}.tmp")
    if temporary_dataset.exists():
        shutil.rmtree(temporary_dataset)
    from datasets import Dataset

    dataset = Dataset.from_json(str(final_jsonl))
    if len(dataset) != written:
        raise BeamRiskError("Arrow risk dataset 行数与 JSONL 不一致")
    dataset.save_to_disk(temporary_dataset)
    temporary_dataset.replace(dataset_path)
    manifest = {
        "schema_version": MINING_SCHEMA_VERSION,
        "status": "completed",
        "reference_epoch": reference_epoch,
        "reference_checkpoint": checkpoint_identity,
        "implementation_sha256": implementation_sha256(config.root),
        "candidate_pool_manifest": str(config.paths.candidate_pool_dir / "manifest.json"),
        "raw_mined_pairs": raw_rows,
        "eligible_after_duplicate_filter": eligible_rows,
        "strict_duplicate_filtered": duplicate_filtered,
        "invalid_final_rank_negatives": (
            structurally_invalid_final_rank_negatives
            + catalog_unexpandable_final_rank_negatives
        ),
        "structurally_invalid_final_rank_negatives": (
            structurally_invalid_final_rank_negatives
        ),
        "catalog_unexpandable_final_rank_negatives": (
            catalog_unexpandable_final_rank_negatives
        ),
        "catalog_expandable_final_rank_negatives": (
            catalog_expandable_final_rank_negatives
        ),
        "selected_final_rank_catalog_counts": dict(
            sorted(selected_final_rank_catalog_counts.items())
        ),
        "catalog_path_diagnostics": str(catalog_diagnostics_path),
        "pairs": written,
        "target_pair_count": config.mining.target_pair_count,
        "minimum_pair_count": config.mining.minimum_pair_count,
        "risk_type_counts": dict(sorted(risk_counts.items())),
        "first_prune_depth_counts": dict(sorted(depth_counts.items())),
        "scanned_state_rows": len(current_states),
        "scanned_state_counts": dict(sorted(current_state_counts.items())),
        "state_diagnostics": str(state_diagnostics_path),
        "risk_pairs_jsonl": str(final_jsonl),
        "risk_pairs_sha256": digest.hexdigest(),
        "dataset": str(dataset_path),
        "rank_manifests": [str(output_dir / "parts" / f"manifest_rank_{rank:03d}.json") for rank in range(config.training.world_size)],
    }
    atomic_write_json(final_manifest_path, manifest)
    atomic_write_json(output_dir / "mining_metrics.json", {
        key: manifest[key]
        for key in (
            "raw_mined_pairs",
            "eligible_after_duplicate_filter",
            "strict_duplicate_filtered",
            "invalid_final_rank_negatives",
            "structurally_invalid_final_rank_negatives",
            "catalog_unexpandable_final_rank_negatives",
            "catalog_expandable_final_rank_negatives",
            "selected_final_rank_catalog_counts",
            "pairs",
            "risk_type_counts",
            "first_prune_depth_counts",
            "scanned_state_rows",
            "scanned_state_counts",
        )
    })
    return manifest


def run_distributed_mining(
    config: BeamRiskConfig,
    *,
    reference_epoch: int,
    checkpoint: Path,
    resume: bool,
) -> dict[str, Any] | None:
    context = initialize_distributed(config.training.world_size)
    try:
        checkpoint = checkpoint.resolve()
        mine_rank_part(
            config,
            reference_epoch=reference_epoch,
            checkpoint=checkpoint,
            context=context,
            resume=resume,
        )
        dist.barrier()
        result: dict[str, Any] | None = None
        if context.is_main:
            result = finalize_mined_parts(
                config,
                reference_epoch=reference_epoch,
                checkpoint=checkpoint,
                resume=resume,
            )
        dist.barrier()
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

"""Paired Qwen teacher-forced and beam-prefix diagnostics on frozen Validation."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.sft import evaluation as ev
from qg_prqk.sft.a0_evaluation import build_index
from qg_prqk.sft.a0_gid_data import HISTORY_ID
from qg_prqk.sft.constrained_decoding import ConstrainedGenerationModel, FinalIdPrefixIndex
from qg_prqk.sft.evaluation_data import SUBSETS, business_key, load_json, read_records, resolve, signature
from qg_prqk.sft.training import project_root

OUTPUT = "qg_prqk/outputs/eval/a0_a4_qwen_prefix_diagnostic_v1"
PLANS = {
    "a0": "qg_prqk/outputs/eval/a0_gid_epoch3_dual_decode_v1/plan.json",
    "a4": "qg_prqk/outputs/eval/sft_epoch3_fixed10k_generalization_v1/a4_gid_parent/plan_2x6000d.json",
}
VARIANTS = {"a0": "a0_gid", "a4": "a4_gid_parent"}
MODES = ("unconstrained", "constrained")
SELECTION_SEED = "qg-qwen-prefix-diagnostic-v1"


def key_hash(record: dict) -> str:
    return hashlib.sha256(json.dumps(business_key(record), ensure_ascii=False).encode()).hexdigest()


def select_indices(records: list[dict], count: int) -> list[int]:
    """Select a deterministic paired sample independent of IDs and outcomes."""
    if len(records) != 10000 or len({business_key(r) for r in records}) != 10000:
        raise ValueError("必须来自固定 10k 唯一业务键")
    return sorted(sorted(range(len(records)), key=lambda i: hashlib.sha256(
        (SELECTION_SEED + key_hash(records[i])).encode()).digest())[:count])


def source_hashes() -> dict:
    paths = [Path(__file__), Path(__file__).with_name("a0_evaluation.py"),
             Path(__file__).with_name("a0_gid_data.py"), Path(__file__).with_name("constrained_decoding.py")]
    return {**ev.source_fingerprints(), **{str(p): sha256_file(p) for p in paths}}


def inspect(per_subset: int) -> dict:
    plans = {name: load_json(resolve(path)) for name, path in PLANS.items()}
    if plans["a0"]["config"]["decoding"] != plans["a4"]["config"]["decoding"]:
        raise ValueError("原评测生成协议不同")
    return dict(status="planned", per_subset=per_subset, selection_seed=SELECTION_SEED,
                plans=plans, plan_hashes={name: sha256_file(resolve(path)) for name, path in PLANS.items()},
                source_hashes=source_hashes(), no_test=True)


def verify_plan(plan: dict) -> None:
    current = inspect(plan["per_subset"])
    if any(plan[key] != current[key] for key in ("plan_hashes", "source_hashes", "selection_seed")):
        raise ValueError("配置、原评测计划或代码变化，拒绝复用")
    for method in PLANS:
        checkpoint = plan["checkpoints"][method]
        for name, receipt in checkpoint["files"].items():
            stat = (Path(checkpoint["path"]) / name).stat()
            if (stat.st_size, stat.st_mtime_ns) != (receipt["bytes"], receipt["mtime_ns"]):
                raise ValueError("完整哈希预检后 checkpoint 文件发生变化")
        if sha256_file(Path(plan["data"][method]["path"])) != plan["data"][method]["sha256"]:
            raise ValueError("诊断配对样本发生变化")


def prepare(per_subset: int, output: Path) -> dict:
    path = output / "plan.json"
    if path.exists():
        plan = load_json(path)
        verify_plan(plan)
        return plan
    plan = inspect(per_subset)
    samples = {name: [] for name in PLANS}
    for subset in SUBSETS:
        records = {}
        for name, original in plan["plans"].items():
            spec = original["data"]["outputs"][subset]
            source = Path(spec["file"])
            if sha256_file(source) != spec["sha256"]:
                raise ValueError("原固定 Validation 样本哈希变化")
            records[name] = read_records(source)
        for a, b in zip(records["a0"], records["a4"], strict=True):
            if (business_key(a), a["target_poi_id"], a["history_length"], HISTORY_ID.sub("<ID>", a["messages"][0]["content"])) != (
                business_key(b), b["target_poi_id"], b["history_length"], HISTORY_ID.sub("<ID>", b["messages"][0]["content"])):
                raise ValueError("A0/A4 样本未逐行配对")
        indices = select_indices(records["a0"], per_subset)
        if indices != select_indices(records["a4"], per_subset):
            raise ValueError("确定性抽样不同")
        for name in PLANS:
            samples[name] += [dict(record=records[name][i], subset=subset, source_line=i + 1,
                                   key_sha256=key_hash(records[name][i])) for i in indices]
        print(f"[准备/{subset}] 同一批 {per_subset} 条", flush=True)
    plan.update(status="prepared", checkpoints={}, data={})
    for name, original in plan["plans"].items():
        config, variant = original["config"], VARIANTS[name]
        checkpoint = ev.validate_checkpoint(config, variant)
        if checkpoint != original["checkpoint"]:
            raise ValueError("模型权重或元数据与原正式评测不同")
        tokenizer, template = ev.load_lf_tokenizer_and_template(Path(checkpoint["path"]), project_root=project_root())
        index = build_index(config, tokenizer) if name == "a0" else ev.build_final_id_index(config, variant, tokenizer)
        for item in samples[name]:
            ev.encode_record(item["record"], variant=variant, tokenizer=tokenizer, template=template, index=index)
        destination = output / f"{name}_samples.json"
        write_json_atomic(destination, {"samples": samples[name]}, overwrite=True)
        plan["data"][name] = dict(path=str(destination), sha256=sha256_file(destination), rows=len(samples[name]))
        plan["checkpoints"][name] = checkpoint
        del tokenizer, template, index
        gc.collect()
        print(f"[准备/{name}] checkpoint 完整哈希、全目录、全部样本 Token 检查通过", flush=True)
    write_json_atomic(path, plan)
    return plan


def target_stages(target: Sequence[int]) -> list[str]:
    """Name target positions including wrappers and optional trailing Dedup."""
    if len(target) not in (12, 13):
        raise ValueError("只支持 GID6+SID3+[D] 的完整目标（含 wrapper/EOS）")
    return ["open", *[f"g{i}" for i in range(1, 7)], "s1", "s2", "s3",
            *(["dedup"] if len(target) == 13 else []), "close", "eos"]


def token_observation(logits: Any, target: int, allowed: Sequence[int]) -> dict:
    """Measure both full-vocabulary and legal-child conditional distributions."""
    import torch

    values = logits.float()
    choices = torch.tensor(allowed, device=values.device)
    if target not in allowed or not bool(torch.isfinite(values).all()):
        raise ValueError("目标不在合法后继或出现非有限 logits")
    selected = values[target]
    rank = 1 + int((values > selected).sum()) + int(((values == selected) & (torch.arange(len(values), device=values.device) < target)).sum())
    legal_values = values[choices]
    legal_rank = 1 + int((legal_values > selected).sum()) + int(((legal_values == selected) & (choices < target)).sum())
    total = torch.logsumexp(values, dim=0)
    legal_total = torch.logsumexp(legal_values, dim=0)
    return dict(rank=rank, legal_rank=legal_rank, nll=float(total-selected), legal_nll=float(legal_total-selected),
                legal_mass=float(torch.exp(legal_total-total)), candidate_count=len(allowed))


def teacher_forced(model: Any, tokenizer: Any, trie: FinalIdPrefixIndex,
                   prompts: list[list[int]], targets: list[list[int]]) -> list[dict]:
    """Score each true next token with the same left-padding position convention."""
    import torch

    inputs, mask = ev.pad_prompts([p + t[:-1] for p, t in zip(prompts, targets, strict=True)], tokenizer.pad_token_id, model.device)
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    keep = max(map(len, targets))
    with torch.inference_mode():
        logits = model(input_ids=inputs, attention_mask=mask, position_ids=positions,
                       use_cache=False, logits_to_keep=keep).logits
    rows = []
    for row, target in enumerate(targets):
        observations = {}
        for j, stage in enumerate(target_stages(target)):
            observations[stage] = token_observation(logits[row, keep-len(target)+j], target[j], trie.allowed_next(tuple(target[:j])))
        rows.append(observations)
    return rows


def beam_prefixes(target: list[int], sequences: list[list[int]]) -> dict:
    """Keep original beam ranks; measure joint geographic and semantic prefixes."""
    stops = dict(gid6=7, gid6_s1=8, gid6_s1_s2=9, gid6_sid3=10, final_id=len(target))
    return {stage: {str(k): any(seq[:stop] == target[:stop] for seq in sequences[:k]) for k in (1, 10)}
            for stage, stop in stops.items()}


class RecordingModel:
    """Observe generation without changing the existing evaluator's scoring."""
    def __init__(self, model: Any):
        self.model, self.device, self.beams = model, model.device, []

    def generate(self, **kwargs: Any) -> Any:
        result = self.model.generate(**kwargs)
        sequences = result.sequences[:, kwargs["input_ids"].shape[1]:].detach().cpu().tolist()
        scores = result.sequences_scores.float().detach().cpu().tolist()
        for offset in range(0, len(sequences), 10):
            ranked = sorted(zip(sequences[offset:offset+10], scores[offset:offset+10], strict=True),
                            key=lambda item: (-item[1], tuple(item[0])))
            self.beams.append(dict(tokens=[x[0] for x in ranked], scores=[x[1] for x in ranked]))
        return result


def require_idle_gpu() -> None:
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    import os
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if not visible.isdigit():
        raise ValueError("诊断仅支持显式单张数字 GPU 编号")
    selected = [row.split(",") for row in rows if row.split(",")[0].strip() == visible]
    if len(selected) != 1 or int(selected[0][1]) < 16000 or int(selected[0][2]) > 5:
        raise ValueError("选定 GPU 不空闲或空闲显存不足 16GB，拒绝追加推理")


def worker(plan: dict, name: str, output: Path) -> None:
    import torch

    verify_plan(plan)
    path = output / f"{name}_result.json"
    if path.exists():
        result = load_json(path)
        if result["signature"] != signature(plan) or result["status"] != "completed":
            raise ValueError("已有结果不兼容")
        return
    require_idle_gpu()
    config, variant = plan["plans"][name]["config"], VARIANTS[name]
    checkpoint = Path(plan["checkpoints"][name]["path"])
    tokenizer, template = ev.load_lf_tokenizer_and_template(checkpoint, project_root=project_root())
    index = build_index(config, tokenizer) if name == "a0" else ev.build_final_id_index(config, variant, tokenizer)
    trie = FinalIdPrefixIndex(index)
    model = ev.load_generation_model(checkpoint, expected_vocab_size=len(tokenizer))
    items = load_json(Path(plan["data"][name]["path"]))["samples"]
    progress_path = output / f"{name}_progress.json"
    progress = load_json(progress_path) if progress_path.exists() else dict(signature=signature(plan), rows=[], batch_size=8)
    if progress["signature"] != signature(plan) or not 1 <= progress["batch_size"] <= 8:
        raise ValueError("断点签名或 batch 不一致")
    expected_keys = [(i["subset"], i["key_sha256"]) for i in items]
    if [(i["subset"], i["key_sha256"]) for i in progress["rows"]] != expected_keys[:len(progress["rows"])]:
        raise ValueError("断点样本错位")
    while len(progress["rows"]) < len(items):
        start = len(progress["rows"])
        chunk = items[start:start+progress["batch_size"]]
        records = [x["record"] for x in chunk]
        encoded = [ev.encode_record(r, variant=variant, tokenizer=tokenizer, template=template, index=index) for r in records]
        prompts = [x[0] for x in encoded]
        targets = [[index.target_open, *index.tokens_by_poi[r["target_poi_id"]], index.target_close, index.eos] for r in records]
        try:
            teacher = teacher_forced(model, tokenizer, trie, prompts, targets)
            rows = [dict(subset=item["subset"], source_line=item["source_line"], key_sha256=item["key_sha256"],
                         target_tokens=target, teacher=observations, generation={})
                    for item, target, observations in zip(chunk, targets, teacher, strict=True)]
            for mode in MODES:
                recorder = RecordingModel(model if mode == "unconstrained" else ConstrainedGenerationModel(model, trie))
                metrics = ev.evaluate_chunk(records, variant=variant, model=recorder, tokenizer=tokenizer,
                                             template=template, index=index, batch_size=len(records), max_new_tokens=13)
                if len(recorder.beams) != len(rows) or (mode == "constrained" and metrics["candidate_count"] != metrics["valid_candidate_count"]):
                    raise ValueError("生成数量不符或约束出现非法 ID")
                for row, beams in zip(rows, recorder.beams, strict=True):
                    row["generation"][mode] = {**beams, "prefixes": beam_prefixes(row["target_tokens"], beams["tokens"])}
            progress["rows"].extend(rows)
        except RuntimeError as error:
            if not ev.is_cuda_oom(error) or progress["batch_size"] == 1:
                raise
            progress["batch_size"] //= 2
            error.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[{name}] OOM，重试当前未入账 batch={progress['batch_size']}", flush=True)
            continue
        write_json_atomic(progress_path, progress, overwrite=True)
        print(f"[{name}] {len(progress['rows'])}/{len(items)}", flush=True)
    write_json_atomic(path, {**progress, "status": "completed"})


def aggregate(rows: list[dict]) -> dict:
    teacher = {}
    for stage in ("s1", "s2", "s3", "dedup", "close"):
        values = [r["teacher"][stage] for r in rows if stage in r["teacher"]]
        if values:
            teacher[stage] = dict(count=len(values),
                top1=sum(v["rank"] == 1 for v in values)/len(values),
                top10=sum(v["rank"] <= 10 for v in values)/len(values),
                legal_top1=sum(v["legal_rank"] == 1 for v in values)/len(values),
                legal_top10=sum(v["legal_rank"] <= 10 for v in values)/len(values),
                nll=sum(v["nll"] for v in values)/len(values),
                legal_nll=sum(v["legal_nll"] for v in values)/len(values),
                legal_mass=sum(v["legal_mass"] for v in values)/len(values),
                mean_candidate_count=sum(v["candidate_count"] for v in values)/len(values),
                singleton_rate=sum(v["candidate_count"] == 1 for v in values)/len(values))
    generation = {mode: {stage: {str(k): sum(r["generation"][mode]["prefixes"][stage][str(k)] for r in rows)/len(rows)
                                           for k in (1, 10)}
                         for stage in ("gid6", "gid6_s1", "gid6_s1_s2", "gid6_sid3", "final_id")} for mode in MODES}
    failure = {}
    for mode in MODES:
        failure[mode] = {}
        for k in (1, 10):
            remaining = 1.0
            rates = {}
            for stage in ("gid6", "gid6_s1", "gid6_s1_s2", "gid6_sid3", "final_id"):
                rate = generation[mode][stage][str(k)]
                rates[stage] = remaining-rate
                remaining = rate
            rates["correct"] = remaining
            failure[mode][str(k)] = rates
    return dict(rows=len(rows), teacher=teacher, generation=generation, first_prefix_failure=failure)


def summarize(plan: dict, output: Path) -> dict:
    results = {}
    for name in PLANS:
        result = load_json(output / f"{name}_result.json")
        if result["status"] != "completed" or result["signature"] != signature(plan):
            raise ValueError("结果未完成或来源不同")
        rows = result["rows"]
        expected = load_json(Path(plan["data"][name]["path"]))["samples"]
        if [(r["subset"], r["key_sha256"]) for r in rows] != [(r["subset"], r["key_sha256"]) for r in expected]:
            raise ValueError("结果条数或业务键不同")
        results[name] = {subset: aggregate([r for r in rows if r["subset"] == subset]) for subset in SUBSETS}
    result = dict(status="completed", per_subset=plan["per_subset"], signature=signature(plan),
                  scope="paired_validation_diagnostic_not_full_metrics", results=results)
    write_json_atomic(output / "summary.json", result, overwrite=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="A0/A4 Qwen 逐层诊断：固定 Validation 配对抽样、正确前缀评分和原 Beam=10 双解码。")
    parser.add_argument("--stage", choices=("plan", "prepare", "worker", "summarize"), default="plan")
    parser.add_argument("--method", choices=tuple(PLANS))
    parser.add_argument("--per-subset", type=int, default=100, help="每组固定抽样数量，1—1000；默认总计 500 条")
    args = parser.parse_args()
    if not 1 <= args.per_subset <= 1000:
        parser.error("per-subset 必须为 1—1000")
    output = resolve(OUTPUT) / f"sample{args.per_subset}"
    if args.stage == "plan":
        inspect(args.per_subset)
        print(f"计划：每组 {args.per_subset} 条，共 {5*args.per_subset} 条；A0/A4 串行使用单卡，结果 {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            prepare(args.per_subset, output)
        else:
            plan = load_json(output / "plan.json")
            verify_plan(plan)
            if args.stage == "worker":
                if args.method is None:
                    parser.error("worker 必须指定 method")
                worker(plan, args.method, output)
            else:
                print(json.dumps(summarize(plan, output), ensure_ascii=False, indent=2))

"""Evaluate both frozen epoch-3 models on the complete final-day Test split."""

from __future__ import annotations

import csv
import fcntl
import gc
import itertools
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.sft import evaluation as ev
from qg_prqk.sft.evaluation_data import SftEvaluationError, load_json, resolve, signature
from qg_prqk.sft.evaluation_suite import validate_hardware
from qg_prqk.sft.full_test_data import (
    VARIANTS, JsonlRecordSequence, file_stat, parse_test_record, scan_paired_test, test_sources,
)
from qg_prqk.sft.training import project_root


def source_fingerprints() -> dict[str, str]:
    root = Path(__file__).parent
    extra = (root / "full_test_data.py", Path(__file__), root.parent / "commands/evaluate_sft_test.py")
    return {**ev.source_fingerprints(), **{str(p.relative_to(resolve("qg_prqk"))): sha256_file(p) for p in extra}}


def encode_test_record(record: Mapping[str, Any], *, variant: str, tokenizer: Any,
                       template: Any, index: ev.FinalIdIndex) -> tuple[list[int], int]:
    """Use the frozen Validation encoding contract, but require the Test split."""
    messages = record.get("messages")
    if (record.get("split"), record.get("identifier_variant")) != ("test", variant):
        raise SftEvaluationError("评测记录 split/variant 不一致")
    if not isinstance(messages, list) or len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        raise SftEvaluationError("Messages 必须为 user + assistant")
    expected = index.tokens_by_poi.get(record.get("target_poi_id"))
    if expected is None:
        raise SftEvaluationError("评测目标不在完整 active POI 库，禁止丢行")
    if not isinstance(record.get("requires_dedup"), bool) or record["requires_dedup"] != (len(expected) == index.base_width + 1):
        raise SftEvaluationError("末位 Dedup 与训练标签契约不一致")
    source, target = ev.encode_prompt_like_training(
        tokenizer=tokenizer, template=template, user_content=messages[0]["content"],
        target_content=messages[1]["content"], cutoff_len=1024,
    )
    if tuple(target) != (index.target_open, *expected, index.target_close):
        raise SftEvaluationError("目标序列不是该 POI 的精确 Final ID")
    return source, index.row_by_tokens[expected]


def evaluate_test_chunk(records: Sequence[Mapping[str, Any]], *, variant: str, model: Any,
                        tokenizer: Any, template: Any, index: ev.FinalIdIndex,
                        batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    """Keep the existing beam-generation math unchanged for full Test chunks."""
    import torch

    examples = [encode_test_record(r, variant=variant, tokenizer=tokenizer, template=template, index=index)
                for r in records]
    metrics = ev.empty_metrics()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        input_ids, attention_mask = ev.pad_prompts([x[0] for x in batch], tokenizer.pad_token_id, model.device)
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
            candidates = [ev.parse_candidate(sequence, float(score), index) for sequence, score in ranked]
            ev.update_metrics(metrics, target_row=target_row, candidates=candidates)
        del generated, input_ids, attention_mask
    return metrics


def write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if load_json(path) != payload:
            raise SftEvaluationError(f"已有产物来源/协议不同，拒绝覆盖：{path}")
    else:
        write_json_atomic(path, payload)


def prepare_plans(config: Mapping[str, Any]) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Check all Test rows and 100 formatted samples before any GPU work."""
    root = resolve(config["output_root"])
    print("[Test 预检] 冻结两份 epoch-3 模型，不进行 checkpoint 选择", flush=True)
    checkpoints = {}
    for variant in VARIANTS:
        checkpoints[variant] = ev.validate_checkpoint(config, variant)
        if checkpoints[variant]["files"]["model.safetensors"]["sha256"] != config["checkpoint_sha256"][variant]:
            raise SftEvaluationError("模型不是已经完成 Validation 评测的冻结 checkpoint")
    data = scan_paired_test(test_sources(config), expected_rows=config["source_rows"])
    write_immutable(root / "test_data_manifest.json", data)
    plans = {}
    for variant in VARIANTS:
        print(f"[{variant}/test] 核验完整 Final ID 与前 100 条训练同模板输入", flush=True)
        checkpoint = checkpoints[variant]
        tokenizer, template = ev.load_lf_tokenizer_and_template(Path(checkpoint["path"]), project_root=project_root())
        index = ev.build_final_id_index(config, variant, tokenizer)
        maximum = 0
        checked = 0
        with Path(data["sources"][variant]["file"]).open("rb") as stream:
            for line_number, line in enumerate(itertools.islice(stream, 100), 1):
                record = parse_test_record(line, line_number=line_number, variant=variant)
                prompt, _ = encode_test_record(record, variant=variant, tokenizer=tokenizer, template=template, index=index)
                maximum = max(maximum, len(prompt))
                checked += 1
        plan = dict(schema_version="qg-prqk-full-test-plan-v1", config=dict(config), variant=variant,
                    checkpoint=checkpoint, data=data, source_sha256=source_fingerprints(),
                    preflight=dict(poi_count=index.poi_count, formatted_samples=checked,
                                   sampled_max_prompt_tokens=maximum, device_count=2,
                                   remaining_rows_checked_per_chunk=True))
        path = root / variant / "plan.json"
        write_immutable(path, plan)
        plans[variant] = (path, plan)
        print(f"[{variant}/test] 全量 {data['rows']:,} 条契约通过；计划：{path}", flush=True)
        del tokenizer, template, index
        gc.collect()
    return plans


def verify_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != "qg-prqk-full-test-plan-v1" or plan.get("variant") not in VARIANTS:
        raise SftEvaluationError("不是全量 Test 计划")
    if (plan["data"]["split"], plan["data"]["date"], plan["data"]["rows"]) != ("test", "2026-07-14", 606682):
        raise SftEvaluationError("Test 计划的日期/行数不正确")
    if source_fingerprints() != plan["source_sha256"]:
        raise SftEvaluationError("Test 源码变化，禁止与旧断点混用")
    for name, expected in plan["checkpoint"]["files"].items():
        if any(file_stat(Path(plan["checkpoint"]["path"]) / name)[k] != expected[k] for k in ("bytes", "mtime_ns")):
            raise SftEvaluationError("checkpoint 在父进程完整哈希核验后变化")
    source = plan["data"]["sources"][plan["variant"]]
    if any(file_stat(Path(source["file"]))[k] != source[k] for k in ("bytes", "mtime_ns")):
        raise SftEvaluationError("Test 文件在父进程预检后变化")


def result_directory(plan: Mapping[str, Any], smoke_limit: int | None) -> Path:
    base = resolve(plan["config"]["output_root"]) / plan["variant"]
    return base / "results" if smoke_limit is None else base / "smoke" / str(smoke_limit)


def run_config(plan: Mapping[str, Any], smoke_limit: int | None) -> dict[str, Any]:
    return dict(plan_signature=signature(plan), variant=plan["variant"], split="test", date="2026-07-14",
                sample_count=smoke_limit or plan["data"]["rows"], smoke_limit=smoke_limit,
                epoch=3.0, decoding="unconstrained_beam_search", num_beams=10, cutoff_len=1024,
                invalid_ids_keep_original_rank=True)


def validate_result(result: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if (result.get("status") != "completed" or result.get("config") != config
            or result.get("signature") != signature(config)
            or result.get("metrics", {}).get("sample_count") != config["sample_count"]):
        raise SftEvaluationError("Test 结果不完整或来源不同，拒绝混用")


def evaluate_test(plan: Mapping[str, Any], *, smoke_limit: int | None = None) -> dict[str, Any]:
    """Resume exact-POI scoring from completed chunks; never drop malformed rows."""
    import torch

    verify_plan(plan)
    variant, config = plan["variant"], plan["config"]
    settings = run_config(plan, smoke_limit)
    target_rows = settings["sample_count"]
    directory = result_directory(plan, smoke_limit)
    result_path, progress_path = directory / "result.json", directory / "progress.json"
    if result_path.exists():
        result = load_json(result_path)
        validate_result(result, settings)
        print(f"[{variant}/test] 已完成 {target_rows:,} 条，跳过", flush=True)
        return result
    progress = dict(signature=signature(settings), next_line=0, metrics=ev.empty_metrics(),
                    inference_seconds=0.0, actual_batch_size=config["decoding"]["batch_size"], peak_memory_bytes=0)
    if progress_path.exists():
        progress = load_json(progress_path)
        done = progress.get("next_line", -1)
        if (progress.get("signature") != signature(settings) or not 0 <= done <= target_rows
                or done != progress.get("metrics", {}).get("sample_count")
                or progress["metrics"].get("candidate_count") != done * 10
                or not 1 <= progress.get("actual_batch_size", 0) <= config["decoding"]["batch_size"]):
            raise SftEvaluationError("Test 断点签名/累计计数/batch 不一致")
    print(f"[{variant}/test] 初始化：只读索引、Tokenizer、Final ID 和模型；续跑位置 {progress['next_line']:,}", flush=True)
    source = plan["data"]["sources"][variant]
    records = JsonlRecordSequence(Path(source["file"]), variant=variant,
                                   expected_rows=plan["data"]["rows"], expected_sha256=source["sha256"])
    tokenizer, template = ev.load_lf_tokenizer_and_template(Path(plan["checkpoint"]["path"]), project_root=project_root())
    index = ev.build_final_id_index(config, variant, tokenizer)
    model = ev.load_generation_model(Path(plan["checkpoint"]["path"]), expected_vocab_size=len(tokenizer))
    batch_size = progress["actual_batch_size"]
    torch.cuda.reset_peak_memory_stats(model.device)
    directory.mkdir(parents=True, exist_ok=True)
    while progress["next_line"] < target_rows:
        start = progress["next_line"]
        stop = min(start + config["decoding"]["chunk_size"], target_rows)
        started = time.perf_counter()
        try:
            metrics = evaluate_test_chunk(records[start:stop], variant=variant, model=model,
                                          tokenizer=tokenizer, template=template, index=index,
                                          batch_size=batch_size, max_new_tokens=config["variants"][variant]["max_new_tokens"])
        except RuntimeError as error:
            if not ev.is_cuda_oom(error) or batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"[{variant}/test] CUDA OOM，batch 降为 {batch_size}，重试未入账 chunk", flush=True)
            error.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
            continue
        ev.merge_metrics(progress["metrics"], metrics)
        progress.update(next_line=stop, actual_batch_size=batch_size,
                        inference_seconds=progress["inference_seconds"] + time.perf_counter() - started,
                        peak_memory_bytes=max(progress["peak_memory_bytes"], torch.cuda.max_memory_allocated(model.device)))
        write_json_atomic(progress_path, progress, overwrite=True)
        speed = stop / max(progress["inference_seconds"], 1e-9)
        print(f"[{variant}/test] {stop:,}/{target_rows:,} ({100 * stop / target_rows:.2f}%)，"
              f"batch={batch_size}，{speed:.2f} 条/秒，剩余推理约 {(target_rows-stop)/speed/60:.1f} 分钟", flush=True)
    result = dict(schema_version="qg-prqk-full-test-result-v1", status="completed", config=settings,
                  signature=signature(settings), metrics=ev.finalize_metrics(progress["metrics"]),
                  performance={k: progress[k] for k in ("inference_seconds", "actual_batch_size", "peak_memory_bytes")})
    write_json_atomic(result_path, result)
    print(f"[{variant}/test] 完成，结果：{result_path}", flush=True)
    return result


def forward_log(stream: TextIO, log: TextIO) -> None:
    """Tee worker lines into its file and the platform's main console."""
    for line in stream:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)


def schedule_tests(plans: Mapping[str, tuple[Path, Mapping[str, Any]]], *, smoke_limit: int | None) -> None:
    gpu_ids = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(gpu_ids) or set(plans) != set(VARIANTS):
        raise SftEvaluationError("全量 Test 要求两个不同可见 GPU 和两个固定分支")
    log_root = resolve("qg_prqk/outputs/run_control/eval_a4_dual_epoch3_full_test_2x6000d")
    if smoke_limit is not None:
        log_root /= f"smoke{smoke_limit}"
    log_root.mkdir(parents=True, exist_ok=True)
    running = {}
    last_heartbeat = time.monotonic()
    try:
        for gpu, variant in zip(gpu_ids, VARIANTS, strict=True):
            plan_path, _ = plans[variant]
            temporary = resolve("qg_prqk/outputs/tmp/qtg" if variant == VARIANTS[0] else "qg_prqk/outputs/tmp/qtn")
            temporary.mkdir(parents=True, exist_ok=True)
            if len(os.fsencode(temporary)) > 64 or not os.access(temporary, os.W_OK):
                raise SftEvaluationError("Test worker TMPDIR 不可写或超过 64 字节")
            command = [sys.executable, str(resolve("qg_prqk/scripts/qg_prqk.py")), "evaluate-sft-test", "--plan", str(plan_path)]
            if smoke_limit is not None:
                command += ["--smoke-limit", str(smoke_limit)]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1",
                   "TMPDIR": str(temporary), "TMP": str(temporary), "TEMP": str(temporary)}
            log = (log_root / f"{variant}.console.log").open("a", encoding="utf-8")
            try:
                process = subprocess.Popen(command, cwd=project_root(), env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1)
            except BaseException:
                log.close()
                raise
            thread = threading.Thread(target=forward_log, args=(process.stdout, log), daemon=True)
            running[variant] = (process, thread, log)
            thread.start()
            (log_root / f"{variant}.pid").write_text(f"{process.pid}\n")
            print(f"[{variant}/test] 启动 GPU {gpu}；每 500 条转发进度到平台主日志", flush=True)
        while running:
            for variant, (process, thread, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                thread.join()
                process.stdout.close()
                log.close()
                (log_root / f"{variant}.exit").write_text(f"{code}\n")
                del running[variant]
                if code != 0:
                    raise SftEvaluationError(f"{variant} Test 失败，exit={code}；保留断点，修复后同命令续跑")
            if running and time.monotonic() - last_heartbeat >= 30:
                print(f"[Test 监督] 正在运行：{', '.join(running)}；初始化期间无样本进度属正常", flush=True)
                last_heartbeat = time.monotonic()
            if running:
                time.sleep(1)
    finally:
        for variant, (process, thread, log) in running.items():
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            thread.join()
            process.stdout.close()
            log.close()
            (log_root / f"{variant}.exit").write_text(f"{process.returncode}\n")


def summarize_tests(plans: Mapping[str, tuple[Path, Mapping[str, Any]]], *, smoke_limit: int | None) -> Path:
    rows = []
    for variant in VARIANTS:
        plan = plans[variant][1]
        path = result_directory(plan, smoke_limit) / "result.json"
        result = load_json(path)
        validate_result(result, run_config(plan, smoke_limit))
        rows.append(dict(variant=variant, **result["metrics"], result_file=str(path)))
    root = resolve(plans[VARIANTS[0]][1]["config"]["output_root"])
    if smoke_limit is not None:
        root /= f"smoke/{smoke_limit}"
    metrics = ("hr@1", "hr@3", "hr@5", "hr@10", "ndcg@3", "ndcg@5", "ndcg@10", "mrr@10", "valid_id_rate")
    summary = dict(status="completed", split="test", date="2026-07-14", epoch=3.0, num_beams=10,
                   decoding="unconstrained_beam_search", smoke_limit=smoke_limit, rows=rows,
                   delta_gid_minus_nogid={m: rows[0][m] - rows[1][m] for m in metrics},
                   business_keys_sha256=plans[VARIANTS[0]][1]["data"]["business_keys_sha256"])
    write_immutable(root / "summary.json", summary)
    path = root / "summary.csv"
    temporary = path.with_suffix(".csv.writing")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "sample_count", *metrics])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
    os.replace(temporary, path)
    return root / "summary.json"


def run_test_suite(config: Mapping[str, Any], *, dry_run: bool, smoke_limit: int | None) -> int:
    root = resolve(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".suite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SftEvaluationError("同一全量 Test 已有进程，请勿重复启动") from error
        plans = prepare_plans(config)
        if dry_run:
            print("全量 Test CPU 预检通过：两版各 606,682 条；未启动 GPU 推理。", flush=True)
            return 0
        validate_hardware()
        schedule_tests(plans, smoke_limit=smoke_limit)
        print(f"两版全量 Test 评测完成：{summarize_tests(plans, smoke_limit=smoke_limit)}", flush=True)
    return 0

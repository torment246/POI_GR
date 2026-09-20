"""Preflight and schedule both SFT branches on one two-GPU node."""

from __future__ import annotations

import csv
import fcntl
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

from qg_prqk.artifacts import write_json_atomic
from qg_prqk.sft.evaluation_data import (
    SUBSETS, SftEvaluationError, load_json, prepare_variant, read_records, resolve, signature,
)
from qg_prqk.sft.evaluation import (
    build_final_id_index, encode_record, load_lf_tokenizer_and_template,
    source_fingerprints, validate_checkpoint,
)
from qg_prqk.sft.training import project_root


GPU_COUNT = 2
VARIANTS = ("a4_gid_parent", "a4_nogid")
PLAN_FILENAME = "plan_2x6000d.json"


def prepare_plan(config: Mapping[str, Any], variant: str) -> dict[str, Any]:
    print(f"[{variant}] 核验 epoch-3 checkpoint 与共同词表", flush=True)
    checkpoint = validate_checkpoint(config, variant)
    print(f"[{variant}] 对齐五个冻结 Validation 子集；已有缓存则只核验缓存", flush=True)
    data = prepare_variant(config, variant)
    tokenizer, template = load_lf_tokenizer_and_template(Path(checkpoint["path"]), project_root=project_root())
    index = build_final_id_index(config, variant, tokenizer)
    max_prompt = 0
    for subset in SUBSETS:
        records = read_records(Path(data["outputs"][subset]["file"]))
        # Full valid had already passed the immutable 1024-token training preflight.
        # Every remaining row is also checked when its generation chunk is encoded.
        for record in records[:8]:
            prompt, _ = encode_record(record, variant=variant, tokenizer=tokenizer, template=template, index=index)
            max_prompt = max(max_prompt, len(prompt))
        print(f"[{variant}/{subset}] {len(records)} 条，固定业务主键/目标与样例 Token 检查通过", flush=True)
    return dict(schema_version="qg-prqk-sft-evaluation-plan-v1", config=dict(config), variant=variant,
                checkpoint=checkpoint, data=data, source_sha256=source_fingerprints(),
                preflight=dict(poi_count=index.poi_count, checked_formatted_samples=40,
                               sampled_max_prompt_tokens=max_prompt, device_count=GPU_COUNT,
                               no_test_read=True, resampling=False))


def validate_hardware() -> None:
    import torch

    count = torch.cuda.device_count()
    if count != GPU_COUNT:
        raise SftEvaluationError(f"本套件需要恰好 2 张可见 RTX 6000D，实际 {count}")
    for gpu in range(count):
        name = torch.cuda.get_device_name(gpu).upper()
        free, total = torch.cuda.mem_get_info(gpu)
        print(f"GPU {gpu}: {name}, 空闲 {free / 1024**3:.1f}/{total / 1024**3:.1f} GiB", flush=True)
        if not ("6000D" in name or "RTX PRO 6000" in name) or free < 36 * 1024**3:
            raise SftEvaluationError("要求两张 RTX 6000D，每张至少 36 GiB 空闲；双分支共用一个调度进程")


def worker_command(plan_path: Path, subset: str, smoke_limit: int | None) -> list[str]:
    command = [sys.executable, str(resolve("qg_prqk/scripts/qg_prqk.py")), "evaluate-sft",
               "--worker-subset", subset, "--plan", str(plan_path)]
    if smoke_limit is not None:
        command += ["--smoke-limit", str(smoke_limit)]
    return command


def schedule_workers(plans: Mapping[str, tuple[Path, Mapping[str, Any]]], *, smoke_limit: int | None) -> None:
    """Keep at most one child per GPU; abort only this suite's children on failure."""
    log_root = resolve("qg_prqk/outputs/run_control/eval_a4_dual_epoch3_2x6000d")
    if smoke_limit is not None:
        log_root /= f"smoke{smoke_limit}"
    gpu_ids = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")]
    if len(gpu_ids) != GPU_COUNT or len(set(gpu_ids)) != GPU_COUNT or not all(gpu_ids):
        raise SftEvaluationError("CUDA_VISIBLE_DEVICES 必须声明两个不同设备")
    if not plans or any(variant not in VARIANTS or plan["variant"] != variant
                        for variant, (_, plan) in plans.items()):
        raise SftEvaluationError("评测计划与分支不一致")
    # Interleave branches in one shared queue; neither branch owns a GPU permanently.
    pending = [(variant, subset) for subset in SUBSETS for variant in plans]
    running: dict[int, tuple[subprocess.Popen, Any, str, str, Path]] = {}
    try:
        while pending or running:
            for gpu in range(GPU_COUNT):
                if gpu in running or not pending:
                    continue
                variant, subset = pending.pop(0)
                plan_path, _ = plans[variant]
                log_dir = log_root / variant
                log_dir.mkdir(parents=True, exist_ok=True)
                tag = "qeg2" if variant == "a4_gid_parent" else "qen2"
                temporary = resolve(f"qg_prqk/outputs/tmp/{tag}{gpu}")
                temporary.mkdir(parents=True, exist_ok=True)
                if len(os.fsencode(temporary)) > 64 or not os.access(temporary, os.W_OK):
                    raise SftEvaluationError("任务 TMPDIR 不可写或超过 64 字节")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids[gpu], "TMPDIR": str(temporary),
                       "TEMP": str(temporary), "TMP": str(temporary), "PYTHONUNBUFFERED": "1",
                       "TOKENIZERS_PARALLELISM": "false", "PYTHONNOUSERSITE": "1"}
                handle = (log_dir / f"{subset}.console.log").open("a", encoding="utf-8")
                try:
                    process = subprocess.Popen(worker_command(plan_path, subset, smoke_limit), env=env,
                                               cwd=project_root(), stdout=handle, stderr=subprocess.STDOUT)
                except BaseException:
                    handle.close()
                    raise
                running[gpu] = (process, handle, variant, subset, log_dir)
                (log_dir / f"{subset}.pid").write_text(f"{process.pid}\n")
                print(f"[{variant}/{subset}] 启动 GPU {gpu_ids[gpu]}，日志 {log_dir / (subset + '.console.log')}", flush=True)
            for gpu, (process, handle, variant, subset, log_dir) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                (log_dir / f"{subset}.exit").write_text(f"{code}\n")
                del running[gpu]
                if code != 0:
                    raise SftEvaluationError(f"{variant}/{subset} 评测失败，exit={code}；修复后用原命令从 chunk 断点继续")
                print(f"[{variant}/{subset}] 完成", flush=True)
            if running:
                time.sleep(1)
    finally:
        for process, handle, _, subset, log_dir in running.values():
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            handle.close()
            (log_dir / f"{subset}.exit").write_text(f"{process.returncode}\n")


def summarize(plan: Mapping[str, Any], *, smoke_limit: int | None = None) -> Path:
    base = resolve(plan["config"]["output_root"]) / plan["variant"]
    result_root = base / ("smoke" if smoke_limit else "results")
    rows = []
    expected_rows = smoke_limit or plan["config"]["subset_size"]
    for subset in SUBSETS:
        path = result_root / subset / "result.json"
        result = load_json(path)
        config, metrics = result.get("config", {}), result.get("metrics", {})
        if result.get("status") != "completed" or config.get("plan_signature") != signature(plan) or config.get("subset") != subset:
            raise SftEvaluationError("汇总结果不完整或来源不同")
        if metrics.get("sample_count") != expected_rows or config.get("num_beams") != 10 or config.get("decoding") != "unconstrained_beam_search" or config.get("smoke_limit") != smoke_limit:
            raise SftEvaluationError("汇总样本数/解码协议不一致")
        rows.append(dict(variant=plan["variant"], subset=subset, **metrics, result_file=str(path)))
    metric_names = ("hr@1", "hr@3", "hr@5", "hr@10", "ndcg@10", "mrr@10", "valid_id_rate")
    macro = {metric: sum(row[metric] for row in rows[1:]) / 4 for metric in metric_names}
    payload = dict(schema_version="qg-prqk-sft-evaluation-summary-v1", status="completed",
                   variant=plan["variant"], epoch=3.0, decoding="unconstrained_beam_search", num_beams=10,
                   smoke_limit=smoke_limit, rows=rows, generalization_macro_average=macro,
                   fixed10k_excluded_from_macro=True, plan_signature=signature(plan))
    output = result_root / "summary.json"
    if output.exists() and load_json(output) != payload:
        raise SftEvaluationError("已有汇总不同，拒绝覆盖")
    if not output.exists():
        write_json_atomic(output, payload)
    csv_path = result_root / "summary.csv"
    temporary = csv_path.with_suffix(".csv.writing")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "subset", "sample_count", *metric_names])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    os.replace(temporary, csv_path)
    return output


def run_suite(config: Mapping[str, Any], variant: str, *, dry_run: bool, smoke_limit: int | None) -> int:
    if variant not in (*VARIANTS, "both"):
        raise SftEvaluationError(f"未知评测分支：{variant}")
    variants = VARIANTS if variant == "both" else (variant,)
    root = resolve(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    plans = {}
    with ExitStack() as stack:
        # Also exclude concurrent single-branch invocations on this shared node.
        lock_paths = [root / ".two_gpu_suite.lock"]
        for name in variants:
            (root / name).mkdir(parents=True, exist_ok=True)
            lock_paths.append(root / name / ".suite.lock")
        for path in lock_paths:
            lock = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SftEvaluationError("同一双卡套件或分支已有进程，请勿重复启动") from error
        for name in variants:
            plan = prepare_plan(config, name)
            # Preserve the earlier four-GPU preflight; aligned data remains shared.
            plan_path = root / name / PLAN_FILENAME
            if plan_path.exists() and load_json(plan_path) != plan:
                raise SftEvaluationError("已保存的双卡评测 plan 不同，禁止覆盖；请用新版本目录")
            if not plan_path.exists():
                write_json_atomic(plan_path, plan)
            plans[name] = (plan_path, plan)
            print(f"[{name}] 双卡计划：{plan_path}", flush=True)
        if dry_run:
            print(f"预检通过：{len(plans)} 个分支 × 5 × 10,000 条，最多 2 个单卡 worker，未启动 GPU 推理。", flush=True)
            return 0
        validate_hardware()
        schedule_workers(plans, smoke_limit=smoke_limit)
        for _, plan in plans.values():
            print(f"五项评测全部完成：{summarize(plan, smoke_limit=smoke_limit)}", flush=True)
    return 0

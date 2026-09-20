"""Frozen-data constrained decoding, two-GPU scheduling and paired summaries."""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.sft import evaluation as ev
from qg_prqk.sft.constrained_decoding import (
    ConstrainedGenerationModel,
    FinalIdPrefixIndex,
)
from qg_prqk.sft.evaluation_data import (
    SUBSETS,
    SftEvaluationError,
    business_key,
    current_context,
    keys_sha256,
    load_json,
    read_records,
    require_hash,
    resolve,
    signature,
)
from qg_prqk.sft.evaluation_suite import VARIANTS, validate_hardware
from qg_prqk.sft.training import project_root

DEFAULT_CONFIG = (
    "qg_prqk/configs/sft/evaluation_epoch3_constrained_fixed10k_generalization_v1.yaml"
)
MODE = "full_catalog_constrained_beam_search"
METRICS = ("hr@1", "hr@3", "hr@5", "hr@10", "ndcg@10", "mrr@10", "valid_id_rate")


def source_fingerprints() -> dict[str, str]:
    extra = (
        Path(__file__),
        Path(__file__).with_name("constrained_decoding.py"),
        resolve("qg_prqk/scripts/evaluate_sft_constrained.py"),
    )
    return {
        **ev.source_fingerprints(),
        **{str(p.relative_to(resolve("qg_prqk"))): sha256_file(p) for p in extra},
    }


def isolated_output(value: str) -> Path:
    path = resolve(value)
    if (
        not path.is_relative_to(resolve("qg_prqk/outputs"))
        or "constrained" not in path.name
    ):
        raise SftEvaluationError("输出必须是 qg_prqk/outputs 下独立的 constrained 目录")
    return path


def immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if load_json(path) != payload:
            raise SftEvaluationError(f"已有产物不同，拒绝覆盖：{path}")
    else:
        write_json_atomic(path, dict(payload))


def prepare_plans(config_path: Path) -> dict[str, dict[str, Any]]:
    """Reuse the exact ten frozen JSONL files; never resample or read Test."""
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != "qg-prqk-constrained-evaluation-v1" or set(
        config["reference_plans"]
    ) != set(VARIANTS):
        raise SftEvaluationError("约束解码配置必须声明两个冻结分支")
    output = isolated_output(config["output_root"])
    logs = isolated_output(config["run_control"])
    plans = {}
    paired_rows = {}
    for variant in VARIANTS:
        spec = config["reference_plans"][variant]
        path = resolve(spec["path"])
        require_hash(path, spec["sha256"])
        reference = load_json(path)
        ev.verify_worker_inputs(reference)
        old_config = reference["config"]
        if (
            reference["variant"] != variant
            or old_config["split"] != "valid"
            or old_config["subset_size"] != 10000
        ):
            raise SftEvaluationError("来源不是该分支的冻结 Validation 五组 10k")
        if output == resolve(old_config["output_root"]) or output.is_relative_to(
            resolve(old_config["output_root"])
        ):
            raise SftEvaluationError("不得写入旧无约束目录")
        run_config = copy.deepcopy(old_config)
        run_config["output_root"] = str(output)
        run_config["decoding"]["mode"] = "constrained"
        checkpoint = Path(reference["checkpoint"]["path"])
        tokenizer, template = ev.load_lf_tokenizer_and_template(
            checkpoint, project_root=project_root()
        )
        index = ev.build_final_id_index(run_config, variant, tokenizer)
        trie = FinalIdPrefixIndex(index)
        if trie.max_length != run_config["variants"][variant]["max_new_tokens"]:
            raise SftEvaluationError("全目录路径长度与冻结生成上限不一致")
        baseline = {}
        all_keys = set()
        for subset in SUBSETS:
            data = reference["data"]["outputs"][subset]
            require_hash(Path(data["file"]), data["sha256"])
            records = read_records(Path(data["file"]))
            keys = [business_key(row) for row in records]
            if (
                len(keys) != 10000
                or len(set(keys)) != 10000
                or keys_sha256(keys) != data["business_keys_sha256"]
                or all_keys.intersection(keys)
            ):
                raise SftEvaluationError("五组数据的数量/顺序/唯一性/互斥不一致")
            all_keys.update(keys)
            pairs = [
                (business_key(r), r["target_poi_id"], current_context(r))
                for r in records
            ]
            if variant == VARIANTS[0]:
                paired_rows[subset] = pairs
            elif pairs != paired_rows.pop(subset):
                raise SftEvaluationError("GID/NoGID 的业务主键、目标或 CURRENT 不一致")
            for record in records[:8]:
                ev.encode_record(
                    record,
                    variant=variant,
                    tokenizer=tokenizer,
                    template=template,
                    index=index,
                )
            result_path = (
                resolve(old_config["output_root"])
                / variant
                / "results"
                / subset
                / "result.json"
            )
            result = load_json(result_path)
            if (
                result["status"] != "completed"
                or result["config"]["plan_signature"] != signature(reference)
                or result["metrics"]["sample_count"] != 10000
            ):
                raise SftEvaluationError("对照无约束结果尚未完成或计划不同")
            baseline[subset] = dict(
                path=str(result_path), sha256=sha256_file(result_path)
            )
            print(
                f"[{variant}/{subset}] 10,000 条冻结样本已核验；不重新抽样", flush=True
            )
        plan = dict(
            schema_version="qg-prqk-constrained-plan-v1",
            variant=variant,
            config=run_config,
            checkpoint=reference["checkpoint"],
            data=reference["data"],
            reference_plan=dict(path=str(path), sha256=spec["sha256"]),
            baseline_results=baseline,
            source_sha256=source_fingerprints(),
            config_sha256=sha256_file(config_path),
            run_control=str(logs),
            trie=dict(
                terminal_paths=index.poi_count,
                max_path_length=trie.max_length,
                includes_dedup=True,
                includes_wrapper_and_eos=True,
                query_or_geo_pruning=False,
                storage="sorted_terminal_paths_bounded_prefix_cache",
            ),
            preflight=dict(
                no_test_read=True, resampling=False, checked_formatted_samples=40
            ),
        )
        immutable_json(output / variant / "plan.json", plan)
        plans[variant] = plan
        del trie, index, tokenizer, template, records
        gc.collect()
    return plans


def verify_plan(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("schema_version") != "qg-prqk-constrained-plan-v1"
        or plan.get("variant") not in VARIANTS
    ):
        raise SftEvaluationError("不是约束解码计划")
    require_hash(Path(plan["reference_plan"]["path"]), plan["reference_plan"]["sha256"])
    ev.verify_worker_inputs(load_json(Path(plan["reference_plan"]["path"])))
    if (
        source_fingerprints() != plan["source_sha256"]
        or plan["config"]["decoding"]["mode"] != "constrained"
    ):
        raise SftEvaluationError("约束解码源码或协议变化，禁止混合断点")


def evaluate_subset(
    plan: Mapping[str, Any], subset: str, smoke_limit: int | None
) -> dict[str, Any]:
    """Adapt the existing chunk loop; reuse its encoding, generation and scoring."""
    import torch

    verify_plan(plan)
    config, variant = plan["config"], plan["variant"]
    data = plan["data"]["outputs"][subset]
    require_hash(Path(data["file"]), data["sha256"])
    records = read_records(Path(data["file"]))
    if len(records) != 10000:
        raise SftEvaluationError("子集必须完整包含 10,000 条")
    if smoke_limit is not None:
        records = records[:smoke_limit]
    directory = (
        resolve(config["output_root"])
        / variant
        / (f"smoke{smoke_limit}" if smoke_limit else "results")
        / subset
    )
    settings = dict(
        plan_signature=signature(plan),
        subset=subset,
        variant=variant,
        decoding=MODE,
        epoch=3.0,
        num_beams=10,
        cutoff_len=1024,
        data=data,
        smoke_limit=smoke_limit,
        sample_count=len(records),
        invalid_ids_keep_original_rank=True,
    )
    run_signature = signature(settings)
    result_path, progress_path = directory / "result.json", directory / "progress.json"
    if result_path.exists():
        result = load_json(result_path)
        if (
            result.get("signature") != run_signature
            or result.get("status") != "completed"
            or result["metrics"]["sample_count"] != len(records)
            or result["metrics"]["valid_id_rate"] != 1
        ):
            raise SftEvaluationError("既有结果不完整或协议不匹配")
        print(f"[{variant}/{subset}] 已完成，复用", flush=True)
        return result
    progress = dict(
        signature=run_signature,
        next_line=0,
        metrics=ev.empty_metrics(),
        inference_seconds=0.0,
        actual_batch_size=config["decoding"]["batch_size"],
        peak_memory_bytes=0,
    )
    if progress_path.exists():
        progress = load_json(progress_path)
        done = progress["next_line"]
        if (
            progress["signature"] != run_signature
            or not 0 <= done <= len(records)
            or progress["metrics"]["sample_count"] != done
            or progress["metrics"]["candidate_count"] != done * 10
            or progress["metrics"]["valid_candidate_count"] != done * 10
        ):
            raise SftEvaluationError("断点签名、行数或合法候选计数不一致")
    tokenizer, template = ev.load_lf_tokenizer_and_template(
        Path(plan["checkpoint"]["path"]), project_root=project_root()
    )
    index = ev.build_final_id_index(config, variant, tokenizer)
    trie = FinalIdPrefixIndex(index)
    model = ConstrainedGenerationModel(
        ev.load_generation_model(
            Path(plan["checkpoint"]["path"]), expected_vocab_size=len(tokenizer)
        ),
        trie,
    )
    batch_size = progress["actual_batch_size"]
    torch.cuda.reset_peak_memory_stats(model.device)
    while progress["next_line"] < len(records):
        start = progress["next_line"]
        stop = min(start + config["decoding"]["chunk_size"], len(records))
        started = time.perf_counter()
        try:
            metrics = ev.evaluate_chunk(
                records[start:stop],
                variant=variant,
                model=model,
                tokenizer=tokenizer,
                template=template,
                index=index,
                batch_size=batch_size,
                max_new_tokens=config["variants"][variant]["max_new_tokens"],
            )
        except RuntimeError as error:
            if not ev.is_cuda_oom(error) or batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            error.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"[{variant}/{subset}] OOM：batch={batch_size}，重试未入账 chunk",
                flush=True,
            )
            continue
        if (
            metrics["sample_count"] != stop - start
            or metrics["candidate_count"] != (stop - start) * 10
            or metrics["valid_candidate_count"] != metrics["candidate_count"]
        ):
            raise SftEvaluationError(
                "约束解码仍出现非法或缺失候选，停止且不写入当前 chunk"
            )
        ev.merge_metrics(progress["metrics"], metrics)
        progress.update(
            next_line=stop,
            actual_batch_size=batch_size,
            inference_seconds=progress["inference_seconds"]
            + time.perf_counter()
            - started,
            peak_memory_bytes=max(
                progress["peak_memory_bytes"],
                torch.cuda.max_memory_allocated(model.device),
            ),
        )
        write_json_atomic(progress_path, progress, overwrite=True)
        print(
            f"[{variant}/{subset}] {stop}/{len(records)}，batch={batch_size}，合法 ID=100%",
            flush=True,
        )
    result = dict(
        schema_version="qg-prqk-constrained-result-v1",
        status="completed",
        config=settings,
        signature=run_signature,
        metrics=ev.finalize_metrics(progress["metrics"]),
        performance={
            key: progress[key]
            for key in ("inference_seconds", "actual_batch_size", "peak_memory_bytes")
        },
    )
    immutable_json(result_path, result)
    del model, trie, index, tokenizer, template
    gc.collect()
    torch.cuda.empty_cache()
    return result


def schedule(
    plans: Mapping[str, Mapping[str, Any]],
    smoke_limit: int | None,
    fixed_only: bool = False,
) -> None:
    """Run one isolated branch per GPU and forward worker progress to platform logs."""
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) != 2 or len(set(devices)) != 2 or not all(devices):
        raise SftEvaluationError("必须声明两张不同的 CUDA GPU")
    processes = []

    def forward(process: Any, path: Path) -> int:
        with path.open("a", encoding="utf-8") as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
        code = process.wait()
        path.with_suffix(".exit").write_text(f"{code}\n")
        return code

    try:
        for gpu, variant in enumerate(VARIANTS):
            plan = plans[variant]
            tag = f"smoke{smoke_limit}" if smoke_limit else "full"
            log_dir = Path(plan["run_control"]) / tag
            log_dir.mkdir(parents=True, exist_ok=True)
            temporary = resolve(f"qg_prqk/outputs/tmp/qec{gpu}")
            temporary.mkdir(parents=True, exist_ok=True)
            if len(os.fsencode(temporary)) > 64 or not os.access(temporary, os.W_OK):
                raise SftEvaluationError("TMPDIR 不可写或超过 64 字节")
            command = [
                sys.executable,
                str(resolve("qg_prqk/scripts/evaluate_sft_constrained.py")),
                "--worker-variant",
                variant,
                "--plan",
                str(resolve(plan["config"]["output_root"]) / variant / "plan.json"),
            ]
            if smoke_limit:
                command += ["--smoke-limit", str(smoke_limit)]
            if fixed_only:
                command += ["--fixed-only"]
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": devices[gpu],
                "TMPDIR": str(temporary),
                "TMP": str(temporary),
                "TEMP": str(temporary),
                "PYTHONUNBUFFERED": "1",
            }
            process = subprocess.Popen(
                command,
                env=env,
                cwd=project_root(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            path = log_dir / f"{variant}.console.log"
            path.with_suffix(".pid").write_text(f"{process.pid}\n")
            processes.append((process, path))
            print(
                f"[{variant}] GPU {devices[gpu]}，阶段 {tag}，日志 {path}", flush=True
            )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(forward, process, path) for process, path in processes
            ]
            for future in as_completed(futures):
                if future.result() != 0:
                    for process, _ in processes:
                        if process.poll() is None:
                            process.terminate()
                    raise SftEvaluationError(
                        "子任务失败，停止另一分支；原命令可续跑已完成 chunk"
                    )
    finally:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def summarize(plans: Mapping[str, Mapping[str, Any]], smoke_limit: int | None) -> Path:
    rows = []
    macros = {}
    for variant, plan in plans.items():
        branch = []
        for subset in SUBSETS:
            mode_dir = f"smoke{smoke_limit}" if smoke_limit else "results"
            path = (
                resolve(plan["config"]["output_root"])
                / variant
                / mode_dir
                / subset
                / "result.json"
            )
            result = load_json(path)
            settings = result["config"]
            if (
                result["status"] != "completed"
                or settings["plan_signature"] != signature(plan)
                or settings["decoding"] != MODE
                or settings["subset"] != subset
                or settings["variant"] != variant
                or settings["smoke_limit"] != smoke_limit
                or result["signature"] != signature(settings)
                or result["metrics"]["sample_count"] != (smoke_limit or 10000)
                or result["metrics"]["valid_id_rate"] != 1
            ):
                raise SftEvaluationError("汇总遇到不完整或不同协议结果")
            ref = plan["baseline_results"][subset]
            require_hash(Path(ref["path"]), ref["sha256"])
            old = load_json(Path(ref["path"]))["metrics"]
            row = dict(
                variant=variant,
                subset=subset,
                constrained=result["metrics"],
                result_file=str(path),
            )
            if smoke_limit is None:
                row.update(
                    unconstrained=old,
                    delta_constrained_minus_unconstrained={
                        m: result["metrics"][m] - old[m] for m in METRICS
                    },
                )
            branch.append(row)
        rows.extend(branch)
        macros[variant] = {
            m: sum(r["constrained"][m] for r in branch[1:]) / 4 for m in METRICS
        }
    root = resolve(plans[VARIANTS[0]]["config"]["output_root"])
    path = root / (
        f"smoke{smoke_limit}_summary.json" if smoke_limit else "summary.json"
    )
    immutable_json(
        path,
        dict(
            status="completed",
            decoding=MODE,
            epoch=3.0,
            num_beams=10,
            smoke_limit=smoke_limit,
            rows=rows,
            generalization_macro_average=macros,
            fixed10k_excluded_from_macro=True,
        ),
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="双卡 6000D：GID/NoGID 全目录约束解码，固定 10k + 四类泛化；不读 Test。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help="冻结来源与隔离输出配置",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="CPU 核验两版目录和十组样本，不启动 GPU"
    )
    parser.add_argument(
        "--smoke-limit", type=int, help="每组仅运行 1–100 条，独立 smoke 输出"
    )
    parser.add_argument("--worker-variant", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fixed-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
        parser.error("smoke-limit 必须在 1–100 之间")
    if (
        bool(args.worker_variant) != bool(args.plan)
        or (args.worker_variant and args.dry_run)
        or (args.fixed_only and not args.worker_variant)
    ):
        parser.error("worker 参数组合不合法")
    try:
        if args.worker_variant:
            plan = load_json(args.plan)
            if plan["variant"] != args.worker_variant:
                raise SftEvaluationError("worker 分支与 plan 不一致")
            lock_path = (
                resolve(plan["config"]["output_root"])
                / plan["variant"]
                / ".worker.lock"
            )
            with lock_path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for subset in SUBSETS[:1] if args.fixed_only else SUBSETS:
                    evaluate_subset(plan, subset, args.smoke_limit)
            return 0
        config_path = resolve(args.config)
        output = isolated_output(yaml.safe_load(config_path.read_text())["output_root"])
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".suite.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plans = prepare_plans(config_path)
            if args.dry_run:
                print(
                    "约束解码 CPU 预检通过：2 个全目录 × 5 组 10k；未启动 GPU 推理。",
                    flush=True,
                )
                return 0
            validate_hardware()
            if args.smoke_limit is None:
                schedule(plans, 2, fixed_only=True)
                print("两版固定集 GPU smoke 通过，开始十组正式约束评测。", flush=True)
            schedule(plans, args.smoke_limit)
            print(
                f"约束解码评测全部完成：{summarize(plans, args.smoke_limit)}",
                flush=True,
            )
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"约束解码评测停止：{error}", file=sys.stderr)
        return 2

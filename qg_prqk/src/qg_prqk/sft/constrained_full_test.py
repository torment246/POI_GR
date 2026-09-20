"""Apply the existing full-catalog constraint to the frozen full Test evaluator."""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Mapping

from qg_prqk.artifacts import write_json_atomic
from qg_prqk.sft import evaluation as ev
from qg_prqk.sft.constrained_decoding import ConstrainedGenerationModel, FinalIdPrefixIndex
from qg_prqk.sft.evaluation_data import SftEvaluationError, load_json, resolve, signature
from qg_prqk.sft.full_test_data import JsonlRecordSequence
from qg_prqk.sft.full_test_evaluation import evaluate_test_chunk
from qg_prqk.sft.training import project_root

MODE = "full_catalog_constrained_beam_search"


def validate_progress(progress: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
    """Reject stale, partial or invalid constrained checkpoints before resuming."""
    count = progress.get("next_line", -1)
    metrics = progress.get("metrics", {})
    if (progress.get("signature") != signature(settings)
            or not 0 <= count <= settings["sample_count"]
            or metrics.get("sample_count") != count
            or metrics.get("candidate_count") != count * 10
            or metrics.get("valid_candidate_count") != count * 10
            or not 1 <= progress.get("actual_batch_size", 0) <= 32):
        raise SftEvaluationError("约束 Test 断点来源、计数或合法率不一致")


def evaluate_constrained_test(
    config: Mapping[str, Any], *, variant: str, source: Mapping[str, Any],
    output: Path, plan_signature: str, smoke_limit: int | None = None,
) -> dict[str, Any]:
    """Reuse exact Test encoding/scoring and add only the existing prefix mask."""
    import torch

    spec = config["variants"][variant]
    settings = dict(plan_signature=plan_signature, variant=variant, split="test", date=config["date"],
                    checkpoint=str(resolve(spec["checkpoint"])), epoch=3.0, num_beams=10,
                    cutoff_len=1024, decoding=MODE, geographic_pruning=False,
                    sample_count=smoke_limit or source["rows"], smoke_limit=smoke_limit)
    output.mkdir(parents=True, exist_ok=True)
    result_path, progress_path = output / "result.json", output / "progress.json"
    if result_path.exists():
        result = load_json(result_path)
        if (result.get("status") != "completed" or result.get("config") != settings
                or result["metrics"]["sample_count"] != settings["sample_count"]
                or result["metrics"]["valid_id_rate"] != 1.0):
            raise SftEvaluationError("已有约束 Test 结果不完整或配置不同")
        return result
    progress = dict(signature=signature(settings), next_line=0, metrics=ev.empty_metrics(),
                    actual_batch_size=32, inference_seconds=0.0, peak_memory_bytes=0)
    if progress_path.exists():
        progress = load_json(progress_path)
    validate_progress(progress, settings)
    records = JsonlRecordSequence(Path(source["file"]), variant=variant,
                                   expected_rows=source["rows"], expected_sha256=source["sha256"])
    checkpoint = resolve(spec["checkpoint"])
    tokenizer, template = ev.load_lf_tokenizer_and_template(checkpoint, project_root=project_root())
    index = ev.build_final_id_index(config, variant, tokenizer)
    trie = FinalIdPrefixIndex(index)
    if trie.max_length != spec["max_new_tokens"]:
        raise SftEvaluationError("约束路径不能在冻结 max_new_tokens 内完整闭合")
    model = ConstrainedGenerationModel(ev.load_generation_model(checkpoint, expected_vocab_size=len(tokenizer)), trie)
    torch.cuda.reset_peak_memory_stats(model.device)
    while progress["next_line"] < settings["sample_count"]:
        start = progress["next_line"]
        stop = min(start + config["decoding"]["chunk_size"], settings["sample_count"])
        started = time.perf_counter()
        try:
            metrics = evaluate_test_chunk(records[start:stop], variant=variant, model=model,
                                          tokenizer=tokenizer, template=template, index=index,
                                          batch_size=progress["actual_batch_size"],
                                          max_new_tokens=spec["max_new_tokens"])
        except RuntimeError as error:
            if not ev.is_cuda_oom(error) or progress["actual_batch_size"] <= 1:
                raise
            progress["actual_batch_size"] //= 2
            error.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[{variant}] OOM，batch 降为 {progress['actual_batch_size']}，重试未入账分块", flush=True)
            continue
        if (metrics["sample_count"] != stop - start or metrics["candidate_count"] != (stop - start) * 10
                or metrics["valid_candidate_count"] != metrics["candidate_count"]):
            raise SftEvaluationError("约束生成出现非法或缺失候选，停止且不入账")
        ev.merge_metrics(progress["metrics"], metrics)
        progress.update(next_line=stop,
                        inference_seconds=progress["inference_seconds"] + time.perf_counter() - started,
                        peak_memory_bytes=max(progress["peak_memory_bytes"], torch.cuda.max_memory_allocated(model.device)))
        write_json_atomic(progress_path, progress, overwrite=True)
        print(f"[{variant}/test/constrained] {stop:,}/{settings['sample_count']:,}", flush=True)
    result = dict(status="completed", config=settings, metrics=ev.finalize_metrics(progress["metrics"]),
                  performance={k: progress[k] for k in ("inference_seconds", "actual_batch_size", "peak_memory_bytes")})
    write_json_atomic(result_path, result)
    return result

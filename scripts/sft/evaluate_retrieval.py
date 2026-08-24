#!/usr/bin/env python3
"""Run resumable full Validation/Test Trie-constrained PID retrieval evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sft.evaluation import (  # noqa: E402
    GenerativeEvalError,
    atomic_write_json,
    authorize_test_file,
    build_fixed_validation_subset,
    build_reference_aligned_validation_subset,
    flatten_result_row,
    freeze_selected_config,
    load_lf_tokenizer_and_template,
    run_full_evaluation,
    select_best_checkpoint,
    validate_checkpoints,
    validate_prompt_template,
    validate_split_manifest,
    write_results_csv,
)
from poi_gr.pid.trie import PidTrieError, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用全量 Final PID Trie 做可断点恢复的生成式 POI 检索评测。"
            "Test 模式只接受已经冻结的 selected_config.json。"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("valid-checkpoints", "valid-beams", "test-final"),
        required=True,
    )
    parser.add_argument("--valid-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--checkpoints", type=Path, nargs="+")
    parser.add_argument("--checkpoint-results", type=Path)
    parser.add_argument("--selected-config", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--trie-dir", type=Path, required=True)
    parser.add_argument(
        "--ssp-predictions-dir",
        type=Path,
        help="GenPOI 邻近分类器对当前评测文件生成的完整 SSP predictions 目录。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beam-sizes", type=int, nargs="+")
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        choices=(128, 64, 32, 16),
        default=128,
    )
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument(
        "--validation-subset-size",
        type=int,
        default=None,
        help=(
            "仅 valid-checkpoints 使用：按 sample_id 字典序固定抽取指定条数，"
            "保存子集 JSONL 和 manifest 后评测全部 checkpoint。"
        ),
    )
    parser.add_argument(
        "--reference-validation-subset",
        type=Path,
        help=(
            "仅 valid-checkpoints 使用：按参考 JSONL 的 order_id + searchid "
            "精确对齐同一批 Validation 样本。"
        ),
    )
    parser.add_argument(
        "--expected-checkpoint-steps",
        type=int,
        nargs="+",
        default=(2290, 4580, 6870, 9158),
        help="依次对应 --checkpoints 的训练步数。",
    )
    parser.add_argument(
        "--expected-checkpoint-epochs",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 1.5, 2.0),
        help="依次对应 --checkpoints 的 epoch。",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="仅做前 N 条技术 Smoke；不会写正式汇总或冻结配置。",
    )
    parser.add_argument(
        "--skip-data-hash",
        action="store_true",
        help="仅用于技术调试；跳过评测 JSONL SHA256 复核。",
    )
    return parser.parse_args()


def resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise GenerativeEvalError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GenerativeEvalError(f"{name} JSON 解析失败") from error
    if not isinstance(value, dict):
        raise GenerativeEvalError(f"{name} 必须是 JSON object")
    return value


def _load_first_records(path: Path, count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise GenerativeEvalError(
                    f"Prompt 校验第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(value, dict):
                raise GenerativeEvalError("评测 JSONL 每行必须是 object")
            records.append(value)
            if len(records) == count:
                break
    if len(records) != count:
        raise GenerativeEvalError(f"评测文件不足 {count} 行")
    return records


def _trie_inputs(trie_dir: Path) -> tuple[Path, dict[str, Any]]:
    manifest = _load_json(trie_dir / "trie_manifest.json", "Trie manifest")
    if manifest.get("status") != "completed":
        raise GenerativeEvalError("Trie manifest 状态不是 completed")
    mapping = Path(manifest.get("input", {}).get("pid_mapping", ""))
    if not mapping.is_file():
        raise GenerativeEvalError("Trie manifest 中的 PID mapping 不存在")
    if manifest.get("leaf_count") != 2_337_178:
        raise GenerativeEvalError("Trie 叶子数必须为 2,337,178")
    return mapping, manifest


def _write_prompt_validation(
    *,
    valid_file: Path,
    tokenizer_path: Path,
    output_dir: Path,
    cutoff_len: int,
    valid_file_sha256: str,
) -> None:
    tokenizer, template = load_lf_tokenizer_and_template(
        tokenizer_path,
        project_root=PROJECT_ROOT,
    )
    result = validate_prompt_template(
        _load_first_records(valid_file, 100),
        split="valid",
        tokenizer=tokenizer,
        template=template,
        cutoff_len=cutoff_len,
    )
    result["valid_file_sha256"] = valid_file_sha256
    result["tokenizer_json_sha256"] = sha256_file(tokenizer_path / "tokenizer.json")
    atomic_write_json(output_dir / "prompt_template_validation.json", result)


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _write_run_error_cases(result: Mapping[str, Any], output_path: Path) -> None:
    progress = _load_json(
        Path(result["run_dir"]) / "progress.json",
        "评测运行进度",
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for error_type in (
            "top1_gid_error",
            "gid_correct_sid_error",
            "base_pid_correct_dedup_error",
            "target_in_top10_not_top1",
            "top10_miss",
        ):
            for item in progress["error_cases"][error_type]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _checkpoint_eval(
    *,
    valid_file: Path,
    valid_rows: int,
    valid_hash: str,
    checkpoint_metadata: Mapping[str, Any],
    tokenizer: Path,
    trie_dir: Path,
    mapping: Path,
    output_dir: Path,
    beams: int,
    returns: int,
    top_k: int,
    batch_size: int,
    chunk_size: int,
    cutoff_len: int,
    smoke_limit: int | None,
    dataset_context: Mapping[str, Any] | None = None,
    ssp_predictions_dir: Path | None = None,
) -> dict[str, Any]:
    result = run_full_evaluation(
        data_file=valid_file,
        split="valid",
        expected_rows=valid_rows,
        data_sha256=valid_hash,
        checkpoint=Path(checkpoint_metadata["path"]),
        epoch=float(checkpoint_metadata["epoch"]),
        validation_loss=checkpoint_metadata.get("validation_loss"),
        tokenizer_path=tokenizer,
        trie_dir=trie_dir,
        mapping_path=mapping,
        output_root=output_dir,
        project_root=PROJECT_ROOT,
        num_beams=beams,
        num_return_sequences=returns,
        top_k=top_k,
        initial_batch_size=batch_size,
        chunk_size=chunk_size,
        cutoff_len=cutoff_len,
        smoke_limit=smoke_limit,
        dataset_context=dataset_context,
        ssp_predictions_dir=ssp_predictions_dir,
    )
    _free_cuda()
    return result


def run_valid_checkpoints(args: argparse.Namespace) -> int:
    valid_file = resolve(args.valid_file)
    checkpoints = [resolve(value) for value in (args.checkpoints or [])]
    tokenizer = resolve(args.tokenizer)
    trie_dir = resolve(args.trie_dir)
    output_dir = resolve(args.output_dir)
    ssp_predictions_dir = resolve(args.ssp_predictions_dir)
    assert valid_file and tokenizer and trie_dir and output_dir
    if args.valid_file is None or not checkpoints:
        raise GenerativeEvalError(
            "valid-checkpoints 模式必须提供 --valid-file 和 --checkpoints"
        )
    if args.length_penalty != 1.0:
        raise GenerativeEvalError("本任务 length_penalty 固定为 1.0")
    if args.num_beams != 10 or args.top_k != 10:
        raise GenerativeEvalError("Checkpoint 对比必须固定 Beam=10、Top-K=10")

    output_dir.mkdir(parents=True, exist_ok=True)
    _, source_valid_rows, source_valid_hash = validate_split_manifest(
        valid_file,
        split="valid",
        verify_hash=not args.skip_data_hash,
    )
    evaluation_file = valid_file
    evaluation_rows = source_valid_rows
    evaluation_hash = source_valid_hash
    dataset_context: dict[str, Any] | None = None
    reference_subset = resolve(args.reference_validation_subset)
    if reference_subset is not None and args.validation_subset_size is not None:
        raise GenerativeEvalError(
            "--reference-validation-subset 与 --validation-subset-size 不能同时使用"
        )
    if reference_subset is not None:
        subset = build_reference_aligned_validation_subset(
            valid_file,
            reference_subset,
            output_dir,
            source_rows=source_valid_rows,
            source_sha256=source_valid_hash,
        )
        if subset.row_count != 10_000:
            raise GenerativeEvalError("本轮参考 Validation 子集必须恰好为 10,000 条")
        evaluation_file = subset.data_path
        evaluation_rows = subset.row_count
        evaluation_hash = subset.sha256
        dataset_context = {
            "scope": "reference_aligned_validation_subset",
            "subset_size": subset.row_count,
            "subset_manifest": str(subset.manifest_path),
            "subset_manifest_sha256": sha256_file(subset.manifest_path),
            "source_file": str(valid_file),
            "source_rows": source_valid_rows,
            "source_sha256": source_valid_hash,
            "reference_file": str(reference_subset),
            "reference_sha256": subset.manifest["reference_sha256"],
            "business_keys_sha256": subset.manifest["business_keys_sha256"],
            "selection_method": subset.manifest["selection_method"],
            "output_order": subset.manifest["output_order"],
        }
    elif args.validation_subset_size is not None:
        if args.validation_subset_size != 10_000:
            raise GenerativeEvalError("本轮固定 Validation 子集大小必须为 10,000")
        subset = build_fixed_validation_subset(
            valid_file,
            output_dir,
            source_rows=source_valid_rows,
            source_sha256=source_valid_hash,
            subset_size=args.validation_subset_size,
        )
        evaluation_file = subset.data_path
        evaluation_rows = subset.row_count
        evaluation_hash = subset.sha256
        dataset_context = {
            "scope": "fixed_validation_subset",
            "subset_size": subset.row_count,
            "subset_manifest": str(subset.manifest_path),
            "subset_manifest_sha256": sha256_file(subset.manifest_path),
            "source_file": str(valid_file),
            "source_rows": source_valid_rows,
            "source_sha256": source_valid_hash,
            "selection_method": subset.manifest["selection_method"],
            "output_order": subset.manifest["output_order"],
        }
    mapping, _ = _trie_inputs(trie_dir)
    metadata = validate_checkpoints(
        checkpoints,
        tokenizer_path=tokenizer,
        expected_steps=args.expected_checkpoint_steps,
        expected_epochs=args.expected_checkpoint_epochs,
    )
    _write_prompt_validation(
        valid_file=evaluation_file,
        tokenizer_path=tokenizer,
        output_dir=output_dir,
        cutoff_len=args.cutoff_len,
        valid_file_sha256=evaluation_hash,
    )

    if args.smoke_limit is not None:
        if args.smoke_limit > 100:
            raise GenerativeEvalError("技术 Smoke 最多允许 100 条")
        result = _checkpoint_eval(
            valid_file=evaluation_file,
            valid_rows=evaluation_rows,
            valid_hash=evaluation_hash,
            checkpoint_metadata=metadata[0],
            tokenizer=tokenizer,
            trie_dir=trie_dir,
            mapping=mapping,
            output_dir=output_dir,
            beams=10,
            returns=10,
            top_k=10,
            batch_size=args.per_device_eval_batch_size,
            chunk_size=min(args.chunk_size, args.smoke_limit),
            cutoff_len=args.cutoff_len,
            smoke_limit=args.smoke_limit,
            dataset_context=dataset_context,
            ssp_predictions_dir=ssp_predictions_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    results = [
        _checkpoint_eval(
            valid_file=evaluation_file,
            valid_rows=evaluation_rows,
            valid_hash=evaluation_hash,
            checkpoint_metadata=item,
            tokenizer=tokenizer,
            trie_dir=trie_dir,
            mapping=mapping,
            output_dir=output_dir,
            beams=10,
            returns=10,
            top_k=10,
            batch_size=args.per_device_eval_batch_size,
            chunk_size=args.chunk_size,
            cutoff_len=args.cutoff_len,
            smoke_limit=None,
            dataset_context=dataset_context,
            ssp_predictions_dir=ssp_predictions_dir,
        )
        for item in metadata
    ]
    best = select_best_checkpoint(results)
    payload = {
        "schema_version": "valid-checkpoint-results-v1",
        "status": "completed",
        "selection_rule": ("NDCG@10 desc, HR@1 desc, HR@10 desc, Validation Loss asc"),
        "evaluation_scope": dataset_context
        or {
            "scope": "full_validation",
            "source_file": str(valid_file),
            "source_rows": source_valid_rows,
            "source_sha256": source_valid_hash,
        },
        "results": results,
        "best_checkpoint": best["config"]["checkpoint_name"],
    }
    atomic_write_json(output_dir / "valid_checkpoint_results.json", payload)
    write_results_csv(
        output_dir / "valid_checkpoint_results.csv",
        [flatten_result_row(result) for result in results],
    )
    _write_run_error_cases(
        best,
        output_dir / "validation_error_cases.jsonl",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_valid_beams(args: argparse.Namespace) -> int:
    valid_file = resolve(args.valid_file)
    checkpoint_results_path = resolve(args.checkpoint_results)
    tokenizer = resolve(args.tokenizer)
    trie_dir = resolve(args.trie_dir)
    output_dir = resolve(args.output_dir)
    assert tokenizer and trie_dir and output_dir
    if valid_file is None or checkpoint_results_path is None:
        raise GenerativeEvalError(
            "valid-beams 模式必须提供 --valid-file 和 --checkpoint-results"
        )
    beam_sizes = args.beam_sizes
    if not beam_sizes or any(value <= 0 for value in beam_sizes):
        raise GenerativeEvalError("valid-beams 必须提供正整数 --beam-sizes")
    if args.length_penalty != 1.0:
        raise GenerativeEvalError("本任务 length_penalty 固定为 1.0")

    _, valid_rows, valid_hash = validate_split_manifest(
        valid_file,
        split="valid",
        verify_hash=not args.skip_data_hash,
    )
    mapping, _ = _trie_inputs(trie_dir)
    checkpoint_payload = _load_json(
        checkpoint_results_path,
        "Validation checkpoint results",
    )
    prior_results = checkpoint_payload.get("results")
    if not isinstance(prior_results, list):
        raise GenerativeEvalError("Checkpoint results 缺少 results")
    best = select_best_checkpoint(prior_results)
    checkpoint_metadata = {
        "path": best["config"]["checkpoint"],
        "epoch": best["config"]["epoch"],
        "validation_loss": best["config"]["validation_loss"],
    }
    beam_results: list[dict[str, Any]] = []
    for beam in beam_sizes:
        returns = 1 if beam == 1 else min(beam, 10)
        if beam == 10:
            reused = next(
                (
                    item
                    for item in prior_results
                    if item["config"]["checkpoint"] == checkpoint_metadata["path"]
                    and item["config"]["num_beams"] == 10
                    and item["config"]["num_return_sequences"] == 10
                ),
                None,
            )
            if reused is not None:
                beam_results.append(reused)
                continue
        beam_results.append(
            _checkpoint_eval(
                valid_file=valid_file,
                valid_rows=valid_rows,
                valid_hash=valid_hash,
                checkpoint_metadata=checkpoint_metadata,
                tokenizer=tokenizer,
                trie_dir=trie_dir,
                mapping=mapping,
                output_dir=output_dir,
                beams=beam,
                returns=returns,
                top_k=10,
                batch_size=args.per_device_eval_batch_size,
                chunk_size=args.chunk_size,
                cutoff_len=args.cutoff_len,
                smoke_limit=args.smoke_limit,
            )
        )
    if args.smoke_limit is not None:
        print(
            json.dumps(
                {"status": "smoke_completed", "results": beam_results},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    payload = {
        "schema_version": "valid-beam-results-v1",
        "status": "completed",
        "checkpoint": checkpoint_metadata["path"],
        "results": beam_results,
    }
    atomic_write_json(output_dir / "valid_beam_results.json", payload)
    write_results_csv(
        output_dir / "valid_beam_results.csv",
        [flatten_result_row(result) for result in beam_results],
    )
    selected = freeze_selected_config(
        checkpoint_result=best,
        beam_results=beam_results,
        output_path=output_dir / "selected_config.json",
    )
    selected_run = next(
        result
        for result in beam_results
        if result["config"]["num_beams"] == selected["num_beams"]
        and result["config"]["num_return_sequences"] == selected["num_return_sequences"]
    )
    progress = _load_json(
        Path(selected_run["run_dir"]) / "progress.json",
        "最佳 Validation 运行进度",
    )
    cases = [
        item
        for error_type in (
            "top1_gid_error",
            "gid_correct_sid_error",
            "base_pid_correct_dedup_error",
            "target_in_top10_not_top1",
            "top10_miss",
        )
        for item in progress["error_cases"][error_type]
    ]
    with (output_dir / "validation_error_cases.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for item in cases:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"beam_results": payload, "selected_config": selected},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_test_final(args: argparse.Namespace) -> int:
    selected_path = resolve(args.selected_config)
    tokenizer = resolve(args.tokenizer)
    trie_dir = resolve(args.trie_dir)
    output_dir = resolve(args.output_dir)
    assert tokenizer and trie_dir and output_dir
    if selected_path is None:
        raise GenerativeEvalError("test-final 必须先提供已冻结的 --selected-config")
    if args.smoke_limit is not None:
        raise GenerativeEvalError(
            "Test 只允许冻结后的单次完整评测，不接受 --smoke-limit"
        )

    # The frozen gate is intentionally loaded and validated before test_file is touched.
    test_file = resolve(args.test_file)
    if test_file is None:
        raise GenerativeEvalError("test-final 模式必须提供 --test-file")
    selected = authorize_test_file(selected_path, test_file)
    if sha256_file(tokenizer / "tokenizer.json") != selected.get(
        "tokenizer_json_sha256"
    ):
        raise GenerativeEvalError("Tokenizer 与冻结配置不一致")
    if sha256_file(trie_dir / "trie_manifest.json") != selected.get(
        "trie_manifest_sha256"
    ):
        raise GenerativeEvalError("Trie 与冻结配置不一致")

    _, test_rows, test_hash = validate_split_manifest(
        test_file,
        split="test",
        verify_hash=not args.skip_data_hash,
    )
    mapping, trie_manifest = _trie_inputs(trie_dir)
    result = run_full_evaluation(
        data_file=test_file,
        split="test",
        expected_rows=test_rows,
        data_sha256=test_hash,
        checkpoint=Path(selected["checkpoint"]),
        epoch=float(selected["epoch"]),
        validation_loss=None,
        tokenizer_path=tokenizer,
        trie_dir=trie_dir,
        mapping_path=mapping,
        output_root=output_dir,
        project_root=PROJECT_ROOT,
        num_beams=int(selected["num_beams"]),
        num_return_sequences=int(selected["num_return_sequences"]),
        top_k=10,
        initial_batch_size=args.per_device_eval_batch_size,
        chunk_size=args.chunk_size,
        cutoff_len=args.cutoff_len,
        smoke_limit=args.smoke_limit,
    )
    if args.smoke_limit is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    atomic_write_json(output_dir / "test_final_results.json", result)
    write_results_csv(
        output_dir / "test_final_results.csv",
        [flatten_result_row(result)],
    )
    progress = _load_json(
        Path(result["run_dir"]) / "progress.json",
        "Test 运行进度",
    )
    with (output_dir / "test_error_cases.jsonl").open("w", encoding="utf-8") as handle:
        for error_type in (
            "top1_gid_error",
            "gid_correct_sid_error",
            "base_pid_correct_dedup_error",
            "target_in_top10_not_top1",
            "top10_miss",
        ):
            for item in progress["error_cases"][error_type]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    runtime_summary = {
        "validation_checkpoint_runs": _load_json(
            output_dir / "valid_checkpoint_results.json",
            "Validation checkpoint results",
        )["results"],
        "validation_beam_runs": _load_json(
            output_dir / "valid_beam_results.json",
            "Validation beam results",
        )["results"],
        "test_run": result,
    }
    atomic_write_json(output_dir / "runtime_summary.json", runtime_summary)
    evaluation_manifest = {
        "schema_version": "sft-evaluation-manifest-v1",
        "status": "completed",
        "selected_config": str(selected_path),
        "selected_config_sha256": sha256_file(selected_path),
        "valid_file": str(test_file.parent / "valid.jsonl"),
        "test_file": str(test_file),
        "test_file_sha256": test_hash,
        "tokenizer": str(tokenizer),
        "tokenizer_json_sha256": sha256_file(tokenizer / "tokenizer.json"),
        "trie_manifest": str(trie_dir / "trie_manifest.json"),
        "trie_manifest_sha256": sha256_file(trie_dir / "trie_manifest.json"),
        "pid_mapping": str(mapping),
        "pid_mapping_sha256": trie_manifest["input"]["pid_mapping_sha256"],
        "completed_at": result["completed_at"],
    }
    atomic_write_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "valid-checkpoints":
            return run_valid_checkpoints(args)
        if args.mode == "valid-beams":
            return run_valid_beams(args)
        return run_test_final(args)
    except (GenerativeEvalError, PidTrieError, OSError, ValueError) as error:
        print(f"生成式检索评测失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run position-wise teacher-forcing diagnostics for generative POI IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from poi_gr.methods.gnpr.eval import (  # noqa: E402
    GnprEvalError,
    load_gnpr_token_ids,
    parse_target_codes as parse_gnpr_target,
)
from poi_gr.methods.tiger.eval import (  # noqa: E402
    TigerEvalError,
    load_tiger_token_ids,
    parse_target_codes as parse_tiger_target,
)
from poi_gr.pid.trie import PidTrieError, load_pid_token_ids, sha256_file  # noqa: E402
from poi_gr.sft.evaluation import (  # noqa: E402
    GenerativeEvalError,
    encode_prompt_like_training,
    load_generation_model,
    load_lf_tokenizer_and_template,
)
from poi_gr.sft.teacher_forcing import (  # noqa: E402
    TeacherForcingError,
    empty_teacher_forcing_metrics,
    finalize_teacher_forcing_metrics,
    score_target_logits,
    update_teacher_forcing_metrics,
)


@dataclass(frozen=True)
class EncodedExample:
    sample_id: str
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    position_names: tuple[str, ...]
    semantic_indices: tuple[int, int, int]
    identifier_indices: tuple[int, ...]
    group: str
    context_indices: tuple[int, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在冻结评测集上统计 SID 各目标位置的 teacher-forcing 准确率。"
    )
    parser.add_argument(
        "--method",
        choices=("tiger", "gnpr", "genpoi"),
        required=True,
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=float, default=3.0)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(32, 16, 8, 4, 2),
        default=16,
    )
    parser.add_argument("--checkpoint-rows", type=int, default=1000)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--skip-model-hash", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise TeacherForcingError(
                    f"评测数据第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(value, dict):
                raise TeacherForcingError(f"评测数据第 {line_number} 行必须为 object")
            records.append(value)
    return records


def assistant_content(record: Mapping[str, Any]) -> tuple[str, str, str]:
    messages = record.get("messages")
    if (
        record.get("split") != "valid"
        or not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise TeacherForcingError("评测样本必须是 valid user+assistant Messages")
    user_content = messages[0].get("content")
    target_content = messages[1].get("content")
    sample_id = record.get("sample_id")
    if not all(isinstance(value, str) and value for value in (user_content, target_content, sample_id)):
        raise TeacherForcingError("Messages content 与 sample_id 必须是非空字符串")
    return user_content, target_content, sample_id


def encode_tiger_example(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    tokens: Any,
    cutoff_len: int,
) -> EncodedExample:
    user_content, target_content, sample_id = assistant_content(record)
    codes = parse_tiger_target(target_content)
    prompt_ids, target_ids = encode_prompt_like_training(
        tokenizer=tokenizer,
        template=template,
        user_content=user_content,
        target_content=target_content,
        cutoff_len=cutoff_len,
    )
    expected = [
        tokens.target_open,
        tokens.s1[codes[0]],
        tokens.s2[codes[1]],
        tokens.s3[codes[2]],
        tokens.collision[codes[3]],
        tokens.target_close,
    ]
    if target_ids != expected:
        raise TeacherForcingError(f"TIGER 目标 Token 编码不一致：{sample_id}")
    return EncodedExample(
        sample_id=sample_id,
        prompt_ids=tuple(int(value) for value in prompt_ids),
        target_ids=tuple((*expected, tokens.eos)),
        position_names=(
            "target_open",
            "sid_1",
            "sid_2",
            "sid_3",
            "disambiguation",
            "target_close",
            "eos",
        ),
        semantic_indices=(1, 2, 3),
        identifier_indices=(1, 2, 3, 4),
        group="fixed_four_token",
    )


def encode_gnpr_example(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    tokens: Any,
    cutoff_len: int,
) -> EncodedExample:
    user_content, target_content, sample_id = assistant_content(record)
    codes = parse_gnpr_target(target_content)
    prompt_ids, target_ids = encode_prompt_like_training(
        tokenizer=tokenizer,
        template=template,
        user_content=user_content,
        target_content=target_content,
        cutoff_len=cutoff_len,
    )
    expected = [
        tokens.target_open,
        tokens.a[codes[0]],
        tokens.b[codes[1]],
        tokens.c[codes[2]],
    ]
    if codes[3] >= 0:
        expected.extend((tokens.dedup[codes[3]], tokens.target_close))
        position_names = (
            "target_open",
            "sid_1",
            "sid_2",
            "sid_3",
            "disambiguation",
            "target_close",
            "eos",
        )
        identifier_indices = (1, 2, 3, 4)
        group = "collision"
    else:
        expected.append(tokens.target_close)
        position_names = (
            "target_open",
            "sid_1",
            "sid_2",
            "sid_3",
            "target_close",
            "eos",
        )
        identifier_indices = (1, 2, 3)
        group = "singleton"
    if target_ids != expected:
        raise TeacherForcingError(f"GNPR 目标 Token 编码不一致：{sample_id}")
    return EncodedExample(
        sample_id=sample_id,
        prompt_ids=tuple(int(value) for value in prompt_ids),
        target_ids=tuple((*expected, tokens.eos)),
        position_names=position_names,
        semantic_indices=(1, 2, 3),
        identifier_indices=identifier_indices,
        group=group,
    )


def _is_member(token_id: int, values: Sequence[int]) -> bool:
    return int(values[0]) <= token_id <= int(values[-1])


def encode_genpoi_example(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    tokens: Any,
    cutoff_len: int,
) -> EncodedExample:
    user_content, target_content, sample_id = assistant_content(record)
    requires_dedup = record.get("requires_dedup")
    if not isinstance(requires_dedup, bool):
        raise TeacherForcingError("GenPOI requires_dedup 必须为 bool")
    prompt_ids, target_ids = encode_prompt_like_training(
        tokenizer=tokenizer,
        template=template,
        user_content=user_content,
        target_content=target_content,
        cutoff_len=cutoff_len,
    )
    expected_length = 10 if requires_dedup else 9
    if len(target_ids) != expected_length:
        raise TeacherForcingError(
            f"GenPOI 目标 PID 长度 {len(target_ids)} != {expected_length}：{sample_id}"
        )
    if any(not _is_member(token_id, tokens.gid) for token_id in target_ids[:6]):
        raise TeacherForcingError(f"GenPOI GID Token 编码不一致：{sample_id}")
    if any(
        not _is_member(target_ids[6 + level], tokens.sid[level])
        for level in range(3)
    ):
        raise TeacherForcingError(f"GenPOI SID Token 编码不一致：{sample_id}")
    if requires_dedup and not _is_member(target_ids[9], tokens.dedup):
        raise TeacherForcingError(f"GenPOI Dedup Token 编码不一致：{sample_id}")

    position_names = (
        "gid_1",
        "gid_2",
        "gid_3",
        "gid_4",
        "gid_5",
        "gid_6",
        "sid_1",
        "sid_2",
        "sid_3",
        *(("disambiguation",) if requires_dedup else ()),
        "eos",
    )
    identifier_indices = tuple(range(expected_length))
    return EncodedExample(
        sample_id=sample_id,
        prompt_ids=tuple(int(value) for value in prompt_ids),
        target_ids=tuple((*target_ids, tokens.eos)),
        position_names=position_names,
        semantic_indices=(6, 7, 8),
        identifier_indices=identifier_indices,
        group="dedup" if requires_dedup else "singleton",
        context_indices=(0, 1, 2, 3, 4, 5),
    )


def encode_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    tokenizer: Any,
    template: Any,
    tokens: Any,
    cutoff_len: int,
) -> list[EncodedExample]:
    encoders = {
        "tiger": encode_tiger_example,
        "gnpr": encode_gnpr_example,
        "genpoi": encode_genpoi_example,
    }
    try:
        encoder = encoders[method]
    except KeyError as error:
        raise TeacherForcingError(f"未知 teacher-forcing 方法：{method}") from error
    return [
        encoder(
            record,
            tokenizer=tokenizer,
            template=template,
            tokens=tokens,
            cutoff_len=cutoff_len,
        )
        for record in records
    ]


def evaluate_batch(
    examples: Sequence[EncodedExample],
    *,
    model: Any,
    tokenizer: Any,
    metrics: dict[str, Any],
) -> None:
    import torch

    combined = [list((*example.prompt_ids, *example.target_ids)) for example in examples]
    maximum = max(len(values) for values in combined)
    input_ids = torch.full(
        (len(examples), maximum),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device=model.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    batch_indices: list[int] = []
    prediction_positions: list[int] = []
    target_ids: list[int] = []
    offsets: list[tuple[int, int]] = []
    for batch_index, (example, values) in enumerate(zip(examples, combined)):
        padding = maximum - len(values)
        input_ids[batch_index, padding:] = torch.tensor(values, device=model.device)
        attention_mask[batch_index, padding:] = 1
        start = len(target_ids)
        for target_index, token_id in enumerate(example.target_ids):
            batch_indices.append(batch_index)
            prediction_positions.append(
                padding + len(example.prompt_ids) - 1 + target_index
            )
            target_ids.append(int(token_id))
        offsets.append((start, len(target_ids)))

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        selected = output.logits[
            torch.tensor(batch_indices, device=model.device),
            torch.tensor(prediction_positions, device=model.device),
        ]
        top1, top10, nll = score_target_logits(
            selected,
            torch.tensor(target_ids, device=model.device),
        )
    top1_values = top1.cpu().tolist()
    top10_values = top10.cpu().tolist()
    nll_values = nll.cpu().tolist()
    for example, (start, stop) in zip(examples, offsets):
        update_teacher_forcing_metrics(
            metrics,
            position_names=example.position_names,
            top1_correct=top1_values[start:stop],
            top10_correct=top10_values[start:stop],
            nll_values=nll_values[start:stop],
            semantic_indices=example.semantic_indices,
            identifier_indices=example.identifier_indices,
            group=example.group,
            context_indices=example.context_indices,
        )
    del output, selected, input_ids, attention_mask


def validate_checkpoint(checkpoint: Path, expected_epoch: float) -> dict[str, Any]:
    state_path = checkpoint / "trainer_state.json"
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    for path in (state_path, config_path, model_path):
        if not path.is_file():
            raise TeacherForcingError(f"checkpoint 缺少文件：{path.name}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    epoch = float(state.get("epoch", -1))
    if abs(epoch - expected_epoch) > 0.001:
        raise TeacherForcingError(f"checkpoint epoch {epoch} != {expected_epoch}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {"epoch": epoch, "vocab_size": int(config["vocab_size"])}


def main() -> int:
    args = parse_args()
    try:
        data_file = resolve(args.data_file)
        checkpoint = resolve(args.checkpoint)
        tokenizer_path = resolve(args.tokenizer)
        output_dir = resolve(args.output_dir)
        if args.checkpoint_rows <= 0:
            raise TeacherForcingError("--checkpoint-rows 必须为正整数")
        if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
            raise TeacherForcingError("--smoke-limit 必须位于 [1, 100]")
        records = load_records(data_file)
        expected_rows = args.smoke_limit or 10_000
        if len(records) < expected_rows or (args.smoke_limit is None and len(records) != 10_000):
            raise TeacherForcingError("正式 teacher-forcing 数据必须恰好为 10,000 条")
        records = records[:expected_rows]
        checkpoint_metadata = validate_checkpoint(checkpoint, args.expected_epoch)
        tokenizer, template = load_lf_tokenizer_and_template(
            tokenizer_path,
            project_root=PROJECT_ROOT,
        )
        if args.method == "tiger":
            tokens, token_metadata = load_tiger_token_ids(tokenizer_path, tokenizer)
        elif args.method == "gnpr":
            tokens, token_metadata = load_gnpr_token_ids(tokenizer_path, tokenizer)
        else:
            tokens, token_metadata = load_pid_token_ids(tokenizer_path)
        if len(tokenizer) != checkpoint_metadata["vocab_size"]:
            raise TeacherForcingError("checkpoint 与 tokenizer 词表大小不一致")

        model_hash = (
            None
            if args.skip_model_hash
            else sha256_file(checkpoint / "model.safetensors")
        )
        config = {
            "schema_version": "sid-teacher-forcing-config-v1",
            "method": args.method,
            "data_file": str(data_file),
            "data_sha256": sha256_file(data_file),
            "rows": len(records),
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": checkpoint_metadata["epoch"],
            "checkpoint_sha256": model_hash,
            "tokenizer": token_metadata,
            "cutoff_len": args.cutoff_len,
            "initial_batch_size": args.batch_size,
            "teacher_forcing": True,
            "beam_search": False,
            "target_positions_only": True,
            "script_sha256": sha256_file(Path(__file__)),
        }
        signature = canonical_sha256(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path = output_dir / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("signature") != signature:
                raise TeacherForcingError("已有 teacher-forcing 运行配置变化")
        else:
            progress = {
                "schema_version": "sid-teacher-forcing-progress-v1",
                "status": "running",
                "signature": signature,
                "config": config,
                "next_line": 0,
                "metrics": empty_teacher_forcing_metrics(),
                "inference_seconds": 0.0,
                "actual_batch_size": args.batch_size,
                "peak_memory_bytes": 0,
            }
            atomic_json(progress_path, progress)

        model = load_generation_model(
            checkpoint,
            expected_vocab_size=len(tokenizer),
        )
        import torch

        torch.cuda.reset_peak_memory_stats(model.device)
        allowed = [value for value in (32, 16, 8, 4, 2) if value <= args.batch_size]
        batch_size = min(int(progress["actual_batch_size"]), allowed[0])
        next_report = (
            (int(progress["next_line"]) // args.checkpoint_rows) + 1
        ) * args.checkpoint_rows
        while int(progress["next_line"]) < len(records):
            start = int(progress["next_line"])
            stop = min(start + batch_size, len(records))
            examples = encode_batch(
                records[start:stop],
                method=args.method,
                tokenizer=tokenizer,
                template=template,
                tokens=tokens,
                cutoff_len=args.cutoff_len,
            )
            started = time.monotonic()
            try:
                evaluate_batch(
                    examples,
                    model=model,
                    tokenizer=tokenizer,
                    metrics=progress["metrics"],
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                smaller = [value for value in allowed if value < batch_size]
                if not smaller:
                    raise TeacherForcingError("batch_size=2 仍然 OOM")
                batch_size = smaller[0]
                progress["actual_batch_size"] = batch_size
                continue
            progress["inference_seconds"] += time.monotonic() - started
            progress["next_line"] = stop
            progress["actual_batch_size"] = batch_size
            progress["peak_memory_bytes"] = max(
                int(progress["peak_memory_bytes"]),
                int(torch.cuda.max_memory_allocated(model.device)),
            )
            if stop >= next_report or stop == len(records):
                atomic_json(progress_path, progress)
                print(f"[{args.method}] {stop:,}/{len(records):,}", flush=True)
                next_report += args.checkpoint_rows

        result = {
            "schema_version": "sid-teacher-forcing-result-v1",
            "status": "completed",
            "config": config,
            "metrics": finalize_teacher_forcing_metrics(progress["metrics"]),
            "performance": {
                "inference_seconds": progress["inference_seconds"],
                "samples_per_second": len(records) / progress["inference_seconds"],
                "peak_memory_bytes": progress["peak_memory_bytes"],
                "actual_batch_size": progress["actual_batch_size"],
            },
        }
        progress["status"] = "completed"
        atomic_json(progress_path, progress)
        atomic_json(output_dir / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (
        TeacherForcingError,
        GenerativeEvalError,
        PidTrieError,
        TigerEvalError,
        GnprEvalError,
        OSError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

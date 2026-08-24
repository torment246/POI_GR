#!/usr/bin/env python
"""Preflight SFT sequence lengths and build reusable LLaMA-Factory caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CACHE_SCHEMA_VERSION = "poi-sft-tokenized-v1"
TRAIN_DATASET = "beijing_order_main_v1_train"
VALID_DATASET = "beijing_order_main_v1_valid"
DEFAULT_MAPPING_FILENAME = "poi_token_mapping.json"
IGNORE_INDEX = -100


class TokenizationPreflightError(ValueError):
    pass


def enforce_safe_cutoff_contract(
    stats: dict[str, Any], requested_cutoff: int
) -> None:
    """Forbid tokenizer-side source truncation for the new 1024-token protocol."""

    over_cutoff = int(stats.get("over_requested_cutoff_count", 0))
    if requested_cutoff >= 1024 and over_cutoff:
        raise TokenizationPreflightError(
            f"仍有 {over_cutoff} 条样本超过 cutoff_len={requested_cutoff}；"
            "拒绝构建会静默截断 CURRENT/当前 Query 的缓存。请先运行 "
            "scripts/sft/build_history_safe_data.py，只删除最早历史后再重新预检"
        )


class LengthHistogram:
    def __init__(self) -> None:
        self.counts = np.zeros(0, dtype=np.int64)
        self.total = 0

    def update(self, values: Sequence[int]) -> None:
        array = np.asarray(values, dtype=np.int64)
        if array.size == 0:
            return
        if np.any(array < 0):
            raise TokenizationPreflightError("Token 长度不能为负数")
        batch_counts = np.bincount(array)
        if batch_counts.size > self.counts.size:
            self.counts = np.pad(
                self.counts, (0, batch_counts.size - self.counts.size)
            )
        self.counts[: batch_counts.size] += batch_counts
        self.total += int(array.size)

    def percentile(self, q: float) -> float:
        if self.total == 0:
            raise TokenizationPreflightError("无法对空数据计算长度分位数")
        rank = (self.total - 1) * q
        low_rank = int(np.floor(rank))
        high_rank = int(np.ceil(rank))
        cumulative = np.cumsum(self.counts)
        low = int(np.searchsorted(cumulative, low_rank + 1))
        high = int(np.searchsorted(cumulative, high_rank + 1))
        return float(low + (high - low) * (rank - low_rank))

    def summary(self, include_p999: bool) -> dict[str, float | int]:
        quantiles = [("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]
        if include_p999:
            quantiles.append(("p99_9", 0.999))
        result: dict[str, float | int] = {
            name: self.percentile(value) for name, value in quantiles
        }
        result["max"] = int(np.flatnonzero(self.counts)[-1])
        return result


@dataclass
class SplitPreflight:
    name: str
    path: Path
    expected_rows: int
    expected_sha256: str
    row_count: int = 0
    sha256: str = ""
    over_128_count: int = 0
    over_requested_cutoff_count: int = 0
    target_truncated_at_requested_cutoff: int = 0
    target_truncated_at_256: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(train_file: Path, valid_file: Path) -> dict[str, Any]:
    manifest_path = train_file.parent / "manifest.json"
    if valid_file.parent != train_file.parent or not manifest_path.is_file():
        raise TokenizationPreflightError(
            "Train/Valid 必须来自同一目录且该目录必须包含 manifest.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TokenizationPreflightError("SFT manifest.json 解析失败") from error
    outputs = manifest.get("outputs")
    for name in ("train.jsonl", "valid.jsonl"):
        if not isinstance(outputs, dict) or name not in outputs:
            raise TokenizationPreflightError(f"Manifest 缺少 {name} 契约")
    return manifest


def _validate_messages(record: dict[str, Any], split: str, line_number: int) -> tuple[str, str]:
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
        or not isinstance(messages[0].get("content"), str)
        or not isinstance(messages[1].get("content"), str)
    ):
        raise TokenizationPreflightError(
            f"{split} 第 {line_number} 行 Messages 格式不合法"
        )
    if record.get("split") != split:
        raise TokenizationPreflightError(
            f"{split} 第 {line_number} 行 split 字段不一致"
        )
    target = messages[1]["content"]
    if "<D_-1>" in target:
        raise TokenizationPreflightError(f"{split} 第 {line_number} 行包含 <D_-1>")
    return messages[0]["content"], target


def _infer_target_length(source: np.ndarray, target: np.ndarray, cutoff: int) -> np.ndarray:
    max_target = np.empty_like(target)
    target_short = target * 2 < cutoff
    source_short = (~target_short) & (source * 2 < cutoff)
    both_long = ~(target_short | source_short)
    max_target[target_short] = cutoff
    max_target[source_short] = cutoff - source[source_short]
    denominator = source[both_long] + target[both_long]
    max_target[both_long] = (
        cutoff * target[both_long] / denominator
    ).astype(np.int64)
    return np.minimum(max_target, target)


def _format_batch(template: Any, users: Sequence[str], targets: Sequence[str]) -> tuple[list[str], list[str]]:
    prompt_texts: list[str] = []
    target_texts: list[str] = []
    prefix_slots = template.format_prefix.apply()
    if prefix_slots:
        raise TokenizationPreflightError("qwen3_nothink 预期没有额外 prefix slot")
    for user, target in zip(users, targets):
        user_slots = template.format_user.apply(content=user, idx="0")
        target_slots = template.format_assistant.apply(content=target)
        if (
            len(user_slots) != 1
            or not isinstance(user_slots[0], str)
            or len(target_slots) != 1
            or not isinstance(target_slots[0], str)
        ):
            raise TokenizationPreflightError(
                "当前 qwen3_nothink 模板不再是单字符串格式，需更新预检实现"
            )
        prompt_texts.append(user_slots[0])
        target_texts.append(target_slots[0])
    return prompt_texts, target_texts


def _process_batch(
    tokenizer: Any,
    template: Any,
    users: Sequence[str],
    targets: Sequence[str],
    requested_cutoff: int,
    input_histogram: LengthHistogram,
    target_histogram: LengthHistogram,
    total_histogram: LengthHistogram,
) -> tuple[int, int, int, int]:
    prompt_texts, target_texts = _format_batch(template, users, targets)
    prompt_encoded = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        return_length=True,
        padding=False,
        truncation=False,
    )
    target_encoded = tokenizer(
        target_texts,
        add_special_tokens=False,
        return_length=True,
        padding=False,
        truncation=False,
    )
    source_lengths = np.asarray(prompt_encoded["length"], dtype=np.int64)
    target_lengths = np.asarray(target_encoded["length"], dtype=np.int64)
    total_lengths = source_lengths + target_lengths
    input_histogram.update(source_lengths)
    target_histogram.update(target_lengths)
    total_histogram.update(total_lengths)
    retained_target = _infer_target_length(
        source_lengths, target_lengths, requested_cutoff
    )
    retained_target_at_256 = _infer_target_length(
        source_lengths, target_lengths, 256
    )
    return (
        int(np.count_nonzero(total_lengths > 128)),
        int(np.count_nonzero(total_lengths > requested_cutoff)),
        int(np.count_nonzero(retained_target < target_lengths)),
        int(np.count_nonzero(retained_target_at_256 < target_lengths)),
    )


def preflight_lengths(
    tokenizer: Any,
    template: Any,
    splits: Sequence[SplitPreflight],
    requested_cutoff: int,
    batch_size: int,
) -> dict[str, Any]:
    """Scan only Train/Valid and return exact untruncated length statistics."""

    input_histogram = LengthHistogram()
    target_histogram = LengthHistogram()
    total_histogram = LengthHistogram()
    for split in splits:
        users: list[str] = []
        targets: list[str] = []
        digest = hashlib.sha256()
        with split.path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                digest.update(raw_line)
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise TokenizationPreflightError(
                        f"{split.name} 第 {line_number} 行不是合法 JSON"
                    ) from error
                user, target = _validate_messages(record, split.name, line_number)
                users.append(user)
                targets.append(target)
                split.row_count += 1
                if len(users) >= batch_size:
                    (
                        over_128,
                        over_requested,
                        truncated,
                        truncated_at_256,
                    ) = _process_batch(
                        tokenizer,
                        template,
                        users,
                        targets,
                        requested_cutoff,
                        input_histogram,
                        target_histogram,
                        total_histogram,
                    )
                    split.over_128_count += over_128
                    split.over_requested_cutoff_count += over_requested
                    split.target_truncated_at_requested_cutoff += truncated
                    split.target_truncated_at_256 += truncated_at_256
                    users.clear()
                    targets.clear()
                if line_number % 1_000_000 == 0:
                    print(
                        f"长度预检 {split.name}: {line_number:,} 行",
                        flush=True,
                    )
            if users:
                (
                    over_128,
                    over_requested,
                    truncated,
                    truncated_at_256,
                ) = _process_batch(
                    tokenizer,
                    template,
                    users,
                    targets,
                    requested_cutoff,
                    input_histogram,
                    target_histogram,
                    total_histogram,
                )
                split.over_128_count += over_128
                split.over_requested_cutoff_count += over_requested
                split.target_truncated_at_requested_cutoff += truncated
                split.target_truncated_at_256 += truncated_at_256
        split.sha256 = digest.hexdigest()
        if split.row_count != split.expected_rows:
            raise TokenizationPreflightError(
                f"{split.name} 行数不一致：{split.row_count} != {split.expected_rows}"
            )
        if split.sha256 != split.expected_sha256:
            raise TokenizationPreflightError(f"{split.name} SHA256 与 Manifest 不一致")

    total_rows = sum(split.row_count for split in splits)
    over_128_count = sum(split.over_128_count for split in splits)
    over_requested_count = sum(
        split.over_requested_cutoff_count for split in splits
    )
    requested_truncated = sum(
        split.target_truncated_at_requested_cutoff for split in splits
    )
    effective_cutoff = 256 if requested_truncated else requested_cutoff
    if effective_cutoff == 256:
        truncated_at_effective = sum(
            split.target_truncated_at_256 for split in splits
        )
        if truncated_at_effective:
            raise TokenizationPreflightError(
                "cutoff_len=256 仍可能截断目标 PID，停止构建缓存"
            )
    else:
        truncated_at_effective = requested_truncated

    return {
        "definition": {
            "input_length": "qwen3_nothink formatted user prompt tokens",
            "target_length": "assistant PID plus qwen3_nothink end token",
            "total_length": "input_length + target_length before truncation",
        },
        "requested_cutoff_len": requested_cutoff,
        "effective_cutoff_len": effective_cutoff,
        "input_length": input_histogram.summary(include_p999=True),
        "target_length": target_histogram.summary(include_p999=False),
        "total_length": total_histogram.summary(include_p999=True),
        "over_128_count": over_128_count,
        "over_128_ratio": over_128_count / total_rows,
        "over_requested_cutoff_count": over_requested_count,
        "over_requested_cutoff_ratio": over_requested_count / total_rows,
        "target_truncated_at_requested_cutoff_count": requested_truncated,
        "target_truncated_at_effective_cutoff_count": truncated_at_effective,
        "splits": {
            split.name: {
                "path": str(split.path.resolve()),
                "rows": split.row_count,
                "sha256": split.sha256,
                "over_128_count": split.over_128_count,
                "over_requested_cutoff_count": (
                    split.over_requested_cutoff_count
                ),
                "target_truncated_at_requested_cutoff_count": (
                    split.target_truncated_at_requested_cutoff
                ),
            }
            for split in splits
        },
    }


def _load_lf_tokenizer_and_template(model_dir: Path) -> tuple[Any, Any, Any]:
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    model_args = ModelArguments(
        model_name_or_path=str(model_dir.resolve()),
        use_fast_tokenizer=True,
    )
    tokenizer_module = load_tokenizer(model_args)
    data_args = DataArguments(
        template="qwen3_nothink",
        train_on_prompt=False,
    )
    template = get_template_and_fix_tokenizer(
        tokenizer_module["tokenizer"], data_args
    )
    return tokenizer_module["tokenizer"], template, tokenizer_module


def _preprocess_smoke_cache(
    *,
    train_file: Path,
    valid_file: Path,
    cache_dir: Path,
    raw_cache_dir: Path,
    tokenizer: Any,
    template: Any,
    tokenizer_module: dict[str, Any],
    cutoff_len: int,
    preprocessing_batch_size: int,
    train_dataset: str,
    valid_dataset: str,
    smoke_train_rows: int,
    smoke_valid_rows: int,
) -> dict[str, int]:
    from datasets import DatasetDict, load_dataset
    from transformers import Seq2SeqTrainingArguments

    from llamafactory.data.loader import _get_preprocessed_dataset
    from llamafactory.data.parser import get_dataset_list
    from llamafactory.data.converter import align_dataset
    from llamafactory.hparams import DataArguments

    dataset_info = {
        train_dataset: {
            "file_name": str(train_file.resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        valid_dataset: {
            "file_name": str(valid_file.resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
    }
    data_args = DataArguments(
        dataset=train_dataset,
        eval_dataset=valid_dataset,
        dataset_dir=".",
        template="qwen3_nothink",
        cutoff_len=cutoff_len,
        train_on_prompt=False,
        packing=True,
        preprocessing_batch_size=preprocessing_batch_size,
        preprocessing_num_workers=1,
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(cache_dir.parent / "smoke-trainer-unused"),
        do_train=True,
        do_eval=True,
        report_to=[],
    )
    datasets = {}
    for name, path, limit, is_eval in (
        (train_dataset, train_file, smoke_train_rows, False),
        (valid_dataset, valid_file, smoke_valid_rows, True),
    ):
        raw = load_dataset(
            "json",
            data_files=[str(path.resolve())],
            split="train",
            cache_dir=str(raw_cache_dir),
        )
        raw = raw.select(range(min(limit, len(raw))))
        attr = get_dataset_list([name], dataset_info)[0]
        aligned = align_dataset(raw, attr, data_args, training_args)
        datasets["validation" if is_eval else "train"] = _get_preprocessed_dataset(
            aligned,
            data_args,
            training_args,
            stage="sft",
            template=template,
            tokenizer=tokenizer,
            processor=tokenizer_module.get("processor"),
            is_eval=is_eval,
        )
    DatasetDict(datasets).save_to_disk(cache_dir)
    return {name: len(dataset) for name, dataset in datasets.items()}


def build_tokenized_cache(
    *,
    model_dir: Path,
    train_file: Path,
    valid_file: Path,
    dataset_dir: Path,
    output_dir: Path,
    cutoff_len: int,
    length_stats: dict[str, Any],
    tokenizer: Any,
    template: Any,
    tokenizer_module: dict[str, Any],
    workers: int,
    preprocessing_batch_size: int,
    train_dataset: str,
    valid_dataset: str,
    mapping_filename: str,
    smoke_train_rows: int,
    smoke_valid_rows: int,
) -> dict[str, Any]:
    """Build the full packed cache plus fixed 10k/2k smoke cache."""

    from llamafactory import __version__ as llamafactory_version
    from llamafactory.data import get_dataset
    from llamafactory.hparams import DataArguments, ModelArguments
    from transformers import Seq2SeqTrainingArguments
    import transformers

    output_dir = output_dir.resolve()
    mapping_path = model_dir / mapping_filename
    if not mapping_path.is_file():
        raise TokenizationPreflightError(f"扩词表映射不存在：{mapping_path}")
    model_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    cache_inputs = {
        "model_dir": str(model_dir.resolve()),
        "extended_tokenizer_sha256": model_mapping["extended_tokenizer_sha256"],
        "train_sha256": length_stats["splits"]["train"]["sha256"],
        "valid_sha256": length_stats["splits"]["valid"]["sha256"],
        "cutoff_len": cutoff_len,
        "packing": True,
        "template": "qwen3_nothink",
        "train_on_prompt": False,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "mapping_filename": mapping_filename,
    }
    manifest_path = output_dir / "cache_manifest.json"
    if output_dir.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TokenizationPreflightError(
                f"Tokenized Cache 已存在但 Manifest 无效，拒绝覆盖：{output_dir}"
            ) from error
        if existing.get("inputs") != cache_inputs:
            raise TokenizationPreflightError(
                "现有 Tokenized Cache 与当前模型、数据或 cutoff 不一致"
            )
        return existing

    build_root = output_dir.with_name(f".{output_dir.name}.building-{os.getpid()}")
    if build_root.exists():
        raise TokenizationPreflightError(f"临时目录已存在：{build_root}")
    full_cache = build_root / "full"
    raw_cache = build_root / "raw"
    unused_output = build_root / "trainer-unused"
    build_root.mkdir(parents=True)
    try:
        model_args = ModelArguments(
            model_name_or_path=str(model_dir.resolve()),
            cache_dir=str(raw_cache),
            use_fast_tokenizer=True,
        )
        data_args = DataArguments(
            dataset=train_dataset,
            eval_dataset=valid_dataset,
            dataset_dir=str(dataset_dir.resolve()),
            template="qwen3_nothink",
            cutoff_len=cutoff_len,
            train_on_prompt=False,
            packing=True,
            preprocessing_batch_size=preprocessing_batch_size,
            preprocessing_num_workers=workers,
            overwrite_cache=False,
            tokenized_path=str(full_cache),
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(unused_output),
            do_train=True,
            do_eval=True,
            report_to=[],
            seed=42,
            data_seed=42,
        )
        dataset_module = get_dataset(
            template,
            model_args,
            data_args,
            training_args,
            stage="sft",
            **tokenizer_module,
        )
        smoke_rows = _preprocess_smoke_cache(
            train_file=train_file,
            valid_file=valid_file,
            cache_dir=full_cache / "smoke",
            raw_cache_dir=raw_cache,
            tokenizer=tokenizer,
            template=template,
            tokenizer_module=tokenizer_module,
            cutoff_len=cutoff_len,
            preprocessing_batch_size=preprocessing_batch_size,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            smoke_train_rows=smoke_train_rows,
            smoke_valid_rows=smoke_valid_rows,
        )
        cache_manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "inputs": cache_inputs,
            "llamafactory_version": llamafactory_version,
            "transformers_version": transformers.__version__,
            "packed_rows": {
                "train": len(dataset_module["train_dataset"]),
                "validation": len(dataset_module["eval_dataset"]),
            },
            "smoke_source_rows": {
                "train": min(
                    smoke_train_rows,
                    int(length_stats["splits"]["train"]["rows"]),
                ),
                "validation": min(
                    smoke_valid_rows,
                    int(length_stats["splits"]["valid"]["rows"]),
                ),
            },
            "smoke_packed_rows": smoke_rows,
        }
        (full_cache / "length_stats.json").write_text(
            json.dumps(length_stats, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (full_cache / "cache_manifest.json").write_text(
            json.dumps(cache_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(raw_cache, ignore_errors=True)
        shutil.rmtree(unused_output, ignore_errors=True)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(full_cache, output_dir)
        shutil.rmtree(build_root, ignore_errors=True)
        return cache_manifest
    except BaseException:
        shutil.rmtree(build_root, ignore_errors=True)
        raise


def validate_sft_tokenization(
    *,
    model_dir: Path,
    train_file: Path,
    valid_file: Path,
    output_dir: Path,
    dataset_dir: Path,
    requested_cutoff: int,
    batch_size: int,
    workers: int,
    preprocessing_batch_size: int,
    train_dataset: str = TRAIN_DATASET,
    valid_dataset: str = VALID_DATASET,
    mapping_filename: str = DEFAULT_MAPPING_FILENAME,
    smoke_train_rows: int = 10_000,
    smoke_valid_rows: int = 2_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if train_file.name == "test.jsonl" or valid_file.name == "test.jsonl":
        raise TokenizationPreflightError("本任务禁止读取 test.jsonl")
    manifest = _load_manifest(train_file, valid_file)
    outputs = manifest["outputs"]
    splits = [
        SplitPreflight(
            "train",
            train_file,
            int(outputs["train.jsonl"]["rows"]),
            outputs["train.jsonl"]["sha256"],
        ),
        SplitPreflight(
            "valid",
            valid_file,
            int(outputs["valid.jsonl"]["rows"]),
            outputs["valid.jsonl"]["sha256"],
        ),
    ]
    tokenizer, template, tokenizer_module = _load_lf_tokenizer_and_template(model_dir)
    mapping_path = model_dir / mapping_filename
    if not mapping_path.is_file():
        raise TokenizationPreflightError(f"扩词表映射不存在：{mapping_path}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    preflight_inputs = {
        "model_dir": str(model_dir.resolve()),
        "extended_tokenizer_sha256": mapping.get("extended_tokenizer_sha256"),
        "train_path": str(train_file.resolve()),
        "train_rows": splits[0].expected_rows,
        "train_sha256": splits[0].expected_sha256,
        "valid_path": str(valid_file.resolve()),
        "valid_rows": splits[1].expected_rows,
        "valid_sha256": splits[1].expected_sha256,
        "requested_cutoff_len": requested_cutoff,
        "template": "qwen3_nothink",
        "train_on_prompt": False,
    }
    preflight_path = output_dir.with_name(f".{output_dir.name}.preflight.json")
    if preflight_path.is_file():
        try:
            preflight_state = json.loads(
                preflight_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise TokenizationPreflightError(
                f"预检状态不是合法 JSON：{preflight_path}"
            ) from error
        if preflight_state.get("inputs") != preflight_inputs:
            raise TokenizationPreflightError(
                f"已有预检状态与当前输入不一致：{preflight_path}"
            )
        stats = preflight_state.get("stats")
        if not isinstance(stats, dict):
            raise TokenizationPreflightError(
                f"已有预检状态缺少 stats：{preflight_path}"
            )
        print(f"复用已完成的长度预检：{preflight_path}", flush=True)
    else:
        stats = preflight_lengths(
            tokenizer,
            template,
            splits,
            requested_cutoff,
            batch_size,
        )
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_preflight = preflight_path.with_name(
            f"{preflight_path.name}.tmp-{os.getpid()}"
        )
        temporary_preflight.write_text(
            json.dumps(
                {"inputs": preflight_inputs, "stats": stats},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_preflight, preflight_path)
        print(f"长度预检状态已保存：{preflight_path}", flush=True)
    enforce_safe_cutoff_contract(stats, requested_cutoff)
    effective_cutoff = int(stats["effective_cutoff_len"])
    cache_manifest = build_tokenized_cache(
        model_dir=model_dir,
        train_file=train_file,
        valid_file=valid_file,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        cutoff_len=effective_cutoff,
        length_stats=stats,
        tokenizer=tokenizer,
        template=template,
        tokenizer_module=tokenizer_module,
        workers=workers,
        preprocessing_batch_size=preprocessing_batch_size,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        mapping_filename=mapping_filename,
        smoke_train_rows=smoke_train_rows,
        smoke_valid_rows=smoke_valid_rows,
    )
    return stats, cache_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="仅扫描 Train/Valid 的 Token 长度并构建可复用的 LLaMA-Factory 缓存。"
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="扩词表后的模型目录")
    parser.add_argument("--train-file", type=Path, required=True, help="Train JSONL")
    parser.add_argument("--valid-file", type=Path, required=True, help="Valid JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Tokenized Cache 目录")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("configs/sft"),
        help="包含 dataset_info.json 的 LLaMA-Factory 数据注册目录",
    )
    parser.add_argument(
        "--cutoff-len",
        type=int,
        default=1024,
        help="整条 Source+Target 的 Token 上限，后续实验默认 1024",
    )
    parser.add_argument("--batch-size", type=int, default=4096, help="长度统计批大小")
    parser.add_argument("--workers", type=int, default=16, help="缓存预处理进程数")
    parser.add_argument(
        "--preprocessing-batch-size",
        type=int,
        default=1000,
        help="LLaMA-Factory packing 分组大小",
    )
    parser.add_argument(
        "--train-dataset",
        default=TRAIN_DATASET,
        help=f"Train 数据注册名，默认 {TRAIN_DATASET}",
    )
    parser.add_argument(
        "--valid-dataset",
        default=VALID_DATASET,
        help=f"Valid 数据注册名，默认 {VALID_DATASET}",
    )
    parser.add_argument(
        "--mapping-filename",
        default=DEFAULT_MAPPING_FILENAME,
        help=f"扩词表映射文件名，默认 {DEFAULT_MAPPING_FILENAME}",
    )
    parser.add_argument(
        "--smoke-train-rows",
        type=int,
        default=10_000,
        help="一并构建的 smoke cache 最大 Train 原始行数",
    )
    parser.add_argument(
        "--smoke-valid-rows",
        type=int,
        default=2_000,
        help="一并构建的 smoke cache 最大 Valid 原始行数",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cutoff_len not in (128, 256, 512, 1024):
        raise SystemExit("--cutoff-len 只允许 128、256、512 或 1024")
    if args.smoke_train_rows <= 0 or args.smoke_valid_rows <= 0:
        raise SystemExit("smoke 行数必须大于 0")
    stats, cache_manifest = validate_sft_tokenization(
        model_dir=args.model_dir.resolve(),
        train_file=args.train_file.resolve(),
        valid_file=args.valid_file.resolve(),
        output_dir=args.output_dir.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        requested_cutoff=args.cutoff_len,
        batch_size=args.batch_size,
        workers=args.workers,
        preprocessing_batch_size=args.preprocessing_batch_size,
        train_dataset=args.train_dataset,
        valid_dataset=args.valid_dataset,
        mapping_filename=args.mapping_filename,
        smoke_train_rows=args.smoke_train_rows,
        smoke_valid_rows=args.smoke_valid_rows,
    )
    print(
        "Token 预检与缓存完成："
        f"cutoff_len={stats['effective_cutoff_len']}，"
        f"超过128={stats['over_128_count']}，"
        f"超过cutoff={stats['over_requested_cutoff_count']}，"
        "目标截断="
        f"{stats['target_truncated_at_effective_cutoff_count']}，"
        f"packed train/valid={cache_manifest['packed_rows']['train']}/"
        f"{cache_manifest['packed_rows']['validation']}"
    )


if __name__ == "__main__":
    main()

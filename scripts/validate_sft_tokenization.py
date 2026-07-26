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
from typing import Any, Iterable, Sequence

import numpy as np


CACHE_SCHEMA_VERSION = "poi-sft-tokenized-v1"
TRAIN_DATASET = "beijing_order_main_v1_train"
VALID_DATASET = "beijing_order_main_v1_valid"
IGNORE_INDEX = -100


class TokenizationPreflightError(ValueError):
    pass


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
) -> tuple[int, int, int]:
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
                    over_128, truncated, truncated_at_256 = _process_batch(
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
                    split.target_truncated_at_requested_cutoff += truncated
                    split.target_truncated_at_256 += truncated_at_256
                    users.clear()
                    targets.clear()
            if users:
                over_128, truncated, truncated_at_256 = _process_batch(
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
        "target_truncated_at_requested_cutoff_count": requested_truncated,
        "target_truncated_at_effective_cutoff_count": truncated_at_effective,
        "splits": {
            split.name: {
                "path": str(split.path.resolve()),
                "rows": split.row_count,
                "sha256": split.sha256,
                "over_128_count": split.over_128_count,
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
) -> dict[str, int]:
    from datasets import DatasetDict, load_dataset
    from transformers import Seq2SeqTrainingArguments

    from llamafactory.data.loader import _get_preprocessed_dataset
    from llamafactory.data.parser import get_dataset_list
    from llamafactory.data.converter import align_dataset
    from llamafactory.hparams import DataArguments

    dataset_info = {
        TRAIN_DATASET: {
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
        VALID_DATASET: {
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
        dataset=f"{TRAIN_DATASET}",
        eval_dataset=f"{VALID_DATASET}",
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
        (TRAIN_DATASET, train_file, 10_000, False),
        (VALID_DATASET, valid_file, 2_000, True),
    ):
        raw = load_dataset(
            "json",
            data_files=[str(path.resolve())],
            split="train",
            cache_dir=str(raw_cache_dir),
        ).select(range(limit))
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
) -> dict[str, Any]:
    """Build the full packed cache plus fixed 10k/2k smoke cache."""

    from llamafactory import __version__ as llamafactory_version
    from llamafactory.data import get_dataset
    from llamafactory.hparams import DataArguments, ModelArguments
    from transformers import Seq2SeqTrainingArguments
    import transformers

    output_dir = output_dir.resolve()
    mapping_path = model_dir / "poi_token_mapping.json"
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
            dataset=TRAIN_DATASET,
            eval_dataset=VALID_DATASET,
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
            "smoke_source_rows": {"train": 10_000, "validation": 2_000},
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
    stats = preflight_lengths(
        tokenizer,
        template,
        splits,
        requested_cutoff,
        batch_size,
    )
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
    parser.add_argument("--cutoff-len", type=int, default=128, help="首选截断长度")
    parser.add_argument("--batch-size", type=int, default=4096, help="长度统计批大小")
    parser.add_argument("--workers", type=int, default=16, help="缓存预处理进程数")
    parser.add_argument(
        "--preprocessing-batch-size",
        type=int,
        default=1000,
        help="LLaMA-Factory packing 分组大小",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cutoff_len not in (128, 256):
        raise SystemExit("--cutoff-len 只允许 128 或 256")
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
    )
    print(
        "Token 预检与缓存完成："
        f"cutoff_len={stats['effective_cutoff_len']}，"
        f"超过128={stats['over_128_count']}，"
        "目标截断="
        f"{stats['target_truncated_at_effective_cutoff_count']}，"
        f"packed train/valid={cache_manifest['packed_rows']['train']}/"
        f"{cache_manifest['packed_rows']['validation']}"
    )


if __name__ == "__main__":
    main()

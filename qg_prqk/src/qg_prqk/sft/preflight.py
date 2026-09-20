"""Run full Train/Valid token preflight and build the SFT cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from qg_prqk.artifacts import sha256_file, utc_now
from qg_prqk.sft.data import SFT_DATA_SCHEMA_VERSION
from qg_prqk.sft.vocabulary import MAPPING_FILENAME, VOCAB_SCHEMA_VERSION


CACHE_SCHEMA_VERSION = "qg-prqk-sft-tokenized-v1"
ALLOWED_VARIANTS = ("a4_gid_parent", "a4_nogid")
REQUIRED_CUTOFF_LEN = 1024


class SftPreflightError(ValueError):
    """Raised when tokenization or cache inputs violate the frozen contract."""


def _ensure_llamafactory_path() -> Path:
    root = Path(__file__).resolve().parents[4]
    source = root / "third_party/LLaMA-Factory/src"
    if not (source / "llamafactory").is_dir():
        raise SftPreflightError(f"缺少本地 LLaMA-Factory：{source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


@dataclass
class _Histogram:
    counts: np.ndarray
    total: int = 0

    @classmethod
    def empty(cls) -> "_Histogram":
        return cls(np.zeros(0, dtype=np.int64))

    def update(self, values: Sequence[int]) -> None:
        array = np.asarray(values, dtype=np.int64)
        if array.size == 0:
            return
        batch = np.bincount(array)
        if batch.size > self.counts.size:
            self.counts = np.pad(self.counts, (0, batch.size - self.counts.size))
        self.counts[: batch.size] += batch
        self.total += int(array.size)

    def percentile(self, quantile: float) -> float:
        if self.total <= 0:
            raise SftPreflightError("空数据不能计算长度分位数")
        rank = (self.total - 1) * quantile
        low_rank = int(np.floor(rank))
        high_rank = int(np.ceil(rank))
        cumulative = np.cumsum(self.counts)
        low = int(np.searchsorted(cumulative, low_rank + 1))
        high = int(np.searchsorted(cumulative, high_rank + 1))
        return float(low + (high - low) * (rank - low_rank))

    def summary(self) -> dict[str, float | int]:
        return {
            "p50": self.percentile(0.50),
            "p90": self.percentile(0.90),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "p99_9": self.percentile(0.999),
            "max": int(np.flatnonzero(self.counts)[-1]),
        }


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SftPreflightError(f"{name} 不是合法 JSON：{path}") from error
    if not isinstance(payload, dict):
        raise SftPreflightError(f"{name} 必须是 JSON object")
    return payload


def _load_sft_contract(data_dir: Path, variant: str) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    manifest = _load_json(manifest_path, "SFT manifest")
    if (
        manifest.get("schema_version") != SFT_DATA_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("scan_mode") != "full"
        or manifest.get("variant") != variant
        or not (data_dir / "_SUCCESS").is_file()
    ):
        raise SftPreflightError("SFT 数据不是对应 variant 的已完成全量产物")
    outputs = manifest.get("outputs")
    for filename in ("train.jsonl", "valid.jsonl"):
        spec = outputs.get(filename) if isinstance(outputs, dict) else None
        path = data_dir / filename
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("rows"), int)
            or spec["rows"] <= 0
            or not isinstance(spec.get("sha256"), str)
            or not path.is_file()
        ):
            raise SftPreflightError(f"SFT manifest 缺少合法 {filename} 契约")
    return manifest


def _load_tokenizer_and_template(model_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    _ensure_llamafactory_path()
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    mapping = _load_json(model_dir / MAPPING_FILENAME, "扩词表映射")
    if mapping.get("schema_version") != VOCAB_SCHEMA_VERSION:
        raise SftPreflightError("扩词表映射版本不符合 QG-PRQK SFT 契约")
    model_args = ModelArguments(
        model_name_or_path=str(model_dir),
        use_fast_tokenizer=True,
    )
    tokenizer_module = load_tokenizer(model_args)
    data_args = DataArguments(template="qwen3_nothink", train_on_prompt=False)
    template = get_template_and_fix_tokenizer(
        tokenizer_module["tokenizer"], data_args
    )
    return tokenizer_module["tokenizer"], template, tokenizer_module


def _format_batch(
    template: Any, users: Sequence[str], targets: Sequence[str]
) -> tuple[list[str], list[str]]:
    if template.format_prefix.apply():
        raise SftPreflightError("qwen3_nothink 模板出现了未预期的 prefix")
    prompts: list[str] = []
    answers: list[str] = []
    for user, target in zip(users, targets, strict=True):
        user_slots = template.format_user.apply(content=user, idx="0")
        target_slots = template.format_assistant.apply(content=target)
        if (
            len(user_slots) != 1
            or not isinstance(user_slots[0], str)
            or len(target_slots) != 1
            or not isinstance(target_slots[0], str)
        ):
            raise SftPreflightError("qwen3_nothink 不再是单字符串格式")
        prompts.append(user_slots[0])
        answers.append(target_slots[0])
    return prompts, answers


def _validate_record(
    record: Mapping[str, Any], split: str, line_number: int, variant: str
) -> tuple[str, str]:
    messages = record.get("messages")
    if (
        record.get("split") != split
        or record.get("identifier_variant") != variant
        or not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(messages[0], dict)
        or not isinstance(messages[1], dict)
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
        or not isinstance(messages[0].get("content"), str)
        or not isinstance(messages[1].get("content"), str)
    ):
        raise SftPreflightError(f"{split}:{line_number} Messages 或 variant 非法")
    target = messages[1]["content"]
    if "<D_-1>" in target:
        raise SftPreflightError(f"{split}:{line_number} 包含非法 <D_-1>")
    if variant == "a4_nogid" and "<G_" in target:
        raise SftPreflightError(f"{split}:{line_number} NoGID target 含 GID")
    return messages[0]["content"], target


def scan_train_valid_lengths(
    *,
    data_dir: Path,
    variant: str,
    manifest: Mapping[str, Any],
    tokenizer: Any,
    template: Any,
    cutoff_len: int,
    batch_size: int,
) -> dict[str, Any]:
    """Hash and tokenize every Train/Valid row without reading Test."""

    if cutoff_len != REQUIRED_CUTOFF_LEN:
        raise SftPreflightError("QG-PRQK SFT cutoff_len 固定为 1024")
    if batch_size <= 0:
        raise SftPreflightError("batch_size 必须大于 0")
    prompt_hist = _Histogram.empty()
    target_hist = _Histogram.empty()
    total_hist = _Histogram.empty()
    split_stats: dict[str, Any] = {}
    over_cutoff = 0
    target_truncated = 0

    for split in ("train", "valid"):
        path = data_dir / f"{split}.jsonl"
        expected = manifest["outputs"][path.name]
        digest = hashlib.sha256()
        rows = 0
        started_at = time.monotonic()
        next_progress_row = 250_000
        split_over = 0
        users: list[str] = []
        targets: list[str] = []

        def consume() -> None:
            nonlocal split_over, over_cutoff, target_truncated
            if not users:
                return
            prompt_texts, answer_texts = _format_batch(template, users, targets)
            prompt_lengths = np.asarray(
                tokenizer(
                    prompt_texts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_length=True,
                )["length"],
                dtype=np.int64,
            )
            target_lengths = np.asarray(
                tokenizer(
                    answer_texts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_length=True,
                )["length"],
                dtype=np.int64,
            )
            totals = prompt_lengths + target_lengths
            count = int(np.count_nonzero(totals > cutoff_len))
            split_over += count
            over_cutoff += count
            target_truncated += count
            prompt_hist.update(prompt_lengths)
            target_hist.update(target_lengths)
            total_hist.update(totals)
            users.clear()
            targets.clear()

        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise SftPreflightError(
                        f"{split}:{line_number} 不是合法 JSON"
                    ) from error
                if not isinstance(record, dict):
                    raise SftPreflightError(f"{split}:{line_number} 必须是 object")
                user, target = _validate_record(record, split, line_number, variant)
                users.append(user)
                targets.append(target)
                rows += 1
                if len(users) >= batch_size:
                    consume()
                if rows >= next_progress_row:
                    elapsed = max(time.monotonic() - started_at, 1e-9)
                    print(
                        json.dumps(
                            {
                                "stage": "sft_token_preflight_progress",
                                "variant": variant,
                                "split": split,
                                "rows": rows,
                                "expected_rows": expected["rows"],
                                "rows_per_second": round(rows / elapsed, 2),
                                "over_cutoff_count": split_over,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    next_progress_row += 250_000
            consume()
        observed_hash = digest.hexdigest()
        if rows != expected["rows"] or observed_hash != expected["sha256"]:
            raise SftPreflightError(f"{split} 行数或 SHA256 与 SFT manifest 不一致")
        split_stats[split] = {
            "path": str(path),
            "rows": rows,
            "sha256": observed_hash,
            "over_cutoff_count": split_over,
            "target_truncated_count": split_over,
        }
        print(
            json.dumps(
                {
                    "stage": "sft_token_preflight_split_completed",
                    "variant": variant,
                    "split": split,
                    "rows": rows,
                    "over_cutoff_count": split_over,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    total_rows = sum(item["rows"] for item in split_stats.values())
    return {
        "definition": {
            "prompt": "qwen3_nothink formatted user prompt",
            "target": "qwen3_nothink formatted assistant target",
            "total": "prompt + target before truncation",
        },
        "cutoff_len": cutoff_len,
        "row_count": total_rows,
        "prompt_length": prompt_hist.summary(),
        "target_length": target_hist.summary(),
        "total_length": total_hist.summary(),
        "over_cutoff_count": over_cutoff,
        "target_truncated_count": target_truncated,
        "splits": split_stats,
    }


def _validate_zero_truncation(stats: Mapping[str, Any]) -> None:
    over = int(stats.get("over_cutoff_count", -1))
    target = int(stats.get("target_truncated_count", -1))
    if over != 0 or target != 0:
        raise SftPreflightError(
            f"全量预检失败：超过 1024={over}，目标截断={target}；拒绝构建缓存"
        )


def _cache_inputs(
    *,
    variant: str,
    model_dir: Path,
    mapping: Mapping[str, Any],
    stats: Mapping[str, Any],
    train_dataset: str,
    valid_dataset: str,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "model_dir": str(model_dir),
        "extended_tokenizer_sha256": mapping["extended_tokenizer_sha256"],
        "train_sha256": stats["splits"]["train"]["sha256"],
        "valid_sha256": stats["splits"]["valid"]["sha256"],
        "cutoff_len": REQUIRED_CUTOFF_LEN,
        "packing": True,
        "template": "qwen3_nothink",
        "train_on_prompt": False,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
    }


def _build_cache(
    *,
    model_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    inputs: Mapping[str, Any],
    stats: Mapping[str, Any],
    tokenizer: Any,
    template: Any,
    tokenizer_module: Mapping[str, Any],
    workers: int,
    preprocessing_batch_size: int,
) -> dict[str, Any]:
    _ensure_llamafactory_path()
    from llamafactory import __version__ as llamafactory_version
    from llamafactory.data import get_dataset
    from llamafactory.hparams import DataArguments, ModelArguments
    from transformers import Seq2SeqTrainingArguments, __version__ as transformers_version

    if output_dir.exists():
        existing = _load_json(output_dir / "cache_manifest.json", "Cache manifest")
        if existing.get("inputs") != dict(inputs):
            raise SftPreflightError("现有 Cache 与当前数据、模型或配置不一致")
        if not (output_dir / "_SUCCESS").is_file():
            raise SftPreflightError("现有 Cache 缺少 _SUCCESS")
        return existing

    temporary = output_dir.with_name(f".{output_dir.name}.building-{os.getpid()}")
    raw_cache = temporary / "raw"
    dataset_cache = temporary / "dataset"
    unused_output = temporary / "trainer-unused"
    if temporary.exists():
        raise SftPreflightError(f"临时目录已存在：{temporary}")
    temporary.mkdir(parents=True)
    try:
        model_args = ModelArguments(
            model_name_or_path=str(model_dir),
            cache_dir=str(raw_cache),
            use_fast_tokenizer=True,
        )
        data_args = DataArguments(
            dataset=str(inputs["train_dataset"]),
            eval_dataset=str(inputs["valid_dataset"]),
            dataset_dir=str(dataset_dir),
            template="qwen3_nothink",
            cutoff_len=REQUIRED_CUTOFF_LEN,
            train_on_prompt=False,
            packing=True,
            preprocessing_batch_size=preprocessing_batch_size,
            preprocessing_num_workers=workers,
            overwrite_cache=False,
            tokenized_path=str(dataset_cache),
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(unused_output),
            do_train=True,
            do_eval=True,
            report_to=[],
            seed=42,
            data_seed=42,
        )
        datasets = get_dataset(
            template,
            model_args,
            data_args,
            training_args,
            stage="sft",
            **tokenizer_module,
        )
        packed_rows = {
            "train": len(datasets["train_dataset"]),
            "validation": len(datasets["eval_dataset"]),
        }
        if min(packed_rows.values()) <= 0:
            raise SftPreflightError("Tokenized Cache 的 Train/Valid 为空")
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "completed",
            "built_at": utc_now(),
            "inputs": dict(inputs),
            "preflight": dict(stats),
            "packed_rows": packed_rows,
            "llamafactory_version": llamafactory_version,
            "transformers_version": transformers_version,
        }
        (dataset_cache / "cache_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (dataset_cache / "_SUCCESS").touch()
        shutil.rmtree(raw_cache, ignore_errors=True)
        shutil.rmtree(unused_output, ignore_errors=True)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(dataset_cache, output_dir)
        shutil.rmtree(temporary, ignore_errors=True)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_sft_cache(
    *,
    variant: str,
    data_dir: Path,
    model_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    train_dataset: str,
    valid_dataset: str,
    batch_size: int = 4096,
    workers: int = 16,
    preprocessing_batch_size: int = 1000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run or safely reuse full preflight, then build an immutable cache."""

    if variant not in ALLOWED_VARIANTS:
        raise SftPreflightError(f"未知 variant：{variant}")
    data_dir = data_dir.resolve()
    model_dir = model_dir.resolve()
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    if Path(train_dataset).name != train_dataset or Path(valid_dataset).name != valid_dataset:
        raise SftPreflightError("Dataset 注册名不能包含路径")
    if "test" in train_dataset.lower() or "test" in valid_dataset.lower():
        raise SftPreflightError("SFT 预检与缓存禁止读取 Test")
    manifest = _load_sft_contract(data_dir, variant)
    mapping = _load_json(model_dir / MAPPING_FILENAME, "扩词表映射")
    tokenizer, template, tokenizer_module = _load_tokenizer_and_template(model_dir)
    expected_preflight_inputs = {
        "variant": variant,
        "model_dir": str(model_dir),
        "extended_tokenizer_sha256": mapping.get("extended_tokenizer_sha256"),
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "cutoff_len": REQUIRED_CUTOFF_LEN,
    }
    state_path = output_dir.with_name(f".{output_dir.name}.preflight.json")
    if state_path.is_file():
        state = _load_json(state_path, "预检状态")
        if state.get("inputs") != expected_preflight_inputs:
            raise SftPreflightError("已有预检状态与当前输入不一致")
        stats = state.get("stats")
        if not isinstance(stats, dict):
            raise SftPreflightError("已有预检状态缺少 stats")
    else:
        stats = scan_train_valid_lengths(
            data_dir=data_dir,
            variant=variant,
            manifest=manifest,
            tokenizer=tokenizer,
            template=template,
            cutoff_len=REQUIRED_CUTOFF_LEN,
            batch_size=batch_size,
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_state = state_path.with_name(f"{state_path.name}.tmp-{os.getpid()}")
        temporary_state.write_text(
            json.dumps(
                {"inputs": expected_preflight_inputs, "stats": stats},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_state, state_path)
    _validate_zero_truncation(stats)
    # Full preflight may use the fast tokenizer's thread pool. Disable that pool
    # before LLaMA-Factory forks dataset workers so cache construction cannot
    # inherit active Rayon threads.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    inputs = _cache_inputs(
        variant=variant,
        model_dir=model_dir,
        mapping=mapping,
        stats=stats,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    )
    cache = _build_cache(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        inputs=inputs,
        stats=stats,
        tokenizer=tokenizer,
        template=template,
        tokenizer_module=tokenizer_module,
        workers=workers,
        preprocessing_batch_size=preprocessing_batch_size,
    )
    return stats, cache

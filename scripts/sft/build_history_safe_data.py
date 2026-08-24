#!/usr/bin/env python
"""Build SFT JSONL files that reserve the token budget for CURRENT and target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
LLAMAFACTORY_ROOT = PROJECT_ROOT / "third_party" / "LLaMA-Factory" / "src"
for import_root in (SRC_ROOT, LLAMAFACTORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from poi_gr.sft.history_budget import (  # noqa: E402
    HistoryBudgetError,
    SequenceLengths,
    fit_history_to_token_budget,
)


SPLIT_FILES = ("train.jsonl", "valid.jsonl", "test.jsonl")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokenizer_and_template(model_dir: Path) -> tuple[Any, Any]:
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    model_args = ModelArguments(
        model_name_or_path=str(model_dir.resolve()),
        use_fast_tokenizer=True,
    )
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    data_args = DataArguments(template="qwen3_nothink", train_on_prompt=False)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    return tokenizer, template


def _format_batch(
    template: Any,
    users: Sequence[str],
    targets: Sequence[str],
) -> tuple[list[str], list[str]]:
    if template.format_prefix.apply():
        raise HistoryBudgetError("qwen3_nothink 预期没有额外 prefix slot")
    prompt_texts: list[str] = []
    target_texts: list[str] = []
    for user, target in zip(users, targets):
        user_slots = template.format_user.apply(content=user, idx="0")
        target_slots = template.format_assistant.apply(content=target)
        if (
            len(user_slots) != 1
            or not isinstance(user_slots[0], str)
            or len(target_slots) != 1
            or not isinstance(target_slots[0], str)
        ):
            raise HistoryBudgetError("qwen3_nothink 模板格式发生变化")
        prompt_texts.append(user_slots[0])
        target_texts.append(target_slots[0])
    return prompt_texts, target_texts


def _measure_batch(
    tokenizer: Any,
    template: Any,
    users: Sequence[str],
    targets: Sequence[str],
) -> list[SequenceLengths]:
    prompt_texts, target_texts = _format_batch(template, users, targets)
    source = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        return_length=True,
        padding=False,
        truncation=False,
    )["length"]
    target = tokenizer(
        target_texts,
        add_special_tokens=False,
        return_length=True,
        padding=False,
        truncation=False,
    )["length"]
    return [
        SequenceLengths(source=int(source_len), target=int(target_len))
        for source_len, target_len in zip(source, target)
    ]


def _validate_record(record: dict[str, Any], split: str, line_number: int) -> tuple[str, str]:
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
        or not isinstance(messages[0].get("content"), str)
        or not isinstance(messages[1].get("content"), str)
    ):
        raise HistoryBudgetError(f"{split} 第 {line_number} 行 messages 不合法")
    if record.get("split") != split:
        raise HistoryBudgetError(f"{split} 第 {line_number} 行 split 字段不一致")
    return messages[0]["content"], messages[1]["content"]


def _flush_batch(
    *,
    records: list[dict[str, Any]],
    raw_lines: list[bytes],
    line_numbers: list[int],
    split: str,
    output_stream: Any,
    output_digest: Any,
    tokenizer: Any,
    template: Any,
    cutoff_len: int,
    stats: dict[str, int],
) -> None:
    users: list[str] = []
    targets: list[str] = []
    for record, line_number in zip(records, line_numbers):
        user, target = _validate_record(record, split, line_number)
        users.append(user)
        targets.append(target)
    lengths = _measure_batch(tokenizer, template, users, targets)

    for record, raw_line, line_number, user, target, original_lengths in zip(
        records,
        raw_lines,
        line_numbers,
        users,
        targets,
        lengths,
    ):
        stats["rows"] += 1
        stats["max_total_tokens_before"] = max(
            stats["max_total_tokens_before"], original_lengths.total
        )
        output_line = raw_line
        final_lengths = original_lengths
        if original_lengths.total > cutoff_len:
            def measure(candidate_user: str, candidate_target: str) -> SequenceLengths:
                return _measure_batch(
                    tokenizer,
                    template,
                    [candidate_user],
                    [candidate_target],
                )[0]

            try:
                result = fit_history_to_token_budget(
                    user,
                    target,
                    cutoff_len,
                    measure,
                )
            except HistoryBudgetError as error:
                raise HistoryBudgetError(
                    f"{split} 第 {line_number} 行无法安全裁剪：{error}"
                ) from error
            record["messages"][0]["content"] = result.user_content
            if "history_length" in record:
                record["history_length"] = result.retained_history_events
            output_line = (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            final_lengths = result.lengths_after
            stats["modified_rows"] += 1
            stats["removed_history_events"] += result.removed_history_events
            stats["max_removed_history_events_per_row"] = max(
                stats["max_removed_history_events_per_row"],
                result.removed_history_events,
            )
        if final_lengths.total > cutoff_len:
            raise AssertionError("安全裁剪后仍超过 cutoff_len")
        stats["max_total_tokens_after"] = max(
            stats["max_total_tokens_after"], final_lengths.total
        )
        output_stream.write(output_line)
        output_digest.update(output_line)


def _process_split(
    *,
    source_path: Path,
    output_path: Path,
    split: str,
    expected: dict[str, Any],
    tokenizer: Any,
    template: Any,
    cutoff_len: int,
    batch_size: int,
) -> dict[str, Any]:
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    stats = {
        "rows": 0,
        "modified_rows": 0,
        "removed_history_events": 0,
        "max_removed_history_events_per_row": 0,
        "max_total_tokens_before": 0,
        "max_total_tokens_after": 0,
    }
    records: list[dict[str, Any]] = []
    raw_lines: list[bytes] = []
    line_numbers: list[int] = []
    with source_path.open("rb") as source_stream, output_path.open("wb") as output_stream:
        for line_number, raw_line in enumerate(source_stream, 1):
            input_digest.update(raw_line)
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise HistoryBudgetError(
                    f"{split} 第 {line_number} 行不是合法 JSON"
                ) from error
            records.append(record)
            raw_lines.append(raw_line)
            line_numbers.append(line_number)
            if len(records) >= batch_size:
                _flush_batch(
                    records=records,
                    raw_lines=raw_lines,
                    line_numbers=line_numbers,
                    split=split,
                    output_stream=output_stream,
                    output_digest=output_digest,
                    tokenizer=tokenizer,
                    template=template,
                    cutoff_len=cutoff_len,
                    stats=stats,
                )
                records.clear()
                raw_lines.clear()
                line_numbers.clear()
            if line_number % 1_000_000 == 0:
                print(f"安全裁剪 {split}: {line_number:,} 行", flush=True)
        if records:
            _flush_batch(
                records=records,
                raw_lines=raw_lines,
                line_numbers=line_numbers,
                split=split,
                output_stream=output_stream,
                output_digest=output_digest,
                tokenizer=tokenizer,
                template=template,
                cutoff_len=cutoff_len,
                stats=stats,
            )

    if stats["rows"] != int(expected["rows"]):
        raise HistoryBudgetError(
            f"{split} 行数不一致：{stats['rows']} != {expected['rows']}"
        )
    if input_digest.hexdigest() != expected["sha256"]:
        raise HistoryBudgetError(f"{split} 输入 SHA256 与源 Manifest 不一致")
    return {
        **stats,
        "input_sha256": input_digest.hexdigest(),
        "output_sha256": output_digest.hexdigest(),
    }


def build_history_safe_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    model_dir: Path,
    cutoff_len: int,
    batch_size: int,
) -> dict[str, Any]:
    if cutoff_len <= 0 or batch_size <= 0:
        raise HistoryBudgetError("cutoff_len 和 batch_size 必须大于 0")
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise HistoryBudgetError("输出目录不能覆盖源数据目录")
    if output_dir.exists():
        raise HistoryBudgetError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    source_manifest_path = source_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise HistoryBudgetError(f"源数据缺少 manifest.json：{source_dir}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_outputs = source_manifest.get("outputs")
    if not isinstance(source_outputs, dict):
        raise HistoryBudgetError("源 Manifest 缺少 outputs")
    history_contract = source_manifest.get("history")
    if (
        not isinstance(history_contract, dict)
        or history_contract.get("order") != "event_time_ascending"
    ):
        raise HistoryBudgetError(
            "源 Manifest 必须声明 history.order=event_time_ascending，"
            "才能确定最早历史事件"
        )

    tokenizer, template = _load_tokenizer_and_template(model_dir)
    build_dir = output_dir.with_name(f".{output_dir.name}.building-{os.getpid()}")
    if build_dir.exists():
        raise HistoryBudgetError(f"临时目录已存在：{build_dir}")
    build_dir.mkdir(parents=True)
    split_stats: dict[str, Any] = {}
    output_entries: dict[str, Any] = {}
    try:
        for filename in SPLIT_FILES:
            expected = source_outputs.get(filename)
            if not isinstance(expected, dict):
                continue
            split = filename.removesuffix(".jsonl")
            source_path = source_dir / filename
            if not source_path.is_file():
                raise HistoryBudgetError(f"源数据缺少 {filename}")
            stats = _process_split(
                source_path=source_path,
                output_path=build_dir / filename,
                split=split,
                expected=expected,
                tokenizer=tokenizer,
                template=template,
                cutoff_len=cutoff_len,
                batch_size=batch_size,
            )
            split_stats[split] = stats
            output_entries[filename] = {
                "rows": stats["rows"],
                "sha256": stats["output_sha256"],
            }

        if not {"train", "valid"}.issubset(split_stats):
            raise HistoryBudgetError("源数据至少必须包含 train.jsonl 和 valid.jsonl")

        special_tokens = source_dir / "special_tokens.json"
        if special_tokens.is_file():
            shutil.copy2(special_tokens, build_dir / special_tokens.name)
            output_entries[special_tokens.name] = {
                "sha256": sha256_file(build_dir / special_tokens.name)
            }

        stats_payload = {
            "cutoff_len": cutoff_len,
            "template": "qwen3_nothink",
            "policy": "remove_oldest_history_events_only",
            "immutable_blocks": ["CURRENT", "assistant_target"],
            "splits": split_stats,
        }
        stats_path = build_dir / "history_budget_stats.json"
        stats_path.write_text(
            json.dumps(stats_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        output_entries[stats_path.name] = {"sha256": sha256_file(stats_path)}

        manifest = deepcopy(source_manifest)
        manifest["built_at"] = datetime.now(timezone.utc).isoformat()
        manifest["source_dataset"] = {
            "path": str(source_dir),
            "manifest_sha256": sha256_file(source_manifest_path),
            "schema_version": source_manifest.get("schema_version"),
        }
        manifest["history_budget"] = {
            "cutoff_len": cutoff_len,
            "template": "qwen3_nothink",
            "history_order": "event_time_ascending",
            "overflow_policy": "remove_oldest_history_events_only",
            "forbid_current_or_target_truncation": True,
        }
        manifest["outputs"] = output_entries
        fingerprint_payload = {
            "source_manifest_sha256": manifest["source_dataset"]["manifest_sha256"],
            "history_budget": manifest["history_budget"],
            "outputs": output_entries,
        }
        manifest["build_fingerprint"] = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        (build_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(build_dir, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "按 qwen3_nothink 的真实 Token 长度构建安全 SFT 数据；超长时只删除最早历史，"
            "不截断 CURRENT、当前 Query 或 Assistant 目标。"
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="源 SFT 数据目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="派生 SFT 数据目录")
    parser.add_argument("--model-dir", type=Path, required=True, help="扩词表后的模型目录")
    parser.add_argument(
        "--cutoff-len",
        type=int,
        default=1024,
        help="整条 Source+Target 的 Token 上限，默认 1024",
    )
    parser.add_argument("--batch-size", type=int, default=4096, help="Token 长度统计批大小")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_history_safe_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        cutoff_len=args.cutoff_len,
        batch_size=args.batch_size,
    )
    print(
        "历史安全 SFT 数据构建完成："
        f"cutoff_len={manifest['history_budget']['cutoff_len']}，"
        f"output={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()

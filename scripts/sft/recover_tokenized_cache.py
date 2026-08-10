#!/usr/bin/env python
"""Recover a sharded SFT cache after preprocessing was interrupted."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLAMAFACTORY_SRC = PROJECT_ROOT / "third_party" / "LLaMA-Factory" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LLAMAFACTORY_SRC) not in sys.path:
    sys.path.insert(0, str(LLAMAFACTORY_SRC))

from scripts.sft.validate_tokenization import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    _load_lf_tokenizer_and_template,
)


PACKED_COLUMNS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "labels",
    "images",
    "videos",
    "audios",
}
ALIGNED_COLUMNS = {
    "_prompt",
    "_response",
    "_system",
    "_tools",
    "_images",
    "_videos",
    "_audios",
}
SHARD_PATTERN = re.compile(
    r"^(?P<prefix>cache-[0-9a-f]+)_(?P<index>\d{5})_of_(?P<count>\d{5})\.arrow$"
)


class CacheRecoveryError(ValueError):
    pass


@dataclass(frozen=True)
class ArrowGroup:
    files: tuple[Path, ...]
    source_rows: int
    columns: frozenset[str]


def _arrow_columns(path: Path) -> frozenset[str]:
    with pa.memory_map(str(path), "r") as source:
        return frozenset(ipc.open_stream(source).schema.names)


def _source_rows(path: Path) -> int:
    info_path = path.parent / "dataset_info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        splits = info["splits"]
        split = next(iter(splits.values()))
        return int(split["num_examples"])
    except (OSError, KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CacheRecoveryError(f"无法读取 Arrow 来源行数：{info_path}") from error


def discover_arrow_group(
    root: Path,
    *,
    source_rows: int,
    required_columns: set[str],
) -> ArrowGroup:
    """Find one complete Arrow cache group by source rows and schema."""

    grouped: dict[tuple[Path, str, int], dict[int, Path]] = {}
    for path in root.rglob("cache-*.arrow"):
        match = SHARD_PATTERN.match(path.name)
        if match is None:
            continue
        count = int(match.group("count"))
        index = int(match.group("index"))
        grouped.setdefault((path.parent, match.group("prefix"), count), {})[
            index
        ] = path

    matches: list[ArrowGroup] = []
    for (parent, _, count), indexed in grouped.items():
        if len(indexed) != count or sorted(indexed) != list(range(count)):
            continue
        files = tuple(indexed[index] for index in range(count))
        if _source_rows(files[0]) != source_rows:
            continue
        columns = _arrow_columns(files[0])
        if columns == required_columns:
            matches.append(
                ArrowGroup(
                    files=files,
                    source_rows=source_rows,
                    columns=columns,
                )
            )

    if len(matches) != 1:
        raise CacheRecoveryError(
            f"预期找到 1 组 source_rows={source_rows}、columns={sorted(required_columns)} "
            f"的完整 Arrow 分片，实际找到 {len(matches)} 组"
        )
    return matches[0]


def _inspect_packed_shard(path_text: str, cutoff_len: int) -> tuple[str, int]:
    path = Path(path_text)
    expected_length = cutoff_len
    row_count = 0
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        if set(reader.schema.names) != PACKED_COLUMNS:
            raise CacheRecoveryError(f"Packed Arrow Schema 不合法：{path}")
        input_index = reader.schema.get_field_index("input_ids")
        label_index = reader.schema.get_field_index("labels")
        for batch in reader:
            row_count += batch.num_rows
            for column_index in (input_index, label_index):
                offsets = np.asarray(batch.column(column_index).offsets)
                lengths = np.diff(offsets)
                if lengths.size and (
                    int(lengths.min()) != expected_length
                    or int(lengths.max()) != expected_length
                ):
                    raise CacheRecoveryError(
                        f"Packed 序列长度不是 {expected_length}：{path}"
                    )
    if row_count == 0:
        raise CacheRecoveryError(f"Packed Arrow 为空：{path}")
    return str(path), row_count


def inspect_packed_shards(
    files: Sequence[Path],
    *,
    cutoff_len: int,
    workers: int,
) -> dict[Path, int]:
    """Validate packed shards in parallel and return exact row counts."""

    row_counts: dict[Path, int] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(files))) as executor:
        futures = {
            executor.submit(_inspect_packed_shard, str(path), cutoff_len): path
            for path in files
        }
        for completed, future in enumerate(as_completed(futures), 1):
            path_text, rows = future.result()
            row_counts[Path(path_text)] = rows
            print(
                f"Packed 分片核验：{completed}/{len(files)}，"
                f"{Path(path_text).name}，{rows:,} 行",
                flush=True,
            )
    return row_counts


def _tokenize_valid_shard(
    input_text: str,
    output_text: str,
    model_text: str,
    cutoff_len: int,
    batch_size: int,
) -> tuple[str, int]:
    from datasets import Dataset
    from llamafactory.data.loader import _get_dataset_processor
    from llamafactory.hparams import DataArguments

    input_path = Path(input_text)
    output_path = Path(output_text)
    if output_path.is_file():
        return _inspect_packed_shard(str(output_path), cutoff_len)

    tokenizer, template, tokenizer_module = _load_lf_tokenizer_and_template(
        Path(model_text)
    )
    data_args = DataArguments(
        template="qwen3_nothink",
        cutoff_len=cutoff_len,
        train_on_prompt=False,
        packing=True,
        preprocessing_batch_size=batch_size,
        preprocessing_num_workers=1,
    )
    processor = _get_dataset_processor(
        data_args,
        "sft",
        template,
        tokenizer,
        tokenizer_module.get("processor"),
        do_generate=False,
    )
    dataset = Dataset.from_file(str(input_path))
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    if temporary_path.exists():
        temporary_path.unlink()
    tokenized = dataset.map(
        processor.preprocess_dataset,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        cache_file_name=str(temporary_path),
        desc=f"Valid {output_path.stem}",
    )
    row_count = len(tokenized)
    del tokenized
    os.replace(temporary_path, output_path)
    _, inspected_rows = _inspect_packed_shard(str(output_path), cutoff_len)
    if inspected_rows != row_count:
        raise CacheRecoveryError(
            f"Valid 分片行数不一致：{row_count} != {inspected_rows}"
        )
    return str(output_path), row_count


def tokenize_valid_shards(
    files: Sequence[Path],
    *,
    work_dir: Path,
    model_dir: Path,
    cutoff_len: int,
    batch_size: int,
    workers: int,
) -> tuple[Path, ...]:
    """Tokenize aligned validation shards with one recoverable output per input."""

    work_dir.mkdir(parents=True, exist_ok=True)
    outputs = tuple(
        work_dir / f"valid-packed-{index:05d}-of-{len(files):05d}.arrow"
        for index in range(len(files))
    )
    completed_paths: set[Path] = set()
    with ProcessPoolExecutor(max_workers=min(workers, len(files))) as executor:
        futures = {
            executor.submit(
                _tokenize_valid_shard,
                str(input_path),
                str(output_path),
                str(model_dir.resolve()),
                cutoff_len,
                batch_size,
            ): output_path
            for input_path, output_path in zip(files, outputs)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            path_text, rows = future.result()
            completed_path = Path(path_text)
            completed_paths.add(completed_path)
            print(
                f"Valid 分片完成：{completed}/{len(files)}，"
                f"{completed_path.name}，{rows:,} 行",
                flush=True,
            )
    if completed_paths != set(outputs):
        raise CacheRecoveryError("Valid 分片集合与输入不一致")
    return outputs


def _fingerprint(files: Sequence[Path], row_counts: dict[Path, int]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(row_counts[path]).encode("ascii"))
    return digest.hexdigest()[:16]


def _dataset_info(split: str, rows: int, size: int) -> dict[str, Any]:
    sequence = lambda dtype: {  # noqa: E731
        "feature": {"dtype": dtype, "_type": "Value"},
        "_type": "Sequence",
    }
    features = {
        "input_ids": sequence("int32"),
        "attention_mask": sequence("int8"),
        "position_ids": sequence("int64"),
        "labels": sequence("int64"),
        "images": {"dtype": "null", "_type": "Value"},
        "videos": {"dtype": "null", "_type": "Value"},
        "audios": {"dtype": "null", "_type": "Value"},
    }
    return {
        "builder_name": "tiger_sft_recovery",
        "citation": "",
        "config_name": "default",
        "dataset_name": "tiger_sft_tokenized",
        "dataset_size": size,
        "description": "Recovered packed TIGER SFT cache.",
        "download_checksums": {},
        "download_size": 0,
        "features": features,
        "homepage": "",
        "license": "",
        "size_in_bytes": size,
        "splits": {
            "train": {
                "name": "train",
                "num_bytes": size,
                "num_examples": rows,
                "dataset_name": f"tiger_sft_tokenized_{split}",
            }
        },
        "version": {
            "version_str": "0.0.0",
            "major": 0,
            "minor": 0,
            "patch": 0,
        },
    }


def _link_split(
    destination: Path,
    files: Sequence[Path],
    row_counts: dict[Path, int],
) -> None:
    destination.mkdir(parents=True)
    linked_names = []
    for index, source in enumerate(files):
        name = f"data-{index:05d}-of-{len(files):05d}.arrow"
        os.link(source, destination / name)
        linked_names.append({"filename": name})
    state = {
        "_data_files": linked_names,
        "_fingerprint": _fingerprint(files, row_counts),
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": "train",
    }
    rows = sum(row_counts[path] for path in files)
    size = sum(path.stat().st_size for path in files)
    (destination / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "dataset_info.json").write_text(
        json.dumps(
            _dataset_info(destination.name, rows, size),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _link_smoke_cache(source: Path, destination: Path) -> dict[str, Any]:
    manifest_path = source / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True)
    for split in ("train", "validation"):
        source_split = source / split
        destination_split = destination / split
        destination_split.mkdir()
        for path in source_split.iterdir():
            if path.is_file():
                os.link(path, destination_split / path.name)
    os.link(source / "dataset_dict.json", destination / "dataset_dict.json")
    return manifest


def recover_cache(
    *,
    interrupted_build_dir: Path,
    output_dir: Path,
    model_dir: Path,
    preflight_path: Path,
    smoke_cache_dir: Path,
    train_dataset: str,
    valid_dataset: str,
    mapping_filename: str,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise CacheRecoveryError(f"正式缓存目录已存在，拒绝覆盖：{output_dir}")
    if not interrupted_build_dir.is_dir():
        raise CacheRecoveryError(f"中断缓存目录不存在：{interrupted_build_dir}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    stats = preflight["stats"]
    cutoff_len = int(stats["effective_cutoff_len"])
    train_rows = int(stats["splits"]["train"]["rows"])
    valid_rows = int(stats["splits"]["valid"]["rows"])

    train_group = discover_arrow_group(
        interrupted_build_dir,
        source_rows=train_rows,
        required_columns=PACKED_COLUMNS,
    )
    valid_group = discover_arrow_group(
        interrupted_build_dir,
        source_rows=valid_rows,
        required_columns=ALIGNED_COLUMNS,
    )
    train_counts = inspect_packed_shards(
        train_group.files,
        cutoff_len=cutoff_len,
        workers=workers,
    )
    valid_files = tokenize_valid_shards(
        valid_group.files,
        work_dir=interrupted_build_dir / "recovery-valid",
        model_dir=model_dir,
        cutoff_len=cutoff_len,
        batch_size=batch_size,
        workers=workers,
    )
    valid_counts = inspect_packed_shards(
        valid_files,
        cutoff_len=cutoff_len,
        workers=workers,
    )

    recovering_dir = output_dir.with_name(
        f".{output_dir.name}.recovering-{os.getpid()}"
    )
    if recovering_dir.exists():
        raise CacheRecoveryError(f"恢复临时目录已存在：{recovering_dir}")
    recovering_dir.mkdir(parents=True)
    try:
        _link_split(recovering_dir / "train", train_group.files, train_counts)
        _link_split(recovering_dir / "validation", valid_files, valid_counts)
        (recovering_dir / "dataset_dict.json").write_text(
            json.dumps({"splits": ["train", "validation"]}) + "\n",
            encoding="utf-8",
        )
        smoke_manifest = _link_smoke_cache(
            smoke_cache_dir,
            recovering_dir / "smoke",
        )
        mapping = json.loads(
            (model_dir / mapping_filename).read_text(encoding="utf-8")
        )
        cache_manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "inputs": {
                "model_dir": str(model_dir.resolve()),
                "extended_tokenizer_sha256": mapping[
                    "extended_tokenizer_sha256"
                ],
                "train_sha256": stats["splits"]["train"]["sha256"],
                "valid_sha256": stats["splits"]["valid"]["sha256"],
                "cutoff_len": cutoff_len,
                "packing": True,
                "template": "qwen3_nothink",
                "train_on_prompt": False,
                "train_dataset": train_dataset,
                "valid_dataset": valid_dataset,
                "mapping_filename": mapping_filename,
            },
            "llamafactory_version": smoke_manifest["llamafactory_version"],
            "transformers_version": smoke_manifest["transformers_version"],
            "packed_rows": {
                "train": sum(train_counts.values()),
                "validation": sum(valid_counts.values()),
            },
            "smoke_source_rows": smoke_manifest["smoke_source_rows"],
            "smoke_packed_rows": smoke_manifest["smoke_packed_rows"],
            "recovered_from_interrupted_shards": True,
        }
        (recovering_dir / "length_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (recovering_dir / "cache_manifest.json").write_text(
            json.dumps(
                cache_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(recovering_dir, output_dir)
        return cache_manifest
    except BaseException:
        shutil.rmtree(recovering_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从中断任务留下的完整 Train packed 分片恢复正式 SFT Cache，"
            "并仅补算 Valid。"
        )
    )
    parser.add_argument("--interrupted-build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--preflight-path", type=Path, required=True)
    parser.add_argument("--smoke-cache-dir", type=Path, required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--valid-dataset", required=True)
    parser.add_argument(
        "--mapping-filename",
        default="tiger_token_mapping.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers <= 0 or args.batch_size <= 0:
        raise CacheRecoveryError("--workers 和 --batch-size 必须为正整数")
    manifest = recover_cache(
        interrupted_build_dir=args.interrupted_build_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        model_dir=args.model_dir.resolve(),
        preflight_path=args.preflight_path.resolve(),
        smoke_cache_dir=args.smoke_cache_dir.resolve(),
        train_dataset=args.train_dataset,
        valid_dataset=args.valid_dataset,
        mapping_filename=args.mapping_filename,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

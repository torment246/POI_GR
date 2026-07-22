#!/usr/bin/env python3
"""Sample a reproducible date-stratified evaluation set from JSON Lines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


SUPPORTED_SUFFIXES = {".json", ".jsonl"}
DATE_FIELD = "source_dt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按 source_dt 分层随机采样 JSON Lines；支持单文件、目录、"
            ".json、.jsonl 和无扩展名 part-* 分片。"
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="输入文件或目录。")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSONL 文件。")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="总采样数，默认 10000。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260721,
        help="随机种子，默认 20260721。",
    )
    return parser.parse_args()


def _is_supported_file(path: Path) -> bool:
    return path.is_file() and (
        path.suffix.lower() in SUPPORTED_SUFFIXES or path.name.startswith("part-")
    )


def discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        files = [input_path] if _is_supported_file(input_path) else []
    elif input_path.is_dir():
        files = sorted(
            (path for path in input_path.rglob("*") if _is_supported_file(path)),
            key=lambda path: path.relative_to(input_path).as_posix(),
        )
    else:
        raise FileNotFoundError(f"输入路径不存在：{input_path}")
    if not files:
        raise ValueError(f"输入路径中没有受支持的 JSON Lines 文件：{input_path}")
    return files


def iter_json_lines(files: list[Path]) -> Iterator[tuple[int, str, dict[str, Any]]]:
    record_index = 0
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw_line = line.rstrip("\r\n")
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON 解析失败：{path}:{line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"记录不是 JSON object：{path}:{line_number}")
                yield record_index, raw_line, record
                record_index += 1


def _stratum_value(record: dict[str, Any]) -> str:
    value = record.get(DATE_FIELD)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"记录缺少非空字段 {DATE_FIELD}")
    return str(value)


def _resize_reservoirs(
    reservoirs: dict[str, list[tuple[int, str]]],
    capacity: int,
    rng: random.Random,
) -> None:
    for source_dt, entries in reservoirs.items():
        if len(entries) > capacity:
            reservoirs[source_dt] = rng.sample(entries, capacity)


def stratified_sample(
    files: list[Path],
    sample_size: int,
    seed: int,
) -> tuple[list[tuple[int, str]], dict[str, int], dict[str, int], int]:
    if sample_size <= 0:
        raise ValueError("--sample-size 必须大于 0")

    sample_rng = random.Random(seed ^ 0x5DEECE66D)
    allocation_rng = random.Random(seed)
    reservoirs: dict[str, list[tuple[int, str]]] = {}
    rows_by_date: Counter[str] = Counter()
    reservoir_capacity = sample_size
    input_rows = 0

    for record_index, raw_line, record in iter_json_lines(files):
        source_dt = _stratum_value(record)
        if source_dt not in reservoirs:
            reservoirs[source_dt] = []
            reservoir_capacity = (sample_size + len(reservoirs) - 1) // len(
                reservoirs
            )
            _resize_reservoirs(reservoirs, reservoir_capacity, sample_rng)

        rows_by_date[source_dt] += 1
        input_rows += 1
        reservoir = reservoirs[source_dt]
        rows_seen = rows_by_date[source_dt]
        item = (record_index, raw_line)
        if len(reservoir) < reservoir_capacity:
            reservoir.append(item)
        else:
            replacement_index = sample_rng.randrange(rows_seen)
            if replacement_index < reservoir_capacity:
                reservoir[replacement_index] = item

        if input_rows % 1_000_000 == 0:
            print(f"已读取 {input_rows:,} 条记录", flush=True)

    dates = sorted(reservoirs)
    if not dates:
        raise ValueError("输入数据没有记录")
    if len(dates) > sample_size:
        raise ValueError("样本数小于日期分层数量，无法保证每个日期都有样本")

    base_quota, extra_count = divmod(sample_size, len(dates))
    extra_dates = set(allocation_rng.sample(dates, extra_count))
    quotas = {
        source_dt: base_quota + int(source_dt in extra_dates) for source_dt in dates
    }

    selected: list[tuple[int, str]] = []
    for source_dt in dates:
        quota = quotas[source_dt]
        available = rows_by_date[source_dt]
        if available < quota:
            raise ValueError(
                f"日期 {source_dt} 只有 {available} 条记录，少于配额 {quota}"
            )
        reservoir = reservoirs[source_dt]
        if len(reservoir) > quota:
            reservoir = sample_rng.sample(reservoir, quota)
        selected.extend(reservoir)

    selected.sort(key=lambda item: item[0])
    if len(selected) != sample_size:
        raise RuntimeError(f"内部采样错误：得到 {len(selected)} 条，期望 {sample_size} 条")
    return selected, dict(rows_by_date), quotas, input_rows


def write_output(output_path: Path, selected: list[tuple[int, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for _, raw_line in selected:
                handle.write(raw_line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _canonical_value(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output(
    output_path: Path,
    sample_size: int,
    expected_quotas: dict[str, int],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    queries: set[str] = set()
    poi_ids: set[str] = set()
    order_ids: set[str] = set()
    output_rows = 0

    for _, _, record in iter_json_lines([output_path]):
        output_rows += 1
        source_dt = _stratum_value(record)
        counts[source_dt] += 1
        if _is_empty(record.get("query")):
            raise ValueError(f"输出第 {output_rows} 条记录的 query 为空")
        if _is_empty(record.get("poi_id")):
            raise ValueError(f"输出第 {output_rows} 条记录的 poi_id 为空")
        queries.add(_canonical_value(record.get("query")))
        poi_ids.add(_canonical_value(record.get("poi_id")))
        order_ids.add(_canonical_value(record.get("order_id")))

    if output_rows != sample_size:
        raise ValueError(f"输出行数为 {output_rows}，期望 {sample_size}")
    if dict(sorted(counts.items())) != dict(sorted(expected_quotas.items())):
        raise ValueError(
            f"输出日期配额不匹配：实际 {dict(counts)}，期望 {expected_quotas}"
        )
    return {
        "output_rows": output_rows,
        "samples_by_date": dict(sorted(counts.items())),
        "unique_query": len(queries),
        "unique_poi_id": len(poi_ids),
        "unique_order_id": len(order_ids),
        "sha256": sha256_file(output_path),
    }


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("输入与输出不能是同一路径")

    files = discover_input_files(input_path)
    selected, input_rows_by_date, quotas, input_rows = stratified_sample(
        files,
        args.sample_size,
        args.seed,
    )
    write_output(output_path, selected)
    validation = validate_output(output_path, args.sample_size, quotas)
    extra_dates = [date for date, quota in quotas.items() if quota > min(quotas.values())]
    summary = {
        "input": str(input_path),
        "input_files": len(files),
        "input_rows": input_rows,
        "input_rows_by_date": dict(sorted(input_rows_by_date.items())),
        "output": str(output_path),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "extra_sample_dates": extra_dates,
        **validation,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

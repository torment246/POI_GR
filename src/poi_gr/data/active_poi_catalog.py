"""Build an active POI catalog from current targets and retained histories."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

try:
    import orjson
except ImportError:  # pragma: no cover - the production environment has orjson.
    orjson = None


SCHEMA_VERSION = "active-poi-catalog-v1"


class ActiveCatalogError(ValueError):
    """Raised when an active catalog input or invariant is invalid."""


@dataclass(frozen=True)
class ActiveCatalogConfig:
    """Input, output, and cardinality contract for one catalog build."""

    orders_dir: Path
    poi_dir: Path
    output_dir: Path
    temp_dir: Path
    order_pattern: str = "part-*.json"
    poi_pattern: str = "part-*.json"
    success_marker: str = "_SUCCESS"
    expected_order_rows: int | None = None
    expected_history_occurrences: int | None = None
    expected_current_unique: int | None = None
    expected_history_unique: int | None = None
    expected_union_unique: int | None = None
    expected_poi_rows: int | None = None
    progress_interval_rows: int = 1_000_000


@dataclass(frozen=True)
class SelectionSummary:
    current_unique: int
    history_unique: int
    intersection_unique: int
    current_only_unique: int
    history_only_unique: int
    union_unique: int
    order_rows: int
    history_occurrences: int
    empty_history_rows: int
    daily_sample_counts: dict[str, int]
    sources: tuple[dict[str, Any], ...]


def _loads(raw_line: bytes, path: Path, line_number: int) -> dict[str, Any]:
    if not raw_line.strip():
        raise ActiveCatalogError(f"{path.name}:{line_number} 是空行")
    try:
        payload = orjson.loads(raw_line) if orjson is not None else json.loads(raw_line)
    except (json.JSONDecodeError, ValueError) as error:
        raise ActiveCatalogError(
            f"{path.name}:{line_number} JSON 解析失败"
        ) from error
    if not isinstance(payload, dict):
        raise ActiveCatalogError(f"{path.name}:{line_number} 不是 JSON object")
    return payload


def _poi_id(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActiveCatalogError(f"{location} 的 poi_id 为空或不是字符串")
    return value


def _discover_files(directory: Path, pattern: str, success_marker: str) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise ActiveCatalogError(f"输入目录不存在：{directory}")
    if success_marker and not (directory / success_marker).is_file():
        raise ActiveCatalogError(f"输入目录缺少完成标记：{directory / success_marker}")
    files = tuple(sorted(directory.glob(pattern)))
    if not files:
        raise ActiveCatalogError(f"{directory} 没有匹配到 {pattern}")
    return files


def _check_expected(name: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise ActiveCatalogError(f"{name}={actual:,}，预期 {expected:,}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=16 * 1024 * 1024) as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(message: str) -> None:
    print(message, flush=True)


def collect_active_poi_ids(
    files: Iterable[Path],
    *,
    progress_interval_rows: int,
    progress: Callable[[str], None] = _progress,
) -> tuple[set[str], SelectionSummary]:
    """Collect the union of current target and retained-history POI IDs."""

    if progress_interval_rows <= 0:
        raise ActiveCatalogError("progress_interval_rows 必须为正整数")

    current_ids: set[str] = set()
    history_ids: set[str] = set()
    daily_counts: Counter[str] = Counter()
    sources: list[dict[str, Any]] = []
    total_rows = 0
    history_occurrences = 0
    empty_history_rows = 0

    for path in files:
        digest = hashlib.sha256()
        file_rows = 0
        file_bytes = 0
        file_history_occurrences = 0
        with path.open("rb", buffering=16 * 1024 * 1024) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                record = _loads(raw_line, path, line_number)
                current_ids.add(
                    _poi_id(record.get("poi_id"), f"{path.name}:{line_number}")
                )

                history = record.get("history_sequence")
                if not isinstance(history, list):
                    raise ActiveCatalogError(
                        f"{path.name}:{line_number} 的 history_sequence 不是列表"
                    )
                history_length = record.get("history_length")
                if (
                    isinstance(history_length, bool)
                    or not isinstance(history_length, int)
                    or history_length != len(history)
                ):
                    raise ActiveCatalogError(
                        f"{path.name}:{line_number} 的 history_length 与序列长度不一致"
                    )
                if not history:
                    empty_history_rows += 1
                for event_index, event in enumerate(history):
                    if not isinstance(event, dict):
                        raise ActiveCatalogError(
                            f"{path.name}:{line_number} history[{event_index}] 不是 object"
                        )
                    history_ids.add(
                        _poi_id(
                            event.get("poi_id"),
                            f"{path.name}:{line_number} history[{event_index}]",
                        )
                    )

                source_dt = record.get("source_dt")
                if not isinstance(source_dt, str) or len(source_dt) != 8:
                    raise ActiveCatalogError(
                        f"{path.name}:{line_number} 的 source_dt 不是 YYYYMMDD"
                    )
                daily_counts[source_dt] += 1

                digest.update(raw_line)
                row_bytes = len(raw_line)
                file_bytes += row_bytes
                file_rows += 1
                total_rows += 1
                file_history_occurrences += len(history)
                history_occurrences += len(history)
                if total_rows % progress_interval_rows == 0:
                    progress(
                        "行为扫描："
                        f"{total_rows:,} 行，目标 {len(current_ids):,}，"
                        f"历史 {len(history_ids):,}"
                    )

        sources.append(
            {
                "name": path.name,
                "rows_scanned": file_rows,
                "history_occurrences": file_history_occurrences,
                "bytes_scanned": file_bytes,
                "sha256": digest.hexdigest(),
            }
        )
        progress(
            f"行为分片完成：{path.name}，累计 {total_rows:,} 行，"
            f"历史事件 {history_occurrences:,}"
        )

    intersection = sum(poi_id in history_ids for poi_id in current_ids)
    current_unique = len(current_ids)
    history_unique = len(history_ids)
    current_ids.update(history_ids)
    union_unique = len(current_ids)
    summary = SelectionSummary(
        current_unique=current_unique,
        history_unique=history_unique,
        intersection_unique=intersection,
        current_only_unique=current_unique - intersection,
        history_only_unique=history_unique - intersection,
        union_unique=union_unique,
        order_rows=total_rows,
        history_occurrences=history_occurrences,
        empty_history_rows=empty_history_rows,
        daily_sample_counts=dict(sorted(daily_counts.items())),
        sources=tuple(sources),
    )
    return current_ids, summary


def _iter_catalog_rows(
    files: Iterable[Path],
) -> Iterator[tuple[Path, int, bytes, str]]:
    for path in files:
        with path.open("rb", buffering=16 * 1024 * 1024) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                record = _loads(raw_line, path, line_number)
                yield (
                    path,
                    line_number,
                    raw_line,
                    _poi_id(record.get("poi_id"), f"{path.name}:{line_number}"),
                )


def filter_poi_catalog(
    files: Iterable[Path],
    active_ids: set[str],
    output_dir: Path,
    *,
    progress_interval_rows: int,
    progress: Callable[[str], None] = _progress,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy selected catalog records verbatim while preserving source row order."""

    matched_ids: set[str] = set()
    output_sources: list[dict[str, Any]] = []
    poi_ids_path = output_dir / "poi_ids.jsonl"
    total_input_rows = 0
    total_output_rows = 0
    current_source: Path | None = None
    source_handle = None
    source_input_digest = hashlib.sha256()
    source_output_digest = hashlib.sha256()
    source_input_rows = 0
    source_output_rows = 0
    source_input_bytes = 0
    source_output_bytes = 0

    def close_source() -> None:
        nonlocal source_handle
        if current_source is None or source_handle is None:
            return
        source_handle.flush()
        os.fsync(source_handle.fileno())
        source_handle.close()
        output_sources.append(
            {
                "input_name": current_source.name,
                "output_name": current_source.name,
                "input_rows": source_input_rows,
                "output_rows": source_output_rows,
                "input_bytes": source_input_bytes,
                "output_bytes": source_output_bytes,
                "input_sha256": source_input_digest.hexdigest(),
                "output_sha256": source_output_digest.hexdigest(),
            }
        )

    with poi_ids_path.open("w", encoding="utf-8", buffering=16 * 1024 * 1024) as ids_handle:
        try:
            for path, line_number, raw_line, poi_id in _iter_catalog_rows(files):
                if current_source != path:
                    close_source()
                    current_source = path
                    source_handle = (output_dir / path.name).open(
                        "wb", buffering=16 * 1024 * 1024
                    )
                    source_input_digest = hashlib.sha256()
                    source_output_digest = hashlib.sha256()
                    source_input_rows = 0
                    source_output_rows = 0
                    source_input_bytes = 0
                    source_output_bytes = 0

                source_input_digest.update(raw_line)
                source_input_rows += 1
                source_input_bytes += len(raw_line)
                total_input_rows += 1

                if poi_id in active_ids:
                    if poi_id in matched_ids:
                        raise ActiveCatalogError(
                            f"全量 POI 主表出现重复选中 ID：{poi_id} "
                            f"({path.name}:{line_number})"
                        )
                    assert source_handle is not None
                    source_handle.write(raw_line)
                    source_output_digest.update(raw_line)
                    source_output_rows += 1
                    source_output_bytes += len(raw_line)
                    total_output_rows += 1
                    matched_ids.add(poi_id)
                    ids_handle.write(json.dumps(poi_id, ensure_ascii=False) + "\n")

                if total_input_rows % progress_interval_rows == 0:
                    progress(
                        f"POI 主表扫描：{total_input_rows:,} 行，"
                        f"已写 {total_output_rows:,}/{len(active_ids):,}"
                    )
        finally:
            close_source()

        ids_handle.flush()
        os.fsync(ids_handle.fileno())

    missing = active_ids - matched_ids
    if missing:
        examples = sorted(missing)[:10]
        raise ActiveCatalogError(
            f"全量 POI 主表缺少 {len(missing):,} 个活跃 ID，示例：{examples}"
        )
    if total_output_rows != len(active_ids):
        raise ActiveCatalogError(
            f"输出 {total_output_rows:,} 行，活跃集合 {len(active_ids):,} 个"
        )

    return output_sources, {
        "input_rows": total_input_rows,
        "output_rows": total_output_rows,
        "poi_ids_sha256": _sha256_file(poi_ids_path),
        "poi_ids_rows": total_output_rows,
    }


def _validate_config(config: ActiveCatalogConfig) -> None:
    if config.output_dir == config.orders_dir or config.output_dir == config.poi_dir:
        raise ActiveCatalogError("输出目录不能覆盖输入目录")
    if config.output_dir.exists():
        raise ActiveCatalogError(f"输出目录已存在，拒绝覆盖：{config.output_dir}")
    if config.temp_dir.exists():
        raise ActiveCatalogError(f"临时目录已存在，拒绝复用：{config.temp_dir}")
    if config.progress_interval_rows <= 0:
        raise ActiveCatalogError("progress_interval_rows 必须为正整数")
    for name, value in asdict(config).items():
        if name.startswith("expected_") and value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ActiveCatalogError(f"{name} 必须是正整数或 null")


def build_active_poi_catalog(
    config: ActiveCatalogConfig,
    *,
    progress: Callable[[str], None] = _progress,
) -> dict[str, Any]:
    """Build and atomically publish an active POI JSONL catalog."""

    _validate_config(config)
    order_files = _discover_files(
        config.orders_dir, config.order_pattern, config.success_marker
    )
    poi_files = _discover_files(config.poi_dir, config.poi_pattern, config.success_marker)
    config.temp_dir.parent.mkdir(parents=True, exist_ok=True)
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir()
    started_at = _utc_now()
    started = time.perf_counter()

    try:
        active_ids, selection = collect_active_poi_ids(
            order_files,
            progress_interval_rows=config.progress_interval_rows,
            progress=progress,
        )
        _check_expected("order_rows", selection.order_rows, config.expected_order_rows)
        _check_expected(
            "history_occurrences",
            selection.history_occurrences,
            config.expected_history_occurrences,
        )
        _check_expected(
            "current_unique", selection.current_unique, config.expected_current_unique
        )
        _check_expected(
            "history_unique", selection.history_unique, config.expected_history_unique
        )
        _check_expected("union_unique", selection.union_unique, config.expected_union_unique)

        output_sources, catalog = filter_poi_catalog(
            poi_files,
            active_ids,
            config.temp_dir,
            progress_interval_rows=config.progress_interval_rows,
            progress=progress,
        )
        _check_expected("poi_rows", catalog["input_rows"], config.expected_poi_rows)
        _check_expected("output_rows", catalog["output_rows"], selection.union_unique)

        selection_payload = asdict(selection)
        selection_payload["sources"] = list(selection.sources)
        stats = {
            **{key: value for key, value in selection_payload.items() if key != "sources"},
            "full_catalog_rows": catalog["input_rows"],
            "retained_ratio": catalog["output_rows"] / catalog["input_rows"],
            "removed_rows": catalog["input_rows"] - catalog["output_rows"],
        }
        _write_json(config.temp_dir / "stats.json", stats)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "selection": {
                "definition": "union(current target POI, retained history POI)",
                "current_scope": "all train/valid/test rows in the 14-day order input",
                "history_scope": "history_sequence exactly as retained in the input",
                "uses_query_or_frequency_for_selection": False,
                "statistics": {
                    key: value
                    for key, value in selection_payload.items()
                    if key not in {"sources", "daily_sample_counts"}
                },
                "daily_sample_counts": selection.daily_sample_counts,
            },
            "input": {
                "orders_dir": str(config.orders_dir),
                "order_pattern": config.order_pattern,
                "order_sources": list(selection.sources),
                "poi_dir": str(config.poi_dir),
                "poi_pattern": config.poi_pattern,
                "poi_sources": output_sources,
            },
            "output": {
                "dir": str(config.output_dir),
                "file_pattern": config.poi_pattern,
                "rows": catalog["output_rows"],
                "poi_ids": "poi_ids.jsonl",
                "poi_ids_rows": catalog["poi_ids_rows"],
                "poi_ids_sha256": catalog["poi_ids_sha256"],
                "stats": "stats.json",
                "success_marker": config.success_marker,
                "records_preserved_verbatim": True,
                "row_order": "sorted source shard name, then original source row order",
            },
            "validation": {
                "all_active_ids_found_in_full_catalog": True,
                "output_poi_ids_unique": True,
                "output_rows_equal_union_unique": True,
                "input_poi_rows": catalog["input_rows"],
            },
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "python": platform.python_version(),
                "orjson": orjson is not None,
            },
        }
        _write_json(config.temp_dir / "manifest.json", manifest)
        (config.temp_dir / config.success_marker).touch()
        os.replace(config.temp_dir, config.output_dir)
        return manifest
    except BaseException:
        if config.temp_dir.exists():
            shutil.rmtree(config.temp_dir)
        raise

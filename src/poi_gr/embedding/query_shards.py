"""Build deterministic Query-to-POI hash shards from the canonical SFT train set."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import orjson
except ImportError:  # pragma: no cover - the supported project environment has orjson
    orjson = None


SCHEMA_VERSION = "train-query-hash-shards-v1"
HASH_ALGORITHM = "blake2b-64"
HASH_PERSONALIZATION = b"poi-query-v1"
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("query", pa.string(), nullable=False),
        pa.field("target_poi_id", pa.string(), nullable=False),
    ]
)


class QueryShardError(RuntimeError):
    """Raised when the source contract or generated shards are invalid."""


@dataclass(frozen=True)
class QueryShardBuildResult:
    output_dir: Path
    manifest_path: Path
    total_rows: int
    num_shards: int
    reused: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise QueryShardError(f"{name} 不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise QueryShardError(f"{name} JSON 非法：{path}") from error
    if not isinstance(payload, dict):
        raise QueryShardError(f"{name} 必须是 JSON object：{path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _parse_json_line(raw_line: bytes, source: str) -> Mapping[str, Any]:
    try:
        payload = orjson.loads(raw_line) if orjson is not None else json.loads(raw_line)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise QueryShardError(f"{source} JSON 非法") from error
    if not isinstance(payload, Mapping):
        raise QueryShardError(f"{source} 必须是 JSON object")
    return payload


def extract_train_query_target(
    record: Mapping[str, Any],
    source: str,
) -> tuple[str, str]:
    """Extract the raw Query and target POI from one canonical train record."""

    if record.get("split") != "train":
        raise QueryShardError(f"{source} split 不是 train")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise QueryShardError(f"{source} messages 必须恰好包含 user/assistant")
    user, assistant = messages
    if not isinstance(user, Mapping) or user.get("role") != "user":
        raise QueryShardError(f"{source} 第一条 message 不是 user")
    if not isinstance(assistant, Mapping) or assistant.get("role") != "assistant":
        raise QueryShardError(f"{source} 第二条 message 不是 assistant")
    content = user.get("content")
    marker = "</QUERY>\n<USER_GID>"
    if (
        not isinstance(content, str)
        or not content.startswith("<QUERY>")
        or marker not in content
        or not content.endswith("</USER_GID>")
    ):
        raise QueryShardError(f"{source} user content 结构非法")
    query, _ = content[len("<QUERY>") :].rsplit(marker, 1)
    if not query.strip():
        raise QueryShardError(f"{source} Query 为空")
    target_poi_id = record.get("target_poi_id")
    if not isinstance(target_poi_id, str) or not target_poi_id.strip():
        raise QueryShardError(f"{source} target_poi_id 为空或不是字符串")
    return query, target_poi_id


def stable_query_shard(query: str, num_shards: int) -> int:
    """Map an exact raw Query to a deterministic shard number."""

    if not isinstance(query, str) or not query.strip():
        raise QueryShardError("Query 必须是非空字符串")
    if num_shards <= 0 or num_shards > 65_536:
        raise QueryShardError("num_shards 必须位于 [1, 65536]")
    digest = hashlib.blake2b(
        query.encode("utf-8"),
        digest_size=8,
        person=HASH_PERSONALIZATION,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % num_shards


def _source_contract(sft_dir: Path) -> tuple[Path, Path, int, str, str]:
    manifest_path = sft_dir / "manifest.json"
    train_path = sft_dir / "train.jsonl"
    manifest = _load_json_object(manifest_path, "SFT manifest")
    if manifest.get("schema_version") != "sft-main-data-v1":
        raise QueryShardError("SFT manifest schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise QueryShardError("SFT manifest 状态不是 completed")
    output = manifest.get("outputs", {}).get("train.jsonl")
    if not isinstance(output, Mapping):
        raise QueryShardError("SFT manifest 缺少 outputs.train.jsonl")
    try:
        expected_rows = int(output["rows"])
    except (KeyError, TypeError, ValueError) as error:
        raise QueryShardError("SFT manifest 的 Train 行数非法") from error
    expected_sha256 = output.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise QueryShardError("SFT manifest 的 Train SHA256 非法")
    if not train_path.is_file():
        raise QueryShardError(f"Train JSONL 不存在：{train_path}")
    return (
        manifest_path,
        train_path,
        expected_rows,
        expected_sha256,
        _sha256_file(manifest_path),
    )


def _shard_name(shard_id: int, num_shards: int) -> str:
    width = max(5, len(str(num_shards)))
    return f"part-{shard_id:0{width}d}-of-{num_shards:0{width}d}.parquet"


def _new_writer(path: Path) -> pq.ParquetWriter:
    return pq.ParquetWriter(
        path,
        PARQUET_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def _flush_shard(
    shard_id: int,
    buffers: list[tuple[list[str], list[str]]],
    writers: list[pq.ParquetWriter | None],
    staging_dir: Path,
    num_shards: int,
) -> None:
    queries, poi_ids = buffers[shard_id]
    if not queries:
        return
    writer = writers[shard_id]
    if writer is None:
        writer = _new_writer(staging_dir / _shard_name(shard_id, num_shards))
        writers[shard_id] = writer
    table = pa.Table.from_arrays(
        [pa.array(queries, type=pa.string()), pa.array(poi_ids, type=pa.string())],
        schema=PARQUET_SCHEMA,
    )
    writer.write_table(table, row_group_size=len(queries))
    queries.clear()
    poi_ids.clear()


def _flush_all_buffers(
    buffers: list[tuple[list[str], list[str]]],
    writers: list[pq.ParquetWriter | None],
    staging_dir: Path,
    num_shards: int,
) -> None:
    for shard_id in range(num_shards):
        _flush_shard(shard_id, buffers, writers, staging_dir, num_shards)


def _close_writers(
    writers: list[pq.ParquetWriter | None],
    staging_dir: Path,
    num_shards: int,
) -> None:
    for shard_id, writer in enumerate(writers):
        if writer is None:
            writer = _new_writer(staging_dir / _shard_name(shard_id, num_shards))
        writer.close()


def _scan_train(
    train_path: Path,
    staging_dir: Path,
    *,
    num_shards: int,
    buffer_rows_per_shard: int,
    show_progress: bool,
    expected_rows: int,
) -> tuple[int, int, str, list[int]]:
    if buffer_rows_per_shard <= 0:
        raise QueryShardError("buffer_rows_per_shard 必须大于 0")
    writers: list[pq.ParquetWriter | None] = [None] * num_shards
    buffers = [([], []) for _ in range(num_shards)]
    shard_rows = [0] * num_shards
    source_digest = hashlib.sha256()
    source_bytes = 0
    total_rows = 0
    progress = tqdm(
        total=expected_rows,
        desc="Train Query hash sharding",
        unit="row",
        disable=not show_progress,
    )
    try:
        with train_path.open("rb", buffering=8 * 1024 * 1024) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                source_digest.update(raw_line)
                source_bytes += len(raw_line)
                if not raw_line.strip():
                    raise QueryShardError(f"{train_path}:{line_number} 是空行")
                source = f"{train_path}:{line_number}"
                record = _parse_json_line(raw_line, source)
                query, target_poi_id = extract_train_query_target(record, source)
                shard_id = stable_query_shard(query, num_shards)
                buffers[shard_id][0].append(query)
                buffers[shard_id][1].append(target_poi_id)
                shard_rows[shard_id] += 1
                total_rows += 1
                if len(buffers[shard_id][0]) >= buffer_rows_per_shard:
                    _flush_shard(
                        shard_id,
                        buffers,
                        writers,
                        staging_dir,
                        num_shards,
                    )
                if total_rows % 100_000 == 0:
                    progress.update(100_000)
        remainder = total_rows % 100_000
        _flush_all_buffers(buffers, writers, staging_dir, num_shards)
        if remainder:
            progress.update(remainder)
    finally:
        progress.close()
        _close_writers(writers, staging_dir, num_shards)
    return total_rows, source_bytes, source_digest.hexdigest(), shard_rows


def _audit_shards(
    directory: Path,
    num_shards: int,
    expected_rows: list[int] | None = None,
    expected_files: list[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    files: list[dict[str, Any]] = []
    total_rows = 0
    for shard_id in range(num_shards):
        name = _shard_name(shard_id, num_shards)
        path = directory / name
        if not path.is_file():
            raise QueryShardError(f"缺少 Query 分片：{path}")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != PARQUET_SCHEMA:
            raise QueryShardError(f"Query 分片 Schema 不一致：{path}")
        rows = parquet.metadata.num_rows
        if expected_rows is not None and rows != expected_rows[shard_id]:
            raise QueryShardError(
                f"Query 分片 {shard_id} 行数 {rows} != 期望 {expected_rows[shard_id]}"
            )
        digest = _sha256_file(path)
        size_bytes = path.stat().st_size
        if expected_files is not None:
            expected = expected_files[shard_id]
            if expected.get("file") != name:
                raise QueryShardError(f"Manifest 的 Query 分片顺序错误：{shard_id}")
            if int(expected.get("rows", -1)) != rows:
                raise QueryShardError(f"Manifest 的 Query 分片行数错误：{shard_id}")
            if expected.get("sha256") != digest:
                raise QueryShardError(f"Query 分片 SHA256 错误：{shard_id}")
            if int(expected.get("size_bytes", -1)) != size_bytes:
                raise QueryShardError(f"Query 分片文件大小错误：{shard_id}")
        files.append(
            {
                "shard_id": shard_id,
                "file": name,
                "rows": rows,
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )
        total_rows += rows
    return files, total_rows


def validate_query_shards(output_dir: Path) -> QueryShardBuildResult:
    """Validate every shard against a completed artifact manifest."""

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = _load_json_object(manifest_path, "Query 分片 manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise QueryShardError("Query 分片 manifest schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise QueryShardError("Query 分片 manifest 状态不是 completed")
    if not (output_dir / "_SUCCESS").is_file():
        raise QueryShardError("Query 分片缺少 _SUCCESS")
    sharding = manifest.get("sharding")
    if not isinstance(sharding, Mapping):
        raise QueryShardError("Query 分片 manifest 缺少 sharding")
    num_shards = int(sharding.get("num_shards", 0))
    expected_files = sharding.get("files")
    if not isinstance(expected_files, list) or len(expected_files) != num_shards:
        raise QueryShardError("Query 分片 manifest 的文件清单非法")
    _, total_rows = _audit_shards(
        output_dir,
        num_shards,
        expected_files=expected_files,
    )
    expected_total = int(sharding.get("total_rows", -1))
    if total_rows != expected_total:
        raise QueryShardError(
            f"Query 分片总行数 {total_rows} != manifest {expected_total}"
        )
    return QueryShardBuildResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        total_rows=total_rows,
        num_shards=num_shards,
        reused=True,
    )


def build_train_query_shards(
    sft_dir: Path,
    output_dir: Path,
    *,
    num_shards: int = 256,
    buffer_rows_per_shard: int = 4_096,
    show_progress: bool = True,
) -> QueryShardBuildResult:
    """Stream the canonical Train JSONL into deterministic Parquet shards."""

    sft_dir = sft_dir.resolve()
    output_dir = output_dir.resolve()
    if num_shards <= 0 or num_shards > 65_536:
        raise QueryShardError("num_shards 必须位于 [1, 65536]")
    manifest_path, train_path, expected_rows, expected_sha256, sft_manifest_sha256 = (
        _source_contract(sft_dir)
    )
    if output_dir.exists():
        result = validate_query_shards(output_dir)
        existing = _load_json_object(
            result.manifest_path,
            "Query 分片 manifest",
        )
        source = existing.get("source")
        if result.num_shards != num_shards:
            raise QueryShardError(
                f"已有产物分片数 {result.num_shards} != 请求 {num_shards}"
            )
        if not isinstance(source, Mapping):
            raise QueryShardError("已有产物 manifest 缺少 source")
        if source.get("sft_manifest_sha256") != sft_manifest_sha256:
            raise QueryShardError("已有产物与当前 SFT manifest 指纹不一致")
        if source.get("train_sha256") != expected_sha256:
            raise QueryShardError("已有产物与当前 Train 指纹不一致")
        return result
    staging_dir = output_dir.with_name(f".{output_dir.name}.building")
    if staging_dir.exists():
        raise QueryShardError(
            f"存在未完成的 Query 分片目录，请先审计后处理：{staging_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()

    total_rows, source_bytes, source_sha256, shard_rows = _scan_train(
        train_path,
        staging_dir,
        num_shards=num_shards,
        buffer_rows_per_shard=buffer_rows_per_shard,
        show_progress=show_progress,
        expected_rows=expected_rows,
    )
    if total_rows != expected_rows:
        raise QueryShardError(
            f"Train 实际行数 {total_rows:,} != manifest {expected_rows:,}"
        )
    if source_sha256 != expected_sha256:
        raise QueryShardError(
            f"Train SHA256 {source_sha256} != manifest {expected_sha256}"
        )
    files, audited_rows = _audit_shards(
        staging_dir,
        num_shards,
        expected_rows=shard_rows,
    )
    if audited_rows != total_rows:
        raise QueryShardError(
            f"Query 分片审计总行数 {audited_rows:,} != Train {total_rows:,}"
        )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "built_at": _utc_now(),
        "source": {
            "sft_dir": str(sft_dir),
            "sft_manifest": str(manifest_path),
            "sft_manifest_sha256": sft_manifest_sha256,
            "train_file": str(train_path),
            "train_rows": total_rows,
            "train_size_bytes": source_bytes,
            "train_sha256": source_sha256,
        },
        "contract": {
            "query_source": "messages[0].content/<QUERY>",
            "query_normalization": "none",
            "target_source": "target_poi_id",
            "output_fields": ["query", "target_poi_id"],
            "split": "train",
        },
        "hash": {
            "algorithm": HASH_ALGORITHM,
            "digest_size_bytes": 8,
            "personalization_utf8": HASH_PERSONALIZATION.decode("ascii"),
            "input": "exact raw Query UTF-8 bytes",
            "mapping": "int.from_bytes(digest, 'big') % num_shards",
        },
        "sharding": {
            "num_shards": num_shards,
            "format": "parquet",
            "compression": "zstd",
            "buffer_rows_per_shard": buffer_rows_per_shard,
            "row_order": "source order preserved within each shard",
            "total_rows": total_rows,
            "min_shard_rows": min(shard_rows),
            "max_shard_rows": max(shard_rows),
            "mean_shard_rows": total_rows / num_shards,
            "files": files,
        },
        "validation": {
            "source_rows_match_manifest": True,
            "source_sha256_matches_manifest": True,
            "all_records_are_train": True,
            "all_queries_non_empty": True,
            "all_target_poi_ids_non_empty": True,
            "shard_row_conservation": True,
            "all_shards_schema_match": True,
            "all_shards_sha256_recorded": True,
        },
    }
    _write_json_atomic(staging_dir / "manifest.json", manifest)
    (staging_dir / "_SUCCESS").touch()
    os.replace(staging_dir, output_dir)
    return QueryShardBuildResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        total_rows=total_rows,
        num_shards=num_shards,
        reused=False,
    )

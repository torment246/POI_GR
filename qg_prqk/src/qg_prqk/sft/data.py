"""Build paired SFT messages from a frozen history corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, utc_now
from qg_prqk.sid.identifiers import (
    FINAL_IDENTIFIER_SCHEMA_VERSION,
    Variant,
    identifier_content,
    identifier_key,
)
from qg_prqk.sid.geo import GEOHASH_ALPHABET


SFT_DATA_SCHEMA_VERSION = "qg-prqk-paired-sft-data-v1"
SPECIAL_TOKENS_SCHEMA_VERSION = "qg-prqk-sft-special-tokens-v1"
SPLITS = ("train", "valid", "test")
VARIANTS: tuple[Variant, Variant] = ("a4_gid_parent", "a4_nogid")
TIGER_CONTENT_PATTERN = re.compile(
    r"<S1_(\d+)><S2_(\d+)><S3_(\d+)><C_(\d+)>"
)
HISTORY_TIGER_PATTERN = re.compile(
    r"<POI_TIGER_ID>"
    r"(<S1_\d+><S2_\d+><S3_\d+><C_\d+>)"
    r"</POI_TIGER_ID>"
)
TARGET_TIGER_PATTERN = re.compile(
    r"<TARGET_POI>"
    r"(<S1_\d+><S2_\d+><S3_\d+><C_\d+>)"
    r"</TARGET_POI>"
)
STRUCTURE_TOKENS = (
    "<USER_ID>",
    "</USER_ID>",
    "<HISTORY>",
    "</HISTORY>",
    "<EVENT>",
    "</EVENT>",
    "<USER_GID>",
    "</USER_GID>",
    "<QUERY>",
    "</QUERY>",
    "<POI_QGPRQK_ID>",
    "</POI_QGPRQK_ID>",
    "<CURRENT>",
    "</CURRENT>",
    "<TARGET_POI>",
    "</TARGET_POI>",
)


class SftDataError(ValueError):
    """Raised when paired QG-PRQK SFT data violates its contract."""


@dataclass
class FinalIdentifierLookup:
    """Memory-backed Final-ID lookup with lazy token serialization."""

    variant: Variant
    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    poi_ids: list[str]
    row_by_poi_id: dict[str, int]
    base_codes: np.ndarray
    dedup_codes: np.ndarray
    requires_dedup: np.ndarray
    content_by_row: list[str | None]

    def content(self, row: int) -> str:
        cached = self.content_by_row[row]
        if cached is None:
            cached = identifier_content(
                self.base_codes[row],
                int(self.dedup_codes[row]),
                variant=self.variant,
            )
            self.content_by_row[row] = cached
        return cached

    def key(self, row: int) -> str:
        return identifier_key(
            self.base_codes[row],
            int(self.dedup_codes[row]),
            variant=self.variant,
        )


@dataclass(frozen=True)
class PairedSftDataResult:
    """Metadata for two completed, row-aligned SFT datasets."""

    manifests: dict[str, dict[str, Any]]
    output_hashes: dict[str, dict[str, str]]
    stats: dict[str, Any]


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise SftDataError(f"{name}不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SftDataError(f"{name}不是合法 JSON：{path}") from error
    if not isinstance(payload, dict):
        raise SftDataError(f"{name}必须是 JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validated_artifact(directory: Path, manifest: Mapping[str, Any], name: str) -> Path:
    spec = manifest.get("artifacts", {}).get(name)
    path = directory / name
    if not isinstance(spec, dict) or not path.is_file():
        raise SftDataError(f"Final-ID manifest 缺少产物：{name}")
    if sha256_file(path) != spec.get("sha256"):
        raise SftDataError(f"Final-ID 产物哈希不一致：{name}")
    return path


def load_final_identifier_lookup(directory: Path, variant: Variant) -> FinalIdentifierLookup:
    """Load and fully validate one frozen QG-PRQK Final-ID directory."""

    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path, "Final-ID manifest")
    if (
        manifest.get("schema_version") != FINAL_IDENTIFIER_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("variant") != variant
    ):
        raise SftDataError("Final-ID manifest 版本、状态或 variant 不一致")
    if not (directory / "_SUCCESS").is_file():
        raise SftDataError("Final-ID 目录缺少 _SUCCESS")
    mapping_path = _validated_artifact(
        directory, manifest, "poi_final_id_mapping.parquet"
    )
    base_path = _validated_artifact(directory, manifest, "base_identifier_codes.npy")
    dedup_path = _validated_artifact(directory, manifest, "dedup_codes.npy")
    requires_path = _validated_artifact(directory, manifest, "requires_dedup.npy")
    base = np.load(base_path, mmap_mode="r", allow_pickle=False)
    dedup = np.load(dedup_path, mmap_mode="r", allow_pickle=False)
    requires = np.load(requires_path, mmap_mode="r", allow_pickle=False)
    poi_count = int(manifest.get("metrics", {}).get("poi_count", 0))
    expected_width = 9 if variant == "a4_gid_parent" else 3
    if base.shape != (poi_count, expected_width) or base.dtype != np.int32:
        raise SftDataError("Final-ID base code shape 或 dtype 不符合契约")
    if dedup.shape != (poi_count,) or dedup.dtype != np.int32:
        raise SftDataError("Final-ID Dedup shape 或 dtype 不符合契约")
    if requires.shape != (poi_count,) or requires.dtype != np.bool_:
        raise SftDataError("Final-ID requires_dedup shape 或 dtype 不符合契约")
    if not np.array_equal(requires, dedup >= 0):
        raise SftDataError("requires_dedup 与 Dedup code 不一致")

    parquet_file = pq.ParquetFile(mapping_path)
    required_columns = ("poi_row_index", "poi_id", "final_id_key")
    if parquet_file.metadata.num_rows != poi_count or not set(required_columns).issubset(
        parquet_file.schema_arrow.names
    ):
        raise SftDataError("Final-ID mapping 行数或字段不符合契约")
    poi_ids: list[str] = []
    offset = 0
    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=list(required_columns))
        values = table.to_pydict()
        count = table.num_rows
        end = offset + count
        if not np.array_equal(
            np.asarray(values["poi_row_index"], dtype=np.int64),
            np.arange(offset, end, dtype=np.int64),
        ):
            raise SftDataError("Final-ID mapping 行号未保持同行同序")
        for local, poi_id in enumerate(values["poi_id"]):
            if not isinstance(poi_id, str) or not poi_id:
                raise SftDataError("Final-ID mapping 包含非法 poi_id")
            row = offset + local
            expected_key = identifier_key(
                base[row], int(dedup[row]), variant=variant
            )
            if values["final_id_key"][local] != expected_key:
                raise SftDataError("Final-ID mapping key 与 code 不一致")
            poi_ids.append(poi_id)
        offset = end
    if offset != poi_count or len(set(poi_ids)) != poi_count:
        raise SftDataError("Final-ID mapping 不完整或 poi_id 不唯一")
    return FinalIdentifierLookup(
        variant=variant,
        directory=directory,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        poi_ids=poi_ids,
        row_by_poi_id={poi_id: row for row, poi_id in enumerate(poi_ids)},
        base_codes=base,
        dedup_codes=dedup,
        requires_dedup=requires,
        content_by_row=[None] * poi_count,
    )


def _tiger_integer_key(codes: Sequence[int], capacities: Sequence[int]) -> int:
    if len(codes) != 4 or len(capacities) != 4:
        raise SftDataError("TIGER identifier 必须是四层")
    key = 0
    for value, capacity in zip(codes, capacities, strict=True):
        value = int(value)
        capacity = int(capacity)
        if not 0 <= value < capacity:
            raise SftDataError("TIGER identifier 超出声明容量")
        key = key * capacity + value
    return key


def _parse_tiger_content(content: str, capacities: Sequence[int]) -> int:
    match = TIGER_CONTENT_PATTERN.fullmatch(content)
    if match is None:
        raise SftDataError(f"非法 TIGER identifier：{content}")
    return _tiger_integer_key([int(value) for value in match.groups()], capacities)


def _load_tiger_reverse_index(
    source_manifest: Mapping[str, Any],
    tiger_identifier_dir: Path,
    qg_poi_ids: Sequence[str],
) -> tuple[dict[int, int], tuple[int, int, int, int], dict[str, Any]]:
    tiger_contract = source_manifest.get("tiger_identifier")
    if not isinstance(tiger_contract, dict):
        raise SftDataError("源 SFT manifest 缺少 TIGER identifier 契约")
    capacities = tuple(int(value) for value in tiger_contract.get("token_capacities", ()))
    if len(capacities) != 4:
        raise SftDataError("源 TIGER token capacities 必须是四层")
    tiger_identifier_dir = tiger_identifier_dir.resolve()
    manifest_path = tiger_identifier_dir / "tiger_id_manifest.json"
    mapping_path = tiger_identifier_dir / "poi_tiger_id_mapping.parquet"
    if sha256_file(manifest_path) != tiger_contract.get("manifest_sha256"):
        raise SftDataError("TIGER manifest 与源 SFT 契约不一致")
    if sha256_file(mapping_path) != tiger_contract.get("mapping_sha256"):
        raise SftDataError("TIGER mapping 与源 SFT 契约不一致")

    row_by_qg_poi = {poi_id: row for row, poi_id in enumerate(qg_poi_ids)}
    row_by_tiger_key: dict[int, int] = {}
    parquet_file = pq.ParquetFile(mapping_path)
    columns = ("poi_id", "s1", "s2", "s3", "collision_code")
    if parquet_file.metadata.num_rows != len(qg_poi_ids) or not set(columns).issubset(
        parquet_file.schema_arrow.names
    ):
        raise SftDataError("TIGER mapping 行数或字段与 QG Final-ID 不一致")
    observed_poi_ids: set[str] = set()
    for row_group in range(parquet_file.num_row_groups):
        values = parquet_file.read_row_group(
            row_group, columns=list(columns)
        ).to_pydict()
        codes = zip(
            values["s1"],
            values["s2"],
            values["s3"],
            values["collision_code"],
            strict=True,
        )
        for poi_id, code in zip(values["poi_id"], codes, strict=True):
            row = row_by_qg_poi.get(poi_id)
            if row is None or poi_id in observed_poi_ids:
                raise SftDataError("TIGER 与 QG POI 集合不一致或存在重复")
            key = _tiger_integer_key(code, capacities)
            if key in row_by_tiger_key:
                raise SftDataError("冻结 TIGER identifier 不是全局唯一")
            row_by_tiger_key[key] = row
            observed_poi_ids.add(poi_id)
    if len(observed_poi_ids) != len(qg_poi_ids):
        raise SftDataError("TIGER 与 QG POI 集合未完全覆盖")
    return row_by_tiger_key, capacities, {
        "directory": str(tiger_identifier_dir),
        "manifest_sha256": tiger_contract["manifest_sha256"],
        "mapping_sha256": tiger_contract["mapping_sha256"],
        "token_capacities": list(capacities),
        "use": "frozen_history_identifier_reverse_lookup_only",
    }


def build_shared_special_tokens(
    *,
    sid_capacity: int,
    dedup_capacity: int,
    user_bucket_count: int,
) -> dict[str, Any]:
    """Return one ordinary-token inventory shared by both SFT variants."""

    if sid_capacity <= 0 or dedup_capacity <= 0 or user_bucket_count <= 0:
        raise SftDataError("SID、Dedup 与用户 Token 容量必须为正数")
    width = max(4, len(str(user_bucket_count - 1)))
    geohash_tokens = [f"<G_{value}>" for value in GEOHASH_ALPHABET]
    user_tokens = [
        f"<U_{value:0{width}d}>" for value in range(user_bucket_count)
    ]
    sid_tokens = {
        f"s{level}": [
            f"<S{level}_{value}>" for value in range(sid_capacity)
        ]
        for level in range(1, 4)
    }
    dedup_tokens = [f"<D_{value}>" for value in range(dedup_capacity)]
    tokens = [
        *STRUCTURE_TOKENS,
        *geohash_tokens,
        *user_tokens,
        *sid_tokens["s1"],
        *sid_tokens["s2"],
        *sid_tokens["s3"],
        *dedup_tokens,
    ]
    if len(tokens) != len(set(tokens)) or "<D_-1>" in tokens:
        raise SftDataError("QG SFT Token 表重复或包含 D_-1")
    if any(token.startswith("<C_") for token in tokens):
        raise SftDataError("QG SFT Token 表不得保留 TIGER C Token")
    return {
        "schema_version": SPECIAL_TOKENS_SCHEMA_VERSION,
        "shared_by_variants": list(VARIANTS),
        "structure_tokens": list(STRUCTURE_TOKENS),
        "geohash_tokens": geohash_tokens,
        "user_bucket_count": user_bucket_count,
        "user_tokens": user_tokens,
        "sid_token_capacities": [sid_capacity] * 3,
        "sid_tokens": sid_tokens,
        "dedup_token_capacity": dedup_capacity,
        "dedup_tokens": dedup_tokens,
        "additional_special_tokens": tokens,
        "token_count": len(tokens),
    }


def _replace_history(
    content: str,
    *,
    expected_count: int,
    row_by_tiger_key: Mapping[int, int],
    tiger_capacities: Sequence[int],
    lookup: FinalIdentifierLookup,
) -> str:
    if not isinstance(content, str):
        raise SftDataError("源 User content 必须是字符串")
    observed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal observed
        observed += 1
        row = row_by_tiger_key.get(
            _parse_tiger_content(match.group(1), tiger_capacities)
        )
        if row is None:
            raise SftDataError("历史 TIGER identifier 不在冻结映射")
        return f"<POI_QGPRQK_ID>{lookup.content(row)}</POI_QGPRQK_ID>"

    transformed = HISTORY_TIGER_PATTERN.sub(replace, content)
    if observed != expected_count:
        raise SftDataError(
            f"history_length={expected_count}，实际解析到 {observed} 个历史 identifier"
        )
    if "<POI_TIGER_ID>" in transformed or "</POI_TIGER_ID>" in transformed:
        raise SftDataError("源 Prompt 仍含未解析的 TIGER history identifier")
    return transformed


def _transform_record(
    source: Mapping[str, Any],
    *,
    split: str,
    row_by_tiger_key: Mapping[int, int],
    tiger_capacities: Sequence[int],
    lookup: FinalIdentifierLookup,
) -> dict[str, Any]:
    if source.get("split") != split:
        raise SftDataError("源样本 split 与文件不一致")
    messages = source.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(messages[0], dict)
        or not isinstance(messages[1], dict)
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise SftDataError("源 messages 必须严格为 user/assistant 两轮")
    history_length = source.get("history_length")
    if isinstance(history_length, bool) or not isinstance(history_length, int):
        raise SftDataError("源 history_length 非法")
    poi_id = source.get("target_poi_id")
    row = lookup.row_by_poi_id.get(poi_id) if isinstance(poi_id, str) else None
    if row is None:
        raise SftDataError("目标 POI 不在 QG Final-ID mapping")
    target_match = TARGET_TIGER_PATTERN.fullmatch(
        str(messages[1].get("content", ""))
    )
    if target_match is None:
        raise SftDataError("源 Assistant 不是合法 TIGER target")
    target_row = row_by_tiger_key.get(
        _parse_tiger_content(target_match.group(1), tiger_capacities)
    )
    if target_row != row:
        raise SftDataError("源 Assistant target 与 target_poi_id 不一致")

    output = {
        key: value
        for key, value in source.items()
        if key not in {"messages", "target_tiger_id_key"}
    }
    output.update(
        {
            "messages": [
                {
                    "role": "user",
                    "content": _replace_history(
                        messages[0].get("content"),
                        expected_count=history_length,
                        row_by_tiger_key=row_by_tiger_key,
                        tiger_capacities=tiger_capacities,
                        lookup=lookup,
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"<TARGET_POI>{lookup.content(row)}</TARGET_POI>",
                },
            ],
            "target_qg_prqk_id_key": lookup.key(row),
            "requires_dedup": bool(lookup.requires_dedup[row]),
            "identifier_variant": lookup.variant,
        }
    )
    if lookup.variant == "a4_nogid" and "<G_" in output["messages"][1]["content"]:
        raise SftDataError("NoGID Assistant target 禁止包含 GID Token")
    return output


def _source_split_contract(
    source_dir: Path, manifest: Mapping[str, Any], split: str
) -> tuple[Path, int, str]:
    path = source_dir / f"{split}.jsonl"
    spec = manifest.get("outputs", {}).get(path.name)
    if not isinstance(spec, dict):
        raise SftDataError(f"源 manifest 缺少 {path.name}")
    rows = spec.get("rows")
    digest = spec.get("sha256")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
        or not isinstance(digest, str)
        or not path.is_file()
    ):
        raise SftDataError(f"源 {path.name} 契约非法")
    return path, rows, digest


def build_paired_sft_data(
    *,
    source_sft_dir: Path,
    tiger_identifier_dir: Path,
    final_identifier_dirs: Mapping[str, Path],
    output_dirs: Mapping[str, Path],
    max_rows_per_split: int | None = None,
    progress_every: int = 250_000,
    progress: Callable[[str], None] | None = None,
) -> PairedSftDataResult:
    """Replace only POI identifiers while preserving every source sample."""

    if set(final_identifier_dirs) != set(VARIANTS) or set(output_dirs) != set(VARIANTS):
        raise SftDataError("必须同时提供 GID-parent 与 NoGID 两个目录")
    if max_rows_per_split is not None and max_rows_per_split <= 0:
        raise SftDataError("max_rows_per_split 必须大于 0")
    if progress_every <= 0:
        raise SftDataError("progress_every 必须大于 0")
    source_sft_dir = source_sft_dir.resolve()
    source_manifest_path = source_sft_dir / "manifest.json"
    source_manifest = _load_json(source_manifest_path, "源 SFT manifest")
    if (
        source_manifest.get("schema_version") != "tiger-map-search-sft-data-v1"
        or source_manifest.get("status") != "completed"
    ):
        raise SftDataError("源数据必须是已完成的冻结 TIGER history10 SFT v1")

    lookups = {
        variant: load_final_identifier_lookup(final_identifier_dirs[variant], variant)
        for variant in VARIANTS
    }
    if lookups["a4_gid_parent"].poi_ids != lookups["a4_nogid"].poi_ids:
        raise SftDataError("两套 QG Final-ID 的 POI 集合或行序不一致")
    row_by_tiger_key, tiger_capacities, tiger_source = _load_tiger_reverse_index(
        source_manifest,
        tiger_identifier_dir,
        lookups["a4_gid_parent"].poi_ids,
    )
    dedup_capacity = max(
        int(lookup.manifest["metrics"]["dedup_token_capacity"])
        for lookup in lookups.values()
    )
    special_tokens = build_shared_special_tokens(
        sid_capacity=512,
        dedup_capacity=dedup_capacity,
        user_bucket_count=2000,
    )

    resolved_outputs = {
        variant: Path(output_dirs[variant]).resolve() for variant in VARIANTS
    }
    if len(set(resolved_outputs.values())) != 2:
        raise SftDataError("两套 SFT 输出目录不能相同")
    for output in resolved_outputs.values():
        if output.exists():
            raise SftDataError(f"输出目录已存在，拒绝覆盖：{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    temporary_dirs = {
        variant: Path(
            tempfile.mkdtemp(
                dir=resolved_outputs[variant].parent,
                prefix=f".{resolved_outputs[variant].name}.building-",
            )
        )
        for variant in VARIANTS
    }

    stats: dict[str, Any] = {
        "source_rows": 0,
        "train_count": 0,
        "valid_count": 0,
        "test_count": 0,
        "history_event_occurrence_count": 0,
        "dedup_target_count": {variant: 0 for variant in VARIANTS},
    }
    source_files: dict[str, Any] = {}
    output_digests = {
        variant: {split: hashlib.sha256() for split in SPLITS}
        for variant in VARIANTS
    }
    try:
        handles = {
            variant: {
                split: (temporary_dirs[variant] / f"{split}.jsonl").open(
                    "wb", buffering=8 * 1024 * 1024
                )
                for split in SPLITS
            }
            for variant in VARIANTS
        }
        try:
            for split in SPLITS:
                source_path, expected_rows, expected_hash = _source_split_contract(
                    source_sft_dir, source_manifest, split
                )
                source_digest = hashlib.sha256()
                rows = 0
                with source_path.open("rb", buffering=8 * 1024 * 1024) as source:
                    for line_number, raw_line in enumerate(source, start=1):
                        if max_rows_per_split is not None and rows >= max_rows_per_split:
                            break
                        source_digest.update(raw_line)
                        rows += 1
                        try:
                            record = json.loads(raw_line)
                        except (UnicodeDecodeError, json.JSONDecodeError) as error:
                            raise SftDataError(
                                f"{source_path.name}:{line_number} JSON 非法"
                            ) from error
                        if not isinstance(record, dict):
                            raise SftDataError("源 SFT 行必须是 JSON object")
                        transformed = {
                            variant: _transform_record(
                                record,
                                split=split,
                                row_by_tiger_key=row_by_tiger_key,
                                tiger_capacities=tiger_capacities,
                                lookup=lookups[variant],
                            )
                            for variant in VARIANTS
                        }
                        invariant_fields = (
                            "sample_id",
                            "order_id",
                            "searchid",
                            "target_poi_id",
                            "history_length",
                            "split",
                        )
                        if any(
                            transformed[VARIANTS[0]].get(field)
                            != transformed[VARIANTS[1]].get(field)
                            for field in invariant_fields
                        ):
                            raise SftDataError("两套 SFT 样本的非标识字段不一致")
                        for variant in VARIANTS:
                            encoded = (
                                json.dumps(
                                    transformed[variant],
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                )
                                + "\n"
                            ).encode("utf-8")
                            handles[variant][split].write(encoded)
                            output_digests[variant][split].update(encoded)
                            if transformed[variant]["requires_dedup"]:
                                stats["dedup_target_count"][variant] += 1
                        stats["source_rows"] += 1
                        stats[f"{split}_count"] += 1
                        stats["history_event_occurrence_count"] += int(
                            record["history_length"]
                        )
                        if progress is not None and rows % progress_every == 0:
                            upper = min(expected_rows, max_rows_per_split or expected_rows)
                            progress(f"{split} 成对转换：{rows:,}/{upper:,}")
                observed_hash = source_digest.hexdigest()
                if max_rows_per_split is None:
                    if rows != expected_rows or observed_hash != expected_hash:
                        raise SftDataError(
                            f"源 {source_path.name} 行数或 SHA256 与 manifest 不一致"
                        )
                elif rows != min(expected_rows, max_rows_per_split):
                    raise SftDataError("smoke 未读取预期的源前缀行数")
                source_files[split] = {
                    "path": str(source_path),
                    "manifest_rows": expected_rows,
                    "manifest_sha256": expected_hash,
                    "scanned_rows": rows,
                    "scanned_sha256": observed_hash,
                    "fully_scanned": max_rows_per_split is None,
                }
                if progress is not None:
                    progress(f"{split} 成对转换完成：{rows:,} 条")
        finally:
            for variant_handles in handles.values():
                for handle in variant_handles.values():
                    handle.close()

        source_stats = _load_json(source_sft_dir / "stats.json", "源 SFT stats")
        manifests: dict[str, dict[str, Any]] = {}
        output_hashes: dict[str, dict[str, str]] = {}
        for variant in VARIANTS:
            temporary = temporary_dirs[variant]
            variant_stats = {
                **source_stats,
                "retained_sample_count": stats["source_rows"],
                "train_count": stats["train_count"],
                "valid_count": stats["valid_count"],
                "test_count": stats["test_count"],
                "history_event_occurrence_count": stats[
                    "history_event_occurrence_count"
                ],
                "dedup_target_count": stats["dedup_target_count"][variant],
                "identifier_variant": variant,
                "source_row_alignment": "exact_one_to_one_in_source_order",
            }
            _write_json(temporary / "special_tokens.json", special_tokens)
            _write_json(temporary / "stats.json", variant_stats)
            outputs = {
                f"{split}.jsonl": {
                    "rows": stats[f"{split}_count"],
                    "sha256": output_digests[variant][split].hexdigest(),
                    "bytes": (temporary / f"{split}.jsonl").stat().st_size,
                }
                for split in SPLITS
            }
            for filename in ("special_tokens.json", "stats.json"):
                outputs[filename] = {
                    "sha256": sha256_file(temporary / filename),
                    "bytes": (temporary / filename).stat().st_size,
                }
            lookup = lookups[variant]
            base_order = (
                [f"G{level}" for level in range(1, 7)]
                + [f"S{level}" for level in range(1, 4)]
                if variant == "a4_gid_parent"
                else [f"S{level}" for level in range(1, 4)]
            )
            manifest = {
                "schema_version": SFT_DATA_SCHEMA_VERSION,
                "status": "completed",
                "built_at": utc_now(),
                "variant": variant,
                "scan_mode": (
                    "full" if max_rows_per_split is None else "smoke_prefix"
                ),
                "max_rows_per_split": max_rows_per_split,
                "task_name": "history_query_request_gid_to_qg_prqk_final_identifier",
                "source_sft": {
                    "directory": str(source_sft_dir),
                    "manifest_sha256": sha256_file(source_manifest_path),
                    "files": source_files,
                    "sample_order": "unchanged",
                    "query_text": "unchanged",
                    "request_gid": "unchanged",
                    "time_split": source_manifest.get("time_split"),
                },
                "source_tiger_identifier": tiger_source,
                "final_identifier": {
                    "directory": str(lookup.directory),
                    "manifest_sha256": lookup.manifest_sha256,
                    "poi_count": len(lookup.poi_ids),
                    "base_token_order": base_order,
                    "dedup_position": "last_if_required",
                    "dedup_token_capacity": int(
                        lookup.manifest["metrics"]["dedup_token_capacity"]
                    ),
                    "final_unique_ratio": lookup.manifest["metrics"][
                        "final_unique_ratio"
                    ],
                },
                "sequence_contract": {
                    "history_and_target_use_same_identifier": True,
                    "history_wrapper": [
                        "<POI_QGPRQK_ID>",
                        "</POI_QGPRQK_ID>",
                    ],
                    "target_wrapper": ["<TARGET_POI>", "</TARGET_POI>"],
                    "dedup_is_always_last": True,
                    "singleton_omits_dedup": True,
                    "nogid_target_contains_gid": False,
                },
                "paired_contract": {
                    "same_source_scan": True,
                    "same_samples_and_order": True,
                    "same_query_history_and_request_context": True,
                    "only_history_and_target_poi_identifiers_differ": True,
                },
                "stats": variant_stats,
                "outputs": outputs,
            }
            _write_json(temporary / "manifest.json", manifest)
            (temporary / "_SUCCESS").touch()
            manifests[variant] = manifest
            output_hashes[variant] = {
                filename: outputs[filename]["sha256"]
                for filename in outputs
            }
            output_hashes[variant]["manifest.json"] = sha256_file(
                temporary / "manifest.json"
            )
        for variant in VARIANTS:
            os.replace(temporary_dirs[variant], resolved_outputs[variant])
    except BaseException:
        for temporary in temporary_dirs.values():
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    return PairedSftDataResult(
        manifests=manifests,
        output_hashes=output_hashes,
        stats=stats,
    )

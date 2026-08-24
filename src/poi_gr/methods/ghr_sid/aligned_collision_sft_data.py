"""Derive SFT Messages for the fixed TIGER-aligned collision suffix."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from poi_gr.methods.ghr_sid.sft_data import (
    _build_tiger_row_index,
    _transform_record,
    _validate_source_contract,
    build_ghr_special_tokens,
)
from poi_gr.methods.tiger.data import load_tiger_id_lookup
from poi_gr.pid.dedup import sha256_file


SCHEMA_VERSION = "ghr-aligned-collision-sft-data-v1"
VARIANT = "tiger_aligned_collision_32x32_v1"
SPLITS = ("train", "valid", "test")
BASE_CAPACITIES = (1024, 1024, 1024)
SUFFIX_CAPACITIES = (32, 32)


class AlignedCollisionSftDataError(ValueError):
    """Raised when aligned-collision SFT data violates its contract."""


@dataclass
class AlignedCollisionLookup:
    """Memory-mapped five-layer codes with lazy text serialization."""

    identifier_dir: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    codes: np.ndarray
    content_by_row: list[str | None]

    def content(self, row: int) -> str:
        cached = self.content_by_row[row]
        if cached is None:
            cached = aligned_identifier_content(self.codes[row])
            self.content_by_row[row] = cached
        return cached


@dataclass(frozen=True)
class AlignedCollisionSftResult:
    """Completed aligned-collision Messages artifacts."""

    manifest: dict[str, Any]
    output_hashes: dict[str, str]


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlignedCollisionSftDataError(f"无法读取{name}：{path}") from error
    if not isinstance(payload, dict):
        raise AlignedCollisionSftDataError(f"{name}必须是 JSON object")
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


def aligned_identifier_content(codes: Sequence[int]) -> str:
    """Serialize one fixed ``[S1,S2,S3,R1,R2]`` identifier."""

    if len(codes) != 5:
        raise AlignedCollisionSftDataError("aligned identifier 必须固定为五层")
    values = tuple(int(value) for value in codes)
    capacities = (*BASE_CAPACITIES, *SUFFIX_CAPACITIES)
    if any(value < 0 or value >= capacity for value, capacity in zip(values, capacities)):
        raise AlignedCollisionSftDataError("aligned identifier code 超出层容量")
    return (
        f"<S1_{values[0]}><S2_{values[1]}><S3_{values[2]}>"
        f"<R1_{values[3]}><R2_{values[4]}>"
    )


def aligned_identifier_tokens() -> tuple[str, ...]:
    """Return the deterministic ordinary-token inventory for five SID layers."""

    tokens: list[str] = []
    for namespace, capacity in zip(
        ("S1", "S2", "S3", "R1", "R2"),
        (*BASE_CAPACITIES, *SUFFIX_CAPACITIES),
    ):
        tokens.extend(f"<{namespace}_{value}>" for value in range(capacity))
    return tuple(tokens)


def _validate_artifact(
    identifier_dir: Path,
    manifest: Mapping[str, Any],
    filename: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    contract = artifacts.get(filename) if isinstance(artifacts, dict) else None
    if not isinstance(contract, dict) or contract.get("path") != filename:
        raise AlignedCollisionSftDataError(f"identifier manifest 缺少 {filename}")
    path = identifier_dir / filename
    if not path.is_file() or sha256_file(path) != contract.get("sha256"):
        raise AlignedCollisionSftDataError(f"identifier 产物哈希不匹配：{filename}")
    return path


def _load_aligned_lookup(
    identifier_dir: Path,
    tiger_poi_ids: np.ndarray,
    tiger_codes: np.ndarray,
) -> AlignedCollisionLookup:
    identifier_dir = identifier_dir.resolve()
    manifest_path = identifier_dir / "manifest.json"
    manifest = _load_json(manifest_path, "aligned identifier manifest")
    if (
        manifest.get("schema_version") != "ghr-aligned-collision-quantizer-v1"
        or manifest.get("status") != "completed"
    ):
        raise AlignedCollisionSftDataError("aligned identifier manifest 状态或版本无效")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or (
        protocol.get("base_layers"), protocol.get("suffix_layers")
    ) != (3, 2):
        raise AlignedCollisionSftDataError("aligned identifier 必须是冻结三层加两层后缀")

    codes_path = _validate_artifact(identifier_dir, manifest, "identifier_codes.npy")
    mapping_path = _validate_artifact(
        identifier_dir, manifest, "poi_identifier_mapping.parquet"
    )
    _validate_artifact(identifier_dir, manifest, "metrics.json")
    codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
    if codes.dtype != np.int32 or codes.shape != (len(tiger_poi_ids), 5):
        raise AlignedCollisionSftDataError(
            f"identifier_codes 应为 {(len(tiger_poi_ids), 5)} int32，实际 {codes.shape} {codes.dtype}"
        )
    for level, capacity in enumerate((*BASE_CAPACITIES, *SUFFIX_CAPACITIES)):
        values = codes[:, level]
        if np.any(values < 0) or np.any(values >= capacity):
            raise AlignedCollisionSftDataError(f"identifier 第 {level + 1} 层超出容量")
    if not np.array_equal(codes[:, :3], tiger_codes[:, :3]):
        raise AlignedCollisionSftDataError("aligned identifier 前三层未严格冻结 TIGER")

    parquet_file = pq.ParquetFile(mapping_path)
    expected_columns = ("poi_id", "s1", "s2", "s3", "r1", "r2")
    if parquet_file.metadata.num_rows != len(tiger_poi_ids) or not set(
        expected_columns
    ).issubset(parquet_file.schema_arrow.names):
        raise AlignedCollisionSftDataError("aligned mapping 行数或字段不符合契约")
    offset = 0
    for group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(group_index, columns=list(expected_columns))
        values = table.to_pydict()
        row_count = table.num_rows
        end = offset + row_count
        poi_ids = np.asarray([int(value) for value in values["poi_id"]], dtype=np.int64)
        if not np.array_equal(poi_ids, tiger_poi_ids[offset:end]):
            raise AlignedCollisionSftDataError("aligned mapping 与 TIGER POI 行序不一致")
        mapping_codes = np.column_stack(
            [np.asarray(values[name], dtype=np.int32) for name in expected_columns[1:]]
        )
        if not np.array_equal(mapping_codes, codes[offset:end]):
            raise AlignedCollisionSftDataError("aligned mapping 与 identifier_codes 不一致")
        offset = end
    if offset != len(tiger_poi_ids):
        raise AlignedCollisionSftDataError("aligned mapping 未完整扫描")

    metrics = _load_json(identifier_dir / "metrics.json", "aligned metrics")
    identifier_metrics = metrics.get("identifier")
    if not isinstance(identifier_metrics, dict) or (
        identifier_metrics.get("layers") != 5
        or identifier_metrics.get("distinct_count") != len(tiger_poi_ids)
        or identifier_metrics.get("within_bucket_duplicate_pair_count") != 0
    ):
        raise AlignedCollisionSftDataError("aligned identifier 唯一性指标未通过")
    return AlignedCollisionLookup(
        identifier_dir=identifier_dir,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        codes=codes,
        content_by_row=[None] * len(tiger_poi_ids),
    )


def build_aligned_collision_sft_data(
    *,
    source_tiger_sft_dir: Path,
    tiger_identifier_dir: Path,
    aligned_identifier_dir: Path,
    output_dir: Path,
    max_rows_per_split: int | None = None,
    progress_every: int = 250_000,
    progress: Callable[[str], None] | None = None,
) -> AlignedCollisionSftResult:
    """Replace only POI identifiers in the frozen TIGER Messages rows."""

    if max_rows_per_split is not None and max_rows_per_split <= 0:
        raise AlignedCollisionSftDataError("max_rows_per_split 必须大于 0")
    if progress_every <= 0:
        raise AlignedCollisionSftDataError("progress_every 必须大于 0")
    source_dir = source_tiger_sft_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise AlignedCollisionSftDataError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    source_manifest, source_manifest_hash = _validate_source_contract(source_dir)

    if progress is not None:
        progress("校验 TIGER 与 aligned-collision identifier 的同行同序契约")
    row_by_tiger_key, tiger_poi_ids, tiger_mapping_hash, tiger_manifest_hash = (
        _build_tiger_row_index(tiger_identifier_dir)
    )
    tiger_lookup, _, _, _ = load_tiger_id_lookup(tiger_identifier_dir.resolve())
    lookup = _load_aligned_lookup(
        aligned_identifier_dir, tiger_poi_ids, tiger_lookup.codes
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}.building-")
    )
    handles: dict[str, Any] = {}
    digests = {split: hashlib.sha256() for split in SPLITS}
    row_counts = {split: 0 for split in SPLITS}
    source_split_hashes: dict[str, str] = {}
    history_occurrences = 0
    try:
        handles = {
            split: (temporary_dir / f"{split}.jsonl").open("wb") for split in SPLITS
        }
        for split in SPLITS:
            source_digest = hashlib.sha256()
            source_path = source_dir / f"{split}.jsonl"
            with source_path.open("rb") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    if max_rows_per_split is not None and line_number > max_rows_per_split:
                        break
                    source_digest.update(raw_line)
                    try:
                        record = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise AlignedCollisionSftDataError(
                            f"{split}:{line_number} 不是合法 JSON"
                        ) from error
                    if not isinstance(record, dict):
                        raise AlignedCollisionSftDataError(
                            f"{split}:{line_number} 必须是 JSON object"
                        )
                    transformed, occurrences = _transform_record(
                        record,
                        split=split,
                        line_number=line_number,
                        row_by_tiger_key=row_by_tiger_key,
                        tiger_poi_ids=tiger_poi_ids,
                        lookups={VARIANT: lookup},
                    )
                    history_occurrences += occurrences
                    encoded = (
                        json.dumps(
                            transformed[VARIANT],
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                    handles[split].write(encoded)
                    digests[split].update(encoded)
                    row_counts[split] += 1
                    if progress is not None and line_number % progress_every == 0:
                        progress(f"aligned SFT：{split} 已完成 {line_number:,} 行")
            source_split_hashes[split] = source_digest.hexdigest()
            expected = source_manifest["outputs"][f"{split}.jsonl"]
            expected_rows = int(expected["rows"])
            if max_rows_per_split is None:
                if row_counts[split] != expected_rows:
                    raise AlignedCollisionSftDataError(f"{split} 行数与 TIGER 不一致")
                if source_split_hashes[split] != expected["sha256"]:
                    raise AlignedCollisionSftDataError(f"{split} TIGER 源哈希不一致")
            elif row_counts[split] != min(max_rows_per_split, expected_rows):
                raise AlignedCollisionSftDataError(f"{split} smoke 行数不符合上限")
            if progress is not None:
                progress(f"aligned SFT：{split} 完成 {row_counts[split]:,} 行")
    except BaseException:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    else:
        for handle in handles.values():
            handle.close()

    try:
        source_stats = _load_json(source_dir / "stats.json", "TIGER SFT stats")
        special_tokens = build_ghr_special_tokens(aligned_identifier_tokens())
        stats = {
            **source_stats,
            "retained_sample_count": sum(row_counts.values()),
            "train_count": row_counts["train"],
            "valid_count": row_counts["valid"],
            "test_count": row_counts["test"],
            "history_event_occurrence_count": history_occurrences,
            "source_tiger_row_alignment": "exact_one_to_one_in_source_order",
            "identifier_variant": VARIANT,
            "identifier_fixed_length": 5,
        }
        _write_json(temporary_dir / "special_tokens.json", special_tokens)
        _write_json(temporary_dir / "stats.json", stats)
        outputs = {
            f"{split}.jsonl": {
                "rows": row_counts[split],
                "sha256": digests[split].hexdigest(),
            }
            for split in SPLITS
        }
        for filename in ("special_tokens.json", "stats.json"):
            outputs[filename] = {"sha256": sha256_file(temporary_dir / filename)}
        fingerprint_payload = {
            "schema_version": SCHEMA_VERSION,
            "source_tiger_manifest_sha256": source_manifest_hash,
            "source_split_sha256": source_split_hashes,
            "tiger_identifier_mapping_sha256": tiger_mapping_hash,
            "tiger_identifier_manifest_sha256": tiger_manifest_hash,
            "aligned_identifier_manifest_sha256": lookup.manifest_sha256,
            "max_rows_per_split": max_rows_per_split,
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "task_name": "ghr_aligned_collision_history_query_gid_poi_to_target",
            "method": "GHR-SID",
            "variant": VARIANT,
            "build_fingerprint": hashlib.sha256(
                json.dumps(
                    fingerprint_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "source_tiger_sft": {
                "path": str(source_dir),
                "manifest_sha256": source_manifest_hash,
                "split_sha256": source_split_hashes,
                "derivation": "one_to_one_identifier_only_remap",
            },
            "tiger_identifier": {
                "path": str(tiger_identifier_dir.resolve()),
                "mapping_sha256": tiger_mapping_hash,
                "manifest_sha256": tiger_manifest_hash,
            },
            "aligned_identifier": {
                "path": str(lookup.identifier_dir),
                "schema_version": lookup.manifest["schema_version"],
                "manifest_sha256": lookup.manifest_sha256,
                "poi_count": len(lookup.codes),
                "token_order": ["S1", "S2", "S3", "R1", "R2"],
                "token_capacities": [*BASE_CAPACITIES, *SUFFIX_CAPACITIES],
                "fixed_length": 5,
                "query_or_order_used_to_build_identifier": False,
            },
            "alignment": {
                "source_order_preserved": True,
                "sample_id_preserved": True,
                "split_preserved": True,
                "metadata_preserved": True,
                "raw_query_preserved": True,
                "request_gid_preserved": True,
                "history_length_preserved": True,
                "changed_fields": [
                    "messages.history_poi_identifier",
                    "messages.target_poi_identifier",
                    "target_tiger_id_key->target_ghr_id_key",
                ],
                "row_counts": row_counts,
                "history_identifier_occurrences_replaced": history_occurrences,
            },
            "input": {
                "scan_mode": "full" if max_rows_per_split is None else "smoke_prefix",
                "max_rows_per_split": max_rows_per_split,
            },
            "history": source_manifest["history"],
            "user_identifier": source_manifest["user_identifier"],
            "time_split": source_manifest["time_split"],
            "prompt": {
                **source_manifest["prompt"],
                "history_event_order": [
                    "request_geohash6_gid",
                    "raw_query",
                    "fixed_five_layer_aligned_collision_id",
                ],
                "target": "fixed_five_layer_unique_aligned_collision_id",
            },
            "processing_rules": source_manifest["processing_rules"],
            "stats": stats,
            "outputs": outputs,
        }
        _write_json(temporary_dir / "manifest.json", manifest)
        filenames = (*[f"{split}.jsonl" for split in SPLITS], "special_tokens.json", "stats.json", "manifest.json")
        output_hashes = {
            filename: sha256_file(temporary_dir / filename) for filename in filenames
        }
        for split in SPLITS:
            if output_hashes[f"{split}.jsonl"] != outputs[f"{split}.jsonl"]["sha256"]:
                raise AlignedCollisionSftDataError(f"{split} 写入后哈希不一致")
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return AlignedCollisionSftResult(manifest=manifest, output_hashes=output_hashes)

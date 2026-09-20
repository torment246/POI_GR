"""Freeze an exact Train-only SHA-priority mining candidate pool."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .config import BeamRiskConfig
from .errors import BeamRiskError
from .io import atomic_write_json, implementation_sha256, read_json, sha256_file
from .schema import business_key


CANDIDATE_SCHEMA_VERSION = "beamrisk-candidate-pool-v1"
SAMPLE_ID_PATTERN = re.compile(br'^\{"sample_id":\s*"([0-9a-f]{64})"')


@dataclass(frozen=True)
class SelectedLine:
    priority: int
    sample_id: str
    offset: int
    length: int
    selection_rank: int


def load_validation_business_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BeamRiskError(
                    f"固定 Validation 第 {line_number} 行 JSON 无效"
                ) from error
            key = business_key(value.get("order_id"), value.get("searchid"))
            if key in keys:
                raise BeamRiskError("固定 Validation 10k 存在重复业务键")
            keys.add(key)
    if len(keys) != 10_000:
        raise BeamRiskError(f"固定 Validation 必须是 10,000 条，实际 {len(keys)}")
    return keys


def _scan_smallest_hash_lines(
    source: Path,
    *,
    expected_rows: int,
    expected_sha256: str,
    pool_size: int,
) -> list[SelectedLine]:
    # Heap root is the currently largest retained SHA priority.
    heap: list[tuple[int, int, int, str]] = []
    source_digest = hashlib.sha256()
    rows = 0
    with source.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            rows += 1
            source_digest.update(line)
            match = SAMPLE_ID_PATTERN.match(line)
            if match is None:
                raise BeamRiskError(f"Train 第 {rows} 行 sample_id 格式或字段顺序无效")
            sample_id = match.group(1).decode("ascii")
            priority = int(sample_id, 16)
            item = (-priority, offset, len(line), sample_id)
            if len(heap) < pool_size:
                heapq.heappush(heap, item)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, item)
            if rows % 500_000 == 0:
                print(
                    f"[candidate-pool] scanned={rows:,}/{expected_rows:,}",
                    flush=True,
                )
    if rows != expected_rows:
        raise BeamRiskError(f"Train 实际行数 {rows:,} != manifest {expected_rows:,}")
    observed_sha256 = source_digest.hexdigest()
    if observed_sha256 != expected_sha256:
        raise BeamRiskError(
            "Train JSONL SHA256 与 manifest 不一致："
            f"{observed_sha256} != {expected_sha256}"
        )
    if len(heap) != pool_size:
        raise BeamRiskError(f"候选池只有 {len(heap):,} 条，要求 {pool_size:,}")
    ordered = sorted(
        (
            (-negative_priority, sample_id, offset, length)
            for negative_priority, offset, length, sample_id in heap
        ),
        key=lambda item: (item[0], item[1]),
    )
    if len({item[1] for item in ordered}) != pool_size:
        raise BeamRiskError("SHA 候选池包含重复 sample_id")
    return [
        SelectedLine(
            priority=priority,
            sample_id=sample_id,
            offset=offset,
            length=length,
            selection_rank=rank,
        )
        for rank, (priority, sample_id, offset, length) in enumerate(ordered)
    ]


def _write_selected_parts(
    source: Path,
    selected: list[SelectedLine],
    *,
    output_dir: Path,
    world_size: int,
    validation_keys: set[str],
) -> list[dict[str, object]]:
    by_offset = sorted(selected, key=lambda item: item.offset)
    temporary_paths = [
        output_dir / f".candidates_part_{rank:03d}.jsonl.tmp"
        for rank in range(world_size)
    ]
    final_paths = [
        output_dir / f"candidates_part_{rank:03d}.jsonl"
        for rank in range(world_size)
    ]
    streams: list[BinaryIO] = []
    digests = [hashlib.sha256() for _ in range(world_size)]
    rows = [0 for _ in range(world_size)]
    selected_business_keys: set[str] = set()
    try:
        streams = [path.open("wb") for path in temporary_paths]
        with source.open("rb") as input_stream:
            for item in by_offset:
                input_stream.seek(item.offset)
                line = input_stream.read(item.length)
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise BeamRiskError(
                        f"候选 sample JSON 解析失败：{item.sample_id}"
                    ) from error
                if value.get("sample_id") != item.sample_id:
                    raise BeamRiskError("候选 byte offset 与 sample_id 不一致")
                if value.get("split") != "train":
                    raise BeamRiskError("候选池只允许 Train 样本")
                key = business_key(value.get("order_id"), value.get("searchid"))
                if key in validation_keys:
                    raise BeamRiskError(f"候选池与固定 Validation 泄漏：{key}")
                if key in selected_business_keys:
                    raise BeamRiskError(f"候选池包含重复业务键：{key}")
                selected_business_keys.add(key)
                rank = item.selection_rank % world_size
                streams[rank].write(line)
                digests[rank].update(line)
                rows[rank] += 1
        for stream in streams:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        streams = []
        for temporary, final in zip(temporary_paths, final_paths):
            temporary.replace(final)
    finally:
        for stream in streams:
            stream.close()
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return [
        {
            "rank": rank,
            "path": final_paths[rank].name,
            "rows": rows[rank],
            "bytes": final_paths[rank].stat().st_size,
            "sha256": digests[rank].hexdigest(),
        }
        for rank in range(world_size)
    ]


def validate_candidate_pool(
    config: BeamRiskConfig,
    *,
    verify_hashes: bool = False,
) -> dict[str, object]:
    manifest_path = config.paths.candidate_pool_dir / "manifest.json"
    manifest = read_json(manifest_path, name="Candidate pool manifest")
    if manifest.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise BeamRiskError("Candidate pool schema_version 无效")
    if manifest.get("status") != "completed":
        raise BeamRiskError("Candidate pool 状态不是 completed")
    if manifest.get("candidate_pool_size") != config.mining.candidate_pool_size:
        raise BeamRiskError("Candidate pool 数量与当前配置不一致")
    if manifest.get("world_size") != config.training.world_size:
        raise BeamRiskError("Candidate pool 分片数与当前配置不一致")
    input_meta = manifest.get("input")
    if not isinstance(input_meta, dict):
        raise BeamRiskError("Candidate pool input manifest 无效")
    expected_input = {
        "train_jsonl": str(config.paths.raw_train),
        "manifest": str(config.paths.raw_train_manifest),
        "fixed_validation_subset": str(config.paths.fixed_validation_subset),
    }
    for name, expected in expected_input.items():
        if input_meta.get(name) != expected:
            raise BeamRiskError(f"Candidate pool 输入 {name} 与当前配置不一致")
    if input_meta.get("raw_train_manifest_sha256") != sha256_file(
        config.paths.raw_train_manifest
    ):
        raise BeamRiskError("Candidate pool 的 Train manifest 已发生变化")
    if input_meta.get("fixed_validation_subset_sha256") != sha256_file(
        config.paths.fixed_validation_subset
    ):
        raise BeamRiskError("Candidate pool 的固定 Validation 已发生变化")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or len(parts) != config.training.world_size:
        raise BeamRiskError("Candidate pool parts 无效")
    observed_rows = 0
    for part in parts:
        if not isinstance(part, dict):
            raise BeamRiskError("Candidate pool part manifest 无效")
        path = config.paths.candidate_pool_dir / str(part.get("path", ""))
        if not path.is_file() or path.stat().st_size != part.get("bytes"):
            raise BeamRiskError(f"Candidate pool part 缺失或字节数变化：{path}")
        if verify_hashes and sha256_file(path) != part.get("sha256"):
            raise BeamRiskError(f"Candidate pool part SHA256 变化：{path}")
        observed_rows += int(part.get("rows", 0))
    if observed_rows != config.mining.candidate_pool_size:
        raise BeamRiskError("Candidate pool 分片总行数不一致")
    return manifest


def build_candidate_pool(
    config: BeamRiskConfig,
    *,
    reuse: bool = False,
) -> dict[str, object]:
    if reuse and (config.paths.candidate_pool_dir / "manifest.json").is_file():
        return validate_candidate_pool(config)
    output_dir = config.paths.candidate_pool_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise BeamRiskError(f"Candidate pool 输出目录非空：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest = read_json(config.paths.raw_train_manifest, name="TIGER data manifest")
    train_output = raw_manifest.get("outputs", {}).get("train.jsonl", {})
    expected_rows = train_output.get("rows")
    if not isinstance(expected_rows, int):
        raise BeamRiskError("TIGER data manifest 缺少 Train rows")
    expected_sha256 = train_output.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise BeamRiskError("TIGER data manifest 缺少 Train SHA256")
    validation_keys = load_validation_business_keys(
        config.paths.fixed_validation_subset
    )
    selected = _scan_smallest_hash_lines(
        config.paths.raw_train,
        expected_rows=expected_rows,
        expected_sha256=expected_sha256,
        pool_size=config.mining.candidate_pool_size,
    )
    parts = _write_selected_parts(
        config.paths.raw_train,
        selected,
        output_dir=output_dir,
        world_size=config.training.world_size,
        validation_keys=validation_keys,
    )
    selected_digest = hashlib.sha256()
    for item in selected:
        selected_digest.update(item.sample_id.encode("ascii"))
        selected_digest.update(b"\n")
    manifest: dict[str, object] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "status": "completed",
        "selection": "smallest_sample_id_sha256_then_rank_mod_world_size",
        "candidate_pool_size": config.mining.candidate_pool_size,
        "world_size": config.training.world_size,
        "implementation_sha256": implementation_sha256(config.root),
        "selected_sample_ids_sha256": selected_digest.hexdigest(),
        "input": {
            "train_jsonl": str(config.paths.raw_train),
            "manifest": str(config.paths.raw_train_manifest),
            "raw_train_manifest_sha256": sha256_file(
                config.paths.raw_train_manifest
            ),
            "expected_rows": expected_rows,
            "expected_sha256": expected_sha256,
            "fixed_validation_subset": str(config.paths.fixed_validation_subset),
            "fixed_validation_subset_sha256": sha256_file(
                config.paths.fixed_validation_subset
            ),
            "fixed_validation_business_keys": len(validation_keys),
            "train_validation_overlap": 0,
        },
        "parts": parts,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest

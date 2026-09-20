"""Build the A0-GID control by changing only identifiers in frozen A4 messages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.identifiers import (
    FINAL_IDENTIFIER_SCHEMA_VERSION,
    _write_mapping,
    assign_optional_dedup,
    identifier_content,
    identifier_key,
)
from qg_prqk.sft.data import SFT_DATA_SCHEMA_VERSION, load_final_identifier_lookup


VARIANT = "a0_gid"
# Reuse the existing nine-code wire format, not the A4 quantizer or provenance.
WIRE_FORMAT = "a4_gid_parent"
HISTORY_ID = re.compile(
    r"<POI_QGPRQK_ID>((?:<G_[0-9bcdefghjkmnpqrstuvwxyz]>){6}"
    r"<S1_\d+><S2_\d+><S3_\d+>(?:<D_\d+>)?)</POI_QGPRQK_ID>"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 必须是对象：{path}")
    return value


def file_state(path: Path) -> dict[str, int]:
    """Track immutable files cheaply between completed platform stages."""
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def checked_artifact(directory: Path, manifest: Mapping[str, Any], name: str) -> Path:
    if Path(name).name != name:
        raise ValueError("产物名称不能包含目录")
    path = directory / name
    if sha256_file(path) != manifest["artifacts"][name]["sha256"]:
        raise ValueError(f"冻结产物哈希不一致：{path}")
    return path


def completed_stage(output: Path, contract: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reuse only a completed, contract-matched stage with unchanged files."""
    if not output.exists():
        return None
    manifest = read_json(output / "manifest.json")
    marker = read_json(output / "_SUCCESS")
    if (
        manifest.get("contract") != dict(contract)
        or manifest.get("status") != "completed"
        or marker.get("manifest_sha256") != sha256_file(output / "manifest.json")
    ):
        raise ValueError(f"已有阶段不完整或输入已变化，拒绝覆盖：{output}")
    for name, spec in manifest["artifacts"].items():
        if Path(name).name != name or file_state(output / name) != spec["stat"]:
            raise ValueError(f"已有阶段产物变化，拒绝复用：{output / name}")
    return manifest


def publish_stage(
    temporary: Path, output: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Publish an immutable stage only after its complete artifact inventory exists."""
    manifest.update(status="completed", built_at=utc_now())
    known_hashes = {
        name: value["sha256"] for name, value in manifest.get("outputs", {}).items()
    }
    manifest["artifacts"] = {
        path.name: {
            "sha256": known_hashes.get(path.name) or sha256_file(path),
            "stat": file_state(path),
        }
        for path in sorted(temporary.iterdir())
        if path.is_file()
    }
    write_json_atomic(temporary / "manifest.json", manifest)
    write_json_atomic(
        temporary / "_SUCCESS",
        {
            "manifest_sha256": sha256_file(temporary / "manifest.json"),
        },
    )
    if output.exists():
        raise ValueError(f"输出在构建期间出现，拒绝覆盖：{output}")
    os.rename(temporary, output)
    return manifest


def build_a0_identifiers(
    *,
    a0_dir: Path,
    a4_id_dir: Path,
    poi_ids_path: Path,
    token_mapping: Mapping[str, Any],
    output: Path,
    contract: Mapping[str, Any],
    expected_rows: int,
) -> dict[str, Any]:
    """Join aligned A0 SID and existing deterministic GID; assign fresh final Dedup."""
    existing = completed_stage(output, contract)
    if existing is not None:
        return existing
    a0 = read_json(a0_dir / "manifest.json")
    if (
        a0.get("status") != "completed"
        or not (a0_dir / "_SUCCESS").is_file()
        or a0["contract"]["gate"] != "full"
        or a0["contract"]["codebook_sizes"] != [512, 512, 512]
    ):
        raise ValueError("A0 必须是已完成的全量 512×3 冻结产物")
    old = load_final_identifier_lookup(a4_id_dir, WIRE_FORMAT)
    sid = np.load(
        checked_artifact(a0_dir, a0, "poi_assignments_s1_s2_s3.npy"), mmap_mode="r"
    )
    selected = np.load(
        checked_artifact(a0_dir, a0, "selected_poi_rows.npy"), mmap_mode="r"
    )
    if (
        len(old.poi_ids) != expected_rows
        or sid.shape != (expected_rows, 3)
        or sid.dtype != np.int32
        or np.any(sid < 0)
        or np.any(sid >= 512)
        or not np.array_equal(selected, np.arange(expected_rows))
    ):
        raise ValueError("A0/A4 行数、SID 范围或 identity 行序不一致")
    if sha256_file(poi_ids_path) != a0["contract"]["source_hashes"]["poi_ids_sha256"]:
        raise ValueError("POI ID 列表与 A0 拟合来源不一致")
    with poi_ids_path.open(encoding="utf-8") as stream:
        poi_ids = [json.loads(line) for line in stream]
    if poi_ids != old.poi_ids:
        raise ValueError("A0/A4 的 POI ID 必须逐行完全一致")
    base = np.column_stack((old.base_codes[:, :6], sid)).astype(np.int32)
    assignment = assign_optional_dedup(base, poi_ids)
    required = {f"<D_{i}>" for i in range(assignment.max_bucket_size)}
    required.update(f"<S{level}_{i}>" for level in (1, 2, 3) for i in range(512))
    missing = sorted(required - set(token_mapping["tokens"]))
    if missing:
        raise ValueError(f"A4 共同初始词表不覆盖 A0，停止而不扩词：{missing[:10]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    np.save(temporary / "base_identifier_codes.npy", base, allow_pickle=False)
    np.save(temporary / "dedup_codes.npy", assignment.codes, allow_pickle=False)
    np.save(temporary / "requires_dedup.npy", assignment.codes >= 0, allow_pickle=False)
    _write_mapping(
        temporary / "poi_final_id_mapping.parquet",
        poi_ids,
        base,
        assignment,
        variant=WIRE_FORMAT,
        chunk_rows=100_000,
    )
    metrics = {
        "poi_count": expected_rows,
        "base_length": 9,
        "base_distinct_count": assignment.base_distinct_count,
        "colliding_poi_count": assignment.colliding_poi_count,
        "max_base_bucket_size": assignment.max_bucket_size,
        "dedup_token_capacity": assignment.max_bucket_size,
        "final_distinct_count": expected_rows,
        "final_unique_ratio": 1.0,
    }
    write_json_atomic(temporary / "metrics.json", metrics)
    return publish_stage(
        temporary,
        output,
        {
            "schema_version": FINAL_IDENTIFIER_SCHEMA_VERSION,
            "variant": VARIANT,
            "contract": dict(contract),
            "metrics": metrics,
            "identifier_contract": {
                "base_token_order": [f"G{i}" for i in range(1, 7)] + ["S1", "S2", "S3"],
                "dedup_position": "last_if_base_identifier_collides",
                "dedup_assignment": "poi_id_lexicographic_zero_based_within_base_bucket",
            },
            "sid_source": {"directory": str(a0_dir), "quantizer": "frozen_A0_POI_only"},
            "gid_source": {
                "directory": str(a4_id_dir),
                "use": "deterministic_GID6_only",
            },
        },
    )


def remap_record(
    source: Mapping[str, Any],
    *,
    split: str,
    old_row_by_content: Mapping[str, int],
    row_by_poi_id: Mapping[str, int],
    new_content: Callable[[int], str],
    new_key: Callable[[int], str],
    requires_dedup: np.ndarray,
) -> dict[str, Any]:
    """Replace only history/target POI IDs and their identifier metadata."""
    if split not in ("train", "valid") or source.get("split") != split:
        raise ValueError("只允许冻结 Train/Valid，split 不一致")
    messages = source.get("messages")
    if (
        source.get("identifier_variant") != WIRE_FORMAT
        or not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise ValueError("源样本必须是 A4 GID-parent 两轮 Messages")
    row = row_by_poi_id.get(source.get("target_poi_id"))
    target = messages[1]["content"]
    if not target.startswith("<TARGET_POI>") or not target.endswith("</TARGET_POI>"):
        raise ValueError("源 target wrapper 非法")
    if row is None or old_row_by_content.get(target[12:-13]) != row:
        raise ValueError("源 target 内容与 target_poi_id 不一致")
    user = messages[0]["content"]
    if user.count("<CURRENT>") != 1 or user.count("</CURRENT>") != 1:
        raise ValueError("源样本必须有一个完整 CURRENT")
    current = user.split("<CURRENT>", 1)[1].split("</CURRENT>", 1)[0]

    def replace(match: re.Match[str]) -> str:
        history_row = old_row_by_content.get(match.group(1))
        if history_row is None:
            raise ValueError("历史 identifier 不在冻结 A4 全目录")
        return f"<POI_QGPRQK_ID>{new_content(history_row)}</POI_QGPRQK_ID>"

    transformed, count = HISTORY_ID.subn(replace, user)
    if (
        count != source.get("history_length")
        or user.count("<POI_QGPRQK_ID>") != count
        or transformed.split("<CURRENT>", 1)[1].split("</CURRENT>", 1)[0] != current
    ):
        raise ValueError("历史长度/wrapper 不匹配，或 CURRENT 被修改")
    result = dict(source)
    result.update(
        messages=[
            {"role": "user", "content": transformed},
            {
                "role": "assistant",
                "content": f"<TARGET_POI>{new_content(row)}</TARGET_POI>",
            },
        ],
        target_qg_prqk_id_key=new_key(row),
        requires_dedup=bool(requires_dedup[row]),
        identifier_variant=VARIANT,
    )
    return result


def build_a0_messages(
    *,
    source_dir: Path,
    a4_id_dir: Path,
    a0_id_dir: Path,
    output: Path,
    contract: Mapping[str, Any],
    datasets: Mapping[str, str],
    expected_rows: Mapping[str, int],
    progress_every: int = 100_000,
) -> dict[str, Any]:
    """Stream the two frozen splits once; never open or generate business Test."""
    existing = completed_stage(output, contract)
    if existing is not None:
        return existing
    source_manifest = read_json(source_dir / "manifest.json")
    if (
        source_manifest.get("status") != "completed"
        or source_manifest.get("variant") != WIRE_FORMAT
        or not (source_dir / "_SUCCESS").is_file()
    ):
        raise ValueError("A4 源数据尚未完成")
    old = load_final_identifier_lookup(a4_id_dir, WIRE_FORMAT)
    base = np.load(a0_id_dir / "base_identifier_codes.npy", mmap_mode="r")
    dedup = np.load(a0_id_dir / "dedup_codes.npy", mmap_mode="r")
    if base.shape != old.base_codes.shape or dedup.shape != old.dedup_codes.shape:
        raise ValueError("A0/A4 Final ID 行数不一致")
    old_rows = {old.content(row): row for row in range(len(old.poi_ids))}
    requires_dedup = dedup >= 0
    if len(old_rows) != len(old.poi_ids):
        raise ValueError("A4 Final ID 非唯一")

    @lru_cache(maxsize=len(old.poi_ids))
    def content(row: int) -> str:
        return identifier_content(base[row], int(dedup[row]), variant=WIRE_FORMAT)

    def key(row: int) -> str:
        return identifier_key(base[row], int(dedup[row]), variant=WIRE_FORMAT)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    outputs: dict[str, Any] = {}
    registry: dict[str, Any] = {}
    for split in ("train", "valid"):
        spec = source_manifest["outputs"][f"{split}.jsonl"]
        if spec["rows"] != expected_rows[split]:
            raise ValueError(f"{split} 源数据行数不符合冻结协议")
        digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        rows = 0
        with (
            (source_dir / f"{split}.jsonl").open("rb") as source,
            (temporary / f"{split}.jsonl").open("wb") as target,
        ):
            for raw in source:
                digest.update(raw)
                record = remap_record(
                    json.loads(raw),
                    split=split,
                    old_row_by_content=old_rows,
                    row_by_poi_id=old.row_by_poi_id,
                    new_content=content,
                    new_key=key,
                    requires_dedup=requires_dedup,
                )
                encoded = (
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
                target.write(encoded)
                output_digest.update(encoded)
                rows += 1
                if rows % progress_every == 0:
                    print(f"A0-GID 数据 {split}: {rows:,}/{spec['rows']:,}", flush=True)
        if rows != expected_rows[split] or digest.hexdigest() != spec["sha256"]:
            raise ValueError(f"{split} 源数据行数或 SHA256 不一致，拒绝发布")
        outputs[f"{split}.jsonl"] = {"rows": rows, "sha256": output_digest.hexdigest()}
        registry[datasets[split]] = {
            "file_name": str(output / f"{split}.jsonl"),
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    write_json_atomic(temporary / "dataset_info.json", registry)
    return publish_stage(
        temporary,
        output,
        {
            "schema_version": SFT_DATA_SCHEMA_VERSION,
            "variant": VARIANT,
            "scan_mode": "full",
            "contract": dict(contract),
            "outputs": outputs,
            "final_identifier": {
                "directory": str(a0_id_dir),
                "manifest_sha256": sha256_file(a0_id_dir / "manifest.json"),
            },
            "paired_contract": {
                "source_variant": WIRE_FORMAT,
                "same_samples_and_order": True,
                "only_history_and_target_poi_identifiers_differ": True,
            },
            "business_test_read": False,
        },
    )

"""Replace frozen TIGER identifiers without rebuilding user interaction inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import re

import numpy as np
from pathlib import Path
from typing import Any, Callable, Mapping

from poi_gr.methods.genpoi.data import GenpoiDataError, build_genpoi_special_tokens
from poi_gr.sft.data import load_pid_lookup, assistant_pid_content, stable_final_pid_key
from poi_gr.methods.tiger.data import load_tiger_id_lookup, tiger_id_content
from poi_gr.pid.dedup import sha256_file

HISTORY = re.compile(r"<POI_TIGER_ID>(<S1_\d+><S2_\d+><S3_\d+><C_\d+>)</POI_TIGER_ID>")
TARGET = re.compile(r"<TARGET_POI>(<S1_\d+><S2_\d+><S3_\d+><C_\d+>)</TARGET_POI>")
GENPOI_HISTORY = re.compile(r"<POI_PID>(?:<G_[0-9bcdefghjkmnpqrstuvwxyz]>){6}<S1_\d+><S2_\d+><S3_\d+>(?:<D_\d+>)?</POI_PID>")


def aligned_special_tokens(sid_codebook_size: int, source: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the exact source user vocabulary, retaining the GenPOI PID grammar."""
    users = source.get("user_tokens")
    if users != [f"<U_{i:04d}>" for i in range(2000)]:
        raise GenpoiDataError("源 TIGER 必须包含完整且有序的 2000 桶用户 Token")
    result = build_genpoi_special_tokens(sid_codebook_size=sid_codebook_size)
    result["structure_tokens"] = ["<USER_ID>", "</USER_ID>", "<EVENT>", "</EVENT>", *result["structure_tokens"]]
    result["user_tokens"] = list(users)
    result["user_bucket_count"] = 2000
    result["additional_special_tokens"] = [
        *result["structure_tokens"], *result["geohash_tokens"], *users,
        *(token for group in result["sid_tokens"].values() for token in group),
        *result["dedup_tokens"],
    ]
    result["token_count"] = len(result["additional_special_tokens"])
    return result


def transform_record(source: dict[str, Any], split: str,
                     replacements: Mapping[str, tuple[str, str, str]]) -> dict[str, Any]:
    """Change only history/target ID content and the method-specific target key."""
    messages = source.get("messages")
    if (source.get("split") != split or not isinstance(messages, list)
            or len(messages) != 2 or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"):
        raise GenpoiDataError("源样本 split 或两轮 messages 契约错误")
    content = messages[0]["content"]
    user = source.get("user_token")
    if (not isinstance(user, str) or re.fullmatch(r"<U_\d{4}>", user) is None
            or int(user[3:7]) >= 2000
            or not content.startswith(f"<USER_ID>{user}</USER_ID>\n")):
        raise GenpoiDataError("源用户哈希与 Prompt 不一致")
    target = TARGET.fullmatch(messages[1]["content"])
    target_info = replacements.get(target.group(1)) if target else None
    if target_info is None or target_info[0] != source.get("target_poi_id"):
        raise GenpoiDataError("源目标 SID 与 target_poi_id 不一致或不存在")

    def replace(match: re.Match[str]) -> str:
        info = replacements.get(match.group(1))
        if info is None:
            raise GenpoiDataError("历史 SID 不在冻结 TIGER mapping")
        return f"<POI_PID>{info[1]}</POI_PID>"

    transformed, count = HISTORY.subn(replace, content)
    if (count != source.get("history_length") or "<POI_TIGER_ID>" in transformed
            or "</POI_TIGER_ID>" in transformed):
        raise GenpoiDataError("历史条数或 SID 语法错误")
    # Audit every row, including queries, GIDs, newlines and complete user tokens.
    if HISTORY.sub("<ID>", content) != GENPOI_HISTORY.sub("<ID>", transformed):
        raise GenpoiDataError("替换 SID 后非标识输入发生变化")
    result = {key: value for key, value in source.items()
              if key not in {"messages", "target_tiger_id_key"}}
    result["messages"] = [dict(messages[0], content=transformed),
                          dict(messages[1], content=target_info[1])]
    result["target_pid_key"] = target_info[2]
    result["requires_dedup"] = "<D_" in target_info[1]
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def build_aligned_sft_data(*, source_dir: Path, tiger_id_dir: Path, pid_dir: Path,
                           output_dir: Path, max_rows_per_split: int | None = None,
                           sid_codebook_size: int = 512,
                           progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Stream all frozen splits, verify input hashes and atomically publish GenPOI data."""
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    staging = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists() or staging.exists():
        raise GenpoiDataError("输出或 .building 目录已存在，拒绝覆盖")
    if max_rows_per_split is not None and max_rows_per_split <= 0:
        raise GenpoiDataError("smoke 行数必须大于零")
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if (source_manifest.get("status") != "completed"
            or source_manifest.get("schema_version") != "tiger-map-search-sft-data-v1"):
        raise GenpoiDataError("源数据不是已完成的 TIGER SFT")
    tiger, _, tiger_hash, tiger_manifest_hash = load_tiger_id_lookup(
        tiger_id_dir, expected_base_codebook_sizes=(512, 512, 512))
    contract = source_manifest["tiger_identifier"]
    if (tiger_hash != contract["mapping_sha256"]
            or tiger_manifest_hash != contract["manifest_sha256"]):
        raise GenpoiDataError("源 SFT 与 TIGER mapping 指纹不一致")
    pid, pid_manifest, pid_hash, pid_manifest_hash = load_pid_lookup(
        pid_dir / "poi_pid_mapping.parquet", pid_dir / "final_pid_manifest.json")
    if sid_codebook_size not in (512, 1024):
        raise GenpoiDataError("SID 容量只支持 512 或 1024")
    if np.any(pid.codes[:, 6:] < 0) or np.any(pid.codes[:, 6:] >= sid_codebook_size):
        raise GenpoiDataError("PID 的 SID 超出声明容量")
    if tiger.poi_count != pid.poi_count or set(tiger.row_by_poi_id) != set(pid.row_by_poi_id):
        raise GenpoiDataError("TIGER 与 GenPOI POI 集合不一致")
    replacements: dict[str, tuple[str, str, str]] = {}
    seen = set()
    for poi_id, row in tiger.row_by_poi_id.items():
        pid_row = pid.row_by_poi_id[poi_id]
        codes, dedup = pid.codes[pid_row], int(pid.dedup_codes[pid_row])
        content = assistant_pid_content(codes, dedup)
        key = stable_final_pid_key(codes, dedup)
        if key in seen:
            raise GenpoiDataError("GenPOI PID 不唯一")
        seen.add(key)
        tiger_content = tiger_id_content(tiger.codes[row])
        if tiger_content in replacements:
            raise GenpoiDataError("源 TIGER SID 不唯一")
        replacements[tiger_content] = (poi_id, content, key)
    source_tokens_path = source_dir / "special_tokens.json"
    if sha256_file(source_tokens_path) != source_manifest["outputs"]["special_tokens.json"]["sha256"]:
        raise GenpoiDataError("源 Token 表指纹不一致")
    tokens = aligned_special_tokens(sid_codebook_size, json.loads(source_tokens_path.read_text()))
    staging.mkdir(parents=True)
    outputs, sources, stats = {}, {}, {"retained_sample_count": 0, "history_length_sum": 0,
                                     "nonempty_history_count": 0, "non_identifier_mismatch_count": 0,
                                     "user_hash_mismatch_count": 0}
    for split in ("train", "valid", "test"):
        filename = f"{split}.jsonl"
        source_digest, output_digest = hashlib.sha256(), hashlib.sha256()
        count = 0
        with (source_dir / filename).open("rb", buffering=8 * 1024 * 1024) as reader, \
             (staging / filename).open("wb", buffering=8 * 1024 * 1024) as writer:
            for raw in reader:
                if max_rows_per_split is not None and count >= max_rows_per_split:
                    break
                source_digest.update(raw)
                source = json.loads(raw)
                record = transform_record(source, split, replacements)
                encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                writer.write(encoded)
                output_digest.update(encoded)
                count += 1
                stats["history_length_sum"] += source["history_length"]
                stats["nonempty_history_count"] += int(source["history_length"] > 0)
                if count % 250000 == 0:
                    progress(f"{split}: {count:,} 行，逐条非 SID 内容一致")
        expected = source_manifest["outputs"][filename]
        if max_rows_per_split is None and (count != expected["rows"]
                or source_digest.hexdigest() != expected["sha256"]):
            raise GenpoiDataError(f"源 {filename} 行数或 SHA256 不一致，拒绝发布")
        if count == 0:
            raise GenpoiDataError(f"{split} 为空")
        outputs[filename] = {"rows": count, "sha256": output_digest.hexdigest()}
        sources[filename] = {"rows": count, "sha256": source_digest.hexdigest(),
                             "complete_file": max_rows_per_split is None}
        stats[f"{split}_count"] = count
        stats["retained_sample_count"] += count
        progress(f"{split} 完成：{count:,} 行")
    stats["average_history_length"] = stats["history_length_sum"] / stats["retained_sample_count"]
    stats["history_coverage_ratio"] = stats["nonempty_history_count"] / stats["retained_sample_count"]
    for name, value in (("special_tokens.json", tokens), ("stats.json", stats)):
        _write_json(staging / name, value)
        outputs[name] = {"sha256": sha256_file(staging / name)}
    manifest = {
        "schema_version": "genpoi-sft-data-v1", "status": "completed",
        "baseline_variant": "centered_geope32_tiger_aligned_user_hash",
        "strict_paper_replication": False,
        "source_sft": {"directory": str(source_dir), "files": sources,
                       "manifest_sha256": sha256_file(source_dir / "manifest.json")},
        "input": {"scan_mode": "full" if max_rows_per_split is None else "smoke",
                  "all_files_fully_scanned": max_rows_per_split is None},
        "tiger_source_identifier": contract,
        "pid": {"schema_version": pid_manifest["schema_version"],
                "poi_count": pid.poi_count, "mapping_sha256": pid_hash,
                "manifest_sha256": pid_manifest_hash},
        "identifier": {"geohash_length": 6, "sid_levels": 3,
                       "sid_codebook_size": sid_codebook_size, "optional_dedup": True},
        "time_split": copy.deepcopy(source_manifest["time_split"]),
        "history": copy.deepcopy(source_manifest["history"]),
        "user_identifier": copy.deepcopy(source_manifest["user_identifier"]),
        "processing_rules": {"only_transform": "history and target POI identifiers",
                             "user_token_copied_without_rehashing": True,
                             "non_identifier_input_audited_every_row": True},
        "stats": stats, "outputs": outputs,
    }
    manifest["build_fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    _write_json(staging / "manifest.json", manifest)
    (staging / "_SUCCESS").touch()
    staging.rename(output_dir)
    return manifest

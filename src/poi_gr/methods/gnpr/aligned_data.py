"""Replace frozen TIGER identifiers without rebuilding user interaction inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from poi_gr.methods.gnpr.data import (
    GnprDataError, build_gnpr_special_tokens, gnpr_id_content, load_gnpr_id_lookup,
)
from poi_gr.methods.tiger.data import load_tiger_id_lookup, tiger_id_content
from poi_gr.pid.dedup import sha256_file

HISTORY = re.compile(r"<POI_TIGER_ID>(<S1_\d+><S2_\d+><S3_\d+><C_\d+>)</POI_TIGER_ID>")
TARGET = re.compile(r"<TARGET_POI>(<S1_\d+><S2_\d+><S3_\d+><C_\d+>)</TARGET_POI>")
GNPR_HISTORY = re.compile(r"<POI_GNPR_ID><a_\d+><b_\d+><c_\d+>(?:<d_\d+>)?</POI_GNPR_ID>")


def aligned_special_tokens(capacities: tuple[int, ...], source: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the exact source user vocabulary, retaining the old GNPR ID grammar."""
    users = source.get("user_tokens")
    if users != [f"<U_{i:04d}>" for i in range(2000)]:
        raise GnprDataError("源 TIGER 必须包含完整且有序的 2000 桶用户 Token")
    result = build_gnpr_special_tokens(capacities)
    result["structure_tokens"] = ["<USER_ID>", "</USER_ID>", *result["structure_tokens"]]
    result["user_tokens"] = list(users)
    result["user_bucket_count"] = 2000
    result["additional_special_tokens"] = [
        *result["structure_tokens"], *result["geohash_tokens"], *users,
        *(token for group in result["item_tokens"].values() for token in group),
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
        raise GnprDataError("源样本 split 或两轮 messages 契约错误")
    content = messages[0]["content"]
    user = source.get("user_token")
    if (not isinstance(user, str) or re.fullmatch(r"<U_\d{4}>", user) is None
            or int(user[3:7]) >= 2000
            or not content.startswith(f"<USER_ID>{user}</USER_ID>\n")):
        raise GnprDataError("源用户哈希与 Prompt 不一致")
    target = TARGET.fullmatch(messages[1]["content"])
    target_info = replacements.get(target.group(1)) if target else None
    if target_info is None or target_info[0] != source.get("target_poi_id"):
        raise GnprDataError("源目标 SID 与 target_poi_id 不一致或不存在")

    def replace(match: re.Match[str]) -> str:
        info = replacements.get(match.group(1))
        if info is None:
            raise GnprDataError("历史 SID 不在冻结 TIGER mapping")
        return f"<POI_GNPR_ID>{info[1]}</POI_GNPR_ID>"

    transformed, count = HISTORY.subn(replace, content)
    if (count != source.get("history_length") or "<POI_TIGER_ID>" in transformed
            or "</POI_TIGER_ID>" in transformed):
        raise GnprDataError("历史条数或 SID 语法错误")
    # Audit every row, including queries, GIDs, newlines and complete user tokens.
    if HISTORY.sub("<ID>", content) != GNPR_HISTORY.sub("<ID>", transformed):
        raise GnprDataError("替换 SID 后非标识输入发生变化")
    result = {key: value for key, value in source.items()
              if key not in {"messages", "target_tiger_id_key"}}
    result["messages"] = [dict(messages[0], content=transformed),
                          dict(messages[1], content=f"<TARGET_POI>{target_info[1]}</TARGET_POI>")]
    result["target_gnpr_id_key"] = target_info[2]
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def build_aligned_sft_data(*, source_dir: Path, tiger_id_dir: Path, gnpr_id_dir: Path,
                           output_dir: Path, max_rows_per_split: int | None = None,
                           progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Stream all frozen splits, verify input hashes and atomically publish GNPR data."""
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    staging = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists() or staging.exists():
        raise GnprDataError("输出或 .building 目录已存在，拒绝覆盖")
    if max_rows_per_split is not None and max_rows_per_split <= 0:
        raise GnprDataError("smoke 行数必须大于零")
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if (source_manifest.get("status") != "completed"
            or source_manifest.get("schema_version") != "tiger-map-search-sft-data-v1"):
        raise GnprDataError("源数据不是已完成的 TIGER SFT")
    tiger, _, tiger_hash, tiger_manifest_hash = load_tiger_id_lookup(
        tiger_id_dir, expected_base_codebook_sizes=(512, 512, 512))
    contract = source_manifest["tiger_identifier"]
    if (tiger_hash != contract["mapping_sha256"]
            or tiger_manifest_hash != contract["manifest_sha256"]):
        raise GnprDataError("源 SFT 与 TIGER mapping 指纹不一致")
    gnpr, gnpr_manifest, gnpr_hash, gnpr_manifest_hash = load_gnpr_id_lookup(gnpr_id_dir)
    if tiger.poi_count != gnpr.poi_count:
        raise GnprDataError("TIGER 与 GNPR POI 数不一致")
    replacements: dict[str, tuple[str, str, str]] = {}
    seen_gnpr = set()
    for poi_id, row in tiger.row_by_poi_id.items():
        gnpr_row = gnpr.row_by_numeric_poi_id.get(int(poi_id))
        if gnpr_row is None or gnpr_row in seen_gnpr:
            raise GnprDataError("TIGER 与 GNPR POI 集合不一致")
        seen_gnpr.add(gnpr_row)
        values = [int(v) for v in gnpr.codes[gnpr_row]]
        content = gnpr_id_content(values)
        key = "-".join(str(v) for v in values[:3])
        if values[3] >= 0:
            key += f"|d{values[3]}"
        tiger_content = tiger_id_content(tiger.codes[row])
        if tiger_content in replacements:
            raise GnprDataError("源 TIGER SID 不唯一")
        replacements[tiger_content] = (poi_id, content, key)
    source_tokens_path = source_dir / "special_tokens.json"
    if sha256_file(source_tokens_path) != source_manifest["outputs"]["special_tokens.json"]["sha256"]:
        raise GnprDataError("源 Token 表指纹不一致")
    tokens = aligned_special_tokens(gnpr.token_capacities, json.loads(source_tokens_path.read_text()))
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
            raise GnprDataError(f"源 {filename} 行数或 SHA256 不一致，拒绝发布")
        if count == 0:
            raise GnprDataError(f"{split} 为空")
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
        "schema_version": "gnpr-map-search-sft-data-v1", "status": "completed",
        "baseline_variant": "map_search_adapted_gnpr_tiger_aligned_user_hash",
        "strict_paper_replication": False,
        "source_sft": {"directory": str(source_dir), "files": sources,
                       "manifest_sha256": sha256_file(source_dir / "manifest.json")},
        "input": {"scan_mode": "full" if max_rows_per_split is None else "smoke",
                  "all_files_fully_scanned": max_rows_per_split is None},
        "tiger_source_identifier": contract,
        "gnpr_identifier": {"schema_version": gnpr_manifest["schema_version"],
                            "poi_count": gnpr.poi_count, "mapping_sha256": gnpr_hash,
                            "manifest_sha256": gnpr_manifest_hash,
                            "token_capacities": list(gnpr.token_capacities)},
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

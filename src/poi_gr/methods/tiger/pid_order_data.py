"""Build paired TIGER SFT data for GID/SID order ablations."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from poi_gr.methods.tiger.data import (
    STRUCTURE_TOKENS,
    TigerDataError,
    load_tiger_id_lookup,
)
from poi_gr.pid.dedup import sha256_file
from poi_gr.pid.geohash import GEOHASH_ALPHABET
from poi_gr.sft.data import PidLookup, SftDataError, load_pid_lookup


SCHEMA_VERSION = "tiger-pid-order-sft-data-v1"
SPECIAL_TOKENS_SCHEMA_VERSION = "tiger-pid-order-special-tokens-v1"
PID_ORDERS = ("gid_sid", "sid_gid")
OUTPUT_FILENAMES = (
    "train.jsonl",
    "valid.jsonl",
    "test.jsonl",
    "special_tokens.json",
    "manifest.json",
    "stats.json",
)
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


class TigerPidOrderDataError(ValueError):
    """Raised when the paired PID-order data contract is violated."""


@dataclass(frozen=True)
class TigerPidOrderDataResult:
    """Metadata for two completed, paired SFT datasets."""

    manifests: dict[str, dict[str, Any]]
    output_hashes: dict[str, dict[str, str]]
    stats: dict[str, Any]


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise TigerPidOrderDataError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerPidOrderDataError(f"{name} JSON 解析失败：{path}") from error
    if not isinstance(value, dict):
        raise TigerPidOrderDataError(f"{name} 必须是 JSON object：{path}")
    return value


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


def build_pid_order_special_tokens(
    *,
    sid_capacities: Sequence[int] = (1024, 1024, 1024),
    dedup_capacity: int = 512,
    user_bucket_count: int = 2000,
) -> dict[str, Any]:
    """Return one shared vocabulary for both PID-order variants."""

    if len(sid_capacities) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in sid_capacities
    ):
        raise TigerPidOrderDataError("sid_capacities 必须包含三个正整数")
    if dedup_capacity <= 0 or user_bucket_count <= 0:
        raise TigerPidOrderDataError("Dedup 与用户 Token 容量必须大于 0")

    structure_tokens = [
        "<POI_PID>" if token == "<POI_TIGER_ID>" else
        "</POI_PID>" if token == "</POI_TIGER_ID>" else token
        for token in STRUCTURE_TOKENS
    ]
    geohash_tokens = [f"<G_{character}>" for character in GEOHASH_ALPHABET]
    width = max(4, len(str(user_bucket_count - 1)))
    user_tokens = [
        f"<U_{value:0{width}d}>" for value in range(user_bucket_count)
    ]
    sid_tokens = {
        f"s{level}": [
            f"<S{level}_{value}>" for value in range(capacity)
        ]
        for level, capacity in enumerate(sid_capacities, start=1)
    }
    dedup_tokens = [f"<D_{value}>" for value in range(dedup_capacity)]
    tokens = [
        *structure_tokens,
        *geohash_tokens,
        *user_tokens,
        *sid_tokens["s1"],
        *sid_tokens["s2"],
        *sid_tokens["s3"],
        *dedup_tokens,
    ]
    if len(tokens) != len(set(tokens)):
        raise TigerPidOrderDataError("PID 顺序消融 Token 表包含重复项")
    if "<D_-1>" in tokens or any(token.startswith("<C_") for token in tokens):
        raise TigerPidOrderDataError("新词表不得包含 D_-1 或旧 TIGER C Token")
    return {
        "schema_version": SPECIAL_TOKENS_SCHEMA_VERSION,
        "shared_by_pid_orders": list(PID_ORDERS),
        "structure_tokens": structure_tokens,
        "geohash_tokens": geohash_tokens,
        "user_bucket_count": user_bucket_count,
        "user_tokens": user_tokens,
        "sid_token_capacities": list(sid_capacities),
        "sid_tokens": sid_tokens,
        "dedup_token_capacity": dedup_capacity,
        "dedup_tokens": dedup_tokens,
        "additional_special_tokens": tokens,
        "token_count": len(tokens),
    }


def pid_content(
    codes: Sequence[int], dedup_code: int, *, order: str
) -> str:
    """Format one unique PID while keeping Dedup last for both variants."""

    if order not in PID_ORDERS:
        raise TigerPidOrderDataError(f"不支持的 PID 顺序：{order}")
    if len(codes) != 9:
        raise TigerPidOrderDataError("PID codes 必须包含 GID6+SID3 九列")
    gid_values = [int(value) for value in codes[:6]]
    sid_values = [int(value) for value in codes[6:]]
    if any(not 0 <= value < 32 for value in gid_values):
        raise TigerPidOrderDataError("GID Token 超出 [0,32)")
    if any(value < 0 for value in sid_values):
        raise TigerPidOrderDataError("SID Token 不能为负数")
    if dedup_code < -1:
        raise TigerPidOrderDataError("Dedup Code 只能为 -1 或非负整数")

    gid = "".join(
        f"<G_{GEOHASH_ALPHABET[value]}>" for value in gid_values
    )
    sid = "".join(
        f"<S{level}_{value}>"
        for level, value in enumerate(sid_values, start=1)
    )
    content = gid + sid if order == "gid_sid" else sid + gid
    if dedup_code >= 0:
        content += f"<D_{dedup_code}>"
    return content


def pid_key(codes: Sequence[int], dedup_code: int, *, order: str) -> str:
    """Serialize one PID in its generation order for evaluation joins."""

    if len(codes) != 9:
        raise TigerPidOrderDataError("PID key 必须包含九个 base Token")
    gid = "-".join(str(int(value)) for value in codes[:6])
    sid = "-".join(str(int(value)) for value in codes[6:])
    base = f"{gid}|{sid}" if order == "gid_sid" else f"{sid}|{gid}"
    return base if dedup_code == -1 else f"{base}|d{dedup_code}"


def _tiger_integer_key(codes: Sequence[int], capacities: Sequence[int]) -> int:
    if len(codes) != 4 or len(capacities) != 4:
        raise TigerPidOrderDataError("TIGER key 必须是四层")
    key = 0
    for value, capacity in zip(codes, capacities):
        value = int(value)
        if not 0 <= value < int(capacity):
            raise TigerPidOrderDataError("TIGER Token 超出声明容量")
        key = key * int(capacity) + value
    return key


def _parse_tiger_content(content: str, capacities: Sequence[int]) -> int:
    match = TIGER_CONTENT_PATTERN.fullmatch(content)
    if match is None:
        raise TigerPidOrderDataError(f"非法四层 TIGER identifier：{content}")
    return _tiger_integer_key(
        [int(value) for value in match.groups()], capacities
    )


def _validate_and_index_identifiers(
    pid_lookup: PidLookup,
    tiger_id_dir: Path,
    source_manifest: Mapping[str, Any],
    pid_manifest: Mapping[str, Any],
    pid_manifest_path: Path,
) -> tuple[dict[int, int], tuple[int, int, int, int], dict[str, str]]:
    source_tiger = source_manifest.get("tiger_identifier")
    if not isinstance(source_tiger, dict):
        raise TigerPidOrderDataError("源 TIGER 数据 manifest 缺少 identifier 契约")
    tiger_lookup, tiger_manifest, mapping_hash, manifest_hash = (
        load_tiger_id_lookup(tiger_id_dir)
    )
    if source_tiger.get("mapping_sha256") != mapping_hash:
        raise TigerPidOrderDataError("源 SFT 与传入 TIGER mapping 不一致")
    if source_tiger.get("manifest_sha256") != manifest_hash:
        raise TigerPidOrderDataError("源 SFT 与传入 TIGER manifest 不一致")
    if tiger_lookup.poi_count != pid_lookup.poi_count:
        raise TigerPidOrderDataError("TIGER 与 Final PID 的 POI 数不一致")

    base_manifest_value = pid_manifest.get("base_pid_source", {}).get("manifest")
    if not isinstance(base_manifest_value, str):
        raise TigerPidOrderDataError("Final PID manifest 缺少 base PID manifest")
    base_manifest_path = Path(base_manifest_value)
    if not base_manifest_path.is_absolute():
        base_manifest_path = pid_manifest_path.resolve().parent / base_manifest_path
    base_manifest = _load_json_object(base_manifest_path, "base PID manifest")
    tiger_sid_hash = tiger_manifest.get("source", {}).get("sid_manifest_sha256")
    if base_manifest.get("sid_source", {}).get("manifest_sha256") != tiger_sid_hash:
        raise TigerPidOrderDataError("Final PID 不是由当前 TIGER 三层 SID 构建")

    capacities = tiger_lookup.token_capacities
    row_by_key: dict[int, int] = {}
    for poi_id, row in tiger_lookup.row_by_poi_id.items():
        pid_row = pid_lookup.row_by_poi_id.get(poi_id)
        if pid_row != row:
            raise TigerPidOrderDataError("TIGER 与 PID 的 POI 行顺序不一致")
        if not np.array_equal(
            tiger_lookup.codes[row, :3], pid_lookup.codes[row, 6:9]
        ):
            raise TigerPidOrderDataError("TIGER SID 与 Final PID SID 不一致")
        key = _tiger_integer_key(tiger_lookup.codes[row], capacities)
        if key in row_by_key:
            raise TigerPidOrderDataError("源 TIGER identifier 不是全局唯一")
        row_by_key[key] = row
    if len(row_by_key) != pid_lookup.poi_count:
        raise TigerPidOrderDataError("TIGER identifier 反向索引不完整")
    del tiger_lookup
    gc.collect()
    return row_by_key, capacities, {
        "mapping_sha256": mapping_hash,
        "manifest_sha256": manifest_hash,
        "sid_manifest_sha256": str(tiger_sid_hash),
    }


def _pid_for_row(pid_lookup: PidLookup, row: int, order: str) -> tuple[str, str]:
    codes = pid_lookup.codes[row]
    dedup_code = int(pid_lookup.dedup_codes[row])
    return (
        pid_content(codes, dedup_code, order=order),
        pid_key(codes, dedup_code, order=order),
    )


def _replace_history_identifiers(
    content: str,
    *,
    expected_count: int,
    order: str,
    row_by_tiger_key: Mapping[int, int],
    tiger_capacities: Sequence[int],
    pid_lookup: PidLookup,
) -> str:
    if not isinstance(content, str):
        raise TigerPidOrderDataError("源 User content 必须是字符串")
    observed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal observed
        observed += 1
        key = _parse_tiger_content(match.group(1), tiger_capacities)
        row = row_by_tiger_key.get(key)
        if row is None:
            raise TigerPidOrderDataError("历史 TIGER identifier 不在冻结映射")
        value, _ = _pid_for_row(pid_lookup, row, order)
        return f"<POI_PID>{value}</POI_PID>"

    transformed = HISTORY_TIGER_PATTERN.sub(replace, content)
    if observed != expected_count:
        raise TigerPidOrderDataError(
            f"history_length={expected_count}，实际解析到 {observed} 个历史 ID"
        )
    if "<POI_TIGER_ID>" in transformed or "</POI_TIGER_ID>" in transformed:
        raise TigerPidOrderDataError("源 Prompt 含无法解析的 TIGER history ID")
    return transformed


def _transform_sample(
    source: Mapping[str, Any],
    *,
    split: str,
    order: str,
    row_by_tiger_key: Mapping[int, int],
    tiger_capacities: Sequence[int],
    pid_lookup: PidLookup,
) -> dict[str, Any]:
    if source.get("split") != split:
        raise TigerPidOrderDataError("源样本 split 与文件不一致")
    messages = source.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(messages[0], dict)
        or not isinstance(messages[1], dict)
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise TigerPidOrderDataError("源 messages 必须严格为 user/assistant 两轮")
    history_length = source.get("history_length")
    if isinstance(history_length, bool) or not isinstance(history_length, int):
        raise TigerPidOrderDataError("源 history_length 非法")
    poi_id = source.get("target_poi_id")
    row = pid_lookup.row_by_poi_id.get(poi_id) if isinstance(poi_id, str) else None
    if row is None:
        raise TigerPidOrderDataError("目标 POI 不在 Final PID 映射")

    target_match = TARGET_TIGER_PATTERN.fullmatch(
        str(messages[1].get("content", ""))
    )
    if target_match is None:
        raise TigerPidOrderDataError("源 Assistant 不是合法 TIGER target")
    target_key = _parse_tiger_content(target_match.group(1), tiger_capacities)
    if row_by_tiger_key.get(target_key) != row:
        raise TigerPidOrderDataError("源 TIGER target 与 target_poi_id 不一致")

    target_content, target_key_value = _pid_for_row(pid_lookup, row, order)
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
                    "content": _replace_history_identifiers(
                        messages[0].get("content"),
                        expected_count=history_length,
                        order=order,
                        row_by_tiger_key=row_by_tiger_key,
                        tiger_capacities=tiger_capacities,
                        pid_lookup=pid_lookup,
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"<TARGET_POI>{target_content}</TARGET_POI>"
                    ),
                },
            ],
            "target_pid_key": target_key_value,
            "requires_dedup": bool(pid_lookup.requires_dedup[row]),
            "pid_order": order,
        }
    )
    return output


def _source_output_contract(
    source_dir: Path, source_manifest: Mapping[str, Any], split: str
) -> tuple[Path, int, str]:
    path = source_dir / f"{split}.jsonl"
    spec = source_manifest.get("outputs", {}).get(path.name)
    if not isinstance(spec, dict):
        raise TigerPidOrderDataError(f"源 manifest 缺少 {path.name}")
    rows = spec.get("rows")
    digest = spec.get("sha256")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
        or not isinstance(digest, str)
    ):
        raise TigerPidOrderDataError(f"源 {path.name} 契约非法")
    return path, rows, digest


def build_tiger_pid_order_sft_data(
    source_sft_dir: Path,
    tiger_id_dir: Path,
    pid_mapping_path: Path,
    pid_manifest_path: Path,
    output_dirs: Mapping[str, Path],
    *,
    max_rows_per_split: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> TigerPidOrderDataResult:
    """Transform one frozen TIGER dataset into two paired PID-order datasets."""

    if set(output_dirs) != set(PID_ORDERS):
        raise TigerPidOrderDataError("output_dirs 必须同时提供 gid_sid 与 sid_gid")
    if max_rows_per_split is not None and max_rows_per_split <= 0:
        raise TigerPidOrderDataError("max_rows_per_split 必须大于 0")
    source_sft_dir = source_sft_dir.resolve()
    tiger_id_dir = tiger_id_dir.resolve()
    resolved_outputs = {
        order: Path(path).resolve() for order, path in output_dirs.items()
    }
    if len(set(resolved_outputs.values())) != 2:
        raise TigerPidOrderDataError("两版输出目录不能相同")
    for path in resolved_outputs.values():
        if path.exists():
            raise TigerPidOrderDataError(f"输出目录已存在：{path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source_sft_dir / "manifest.json"
    source_manifest = _load_json_object(source_manifest_path, "源 TIGER SFT manifest")
    if source_manifest.get("schema_version") != "tiger-map-search-sft-data-v1":
        raise TigerPidOrderDataError("源数据必须是正式 TIGER history10 SFT v1")
    if source_manifest.get("status") != "completed":
        raise TigerPidOrderDataError("源 TIGER SFT manifest 状态不是 completed")

    try:
        pid_lookup, pid_manifest, pid_mapping_hash, pid_manifest_hash = (
            load_pid_lookup(pid_mapping_path, pid_manifest_path)
        )
    except SftDataError as error:
        raise TigerPidOrderDataError(str(error)) from error
    try:
        row_by_tiger_key, tiger_capacities, tiger_hashes = (
            _validate_and_index_identifiers(
                pid_lookup,
                tiger_id_dir,
                source_manifest,
                pid_manifest,
                Path(pid_manifest_path),
            )
        )
    except TigerDataError as error:
        raise TigerPidOrderDataError(str(error)) from error

    dedup_capacity = int(
        pid_manifest.get("dedup_assignment", {}).get("dedup_token_capacity", 0)
    )
    special_tokens = build_pid_order_special_tokens(
        sid_capacities=(1024, 1024, 1024),
        dedup_capacity=dedup_capacity,
        user_bucket_count=2000,
    )
    stats: dict[str, Any] = {
        "source_rows": 0,
        "train_count": 0,
        "valid_count": 0,
        "test_count": 0,
        "history_event_occurrence_count": 0,
        "requires_dedup_sample_count": 0,
        "singleton_sample_count": 0,
    }
    source_files: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    output_hashes: dict[str, dict[str, str]] = {}

    temporary_contexts = {
        order: tempfile.TemporaryDirectory(
            dir=path.parent, prefix=f".{path.name}.building-"
        )
        for order, path in resolved_outputs.items()
    }
    temporary_dirs = {
        order: Path(context.name) for order, context in temporary_contexts.items()
    }
    try:
        handles = {
            order: {
                split: (temporary_dirs[order] / f"{split}.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                )
                for split in ("train", "valid", "test")
            }
            for order in PID_ORDERS
        }
        try:
            for split in ("train", "valid", "test"):
                source_path, expected_rows, expected_hash = _source_output_contract(
                    source_sft_dir, source_manifest, split
                )
                digest = hashlib.sha256()
                rows = 0
                with source_path.open("rb") as stream:
                    for line_number, raw_line in enumerate(stream, start=1):
                        if (
                            max_rows_per_split is not None
                            and rows >= max_rows_per_split
                        ):
                            break
                        digest.update(raw_line)
                        rows += 1
                        try:
                            source = json.loads(raw_line)
                        except json.JSONDecodeError as error:
                            raise TigerPidOrderDataError(
                                f"{source_path.name}:{line_number} JSON 非法"
                            ) from error
                        if not isinstance(source, dict):
                            raise TigerPidOrderDataError("源 SFT 样本必须是 object")
                        transformed = {
                            order: _transform_sample(
                                source,
                                split=split,
                                order=order,
                                row_by_tiger_key=row_by_tiger_key,
                                tiger_capacities=tiger_capacities,
                                pid_lookup=pid_lookup,
                            )
                            for order in PID_ORDERS
                        }
                        fairness_fields = (
                            "sample_id",
                            "order_id",
                            "searchid",
                            "target_poi_id",
                            "history_length",
                            "split",
                            "requires_dedup",
                        )
                        if any(
                            transformed["gid_sid"].get(field)
                            != transformed["sid_gid"].get(field)
                            for field in fairness_fields
                        ):
                            raise TigerPidOrderDataError("两版样本公平字段不一致")
                        for order in PID_ORDERS:
                            handles[order][split].write(
                                json.dumps(
                                    transformed[order],
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                        stats["source_rows"] += 1
                        stats[f"{split}_count"] += 1
                        history_length = int(source["history_length"])
                        stats["history_event_occurrence_count"] += history_length
                        if transformed["gid_sid"]["requires_dedup"]:
                            stats["requires_dedup_sample_count"] += 1
                        else:
                            stats["singleton_sample_count"] += 1
                        if progress is not None and rows % 500_000 == 0:
                            progress(
                                f"{split} 成对转换：{rows:,}/"
                                f"{min(expected_rows, max_rows_per_split or expected_rows):,}"
                            )
                actual_hash = digest.hexdigest()
                if max_rows_per_split is None:
                    if rows != expected_rows or actual_hash != expected_hash:
                        raise TigerPidOrderDataError(
                            f"源 {source_path.name} 行数或 SHA256 与 manifest 不一致"
                        )
                elif rows != min(expected_rows, max_rows_per_split):
                    raise TigerPidOrderDataError("smoke 未读到预期前缀行数")
                source_files[split] = {
                    "path": str(source_path),
                    "manifest_rows": expected_rows,
                    "manifest_sha256": expected_hash,
                    "scanned_rows": rows,
                    "scanned_prefix_sha256": actual_hash,
                    "fully_scanned": max_rows_per_split is None,
                }
                if progress is not None:
                    progress(f"{split} 成对转换完成：{rows:,} 条")
        finally:
            for split_handles in handles.values():
                for handle in split_handles.values():
                    handle.close()

        if stats["source_rows"] != sum(
            stats[f"{split}_count"] for split in ("train", "valid", "test")
        ):
            raise TigerPidOrderDataError("成对输出样本数不守恒")
        if stats["source_rows"] != (
            stats["requires_dedup_sample_count"]
            + stats["singleton_sample_count"]
        ):
            raise TigerPidOrderDataError("Dedup 样本计数不守恒")

        source_manifest_hash = sha256_file(source_manifest_path)
        for order in PID_ORDERS:
            temporary_dir = temporary_dirs[order]
            _write_json(temporary_dir / "special_tokens.json", special_tokens)
            variant_stats = {
                **stats,
                "pid_order": order,
                "dedup_is_always_last": True,
            }
            _write_json(temporary_dir / "stats.json", variant_stats)
            artifacts = {
                name: {
                    "sha256": sha256_file(temporary_dir / name),
                    **(
                        {"rows": stats[f"{name[:-6]}_count"]}
                        if name.endswith(".jsonl")
                        else {}
                    ),
                }
                for name in OUTPUT_FILENAMES
                if name != "manifest.json"
            }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "task_name": "tiger_history10_query_gid_to_unique_geo_semantic_pid",
                "pid_order": order,
                "scan_mode": (
                    "full" if max_rows_per_split is None else "smoke_prefix"
                ),
                "max_rows_per_split": max_rows_per_split,
                "source_sft": {
                    "directory": str(source_sft_dir),
                    "schema_version": source_manifest["schema_version"],
                    "manifest_sha256": source_manifest_hash,
                    "files": source_files,
                },
                "tiger_identifier": {
                    **tiger_hashes,
                    "token_order": ["S1", "S2", "S3", "C"],
                    "use": "source_contract_and_history_reverse_lookup_only",
                },
                "final_pid_mapping": {
                    "mapping_path": str(Path(pid_mapping_path).resolve()),
                    "mapping_sha256": pid_mapping_hash,
                    "manifest_path": str(Path(pid_manifest_path).resolve()),
                    "manifest_sha256": pid_manifest_hash,
                    "poi_count": pid_lookup.poi_count,
                    "base_collision_key": ["G1", "G2", "G3", "G4", "G5", "G6", "S1", "S2", "S3"],
                    "dedup_scope": "within_each_gid6_sid3_bucket",
                    "dedup_token_capacity": dedup_capacity,
                },
                "sequence_contract": {
                    "base_order": (
                        ["G1", "G2", "G3", "G4", "G5", "G6", "S1", "S2", "S3"]
                        if order == "gid_sid"
                        else ["S1", "S2", "S3", "G1", "G2", "G3", "G4", "G5", "G6"]
                    ),
                    "dedup_position": "last_if_required",
                    "singleton_length": 9,
                    "colliding_length": 10,
                    "history_and_target_use_same_pid_order": True,
                    "request_gid_is_unchanged": True,
                    "history_wrapper": ["<POI_PID>", "</POI_PID>"],
                    "target_wrapper": ["<TARGET_POI>", "</TARGET_POI>"],
                },
                "fairness_contract": {
                    "paired_source_scan": True,
                    "same_samples_and_order": True,
                    "same_pid_mapping_and_dedup_codes": True,
                    "same_special_tokens_and_initial_model": True,
                    "only_poi_pid_gid_sid_order_differs": True,
                },
                "stats": variant_stats,
                "outputs": artifacts,
            }
            _write_json(temporary_dir / "manifest.json", manifest)
            manifests[order] = manifest
            output_hashes[order] = {
                name: sha256_file(temporary_dir / name)
                for name in OUTPUT_FILENAMES
            }

        for order in PID_ORDERS:
            os.replace(temporary_dirs[order], resolved_outputs[order])
            temporary_contexts[order]._finalizer.detach()
    finally:
        for context in temporary_contexts.values():
            context.cleanup()

    return TigerPidOrderDataResult(
        manifests=manifests,
        output_hashes=output_hashes,
        stats=stats,
    )

"""Contracts for frozen Beijing inputs and serialized POI identifiers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qg_prqk.config import DataContractConfig, FrozenInputs, IdentifierConfig


class QGPRQKDataContractError(ValueError):
    """Raised when an input record or manifest breaks the frozen contract."""


def load_json_object(path: Path, name: str) -> dict[str, Any]:
    """Load a JSON object with a contract-oriented error."""

    if not path.is_file():
        raise QGPRQKDataContractError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QGPRQKDataContractError(f"{name} JSON 非法：{path}") from error
    if not isinstance(value, dict):
        raise QGPRQKDataContractError(f"{name} 必须是 JSON object")
    return value


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QGPRQKDataContractError(f"{name} 必须是非空字符串")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QGPRQKDataContractError(f"{name} 必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise QGPRQKDataContractError(f"{name} 必须是有限数值")
    return result


def validate_poi_record(
    record: Mapping[str, Any], contract: DataContractConfig, source: str
) -> str:
    """Validate one real Beijing POI row and return its POI ID."""

    if not isinstance(record, Mapping):
        raise QGPRQKDataContractError(f"{source} 必须是 JSON object")
    missing = sorted(set(contract.poi_required_fields) - set(record))
    if missing:
        raise QGPRQKDataContractError(f"{source} 缺少 POI 字段：{missing}")
    unknown = sorted(set(record) - set(contract.poi_catalog_fields))
    if unknown:
        raise QGPRQKDataContractError(f"{source} 出现未登记 POI 字段：{unknown}")
    poi_id = _non_empty_string(record.get("poi_id"), f"{source}.poi_id")
    alias = record.get("alias")
    if alias is not None and not isinstance(alias, str):
        raise QGPRQKDataContractError(f"{source}.alias 必须是字符串或 null")
    for field in (
        "displayname",
        "category",
        "category_code",
        "address",
        "city",
        "source_dt",
        "text",
    ):
        if not isinstance(record.get(field), str):
            raise QGPRQKDataContractError(f"{source}.{field} 必须是字符串")
    lat = _finite_number(record.get("lat"), f"{source}.lat")
    lng = _finite_number(record.get("lng"), f"{source}.lng")
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lng <= 180.0:
        raise QGPRQKDataContractError(f"{source} 经纬度超出合法范围")
    return poi_id


def extract_train_query_target(
    record: Mapping[str, Any], source: str
) -> tuple[str, str]:
    """Extract raw Query and target POI from one canonical SFT Train row."""

    if record.get("split") != "train":
        raise QGPRQKDataContractError(f"{source} split 不是 train")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise QGPRQKDataContractError(
            f"{source} messages 必须恰好包含 user/assistant"
        )
    user, assistant = messages
    if not isinstance(user, Mapping) or user.get("role") != "user":
        raise QGPRQKDataContractError(f"{source} 第一条 message 不是 user")
    if not isinstance(assistant, Mapping) or assistant.get("role") != "assistant":
        raise QGPRQKDataContractError(f"{source} 第二条 message 不是 assistant")
    content = user.get("content")
    marker = "</QUERY>\n<USER_GID>"
    if (
        not isinstance(content, str)
        or not content.startswith("<QUERY>")
        or marker not in content
        or not content.endswith("</USER_GID>")
    ):
        raise QGPRQKDataContractError(f"{source} user content 结构非法")
    query, _ = content[len("<QUERY>") :].rsplit(marker, 1)
    if not query.strip():
        raise QGPRQKDataContractError(f"{source} Query 为空")
    target_poi_id = _non_empty_string(
        record.get("target_poi_id"), f"{source}.target_poi_id"
    )
    return query, target_poi_id


def validate_train_record(
    record: Mapping[str, Any], contract: DataContractConfig, source: str
) -> tuple[str, str]:
    """Validate one canonical Train row and return Query/target POI."""

    missing = sorted(set(contract.train_required_fields) - set(record))
    if missing:
        raise QGPRQKDataContractError(f"{source} 缺少 Train 字段：{missing}")
    for field in ("sample_id", "order_id", "searchid"):
        _non_empty_string(record.get(field), f"{source}.{field}")
    return extract_train_query_target(record, source)


def validate_embedding_manifest(
    manifest: Mapping[str, Any],
    contract: DataContractConfig,
    frozen: FrozenInputs,
) -> tuple[int, int, str]:
    """Validate the frozen BGE manifest without reading the full array."""

    if manifest.get("status") != "completed":
        raise QGPRQKDataContractError("Embedding manifest 状态不是 completed")
    input_spec = manifest.get("input")
    output_spec = manifest.get("output")
    if not isinstance(input_spec, Mapping) or not isinstance(output_spec, Mapping):
        raise QGPRQKDataContractError("Embedding manifest 缺少 input/output")
    shape = output_spec.get("shape")
    if shape != [contract.expected_poi_rows, contract.embedding_dim]:
        raise QGPRQKDataContractError("Embedding manifest shape 与冻结配置不一致")
    dtype = output_spec.get("dtype")
    if dtype != contract.embedding_dtype:
        raise QGPRQKDataContractError("Embedding manifest dtype 与冻结配置不一致")
    if input_spec.get("total_rows") != contract.expected_poi_rows:
        raise QGPRQKDataContractError("Embedding manifest POI 行数不一致")
    if input_spec.get("fingerprint") != frozen.embedding_fingerprint:
        raise QGPRQKDataContractError("Embedding fingerprint 与冻结配置不一致")
    return int(shape[0]), int(shape[1]), str(dtype)


def validate_sft_manifest(
    manifest: Mapping[str, Any],
    contract: DataContractConfig,
    frozen: FrozenInputs,
) -> dict[str, int]:
    """Validate the canonical temporal split and frozen file hashes."""

    if manifest.get("schema_version") != "sft-main-data-v1":
        raise QGPRQKDataContractError("SFT manifest schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise QGPRQKDataContractError("SFT manifest 状态不是 completed")
    split = manifest.get("time_split")
    outputs = manifest.get("outputs")
    if not isinstance(split, Mapping) or not isinstance(outputs, Mapping):
        raise QGPRQKDataContractError("SFT manifest 缺少 time_split/outputs")
    expected_split = {
        "field": "create_time",
        "source_dt_usage": "verification_only",
        "train": {"start": contract.train_date_start, "end": contract.train_date_end},
        "valid": contract.valid_date,
        "test": contract.test_date,
    }
    if dict(split) != expected_split:
        raise QGPRQKDataContractError("SFT 时间切分与冻结配置不一致")
    expected_hashes = {
        "train.jsonl": frozen.train_sha256,
        "valid.jsonl": frozen.valid_sha256,
        "test.jsonl": frozen.test_sha256,
    }
    rows: dict[str, int] = {}
    for filename, expected_hash in expected_hashes.items():
        spec = outputs.get(filename)
        if not isinstance(spec, Mapping):
            raise QGPRQKDataContractError(f"SFT manifest 缺少 {filename}")
        if spec.get("sha256") != expected_hash:
            raise QGPRQKDataContractError(f"{filename} SHA256 与冻结配置不一致")
        value = spec.get("rows")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise QGPRQKDataContractError(f"{filename} 行数非法")
        rows[filename] = value
    return rows


def validate_aligned_row_counts(
    *, poi_rows: int, embedding_rows: int, poi_id_rows: int
) -> None:
    """Require POI, embedding, and POI-ID row counts to be identical."""

    if poi_rows <= 0 or embedding_rows <= 0 or poi_id_rows <= 0:
        raise QGPRQKDataContractError("对齐行数必须为正整数")
    if len({poi_rows, embedding_rows, poi_id_rows}) != 1:
        raise QGPRQKDataContractError(
            "POI、Embedding 和 POI ID 行数不一致："
            f"{poi_rows}/{embedding_rows}/{poi_id_rows}"
        )


def validate_final_pid(
    tokens: Sequence[int], requires_dedup: bool, contract: IdentifierConfig
) -> tuple[int, ...]:
    """Validate one GID6+SID3+[D] identifier."""

    if isinstance(tokens, (str, bytes)):
        raise QGPRQKDataContractError("Final PID 必须是整数序列")
    values = tuple(tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise QGPRQKDataContractError("Final PID 只能包含整数")
    expected_length = (
        contract.collision_length if requires_dedup else contract.singleton_length
    )
    if len(values) != expected_length:
        raise QGPRQKDataContractError(
            f"Final PID 长度必须为 {expected_length}，实际为 {len(values)}"
        )
    if any(not 0 <= value < contract.gid_codebook_size for value in values[:6]):
        raise QGPRQKDataContractError("GID code 必须位于 [0,31]")
    for level, (value, size) in enumerate(
        zip(values[6:9], contract.sid_codebook_sizes, strict=True), start=1
    ):
        if not 0 <= value < size:
            raise QGPRQKDataContractError(
                f"S{level} code 必须位于 [0,{size - 1}]"
            )
    if requires_dedup and not 0 <= values[9] < contract.dedup_capacity:
        raise QGPRQKDataContractError(
            f"Dedup code 必须位于 [0,{contract.dedup_capacity - 1}]"
        )
    return values

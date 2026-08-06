"""Paper-aligned unconstrained beam-search evaluation for TIGER identifiers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..generative_eval import GenerativeEvalError, encode_prompt_like_training
from ..pid_trie import sha256_file


TARGET_PATTERN_PARTS = ("target_open", "s1", "s2", "s3", "collision", "target_close")
METRIC_KS = (1, 3, 5, 10)


class TigerEvalError(ValueError):
    """Raised when TIGER evaluation violates its fixed identifier protocol."""


@dataclass(frozen=True)
class TigerTokenIds:
    target_open: int
    target_close: int
    s1: tuple[int, ...]
    s2: tuple[int, ...]
    s3: tuple[int, ...]
    collision: tuple[int, ...]
    eos: int

    @property
    def expected_sequence_length(self) -> int:
        return 7


@dataclass(frozen=True)
class TigerExample:
    sample_id: str
    target_poi_id: str
    target_codes: tuple[int, int, int, int]
    prompt_ids: tuple[int, ...]


@dataclass(frozen=True)
class TigerCandidate:
    codes: tuple[int, int, int, int] | None
    poi_row: int | None
    score: float
    error: str | None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_tiger_token_ids(
    tokenizer_path: Path,
    tokenizer: Any,
) -> tuple[TigerTokenIds, dict[str, Any]]:
    """Validate TIGER tokens and return stable position-specific ID tables."""

    mapping_path = tokenizer_path / "tiger_token_mapping.json"
    if not mapping_path.is_file():
        raise TigerEvalError(f"TIGER Token mapping 不存在：{mapping_path}")
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerEvalError("TIGER Token mapping JSON 非法") from error
    token_mapping = mapping.get("tokens")
    if not isinstance(token_mapping, Mapping):
        raise TigerEvalError("TIGER Token mapping 缺少 tokens")
    if len(tokenizer) != mapping.get("new_vocab_size"):
        raise TigerEvalError("TIGER Tokenizer 词表大小与 mapping 不一致")

    def token_id(token: str) -> int:
        expected = token_mapping.get(token)
        actual = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if not isinstance(expected, int) or actual != expected or encoded != [expected]:
            raise TigerEvalError(f"TIGER Token 不是稳定原子 Token：{token}")
        return expected

    values = TigerTokenIds(
        target_open=token_id("<TARGET_POI>"),
        target_close=token_id("</TARGET_POI>"),
        s1=tuple(token_id(f"<S1_{code}>") for code in range(1024)),
        s2=tuple(token_id(f"<S2_{code}>") for code in range(1024)),
        s3=tuple(token_id(f"<S3_{code}>") for code in range(1024)),
        collision=tuple(token_id(f"<C_{code}>") for code in range(306)),
        eos=int(tokenizer.eos_token_id),
    )
    flattened = (
        values.target_open,
        values.target_close,
        *values.s1,
        *values.s2,
        *values.s3,
        *values.collision,
    )
    if len(set(flattened)) != len(flattened):
        raise TigerEvalError("TIGER 输出 Token ID 必须互不重复")
    for name, token_range in (
        ("S1", values.s1),
        ("S2", values.s2),
        ("S3", values.s3),
        ("Collision", values.collision),
    ):
        if any(right != left + 1 for left, right in zip(token_range, token_range[1:])):
            raise TigerEvalError(f"TIGER {name} Token ID 必须是连续区间")
    return values, {
        "mapping_path": str(mapping_path.resolve()),
        "mapping_sha256": sha256_file(mapping_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "vocab_size": len(tokenizer),
    }


class TigerIdIndex:
    """Compact exact lookup from a four-code TIGER identifier to POI row."""

    def __init__(
        self,
        *,
        sorted_keys: np.ndarray,
        sorted_rows: np.ndarray,
        poi_ids: Any,
    ) -> None:
        self.sorted_keys = sorted_keys
        self.sorted_rows = sorted_rows
        self.poi_ids = poi_ids
        self.row_count = len(sorted_rows)

    @staticmethod
    def pack(codes: Sequence[int]) -> int:
        if len(codes) != 4:
            return -1
        s1, s2, s3, collision = (int(value) for value in codes)
        if not (0 <= s1 < 1024 and 0 <= s2 < 1024 and 0 <= s3 < 1024):
            return -1
        if not 0 <= collision < 306:
            return -1
        return (((s1 * 1024) + s2) * 1024 + s3) * 306 + collision

    def lookup(self, codes: Sequence[int]) -> int:
        key = self.pack(codes)
        if key < 0:
            return -1
        index = int(np.searchsorted(self.sorted_keys, key))
        if index >= self.row_count or int(self.sorted_keys[index]) != key:
            return -1
        return int(self.sorted_rows[index])

    def poi_id(self, row: int) -> str:
        if row < 0 or row >= self.row_count:
            raise TigerEvalError(f"POI row 越界：{row}")
        return str(self.poi_ids[row].as_py())


def load_tiger_id_index(identifier_dir: Path) -> tuple[TigerIdIndex, dict[str, Any]]:
    """Load and verify the frozen TIGER identifier lookup table."""

    identifier_dir = identifier_dir.resolve()
    manifest_path = identifier_dir / "tiger_id_manifest.json"
    if not manifest_path.is_file():
        raise TigerEvalError(f"TIGER identifier manifest 不存在：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerEvalError("TIGER identifier manifest JSON 非法") from error
    if manifest.get("schema_version") != "tiger-item-identifier-v1":
        raise TigerEvalError("TIGER identifier schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise TigerEvalError("TIGER identifier 状态不是 completed")
    if manifest.get("tiger_ids", {}).get("fixed_length") != 4:
        raise TigerEvalError("TIGER identifier 必须固定为四层")
    if manifest.get("tiger_ids", {}).get("token_capacities") != [1024, 1024, 1024, 306]:
        raise TigerEvalError("TIGER identifier 容量与当前评测协议不一致")
    mapping_path = identifier_dir / manifest["mapping"]["path"]
    ids_path = identifier_dir / manifest["tiger_ids"]["path"]
    for path, expected_hash, name in (
        (mapping_path, manifest["mapping"]["sha256"], "mapping"),
        (ids_path, manifest["tiger_ids"]["sha256"], "tiger_ids"),
    ):
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise TigerEvalError(f"TIGER {name} 文件或 SHA256 不一致")
    codes = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    expected_rows = int(manifest["mapping"]["rows"])
    if codes.shape != (expected_rows, 4) or codes.dtype != np.int32:
        raise TigerEvalError("TIGER identifier NPY shape/dtype 不一致")
    keys = (
        (
            (codes[:, 0].astype(np.int64) * 1024 + codes[:, 1]) * 1024
            + codes[:, 2]
        )
        * 306
        + codes[:, 3]
    )
    order = np.argsort(keys, kind="stable")
    sorted_keys = np.asarray(keys[order], dtype=np.int64)
    if np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise TigerEvalError("TIGER identifier key 必须全局唯一")
    mapping = pq.read_table(mapping_path, columns=["poi_id"])
    poi_ids = mapping["poi_id"]
    if len(poi_ids) != expected_rows or poi_ids.null_count:
        raise TigerEvalError("TIGER POI mapping 行数或空值非法")
    if pc.count_distinct(poi_ids).as_py() != expected_rows:
        raise TigerEvalError("TIGER POI mapping 的 poi_id 必须唯一")
    return (
        TigerIdIndex(
            sorted_keys=sorted_keys,
            sorted_rows=np.asarray(order, dtype=np.int64),
            poi_ids=poi_ids,
        ),
        {
            "identifier_manifest": str(manifest_path),
            "identifier_manifest_sha256": sha256_file(manifest_path),
            "mapping": str(mapping_path),
            "mapping_sha256": manifest["mapping"]["sha256"],
            "rows": expected_rows,
        },
    )


def parse_target_codes(content: str) -> tuple[int, int, int, int]:
    """Parse the exact TIGER assistant target serialization."""

    import re

    match = re.fullmatch(
        r"<TARGET_POI><S1_(\d+)><S2_(\d+)><S3_(\d+)><C_(\d+)></TARGET_POI>",
        content,
    )
    if match is None:
        raise TigerEvalError("Assistant content 不是严格 TIGER Target 格式")
    codes = tuple(int(value) for value in match.groups())
    if TigerIdIndex.pack(codes) < 0:
        raise TigerEvalError("Assistant TIGER identifier 超出码本范围")
    return codes  # type: ignore[return-value]


def encode_tiger_record(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    cutoff_len: int,
    split: str = "valid",
) -> TigerExample:
    """Validate one TIGER SFT sample and recreate its training prompt."""

    messages = record.get("messages")
    if (
        record.get("split") != split
        or not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise TigerEvalError("TIGER 评测样本必须是指定 split 的 user+assistant")
    user_content = messages[0].get("content")
    target_content = messages[1].get("content")
    if not isinstance(user_content, str) or not isinstance(target_content, str):
        raise TigerEvalError("TIGER Messages content 必须是字符串")
    target_codes = parse_target_codes(target_content)
    try:
        prompt_ids, target_ids = encode_prompt_like_training(
            tokenizer=tokenizer,
            template=template,
            user_content=user_content,
            target_content=target_content,
            cutoff_len=cutoff_len,
        )
    except GenerativeEvalError as error:
        raise TigerEvalError(str(error)) from error
    if len(target_ids) != 6:
        raise TigerEvalError("TIGER Assistant Target 必须编码为 6 个原子 Token")
    if target_content in tokenizer.decode(prompt_ids, skip_special_tokens=False):
        raise TigerEvalError("TIGER Prompt 泄露目标 identifier")
    sample_id = record.get("sample_id")
    target_poi_id = record.get("target_poi_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise TigerEvalError("sample_id 必须是非空字符串")
    if not isinstance(target_poi_id, str) or not target_poi_id:
        raise TigerEvalError("target_poi_id 必须是非空字符串")
    return TigerExample(
        sample_id=sample_id,
        target_poi_id=target_poi_id,
        target_codes=target_codes,
        prompt_ids=tuple(int(value) for value in prompt_ids),
    )


def parse_generated_candidate(
    sequence: Sequence[int],
    score: float,
    *,
    tokens: TigerTokenIds,
    index: TigerIdIndex,
) -> TigerCandidate:
    """Parse one unconstrained beam and keep invalid IDs as ranked slots."""

    values = [int(value) for value in sequence]
    if tokens.eos not in values:
        return TigerCandidate(None, None, float(score), "missing_eos")
    eos_index = values.index(tokens.eos)
    generated = values[: eos_index + 1]
    if len(generated) != tokens.expected_sequence_length:
        return TigerCandidate(None, None, float(score), "invalid_length")
    if generated[0] != tokens.target_open or generated[5] != tokens.target_close:
        return TigerCandidate(None, None, float(score), "invalid_structure")
    token_ranges = (tokens.s1, tokens.s2, tokens.s3, tokens.collision)
    codes = tuple(
        generated[position + 1] - token_range[0]
        for position, token_range in enumerate(token_ranges)
    )
    if any(
        code < 0 or code >= len(token_range)
        for code, token_range in zip(codes, token_ranges)
    ):
        return TigerCandidate(None, None, float(score), "invalid_position_token")
    row = index.lookup(codes)
    if row < 0:
        return TigerCandidate(codes, None, float(score), "identifier_not_in_corpus")
    return TigerCandidate(codes, row, float(score), None)  # type: ignore[arg-type]


def empty_metrics() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "hit_sums": {str(k): 0 for k in METRIC_KS},
        "ndcg_sums": {str(k): 0.0 for k in METRIC_KS},
        "candidate_count": 0,
        "valid_candidate_count": 0,
        "invalid_error_counts": {},
        "samples_with_invalid_candidate": 0,
        "target_rank_histogram": {},
    }


def update_metrics(
    metrics: dict[str, Any],
    *,
    target_row: int,
    candidates: Sequence[TigerCandidate],
) -> int | None:
    """Update Recall/NDCG while invalid beams retain their original ranks."""

    target_rank = next(
        (
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if candidate.poi_row == target_row
        ),
        None,
    )
    metrics["sample_count"] += 1
    metrics["candidate_count"] += len(candidates)
    invalid = [candidate for candidate in candidates if candidate.error is not None]
    metrics["valid_candidate_count"] += len(candidates) - len(invalid)
    metrics["samples_with_invalid_candidate"] += int(bool(invalid))
    for candidate in invalid:
        counts = metrics["invalid_error_counts"]
        counts[candidate.error] = counts.get(candidate.error, 0) + 1
    rank_key = str(target_rank) if target_rank is not None else "miss"
    histogram = metrics["target_rank_histogram"]
    histogram[rank_key] = histogram.get(rank_key, 0) + 1
    for k in METRIC_KS:
        hit = int(target_rank is not None and target_rank <= k)
        metrics["hit_sums"][str(k)] += hit
        if hit:
            metrics["ndcg_sums"][str(k)] += 1.0 / math.log2(target_rank + 1)
    return target_rank


def merge_metrics(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key in (
        "sample_count",
        "candidate_count",
        "valid_candidate_count",
        "samples_with_invalid_candidate",
    ):
        target[key] += int(source[key])
    for name in ("hit_sums", "ndcg_sums", "invalid_error_counts", "target_rank_histogram"):
        for key, value in source[name].items():
            target[name][key] = target[name].get(key, 0) + value


def finalize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    samples = int(metrics["sample_count"])
    candidates = int(metrics["candidate_count"])
    if samples <= 0 or candidates <= 0:
        raise TigerEvalError("TIGER 评测指标不能为空")
    result: dict[str, Any] = {
        "sample_count": samples,
        "invalid_id_rate": 1 - int(metrics["valid_candidate_count"]) / candidates,
        "valid_id_rate": int(metrics["valid_candidate_count"]) / candidates,
        "samples_with_invalid_id_rate": int(metrics["samples_with_invalid_candidate"]) / samples,
        "invalid_error_counts": metrics["invalid_error_counts"],
        "target_rank_histogram": metrics["target_rank_histogram"],
    }
    for k in METRIC_KS:
        result[f"hr@{k}"] = metrics["hit_sums"][str(k)] / samples
        result[f"ndcg@{k}"] = metrics["ndcg_sums"][str(k)] / samples
    return result

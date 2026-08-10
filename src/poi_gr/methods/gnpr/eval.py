"""Unconstrained generative evaluation for conditional-length GNPR IDs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ...pid.trie import sha256_file
from ...sft.evaluation import GenerativeEvalError, encode_prompt_like_training


METRIC_KS = (1, 3, 5, 10)


class GnprEvalError(ValueError):
    """Raised when GNPR evaluation violates its frozen protocol."""


@dataclass(frozen=True)
class GnprTokenIds:
    target_open: int
    target_close: int
    a: tuple[int, ...]
    b: tuple[int, ...]
    c: tuple[int, ...]
    dedup: tuple[int, ...]
    eos: int

    @property
    def max_sequence_length(self) -> int:
        return 7


@dataclass(frozen=True)
class GnprExample:
    sample_id: str
    target_poi_id: str
    target_codes: tuple[int, int, int, int]
    prompt_ids: tuple[int, ...]


@dataclass(frozen=True)
class GnprCandidate:
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


def load_gnpr_token_ids(
    tokenizer_path: Path,
    tokenizer: Any,
) -> tuple[GnprTokenIds, dict[str, Any]]:
    """Validate GNPR tokens and return position-specific token ID tables."""

    mapping_path = tokenizer_path / "poi_token_mapping.json"
    if not mapping_path.is_file():
        raise GnprEvalError(f"GNPR Token mapping 不存在：{mapping_path}")
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GnprEvalError("GNPR Token mapping JSON 非法") from error
    if mapping.get("schema_version") != "gnpr-vocab-v1":
        raise GnprEvalError("GNPR Token mapping schema_version 不兼容")
    token_mapping = mapping.get("tokens")
    if not isinstance(token_mapping, Mapping):
        raise GnprEvalError("GNPR Token mapping 缺少 tokens")
    if len(tokenizer) != mapping.get("new_vocab_size"):
        raise GnprEvalError("GNPR Tokenizer 词表大小与 mapping 不一致")

    def token_id(token: str) -> int:
        expected = token_mapping.get(token)
        actual = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if not isinstance(expected, int) or actual != expected or encoded != [expected]:
            raise GnprEvalError(f"GNPR Token 不是稳定原子 Token：{token}")
        return expected

    values = GnprTokenIds(
        target_open=token_id("<TARGET_POI>"),
        target_close=token_id("</TARGET_POI>"),
        a=tuple(token_id(f"<a_{code}>") for code in range(512)),
        b=tuple(token_id(f"<b_{code}>") for code in range(512)),
        c=tuple(token_id(f"<c_{code}>") for code in range(512)),
        dedup=tuple(token_id(f"<d_{code}>") for code in range(223)),
        eos=int(tokenizer.eos_token_id),
    )
    flattened = (
        values.target_open,
        values.target_close,
        *values.a,
        *values.b,
        *values.c,
        *values.dedup,
    )
    if len(set(flattened)) != len(flattened):
        raise GnprEvalError("GNPR 输出 Token ID 必须互不重复")
    for name, token_range in (
        ("a", values.a),
        ("b", values.b),
        ("c", values.c),
        ("d", values.dedup),
    ):
        if any(right != left + 1 for left, right in zip(token_range, token_range[1:])):
            raise GnprEvalError(f"GNPR {name} Token ID 必须是连续区间")
    return values, {
        "mapping_path": str(mapping_path.resolve()),
        "mapping_sha256": sha256_file(mapping_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "vocab_size": len(tokenizer),
    }


class GnprIdIndex:
    """Compact exact lookup for three-token singleton and four-token GNPR IDs."""

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
        a, b, c, dedup = (int(value) for value in codes)
        if not (0 <= a < 512 and 0 <= b < 512 and 0 <= c < 512):
            return -1
        if not -1 <= dedup < 223:
            return -1
        return (((a * 512) + b) * 512 + c) * 224 + dedup + 1

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
            raise GnprEvalError(f"POI row 越界：{row}")
        return str(self.poi_ids[row].as_py())


def load_gnpr_id_index(identifier_dir: Path) -> tuple[GnprIdIndex, dict[str, Any]]:
    """Load and verify the frozen GNPR identifier lookup table."""

    identifier_dir = identifier_dir.resolve()
    manifest_path = identifier_dir / "gnpr_id_manifest.json"
    if not manifest_path.is_file():
        raise GnprEvalError(f"GNPR identifier manifest 不存在：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GnprEvalError("GNPR identifier manifest JSON 非法") from error
    if manifest.get("schema_version") != "gnpr-poi-identifier-v1":
        raise GnprEvalError("GNPR identifier schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise GnprEvalError("GNPR identifier 状态不是 completed")
    details = manifest.get("gnpr_ids", {})
    if details.get("codebook_capacities") != [512, 512, 512]:
        raise GnprEvalError("GNPR 三层码本容量与评测协议不一致")
    if details.get("dedup_token_capacity") != 223:
        raise GnprEvalError("GNPR Dedup Token 容量与评测协议不一致")
    mapping_path = identifier_dir / manifest["mapping"]["path"]
    ids_path = identifier_dir / details["path"]
    for path, expected_hash, name in (
        (mapping_path, manifest["mapping"]["sha256"], "mapping"),
        (ids_path, details["sha256"], "gnpr_ids"),
    ):
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise GnprEvalError(f"GNPR {name} 文件或 SHA256 不一致")
    codes = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    expected_rows = int(manifest["mapping"]["rows"])
    if codes.shape != (expected_rows, 4) or codes.dtype != np.int32:
        raise GnprEvalError("GNPR identifier NPY shape/dtype 不一致")
    keys = (
        ((codes[:, 0].astype(np.int64) * 512 + codes[:, 1]) * 512 + codes[:, 2])
        * 224
        + codes[:, 3]
        + 1
    )
    order = np.argsort(keys, kind="stable")
    sorted_keys = np.asarray(keys[order], dtype=np.int64)
    if np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise GnprEvalError("GNPR identifier key 必须全局唯一")
    mapping = pq.read_table(mapping_path, columns=["poi_id"])
    poi_ids = mapping["poi_id"]
    if len(poi_ids) != expected_rows or poi_ids.null_count:
        raise GnprEvalError("GNPR POI mapping 行数或空值非法")
    if pc.count_distinct(poi_ids).as_py() != expected_rows:
        raise GnprEvalError("GNPR POI mapping 的 poi_id 必须唯一")
    return (
        GnprIdIndex(
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
            "singleton_length": 3,
            "collision_length": 4,
        },
    )


def parse_target_codes(content: str) -> tuple[int, int, int, int]:
    """Parse the exact conditional-length GNPR target serialization."""

    match = re.fullmatch(
        r"<TARGET_POI><a_(\d+)><b_(\d+)><c_(\d+)>(?:<d_(\d+)>)?</TARGET_POI>",
        content,
    )
    if match is None:
        raise GnprEvalError("Assistant content 不是严格 GNPR Target 格式")
    a, b, c, dedup = match.groups()
    codes = (int(a), int(b), int(c), -1 if dedup is None else int(dedup))
    if GnprIdIndex.pack(codes) < 0:
        raise GnprEvalError("Assistant GNPR identifier 超出码本范围")
    return codes


def encode_gnpr_record(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    cutoff_len: int,
    split: str = "valid",
) -> GnprExample:
    """Validate one GNPR SFT sample and recreate its training prompt."""

    messages = record.get("messages")
    if (
        record.get("split") != split
        or not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise GnprEvalError("GNPR 评测样本必须是指定 split 的 user+assistant")
    user_content = messages[0].get("content")
    target_content = messages[1].get("content")
    if not isinstance(user_content, str) or not isinstance(target_content, str):
        raise GnprEvalError("GNPR Messages content 必须是字符串")
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
        raise GnprEvalError(str(error)) from error
    expected_target_tokens = 5 if target_codes[3] < 0 else 6
    if len(target_ids) != expected_target_tokens:
        raise GnprEvalError("GNPR Assistant Target 原子 Token 数量不一致")
    if target_content in tokenizer.decode(prompt_ids, skip_special_tokens=False):
        raise GnprEvalError("GNPR Prompt 泄露目标 identifier")
    sample_id = record.get("sample_id")
    target_poi_id = record.get("target_poi_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise GnprEvalError("sample_id 必须是非空字符串")
    if not isinstance(target_poi_id, str) or not target_poi_id:
        raise GnprEvalError("target_poi_id 必须是非空字符串")
    return GnprExample(
        sample_id=sample_id,
        target_poi_id=target_poi_id,
        target_codes=target_codes,
        prompt_ids=tuple(int(value) for value in prompt_ids),
    )


def parse_generated_candidate(
    sequence: Sequence[int],
    score: float,
    *,
    tokens: GnprTokenIds,
    index: GnprIdIndex,
) -> GnprCandidate:
    """Parse an unconstrained beam while preserving invalid rank slots."""

    values = [int(value) for value in sequence]
    if tokens.eos not in values:
        return GnprCandidate(None, None, float(score), "missing_eos")
    generated = values[: values.index(tokens.eos) + 1]
    if len(generated) == 6:
        if generated[0] != tokens.target_open or generated[4] != tokens.target_close:
            return GnprCandidate(None, None, float(score), "invalid_structure")
        token_ranges = (tokens.a, tokens.b, tokens.c)
        position_ids = generated[1:4]
        dedup = -1
    elif len(generated) == 7:
        if generated[0] != tokens.target_open or generated[5] != tokens.target_close:
            return GnprCandidate(None, None, float(score), "invalid_structure")
        token_ranges = (tokens.a, tokens.b, tokens.c, tokens.dedup)
        position_ids = generated[1:5]
        dedup = None
    else:
        return GnprCandidate(None, None, float(score), "invalid_length")
    parsed = tuple(
        token_id - token_range[0]
        for token_id, token_range in zip(position_ids, token_ranges)
    )
    if any(
        code < 0 or code >= len(token_range)
        for code, token_range in zip(parsed, token_ranges)
    ):
        return GnprCandidate(None, None, float(score), "invalid_position_token")
    codes = (
        int(parsed[0]),
        int(parsed[1]),
        int(parsed[2]),
        int(dedup if dedup is not None else parsed[3]),
    )
    row = index.lookup(codes)
    if row < 0:
        return GnprCandidate(codes, None, float(score), "identifier_not_in_corpus")
    return GnprCandidate(codes, row, float(score), None)


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
    candidates: Sequence[GnprCandidate],
) -> int | None:
    """Update HR/NDCG while invalid beams retain their original ranks."""

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
        raise GnprEvalError("GNPR 评测指标不能为空")
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

"""Beam-search evaluation utilities for TIGER-style identifiers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ...sft.evaluation import GenerativeEvalError, encode_prompt_like_training
from ...pid.trie import sha256_file


TARGET_PATTERN_PARTS = ("target_open", "s1", "s2", "s3", "collision", "target_close")
METRIC_KS = (1, 3, 5, 10)
DEFAULT_TOKEN_CAPACITIES = (1024, 1024, 1024, 306)


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
    base_bucket: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class TigerBucketRanking:
    """Rank one target among generated expandable three-code buckets."""

    raw_slot_target_rank: int | None
    unique_target_rank: int | None
    unique_buckets: tuple[tuple[int, int, int], ...]
    unique_bucket_first_beam_ranks: tuple[int, ...]
    unique_bucket_sizes: tuple[int, ...]
    duplicate_bucket_candidates: int
    non_expandable_candidates: int


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
    *,
    token_capacities: Sequence[int] = DEFAULT_TOKEN_CAPACITIES,
    mapping_filename: str = "tiger_token_mapping.json",
) -> tuple[TigerTokenIds, dict[str, Any]]:
    """Validate TIGER tokens and return stable position-specific ID tables."""

    capacities = _validate_token_capacities(token_capacities)
    mapping_path = tokenizer_path / mapping_filename
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
        s1=tuple(token_id(f"<S1_{code}>") for code in range(capacities[0])),
        s2=tuple(token_id(f"<S2_{code}>") for code in range(capacities[1])),
        s3=tuple(token_id(f"<S3_{code}>") for code in range(capacities[2])),
        collision=tuple(token_id(f"<C_{code}>") for code in range(capacities[3])),
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
        "token_capacities": list(capacities),
    }


def _validate_token_capacities(values: Sequence[int]) -> tuple[int, int, int, int]:
    capacities = tuple(int(value) for value in values)
    if len(capacities) != 4 or any(value <= 0 for value in capacities):
        raise TigerEvalError("TIGER token capacities 必须包含四个正整数")
    return capacities  # type: ignore[return-value]


class TigerIdIndex:
    """Compact exact lookup from a four-code TIGER identifier to POI row."""

    def __init__(
        self,
        *,
        sorted_keys: np.ndarray,
        sorted_rows: np.ndarray,
        poi_ids: Any,
        token_capacities: Sequence[int] = DEFAULT_TOKEN_CAPACITIES,
    ) -> None:
        self.sorted_keys = sorted_keys
        self.sorted_rows = sorted_rows
        self.poi_ids = poi_ids
        self.row_count = len(sorted_rows)
        self.token_capacities = _validate_token_capacities(token_capacities)

    @staticmethod
    def pack(
        codes: Sequence[int],
        token_capacities: Sequence[int] = DEFAULT_TOKEN_CAPACITIES,
    ) -> int:
        if len(codes) != 4:
            return -1
        capacities = _validate_token_capacities(token_capacities)
        s1, s2, s3, collision = (int(value) for value in codes)
        if not all(
            0 <= code < capacity
            for code, capacity in zip((s1, s2, s3, collision), capacities)
        ):
            return -1
        return (((s1 * capacities[1]) + s2) * capacities[2] + s3) * capacities[
            3
        ] + collision

    def lookup(self, codes: Sequence[int]) -> int:
        key = self.pack(codes, self.token_capacities)
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

    def bucket_bounds(self, codes: Sequence[int]) -> tuple[int, int]:
        """Return sorted-key bounds for one three-code semantic bucket."""

        if len(codes) != 3:
            return 0, 0
        capacities = self.token_capacities
        s1, s2, s3 = (int(value) for value in codes)
        if not all(
            0 <= code < capacity for code, capacity in zip((s1, s2, s3), capacities[:3])
        ):
            return 0, 0
        prefix_key = (s1 * capacities[1] + s2) * capacities[2] + s3
        lower = prefix_key * capacities[3]
        upper = lower + capacities[3]
        start = int(np.searchsorted(self.sorted_keys, lower, side="left"))
        stop = int(np.searchsorted(self.sorted_keys, upper, side="left"))
        return start, stop

    def bucket_size(self, codes: Sequence[int]) -> int:
        """Return the number of corpus POIs sharing a three-code prefix."""

        start, stop = self.bucket_bounds(codes)
        return stop - start


class TigerLegalPathConstraint:
    """Restrict generation to prefixes of identifiers present in the corpus."""

    def __init__(
        self,
        *,
        index: TigerIdIndex,
        tokens: TigerTokenIds,
        prompt_width: int,
        next_token_cache: dict[tuple[int, ...], list[int]] | None = None,
    ) -> None:
        if prompt_width <= 0:
            raise TigerEvalError("prompt_width 必须为正整数")
        if tuple(len(values) for values in self._token_ranges(tokens)) != (
            index.token_capacities
        ):
            raise TigerEvalError("合法路径约束的 Token 容量与 identifier 不一致")
        self.index = index
        self.tokens = tokens
        self.prompt_width = prompt_width
        self._next_token_cache = (
            next_token_cache if next_token_cache is not None else {}
        )

    @staticmethod
    def _token_ranges(tokens: TigerTokenIds) -> tuple[tuple[int, ...], ...]:
        return tokens.s1, tokens.s2, tokens.s3, tokens.collision

    def _decode_prefix_codes(self, generated: Sequence[int]) -> tuple[int, ...]:
        if not generated or int(generated[0]) != self.tokens.target_open:
            raise TigerEvalError("合法路径生成必须以 <TARGET_POI> 开始")
        code_tokens = generated[1:]
        if len(code_tokens) > 4:
            raise TigerEvalError("合法路径 SID Prefix 超过四层")
        codes: list[int] = []
        for token_id, token_range in zip(
            code_tokens,
            self._token_ranges(self.tokens),
        ):
            code = int(token_id) - token_range[0]
            if code < 0 or code >= len(token_range):
                raise TigerEvalError("生成序列离开合法的分层 Token 位置")
            codes.append(code)
        return tuple(codes)

    def _allowed_code_tokens(self, prefix_codes: tuple[int, ...]) -> list[int]:
        cached = self._next_token_cache.get(prefix_codes)
        if cached is not None:
            return cached
        capacities = self.index.token_capacities
        depth = len(prefix_codes)
        if depth >= len(capacities):
            raise TigerEvalError("完整 identifier 不再包含下一级 SID Token")

        prefix_value = 0
        for code, capacity in zip(prefix_codes, capacities):
            prefix_value = prefix_value * capacity + code
        suffix_size = math.prod(capacities[depth:])
        lower = prefix_value * suffix_size
        upper = lower + suffix_size
        start = int(np.searchsorted(self.index.sorted_keys, lower, side="left"))
        stop = int(np.searchsorted(self.index.sorted_keys, upper, side="left"))
        if start >= stop:
            raise TigerEvalError(f"identifier Prefix 不在冻结语料库中：{prefix_codes}")

        divisor = math.prod(capacities[depth + 1 :])
        next_codes = (self.index.sorted_keys[start:stop] // divisor) % capacities[depth]
        distinct = next_codes[
            np.concatenate((np.asarray([True]), next_codes[1:] != next_codes[:-1]))
        ]
        token_range = self._token_ranges(self.tokens)[depth]
        allowed = [token_range[int(code)] for code in distinct]
        self._next_token_cache[prefix_codes] = allowed
        return allowed

    def allowed_next(self, generated: Sequence[int]) -> list[int]:
        """Return the exact legal next-token set for one generated prefix."""

        values = tuple(int(value) for value in generated)
        if not values:
            return [self.tokens.target_open]
        if len(values) <= 4:
            return self._allowed_code_tokens(self._decode_prefix_codes(values))
        if len(values) == 5:
            codes = self._decode_prefix_codes(values)
            if self.index.lookup(codes) < 0:
                raise TigerEvalError(f"完整 identifier 不在冻结语料库中：{codes}")
            return [self.tokens.target_close]
        if len(values) == 6:
            if int(values[-1]) != self.tokens.target_close:
                raise TigerEvalError("完整 identifier 后必须生成 </TARGET_POI>")
            self._decode_prefix_codes(values[:-1])
            return [self.tokens.eos]
        if len(values) == 7 and int(values[-1]) == self.tokens.eos:
            return [self.tokens.eos]
        raise TigerEvalError(f"生成序列超过合法 TIGER 路径：{values}")

    def __call__(self, _batch_id: int, input_ids: Any) -> list[int]:
        generated = input_ids[self.prompt_width :]
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        return self.allowed_next(generated)


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
    capacities = _validate_token_capacities(
        manifest.get("tiger_ids", {}).get("token_capacities", ())
    )
    if tuple(manifest.get("source", {}).get("codebook_sizes", ())) != capacities[:3]:
        raise TigerEvalError("TIGER identifier 与来源 SID 容量不一致")
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
        (codes[:, 0].astype(np.int64) * capacities[1] + codes[:, 1]) * capacities[2]
        + codes[:, 2]
    ) * capacities[3] + codes[:, 3]
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
            token_capacities=capacities,
        ),
        {
            "identifier_manifest": str(manifest_path),
            "identifier_manifest_sha256": sha256_file(manifest_path),
            "mapping": str(mapping_path),
            "mapping_sha256": manifest["mapping"]["sha256"],
            "rows": expected_rows,
            "token_capacities": list(capacities),
        },
    )


def parse_target_codes(
    content: str,
    token_capacities: Sequence[int] = DEFAULT_TOKEN_CAPACITIES,
) -> tuple[int, int, int, int]:
    """Parse the exact TIGER assistant target serialization."""

    import re

    match = re.fullmatch(
        r"<TARGET_POI><S1_(\d+)><S2_(\d+)><S3_(\d+)><C_(\d+)></TARGET_POI>",
        content,
    )
    if match is None:
        raise TigerEvalError("Assistant content 不是严格 TIGER Target 格式")
    codes = tuple(int(value) for value in match.groups())
    if TigerIdIndex.pack(codes, token_capacities) < 0:
        raise TigerEvalError("Assistant TIGER identifier 超出码本范围")
    return codes  # type: ignore[return-value]


def encode_tiger_record(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    cutoff_len: int,
    split: str = "valid",
    token_capacities: Sequence[int] = DEFAULT_TOKEN_CAPACITIES,
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
    target_codes = parse_target_codes(target_content, token_capacities)
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
    base_bucket = parse_generated_base_bucket(values, tokens=tokens)
    if tokens.eos not in values:
        return TigerCandidate(
            None, None, float(score), "missing_eos", base_bucket=base_bucket
        )
    eos_index = values.index(tokens.eos)
    generated = values[: eos_index + 1]
    if len(generated) != tokens.expected_sequence_length:
        return TigerCandidate(
            None, None, float(score), "invalid_length", base_bucket=base_bucket
        )
    if generated[0] != tokens.target_open or generated[5] != tokens.target_close:
        return TigerCandidate(
            None, None, float(score), "invalid_structure", base_bucket=base_bucket
        )
    token_ranges = (tokens.s1, tokens.s2, tokens.s3, tokens.collision)
    codes = tuple(
        generated[position + 1] - token_range[0]
        for position, token_range in enumerate(token_ranges)
    )
    if any(
        code < 0 or code >= len(token_range)
        for code, token_range in zip(codes, token_ranges)
    ):
        return TigerCandidate(
            None,
            None,
            float(score),
            "invalid_position_token",
            base_bucket=base_bucket,
        )
    row = index.lookup(codes)
    if row < 0:
        return TigerCandidate(
            codes,
            None,
            float(score),
            "identifier_not_in_corpus",
            base_bucket=base_bucket,
        )
    return TigerCandidate(  # type: ignore[arg-type]
        codes,
        row,
        float(score),
        None,
        base_bucket=base_bucket,
    )


def parse_generated_base_bucket(
    sequence: Sequence[int],
    *,
    tokens: TigerTokenIds,
) -> tuple[int, int, int] | None:
    """Parse only the generated open/S1/S2/S3 prefix, independent of its suffix."""

    values = [int(value) for value in sequence]
    if len(values) < 4 or values[0] != tokens.target_open:
        return None
    token_ranges = (tokens.s1, tokens.s2, tokens.s3)
    codes = tuple(
        values[position + 1] - token_range[0]
        for position, token_range in enumerate(token_ranges)
    )
    if any(
        code < 0 or code >= len(token_range)
        for code, token_range in zip(codes, token_ranges)
    ):
        return None
    return codes  # type: ignore[return-value]


def rank_candidate_buckets(
    candidates: Sequence[TigerCandidate],
    *,
    target_codes: Sequence[int],
    index: TigerIdIndex,
) -> TigerBucketRanking:
    """Collapse ranked full IDs into first-occurrence expandable base buckets."""

    if len(target_codes) != 4:
        raise TigerEvalError("目标 TIGER identifier 必须包含四层")
    target_bucket = tuple(int(value) for value in target_codes[:3])
    if index.bucket_size(target_bucket) <= 0:
        raise TigerEvalError(f"目标基础 SID 桶不在冻结目录中：{target_bucket}")

    raw_slot_target_rank: int | None = None
    unique_buckets: list[tuple[int, int, int]] = []
    unique_bucket_first_beam_ranks: list[int] = []
    unique_bucket_sizes: list[int] = []
    seen: set[tuple[int, int, int]] = set()
    duplicate_bucket_candidates = 0
    non_expandable_candidates = 0

    for beam_rank, candidate in enumerate(candidates, start=1):
        bucket = candidate.base_bucket
        if bucket is None and candidate.codes is not None:
            bucket = tuple(int(value) for value in candidate.codes[:3])
        if bucket is None:
            non_expandable_candidates += 1
            continue
        bucket_size = index.bucket_size(bucket)
        if bucket_size <= 0:
            non_expandable_candidates += 1
            continue
        if bucket == target_bucket and raw_slot_target_rank is None:
            raw_slot_target_rank = beam_rank
        if bucket in seen:
            duplicate_bucket_candidates += 1
            continue
        seen.add(bucket)
        unique_buckets.append(bucket)
        unique_bucket_first_beam_ranks.append(beam_rank)
        unique_bucket_sizes.append(bucket_size)

    unique_target_rank = next(
        (
            rank
            for rank, bucket in enumerate(unique_buckets, start=1)
            if bucket == target_bucket
        ),
        None,
    )
    return TigerBucketRanking(
        raw_slot_target_rank=raw_slot_target_rank,
        unique_target_rank=unique_target_rank,
        unique_buckets=tuple(unique_buckets),
        unique_bucket_first_beam_ranks=tuple(unique_bucket_first_beam_ranks),
        unique_bucket_sizes=tuple(unique_bucket_sizes),
        duplicate_bucket_candidates=duplicate_bucket_candidates,
        non_expandable_candidates=non_expandable_candidates,
    )


def empty_bucket_metrics() -> dict[str, Any]:
    """Return a mergeable accumulator for three-code bucket retrieval."""

    return {
        "sample_count": 0,
        "candidate_count": 0,
        "raw_slot_hit_sums": {str(k): 0 for k in METRIC_KS},
        "unique_hit_sums": {str(k): 0 for k in METRIC_KS},
        "raw_slot_target_rank_histogram": {},
        "unique_target_rank_histogram": {},
        "unique_bucket_count_sum": 0,
        "unique_bucket_count_min": None,
        "unique_bucket_count_max": 0,
        "duplicate_bucket_candidates": 0,
        "non_expandable_candidates": 0,
        "target_bucket_size_sum": 0,
        "target_bucket_size_max": 0,
        "target_bucket_size_histogram": {},
    }


def _bucket_size_group(size: int) -> str:
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    if size <= 50:
        return "11-50"
    return "51+"


def update_bucket_metrics(
    metrics: dict[str, Any],
    *,
    target_codes: Sequence[int],
    candidates: Sequence[TigerCandidate],
    index: TigerIdIndex,
) -> TigerBucketRanking:
    """Update raw-slot and unique-expandable bucket recall accumulators."""

    ranking = rank_candidate_buckets(
        candidates,
        target_codes=target_codes,
        index=index,
    )
    target_bucket_size = index.bucket_size(target_codes[:3])
    unique_count = len(ranking.unique_buckets)
    metrics["sample_count"] += 1
    metrics["candidate_count"] += len(candidates)
    metrics["unique_bucket_count_sum"] += unique_count
    current_min = metrics["unique_bucket_count_min"]
    metrics["unique_bucket_count_min"] = (
        unique_count if current_min is None else min(int(current_min), unique_count)
    )
    metrics["unique_bucket_count_max"] = max(
        int(metrics["unique_bucket_count_max"]), unique_count
    )
    metrics["duplicate_bucket_candidates"] += ranking.duplicate_bucket_candidates
    metrics["non_expandable_candidates"] += ranking.non_expandable_candidates
    metrics["target_bucket_size_sum"] += target_bucket_size
    metrics["target_bucket_size_max"] = max(
        int(metrics["target_bucket_size_max"]), target_bucket_size
    )
    bucket_group = _bucket_size_group(target_bucket_size)
    bucket_histogram = metrics["target_bucket_size_histogram"]
    bucket_histogram[bucket_group] = bucket_histogram.get(bucket_group, 0) + 1

    for name, target_rank in (
        ("raw_slot", ranking.raw_slot_target_rank),
        ("unique", ranking.unique_target_rank),
    ):
        rank_key = str(target_rank) if target_rank is not None else "miss"
        histogram = metrics[f"{name}_target_rank_histogram"]
        histogram[rank_key] = histogram.get(rank_key, 0) + 1
        hit_sums = metrics[f"{name}_hit_sums"]
        for k in METRIC_KS:
            hit_sums[str(k)] += int(target_rank is not None and target_rank <= k)
    return ranking


def merge_bucket_metrics(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Merge one chunk-level bucket accumulator into a run accumulator."""

    for key in (
        "sample_count",
        "candidate_count",
        "unique_bucket_count_sum",
        "duplicate_bucket_candidates",
        "non_expandable_candidates",
        "target_bucket_size_sum",
    ):
        target[key] += int(source[key])
    target["unique_bucket_count_max"] = max(
        int(target["unique_bucket_count_max"]),
        int(source["unique_bucket_count_max"]),
    )
    target["target_bucket_size_max"] = max(
        int(target["target_bucket_size_max"]),
        int(source["target_bucket_size_max"]),
    )
    source_min = source["unique_bucket_count_min"]
    if source_min is not None:
        target_min = target["unique_bucket_count_min"]
        target["unique_bucket_count_min"] = (
            int(source_min)
            if target_min is None
            else min(int(target_min), int(source_min))
        )
    for name in (
        "raw_slot_hit_sums",
        "unique_hit_sums",
        "raw_slot_target_rank_histogram",
        "unique_target_rank_histogram",
        "target_bucket_size_histogram",
    ):
        for key, value in source[name].items():
            target[name][key] = target[name].get(key, 0) + value


def finalize_bucket_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Convert accumulated bucket counts into auditable retrieval rates."""

    samples = int(metrics["sample_count"])
    candidates = int(metrics["candidate_count"])
    if samples <= 0 or candidates <= 0:
        raise TigerEvalError("TIGER Bucket 评测指标不能为空")
    result: dict[str, Any] = {
        "sample_count": samples,
        "candidate_count": candidates,
        "bucket_definition": "first three codes [S1,S2,S3]",
        "prefix_parse_rule": (
            "parse <TARGET_POI>, S1, S2 and S3 independently of collision, close and EOS"
        ),
        "raw_slot_ranking": "full-ID beam slots retained; duplicate buckets retain slots",
        "unique_ranking": (
            "expandable corpus buckets deduplicated by first beam occurrence; "
            "non-expandable candidates removed"
        ),
        "raw_slot_target_rank_histogram": metrics["raw_slot_target_rank_histogram"],
        "unique_target_rank_histogram": metrics["unique_target_rank_histogram"],
        "unique_bucket_count_mean": int(metrics["unique_bucket_count_sum"]) / samples,
        "unique_bucket_count_min": int(metrics["unique_bucket_count_min"]),
        "unique_bucket_count_max": int(metrics["unique_bucket_count_max"]),
        "duplicate_bucket_candidate_rate": int(metrics["duplicate_bucket_candidates"])
        / candidates,
        "non_expandable_candidate_rate": int(metrics["non_expandable_candidates"])
        / candidates,
        "target_bucket_size_mean": int(metrics["target_bucket_size_sum"]) / samples,
        "target_bucket_size_max": int(metrics["target_bucket_size_max"]),
        "target_bucket_size_histogram": metrics["target_bucket_size_histogram"],
    }
    for k in METRIC_KS:
        result[f"raw_slot_bucket_hr@{k}"] = (
            metrics["raw_slot_hit_sums"][str(k)] / samples
        )
        result[f"unique_bucket_hr@{k}"] = metrics["unique_hit_sums"][str(k)] / samples
    return result


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
    for name in (
        "hit_sums",
        "ndcg_sums",
        "invalid_error_counts",
        "target_rank_histogram",
    ):
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
        "samples_with_invalid_id_rate": int(metrics["samples_with_invalid_candidate"])
        / samples,
        "invalid_error_counts": metrics["invalid_error_counts"],
        "target_rank_histogram": metrics["target_rank_histogram"],
    }
    for k in METRIC_KS:
        result[f"hr@{k}"] = metrics["hit_sums"][str(k)] / samples
        result[f"ndcg@{k}"] = metrics["ndcg_sums"][str(k)] / samples
    return result

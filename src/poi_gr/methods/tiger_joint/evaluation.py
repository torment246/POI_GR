"""Bucket-retrieval evaluation helpers for a frozen TIGER-Joint checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from poi_gr.methods.tiger_joint.data import SidTokenLayout
from poi_gr.methods.tiger_joint.preparation import DynamicTigerExample
from poi_gr.sid.evaluation import compute_basic_metrics, compute_layer_metrics


BUCKET_CUTOFFS = (1, 3, 5, 10)


class TigerJointEvaluationError(ValueError):
    """Raised when a joint SID or generated candidate violates evaluation rules."""


@dataclass(frozen=True)
class JointBucketIndex:
    """Compact lookup over all expandable three-code catalog buckets."""

    sorted_keys: np.ndarray
    bucket_sizes: np.ndarray
    codebook_sizes: tuple[int, int, int]

    def __post_init__(self) -> None:
        if (
            self.sorted_keys.ndim != 1
            or self.bucket_sizes.shape != self.sorted_keys.shape
            or self.sorted_keys.dtype != np.int64
            or self.bucket_sizes.dtype != np.int64
            or not len(self.sorted_keys)
        ):
            raise TigerJointEvaluationError("Bucket index shape/dtype 无效")
        if len(self.codebook_sizes) != 3 or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.codebook_sizes
        ):
            raise TigerJointEvaluationError("Bucket index 码本容量无效")
        if np.any(self.sorted_keys[1:] <= self.sorted_keys[:-1]):
            raise TigerJointEvaluationError("Bucket key 必须严格递增")
        if np.any(self.bucket_sizes <= 0):
            raise TigerJointEvaluationError("Bucket size 必须为正数")

    @classmethod
    def from_codes(
        cls,
        codes: np.ndarray,
        codebook_sizes: Sequence[int],
    ) -> "JointBucketIndex":
        """Build an index without materializing a POI-ID dictionary."""

        values = np.asarray(codes)
        sizes = tuple(int(size) for size in codebook_sizes)
        if values.ndim != 2 or values.shape[1] != 3 or len(sizes) != 3:
            raise TigerJointEvaluationError("全量 SID 必须是 [N, 3]")
        if values.dtype.kind not in {"i", "u"} or not len(values):
            raise TigerJointEvaluationError("全量 SID 必须是非空整数数组")
        for level, size in enumerate(sizes):
            level_values = values[:, level]
            if int(level_values.min()) < 0 or int(level_values.max()) >= size:
                raise TigerJointEvaluationError(f"S{level + 1} code 超出容量")
        keys = cls.pack_array(values, sizes)
        sorted_keys, counts = np.unique(keys, return_counts=True)
        return cls(
            sorted_keys=np.asarray(sorted_keys, dtype=np.int64),
            bucket_sizes=np.asarray(counts, dtype=np.int64),
            codebook_sizes=(sizes[0], sizes[1], sizes[2]),
        )

    @staticmethod
    def pack_array(
        codes: np.ndarray,
        codebook_sizes: Sequence[int],
    ) -> np.ndarray:
        """Pack `[N, 3]` codes into collision-free signed 64-bit keys."""

        values = np.asarray(codes, dtype=np.int64)
        sizes = tuple(int(size) for size in codebook_sizes)
        return (
            (values[:, 0] * sizes[1] + values[:, 1]) * sizes[2]
            + values[:, 2]
        )

    def pack(self, codes: Sequence[int]) -> int:
        """Validate and pack one three-code bucket."""

        if len(codes) != 3:
            raise TigerJointEvaluationError("Bucket code 数必须为 3")
        normalized = tuple(int(value) for value in codes)
        if any(
            value < 0 or value >= self.codebook_sizes[level]
            for level, value in enumerate(normalized)
        ):
            raise TigerJointEvaluationError("Bucket code 超出容量")
        return (
            (normalized[0] * self.codebook_sizes[1] + normalized[1])
            * self.codebook_sizes[2]
            + normalized[2]
        )

    def bucket_size(self, codes: Sequence[int]) -> int:
        """Return zero for a syntactically valid path absent from the catalog."""

        key = self.pack(codes)
        position = int(np.searchsorted(self.sorted_keys, key))
        if position >= len(self.sorted_keys) or int(self.sorted_keys[position]) != key:
            return 0
        return int(self.bucket_sizes[position])


class JointLegalPathConstraint:
    """Restrict dynamic SID generation to buckets present in the frozen catalog."""

    def __init__(
        self,
        *,
        index: JointBucketIndex,
        token_layout: SidTokenLayout,
        target_open_token_id: int,
        target_close_token_id: int,
        eos_token_id: int,
        prompt_width: int,
        next_token_cache: dict[tuple[int, ...], list[int]] | None = None,
    ) -> None:
        if prompt_width <= 0:
            raise TigerJointEvaluationError("prompt_width 必须为正整数")
        if tuple(len(level) for level in token_layout.level_token_ids) != (
            index.codebook_sizes
        ):
            raise TigerJointEvaluationError("合法路径约束的 Token 容量与 Bucket 不一致")
        self.index = index
        self.token_layout = token_layout
        self.target_open_token_id = int(target_open_token_id)
        self.target_close_token_id = int(target_close_token_id)
        self.eos_token_id = int(eos_token_id)
        self.prompt_width = int(prompt_width)
        self._code_by_token = tuple(
            {int(token_id): code for code, token_id in enumerate(level)}
            for level in token_layout.level_token_ids
        )
        self._next_token_cache = (
            next_token_cache if next_token_cache is not None else {}
        )

    def _decode_prefix_codes(self, generated: Sequence[int]) -> tuple[int, ...]:
        if not generated or int(generated[0]) != self.target_open_token_id:
            raise TigerJointEvaluationError("合法路径生成必须以 <TARGET_POI> 开始")
        code_tokens = generated[1:]
        if len(code_tokens) > 3:
            raise TigerJointEvaluationError("合法路径 SID Prefix 超过三级")
        codes: list[int] = []
        for level, token_id in enumerate(code_tokens):
            code = self._code_by_token[level].get(int(token_id))
            if code is None:
                raise TigerJointEvaluationError("生成序列离开合法的分层 SID Token 位置")
            codes.append(code)
        return tuple(codes)

    def _allowed_code_tokens(self, prefix_codes: tuple[int, ...]) -> list[int]:
        cached = self._next_token_cache.get(prefix_codes)
        if cached is not None:
            return cached
        capacities = self.index.codebook_sizes
        depth = len(prefix_codes)
        if depth >= len(capacities):
            raise TigerJointEvaluationError("完整 Bucket 不再包含下一级 SID Token")

        prefix_value = 0
        for code, capacity in zip(prefix_codes, capacities):
            prefix_value = prefix_value * capacity + code
        suffix_size = math.prod(capacities[depth:])
        lower = prefix_value * suffix_size
        upper = lower + suffix_size
        start = int(np.searchsorted(self.index.sorted_keys, lower, side="left"))
        stop = int(np.searchsorted(self.index.sorted_keys, upper, side="left"))
        if start >= stop:
            raise TigerJointEvaluationError(
                f"Bucket Prefix 不在冻结目录中：{prefix_codes}"
            )

        divisor = math.prod(capacities[depth + 1 :])
        next_codes = (
            self.index.sorted_keys[start:stop] // divisor
        ) % capacities[depth]
        distinct = next_codes[
            np.concatenate(
                (np.asarray([True]), next_codes[1:] != next_codes[:-1])
            )
        ]
        token_ids = self.token_layout.level_token_ids[depth]
        allowed = [int(token_ids[int(code)]) for code in distinct]
        self._next_token_cache[prefix_codes] = allowed
        return allowed

    def allowed_next(self, generated: Sequence[int]) -> list[int]:
        """Return the exact legal next-token set for one generated prefix."""

        values = tuple(int(value) for value in generated)
        if not values:
            return [self.target_open_token_id]
        if len(values) <= 3:
            return self._allowed_code_tokens(self._decode_prefix_codes(values))
        if len(values) == 4:
            codes = self._decode_prefix_codes(values)
            if self.index.bucket_size(codes) <= 0:
                raise TigerJointEvaluationError(
                    f"完整 Bucket 不在冻结目录中：{codes}"
                )
            return [self.target_close_token_id]
        if len(values) == 5:
            if values[-1] != self.target_close_token_id:
                raise TigerJointEvaluationError("完整 Bucket 后必须生成 </TARGET_POI>")
            self._decode_prefix_codes(values[:-1])
            return [self.eos_token_id]
        if len(values) == 6 and values[-1] == self.eos_token_id:
            return [self.eos_token_id]
        raise TigerJointEvaluationError(f"生成序列超过合法 Bucket 路径：{values}")

    def __call__(self, _batch_id: int, input_ids: Any) -> list[int]:
        generated = input_ids[self.prompt_width :]
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        return self.allowed_next(generated)


@dataclass(frozen=True)
class JointGeneratedCandidate:
    """One raw Beam candidate parsed only through its three-code prefix."""

    sequence_token_ids: tuple[int, ...]
    score: float
    bucket: tuple[int, int, int] | None
    prefix_error: str | None
    target_close_valid: bool


@dataclass(frozen=True)
class JointBucketRanking:
    """Raw-slot and deduplicated expandable-bucket ranks for one request."""

    raw_slot_target_rank: int | None
    unique_target_rank: int | None
    unique_buckets: tuple[tuple[int, int, int], ...]
    unique_bucket_first_beam_ranks: tuple[int, ...]
    unique_bucket_sizes: tuple[int, ...]
    prefix_invalid_candidates: int
    nonexpandable_candidates: int
    duplicate_bucket_candidates: int
    complete_format_valid_candidates: int


class JointSidCandidateParser:
    """Parse dynamic S1/S2/S3 vocabulary IDs without constraining decoding."""

    def __init__(
        self,
        *,
        token_layout: SidTokenLayout,
        target_open_token_id: int,
        target_close_token_id: int,
    ) -> None:
        self.target_open_token_id = int(target_open_token_id)
        self.target_close_token_id = int(target_close_token_id)
        self.code_by_token = tuple(
            {int(token_id): code for code, token_id in enumerate(level)}
            for level in token_layout.level_token_ids
        )

    def parse(
        self,
        sequence_token_ids: Sequence[int],
        score: float,
    ) -> JointGeneratedCandidate:
        """Accept a bucket when `<TARGET_POI>,S1,S2,S3` is valid.

        The closing token is audited separately and never removes an otherwise
        valid bucket prefix, matching the existing TIGER Bucket-HR protocol.
        """

        tokens = tuple(int(value) for value in sequence_token_ids)
        close_valid = (
            len(tokens) >= 5 and tokens[4] == self.target_close_token_id
        )
        if len(tokens) < 4:
            return JointGeneratedCandidate(
                tokens, float(score), None, "too_short_for_bucket_prefix", close_valid
            )
        if tokens[0] != self.target_open_token_id:
            return JointGeneratedCandidate(
                tokens, float(score), None, "missing_target_open", close_valid
            )
        codes: list[int] = []
        for level in range(3):
            code = self.code_by_token[level].get(tokens[level + 1])
            if code is None:
                return JointGeneratedCandidate(
                    tokens,
                    float(score),
                    None,
                    f"invalid_s{level + 1}_token",
                    close_valid,
                )
            codes.append(code)
        return JointGeneratedCandidate(
            sequence_token_ids=tokens,
            score=float(score),
            bucket=(codes[0], codes[1], codes[2]),
            prefix_error=None,
            target_close_valid=close_valid,
        )


def rank_candidate_buckets(
    *,
    target_codes: Sequence[int],
    candidates: Sequence[JointGeneratedCandidate],
    index: JointBucketIndex,
) -> JointBucketRanking:
    """Preserve raw Beam ranks, then compact only valid unique catalog buckets."""

    target = tuple(int(value) for value in target_codes)
    if len(target) != 3 or index.bucket_size(target) <= 0:
        raise TigerJointEvaluationError("目标 Bucket 不在全量 epoch 3 SID 目录中")
    raw_rank: int | None = None
    unique_rank: int | None = None
    seen: set[tuple[int, int, int]] = set()
    unique_buckets: list[tuple[int, int, int]] = []
    first_ranks: list[int] = []
    bucket_sizes: list[int] = []
    prefix_invalid = 0
    nonexpandable = 0
    duplicate = 0
    complete_valid = 0
    for beam_rank, candidate in enumerate(candidates, start=1):
        if candidate.bucket is None:
            prefix_invalid += 1
            continue
        if candidate.target_close_valid:
            complete_valid += 1
        size = index.bucket_size(candidate.bucket)
        if size <= 0:
            nonexpandable += 1
            continue
        if candidate.bucket == target and raw_rank is None:
            raw_rank = beam_rank
        if candidate.bucket in seen:
            duplicate += 1
            continue
        seen.add(candidate.bucket)
        unique_buckets.append(candidate.bucket)
        first_ranks.append(beam_rank)
        bucket_sizes.append(size)
        if candidate.bucket == target and unique_rank is None:
            unique_rank = len(unique_buckets)
    return JointBucketRanking(
        raw_slot_target_rank=raw_rank,
        unique_target_rank=unique_rank,
        unique_buckets=tuple(unique_buckets),
        unique_bucket_first_beam_ranks=tuple(first_ranks),
        unique_bucket_sizes=tuple(bucket_sizes),
        prefix_invalid_candidates=prefix_invalid,
        nonexpandable_candidates=nonexpandable,
        duplicate_bucket_candidates=duplicate,
        complete_format_valid_candidates=complete_valid,
    )


def empty_bucket_metrics() -> dict[str, Any]:
    """Return a mergeable accumulator for fixed Beam=10 Bucket-HR."""

    return {
        "samples": 0,
        "candidates": 0,
        "raw_hit_sums": {str(k): 0 for k in BUCKET_CUTOFFS},
        "unique_hit_sums": {str(k): 0 for k in BUCKET_CUTOFFS},
        "unique_bucket_count_sum": 0,
        "unique_bucket_count_min": None,
        "unique_bucket_count_max": 0,
        "prefix_invalid_candidates": 0,
        "nonexpandable_candidates": 0,
        "duplicate_bucket_candidates": 0,
        "complete_format_valid_candidates": 0,
        "target_bucket_size_sum": 0,
        "target_bucket_size_max": 0,
        "target_bucket_size_histogram": {},
    }


def _bucket_size_group(size: int) -> str:
    if size == 1:
        return "1"
    if size <= 5:
        return "2-5"
    if size <= 10:
        return "6-10"
    if size <= 50:
        return "11-50"
    return "51+"


def update_bucket_metrics(
    metrics: dict[str, Any],
    *,
    target_codes: Sequence[int],
    candidates: Sequence[JointGeneratedCandidate],
    index: JointBucketIndex,
) -> JointBucketRanking:
    """Update one request and return its auditable candidate ranking."""

    ranking = rank_candidate_buckets(
        target_codes=target_codes,
        candidates=candidates,
        index=index,
    )
    metrics["samples"] += 1
    metrics["candidates"] += len(candidates)
    for cutoff in BUCKET_CUTOFFS:
        if (
            ranking.raw_slot_target_rank is not None
            and ranking.raw_slot_target_rank <= cutoff
        ):
            metrics["raw_hit_sums"][str(cutoff)] += 1
        if ranking.unique_target_rank is not None and ranking.unique_target_rank <= cutoff:
            metrics["unique_hit_sums"][str(cutoff)] += 1
    unique_count = len(ranking.unique_buckets)
    metrics["unique_bucket_count_sum"] += unique_count
    current_min = metrics["unique_bucket_count_min"]
    metrics["unique_bucket_count_min"] = (
        unique_count if current_min is None else min(int(current_min), unique_count)
    )
    metrics["unique_bucket_count_max"] = max(
        int(metrics["unique_bucket_count_max"]), unique_count
    )
    for name in (
        "prefix_invalid_candidates",
        "nonexpandable_candidates",
        "duplicate_bucket_candidates",
        "complete_format_valid_candidates",
    ):
        metrics[name] += int(getattr(ranking, name))
    target_size = index.bucket_size(target_codes)
    metrics["target_bucket_size_sum"] += target_size
    metrics["target_bucket_size_max"] = max(
        int(metrics["target_bucket_size_max"]), target_size
    )
    group = _bucket_size_group(target_size)
    histogram = metrics["target_bucket_size_histogram"]
    histogram[group] = int(histogram.get(group, 0)) + 1
    return ranking


def finalize_bucket_metrics(
    metrics: Mapping[str, Any],
    *,
    legal_path_constraint: bool = False,
) -> dict[str, Any]:
    """Convert accumulated counts to explicit percentages-as-ratios."""

    samples = int(metrics["samples"])
    candidates = int(metrics["candidates"])
    if samples <= 0 or candidates <= 0:
        raise TigerJointEvaluationError("Bucket 指标没有样本或候选")
    prefix_invalid = int(metrics["prefix_invalid_candidates"])
    nonexpandable = int(metrics["nonexpandable_candidates"])
    duplicate = int(metrics["duplicate_bucket_candidates"])
    complete_valid = int(metrics["complete_format_valid_candidates"])
    result: dict[str, Any] = {
        "sample_count": samples,
        "candidate_count": candidates,
        "bucket_definition": "three current epoch-3 RQ-VAE codes [S1,S2,S3]",
        "decoding": (
            "epoch3_catalog_legal_path_constrained_beam_search"
            if legal_path_constraint
            else "unconstrained_beam_search_then_epoch3_catalog_bucket_lookup"
        ),
        "raw_slot_ranking": "invalid and duplicate candidates retain Beam slots",
        "unique_bucket_ranking": (
            "remove invalid/nonexpandable paths and deduplicate by first Beam rank"
        ),
        "prefix_valid_candidate_rate": (candidates - prefix_invalid) / candidates,
        "complete_target_format_valid_candidate_rate": complete_valid / candidates,
        "expandable_candidate_rate": (
            candidates - prefix_invalid - nonexpandable
        )
        / candidates,
        "nonexpandable_candidate_rate": nonexpandable / candidates,
        "duplicate_expandable_bucket_candidate_rate": duplicate / candidates,
        "unique_bucket_count_mean": int(metrics["unique_bucket_count_sum"])
        / samples,
        "unique_bucket_count_min": int(metrics["unique_bucket_count_min"]),
        "unique_bucket_count_max": int(metrics["unique_bucket_count_max"]),
        "target_bucket_size_mean": int(metrics["target_bucket_size_sum"]) / samples,
        "target_bucket_size_max": int(metrics["target_bucket_size_max"]),
        "target_bucket_size_histogram": dict(
            metrics["target_bucket_size_histogram"]
        ),
    }
    for cutoff in BUCKET_CUTOFFS:
        result[f"raw_slot_bucket_hr@{cutoff}"] = (
            int(metrics["raw_hit_sums"][str(cutoff)]) / samples
        )
        result[f"unique_bucket_hr@{cutoff}"] = (
            int(metrics["unique_hit_sums"][str(cutoff)]) / samples
        )
    return result


def materialize_joint_prompt(
    example: DynamicTigerExample,
    *,
    sid_codes: np.ndarray,
    token_layout: SidTokenLayout,
) -> tuple[int, ...]:
    """Patch frozen epoch-3 history SIDs and return the training-identical prompt."""

    codes = np.asarray(sid_codes)
    if codes.ndim != 2 or codes.shape[1] != 3 or codes.dtype.kind not in {"i", "u"}:
        raise TigerJointEvaluationError("SID codes 必须是 [N, 3] 整数数组")
    source_length = int(example.target_sid_positions[0]) - 1
    if source_length <= 0 or source_length > len(example.input_ids):
        raise TigerJointEvaluationError("动态目标起始位置无效")
    prompt = list(example.input_ids[:source_length])
    if len(example.history_poi_rows) != len(example.history_sid_positions):
        raise TigerJointEvaluationError("历史 POI 行与槽位数量不一致")
    for poi_row, positions in zip(
        example.history_poi_rows,
        example.history_sid_positions,
        strict=True,
    ):
        if poi_row < 0 or poi_row >= len(codes):
            raise TigerJointEvaluationError("历史 POI 行超出 SID 目录")
        for level, position in enumerate(positions):
            code = int(codes[poi_row, level])
            prompt[position] = int(token_layout.level_token_ids[level][code])
    return tuple(prompt)


def compute_joint_sid_metrics(
    codes: np.ndarray,
    codebook_sizes: Sequence[int],
) -> dict[str, Any]:
    """Summarize full-catalog utilization, entropy, prefixes, and collisions."""

    values = np.asarray(codes)
    sizes = tuple(int(size) for size in codebook_sizes)
    if values.ndim != 2 or values.shape[1] != 3 or len(sizes) != 3:
        raise TigerJointEvaluationError("SID 静态诊断要求 [N, 3]")
    basic, _, _, full_bucket_sizes = compute_basic_metrics(values)
    layers = compute_layer_metrics(values, sizes)
    enriched_layers: list[dict[str, Any]] = []
    for level, layer in enumerate(layers):
        counts = np.bincount(
            np.asarray(values[:, level], dtype=np.int64), minlength=sizes[level]
        )
        probabilities = counts[counts > 0].astype(np.float64) / len(values)
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        enriched_layers.append(
            {
                **layer,
                "entropy_perplexity": math.exp(entropy),
                "kish_effective_code_count": float(
                    1.0 / np.sum(probabilities * probabilities)
                ),
                "max_code_count": int(counts.max()),
                "max_code_share": float(counts.max() / len(values)),
            }
        )
    prefixes: list[dict[str, Any]] = []
    for depth in range(1, 4):
        prefix_basic, _, _, _ = compute_basic_metrics(values[:, :depth])
        prefixes.append({"depth": depth, **prefix_basic})
    bucket_probabilities = full_bucket_sizes.astype(np.float64) / len(values)
    bucket_entropy = float(
        -np.sum(bucket_probabilities * np.log(bucket_probabilities))
    )
    return {
        "schema_version": "tiger-joint-sid-diagnostics-v1",
        "status": "completed",
        "shape": [int(value) for value in values.shape],
        "dtype": str(values.dtype),
        "codebook_sizes": list(sizes),
        "basic": basic,
        "layers": enriched_layers,
        "prefixes": prefixes,
        "full_bucket_entropy": bucket_entropy,
        "full_bucket_entropy_perplexity": math.exp(bucket_entropy),
        "full_bucket_kish_effective_count": float(
            1.0 / np.sum(bucket_probabilities * bucket_probabilities)
        ),
    }

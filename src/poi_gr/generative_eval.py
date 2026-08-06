"""Trie-constrained generative retrieval evaluation for unique Final PIDs."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pid_trie import (
    CompactPidTrie,
    GEOHASH_ALPHABET,
    PidTokenIds,
    TriePrefilledPrefixConstraint,
    TriePrefixConstraint,
    load_pid_token_ids,
    sha256_file,
)
from .methods.genpoi_proximity import (
    GenPoiProximityError,
    effective_prefix_length,
    extract_current_query_and_gid,
)


EVALUATION_SCHEMA_VERSION = "final-pid-generative-eval-v1"
DEFAULT_KS = (1, 3, 5, 10)
ERROR_TYPES = (
    "top1_gid_error",
    "gid_correct_sid_error",
    "base_pid_correct_dedup_error",
    "target_in_top10_not_top1",
    "top10_miss",
)
PID_TOKEN_PATTERN = re.compile(r"<(?:G_[0-9bcdefghjkmnpqrstuvwxyz]|S[123]_\d+|D_\d+)>")


class GenerativeEvalError(ValueError):
    """Raised when evaluation inputs or frozen protocol are inconsistent."""


@dataclass(frozen=True)
class RankedCandidate:
    pid_tokens: tuple[int, ...]
    poi_row: int
    score: float


@dataclass(frozen=True)
class CandidateBatch:
    candidates: tuple[RankedCandidate, ...]
    raw_count: int
    valid_structure_count: int
    valid_pid_count: int
    duplicate_count: int
    generated_lengths: tuple[int, ...]


@dataclass(frozen=True)
class PromptExample:
    sample_id: str
    user_content: str
    target_content: str
    target_pid_tokens: tuple[int, ...]
    prompt_ids: tuple[int, ...]
    target_poi_id: str
    requires_dedup: bool
    target_present_in_causal_history: bool
    split: str


@dataclass(frozen=True)
class FixedValidationSubset:
    data_path: Path
    manifest_path: Path
    row_count: int
    sha256: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SspPrediction:
    user_gid: tuple[str, ...]
    predicted_lambda: int
    prefill_length: int


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_valid_structure(pid_tokens: Sequence[int], token_ids: PidTokenIds) -> bool:
    if len(pid_tokens) not in (9, 10):
        return False
    gid_low, gid_high = int(token_ids.gid[0]), int(token_ids.gid[-1])
    if any(not gid_low <= pid_tokens[index] <= gid_high for index in range(6)):
        return False
    if any(
        not int(token_ids.sid[index][0])
        <= pid_tokens[index + 6]
        <= int(token_ids.sid[index][-1])
        for index in range(3)
    ):
        return False
    if len(pid_tokens) == 10 and not (
        int(token_ids.dedup[0]) <= pid_tokens[9] <= int(token_ids.dedup[-1])
    ):
        return False
    return True


def rank_and_deduplicate_candidates(
    generated_sequences: Sequence[Sequence[int]],
    sequence_scores: Sequence[float],
    *,
    trie: CompactPidTrie,
    token_ids: PidTokenIds,
    top_k: int,
) -> CandidateBatch:
    """Validate, score-sort, and stably deduplicate generated Final PIDs."""

    if len(generated_sequences) != len(sequence_scores):
        raise GenerativeEvalError("生成序列数与 sequence score 数量不一致")
    if top_k <= 0:
        raise GenerativeEvalError("top_k 必须为正整数")

    raw: list[RankedCandidate] = []
    valid_structure_count = 0
    generated_lengths: list[int] = []
    for sequence, score in zip(generated_sequences, sequence_scores):
        values = [int(value) for value in sequence]
        try:
            eos_index = values.index(token_ids.eos)
        except ValueError:
            generated_lengths.append(len(values))
            continue
        path = values[: eos_index + 1]
        pid = tuple(path[:-1])
        generated_lengths.append(len(path))
        if not _is_valid_structure(pid, token_ids):
            continue
        valid_structure_count += 1
        poi_row = trie.lookup(path)
        if poi_row < 0:
            continue
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise GenerativeEvalError("sequence_scores 出现 NaN 或 Inf")
        raw.append(
            RankedCandidate(
                pid_tokens=pid,
                poi_row=poi_row,
                score=numeric_score,
            )
        )

    raw.sort(key=lambda item: (-item.score, item.pid_tokens))
    deduplicated_all: list[RankedCandidate] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in raw:
        if candidate.pid_tokens in seen:
            continue
        seen.add(candidate.pid_tokens)
        deduplicated_all.append(candidate)
    deduplicated = deduplicated_all[:top_k]
    return CandidateBatch(
        candidates=tuple(deduplicated),
        raw_count=len(generated_sequences),
        valid_structure_count=valid_structure_count,
        valid_pid_count=len(raw),
        duplicate_count=len(raw) - len(deduplicated_all),
        generated_lengths=tuple(generated_lengths),
    )


def hit_and_ndcg(
    target_pid: Sequence[int],
    candidates: Sequence[RankedCandidate],
    ks: Sequence[int] = DEFAULT_KS,
    *,
    available_candidates: int,
) -> dict[str, float | int | None]:
    """Return single-relevance HR/NDCG values with unavailable K as null."""

    target = tuple(int(value) for value in target_pid)
    rank = next(
        (
            index
            for index, candidate in enumerate(candidates, start=1)
            if candidate.pid_tokens == target
        ),
        None,
    )
    result: dict[str, float | int | None] = {"target_rank": rank}
    for k in ks:
        if available_candidates < k:
            result[f"hr@{k}"] = None
            result[f"ndcg@{k}"] = None
            continue
        hit = int(rank is not None and rank <= k)
        result[f"hr@{k}"] = hit
        result[f"ndcg@{k}"] = (
            1.0 / math.log2(rank + 1) if hit and rank is not None else 0.0
        )
    return result


class RetrievalMetricsAccumulator:
    """Mergeable sufficient statistics for retrieval and diagnostic metrics."""

    def __init__(
        self,
        *,
        available_candidates: int,
        ks: Sequence[int] = DEFAULT_KS,
    ) -> None:
        self.available_candidates = int(available_candidates)
        self.ks = tuple(int(k) for k in ks)
        self.sample_count = 0
        self.hit_sums = {k: 0 for k in self.ks}
        self.ndcg_sums = {k: 0.0 for k in self.ks}
        self.layer_correct = np.zeros(9, dtype=np.int64)
        self.gid_exact = 0
        self.sid_exact = 0
        self.base_exact = 0
        self.final_exact = 0
        self.group_counts = {"singleton": 0, "dedup": 0}
        self.group_hits = {name: {k: 0 for k in self.ks} for name in self.group_counts}
        self.group_ndcg10 = {"singleton": 0.0, "dedup": 0.0}
        self.dedup_condition_count = 0
        self.dedup_condition_correct = 0
        self.raw_candidate_count = 0
        self.valid_structure_count = 0
        self.valid_pid_count = 0
        self.duplicate_count = 0
        self.returned_count_hist = Counter()
        self.generated_length_hist = Counter()

    def update(
        self,
        target_pid: Sequence[int],
        batch: CandidateBatch,
    ) -> dict[str, float | int | None]:
        target = tuple(int(value) for value in target_pid)
        if len(target) not in (9, 10):
            raise GenerativeEvalError("目标 Final PID 长度必须为 9 或 10")
        metrics = hit_and_ndcg(
            target,
            batch.candidates,
            self.ks,
            available_candidates=self.available_candidates,
        )
        self.sample_count += 1
        for k in self.ks:
            if metrics[f"hr@{k}"] is not None:
                self.hit_sums[k] += int(metrics[f"hr@{k}"])
                self.ndcg_sums[k] += float(metrics[f"ndcg@{k}"])

        top1 = batch.candidates[0].pid_tokens if batch.candidates else ()
        for index in range(9):
            self.layer_correct[index] += int(
                len(top1) > index and top1[index] == target[index]
            )
        self.gid_exact += int(len(top1) >= 6 and top1[:6] == target[:6])
        self.sid_exact += int(len(top1) >= 9 and top1[6:9] == target[6:9])
        base_correct = len(top1) >= 9 and top1[:9] == target[:9]
        self.base_exact += int(base_correct)
        final_correct = top1 == target
        self.final_exact += int(final_correct)

        group = "dedup" if len(target) == 10 else "singleton"
        self.group_counts[group] += 1
        for k in self.ks:
            value = metrics[f"hr@{k}"]
            if value is not None:
                self.group_hits[group][k] += int(value)
        if metrics.get("ndcg@10") is not None:
            self.group_ndcg10[group] += float(metrics["ndcg@10"])

        if group == "dedup" and base_correct:
            self.dedup_condition_count += 1
            self.dedup_condition_correct += int(
                len(top1) == 10 and top1[9] == target[9]
            )

        self.raw_candidate_count += batch.raw_count
        self.valid_structure_count += batch.valid_structure_count
        self.valid_pid_count += batch.valid_pid_count
        self.duplicate_count += batch.duplicate_count
        self.returned_count_hist[len(batch.candidates)] += 1
        self.generated_length_hist.update(batch.generated_lengths)
        return metrics

    def merge(self, other: "RetrievalMetricsAccumulator") -> None:
        if (
            self.available_candidates != other.available_candidates
            or self.ks != other.ks
        ):
            raise GenerativeEvalError("不能合并候选容量或 K 定义不同的指标")
        self.sample_count += other.sample_count
        for k in self.ks:
            self.hit_sums[k] += other.hit_sums[k]
            self.ndcg_sums[k] += other.ndcg_sums[k]
        self.layer_correct += other.layer_correct
        self.gid_exact += other.gid_exact
        self.sid_exact += other.sid_exact
        self.base_exact += other.base_exact
        self.final_exact += other.final_exact
        for group in self.group_counts:
            self.group_counts[group] += other.group_counts[group]
            for k in self.ks:
                self.group_hits[group][k] += other.group_hits[group][k]
            self.group_ndcg10[group] += other.group_ndcg10[group]
        self.dedup_condition_count += other.dedup_condition_count
        self.dedup_condition_correct += other.dedup_condition_correct
        self.raw_candidate_count += other.raw_candidate_count
        self.valid_structure_count += other.valid_structure_count
        self.valid_pid_count += other.valid_pid_count
        self.duplicate_count += other.duplicate_count
        self.returned_count_hist.update(other.returned_count_hist)
        self.generated_length_hist.update(other.generated_length_hist)

    @staticmethod
    def _hist_percentile(histogram: Counter[int], q: float) -> float | None:
        total = sum(histogram.values())
        if total == 0:
            return None
        rank = (total - 1) * q
        low = int(math.floor(rank))
        high = int(math.ceil(rank))

        def value_at(target: int) -> int:
            cumulative = 0
            for value in sorted(histogram):
                cumulative += histogram[value]
                if cumulative > target:
                    return value
            raise AssertionError("histogram rank overflow")

        low_value = value_at(low)
        high_value = value_at(high)
        return low_value + (high_value - low_value) * (rank - low)

    def to_state(self) -> dict[str, Any]:
        return {
            "available_candidates": self.available_candidates,
            "ks": list(self.ks),
            "sample_count": self.sample_count,
            "hit_sums": {str(k): self.hit_sums[k] for k in self.ks},
            "ndcg_sums": {str(k): self.ndcg_sums[k] for k in self.ks},
            "layer_correct": self.layer_correct.tolist(),
            "gid_exact": self.gid_exact,
            "sid_exact": self.sid_exact,
            "base_exact": self.base_exact,
            "final_exact": self.final_exact,
            "group_counts": self.group_counts,
            "group_hits": {
                group: {str(k): values[k] for k in self.ks}
                for group, values in self.group_hits.items()
            },
            "group_ndcg10": self.group_ndcg10,
            "dedup_condition_count": self.dedup_condition_count,
            "dedup_condition_correct": self.dedup_condition_correct,
            "raw_candidate_count": self.raw_candidate_count,
            "valid_structure_count": self.valid_structure_count,
            "valid_pid_count": self.valid_pid_count,
            "duplicate_count": self.duplicate_count,
            "returned_count_hist": {
                str(key): value for key, value in self.returned_count_hist.items()
            },
            "generated_length_hist": {
                str(key): value for key, value in self.generated_length_hist.items()
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "RetrievalMetricsAccumulator":
        instance = cls(
            available_candidates=int(state["available_candidates"]),
            ks=[int(k) for k in state["ks"]],
        )
        instance.sample_count = int(state["sample_count"])
        instance.hit_sums = {k: int(state["hit_sums"][str(k)]) for k in instance.ks}
        instance.ndcg_sums = {k: float(state["ndcg_sums"][str(k)]) for k in instance.ks}
        instance.layer_correct = np.asarray(state["layer_correct"], dtype=np.int64)
        for name in (
            "gid_exact",
            "sid_exact",
            "base_exact",
            "final_exact",
            "dedup_condition_count",
            "dedup_condition_correct",
            "raw_candidate_count",
            "valid_structure_count",
            "valid_pid_count",
            "duplicate_count",
        ):
            setattr(instance, name, int(state[name]))
        instance.group_counts = {
            key: int(value) for key, value in state["group_counts"].items()
        }
        instance.group_hits = {
            group: {k: int(values[str(k)]) for k in instance.ks}
            for group, values in state["group_hits"].items()
        }
        instance.group_ndcg10 = {
            key: float(value) for key, value in state["group_ndcg10"].items()
        }
        instance.returned_count_hist = Counter(
            {
                int(key): int(value)
                for key, value in state["returned_count_hist"].items()
            }
        )
        instance.generated_length_hist = Counter(
            {
                int(key): int(value)
                for key, value in state["generated_length_hist"].items()
            }
        )
        return instance

    def finalize(self) -> dict[str, Any]:
        if self.sample_count == 0:
            raise GenerativeEvalError("不能汇总空评测")
        retrieval: dict[str, float | None] = {}
        for k in self.ks:
            if self.available_candidates < k:
                retrieval[f"hr@{k}"] = None
                retrieval[f"ndcg@{k}"] = None
            else:
                retrieval[f"hr@{k}"] = self.hit_sums[k] / self.sample_count
                retrieval[f"ndcg@{k}"] = self.ndcg_sums[k] / self.sample_count
        if retrieval.get("hr@1") != retrieval.get("ndcg@1"):
            raise GenerativeEvalError("NDCG@1 必须严格等于 HR@1")
        if retrieval.get("hr@1") != self.final_exact / self.sample_count:
            raise GenerativeEvalError("Final PID Exact Match 必须等于 HR@1")

        groups: dict[str, Any] = {}
        for group, count in self.group_counts.items():
            groups[group] = {
                "sample_count": count,
                **{
                    f"hr@{k}": (
                        None
                        if self.available_candidates < k
                        else self.group_hits[group][k] / count
                        if count
                        else None
                    )
                    for k in self.ks
                },
                "ndcg@10": (
                    self.group_ndcg10[group] / count
                    if count and self.available_candidates >= 10
                    else None
                ),
            }

        returned_total = sum(
            count * frequency for count, frequency in self.returned_count_hist.items()
        )
        generated_total = sum(
            length * frequency
            for length, frequency in self.generated_length_hist.items()
        )
        generated_count = sum(self.generated_length_hist.values())
        return {
            "sample_count": self.sample_count,
            **retrieval,
            "generation_validity": {
                "valid_structure_ratio": (
                    self.valid_structure_count / self.raw_candidate_count
                    if self.raw_candidate_count
                    else None
                ),
                "valid_pid_ratio": (
                    self.valid_pid_count / self.raw_candidate_count
                    if self.raw_candidate_count
                    else None
                ),
                "returned_candidate_count_mean": returned_total / self.sample_count,
                "returned_candidate_count_p50": self._hist_percentile(
                    self.returned_count_hist, 0.50
                ),
                "returned_candidate_count_p90": self._hist_percentile(
                    self.returned_count_hist, 0.90
                ),
                "returned_candidate_count_min": min(self.returned_count_hist),
                "duplicate_candidate_ratio": (
                    self.duplicate_count / self.raw_candidate_count
                    if self.raw_candidate_count
                    else None
                ),
            },
            "top1_diagnostics": {
                **{
                    f"g{index + 1}_token_accuracy": (
                        int(self.layer_correct[index]) / self.sample_count
                    )
                    for index in range(6)
                },
                "gid6_exact_match": self.gid_exact / self.sample_count,
                **{
                    f"s{index + 1}_token_accuracy": (
                        int(self.layer_correct[index + 6]) / self.sample_count
                    )
                    for index in range(3)
                },
                "sid3_exact_match": self.sid_exact / self.sample_count,
                "base_pid9_exact_match": self.base_exact / self.sample_count,
                "final_pid_exact_match": self.final_exact / self.sample_count,
            },
            "groups": groups,
            "dedup_conditional_accuracy": {
                "condition_sample_count": self.dedup_condition_count,
                "correct_count": self.dedup_condition_correct,
                "accuracy": (
                    self.dedup_condition_correct / self.dedup_condition_count
                    if self.dedup_condition_count
                    else None
                ),
            },
            "average_generation_length": (
                generated_total / generated_count if generated_count else None
            ),
        }


def classify_error_types(
    target_pid: Sequence[int],
    candidates: Sequence[RankedCandidate],
) -> list[str]:
    target = tuple(int(value) for value in target_pid)
    top1 = candidates[0].pid_tokens if candidates else ()
    target_rank = next(
        (
            index
            for index, candidate in enumerate(candidates, start=1)
            if candidate.pid_tokens == target
        ),
        None,
    )
    result: list[str] = []
    if len(top1) < 6 or top1[:6] != target[:6]:
        result.append("top1_gid_error")
    elif len(top1) < 9 or top1[6:9] != target[6:9]:
        result.append("gid_correct_sid_error")
    elif len(target) == 10 and top1[:9] == target[:9] and top1 != target:
        result.append("base_pid_correct_dedup_error")
    if target_rank is not None and 1 < target_rank <= 10:
        result.append("target_in_top10_not_top1")
    if target_rank is None or target_rank > 10:
        result.append("top10_miss")
    return result


def parse_user_fields(user_content: str) -> tuple[str, str]:
    try:
        query, user_gid = extract_current_query_and_gid(user_content)
    except GenPoiProximityError:
        return "", ""
    return query, "".join(f"<G_{value}>" for value in user_gid)


def load_lf_tokenizer_and_template(
    tokenizer_path: Path,
    *,
    project_root: Path,
) -> tuple[Any, Any]:
    """Load the exact local LLaMA-Factory 0.9.4 qwen3_nothink template."""

    source = project_root / "third_party" / "LLaMA-Factory" / "src"
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        from llamafactory.data import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments, ModelArguments
        from llamafactory.model import load_tokenizer
    except ImportError as error:
        raise GenerativeEvalError("无法导入本地 LLaMA-Factory") from error

    model_args = ModelArguments(
        model_name_or_path=str(tokenizer_path.resolve()),
        use_fast_tokenizer=True,
        trust_remote_code=False,
    )
    module = load_tokenizer(model_args)
    tokenizer = module["tokenizer"]
    data_args = DataArguments(
        template="qwen3_nothink",
        train_on_prompt=False,
        cutoff_len=128,
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, template


def encode_prompt_like_training(
    *,
    tokenizer: Any,
    template: Any,
    user_content: str,
    target_content: str,
    cutoff_len: int,
) -> tuple[list[int], list[int]]:
    """Return the exact retained training source IDs and unformatted PID IDs."""

    try:
        from llamafactory.data.processor.processor_utils import infer_seqlen
    except ImportError as error:
        raise GenerativeEvalError("无法导入 LLaMA-Factory infer_seqlen") from error
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target_content},
    ]
    source_ids, formatted_target_ids = template.encode_oneturn(
        tokenizer,
        messages,
        system=None,
        tools=None,
    )
    source_len, target_len = infer_seqlen(
        len(source_ids),
        len(formatted_target_ids),
        cutoff_len,
    )
    if target_len != len(formatted_target_ids):
        raise GenerativeEvalError("cutoff_len 会截断 Assistant Final PID")
    target_pid_ids = tokenizer.encode(target_content, add_special_tokens=False)
    return source_ids[:source_len], target_pid_ids


def target_pid_is_confined_to_causal_history(
    user_content: str,
    target_content: str,
) -> bool:
    """Return whether a repeated target PID occurs only in history POI fields."""

    if (
        user_content.count("<HISTORY>") != 1
        or user_content.count("</HISTORY>") != 1
        or user_content.count("<CURRENT>") != 1
        or user_content.count("</CURRENT>") != 1
    ):
        return False
    history_open, remainder = user_content.split("<HISTORY>", maxsplit=1)
    history_content, current_content = remainder.split("</HISTORY>", maxsplit=1)
    if history_open.strip() or target_content in current_content:
        return False
    pid_fields = re.findall(r"<POI_PID>(.*?)</POI_PID>", history_content)
    if not any(target_content in value for value in pid_fields):
        return False
    history_without_pid_fields = re.sub(
        r"<POI_PID>.*?</POI_PID>",
        "",
        history_content,
    )
    return target_content not in history_without_pid_fields


def validate_and_encode_record(
    record: Mapping[str, Any],
    *,
    expected_split: str,
    tokenizer: Any,
    template: Any,
    cutoff_len: int,
) -> PromptExample:
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(messages[0], Mapping)
        or not isinstance(messages[1], Mapping)
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
        or not isinstance(messages[0].get("content"), str)
        or not isinstance(messages[1].get("content"), str)
    ):
        raise GenerativeEvalError("评测样本 Messages 必须严格为 user + assistant")
    if record.get("split") != expected_split:
        raise GenerativeEvalError("评测样本 split 与当前运行不一致")
    user_content = messages[0]["content"]
    target_content = messages[1]["content"]
    if "<D_-1>" in target_content:
        raise GenerativeEvalError("目标 PID 不得包含 <D_-1>")
    requires_dedup = record.get("requires_dedup")
    if not isinstance(requires_dedup, bool):
        raise GenerativeEvalError("requires_dedup 必须为 bool")
    prompt_ids, target_pid_ids = encode_prompt_like_training(
        tokenizer=tokenizer,
        template=template,
        user_content=user_content,
        target_content=target_content,
        cutoff_len=cutoff_len,
    )
    expected_length = 10 if requires_dedup else 9
    if len(target_pid_ids) != expected_length:
        raise GenerativeEvalError(
            f"目标 PID Token 数应为 {expected_length}，实际为 {len(target_pid_ids)}"
        )
    parsed_tokens = PID_TOKEN_PATTERN.findall(target_content)
    if (
        len(parsed_tokens) != expected_length
        or "".join(parsed_tokens) != target_content
    ):
        raise GenerativeEvalError("Assistant content 不是纯 Final PID Token 序列")
    decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    target_present_in_prompt = target_content in decoded_prompt
    target_present_in_causal_history = False
    if target_present_in_prompt:
        target_present_in_causal_history = target_pid_is_confined_to_causal_history(
            user_content,
            target_content,
        )
        if not target_present_in_causal_history:
            raise GenerativeEvalError("评测 Prompt 泄露目标 Final PID")
    sample_id = record.get("sample_id")
    target_poi_id = record.get("target_poi_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise GenerativeEvalError("sample_id 必须为非空字符串")
    if not isinstance(target_poi_id, str) or not target_poi_id:
        raise GenerativeEvalError("target_poi_id 必须为非空字符串")
    return PromptExample(
        sample_id=sample_id,
        user_content=user_content,
        target_content=target_content,
        target_pid_tokens=tuple(int(value) for value in target_pid_ids),
        prompt_ids=tuple(int(value) for value in prompt_ids),
        target_poi_id=target_poi_id,
        requires_dedup=requires_dedup,
        target_present_in_causal_history=target_present_in_causal_history,
        split=expected_split,
    )


def validate_prompt_template(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    tokenizer: Any,
    template: Any,
    cutoff_len: int,
) -> dict[str, Any]:
    """Prove evaluation Prompt IDs match the training template on 100 rows."""

    if len(records) < 100:
        raise GenerativeEvalError("Prompt 同构校验至少需要 100 条样本")
    rows: list[dict[str, Any]] = []
    prompts: list[tuple[int, ...]] = []
    for record in records[:100]:
        example = validate_and_encode_record(
            record,
            expected_split=split,
            tokenizer=tokenizer,
            template=template,
            cutoff_len=cutoff_len,
        )
        user_slots = template.format_user.apply(
            content=example.user_content,
            idx="0",
        )
        if len(user_slots) != 1 or not isinstance(user_slots[0], str):
            raise GenerativeEvalError("qwen3_nothink User 模板结构发生变化")
        direct_ids = tokenizer.encode(user_slots[0], add_special_tokens=False)
        if direct_ids[: len(example.prompt_ids)] != list(example.prompt_ids):
            raise GenerativeEvalError("评测 Prompt Token IDs 与训练模板不一致")
        decoded = tokenizer.decode(example.prompt_ids, skip_special_tokens=False)
        if "<think>" in decoded or "</think>" in decoded:
            raise GenerativeEvalError("qwen3_nothink Prompt 不得包含 think 标记")
        if not decoded.endswith("<|im_start|>assistant\n"):
            raise GenerativeEvalError("Assistant generation prompt 位置不正确")
        prompts.append(example.prompt_ids)
        rows.append(
            {
                "sample_id": example.sample_id,
                "prompt_token_count": len(example.prompt_ids),
                "prompt_ids_sha256": hashlib.sha256(
                    np.asarray(example.prompt_ids, dtype=np.int32).tobytes()
                ).hexdigest(),
                "target_absent": not example.target_present_in_causal_history,
                "target_present_only_in_causal_history": (
                    example.target_present_in_causal_history
                ),
                "assistant_generation_prompt_correct": True,
                "thinking_disabled": True,
            }
        )

    width = max(len(ids) for ids in prompts)
    attention_masks = [[0] * (width - len(ids)) + [1] * len(ids) for ids in prompts]
    if tokenizer.padding_side != "left":
        raise GenerativeEvalError("批量 generate 必须使用 left padding")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise GenerativeEvalError("pad_token_id 和 eos_token_id 必须存在")
    if any(
        mask != sorted(mask) or sum(mask) != len(prompt)
        for mask, prompt in zip(attention_masks, prompts)
    ):
        raise GenerativeEvalError("Attention mask 与左 Padding 不一致")
    return {
        "schema_version": "qwen3-nothink-prompt-validation-v1",
        "status": "passed",
        "validated_sample_count": 100,
        "split": split,
        "template": "qwen3_nothink",
        "thinking_enabled": False,
        "cutoff_len": cutoff_len,
        "padding_side": tokenizer.padding_side,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "checks": rows,
    }


def validate_split_manifest(
    data_file: Path,
    *,
    split: str,
    verify_hash: bool = True,
) -> tuple[dict[str, Any], int, str]:
    if split not in ("valid", "test"):
        raise GenerativeEvalError("split 只允许 valid 或 test")
    data_file = data_file.resolve()
    manifest_path = data_file.parent / "manifest.json"
    if not manifest_path.is_file():
        raise GenerativeEvalError(f"SFT manifest 不存在：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GenerativeEvalError("SFT manifest JSON 解析失败") from error
    filename = f"{split}.jsonl"
    spec = manifest.get("outputs", {}).get(filename)
    if not isinstance(spec, Mapping):
        raise GenerativeEvalError(f"SFT manifest 缺少 {filename}")
    expected_path = (manifest_path.parent / filename).resolve()
    if data_file != expected_path:
        raise GenerativeEvalError(f"{split} 文件路径与 SFT manifest 不一致")
    expected_count = 597_421 if split == "valid" else 606_682
    if spec.get("rows") != expected_count:
        raise GenerativeEvalError(
            f"{split} 样本数必须为 {expected_count:,}，manifest 为 {spec.get('rows')}"
        )
    split_rule = manifest.get("time_split", {})
    expected_date = "2026-07-13" if split == "valid" else "2026-07-14"
    declared_date = split_rule.get(split)
    if declared_date != expected_date:
        raise GenerativeEvalError(f"{split} 日期必须为 {expected_date}")
    expected_hash = spec.get("sha256")
    if not isinstance(expected_hash, str):
        raise GenerativeEvalError(f"{split} manifest 缺少 SHA256")
    if verify_hash and sha256_file(data_file) != expected_hash:
        raise GenerativeEvalError(f"{split} 文件 SHA256 与 manifest 不一致")
    return manifest, expected_count, expected_hash


def build_fixed_validation_subset(
    valid_file: Path,
    output_dir: Path,
    *,
    source_rows: int,
    source_sha256: str,
    subset_size: int,
) -> FixedValidationSubset:
    """Persist a deterministic Validation subset selected by smallest sample IDs."""

    if subset_size <= 0 or subset_size > source_rows:
        raise GenerativeEvalError("Validation 子集大小必须位于 (0, source_rows]")
    valid_file = valid_file.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"validation_subset_{subset_size}.jsonl"
    manifest_path = output_dir / f"validation_subset_{subset_size}_manifest.json"

    if data_path.exists() or manifest_path.exists():
        if not data_path.is_file() or not manifest_path.is_file():
            raise GenerativeEvalError("Validation 子集文件与 manifest 必须同时存在")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise GenerativeEvalError(
                "Validation 子集 manifest JSON 解析失败"
            ) from error
        expected = {
            "schema_version": "fixed-validation-subset-v1",
            "status": "completed",
            "source_file": str(valid_file),
            "source_rows": source_rows,
            "source_sha256": source_sha256,
            "subset_size": subset_size,
            "selection_method": "lowest_sample_id_lexicographic",
            "output_order": "source_row_ascending",
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise GenerativeEvalError("已有 Validation 子集 manifest 与当前输入不一致")
        actual_hash = sha256_file(data_path)
        if manifest.get("output_sha256") != actual_hash:
            raise GenerativeEvalError("已有 Validation 子集 SHA256 与 manifest 不一致")
        with data_path.open("rb") as handle:
            actual_rows = sum(1 for _ in handle)
        if actual_rows != subset_size or manifest.get("output_rows") != subset_size:
            raise GenerativeEvalError("已有 Validation 子集行数与 manifest 不一致")
        return FixedValidationSubset(
            data_path=data_path,
            manifest_path=manifest_path,
            row_count=subset_size,
            sha256=actual_hash,
            manifest=manifest,
        )

    def sample_id_rows() -> Any:
        with valid_file.open("rb") as handle:
            for row_index, raw_line in enumerate(handle):
                try:
                    record = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise GenerativeEvalError(
                        f"Validation 第 {row_index + 1} 行 JSON 非法"
                    ) from error
                sample_id = record.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise GenerativeEvalError(
                        f"Validation 第 {row_index + 1} 行 sample_id 非法"
                    )
                yield sample_id, row_index

    selected = heapq.nsmallest(
        subset_size,
        sample_id_rows(),
        key=lambda item: (item[0], item[1]),
    )
    if len(selected) != subset_size:
        raise GenerativeEvalError(
            f"Validation 实际行数不足 {subset_size:,}，无法构建固定子集"
        )
    selected_ids = [item[0] for item in selected]
    if len(set(selected_ids)) != subset_size:
        raise GenerativeEvalError("选中的 Validation sample_id 存在重复")
    selected_rows = sorted(item[1] for item in selected)
    selected_row_set = set(selected_rows)
    temporary = data_path.with_name(f".{data_path.name}.tmp")
    digest = hashlib.sha256()
    written_rows = 0
    with valid_file.open("rb") as source, temporary.open("wb") as destination:
        for row_index, raw_line in enumerate(source):
            if row_index not in selected_row_set:
                continue
            destination.write(raw_line)
            digest.update(raw_line)
            written_rows += 1
        destination.flush()
        os.fsync(destination.fileno())
    if written_rows != subset_size:
        temporary.unlink(missing_ok=True)
        raise GenerativeEvalError(
            f"Validation 子集写入 {written_rows:,}/{subset_size:,} 行"
        )
    os.replace(temporary, data_path)
    indices_bytes = np.asarray(selected_rows, dtype=np.int64).tobytes()
    sample_ids_bytes = "".join(f"{value}\n" for value in selected_ids).encode("utf-8")
    manifest = {
        "schema_version": "fixed-validation-subset-v1",
        "status": "completed",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "split": "valid",
        "date": "2026-07-13",
        "source_file": str(valid_file),
        "source_rows": source_rows,
        "source_sha256": source_sha256,
        "subset_size": subset_size,
        "selection_method": "lowest_sample_id_lexicographic",
        "selection_tie_break": "source_row_ascending",
        "output_order": "source_row_ascending",
        "selected_row_indices_sha256": hashlib.sha256(indices_bytes).hexdigest(),
        "selected_sample_ids_sha256": hashlib.sha256(sample_ids_bytes).hexdigest(),
        "output_file": str(data_path),
        "output_rows": written_rows,
        "output_sha256": digest.hexdigest(),
    }
    atomic_write_json(manifest_path, manifest)
    return FixedValidationSubset(
        data_path=data_path,
        manifest_path=manifest_path,
        row_count=subset_size,
        sha256=digest.hexdigest(),
        manifest=manifest,
    )


def build_reference_aligned_validation_subset(
    valid_file: Path,
    reference_subset: Path,
    output_dir: Path,
    *,
    source_rows: int,
    source_sha256: str,
) -> FixedValidationSubset:
    """Align a method-specific Validation file to a frozen business-key subset."""

    valid_file = valid_file.resolve()
    reference_subset = reference_subset.resolve()
    output_dir = output_dir.resolve()
    if not reference_subset.is_file():
        raise GenerativeEvalError(f"参考 Validation 子集不存在：{reference_subset}")
    reference_sha256 = sha256_file(reference_subset)

    def identity(record: Mapping[str, Any], source: str) -> tuple[str, str]:
        values: list[str] = []
        for field in ("order_id", "searchid"):
            value = record.get(field)
            if value is None or not str(value).strip():
                raise GenerativeEvalError(f"{source} 缺少有效 {field}")
            values.append(str(value))
        return values[0], values[1]

    reference_keys: list[tuple[str, str]] = []
    reference_targets: dict[tuple[str, str], str] = {}
    with reference_subset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise GenerativeEvalError(
                    f"参考 Validation 第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(record, Mapping):
                raise GenerativeEvalError("参考 Validation 每行必须是 object")
            key = identity(record, f"参考 Validation 第 {line_number} 行")
            if key in reference_targets:
                raise GenerativeEvalError(f"参考 Validation 业务主键重复：{key}")
            target = record.get("target_poi_id")
            if target is None or not str(target).strip():
                raise GenerativeEvalError(
                    f"参考 Validation 第 {line_number} 行缺少 target_poi_id"
                )
            reference_keys.append(key)
            reference_targets[key] = str(target)
    if not reference_keys:
        raise GenerativeEvalError("参考 Validation 子集不能为空")

    subset_size = len(reference_keys)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"validation_subset_{subset_size}.jsonl"
    manifest_path = output_dir / f"validation_subset_{subset_size}_manifest.json"
    expected_manifest = {
        "schema_version": "reference-aligned-validation-subset-v1",
        "status": "completed",
        "source_file": str(valid_file),
        "source_rows": source_rows,
        "source_sha256": source_sha256,
        "reference_file": str(reference_subset),
        "reference_rows": subset_size,
        "reference_sha256": reference_sha256,
        "subset_size": subset_size,
        "selection_method": "exact_order_id_searchid_match",
        "output_order": "reference_row_ascending",
    }
    if data_path.exists() or manifest_path.exists():
        if not data_path.is_file() or not manifest_path.is_file():
            raise GenerativeEvalError("Validation 子集文件与 manifest 必须同时存在")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise GenerativeEvalError(
                "Validation 子集 manifest JSON 解析失败"
            ) from error
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise GenerativeEvalError("已有 Validation 子集 manifest 与当前输入不一致")
        output_sha256 = sha256_file(data_path)
        if manifest.get("output_sha256") != output_sha256:
            raise GenerativeEvalError("已有 Validation 子集 SHA256 与 manifest 不一致")
        with data_path.open("rb") as handle:
            actual_rows = sum(1 for _ in handle)
        if actual_rows != subset_size or manifest.get("output_rows") != subset_size:
            raise GenerativeEvalError("已有 Validation 子集行数与 manifest 不一致")
        return FixedValidationSubset(
            data_path=data_path,
            manifest_path=manifest_path,
            row_count=subset_size,
            sha256=output_sha256,
            manifest=manifest,
        )

    matched_lines: dict[tuple[str, str], bytes] = {}
    actual_source_rows = 0
    with valid_file.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            actual_source_rows += 1
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise GenerativeEvalError(
                    f"Validation 第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(record, Mapping):
                raise GenerativeEvalError("Validation 每行必须是 object")
            key = identity(record, f"Validation 第 {line_number} 行")
            if key not in reference_targets:
                continue
            if key in matched_lines:
                raise GenerativeEvalError(f"Validation 业务主键重复：{key}")
            target = record.get("target_poi_id")
            if str(target) != reference_targets[key]:
                raise GenerativeEvalError(f"同一业务主键的目标 POI 不一致：{key}")
            matched_lines[key] = raw_line
    if actual_source_rows != source_rows:
        raise GenerativeEvalError(
            f"Validation 实际行数 {actual_source_rows:,} 与 manifest {source_rows:,} 不一致"
        )
    missing = [key for key in reference_keys if key not in matched_lines]
    if missing:
        raise GenerativeEvalError(
            f"Validation 缺少 {len(missing):,} 条参考样本，例如：{missing[0]}"
        )

    temporary = data_path.with_name(f".{data_path.name}.tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as destination:
        for key in reference_keys:
            raw_line = matched_lines[key]
            destination.write(raw_line)
            digest.update(raw_line)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, data_path)
    key_bytes = "".join(
        f"{order_id}\t{searchid}\n" for order_id, searchid in reference_keys
    ).encode("utf-8")
    manifest = {
        **expected_manifest,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "split": "valid",
        "date": "2026-07-13",
        "business_keys_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "target_poi_mismatch_count": 0,
        "output_file": str(data_path),
        "output_rows": subset_size,
        "output_sha256": digest.hexdigest(),
    }
    atomic_write_json(manifest_path, manifest)
    return FixedValidationSubset(
        data_path=data_path,
        manifest_path=manifest_path,
        row_count=subset_size,
        sha256=digest.hexdigest(),
        manifest=manifest,
    )


def load_ssp_predictions(
    predictions_dir: Path,
    *,
    evaluation_data_sha256: str,
    expected_rows: int,
) -> tuple[dict[str, SspPrediction], dict[str, Any]]:
    """Validate and load target-free SSP predictions for one evaluation file."""

    import pyarrow.parquet as pq

    predictions_dir = predictions_dir.resolve()
    manifest_path = predictions_dir / "manifest.json"
    predictions_path = predictions_dir / "predictions.parquet"
    if not manifest_path.is_file() or not predictions_path.is_file():
        raise GenerativeEvalError("SSP predictions 目录缺少 manifest 或 Parquet")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GenerativeEvalError("SSP predictions manifest JSON 非法") from error
    schema_version = manifest.get("schema_version")
    if schema_version not in {
        "genpoi-proximity-predictions-v1",
        "genpoi-beijing-ssp-predictions-v1",
    }:
        raise GenerativeEvalError("SSP predictions schema_version 不兼容")
    expected_manifest = {
        "status": "completed",
        "rows": expected_rows,
        "limit": None,
        "source_validation_jsonl_sha256": evaluation_data_sha256,
        "target_fields_used": [],
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise GenerativeEvalError("SSP predictions 与当前正式评测数据不一致")
    if Path(manifest.get("output_file", "")).resolve() != predictions_path:
        raise GenerativeEvalError("SSP predictions 输出路径与 manifest 不一致")
    if sha256_file(predictions_path) != manifest.get("output_sha256"):
        raise GenerativeEvalError("SSP predictions Parquet SHA256 不一致")
    table = pq.read_table(
        predictions_path,
        columns=["sample_id", "user_gid", "predicted_lambda", "prefill_length"],
    )
    if table.num_rows != expected_rows:
        raise GenerativeEvalError("SSP predictions Parquet 行数不一致")
    values: dict[str, SspPrediction] = {}
    for sample_id, user_gid, predicted_lambda, prefill_length in zip(
        table["sample_id"].to_pylist(),
        table["user_gid"].to_pylist(),
        table["predicted_lambda"].to_pylist(),
        table["prefill_length"].to_pylist(),
    ):
        if not isinstance(sample_id, str) or not sample_id:
            raise GenerativeEvalError("SSP prediction sample_id 非法")
        if sample_id in values:
            raise GenerativeEvalError(f"SSP prediction sample_id 重复：{sample_id}")
        if not isinstance(user_gid, str) or len(user_gid) != 6:
            raise GenerativeEvalError(f"SSP prediction user_gid 非法：{sample_id}")
        level = int(predicted_lambda)
        prefix = int(prefill_length)
        if schema_version == "genpoi-proximity-predictions-v1":
            if manifest.get("gamma") != 2:
                raise GenerativeEvalError("论文原版 SSP predictions gamma 必须为 2")
            if prefix != effective_prefix_length(level, gamma=2):
                raise GenerativeEvalError(f"SSP prediction 前缀长度非法：{sample_id}")
        else:
            expected_beijing_contract = {
                "selection_strategy": "direct_ordinal_safe_prefix",
                "useful_prefix_depths": [3, 4, 5, 6],
            }
            if any(
                manifest.get(key) != value
                for key, value in expected_beijing_contract.items()
            ):
                raise GenerativeEvalError("北京 SSP predictions 安全前缀契约不一致")
            if prefix not in {0, 3, 4, 5, 6} or level != prefix:
                raise GenerativeEvalError(f"北京 SSP prediction 前缀长度非法：{sample_id}")
        values[sample_id] = SspPrediction(
            user_gid=tuple(user_gid),
            predicted_lambda=level,
            prefill_length=prefix,
        )
    return values, manifest


def authorize_test_file(
    selected_config_path: Path,
    test_file: Path,
) -> dict[str, Any]:
    """Validate the frozen selection gate before any Test-file access."""

    if not selected_config_path.is_file():
        raise GenerativeEvalError("selected_config 不存在，禁止读取 Test 文件")
    try:
        selected = json.loads(selected_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GenerativeEvalError("selected_config JSON 解析失败") from error
    if not isinstance(selected, dict) or selected.get("status") != "frozen":
        raise GenerativeEvalError("selected_config 尚未冻结，禁止读取 Test 文件")
    if not test_file.is_file():
        raise GenerativeEvalError(f"Test 文件不存在：{test_file}")
    return selected


def validate_checkpoints(
    checkpoints: Sequence[Path],
    *,
    tokenizer_path: Path,
    expected_steps: Sequence[int] = (2290, 4580, 6870, 9158),
    expected_epochs: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    expected_vocab_size: int | None = None,
) -> list[dict[str, Any]]:
    if not checkpoints or len(checkpoints) != len(expected_steps):
        raise GenerativeEvalError("Checkpoint 数量必须与 expected_steps 一致且非空")
    if len(expected_epochs) != len(expected_steps):
        raise GenerativeEvalError("expected_epochs 数量必须与 expected_steps 一致")
    mapping_payload = json.loads(
        (tokenizer_path / "poi_token_mapping.json").read_text(encoding="utf-8")
    )
    mapping = mapping_payload["tokens"]
    vocab_size = expected_vocab_size or mapping_payload.get("new_vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise GenerativeEvalError("无法从 Token mapping 确定预期 vocab_size")
    result: list[dict[str, Any]] = []
    for checkpoint, expected_step, expected_epoch in zip(
        checkpoints, expected_steps, expected_epochs
    ):
        checkpoint = checkpoint.resolve()
        if checkpoint.name != f"checkpoint-{expected_step}":
            raise GenerativeEvalError(f"Checkpoint 顺序或步数错误：{checkpoint.name}")
        required = (
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "trainer_state.json",
        )
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise GenerativeEvalError(f"{checkpoint} 缺少文件：{', '.join(missing)}")
        state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        actual_epoch = float(state.get("epoch", -1))
        if abs(actual_epoch - expected_epoch) > 0.001:
            raise GenerativeEvalError(
                f"{checkpoint.name} epoch {actual_epoch} 与 {expected_epoch} 不一致"
            )
        added = json.loads(
            (checkpoint / "added_tokens.json").read_text(encoding="utf-8")
        )
        for token, token_id in mapping.items():
            if added.get(token) != token_id:
                raise GenerativeEvalError(f"{checkpoint.name} Token ID 不一致：{token}")
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if config.get("vocab_size") != vocab_size:
            raise GenerativeEvalError(f"{checkpoint.name} vocab_size 不一致")
        eval_loss = None
        for item in state.get("log_history", []):
            if item.get("step") == expected_step and "eval_loss" in item:
                eval_loss = float(item["eval_loss"])
        if eval_loss is None:
            final_eval_path = checkpoint.parent / "eval_results.json"
            if final_eval_path.is_file():
                final_eval = json.loads(final_eval_path.read_text(encoding="utf-8"))
                if abs(float(final_eval.get("epoch", -1)) - actual_epoch) <= 0.001:
                    eval_loss = float(final_eval["eval_loss"])
        if eval_loss is None:
            raise GenerativeEvalError(f"{checkpoint.name} 缺少对应完整 Validation Loss")
        result.append(
            {
                "path": str(checkpoint),
                "name": checkpoint.name,
                "step": expected_step,
                "epoch": actual_epoch,
                "model_sha256": sha256_file(checkpoint / "model.safetensors"),
                "config_sha256": sha256_file(checkpoint / "config.json"),
                "tokenizer_json_sha256": sha256_file(checkpoint / "tokenizer.json"),
                "validation_loss": eval_loss,
            }
        )
    tokenizer_hashes = {item["tokenizer_json_sha256"] for item in result}
    if len(tokenizer_hashes) != 1:
        raise GenerativeEvalError("各 checkpoint 的 tokenizer 不一致")
    return result


def select_best_checkpoint(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not results:
        raise GenerativeEvalError("Checkpoint 选择至少需要一个完整 Validation 结果")
    if any(item.get("status") != "completed" for item in results):
        raise GenerativeEvalError("存在未完成的 Checkpoint Validation 结果")

    def key(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
        metrics = item["metrics"]
        eval_loss = item.get(
            "validation_loss", item.get("config", {}).get("validation_loss")
        )
        return (
            float(metrics["ndcg@10"]),
            float(metrics["hr@1"]),
            float(metrics["hr@10"]),
            -float(eval_loss) if eval_loss is not None else -math.inf,
        )

    return max(results, key=key)


def write_results_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise GenerativeEvalError("CSV 结果不能为空")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


class RunProgressStore:
    """Atomic, signature-bound chunk progress that never double-counts rows."""

    def __init__(
        self,
        run_dir: Path,
        *,
        config: Mapping[str, Any],
        available_candidates: int,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.run_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        self.config = dict(config)
        self.signature = canonical_sha256(self.config)
        self.progress_path = self.run_dir / "progress.json"
        self.config_path = self.run_dir / "run_config.json"
        self.available_candidates = available_candidates
        if self.progress_path.exists():
            state = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if state.get("config_signature") != self.signature:
                raise GenerativeEvalError(
                    f"运行配置已变化，不能复用旧进度：{self.run_dir}"
                )
            self.state = state
        else:
            self.state = {
                "schema_version": "generative-eval-progress-v1",
                "status": "running",
                "config_signature": self.signature,
                "next_line": 0,
                "next_byte": 0,
                "chunk_count": 0,
                "metrics_state": RetrievalMetricsAccumulator(
                    available_candidates=available_candidates
                ).to_state(),
                "error_cases": {name: [] for name in ERROR_TYPES},
                "inference_seconds": 0.0,
                "generated_token_count": 0,
                "actual_batch_size": None,
                "peak_memory_bytes": 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(
                self.config_path, {**self.config, "signature": self.signature}
            )
            atomic_write_json(self.progress_path, self.state)

    @property
    def next_line(self) -> int:
        return int(self.state["next_line"])

    @property
    def next_byte(self) -> int:
        return int(self.state["next_byte"])

    @property
    def metrics(self) -> RetrievalMetricsAccumulator:
        return RetrievalMetricsAccumulator.from_state(self.state["metrics_state"])

    @property
    def error_cases(self) -> dict[str, list[dict[str, Any]]]:
        return {
            name: list(self.state["error_cases"].get(name, [])) for name in ERROR_TYPES
        }

    def commit_chunk(
        self,
        *,
        start_line: int,
        end_line: int,
        end_byte: int,
        metrics: RetrievalMetricsAccumulator,
        error_cases: Mapping[str, Sequence[Mapping[str, Any]]],
        inference_seconds: float,
        generated_token_count: int,
        actual_batch_size: int,
        peak_memory_bytes: int,
    ) -> None:
        if start_line != self.next_line or end_line <= start_line:
            raise GenerativeEvalError("分片提交行号不连续，拒绝重复计数")
        if metrics.sample_count != end_line - start_line:
            raise GenerativeEvalError("分片指标样本数与行号范围不一致")
        chunk_index = int(self.state["chunk_count"])
        chunk_payload = {
            "config_signature": self.signature,
            "chunk_index": chunk_index,
            "start_line": start_line,
            "end_line": end_line,
            "end_byte": end_byte,
            "metrics_state": metrics.to_state(),
            "error_cases": {
                name: list(error_cases.get(name, [])) for name in ERROR_TYPES
            },
            "inference_seconds": inference_seconds,
            "generated_token_count": generated_token_count,
            "actual_batch_size": actual_batch_size,
            "peak_memory_bytes": peak_memory_bytes,
        }
        chunk_path = self.chunks_dir / (
            f"chunk_{chunk_index:06d}_{start_line:09d}_{end_line:09d}.json"
        )
        atomic_write_json(chunk_path, chunk_payload)

        cumulative = self.metrics
        cumulative.merge(metrics)
        combined_errors = self.error_cases
        for name in ERROR_TYPES:
            remaining = max(100 - len(combined_errors[name]), 0)
            combined_errors[name].extend(list(error_cases.get(name, []))[:remaining])
        self.state.update(
            {
                "next_line": end_line,
                "next_byte": end_byte,
                "chunk_count": chunk_index + 1,
                "metrics_state": cumulative.to_state(),
                "error_cases": combined_errors,
                "inference_seconds": float(self.state["inference_seconds"])
                + inference_seconds,
                "generated_token_count": int(self.state["generated_token_count"])
                + generated_token_count,
                "actual_batch_size": (
                    actual_batch_size
                    if self.state["actual_batch_size"] is None
                    else min(int(self.state["actual_batch_size"]), actual_batch_size)
                ),
                "peak_memory_bytes": max(
                    int(self.state["peak_memory_bytes"]), peak_memory_bytes
                ),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(self.progress_path, self.state)

    def finalize(self, expected_rows: int) -> dict[str, Any]:
        if self.next_line != expected_rows:
            raise GenerativeEvalError(
                f"运行仅覆盖 {self.next_line:,}/{expected_rows:,} 条样本"
            )
        metrics = self.metrics.finalize()
        seconds = float(self.state["inference_seconds"])
        result = {
            "status": "completed",
            "config_signature": self.signature,
            "config": self.config,
            "metrics": metrics,
            "performance": {
                "total_seconds": seconds,
                "samples_per_second": expected_rows / seconds if seconds else None,
                "generated_tokens_per_second": (
                    int(self.state["generated_token_count"]) / seconds
                    if seconds
                    else None
                ),
                "average_latency_seconds": (
                    seconds / expected_rows if expected_rows else None
                ),
                "actual_batch_size": self.state["actual_batch_size"],
                "peak_memory_bytes": self.state["peak_memory_bytes"],
                "trie_load_memory_bytes": self.config.get("trie_load_memory_bytes"),
                "average_generation_length": metrics["average_generation_length"],
            },
            "chunk_count": self.state["chunk_count"],
            "covered_rows": self.next_line,
            "error_case_counts": {
                name: len(values) for name, values in self.error_cases.items()
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(self.run_dir),
        }
        self.state["status"] = "completed"
        self.state["completed_at"] = result["completed_at"]
        atomic_write_json(self.progress_path, self.state)
        atomic_write_json(self.run_dir / "result.json", result)
        return result


def read_jsonl_chunk(
    path: Path,
    *,
    start_byte: int,
    max_rows: int,
) -> tuple[list[dict[str, Any]], int]:
    if max_rows <= 0:
        raise GenerativeEvalError("chunk size 必须为正整数")
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        handle.seek(start_byte)
        for _ in range(max_rows):
            raw_line = handle.readline()
            if not raw_line:
                break
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise GenerativeEvalError("评测 JSONL 包含非法行") from error
            if not isinstance(record, dict):
                raise GenerativeEvalError("评测 JSONL 每行必须是 object")
            records.append(record)
        end_byte = handle.tell()
    return records, end_byte


def freeze_selected_config(
    *,
    checkpoint_result: Mapping[str, Any],
    beam_results: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Freeze one immutable test configuration from completed full Validation."""

    eligible = [
        item
        for item in beam_results
        if item.get("status") == "completed"
        and item.get("metrics", {}).get("ndcg@10") is not None
    ]
    if not eligible:
        raise GenerativeEvalError("没有完整 Top-10 Validation Beam 结果可冻结")
    selected_beam = max(
        eligible,
        key=lambda item: (
            float(item["metrics"]["ndcg@10"]),
            float(item["metrics"]["hr@1"]),
            float(item["performance"]["samples_per_second"]),
            -float(item["performance"]["peak_memory_bytes"]),
        ),
    )
    frozen = {
        "schema_version": "sft-retrieval-selected-config-v1",
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "selection_source": "full_2026-07-13_validation",
        "checkpoint": checkpoint_result["config"]["checkpoint"],
        "checkpoint_name": checkpoint_result["config"]["checkpoint_name"],
        "epoch": checkpoint_result["config"]["epoch"],
        "num_beams": selected_beam["config"]["num_beams"],
        "num_return_sequences": selected_beam["config"]["num_return_sequences"],
        "max_new_tokens": 11,
        "length_penalty": 1.0,
        "do_sample": False,
        "early_stopping": True,
        "renormalize_logits": True,
        "dtype": "bfloat16",
        "trie_manifest_sha256": selected_beam["config"]["trie_manifest_sha256"],
        "tokenizer_json_sha256": selected_beam["config"]["tokenizer_json_sha256"],
        "validation_metrics": selected_beam["metrics"],
        "selection_rule": ("NDCG@10 desc, HR@1 desc, samples/s desc, peak memory asc"),
    }
    frozen["config_signature"] = canonical_sha256(frozen)
    atomic_write_json(output_path, frozen)
    return frozen


def load_aligned_poi_ids(mapping_path: Path, expected_rows: int) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(mapping_path, columns=["poi_id"])
    if table.num_rows != expected_rows:
        raise GenerativeEvalError("PID mapping 行数与 Trie 叶子数不一致")
    values = table["poi_id"].to_pylist()
    if any(not isinstance(value, str) or not value for value in values):
        raise GenerativeEvalError("PID mapping 包含非法 poi_id")
    return values


def _pad_prompt_batch(
    prompt_ids: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    width = max(len(values) for values in prompt_ids)
    input_ids = torch.full(
        (len(prompt_ids), width),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(prompt_ids), width),
        dtype=torch.long,
        device=device,
    )
    for index, values in enumerate(prompt_ids):
        length = len(values)
        input_ids[index, width - length :] = torch.as_tensor(
            values,
            dtype=torch.long,
            device=device,
        )
        attention_mask[index, width - length :] = 1
    return input_ids, attention_mask


def _pid_text(tokenizer: Any, pid_tokens: Sequence[int]) -> str:
    return "".join(tokenizer.convert_ids_to_tokens(list(pid_tokens)))


def _build_error_case(
    *,
    error_type: str,
    example: PromptExample,
    candidates: Sequence[RankedCandidate],
    poi_ids: Sequence[str],
    tokenizer: Any,
    checkpoint_name: str,
    beam_size: int,
) -> dict[str, Any]:
    query, user_gid = parse_user_fields(example.user_content)
    target = example.target_pid_tokens
    target_rank = next(
        (
            index
            for index, candidate in enumerate(candidates, start=1)
            if candidate.pid_tokens == target
        ),
        None,
    )
    return {
        "sample_id": example.sample_id,
        "query": query,
        "user_gid": user_gid,
        "target_poi_id": example.target_poi_id,
        "target_pid": example.target_content,
        "predicted_topk_poi_ids": [
            poi_ids[candidate.poi_row] for candidate in candidates
        ],
        "predicted_topk_pids": [
            _pid_text(tokenizer, candidate.pid_tokens) for candidate in candidates
        ],
        "sequence_scores": [candidate.score for candidate in candidates],
        "target_rank": target_rank,
        "error_type": error_type,
        "split": example.split,
        "checkpoint": checkpoint_name,
        "beam_size": beam_size,
    }


def _is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message)


def evaluate_records_chunk(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    model: Any,
    tokenizer: Any,
    template: Any,
    trie: CompactPidTrie,
    token_ids: PidTokenIds,
    poi_ids: Sequence[str],
    checkpoint_name: str,
    num_beams: int,
    num_return_sequences: int,
    top_k: int,
    cutoff_len: int,
    batch_size: int,
    existing_error_counts: Mapping[str, int],
    ssp_predictions: Mapping[str, SspPrediction] | None = None,
) -> tuple[
    RetrievalMetricsAccumulator,
    dict[str, list[dict[str, Any]]],
    float,
    int,
    int,
]:
    """Evaluate one not-yet-committed chunk; failures leave no partial state."""

    import torch

    examples = [
        validate_and_encode_record(
            record,
            expected_split=split,
            tokenizer=tokenizer,
            template=template,
            cutoff_len=cutoff_len,
        )
        for record in records
    ]
    for example in examples:
        row = trie.lookup((*example.target_pid_tokens, token_ids.eos))
        if row < 0:
            raise GenerativeEvalError(f"目标 PID 不在全量 Trie：{example.sample_id}")
        if poi_ids[row] != example.target_poi_id:
            raise GenerativeEvalError(
                f"目标 PID 与 poi_id 映射不一致：{example.sample_id}"
            )

    accumulator = RetrievalMetricsAccumulator(available_candidates=num_return_sequences)
    error_cases = {name: [] for name in ERROR_TYPES}
    generated_token_count = 0
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)

    prepared: list[tuple[PromptExample, tuple[int, ...]]] = []
    for example in examples:
        prefix: tuple[int, ...] = ()
        if ssp_predictions is not None:
            prediction = ssp_predictions.get(example.sample_id)
            if prediction is None:
                raise GenerativeEvalError(
                    f"SSP predictions 缺少样本：{example.sample_id}"
                )
            try:
                _, current_gid = extract_current_query_and_gid(example.user_content)
            except GenPoiProximityError as error:
                raise GenerativeEvalError(
                    f"无法解析 CURRENT 地理信息：{example.sample_id}"
                ) from error
            if current_gid != prediction.user_gid:
                raise GenerativeEvalError(
                    f"SSP prediction user_gid 与评测样本不一致：{example.sample_id}"
                )
            prefix = tuple(
                int(token_ids.gid[GEOHASH_ALPHABET.index(value)])
                for value in current_gid[: prediction.prefill_length]
            )
            if trie.traverse(prefix) < 0:
                raise GenerativeEvalError(
                    f"SSP 预填 GID Prefix 不在 POI Trie：{example.sample_id}"
                )
        prepared.append((example, prefix))

    groups: dict[int, list[tuple[PromptExample, tuple[int, ...]]]] = {}
    for item in prepared:
        groups.setdefault(len(item[1]), []).append(item)

    for prefix_length in sorted(groups):
        group = groups[prefix_length]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            batch_examples = [item[0] for item in batch]
            prefilled_prefixes = [item[1] for item in batch]
            model_inputs = [
                (*example.prompt_ids, *prefix)
                for example, prefix in batch
            ]
            input_ids, attention_mask = _pad_prompt_batch(
                model_inputs,
                pad_token_id=tokenizer.pad_token_id,
                device=model.device,
            )
            prompt_width = int(input_ids.shape[1])
            if ssp_predictions is None:
                constraint: Any = TriePrefixConstraint(
                    trie,
                    prompt_width,
                    tokenizer.eos_token_id,
                )
            else:
                constraint = TriePrefilledPrefixConstraint(
                    trie,
                    prompt_width,
                    tokenizer.eos_token_id,
                    prefilled_prefixes,
                )
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "do_sample": False,
                "num_beams": num_beams,
                "num_return_sequences": num_return_sequences,
                "max_new_tokens": 11 - prefix_length,
                "length_penalty": 1.0,
                "early_stopping": True if num_beams > 1 else False,
                "renormalize_logits": True,
                "return_dict_in_generate": True,
                "output_scores": True,
                "prefix_allowed_tokens_fn": constraint,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            with torch.inference_mode():
                generated = model.generate(**generation_kwargs)
            remaining_sequences = (
                generated.sequences[:, prompt_width:].detach().cpu().tolist()
            )
            sequences = [
                [*prefilled_prefixes[index // num_return_sequences], *sequence]
                for index, sequence in enumerate(remaining_sequences)
            ]
            if getattr(generated, "sequences_scores", None) is None:
                scores = [0.0] * len(sequences)
            else:
                scores = generated.sequences_scores.float().detach().cpu().tolist()
            expected_sequences = len(batch_examples) * num_return_sequences
            if len(sequences) != expected_sequences:
                raise GenerativeEvalError("generate 返回的候选数量不符合配置")

            for index, example in enumerate(batch_examples):
                begin = index * num_return_sequences
                end = begin + num_return_sequences
                candidate_batch = rank_and_deduplicate_candidates(
                    sequences[begin:end],
                    scores[begin:end],
                    trie=trie,
                    token_ids=token_ids,
                    top_k=top_k,
                )
                accumulator.update(example.target_pid_tokens, candidate_batch)
                generated_token_count += sum(candidate_batch.generated_lengths) - (
                    prefix_length * len(candidate_batch.generated_lengths)
                )
                for error_type in classify_error_types(
                    example.target_pid_tokens,
                    candidate_batch.candidates,
                ):
                    if (
                        existing_error_counts.get(error_type, 0)
                        + len(error_cases[error_type])
                        >= 100
                    ):
                        continue
                    error_cases[error_type].append(
                        _build_error_case(
                            error_type=error_type,
                            example=example,
                            candidates=candidate_batch.candidates,
                            poi_ids=poi_ids,
                            tokenizer=tokenizer,
                            checkpoint_name=checkpoint_name,
                            beam_size=num_beams,
                        )
                    )
            del generated, input_ids, attention_mask

    elapsed = time.monotonic() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(model.device))
        if torch.cuda.is_available()
        else 0
    )
    return (
        accumulator,
        error_cases,
        elapsed,
        generated_token_count,
        peak_memory,
    )


def load_generation_model(checkpoint: Path, *, expected_vocab_size: int) -> Any:
    """Load one full-finetuned checkpoint on a single CUDA device."""

    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise GenerativeEvalError("完整生成评测需要 CUDA GPU，当前环境不可用")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(torch.device("cuda:0"))
    model.eval()
    if model.config.vocab_size != expected_vocab_size:
        raise GenerativeEvalError(
            "Checkpoint 模型词表大小与评测 tokenizer 不一致："
            f"{model.config.vocab_size} != {expected_vocab_size}"
        )
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model


def run_full_evaluation(
    *,
    data_file: Path,
    split: str,
    expected_rows: int,
    data_sha256: str,
    checkpoint: Path,
    epoch: float,
    validation_loss: float | None,
    tokenizer_path: Path,
    trie_dir: Path,
    mapping_path: Path,
    output_root: Path,
    project_root: Path,
    num_beams: int,
    num_return_sequences: int,
    top_k: int,
    initial_batch_size: int,
    chunk_size: int,
    cutoff_len: int = 128,
    smoke_limit: int | None = None,
    dataset_context: Mapping[str, Any] | None = None,
    ssp_predictions_dir: Path | None = None,
) -> dict[str, Any]:
    """Run or resume one signature-bound checkpoint/beam evaluation."""

    if num_beams < num_return_sequences:
        raise GenerativeEvalError("num_beams 不得小于 num_return_sequences")
    if initial_batch_size not in (128, 64, 32, 16):
        raise GenerativeEvalError("Eval Batch Size 只允许 128、64、32、16")
    checkpoint = checkpoint.resolve()
    checkpoint_name = checkpoint.name
    trie_manifest_path = trie_dir.resolve() / "trie_manifest.json"
    if not trie_manifest_path.is_file():
        raise GenerativeEvalError(f"Trie manifest 不存在：{trie_manifest_path}")
    trie_manifest = json.loads(trie_manifest_path.read_text(encoding="utf-8"))
    ssp_predictions: dict[str, SspPrediction] | None = None
    ssp_manifest: dict[str, Any] | None = None
    if ssp_predictions_dir is not None:
        ssp_predictions, ssp_manifest = load_ssp_predictions(
            ssp_predictions_dir,
            evaluation_data_sha256=data_sha256,
            expected_rows=expected_rows,
        )
    tokenizer_json = tokenizer_path.resolve() / "tokenizer.json"
    config = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "split": split,
        "data_file": str(data_file.resolve()),
        "data_sha256": data_sha256,
        "expected_rows": expected_rows if smoke_limit is None else smoke_limit,
        "full_dataset_rows": expected_rows,
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint_name,
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "trie_dir": str(trie_dir.resolve()),
        "trie_manifest_sha256": sha256_file(trie_manifest_path),
        "trie_load_memory_bytes": trie_manifest.get("load_memory_bytes"),
        "evaluator_code_sha256": sha256_file(
            project_root / "src" / "poi_gr" / "generative_eval.py"
        ),
        "trie_code_sha256": sha256_file(
            project_root / "src" / "poi_gr" / "pid_trie.py"
        ),
        "pid_mapping": str(mapping_path.resolve()),
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "top_k": top_k,
        "do_sample": False,
        "max_new_tokens": 11,
        "length_penalty": 1.0,
        "early_stopping": True if num_beams > 1 else False,
        "renormalize_logits": True,
        "dtype": "bfloat16",
        "cutoff_len": cutoff_len,
        "chunk_size": chunk_size,
        "initial_batch_size": initial_batch_size,
        "smoke_limit": smoke_limit,
        "dataset_context": dict(dataset_context or {}),
        "constraint_mode": (
            "tcg_only"
            if ssp_manifest is None
            else (
                "tcg_ssp_beijing_ordinal"
                if ssp_manifest.get("schema_version")
                == "genpoi-beijing-ssp-predictions-v1"
                else "tcg_ssp"
            )
        ),
        "ssp_gamma": ssp_manifest.get("gamma") if ssp_manifest else None,
        "ssp_predictions_dir": (
            str(ssp_predictions_dir.resolve()) if ssp_predictions_dir else None
        ),
        "ssp_predictions_manifest_sha256": (
            sha256_file(ssp_predictions_dir.resolve() / "manifest.json")
            if ssp_predictions_dir
            else None
        ),
        "ssp_head_sha256": ssp_manifest.get("head_sha256") if ssp_manifest else None,
    }
    run_name = f"{split}_{checkpoint_name}_beam{num_beams}_return{num_return_sequences}"
    if smoke_limit is not None:
        run_name += f"_smoke{smoke_limit}"
    elif dataset_context and dataset_context.get("subset_size"):
        run_name += f"_subset{int(dataset_context['subset_size'])}"
    if ssp_manifest is not None:
        run_name += "_ssp"
    store = RunProgressStore(
        output_root / "runs" / run_name,
        config=config,
        available_candidates=num_return_sequences,
    )
    target_rows = smoke_limit if smoke_limit is not None else expected_rows
    if store.next_line == target_rows:
        result_path = store.run_dir / "result.json"
        if result_path.is_file():
            return json.loads(result_path.read_text(encoding="utf-8"))
        return store.finalize(target_rows)

    trie = CompactPidTrie.load(trie_dir, mmap=True)
    token_ids, _ = load_pid_token_ids(tokenizer_path)
    if trie.leaf_count != 2_337_178:
        raise GenerativeEvalError("全量 Final PID Trie 叶子数必须为 2,337,178")
    poi_ids = load_aligned_poi_ids(mapping_path, trie.leaf_count)
    tokenizer, template = load_lf_tokenizer_and_template(
        tokenizer_path,
        project_root=project_root,
    )
    model = load_generation_model(checkpoint, expected_vocab_size=len(tokenizer))
    allowed_batch_sizes = [
        value for value in (128, 64, 32, 16) if value <= initial_batch_size
    ]
    if store.state.get("actual_batch_size") is not None:
        safe = int(store.state["actual_batch_size"])
        allowed_batch_sizes = [value for value in allowed_batch_sizes if value <= safe]
    if not allowed_batch_sizes:
        raise GenerativeEvalError("没有符合协议的可用 Eval Batch Size")
    batch_size = allowed_batch_sizes[0]

    while store.next_line < target_rows:
        rows_to_read = min(chunk_size, target_rows - store.next_line)
        records, end_byte = read_jsonl_chunk(
            data_file,
            start_byte=store.next_byte,
            max_rows=rows_to_read,
        )
        if len(records) != rows_to_read:
            raise GenerativeEvalError(
                f"评测文件提前结束：需要 {rows_to_read} 行，读取 {len(records)} 行"
            )
        start_line = store.next_line
        existing_counts = {
            name: len(values) for name, values in store.error_cases.items()
        }
        while True:
            try:
                (
                    metrics,
                    error_cases,
                    seconds,
                    generated_tokens,
                    peak_memory,
                ) = evaluate_records_chunk(
                    records,
                    split=split,
                    model=model,
                    tokenizer=tokenizer,
                    template=template,
                    trie=trie,
                    token_ids=token_ids,
                    poi_ids=poi_ids,
                    checkpoint_name=checkpoint_name,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    top_k=top_k,
                    cutoff_len=cutoff_len,
                    batch_size=batch_size,
                    existing_error_counts=existing_counts,
                    ssp_predictions=ssp_predictions,
                )
                break
            except RuntimeError as error:
                if not _is_cuda_oom(error):
                    raise
                import torch

                torch.cuda.empty_cache()
                smaller = [value for value in allowed_batch_sizes if value < batch_size]
                if not smaller:
                    raise GenerativeEvalError(
                        "Eval Batch Size=16 仍然 OOM，停止当前及后续评测"
                    ) from error
                batch_size = smaller[0]
                print(
                    f"CUDA OOM：当前未提交分片改用 batch_size={batch_size} 重跑",
                    file=sys.stderr,
                    flush=True,
                )
        store.commit_chunk(
            start_line=start_line,
            end_line=start_line + len(records),
            end_byte=end_byte,
            metrics=metrics,
            error_cases=error_cases,
            inference_seconds=seconds,
            generated_token_count=generated_tokens,
            actual_batch_size=batch_size,
            peak_memory_bytes=peak_memory,
        )
        print(
            f"[{run_name}] {store.next_line:,}/{target_rows:,} "
            f"({store.next_line / target_rows:.2%})",
            file=sys.stderr,
            flush=True,
        )
    return store.finalize(target_rows)


def flatten_result_row(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    performance = result["performance"]
    config = result["config"]
    return {
        "split": config["split"],
        "checkpoint": config["checkpoint_name"],
        "epoch": config["epoch"],
        "beam": config["num_beams"],
        "num_return_sequences": config["num_return_sequences"],
        "hr@1": metrics["hr@1"],
        "hr@3": metrics["hr@3"],
        "hr@5": metrics["hr@5"],
        "hr@10": metrics["hr@10"],
        "ndcg@1": metrics["ndcg@1"],
        "ndcg@3": metrics["ndcg@3"],
        "ndcg@5": metrics["ndcg@5"],
        "ndcg@10": metrics["ndcg@10"],
        "samples_per_second": performance["samples_per_second"],
        "average_latency_seconds": performance["average_latency_seconds"],
        "peak_memory_bytes": performance["peak_memory_bytes"],
        "actual_batch_size": performance["actual_batch_size"],
    }

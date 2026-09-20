"""Full Train/Valid preflight for dynamic TIGER-Joint templates."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from poi_gr.methods.tiger_joint.preparation import (
    DYNAMIC_TARGET_IDENTIFIER,
    DynamicTextRecord,
    DynamicTokenIds,
    TigerJointPreparationError,
    parse_sid_free_record,
    parse_sid_free_records,
)

try:
    import orjson
except ImportError:  # pragma: no cover - supported environment has orjson
    orjson = None


class TigerJointPreflightError(TigerJointPreparationError):
    """Raised when a full dynamic-template preflight violates its gate."""


@dataclass
class IntegerHistogram:
    """Bounded integer histogram with deterministic percentile summaries."""

    counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    total: int = 0

    def update(self, values: Sequence[int] | np.ndarray) -> None:
        array = np.asarray(values, dtype=np.int64)
        if not array.size:
            return
        if array.ndim != 1 or np.any(array < 0):
            raise TigerJointPreflightError("长度 histogram 只接受一维非负整数")
        batch_counts = np.bincount(array)
        if batch_counts.size > self.counts.size:
            self.counts = np.pad(
                self.counts,
                (0, batch_counts.size - self.counts.size),
            )
        self.counts[: batch_counts.size] += batch_counts
        self.total += int(array.size)

    def merge(self, other: "IntegerHistogram") -> None:
        if other.counts.size > self.counts.size:
            self.counts = np.pad(
                self.counts,
                (0, other.counts.size - self.counts.size),
            )
        self.counts[: other.counts.size] += other.counts
        self.total += other.total

    def percentile(self, quantile: float) -> float:
        if not 0 <= quantile <= 1 or self.total <= 0:
            raise TigerJointPreflightError("无法计算空 histogram 的分位数")
        rank = (self.total - 1) * quantile
        low_rank = int(np.floor(rank))
        high_rank = int(np.ceil(rank))
        cumulative = np.cumsum(self.counts)
        low = int(np.searchsorted(cumulative, low_rank + 1))
        high = int(np.searchsorted(cumulative, high_rank + 1))
        return float(low + (high - low) * (rank - low_rank))

    def summary(self) -> dict[str, float | int]:
        if self.total <= 0:
            raise TigerJointPreflightError("长度 histogram 不能为空")
        nonzero = np.flatnonzero(self.counts)
        return {
            "min": int(nonzero[0]),
            "p50": self.percentile(0.50),
            "p90": self.percentile(0.90),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "p99_9": self.percentile(0.999),
            "max": int(nonzero[-1]),
        }


def _require_atomic_token(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if (
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or encoded != [token_id]
    ):
        raise TigerJointPreflightError(f"Tokenizer Token 不是原子项：{token}")
    return token_id


@dataclass(frozen=True)
class DynamicPreflightTemplate:
    """Frozen qwen3_nothink strings and IDs for vectorized tokenization."""

    source_prefix: str
    source_suffix: str
    target_text: str
    target_token_ids: tuple[int, ...]
    current_open: int
    token_ids: DynamicTokenIds

    @classmethod
    def from_runtime(
        cls,
        *,
        tokenizer: Any,
        template: Any,
        token_ids: DynamicTokenIds,
    ) -> "DynamicPreflightTemplate":
        sentinel = "__TIGER_JOINT_DYNAMIC_CONTENT_SENTINEL__"
        if template.format_prefix.apply():
            raise TigerJointPreflightError("qwen3_nothink 不应包含额外 prefix slot")
        user_slots = template.format_user.apply(content=sentinel, idx="0")
        target_slots = template.format_assistant.apply(
            content=DYNAMIC_TARGET_IDENTIFIER
        )
        if (
            len(user_slots) != 1
            or not isinstance(user_slots[0], str)
            or user_slots[0].count(sentinel) != 1
            or len(target_slots) != 1
            or not isinstance(target_slots[0], str)
        ):
            raise TigerJointPreflightError("qwen3_nothink 不再是单字符串模板")
        source_prefix, source_suffix = user_slots[0].split(sentinel)
        source_ids, target_ids = template.encode_oneturn(
            tokenizer,
            [
                {"role": "user", "content": sentinel},
                {
                    "role": "assistant",
                    "content": DYNAMIC_TARGET_IDENTIFIER,
                },
            ],
            system=None,
            tools=None,
        )
        if source_ids != tokenizer.encode(
            user_slots[0], add_special_tokens=False
        ) or target_ids != tokenizer.encode(target_slots[0], add_special_tokens=False):
            raise TigerJointPreflightError("批量格式化与 encode_oneturn Token 不等价")
        target_open_positions = [
            index
            for index, token_id in enumerate(target_ids)
            if token_id == token_ids.target_open
        ]
        if len(target_open_positions) != 1:
            raise TigerJointPreflightError("动态 Target wrapper 数量无效")
        target_open = target_open_positions[0]
        expected_target = (
            *token_ids.placeholders,
            token_ids.target_close,
        )
        if tuple(target_ids[target_open + 1 : target_open + 5]) != (expected_target):
            raise TigerJointPreflightError("动态 Target 三层槽位无效")
        if sum(
            token_id == token_ids.target_close for token_id in target_ids
        ) != 1 or any(
            sum(token_id == placeholder for token_id in target_ids) != 1
            for placeholder in token_ids.placeholders
        ):
            raise TigerJointPreflightError("动态 Target 槽位 Token 数量无效")
        return cls(
            source_prefix=source_prefix,
            source_suffix=source_suffix,
            target_text=target_slots[0],
            target_token_ids=tuple(int(value) for value in target_ids),
            current_open=_require_atomic_token(tokenizer, "<CURRENT>"),
            token_ids=token_ids,
        )

    def format_source(self, user_content: str) -> str:
        return f"{self.source_prefix}{user_content}{self.source_suffix}"

    def validate_source_token_ids(
        self,
        source_ids: Sequence[int],
        *,
        history_count: int,
    ) -> None:
        values = (
            source_ids if isinstance(source_ids, (list, tuple)) else tuple(source_ids)
        )
        if values.count(self.token_ids.history_open) != history_count:
            raise TigerJointPreflightError(
                "Tokenized 历史 wrapper 数量与 history_length 不一致"
            )
        expected_history = (
            *self.token_ids.placeholders,
            self.token_ids.history_close,
        )
        history_open_positions: list[int] = []
        search_from = 0
        for _ in range(history_count):
            open_position = values.index(
                self.token_ids.history_open,
                search_from,
            )
            history_open_positions.append(open_position)
            search_from = open_position + 1
            if tuple(values[open_position + 1 : open_position + 5]) != (
                expected_history
            ):
                raise TigerJointPreflightError("Tokenized 历史三层槽位无效")
        if values.count(self.token_ids.history_close) != history_count:
            raise TigerJointPreflightError("Tokenized 历史 close 数量无效")
        for placeholder in self.token_ids.placeholders:
            if values.count(placeholder) != history_count:
                raise TigerJointPreflightError("SID placeholder 出现在动态历史槽位之外")
        if (
            values.count(self.current_open) != 1
            or values.count(self.token_ids.current_close) != 1
        ):
            raise TigerJointPreflightError("Tokenized CURRENT wrapper 无效")
        current_open_position = values.index(self.current_open)
        current_close_position = values.index(self.token_ids.current_close)
        if current_open_position >= current_close_position:
            raise TigerJointPreflightError("Tokenized CURRENT wrapper 无效")
        if history_open_positions and (
            history_open_positions[-1] >= current_open_position
        ):
            raise TigerJointPreflightError("历史 SID 槽位必须位于 CURRENT 之前")
        if values.count(self.token_ids.target_open) or values.count(
            self.token_ids.target_close
        ):
            raise TigerJointPreflightError("Source 中出现了 Target structural Token")


def _retained_target_lengths(
    source_lengths: np.ndarray,
    target_lengths: np.ndarray,
    cutoff_len: int,
) -> np.ndarray:
    max_target = np.empty_like(target_lengths)
    target_short = target_lengths * 2 < cutoff_len
    source_short = (~target_short) & (source_lengths * 2 < cutoff_len)
    both_long = ~(target_short | source_short)
    max_target[target_short] = cutoff_len
    max_target[source_short] = cutoff_len - source_lengths[source_short]
    denominator = source_lengths[both_long] + target_lengths[both_long]
    max_target[both_long] = (
        cutoff_len * target_lengths[both_long] / denominator
    ).astype(np.int64)
    return np.minimum(max_target, target_lengths)


@dataclass
class DynamicSplitPreflight:
    """Mutable bounded statistics for one fully streamed split."""

    name: str
    path: Path
    expected_rows: int
    expected_sha256: str
    scope: str
    source_lengths: IntegerHistogram = field(default_factory=IntegerHistogram)
    target_lengths: IntegerHistogram = field(default_factory=IntegerHistogram)
    total_lengths: IntegerHistogram = field(default_factory=IntegerHistogram)
    history_length_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(11, dtype=np.int64)
    )
    rows: int = 0
    history_events: int = 0
    poi_row_references: int = 0
    dynamic_sid_slots: int = 0
    over_cutoff_count: int = 0
    target_truncated_count: int = 0
    sha256: str = ""
    seconds: float = 0.0

    def update_batch(
        self,
        records: Sequence[DynamicTextRecord],
        source_token_ids: Sequence[Sequence[int]],
        source_lengths: np.ndarray,
        target_length: int,
        *,
        template_contract: DynamicPreflightTemplate,
        cutoff_len: int,
    ) -> None:
        if len(records) != len(source_token_ids) or len(records) != len(source_lengths):
            raise TigerJointPreflightError("预检 batch 行数不守恒")
        history_lengths = np.asarray(
            [len(record.history_poi_rows) for record in records],
            dtype=np.int64,
        )
        if np.any(history_lengths < 0) or np.any(history_lengths > 10):
            raise TigerJointPreflightError("history_length 超出 0～10")
        for record, token_ids, history_count in zip(
            records,
            source_token_ids,
            history_lengths,
            strict=True,
        ):
            if record.target_content != DYNAMIC_TARGET_IDENTIFIER:
                raise TigerJointPreflightError("动态 Target 模板发生漂移")
            template_contract.validate_source_token_ids(
                token_ids,
                history_count=int(history_count),
            )
        target_lengths = np.full(len(records), target_length, dtype=np.int64)
        total_lengths = source_lengths + target_lengths
        retained_targets = _retained_target_lengths(
            source_lengths,
            target_lengths,
            cutoff_len,
        )
        self.source_lengths.update(source_lengths)
        self.target_lengths.update(target_lengths)
        self.total_lengths.update(total_lengths)
        self.history_length_counts += np.bincount(history_lengths, minlength=11)
        batch_rows = len(records)
        batch_history = int(history_lengths.sum())
        self.rows += batch_rows
        self.history_events += batch_history
        self.poi_row_references += batch_history + batch_rows
        self.dynamic_sid_slots += 3 * (batch_history + batch_rows)
        self.over_cutoff_count += int(np.count_nonzero(total_lengths > cutoff_len))
        self.target_truncated_count += int(
            np.count_nonzero(retained_targets < target_lengths)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path.resolve()),
            "scope": self.scope,
            "rows": self.rows,
            "expected_rows": self.expected_rows,
            "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
            "source_length": self.source_lengths.summary(),
            "target_length": self.target_lengths.summary(),
            "total_length": self.total_lengths.summary(),
            "history_length_counts": {
                str(index): int(value)
                for index, value in enumerate(self.history_length_counts)
            },
            "history_events": self.history_events,
            "poi_row_references": self.poi_row_references,
            "dynamic_sid_slots": self.dynamic_sid_slots,
            "old_collision_token_count": 0,
            "slot_validation_rows": self.rows,
            "over_cutoff_count": self.over_cutoff_count,
            "target_truncated_count": self.target_truncated_count,
            "seconds": self.seconds,
            "rows_per_second": self.rows / self.seconds,
        }


ProgressCallback = Callable[[str, int, float, float, int, int], None]


def _loads_json_object(raw_line: bytes, split: str, line_number: int) -> dict[str, Any]:
    try:
        value = orjson.loads(raw_line) if orjson is not None else json.loads(raw_line)
    except (json.JSONDecodeError, ValueError) as error:
        raise TigerJointPreflightError(
            f"{split} 第 {line_number} 行不是合法 JSON"
        ) from error
    if not isinstance(value, dict):
        raise TigerJointPreflightError(f"{split} 第 {line_number} 行必须是 JSON object")
    return value


def _parse_batch_with_line_context(
    records: Sequence[Mapping[str, Any]],
    line_numbers: Sequence[int],
    *,
    split: str,
) -> list[DynamicTextRecord]:
    try:
        return parse_sid_free_records(
            records,
            expected_split=split,
        )
    except TigerJointPreparationError as batch_error:
        for record, line_number in zip(records, line_numbers, strict=True):
            try:
                parse_sid_free_record(
                    record,
                    expected_split=split,
                )
            except TigerJointPreparationError as error:
                raise TigerJointPreflightError(
                    f"{split} 第 {line_number} 行 SID-free 校验失败：{error}"
                ) from error
        raise TigerJointPreflightError(
            f"{split} batch SID-free 校验失败：{batch_error}"
        ) from batch_error


def _process_record_batch(
    records: Sequence[Mapping[str, Any]],
    line_numbers: Sequence[int],
    *,
    split_stats: DynamicSplitPreflight,
    tokenizer: Any,
    template_contract: DynamicPreflightTemplate,
    cutoff_len: int,
) -> None:
    converted = _parse_batch_with_line_context(
        records,
        line_numbers,
        split=split_stats.name,
    )
    source_texts = [
        template_contract.format_source(record.user_content) for record in converted
    ]
    encoded = tokenizer(
        source_texts,
        add_special_tokens=False,
        return_length=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        padding=False,
        truncation=False,
    )
    source_token_ids = encoded.get("input_ids")
    raw_lengths = encoded.get("length")
    if not isinstance(source_token_ids, list) or not isinstance(raw_lengths, list):
        raise TigerJointPreflightError("Tokenizer batch 输出格式无效")
    source_lengths = np.asarray(raw_lengths, dtype=np.int64)
    if any(
        len(token_ids) != int(length)
        for token_ids, length in zip(source_token_ids, source_lengths, strict=True)
    ):
        raise TigerJointPreflightError("Tokenizer length 与 input_ids 不一致")
    split_stats.update_batch(
        converted,
        source_token_ids,
        source_lengths,
        len(template_contract.target_token_ids),
        template_contract=template_contract,
        cutoff_len=cutoff_len,
    )


def scan_dynamic_preflight_split(
    *,
    name: str,
    path: Path,
    expected_rows: int,
    expected_sha256: str,
    tokenizer: Any,
    template_contract: DynamicPreflightTemplate,
    cutoff_len: int = 1024,
    batch_size: int = 4096,
    max_rows: int | None = None,
    progress_every: int = 250_000,
    progress_callback: ProgressCallback | None = None,
) -> DynamicSplitPreflight:
    """Stream one split, validate every mapping/slot, and collect exact lengths."""

    if name not in {"train", "valid"}:
        raise TigerJointPreflightError("全量预检只允许 train/valid")
    if path.name == "test.jsonl":
        raise TigerJointPreflightError("全量预检禁止读取 test.jsonl")
    if cutoff_len != 1024:
        raise TigerJointPreflightError("新的历史联合训练 cutoff_len 必须是 1024")
    if batch_size <= 0 or progress_every <= 0:
        raise TigerJointPreflightError("batch_size/progress_every 必须为正数")
    if max_rows is not None and max_rows <= 0:
        raise TigerJointPreflightError("max_rows 必须为正数")
    if not path.is_file():
        raise TigerJointPreflightError(f"{name} 文件不存在：{path}")
    stats = DynamicSplitPreflight(
        name=name,
        path=path,
        expected_rows=expected_rows,
        expected_sha256=expected_sha256,
        scope="full" if max_rows is None else "prefix_smoke",
    )
    records: list[Mapping[str, Any]] = []
    line_numbers: list[int] = []
    digest = hashlib.sha256()
    started = time.perf_counter()
    next_progress = progress_every
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if max_rows is not None and stats.rows + len(records) >= max_rows:
                break
            digest.update(raw_line)
            records.append(_loads_json_object(raw_line, name, line_number))
            line_numbers.append(line_number)
            if len(records) == batch_size:
                _process_record_batch(
                    records,
                    line_numbers,
                    split_stats=stats,
                    tokenizer=tokenizer,
                    template_contract=template_contract,
                    cutoff_len=cutoff_len,
                )
                records.clear()
                line_numbers.clear()
            processed = stats.rows + len(records)
            if processed >= next_progress:
                elapsed = time.perf_counter() - started
                if progress_callback is not None:
                    progress_callback(
                        name,
                        processed,
                        elapsed,
                        processed / elapsed,
                        stats.total_lengths.summary()["max"] if stats.rows else 0,
                        stats.over_cutoff_count,
                    )
                next_progress += progress_every
    if records:
        _process_record_batch(
            records,
            line_numbers,
            split_stats=stats,
            tokenizer=tokenizer,
            template_contract=template_contract,
            cutoff_len=cutoff_len,
        )
    stats.sha256 = digest.hexdigest()
    stats.seconds = time.perf_counter() - started
    if max_rows is None:
        if stats.rows != expected_rows:
            raise TigerJointPreflightError(
                f"{name} 行数不一致：{stats.rows} != {expected_rows}"
            )
        if stats.sha256 != expected_sha256:
            raise TigerJointPreflightError(f"{name} SHA256 与 manifest 不一致")
    elif stats.rows != min(max_rows, expected_rows):
        raise TigerJointPreflightError(f"{name} prefix 行数不一致：{stats.rows}")
    return stats


def combine_dynamic_preflight_splits(
    splits: Sequence[DynamicSplitPreflight],
) -> dict[str, Any]:
    """Combine Train/Valid histograms and enforce the formal zero gates."""

    if {split.name for split in splits} != {"train", "valid"}:
        raise TigerJointPreflightError("必须同时提供 train 与 valid 预检")
    source_lengths = IntegerHistogram()
    target_lengths = IntegerHistogram()
    total_lengths = IntegerHistogram()
    for split in splits:
        source_lengths.merge(split.source_lengths)
        target_lengths.merge(split.target_lengths)
        total_lengths.merge(split.total_lengths)
    rows = sum(split.rows for split in splits)
    over_cutoff = sum(split.over_cutoff_count for split in splits)
    target_truncated = sum(split.target_truncated_count for split in splits)
    return {
        "rows": rows,
        "source_length": source_lengths.summary(),
        "target_length": target_lengths.summary(),
        "total_length": total_lengths.summary(),
        "history_events": sum(split.history_events for split in splits),
        "poi_row_references": sum(split.poi_row_references for split in splits),
        "dynamic_sid_slots": sum(split.dynamic_sid_slots for split in splits),
        "old_collision_token_count": 0,
        "slot_validation_rows": rows,
        "over_cutoff_count": over_cutoff,
        "target_truncated_count": target_truncated,
        "formal_zero_gate_passed": over_cutoff == 0 and target_truncated == 0,
    }


def build_fresh_initialization_contract(*, row_count: int) -> dict[str, Any]:
    """Create the method-owned deterministic split without reading old SID runs."""

    from poi_gr.sid.training import create_fixed_indices

    if row_count <= 1:
        raise TigerJointPreflightError("KMeans 契约 row_count 必须大于 1")
    seed = 42
    validation_ratio = 0.01
    kmeans_sample_size = 500_000
    validation_mask, validation_indices, sample_indices, metadata = (
        create_fixed_indices(
            row_count=row_count,
            validation_ratio=validation_ratio,
            kmeans_sample_size=kmeans_sample_size,
            seed=seed,
        )
    )
    del validation_mask, validation_indices, sample_indices
    return {
        "schema_version": "tiger-joint-fresh-initialization-v1",
        "source": "method_owned_constants_plus_bge_row_count",
        "legacy_artifacts_loaded": [],
        "fresh_rqvae": {
            "input_dim": 1024,
            "encoder": [1024, 512, 256],
            "decoder": [256, 512, 1024],
            "codebook_sizes": [1024, 1024, 1024],
            "checkpoint_loaded": False,
        },
        "fresh_qwen": {
            "base_model": "models/Qwen3-0.6B",
            "old_tiger_vocab_loaded": False,
            "old_tiger_checkpoint_loaded": False,
            "new_token_rows_initialized_from_base_model": True,
        },
        "random_seed": seed,
        "construction_order": [
            "seed_all_rngs",
            "construct_fresh_rqvae",
            "extend_vanilla_qwen_with_fresh_joint_tokens",
            "construct_query_projection",
        ],
        "fixed_split": metadata,
        "kmeans": {
            "backend": "faiss_gpu",
            "sample_rows": kmeans_sample_size,
            "sample_indices_sha256": metadata["kmeans_sample_indices_sha256"],
            "residual_mode": "sequential",
            "iterations": 20,
            "batch_size": 8192,
            "max_points_per_centroid": 2048,
        },
    }

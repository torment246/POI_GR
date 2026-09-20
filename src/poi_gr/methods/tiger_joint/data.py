"""Dynamic Semantic-ID batch contracts for TIGER joint training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


IGNORE_INDEX = -100
SID_LEVEL_COUNT = 3
FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


class TigerJointDataError(ValueError):
    """Raised when a dynamic TIGER-Joint batch violates its contract."""


def _has_true(value: torch.Tensor) -> bool:
    return bool(value.detach().any().item())


def _require_long_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long:
        raise TigerJointDataError(f"{name} 必须是 torch.long tensor")


def _validate_active_positions(
    attention_mask: torch.Tensor,
    sample_indices: torch.Tensor,
    positions: torch.Tensor,
    name: str,
) -> None:
    if not positions.numel():
        return
    active = attention_mask.bool()[sample_indices.unsqueeze(1), positions]
    if not bool(active.all().item()):
        raise TigerJointDataError(f"{name} 必须全部落在有效 Token 位置")


@dataclass(frozen=True)
class DynamicSidTemplate:
    """One padded behavior batch with mutable three-level SID slots."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    query_state_positions: torch.Tensor
    history_sample_indices: torch.Tensor
    history_poi_rows: torch.Tensor
    history_sid_positions: torch.Tensor
    target_poi_rows: torch.Tensor
    target_sid_positions: torch.Tensor
    cutoff_len: int = 1024
    max_history_events: int = 10

    def __post_init__(self) -> None:
        _require_long_tensor(self.input_ids, "input_ids")
        _require_long_tensor(self.labels, "labels")
        if self.attention_mask.dtype not in {torch.bool, torch.long}:
            raise TigerJointDataError(
                "attention_mask 必须是 torch.bool 或 torch.long tensor"
            )
        for name, value in (
            ("query_state_positions", self.query_state_positions),
            ("history_sample_indices", self.history_sample_indices),
            ("history_poi_rows", self.history_poi_rows),
            ("history_sid_positions", self.history_sid_positions),
            ("target_poi_rows", self.target_poi_rows),
            ("target_sid_positions", self.target_sid_positions),
        ):
            _require_long_tensor(value, name)

        if self.input_ids.ndim != 2 or not min(self.input_ids.shape):
            raise TigerJointDataError("input_ids 必须是非空 [batch, sequence] tensor")
        if self.labels.shape != self.input_ids.shape:
            raise TigerJointDataError("labels shape 必须与 input_ids 一致")
        if self.attention_mask.shape != self.input_ids.shape:
            raise TigerJointDataError("attention_mask shape 必须与 input_ids 一致")
        if (
            isinstance(self.cutoff_len, bool)
            or not isinstance(self.cutoff_len, int)
            or self.cutoff_len <= 0
        ):
            raise TigerJointDataError("cutoff_len 必须是正整数")
        if (
            isinstance(self.max_history_events, bool)
            or not isinstance(self.max_history_events, int)
            or self.max_history_events <= 0
        ):
            raise TigerJointDataError("max_history_events 必须是正整数")

        batch_size, sequence_length = self.input_ids.shape
        if sequence_length > self.cutoff_len:
            raise TigerJointDataError(
                f"sequence length {sequence_length} 超过 cutoff_len={self.cutoff_len}"
            )
        if self.query_state_positions.shape != (batch_size,):
            raise TigerJointDataError("query_state_positions shape 必须是 [batch]")
        if self.target_poi_rows.shape != (batch_size,):
            raise TigerJointDataError("target_poi_rows shape 必须是 [batch]")
        if self.target_sid_positions.shape != (
            batch_size,
            SID_LEVEL_COUNT,
        ):
            raise TigerJointDataError("target_sid_positions shape 必须是 [batch, 3]")

        if self.history_poi_rows.ndim != 1:
            raise TigerJointDataError("history_poi_rows shape 必须是 [history]")
        history_count = self.history_poi_rows.shape[0]
        if self.history_sample_indices.shape != (history_count,):
            raise TigerJointDataError("history_sample_indices shape 必须是 [history]")
        if self.history_sid_positions.shape != (
            history_count,
            SID_LEVEL_COUNT,
        ):
            raise TigerJointDataError("history_sid_positions shape 必须是 [history, 3]")

        tensors = (
            self.labels,
            self.attention_mask,
            self.query_state_positions,
            self.history_sample_indices,
            self.history_poi_rows,
            self.history_sid_positions,
            self.target_poi_rows,
            self.target_sid_positions,
        )
        if any(value.device != self.input_ids.device for value in tensors):
            raise TigerJointDataError("动态模板中的 tensor 必须位于同一 device")
        if _has_true(self.input_ids < 0):
            raise TigerJointDataError("input_ids 不能包含负数")
        invalid_attention = (self.attention_mask != 0) & (self.attention_mask != 1)
        if _has_true(invalid_attention):
            raise TigerJointDataError("attention_mask 只能包含 0/1")
        invalid_labels = (self.labels < 0) & (self.labels != IGNORE_INDEX)
        if _has_true(invalid_labels):
            raise TigerJointDataError("labels 只能包含非负 Token ID 或 -100")
        if _has_true(self.labels[~self.attention_mask.bool()] != IGNORE_INDEX):
            raise TigerJointDataError("padding 位置的 labels 必须是 -100")
        if _has_true(self.target_poi_rows < 0) or _has_true(self.history_poi_rows < 0):
            raise TigerJointDataError("POI 行号不能为负数")

        if _has_true(self.query_state_positions < 0) or _has_true(
            self.query_state_positions >= sequence_length
        ):
            raise TigerJointDataError("query_state_positions 超出序列范围")
        if history_count and (
            _has_true(self.history_sample_indices < 0)
            or _has_true(self.history_sample_indices >= batch_size)
        ):
            raise TigerJointDataError("history_sample_indices 超出 batch 范围")
        if history_count:
            history_counts = torch.bincount(
                self.history_sample_indices,
                minlength=batch_size,
            )
            if _has_true(history_counts > self.max_history_events):
                raise TigerJointDataError("单条样本的历史数量超过 max_history_events")
        for name, positions in (
            ("history_sid_positions", self.history_sid_positions),
            ("target_sid_positions", self.target_sid_positions),
        ):
            if _has_true(positions < 0) or _has_true(positions >= sequence_length):
                raise TigerJointDataError(f"{name} 超出序列范围")

        batch_indices = torch.arange(batch_size, device=self.input_ids.device)
        if not bool(
            self.attention_mask.bool()[batch_indices, self.query_state_positions]
            .all()
            .item()
        ):
            raise TigerJointDataError(
                "query_state_positions 必须全部落在有效 Token 位置"
            )
        _validate_active_positions(
            self.attention_mask,
            self.history_sample_indices,
            self.history_sid_positions,
            "history_sid_positions",
        )
        _validate_active_positions(
            self.attention_mask,
            batch_indices,
            self.target_sid_positions,
            "target_sid_positions",
        )
        if history_count and _has_true(
            self.history_sid_positions
            >= self.query_state_positions[self.history_sample_indices].unsqueeze(1)
        ):
            raise TigerJointDataError("历史 SID 槽位必须位于 query state 之前")
        if _has_true(
            self.target_sid_positions <= self.query_state_positions.unsqueeze(1)
        ):
            raise TigerJointDataError("目标 SID 槽位必须位于 query state 之后")
        for name, positions in (
            ("history_sid_positions", self.history_sid_positions),
            ("target_sid_positions", self.target_sid_positions),
        ):
            if positions.shape[0] and _has_true(positions[:, 1:] <= positions[:, :-1]):
                raise TigerJointDataError(f"{name} 必须按 S1/S2/S3 严格递增")
        if history_count and _has_true(
            self.labels[
                self.history_sample_indices.unsqueeze(1),
                self.history_sid_positions,
            ]
            != IGNORE_INDEX
        ):
            raise TigerJointDataError("历史 SID 输入不能进入 Assistant loss")
        if _has_true(
            self.labels[batch_indices, self.query_state_positions] != IGNORE_INDEX
        ):
            raise TigerJointDataError("query state 不能进入 Assistant loss")

        history_slot_keys = (
            self.history_sample_indices.unsqueeze(1) * sequence_length
            + self.history_sid_positions
        ).reshape(-1)
        target_slot_keys = (
            batch_indices.unsqueeze(1) * sequence_length + self.target_sid_positions
        ).reshape(-1)
        all_slot_keys = torch.cat((history_slot_keys, target_slot_keys))
        if torch.unique(all_slot_keys).numel() != all_slot_keys.numel():
            raise TigerJointDataError("历史与目标 SID 槽位不能重叠")
        query_keys = batch_indices * sequence_length + self.query_state_positions
        if _has_true(torch.isin(query_keys, all_slot_keys)):
            raise TigerJointDataError("query state 位置不能与 SID 槽位重叠")


@dataclass(frozen=True)
class JointTigerBatch:
    """Runtime embeddings paired with one validated dynamic SID template."""

    template: DynamicSidTemplate
    history_embeddings: torch.Tensor
    target_embeddings: torch.Tensor
    catalog_poi_rows: torch.Tensor
    catalog_embeddings: torch.Tensor

    def __post_init__(self) -> None:
        _require_long_tensor(self.catalog_poi_rows, "catalog_poi_rows")
        for name, value in (
            ("history_embeddings", self.history_embeddings),
            ("target_embeddings", self.target_embeddings),
            ("catalog_embeddings", self.catalog_embeddings),
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise TigerJointDataError(f"{name} 必须是二维 tensor")
            if value.dtype not in FLOAT_DTYPES:
                raise TigerJointDataError(f"{name} 必须是浮点 tensor")
            if _has_true(~torch.isfinite(value)):
                raise TigerJointDataError(f"{name} 包含 NaN 或 Inf")

        batch_size = self.template.input_ids.shape[0]
        history_count = self.template.history_poi_rows.shape[0]
        if self.catalog_poi_rows.ndim != 1:
            raise TigerJointDataError("catalog_poi_rows 必须是 [catalog] tensor")
        catalog_count = self.catalog_poi_rows.shape[0]
        if self.history_embeddings.shape[0] != history_count:
            raise TigerJointDataError(
                "history_embeddings 行数必须与 history_poi_rows 一致"
            )
        if self.target_embeddings.shape[0] != batch_size:
            raise TigerJointDataError("target_embeddings 行数必须与行为 batch 一致")
        if catalog_count <= 0:
            raise TigerJointDataError("catalog_poi_rows 必须是非空 [catalog] tensor")
        if self.catalog_embeddings.shape[0] != catalog_count:
            raise TigerJointDataError(
                "catalog_embeddings 行数必须与 catalog_poi_rows 一致"
            )
        dimensions = {
            self.history_embeddings.shape[1],
            self.target_embeddings.shape[1],
            self.catalog_embeddings.shape[1],
        }
        if len(dimensions) != 1 or next(iter(dimensions)) <= 0:
            raise TigerJointDataError("三类 Embedding 的维度必须相同且为正数")
        tensors = (
            self.history_embeddings,
            self.target_embeddings,
            self.catalog_poi_rows,
            self.catalog_embeddings,
        )
        if any(value.device != self.template.input_ids.device for value in tensors):
            raise TigerJointDataError("模板与运行时 tensor 必须位于同一 device")
        if _has_true(self.catalog_poi_rows < 0):
            raise TigerJointDataError("catalog_poi_rows 不能包含负数")

    @property
    def embedding_dim(self) -> int:
        return int(self.target_embeddings.shape[1])


@dataclass(frozen=True)
class SidTokenLayout:
    """Map each RQ code level to its existing atomic vocabulary tokens."""

    level_token_ids: tuple[tuple[int, ...], ...]

    def __init__(self, level_token_ids: Sequence[Sequence[int]]) -> None:
        normalized = tuple(tuple(level) for level in level_token_ids)
        object.__setattr__(self, "level_token_ids", normalized)
        self.__post_init__()

    def __post_init__(self) -> None:
        if len(self.level_token_ids) != SID_LEVEL_COUNT:
            raise TigerJointDataError("SID Token layout 必须恰好包含三层")
        flattened: list[int] = []
        for level_index, token_ids in enumerate(self.level_token_ids):
            if not token_ids:
                raise TigerJointDataError(f"第 {level_index + 1} 层 Token 表不能为空")
            if any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            ):
                raise TigerJointDataError("SID Token ID 必须是非负整数")
            flattened.extend(token_ids)
        if len(flattened) != len(set(flattened)):
            raise TigerJointDataError("三层 SID Token ID 必须全局互不重复")

    @property
    def codebook_sizes(self) -> tuple[int, int, int]:
        sizes = tuple(len(level) for level in self.level_token_ids)
        return sizes[0], sizes[1], sizes[2]

    def tokens_for_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert `[items, 3]` RQ codes to vocabulary token IDs."""

        _require_long_tensor(codes, "codes")
        if codes.ndim != 2 or codes.shape[1] != SID_LEVEL_COUNT:
            raise TigerJointDataError("codes shape 必须是 [items, 3]")
        token_ids = torch.empty_like(codes)
        for level_index, level_tokens in enumerate(self.level_token_ids):
            level_codes = codes[:, level_index]
            if _has_true(level_codes < 0) or _has_true(
                level_codes >= len(level_tokens)
            ):
                raise TigerJointDataError(
                    f"第 {level_index + 1} 层 code 超出 Token 容量"
                )
            table = torch.tensor(
                level_tokens,
                dtype=torch.long,
                device=codes.device,
            )
            token_ids[:, level_index] = table[level_codes]
        return token_ids


@dataclass(frozen=True)
class MaterializedSidBatch:
    """Input IDs and labels after current RQ codes replace all SID slots."""

    input_ids: torch.Tensor
    labels: torch.Tensor


def materialize_dynamic_sid_batch(
    template: DynamicSidTemplate,
    history_codes: torch.Tensor,
    target_codes: torch.Tensor,
    token_layout: SidTokenLayout,
) -> MaterializedSidBatch:
    """Patch current hard SID codes into history inputs and target labels."""

    expected_history_shape = (
        template.history_poi_rows.shape[0],
        SID_LEVEL_COUNT,
    )
    expected_target_shape = (
        template.input_ids.shape[0],
        SID_LEVEL_COUNT,
    )
    _require_long_tensor(history_codes, "history_codes")
    _require_long_tensor(target_codes, "target_codes")
    if history_codes.shape != expected_history_shape:
        raise TigerJointDataError(
            f"history_codes shape 必须是 {expected_history_shape}"
        )
    if target_codes.shape != expected_target_shape:
        raise TigerJointDataError(f"target_codes shape 必须是 {expected_target_shape}")
    if history_codes.device != template.input_ids.device or (
        target_codes.device != template.input_ids.device
    ):
        raise TigerJointDataError("SID codes 与模板必须位于同一 device")

    history_token_ids = token_layout.tokens_for_codes(history_codes)
    target_token_ids = token_layout.tokens_for_codes(target_codes)
    input_ids = template.input_ids.clone()
    labels = template.labels.clone()
    target_sample_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
    for level_index in range(SID_LEVEL_COUNT):
        if history_codes.shape[0]:
            input_ids[
                template.history_sample_indices,
                template.history_sid_positions[:, level_index],
            ] = history_token_ids[:, level_index]
        target_positions = template.target_sid_positions[:, level_index]
        input_ids[target_sample_indices, target_positions] = target_token_ids[
            :, level_index
        ]
        labels[target_sample_indices, target_positions] = target_token_ids[
            :, level_index
        ]
    return MaterializedSidBatch(input_ids=input_ids, labels=labels)

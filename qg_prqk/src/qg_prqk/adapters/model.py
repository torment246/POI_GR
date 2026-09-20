"""Query-only residual adapter and masked weighted InfoNCE."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from qg_prqk.config import QueryAdapterConfig


class QueryAdapterError(ValueError):
    """Raised when Query Adapter tensors violate the frozen contract."""


class ResidualQueryAdapter(nn.Module):
    """LayerNorm→bottleneck MLP→residual add→L2 normalize."""

    def __init__(
        self,
        embedding_dim: int,
        bottleneck: int,
        *,
        residual_scale: float = 1.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or bottleneck <= 0:
            raise QueryAdapterError("embedding_dim/bottleneck 必须为正数")
        if residual_scale < 0 or not 0 <= dropout < 1:
            raise QueryAdapterError("residual_scale/dropout 非法")
        self.embedding_dim = embedding_dim
        self.bottleneck = bottleneck
        self.residual_scale = float(residual_scale)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.down = nn.Linear(embedding_dim, bottleneck)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, embedding_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.embedding_dim:
            raise QueryAdapterError("Query embedding shape 必须为 [batch, embedding_dim]")
        residual = self.up(
            self.dropout(self.activation(self.down(self.layer_norm(query_embeddings))))
        )
        return F.normalize(
            query_embeddings + self.residual_scale * residual,
            p=2,
            dim=1,
            eps=1e-8,
        )


def build_negative_valid_mask(
    candidate_poi_rows: torch.Tensor,
    target_poi_rows: torch.Tensor,
    reasonable_positive_rows: Sequence[Iterable[int]],
) -> torch.Tensor:
    """Mask the target and every P2 false-negative target out of negatives."""

    if candidate_poi_rows.ndim != 2 or candidate_poi_rows.dtype != torch.long:
        raise QueryAdapterError("candidate_poi_rows 必须是 int64[batch, negatives]")
    if target_poi_rows.dtype != torch.long or target_poi_rows.shape != (
        candidate_poi_rows.shape[0],
    ):
        raise QueryAdapterError("target_poi_rows 必须是 int64[batch]")
    if target_poi_rows.device != candidate_poi_rows.device:
        raise QueryAdapterError("candidate/target POI rows 必须位于同一 device")
    if len(reasonable_positive_rows) != candidate_poi_rows.shape[0]:
        raise QueryAdapterError("reasonable_positive_rows 长度必须等于 batch")
    mask = candidate_poi_rows.ge(0) & candidate_poi_rows.ne(
        target_poi_rows.unsqueeze(1)
    )
    for row_index, positives in enumerate(reasonable_positive_rows):
        values = sorted({int(value) for value in positives})
        if values:
            positive_tensor = torch.tensor(
                values,
                dtype=torch.long,
                device=candidate_poi_rows.device,
            )
            mask[row_index] &= ~candidate_poi_rows[row_index].unsqueeze(1).eq(
                positive_tensor.unsqueeze(0)
            ).any(dim=1)
    return mask


def build_in_batch_valid_mask(
    batch_target_rows: torch.Tensor,
    reasonable_positive_rows: Sequence[Iterable[int]],
) -> torch.Tensor:
    """Mask duplicate targets and Query-specific reasonable positives in a batch."""

    if batch_target_rows.dtype != torch.long or batch_target_rows.ndim != 1:
        raise QueryAdapterError("batch_target_rows 必须是 int64[batch]")
    if len(reasonable_positive_rows) != len(batch_target_rows):
        raise QueryAdapterError("in-batch reasonable positives 长度不一致")
    mask = batch_target_rows.unsqueeze(0).ne(batch_target_rows.unsqueeze(1))
    columns_by_target: dict[int, list[int]] = {}
    for column, target in enumerate(batch_target_rows.tolist()):
        columns_by_target.setdefault(int(target), []).append(column)
    for row, positives in enumerate(reasonable_positive_rows):
        masked_columns = sorted(
            {
                column
                for positive in positives
                for column in columns_by_target.get(int(positive), ())
            }
        )
        if masked_columns:
            mask[row, masked_columns] = False
    return mask


def weighted_info_nce(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    negative_valid_mask: torch.Tensor,
    query_weights: torch.Tensor,
    *,
    temperature: float,
    in_batch_positive_embeddings: torch.Tensor | None = None,
    in_batch_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute weighted InfoNCE over fixed and optional dynamic in-batch negatives."""

    if query_embeddings.ndim != 2 or positive_embeddings.shape != query_embeddings.shape:
        raise QueryAdapterError("Query/positive embedding shape 必须相同且为二维")
    batch, dimension = query_embeddings.shape
    if negative_embeddings.ndim != 3 or negative_embeddings.shape[:2] != (
        batch,
        negative_valid_mask.shape[1],
    ) or negative_embeddings.shape[2] != dimension:
        raise QueryAdapterError("negative_embeddings shape 必须为 [batch, negatives, dim]")
    if negative_valid_mask.dtype != torch.bool or negative_valid_mask.shape != negative_embeddings.shape[:2]:
        raise QueryAdapterError("negative_valid_mask shape/dtype 非法")
    if query_weights.shape != (batch,) or not torch.is_floating_point(query_weights):
        raise QueryAdapterError("query_weights 必须是浮点 [batch]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise QueryAdapterError("temperature 必须为正有限数")
    if not bool(negative_valid_mask.any(dim=1).all().item()):
        raise QueryAdapterError("每个 Query 至少需要一个 mask 后有效负样本")
    if not bool(torch.isfinite(query_weights).all().item()) or bool(
        (query_weights < 0).any().item()
    ) or float(query_weights.sum().item()) <= 0:
        raise QueryAdapterError("query_weights 必须有限、非负且总和为正")

    queries = F.normalize(query_embeddings.float(), p=2, dim=1, eps=1e-8)
    positives = F.normalize(positive_embeddings.float(), p=2, dim=1, eps=1e-8)
    negatives = F.normalize(negative_embeddings.float(), p=2, dim=2, eps=1e-8)
    positive_logits = (queries * positives).sum(dim=1, keepdim=True) / temperature
    negative_logits = torch.einsum("bd,bnd->bn", queries, negatives) / temperature
    negative_logits = negative_logits.masked_fill(~negative_valid_mask, -torch.inf)
    logit_parts = [positive_logits, negative_logits]
    if (in_batch_positive_embeddings is None) != (in_batch_valid_mask is None):
        raise QueryAdapterError("in-batch embedding/mask 必须同时提供")
    if in_batch_positive_embeddings is not None and in_batch_valid_mask is not None:
        if in_batch_positive_embeddings.shape != (batch, dimension):
            raise QueryAdapterError("in-batch positive embedding shape 非法")
        if in_batch_valid_mask.dtype != torch.bool or in_batch_valid_mask.shape != (
            batch,
            batch,
        ):
            raise QueryAdapterError("in-batch valid mask shape/dtype 非法")
        batch_positives = F.normalize(
            in_batch_positive_embeddings.float(), p=2, dim=1, eps=1e-8
        )
        in_batch_logits = queries @ batch_positives.T / temperature
        logit_parts.append(
            in_batch_logits.masked_fill(~in_batch_valid_mask, -torch.inf)
        )
    logits = torch.cat(logit_parts, dim=1)
    per_query = torch.logsumexp(logits, dim=1) - positive_logits.squeeze(1)
    normalized_weights = query_weights.float() / query_weights.float().sum()
    return (normalized_weights * per_query).sum()


def fixed_candidate_metrics(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    negative_valid_mask: torch.Tensor,
) -> dict[str, float]:
    """Measure positive rank and margin against the prepared hard negatives."""

    if not bool(negative_valid_mask.any(dim=1).all().item()):
        raise QueryAdapterError("每个 Query 至少需要一个有效评测负样本")
    queries = F.normalize(query_embeddings.float(), p=2, dim=1, eps=1e-8)
    positives = F.normalize(positive_embeddings.float(), p=2, dim=1, eps=1e-8)
    negatives = F.normalize(negative_embeddings.float(), p=2, dim=2, eps=1e-8)
    positive_scores = (queries * positives).sum(dim=1)
    negative_scores = torch.einsum("bd,bnd->bn", queries, negatives).masked_fill(
        ~negative_valid_mask, -torch.inf
    )
    ranks = 1 + negative_scores.ge(positive_scores.unsqueeze(1)).sum(dim=1)
    hardest = negative_scores.max(dim=1).values
    return {
        "recall_at_1": float(ranks.le(1).float().mean().item()),
        "recall_at_10": float(ranks.le(10).float().mean().item()),
        "recall_at_50": float(ranks.le(50).float().mean().item()),
        "mean_positive_score": float(positive_scores.mean().item()),
        "mean_hardest_negative_score": float(hardest.mean().item()),
        "mean_hard_margin": float((positive_scores - hardest).mean().item()),
    }


def stable_train_dev_split(
    query_ids: Sequence[int], *, dev_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Create a deterministic fixed-fraction dev split from Train Query IDs."""

    if not 0 < dev_fraction < 1 or len(query_ids) < 2:
        raise QueryAdapterError("dev_fraction/query_ids 非法")
    scored: list[tuple[int, int]] = []
    for index, query_id in enumerate(query_ids):
        if query_id < 0:
            raise QueryAdapterError("query_id 不能为负数")
        digest = hashlib.blake2b(
            f"{seed}:{query_id}".encode("ascii"),
            digest_size=8,
            person=b"qg-dev-v1",
        ).digest()
        scored.append((int.from_bytes(digest, "big"), index))
    dev_size = min(len(query_ids) - 1, max(1, round(len(query_ids) * dev_fraction)))
    dev = sorted(index for _, index in sorted(scored)[:dev_size])
    dev_set = set(dev)
    train = [index for index in range(len(query_ids)) if index not in dev_set]
    return train, dev


@dataclass(frozen=True)
class AdapterTrainingResult:
    raw_metrics: dict[str, float]
    adapted_metrics: dict[str, float]
    epoch_losses: tuple[float, ...]


def fit_query_adapter(
    model: ResidualQueryAdapter,
    raw_queries: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    negative_valid_mask: torch.Tensor,
    query_weights: torch.Tensor,
    target_poi_rows: torch.Tensor,
    reasonable_positive_rows: Sequence[Iterable[int]],
    train_indices: Sequence[int],
    dev_indices: Sequence[int],
    config: QueryAdapterConfig,
    *,
    device: torch.device,
    batch_size: int | None = None,
    seed: int = 42,
) -> AdapterTrainingResult:
    """Fit the Adapter on prepared Train-only tensors and evaluate fixed dev."""

    if not train_indices or not dev_indices:
        raise QueryAdapterError("Train/dev indices 都不能为空")
    if raw_queries.shape != positives.shape or raw_queries.ndim != 2:
        raise QueryAdapterError("raw_queries/positives shape 非法")
    if len(raw_queries) != len(query_weights) or negatives.shape[0] != len(raw_queries):
        raise QueryAdapterError("Adapter 训练 tensor 首维不一致")
    if target_poi_rows.dtype != torch.long or target_poi_rows.shape != (
        len(raw_queries),
    ):
        raise QueryAdapterError("target_poi_rows 必须是 int64[rows]")
    if len(reasonable_positive_rows) != len(raw_queries):
        raise QueryAdapterError("reasonable_positive_rows 长度必须等于 rows")
    effective_batch = batch_size or config.batch_size
    if effective_batch <= 0:
        raise QueryAdapterError("batch_size 必须为正数")
    torch.manual_seed(seed)
    model.to(device)
    raw_queries = raw_queries.to(device)
    positives = positives.to(device)
    negatives = negatives.to(device)
    negative_valid_mask = negative_valid_mask.to(device)
    query_weights = query_weights.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    dev_tensor = torch.tensor(dev_indices, dtype=torch.long, device=device)
    with torch.no_grad():
        raw_metrics = fixed_candidate_metrics(
            raw_queries[dev_tensor],
            positives[dev_tensor],
            negatives[dev_tensor],
            negative_valid_mask[dev_tensor],
        )
    losses: list[float] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_tensor_cpu = torch.tensor(train_indices, dtype=torch.long)
    for _ in range(config.epochs):
        model.train()
        permutation = train_tensor_cpu[
            torch.randperm(len(train_tensor_cpu), generator=generator)
        ]
        weighted_loss = 0.0
        seen = 0
        for start in range(0, len(permutation), effective_batch):
            batch_indices = permutation[start : start + effective_batch]
            indices = batch_indices.to(device)
            batch_targets = target_poi_rows[batch_indices]
            batch_reasonable = [
                reasonable_positive_rows[int(index)] for index in batch_indices
            ]
            in_batch_mask = build_in_batch_valid_mask(
                batch_targets,
                batch_reasonable,
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            amp_enabled = device.type == "cuda" and config.precision in {
                "bf16",
                "fp16",
            }
            amp_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                adapted = model(raw_queries[indices])
                loss = weighted_info_nce(
                    adapted,
                    positives[indices],
                    negatives[indices],
                    negative_valid_mask[indices],
                    query_weights[indices],
                    temperature=config.temperature,
                    in_batch_positive_embeddings=positives[indices],
                    in_batch_valid_mask=in_batch_mask,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            weighted_loss += float(loss.detach().item()) * len(indices)
            seen += len(indices)
        losses.append(weighted_loss / seen)
    model.eval()
    with torch.no_grad():
        adapted_metrics = fixed_candidate_metrics(
            model(raw_queries[dev_tensor]),
            positives[dev_tensor],
            negatives[dev_tensor],
            negative_valid_mask[dev_tensor],
        )
    return AdapterTrainingResult(
        raw_metrics=raw_metrics,
        adapted_metrics=adapted_metrics,
        epoch_losses=tuple(losses),
    )


def recommend_identity_fallback(
    raw_overall: dict[str, float],
    adapted_overall: dict[str, float],
    raw_hard: dict[str, float],
    adapted_hard: dict[str, float],
) -> bool:
    """Recommend identity only when neither overall nor hard subset improves."""

    overall_improved = (
        adapted_overall["recall_at_10"] > raw_overall["recall_at_10"]
        or adapted_overall["mean_hard_margin"] > raw_overall["mean_hard_margin"]
    )
    hard_improved = (
        adapted_hard["recall_at_10"] > raw_hard["recall_at_10"]
        or adapted_hard["mean_hard_margin"] > raw_hard["mean_hard_margin"]
    )
    return not overall_improved and not hard_improved


def adapter_gate_passes(
    raw_overall: dict[str, float],
    adapted_overall: dict[str, float],
    raw_hard: dict[str, float],
    adapted_hard: dict[str, float],
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Require non-regression plus a Recall@10 or margin gain in both subsets."""

    def improved(raw: dict[str, float], adapted: dict[str, float]) -> bool:
        recall_delta = adapted["recall_at_10"] - raw["recall_at_10"]
        margin_delta = adapted["mean_hard_margin"] - raw["mean_hard_margin"]
        return (
            recall_delta >= -tolerance
            and margin_delta >= -tolerance
            and (recall_delta > tolerance or margin_delta > tolerance)
        )

    return improved(raw_overall, adapted_overall) and improved(
        raw_hard, adapted_hard
    )

"""Minimal end-to-end objective for jointly training TIGER SID and Qwen."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from poi_gr.methods.tiger_joint.data import (
    IGNORE_INDEX,
    JointTigerBatch,
    MaterializedSidBatch,
    SID_LEVEL_COUNT,
    SidTokenLayout,
    TigerJointDataError,
    materialize_dynamic_sid_batch,
)
from poi_gr.sid.rqvae import RQVAE


@dataclass(frozen=True)
class JointGeneratorOutput:
    """Causal logits and only the last hidden layer needed by joint training."""

    logits: torch.Tensor
    last_hidden_state: torch.Tensor


class HuggingFaceCausalLMJointAdapter(nn.Module):
    """Expose Qwen logits and last hidden state without retaining every layer."""

    def __init__(self, causal_lm: nn.Module) -> None:
        super().__init__()
        if not isinstance(causal_lm, nn.Module):
            raise TypeError("causal_lm 必须是 torch.nn.Module")
        if not isinstance(getattr(causal_lm, "model", None), nn.Module):
            raise ValueError("causal_lm 必须暴露 Hugging Face base model 属性 model")
        output_head = self._resolve_output_head(causal_lm)
        if not isinstance(output_head, nn.Module):
            raise ValueError("causal_lm 必须暴露可用的 output embedding/lm_head")
        self.causal_lm = causal_lm

    @staticmethod
    def _resolve_output_head(causal_lm: nn.Module) -> nn.Module | None:
        get_output_embeddings = getattr(causal_lm, "get_output_embeddings", None)
        if callable(get_output_embeddings):
            output_head = get_output_embeddings()
            if isinstance(output_head, nn.Module):
                return output_head
        output_head = getattr(causal_lm, "lm_head", None)
        return output_head if isinstance(output_head, nn.Module) else None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> JointGeneratorOutput:
        base_output = self.causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        last_hidden_state = getattr(base_output, "last_hidden_state", None)
        if not isinstance(last_hidden_state, torch.Tensor):
            raise RuntimeError("base model 未返回 last_hidden_state")
        output_head = self._resolve_output_head(self.causal_lm)
        if output_head is None:
            raise RuntimeError("causal_lm output head 在前向时不可用")
        return JointGeneratorOutput(
            logits=output_head(last_hidden_state),
            last_hidden_state=last_hidden_state,
        )


def causal_language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute next-token loss for pre-masked Assistant-only labels."""

    if logits.ndim != 3:
        raise TigerJointDataError("logits shape 必须是 [batch, sequence, vocab]")
    if labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise TigerJointDataError("labels shape 必须与 logits 前两维一致")
    if labels.dtype != torch.long:
        raise TigerJointDataError("labels 必须是 torch.long tensor")
    if logits.device != labels.device:
        raise TigerJointDataError("logits 与 labels 必须位于同一 device")
    if bool((labels[:, 0] != IGNORE_INDEX).any().item()):
        raise TigerJointDataError("序列首位 label 无法由 causal LM 预测，必须为 -100")

    shifted_labels = labels[:, 1:].contiguous()
    supervised = shifted_labels != IGNORE_INDEX
    if not bool(supervised.any().item()):
        raise TigerJointDataError("batch 中没有可监督的 Assistant Token")
    supervised_labels = shifted_labels[supervised]
    if bool((supervised_labels < 0).any().item()) or bool(
        (supervised_labels >= logits.shape[2]).any().item()
    ):
        raise TigerJointDataError("labels 包含超出 generator vocabulary 的 Token ID")
    shifted_logits = logits[:, :-1, :].contiguous().float()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def teacher_forced_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return supervised-token accuracy and per-example exact-match rate."""

    if logits.ndim != 3:
        raise TigerJointDataError("logits shape 必须是 [batch, sequence, vocab]")
    if labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise TigerJointDataError("labels shape 必须与 logits 前两维一致")
    if labels.dtype != torch.long:
        raise TigerJointDataError("labels 必须是 torch.long tensor")
    if logits.device != labels.device:
        raise TigerJointDataError("logits 与 labels 必须位于同一 device")
    if bool((labels[:, 0] != IGNORE_INDEX).any().item()):
        raise TigerJointDataError("序列首位 label 无法由 causal LM 预测，必须为 -100")

    shifted_labels = labels[:, 1:]
    supervised = shifted_labels != IGNORE_INDEX
    if not bool(supervised.any().item()):
        raise TigerJointDataError("batch 中没有可监督的 Assistant Token")
    if not bool(supervised.any(dim=1).all().item()):
        raise TigerJointDataError("每个样本都必须至少包含一个 Assistant Token")
    supervised_labels = shifted_labels[supervised]
    if bool((supervised_labels < 0).any().item()) or bool(
        (supervised_labels >= logits.shape[2]).any().item()
    ):
        raise TigerJointDataError("labels 包含超出 generator vocabulary 的 Token ID")

    predictions = logits[:, :-1, :].argmax(dim=-1)
    correct = predictions.eq(shifted_labels) & supervised
    token_accuracy = correct.sum(dtype=torch.float32) / supervised.sum()
    exact_match = (correct | ~supervised).all(dim=1).float().mean()
    return token_accuracy, exact_match


def multi_positive_info_nce(
    query_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    target_poi_rows: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Use every duplicate target POI in a batch as a positive, not a negative."""

    if query_embeddings.ndim != 2 or item_embeddings.ndim != 2:
        raise TigerJointDataError("对齐 Embedding 必须是二维 tensor")
    if query_embeddings.shape != item_embeddings.shape or not min(
        query_embeddings.shape
    ):
        raise TigerJointDataError("Query 与 Item Embedding shape 必须相同且非空")
    if target_poi_rows.dtype != torch.long or target_poi_rows.shape != (
        query_embeddings.shape[0],
    ):
        raise TigerJointDataError("target_poi_rows shape 必须是 [batch]")
    if (
        query_embeddings.device != item_embeddings.device
        or target_poi_rows.device != query_embeddings.device
    ):
        raise TigerJointDataError("对齐输入必须位于同一 device")
    if temperature <= 0:
        raise TigerJointDataError("InfoNCE temperature 必须为正数")

    queries = F.normalize(query_embeddings.float(), p=2, dim=1, eps=1e-8)
    items = F.normalize(item_embeddings.float(), p=2, dim=1, eps=1e-8)
    logits = queries @ items.t() / temperature
    positive_mask = target_poi_rows.unsqueeze(1).eq(target_poi_rows.unsqueeze(0))
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    return (
        torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    ).mean()


@dataclass(frozen=True)
class JointTigerLossOutput:
    """Joint loss components and current dynamic SID materialization."""

    total_loss: torch.Tensor
    generation_loss: torch.Tensor
    alignment_loss: torch.Tensor
    rq_loss: torch.Tensor
    teacher_forced_token_accuracy: torch.Tensor
    teacher_forced_exact_match: torch.Tensor
    teacher_forced_sid_token_accuracy: torch.Tensor
    teacher_forced_static_token_accuracy: torch.Tensor
    materialized: MaterializedSidBatch
    history_codes: torch.Tensor
    target_codes: torch.Tensor


class JointTigerTrainingModule(nn.Module):
    """Own every trainable joint component behind one DDP forward boundary."""

    def __init__(
        self,
        *,
        generator: nn.Module,
        rqvae: RQVAE,
        query_projection: nn.Module,
        token_layout: SidTokenLayout,
        alignment_weight: float,
        rq_weight: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.rqvae = rqvae
        self.query_projection = query_projection
        self.token_layout = token_layout
        self.alignment_weight = alignment_weight
        self.rq_weight = rq_weight
        self.temperature = temperature

    def forward(self, batch: JointTigerBatch) -> JointTigerLossOutput:
        return compute_joint_tiger_loss(
            generator=self.generator,
            rqvae=self.rqvae,
            query_projection=self.query_projection,
            batch=batch,
            token_layout=self.token_layout,
            alignment_weight=self.alignment_weight,
            rq_weight=self.rq_weight,
            temperature=self.temperature,
        )


def _validate_generator_output(
    output: JointGeneratorOutput,
    *,
    batch_size: int,
    sequence_length: int,
) -> None:
    if not isinstance(output, JointGeneratorOutput):
        raise TypeError("generator 必须返回 JointGeneratorOutput")
    if output.logits.ndim != 3 or output.logits.shape[:2] != (
        batch_size,
        sequence_length,
    ):
        raise TigerJointDataError("generator logits shape 无效")
    if output.last_hidden_state.ndim != 3 or (
        output.last_hidden_state.shape[:2] != (batch_size, sequence_length)
    ):
        raise TigerJointDataError("generator last_hidden_state shape 无效")


def compute_joint_tiger_loss(
    *,
    generator: nn.Module,
    rqvae: RQVAE,
    query_projection: nn.Module,
    batch: JointTigerBatch,
    token_layout: SidTokenLayout,
    alignment_weight: float,
    rq_weight: float,
    temperature: float = 0.07,
) -> JointTigerLossOutput:
    """Materialize current SIDs and compute one differentiable joint objective."""

    if alignment_weight <= 0 or rq_weight <= 0:
        raise TigerJointDataError("联合训练的 alignment_weight 与 rq_weight 必须为正数")
    if batch.embedding_dim != rqvae.input_dim:
        raise TigerJointDataError(
            f"Embedding dim {batch.embedding_dim} 与 RQ-VAE input_dim "
            f"{rqvae.input_dim} 不一致"
        )
    if tuple(rqvae.codebook_sizes) != token_layout.codebook_sizes:
        raise TigerJointDataError(
            "RQ-VAE codebook sizes 与 SID Token layout 容量不一致"
        )

    target_latent = rqvae.encode(batch.target_embeddings)
    target_quantizer = rqvae.quantizer(
        target_latent,
        compute_diversity=False,
    )
    target_codes = target_quantizer.codes
    history_count = batch.history_embeddings.shape[0]
    if history_count:
        with torch.no_grad():
            history_codes = rqvae.encode_codes(batch.history_embeddings)
    else:
        history_codes = torch.empty(
            (0, SID_LEVEL_COUNT),
            dtype=torch.long,
            device=batch.target_embeddings.device,
        )
    materialized = materialize_dynamic_sid_batch(
        batch.template,
        history_codes,
        target_codes,
        token_layout,
    )

    raw_generator_output = generator(
        input_ids=materialized.input_ids,
        attention_mask=batch.template.attention_mask,
    )
    _validate_generator_output(
        raw_generator_output,
        batch_size=materialized.input_ids.shape[0],
        sequence_length=materialized.input_ids.shape[1],
    )
    generation_loss = causal_language_model_loss(
        raw_generator_output.logits,
        materialized.labels,
    )
    teacher_token_accuracy, teacher_exact_match = teacher_forced_accuracy(
        raw_generator_output.logits,
        materialized.labels,
    )
    shifted_predictions = raw_generator_output.logits[:, :-1].argmax(dim=-1)
    shifted_labels = materialized.labels[:, 1:]
    sid_supervision = torch.zeros_like(materialized.labels, dtype=torch.bool)
    target_samples = torch.arange(
        materialized.input_ids.shape[0],
        device=materialized.input_ids.device,
    ).unsqueeze(1)
    sid_supervision[
        target_samples,
        batch.template.target_sid_positions,
    ] = True
    shifted_sid_supervision = sid_supervision[:, 1:]
    shifted_all_supervision = shifted_labels != IGNORE_INDEX
    shifted_static_supervision = shifted_all_supervision & ~shifted_sid_supervision
    if not bool(shifted_sid_supervision.any().item()) or not bool(
        shifted_static_supervision.any().item()
    ):
        raise TigerJointDataError("目标必须同时包含动态 SID 与静态监督 Token")
    teacher_sid_token_accuracy = (
        shifted_predictions[shifted_sid_supervision]
        .eq(shifted_labels[shifted_sid_supervision])
        .float()
        .mean()
    )
    teacher_static_token_accuracy = (
        shifted_predictions[shifted_static_supervision]
        .eq(shifted_labels[shifted_static_supervision])
        .float()
        .mean()
    )

    sample_indices = torch.arange(
        materialized.input_ids.shape[0],
        device=materialized.input_ids.device,
    )
    query_hidden = raw_generator_output.last_hidden_state[
        sample_indices,
        batch.template.query_state_positions,
    ]
    projection_parameter = next(query_projection.parameters(), None)
    if projection_parameter is not None:
        query_hidden = query_hidden.to(dtype=projection_parameter.dtype)
    query_embeddings = query_projection(query_hidden)
    if query_embeddings.ndim != 2 or query_embeddings.shape != (
        materialized.input_ids.shape[0],
        rqvae.latent_dim,
    ):
        raise TigerJointDataError(
            "query_projection 输出必须是 [batch, rqvae.latent_dim]"
        )
    alignment_loss = multi_positive_info_nce(
        query_embeddings,
        target_quantizer.quantized_st,
        batch.template.target_poi_rows,
        temperature=temperature,
    )

    catalog_output = rqvae(batch.catalog_embeddings)
    rq_loss = catalog_output.total_loss
    total_loss = (
        generation_loss + alignment_weight * alignment_loss + rq_weight * rq_loss
    )
    components = torch.stack(
        (
            total_loss.float(),
            generation_loss.float(),
            alignment_loss.float(),
            rq_loss.float(),
        )
    )
    if not bool(torch.isfinite(components).all().item()):
        raise FloatingPointError("TIGER-Joint loss 出现 NaN 或 Inf")
    return JointTigerLossOutput(
        total_loss=total_loss,
        generation_loss=generation_loss,
        alignment_loss=alignment_loss,
        rq_loss=rq_loss,
        teacher_forced_token_accuracy=teacher_token_accuracy.detach(),
        teacher_forced_exact_match=teacher_exact_match.detach(),
        teacher_forced_sid_token_accuracy=teacher_sid_token_accuracy.detach(),
        teacher_forced_static_token_accuracy=(teacher_static_token_accuracy.detach()),
        materialized=materialized,
        history_codes=history_codes.detach(),
        target_codes=target_codes.detach(),
    )

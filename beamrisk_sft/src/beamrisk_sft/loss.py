"""Beam survival/final-rank pairwise objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .errors import BeamRiskError
from .scoring import score_completion_paths


@dataclass(frozen=True)
class BeamRiskLossOutput:
    loss: torch.Tensor
    per_pair_loss: torch.Tensor
    positive_scores: torch.Tensor
    negative_scores: torch.Tensor
    margins: torch.Tensor


def beam_risk_loss_from_scores(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    temperature: float,
    margin: float = 0.0,
) -> BeamRiskLossOutput:
    if positive_scores.ndim != 1 or positive_scores.shape != negative_scores.shape:
        raise BeamRiskError("BeamRisk 正负分数必须是相同 shape 的一维 Tensor")
    if not positive_scores.numel():
        raise BeamRiskError("BeamRisk batch 不能为空")
    if temperature <= 0:
        raise BeamRiskError("BeamRisk temperature 必须为正数")
    if not bool(torch.isfinite(positive_scores).all().item()) or not bool(
        torch.isfinite(negative_scores).all().item()
    ):
        raise BeamRiskError("BeamRisk 路径分数出现 NaN/Inf")
    margins = negative_scores - positive_scores
    per_pair = F.softplus((margins + float(margin)) / float(temperature))
    return BeamRiskLossOutput(
        loss=per_pair.mean(),
        per_pair_loss=per_pair,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        margins=margins,
    )


def score_and_compute_beam_risk_loss(
    model: Any,
    prompts: Sequence[Sequence[int]],
    positive_paths: Sequence[Sequence[int]],
    negative_paths: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    temperature: float,
    margin: float,
    device: torch.device | None = None,
) -> BeamRiskLossOutput:
    if not (
        len(prompts) == len(positive_paths) == len(negative_paths) and prompts
    ):
        raise BeamRiskError("BeamRisk prompt/positive/negative batch 数量不一致")
    if any(len(positive) != len(negative) for positive, negative in zip(positive_paths, negative_paths)):
        raise BeamRiskError("每个 BeamRisk pair 的正负路径长度必须一致")
    combined_prompts = [*prompts, *prompts]
    combined_paths = [*positive_paths, *negative_paths]
    scores = score_completion_paths(
        model,
        combined_prompts,
        combined_paths,
        pad_token_id=pad_token_id,
        device=device,
    ).sums
    count = len(prompts)
    return beam_risk_loss_from_scores(
        scores[:count],
        scores[count:],
        temperature=temperature,
        margin=margin,
    )

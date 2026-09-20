"""Memory-bounded differentiable completion-path scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .errors import BeamRiskError


@dataclass(frozen=True)
class PathScores:
    sums: torch.Tensor
    token_log_probs: torch.Tensor
    token_mask: torch.Tensor


def collate_completion_paths(
    prompts: Sequence[Sequence[int]],
    completions: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if len(prompts) != len(completions) or not prompts:
        raise BeamRiskError("Prompt 与 completion 必须等长且非空")
    if any(not prompt for prompt in prompts) or any(not path for path in completions):
        raise BeamRiskError("Prompt/completion 不能是空 Token 序列")
    width = max(len(prompt) + len(path) for prompt, path in zip(prompts, completions))
    max_completion = max(len(path) for path in completions)
    if width <= max_completion:
        raise BeamRiskError("每条 completion 前必须至少有一个 Prompt Token")
    count = len(prompts)
    input_ids = torch.full(
        (count, width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((count, width), dtype=torch.long, device=device)
    targets = torch.full(
        (count, max_completion),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    target_mask = torch.zeros(
        (count, max_completion), dtype=torch.bool, device=device
    )
    for index, (prompt, completion) in enumerate(zip(prompts, completions)):
        values = [int(token) for token in (*prompt, *completion)]
        if any(token < 0 for token in values):
            raise BeamRiskError("Token ID 不得为负数")
        start = width - len(values)
        input_ids[index, start:] = torch.tensor(values, dtype=torch.long, device=device)
        attention_mask[index, start:] = 1
        target_start = max_completion - len(completion)
        targets[index, target_start:] = torch.tensor(
            completion, dtype=torch.long, device=device
        )
        target_mask[index, target_start:] = True
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    # These shared positions predict the final max_completion input tokens.
    logit_positions = torch.arange(
        width - max_completion - 1,
        width - 1,
        dtype=torch.long,
        device=device,
    )
    return input_ids, attention_mask, position_ids, targets, target_mask, logit_positions


def score_completion_paths(
    model: Any,
    prompts: Sequence[Sequence[int]],
    completions: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device | None = None,
) -> PathScores:
    if device is None:
        parameter = next(model.parameters(), None)
        if parameter is None:
            raise BeamRiskError("无法从无参数模型推断 device")
        device = parameter.device
    (
        input_ids,
        attention_mask,
        position_ids,
        targets,
        target_mask,
        logit_positions,
    ) = collate_completion_paths(
        prompts,
        completions,
        pad_token_id=pad_token_id,
        device=device,
    )
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
        logits_to_keep=logit_positions,
    )
    logits = getattr(output, "logits", None)
    expected_shape = (*targets.shape,)
    if not isinstance(logits, torch.Tensor) or logits.shape[:2] != expected_shape:
        raise BeamRiskError(
            "模型 logits_to_keep 返回 shape 无效："
            f"{getattr(logits, 'shape', None)} vs {expected_shape}"
        )
    safe_targets = targets.masked_fill(~target_mask, 0)
    token_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
        dim=-1,
        index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~target_mask, 0.0)
    return PathScores(
        sums=token_log_probs.sum(dim=-1),
        token_log_probs=token_log_probs,
        token_mask=target_mask,
    )

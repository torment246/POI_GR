"""Teacher-forced token diagnostics for generative POI identifiers."""

from __future__ import annotations

import math
from typing import Any, Sequence


class TeacherForcingError(ValueError):
    """Raised when a teacher-forcing observation violates its contract."""


def empty_teacher_forcing_metrics() -> dict[str, Any]:
    """Return an empty JSON-serializable diagnostic accumulator."""

    return {
        "sample_count": 0,
        "target_token_count": 0,
        "target_nll_sum": 0.0,
        "positions": {},
        "aliases": {},
        "context_prefixes": {},
        "semantic_prefixes": {
            str(depth): {"count": 0, "top1_all_sum": 0, "top10_all_sum": 0}
            for depth in (1, 2, 3)
        },
        "full_identifier": {
            "count": 0,
            "top1_all_sum": 0,
            "top10_all_sum": 0,
        },
        "conditional_suffix": {
            "count": 0,
            "top1_all_sum": 0,
            "top10_all_sum": 0,
        },
        "conditional_tail": {
            "count": 0,
            "top1_all_sum": 0,
            "top10_all_sum": 0,
        },
        "full_serialization": {
            "count": 0,
            "top1_all_sum": 0,
            "top10_all_sum": 0,
        },
        "groups": {},
    }


def _empty_slot() -> dict[str, float | int]:
    return {"count": 0, "top1_sum": 0, "top10_sum": 0, "nll_sum": 0.0}


def _update_slot(
    slot: dict[str, float | int],
    *,
    top1: bool,
    top10: bool,
    nll: float,
) -> None:
    slot["count"] = int(slot["count"]) + 1
    slot["top1_sum"] = int(slot["top1_sum"]) + int(top1)
    slot["top10_sum"] = int(slot["top10_sum"]) + int(top10)
    slot["nll_sum"] = float(slot["nll_sum"]) + float(nll)


def update_teacher_forcing_metrics(
    metrics: dict[str, Any],
    *,
    position_names: Sequence[str],
    top1_correct: Sequence[bool],
    top10_correct: Sequence[bool],
    nll_values: Sequence[float],
    semantic_indices: Sequence[int],
    identifier_indices: Sequence[int],
    group: str,
    context_indices: Sequence[int] = (),
    conditional_tail_indices: Sequence[int] = (),
) -> None:
    """Add one teacher-forced target sequence to the accumulator."""

    length = len(position_names)
    if not (
        length
        == len(top1_correct)
        == len(top10_correct)
        == len(nll_values)
    ):
        raise TeacherForcingError("逐位置预测结果长度不一致")
    if length == 0 or len(semantic_indices) != 3:
        raise TeacherForcingError("目标序列必须包含三层 semantic identifier")
    all_indices = (
        *context_indices,
        *semantic_indices,
        *identifier_indices,
        *conditional_tail_indices,
    )
    if any(index < 0 or index >= length for index in all_indices):
        raise TeacherForcingError("目标位置索引越界")
    if not group:
        raise TeacherForcingError("group 不能为空")
    if context_indices and tuple(context_indices) != tuple(
        range(context_indices[0], context_indices[0] + len(context_indices))
    ):
        raise TeacherForcingError("context 位置必须连续且有序")
    if conditional_tail_indices and tuple(conditional_tail_indices) != tuple(
        range(
            conditional_tail_indices[0],
            conditional_tail_indices[0] + len(conditional_tail_indices),
        )
    ):
        raise TeacherForcingError("conditional tail 位置必须连续且有序")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in nll_values):
        raise TeacherForcingError("NLL 必须为有限非负数")

    metrics["sample_count"] += 1
    metrics["target_token_count"] += length
    metrics["target_nll_sum"] += sum(float(value) for value in nll_values)
    for index, name in enumerate(position_names):
        if not name:
            raise TeacherForcingError("position name 不能为空")
        slot = metrics["positions"].setdefault(name, _empty_slot())
        _update_slot(
            slot,
            top1=bool(top1_correct[index]),
            top10=bool(top10_correct[index]),
            nll=float(nll_values[index]),
        )

    decision_index = int(semantic_indices[-1]) + 1
    if decision_index < length:
        slot = metrics["aliases"].setdefault("after_sid3_decision", _empty_slot())
        _update_slot(
            slot,
            top1=bool(top1_correct[decision_index]),
            top10=bool(top10_correct[decision_index]),
            nll=float(nll_values[decision_index]),
        )

    for depth in range(1, len(context_indices) + 1):
        indices = context_indices[:depth]
        prefix = metrics["context_prefixes"].setdefault(
            str(depth),
            {"count": 0, "top1_all_sum": 0, "top10_all_sum": 0},
        )
        prefix["count"] += 1
        prefix["top1_all_sum"] += int(
            all(bool(top1_correct[index]) for index in indices)
        )
        prefix["top10_all_sum"] += int(
            all(bool(top10_correct[index]) for index in indices)
        )

    for depth in (1, 2, 3):
        indices = semantic_indices[:depth]
        prefix = metrics["semantic_prefixes"][str(depth)]
        prefix["count"] += 1
        prefix["top1_all_sum"] += int(
            all(bool(top1_correct[index]) for index in indices)
        )
        prefix["top10_all_sum"] += int(
            all(bool(top10_correct[index]) for index in indices)
        )

    identifier = metrics["full_identifier"]
    identifier["count"] += 1
    identifier["top1_all_sum"] += int(
        all(bool(top1_correct[index]) for index in identifier_indices)
    )
    identifier["top10_all_sum"] += int(
        all(bool(top10_correct[index]) for index in identifier_indices)
    )

    suffix_indices = tuple(
        index for index in identifier_indices if index > int(semantic_indices[-1])
    )
    if suffix_indices:
        suffix = metrics["conditional_suffix"]
        suffix["count"] += 1
        suffix["top1_all_sum"] += int(
            all(bool(top1_correct[index]) for index in suffix_indices)
        )
        suffix["top10_all_sum"] += int(
            all(bool(top10_correct[index]) for index in suffix_indices)
        )

    if conditional_tail_indices:
        tail = metrics["conditional_tail"]
        tail["count"] += 1
        tail["top1_all_sum"] += int(
            all(bool(top1_correct[index]) for index in conditional_tail_indices)
        )
        tail["top10_all_sum"] += int(
            all(bool(top10_correct[index]) for index in conditional_tail_indices)
        )

    serialization = metrics["full_serialization"]
    serialization["count"] += 1
    serialization["top1_all_sum"] += int(all(top1_correct))
    serialization["top10_all_sum"] += int(all(top10_correct))

    group_slot = metrics["groups"].setdefault(
        group,
        {
            "sample_count": 0,
            "target_token_count": 0,
            "target_nll_sum": 0.0,
            "identifier_top1_all_sum": 0,
            "identifier_top10_all_sum": 0,
            "suffix_count": 0,
            "suffix_top1_all_sum": 0,
            "suffix_top10_all_sum": 0,
            "tail_count": 0,
            "tail_top1_all_sum": 0,
            "tail_top10_all_sum": 0,
            "positions": {},
        },
    )
    group_slot["sample_count"] += 1
    group_slot["target_token_count"] += length
    group_slot["target_nll_sum"] += sum(float(value) for value in nll_values)
    group_slot["identifier_top1_all_sum"] += int(
        all(bool(top1_correct[index]) for index in identifier_indices)
    )
    group_slot["identifier_top10_all_sum"] += int(
        all(bool(top10_correct[index]) for index in identifier_indices)
    )
    if suffix_indices:
        group_slot["suffix_count"] += 1
        group_slot["suffix_top1_all_sum"] += int(
            all(bool(top1_correct[index]) for index in suffix_indices)
        )
        group_slot["suffix_top10_all_sum"] += int(
            all(bool(top10_correct[index]) for index in suffix_indices)
        )
    if conditional_tail_indices:
        group_slot["tail_count"] += 1
        group_slot["tail_top1_all_sum"] += int(
            all(bool(top1_correct[index]) for index in conditional_tail_indices)
        )
        group_slot["tail_top10_all_sum"] += int(
            all(bool(top10_correct[index]) for index in conditional_tail_indices)
        )
    for index, name in enumerate(position_names):
        slot = group_slot["positions"].setdefault(name, _empty_slot())
        _update_slot(
            slot,
            top1=bool(top1_correct[index]),
            top10=bool(top10_correct[index]),
            nll=float(nll_values[index]),
        )


def _finalize_slot(slot: dict[str, float | int]) -> dict[str, float | int]:
    count = int(slot["count"])
    if count <= 0:
        raise TeacherForcingError("位置指标 count 必须为正整数")
    mean_nll = float(slot["nll_sum"]) / count
    return {
        "count": count,
        "top1_accuracy": int(slot["top1_sum"]) / count,
        "top10_accuracy": int(slot["top10_sum"]) / count,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 80.0)),
    }


def _finalize_exact(slot: dict[str, int]) -> dict[str, float | int]:
    count = int(slot["count"])
    if count <= 0:
        raise TeacherForcingError("序列指标 count 必须为正整数")
    return {
        "count": count,
        "top1_all_accuracy": int(slot["top1_all_sum"]) / count,
        "top10_all_accuracy": int(slot["top10_all_sum"]) / count,
    }


def finalize_teacher_forcing_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert accumulator sums to reported teacher-forcing metrics."""

    sample_count = int(metrics["sample_count"])
    token_count = int(metrics["target_token_count"])
    if sample_count <= 0 or token_count <= 0:
        raise TeacherForcingError("没有可汇总的 teacher-forcing 样本")
    mean_nll = float(metrics["target_nll_sum"]) / token_count
    groups: dict[str, Any] = {}
    for name, slot in sorted(metrics["groups"].items()):
        group_samples = int(slot["sample_count"])
        group_tokens = int(slot["target_token_count"])
        group_nll = float(slot["target_nll_sum"]) / group_tokens
        groups[name] = {
            "sample_count": group_samples,
            "target_token_count": group_tokens,
            "mean_nll": group_nll,
            "perplexity": math.exp(min(group_nll, 80.0)),
            "identifier_top1_all_accuracy": (
                int(slot["identifier_top1_all_sum"]) / group_samples
            ),
            "identifier_top10_all_accuracy": (
                int(slot["identifier_top10_all_sum"]) / group_samples
            ),
            "conditional_suffix": (
                {
                    "count": int(slot["suffix_count"]),
                    "top1_all_accuracy": (
                        int(slot["suffix_top1_all_sum"]) / int(slot["suffix_count"])
                    ),
                    "top10_all_accuracy": (
                        int(slot["suffix_top10_all_sum"]) / int(slot["suffix_count"])
                    ),
                }
                if int(slot["suffix_count"]) > 0
                else None
            ),
            "conditional_tail": (
                {
                    "count": int(slot.get("tail_count", 0)),
                    "top1_all_accuracy": (
                        int(slot.get("tail_top1_all_sum", 0))
                        / int(slot["tail_count"])
                    ),
                    "top10_all_accuracy": (
                        int(slot.get("tail_top10_all_sum", 0))
                        / int(slot["tail_count"])
                    ),
                }
                if int(slot.get("tail_count", 0)) > 0
                else None
            ),
            "positions": {
                position_name: _finalize_slot(position_slot)
                for position_name, position_slot in sorted(
                    slot.get("positions", {}).items()
                )
            },
        }
    return {
        "sample_count": sample_count,
        "target_token_count": token_count,
        "mean_target_nll": mean_nll,
        "target_perplexity": math.exp(min(mean_nll, 80.0)),
        "positions": {
            name: _finalize_slot(slot)
            for name, slot in sorted(metrics["positions"].items())
        },
        "aliases": {
            name: _finalize_slot(slot)
            for name, slot in sorted(metrics["aliases"].items())
        },
        "context_prefixes": {
            name: _finalize_exact(slot)
            for name, slot in metrics["context_prefixes"].items()
        },
        "semantic_prefixes": {
            name: _finalize_exact(slot)
            for name, slot in metrics["semantic_prefixes"].items()
        },
        "full_identifier": _finalize_exact(metrics["full_identifier"]),
        "conditional_suffix": (
            _finalize_exact(metrics["conditional_suffix"])
            if int(metrics["conditional_suffix"]["count"]) > 0
            else None
        ),
        "conditional_tail": (
            _finalize_exact(metrics["conditional_tail"])
            if int(metrics["conditional_tail"]["count"]) > 0
            else None
        ),
        "full_serialization": _finalize_exact(metrics["full_serialization"]),
        "groups": groups,
    }


def score_target_logits(logits: Any, target_ids: Any) -> tuple[Any, Any, Any]:
    """Return target Top-1, Top-10 and NLL tensors for selected logits."""

    import torch

    if logits.ndim != 2 or target_ids.ndim != 1 or logits.shape[0] != target_ids.shape[0]:
        raise TeacherForcingError("logits 与 target_ids shape 不兼容")
    if logits.shape[1] < 10:
        raise TeacherForcingError("词表大小必须至少为 10")
    targets = target_ids.to(device=logits.device, dtype=torch.long)
    top_ids = torch.topk(logits, k=10, dim=-1).indices
    top1 = top_ids[:, 0].eq(targets)
    top10 = top_ids.eq(targets[:, None]).any(dim=-1)
    float_logits = logits.float()
    target_logits = float_logits.gather(1, targets[:, None]).squeeze(1)
    nll = torch.logsumexp(float_logits, dim=-1) - target_logits
    return top1, top10, nll

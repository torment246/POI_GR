"""Exact live-Beam tracing and survive-then-rank state classification."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterator, Sequence

from .errors import BeamRiskError


@dataclass(frozen=True)
class LiveBeamStep:
    """The non-finished beams retained for the next decoding iteration."""

    step: int
    prefixes: tuple[tuple[tuple[int, ...], ...], ...]
    scores: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RiskDecision:
    """One mutually exclusive Beam-risk state for a request."""

    state: str
    risk_type: str | None
    first_prune_depth: int | None
    gold_rank: int | None
    positive_token_ids: tuple[int, ...] | None
    negative_token_ids: tuple[int, ...] | None
    boundary_rank: int | None
    reference_negative_score: float | None


class LiveBeamRecorder:
    """Temporarily hook Transformers' retained-live-beam operation.

    Transformers 4.52 performs Beam search inside ``GenerationMixin._beam_search``.
    Hooking ``_get_running_beams_for_next_iteration`` records the exact beams that
    survive EOS/stopping filtering, instead of reproducing an approximate decoder.
    """

    def __init__(self, model: Any, *, prompt_width: int, expected_beams: int) -> None:
        if prompt_width <= 0 or expected_beams <= 0:
            raise BeamRiskError("Beam recorder 的 prompt_width/expected_beams 必须为正")
        self.model = model
        self.prompt_width = prompt_width
        self.expected_beams = expected_beams
        self.steps: list[LiveBeamStep] = []
        self._attribute = "_get_running_beams_for_next_iteration"
        self._had_instance_attribute = False
        self._previous_instance_attribute: Any = None

    def __enter__(self) -> "LiveBeamRecorder":
        if not hasattr(self.model, self._attribute):
            raise BeamRiskError(
                "当前 Transformers 模型没有可追踪的 live-Beam 扩展点"
            )
        instance_dictionary = getattr(self.model, "__dict__", {})
        self._had_instance_attribute = self._attribute in instance_dictionary
        if self._had_instance_attribute:
            self._previous_instance_attribute = instance_dictionary[self._attribute]
        original = getattr(self.model, self._attribute)
        recorder = self

        def wrapped(model_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 3:
                raise BeamRiskError("Transformers live-Beam 返回契约发生变化")
            running_sequences, running_scores, _ = result
            step = len(recorder.steps) + 1
            if running_sequences.ndim != 3 or running_scores.ndim != 2:
                raise BeamRiskError("Transformers live-Beam Tensor shape 无效")
            if running_sequences.shape[:2] != running_scores.shape:
                raise BeamRiskError("Transformers live-Beam sequence/score shape 不一致")
            if int(running_sequences.shape[1]) != recorder.expected_beams:
                raise BeamRiskError("实际 Beam 数与配置不一致")
            end = recorder.prompt_width + step
            if end > int(running_sequences.shape[2]):
                raise BeamRiskError("live-Beam 静态序列宽度不足")
            generated = (
                running_sequences[:, :, recorder.prompt_width:end]
                .detach()
                .cpu()
                .tolist()
            )
            scores = running_scores.detach().float().cpu().tolist()
            recorder.steps.append(
                LiveBeamStep(
                    step=step,
                    prefixes=tuple(
                        tuple(tuple(int(token) for token in beam) for beam in sample)
                        for sample in generated
                    ),
                    scores=tuple(
                        tuple(float(score) for score in sample) for sample in scores
                    ),
                )
            )
            return result

        setattr(self.model, self._attribute, MethodType(wrapped, self.model))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._had_instance_attribute:
            setattr(
                self.model,
                self._attribute,
                self._previous_instance_attribute,
            )
        else:
            delattr(self.model, self._attribute)


@contextmanager
def record_live_beams(
    model: Any,
    *,
    prompt_width: int,
    expected_beams: int,
) -> Iterator[LiveBeamRecorder]:
    with LiveBeamRecorder(
        model,
        prompt_width=prompt_width,
        expected_beams=expected_beams,
    ) as recorder:
        yield recorder


def _rank(prefixes: Sequence[Sequence[int]], target: Sequence[int]) -> int | None:
    target_tuple = tuple(int(token) for token in target)
    return next(
        (
            index
            for index, prefix in enumerate(prefixes, start=1)
            if tuple(int(token) for token in prefix) == target_tuple
        ),
        None,
    )


def classify_beam_risk(
    *,
    gold_token_ids: Sequence[int],
    sid_positions: Sequence[int],
    live_steps: Sequence[LiveBeamStep],
    sample_index: int,
    final_sequences: Sequence[Sequence[int]],
) -> RiskDecision:
    """Classify one request using exact retained beams.

    ``sid_positions`` are zero-based locations of S1/S2/S3/C inside the
    generated Assistant sequence. Only a first prune at those four semantic
    positions becomes a survival pair. Static-wrapper failures are reported but
    deliberately not trained as SID risk.
    """

    gold = tuple(int(token) for token in gold_token_ids)
    positions = tuple(int(position) for position in sid_positions)
    if not gold or len(positions) != 4 or tuple(sorted(positions)) != positions:
        raise BeamRiskError("gold target 或四个 SID 位置无效")
    if positions[0] < 0 or positions[-1] >= len(gold):
        raise BeamRiskError("SID 位置超出 gold target")
    if len(live_steps) <= positions[-1]:
        raise BeamRiskError("Beam trace 未覆盖 C Token")
    if sample_index < 0:
        raise BeamRiskError("sample_index 必须非负")

    # A failure before S1 is a formatting/static-token issue, not an SID pair.
    for step_number in range(1, positions[0] + 1):
        step = live_steps[step_number - 1]
        try:
            prefixes = step.prefixes[sample_index]
        except IndexError as error:
            raise BeamRiskError("sample_index 超出 Beam trace batch") from error
        if _rank(prefixes, gold[:step_number]) is None:
            return RiskDecision(
                state="pre_sid_prune",
                risk_type=None,
                first_prune_depth=None,
                gold_rank=None,
                positive_token_ids=None,
                negative_token_ids=None,
                boundary_rank=None,
                reference_negative_score=None,
            )

    for depth, position in enumerate(positions, start=1):
        generated_length = position + 1
        step = live_steps[generated_length - 1]
        prefixes = step.prefixes[sample_index]
        scores = step.scores[sample_index]
        if len(prefixes) != len(scores) or not prefixes:
            raise BeamRiskError("Beam trace prefix/score 数量不一致")
        gold_prefix = gold[:generated_length]
        gold_rank = _rank(prefixes, gold_prefix)
        if gold_rank is None:
            return RiskDecision(
                state="miss@10",
                risk_type="first_prune",
                first_prune_depth=depth,
                gold_rank=None,
                positive_token_ids=gold_prefix,
                negative_token_ids=tuple(prefixes[-1]),
                boundary_rank=len(prefixes),
                reference_negative_score=float(scores[-1]),
            )

    final_rank = next(
        (
            index
            for index, sequence in enumerate(final_sequences, start=1)
            if tuple(int(token) for token in sequence[: len(gold)]) == gold
        ),
        None,
    )
    if final_rank == 1:
        return RiskDecision(
            state="hit@1",
            risk_type=None,
            first_prune_depth=None,
            gold_rank=1,
            positive_token_ids=None,
            negative_token_ids=None,
            boundary_rank=None,
            reference_negative_score=None,
        )
    if final_rank is not None:
        if not final_sequences:
            raise BeamRiskError("final rank 有效但没有 final sequence")
        return RiskDecision(
            state="hit@10_not@1",
            risk_type="final_rank",
            first_prune_depth=None,
            gold_rank=final_rank,
            positive_token_ids=gold,
            negative_token_ids=tuple(
                int(token) for token in final_sequences[0][: len(gold)]
            ),
            boundary_rank=1,
            reference_negative_score=None,
        )
    return RiskDecision(
        state="miss_after_sid",
        risk_type=None,
        first_prune_depth=None,
        gold_rank=None,
        positive_token_ids=None,
        negative_token_ids=None,
        boundary_rank=None,
        reference_negative_score=None,
    )

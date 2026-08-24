"""History-aware sequence budgeting for map-search SFT prompts."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


HISTORY_OPEN = "<HISTORY>"
HISTORY_CLOSE = "</HISTORY>"
CURRENT_OPEN = "<CURRENT>"
CURRENT_CLOSE = "</CURRENT>"

_WRAPPED_EVENT_PATTERN = re.compile(r"<EVENT>.*?</EVENT>", re.DOTALL)
_GENPOI_EVENT_PATTERN = re.compile(
    r"<USER_GID>.*?</USER_GID>\s*"
    r"<QUERY>.*?</QUERY>\s*"
    r"<POI_PID>.*?</POI_PID>",
    re.DOTALL,
)


class HistoryBudgetError(ValueError):
    """Raised when a prompt cannot be made safe by removing history events."""


@dataclass(frozen=True)
class SequenceLengths:
    """Formatted source, target, and total token lengths."""

    source: int
    target: int

    @property
    def total(self) -> int:
        return self.source + self.target


@dataclass(frozen=True)
class HistoryBudgetResult:
    """A prompt fitted to a token budget without modifying current intent."""

    user_content: str
    original_history_events: int
    retained_history_events: int
    removed_history_events: int
    lengths_before: SequenceLengths
    lengths_after: SequenceLengths


@dataclass(frozen=True)
class _ParsedPrompt:
    before_history_body: str
    history_events: tuple[str, ...]
    after_history_body: str

    def render(self, retained_events: tuple[str, ...]) -> str:
        history_body = "\n" + "\n".join(retained_events) + "\n"
        return self.before_history_body + history_body + self.after_history_body


LengthMeasure = Callable[[str, str], SequenceLengths]


def _require_single_marker(content: str, marker: str) -> int:
    if content.count(marker) != 1:
        raise HistoryBudgetError(f"Prompt 必须且只能包含一个 {marker}")
    return content.index(marker)


def _parse_events(history_body: str) -> tuple[str, ...]:
    stripped = history_body.strip()
    if not stripped:
        return ()

    pattern = (
        _WRAPPED_EVENT_PATTERN
        if "<EVENT>" in stripped
        else _GENPOI_EVENT_PATTERN
    )
    matches = list(pattern.finditer(stripped))
    if not matches:
        raise HistoryBudgetError("无法识别 HISTORY 中的事件结构")

    cursor = 0
    events: list[str] = []
    for match in matches:
        if stripped[cursor : match.start()].strip():
            raise HistoryBudgetError("HISTORY 中包含无法归属到事件的内容")
        events.append(match.group(0).strip())
        cursor = match.end()
    if stripped[cursor:].strip():
        raise HistoryBudgetError("HISTORY 末尾包含无法归属到事件的内容")
    return tuple(events)


def _parse_prompt(user_content: str) -> _ParsedPrompt:
    history_open = _require_single_marker(user_content, HISTORY_OPEN)
    history_close = _require_single_marker(user_content, HISTORY_CLOSE)
    current_open = _require_single_marker(user_content, CURRENT_OPEN)
    current_close = _require_single_marker(user_content, CURRENT_CLOSE)
    history_body_start = history_open + len(HISTORY_OPEN)

    if not (
        history_body_start <= history_close < current_open < current_close
    ):
        raise HistoryBudgetError("HISTORY/CURRENT 标记顺序不合法")

    return _ParsedPrompt(
        before_history_body=user_content[:history_body_start],
        history_events=_parse_events(user_content[history_body_start:history_close]),
        after_history_body=user_content[history_close:],
    )


def fit_history_to_token_budget(
    user_content: str,
    target_content: str,
    cutoff_len: int,
    measure_lengths: LengthMeasure,
) -> HistoryBudgetResult:
    """Fit a prompt by removing only the oldest history events.

    History events are expected in ascending event-time order, matching the
    repository data contract. The CURRENT block and assistant target are never
    edited. If those immutable parts alone exceed the budget, the function
    raises instead of falling back to tokenizer-side truncation.
    """

    if cutoff_len <= 0:
        raise HistoryBudgetError("cutoff_len 必须大于 0")
    parsed = _parse_prompt(user_content)
    lengths_before = measure_lengths(user_content, target_content)
    if lengths_before.total <= cutoff_len:
        return HistoryBudgetResult(
            user_content=user_content,
            original_history_events=len(parsed.history_events),
            retained_history_events=len(parsed.history_events),
            removed_history_events=0,
            lengths_before=lengths_before,
            lengths_after=lengths_before,
        )

    events = parsed.history_events
    for removed_count in range(1, len(events) + 1):
        retained = events[removed_count:]
        candidate = parsed.render(retained)
        candidate_lengths = measure_lengths(candidate, target_content)
        if candidate_lengths.total <= cutoff_len:
            return HistoryBudgetResult(
                user_content=candidate,
                original_history_events=len(events),
                retained_history_events=len(retained),
                removed_history_events=removed_count,
                lengths_before=lengths_before,
                lengths_after=candidate_lengths,
            )

    no_history_content = parsed.render(())
    no_history_lengths = measure_lengths(no_history_content, target_content)
    raise HistoryBudgetError(
        "删除全部历史后仍超过 cutoff_len："
        f"{no_history_lengths.total} > {cutoff_len}；"
        "拒绝截断 CURRENT、当前 Query 或 Assistant 目标"
    )

"""Define Beijing-specific safe-prefix targets and GenPOI SSP calibration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


USEFUL_PREFIX_DEPTHS = (3, 4, 5, 6)


class BeijingSspError(ValueError):
    """Raised when Beijing SSP inputs violate the safe-prefix contract."""


@dataclass(frozen=True)
class SafePrefixMetrics:
    """Aggregate safety and coverage metrics for selected GID prefixes."""

    rows: int
    target_retention_rate: float
    unsafe_pruning_rate: float
    useful_prefix_rate: float
    strong_prefix_rate: float
    mean_prefix_depth: float
    prefix_depth_counts: dict[str, int]


def ordinal_targets(
    proximity_level: int,
    *,
    depths: Sequence[int] = USEFUL_PREFIX_DEPTHS,
) -> tuple[int, ...]:
    """Encode one common-prefix level as nested binary depth targets."""

    _validate_depths(depths)
    if not 0 <= proximity_level <= 6:
        raise BeijingSspError("proximity_level 必须位于 [0, 6]")
    return tuple(int(proximity_level >= depth) for depth in depths)


def select_safe_prefix_depth(
    probabilities: Sequence[float],
    thresholds: Mapping[int, float],
    *,
    depths: Sequence[int] = USEFUL_PREFIX_DEPTHS,
) -> int:
    """Select the deepest prefix whose cumulative decisions all pass."""

    _validate_depths(depths)
    if len(probabilities) != len(depths):
        raise BeijingSspError("probabilities 数量必须与 depths 一致")
    selected = 0
    for depth, probability in zip(depths, probabilities):
        threshold = thresholds.get(depth)
        if threshold is None or not 0.0 <= float(threshold) <= 1.0:
            raise BeijingSspError(f"depth={depth} 缺少合法阈值")
        value = float(probability)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise BeijingSspError("probability 必须是 [0, 1] 内有限值")
        if value <= float(threshold):
            break
        selected = depth
    return selected


def calibrate_safe_prefix_thresholds(
    probabilities: np.ndarray,
    proximity_levels: np.ndarray,
    *,
    max_unsafe_rate: float = 0.001,
    depths: Sequence[int] = USEFUL_PREFIX_DEPTHS,
) -> dict[int, float]:
    """Calibrate conservative per-depth thresholds under a union-bound budget."""

    _validate_depths(depths)
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(proximity_levels, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(depths):
        raise BeijingSspError("probabilities shape 必须为 [rows, len(depths)]")
    if labels.ndim != 1 or labels.shape[0] != values.shape[0] or labels.size == 0:
        raise BeijingSspError("proximity_levels 必须是一维非空且与 probabilities 对齐")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise BeijingSspError("probabilities 必须全部位于 [0, 1]")
    if np.any((labels < 0) | (labels > 6)):
        raise BeijingSspError("proximity_levels 必须位于 [0, 6]")
    if not 0.0 <= max_unsafe_rate < 1.0:
        raise BeijingSspError("max_unsafe_rate 必须位于 [0, 1)")

    total_budget = int(np.floor(labels.size * max_unsafe_rate))
    base_budget, remainder = divmod(total_budget, len(depths))
    thresholds: dict[int, float] = {}
    for index, depth in enumerate(depths):
        budget = base_budget + int(index < remainder)
        negative_scores = np.sort(values[labels < depth, index])[::-1]
        if negative_scores.size == 0:
            thresholds[depth] = 0.0
        elif budget >= negative_scores.size:
            thresholds[depth] = 0.0
        else:
            thresholds[depth] = float(negative_scores[budget])
    return thresholds


def evaluate_safe_prefixes(
    proximity_levels: Sequence[int],
    selected_depths: Sequence[int],
) -> SafePrefixMetrics:
    """Evaluate whether selected prefixes retain each target POI."""

    if len(proximity_levels) != len(selected_depths) or len(proximity_levels) == 0:
        raise BeijingSspError("标签和预测必须非空且等长")
    counts: Counter[int] = Counter()
    retained = 0
    depth_sum = 0
    for level, depth in zip(proximity_levels, selected_depths):
        level = int(level)
        depth = int(depth)
        if not 0 <= level <= 6:
            raise BeijingSspError("proximity_level 必须位于 [0, 6]")
        if depth not in (0, *USEFUL_PREFIX_DEPTHS):
            raise BeijingSspError("selected_depth 必须为 0、3、4、5 或 6")
        counts[depth] += 1
        retained += int(depth <= level)
        depth_sum += depth
    rows = len(proximity_levels)
    retention_rate = retained / rows
    return SafePrefixMetrics(
        rows=rows,
        target_retention_rate=retention_rate,
        unsafe_pruning_rate=1.0 - retention_rate,
        useful_prefix_rate=sum(count for depth, count in counts.items() if depth >= 3)
        / rows,
        strong_prefix_rate=sum(count for depth, count in counts.items() if depth >= 4)
        / rows,
        mean_prefix_depth=depth_sum / rows,
        prefix_depth_counts={
            str(depth): counts[depth] for depth in (0, *USEFUL_PREFIX_DEPTHS)
        },
    )


def _validate_depths(depths: Sequence[int]) -> None:
    normalized = tuple(int(depth) for depth in depths)
    if normalized != tuple(sorted(set(normalized))):
        raise BeijingSspError("depths 必须严格递增且不重复")
    if not normalized or normalized[0] < 1 or normalized[-1] > 6:
        raise BeijingSspError("depths 必须位于 [1, 6]")

"""Exact shortest distinguishing descriptions for one fixed collision catalog."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

import numpy as np


class MinimumDescriptionError(ValueError):
    """Raised when minimum-description inputs violate the contract."""


@dataclass(frozen=True)
class CollisionGrouping:
    """Rows grouped by the fixed identifier prefix preceding entity relations."""

    collision_rows: np.ndarray
    group_ids: np.ndarray
    group_sizes: np.ndarray
    member_order: np.ndarray
    group_offsets: np.ndarray


@dataclass(frozen=True)
class MinimumDescriptionCompilation:
    """Per-POI minimum relation descriptions in collision-row order."""

    path_types: np.ndarray
    path_values: np.ndarray
    resolved: np.ndarray
    metrics: dict[str, object]


@dataclass(frozen=True)
class _Candidate:
    relation: int
    same_mask: int
    token_cost: int


def base1024_digit_count(value: int) -> int:
    """Return the exact number of base-1024 value tokens."""

    if value < 0:
        raise MinimumDescriptionError("关系值不能为负数")
    digits = 1
    while value >= 1024:
        value //= 1024
        digits += 1
    return digits


def relation_token_cost(value: int) -> int:
    """Count one relation-type token plus its base-1024 value tokens."""

    return 1 + base1024_digit_count(value)


def _path_keys(
    path_types: np.ndarray, path_values: np.ndarray
) -> list[Hashable]:
    return [
        tuple(
            (int(relation_type), int(value))
            for relation_type, value in zip(types, values, strict=True)
            if relation_type >= 0
        )
        for types, values in zip(path_types, path_values, strict=True)
    ]


def _collision_stats(
    group_ids: np.ndarray, keys: Sequence[Hashable]
) -> tuple[dict[str, int], np.ndarray]:
    counts = Counter(
        (int(group_id), key)
        for group_id, key in zip(group_ids, keys, strict=True)
    )
    collision_sizes = [count for count in counts.values() if count > 1]
    resolved = np.asarray(
        [
            counts[(int(group_id), key)] == 1
            for group_id, key in zip(group_ids, keys, strict=True)
        ],
        dtype=np.bool_,
    )
    return (
        {
            "collision_excess": int(sum(size - 1 for size in collision_sizes)),
            "collision_poi_count": int(sum(collision_sizes)),
            "collision_group_count": len(collision_sizes),
            "max_collision_group_size": max(collision_sizes, default=1),
            "distinct_count": len(counts),
        },
        resolved,
    )


def _members(grouping: CollisionGrouping, group_id: int) -> np.ndarray:
    start = int(grouping.group_offsets[group_id])
    end = int(grouping.group_offsets[group_id + 1])
    return grouping.member_order[start:end]


def _same_mask(equal: np.ndarray) -> int:
    packed = np.packbits(equal, bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little")


def _reduce_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    by_mask: dict[int, _Candidate] = {}
    for candidate in candidates:
        current = by_mask.get(candidate.same_mask)
        if current is None or (
            candidate.token_cost,
            candidate.relation,
        ) < (
            current.token_cost,
            current.relation,
        ):
            by_mask[candidate.same_mask] = candidate
    unique = sorted(
        by_mask.values(), key=lambda item: (item.token_cost, item.relation)
    )
    retained: list[_Candidate] = []
    for candidate in unique:
        dominated = False
        for other in unique:
            if other is candidate:
                continue
            if (other.same_mask | candidate.same_mask) != candidate.same_mask:
                continue
            if (other.token_cost, other.relation) <= (
                candidate.token_cost,
                candidate.relation,
            ):
                dominated = True
                break
        if not dominated:
            retained.append(candidate)
    return retained


def _best_combination(
    candidates: list[_Candidate],
    *,
    max_pairs: int,
    max_tokens: int,
) -> tuple[int, ...] | None:
    best_key: tuple[int, int, tuple[int, ...]] | None = None
    best: tuple[int, ...] | None = None

    def consider(items: tuple[_Candidate, ...]) -> None:
        nonlocal best_key, best
        cost = sum(item.token_cost for item in items)
        if cost > max_tokens:
            return
        remaining = items[0].same_mask
        for item in items[1:]:
            remaining &= item.same_mask
        if remaining:
            return
        relations = tuple(sorted(item.relation for item in items))
        key = (cost, len(items), relations)
        if best_key is None or key < best_key:
            best_key = key
            best = relations

    for candidate in candidates:
        consider((candidate,))
    if max_pairs >= 2:
        for first_index, first in enumerate(candidates):
            for second in candidates[first_index + 1 :]:
                if first.token_cost + second.token_cost > max_tokens:
                    continue
                consider((first, second))
    if max_pairs >= 3:
        for first_index, first in enumerate(candidates):
            for second_index in range(first_index + 1, len(candidates)):
                second = candidates[second_index]
                partial_cost = first.token_cost + second.token_cost
                if partial_cost + 2 > max_tokens:
                    continue
                intersection = first.same_mask & second.same_mask
                if not intersection:
                    continue
                for third in candidates[second_index + 1 :]:
                    if partial_cost + third.token_cost > max_tokens:
                        continue
                    if intersection & third.same_mask:
                        continue
                    consider((first, second, third))
    return best


def _solve_row(
    relations: np.ndarray,
    row: int,
    semantic_tiers: np.ndarray,
    *,
    max_pairs: int,
    max_tokens: int,
) -> tuple[int, ...] | None:
    peers = np.concatenate(
        (
            np.arange(0, row, dtype=np.int64),
            np.arange(row + 1, len(relations), dtype=np.int64),
        )
    )
    full_mask = (1 << len(peers)) - 1
    present = np.flatnonzero(relations[row] >= 0)
    candidates: list[tuple[_Candidate, int]] = []
    for relation in present:
        value = int(relations[row, relation])
        mask = _same_mask(relations[peers, relation] == value)
        if mask == full_mask:
            continue
        candidates.append(
            (
                _Candidate(
                    relation=int(relation),
                    same_mask=mask,
                    token_cost=relation_token_cost(value),
                ),
                int(semantic_tiers[relation]),
            )
        )
    if not candidates:
        return None
    for tier in sorted({tier for _, tier in candidates}):
        eligible = [
            candidate
            for candidate, candidate_tier in candidates
            if candidate_tier <= tier
        ]
        result = _best_combination(
            _reduce_candidates(eligible),
            max_pairs=max_pairs,
            max_tokens=max_tokens,
        )
        if result is not None:
            return result
    return None


def compile_minimum_descriptions(
    *,
    grouping: CollisionGrouping,
    relation_values: np.ndarray,
    semantic_tiers: np.ndarray,
    max_pairs: int = 3,
    max_tokens: int = 6,
    progress: Callable[[str], None] | None = None,
) -> MinimumDescriptionCompilation:
    """Compile exact shortest per-POI descriptions without Query signals."""

    relations = np.asarray(relation_values)
    tiers = np.asarray(semantic_tiers)
    if relations.ndim != 2 or relations.shape[0] != len(grouping.group_ids):
        raise MinimumDescriptionError("relation_values 与 grouping 行数不一致")
    if tiers.shape != (relations.shape[1],):
        raise MinimumDescriptionError("semantic_tiers shape 非法")
    if max_pairs not in {1, 2, 3}:
        raise MinimumDescriptionError("max_pairs 当前只支持 1/2/3")
    if max_tokens < 2:
        raise MinimumDescriptionError("max_tokens 至少为 2")
    path_types = np.full(
        (len(relations), max_pairs), -1, dtype=np.int16
    )
    path_values = np.full(
        (len(relations), max_pairs), -1, dtype=np.int64
    )
    solved_depths: Counter[int] = Counter()
    no_description_count = 0
    group_count = len(grouping.group_sizes)
    for group_id in range(group_count):
        members = _members(grouping, group_id)
        bucket = relations[members]
        for local_row, global_row in enumerate(members):
            selected = _solve_row(
                bucket,
                local_row,
                tiers,
                max_pairs=max_pairs,
                max_tokens=max_tokens,
            )
            if selected is None:
                no_description_count += 1
                continue
            solved_depths[len(selected)] += 1
            for depth, relation in enumerate(selected):
                path_types[global_row, depth] = relation
                path_values[global_row, depth] = relations[global_row, relation]
        if progress is not None and (
            (group_id + 1) % 20_000 == 0 or group_id + 1 == group_count
        ):
            progress(
                f"最短描述已编译 {group_id + 1:,}/{group_count:,} 个 G6 碰撞组"
            )

    keys = _path_keys(path_types, path_values)
    stats, resolved = _collision_stats(grouping.group_ids, keys)
    pair_lengths = np.sum(path_types >= 0, axis=1)
    token_lengths = np.zeros(len(relations), dtype=np.int64)
    for row in range(len(relations)):
        token_lengths[row] = sum(
            relation_token_cost(int(value))
            for relation, value in zip(
                path_types[row], path_values[row], strict=True
            )
            if relation >= 0
        )
    stats.update(
        {
            "max_pairs": max_pairs,
            "max_tokens": max_tokens,
            "solved_description_count_by_pair_length": {
                str(length): int(count)
                for length, count in sorted(solved_depths.items())
            },
            "no_nonempty_description_count": no_description_count,
            "resolved_poi_count": int(np.count_nonzero(resolved)),
            "mean_pair_count": float(pair_lengths.mean()),
            "max_pair_count": int(pair_lengths.max(initial=0)),
            "mean_relation_token_count": float(token_lengths.mean()),
            "max_relation_token_count": int(token_lengths.max(initial=0)),
            "pair_count_distribution": {
                str(int(value)): int(count)
                for value, count in sorted(Counter(pair_lengths.tolist()).items())
            },
            "token_count_distribution": {
                str(int(value)): int(count)
                for value, count in sorted(Counter(token_lengths.tolist()).items())
            },
        }
    )
    return MinimumDescriptionCompilation(
        path_types=path_types,
        path_values=path_values,
        resolved=resolved,
        metrics=stats,
    )

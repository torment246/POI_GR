"""Deterministic hard-negative sources with mandatory false-negative masking."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class HardNegativeError(ValueError):
    """Raised when hard-negative inputs violate the P3 contract."""


@dataclass(frozen=True)
class PoiMetadata:
    row: int
    poi_id: str
    displayname: str
    alias: str
    category: str
    category_code: str
    address: str
    lat: float
    lng: float


@dataclass(frozen=True)
class NegativeCandidate:
    poi_row: int
    source: str
    score: float


def _validate_forbidden(target_row: int, false_negative_rows: Iterable[int]) -> set[int]:
    if target_row < 0:
        raise HardNegativeError("target_row 不能为负数")
    forbidden = {int(row) for row in false_negative_rows}
    if any(row < 0 for row in forbidden):
        raise HardNegativeError("false-negative POI row 不能为负数")
    forbidden.add(target_row)
    return forbidden


def in_batch_candidates(
    target_row: int,
    batch_target_rows: Sequence[int],
    false_negative_rows: Iterable[int],
) -> list[NegativeCandidate]:
    """Use distinct batch targets after removing every reasonable positive."""

    forbidden = _validate_forbidden(target_row, false_negative_rows)
    return [
        NegativeCandidate(row, "in_batch", 1.0)
        for row in sorted({int(value) for value in batch_target_rows} - forbidden)
    ]


def semantic_ann_candidates(
    target_row: int,
    poi_embeddings: np.ndarray,
    *,
    top_k: int,
    false_negative_rows: Iterable[int] = (),
    chunk_rows: int = 32_768,
) -> list[NegativeCandidate]:
    """Find exact cosine neighbors of the target POI with bounded RAM."""

    if poi_embeddings.ndim != 2 or not min(poi_embeddings.shape):
        raise HardNegativeError("POI embedding 必须是非空二维数组")
    if not 0 <= target_row < len(poi_embeddings):
        raise HardNegativeError("target_row 超出 POI embedding 范围")
    if top_k <= 0 or chunk_rows <= 0:
        raise HardNegativeError("top_k/chunk_rows 必须为正数")
    forbidden = _validate_forbidden(target_row, false_negative_rows)
    target = np.asarray(poi_embeddings[target_row], dtype=np.float32)
    target_norm = float(np.linalg.norm(target))
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise HardNegativeError("目标 POI embedding 是零范数或非有限值")
    target /= target_norm
    best_rows = np.empty(0, dtype=np.int64)
    best_scores = np.empty(0, dtype=np.float32)
    for start in range(0, len(poi_embeddings), chunk_rows):
        stop = min(start + chunk_rows, len(poi_embeddings))
        chunk = np.asarray(poi_embeddings[start:stop], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise HardNegativeError("POI embedding 包含 NaN/Inf")
        norms = np.linalg.norm(chunk, axis=1)
        scores = chunk @ target / np.maximum(norms, 1e-12)
        rows = np.arange(start, stop, dtype=np.int64)
        local_forbidden = np.fromiter(
            (row in forbidden for row in rows), dtype=np.bool_, count=len(rows)
        )
        scores[local_forbidden] = -np.inf
        keep = min(top_k, len(scores))
        if keep:
            indices = np.lexsort((rows, -scores))[:keep]
            finite = np.isfinite(scores[indices])
            best_rows = np.concatenate((best_rows, rows[indices][finite]))
            best_scores = np.concatenate((best_scores, scores[indices][finite]))
        if len(best_rows) > top_k:
            order = np.lexsort((best_rows, -best_scores))[:top_k]
            best_rows = best_rows[order]
            best_scores = best_scores[order]
    order = np.lexsort((best_rows, -best_scores))[:top_k]
    return [
        NegativeCandidate(int(best_rows[index]), "semantic_ann", float(best_scores[index]))
        for index in order
    ]


def _surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def _character_ngrams(value: str, width: int = 2) -> set[str]:
    compact = _surface(value).replace(" ", "")
    if not compact:
        return set()
    if len(compact) < width:
        return {compact}
    return {compact[index : index + width] for index in range(len(compact) - width + 1)}


def character_ngram_jaccard(left: str, right: str) -> float:
    """Measure conservative surface overlap without segmentation or rewriting."""

    left_ngrams = _character_ngrams(left)
    right_ngrams = _character_ngrams(right)
    if not left_ngrams or not right_ngrams:
        return 0.0
    return len(left_ngrams & right_ngrams) / len(left_ngrams | right_ngrams)


def _ngram_set_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    intersection = sum(value in larger for value in smaller)
    return intersection / (len(left) + len(right) - intersection)


def lexical_metadata_candidates(
    target: PoiMetadata,
    catalog: Sequence[PoiMetadata],
    *,
    top_k: int,
    false_negative_rows: Iterable[int] = (),
) -> list[NegativeCandidate]:
    """Rank name/alias overlap, then category and address signals."""

    if top_k <= 0:
        raise HardNegativeError("top_k 必须为正数")
    forbidden = _validate_forbidden(target.row, false_negative_rows)
    target_names = (
        _character_ngrams(target.displayname),
        _character_ngrams(target.alias),
    )
    target_address = _character_ngrams(target.address)
    ranked: list[tuple[float, int, int, float, int]] = []
    for candidate in catalog:
        if candidate.row in forbidden:
            continue
        candidate_names = (
            _character_ngrams(candidate.displayname),
            _character_ngrams(candidate.alias),
        )
        name_alias = max(
            _ngram_set_jaccard(left, right)
            for left in target_names
            for right in candidate_names
        )
        category_code_match = int(
            bool(target.category_code)
            and target.category_code == candidate.category_code
        )
        category_match = int(
            bool(target.category) and target.category == candidate.category
        )
        address = _ngram_set_jaccard(
            target_address, _character_ngrams(candidate.address)
        )
        if name_alias <= 0 and not category_code_match and not category_match and address <= 0:
            continue
        ranked.append(
            (name_alias, category_code_match, category_match, address, candidate.row)
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4]))
    return [
        NegativeCandidate(
            poi_row=row,
            source="lexical_metadata",
            score=float(name_alias),
        )
        for name_alias, _, _, _, row in ranked[:top_k]
    ]


def haversine_meters(left: PoiMetadata, right: PoiMetadata) -> float:
    """Return deterministic great-circle distance for Beijing-scale mining."""

    lat1, lon1, lat2, lon2 = map(
        math.radians, (left.lat, left.lng, right.lat, right.lng)
    )
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * 6_371_008.8 * math.asin(min(1.0, math.sqrt(value)))


def local_geo_candidates(
    target: PoiMetadata,
    catalog: Sequence[PoiMetadata],
    *,
    top_k: int,
    false_negative_rows: Iterable[int] = (),
) -> list[NegativeCandidate]:
    """Prefer nearby same-category POIs without using area/layer fields."""

    if top_k <= 0:
        raise HardNegativeError("top_k 必须为正数")
    forbidden = _validate_forbidden(target.row, false_negative_rows)
    ranked: list[tuple[int, int, float, int]] = []
    for candidate in catalog:
        if candidate.row in forbidden:
            continue
        distance = haversine_meters(target, candidate)
        if not math.isfinite(distance):
            raise HardNegativeError("POI 经纬度产生非有限距离")
        code_match = int(
            bool(target.category_code)
            and target.category_code == candidate.category_code
        )
        category_match = int(
            bool(target.category) and target.category == candidate.category
        )
        ranked.append((code_match, category_match, distance, candidate.row))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return [
        NegativeCandidate(
            poi_row=row,
            source="local_geo",
            score=float(-distance),
        )
        for _, _, distance, row in ranked[:top_k]
    ]


def merge_negative_sources(
    *,
    target_row: int,
    false_negative_rows: Iterable[int],
    in_batch: Sequence[NegativeCandidate],
    semantic_ann: Sequence[NegativeCandidate],
    lexical_metadata: Sequence[NegativeCandidate],
    local_geo: Sequence[NegativeCandidate],
    semantic_budget: int,
    lexical_budget: int,
    local_geo_budget: int,
) -> list[NegativeCandidate]:
    """Merge four sources by frozen priority and remove duplicates/positives."""

    if min(semantic_budget, lexical_budget, local_geo_budget) < 0:
        raise HardNegativeError("负样本 budget 不能为负数")
    forbidden = _validate_forbidden(target_row, false_negative_rows)
    result: list[NegativeCandidate] = []
    seen = set(forbidden)
    for candidates, budget in (
        (in_batch, len(in_batch)),
        (semantic_ann, semantic_budget),
        (lexical_metadata, lexical_budget),
        (local_geo, local_geo_budget),
    ):
        if budget == 0:
            continue
        selected = 0
        for candidate in candidates:
            if candidate.poi_row in seen:
                continue
            seen.add(candidate.poi_row)
            result.append(candidate)
            selected += 1
            if selected >= budget:
                break
    return result

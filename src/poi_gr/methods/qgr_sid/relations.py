"""Deterministic, high-precision numeric relation extraction for QGR-SID."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


RELATION_SCHEMA_VERSION = "qgr-sid-numeric-relations-v1"

RELATION_TYPES = (
    "R_PHASE",
    "R_ZONE_NUM",
    "R_ZONE_ALPHA",
    "R_ZONE_DIR",
    "R_BUILDING",
    "R_ADDRESS_NO",
    "R_QUALIFIER",
    "R_SUBNO",
    "R_UNIT",
    "R_FLOOR",
    "R_BASEMENT",
    "R_ROOM",
    "R_SHOP",
    "R_ENTITY_NO",
    "R_ENTRANCE_DIR",
    "R_ENTRANCE_ALPHA",
    "R_ENTRANCE_NO",
)

DIRECT_NUMERIC_RELATION_TYPES = frozenset(
    {
        "R_PHASE",
        "R_ZONE_NUM",
        "R_BUILDING",
        "R_ADDRESS_NO",
        "R_SUBNO",
        "R_UNIT",
        "R_FLOOR",
        "R_BASEMENT",
        "R_ROOM",
        "R_SHOP",
        "R_ENTITY_NO",
        "R_ENTRANCE_NO",
    }
)

SOURCE_PRIORITIES = {
    "displayname": 3,
    "address": 2,
    "alias": 1,
}

SOURCE_CONFIDENCE = {
    "displayname": 0.99,
    "address": 0.98,
    "alias": 0.92,
}

_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_QUALIFIERS = {
    "甲": 1,
    "乙": 2,
    "丙": 3,
    "丁": 4,
    "戊": 5,
    "己": 6,
    "庚": 7,
    "辛": 8,
    "壬": 9,
    "癸": 10,
}
_DIRECTIONS = {
    "北": 0,
    "东北": 1,
    "东": 2,
    "东南": 3,
    "南": 4,
    "西南": 5,
    "西": 6,
    "西北": 7,
    "中": 8,
}

_NUMBER_CHARACTER = r"0-9零〇一二两三四五六七八九十百千万"
_NUMBER = (
    rf"(?<![{_NUMBER_CHARACTER}])"
    rf"(?:[0-9]{{1,7}}|[零〇一二两三四五六七八九十百千万]{{1,10}})"
    rf"(?![{_NUMBER_CHARACTER}])"
)
_SHORT_NUMBER = (
    rf"(?<![{_NUMBER_CHARACTER}])"
    rf"(?:[0-9]{{1,2}}|[零〇一二两三四五六七八九十]{{1,3}})"
    rf"(?![{_NUMBER_CHARACTER}])"
)
_DIRECTION = r"(?:东北|东南|西南|西北|北|东|南|西|中)"
_QUALIFIER = r"[甲乙丙丁戊己庚辛壬癸]"


class NumericRelationError(ValueError):
    """Raised when raw POI relation inputs violate the extractor contract."""


@dataclass(frozen=True, order=True)
class NumericRelation:
    """One selected POI-level relation whose value is globally numeric."""

    relation_type: str
    numeric_value: int
    source_field: str
    source_span: str
    confidence: float
    normalization_rule: str


@dataclass(frozen=True, order=True)
class RelationConflict:
    """Multiple values observed for the same relation type on one POI."""

    relation_type: str
    candidate_values: tuple[int, ...]
    selected_value: int | None
    highest_conflicting_priority: int


@dataclass(frozen=True)
class ExtractionResult:
    """Stable relations plus conflicts retained for audit and filtering."""

    relations: tuple[NumericRelation, ...]
    conflicts: tuple[RelationConflict, ...]
    evidence_count: int


@dataclass(frozen=True)
class QueryRelationFeatures:
    """Typed relation pairs and boundary-delimited numeric values from one Query."""

    typed_relations: tuple[tuple[str, int], ...]
    numeric_values: tuple[int, ...]


@dataclass(frozen=True)
class _Evidence:
    relation_type: str
    numeric_value: int
    source_field: str
    source_span: str
    confidence: float
    normalization_rule: str
    source_priority: int


def normalize_relation_text(value: Any) -> str:
    """Normalize relation text without converting untriggered name characters."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).upper()
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    return re.sub(r"\s+", "", text)


def parse_contextual_integer(value: str) -> int:
    """Parse Arabic or Chinese digits after an explicit relation trigger matched."""

    normalized = normalize_relation_text(value)
    if not normalized:
        raise NumericRelationError("关系数值不能为空")
    if normalized.isascii() and normalized.isdigit():
        return int(normalized)
    if any(character not in _DIGITS and character not in _UNITS for character in normalized):
        raise NumericRelationError(f"无法解析关系数值：{value!r}")

    if not any(character in _UNITS for character in normalized):
        return int("".join(str(_DIGITS[character]) for character in normalized))

    total = 0
    section = 0
    pending_digit: int | None = None
    for character in normalized:
        if character in _DIGITS:
            pending_digit = _DIGITS[character]
            continue
        unit = _UNITS[character]
        if unit == 10000:
            section += 0 if pending_digit is None else pending_digit
            total += (section or 1) * unit
            section = 0
            pending_digit = None
            continue
        section += (1 if pending_digit is None else pending_digit) * unit
        pending_digit = None
    return total + section + (0 if pending_digit is None else pending_digit)


def _add_number_match(
    evidence: list[_Evidence],
    *,
    relation_type: str,
    match: re.Match[str],
    source_field: str,
    rule: str,
    group: str = "number",
) -> None:
    evidence.append(
        _Evidence(
            relation_type=relation_type,
            numeric_value=parse_contextual_integer(match.group(group)),
            source_field=source_field,
            source_span=match.group(0),
            confidence=SOURCE_CONFIDENCE[source_field],
            normalization_rule=rule,
            source_priority=SOURCE_PRIORITIES[source_field],
        )
    )


def _find_numeric(
    evidence: list[_Evidence],
    *,
    text: str,
    source_field: str,
    relation_type: str,
    pattern: str,
    rule: str,
    flags: int = 0,
) -> None:
    for match in re.finditer(pattern, text, flags):
        _add_number_match(
            evidence,
            relation_type=relation_type,
            match=match,
            source_field=source_field,
            rule=rule,
        )


def _extract_from_text(text: str, source_field: str) -> list[_Evidence]:
    evidence: list[_Evidence] = []
    if not text:
        return evidence

    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_PHASE",
        pattern=rf"第?(?P<number>{_NUMBER})期",
        rule="explicit_phase_number",
    )
    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_ZONE_NUM",
        pattern=rf"(?P<number>{_NUMBER})(?:号)?区",
        rule="explicit_numeric_zone",
    )
    for match in re.finditer(r"(?<![A-Z])(?P<alpha>[A-Z])区", text):
        evidence.append(
            _Evidence(
                relation_type="R_ZONE_ALPHA",
                numeric_value=ord(match.group("alpha")) - ord("A") + 1,
                source_field=source_field,
                source_span=match.group(0),
                confidence=SOURCE_CONFIDENCE[source_field],
                normalization_rule="alpha_zone_a1_z26",
                source_priority=SOURCE_PRIORITIES[source_field],
            )
        )
    for match in re.finditer(rf"(?P<direction>{_DIRECTION})区", text):
        evidence.append(
            _Evidence(
                relation_type="R_ZONE_DIR",
                numeric_value=_DIRECTIONS[match.group("direction")],
                source_field=source_field,
                source_span=match.group(0),
                confidence=SOURCE_CONFIDENCE[source_field],
                normalization_rule="direction_zone_fixed_code",
                source_priority=SOURCE_PRIORITIES[source_field],
            )
        )

    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_BUILDING",
        pattern=rf"第?(?P<number>{_NUMBER})(?:号)?(?:楼|栋|幢|座)",
        rule="explicit_building_number",
    )
    address_pattern = re.compile(
        rf"(?P<qualifier>{_QUALIFIER})?"
        rf"(?P<number>{_NUMBER})"
        rf"(?:-(?P<subnumber>{_NUMBER}))?号"
        rf"(?!楼|栋|幢|座|铺|商铺|铺位|摊位|门|口)"
    )
    for match in address_pattern.finditer(text):
        _add_number_match(
            evidence,
            relation_type="R_ADDRESS_NO",
            match=match,
            source_field=source_field,
            rule="explicit_address_number",
        )
        qualifier = match.group("qualifier")
        if qualifier:
            evidence.append(
                _Evidence(
                    relation_type="R_QUALIFIER",
                    numeric_value=_QUALIFIERS[qualifier],
                    source_field=source_field,
                    source_span=qualifier,
                    confidence=SOURCE_CONFIDENCE[source_field],
                    normalization_rule="address_qualifier_fixed_code",
                    source_priority=SOURCE_PRIORITIES[source_field],
                )
            )
        if match.group("subnumber"):
            _add_number_match(
                evidence,
                relation_type="R_SUBNO",
                match=match,
                source_field=source_field,
                rule="hyphenated_address_subnumber",
                group="subnumber",
            )

    qualifier_pattern = re.compile(
        rf"(?P<qualifier>{_QUALIFIER})(?={_NUMBER}(?:号)?(?:楼|栋|幢|座))"
    )
    for match in qualifier_pattern.finditer(text):
        qualifier = match.group("qualifier")
        evidence.append(
            _Evidence(
                relation_type="R_QUALIFIER",
                numeric_value=_QUALIFIERS[qualifier],
                source_field=source_field,
                source_span=qualifier,
                confidence=SOURCE_CONFIDENCE[source_field],
                normalization_rule="building_qualifier_fixed_code",
                source_priority=SOURCE_PRIORITIES[source_field],
            )
        )

    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_UNIT",
        pattern=rf"(?P<number>{_NUMBER})单元",
        rule="explicit_unit_number",
    )
    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_BASEMENT",
        pattern=(
            rf"(?<![A-Z0-9])B(?P<number>{_SHORT_NUMBER})"
            rf"(?![{_NUMBER_CHARACTER}])(?:层|楼)?"
        ),
        rule="latin_b_basement_number",
    )
    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_BASEMENT",
        pattern=rf"地下第?(?P<number>{_NUMBER})(?:层|楼)?",
        rule="explicit_underground_floor",
    )
    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_FLOOR",
        pattern=rf"(?<![B下])第?(?P<number>{_NUMBER})(?:层|楼层)",
        rule="explicit_above_ground_floor",
    )
    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_FLOOR",
        pattern=(
            rf"(?<![A-Z0-9])F(?P<number>{_SHORT_NUMBER})"
            rf"(?![{_NUMBER_CHARACTER}])(?:层)?"
        ),
        rule="latin_f_floor_number",
    )
    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_FLOOR",
        pattern=(
            rf"(?<![{_NUMBER_CHARACTER}])(?P<number>{_SHORT_NUMBER})F"
            rf"(?![A-Z0-9])(?:层)?"
        ),
        rule="suffix_f_floor_number",
    )
    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_ROOM",
        pattern=rf"(?<![A-Z])(?P<number>{_NUMBER})(?:室|房)",
        rule="explicit_room_number",
    )
    _find_numeric(
        evidence,
        text=text,
        source_field=source_field,
        relation_type="R_SHOP",
        pattern=rf"(?P<number>{_NUMBER})(?:号)?(?:商铺|铺位|摊位|铺)",
        rule="explicit_shop_number",
    )

    if source_field == "displayname":
        for match in re.finditer(rf"^(?P<number>{_NUMBER})(?=[(]|$)", text):
            _add_number_match(
                evidence,
                relation_type="R_ENTITY_NO",
                match=match,
                source_field=source_field,
                rule="leading_numeric_entity_name",
            )

    entrance_pattern = re.compile(
        rf"(?P<direction>{_DIRECTION})(?P<number>{_SHORT_NUMBER})?"
        rf"(?:号)?(?:门|口)"
    )
    for match in entrance_pattern.finditer(text):
        evidence.append(
            _Evidence(
                relation_type="R_ENTRANCE_DIR",
                numeric_value=_DIRECTIONS[match.group("direction")],
                source_field=source_field,
                source_span=match.group(0),
                confidence=SOURCE_CONFIDENCE[source_field],
                normalization_rule="entrance_direction_fixed_code",
                source_priority=SOURCE_PRIORITIES[source_field],
            )
        )
        if match.group("number"):
            _add_number_match(
                evidence,
                relation_type="R_ENTRANCE_NO",
                match=match,
                source_field=source_field,
                rule="directional_entrance_number",
            )

    alpha_entrance_pattern = re.compile(
        rf"(?<![A-Z])(?P<alpha>[A-Z])(?P<number>{_SHORT_NUMBER})?"
        rf"(?:号)?(?:门|口)"
    )
    for match in alpha_entrance_pattern.finditer(text):
        evidence.append(
            _Evidence(
                relation_type="R_ENTRANCE_ALPHA",
                numeric_value=ord(match.group("alpha")) - ord("A") + 1,
                source_field=source_field,
                source_span=match.group(0),
                confidence=SOURCE_CONFIDENCE[source_field],
                normalization_rule="alpha_entrance_a1_z26",
                source_priority=SOURCE_PRIORITIES[source_field],
            )
        )
        if match.group("number"):
            _add_number_match(
                evidence,
                relation_type="R_ENTRANCE_NO",
                match=match,
                source_field=source_field,
                rule="alpha_entrance_number",
            )

    numeric_entrance_pattern = re.compile(
        rf"(?<![A-Z北东南西中])(?P<number>{_SHORT_NUMBER})(?:号)?(?:门|口)"
    )
    for match in numeric_entrance_pattern.finditer(text):
        _add_number_match(
            evidence,
            relation_type="R_ENTRANCE_NO",
            match=match,
            source_field=source_field,
            rule="numeric_entrance_number",
        )
    return evidence


def _iter_sources(poi: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    for field in ("displayname", "address"):
        text = normalize_relation_text(poi.get(field))
        if text:
            yield field, text
    alias = normalize_relation_text(poi.get("alias"))
    if alias:
        yield "alias", alias


def extract_numeric_relations(poi: Mapping[str, Any]) -> ExtractionResult:
    """Extract stable numeric relations using deterministic source precedence.

    The highest source level wins only when it has one distinct value. Lower-level
    disagreements are retained as conflicts but cannot override a unique display
    name or address value. Ambiguity at the highest present level suppresses the
    relation from the stable result.
    """

    evidence = [
        item
        for source_field, text in _iter_sources(poi)
        for item in _extract_from_text(text, source_field)
    ]
    deduplicated = {
        (
            item.relation_type,
            item.numeric_value,
            item.source_field,
            item.source_span,
            item.normalization_rule,
        ): item
        for item in evidence
    }
    evidence = sorted(
        deduplicated.values(),
        key=lambda item: (
            item.relation_type,
            item.numeric_value,
            -item.source_priority,
            item.source_field,
            item.source_span,
            item.normalization_rule,
        ),
    )

    by_type: dict[str, list[_Evidence]] = defaultdict(list)
    for item in evidence:
        by_type[item.relation_type].append(item)

    selected: list[NumericRelation] = []
    conflicts: list[RelationConflict] = []
    for relation_type in RELATION_TYPES:
        items = by_type.get(relation_type, [])
        if not items:
            continue
        all_values = tuple(sorted({item.numeric_value for item in items}))
        highest_priority = max(item.source_priority for item in items)
        highest_items = [
            item for item in items if item.source_priority == highest_priority
        ]
        highest_values = tuple(
            sorted({item.numeric_value for item in highest_items})
        )
        selected_value = highest_values[0] if len(highest_values) == 1 else None
        if len(all_values) > 1:
            conflicts.append(
                RelationConflict(
                    relation_type=relation_type,
                    candidate_values=all_values,
                    selected_value=selected_value,
                    highest_conflicting_priority=highest_priority,
                )
            )
        if selected_value is None:
            continue
        selected_item = min(
            (
                item
                for item in highest_items
                if item.numeric_value == selected_value
            ),
            key=lambda item: (
                -item.confidence,
                item.source_field,
                item.source_span,
                item.normalization_rule,
            ),
        )
        selected.append(
            NumericRelation(
                relation_type=relation_type,
                numeric_value=selected_value,
                source_field=selected_item.source_field,
                source_span=selected_item.source_span,
                confidence=selected_item.confidence,
                normalization_rule=selected_item.normalization_rule,
            )
        )
    return ExtractionResult(
        relations=tuple(selected),
        conflicts=tuple(conflicts),
        evidence_count=len(evidence),
    )


def extract_query_relation_features(query: Any) -> QueryRelationFeatures:
    """Extract conservative relation signals available in a raw search Query."""

    normalized = normalize_relation_text(query)
    result = extract_numeric_relations({"displayname": normalized})
    typed_relations = tuple(
        sorted(
            (item.relation_type, item.numeric_value)
            for item in result.relations
        )
    )
    numeric_values = {
        int(match.group(0))
        for match in re.finditer(r"(?<![0-9])[0-9]{1,7}(?![0-9])", normalized)
    }
    numeric_values.update(
        value
        for relation_type, value in typed_relations
        if relation_type in DIRECT_NUMERIC_RELATION_TYPES
    )
    return QueryRelationFeatures(
        typed_relations=typed_relations,
        numeric_values=tuple(sorted(numeric_values)),
    )

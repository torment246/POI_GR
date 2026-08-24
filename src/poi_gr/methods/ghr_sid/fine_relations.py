"""Fine-grained static entity relations used after G6 and coarse R3."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from poi_gr.methods.qgr_sid.relations import (
    normalize_relation_text,
    parse_contextual_integer,
)


FINE_RELATION_SCHEMA_VERSION = "ghr-sid-fine-entity-relations-v3"
BASE_FINE_RELATION_TYPES = (
    "R_FINE_INNER_NO",
    "R_FINE_PARKING_SLOT",
    "R_FINE_ROW",
    "R_FINE_STORE_NO",
    "R_FINE_BOOTH_NO",
    "R_FINE_CHILD_NO",
    "R_FINE_ALPHA_PREFIX",
    "R_FINE_ALPHA_ENTITY_NO",
    "R_FINE_ADDRESS_MAIN_NO",
    "R_FINE_BUILDING_NO",
    "R_FINE_COMPOUND_HEAD_NO",
    "R_FINE_COMPOUND_TAIL_NO",
    "R_FINE_BUILDING_CODE",
    "R_FINE_ENTITY_CODE",
    "R_FINE_CATEGORY_MAJOR",
    "R_FINE_CATEGORY_MID",
    "R_FINE_CATEGORY_LEAF",
    "R_FINE_LAYER",
    "R_FINE_ROAD_ENTITY",
    "R_FINE_PARENT_ENTITY",
    "R_FINE_CHILD_ENTITY",
)
ENTITY_CHARACTER_DEPTH = 12
PARENT_CHARACTER_RELATION_TYPES = tuple(
    f"R_FINE_PARENT_PREFIX_CHAR_{position:02d}"
    for position in range(1, ENTITY_CHARACTER_DEPTH + 1)
) + tuple(
    f"R_FINE_PARENT_SUFFIX_CHAR_{position:02d}"
    for position in range(1, ENTITY_CHARACTER_DEPTH + 1)
)
CHILD_CHARACTER_RELATION_TYPES = tuple(
    f"R_FINE_CHILD_PREFIX_CHAR_{position:02d}"
    for position in range(1, ENTITY_CHARACTER_DEPTH + 1)
) + tuple(
    f"R_FINE_CHILD_SUFFIX_CHAR_{position:02d}"
    for position in range(1, ENTITY_CHARACTER_DEPTH + 1)
)
CHARACTER_RELATION_TYPES = (
    PARENT_CHARACTER_RELATION_TYPES + CHILD_CHARACTER_RELATION_TYPES
)
FINE_RELATION_TYPES = BASE_FINE_RELATION_TYPES + CHARACTER_RELATION_TYPES
NUMERIC_RELATION_COUNT = 14
CATEGORY_RELATION_COUNT = 4
ROAD_RELATION_COUNT = 1
BASE_FINE_RELATION_COUNT = len(BASE_FINE_RELATION_TYPES)
MAX_FINE_VALUE = np.iinfo(np.int16).max
CHARACTER_VOCABULARY_NAME = "R_FINE_ENTITY_CHARACTER_VALUE"
VOCAB_RELATION_NAMES = (
    "R_FINE_CATEGORY_LEAF",
    "R_FINE_ROAD_ENTITY",
    "R_FINE_PARENT_ENTITY",
    "R_FINE_CHILD_ENTITY",
)

_NUMBER = r"(?:[0-9]{1,7}|[零〇一二两三四五六七八九十百千万]{1,10})"
_CODE_RE = re.compile(r"^[A-Z]{0,2}[-_]?0*[0-9]{1,6}$")
_ALNUM_ENTITY_CODE_RE = re.compile(
    r"(?<![A-Z0-9])"
    r"(?P<prefix>[A-Z]{0,2})[-_]?"
    r"(?P<number>[0-9]{1,6})"
    r"(?P<suffix>[A-Z]{0,2})"
    r"(?![A-Z0-9])"
)
_BASE36_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ALNUM_CODE_LENGTH_BITS = 4
_ALNUM_CODE_LENGTH_MASK = (1 << _ALNUM_CODE_LENGTH_BITS) - 1
_NUMERIC_COMPOUND_CODE_RE = re.compile(
    r"^[0-9]{1,5}(?:-[0-9]{1,5}){1,2}$"
)
_BRACKET_RE = re.compile(r"^(?P<outside>.+?)[(（](?P<inside>[^()（）]{1,100})[)）]$")
_DIRECTION_SUFFIX_RE = re.compile(
    r"^(?P<parent>.+?)(?P<child>东北门|东南门|西南门|西北门|东门|南门|西门|北门|正门|侧门)$"
)
_ROAD_RE = re.compile(
    r"([A-Z0-9\u3400-\u9fff·]{1,32}(?:高速公路|公路|大道|大街|胡同|路|街|道|巷))"
)


class FineRelationError(ValueError):
    """Raised when fine entity relation inputs violate the contract."""


@dataclass(frozen=True)
class RawFineRelations:
    """Raw numeric and lexical relation candidates for one POI."""

    numeric_values: tuple[int, ...]
    category_major: int
    category_mid: int
    category_leaf: str
    layer: int
    road_entity: str
    parent_entity: str
    child_entity: str


@dataclass(frozen=True)
class FineRelationMatrix:
    """Dense numeric relation matrix plus reversible global vocabularies."""

    values: np.ndarray
    vocabularies: dict[str, tuple[str, ...]]
    metrics: dict[str, Any]


def normalize_entity_label(value: Any) -> str:
    """Canonicalize a relation label while retaining its lexical semantics."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).upper()
    text = text.replace("—", "-").replace("–", "-").replace("－", "-")
    text = re.sub(r"\s+", "", text)
    text = text.strip("-_:：,，.。;；/|｜[]【】")
    return text[:160]


def _safe_number(value: str) -> int:
    try:
        number = parse_contextual_integer(value)
    except ValueError:
        return -1
    return number if 0 <= number <= MAX_FINE_VALUE else -1


def _last_trigger_number(text: str, suffix_pattern: str) -> int:
    matches = list(
        re.finditer(
            rf"(?P<number>{_NUMBER})(?:号)?(?:{suffix_pattern})",
            text,
        )
    )
    return -1 if not matches else _safe_number(matches[-1].group("number"))


def _inner_number(text: str) -> int:
    matches = list(re.finditer(rf"(?P<number>{_NUMBER})号", text))
    if len(matches) < 2:
        return -1
    return _safe_number(matches[-1].group("number"))


def _alpha_code(text: str) -> tuple[int, int]:
    matches = list(
        re.finditer(
            r"(?<![A-Z0-9])(?P<alpha>[A-Z]{1,2})[-_]?0*(?P<number>[0-9]{1,5})(?![A-Z0-9])",
            text,
        )
    )
    if not matches:
        return -1, -1
    match = matches[-1]
    prefix = 0
    for character in match.group("alpha"):
        prefix = prefix * 27 + ord(character) - ord("A") + 1
    number = int(match.group("number"))
    if prefix > MAX_FINE_VALUE or number > MAX_FINE_VALUE:
        return -1, -1
    return prefix, number


def _encode_alnum_entity_code(code: str) -> int:
    """Pack a canonical alphanumeric entity code into one reversible integer."""

    if not 1 <= len(code) <= _ALNUM_CODE_LENGTH_MASK:
        return -1
    encoded = (int(code, 36) << _ALNUM_CODE_LENGTH_BITS) | len(code)
    return encoded if encoded <= np.iinfo(np.int64).max else -1


def _decode_alnum_entity_code(value: int) -> str:
    length = value & _ALNUM_CODE_LENGTH_MASK
    remaining = value >> _ALNUM_CODE_LENGTH_BITS
    if length <= 0 or remaining <= 0:
        return str(value)
    characters: list[str] = []
    while remaining:
        remaining, digit = divmod(remaining, 36)
        characters.append(_BASE36_ALPHABET[digit])
    return "".join(reversed(characters)).zfill(length)


def _alnum_entity_codes(text: str) -> tuple[int, int]:
    """Return role-aware building and generic alphanumeric entity codes."""

    matches = [
        match
        for match in _ALNUM_ENTITY_CODE_RE.finditer(text)
        if match.group("prefix") or match.group("suffix")
    ]
    if not matches:
        return -1, -1
    building_code = -1
    entity_code = -1
    for match in matches:
        code = "".join(
            (
                match.group("prefix"),
                match.group("number"),
                match.group("suffix"),
            )
        )
        encoded = _encode_alnum_entity_code(code)
        tail = text[match.end() :]
        if re.match(r"(?:号)?(?:楼|栋|幢|座)", tail):
            building_code = encoded
        else:
            entity_code = encoded
    return building_code, entity_code


def _remove_alnum_entity_codes(text: str) -> str:
    """Remove complete alphanumeric codes before generic number extraction."""

    def replace(match: re.Match[str]) -> str:
        if match.group("prefix") or match.group("suffix"):
            return ""
        return match.group(0)

    return _ALNUM_ENTITY_CODE_RE.sub(replace, text)


def _prefer_display_value(display_value: int, address_value: int) -> int:
    return display_value if display_value >= 0 else address_value


def _address_main_number(text: str) -> int:
    matches = list(
        re.finditer(
            rf"(?P<main>{_NUMBER})(?:-(?P<sub>{_NUMBER}))?号"
            r"(?!楼|栋|幢|座|铺|商铺|铺位|摊位|门|口)",
            text,
        )
    )
    return -1 if not matches else _safe_number(matches[-1].group("main"))


def _building_number(text: str) -> int:
    matches = list(
        re.finditer(
            rf"第?(?P<number>{_NUMBER})(?:号)?(?:楼|栋|幢|座)",
            text,
        )
    )
    return -1 if not matches else _safe_number(matches[-1].group("number"))


def _compound_numbers(text: str) -> tuple[int, int]:
    """Return the head/tail of the last two/three-level numeric entity code."""

    matches = list(
        re.finditer(
            r"(?<![0-9])(?P<head>[0-9]{1,5})-"
            r"(?P<middle>[0-9]{1,5})"
            r"(?:-(?P<tail>[0-9]{1,5}))?(?![0-9])",
            text,
        )
    )
    if not matches:
        return -1, -1
    match = matches[-1]
    tail = match.group("tail") or match.group("middle")
    return _safe_number(match.group("head")), _safe_number(tail)


def split_parent_child(displayname: Any) -> tuple[str, str]:
    """Infer a deterministic parent/child entity relation from a POI name."""

    name = normalize_entity_label(displayname)
    if not name:
        return "", ""
    bracket = _BRACKET_RE.fullmatch(name)
    if bracket:
        outside = normalize_entity_label(bracket.group("outside"))
        inside = normalize_entity_label(bracket.group("inside"))
        if _CODE_RE.fullmatch(outside) or _NUMERIC_COMPOUND_CODE_RE.fullmatch(
            outside
        ):
            return inside, outside
        return outside, inside

    if "-" in name and not _CODE_RE.fullmatch(name):
        parent, child = name.rsplit("-", 1)
        parent = normalize_entity_label(parent)
        child = normalize_entity_label(child)
        if parent and child:
            return parent, child

    direction = _DIRECTION_SUFFIX_RE.fullmatch(name)
    if direction:
        return (
            normalize_entity_label(direction.group("parent")),
            normalize_entity_label(direction.group("child")),
        )

    address_numbers = list(re.finditer(rf"{_NUMBER}号", name))
    if len(address_numbers) >= 2:
        split = address_numbers[0].end()
        parent = normalize_entity_label(name[:split])
        child = normalize_entity_label(name[split:])
        if parent and child:
            return parent, child
    return name, ""


def extract_road_entity(address: Any) -> str:
    """Extract the last explicit road-like entity from a static address."""

    text = normalize_entity_label(address)
    matches = list(_ROAD_RE.finditer(text))
    if not matches:
        return ""
    road = matches[-1].group(1)
    for marker in ("北京市", "市", "区", "县", "街道", "镇", "乡"):
        if marker in road:
            road = road.rsplit(marker, 1)[-1]
    return normalize_entity_label(road)


def _category_parts(category_code: Any) -> tuple[int, int, str]:
    value = normalize_entity_label(category_code)
    if not value.isdigit() or len(value) < 2:
        return -1, -1, ""
    major = int(value[:2])
    mid = int(value[:4]) if len(value) >= 4 else -1
    return major, mid, value


def extract_raw_fine_relations(poi: Mapping[str, Any]) -> RawFineRelations:
    """Extract typed numeric and entity relations without Query or POI ID."""

    displayname = normalize_relation_text(poi.get("displayname"))
    address = normalize_relation_text(poi.get("address"))
    parent, child = split_parent_child(poi.get("displayname"))
    building_code, entity_code = _alnum_entity_codes(child)
    display_building_code, display_entity_code = _alnum_entity_codes(displayname)
    if building_code < 0:
        building_code = display_building_code
    if entity_code < 0:
        entity_code = display_entity_code
    if building_code >= 0 or entity_code >= 0:
        alpha_prefix, alpha_number = -1, -1
    else:
        alpha_prefix, alpha_number = _alpha_code(displayname)
    child_without_alnum_code = _remove_alnum_entity_codes(child)
    child_number_matches = re.findall(
        rf"(?P<number>{_NUMBER})", child_without_alnum_code
    )
    child_number = (
        -1 if not child_number_matches else _safe_number(child_number_matches[-1])
    )
    category_major, category_mid, category_leaf = _category_parts(
        poi.get("category_code")
    )
    display_compound_head, display_compound_tail = _compound_numbers(displayname)
    address_compound_head, address_compound_tail = _compound_numbers(address)
    try:
        layer = int(str(poi.get("layer", "")))
    except (TypeError, ValueError, OverflowError):
        layer = -1
    if not 0 <= layer <= MAX_FINE_VALUE:
        layer = -1
    numeric_values = (
        _inner_number(displayname),
        _last_trigger_number(displayname, "车位"),
        _last_trigger_number(displayname, "排"),
        _last_trigger_number(displayname, "店|门店"),
        _last_trigger_number(displayname, "柜台|柜|档口|摊位"),
        child_number,
        alpha_prefix,
        alpha_number,
        _prefer_display_value(
            _address_main_number(displayname), _address_main_number(address)
        ),
        _prefer_display_value(
            _building_number(displayname), _building_number(address)
        ),
        _prefer_display_value(display_compound_head, address_compound_head),
        _prefer_display_value(display_compound_tail, address_compound_tail),
        building_code,
        entity_code,
    )
    return RawFineRelations(
        numeric_values=numeric_values,
        category_major=category_major,
        category_mid=category_mid,
        category_leaf=category_leaf,
        layer=layer,
        road_entity=extract_road_entity(poi.get("address")),
        parent_entity=parent,
        child_entity=child,
    )


def build_shared_vocabulary(
    values: Sequence[str],
    *,
    min_support: int,
    max_values: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Build a deterministic support-bounded vocabulary with no singleton IDs."""

    if min_support < 2:
        raise FineRelationError("实体关系词表 min_support 必须至少为 2")
    if not 1 <= max_values <= MAX_FINE_VALUE:
        raise FineRelationError("实体关系词表 max_values 超出 int16 范围")
    counts = Counter(value for value in values if value)
    eligible = [
        (value, count) for value, count in counts.items() if count >= min_support
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    retained = eligible[:max_values]
    vocabulary = tuple(value for value, _ in retained)
    retained_set = set(vocabulary)
    metrics = {
        "observed_value_count": len(counts),
        "singleton_value_count": sum(count == 1 for count in counts.values()),
        "eligible_value_count": len(eligible),
        "retained_value_count": len(vocabulary),
        "truncated_value_count": max(0, len(eligible) - len(vocabulary)),
        "covered_poi_count": sum(counts[value] for value in retained_set),
        "max_support": max(counts.values(), default=0),
        "min_support": min_support,
        "max_values": max_values,
    }
    return vocabulary, metrics


def build_fine_relation_matrix(
    raw_relations: Sequence[RawFineRelations],
    *,
    entity_min_support: int = 2,
    entity_vocab_size: int = 4096,
) -> FineRelationMatrix:
    """Convert raw relations into an int16 matrix with reversible vocab IDs."""

    rows = len(raw_relations)
    values = np.full((rows, len(FINE_RELATION_TYPES)), -1, dtype=np.int64)
    if not rows:
        raise FineRelationError("细粒度关系输入不能为空")
    for row, relation in enumerate(raw_relations):
        values[row, :NUMERIC_RELATION_COUNT] = relation.numeric_values
        values[row, NUMERIC_RELATION_COUNT] = relation.category_major
        values[row, NUMERIC_RELATION_COUNT + 1] = relation.category_mid
        values[row, NUMERIC_RELATION_COUNT + 3] = relation.layer

    category_values = sorted(
        {relation.category_leaf for relation in raw_relations if relation.category_leaf}
    )
    if len(category_values) > MAX_FINE_VALUE:
        raise FineRelationError("category leaf 词表超出 int16 范围")
    category_vocab = tuple(category_values)
    category_to_id = {
        value: index + 1 for index, value in enumerate(category_vocab)
    }
    lexical_inputs = {
        "R_FINE_ROAD_ENTITY": [relation.road_entity for relation in raw_relations],
        "R_FINE_PARENT_ENTITY": [
            relation.parent_entity for relation in raw_relations
        ],
        "R_FINE_CHILD_ENTITY": [relation.child_entity for relation in raw_relations],
    }
    vocabularies: dict[str, tuple[str, ...]] = {
        "R_FINE_CATEGORY_LEAF": category_vocab
    }
    vocab_metrics: dict[str, Any] = {
        "R_FINE_CATEGORY_LEAF": {
            "observed_value_count": len(category_vocab),
            "retained_value_count": len(category_vocab),
            "covered_poi_count": sum(bool(value) for value in (
                relation.category_leaf for relation in raw_relations
            )),
            "min_support": 1,
            "max_values": MAX_FINE_VALUE,
        }
    }
    lexical_columns = {
        "R_FINE_ROAD_ENTITY": NUMERIC_RELATION_COUNT + CATEGORY_RELATION_COUNT,
        "R_FINE_PARENT_ENTITY": (
            NUMERIC_RELATION_COUNT + CATEGORY_RELATION_COUNT + 1
        ),
        "R_FINE_CHILD_ENTITY": (
            NUMERIC_RELATION_COUNT + CATEGORY_RELATION_COUNT + 2
        ),
    }
    for name, raw_values in lexical_inputs.items():
        vocabulary, metrics = build_shared_vocabulary(
            raw_values,
            min_support=entity_min_support,
            max_values=entity_vocab_size,
        )
        vocabularies[name] = vocabulary
        vocab_metrics[name] = metrics
        value_to_id = {value: index + 1 for index, value in enumerate(vocabulary)}
        column = lexical_columns[name]
        values[:, column] = np.asarray(
            [value_to_id.get(value, -1) for value in raw_values], dtype=np.int64
        )
    values[:, NUMERIC_RELATION_COUNT + 2] = np.asarray(
        [category_to_id.get(relation.category_leaf, -1) for relation in raw_relations],
        dtype=np.int64,
    )
    entity_characters = sorted(
        {
            character
            for relation in raw_relations
            for label in (relation.parent_entity, relation.child_entity)
            for character in label
        }
    )
    if len(entity_characters) > MAX_FINE_VALUE:
        raise FineRelationError("实体字符词表超出 int16 范围")
    character_vocabulary = tuple(entity_characters)
    character_to_id = {
        character: index + 1
        for index, character in enumerate(character_vocabulary)
    }
    vocabularies[CHARACTER_VOCABULARY_NAME] = character_vocabulary
    vocab_metrics[CHARACTER_VOCABULARY_NAME] = {
        "observed_value_count": len(character_vocabulary),
        "retained_value_count": len(character_vocabulary),
        "covered_parent_poi_count": sum(
            bool(relation.parent_entity) for relation in raw_relations
        ),
        "covered_child_poi_count": sum(
            bool(relation.child_entity) for relation in raw_relations
        ),
        "min_support": 1,
        "max_values": MAX_FINE_VALUE,
        "assignment": "unicode_lexical_order",
    }

    def assign_character_positions(
        *, labels: Sequence[str], start_column: int
    ) -> None:
        for row, label in enumerate(labels):
            prefix = label[:ENTITY_CHARACTER_DEPTH]
            suffix = label[-ENTITY_CHARACTER_DEPTH:][::-1]
            for position, character in enumerate(prefix):
                values[row, start_column + position] = character_to_id[character]
            suffix_start = start_column + ENTITY_CHARACTER_DEPTH
            for position, character in enumerate(suffix):
                values[row, suffix_start + position] = character_to_id[character]

    parent_start = BASE_FINE_RELATION_COUNT
    child_start = parent_start + len(PARENT_CHARACTER_RELATION_TYPES)
    assign_character_positions(
        labels=[relation.parent_entity for relation in raw_relations],
        start_column=parent_start,
    )
    assign_character_positions(
        labels=[relation.child_entity for relation in raw_relations],
        start_column=child_start,
    )
    coverage = {
        relation_type: {
            "poi_count": int(np.count_nonzero(values[:, index] >= 0)),
            "poi_ratio": float(np.mean(values[:, index] >= 0)),
            "distinct_value_count": int(
                len(np.unique(values[values[:, index] >= 0, index]))
            ),
        }
        for index, relation_type in enumerate(FINE_RELATION_TYPES)
    }
    return FineRelationMatrix(
        values=values,
        vocabularies=vocabularies,
        metrics={
            "schema_version": FINE_RELATION_SCHEMA_VERSION,
            "entity_min_support": entity_min_support,
            "entity_vocab_size": entity_vocab_size,
            "coverage": coverage,
            "vocabularies": vocab_metrics,
        },
    )


def decode_fine_relation_value(
    relation_type: str,
    value: int,
    vocabularies: Mapping[str, Sequence[str]],
) -> str | int:
    """Decode vocabulary-backed values for audit cases."""

    if relation_type in {"R_FINE_BUILDING_CODE", "R_FINE_ENTITY_CODE"}:
        return _decode_alnum_entity_code(value)
    vocabulary_name = relation_type
    if relation_type in CHARACTER_RELATION_TYPES:
        vocabulary_name = CHARACTER_VOCABULARY_NAME
    elif relation_type not in VOCAB_RELATION_NAMES:
        return value
    vocabulary = vocabularies.get(vocabulary_name, ())
    if not 1 <= value <= len(vocabulary):
        return value
    return vocabulary[value - 1]

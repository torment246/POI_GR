"""Unified entity-role candidates used directly after the G6 branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from poi_gr.methods.ghr_sid.fine_relations import (
    CHARACTER_VOCABULARY_NAME,
    CHILD_CHARACTER_RELATION_TYPES,
    FINE_RELATION_TYPES,
    PARENT_CHARACTER_RELATION_TYPES,
    FineRelationMatrix,
    decode_fine_relation_value,
)
from poi_gr.methods.qgr_sid.relations import RELATION_TYPES


UNIFIED_RELATION_SCHEMA_VERSION = "ghr-sid-g6-entity-relations-v1"

CORE_RELATION_TYPES = (
    "R_PHASE",
    "R_ZONE_NUM",
    "R_ZONE_ALPHA",
    "R_ZONE_DIR",
    "R_BUILDING",
    "R_ADDRESS",
    "R_QUALIFIER",
    "R_SUBNO",
    "R_UNIT",
    "R_FLOOR",
    "R_BASEMENT",
    "R_ROOM",
    "R_SHOP",
    "R_ENTITY",
    "R_ENTRANCE_DIR",
    "R_ENTRANCE_ALPHA",
    "R_ENTRANCE_NO",
    "R_INNER",
    "R_PARKING",
    "R_ROW",
    "R_STORE",
    "R_BOOTH",
    "R_CHILD_NO",
    "R_COMPOUND_HEAD",
    "R_COMPOUND_TAIL",
    "R_ROAD",
    "R_PARENT",
    "R_CHILD",
    "R_CATEGORY_MAJOR",
    "R_CATEGORY_MID",
    "R_CATEGORY_LEAF",
    "R_LAYER",
)


def _compact_character_type(relation_type: str) -> str:
    role = "PARENT" if "PARENT" in relation_type else "CHILD"
    direction = "P" if "PREFIX" in relation_type else "S"
    position = relation_type.rsplit("_", 1)[-1]
    return f"R_{role}_{direction}{position}"


POSITION_RELATION_TYPES = tuple(
    _compact_character_type(relation_type)
    for relation_type in (
        PARENT_CHARACTER_RELATION_TYPES + CHILD_CHARACTER_RELATION_TYPES
    )
)
UNIFIED_RELATION_TYPES = CORE_RELATION_TYPES + POSITION_RELATION_TYPES
UNIFIED_RELATION_INDEX = {
    relation_type: index
    for index, relation_type in enumerate(UNIFIED_RELATION_TYPES)
}

SEMANTIC_TIERS = np.asarray(
    [
        0
        if relation_type
        not in {
            "R_CATEGORY_MAJOR",
            "R_CATEGORY_MID",
            "R_CATEGORY_LEAF",
            "R_LAYER",
        }
        else 1
        for relation_type in CORE_RELATION_TYPES
    ]
    + [2] * len(POSITION_RELATION_TYPES),
    dtype=np.int8,
)

CHARACTER_VOCABULARY = "R_CHARACTER_VALUE"
VOCABULARY_RELATION_TYPES = frozenset(
    {"R_CATEGORY_LEAF", "R_ROAD", "R_PARENT", "R_CHILD"}
)


class UnifiedRelationError(ValueError):
    """Raised when unified relation inputs violate the schema."""


@dataclass(frozen=True)
class UnifiedRelationMatrix:
    """Dense post-G6 relation values and reversible shared vocabularies."""

    values: np.ndarray
    vocabularies: dict[str, tuple[str, ...]]
    metrics: dict[str, Any]


def _first_available(*values: np.ndarray) -> np.ndarray:
    if not values:
        raise UnifiedRelationError("至少需要一个候选数组")
    output = np.full(len(values[0]), -1, dtype=np.int64)
    for candidate in values:
        candidate = np.asarray(candidate, dtype=np.int64)
        if candidate.shape != output.shape:
            raise UnifiedRelationError("待合并关系 shape 不一致")
        take = (output < 0) & (candidate >= 0)
        output[take] = candidate[take]
    return output


def _copy_column(
    output: np.ndarray,
    relation_type: str,
    source: np.ndarray,
) -> None:
    output[:, UNIFIED_RELATION_INDEX[relation_type]] = np.asarray(
        source, dtype=np.int64
    )


def build_unified_relation_matrix(
    *,
    strict_relations: np.ndarray,
    fine_relations: FineRelationMatrix,
) -> UnifiedRelationMatrix:
    """Merge the old R3 features and fine features into one role ontology."""

    strict = np.asarray(strict_relations)
    fine = np.asarray(fine_relations.values)
    if strict.ndim != 2 or strict.shape[1] != len(RELATION_TYPES):
        raise UnifiedRelationError("strict_relations shape 非法")
    if fine.shape != (strict.shape[0], len(FINE_RELATION_TYPES)):
        raise UnifiedRelationError("fine_relations shape 非法")
    rows = strict.shape[0]
    values = np.full(
        (rows, len(UNIFIED_RELATION_TYPES)), -1, dtype=np.int64
    )
    strict_index = {name: index for index, name in enumerate(RELATION_TYPES)}
    fine_index = {
        name: index for index, name in enumerate(FINE_RELATION_TYPES)
    }

    direct_strict = {
        "R_PHASE": "R_PHASE",
        "R_ZONE_NUM": "R_ZONE_NUM",
        "R_ZONE_ALPHA": "R_ZONE_ALPHA",
        "R_ZONE_DIR": "R_ZONE_DIR",
        "R_QUALIFIER": "R_QUALIFIER",
        "R_SUBNO": "R_SUBNO",
        "R_UNIT": "R_UNIT",
        "R_FLOOR": "R_FLOOR",
        "R_BASEMENT": "R_BASEMENT",
        "R_ROOM": "R_ROOM",
        "R_SHOP": "R_SHOP",
        "R_ENTRANCE_DIR": "R_ENTRANCE_DIR",
        "R_ENTRANCE_ALPHA": "R_ENTRANCE_ALPHA",
        "R_ENTRANCE_NO": "R_ENTRANCE_NO",
    }
    for target, source in direct_strict.items():
        _copy_column(values, target, strict[:, strict_index[source]])

    numeric_building = _first_available(
        strict[:, strict_index["R_BUILDING"]],
        fine[:, fine_index["R_FINE_BUILDING_NO"]],
    )
    building = np.where(numeric_building >= 0, numeric_building * 2, -1)
    building_code = fine[:, fine_index["R_FINE_BUILDING_CODE"]]
    building = np.where(building_code >= 0, building_code * 2 + 1, building)
    _copy_column(values, "R_BUILDING", building)

    address = _first_available(
        strict[:, strict_index["R_ADDRESS_NO"]],
        fine[:, fine_index["R_FINE_ADDRESS_MAIN_NO"]],
    )
    _copy_column(values, "R_ADDRESS", address)

    numeric_entity = strict[:, strict_index["R_ENTITY_NO"]].astype(
        np.int64, copy=False
    )
    entity = np.where(numeric_entity >= 0, numeric_entity * 2, -1)
    entity_code = fine[:, fine_index["R_FINE_ENTITY_CODE"]]
    entity = np.where(entity_code >= 0, entity_code * 2 + 1, entity)
    _copy_column(values, "R_ENTITY", entity)

    fine_mapping = {
        "R_INNER": "R_FINE_INNER_NO",
        "R_PARKING": "R_FINE_PARKING_SLOT",
        "R_ROW": "R_FINE_ROW",
        "R_STORE": "R_FINE_STORE_NO",
        "R_BOOTH": "R_FINE_BOOTH_NO",
        "R_CHILD_NO": "R_FINE_CHILD_NO",
        "R_COMPOUND_HEAD": "R_FINE_COMPOUND_HEAD_NO",
        "R_COMPOUND_TAIL": "R_FINE_COMPOUND_TAIL_NO",
        "R_CATEGORY_MAJOR": "R_FINE_CATEGORY_MAJOR",
        "R_CATEGORY_MID": "R_FINE_CATEGORY_MID",
        "R_CATEGORY_LEAF": "R_FINE_CATEGORY_LEAF",
        "R_LAYER": "R_FINE_LAYER",
        "R_ROAD": "R_FINE_ROAD_ENTITY",
        "R_PARENT": "R_FINE_PARENT_ENTITY",
        "R_CHILD": "R_FINE_CHILD_ENTITY",
    }
    for target, source in fine_mapping.items():
        _copy_column(values, target, fine[:, fine_index[source]])

    for source in (
        PARENT_CHARACTER_RELATION_TYPES + CHILD_CHARACTER_RELATION_TYPES
    ):
        target = _compact_character_type(source)
        _copy_column(values, target, fine[:, fine_index[source]])

    vocabularies = {
        "R_CATEGORY_LEAF": fine_relations.vocabularies.get(
            "R_FINE_CATEGORY_LEAF", ()
        ),
        "R_ROAD": fine_relations.vocabularies.get("R_FINE_ROAD_ENTITY", ()),
        "R_PARENT": fine_relations.vocabularies.get(
            "R_FINE_PARENT_ENTITY", ()
        ),
        "R_CHILD": fine_relations.vocabularies.get(
            "R_FINE_CHILD_ENTITY", ()
        ),
        CHARACTER_VOCABULARY: fine_relations.vocabularies.get(
            CHARACTER_VOCABULARY_NAME, ()
        ),
    }
    coverage = {
        relation_type: {
            "poi_count": int(np.count_nonzero(values[:, index] >= 0)),
            "poi_ratio": float(np.mean(values[:, index] >= 0)),
            "distinct_value_count": int(
                len(np.unique(values[values[:, index] >= 0, index]))
            ),
            "semantic_tier": int(SEMANTIC_TIERS[index]),
        }
        for index, relation_type in enumerate(UNIFIED_RELATION_TYPES)
    }
    return UnifiedRelationMatrix(
        values=values,
        vocabularies=vocabularies,
        metrics={
            "schema_version": UNIFIED_RELATION_SCHEMA_VERSION,
            "relation_count": len(UNIFIED_RELATION_TYPES),
            "semantic_tiers": {
                relation_type: int(SEMANTIC_TIERS[index])
                for index, relation_type in enumerate(UNIFIED_RELATION_TYPES)
            },
            "coverage": coverage,
            "fine_relation_features": fine_relations.metrics,
        },
    )


def _decode_namespaced_entity(relation_type: str, value: int) -> str | int:
    if value % 2 == 0:
        return value // 2
    fine_type = (
        "R_FINE_BUILDING_CODE"
        if relation_type == "R_BUILDING"
        else "R_FINE_ENTITY_CODE"
    )
    return decode_fine_relation_value(fine_type, (value - 1) // 2, {})


def decode_unified_relation_value(
    relation_type: str,
    value: int,
    vocabularies: Mapping[str, Sequence[str]],
) -> str | int:
    """Decode one unified numeric value for cases and manual audits."""

    if relation_type in {"R_BUILDING", "R_ENTITY"}:
        return _decode_namespaced_entity(relation_type, value)
    vocabulary_name = relation_type
    if relation_type in POSITION_RELATION_TYPES:
        vocabulary_name = CHARACTER_VOCABULARY
    elif relation_type not in VOCABULARY_RELATION_TYPES:
        return value
    vocabulary = vocabularies.get(vocabulary_name, ())
    if not 1 <= value <= len(vocabulary):
        return value
    return vocabulary[value - 1]

"""Compile shared strong-relation trees inside frozen post-G6 buckets."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np

from poi_gr.methods.ghr_sid.fine_relations import RawFineRelations
from poi_gr.methods.ghr_sid.minimum_description import (
    CollisionGrouping,
    relation_token_cost,
)
from poi_gr.methods.ghr_sid.unified_relations import (
    UNIFIED_RELATION_INDEX,
    UnifiedRelationMatrix,
    decode_unified_relation_value,
)


RELATION_TREE_SCHEMA_VERSION = "ghr-sid-strong-relation-tree-v1"

# This is a hierarchy, not a feature-ranking list.  Weak fallbacks used by EX9
# (category/layer, full lexical entities, and character positions) are omitted.
TREE_RELATION_TYPES = (
    "R_PHASE",
    "R_ADDRESS",
    "R_SUBNO",
    "R_ZONE_DIR",
    "R_ZONE_ALPHA",
    "R_ZONE_NUM",
    "R_BUILDING",
    "R_UNIT",
    "R_FLOOR",
    "R_BASEMENT",
    "R_ROOM",
    "R_SHOP",
    "R_STORE",
    "R_BOOTH",
    "R_PARKING",
    "R_ROW",
    "R_ENTRANCE_DIR",
    "R_ENTRANCE_ALPHA",
    "R_ENTRANCE_NO",
    "R_QUALIFIER",
    "R_ENTITY",
)
TREE_RELATION_INDEX = {
    relation_type: index
    for index, relation_type in enumerate(TREE_RELATION_TYPES)
}
DEDUP_RELATION_TYPE_ID = len(TREE_RELATION_TYPES)
DEDUP_RELATION_TYPE = "D"

# A varying descendant activates its available ancestors, even when an ancestor
# is constant in the bucket.  This retains paths such as 4区 -> 20号楼 instead
# of reducing them back to the shortest leaf-only description.
_SITE_ANCESTORS = (
    "R_PHASE",
    "R_ADDRESS",
    "R_SUBNO",
    "R_ZONE_DIR",
    "R_ZONE_ALPHA",
    "R_ZONE_NUM",
)
_BUILDING_ANCESTORS = _SITE_ANCESTORS + ("R_BUILDING",)
_UNIT_ANCESTORS = _BUILDING_ANCESTORS + ("R_UNIT",)
_FLOOR_ANCESTORS = _UNIT_ANCESTORS + ("R_FLOOR", "R_BASEMENT")
TREE_ANCESTORS: dict[str, tuple[str, ...]] = {
    "R_BUILDING": _SITE_ANCESTORS,
    "R_UNIT": _BUILDING_ANCESTORS,
    "R_FLOOR": _UNIT_ANCESTORS,
    "R_BASEMENT": _UNIT_ANCESTORS,
    "R_ROOM": _FLOOR_ANCESTORS,
    "R_SHOP": _BUILDING_ANCESTORS,
    "R_STORE": _BUILDING_ANCESTORS,
    "R_BOOTH": _BUILDING_ANCESTORS,
    "R_PARKING": _SITE_ANCESTORS,
    "R_ROW": _SITE_ANCESTORS + ("R_PARKING",),
    "R_ENTRANCE_DIR": _SITE_ANCESTORS,
    "R_ENTRANCE_ALPHA": _SITE_ANCESTORS,
    "R_ENTRANCE_NO": _SITE_ANCESTORS,
    "R_QUALIFIER": _SITE_ANCESTORS,
    "R_ENTITY": _SITE_ANCESTORS,
}

_EXPLICIT_BUILDING_CHILD_RE = re.compile(
    r"(?:第?[A-Z]{0,2}[0-9零〇一二两三四五六七八九十百千万]+[A-Z]{0,2}号?"
    r"(?:楼|栋|幢|座))$"
)


class RelationTreeError(ValueError):
    """Raised when relation-tree inputs violate the frozen contract."""


@dataclass(frozen=True)
class RelationTreeCompilation:
    """Variable-length semantic paths plus leaf-local deterministic Dedup."""

    path_types: np.ndarray
    path_values: np.ndarray
    path_lengths: np.ndarray
    semantic_resolved: np.ndarray
    dedup_codes: np.ndarray
    metrics: dict[str, Any]


def _members(grouping: CollisionGrouping, group_id: int) -> np.ndarray:
    start = int(grouping.group_offsets[group_id])
    end = int(grouping.group_offsets[group_id + 1])
    return grouping.member_order[start:end]


def _distribution(values: np.ndarray) -> dict[str, int]:
    return {
        str(int(value)): int(count)
        for value, count in sorted(Counter(values.tolist()).items())
    }


def _path_key(
    path_types: np.ndarray,
    path_values: np.ndarray,
    path_length: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(path_types[index]), int(path_values[index]))
        for index in range(path_length)
    )


def _is_explicit_building_child(relation: RawFineRelations) -> bool:
    return bool(_EXPLICIT_BUILDING_CHILD_RE.fullmatch(relation.child_entity))


def build_tree_relation_values(
    relation_matrix: UnifiedRelationMatrix,
    *,
    raw_relations: Sequence[RawFineRelations] | None = None,
) -> np.ndarray:
    """Project the unified matrix onto strong roles and normalize building children."""

    source = np.asarray(relation_matrix.values)
    if source.ndim != 2:
        raise RelationTreeError("统一关系矩阵必须为二维")
    source_indices = [
        UNIFIED_RELATION_INDEX[relation_type]
        for relation_type in TREE_RELATION_TYPES
    ]
    values = np.empty(
        (len(source), len(TREE_RELATION_TYPES)), dtype=np.int64
    )
    for target_index, source_index in enumerate(source_indices):
        values[:, target_index] = source[:, source_index]
    if raw_relations is None:
        return values
    if len(raw_relations) != len(values):
        raise RelationTreeError("raw_relations 与统一关系矩阵行数不一致")

    building = TREE_RELATION_INDEX["R_BUILDING"]
    child_values = source[:, UNIFIED_RELATION_INDEX["R_CHILD_NO"]]
    for row, relation in enumerate(raw_relations):
        child_value = int(child_values[row])
        if (
            values[row, building] < 0
            and child_value >= 0
            and _is_explicit_building_child(relation)
        ):
            # Unified R_BUILDING uses even values for numeric building numbers
            # and odd values for packed alphanumeric building codes.
            values[row, building] = child_value * 2
    return values


def _active_relations(bucket: np.ndarray) -> np.ndarray:
    """Return bucket-shared roles: varying roles plus present semantic ancestors."""

    varying: set[str] = set()
    for relation_index, relation_type in enumerate(TREE_RELATION_TYPES):
        # Missing is an explicit partition only for activation.  Missing values
        # are not serialized as fake relation values in the final path.
        if len(np.unique(bucket[:, relation_index])) > 1:
            varying.add(relation_type)
    active = set(varying)
    for relation_type in varying:
        active.update(TREE_ANCESTORS.get(relation_type, ()))
    return np.asarray(
        [
            index
            for index, relation_type in enumerate(TREE_RELATION_TYPES)
            if relation_type in active and np.any(bucket[:, index] >= 0)
        ],
        dtype=np.int16,
    )


def compile_relation_trees(
    *,
    grouping: CollisionGrouping,
    relation_values: np.ndarray,
    poi_ids: np.ndarray,
    progress: Callable[[str], None] | None = None,
) -> RelationTreeCompilation:
    """Build shared ordered relation paths, then add Dedup only at residual leaves."""

    relations = np.asarray(relation_values)
    poi_ids = np.asarray(poi_ids)
    rows = len(grouping.group_ids)
    if relations.shape != (rows, len(TREE_RELATION_TYPES)):
        raise RelationTreeError("relation_values shape 非法")
    if poi_ids.shape != (rows,) or poi_ids.dtype.kind not in {"i", "u"}:
        raise RelationTreeError("poi_ids 必须是与关系矩阵对齐的一维整数数组")
    if len(np.unique(poi_ids)) != rows:
        raise RelationTreeError("poi_ids 必须全局唯一")

    max_pairs = len(TREE_RELATION_TYPES)
    path_types = np.full((rows, max_pairs), -1, dtype=np.int16)
    path_values = np.full((rows, max_pairs), -1, dtype=np.int64)
    path_lengths = np.zeros(rows, dtype=np.uint8)
    dedup_codes = np.full(rows, -1, dtype=np.int32)
    active_group_counts: Counter[str] = Counter()

    for group_id in range(len(grouping.group_sizes)):
        members = _members(grouping, group_id)
        active = _active_relations(relations[members])
        for relation_index in active:
            active_group_counts[TREE_RELATION_TYPES[int(relation_index)]] += 1
        for row in members:
            present = active[relations[row, active] >= 0]
            length = len(present)
            path_lengths[row] = length
            if length:
                path_types[row, :length] = present
                path_values[row, :length] = relations[row, present]

        leaves: dict[Hashable, list[int]] = defaultdict(list)
        for row in members:
            length = int(path_lengths[row])
            leaves[
                _path_key(path_types[row], path_values[row], length)
            ].append(int(row))
        for leaf_members in leaves.values():
            if len(leaf_members) < 2:
                continue
            ordered = sorted(leaf_members, key=lambda row: int(poi_ids[row]))
            for code, row in enumerate(ordered):
                dedup_codes[row] = code

        if progress is not None and (
            (group_id + 1) % 20_000 == 0
            or group_id + 1 == len(grouping.group_sizes)
        ):
            progress(
                f"关系树已编译 {group_id + 1:,}/{len(grouping.group_sizes):,} "
                "个 G6 碰撞桶"
            )

    semantic_counts = Counter(
        (
            int(grouping.group_ids[row]),
            _path_key(
                path_types[row], path_values[row], int(path_lengths[row])
            ),
        )
        for row in range(rows)
    )
    semantic_resolved = np.asarray(
        [
            semantic_counts[
                (
                    int(grouping.group_ids[row]),
                    _path_key(
                        path_types[row],
                        path_values[row],
                        int(path_lengths[row]),
                    ),
                )
            ]
            == 1
            for row in range(rows)
        ],
        dtype=np.bool_,
    )
    collision_sizes = [count for count in semantic_counts.values() if count > 1]
    if int(np.count_nonzero(dedup_codes >= 0)) != sum(collision_sizes):
        raise RelationTreeError("Dedup 覆盖未与语义残留叶子对齐")

    final_keys = {
        (
            int(grouping.group_ids[row]),
            _path_key(
                path_types[row], path_values[row], int(path_lengths[row])
            ),
            int(dedup_codes[row]),
        )
        for row in range(rows)
    }
    if len(final_keys) != rows:
        raise RelationTreeError("追加叶子 Dedup 后 SID 仍不唯一")

    relation_token_lengths = np.zeros(rows, dtype=np.int64)
    for row in range(rows):
        relation_token_lengths[row] = sum(
            relation_token_cost(int(value))
            for value in path_values[row, : int(path_lengths[row])]
        )
    dedup_token_lengths = np.asarray(
        [
            0 if code < 0 else relation_token_cost(int(code))
            for code in dedup_codes
        ],
        dtype=np.int64,
    )
    total_tree_tokens = relation_token_lengths + dedup_token_lengths
    emitted = {
        relation_type: int(
            np.count_nonzero(path_types == relation_index)
        )
        for relation_index, relation_type in enumerate(TREE_RELATION_TYPES)
    }
    metrics: dict[str, Any] = {
        "schema_version": RELATION_TREE_SCHEMA_VERSION,
        "semantic_distinct_leaf_count": len(semantic_counts),
        "semantic_collision_excess": int(sum(size - 1 for size in collision_sizes)),
        "semantic_collision_poi_count": int(sum(collision_sizes)),
        "semantic_collision_leaf_count": len(collision_sizes),
        "max_semantic_collision_leaf_size": max(collision_sizes, default=1),
        "semantic_resolved_poi_count": int(np.count_nonzero(semantic_resolved)),
        "dedup_poi_count": int(np.count_nonzero(dedup_codes >= 0)),
        "dedup_leaf_count": len(collision_sizes),
        "max_dedup_code": int(dedup_codes.max(initial=-1)),
        "final_distinct_count": len(final_keys),
        "final_collision_excess": rows - len(final_keys),
        "relation_pair_distribution": _distribution(path_lengths),
        "relation_token_distribution": _distribution(relation_token_lengths),
        "dedup_token_distribution": _distribution(dedup_token_lengths),
        "tree_suffix_token_distribution": _distribution(total_tree_tokens),
        "mean_relation_pair_count": float(path_lengths.mean()),
        "max_relation_pair_count": int(path_lengths.max(initial=0)),
        "mean_relation_token_count": float(relation_token_lengths.mean()),
        "max_relation_token_count": int(relation_token_lengths.max(initial=0)),
        "mean_tree_suffix_token_count": float(total_tree_tokens.mean()),
        "max_tree_suffix_token_count": int(total_tree_tokens.max(initial=0)),
        "active_group_count_by_type": {
            relation_type: int(active_group_counts[relation_type])
            for relation_type in TREE_RELATION_TYPES
        },
        "emitted_pair_count_by_type": emitted,
    }
    return RelationTreeCompilation(
        path_types=path_types,
        path_values=path_values,
        path_lengths=path_lengths,
        semantic_resolved=semantic_resolved,
        dedup_codes=dedup_codes,
        metrics=metrics,
    )


def decoded_path(
    row: int,
    compilation: RelationTreeCompilation,
    vocabularies: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Decode one numeric tree path for a human-auditable case file."""

    output: list[dict[str, Any]] = []
    for depth in range(int(compilation.path_lengths[row])):
        relation_id = int(compilation.path_types[row, depth])
        relation_type = TREE_RELATION_TYPES[relation_id]
        value = int(compilation.path_values[row, depth])
        output.append(
            {
                "type": relation_type,
                "type_id": relation_id,
                "value": value,
                "decoded": decode_unified_relation_value(
                    relation_type, value, vocabularies
                ),
                "tokens": relation_token_cost(value),
            }
        )
    return output


def build_relation_trie(
    rows: Sequence[int],
    compilation: RelationTreeCompilation,
    vocabularies: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Materialize a compact ordered trie for selected audit buckets."""

    root: dict[tuple[int, int], Any] = {}
    leaf_rows: dict[int, list[int]] = defaultdict(list)
    for raw_row in rows:
        row = int(raw_row)
        cursor = root
        for depth in range(int(compilation.path_lengths[row])):
            relation_id = int(compilation.path_types[row, depth])
            value = int(compilation.path_values[row, depth])
            node = cursor.setdefault(
                (relation_id, value), {"children": {}, "rows": []}
            )
            node["rows"].append(row)
            cursor = node["children"]
        leaf_rows[id(cursor)].append(row)

    def serialize(
        children: Mapping[tuple[int, int], dict[str, Any]],
        terminal_rows: Sequence[int],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if terminal_rows:
            dedup = [
                {
                    "type": DEDUP_RELATION_TYPE,
                    "type_id": DEDUP_RELATION_TYPE_ID,
                    "value": int(compilation.dedup_codes[row]),
                    "poi_count": 1,
                }
                for row in sorted(
                    terminal_rows,
                    key=lambda item: int(compilation.dedup_codes[item]),
                )
                if compilation.dedup_codes[row] >= 0
            ]
            if dedup:
                result.extend(dedup)
            else:
                result.append(
                    {"type": "EOS", "poi_count": len(terminal_rows)}
                )
        for (relation_id, value), node in sorted(
            children.items(), key=lambda item: item[0]
        ):
            relation_type = TREE_RELATION_TYPES[relation_id]
            child_cursor = node["children"]
            child_terminal = leaf_rows.get(id(child_cursor), [])
            result.append(
                {
                    "type": relation_type,
                    "type_id": relation_id,
                    "value": value,
                    "decoded": decode_unified_relation_value(
                        relation_type, value, vocabularies
                    ),
                    "poi_count": len(node["rows"]),
                    "children": serialize(child_cursor, child_terminal),
                }
            )
        return result

    return serialize(root, leaf_rows.get(id(root), []))

"""Build unique variable-length GHR-SID identifiers for SFT."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from poi_gr.methods.ghr_sid.entity_structure import _load_structure_arrays
from poi_gr.methods.ghr_sid.g6_entity_structure import (
    EXPECTED_POST_G6_GROUP_COUNT,
    EXPECTED_POST_G6_POI_COUNT,
    reconstruct_post_g6_groups,
    validate_g6_entity_structure_output,
)
from poi_gr.methods.ghr_sid.g6_relation_tree import (
    validate_g6_relation_tree_output,
)
from poi_gr.methods.ghr_sid.relation_tree import TREE_RELATION_TYPES
from poi_gr.methods.ghr_sid.unified_relations import UNIFIED_RELATION_TYPES
from poi_gr.methods.tiger.data import load_tiger_id_lookup
from poi_gr.methods.tiger.identifier import sha256_file


GHR_IDENTIFIER_SCHEMA_VERSION = "ghr-sid-identifier-v1"
GHR_IDENTIFIER_METRICS_SCHEMA_VERSION = "ghr-sid-identifier-metrics-v1"
GHR_IDENTIFIER_VOCAB_SCHEMA_VERSION = "ghr-sid-identifier-vocab-v1"
VALUE_RADIX = 1024
VARIANT_EXP09 = "exp09"
VARIANT_EXP14 = "exp14"
SUPPORTED_VARIANTS = (VARIANT_EXP09, VARIANT_EXP14)
OUTPUT_FILENAMES = (
    "poi_ids.npy",
    "identifier_token_codes.npy",
    "identifier_lengths.npy",
    "dedup_codes.npy",
    "poi_ghr_id_mapping.parquet",
    "identifier_vocabulary.json",
    "ghr_id_manifest.json",
    "metrics.json",
    "_SUCCESS",
)


class GhrIdentifierError(ValueError):
    """Raised when a GHR identifier input or output violates its contract."""


@dataclass(frozen=True)
class IdentifierVocabulary:
    """Logical identifier-token vocabulary and namespace offsets."""

    tokens: tuple[str, ...]
    offsets: dict[str, int]
    relation_type_count: int


@dataclass(frozen=True)
class GhrIdentifierResult:
    """Completed unique GHR identifier artifacts."""

    manifest: dict[str, Any]
    metrics: dict[str, Any]
    output_dir: Path


def base1024_digits(value: int) -> tuple[int, ...]:
    """Serialize one non-negative integer as high-order-first base-1024 digits."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise GhrIdentifierError("关系值必须是整数")
    value = int(value)
    if value < 0:
        raise GhrIdentifierError("关系值不能为负数")
    digits = [value % VALUE_RADIX]
    value //= VALUE_RADIX
    while value:
        digits.append(value % VALUE_RADIX)
        value //= VALUE_RADIX
    return tuple(reversed(digits))


def build_identifier_vocabulary(relation_type_count: int) -> IdentifierVocabulary:
    """Return stable logical tokens for one GHR relation schema."""

    if relation_type_count <= 0:
        raise GhrIdentifierError("relation_type_count 必须大于 0")
    tokens: list[str] = []
    offsets: dict[str, int] = {}

    def extend(name: str, values: Sequence[str]) -> None:
        offsets[name] = len(tokens)
        tokens.extend(values)

    extend("s1", [f"<S1_{value}>" for value in range(1024)])
    extend("s2", [f"<S2_{value}>" for value in range(1024)])
    extend("s3", [f"<S3_{value}>" for value in range(1024)])
    geohash_alphabet = "0123456789bcdefghjkmnpqrstuvwxyz"
    extend("geo", [f"<G_{value}>" for value in geohash_alphabet])
    extend("relation", [f"<R_{value}>" for value in range(relation_type_count)])
    extend("value", [f"<V_{value}>" for value in range(VALUE_RADIX)])
    extend("dedup", ["<D>"])
    if len(tokens) != len(set(tokens)):
        raise GhrIdentifierError("GHR identifier 词表存在重复 Token")
    return IdentifierVocabulary(
        tokens=tuple(tokens),
        offsets=offsets,
        relation_type_count=relation_type_count,
    )


def serialize_identifier_codes(
    *,
    base_sid: Sequence[int],
    geo_codes: Sequence[int],
    relation_types: Sequence[int],
    relation_values: Sequence[int],
    dedup_code: int,
    vocabulary: IdentifierVocabulary,
) -> tuple[int, ...]:
    """Compose one namespace-safe variable-length logical token sequence."""

    if len(base_sid) != 3:
        raise GhrIdentifierError("GHR base SID 必须恰好三层")
    if len(relation_types) != len(relation_values):
        raise GhrIdentifierError("关系类型和值长度不一致")
    base = tuple(int(value) for value in base_sid)
    if any(value < 0 or value >= 1024 for value in base):
        raise GhrIdentifierError("GHR base SID 超出 [0,1024)")

    output = [
        vocabulary.offsets["s1"] + base[0],
        vocabulary.offsets["s2"] + base[1],
        vocabulary.offsets["s3"] + base[2],
    ]
    for value in geo_codes:
        numeric = int(value)
        if numeric < 0 or numeric >= 32:
            raise GhrIdentifierError("G6 路径码超出 [0,32)")
        output.append(vocabulary.offsets["geo"] + numeric)
    for relation_type, relation_value in zip(
        relation_types, relation_values, strict=True
    ):
        relation_type = int(relation_type)
        if relation_type < 0 or relation_type >= vocabulary.relation_type_count:
            raise GhrIdentifierError("关系类型超出冻结 schema")
        output.append(vocabulary.offsets["relation"] + relation_type)
        output.extend(
            vocabulary.offsets["value"] + digit
            for digit in base1024_digits(int(relation_value))
        )
    if dedup_code >= 0:
        output.append(vocabulary.offsets["dedup"])
        output.extend(
            vocabulary.offsets["value"] + digit
            for digit in base1024_digits(int(dedup_code))
        )
    return tuple(output)


def assign_leaf_dedup_codes(
    *,
    group_ids: np.ndarray,
    path_types: np.ndarray,
    path_values: np.ndarray,
    path_lengths: np.ndarray,
    poi_ids: np.ndarray,
) -> np.ndarray:
    """Assign D0... only inside residual semantic leaves, by numeric POI ID."""

    group_ids = np.asarray(group_ids)
    path_types = np.asarray(path_types)
    path_values = np.asarray(path_values)
    path_lengths = np.asarray(path_lengths)
    poi_ids = np.asarray(poi_ids)
    rows = len(group_ids)
    if (
        path_types.shape != path_values.shape
        or path_types.shape[0] != rows
        or path_lengths.shape != (rows,)
        or poi_ids.shape != (rows,)
    ):
        raise GhrIdentifierError("叶子 Dedup 输入 shape 不一致")
    if poi_ids.dtype.kind not in {"i", "u"} or len(np.unique(poi_ids)) != rows:
        raise GhrIdentifierError("叶子 Dedup 要求唯一整数 POI ID")

    members_by_group: dict[int, list[int]] = defaultdict(list)
    for row, group_id in enumerate(group_ids):
        members_by_group[int(group_id)].append(row)
    dedup = np.full(rows, -1, dtype=np.int32)
    for members in members_by_group.values():
        leaves: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
        for row in members:
            length = int(path_lengths[row])
            key = tuple(
                (int(path_types[row, depth]), int(path_values[row, depth]))
                for depth in range(length)
            )
            leaves[key].append(row)
        for leaf in leaves.values():
            if len(leaf) < 2:
                continue
            for code, row in enumerate(
                sorted(leaf, key=lambda item: int(poi_ids[item]))
            ):
                dedup[row] = code
    return dedup


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise GhrIdentifierError(f"{name} 不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GhrIdentifierError(f"{name} JSON 解析失败：{path}") from error
    if not isinstance(payload, dict):
        raise GhrIdentifierError(f"{name} 必须是 JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _distribution(values: np.ndarray) -> dict[str, int]:
    return {
        str(int(value)): int(count)
        for value, count in sorted(Counter(values.tolist()).items())
    }


def _array_contract(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _file_contract(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _relation_source(
    variant: str, relation_dir: Path
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    if variant == VARIANT_EXP09:
        validate_g6_entity_structure_output(relation_dir)
        relation_types = np.load(
            relation_dir / "relation_path_types.npy", mmap_mode="r", allow_pickle=False
        )
        relation_values = np.load(
            relation_dir / "relation_path_values.npy", mmap_mode="r", allow_pickle=False
        )
        path_lengths = np.sum(relation_types >= 0, axis=1).astype(np.uint8)
        relation_type_count = len(UNIFIED_RELATION_TYPES)
        relation_names = list(UNIFIED_RELATION_TYPES)
        source_dedup = np.full(len(path_lengths), -1, dtype=np.int32)
    elif variant == VARIANT_EXP14:
        validate_g6_relation_tree_output(relation_dir)
        relation_types = np.load(
            relation_dir / "relation_path_types.npy", mmap_mode="r", allow_pickle=False
        )
        relation_values = np.load(
            relation_dir / "relation_path_values.npy", mmap_mode="r", allow_pickle=False
        )
        path_lengths = np.asarray(
            np.load(
                relation_dir / "relation_path_lengths.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            dtype=np.uint8,
        )
        relation_type_count = len(TREE_RELATION_TYPES)
        relation_names = list(TREE_RELATION_TYPES)
        source_dedup = np.asarray(
            np.load(
                relation_dir / "dedup_codes.npy", mmap_mode="r", allow_pickle=False
            ),
            dtype=np.int32,
        )
    else:
        raise GhrIdentifierError(f"未知 GHR variant：{variant}")

    post_rows = np.load(
        relation_dir / "post_g6_collision_rows.npy", mmap_mode="r", allow_pickle=False
    )
    post_groups = np.load(
        relation_dir / "post_g6_group_ids.npy", mmap_mode="r", allow_pickle=False
    )
    post_poi_ids = np.load(
        relation_dir / "post_g6_poi_ids.npy", mmap_mode="r", allow_pickle=False
    )
    expected = (EXPECTED_POST_G6_POI_COUNT,)
    if (
        post_rows.shape != expected
        or post_groups.shape != expected
        or post_poi_ids.shape != expected
        or path_lengths.shape != expected
        or relation_types.shape != relation_values.shape
        or relation_types.shape[0] != expected[0]
        or source_dedup.shape != expected
    ):
        raise GhrIdentifierError("关系实验 post-G6 数组 shape 不一致")
    if np.any(path_lengths > relation_types.shape[1]):
        raise GhrIdentifierError("关系路径长度超过数组宽度")
    for row in range(len(path_lengths)):
        length = int(path_lengths[row])
        if np.any(relation_types[row, :length] < 0) or np.any(
            relation_values[row, :length] < 0
        ):
            raise GhrIdentifierError("关系路径有效前缀包含负值")
        if np.any(relation_types[row, length:] >= 0):
            raise GhrIdentifierError("关系路径不是左对齐前缀")
    return (
        np.asarray(post_rows),
        np.asarray(post_groups),
        np.asarray(post_poi_ids),
        relation_types,
        relation_values,
        relation_type_count,
        relation_names,
        path_lengths,
        source_dedup,
    )


def _mapping_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("poi_id", pa.int64(), nullable=False),
            pa.field("s1", pa.int32(), nullable=False),
            pa.field("s2", pa.int32(), nullable=False),
            pa.field("s3", pa.int32(), nullable=False),
            pa.field("identifier_length", pa.uint8(), nullable=False),
            pa.field("geo_length", pa.uint8(), nullable=False),
            pa.field("relation_pair_count", pa.uint8(), nullable=False),
            pa.field("dedup_code", pa.int32(), nullable=True),
            pa.field("ghr_id_key", pa.string(), nullable=False),
        ]
    )


def _write_mapping(
    path: Path,
    *,
    poi_ids: np.ndarray,
    base_codes: np.ndarray,
    token_codes: np.ndarray,
    lengths: np.ndarray,
    geo_lengths: np.ndarray,
    relation_pair_counts: np.ndarray,
    dedup_codes: np.ndarray,
    tokens: Sequence[str],
    chunk_rows: int,
) -> None:
    schema = _mapping_schema()
    writer = pq.ParquetWriter(
        path,
        schema,
        version="2.6",
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=True,
    )
    try:
        for start in range(0, len(poi_ids), chunk_rows):
            end = min(start + chunk_rows, len(poi_ids))
            keys = [
                "".join(tokens[int(code)] for code in token_codes[row, : int(lengths[row])])
                for row in range(start, end)
            ]
            chunk_dedup = np.asarray(dedup_codes[start:end], dtype=np.int32)
            table = pa.Table.from_arrays(
                [
                    pa.array(poi_ids[start:end], type=pa.int64()),
                    pa.array(base_codes[start:end, 0], type=pa.int32()),
                    pa.array(base_codes[start:end, 1], type=pa.int32()),
                    pa.array(base_codes[start:end, 2], type=pa.int32()),
                    pa.array(lengths[start:end], type=pa.uint8()),
                    pa.array(geo_lengths[start:end], type=pa.uint8()),
                    pa.array(relation_pair_counts[start:end], type=pa.uint8()),
                    pa.array(
                        chunk_dedup,
                        mask=chunk_dedup < 0,
                        type=pa.int32(),
                    ),
                    pa.array(keys, type=pa.string()),
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=chunk_rows)
    finally:
        writer.close()


def build_ghr_identifiers(
    *,
    project_root: Path,
    variant: str,
    tiger_id_dir: Path,
    structure_dir: Path,
    relation_dir: Path,
    output_dir: Path,
    chunk_rows: int = 100_000,
    progress: Callable[[str], None] | None = None,
) -> GhrIdentifierResult:
    """Compose and persist one full-catalog unique GHR identifier mapping."""

    started = time.perf_counter()
    project_root = project_root.resolve()
    tiger_id_dir = tiger_id_dir.resolve()
    structure_dir = structure_dir.resolve()
    relation_dir = relation_dir.resolve()
    output_dir = output_dir.resolve()
    if variant not in SUPPORTED_VARIANTS:
        raise GhrIdentifierError(f"variant 必须是 {SUPPORTED_VARIANTS}")
    if chunk_rows <= 0:
        raise GhrIdentifierError("chunk_rows 必须大于 0")
    if output_dir.exists():
        raise GhrIdentifierError(f"输出目录已存在，拒绝覆盖：{output_dir}")

    if progress is not None:
        progress("校验 TIGER、G6 和关系实验输入")
    tiger_lookup, tiger_manifest, tiger_mapping_hash, tiger_manifest_hash = (
        load_tiger_id_lookup(tiger_id_dir)
    )
    poi_count = tiger_lookup.poi_count
    poi_ids = np.empty(poi_count, dtype=np.int64)
    for poi_id, row in tiger_lookup.row_by_poi_id.items():
        if not poi_id.isdecimal():
            raise GhrIdentifierError(f"POI ID 不是数字：{poi_id}")
        poi_ids[row] = int(poi_id)
    if len(np.unique(poi_ids)) != poi_count:
        raise GhrIdentifierError("TIGER mapping 的数字 POI ID 不唯一")
    base_codes = np.asarray(tiger_lookup.codes[:, :3], dtype=np.int32)
    del tiger_lookup

    structure_arrays, structure_manifest, _ = _load_structure_arrays(structure_dir)
    grouping = reconstruct_post_g6_groups(
        base_sid_keys=structure_arrays["base_sid_keys"],
        gid8_codes=structure_arrays["collision_gid8_codes"],
        resolution_stage=structure_arrays["resolution_stage"],
    )
    if (
        len(grouping.collision_rows) != EXPECTED_POST_G6_POI_COUNT
        or len(grouping.group_sizes) != EXPECTED_POST_G6_GROUP_COUNT
    ):
        raise GhrIdentifierError("post-G6 冻结规模发生变化")

    (
        source_post_rows,
        source_post_groups,
        source_post_poi_ids,
        relation_types,
        relation_values,
        relation_type_count,
        relation_names,
        path_lengths,
        source_dedup,
    ) = _relation_source(variant, relation_dir)
    if not np.array_equal(source_post_rows, grouping.collision_rows):
        raise GhrIdentifierError("关系实验 post-G6 collision row 与冻结结构不一致")
    if not np.array_equal(source_post_groups, grouping.group_ids):
        raise GhrIdentifierError("关系实验 post-G6 group ID 与冻结结构不一致")
    expected_post_poi = np.asarray(structure_arrays["collision_poi_ids"])[
        grouping.collision_rows
    ]
    if not np.array_equal(source_post_poi_ids, expected_post_poi):
        raise GhrIdentifierError("关系实验 post-G6 POI 行序与冻结结构不一致")

    recomputed_dedup = assign_leaf_dedup_codes(
        group_ids=source_post_groups,
        path_types=relation_types,
        path_values=relation_values,
        path_lengths=path_lengths,
        poi_ids=source_post_poi_ids,
    )
    if variant == VARIANT_EXP14 and not np.array_equal(
        recomputed_dedup, source_dedup
    ):
        raise GhrIdentifierError("EXP-14 叶子 Dedup 与重新计算结果不一致")
    post_dedup = recomputed_dedup

    collision_poi_ids = np.asarray(structure_arrays["collision_poi_ids"])
    sort_order = np.argsort(poi_ids, kind="stable")
    sorted_poi_ids = poi_ids[sort_order]
    collision_positions = np.searchsorted(sorted_poi_ids, collision_poi_ids)
    if np.any(collision_positions >= poi_count) or not np.array_equal(
        sorted_poi_ids[collision_positions], collision_poi_ids
    ):
        raise GhrIdentifierError("G6 碰撞 POI 无法完整映射到 TIGER 全目录")
    collision_full_rows = sort_order[collision_positions]
    if len(np.unique(collision_full_rows)) != len(collision_full_rows):
        raise GhrIdentifierError("G6 collision row 映射重复")
    post_full_rows = collision_full_rows[grouping.collision_rows]

    full_geo_lengths = np.zeros(poi_count, dtype=np.uint8)
    full_geo_lengths[collision_full_rows] = np.asarray(
        structure_arrays["coarse_geo_lengths"], dtype=np.uint8
    )
    full_relation_pairs = np.zeros(poi_count, dtype=np.uint8)
    full_relation_pairs[post_full_rows] = path_lengths
    full_dedup = np.full(poi_count, -1, dtype=np.int32)
    full_dedup[post_full_rows] = post_dedup

    vocabulary = build_identifier_vocabulary(relation_type_count)
    per_post_lengths = np.empty(len(path_lengths), dtype=np.uint8)
    max_identifier_length = 3
    for row in range(len(path_lengths)):
        relation_token_count = sum(
            1 + len(base1024_digits(int(relation_values[row, depth])))
            for depth in range(int(path_lengths[row]))
        )
        dedup_token_count = (
            0
            if post_dedup[row] < 0
            else 1 + len(base1024_digits(int(post_dedup[row])))
        )
        full_row = int(post_full_rows[row])
        length = (
            3
            + int(full_geo_lengths[full_row])
            + relation_token_count
            + dedup_token_count
        )
        per_post_lengths[row] = length
        max_identifier_length = max(max_identifier_length, length)
    collision_lengths = 3 + np.asarray(
        structure_arrays["coarse_geo_lengths"], dtype=np.uint8
    ).astype(np.int16)
    max_identifier_length = max(
        max_identifier_length, int(collision_lengths.max(initial=3))
    )

    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging_dir.exists():
        raise GhrIdentifierError(f"暂存目录已存在：{staging_dir}")
    staging_dir.mkdir(parents=True)
    try:
        if progress is not None:
            progress("编译全目录变长 identifier Token 数组")
        codes_path = staging_dir / "identifier_token_codes.npy"
        token_codes = np.lib.format.open_memmap(
            codes_path,
            mode="w+",
            dtype=np.int32,
            shape=(poi_count, max_identifier_length),
        )
        token_codes.fill(-1)
        lengths = np.full(poi_count, 3, dtype=np.uint8)
        token_codes[:, 0] = vocabulary.offsets["s1"] + base_codes[:, 0]
        token_codes[:, 1] = vocabulary.offsets["s2"] + base_codes[:, 1]
        token_codes[:, 2] = vocabulary.offsets["s3"] + base_codes[:, 2]

        coarse_codes = np.asarray(structure_arrays["coarse_geo_codes"])
        coarse_lengths = np.asarray(structure_arrays["coarse_geo_lengths"])
        for collision_row, full_row in enumerate(collision_full_rows):
            geo_length = int(coarse_lengths[collision_row])
            if geo_length:
                geo = coarse_codes[collision_row, :geo_length]
                if np.any(geo < 0) or np.any(geo >= 32):
                    raise GhrIdentifierError("冻结 G6 分支包含非法码")
                token_codes[full_row, 3 : 3 + geo_length] = (
                    vocabulary.offsets["geo"] + geo
                )
                lengths[full_row] = 3 + geo_length

        for post_row, full_row in enumerate(post_full_rows):
            geo_length = int(full_geo_lengths[full_row])
            relation_length = int(path_lengths[post_row])
            sequence = serialize_identifier_codes(
                base_sid=base_codes[full_row],
                geo_codes=coarse_codes[
                    int(grouping.collision_rows[post_row]), :geo_length
                ],
                relation_types=relation_types[post_row, :relation_length],
                relation_values=relation_values[post_row, :relation_length],
                dedup_code=int(post_dedup[post_row]),
                vocabulary=vocabulary,
            )
            token_codes[full_row, : len(sequence)] = sequence
            lengths[full_row] = len(sequence)
        token_codes.flush()

        if int(lengths.max()) != max_identifier_length:
            raise GhrIdentifierError("最终 identifier 最大长度与预计算不一致")
        if np.any(token_codes[:, 0] < vocabulary.offsets["s1"]):
            raise GhrIdentifierError("identifier 首 Token 非法")
        for row in range(poi_count):
            length = int(lengths[row])
            if np.any(token_codes[row, :length] < 0) or np.any(
                token_codes[row, length:] >= 0
            ):
                raise GhrIdentifierError("identifier Token 不是左对齐有效前缀")
        if int(token_codes.max()) >= len(vocabulary.tokens):
            raise GhrIdentifierError("identifier logical Token 超出词表")

        if progress is not None:
            progress("验证分层唯一性并写入 POI mapping")
        # Exact construction proof: base singleton buckets are already unique;
        # all non-singletons are partitioned by the frozen G6 path; every
        # remaining post-G6 semantic leaf has a continuous local D assignment.
        tiger_bucket_sizes = np.bincount(
            np.unique(base_codes, axis=0, return_inverse=True)[1]
        )
        if int(np.sum(tiger_bucket_sizes)) != poi_count:
            raise GhrIdentifierError("TIGER base 桶计数不守恒")
        if np.any(post_dedup >= 0):
            for group_id in range(len(grouping.group_sizes)):
                start = int(grouping.group_offsets[group_id])
                end = int(grouping.group_offsets[group_id + 1])
                members = grouping.member_order[start:end]
                final_keys = []
                for row in members:
                    length = int(path_lengths[row])
                    final_keys.append(
                        (
                            tuple(
                                (
                                    int(relation_types[row, depth]),
                                    int(relation_values[row, depth]),
                                )
                                for depth in range(length)
                            ),
                            int(post_dedup[row]),
                        )
                    )
                if len(final_keys) != len(set(final_keys)):
                    raise GhrIdentifierError("追加局部 D 后 post-G6 组仍不唯一")

        poi_ids_path = staging_dir / "poi_ids.npy"
        lengths_path = staging_dir / "identifier_lengths.npy"
        dedup_path = staging_dir / "dedup_codes.npy"
        with poi_ids_path.open("wb") as handle:
            np.save(handle, poi_ids, allow_pickle=False)
        with lengths_path.open("wb") as handle:
            np.save(handle, lengths, allow_pickle=False)
        with dedup_path.open("wb") as handle:
            np.save(handle, full_dedup, allow_pickle=False)

        vocab_path = staging_dir / "identifier_vocabulary.json"
        vocab_payload = {
            "schema_version": GHR_IDENTIFIER_VOCAB_SCHEMA_VERSION,
            "value_radix": VALUE_RADIX,
            "value_digit_order": "most_significant_first",
            "relation_type_count": relation_type_count,
            "relation_types": relation_names,
            "offsets": vocabulary.offsets,
            "tokens": list(vocabulary.tokens),
            "token_count": len(vocabulary.tokens),
            "dedup_serialization": ["<D>", "one_or_more_<V_n>"],
        }
        _write_json(vocab_path, vocab_payload)

        mapping_path = staging_dir / "poi_ghr_id_mapping.parquet"
        _write_mapping(
            mapping_path,
            poi_ids=poi_ids,
            base_codes=base_codes,
            token_codes=token_codes,
            lengths=lengths,
            geo_lengths=full_geo_lengths,
            relation_pair_counts=full_relation_pairs,
            dedup_codes=full_dedup,
            tokens=vocabulary.tokens,
            chunk_rows=chunk_rows,
        )

        dedup_count = int(np.count_nonzero(full_dedup >= 0))
        metrics: dict[str, Any] = {
            "schema_version": GHR_IDENTIFIER_METRICS_SCHEMA_VERSION,
            "status": "completed",
            "variant": variant,
            "poi_count": poi_count,
            "distinct_identifier_count": poi_count,
            "identifier_unique_ratio": 1.0,
            "tiger_base_singleton_poi_count": poi_count - len(collision_poi_ids),
            "tiger_collision_poi_count": len(collision_poi_ids),
            "g6_resolved_poi_count": len(collision_poi_ids) - len(post_full_rows),
            "post_g6_poi_count": len(post_full_rows),
            "post_g6_group_count": len(grouping.group_sizes),
            "relation_semantic_singleton_poi_count": int(
                len(post_full_rows) - dedup_count
            ),
            "dedup_poi_count": dedup_count,
            "dedup_catalog_ratio": dedup_count / poi_count,
            "max_dedup_code": int(full_dedup.max(initial=-1)),
            "identifier_length_distribution_excluding_eos": _distribution(lengths),
            "mean_identifier_length_excluding_eos": float(lengths.mean()),
            "max_identifier_length_excluding_eos": int(lengths.max()),
            "max_full_target_tokens_including_eos": int(lengths.max()) + 1,
            "relation_pair_distribution_full_catalog": _distribution(
                full_relation_pairs
            ),
            "query_or_order_used_to_build_identifier": False,
            "dedup_assignment": "numeric_poi_id_ascending_within_residual_semantic_leaf",
            "build_seconds": time.perf_counter() - started,
        }
        metrics_path = staging_dir / "metrics.json"
        _write_json(metrics_path, metrics)

        source_manifest_path = relation_dir / "manifest.json"
        structure_manifest_path = structure_dir / "manifest.json"
        arrays = {
            "poi_ids": _array_contract(poi_ids_path, poi_ids),
            "identifier_token_codes": _array_contract(codes_path, token_codes),
            "identifier_lengths": _array_contract(lengths_path, lengths),
            "dedup_codes": _array_contract(dedup_path, full_dedup),
        }
        manifest: dict[str, Any] = {
            "schema_version": GHR_IDENTIFIER_SCHEMA_VERSION,
            "status": "completed",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "method": "GHR-SID",
            "variant": variant,
            "protocol": "TIGER3->G6->relations->optional_leaf_D->EOS",
            "query_or_order_used_to_build_identifier": False,
            "sources": {
                "tiger_identifier": {
                    "path": str(tiger_id_dir),
                    "schema_version": tiger_manifest["schema_version"],
                    "mapping_sha256": tiger_mapping_hash,
                    "manifest_sha256": tiger_manifest_hash,
                },
                "g6_structure": {
                    "path": str(structure_dir),
                    "schema_version": structure_manifest["schema_version"],
                    "manifest_sha256": sha256_file(structure_manifest_path),
                },
                "relation_experiment": {
                    "path": str(relation_dir),
                    "manifest_sha256": sha256_file(source_manifest_path),
                },
            },
            "identifier": {
                "poi_count": poi_count,
                "base_sid_codebook_sizes": [1024, 1024, 1024],
                "variable_length": True,
                "minimum_length_excluding_eos": int(lengths.min()),
                "maximum_length_excluding_eos": int(lengths.max()),
                "eos_token": "</TARGET_POI>",
                "relation_type_count": relation_type_count,
                "value_radix": VALUE_RADIX,
                "value_digit_order": "most_significant_first",
                "dedup_type_token": "<D>",
                "dedup_only_on_residual_semantic_leaf": True,
                "dedup_assignment": "numeric_poi_id_ascending",
            },
            "outputs": {
                "arrays": arrays,
                "mapping": {
                    **_file_contract(mapping_path),
                    "rows": poi_count,
                    "poi_id_unique": True,
                    "ghr_id_key_unique": True,
                },
                "vocabulary": _file_contract(vocab_path),
                "metrics": _file_contract(metrics_path),
                "success": {"path": "_SUCCESS"},
            },
        }
        manifest_path = staging_dir / "ghr_id_manifest.json"
        _write_json(manifest_path, manifest)
        (staging_dir / "_SUCCESS").touch()
        os.replace(staging_dir, output_dir)
    except BaseException:
        if staging_dir.exists():
            for path in staging_dir.iterdir():
                path.unlink(missing_ok=True)
            staging_dir.rmdir()
        raise

    return GhrIdentifierResult(
        manifest=manifest,
        metrics=metrics,
        output_dir=output_dir,
    )


def validate_ghr_identifier_output(output_dir: Path) -> dict[str, Any]:
    """Rehash and validate one completed GHR identifier directory."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "ghr_id_manifest.json", "GHR manifest")
    if manifest.get("schema_version") != GHR_IDENTIFIER_SCHEMA_VERSION:
        raise GhrIdentifierError("GHR identifier schema_version 不匹配")
    if manifest.get("status") != "completed" or not (output_dir / "_SUCCESS").is_file():
        raise GhrIdentifierError("GHR identifier 未完成")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise GhrIdentifierError("GHR manifest 缺少 outputs")
    validated = 0
    arrays: dict[str, np.ndarray] = {}
    for name, contract in outputs.get("arrays", {}).items():
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file() or sha256_file(path) != contract.get("sha256"):
            raise GhrIdentifierError(f"GHR 输出数组缺失或 SHA 错误：{name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != contract.get("shape") or str(array.dtype) != contract.get("dtype"):
            raise GhrIdentifierError(f"GHR 输出数组 shape/dtype 错误：{name}")
        arrays[name] = array
        validated += 1
    for name in ("mapping", "vocabulary", "metrics"):
        contract = outputs.get(name)
        if not isinstance(contract, dict):
            raise GhrIdentifierError(f"GHR manifest 缺少 {name}")
        path = output_dir / str(contract.get("path", ""))
        if not path.is_file() or sha256_file(path) != contract.get("sha256"):
            raise GhrIdentifierError(f"GHR 输出缺失或 SHA 错误：{name}")
        validated += 1
    poi_ids = arrays["poi_ids"]
    codes = arrays["identifier_token_codes"]
    lengths = arrays["identifier_lengths"]
    if len(np.unique(poi_ids)) != len(poi_ids):
        raise GhrIdentifierError("GHR 输出 POI ID 不唯一")
    if codes.shape[0] != len(poi_ids) or lengths.shape != (len(poi_ids),):
        raise GhrIdentifierError("GHR 输出主数组行数不一致")
    vocabulary = _load_json(
        output_dir / outputs["vocabulary"]["path"], "identifier vocabulary"
    )
    token_count = vocabulary.get("token_count")
    if np.any(lengths < 3) or np.any(lengths > codes.shape[1]):
        raise GhrIdentifierError("GHR identifier 长度非法")
    if int(codes.max()) >= token_count or int(codes.min()) < -1:
        raise GhrIdentifierError("GHR logical Token 超出词表")
    parquet_rows = pq.ParquetFile(
        output_dir / outputs["mapping"]["path"]
    ).metadata.num_rows
    if parquet_rows != len(poi_ids):
        raise GhrIdentifierError("GHR mapping 行数不一致")
    return {
        "status": "validated",
        "variant": manifest["variant"],
        "poi_count": len(poi_ids),
        "validated_output_count": validated + 1,
    }

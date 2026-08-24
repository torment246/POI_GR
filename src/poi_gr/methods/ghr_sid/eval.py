"""Unconstrained beam evaluation utilities for variable-length GHR identifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from poi_gr.pid.trie import CompactPidTrie, sha256_file
from poi_gr.sft.evaluation import GenerativeEvalError, encode_prompt_like_training


FNV_OFFSET_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1
ALIGNED_COLLISION_SCHEMA_VERSION = "ghr-aligned-collision-quantizer-v1"
ALIGNED_COLLISION_CAPACITIES = (1024, 1024, 1024, 32, 32)
ALIGNED_COLLISION_PREFIXES = ("S1", "S2", "S3", "R1", "R2")


class GhrEvalError(ValueError):
    """Raised when GHR evaluation violates its frozen identifier contract."""


def _aligned_identifier_tokens() -> tuple[str, ...]:
    return tuple(
        f"<{prefix}_{code}>"
        for prefix, capacity in zip(
            ALIGNED_COLLISION_PREFIXES,
            ALIGNED_COLLISION_CAPACITIES,
            strict=True,
        )
        for code in range(capacity)
    )


def _aligned_codes_to_logical(codes: np.ndarray) -> np.ndarray:
    """Map fixed-position aligned codes into the shared logical-token space."""

    if codes.ndim != 2 or codes.shape[1] != len(ALIGNED_COLLISION_CAPACITIES):
        raise GhrEvalError("aligned identifier codes 必须是五层矩阵")
    if codes.dtype != np.int32:
        raise GhrEvalError("aligned identifier codes 必须是 int32")
    offsets = np.cumsum(
        np.asarray((0, *ALIGNED_COLLISION_CAPACITIES[:-1]), dtype=np.int32)
    )
    logical = np.array(codes, dtype=np.int32, copy=True)
    for level, capacity in enumerate(ALIGNED_COLLISION_CAPACITIES):
        values = logical[:, level]
        if np.any(values < 0) or np.any(values >= capacity):
            raise GhrEvalError(f"aligned identifier 第 {level + 1} 层超出容量")
        values += offsets[level]
    return logical


@dataclass(frozen=True)
class GhrTokenIds:
    target_open: int
    target_close: int
    logical_to_token: tuple[int, ...]
    token_to_logical: Mapping[int, int]
    eos: int
    minimum_identifier_length: int
    maximum_identifier_length: int

    @property
    def maximum_sequence_length(self) -> int:
        """Return open + identifier + close + EOS generation budget."""

        return self.maximum_identifier_length + 3


@dataclass(frozen=True)
class GhrExample:
    sample_id: str
    target_poi_id: str
    target_codes: tuple[int, ...]
    prompt_ids: tuple[int, ...]


@dataclass(frozen=True)
class GhrCandidate:
    codes: tuple[int, ...] | None
    poi_row: int | None
    score: float
    error: str | None


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise GhrEvalError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GhrEvalError(f"{name} JSON 非法：{path}") from error
    if not isinstance(value, dict):
        raise GhrEvalError(f"{name} 必须是 JSON object")
    return value


def _hash_codes(values: Sequence[int]) -> int:
    result = FNV_OFFSET_BASIS
    for value in values:
        result ^= int(value) + 1
        result = (result * FNV_PRIME) & UINT64_MASK
    result ^= len(values)
    return (result * FNV_PRIME) & UINT64_MASK


def _hash_code_matrix(codes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    hashes = np.full(len(lengths), FNV_OFFSET_BASIS, dtype=np.uint64)
    prime = np.uint64(FNV_PRIME)
    with np.errstate(over="ignore"):
        for position in range(codes.shape[1]):
            active = lengths > position
            values = codes[active, position].astype(np.uint64, copy=False) + 1
            hashes[active] = (hashes[active] ^ values) * prime
        hashes = (hashes ^ lengths.astype(np.uint64, copy=False)) * prime
    return hashes


class GhrIdIndex:
    """Compact exact lookup for frozen variable-length GHR identifiers."""

    def __init__(
        self,
        *,
        codes: np.ndarray,
        lengths: np.ndarray,
        poi_ids: np.ndarray,
    ) -> None:
        if codes.ndim != 2 or codes.dtype != np.int32:
            raise GhrEvalError("GHR identifier codes 必须是二维 int32")
        if lengths.shape != (codes.shape[0],) or lengths.dtype != np.uint8:
            raise GhrEvalError("GHR identifier lengths shape/dtype 不一致")
        if poi_ids.shape != (codes.shape[0],) or poi_ids.dtype != np.int64:
            raise GhrEvalError("GHR POI IDs shape/dtype 不一致")
        if np.any(lengths == 0) or np.any(lengths > codes.shape[1]):
            raise GhrEvalError("GHR identifier length 越界")
        positions = np.arange(codes.shape[1], dtype=np.int64)[None, :]
        active = positions < lengths.astype(np.int64)[:, None]
        if np.any(codes[active] < 0) or np.any(codes[~active] != -1):
            raise GhrEvalError("GHR identifier code 或 padding 非法")
        self.codes = codes
        self.lengths = lengths
        self.poi_ids = poi_ids
        self.row_count = int(codes.shape[0])
        self.maximum_identifier_length = int(codes.shape[1])
        hashes = _hash_code_matrix(codes, lengths)
        order = np.argsort(hashes, kind="stable")
        self.sorted_hashes = np.asarray(hashes[order], dtype=np.uint64)
        self.sorted_rows = np.asarray(order, dtype=np.int64)

    def lookup(self, values: Sequence[int]) -> int:
        codes = tuple(int(value) for value in values)
        if not codes or len(codes) > self.maximum_identifier_length:
            return -1
        key = np.uint64(_hash_codes(codes))
        start = int(np.searchsorted(self.sorted_hashes, key, side="left"))
        stop = int(np.searchsorted(self.sorted_hashes, key, side="right"))
        for position in range(start, stop):
            row = int(self.sorted_rows[position])
            if int(self.lengths[row]) != len(codes):
                continue
            if np.array_equal(self.codes[row, : len(codes)], codes):
                return row
        return -1

    def poi_id(self, row: int) -> str:
        if row < 0 or row >= self.row_count:
            raise GhrEvalError(f"GHR POI row 越界：{row}")
        return str(int(self.poi_ids[row]))


class GhrPrefixIndex:
    """Compact corpus lookup for frozen three-layer TIGER prefixes."""

    def __init__(self, codes: np.ndarray) -> None:
        if codes.ndim != 2 or codes.shape[1] < 3 or codes.dtype != np.int32:
            raise GhrEvalError("GHR prefix index 需要至少三层 int32 identifier")
        prefixes = np.asarray(codes[:, :3], dtype=np.int64)
        self.radix = int(prefixes.max(initial=0)) + 1
        packed = (
            (prefixes[:, 0] * self.radix + prefixes[:, 1]) * self.radix
            + prefixes[:, 2]
        )
        sorted_keys = np.sort(packed, kind="stable")
        boundaries = np.flatnonzero(
            np.concatenate((np.asarray([True]), sorted_keys[1:] != sorted_keys[:-1]))
        )
        self.keys = sorted_keys[boundaries]
        self.counts = np.diff(
            np.append(boundaries, len(sorted_keys))
        ).astype(np.int32, copy=False)

    def bucket_size(self, codes: Sequence[int]) -> int:
        """Return corpus POI count for one logical three-code prefix."""

        if len(codes) != 3:
            return 0
        values = tuple(int(value) for value in codes)
        if any(value < 0 or value >= self.radix for value in values):
            return 0
        key = (values[0] * self.radix + values[1]) * self.radix + values[2]
        position = int(np.searchsorted(self.keys, key))
        if position >= len(self.keys) or int(self.keys[position]) != key:
            return 0
        return int(self.counts[position])


def parse_generated_base_bucket(
    sequence: Sequence[int],
    *,
    tokens: GhrTokenIds,
) -> tuple[int, int, int] | None:
    """Parse GHR open/S1/S2/S3 independently of suffix, close and EOS."""

    values = [int(value) for value in sequence]
    if len(values) < 4 or values[0] != tokens.target_open:
        return None
    try:
        return tuple(tokens.token_to_logical[value] for value in values[1:4])  # type: ignore[return-value]
    except KeyError:
        return None


def build_ghr_compact_trie(
    index: GhrIdIndex,
    *,
    progress: Callable[[str], None] | None = None,
) -> CompactPidTrie:
    """Build a compact CSR Trie over frozen variable-length GHR code paths."""

    progress = progress or (lambda _: None)
    codes = index.codes
    lengths = index.lengths
    maximum_depth = index.maximum_identifier_length
    progress("正在对 GHR identifier 做稳定字典序排序……")
    sort_keys = tuple(codes[:, column] for column in range(maximum_depth - 1, -1, -1))
    sorted_rows = np.lexsort(sort_keys)
    sorted_lengths = lengths[sorted_rows]
    row_nodes = np.zeros(index.row_count, dtype=np.int32)
    next_node_id = 1
    edge_parent_parts: list[np.ndarray] = []
    edge_token_parts: list[np.ndarray] = []
    edge_child_parts: list[np.ndarray] = []
    terminal_node_parts: list[np.ndarray] = []
    terminal_row_parts: list[np.ndarray] = []

    for depth in range(maximum_depth):
        positions = np.flatnonzero(sorted_lengths > depth)
        if positions.size == 0:
            break
        rows = sorted_rows[positions]
        token_codes = np.asarray(codes[rows, depth], dtype=np.int32)
        parents = row_nodes[positions]
        boundaries = np.empty(token_codes.size, dtype=bool)
        boundaries[0] = True
        boundaries[1:] = (parents[1:] != parents[:-1]) | (
            token_codes[1:] != token_codes[:-1]
        )
        group_index = np.cumsum(boundaries, dtype=np.int64) - 1
        child_count = int(group_index[-1]) + 1
        if next_node_id + child_count > np.iinfo(np.int32).max:
            raise GhrEvalError("GHR Trie 节点数超过 int32 容量")
        child_nodes = (group_index + next_node_id).astype(np.int32)
        edge_parent_parts.append(parents[boundaries].astype(np.int32, copy=True))
        edge_token_parts.append(token_codes[boundaries].astype(np.int32, copy=True))
        edge_child_parts.append(
            np.arange(next_node_id, next_node_id + child_count, dtype=np.int32)
        )
        row_nodes[positions] = child_nodes
        next_node_id += child_count

        terminal_positions = positions[sorted_lengths[positions] == depth + 1]
        if terminal_positions.size:
            terminal_node_parts.append(row_nodes[terminal_positions].copy())
            terminal_row_parts.append(
                sorted_rows[terminal_positions].astype(np.int32, copy=True)
            )
        progress(
            f"GHR Trie 深度 {depth + 1}/{maximum_depth}：新增 {child_count:,} 节点，"
            f"累计 {next_node_id:,} 节点"
        )

    edge_parents = np.concatenate(edge_parent_parts)
    child_token_ids = np.concatenate(edge_token_parts)
    child_node_ids = np.concatenate(edge_child_parts)
    terminal_node_ids = np.concatenate(terminal_node_parts)
    terminal_poi_rows = np.concatenate(terminal_row_parts)
    if edge_parents.size != next_node_id - 1:
        raise GhrEvalError("GHR Trie 边数必须等于节点数减一")
    if np.any(edge_parents[1:] < edge_parents[:-1]):
        raise GhrEvalError("GHR Trie 父节点顺序异常")
    if terminal_node_ids.size != index.row_count:
        raise GhrEvalError("GHR Trie 叶子数与 identifier 数不一致")
    if np.unique(terminal_node_ids).size != index.row_count:
        raise GhrEvalError("GHR identifier 必须全局唯一")

    child_counts = np.bincount(edge_parents, minlength=next_node_id)
    child_offsets = np.empty(next_node_id + 1, dtype=np.int64)
    child_offsets[0] = 0
    np.cumsum(child_counts, out=child_offsets[1:])
    terminal_order = np.argsort(terminal_node_ids, kind="stable")
    return CompactPidTrie(
        child_offsets=child_offsets,
        child_token_ids=child_token_ids,
        child_node_ids=child_node_ids,
        terminal_node_ids=terminal_node_ids[terminal_order],
        terminal_poi_rows=terminal_poi_rows[terminal_order],
    )


class GhrLegalPathConstraint:
    """Restrict generation to exact prefixes of frozen GHR identifiers."""

    def __init__(
        self,
        *,
        trie: CompactPidTrie,
        tokens: GhrTokenIds,
        prompt_width: int,
        next_token_cache: dict[tuple[int, ...], list[int]] | None = None,
    ) -> None:
        if prompt_width <= 0:
            raise GhrEvalError("prompt_width 必须为正整数")
        self.trie = trie
        self.tokens = tokens
        self.prompt_width = prompt_width
        self._next_token_cache = (
            next_token_cache if next_token_cache is not None else {}
        )

    def _decode_code_tokens(self, values: Sequence[int]) -> tuple[int, ...]:
        try:
            return tuple(self.tokens.token_to_logical[int(value)] for value in values)
        except KeyError as error:
            raise GhrEvalError("生成序列离开 GHR identifier Token 集") from error

    def _allowed_after_codes(self, code_tokens: tuple[int, ...]) -> list[int]:
        cached = self._next_token_cache.get(code_tokens)
        if cached is not None:
            return cached
        logical_codes = self._decode_code_tokens(code_tokens)
        node_id = self.trie.traverse(logical_codes)
        if node_id < 0:
            raise GhrEvalError(f"identifier Prefix 不在冻结语料库中：{logical_codes}")
        children = self.trie.children(node_id)
        allowed = [self.tokens.logical_to_token[int(code)] for code in children]
        if self.trie.terminal_poi_row(node_id) >= 0:
            allowed.append(self.tokens.target_close)
        if not allowed:
            raise GhrEvalError(f"GHR Trie 非 terminal 节点没有合法后继：{node_id}")
        if len(code_tokens) <= 6:
            self._next_token_cache[code_tokens] = allowed
        return allowed

    def allowed_next(self, generated: Sequence[int]) -> list[int]:
        """Return legal next tokens for one generated-only prefix."""

        values = tuple(int(value) for value in generated)
        if not values:
            return [self.tokens.target_open]
        if values[0] != self.tokens.target_open:
            raise GhrEvalError("合法路径生成必须以 <TARGET_POI> 开始")
        body = values[1:]
        if not body:
            return self._allowed_after_codes(())
        if self.tokens.eos in body:
            eos_index = body.index(self.tokens.eos)
            if any(value != self.tokens.eos for value in body[eos_index:]):
                raise GhrEvalError("GHR EOS 后只能继续填充 EOS")
            if eos_index < 1 or body[eos_index - 1] != self.tokens.target_close:
                raise GhrEvalError("GHR EOS 前必须是 </TARGET_POI>")
            logical_codes = self._decode_code_tokens(body[: eos_index - 1])
            if self.trie.lookup(logical_codes) < 0:
                raise GhrEvalError(f"完整 identifier 不在冻结语料库中：{logical_codes}")
            return [self.tokens.eos]
        if body[-1] == self.tokens.target_close:
            logical_codes = self._decode_code_tokens(body[:-1])
            if self.trie.lookup(logical_codes) < 0:
                raise GhrEvalError(f"完整 identifier 不在冻结语料库中：{logical_codes}")
            return [self.tokens.eos]
        if self.tokens.target_close in body or self.tokens.eos in body:
            raise GhrEvalError("GHR identifier 内部出现结构 Token")
        if len(body) > self.tokens.maximum_identifier_length:
            raise GhrEvalError("GHR identifier Prefix 超过最大长度")
        return self._allowed_after_codes(body)

    def __call__(self, _batch_id: int, input_ids: Any) -> list[int]:
        generated = input_ids[self.prompt_width :]
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        return self.allowed_next(generated)


def _validate_aligned_artifact(
    identifier_dir: Path,
    artifacts: Mapping[str, Any],
    filename: str,
) -> Path:
    spec = artifacts.get(filename)
    if not isinstance(spec, Mapping) or spec.get("path") != filename:
        raise GhrEvalError(f"aligned manifest 缺少产物：{filename}")
    path = identifier_dir / filename
    if not path.is_file() or sha256_file(path) != spec.get("sha256"):
        raise GhrEvalError(f"aligned 产物文件或 SHA256 不一致：{filename}")
    return path


def _load_aligned_collision_id_index(
    identifier_dir: Path,
) -> tuple[GhrIdIndex, dict[str, Any]]:
    manifest_path = identifier_dir / "manifest.json"
    manifest = _load_json(manifest_path, "aligned identifier manifest")
    if (
        manifest.get("schema_version") != ALIGNED_COLLISION_SCHEMA_VERSION
        or manifest.get("status") != "completed"
    ):
        raise GhrEvalError("aligned identifier manifest 状态或版本无效")
    protocol = manifest.get("protocol")
    artifacts = manifest.get("artifacts")
    if not isinstance(protocol, Mapping) or not isinstance(artifacts, Mapping):
        raise GhrEvalError("aligned identifier manifest 缺少 protocol/artifacts")
    if (protocol.get("base_layers"), protocol.get("suffix_layers")) != (3, 2):
        raise GhrEvalError("aligned identifier 必须是冻结三层加两层后缀")

    codes_path = _validate_aligned_artifact(
        identifier_dir, artifacts, "identifier_codes.npy"
    )
    mapping_path = _validate_aligned_artifact(
        identifier_dir, artifacts, "poi_identifier_mapping.parquet"
    )
    metrics_path = _validate_aligned_artifact(identifier_dir, artifacts, "metrics.json")
    metrics = _load_json(metrics_path, "aligned identifier metrics")
    base_metrics = metrics.get("base_tiger")
    identifier_metrics = metrics.get("identifier")
    if not isinstance(base_metrics, Mapping) or not isinstance(
        identifier_metrics, Mapping
    ):
        raise GhrEvalError("aligned metrics 缺少 base_tiger/identifier")
    row_count = int(base_metrics.get("poi_count", -1))
    if (
        row_count <= 0
        or identifier_metrics.get("layers") != 5
        or identifier_metrics.get("distinct_count") != row_count
        or identifier_metrics.get("within_bucket_duplicate_pair_count") != 0
    ):
        raise GhrEvalError("aligned identifier 唯一性指标未通过")

    raw_codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
    if raw_codes.shape != (row_count, 5) or raw_codes.dtype != np.int32:
        raise GhrEvalError("aligned identifier_codes shape/dtype 不一致")
    logical_codes = _aligned_codes_to_logical(raw_codes)
    lengths = np.full(row_count, 5, dtype=np.uint8)

    parquet_file = pq.ParquetFile(mapping_path)
    columns = ("poi_id", "s1", "s2", "s3", "r1", "r2")
    if parquet_file.metadata.num_rows != row_count or not set(columns).issubset(
        parquet_file.schema_arrow.names
    ):
        raise GhrEvalError("aligned mapping 行数或字段不符合契约")
    poi_ids = np.empty(row_count, dtype=np.int64)
    offset = 0
    for group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(group_index, columns=list(columns))
        values = table.to_pydict()
        end = offset + table.num_rows
        try:
            poi_ids[offset:end] = np.asarray(values["poi_id"], dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as error:
            raise GhrEvalError("aligned mapping poi_id 不是 int64 数字字符串") from error
        mapping_codes = np.column_stack(
            [np.asarray(values[name], dtype=np.int32) for name in columns[1:]]
        )
        if not np.array_equal(mapping_codes, raw_codes[offset:end]):
            raise GhrEvalError("aligned mapping 与 identifier_codes 不一致")
        offset = end
    if offset != row_count or np.unique(poi_ids).size != row_count:
        raise GhrEvalError("aligned mapping 未完整扫描或 poi_id 不唯一")

    index = GhrIdIndex(codes=logical_codes, lengths=lengths, poi_ids=poi_ids)
    return index, {
        "identifier_manifest": str(manifest_path),
        "identifier_manifest_sha256": sha256_file(manifest_path),
        "mapping": str(mapping_path),
        "mapping_sha256": artifacts["poi_identifier_mapping.parquet"]["sha256"],
        "metrics": str(metrics_path),
        "metrics_sha256": artifacts["metrics.json"]["sha256"],
        "variant": "tiger_aligned_collision_32x32_v1",
        "format": "fixed_five_layer_aligned_collision",
        "rows": index.row_count,
        "minimum_identifier_length": 5,
        "maximum_identifier_length": 5,
        "token_capacities": list(ALIGNED_COLLISION_CAPACITIES),
    }


def load_ghr_id_index(identifier_dir: Path) -> tuple[GhrIdIndex, dict[str, Any]]:
    """Load a frozen variable-length or fixed aligned-collision GHR index."""

    identifier_dir = identifier_dir.resolve()
    manifest_path = identifier_dir / "ghr_id_manifest.json"
    if not manifest_path.is_file():
        aligned_manifest = identifier_dir / "manifest.json"
        if aligned_manifest.is_file():
            return _load_aligned_collision_id_index(identifier_dir)
        raise GhrEvalError(f"GHR identifier manifest 不存在：{manifest_path}")
    manifest = _load_json(manifest_path, "GHR identifier manifest")
    if manifest.get("schema_version") != "ghr-sid-identifier-v1":
        raise GhrEvalError("GHR identifier schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise GhrEvalError("GHR identifier 状态不是 completed")
    identifier = manifest.get("identifier")
    outputs = manifest.get("outputs")
    if not isinstance(identifier, Mapping) or not isinstance(outputs, Mapping):
        raise GhrEvalError("GHR manifest 缺少 identifier/outputs")
    if identifier.get("variable_length") is not True:
        raise GhrEvalError("GHR identifier 必须是变长序列")
    arrays = outputs.get("arrays")
    vocabulary = outputs.get("vocabulary")
    if not isinstance(arrays, Mapping) or not isinstance(vocabulary, Mapping):
        raise GhrEvalError("GHR manifest 缺少 arrays/vocabulary")

    loaded: dict[str, np.ndarray] = {}
    for name in ("identifier_token_codes", "identifier_lengths", "poi_ids"):
        spec = arrays.get(name)
        if not isinstance(spec, Mapping):
            raise GhrEvalError(f"GHR manifest 缺少数组：{name}")
        path = identifier_dir / str(spec.get("path", ""))
        if not path.is_file() or sha256_file(path) != spec.get("sha256"):
            raise GhrEvalError(f"GHR 数组文件或 SHA256 不一致：{name}")
        loaded[name] = np.load(path, mmap_mode="r", allow_pickle=False)

    vocabulary_path = identifier_dir / str(vocabulary.get("path", ""))
    if not vocabulary_path.is_file() or sha256_file(vocabulary_path) != vocabulary.get(
        "sha256"
    ):
        raise GhrEvalError("GHR identifier vocabulary 文件或 SHA256 不一致")
    index = GhrIdIndex(
        codes=loaded["identifier_token_codes"],
        lengths=loaded["identifier_lengths"],
        poi_ids=loaded["poi_ids"],
    )
    expected_rows = int(identifier.get("poi_count", -1))
    if index.row_count != expected_rows:
        raise GhrEvalError("GHR identifier POI 行数不一致")
    return index, {
        "identifier_manifest": str(manifest_path),
        "identifier_manifest_sha256": sha256_file(manifest_path),
        "identifier_vocabulary": str(vocabulary_path),
        "identifier_vocabulary_sha256": vocabulary.get("sha256"),
        "variant": manifest.get("variant"),
        "rows": index.row_count,
        "minimum_identifier_length": int(
            identifier.get("minimum_length_excluding_eos", -1)
        ),
        "maximum_identifier_length": int(
            identifier.get("maximum_length_excluding_eos", -1)
        ),
    }


def load_ghr_token_ids(
    tokenizer_path: Path,
    tokenizer: Any,
    *,
    identifier_dir: Path,
    token_source_path: Path,
    mapping_filename: str = "poi_token_mapping.json",
) -> tuple[GhrTokenIds, dict[str, Any]]:
    """Bind logical GHR vocabulary codes to stable atomic tokenizer IDs."""

    tokenizer_path = tokenizer_path.resolve()
    mapping_path = tokenizer_path / mapping_filename
    mapping = _load_json(mapping_path, "GHR Token mapping")
    token_mapping = mapping.get("tokens")
    if not isinstance(token_mapping, Mapping):
        raise GhrEvalError("GHR Token mapping 缺少 tokens")
    if len(tokenizer) != mapping.get("new_vocab_size"):
        raise GhrEvalError("GHR Tokenizer 词表大小与 mapping 不一致")
    token_source_path = token_source_path.resolve()
    token_source = _load_json(token_source_path, "GHR Special Token source")
    source_tokens = token_source.get("additional_special_tokens")
    if (
        token_source.get("schema_version") != "ghr-map-search-special-tokens-v1"
        or not isinstance(source_tokens, list)
        or len(source_tokens) != token_source.get("token_count")
        or len(source_tokens) != len(set(source_tokens))
    ):
        raise GhrEvalError("GHR Special Token source 契约非法")
    if mapping.get("token_source_sha256") != sha256_file(token_source_path):
        raise GhrEvalError("GHR Token mapping 未绑定当前 SFT Special Token source")
    if set(source_tokens) != set(token_mapping):
        raise GhrEvalError("GHR Special Token source 与 Token mapping 不一致")
    ghr_manifest_path = identifier_dir.resolve() / "ghr_id_manifest.json"
    aligned_manifest_path = identifier_dir.resolve() / "manifest.json"
    vocabulary_path: Path | None = None
    if ghr_manifest_path.is_file():
        identifier_manifest = _load_json(
            ghr_manifest_path, "GHR identifier manifest"
        )
        outputs = identifier_manifest.get("outputs", {})
        vocabulary_spec = outputs.get("vocabulary", {})
        vocabulary_path = identifier_dir.resolve() / str(
            vocabulary_spec.get("path", "")
        )
        vocabulary = _load_json(vocabulary_path, "GHR identifier vocabulary")
        logical_tokens = vocabulary.get("tokens")
        identifier = identifier_manifest.get("identifier", {})
        minimum_identifier_length = int(
            identifier.get("minimum_length_excluding_eos")
        )
        maximum_identifier_length = int(
            identifier.get("maximum_length_excluding_eos")
        )
        identifier_format = "variable_length_ghr"
    elif aligned_manifest_path.is_file():
        identifier_manifest = _load_json(
            aligned_manifest_path, "aligned identifier manifest"
        )
        if (
            identifier_manifest.get("schema_version")
            != ALIGNED_COLLISION_SCHEMA_VERSION
            or identifier_manifest.get("status") != "completed"
        ):
            raise GhrEvalError("aligned identifier manifest 状态或版本无效")
        logical_tokens = list(_aligned_identifier_tokens())
        minimum_identifier_length = 5
        maximum_identifier_length = 5
        identifier_format = "fixed_five_layer_aligned_collision"
    else:
        raise GhrEvalError("GHR identifier manifest 不存在")
    if not isinstance(logical_tokens, list) or not logical_tokens:
        raise GhrEvalError("GHR identifier vocabulary 缺少 tokens")
    if token_source.get("identifier_tokens") != logical_tokens:
        raise GhrEvalError("GHR SFT identifier tokens 未绑定当前 identifier vocabulary")

    def atomic_id(token: str) -> int:
        expected = token_mapping.get(token)
        actual = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if not isinstance(expected, int) or actual != expected or encoded != [expected]:
            raise GhrEvalError(f"GHR Token 不是稳定原子 Token：{token}")
        return expected

    logical_to_token = tuple(atomic_id(str(token)) for token in logical_tokens)
    token_to_logical = {token_id: code for code, token_id in enumerate(logical_to_token)}
    if len(token_to_logical) != len(logical_to_token):
        raise GhrEvalError("GHR identifier Token ID 必须互不重复")
    values = GhrTokenIds(
        target_open=atomic_id("<TARGET_POI>"),
        target_close=atomic_id("</TARGET_POI>"),
        logical_to_token=logical_to_token,
        token_to_logical=token_to_logical,
        eos=int(tokenizer.eos_token_id),
        minimum_identifier_length=minimum_identifier_length,
        maximum_identifier_length=maximum_identifier_length,
    )
    if values.minimum_identifier_length <= 0 or (
        values.maximum_identifier_length < values.minimum_identifier_length
    ):
        raise GhrEvalError("GHR identifier 长度范围非法")
    return values, {
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "token_source_path": str(token_source_path),
        "token_source_sha256": sha256_file(token_source_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "identifier_vocabulary_sha256": (
            sha256_file(vocabulary_path) if vocabulary_path is not None else None
        ),
        "identifier_format": identifier_format,
        "vocab_size": len(tokenizer),
        "logical_identifier_token_count": len(logical_to_token),
        "minimum_identifier_length": values.minimum_identifier_length,
        "maximum_identifier_length": values.maximum_identifier_length,
        "maximum_sequence_length": values.maximum_sequence_length,
    }


def _parse_target_content(
    content: str,
    *,
    tokenizer: Any,
    tokens: GhrTokenIds,
) -> tuple[int, ...]:
    encoded = tokenizer.encode(content, add_special_tokens=False)
    if len(encoded) < 2 or encoded[0] != tokens.target_open or encoded[-1] != tokens.target_close:
        raise GhrEvalError("Assistant content 不是严格 GHR Target 格式")
    identifier_ids = encoded[1:-1]
    if not (
        tokens.minimum_identifier_length
        <= len(identifier_ids)
        <= tokens.maximum_identifier_length
    ):
        raise GhrEvalError("Assistant GHR identifier 长度越界")
    try:
        return tuple(tokens.token_to_logical[int(value)] for value in identifier_ids)
    except KeyError as error:
        raise GhrEvalError("Assistant GHR identifier 包含非法 Token") from error


def encode_ghr_record(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    template: Any,
    tokens: GhrTokenIds,
    cutoff_len: int,
    split: str = "valid",
) -> GhrExample:
    """Validate one GHR sample and recreate its leakage-free training prompt."""

    messages = record.get("messages")
    if (
        record.get("split") != split
        or not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise GhrEvalError("GHR 评测样本必须是指定 split 的 user+assistant")
    user_content = messages[0].get("content")
    target_content = messages[1].get("content")
    if not isinstance(user_content, str) or not isinstance(target_content, str):
        raise GhrEvalError("GHR Messages content 必须是字符串")
    target_codes = _parse_target_content(
        target_content, tokenizer=tokenizer, tokens=tokens
    )
    expected_key = record.get("target_ghr_id_key")
    actual_key = target_content[len("<TARGET_POI>") : -len("</TARGET_POI>")]
    if expected_key != actual_key:
        raise GhrEvalError("target_ghr_id_key 与 Assistant content 不一致")
    try:
        prompt_ids, target_ids = encode_prompt_like_training(
            tokenizer=tokenizer,
            template=template,
            user_content=user_content,
            target_content=target_content,
            cutoff_len=cutoff_len,
        )
    except GenerativeEvalError as error:
        raise GhrEvalError(str(error)) from error
    if tuple(target_ids) != tuple(tokenizer.encode(target_content, add_special_tokens=False)):
        raise GhrEvalError("GHR Target 编码与训练模板不一致")
    if target_content in tokenizer.decode(prompt_ids, skip_special_tokens=False):
        raise GhrEvalError("GHR Prompt 泄露目标 identifier")
    sample_id = record.get("sample_id")
    target_poi_id = record.get("target_poi_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise GhrEvalError("sample_id 必须是非空字符串")
    if not isinstance(target_poi_id, str) or not target_poi_id:
        raise GhrEvalError("target_poi_id 必须是非空字符串")
    return GhrExample(
        sample_id=sample_id,
        target_poi_id=target_poi_id,
        target_codes=target_codes,
        prompt_ids=tuple(int(value) for value in prompt_ids),
    )


def parse_generated_candidate(
    sequence: Sequence[int],
    score: float,
    *,
    tokens: GhrTokenIds,
    index: GhrIdIndex,
) -> GhrCandidate:
    """Parse one unconstrained beam while retaining invalid ranked slots."""

    values = [int(value) for value in sequence]
    if tokens.eos not in values:
        return GhrCandidate(None, None, float(score), "missing_eos")
    eos_index = values.index(tokens.eos)
    generated = values[: eos_index + 1]
    if not (
        tokens.minimum_identifier_length + 3
        <= len(generated)
        <= tokens.maximum_sequence_length
    ):
        return GhrCandidate(None, None, float(score), "invalid_length")
    if generated[0] != tokens.target_open or generated[-2] != tokens.target_close:
        return GhrCandidate(None, None, float(score), "invalid_structure")
    try:
        codes = tuple(tokens.token_to_logical[value] for value in generated[1:-2])
    except KeyError:
        return GhrCandidate(None, None, float(score), "invalid_identifier_token")
    row = index.lookup(codes)
    if row < 0:
        return GhrCandidate(codes, None, float(score), "identifier_not_in_corpus")
    return GhrCandidate(codes, row, float(score), None)

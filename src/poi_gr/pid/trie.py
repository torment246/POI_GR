"""Compact integer Trie for globally unique Final PID sequences."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from transformers import AutoTokenizer


TRIE_SCHEMA_VERSION = "final-pid-trie-v1"
GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
PID_ORDERS = ("gid_sid", "sid_gid")
TRIE_FILENAMES = (
    "child_offsets.npy",
    "child_token_ids.npy",
    "child_node_ids.npy",
    "terminal_node_ids.npy",
    "terminal_poi_rows.npy",
)


class PidTrieError(ValueError):
    """Raised when Final PID or Trie inputs violate the declared contract."""


@dataclass(frozen=True)
class FinalPidInput:
    mapping_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    codes_path: Path
    codes: np.ndarray
    poi_count: int


@dataclass(frozen=True)
class PidTokenIds:
    gid: np.ndarray
    sid: tuple[np.ndarray, np.ndarray, np.ndarray]
    dedup: np.ndarray
    eos: int


@dataclass(frozen=True)
class TrieBuildResult:
    manifest_path: Path
    manifest: dict[str, Any]


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise PidTrieError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PidTrieError(f"{name} JSON 解析失败：{path}") from error
    if not isinstance(value, dict):
        raise PidTrieError(f"{name} 必须是 JSON object：{path}")
    return value


def _manifest_output_spec(
    manifest: Mapping[str, Any], filename: str
) -> Mapping[str, Any]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PidTrieError("Final PID manifest 缺少 outputs")
    spec = outputs.get(filename)
    if not isinstance(spec, Mapping):
        raise PidTrieError(f"Final PID manifest 缺少 outputs.{filename}")
    return spec


def load_final_pid_input(
    mapping_path: Path,
    manifest_path: Path,
    *,
    verify_hashes: bool = True,
) -> FinalPidInput:
    """Load and validate the fixed-width Final PID array and mapping metadata."""

    mapping_path = mapping_path.resolve()
    manifest_path = manifest_path.resolve()
    manifest = _load_json_object(manifest_path, "Final PID manifest")
    if manifest.get("schema_version") != "dedup-pid-v1":
        raise PidTrieError("Final PID schema_version 必须是 dedup-pid-v1")
    if manifest.get("status") != "completed":
        raise PidTrieError("Final PID manifest 状态不是 completed")
    poi_count = manifest.get("poi_count")
    if isinstance(poi_count, bool) or not isinstance(poi_count, int) or poi_count <= 0:
        raise PidTrieError("Final PID manifest.poi_count 必须是正整数")

    definition = manifest.get("final_pid_definition")
    if not isinstance(definition, Mapping):
        raise PidTrieError("Final PID manifest 缺少 final_pid_definition")
    if definition.get("base_order") != [
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "S1",
        "S2",
        "S3",
    ]:
        raise PidTrieError("Final PID base_order 必须为 G1..G6,S1..S3")
    if definition.get("singleton_length") != 9:
        raise PidTrieError("Final PID singleton_length 必须为 9")
    if definition.get("colliding_length") != 10:
        raise PidTrieError("Final PID colliding_length 必须为 10")
    if definition.get("fixed_width_singleton_value") != -1:
        raise PidTrieError("Final PID 单例 sentinel 必须为 -1")

    mapping_spec = _manifest_output_spec(manifest, "poi_pid_mapping.parquet")
    declared_mapping = (manifest_path.parent / "poi_pid_mapping.parquet").resolve()
    if mapping_path != declared_mapping:
        raise PidTrieError(
            f"PID mapping 路径与 manifest 不一致：{mapping_path} != {declared_mapping}"
        )
    if not mapping_path.is_file():
        raise PidTrieError(f"PID mapping 不存在：{mapping_path}")
    metadata = pq.read_metadata(mapping_path)
    if metadata.num_rows != poi_count:
        raise PidTrieError(
            f"PID mapping 行数 {metadata.num_rows} 与 poi_count {poi_count} 不一致"
        )
    if mapping_spec.get("shape", [None])[0] != poi_count:
        raise PidTrieError("PID mapping manifest shape 与 poi_count 不一致")

    codes_spec = _manifest_output_spec(manifest, "final_pid_codes.npy")
    codes_path = (manifest_path.parent / "final_pid_codes.npy").resolve()
    if not codes_path.is_file():
        raise PidTrieError(f"final_pid_codes.npy 不存在：{codes_path}")
    try:
        codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PidTrieError(f"Final PID NPY 读取失败：{codes_path}") from error
    if codes.shape != (poi_count, 10) or codes.dtype != np.int32:
        raise PidTrieError(
            f"Final PID 必须是 [{poi_count},10] int32，实际为 "
            f"{codes.shape} {codes.dtype}"
        )
    if codes_spec.get("shape") != [poi_count, 10]:
        raise PidTrieError("Final PID manifest shape 与实际定义不一致")
    if codes_spec.get("dtype") != "int32":
        raise PidTrieError("Final PID manifest dtype 必须为 int32")

    base = codes[:, :9]
    if np.any(base[:, :6] < 0) or np.any(base[:, :6] >= 32):
        raise PidTrieError("G1..G6 Token 必须位于 [0,31]")
    if np.any(base[:, 6:] < 0) or np.any(base[:, 6:] >= 1024):
        raise PidTrieError("S1..S3 Token 必须位于 [0,1023]")
    dedup = codes[:, 9]
    if np.any(dedup < -1) or np.any(dedup >= 512):
        raise PidTrieError("Dedup Code 必须为 -1 或 [0,511]")

    if verify_hashes:
        expected_mapping_hash = mapping_spec.get("sha256")
        if sha256_file(mapping_path) != expected_mapping_hash:
            raise PidTrieError("PID mapping SHA256 与 manifest 不一致")
        expected_codes_hash = codes_spec.get("sha256")
        if sha256_file(codes_path) != expected_codes_hash:
            raise PidTrieError("Final PID NPY SHA256 与 manifest 不一致")

    poi_column = pq.read_table(mapping_path, columns=["poi_id"])["poi_id"]
    if poi_column.null_count:
        raise PidTrieError("PID mapping 的 poi_id 不允许为空")
    if pc.count_distinct(poi_column).as_py() != poi_count:
        raise PidTrieError("PID mapping 的 poi_id 必须全局唯一")

    return FinalPidInput(
        mapping_path=mapping_path,
        manifest_path=manifest_path,
        manifest=manifest,
        codes_path=codes_path,
        codes=codes,
        poi_count=poi_count,
    )


def load_pid_token_ids(
    tokenizer_path: Path,
    *,
    mapping_filename: str = "poi_token_mapping.json",
) -> tuple[PidTokenIds, dict[str, Any]]:
    """Validate the trained tokenizer and return dense PID-code lookup tables."""

    tokenizer_path = tokenizer_path.resolve()
    mapping_path = tokenizer_path / mapping_filename
    mapping = _load_json_object(mapping_path, "POI Token mapping")
    token_mapping = mapping.get("tokens")
    if not isinstance(token_mapping, Mapping):
        raise PidTrieError("poi_token_mapping.json 的 tokens 必须是 JSON object")
    if "<D_-1>" in token_mapping:
        raise PidTrieError("Token 表不得包含 <D_-1>")

    sid_sizes = []
    for level in (1, 2, 3):
        codes = {token for token in token_mapping if token.startswith(f"<S{level}_")}
        size = len(codes)
        if size not in (512, 1024) or codes != {f"<S{level}_{i}>" for i in range(size)}:
            raise PidTrieError(f"S{level} 必须包含完整连续的 512 或 1024 个 SID Token")
        sid_sizes.append(size)
    required_pid_tokens = {
        *(f"<G_{char}>" for char in GEOHASH_ALPHABET),
        *(f"<S{level}_{code}>" for level, size in enumerate(sid_sizes, 1) for code in range(size)),
        *(f"<D_{code}>" for code in range(512)),
    }
    missing = required_pid_tokens.difference(token_mapping)
    if missing:
        example = sorted(missing)[0]
        raise PidTrieError(f"poi_token_mapping.json 缺少 PID Token：{example}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if len(tokenizer) != mapping.get("new_vocab_size"):
        raise PidTrieError("Tokenizer 词表大小与 poi_token_mapping.json 不一致")

    def token_id(token: str) -> int:
        expected = token_mapping.get(token)
        actual = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if not isinstance(expected, int) or actual != expected or encoded != [expected]:
            raise PidTrieError(f"Token 不是稳定原子 Token：{token}")
        return expected

    gid = np.asarray(
        [token_id(f"<G_{char}>") for char in GEOHASH_ALPHABET], dtype=np.int32
    )
    sid = tuple(
        np.asarray(
            [token_id(f"<S{level}_{code}>") for code in range(sid_sizes[level - 1])],
            dtype=np.int32,
        )
        for level in (1, 2, 3)
    )
    dedup = np.asarray(
        [token_id(f"<D_{code}>") for code in range(512)],
        dtype=np.int32,
    )
    if tokenizer.eos_token_id is None:
        raise PidTrieError("Tokenizer 缺少 eos_token_id")
    all_pid_ids = np.concatenate((gid, *sid, dedup))
    if np.unique(all_pid_ids).size != all_pid_ids.size:
        raise PidTrieError("GID、SID、Dedup Token ID 必须互不重复")
    for name, values in (
        ("GID", gid),
        ("S1", sid[0]),
        ("S2", sid[1]),
        ("S3", sid[2]),
        ("Dedup", dedup),
    ):
        if np.any(values[1:] <= values[:-1]):
            raise PidTrieError(f"{name} Token ID 必须按 Code 严格递增")
        if np.any(np.diff(values) != 1):
            raise PidTrieError(f"{name} Token ID 必须是连续区间")
    metadata = {
        "path": str(tokenizer_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "mapping_token_count": len(token_mapping),
        "pid_token_count": len(required_pid_tokens),
        "vocab_size": len(tokenizer),
        "eos_token_id": int(tokenizer.eos_token_id),
    }
    return (
        PidTokenIds(
            gid=gid,
            sid=(sid[0], sid[1], sid[2]),
            dedup=dedup,
            eos=int(tokenizer.eos_token_id),
        ),
        metadata,
    )


def _base_code_columns(pid_order: str) -> tuple[int, ...]:
    if pid_order == "gid_sid":
        return tuple(range(9))
    if pid_order == "sid_gid":
        return (6, 7, 8, 0, 1, 2, 3, 4, 5)
    raise PidTrieError(f"不支持的 PID 顺序：{pid_order}")


def _depth_tokens(
    codes: np.ndarray,
    sorted_rows: np.ndarray,
    depth: int,
    token_ids: PidTokenIds,
    *,
    pid_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_columns = _base_code_columns(pid_order)
    if depth < 9:
        positions = np.arange(sorted_rows.size, dtype=np.int64)
        column = base_columns[depth]
        if column < 6:
            tokens = token_ids.gid[codes[sorted_rows, column]]
        else:
            tokens = token_ids.sid[column - 6][codes[sorted_rows, column]]
    elif depth == 9:
        positions = np.arange(sorted_rows.size, dtype=np.int64)
        dedup = codes[sorted_rows, 9]
        tokens = np.full(sorted_rows.size, token_ids.eos, dtype=np.int32)
        has_dedup = dedup >= 0
        tokens[has_dedup] = token_ids.dedup[dedup[has_dedup]]
    elif depth == 10:
        positions = np.flatnonzero(codes[sorted_rows, 9] >= 0)
        tokens = np.full(positions.size, token_ids.eos, dtype=np.int32)
    else:
        raise AssertionError(f"unexpected Trie depth: {depth}")
    return positions, np.asarray(tokens, dtype=np.int32)


def build_compact_trie_arrays(
    final_pid_codes: np.ndarray,
    token_ids: PidTokenIds,
    *,
    pid_order: str = "gid_sid",
    progress: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """Build breadth-first CSR Trie arrays without Python objects per node."""

    progress = progress or (lambda _: None)
    codes = np.asarray(final_pid_codes)
    if codes.ndim != 2 or codes.shape[1] != 10:
        raise PidTrieError(f"Final PID shape 必须是 [N,10]，实际为 {codes.shape}")
    if codes.dtype != np.int32:
        raise PidTrieError(f"Final PID dtype 必须是 int32，实际为 {codes.dtype}")
    if codes.shape[0] == 0:
        raise PidTrieError("Final PID 不允许为空")
    if np.any(codes[:, :9] < 0):
        raise PidTrieError("Final PID 前九层不允许负 Token")
    if np.any(codes[:, 9] < -1):
        raise PidTrieError("Dedup Code 仅允许 -1 或非负整数")

    base_columns = _base_code_columns(pid_order)
    progress(f"正在按 {pid_order} 对 Final PID 做稳定字典序排序……")
    logical_columns = (*base_columns, 9)
    sort_keys = tuple(codes[:, column] for column in reversed(logical_columns))
    sorted_rows = np.lexsort(sort_keys)
    row_nodes = np.zeros(codes.shape[0], dtype=np.int32)
    next_node_id = 1
    edge_parent_parts: list[np.ndarray] = []
    edge_token_parts: list[np.ndarray] = []
    edge_child_parts: list[np.ndarray] = []
    terminal_node_parts: list[np.ndarray] = []
    terminal_row_parts: list[np.ndarray] = []

    for depth in range(11):
        positions, tokens = _depth_tokens(
            codes,
            sorted_rows,
            depth,
            token_ids,
            pid_order=pid_order,
        )
        parents = row_nodes[positions]
        if tokens.size == 0:
            raise PidTrieError(f"Trie 第 {depth + 1} 层没有有效路径")
        boundaries = np.empty(tokens.size, dtype=bool)
        boundaries[0] = True
        boundaries[1:] = (parents[1:] != parents[:-1]) | (tokens[1:] != tokens[:-1])
        group_index = np.cumsum(boundaries, dtype=np.int64) - 1
        child_count = int(group_index[-1]) + 1
        if next_node_id + child_count > np.iinfo(np.int32).max:
            raise PidTrieError("Trie 节点数超过 int32 容量")
        child_nodes = (group_index + next_node_id).astype(np.int32)
        boundary_children = np.arange(
            next_node_id,
            next_node_id + child_count,
            dtype=np.int32,
        )
        edge_parent_parts.append(parents[boundaries].astype(np.int32, copy=True))
        edge_token_parts.append(tokens[boundaries].astype(np.int32, copy=True))
        edge_child_parts.append(boundary_children)
        row_nodes[positions] = child_nodes
        next_node_id += child_count

        if depth == 9:
            singleton_positions = positions[codes[sorted_rows, 9] == -1]
            terminal_node_parts.append(row_nodes[singleton_positions].copy())
            terminal_row_parts.append(sorted_rows[singleton_positions].astype(np.int32))
        elif depth == 10:
            terminal_node_parts.append(row_nodes[positions].copy())
            terminal_row_parts.append(sorted_rows[positions].astype(np.int32))

        progress(
            f"Trie 深度 {depth + 1}/11：新增 {child_count:,} 节点，"
            f"累计 {next_node_id:,} 节点"
        )

    edge_parents = np.concatenate(edge_parent_parts)
    child_token_ids = np.concatenate(edge_token_parts)
    child_node_ids = np.concatenate(edge_child_parts)
    terminal_node_ids = np.concatenate(terminal_node_parts)
    terminal_poi_rows = np.concatenate(terminal_row_parts)
    node_count = next_node_id

    if edge_parents.size != node_count - 1:
        raise PidTrieError("Trie 边数必须等于节点数减一")
    if np.any(edge_parents[1:] < edge_parents[:-1]):
        raise PidTrieError("Trie CSR 构建时父节点顺序异常")
    if terminal_node_ids.size != codes.shape[0]:
        raise PidTrieError("Trie 叶子数与 POI 数不一致，Final PID 可能重复")
    if np.unique(terminal_node_ids).size != codes.shape[0]:
        raise PidTrieError("一个 Trie 叶子映射了多个 POI")
    if np.unique(terminal_poi_rows).size != codes.shape[0]:
        raise PidTrieError("Trie terminal POI row 映射不唯一")

    child_counts = np.bincount(edge_parents, minlength=node_count)
    child_offsets = np.empty(node_count + 1, dtype=np.int64)
    child_offsets[0] = 0
    np.cumsum(child_counts, out=child_offsets[1:])
    terminal_order = np.argsort(terminal_node_ids, kind="stable")
    terminal_node_ids = terminal_node_ids[terminal_order]
    terminal_poi_rows = terminal_poi_rows[terminal_order]

    return {
        "child_offsets": child_offsets,
        "child_token_ids": child_token_ids,
        "child_node_ids": child_node_ids,
        "terminal_node_ids": terminal_node_ids,
        "terminal_poi_rows": terminal_poi_rows,
    }


class CompactPidTrie:
    """Read-only CSR Trie used by constrained generation."""

    def __init__(
        self,
        child_offsets: np.ndarray,
        child_token_ids: np.ndarray,
        child_node_ids: np.ndarray,
        terminal_node_ids: np.ndarray,
        terminal_poi_rows: np.ndarray,
    ) -> None:
        self.child_offsets = child_offsets
        self.child_token_ids = child_token_ids
        self.child_node_ids = child_node_ids
        self.terminal_node_ids = terminal_node_ids
        self.terminal_poi_rows = terminal_poi_rows
        self.node_count = int(child_offsets.size - 1)
        self.edge_count = int(child_token_ids.size)
        self.leaf_count = int(terminal_node_ids.size)
        self.memory_bytes = sum(
            int(array.nbytes)
            for array in (
                child_offsets,
                child_token_ids,
                child_node_ids,
                terminal_node_ids,
                terminal_poi_rows,
            )
        )
        self._validate()

    def _validate(self) -> None:
        if self.child_offsets.ndim != 1 or self.child_offsets.dtype != np.int64:
            raise PidTrieError("child_offsets 必须是一维 int64")
        for name, array in (
            ("child_token_ids", self.child_token_ids),
            ("child_node_ids", self.child_node_ids),
            ("terminal_node_ids", self.terminal_node_ids),
            ("terminal_poi_rows", self.terminal_poi_rows),
        ):
            if array.ndim != 1 or array.dtype != np.int32:
                raise PidTrieError(f"{name} 必须是一维 int32")
        if self.child_node_ids.size != self.edge_count:
            raise PidTrieError("child Token 与 child node 数量不一致")
        if int(self.child_offsets[0]) != 0:
            raise PidTrieError("Trie 根 offset 必须为 0")
        if int(self.child_offsets[-1]) != self.edge_count:
            raise PidTrieError("Trie 最后 offset 必须等于边数")
        if np.any(self.child_offsets[1:] < self.child_offsets[:-1]):
            raise PidTrieError("Trie offsets 必须单调非降")
        if np.any(self.child_node_ids <= 0) or np.any(
            self.child_node_ids >= self.node_count
        ):
            raise PidTrieError("Trie child node 超出范围")
        if self.terminal_node_ids.size != self.terminal_poi_rows.size:
            raise PidTrieError("Trie terminal node 与 POI row 数量不一致")
        if np.any(self.terminal_node_ids[1:] <= self.terminal_node_ids[:-1]):
            raise PidTrieError("Trie terminal node 必须严格递增")

    @classmethod
    def load(cls, trie_dir: Path, *, mmap: bool = True) -> "CompactPidTrie":
        trie_dir = trie_dir.resolve()
        manifest = _load_json_object(trie_dir / "trie_manifest.json", "Trie manifest")
        if manifest.get("schema_version") != TRIE_SCHEMA_VERSION:
            raise PidTrieError("Trie manifest schema_version 不受支持")
        if manifest.get("status") != "completed":
            raise PidTrieError("Trie manifest 状态不是 completed")
        arrays: dict[str, np.ndarray] = {}
        for filename in TRIE_FILENAMES:
            path = trie_dir / filename
            if not path.is_file():
                raise PidTrieError(f"Trie 数据文件缺失：{path}")
            spec = manifest.get("files", {}).get(filename)
            if not isinstance(spec, Mapping):
                raise PidTrieError(f"Trie manifest 缺少文件定义：{filename}")
            array = np.load(
                path,
                mmap_mode="r" if mmap else None,
                allow_pickle=False,
            )
            if list(array.shape) != spec.get("shape") or str(array.dtype) != spec.get(
                "dtype"
            ):
                raise PidTrieError(f"Trie 文件 shape/dtype 不一致：{filename}")
            arrays[filename.removesuffix(".npy")] = array
        trie = cls(**arrays)
        if (
            trie.node_count != manifest.get("node_count")
            or trie.edge_count != manifest.get("edge_count")
            or trie.leaf_count != manifest.get("leaf_count")
        ):
            raise PidTrieError("Trie 实际规模与 manifest 不一致")
        return trie

    def children(self, node_id: int) -> np.ndarray:
        if node_id < 0 or node_id >= self.node_count:
            return np.empty(0, dtype=np.int32)
        start = int(self.child_offsets[node_id])
        end = int(self.child_offsets[node_id + 1])
        return self.child_token_ids[start:end]

    def advance(self, node_id: int, token_id: int) -> int:
        if node_id < 0 or node_id >= self.node_count:
            return -1
        start = int(self.child_offsets[node_id])
        end = int(self.child_offsets[node_id + 1])
        tokens = self.child_token_ids[start:end]
        index = int(np.searchsorted(tokens, token_id))
        if index >= tokens.size or int(tokens[index]) != token_id:
            return -1
        return int(self.child_node_ids[start + index])

    def traverse(self, token_ids: Sequence[int]) -> int:
        node_id = 0
        for token_id in token_ids:
            node_id = self.advance(node_id, int(token_id))
            if node_id < 0:
                return -1
        return node_id

    def terminal_poi_row(self, node_id: int) -> int:
        index = int(np.searchsorted(self.terminal_node_ids, node_id))
        if (
            index >= self.terminal_node_ids.size
            or int(self.terminal_node_ids[index]) != node_id
        ):
            return -1
        return int(self.terminal_poi_rows[index])

    def lookup(self, token_ids: Sequence[int]) -> int:
        return self.terminal_poi_row(self.traverse(token_ids))


class TriePrefixConstraint:
    """Transformers-compatible prefix callback over generated-only tokens."""

    def __init__(
        self,
        trie: CompactPidTrie,
        prompt_width: int,
        eos_token_id: int,
    ) -> None:
        if prompt_width <= 0:
            raise PidTrieError("prompt_width 必须为正整数")
        self.trie = trie
        self.prompt_width = prompt_width
        self.eos_token_id = eos_token_id

    def __call__(self, _batch_id: int, input_ids: Any) -> list[int]:
        generated = input_ids[self.prompt_width :]
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        node_id = self.trie.traverse(generated)
        if node_id < 0:
            raise PidTrieError(f"生成序列离开 Final PID Trie：{generated}")
        children = self.trie.children(node_id).astype(int).tolist()
        if children:
            return children
        if self.trie.terminal_poi_row(node_id) >= 0:
            return [self.eos_token_id]
        raise PidTrieError(f"非 terminal Trie 节点没有合法子节点：{node_id}")


class TriePrefilledPrefixConstraint:
    """Trie callback whose per-example prefix is already present in model input."""

    def __init__(
        self,
        trie: CompactPidTrie,
        prompt_width: int,
        eos_token_id: int,
        prefilled_prefixes: Sequence[Sequence[int]],
    ) -> None:
        if prompt_width <= 0:
            raise PidTrieError("prompt_width 必须为正整数")
        if not prefilled_prefixes:
            raise PidTrieError("prefilled_prefixes 不能为空")
        self.trie = trie
        self.prompt_width = prompt_width
        self.eos_token_id = eos_token_id
        self.prefilled_prefixes = tuple(
            tuple(int(token_id) for token_id in prefix)
            for prefix in prefilled_prefixes
        )
        for prefix in self.prefilled_prefixes:
            if trie.traverse(prefix) < 0:
                raise PidTrieError(f"预填 GID Prefix 不在 Final PID Trie：{prefix}")

    def __call__(self, batch_id: int, input_ids: Any) -> list[int]:
        if batch_id < 0 or batch_id >= len(self.prefilled_prefixes):
            raise PidTrieError(f"batch_id 超出预填 Prefix 范围：{batch_id}")
        generated = input_ids[self.prompt_width :]
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        path = (*self.prefilled_prefixes[batch_id], *generated)
        node_id = self.trie.traverse(path)
        if node_id < 0:
            raise PidTrieError(f"生成序列离开预填 Final PID Trie：{path}")
        children = self.trie.children(node_id).astype(int).tolist()
        if children:
            return children
        if self.trie.terminal_poi_row(node_id) >= 0:
            return [self.eos_token_id]
        raise PidTrieError(f"非 terminal Trie 节点没有合法子节点：{node_id}")


def build_pid_trie(
    mapping_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    *,
    pid_order: str = "gid_sid",
    verify_hashes: bool = True,
    progress: Callable[[str], None] | None = None,
) -> TrieBuildResult:
    """Validate inputs, build compact arrays, and persist a deterministic Trie."""

    progress = progress or (lambda _: None)
    started = time.monotonic()
    final_pid = load_final_pid_input(
        mapping_path,
        manifest_path,
        verify_hashes=verify_hashes,
    )
    token_ids, tokenizer_metadata = load_pid_token_ids(tokenizer_path)
    arrays = build_compact_trie_arrays(
        final_pid.codes,
        token_ids,
        pid_order=pid_order,
        progress=progress,
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        path = output_dir / filename
        temporary = output_dir / f".{filename}.tmp"
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        files[filename] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    node_count = int(arrays["child_offsets"].size - 1)
    edge_count = int(arrays["child_token_ids"].size)
    leaf_count = int(arrays["terminal_node_ids"].size)
    load_memory_bytes = sum(int(array.nbytes) for array in arrays.values())
    root_end = int(arrays["child_offsets"][1])
    root_token_ids = arrays["child_token_ids"][:root_end].astype(int).tolist()
    manifest = {
        "schema_version": TRIE_SCHEMA_VERSION,
        "status": "completed",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": time.monotonic() - started,
        "input": {
            "pid_mapping": str(final_pid.mapping_path),
            "pid_mapping_sha256": sha256_file(final_pid.mapping_path),
            "pid_manifest": str(final_pid.manifest_path),
            "pid_manifest_sha256": sha256_file(final_pid.manifest_path),
            "final_pid_codes": str(final_pid.codes_path),
            "final_pid_codes_sha256": sha256_file(final_pid.codes_path),
            "poi_count": final_pid.poi_count,
        },
        "tokenizer": tokenizer_metadata,
        "path_definition": {
            "pid_order": pid_order,
            "singleton": (
                "G1..G6,S1..S3,EOS"
                if pid_order == "gid_sid"
                else "S1..S3,G1..G6,EOS"
            ),
            "dedup": (
                "G1..G6,S1..S3,D,EOS"
                if pid_order == "gid_sid"
                else "S1..S3,G1..G6,D,EOS"
            ),
            "singleton_sentinel": -1,
            "singleton_sentinel_is_token": False,
            "root_token_ids": root_token_ids,
        },
        "node_count": node_count,
        "edge_count": edge_count,
        "leaf_count": leaf_count,
        "file_size_bytes": sum(spec["size_bytes"] for spec in files.values()),
        "load_memory_bytes": load_memory_bytes,
        "files": files,
    }
    manifest_path_out = output_dir / "trie_manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TrieBuildResult(manifest_path=manifest_path_out, manifest=manifest)

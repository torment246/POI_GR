"""Materialize frozen BGE and category assets in active-POI row order."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import yaml

from qg_prqk.artifacts import sha256_file, write_json_atomic


CONFIG_SCHEMA_VERSION = "qg-prqk-active-poi-assets-config-v1"
ASSET_SCHEMA_VERSION = "qg-prqk-active-poi-assets-v1"
ROW_ORDER = "BGE-M3 poi_ids.jsonl order"


class ActivePoiAssetsError(RuntimeError):
    """Raised when active-POI assets violate their frozen contract."""


@dataclass(frozen=True)
class ActivePoiAssetPaths:
    project_root: Path
    active_catalog: Path
    active_catalog_manifest: Path
    active_poi_ids: Path
    source_embeddings: Path
    source_poi_ids: Path
    source_embedding_manifest: Path
    source_category_indices: Path
    source_category_manifest: Path
    output_dir: Path


@dataclass(frozen=True)
class ActivePoiAssetInputs:
    source_rows: int
    active_rows: int
    embedding_dim: int
    embedding_dtype: str
    active_catalog_manifest_sha256: str
    active_poi_ids_sha256: str
    source_poi_ids_sha256: str
    source_embedding_manifest_sha256: str
    source_category_indices_sha256: str
    source_category_manifest_sha256: str
    category_vocab_count: int


@dataclass(frozen=True)
class ActivePoiAssetsConfig:
    source_path: Path
    source_sha256: str
    paths: ActivePoiAssetPaths
    frozen: ActivePoiAssetInputs
    chunk_rows: int

    def signature(self) -> str:
        """Return a deterministic signature for the resolved asset contract."""

        payload = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "source_sha256": self.source_sha256,
            "paths": {
                key: str(value)
                for key, value in self.paths.__dict__.items()
            },
            "frozen": dict(self.frozen.__dict__),
            "chunk_rows": self.chunk_rows,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActivePoiAssetsError(f"{label} 必须是 mapping")
    return value


def _string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ActivePoiAssetsError(f"{label}.{key} 必须是非空字符串")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActivePoiAssetsError(f"{label}.{key} 必须是正整数")
    return value


def _sha256(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _string(mapping, key, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ActivePoiAssetsError(f"{label}.{key} 必须是小写 SHA256")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ActivePoiAssetsError(f"{label} 必须位于 {root}") from error


def load_active_poi_assets_config(path: Path) -> ActivePoiAssetsConfig:
    """Load the strict active-POI asset materialization contract."""

    source_path = path.resolve()
    try:
        raw_bytes = source_path.read_bytes()
        root = _mapping(yaml.safe_load(raw_bytes), "配置根")
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ActivePoiAssetsError(f"配置读取失败：{source_path}") from error
    if root.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ActivePoiAssetsError("schema_version 不匹配")
    paths_raw = _mapping(root.get("paths"), "paths")
    frozen_raw = _mapping(root.get("frozen_inputs"), "frozen_inputs")
    runtime_raw = _mapping(root.get("runtime"), "runtime")
    project_root = _resolve(source_path.parent, _string(paths_raw, "project_root", "paths"))
    paths = ActivePoiAssetPaths(
        project_root=project_root,
        active_catalog=_resolve(project_root, _string(paths_raw, "active_catalog", "paths")),
        active_catalog_manifest=_resolve(
            project_root, _string(paths_raw, "active_catalog_manifest", "paths")
        ),
        active_poi_ids=_resolve(project_root, _string(paths_raw, "active_poi_ids", "paths")),
        source_embeddings=_resolve(
            project_root, _string(paths_raw, "source_embeddings", "paths")
        ),
        source_poi_ids=_resolve(project_root, _string(paths_raw, "source_poi_ids", "paths")),
        source_embedding_manifest=_resolve(
            project_root, _string(paths_raw, "source_embedding_manifest", "paths")
        ),
        source_category_indices=_resolve(
            project_root, _string(paths_raw, "source_category_indices", "paths")
        ),
        source_category_manifest=_resolve(
            project_root, _string(paths_raw, "source_category_manifest", "paths")
        ),
        output_dir=_resolve(project_root, _string(paths_raw, "output_dir", "paths")),
    )
    _inside(paths.output_dir, (project_root / "qg_prqk/outputs").resolve(), "paths.output_dir")
    frozen = ActivePoiAssetInputs(
        source_rows=_positive_int(frozen_raw, "source_rows", "frozen_inputs"),
        active_rows=_positive_int(frozen_raw, "active_rows", "frozen_inputs"),
        embedding_dim=_positive_int(frozen_raw, "embedding_dim", "frozen_inputs"),
        embedding_dtype=_string(frozen_raw, "embedding_dtype", "frozen_inputs"),
        active_catalog_manifest_sha256=_sha256(
            frozen_raw, "active_catalog_manifest_sha256", "frozen_inputs"
        ),
        active_poi_ids_sha256=_sha256(
            frozen_raw, "active_poi_ids_sha256", "frozen_inputs"
        ),
        source_poi_ids_sha256=_sha256(
            frozen_raw, "source_poi_ids_sha256", "frozen_inputs"
        ),
        source_embedding_manifest_sha256=_sha256(
            frozen_raw, "source_embedding_manifest_sha256", "frozen_inputs"
        ),
        source_category_indices_sha256=_sha256(
            frozen_raw, "source_category_indices_sha256", "frozen_inputs"
        ),
        source_category_manifest_sha256=_sha256(
            frozen_raw, "source_category_manifest_sha256", "frozen_inputs"
        ),
        category_vocab_count=_positive_int(
            frozen_raw, "category_vocab_count", "frozen_inputs"
        ),
    )
    if frozen.embedding_dtype != "float16":
        raise ActivePoiAssetsError("embedding_dtype 必须为 float16")
    if frozen.active_rows >= frozen.source_rows:
        raise ActivePoiAssetsError("active_rows 必须小于 source_rows")
    return ActivePoiAssetsConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        paths=paths,
        frozen=frozen,
        chunk_rows=_positive_int(runtime_raw, "chunk_rows", "runtime"),
    )


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActivePoiAssetsError(f"{label} 无法读取：{path}") from error
    return _mapping(payload, label)


def _read_json_id(raw_line: bytes, label: str) -> str:
    try:
        value = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActivePoiAssetsError(f"{label} 不是合法 JSON 字符串") from error
    if not isinstance(value, str) or not value:
        raise ActivePoiAssetsError(f"{label} 不是非空 POI ID")
    return value


def _iter_ids(path: Path, label: str) -> Iterator[str]:
    with path.open("rb", buffering=16 * 1024 * 1024) as handle:
        for row, raw_line in enumerate(handle):
            yield _read_json_id(raw_line, f"{label}:{row + 1}")


def resolve_active_source_rows(
    source_poi_ids: Path,
    active_poi_ids: Path,
    *,
    source_rows: int,
    active_rows: int,
) -> np.ndarray:
    """Resolve an active ID subsequence to monotonic source BGE row indices."""

    active_iterator = _iter_ids(active_poi_ids, "active_poi_ids")
    try:
        expected_active = next(active_iterator)
    except StopIteration as error:
        raise ActivePoiAssetsError("active_poi_ids 为空") from error
    result = np.empty(active_rows, dtype=np.int64)
    matched = 0
    scanned = 0
    previous_active: str | None = None
    for source_row, source_id in enumerate(_iter_ids(source_poi_ids, "source_poi_ids")):
        scanned = source_row + 1
        if source_id != expected_active:
            continue
        if expected_active == previous_active:
            raise ActivePoiAssetsError(f"active POI ID 重复：{expected_active}")
        if matched >= active_rows:
            raise ActivePoiAssetsError("active_poi_ids 行数超过冻结值")
        result[matched] = source_row
        matched += 1
        previous_active = expected_active
        try:
            expected_active = next(active_iterator)
        except StopIteration:
            expected_active = ""
    if scanned != source_rows:
        raise ActivePoiAssetsError(
            f"source_poi_ids 行数 {scanned} != {source_rows}"
        )
    if expected_active:
        raise ActivePoiAssetsError(
            f"active POI 不在冻结 BGE 行序中或顺序错误：{expected_active}"
        )
    try:
        extra_active = next(active_iterator)
    except StopIteration:
        extra_active = None
    if extra_active is not None or matched != active_rows:
        raise ActivePoiAssetsError(
            f"active_poi_ids 行数 {matched} != {active_rows}"
        )
    if matched > 1 and np.any(result[1:] <= result[:-1]):
        raise ActivePoiAssetsError("active source rows 不是严格递增")
    return result


def _validate_inputs(config: ActivePoiAssetsConfig) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    frozen = config.frozen
    paths = config.paths
    expected_hashes = (
        (paths.active_catalog_manifest, frozen.active_catalog_manifest_sha256, "active catalog manifest"),
        (paths.active_poi_ids, frozen.active_poi_ids_sha256, "active poi_ids"),
        (paths.source_poi_ids, frozen.source_poi_ids_sha256, "source poi_ids"),
        (paths.source_embedding_manifest, frozen.source_embedding_manifest_sha256, "source embedding manifest"),
        (paths.source_category_indices, frozen.source_category_indices_sha256, "source category indices"),
        (paths.source_category_manifest, frozen.source_category_manifest_sha256, "source category manifest"),
    )
    for path, expected, label in expected_hashes:
        if not path.is_file() or sha256_file(path) != expected:
            raise ActivePoiAssetsError(f"{label} 缺失或 SHA256 不一致：{path}")
    active_manifest = _load_json(paths.active_catalog_manifest, "active catalog manifest")
    active_output = _mapping(active_manifest.get("output"), "active manifest.output")
    if (
        active_manifest.get("status") != "completed"
        or int(active_output.get("rows", -1)) != frozen.active_rows
        or active_output.get("poi_ids_sha256") != frozen.active_poi_ids_sha256
        or active_manifest.get("validation", {}).get("all_active_ids_found_in_full_catalog") is not True
    ):
        raise ActivePoiAssetsError("active catalog manifest 合同不一致")
    embedding_manifest = _load_json(paths.source_embedding_manifest, "source embedding manifest")
    embedding_output = _mapping(embedding_manifest.get("output"), "source embedding.output")
    embedding_input = _mapping(embedding_manifest.get("input"), "source embedding.input")
    if (
        embedding_manifest.get("status") != "completed"
        or embedding_output.get("shape") != [frozen.source_rows, frozen.embedding_dim]
        or embedding_output.get("dtype") != frozen.embedding_dtype
        or int(embedding_input.get("total_rows", -1)) != frozen.source_rows
    ):
        raise ActivePoiAssetsError("source embedding manifest 合同不一致")
    source_embeddings = np.load(paths.source_embeddings, mmap_mode="r", allow_pickle=False)
    if source_embeddings.shape != (frozen.source_rows, frozen.embedding_dim) or str(source_embeddings.dtype) != frozen.embedding_dtype:
        raise ActivePoiAssetsError("source embeddings shape/dtype 不一致")
    del source_embeddings
    source_categories = np.load(paths.source_category_indices, mmap_mode="r", allow_pickle=False)
    if source_categories.shape != (frozen.source_rows,) or source_categories.dtype != np.int32:
        raise ActivePoiAssetsError("source category_indices 必须是 int32[source_rows]")
    del source_categories
    category_manifest = _load_json(paths.source_category_manifest, "source category manifest")
    category_stats = _mapping(category_manifest.get("stats"), "source category.stats")
    if (
        category_manifest.get("status") != "completed"
        or category_manifest.get("row_order") != ROW_ORDER
        or int(category_stats.get("output_rows", -1)) != frozen.source_rows
        or int(category_stats.get("category_vocab_count", -1)) != frozen.category_vocab_count
        or int(category_stats.get("invalid_category_count", -1)) != 0
    ):
        raise ActivePoiAssetsError("source category manifest 合同不一致")
    return active_manifest, embedding_manifest, category_manifest


def _catalog_sources(active_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_sources = active_manifest.get("input", {}).get("poi_sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ActivePoiAssetsError("active catalog manifest 缺少 POI 输出分片")
    sources: list[dict[str, Any]] = []
    for item in raw_sources:
        source = _mapping(item, "active catalog poi_source")
        sources.append(
            {
                "name": str(source["output_name"]),
                "rows_scanned": int(source["output_rows"]),
                "bytes_scanned": int(source["output_bytes"]),
                "sha256_scanned": str(source["output_sha256"]),
            }
        )
    return sources


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_active_arrays_sequentially(
    *,
    source_embeddings: np.ndarray,
    source_categories: np.ndarray,
    source_rows: np.ndarray,
    embeddings_path: Path,
    active_categories: np.ndarray,
    chunk_rows: int,
) -> None:
    """Copy selected rows while reading the source NPY only in contiguous blocks."""

    active_start = 0
    source_count = len(source_embeddings)
    shape = (len(source_rows), source_embeddings.shape[1])
    header = {
        "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
        "fortran_order": False,
        "shape": shape,
    }
    with embeddings_path.open("wb", buffering=16 * 1024 * 1024) as handle:
        np.lib.format.write_array_header_2_0(handle, header)
        for source_start in range(0, source_count, chunk_rows):
            source_stop = min(source_start + chunk_rows, source_count)
            active_stop = int(
                np.searchsorted(source_rows, source_stop, side="left")
            )
            if active_stop > active_start:
                local_rows = (
                    source_rows[active_start:active_stop] - source_start
                )
                embedding_block = np.array(
                    source_embeddings[source_start:source_stop],
                    dtype=np.float16,
                    copy=True,
                    order="C",
                )
                if not np.isfinite(embedding_block).all():
                    raise ActivePoiAssetsError(
                        "source embedding 含非有限值："
                        f"rows {source_start}:{source_stop}"
                    )
                selected_embeddings = np.ascontiguousarray(
                    embedding_block[local_rows], dtype=np.float16
                )
                handle.write(memoryview(selected_embeddings).cast("B"))
                category_block = np.asarray(
                    source_categories[source_start:source_stop], dtype=np.int32
                )
                active_categories[active_start:active_stop] = category_block[
                    local_rows
                ]
                active_start = active_stop
            if (
                source_stop % (chunk_rows * 16) == 0
                or source_stop == source_count
            ):
                print(
                    "active asset 顺序切片："
                    f"source {source_stop:,}/{source_count:,}，"
                    f"active {active_start:,}/{len(source_rows):,}",
                    flush=True,
                )
    if active_start != len(source_rows):
        raise ActivePoiAssetsError("顺序切片未覆盖全部 active source rows")


def _validate_active_arrays_sequentially(
    *,
    source_embeddings: np.ndarray,
    source_categories: np.ndarray,
    source_rows: np.ndarray,
    active_embeddings: np.ndarray,
    active_categories: np.ndarray,
    chunk_rows: int,
) -> None:
    """Compare active arrays while retaining only one contiguous source block."""

    active_start = 0
    source_count = len(source_embeddings)
    for source_start in range(0, source_count, chunk_rows):
        source_stop = min(source_start + chunk_rows, source_count)
        active_stop = int(np.searchsorted(source_rows, source_stop, side="left"))
        if active_stop == active_start:
            continue
        local_rows = source_rows[active_start:active_stop] - source_start
        embedding_block = np.array(
            source_embeddings[source_start:source_stop],
            dtype=np.float16,
            copy=True,
            order="C",
        )
        expected_embeddings = embedding_block[local_rows]
        if not np.array_equal(
            active_embeddings[active_start:active_stop], expected_embeddings
        ):
            raise ActivePoiAssetsError(
                f"active embeddings 与冻结来源不一致：{active_start}:{active_stop}"
            )
        expected_categories = np.asarray(
            source_categories[source_start:source_stop], dtype=np.int32
        )[local_rows]
        if not np.array_equal(
            active_categories[active_start:active_stop], expected_categories
        ):
            raise ActivePoiAssetsError(
                f"active categories 与冻结来源不一致：{active_start}:{active_stop}"
            )
        active_start = active_stop
    if active_start != len(source_rows):
        raise ActivePoiAssetsError("独立校验未覆盖全部 active source rows")


def build_active_poi_assets(config: ActivePoiAssetsConfig) -> Mapping[str, Any]:
    """Build active-row BGE/category arrays and reproducibility manifests."""

    output_dir = config.paths.output_dir
    if output_dir.exists():
        raise ActivePoiAssetsError(f"输出已存在，拒绝覆盖：{output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.building")
    if staging.exists():
        raise ActivePoiAssetsError(f"暂存目录已存在，拒绝覆盖：{staging}")
    active_manifest, source_embedding_manifest, _ = _validate_inputs(config)
    started_at = _utc_now()
    source_rows = resolve_active_source_rows(
        config.paths.source_poi_ids,
        config.paths.active_poi_ids,
        source_rows=config.frozen.source_rows,
        active_rows=config.frozen.active_rows,
    )
    staging.mkdir(parents=True)
    try:
        source_rows_path = staging / "source_rows.npy"
        np.save(source_rows_path, source_rows, allow_pickle=False)
        shutil.copyfile(config.paths.active_poi_ids, staging / "poi_ids.jsonl")

        source_embeddings = np.load(
            config.paths.source_embeddings, mmap_mode="r", allow_pickle=False
        )
        embeddings_path = staging / "embeddings.npy"
        source_categories = np.load(
            config.paths.source_category_indices, mmap_mode="r", allow_pickle=False
        )
        active_categories = np.empty(config.frozen.active_rows, dtype=np.int32)
        _copy_active_arrays_sequentially(
            source_embeddings=source_embeddings,
            source_categories=source_categories,
            source_rows=source_rows,
            embeddings_path=embeddings_path,
            active_categories=active_categories,
            chunk_rows=config.chunk_rows,
        )
        del source_embeddings, source_categories
        category_path = staging / "category_indices.npy"
        np.save(category_path, active_categories, allow_pickle=False)
        unique_categories = np.unique(active_categories)
        if (
            len(unique_categories) == 0
            or int(unique_categories[0]) < 0
            or int(unique_categories[-1]) >= config.frozen.category_vocab_count
        ):
            raise ActivePoiAssetsError("active POI category index 超出冻结词表范围")
        missing_category_indices = sorted(
            set(range(config.frozen.category_vocab_count))
            - set(int(value) for value in unique_categories)
        )

        ids_sha = sha256_file(staging / "poi_ids.jsonl")
        embeddings_sha = sha256_file(embeddings_path)
        rows_sha = sha256_file(source_rows_path)
        category_sha = sha256_file(category_path)
        catalog_sources = _catalog_sources(active_manifest)
        source_model = dict(_mapping(source_embedding_manifest.get("model"), "source embedding.model"))
        embedding_manifest = {
            "schema_version": "qg-prqk-active-bge-subset-v1",
            "status": "completed",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "job_name": "qg_prqk_beijing_active_poi_bge_m3_v1",
            "input": {
                "dir": str(config.paths.active_catalog),
                "file_pattern": "part-*.json",
                "fingerprint": hashlib.sha256(
                    f"{config.frozen.active_catalog_manifest_sha256}:{config.frozen.source_embedding_manifest_sha256}".encode("ascii")
                ).hexdigest(),
                "id_field": "poi_id",
                "text_field": "text",
                "total_rows": config.frozen.active_rows,
                "sources": catalog_sources,
            },
            "model": source_model,
            "output": {
                "dir": str(output_dir),
                "dtype": config.frozen.embedding_dtype,
                "embeddings": "embeddings.npy",
                "poi_ids": "poi_ids.jsonl",
                "shape": [config.frozen.active_rows, config.frozen.embedding_dim],
                "embeddings_sha256": embeddings_sha,
                "poi_ids_sha256": ids_sha,
            },
            "derivation": {
                "operation": "exact_row_subset_without_reencoding",
                "source_embedding_manifest_sha256": config.frozen.source_embedding_manifest_sha256,
                "source_poi_ids_sha256": config.frozen.source_poi_ids_sha256,
                "source_rows_sha256": rows_sha,
                "records_preserved_verbatim": active_manifest.get("output", {}).get("records_preserved_verbatim"),
            },
        }
        write_json_atomic(staging / "embedding_manifest.json", embedding_manifest)
        embedding_manifest_sha = sha256_file(staging / "embedding_manifest.json")
        category_manifest = {
            "schema_version": "qg-prqk-active-category-input-v1",
            "status": "completed",
            "method": "QG-PRQK active POI category row subset",
            "catalog_filter": {
                "definition": active_manifest.get("selection", {}).get("definition"),
                "manifest_sha256": config.frozen.active_catalog_manifest_sha256,
            },
            "inputs": {
                "source_category_manifest": str(config.paths.source_category_manifest),
                "source_category_manifest_sha256": config.frozen.source_category_manifest_sha256,
                "active_embedding_manifest_sha256": embedding_manifest_sha,
            },
            "outputs": {"category_indices": "category_indices.npy"},
            "row_order": ROW_ORDER,
            "sha256": {"category_indices": category_sha},
            "stats": {
                "category_vocab_count": config.frozen.category_vocab_count,
                "observed_category_count": len(unique_categories),
                "missing_category_indices": missing_category_indices,
                "invalid_category_count": 0,
                "output_rows": config.frozen.active_rows,
                "source_embedding_rows": config.frozen.active_rows,
                "source_feature_rows": config.frozen.active_rows,
            },
        }
        write_json_atomic(staging / "category_manifest.json", category_manifest)
        category_manifest_sha = sha256_file(staging / "category_manifest.json")
        asset_manifest = {
            "schema_version": ASSET_SCHEMA_VERSION,
            "status": "completed",
            "created_at": _utc_now(),
            "config": {
                "path": str(config.source_path),
                "sha256": config.source_sha256,
                "resolved_signature": config.signature(),
            },
            "row_order": {
                "definition": "active poi_ids strict subsequence of frozen full BGE poi_ids",
                "source_rows": config.frozen.source_rows,
                "active_rows": config.frozen.active_rows,
                "source_rows_sha256": rows_sha,
                "strictly_increasing": True,
            },
            "artifacts": {
                "embeddings.npy": {"sha256": embeddings_sha, "shape": [config.frozen.active_rows, config.frozen.embedding_dim], "dtype": config.frozen.embedding_dtype},
                "poi_ids.jsonl": {"sha256": ids_sha, "rows": config.frozen.active_rows},
                "source_rows.npy": {"sha256": rows_sha, "shape": [config.frozen.active_rows], "dtype": "int64"},
                "category_indices.npy": {"sha256": category_sha, "shape": [config.frozen.active_rows], "dtype": "int32"},
                "embedding_manifest.json": {"sha256": embedding_manifest_sha},
                "category_manifest.json": {"sha256": category_manifest_sha},
            },
            "validation": {
                "active_catalog_manifest_sha256": config.frozen.active_catalog_manifest_sha256,
                "active_poi_ids_sha256": config.frozen.active_poi_ids_sha256,
                "source_embedding_manifest_sha256": config.frozen.source_embedding_manifest_sha256,
                "source_category_manifest_sha256": config.frozen.source_category_manifest_sha256,
                "embedding_values_copied_exactly": True,
                "category_values_copied_exactly": True,
                "all_active_category_indices_valid": True,
                "observed_category_count": len(unique_categories),
                "missing_category_indices": missing_category_indices,
            },
        }
        write_json_atomic(staging / "manifest.json", asset_manifest)
        (staging / "_SUCCESS").touch()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _load_json(output_dir / "manifest.json", "active asset manifest")


def validate_active_poi_assets(config: ActivePoiAssetsConfig) -> Mapping[str, Any]:
    """Independently validate active-row arrays, hashes, and source alignment."""

    output_dir = config.paths.output_dir
    if not (output_dir / "_SUCCESS").is_file():
        raise ActivePoiAssetsError("active POI assets 缺少 _SUCCESS")
    manifest = _load_json(output_dir / "manifest.json", "active asset manifest")
    if (
        manifest.get("schema_version") != ASSET_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("config", {}).get("resolved_signature") != config.signature()
    ):
        raise ActivePoiAssetsError("active asset manifest 与当前配置不一致")
    _validate_inputs(config)
    artifacts = _mapping(manifest.get("artifacts"), "active asset artifacts")
    for name, metadata in artifacts.items():
        item = _mapping(metadata, f"artifacts.{name}")
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ActivePoiAssetsError(f"active asset 缺失或 SHA256 不一致：{path}")
    expected_rows = resolve_active_source_rows(
        config.paths.source_poi_ids,
        config.paths.active_poi_ids,
        source_rows=config.frozen.source_rows,
        active_rows=config.frozen.active_rows,
    )
    actual_rows = np.load(output_dir / "source_rows.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(actual_rows, expected_rows):
        raise ActivePoiAssetsError("source_rows 与冻结 ID 行序不一致")
    source_embeddings = np.load(config.paths.source_embeddings, mmap_mode="r", allow_pickle=False)
    active_embeddings = np.load(output_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    source_categories = np.load(config.paths.source_category_indices, mmap_mode="r", allow_pickle=False)
    active_categories = np.load(output_dir / "category_indices.npy", mmap_mode="r", allow_pickle=False)
    _validate_active_arrays_sequentially(
        source_embeddings=source_embeddings,
        source_categories=source_categories,
        source_rows=actual_rows,
        active_embeddings=active_embeddings,
        active_categories=active_categories,
        chunk_rows=config.chunk_rows,
    )
    return manifest

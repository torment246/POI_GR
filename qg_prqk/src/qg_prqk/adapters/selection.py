"""Full-data adapter selection, holdout evaluation, and final retraining."""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.hard_negative_mining import (
    NegativeCandidate,
    lexical_metadata_candidates,
    local_geo_candidates,
    merge_negative_sources,
)
from qg_prqk.adapters.retrieval import ExactPoiIndex
from qg_prqk.adapters.data import (
    D3Query,
    load_false_negative_ids,
    load_selected_metadata,
    resolve_poi_rows,
    select_d3_queries,
    write_selection,
)
from qg_prqk.adapters.selection_config import AdapterSelectionConfig
from qg_prqk.adapters.gate import (
    SOURCE_CODES,
    _adapt_queries,
    _artifact,
    _csr,
    _event,
    _ranks_and_margins,
    _reasonable_positive_rows,
    _validate_frozen_inputs,
    _without_self,
    validate_query_adapter_gate,
)
from qg_prqk.adapters.cache import build_d3_query_cache
from qg_prqk.adapters.model import (
    ResidualQueryAdapter,
    build_in_batch_valid_mask,
    build_negative_valid_mask,
    weighted_info_nce,
)


SCHEMA_VERSION = "qg-prqk-p3a-full-v1"
HOLDOUT_SCHEMA_VERSION = "qg-prqk-p3a-full-holdout-v1"
SELECT_SCHEMA_VERSION = "qg-prqk-p3a-full-select-v1"
FINAL_SCHEMA_VERSION = "qg-prqk-p3a-full-final-v1"
CHECKPOINT_SCHEMA_VERSION = "qg-prqk-query-adapter-v1"
FULL_BUNDLE_SCHEMA_VERSION = "qg-prqk-p3a-full-training-data-v1"
FULL_BUNDLE_EMBEDDING_BATCH_ROWS = 2_048
FULL_INITIALIZATION_POLICY = (
    "User-confirmed 2026-09-05: fresh seeded initialization with the Gate "
    "architecture and zero-output residual; SELECT and FINAL reload the same "
    "saved initial state. Historical Gate initial weights were not saved "
    "and are not claimed to be reproduced."
)


class AdapterSelectionError(RuntimeError):
    """Raised when P3A-FULL violates its frozen execution protocol."""


@dataclass
class TrainingTensors:
    """Memory-mapped full-D3 arrays shared by two fresh training runs."""

    raw_queries: np.ndarray
    positives: np.ndarray
    negatives: np.ndarray
    negative_valid_mask: torch.Tensor
    query_weights: np.ndarray
    target_rows_cpu: torch.Tensor
    reasonable_positive_rows: list[np.ndarray]
    query_ids: tuple[int, ...]


def partition_full_d3(
    queries: Sequence[D3Query],
    gate_query_ids: Sequence[int],
    *,
    gate_exclusion_rows: int,
    holdout_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic holdout and SELECT-train row indices."""

    if len(queries) <= gate_exclusion_rows + holdout_rows:
        raise AdapterSelectionError("D3 数量不足以划分 Gate 排除集和 internal holdout")
    if len(gate_query_ids) != gate_exclusion_rows:
        raise AdapterSelectionError("50k Gate Query ID 数量与冻结配置不一致")
    query_ids = [query.query_id for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise AdapterSelectionError("D3 Query ID 不唯一")
    ordered = sorted(queries, key=lambda value: (value.selection_hash, value.query_id))
    if list(queries) != ordered:
        raise AdapterSelectionError("D3 Query 未按冻结 selection hash 排序")
    if query_ids[:gate_exclusion_rows] != [int(value) for value in gate_query_ids]:
        raise AdapterSelectionError("当前 50k Gate 不是全量 D3 哈希排序的固定前缀")
    holdout = np.arange(
        gate_exclusion_rows,
        gate_exclusion_rows + holdout_rows,
        dtype=np.int64,
    )
    train = np.concatenate(
        (
            np.arange(0, gate_exclusion_rows, dtype=np.int64),
            np.arange(gate_exclusion_rows + holdout_rows, len(queries), dtype=np.int64),
        )
    )
    if len(np.intersect1d(train, holdout)) or len(train) + len(holdout) != len(queries):
        raise AdapterSelectionError("P3A-FULL SELECT/holdout 划分不守恒")
    return holdout, train


def full_retrieval_metrics(
    ranks: np.ndarray,
    difficult_mask: np.ndarray,
    raw_embeddings: np.ndarray,
    view_embeddings: np.ndarray,
) -> dict[str, Any]:
    """Compute the approved exact retrieval and Query-drift metrics."""

    ranks = np.asarray(ranks)
    difficult_mask = np.asarray(difficult_mask, dtype=np.bool_)
    raw = np.asarray(raw_embeddings, dtype=np.float32)
    view = np.asarray(view_embeddings, dtype=np.float32)
    if ranks.ndim != 1 or len(ranks) == 0:
        raise AdapterSelectionError("评测 rank 必须是非空一维数组")
    if difficult_mask.shape != ranks.shape or not bool(difficult_mask.any()):
        raise AdapterSelectionError("困难子集必须是非空且与 rank 对齐")
    if raw.shape != view.shape or raw.ndim != 2 or len(raw) != len(ranks):
        raise AdapterSelectionError("Query drift embedding shape 非法")
    raw_norms = np.linalg.norm(raw, axis=1, keepdims=True)
    view_norms = np.linalg.norm(view, axis=1, keepdims=True)
    if np.any(raw_norms <= 0) or np.any(view_norms <= 0):
        raise AdapterSelectionError("Query drift embedding 包含零范数")
    cosine = np.einsum(
        "bd,bd->b",
        raw / raw_norms,
        view / view_norms,
    )
    drift = 1.0 - np.clip(cosine, -1.0, 1.0)
    reciprocal = np.where(ranks <= 10, 1.0 / ranks.astype(np.float64), 0.0)
    difficult_ranks = ranks[difficult_mask]
    return {
        "rows": int(len(ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "recall_at_20": float(np.mean(ranks <= 20)),
        "mrr_at_10": float(np.mean(reciprocal)),
        "difficult_rows": int(difficult_mask.sum()),
        "difficult_recall_at_10": float(np.mean(difficult_ranks <= 10)),
        "query_embedding_cosine_drift": {
            "definition": "1 - cosine(raw_bge_query, evaluated_query_view)",
            "mean": float(np.mean(drift)),
            "p50": float(np.quantile(drift, 0.50)),
            "p90": float(np.quantile(drift, 0.90)),
            "p99": float(np.quantile(drift, 0.99)),
            "max": float(np.max(drift)),
        },
    }


def select_best_epoch(epoch_records: Sequence[Mapping[str, Any]]) -> int:
    """Select by R@10, then R@1, difficult R@10, then earliest epoch."""

    if not epoch_records:
        raise AdapterSelectionError("P3A-FULL-SELECT 没有 epoch 记录")

    def key(record: Mapping[str, Any]) -> tuple[float, float, float, int]:
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            raise AdapterSelectionError("SELECT epoch 缺少 metrics")
        try:
            epoch = int(record["epoch"])
            return (
                float(metrics["recall_at_10"]),
                float(metrics["recall_at_1"]),
                float(metrics["difficult_recall_at_10"]),
                -epoch,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AdapterSelectionError("SELECT epoch metrics 非法") from error

    return int(max(epoch_records, key=key)["epoch"])


def _read_gate_query_ids(path: Path) -> list[int]:
    table = pq.read_table(path, columns=["query_id"])
    return [int(value) for value in table.column(0).to_pylist()]


def _save_checkpoint(
    path: Path,
    model: ResidualQueryAdapter,
    config: AdapterSelectionConfig,
    *,
    role: str,
    trained_epochs: int,
    initial_state_sha256: str,
) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "embedding_dim": model.embedding_dim,
            "adapter_config": asdict(config.base.adapter),
            "p3a_config_signature": config.base.signature(),
            "p3a_full_config_signature": config.signature(),
            "role": role,
            "trained_epochs": trained_epochs,
            "initial_state_sha256": initial_state_sha256,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        temporary,
    )
    os.replace(temporary, path)


def _load_adapter_checkpoint(
    path: Path,
    config: AdapterSelectionConfig,
    *,
    device: torch.device,
) -> ResidualQueryAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("adapter_config") != asdict(config.base.adapter)
        or int(payload.get("embedding_dim", -1))
        != config.base.category_config.frozen.poi_embedding_dim
    ):
        raise AdapterSelectionError(f"Adapter checkpoint 合同不一致：{path}")
    model = ResidualQueryAdapter(
        int(payload["embedding_dim"]),
        config.base.adapter.bottleneck,
        residual_scale=config.base.adapter.residual_scale,
        dropout=config.base.adapter.dropout,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _save_initial_state(
    path: Path,
    config: AdapterSelectionConfig,
    embedding_dim: int,
) -> str:
    torch.manual_seed(config.base.seed)
    model = ResidualQueryAdapter(
        embedding_dim,
        config.base.adapter.bottleneck,
        residual_scale=config.base.adapter.residual_scale,
        dropout=config.base.adapter.dropout,
    )
    if any(bool(value.any()) for value in (model.up.weight, model.up.bias)):
        raise AdapterSelectionError("fresh Adapter 的 zero-output residual 初始化失效")
    temporary = path.with_name(f".{path.name}.writing")
    torch.save(
        {
            "schema_version": "qg-prqk-query-adapter-initial-state-v1",
            "embedding_dim": embedding_dim,
            "adapter_config": asdict(config.base.adapter),
            "seed": config.base.seed,
            "initialization": FULL_INITIALIZATION_POLICY,
            "historical_gate_initial_weights_reproduced": False,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        temporary,
    )
    os.replace(temporary, path)
    return sha256_file(path)


def _fresh_adapter(
    initial_path: Path,
    config: AdapterSelectionConfig,
    *,
    device: torch.device,
) -> ResidualQueryAdapter:
    payload = torch.load(initial_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version")
        != "qg-prqk-query-adapter-initial-state-v1"
        or payload.get("adapter_config") != asdict(config.base.adapter)
        or int(payload.get("seed", -1)) != config.base.seed
    ):
        raise AdapterSelectionError("P3A-FULL initial Adapter state 合同不一致")
    model = ResidualQueryAdapter(
        int(payload["embedding_dim"]),
        config.base.adapter.bottleneck,
        residual_scale=config.base.adapter.residual_scale,
        dropout=config.base.adapter.dropout,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    return model


def _build_full_bundle(
    *,
    config: AdapterSelectionConfig,
    queries: Sequence[D3Query],
    raw_queries_path: Path,
    poi_embeddings: np.ndarray,
    target_rows: np.ndarray,
    false_negative_rows: Sequence[Sequence[int]],
    positive_ann_rows: np.ndarray,
    positive_ann_scores: np.ndarray,
    metadata: Mapping[int, Any],
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Build the frozen Gate bundle contract with bounded resident memory."""

    started = time.perf_counter()
    if output_dir.exists():
        raise AdapterSelectionError(f"P3A-FULL training bundle 已存在：{output_dir}")
    temporary_dir = output_dir.with_name(f".{output_dir.name}.writing")
    rows = len(queries)
    fixed_count = (
        config.base.adapter.semantic_ann_negatives
        + config.base.adapter.lexical_metadata_negatives
        + config.base.adapter.local_geo_negatives
    )
    candidate_path = temporary_dir / "candidate_poi_rows.npy"
    source_path = temporary_dir / "negative_source_codes.npy"
    reuse_candidates = temporary_dir.is_dir()
    if reuse_candidates:
        try:
            candidate_rows = np.load(candidate_path, mmap_mode="r+")
            source_codes = np.load(source_path, mmap_mode="r+")
        except (OSError, ValueError) as error:
            raise AdapterSelectionError("中断 bundle 的候选 memmap 无法复用") from error
        if (
            candidate_rows.shape != (rows, fixed_count)
            or candidate_rows.dtype != np.int64
            or source_codes.shape != (rows, fixed_count)
            or source_codes.dtype != np.int8
            or int(candidate_rows.min()) < 0
            or int(candidate_rows.max()) >= len(poi_embeddings)
            or not np.isin(source_codes, tuple(SOURCE_CODES.values())).all()
            or np.any(candidate_rows == target_rows[:, None])
        ):
            raise AdapterSelectionError("中断 bundle 的候选 memmap 合同不一致")
        source_totals = {
            name: int(np.count_nonzero(source_codes == code))
            for name, code in SOURCE_CODES.items()
        }
        _event(
            log_path,
            "negative_mining_reused_from_interrupted_stream",
            rows=rows,
            source_counts=source_totals,
        )
    else:
        temporary_dir.mkdir()
        candidate_rows = np.lib.format.open_memmap(
            candidate_path,
            mode="w+",
            dtype=np.int64,
            shape=(rows, fixed_count),
        )
        source_codes = np.lib.format.open_memmap(
            source_path,
            mode="w+",
            dtype=np.int8,
            shape=(rows, fixed_count),
        )
        source_totals = {name: 0 for name in SOURCE_CODES}
        for index, target_row in enumerate(target_rows):
            forbidden = false_negative_rows[index]
            semantic = [
                NegativeCandidate(int(row), "semantic_ann", float(score))
                for row, score in zip(
                    positive_ann_rows[index],
                    positive_ann_scores[index],
                    strict=True,
                )
            ]
            pool = [metadata[int(row)] for row in positive_ann_rows[index]]
            target = metadata[int(target_row)]
            lexical = lexical_metadata_candidates(
                target,
                pool,
                top_k=config.base.ann.top_k,
                false_negative_rows=forbidden,
            )
            local_geo = local_geo_candidates(
                target,
                pool,
                top_k=config.base.ann.top_k,
                false_negative_rows=forbidden,
            )
            merged = merge_negative_sources(
                target_row=int(target_row),
                false_negative_rows=forbidden,
                in_batch=(),
                semantic_ann=semantic,
                lexical_metadata=lexical,
                local_geo=local_geo,
                semantic_budget=config.base.adapter.semantic_ann_negatives,
                lexical_budget=config.base.adapter.lexical_metadata_negatives,
                local_geo_budget=config.base.adapter.local_geo_negatives,
            )
            if len(merged) != fixed_count:
                raise AdapterSelectionError(
                    f"Query {queries[index].query_id} 只得到 "
                    f"{len(merged)}/{fixed_count} 个固定负例"
                )
            candidate_rows[index] = [
                candidate.poi_row for candidate in merged
            ]
            source_codes[index] = [
                SOURCE_CODES[candidate.source] for candidate in merged
            ]
            for candidate in merged:
                source_totals[candidate.source] += 1
            if (index + 1) % 5_000 == 0:
                candidate_rows.flush()
                source_codes.flush()
                _event(
                    log_path,
                    "negative_mining_progress",
                    next_row=index + 1,
                    rows=rows,
                )
        candidate_rows.flush()
        source_codes.flush()

    embedding_dim = int(poi_embeddings.shape[1])
    requested_rows = np.concatenate(
        (target_rows, np.asarray(candidate_rows).reshape(-1))
    )
    unique_rows, inverse = np.unique(requested_rows, return_inverse=True)
    gathered = np.empty((len(unique_rows), embedding_dim), dtype=np.float16)
    for start in range(0, len(poi_embeddings), 32_768):
        stop = min(start + 32_768, len(poi_embeddings))
        left = int(np.searchsorted(unique_rows, start, side="left"))
        right = int(np.searchsorted(unique_rows, stop, side="left"))
        if left != right:
            block = np.asarray(poi_embeddings[start:stop], dtype=np.float16)
            gathered[left:right] = block[unique_rows[left:right] - start]
        if stop % 262_144 == 0 or stop == len(poi_embeddings):
            _event(
                log_path,
                "bundle_unique_embedding_scan_progress",
                next_poi_row=stop,
                poi_rows=len(poi_embeddings),
                unique_embedding_rows=len(unique_rows),
            )
    target_inverse = inverse[:rows]
    candidate_inverse = inverse[rows:].reshape(rows, fixed_count)
    del requested_rows, inverse
    positive_path = temporary_dir / "positive_embeddings.npy"
    negative_path = temporary_dir / "negative_embeddings.npy"
    positive_header = {
        "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
        "fortran_order": False,
        "shape": (rows, embedding_dim),
    }
    negative_header = {
        "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
        "fortran_order": False,
        "shape": (rows, fixed_count, embedding_dim),
    }
    with (
        positive_path.open("wb", buffering=16 * 1024 * 1024) as positive_handle,
        negative_path.open("wb", buffering=16 * 1024 * 1024) as negative_handle,
    ):
        np.lib.format.write_array_header_2_0(positive_handle, positive_header)
        np.lib.format.write_array_header_2_0(negative_handle, negative_header)
        for start in range(0, rows, FULL_BUNDLE_EMBEDDING_BATCH_ROWS):
            stop = min(start + FULL_BUNDLE_EMBEDDING_BATCH_ROWS, rows)
            positive_block = np.ascontiguousarray(
                gathered[target_inverse[start:stop]], dtype=np.float16
            )
            negative_block = np.ascontiguousarray(
                gathered[candidate_inverse[start:stop]], dtype=np.float16
            )
            if not np.isfinite(positive_block).all() or not np.isfinite(
                negative_block
            ).all():
                raise AdapterSelectionError("P3A-FULL training embedding 包含 NaN/Inf")
            positive_handle.write(memoryview(positive_block).cast("B"))
            negative_handle.write(memoryview(negative_block).cast("B"))
            _event(
                log_path,
                "bundle_embedding_progress",
                next_row=stop,
                rows=rows,
            )
    del gathered

    offsets, fn_rows = _csr(false_negative_rows)
    query_weights = np.asarray(
        [query.query_weight for query in queries], dtype=np.float32
    )
    query_ids = np.asarray([query.query_id for query in queries], dtype=np.int64)
    if not np.isfinite(query_weights).all() or np.any(query_weights < 0):
        raise AdapterSelectionError("P3A-FULL query weight 必须有限且非负")
    arrays = {
        "target_poi_rows.npy": np.asarray(target_rows, dtype=np.int64),
        "false_negative_offsets.npy": offsets,
        "false_negative_rows.npy": fn_rows,
        "query_weights.npy": query_weights,
        "query_ids.npy": query_ids,
    }
    for name, values in arrays.items():
        np.save(temporary_dir / name, values)
    del candidate_rows, source_codes
    gc.collect()

    artifact_paths = sorted(temporary_dir.glob("*.npy"))
    manifest = {
        "schema_version": FULL_BUNDLE_SCHEMA_VERSION,
        "status": "completed",
        "built_at": utc_now(),
        "rows": rows,
        "embedding_dim": embedding_dim,
        "fixed_negatives_per_query": fixed_count,
        "raw_queries": _artifact(raw_queries_path),
        "arrays": {path.name: _artifact(path) for path in artifact_paths},
        "source_counts": source_totals,
        "false_negative_rows": len(fn_rows),
        "storage": "directory_npy_buffered_sequential_write_mmap_read",
        "materialization_batch_rows": FULL_BUNDLE_EMBEDDING_BATCH_ROWS,
        "unique_embedding_rows_gathered": len(unique_rows),
        "embedding_gather_order": "single_sequential_poi_scan_then_inverse_restore",
        "candidate_mining_reused_after_interruption": reuse_candidates,
        "valid_and_test_read": False,
    }
    write_json_atomic(temporary_dir / "manifest.json", manifest)
    (temporary_dir / "_SUCCESS").touch()
    os.replace(temporary_dir, output_dir)
    return {
        "schema_version": FULL_BUNDLE_SCHEMA_VERSION,
        "rows": rows,
        "fixed_negatives_per_query": fixed_count,
        "source_counts": source_totals,
        "false_negative_rows": len(fn_rows),
        "embedding_storage": "directory_npy_buffered_sequential_write_mmap_read",
        "embedding_materialization_batch_rows": FULL_BUNDLE_EMBEDDING_BATCH_ROWS,
        "unique_embedding_rows_gathered": len(unique_rows),
        "embedding_gather_order": "single_sequential_poi_scan_then_inverse_restore",
        "candidate_mining_reused_after_interruption": reuse_candidates,
        "algorithm_changed_from_gate": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _load_full_bundle(
    bundle_dir: Path,
    raw_queries_path: Path,
) -> dict[str, np.ndarray]:
    """Load the full-scale bundle as read-only memory maps."""

    manifest_path = bundle_dir / "manifest.json"
    if not (bundle_dir / "_SUCCESS").is_file() or not manifest_path.is_file():
        raise AdapterSelectionError("P3A-FULL training bundle 未完成")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterSelectionError("P3A-FULL training bundle manifest 非法") from error
    if (
        manifest.get("schema_version") != FULL_BUNDLE_SCHEMA_VERSION
        or manifest.get("status") != "completed"
    ):
        raise AdapterSelectionError("P3A-FULL training bundle schema/status 不一致")
    names = (
        "positive_embeddings",
        "negative_embeddings",
        "candidate_poi_rows",
        "negative_source_codes",
        "target_poi_rows",
        "false_negative_offsets",
        "false_negative_rows",
        "query_weights",
        "query_ids",
    )
    try:
        bundle = {
            name: np.load(bundle_dir / f"{name}.npy", mmap_mode="r")
            for name in names
        }
        bundle["raw_queries"] = np.load(raw_queries_path, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise AdapterSelectionError("P3A-FULL training bundle array 无法 mmap") from error
    rows = int(manifest["rows"])
    embedding_dim = int(manifest["embedding_dim"])
    fixed_count = int(manifest["fixed_negatives_per_query"])
    expected_shapes = {
        "raw_queries": (rows, embedding_dim),
        "positive_embeddings": (rows, embedding_dim),
        "negative_embeddings": (rows, fixed_count, embedding_dim),
        "candidate_poi_rows": (rows, fixed_count),
        "negative_source_codes": (rows, fixed_count),
        "target_poi_rows": (rows,),
        "false_negative_offsets": (rows + 1,),
        "query_weights": (rows,),
        "query_ids": (rows,),
    }
    for name, shape in expected_shapes.items():
        if bundle[name].shape != shape:
            raise AdapterSelectionError(f"P3A-FULL bundle {name} shape 非法")
    false_stop = int(bundle["false_negative_offsets"][-1])
    if bundle["false_negative_rows"].shape != (false_stop,):
        raise AdapterSelectionError("P3A-FULL bundle false-negative CSR 非法")
    return bundle


def _validate_full_bundle(bundle_dir: Path, raw_queries_path: Path) -> None:
    """Rehash every mmap array and validate the completed bundle contract."""

    bundle = _load_full_bundle(bundle_dir, raw_queries_path)
    manifest = json.loads(
        (bundle_dir / "manifest.json").read_text(encoding="utf-8")
    )
    raw_artifact = manifest.get("raw_queries", {})
    if sha256_file(raw_queries_path) != raw_artifact.get("sha256"):
        raise AdapterSelectionError("P3A-FULL bundle Raw Query SHA256 失败")
    artifacts = manifest.get("arrays", {})
    expected_names = {
        f"{name}.npy"
        for name in (
            "positive_embeddings",
            "negative_embeddings",
            "candidate_poi_rows",
            "negative_source_codes",
            "target_poi_rows",
            "false_negative_offsets",
            "false_negative_rows",
            "query_weights",
            "query_ids",
        )
    }
    if set(artifacts) != expected_names:
        raise AdapterSelectionError("P3A-FULL bundle artifact 列表不一致")
    for name, artifact in artifacts.items():
        path = bundle_dir / name
        values = np.load(path, mmap_mode="r")
        if (
            sha256_file(path) != artifact.get("sha256")
            or list(values.shape) != artifact.get("shape")
            or str(values.dtype) != artifact.get("dtype")
        ):
            raise AdapterSelectionError(f"P3A-FULL bundle artifact 校验失败：{name}")
    del bundle


def _prepare_training_tensors(
    bundle_path: Path,
    raw_queries_path: Path,
) -> TrainingTensors:
    bundle = _load_full_bundle(bundle_path, raw_queries_path)
    reasonable = _reasonable_positive_rows(bundle)
    candidate_rows = torch.from_numpy(
        np.asarray(bundle["candidate_poi_rows"], dtype=np.int64).copy()
    )
    target_rows = torch.from_numpy(
        np.asarray(bundle["target_poi_rows"], dtype=np.int64).copy()
    )
    valid_mask = build_negative_valid_mask(candidate_rows, target_rows, reasonable)
    if not bool(valid_mask.any(dim=1).all().item()):
        raise AdapterSelectionError("至少一个 D3 Query 在 false-negative mask 后没有负例")
    query_ids = tuple(int(value) for value in bundle["query_ids"])
    tensors = TrainingTensors(
        raw_queries=bundle["raw_queries"],
        positives=bundle["positive_embeddings"],
        negatives=bundle["negative_embeddings"],
        negative_valid_mask=valid_mask,
        query_weights=bundle["query_weights"],
        target_rows_cpu=target_rows,
        reasonable_positive_rows=reasonable,
        query_ids=query_ids,
    )
    del candidate_rows
    gc.collect()
    return tensors


EpochCallback = Callable[[int, ResidualQueryAdapter, float], None]


def _train_epochs(
    model: ResidualQueryAdapter,
    tensors: TrainingTensors,
    train_indices: np.ndarray,
    config: AdapterSelectionConfig,
    *,
    epochs: int,
    device: torch.device,
    log_path: Path,
    run_role: str,
    epoch_callback: EpochCallback | None = None,
) -> list[float]:
    if epochs <= 0 or not len(train_indices):
        raise AdapterSelectionError("Adapter 训练 epoch/rows 必须为正")
    if len(np.unique(train_indices)) != len(train_indices):
        raise AdapterSelectionError("Adapter 训练 row index 重复")
    torch.manual_seed(config.base.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.base.adapter.learning_rate,
        weight_decay=config.base.adapter.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.base.seed)
    train_tensor = torch.from_numpy(np.asarray(train_indices, dtype=np.int64))
    losses: list[float] = []
    amp_enabled = device.type == "cuda" and config.base.adapter.precision in {
        "bf16",
        "fp16",
    }
    amp_dtype = (
        torch.bfloat16
        if config.base.adapter.precision == "bf16"
        else torch.float16
    )
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = train_tensor[
            torch.randperm(len(train_tensor), generator=generator)
        ]
        weighted_loss = 0.0
        seen = 0
        batches = 0
        for start in range(0, len(permutation), config.base.adapter.batch_size):
            batch_indices = permutation[
                start : start + config.base.adapter.batch_size
            ]
            numpy_indices = batch_indices.numpy()
            batch_targets = tensors.target_rows_cpu[batch_indices]
            batch_reasonable = [
                tensors.reasonable_positive_rows[int(index)]
                for index in batch_indices
            ]
            in_batch_mask = build_in_batch_valid_mask(
                batch_targets,
                batch_reasonable,
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            raw_batch = torch.from_numpy(
                np.asarray(tensors.raw_queries[numpy_indices], dtype=np.float32)
            ).to(device)
            positive_batch = torch.from_numpy(
                np.asarray(tensors.positives[numpy_indices], dtype=np.float32)
            ).to(device)
            negative_batch = torch.from_numpy(
                np.asarray(tensors.negatives[numpy_indices], dtype=np.float32)
            ).to(device)
            valid_batch = tensors.negative_valid_mask[batch_indices].to(device)
            weight_batch = torch.from_numpy(
                np.asarray(tensors.query_weights[numpy_indices], dtype=np.float32)
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                adapted = model(raw_batch)
                loss = weighted_info_nce(
                    adapted,
                    positive_batch,
                    negative_batch,
                    valid_batch,
                    weight_batch,
                    temperature=config.base.adapter.temperature,
                    in_batch_positive_embeddings=positive_batch,
                    in_batch_valid_mask=in_batch_mask,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), config.base.adapter.max_grad_norm
            )
            optimizer.step()
            weighted_loss += float(loss.detach().item()) * len(batch_indices)
            seen += len(batch_indices)
            batches += 1
            if batches % 25 == 0:
                _event(
                    log_path,
                    "adapter_training_progress",
                    role=run_role,
                    epoch=epoch,
                    seen=seen,
                    rows=len(train_indices),
                )
        epoch_loss = weighted_loss / seen
        losses.append(epoch_loss)
        _event(
            log_path,
            "adapter_epoch_completed",
            role=run_role,
            epoch=epoch,
            loss=epoch_loss,
        )
        if epoch_callback is not None:
            epoch_callback(epoch, model, epoch_loss)
    model.eval()
    return losses


def _evaluate_view(
    *,
    name: str,
    embeddings_path: Path,
    raw_embeddings: np.ndarray,
    index: ExactPoiIndex,
    poi_embeddings: np.ndarray,
    target_rows: np.ndarray,
    false_negative_rows: Sequence[Sequence[int]],
    difficult_mask: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    embeddings = np.load(embeddings_path, mmap_mode="r")
    scores, rows = index.search(embeddings, top_k=100)
    rows_path = output_dir / f"{name}_ann_top100_rows.npy"
    scores_path = output_dir / f"{name}_ann_top100_scores.npy"
    np.save(rows_path, rows.astype(np.int32))
    np.save(scores_path, scores.astype(np.float32))
    ranks, positive_scores, hardest_scores = _ranks_and_margins(
        embeddings,
        poi_embeddings,
        rows,
        scores,
        target_rows,
        false_negative_rows,
    )
    metrics = full_retrieval_metrics(
        ranks,
        difficult_mask,
        raw_embeddings,
        embeddings,
    )
    metrics["mean_positive_score"] = float(np.mean(positive_scores))
    metrics["mean_hardest_negative_score"] = float(np.mean(hardest_scores))
    metrics["mean_hard_margin"] = float(
        np.mean(positive_scores - hardest_scores)
    )
    metrics["false_negative_mask_applied"] = True
    metrics["retrieval"] = "exact frozen candidate-catalog POI inner product Top-100"
    write_json_atomic(output_dir / f"{name}_metrics.json", metrics)
    return metrics, ranks


def _embedding_drift(
    model: ResidualQueryAdapter,
    raw_queries: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, Any]:
    values = np.empty(len(raw_queries), dtype=np.float32)
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(raw_queries), batch_size):
            stop = min(start + batch_size, len(raw_queries))
            raw = torch.from_numpy(
                np.asarray(raw_queries[start:stop], dtype=np.float32)
            ).to(device)
            adapted = model(raw)
            cosine = torch.sum(
                torch.nn.functional.normalize(raw.float(), dim=1)
                * adapted.float(),
                dim=1,
            )
            values[start:stop] = (1.0 - cosine).cpu().numpy()
    values = np.maximum(values, 0.0)
    return {
        "definition": "1 - cosine(raw_bge_query, final_adapted_query)",
        "rows": int(len(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _relative_artifact(path: Path, root: Path) -> dict[str, Any]:
    artifact = _artifact(path)
    artifact["file"] = str(path.relative_to(root))
    return artifact


def run_adapter_selection(
    config: AdapterSelectionConfig,
    *,
    resume_incomplete: bool = False,
) -> dict[str, Any]:
    """Run the approved P3A-FULL-SELECT and P3A-FULL-FINAL protocol."""

    output_dir = config.output_dir.resolve()
    resuming = output_dir.exists()
    if resuming and not resume_incomplete:
        raise AdapterSelectionError(f"P3A-FULL 输出已存在，拒绝覆盖：{output_dir}")
    if resuming and (output_dir / "_SUCCESS").exists():
        raise AdapterSelectionError("P3A-FULL 输出已完成，不允许 incomplete resume")
    validate_query_adapter_gate(config.gate, gate="medium")
    gate_manifest_path = config.gate_dir / "manifest.json"
    if sha256_file(gate_manifest_path) != config.gate_manifest_sha256:
        raise AdapterSelectionError("50k Gate manifest SHA256 与 FULL 冻结配置不一致")
    if not resuming:
        output_dir.mkdir(parents=True)
    holdout_dir = output_dir / "internal_holdout"
    comparison_dir = output_dir / "comparison"
    select_dir = output_dir / "select"
    final_dir = output_dir / "final"
    for path in (holdout_dir, comparison_dir, select_dir, final_dir):
        if resuming:
            if not path.is_dir():
                raise AdapterSelectionError(f"incomplete resume 缺少目录：{path}")
        else:
            path.mkdir()
    log_path = output_dir / "run_log.jsonl"
    started = time.perf_counter()
    _event(
        log_path,
        "resumed_after_cgroup_oom" if resuming else "started",
        phase="P3A-FULL",
        max_epochs=config.max_epochs,
        precomputed_artifacts_reused=resuming,
        algorithm_changed=False,
    )
    frozen = _validate_frozen_inputs(config.base)
    _event(log_path, "frozen_inputs_validated", **frozen)

    stats_dir = config.base.query_depth_dir / "query_category_stats.parquet"
    queries, full_d3_rows = select_d3_queries(
        stats_dir,
        limit=config.d3_rows,
        seed=config.holdout_seed,
    )
    if full_d3_rows != config.d3_rows or len(queries) != config.d3_rows:
        raise AdapterSelectionError(
            f"D3 Exact-Core 数量 {full_d3_rows}/{len(queries)} != {config.d3_rows}"
        )
    gate_selection_path = config.gate_dir / "d3_selection.parquet"
    gate_query_ids = _read_gate_query_ids(gate_selection_path)
    holdout_indices, select_train_indices = partition_full_d3(
        queries,
        gate_query_ids,
        gate_exclusion_rows=config.gate_exclusion_rows,
        holdout_rows=config.internal_holdout_rows,
    )
    holdout_queries = [queries[index] for index in holdout_indices]
    holdout_query_ids = np.asarray(
        [query.query_id for query in holdout_queries], dtype=np.int64
    )
    query_ids_path = holdout_dir / "query_ids.npy"
    full_rows_path = holdout_dir / "full_d3_rows.npy"
    if resuming:
        saved_ids = np.load(query_ids_path)
        saved_rows = np.load(full_rows_path)
        if not np.array_equal(saved_ids, holdout_query_ids) or not np.array_equal(
            saved_rows, holdout_indices
        ):
            raise AdapterSelectionError("incomplete resume 的 holdout Query ID/行号不一致")
    else:
        np.save(query_ids_path, holdout_query_ids)
        np.save(full_rows_path, holdout_indices)

    false_ids = load_false_negative_ids(
        config.base.p2_query_stats / "false_negative_mask",
        (query.query_id for query in queries),
    )
    required_ids = {query.target_poi_id for query in queries}
    required_ids.update(
        poi_id for values in false_ids.values() for poi_id in values
    )
    id_to_row = resolve_poi_rows(
        config.base.poi_ids,
        required_ids,
        expected_rows=config.base.category_config.frozen.poi_rows,
    )
    target_rows = np.asarray(
        [id_to_row[query.target_poi_id] for query in queries], dtype=np.int64
    )
    false_rows = [
        tuple(id_to_row[poi_id] for poi_id in false_ids[query.query_id])
        for query in queries
    ]
    if any(
        int(target) not in set(reasonable)
        for target, reasonable in zip(target_rows, false_rows, strict=True)
    ):
        raise AdapterSelectionError("D3 primary target 未出现在 false-negative 保护集合")
    full_selection_path = output_dir / "d3_full_selection.parquet"
    holdout_selection_path = holdout_dir / "holdout_selection.parquet"
    holdout_manifest_path = holdout_dir / "manifest.json"
    if resuming:
        try:
            holdout_manifest = json.loads(
                holdout_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdapterSelectionError("incomplete resume 的 holdout manifest 非法") from error
        if (
            not (holdout_dir / "_SUCCESS").is_file()
            or holdout_manifest.get("schema_version") != HOLDOUT_SCHEMA_VERSION
            or holdout_manifest.get("selection", {}).get("holdout_rows")
            != len(holdout_indices)
            or holdout_manifest.get("source_hashes", {}).get(
                "full_d3_selection_sha256"
            )
            != sha256_file(full_selection_path)
        ):
            raise AdapterSelectionError("incomplete resume 的 holdout 冻结证据不一致")
    else:
        write_selection(full_selection_path, queries, target_rows, false_rows)
        write_selection(
            holdout_selection_path,
            holdout_queries,
            target_rows[holdout_indices],
            [false_rows[index] for index in holdout_indices],
        )
        holdout_manifest = {
            "schema_version": HOLDOUT_SCHEMA_VERSION,
            "status": "completed",
            "built_at": utc_now(),
            "population": "Train-date D3 Exact-Core only",
            "selection": {
                "algorithm": config.holdout_selection,
                "seed": config.holdout_seed,
                "full_d3_rows": config.d3_rows,
                "gate_excluded_rows": config.gate_exclusion_rows,
                "holdout_rows": len(holdout_indices),
                "select_train_rows": len(select_train_indices),
                "first_hash": f"{holdout_queries[0].selection_hash:016x}",
                "last_hash": f"{holdout_queries[-1].selection_hash:016x}",
                "disjoint_from_gate": True,
                "participates_in_select_model_updates": False,
                "participates_in_final_model_updates": True,
            },
            "source_access": {
                "train_date_derived_stats_read": True,
                "business_validation_read": False,
                "business_test_read": False,
            },
            "source_hashes": {
                "p2_manifest_sha256": frozen["p2_manifest_sha256"],
                "p2_5_manifest_sha256": frozen["p2_5_manifest_sha256"],
                "gate_manifest_sha256": config.gate_manifest_sha256,
                "gate_selection_sha256": sha256_file(gate_selection_path),
                "full_d3_selection_sha256": sha256_file(full_selection_path),
            },
            "artifacts": {
                path.name: _relative_artifact(path, holdout_dir)
                for path in (
                    query_ids_path,
                    full_rows_path,
                    holdout_selection_path,
                )
            },
        }
        write_json_atomic(holdout_manifest_path, holdout_manifest)
        (holdout_dir / "_SUCCESS").touch()
        _event(
            log_path,
            "holdout_frozen",
            holdout_rows=len(holdout_indices),
            select_train_rows=len(select_train_indices),
        )

    raw_query_path = output_dir / "raw_query_embeddings_d3.npy"
    if resuming:
        raw_queries = np.load(raw_query_path, mmap_mode="r")
        expected_shape = (
            config.d3_rows,
            config.base.category_config.frozen.poi_embedding_dim,
        )
        if raw_queries.shape != expected_shape or str(raw_queries.dtype) != str(
            np.dtype(config.base.query_embedding.output_dtype)
        ):
            raise AdapterSelectionError("incomplete resume 的 D3 Query embedding 合同不一致")
        sample_rows = np.linspace(
            0,
            len(raw_queries) - 1,
            min(2_048, len(raw_queries)),
            dtype=np.int64,
        )
        norms = np.linalg.norm(
            np.asarray(raw_queries[sample_rows], dtype=np.float32), axis=1
        )
        if float(np.max(np.abs(norms - 1.0))) > 2e-3:
            raise AdapterSelectionError("incomplete resume 的 Query embedding 未保持 L2 normalize")
        encoding = {
            "rows": len(raw_queries),
            "embedding_dim": raw_queries.shape[1],
            "dtype": str(raw_queries.dtype),
            "actual_device": "cuda (reused after cgroup OOM)",
            "model_load_seconds": None,
            "elapsed_seconds": None,
            "l2_norm_min": float(norms.min()),
            "l2_norm_mean": float(norms.mean()),
            "l2_norm_max": float(norms.max()),
            "resumed_precomputed_artifact": True,
        }
    else:
        cache_path, encoding = build_d3_query_cache(config, queries)
        temporary = raw_query_path.with_name(f".{raw_query_path.name}.writing")
        with cache_path.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
        os.replace(temporary, raw_query_path)
        if sha256_file(raw_query_path) != encoding["output_sha256"]:
            raise AdapterSelectionError("发布到 FULL 的 Query cache 哈希不一致")
        _event(log_path, "query_cache_reused", rows=len(queries), source=str(cache_path))
    raw_queries = np.load(raw_query_path, mmap_mode="r")
    holdout_raw_path = comparison_dir / "raw_bge_query_embeddings.npy"
    if resuming:
        saved_holdout_raw = np.load(holdout_raw_path, mmap_mode="r")
        if saved_holdout_raw.shape != (
            config.internal_holdout_rows,
            raw_queries.shape[1],
        ) or not np.array_equal(
            np.asarray(saved_holdout_raw),
            np.asarray(raw_queries[holdout_indices]),
        ):
            raise AdapterSelectionError("incomplete resume 的 holdout Raw BGE 不一致")
    else:
        np.save(holdout_raw_path, np.asarray(raw_queries[holdout_indices]))
    holdout_raw = np.load(holdout_raw_path, mmap_mode="r")
    holdout_targets = target_rows[holdout_indices]
    holdout_false_rows = [false_rows[index] for index in holdout_indices]

    poi_embeddings = np.load(config.base.poi_embeddings, mmap_mode="r")
    index_started = time.perf_counter()
    index = ExactPoiIndex.build(poi_embeddings, config.base.ann)
    _event(
        log_path,
        "poi_index_built",
        rows=len(poi_embeddings),
        elapsed_seconds=time.perf_counter() - index_started,
    )
    raw_rows_path = comparison_dir / "raw_bge_ann_top100_rows.npy"
    raw_scores_path = comparison_dir / "raw_bge_ann_top100_scores.npy"
    raw_metrics_path = comparison_dir / "raw_bge_metrics.json"
    if resuming:
        raw_rows = np.load(raw_rows_path, mmap_mode="r")
        raw_scores = np.load(raw_scores_path, mmap_mode="r")
        if raw_rows.shape != (config.internal_holdout_rows, 100) or raw_scores.shape != (
            config.internal_holdout_rows,
            100,
        ):
            raise AdapterSelectionError("incomplete resume 的 Raw BGE ANN shape 非法")
    else:
        raw_scores, raw_rows = index.search(holdout_raw, top_k=100)
        np.save(raw_rows_path, raw_rows.astype(np.int32))
        np.save(raw_scores_path, raw_scores.astype(np.float32))
    raw_ranks, raw_positive, raw_hardest = _ranks_and_margins(
        holdout_raw,
        poi_embeddings,
        raw_rows,
        raw_scores,
        holdout_targets,
        holdout_false_rows,
    )
    difficult_mask = raw_ranks > 1
    if resuming:
        try:
            raw_metrics = json.loads(raw_metrics_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdapterSelectionError("incomplete resume 的 Raw BGE metrics 非法") from error
    else:
        raw_metrics = full_retrieval_metrics(
            raw_ranks,
            difficult_mask,
            holdout_raw,
            holdout_raw,
        )
        raw_metrics.update(
            {
                "mean_positive_score": float(np.mean(raw_positive)),
                "mean_hardest_negative_score": float(np.mean(raw_hardest)),
                "mean_hard_margin": float(np.mean(raw_positive - raw_hardest)),
                "false_negative_mask_applied": True,
                "retrieval": "exact frozen candidate-catalog POI inner product Top-100",
            }
        )
        write_json_atomic(raw_metrics_path, raw_metrics)

    device = torch.device(f"cuda:{config.base.ann.gpu_id}")
    gate_embeddings_path = comparison_dir / "gate50k_query_embeddings.npy"
    if resuming:
        gate_embeddings = np.load(gate_embeddings_path, mmap_mode="r")
        if gate_embeddings.shape != holdout_raw.shape:
            raise AdapterSelectionError("incomplete resume 的 Gate Query embedding shape 非法")
        gate_metrics_path = comparison_dir / "gate50k_metrics.json"
        try:
            gate_metrics = json.loads(
                gate_metrics_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdapterSelectionError("incomplete resume 的 Gate metrics 非法") from error
        for path in (
            comparison_dir / "gate50k_ann_top100_rows.npy",
            comparison_dir / "gate50k_ann_top100_scores.npy",
        ):
            if np.load(path, mmap_mode="r").shape != (
                config.internal_holdout_rows,
                100,
            ):
                raise AdapterSelectionError("incomplete resume 的 Gate ANN shape 非法")
        _event(log_path, "raw_and_gate50k_holdout_reused")
    else:
        gate_model = _load_adapter_checkpoint(
            config.gate_dir / "query_adapter_exact.pt",
            config,
            device=device,
        )
        _adapt_queries(
            gate_model,
            holdout_raw_path,
            gate_embeddings_path,
            device=device,
        )
        gate_metrics, _ = _evaluate_view(
            name="gate50k",
            embeddings_path=gate_embeddings_path,
            raw_embeddings=holdout_raw,
            index=index,
            poi_embeddings=poi_embeddings,
            target_rows=holdout_targets,
            false_negative_rows=holdout_false_rows,
            difficult_mask=difficult_mask,
            output_dir=comparison_dir,
        )
        del gate_model
        torch.cuda.empty_cache()
        _event(log_path, "raw_and_gate50k_holdout_evaluated")

    unique_target_rows, target_inverse = np.unique(
        target_rows,
        return_inverse=True,
    )
    positive_rows_path = output_dir / "positive_poi_ann_top100_rows.npy"
    positive_scores_path = output_dir / "positive_poi_ann_top100_scores.npy"
    if resuming:
        positive_rows = np.load(positive_rows_path, mmap_mode="r")
        positive_scores = np.load(positive_scores_path, mmap_mode="r")
        expected_positive_shape = (config.d3_rows, config.base.ann.top_k)
        if (
            positive_rows.shape != expected_positive_shape
            or positive_scores.shape != expected_positive_shape
        ):
            raise AdapterSelectionError("incomplete resume 的 positive POI ANN shape 非法")
        if np.any(
            np.asarray(positive_rows[:, 0], dtype=np.int64) == target_rows
        ):
            raise AdapterSelectionError("incomplete resume 的 positive POI ANN 未去除 self")
        _event(
            log_path,
            "positive_poi_ann_reused",
            rows=len(positive_rows),
            unique_target_rows=len(unique_target_rows),
        )
    else:
        unique_positive_scores, unique_positive_rows = index.search_poi_rows(
            unique_target_rows,
            top_k=config.base.ann.top_k + 1,
        )
        positive_scores_all = unique_positive_scores[target_inverse]
        positive_rows_all = unique_positive_rows[target_inverse]
        positive_rows, positive_scores = _without_self(
            positive_rows_all,
            positive_scores_all,
            target_rows,
            top_k=config.base.ann.top_k,
        )
        np.save(positive_rows_path, positive_rows.astype(np.int32))
        np.save(positive_scores_path, positive_scores.astype(np.float32))
        del (
            positive_rows_all,
            positive_scores_all,
            unique_positive_rows,
            unique_positive_scores,
        )
    bundle_path = output_dir / "adapter_training_bundle"
    bundle_temporary_dir = bundle_path.with_name(f".{bundle_path.name}.writing")
    if bundle_temporary_dir.is_dir():
        metadata: Mapping[int, Any] = {}
    else:
        required_metadata_rows = set(int(value) for value in target_rows)
        required_metadata_rows.update(
            int(value) for value in np.unique(positive_rows)
        )
        expected_ids_by_row = {
            row: poi_id
            for poi_id, row in id_to_row.items()
            if row in required_metadata_rows
        }
        metadata = load_selected_metadata(
            config.base.poi_catalog,
            required_metadata_rows,
            expected_ids_by_row,
            expected_rows=config.base.category_config.frozen.poi_rows,
        )
    bundle_summary = _build_full_bundle(
        config=config,
        queries=queries,
        raw_queries_path=raw_query_path,
        poi_embeddings=poi_embeddings,
        target_rows=target_rows,
        false_negative_rows=false_rows,
        positive_ann_rows=positive_rows,
        positive_ann_scores=positive_scores,
        metadata=metadata,
        output_dir=bundle_path,
        log_path=log_path,
    )
    bundle_summary["unique_positive_poi_rows_searched"] = len(unique_target_rows)
    del (
        metadata,
        positive_rows,
        positive_scores,
        target_inverse,
        unique_target_rows,
    )
    gc.collect()
    _event(log_path, "training_bundle_completed", **bundle_summary)

    initial_path = output_dir / "initial_adapter_state.pt"
    initial_sha = _save_initial_state(
        initial_path,
        config,
        raw_queries.shape[1],
    )
    torch.cuda.reset_peak_memory_stats(config.base.ann.gpu_id)
    tensors = _prepare_training_tensors(bundle_path, raw_query_path)
    if tensors.query_ids != tuple(query.query_id for query in queries):
        raise AdapterSelectionError("training bundle Query 顺序与全量 D3 selection 不一致")
    epoch_records: list[dict[str, Any]] = []
    select_model = _fresh_adapter(initial_path, config, device=device)

    def evaluate_select_epoch(
        epoch: int,
        model: ResidualQueryAdapter,
        epoch_loss: float,
    ) -> None:
        epoch_dir = select_dir / "epochs" / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(parents=True)
        checkpoint_path = epoch_dir / "query_adapter_exact_select.pt"
        _save_checkpoint(
            checkpoint_path,
            model,
            config,
            role="P3A-FULL-SELECT",
            trained_epochs=epoch,
            initial_state_sha256=initial_sha,
        )
        embeddings_path = epoch_dir / "adapted_holdout_embeddings.npy"
        _adapt_queries(
            model,
            holdout_raw_path,
            embeddings_path,
            device=device,
        )
        metrics, _ = _evaluate_view(
            name="full_select",
            embeddings_path=embeddings_path,
            raw_embeddings=holdout_raw,
            index=index,
            poi_embeddings=poi_embeddings,
            target_rows=holdout_targets,
            false_negative_rows=holdout_false_rows,
            difficult_mask=difficult_mask,
            output_dir=epoch_dir,
        )
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "metrics": metrics,
            "checkpoint": str(checkpoint_path.relative_to(output_dir)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        epoch_records.append(record)
        _event(
            log_path,
            "select_epoch_evaluated",
            epoch=epoch,
            recall_at_10=metrics["recall_at_10"],
            recall_at_1=metrics["recall_at_1"],
            difficult_recall_at_10=metrics["difficult_recall_at_10"],
        )

    select_losses = _train_epochs(
        select_model,
        tensors,
        select_train_indices,
        config,
        epochs=config.max_epochs,
        device=device,
        log_path=log_path,
        run_role="P3A-FULL-SELECT",
        epoch_callback=evaluate_select_epoch,
    )
    best_epoch = select_best_epoch(epoch_records)
    best_record = next(
        record for record in epoch_records if record["epoch"] == best_epoch
    )
    best_epoch_dir = select_dir / "epochs" / f"epoch_{best_epoch:02d}"
    best_checkpoint_path = select_dir / "query_adapter_exact_select_best.pt"
    shutil.copy2(
        best_epoch_dir / "query_adapter_exact_select.pt",
        best_checkpoint_path,
    )
    best_embeddings_path = comparison_dir / "full_select_query_embeddings.npy"
    shutil.copy2(
        best_epoch_dir / "adapted_holdout_embeddings.npy",
        best_embeddings_path,
    )
    for suffix in ("rows.npy", "scores.npy"):
        shutil.copy2(
            best_epoch_dir / f"full_select_ann_top100_{suffix}",
            comparison_dir / f"full_select_ann_top100_{suffix}",
        )
    select_metrics = {
        "schema_version": SELECT_SCHEMA_VERSION,
        "status": "completed",
        "population": "fixed Train-only internal holdout",
        "train_rows": len(select_train_indices),
        "holdout_rows": len(holdout_indices),
        "holdout_participates_in_model_updates": False,
        "fresh_initial_state_sha256": initial_sha,
        "continued_from_gate_checkpoint": False,
        "max_epochs": config.max_epochs,
        "selection_order": list(config.checkpoint_selection),
        "best_epoch": best_epoch,
        "best_metrics": best_record["metrics"],
        "epoch_records": epoch_records,
        "epoch_losses": select_losses,
        "peak_cuda_memory_bytes": int(
            torch.cuda.max_memory_allocated(config.base.ann.gpu_id)
        ),
    }
    write_json_atomic(select_dir / "selection_metrics.json", select_metrics)
    select_artifacts = [
        best_checkpoint_path,
        select_dir / "selection_metrics.json",
    ]
    select_manifest = {
        "schema_version": SELECT_SCHEMA_VERSION,
        "status": "completed",
        "built_at": utc_now(),
        "config_sha256": config.source_sha256,
        "initial_state_sha256": initial_sha,
        "gate_checkpoint_role": "evaluation_only",
        "continued_from_gate_checkpoint": False,
        "holdout_manifest_sha256": sha256_file(holdout_dir / "manifest.json"),
        "selection": select_metrics,
        "source_access": {
            "business_validation_read": False,
            "business_test_read": False,
        },
        "artifacts": {
            str(path.relative_to(select_dir)): _relative_artifact(path, select_dir)
            for path in select_artifacts
        },
    }
    write_json_atomic(select_dir / "manifest.json", select_manifest)
    (select_dir / "_SUCCESS").touch()

    comparison = {
        "schema_version": "qg-prqk-p3a-full-comparison-v1",
        "status": "completed",
        "population": "same fixed 20,000 Train-only internal holdout",
        "difficult_definition": "Raw BGE Query misses Recall@1 after FN mask",
        "views": {
            "raw_bge_query": raw_metrics,
            "gate50k_adapter": gate_metrics,
            "p3a_full_select_adapter": best_record["metrics"],
        },
        "selected_epoch": best_epoch,
        "gate_checkpoint_role": "evaluation_only",
        "false_negative_mask_applied": True,
        "candidate_catalog_exact_retrieval": True,
        "candidate_catalog_rows": config.base.category_config.frozen.poi_rows,
        "candidate_catalog_scope": "frozen active POI closed set",
    }
    write_json_atomic(comparison_dir / "three_view_comparison.json", comparison)
    _event(log_path, "select_completed", best_epoch=best_epoch)

    del select_model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(config.base.ann.gpu_id)
    final_model = _fresh_adapter(initial_path, config, device=device)
    all_indices = np.arange(config.d3_rows, dtype=np.int64)
    final_started = time.perf_counter()
    final_losses = _train_epochs(
        final_model,
        tensors,
        all_indices,
        config,
        epochs=best_epoch,
        device=device,
        log_path=log_path,
        run_role="P3A-FULL-FINAL",
    )
    final_checkpoint_path = final_dir / "query_adapter_exact_final.pt"
    _save_checkpoint(
        final_checkpoint_path,
        final_model,
        config,
        role="P3A-FULL-FINAL",
        trained_epochs=best_epoch,
        initial_state_sha256=initial_sha,
    )
    final_metrics = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "status": "completed",
        "training_rows": config.d3_rows,
        "fixed_epochs": best_epoch,
        "epoch_losses": final_losses,
        "model_selection_performed": False,
        "selection_source": "P3A-FULL-SELECT best_epoch",
        "fresh_initial_state_sha256": initial_sha,
        "continued_from_select_or_gate_checkpoint": False,
        "all_d3_query_embedding_cosine_drift": _embedding_drift(
            final_model,
            tensors.raw_queries,
            batch_size=4096,
        ),
        "peak_cuda_memory_bytes": int(
            torch.cuda.max_memory_allocated(config.base.ann.gpu_id)
        ),
        "elapsed_seconds": time.perf_counter() - final_started,
    }
    final_metrics_path = final_dir / "metrics.json"
    write_json_atomic(final_metrics_path, final_metrics)
    final_config_path = final_dir / "config.yaml"
    shutil.copy2(config.source_path, final_config_path)
    final_resolved_path = final_dir / "config_resolved.json"
    write_json_atomic(final_resolved_path, config.resolved_payload())
    final_artifacts = [
        final_checkpoint_path,
        final_metrics_path,
        final_config_path,
        final_resolved_path,
    ]
    final_manifest = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "status": "completed",
        "built_at": utc_now(),
        "config_sha256": config.source_sha256,
        "config_signature": config.signature(),
        "source_hashes": {
            "p2_manifest_sha256": frozen["p2_manifest_sha256"],
            "p2_5_manifest_sha256": frozen["p2_5_manifest_sha256"],
            "holdout_manifest_sha256": sha256_file(holdout_dir / "manifest.json"),
            "select_manifest_sha256": sha256_file(select_dir / "manifest.json"),
            "full_d3_selection_sha256": sha256_file(full_selection_path),
            "training_bundle_manifest_sha256": sha256_file(
                bundle_path / "manifest.json"
            ),
            "initial_state_sha256": initial_sha,
        },
        "training": final_metrics,
        "query_view_policy": dict(config.query_view_policy),
        "default_scope": "D3 Exact Query only",
        "d1_d2_validated_or_modified": False,
        "source_access": {
            "business_validation_read": False,
            "business_test_read": False,
        },
        "artifacts": {
            path.name: _relative_artifact(path, final_dir)
            for path in final_artifacts
        },
    }
    write_json_atomic(final_dir / "manifest.json", final_manifest)
    (final_dir / "_SUCCESS").touch()
    _event(log_path, "final_completed", fixed_epochs=best_epoch)

    del final_model, tensors
    gc.collect()
    torch.cuda.empty_cache()
    core_artifact_paths = [
        full_selection_path,
        raw_query_path,
        positive_rows_path,
        positive_scores_path,
        bundle_path / "manifest.json",
        initial_path,
        holdout_dir / "manifest.json",
        select_dir / "manifest.json",
        final_dir / "manifest.json",
        comparison_dir / "three_view_comparison.json",
    ]
    artifacts = {
        str(path.relative_to(output_dir)): _relative_artifact(path, output_dir)
        for path in core_artifact_paths
    }
    _event(log_path, "artifacts_hashed", artifact_count=len(artifacts))
    artifacts[log_path.name] = _relative_artifact(log_path, output_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "phase": "P3A-FULL",
        "built_at": utc_now(),
        "config": {
            "path": str(config.source_path),
            "sha256": config.source_sha256,
            "signature": config.signature(),
            "resolved": config.resolved_payload(),
        },
        "inputs": {
            **frozen,
            "gate_manifest_path": str(gate_manifest_path),
            "gate_manifest_sha256": config.gate_manifest_sha256,
            "gate_artifacts_preserved": True,
        },
        "partition": {
            "full_d3_rows": config.d3_rows,
            "gate_exclusion_rows": config.gate_exclusion_rows,
            "internal_holdout_rows": config.internal_holdout_rows,
            "select_train_rows": len(select_train_indices),
            "holdout_disjoint_from_gate": True,
            "holdout_updates_select": False,
            "source": "Train-date D3 Exact-Core only",
            "seed": config.holdout_seed,
        },
        "initialization": {
            "policy": FULL_INITIALIZATION_POLICY,
            "seed": config.base.seed,
            "initial_state_sha256": initial_sha,
            "select_final_share_saved_initial_state": True,
            "historical_gate_initial_weights_reproduced": False,
        },
        "query_encoding": encoding,
        "ann": {
            **asdict(config.base.ann),
            "exact": True,
            "metric": "inner_product_on_l2_normalized_vectors",
        },
        "negative_mining": bundle_summary,
        "select": select_metrics,
        "comparison": comparison,
        "final": final_metrics,
        "query_view_policy": dict(config.query_view_policy),
        "decision": {
            "default_query_view": "final_adapter",
            "default_scope": "D3 Exact Query only",
            "current_status": "HOLD_FOR_P3A_FULL_REVIEW",
            "p4_cat_started": False,
            "rq_kmeans_started": False,
        },
        "source_access": {
            "train_date_derived_p2_p2_5_read": True,
            "business_validation_read": False,
            "business_test_read": False,
            "d1_d2_representation_validated_or_modified": False,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "valid_and_test_read": False,
            "resumed_after_cgroup_oom": resuming,
            "precomputed_artifacts_reused": resuming,
        },
        "artifacts": artifacts,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").touch()
    return manifest


def validate_adapter_selection(config: AdapterSelectionConfig) -> dict[str, Any]:
    """Validate completed P3A-FULL outputs, hashes, partitions, and roles."""

    output_dir = config.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file() or not (output_dir / "_SUCCESS").is_file():
        raise AdapterSelectionError("P3A-FULL 输出缺少 manifest.json 或 _SUCCESS")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterSelectionError("P3A-FULL manifest 非法") from error
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != "P3A-FULL"
        or manifest.get("config", {}).get("signature") != config.signature()
    ):
        raise AdapterSelectionError("P3A-FULL manifest 版本、状态或配置不一致")
    partition = manifest.get("partition", {})
    expected_partition = {
        "full_d3_rows": config.d3_rows,
        "gate_exclusion_rows": config.gate_exclusion_rows,
        "internal_holdout_rows": config.internal_holdout_rows,
        "select_train_rows": config.d3_rows - config.internal_holdout_rows,
        "holdout_disjoint_from_gate": True,
        "holdout_updates_select": False,
    }
    if any(partition.get(key) != value for key, value in expected_partition.items()):
        raise AdapterSelectionError("P3A-FULL partition manifest 不一致")
    if any(
        manifest.get("source_access", {}).get(key) is not False
        for key in (
            "business_validation_read",
            "business_test_read",
            "d1_d2_representation_validated_or_modified",
        )
    ):
        raise AdapterSelectionError("P3A-FULL 数据隔离或 D1/D2 边界证据非法")
    if manifest.get("query_view_policy") != dict(config.query_view_policy):
        raise AdapterSelectionError("P3A-FULL Query view policy 不一致")
    _validate_full_bundle(
        output_dir / "adapter_training_bundle",
        output_dir / "raw_query_embeddings_d3.npy",
    )
    holdout_ids = np.load(output_dir / "internal_holdout/query_ids.npy")
    full_rows = np.load(output_dir / "internal_holdout/full_d3_rows.npy")
    if (
        holdout_ids.shape != (config.internal_holdout_rows,)
        or len(np.unique(holdout_ids)) != config.internal_holdout_rows
        or full_rows.tolist()
        != list(
            range(
                config.gate_exclusion_rows,
                config.gate_exclusion_rows + config.internal_holdout_rows,
            )
        )
    ):
        raise AdapterSelectionError("P3A-FULL holdout Query ID/全量行号非法")
    gate_ids = np.asarray(
        _read_gate_query_ids(config.gate_dir / "d3_selection.parquet"),
        dtype=np.int64,
    )
    if len(np.intersect1d(gate_ids, holdout_ids)):
        raise AdapterSelectionError("P3A-FULL holdout 与 50k Gate 有交集")
    select = manifest.get("select", {})
    if (
        select.get("continued_from_gate_checkpoint") is not False
        or select.get("holdout_participates_in_model_updates") is not False
        or select.get("best_epoch") != select_best_epoch(select.get("epoch_records", []))
    ):
        raise AdapterSelectionError("P3A-FULL-SELECT 初始化、隔离或 best epoch 非法")
    final = manifest.get("final", {})
    if (
        final.get("training_rows") != config.d3_rows
        or final.get("fixed_epochs") != select.get("best_epoch")
        or final.get("model_selection_performed") is not False
        or final.get("continued_from_select_or_gate_checkpoint") is not False
    ):
        raise AdapterSelectionError("P3A-FULL-FINAL 训练合同非法")
    initial_sha = sha256_file(output_dir / "initial_adapter_state.pt")
    if (
        select.get("fresh_initial_state_sha256") != initial_sha
        or final.get("fresh_initial_state_sha256") != initial_sha
    ):
        raise AdapterSelectionError("SELECT/FINAL 没有共享同一 initial state")
    for name, artifact in manifest.get("artifacts", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise AdapterSelectionError(f"P3A-FULL artifact SHA256 失败：{name}")
    for child in ("internal_holdout", "select", "final"):
        if not (output_dir / child / "manifest.json").is_file() or not (
            output_dir / child / "_SUCCESS"
        ).is_file():
            raise AdapterSelectionError(f"P3A-FULL 子阶段未完成：{child}")
    if sha256_file(config.gate_dir / "manifest.json") != config.gate_manifest_sha256:
        raise AdapterSelectionError("50k Gate 在 P3A-FULL 后发生变化")
    return manifest

"""End-to-end engineering gate for the exact-query adapter."""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from qg_prqk.adapters.training import (
    TRAINING_DATA_SCHEMA_VERSION,
    load_training_bundle,
)
from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.data.query_supervision import validate_category_depth
from qg_prqk.hard_negative_mining import (
    NegativeCandidate,
    lexical_metadata_candidates,
    local_geo_candidates,
    merge_negative_sources,
)
from qg_prqk.adapters.retrieval import ExactPoiIndex
from qg_prqk.adapters.config import QueryAdapterGateConfig
from qg_prqk.adapters.data import (
    D3Query,
    load_false_negative_ids,
    load_selected_metadata,
    resolve_poi_rows,
    select_d3_queries,
    write_selection,
)
from qg_prqk.adapters.model import (
    ResidualQueryAdapter,
    adapter_gate_passes,
    build_negative_valid_mask,
    fit_query_adapter,
    stable_train_dev_split,
)
from qg_prqk.data.query_embeddings import load_text_encoder
from qg_prqk.data.query_statistics import validate_query_stats


SCHEMA_VERSION = "qg-prqk-p3a-adapter-gate-v1"
SOURCE_CODES = {"semantic_ann": 1, "lexical_metadata": 2, "local_geo": 3}


class QueryAdapterGateError(RuntimeError):
    """Raised when the real P3A Gate cannot be completed safely."""


def normalize_encoded_queries(vectors: np.ndarray) -> np.ndarray:
    """Apply an explicit float32 L2 pass after BF16 model inference."""

    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not min(values.shape) or not np.isfinite(values).all():
        raise QueryAdapterGateError("Query encoder 输出必须是有限二维数组")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise QueryAdapterGateError("Query encoder 输出包含零范数向量")
    return values / norms


def _event(log_path: Path, stage: str, **payload: Any) -> None:
    record = {"at": utc_now(), "stage": stage, **payload}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _output_dir(config: QueryAdapterGateConfig, gate: str) -> Path:
    if gate == "sample":
        return config.output_dir / "p3a_sample/query_adapter_exact"
    if gate == "medium":
        return config.output_dir / "query_adapter_exact"
    raise QueryAdapterGateError("P3A gate 只允许 sample 或 medium")


def _limit(config: QueryAdapterGateConfig, gate: str) -> int:
    return (
        config.sample_query_limit if gate == "sample" else config.gate_query_limit
    )


def _validate_frozen_inputs(config: QueryAdapterGateConfig) -> dict[str, Any]:
    p2 = validate_query_stats(config.p2_query_stats)
    p2_manifest_sha = sha256_file(p2.manifest_path)
    if p2_manifest_sha != config.category_config.frozen.p2_manifest_sha256:
        raise QueryAdapterGateError("P2 manifest SHA256 与 v2.1 冻结值不一致")
    p2_5 = validate_category_depth(config.query_depth_dir, config.category_config)
    p2_5_manifest_sha = sha256_file(p2_5.manifest_path)
    if p2_5_manifest_sha != config.p2_5_manifest_sha256:
        raise QueryAdapterGateError("P2.5 manifest SHA256 与 v2.1 冻结值不一致")
    values = np.load(config.poi_embeddings, mmap_mode="r")
    frozen = config.category_config.frozen
    if list(values.shape) != [frozen.poi_rows, frozen.poi_embedding_dim]:
        raise QueryAdapterGateError("POI embedding shape 与冻结合同不一致")
    if str(values.dtype) != frozen.poi_embedding_dtype:
        raise QueryAdapterGateError("POI embedding dtype 与冻结合同不一致")
    return {
        "p2_manifest_path": str(p2.manifest_path),
        "p2_manifest_sha256": p2_manifest_sha,
        "p2_5_manifest_path": str(p2_5.manifest_path),
        "p2_5_manifest_sha256": p2_5_manifest_sha,
        "p2_unique_queries": p2.unique_queries,
        "p2_5_unique_queries": p2_5.unique_queries,
        "poi_embedding_shape": list(values.shape),
        "poi_embedding_dtype": str(values.dtype),
    }


def _encode_queries(
    config: QueryAdapterGateConfig,
    queries: Sequence[D3Query],
    output_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    encoder, actual_device, load_seconds = load_text_encoder(config.query_embedding)
    dimension = encoder.get_sentence_embedding_dimension()
    expected_dim = config.category_config.frozen.poi_embedding_dim
    if dimension != expected_dim:
        raise QueryAdapterGateError(f"Query encoder 维度 {dimension} != {expected_dim}")
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.dtype(config.query_embedding.output_dtype),
        shape=(len(queries), expected_dim),
    )
    texts = [query.normalized_query for query in queries]
    next_row = 0
    for start in range(0, len(texts), config.query_embedding.encode_buffer_size):
        stop = min(start + config.query_embedding.encode_buffer_size, len(texts))
        vectors = np.asarray(
            encoder.encode(
                texts[start:stop],
                batch_size=config.query_embedding.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                prompt_name=config.query_embedding.prompt_name,
            ),
            dtype=np.float32,
        )
        if vectors.shape != (stop - start, expected_dim):
            raise QueryAdapterGateError("Query encoder 输出 shape 非法")
        vectors = normalize_encoded_queries(vectors)
        output[start:stop] = vectors.astype(output.dtype, copy=False)
        output.flush()
        next_row = stop
        _event(log_path, "query_encoding_progress", next_row=next_row, rows=len(texts))
    del output
    del encoder
    gc.collect()
    values = np.load(output_path, mmap_mode="r")
    sample_rows = np.linspace(0, len(values) - 1, min(2048, len(values)), dtype=np.int64)
    norms = np.linalg.norm(np.asarray(values[sample_rows], dtype=np.float32), axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 2e-3:
        raise QueryAdapterGateError("Query cache 未保持 L2 normalize")
    return {
        "rows": len(values),
        "embedding_dim": values.shape[1],
        "dtype": str(values.dtype),
        "actual_device": actual_device,
        "model_load_seconds": load_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "l2_norm_min": float(norms.min()),
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_max": float(norms.max()),
    }


def _without_self(
    rows: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    result_rows = np.empty((len(rows), top_k), dtype=np.int64)
    result_scores = np.empty((len(rows), top_k), dtype=np.float32)
    for index in range(len(rows)):
        keep = rows[index] != targets[index]
        filtered_rows = rows[index][keep]
        filtered_scores = scores[index][keep]
        if len(filtered_rows) < top_k:
            raise QueryAdapterGateError("positive POI ANN 去除 self 后不足 Top-K")
        result_rows[index] = filtered_rows[:top_k]
        result_scores[index] = filtered_scores[:top_k]
    return result_rows, result_scores


def _csr(values: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(values) + 1, dtype=np.int64)
    flattened: list[int] = []
    for index, rows in enumerate(values):
        flattened.extend(int(value) for value in rows)
        offsets[index + 1] = len(flattened)
    return offsets, np.asarray(flattened, dtype=np.int64)


def _build_bundle(
    *,
    config: QueryAdapterGateConfig,
    queries: Sequence[D3Query],
    raw_queries_path: Path,
    poi_embeddings: np.ndarray,
    target_rows: np.ndarray,
    false_negative_rows: Sequence[Sequence[int]],
    positive_ann_rows: np.ndarray,
    positive_ann_scores: np.ndarray,
    metadata: Mapping[int, Any],
    output_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    fixed_count = (
        config.adapter.semantic_ann_negatives
        + config.adapter.lexical_metadata_negatives
        + config.adapter.local_geo_negatives
    )
    candidate_rows = np.empty((len(queries), fixed_count), dtype=np.int64)
    source_codes = np.empty((len(queries), fixed_count), dtype=np.int8)
    source_totals = {name: 0 for name in SOURCE_CODES}
    for index, target_row in enumerate(target_rows):
        forbidden = false_negative_rows[index]
        semantic = [
            NegativeCandidate(int(row), "semantic_ann", float(score))
            for row, score in zip(
                positive_ann_rows[index], positive_ann_scores[index], strict=True
            )
        ]
        pool = [metadata[int(row)] for row in positive_ann_rows[index]]
        target = metadata[int(target_row)]
        lexical = lexical_metadata_candidates(
            target,
            pool,
            top_k=config.ann.top_k,
            false_negative_rows=forbidden,
        )
        local_geo = local_geo_candidates(
            target,
            pool,
            top_k=config.ann.top_k,
            false_negative_rows=forbidden,
        )
        merged = merge_negative_sources(
            target_row=int(target_row),
            false_negative_rows=forbidden,
            in_batch=(),
            semantic_ann=semantic,
            lexical_metadata=lexical,
            local_geo=local_geo,
            semantic_budget=config.adapter.semantic_ann_negatives,
            lexical_budget=config.adapter.lexical_metadata_negatives,
            local_geo_budget=config.adapter.local_geo_negatives,
        )
        if len(merged) != fixed_count:
            raise QueryAdapterGateError(
                f"Query {queries[index].query_id} 只得到 {len(merged)}/{fixed_count} 个固定负例"
            )
        candidate_rows[index] = [candidate.poi_row for candidate in merged]
        source_codes[index] = [SOURCE_CODES[candidate.source] for candidate in merged]
        for candidate in merged:
            source_totals[candidate.source] += 1
        if (index + 1) % 5000 == 0:
            _event(log_path, "negative_mining_progress", next_row=index + 1, rows=len(queries))

    requested_rows = np.concatenate((target_rows, candidate_rows.reshape(-1)))
    unique_rows, inverse = np.unique(requested_rows, return_inverse=True)
    gathered = np.empty(
        (len(unique_rows), poi_embeddings.shape[1]), dtype=np.float16
    )
    for start in range(0, len(poi_embeddings), 32_768):
        stop = min(start + 32_768, len(poi_embeddings))
        left = int(np.searchsorted(unique_rows, start, side="left"))
        right = int(np.searchsorted(unique_rows, stop, side="left"))
        if left == right:
            continue
        block = np.asarray(poi_embeddings[start:stop], dtype=np.float16)
        gathered[left:right] = block[unique_rows[left:right] - start]
    target_inverse = inverse[: len(target_rows)]
    candidate_inverse = inverse[len(target_rows) :].reshape(candidate_rows.shape)
    positive_embeddings = gathered[target_inverse]
    negative_embeddings = gathered[candidate_inverse]
    offsets, fn_rows = _csr(false_negative_rows)
    raw_queries = np.load(raw_queries_path, mmap_mode="r")
    temporary_bundle = output_path.with_name(f".{output_path.name}.writing.npz")
    np.savez(
        temporary_bundle,
        schema_version=np.asarray(TRAINING_DATA_SCHEMA_VERSION),
        raw_queries=raw_queries,
        positive_embeddings=positive_embeddings,
        negative_embeddings=negative_embeddings,
        candidate_poi_rows=candidate_rows,
        negative_source_codes=source_codes,
        target_poi_rows=target_rows,
        false_negative_offsets=offsets,
        false_negative_rows=fn_rows,
        query_weights=np.asarray(
            [query.query_weight for query in queries], dtype=np.float32
        ),
        query_ids=np.asarray([query.query_id for query in queries], dtype=np.int64),
    )
    del negative_embeddings, positive_embeddings, gathered
    os.replace(temporary_bundle, output_path)
    return {
        "schema_version": TRAINING_DATA_SCHEMA_VERSION,
        "rows": len(queries),
        "fixed_negatives_per_query": fixed_count,
        "source_counts": source_totals,
        "false_negative_rows": len(fn_rows),
        "unique_embedding_rows_gathered": len(unique_rows),
        "embedding_gather_order": "single_sequential_poi_scan_then_inverse_restore",
        "elapsed_seconds": time.perf_counter() - started,
    }


def _reasonable_positive_rows(bundle: dict[str, np.ndarray]) -> list[np.ndarray]:
    offsets = bundle["false_negative_offsets"]
    values = bundle["false_negative_rows"]
    return [
        values[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(len(offsets) - 1)
    ]


def _train_adapter(
    config: QueryAdapterGateConfig,
    bundle_path: Path,
    checkpoint_path: Path,
    log_path: Path,
) -> tuple[ResidualQueryAdapter, list[int], list[int], dict[str, Any]]:
    started = time.perf_counter()
    bundle = load_training_bundle(bundle_path)
    reasonable = _reasonable_positive_rows(bundle)
    candidate_rows = torch.from_numpy(bundle["candidate_poi_rows"].astype(np.int64))
    target_rows = torch.from_numpy(bundle["target_poi_rows"].astype(np.int64))
    valid_mask = build_negative_valid_mask(candidate_rows, target_rows, reasonable)
    if not bool(valid_mask.any(dim=1).all().item()):
        raise QueryAdapterGateError("至少一个 Query 在 false-negative mask 后没有负例")
    query_ids = [int(value) for value in bundle["query_ids"]]
    train_indices, dev_indices = stable_train_dev_split(
        query_ids,
        dev_fraction=config.adapter.dev_fraction,
        seed=config.seed,
    )
    model = ResidualQueryAdapter(
        bundle["raw_queries"].shape[1],
        config.adapter.bottleneck,
        residual_scale=config.adapter.residual_scale,
        dropout=config.adapter.dropout,
    )
    torch.cuda.reset_peak_memory_stats(config.ann.gpu_id)
    result = fit_query_adapter(
        model,
        torch.from_numpy(bundle["raw_queries"].astype(np.float32)),
        torch.from_numpy(bundle["positive_embeddings"].astype(np.float32)),
        torch.from_numpy(bundle["negative_embeddings"].astype(np.float32)),
        valid_mask,
        torch.from_numpy(bundle["query_weights"].astype(np.float32)),
        target_rows,
        reasonable,
        train_indices,
        dev_indices,
        config.adapter,
        device=torch.device(f"cuda:{config.ann.gpu_id}"),
        batch_size=config.adapter.batch_size,
        seed=config.seed,
    )
    for epoch, loss in enumerate(result.epoch_losses, start=1):
        _event(log_path, "adapter_epoch", epoch=epoch, loss=loss)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.writing")
    torch.save(
        {
            "schema_version": "qg-prqk-query-adapter-v1",
            "embedding_dim": bundle["raw_queries"].shape[1],
            "adapter_config": asdict(config.adapter),
            "p3a_config_signature": config.signature(),
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    training = {
        "train_rows": len(train_indices),
        "dev_rows": len(dev_indices),
        "epoch_losses": list(result.epoch_losses),
        "fixed_candidate_dev_raw": result.raw_metrics,
        "fixed_candidate_dev_adapted": result.adapted_metrics,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(config.ann.gpu_id)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    del bundle
    gc.collect()
    torch.cuda.empty_cache()
    return model, train_indices, dev_indices, training


def _adapt_queries(
    model: ResidualQueryAdapter,
    raw_path: Path,
    output_path: Path,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> None:
    raw = np.load(raw_path, mmap_mode="r")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16, shape=raw.shape
    )
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(raw), batch_size):
            stop = min(start + batch_size, len(raw))
            inputs = torch.from_numpy(
                np.asarray(raw[start:stop], dtype=np.float32)
            ).to(device)
            output[start:stop] = model(inputs).float().cpu().numpy().astype(np.float16)
    output.flush()
    del output


def _ranks_and_margins(
    query_embeddings: np.ndarray,
    poi_embeddings: np.ndarray,
    ann_rows: np.ndarray,
    ann_scores: np.ndarray,
    target_rows: np.ndarray,
    false_negative_rows: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks = np.full(len(target_rows), ann_rows.shape[1] + 1, dtype=np.int32)
    hardest = np.full(len(target_rows), -np.inf, dtype=np.float32)
    for index in range(len(target_rows)):
        target = int(target_rows[index])
        forbidden = set(int(value) for value in false_negative_rows[index])
        forbidden.discard(target)
        filtered_rank = 0
        for row, score in zip(ann_rows[index], ann_scores[index], strict=True):
            candidate = int(row)
            if candidate in forbidden:
                continue
            filtered_rank += 1
            if candidate != target and not np.isfinite(hardest[index]):
                hardest[index] = float(score)
            if candidate == target and ranks[index] == ann_rows.shape[1] + 1:
                ranks[index] = filtered_rank
            if ranks[index] <= ann_rows.shape[1] and np.isfinite(hardest[index]):
                break
        if not np.isfinite(hardest[index]):
            raise QueryAdapterGateError("ANN Top-K 在 false-negative mask 后无有效负例")
    positive_scores = np.empty(len(target_rows), dtype=np.float32)
    for start in range(0, len(target_rows), 4096):
        stop = min(start + 4096, len(target_rows))
        queries = np.asarray(query_embeddings[start:stop], dtype=np.float32)
        positives = np.asarray(
            poi_embeddings[target_rows[start:stop]], dtype=np.float32
        )
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        positive_norms = np.linalg.norm(positives, axis=1, keepdims=True)
        queries = queries / np.maximum(query_norms, 1e-12)
        positives = positives / np.maximum(positive_norms, 1e-12)
        positive_scores[start:stop] = np.einsum("bd,bd->b", queries, positives)
    return ranks, positive_scores, hardest


def retrieval_metrics(
    ranks: np.ndarray,
    positive_scores: np.ndarray,
    hardest_scores: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    """Compute the fixed P3A exact retrieval metrics for one dev subset."""

    selected = np.flatnonzero(mask)
    if not len(selected):
        raise QueryAdapterGateError("评测子集为空")
    subset_ranks = ranks[selected]
    margins = positive_scores[selected] - hardest_scores[selected]
    return {
        "rows": int(len(selected)),
        "recall_at_1": float(np.mean(subset_ranks <= 1)),
        "recall_at_10": float(np.mean(subset_ranks <= 10)),
        "recall_at_50": float(np.mean(subset_ranks <= 50)),
        "recall_at_100": float(np.mean(subset_ranks <= 100)),
        "mean_positive_score": float(np.mean(positive_scores[selected])),
        "mean_hardest_negative_score": float(np.mean(hardest_scores[selected])),
        "mean_hard_margin": float(np.mean(margins)),
    }


def _evaluate(
    *,
    config: QueryAdapterGateConfig,
    queries: Sequence[D3Query],
    dev_indices: Sequence[int],
    raw_queries: np.ndarray,
    adapted_queries: np.ndarray,
    poi_embeddings: np.ndarray,
    raw_ann_rows: np.ndarray,
    raw_ann_scores: np.ndarray,
    adapted_ann_rows: np.ndarray,
    adapted_ann_scores: np.ndarray,
    target_rows: np.ndarray,
    false_negative_rows: Sequence[Sequence[int]],
) -> dict[str, Any]:
    raw_ranks, raw_positive, raw_hardest = _ranks_and_margins(
        raw_queries,
        poi_embeddings,
        raw_ann_rows,
        raw_ann_scores,
        target_rows,
        false_negative_rows,
    )
    adapted_ranks, adapted_positive, adapted_hardest = _ranks_and_margins(
        adapted_queries,
        poi_embeddings,
        adapted_ann_rows,
        adapted_ann_scores,
        target_rows,
        false_negative_rows,
    )
    dev = np.zeros(len(queries), dtype=np.bool_)
    dev[np.asarray(dev_indices, dtype=np.int64)] = True
    counts = np.asarray([query.query_count for query in queries], dtype=np.int64)
    masks = {
        "overall": dev,
        "head": dev & (counts >= config.evaluation.head_min_query_count),
        "tail": dev & (counts <= config.evaluation.tail_max_query_count),
        "difficult": dev & (raw_ranks > 1),
    }
    raw = {
        name: retrieval_metrics(raw_ranks, raw_positive, raw_hardest, mask)
        for name, mask in masks.items()
    }
    adapted = {
        name: retrieval_metrics(
            adapted_ranks, adapted_positive, adapted_hardest, mask
        )
        for name, mask in masks.items()
    }
    passed = adapter_gate_passes(
        raw["overall"],
        adapted["overall"],
        raw["difficult"],
        adapted["difficult"],
    )
    return {
        "population": "Train-only stable 5% internal dev",
        "false_negative_mask_applied": True,
        "subsets": {
            "head": f"query_count >= {config.evaluation.head_min_query_count}",
            "tail": f"query_count <= {config.evaluation.tail_max_query_count}",
            "difficult": config.evaluation.difficult_definition,
        },
        "primary_metric": config.evaluation.primary_metric,
        "raw": raw,
        "adapted": adapted,
        "gate_passed": passed,
        "selected_query_view": "adapter" if passed else "identity_raw_bge",
    }


def _artifact(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r")
        result.update({"shape": list(values.shape), "dtype": str(values.dtype)})
    return result


def run_query_adapter_gate(config: QueryAdapterGateConfig, *, gate: str) -> dict[str, Any]:
    """Build, train, evaluate, and persist one sample or 50k P3A Gate."""

    output_dir = _output_dir(config, gate).resolve()
    if output_dir.exists():
        raise QueryAdapterGateError(f"P3A 输出已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    log_path = output_dir / "run_log.jsonl"
    started = time.perf_counter()
    _event(log_path, "started", gate=gate, query_limit=_limit(config, gate))
    frozen = _validate_frozen_inputs(config)
    _event(log_path, "frozen_inputs_validated", **frozen)

    stats_dir = config.query_depth_dir / "query_category_stats.parquet"
    queries, full_d3_rows = select_d3_queries(
        stats_dir,
        limit=_limit(config, gate),
        seed=config.seed,
    )
    if full_d3_rows != 291_590:
        raise QueryAdapterGateError(f"D3 总数 {full_d3_rows} != 291590")
    false_ids = load_false_negative_ids(
        config.p2_query_stats / "false_negative_mask",
        (query.query_id for query in queries),
    )
    required_ids = {query.target_poi_id for query in queries}
    required_ids.update(
        poi_id for values in false_ids.values() for poi_id in values
    )
    id_to_row = resolve_poi_rows(
        config.poi_ids,
        required_ids,
        expected_rows=config.category_config.frozen.poi_rows,
    )
    target_rows = np.asarray(
        [id_to_row[query.target_poi_id] for query in queries], dtype=np.int64
    )
    false_rows = [
        tuple(id_to_row[poi_id] for poi_id in false_ids[query.query_id])
        for query in queries
    ]
    if any(int(target) not in set(rows) for target, rows in zip(target_rows, false_rows, strict=True)):
        raise QueryAdapterGateError("D3 primary target 未出现在 P2 false-negative 保护集合")
    selection_path = output_dir / "d3_selection.parquet"
    write_selection(selection_path, queries, target_rows, false_rows)
    _event(
        log_path,
        "d3_selected",
        full_d3_rows=full_d3_rows,
        selected_rows=len(queries),
        false_negative_rows=sum(len(values) for values in false_rows),
    )

    raw_query_path = output_dir / "raw_query_embeddings_d3.npy"
    encoding = _encode_queries(config, queries, raw_query_path, log_path)
    _event(log_path, "query_cache_completed", **encoding)

    poi_embeddings = np.load(config.poi_embeddings, mmap_mode="r")
    raw_queries = np.load(raw_query_path, mmap_mode="r")
    ann_started = time.perf_counter()
    index = ExactPoiIndex.build(poi_embeddings, config.ann)
    _event(
        log_path,
        "poi_index_built",
        rows=len(poi_embeddings),
        backend=config.ann.backend,
        elapsed_seconds=time.perf_counter() - ann_started,
    )
    raw_scores, raw_rows = index.search(raw_queries)
    raw_rows_path = output_dir / "raw_query_ann_top100_rows.npy"
    raw_scores_path = output_dir / "raw_query_ann_top100_scores.npy"
    np.save(raw_rows_path, raw_rows.astype(np.int32))
    np.save(raw_scores_path, raw_scores.astype(np.float32))
    positive_scores_all, positive_rows_all = index.search_poi_rows(
        target_rows, top_k=config.ann.top_k + 1
    )
    positive_rows, positive_scores = _without_self(
        positive_rows_all,
        positive_scores_all,
        target_rows,
        top_k=config.ann.top_k,
    )
    positive_rows_path = output_dir / "positive_poi_ann_top100_rows.npy"
    positive_scores_path = output_dir / "positive_poi_ann_top100_scores.npy"
    np.save(positive_rows_path, positive_rows.astype(np.int32))
    np.save(positive_scores_path, positive_scores.astype(np.float32))
    _event(log_path, "raw_and_positive_ann_completed", rows=len(queries), top_k=config.ann.top_k)

    required_metadata_rows = set(int(value) for value in target_rows)
    required_metadata_rows.update(
        int(value) for value in np.unique(positive_rows)
    )
    expected_ids_by_row = {
        row: poi_id for poi_id, row in id_to_row.items() if row in required_metadata_rows
    }
    metadata = load_selected_metadata(
        config.poi_catalog,
        required_metadata_rows,
        expected_ids_by_row,
        expected_rows=config.category_config.frozen.poi_rows,
    )
    _event(log_path, "poi_metadata_loaded", selected_rows=len(metadata))

    bundle_path = output_dir / "adapter_training_bundle.npz"
    bundle = _build_bundle(
        config=config,
        queries=queries,
        raw_queries_path=raw_query_path,
        poi_embeddings=poi_embeddings,
        target_rows=target_rows,
        false_negative_rows=false_rows,
        positive_ann_rows=positive_rows,
        positive_ann_scores=positive_scores,
        metadata=metadata,
        output_path=bundle_path,
        log_path=log_path,
    )
    del metadata
    gc.collect()
    _event(log_path, "training_bundle_completed", **bundle)

    checkpoint_path = output_dir / "query_adapter_exact.pt"
    model, train_indices, dev_indices, training = _train_adapter(
        config, bundle_path, checkpoint_path, log_path
    )
    _event(log_path, "adapter_training_completed", **training)

    adapted_path = output_dir / "adapted_query_embeddings_d3.npy"
    device = torch.device(f"cuda:{config.ann.gpu_id}")
    _adapt_queries(model, raw_query_path, adapted_path, device=device)
    adapted_queries = np.load(adapted_path, mmap_mode="r")
    adapted_scores, adapted_rows = index.search(adapted_queries)
    adapted_rows_path = output_dir / "adapted_query_ann_top100_rows.npy"
    adapted_scores_path = output_dir / "adapted_query_ann_top100_scores.npy"
    np.save(adapted_rows_path, adapted_rows.astype(np.int32))
    np.save(adapted_scores_path, adapted_scores.astype(np.float32))
    evaluation = _evaluate(
        config=config,
        queries=queries,
        dev_indices=dev_indices,
        raw_queries=raw_queries,
        adapted_queries=adapted_queries,
        poi_embeddings=poi_embeddings,
        raw_ann_rows=raw_rows,
        raw_ann_scores=raw_scores,
        adapted_ann_rows=adapted_rows,
        adapted_ann_scores=adapted_scores,
        target_rows=target_rows,
        false_negative_rows=false_rows,
    )
    evaluation_path = output_dir / "raw_vs_adapted_eval.json"
    write_json_atomic(evaluation_path, evaluation)
    _event(
        log_path,
        "evaluation_completed",
        gate_passed=evaluation["gate_passed"],
        selected_query_view=evaluation["selected_query_view"],
    )
    del model, index
    torch.cuda.empty_cache()

    artifact_paths = [
        selection_path,
        raw_query_path,
        raw_rows_path,
        raw_scores_path,
        positive_rows_path,
        positive_scores_path,
        bundle_path,
        checkpoint_path,
        adapted_path,
        adapted_rows_path,
        adapted_scores_path,
        evaluation_path,
    ]
    artifacts = {path.name: _artifact(path) for path in artifact_paths}
    _event(log_path, "artifacts_hashed", artifact_count=len(artifacts))
    artifacts[log_path.name] = _artifact(log_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "phase": "P3A",
        "gate": gate,
        "gate_scope": "sample" if gate == "sample" else "at_most_50000_d3",
        "built_at": utc_now(),
        "config": {
            "path": str(config.source_path),
            "sha256": config.source_sha256,
            "p3a_signature": config.signature(),
            "resolved": config.resolved_payload(),
        },
        "inputs": {
            **frozen,
            "poi_embeddings": str(config.poi_embeddings),
            "poi_ids": str(config.poi_ids),
            "poi_catalog": str(config.poi_catalog),
            "poi_ids_sha256": config.category_config.frozen.poi_ids_sha256,
        },
        "source_access": {
            "p2_query_category_stats_read": True,
            "p2_false_negative_mask_read": True,
            "poi_embedding_values_read": True,
            "poi_metadata_fields_read": [
                "poi_id",
                "displayname",
                "alias",
                "category",
                "category_code",
                "address",
                "lat",
                "lng",
            ],
            "raw_train_read": False,
            "validation_read": False,
            "test_read": False,
            "poi_metadata_fields_excluded": ["area", "layer", "click_score"],
        },
        "selection": {
            "algorithm": config.selection,
            "full_d3_rows": full_d3_rows,
            "selected_rows": len(queries),
            "first_hash": f"{queries[0].selection_hash:016x}",
            "last_hash": f"{queries[-1].selection_hash:016x}",
            "train_rows": len(train_indices),
            "dev_rows": len(dev_indices),
        },
        "query_encoding": encoding,
        "ann": {
            **asdict(config.ann),
            "metric": "inner_product_on_l2_normalized_vectors",
            "exact": True,
            "float32_l2_normalize_before_add_and_search": True,
            "raw_query_top100": True,
            "adapted_query_top100": True,
            "positive_poi_top100": True,
        },
        "negative_mining": bundle,
        "training": training,
        "evaluation": evaluation,
        "decision": {
            "gate_passed": evaluation["gate_passed"],
            "query_view": evaluation["selected_query_view"],
            "full_291590_requires_user_confirmation": True,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "valid_and_test_read": False,
        },
        "artifacts": artifacts,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").touch()
    return manifest


def validate_query_adapter_gate(config: QueryAdapterGateConfig, *, gate: str) -> dict[str, Any]:
    """Independently validate a completed P3A sample or medium artifact."""

    output_dir = _output_dir(config, gate).resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file() or not (output_dir / "_SUCCESS").is_file():
        raise QueryAdapterGateError("P3A 输出缺少 manifest.json 或 _SUCCESS")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QueryAdapterGateError("P3A manifest 非法") from error
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("phase") != "P3A"
        or manifest.get("gate") != gate
        or manifest.get("config", {}).get("p3a_signature") != config.signature()
    ):
        raise QueryAdapterGateError("P3A manifest 版本、状态或配置不一致")
    access = manifest.get("source_access", {})
    if any(access.get(key) is not False for key in ("raw_train_read", "validation_read", "test_read")):
        raise QueryAdapterGateError("P3A Train-only 隔离证据非法")
    if access.get("poi_metadata_fields_excluded") != ["area", "layer", "click_score"]:
        raise QueryAdapterGateError("P3A 禁用字段证据非法")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise QueryAdapterGateError("P3A manifest 缺少 artifacts")
    for name, info in artifacts.items():
        if not isinstance(info, Mapping):
            raise QueryAdapterGateError(f"P3A artifact {name} 描述非法")
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != info.get("sha256"):
            raise QueryAdapterGateError(f"P3A artifact hash 不一致：{name}")
        if path.suffix == ".npy":
            values = np.load(path, mmap_mode="r")
            if list(values.shape) != info.get("shape") or str(values.dtype) != info.get("dtype"):
                raise QueryAdapterGateError(f"P3A NPY 合同不一致：{name}")
    selected = int(manifest.get("selection", {}).get("selected_rows", -1))
    if selected != _limit(config, gate):
        raise QueryAdapterGateError("P3A selected_rows 与 gate limit 不一致")
    if gate == "medium" and selected > 50_000:
        raise QueryAdapterGateError("P3A medium 超过 50,000 D3")
    return manifest

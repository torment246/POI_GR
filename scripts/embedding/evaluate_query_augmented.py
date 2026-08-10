#!/usr/bin/env python3
"""Evaluate sparse Train-Query augmentation of frozen BGE POI embeddings."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding.query_augmentation import (  # noqa: E402
    validate_query_poi_aggregates,
)
from poi_gr.embedding.query_augmented_eval import (  # noqa: E402
    adaptive_alpha_from_effective_count,
    fuse_embedding_chunk,
    metrics_by_masks,
    paired_rank_comparison,
    rank_metrics,
    target_ranks_from_topk,
)
from poi_gr.embedding.query_category_residual import (  # noqa: E402
    validate_query_category_residual,
)
from poi_gr.embedding.query_heterogeneity import (  # noqa: E402
    validate_query_heterogeneity,
)


class EvaluationError(RuntimeError):
    """Raised when a frozen input or exact evaluation run is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在冻结的 10k Query 上评测 E1/E2/E3 Query 增强 POI 向量。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/embedding/embedding_query_augmented_eval_v1.yaml"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅执行冻结输入与对齐校验，不创建 Faiss 索引。",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{name} 必须是 mapping")
    return value


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationError(f"{name} 不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise EvaluationError(f"{name} 必须是 JSON object：{path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationError(f"配置不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        return _mapping(yaml.safe_load(handle), "配置根节点")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sha(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise EvaluationError(f"{name} 不存在：{path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise EvaluationError(f"{name} SHA256 {actual} != 冻结值 {expected}")
    return actual


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def _load_eval_records(
    path: Path,
    expected_rows: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    order_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise EvaluationError(f"评测集第 {row_number} 行不是 object")
            for field in ("query", "poi_id", "order_id"):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise EvaluationError(
                        f"评测集第 {row_number} 行的 {field} 非法"
                    )
            if record["order_id"] in order_ids:
                raise EvaluationError(
                    f"评测集存在重复 order_id：{record['order_id']}"
                )
            order_ids.add(record["order_id"])
            records.append(record)
    if len(records) != expected_rows:
        raise EvaluationError(
            f"评测集行数 {len(records)} != 冻结值 {expected_rows}"
        )
    return records


def _validate_query_mapping(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, record in enumerate(records):
            line = handle.readline()
            if not line:
                raise EvaluationError("Query mapping 比评测集短")
            mapping = json.loads(line)
            expected = {
                "row_index": row_index,
                "order_id": record["order_id"],
                "target_poi_id": record["poi_id"],
            }
            for name, value in expected.items():
                if mapping.get(name) != value:
                    raise EvaluationError(
                        f"Query mapping 第 {row_index + 1} 行 {name} 不一致"
                    )
        if handle.readline():
            raise EvaluationError("Query mapping 比评测集长")


def _map_target_poi_rows(
    poi_ids_path: Path,
    records: list[dict[str, Any]],
    expected_rows: int,
) -> np.ndarray:
    target_ids = {record["poi_id"] for record in records}
    rows_by_id: dict[str, int] = {}
    row_count = 0
    with poi_ids_path.open("r", encoding="utf-8") as handle:
        for row_count, line in enumerate(handle, start=1):
            poi_id = json.loads(line)
            if not isinstance(poi_id, str) or not poi_id:
                raise EvaluationError(f"POI ID 第 {row_count} 行非法")
            if poi_id in target_ids:
                if poi_id in rows_by_id:
                    raise EvaluationError(f"目标 POI ID 重复：{poi_id}")
                rows_by_id[poi_id] = row_count - 1
    if row_count != expected_rows:
        raise EvaluationError(
            f"POI ID 行数 {row_count} != 冻结值 {expected_rows}"
        )
    missing = target_ids.difference(rows_by_id)
    if missing:
        raise EvaluationError(f"有 {len(missing)} 个目标 POI 不在候选库")
    return np.fromiter(
        (rows_by_id[record["poi_id"]] for record in records),
        dtype=np.int64,
        count=len(records),
    )


def _query_length_masks(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    lengths = np.fromiter(
        (len(record["query"].strip()) for record in records),
        dtype=np.int32,
        count=len(records),
    )
    return {
        "1-2": (lengths >= 1) & (lengths <= 2),
        "3-5": (lengths >= 3) & (lengths <= 5),
        "6-10": (lengths >= 6) & (lengths <= 10),
        ">10": lengths > 10,
    }


def _count_masks(
    values: np.ndarray,
    *,
    kind: str,
) -> dict[str, np.ndarray]:
    if kind == "unique_query_count":
        return {
            "0": values == 0,
            "1-5": (values >= 1) & (values <= 5),
            "6-20": (values >= 6) & (values <= 20),
            ">20": values > 20,
        }
    if kind == "train_order_count":
        return {
            "0": values == 0,
            "1-5": (values >= 1) & (values <= 5),
            "6-20": (values >= 6) & (values <= 20),
            "21-100": (values >= 21) & (values <= 100),
            ">100": values > 100,
        }
    raise EvaluationError(f"未知诊断分桶：{kind}")


def _validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    input_config = _mapping(config.get("input"), "input")
    expected_rows = int(input_config["expected_eval_rows"])
    expected_poi_rows = int(input_config["expected_poi_rows"])
    eval_path = _resolve(input_config["eval_data"])
    query_path = _resolve(input_config["query_embeddings"])
    mapping_path = _resolve(input_config["query_mapping"])
    poi_dir = _resolve(input_config["poi_embedding_dir"])
    poi_embeddings_path = poi_dir / "embeddings.npy"
    poi_ids_path = poi_dir / "poi_ids.jsonl"
    aggregates_dir = _resolve(input_config["aggregates_dir"])
    aggregates_manifest_path = aggregates_dir / "manifest.json"
    baseline_results_path = (
        _resolve(input_config["baseline_run_dir"]) / "retrieval_results.npz"
    )
    heterogeneity_dir = None
    heterogeneity_manifest_path = None
    if input_config.get("heterogeneity_dir") is not None:
        heterogeneity_dir = _resolve(input_config["heterogeneity_dir"])
        heterogeneity_manifest_path = heterogeneity_dir / "manifest.json"
    reference_results_path = None
    if input_config.get("reference_run_dir") is not None:
        reference_results_path = (
            _resolve(input_config["reference_run_dir"])
            / "retrieval_results.npz"
        )
    category_residual_dir = None
    category_residual_manifest_path = None
    category_indices_path = None
    if input_config.get("category_residual_dir") is not None:
        category_residual_dir = _resolve(input_config["category_residual_dir"])
        category_residual_manifest_path = (
            category_residual_dir / "manifest.json"
        )
        category_indices_path = _resolve(input_config["category_indices"])

    hashes = {
        "eval_data": _check_sha(
            eval_path,
            input_config["expected_eval_sha256"],
            "冻结评测集",
        ),
        "query_embeddings": _check_sha(
            query_path,
            input_config["expected_query_embeddings_sha256"],
            "冻结 Query embedding",
        ),
        "query_mapping": _check_sha(
            mapping_path,
            input_config["expected_query_mapping_sha256"],
            "冻结 Query mapping",
        ),
        "poi_embeddings": _check_sha(
            poi_embeddings_path,
            input_config["expected_poi_embeddings_sha256"],
            "冻结 POI embedding",
        ),
        "poi_ids": _check_sha(
            poi_ids_path,
            input_config["expected_poi_ids_sha256"],
            "冻结 POI ID",
        ),
        "aggregates_manifest": _check_sha(
            aggregates_manifest_path,
            input_config["expected_aggregates_manifest_sha256"],
            "冻结 E1/E2 聚合 manifest",
        ),
        "baseline_results": _check_sha(
            baseline_results_path,
            input_config["expected_baseline_results_sha256"],
            "冻结 E0 检索结果",
        ),
    }
    if heterogeneity_manifest_path is not None:
        hashes["heterogeneity_manifest"] = _check_sha(
            heterogeneity_manifest_path,
            input_config["expected_heterogeneity_manifest_sha256"],
            "冻结 E3 异质性特征 manifest",
        )
    if reference_results_path is not None:
        hashes["reference_results"] = _check_sha(
            reference_results_path,
            input_config["expected_reference_results_sha256"],
            "冻结 E2 对照检索结果",
        )
    if category_residual_manifest_path is not None:
        hashes["category_residual_manifest"] = _check_sha(
            category_residual_manifest_path,
            input_config["expected_category_residual_manifest_sha256"],
            "冻结 E4 类别残差 manifest",
        )
        hashes["category_indices"] = _check_sha(
            category_indices_path,
            input_config["expected_category_indices_sha256"],
            "冻结 E4 类别行号",
        )
    validate_query_poi_aggregates(aggregates_dir)
    if heterogeneity_dir is not None:
        validate_query_heterogeneity(
            heterogeneity_dir,
            expected_aggregates_manifest_sha256=(
                input_config["expected_aggregates_manifest_sha256"]
            ),
        )
    if category_residual_dir is not None:
        validate_query_category_residual(
            category_residual_dir,
            expected_aggregates_manifest_sha256=(
                input_config["expected_aggregates_manifest_sha256"]
            ),
            expected_category_indices_sha256=(
                input_config["expected_category_indices_sha256"]
            ),
        )
    records = _load_eval_records(eval_path, expected_rows)
    _validate_query_mapping(mapping_path, records)

    query_embeddings = np.load(query_path, mmap_mode="r", allow_pickle=False)
    poi_embeddings = np.load(
        poi_embeddings_path, mmap_mode="r", allow_pickle=False
    )
    if query_embeddings.shape != (expected_rows, poi_embeddings.shape[1]):
        raise EvaluationError("Query embedding shape 与评测集/POI 维度不一致")
    if poi_embeddings.shape[0] != expected_poi_rows:
        raise EvaluationError("POI embedding 行数与配置不一致")
    if str(query_embeddings.dtype) != "float16" or str(
        poi_embeddings.dtype
    ) != "float16":
        raise EvaluationError("冻结 Query/POI embedding 必须是 float16")
    target_rows = _map_target_poi_rows(
        poi_ids_path, records, expected_poi_rows
    )

    aggregate_manifest = _load_json(
        aggregates_manifest_path, "E1/E2 聚合 manifest"
    )
    covered_rows = np.load(
        aggregates_dir / "covered_poi_rows.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    position_by_poi_row = np.full(expected_poi_rows, -1, dtype=np.int32)
    position_by_poi_row[covered_rows] = np.arange(
        len(covered_rows), dtype=np.int32
    )
    category_by_aggregate = None
    category_query_means = None
    if category_residual_dir is not None:
        category_indices = np.load(
            category_indices_path, mmap_mode="r", allow_pickle=False
        )
        category_query_means = np.load(
            category_residual_dir / "category_query_means.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if category_indices.shape != (expected_poi_rows,) or not (
            np.issubdtype(category_indices.dtype, np.integer)
        ):
            raise EvaluationError("E4 类别行号 shape/dtype 非法")
        if (
            category_query_means.ndim != 2
            or category_query_means.shape[1] != poi_embeddings.shape[1]
            or str(category_query_means.dtype) != "float32"
        ):
            raise EvaluationError("E4 类别中心 shape/dtype 非法")
        category_by_aggregate = np.asarray(
            category_indices[covered_rows], dtype=np.int32
        )
        if np.any(category_by_aggregate < 0) or np.any(
            category_by_aggregate >= len(category_query_means)
        ):
            raise EvaluationError("E4 聚合 POI 类别行号越界")
    effective_query_count = None
    if heterogeneity_dir is not None:
        effective_query_count = np.load(
            heterogeneity_dir / "e2_effective_query_count.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if effective_query_count.shape != (len(covered_rows),):
            raise EvaluationError("E3 有效 Query 数与覆盖 POI 数不一致")
    target_positions = position_by_poi_row[target_rows]
    unique_query_count = np.zeros(expected_rows, dtype=np.int32)
    train_order_count = np.zeros(expected_rows, dtype=np.int64)
    covered_targets = target_positions >= 0
    unique_query_values = np.load(
        aggregates_dir / "unique_query_count.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    train_order_values = np.load(
        aggregates_dir / "train_order_count.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    unique_query_count[covered_targets] = unique_query_values[
        target_positions[covered_targets]
    ]
    train_order_count[covered_targets] = train_order_values[
        target_positions[covered_targets]
    ]

    with np.load(baseline_results_path, allow_pickle=False) as baseline:
        baseline_topk = np.asarray(baseline["topk_indices"])
        baseline_ranks = np.asarray(baseline["target_ranks"])
    if baseline_topk.shape != (expected_rows, 20):
        raise EvaluationError("E0 top-k shape 非法")
    rebuilt_baseline_ranks = target_ranks_from_topk(
        baseline_topk, target_rows
    )
    if not np.array_equal(baseline_ranks, rebuilt_baseline_ranks):
        raise EvaluationError("E0 target rank 与冻结目标 POI 行号不一致")

    reference_ranks = None
    reference_metrics = None
    if reference_results_path is not None:
        with np.load(reference_results_path, allow_pickle=False) as reference:
            reference_topk = np.asarray(reference["topk_indices"])
            reference_ranks = np.asarray(reference["target_ranks"])
        if reference_topk.shape != (expected_rows, 20):
            raise EvaluationError("E2 对照 top-k shape 非法")
        rebuilt_reference_ranks = target_ranks_from_topk(
            reference_topk, target_rows
        )
        if not np.array_equal(reference_ranks, rebuilt_reference_ranks):
            raise EvaluationError("E2 对照 target rank 与冻结目标 POI 行号不一致")
        reference_metrics = rank_metrics(reference_ranks)

    paths = {
        "eval_data": eval_path,
        "query_embeddings": query_path,
        "query_mapping": mapping_path,
        "poi_embeddings": poi_embeddings_path,
        "poi_ids": poi_ids_path,
        "aggregates_dir": aggregates_dir,
        "aggregates_manifest": aggregates_manifest_path,
        "baseline_results": baseline_results_path,
    }
    if heterogeneity_dir is not None:
        paths["heterogeneity_dir"] = heterogeneity_dir
    if reference_results_path is not None:
        paths["reference_results"] = reference_results_path
    if category_residual_dir is not None:
        paths["category_residual_dir"] = category_residual_dir
        paths["category_indices"] = category_indices_path

    return {
        "paths": paths,
        "hashes": hashes,
        "records": records,
        "query_embeddings": query_embeddings,
        "poi_embeddings": poi_embeddings,
        "target_rows": target_rows,
        "position_by_poi_row": position_by_poi_row,
        "baseline_ranks": baseline_ranks,
        "baseline_metrics": rank_metrics(baseline_ranks),
        "reference_ranks": reference_ranks,
        "reference_metrics": reference_metrics,
        "effective_query_count": effective_query_count,
        "category_by_aggregate": category_by_aggregate,
        "category_query_means": category_query_means,
        "query_length_masks": _query_length_masks(records),
        "unique_query_masks": _count_masks(
            unique_query_count, kind="unique_query_count"
        ),
        "train_order_masks": _count_masks(
            train_order_count, kind="train_order_count"
        ),
        "aggregate_manifest": aggregate_manifest,
        "covered_eval_rows": int(covered_targets.sum()),
        "unique_eval_target_pois": len({record["poi_id"] for record in records}),
    }


def _exact_search(
    poi_embeddings: np.ndarray,
    query_aggregates: np.ndarray,
    position_by_poi_row: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    alpha: float | np.ndarray,
    run_label: str,
    fusion_chunk_rows: int,
    faiss_config: dict[str, Any],
    category_by_aggregate: np.ndarray | None = None,
    category_query_means: np.ndarray | None = None,
    residual_beta: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import faiss
    import torch

    if faiss_config.get("type") != "IndexFlatIP":
        raise EvaluationError("正式评测只允许 IndexFlatIP")
    if faiss_config.get("device") != "gpu":
        raise EvaluationError("正式评测必须使用 GPU Faiss")
    if bool(faiss_config.get("use_float16")):
        raise EvaluationError("精确评测禁止 Faiss float16 index")
    gpu_id = int(faiss_config.get("gpu_id", 0))
    if not torch.cuda.is_available() or gpu_id >= faiss.get_num_gpus():
        raise EvaluationError(f"GPU {gpu_id} 对 PyTorch/Faiss 不可用")
    if not hasattr(faiss, "StandardGpuResources"):
        raise EvaluationError("当前 Faiss 没有 GPU 接口")

    rows, dimension = poi_embeddings.shape
    initial_add_rows = min(
        int(faiss_config["initial_add_rows"]), rows
    )
    if initial_add_rows <= 0:
        raise EvaluationError("initial_add_rows 必须大于 0")
    storage_bytes = rows * dimension * np.dtype(np.float32).itemsize
    expansion_bytes = (
        storage_bytes
        + initial_add_rows * dimension * np.dtype(np.float32).itemsize
    )
    headroom_bytes = int(
        float(faiss_config["memory_headroom_gib"]) * 1024**3
    )
    free_before, total_memory = torch.cuda.mem_get_info(gpu_id)
    required_bytes = expansion_bytes + headroom_bytes
    if free_before < required_bytes:
        raise EvaluationError(
            "GPU 显存不足："
            f"空闲 {free_before / 1024**3:.2f} GiB，"
            f"要求 {required_bytes / 1024**3:.2f} GiB"
        )

    resources = faiss.StandardGpuResources()
    index_config = faiss.GpuIndexFlatConfig()
    index_config.device = gpu_id
    index_config.useFloat16 = False
    index = faiss.GpuIndexFlatIP(resources, dimension, index_config)
    minimum_free = free_before
    add_ranges = [(0, initial_add_rows)]
    if initial_add_rows < rows:
        add_ranges.append((initial_add_rows, rows))
    build_started = time.perf_counter()
    progress = tqdm(total=rows, desc=f"{run_label} GPU add", unit="poi")
    add_batch_rows: list[int] = []
    for start, stop in add_ranges:
        batch = fuse_embedding_chunk(
            poi_embeddings,
            query_aggregates,
            position_by_poi_row,
            start=start,
            stop=stop,
            alpha=alpha,
            fusion_chunk_rows=fusion_chunk_rows,
            category_by_aggregate=category_by_aggregate,
            category_query_means=category_query_means,
            residual_beta=residual_beta,
        )
        index.add(batch)
        del batch
        current_free, _ = torch.cuda.mem_get_info(gpu_id)
        minimum_free = min(minimum_free, current_free)
        add_batch_rows.append(stop - start)
        progress.update(stop - start)
    progress.close()
    resources.syncDefaultStreamCurrentDevice()
    build_seconds = time.perf_counter() - build_started
    if type(index).__name__ != "GpuIndexFlatIP" or index.ntotal != rows:
        raise EvaluationError("Faiss GPU index 类型或向量数非法")

    top_k = int(faiss_config["top_k"])
    query_batch_size = int(faiss_config["query_batch_size"])
    topk_scores = np.empty((len(query_embeddings), top_k), dtype=np.float32)
    topk_indices = np.empty((len(query_embeddings), top_k), dtype=np.int64)
    search_started = time.perf_counter()
    progress = tqdm(
        total=len(query_embeddings), desc=f"{run_label} search", unit="query"
    )
    for start in range(0, len(query_embeddings), query_batch_size):
        stop = min(start + query_batch_size, len(query_embeddings))
        batch = np.ascontiguousarray(
            query_embeddings[start:stop], dtype=np.float32
        )
        scores, indices = index.search(batch, top_k)
        topk_scores[start:stop] = scores
        topk_indices[start:stop] = indices
        del batch
        current_free, _ = torch.cuda.mem_get_info(gpu_id)
        minimum_free = min(minimum_free, current_free)
        progress.update(stop - start)
    progress.close()
    search_seconds = time.perf_counter() - search_started
    metrics = {
        "type": type(index).__name__,
        "device": "gpu",
        "gpu_id": gpu_id,
        "gpu_name": torch.cuda.get_device_name(gpu_id),
        "faiss_version": faiss.__version__,
        "use_float16": False,
        "ntotal": int(index.ntotal),
        "dimension": dimension,
        "top_k": top_k,
        "query_batch_size": query_batch_size,
        "add_batch_rows": add_batch_rows,
        "build_seconds": build_seconds,
        "search_seconds": search_seconds,
        "index_storage_estimated_bytes": storage_bytes,
        "index_expansion_peak_estimated_bytes": expansion_bytes,
        "memory_headroom_bytes": headroom_bytes,
        "memory_required_bytes": required_bytes,
        "gpu_memory_total_bytes": int(total_memory),
        "gpu_memory_free_before_bytes": int(free_before),
        "gpu_memory_min_free_bytes": int(minimum_free),
        "gpu_memory_peak_delta_bytes": int(max(0, free_before - minimum_free)),
        "memory_check_status": "passed",
    }
    del index
    del resources
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(gpu_id)
    free_after, _ = torch.cuda.mem_get_info(gpu_id)
    metrics["gpu_memory_free_after_bytes"] = int(free_after)
    return topk_indices, topk_scores, metrics


def _alpha_label(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")


def _run_signature(
    config_sha256: str,
    method: str,
    alpha: float,
    input_hashes: dict[str, str],
) -> str:
    payload = {
        "config_sha256": config_sha256,
        "method": method,
        "alpha": alpha,
        "input_hashes": input_hashes,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _run_one(
    *,
    method: str,
    method_order: int,
    aggregate_dir: Path,
    aggregate_filename: str,
    alpha: float,
    residual_beta: float | None,
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    validated: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    run_dir = output_root / method / _alpha_label(alpha)
    signature = _run_signature(
        config_sha256, method, alpha, validated["hashes"]
    )
    if run_dir.is_dir():
        manifest = _load_json(run_dir / "run_manifest.json", "已有运行 manifest")
        if manifest.get("status") != "completed":
            raise EvaluationError(f"已有运行未完成：{run_dir}")
        if manifest.get("signature") != signature:
            raise EvaluationError(f"已有运行 signature 不一致：{run_dir}")
        return {
            "method": method,
            "method_order": method_order,
            "alpha": alpha,
            "residual_beta": residual_beta,
            "metrics": _load_json(run_dir / "metrics.json", "已有 metrics"),
            "run_dir": str(run_dir),
            "reused": True,
        }
    staging_dir = run_dir.with_name(f".{run_dir.name}.building")
    if staging_dir.exists():
        failed_manifest_path = staging_dir / "run_manifest.json"
        failed_manifest = _load_json(
            failed_manifest_path, "未完成运行 manifest"
        )
        if (
            failed_manifest.get("status") != "failed"
            or failed_manifest.get("signature") != signature
        ):
            raise EvaluationError(f"存在不可自动恢复的 staging：{staging_dir}")
        archived_dir = staging_dir.with_name(
            f"{staging_dir.name}.failed-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        )
        os.replace(staging_dir, archived_dir)
    staging_dir.mkdir(parents=True)
    started_at = _utc_now()
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "experiment_id": config["experiment_id"],
        "method": method,
        "method_order": method_order,
        "alpha": alpha,
        "residual_beta": residual_beta,
        "signature": signature,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "git": _git_state(),
        "input_hashes": validated["hashes"],
    }
    _write_json_atomic(staging_dir / "run_manifest.json", manifest)
    try:
        aggregate_path = aggregate_dir / aggregate_filename
        query_aggregates = np.load(
            aggregate_path, mmap_mode="r", allow_pickle=False
        )
        fusion_config = _mapping(config.get("fusion"), "fusion")
        faiss_config = _mapping(config.get("faiss"), "faiss")
        topk_indices, topk_scores, faiss_metrics = _exact_search(
            validated["poi_embeddings"],
            query_aggregates,
            validated["position_by_poi_row"],
            validated["query_embeddings"],
            alpha=alpha,
            run_label=f"alpha={alpha:.2f}",
            fusion_chunk_rows=int(fusion_config["fusion_chunk_rows"]),
            faiss_config=faiss_config,
            category_by_aggregate=validated["category_by_aggregate"],
            category_query_means=validated["category_query_means"],
            residual_beta=residual_beta,
        )
        target_ranks = target_ranks_from_topk(
            topk_indices, validated["target_rows"]
        )
        metrics = rank_metrics(target_ranks)
        metric_names = (
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "hit_at_10",
            "hit_at_20",
            "mrr_at_10",
            "ndcg_at_10",
        )
        metrics.update(
            {
                "query_length_buckets": metrics_by_masks(
                    target_ranks, validated["query_length_masks"]
                ),
                "target_unique_query_count_buckets": metrics_by_masks(
                    target_ranks, validated["unique_query_masks"]
                ),
                "target_train_order_count_buckets": metrics_by_masks(
                    target_ranks, validated["train_order_masks"]
                ),
                "paired_vs_e0": paired_rank_comparison(
                    validated["baseline_ranks"],
                    target_ranks,
                    top_k=int(faiss_config["top_k"]),
                ),
            }
        )
        if validated["reference_ranks"] is not None:
            metrics["paired_vs_reference"] = paired_rank_comparison(
                validated["reference_ranks"],
                target_ranks,
                top_k=int(faiss_config["top_k"]),
            )
            metrics["deltas_vs_reference"] = {
                name: float(
                    metrics[name] - validated["reference_metrics"][name]
                )
                for name in metric_names
            }
        metrics_path = staging_dir / "metrics.json"
        results_path = staging_dir / "retrieval_results.npz"
        _write_json_atomic(metrics_path, metrics)
        _save_npz_atomic(
            results_path,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            target_ranks=target_ranks,
        )
        elapsed = time.perf_counter() - started
        manifest.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "fusion": {
                    "formula": (
                        "normalize((1-alpha)*c_i + "
                        "alpha*normalize(q_i-beta*mean_category(i)))"
                        if residual_beta is not None
                        else "normalize((1-alpha)*c_i + alpha*q_i)"
                    ),
                    "uncovered_policy": "keep frozen c_i unchanged",
                    "aggregate_file": str(aggregate_path),
                    "residual_beta": residual_beta,
                },
                "faiss": faiss_metrics,
                "metrics": metrics,
                "timing_seconds": {"total": elapsed},
                "outputs": {
                    "metrics": "metrics.json",
                    "retrieval_results": "retrieval_results.npz",
                    "retrieval_results_sha256": _sha256_file(results_path),
                },
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                },
            }
        )
        _write_json_atomic(staging_dir / "run_manifest.json", manifest)
        os.replace(staging_dir, run_dir)
        return {
            "method": method,
            "method_order": method_order,
            "alpha": alpha,
            "residual_beta": residual_beta,
            "metrics": metrics,
            "run_dir": str(run_dir),
            "reused": False,
        }
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "timing_seconds": {"total": time.perf_counter() - started},
            }
        )
        _write_json_atomic(staging_dir / "run_manifest.json", manifest)
        raise


def _distribution(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(
        np.asarray(values, dtype=np.float64),
        [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
    )
    return {
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(np.mean(values, dtype=np.float64)),
    }


def _adaptive_candidate(
    value: Any,
    candidate_order: int,
) -> dict[str, Any]:
    candidate = _mapping(value, f"fusion.adaptive_candidates[{candidate_order}]")
    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str) or not candidate_id or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for char in candidate_id
    ):
        raise EvaluationError(f"E3 candidate id 非法：{candidate_id!r}")
    parsed = {
        "id": candidate_id,
        "alpha_min": float(candidate["alpha_min"]),
        "alpha_max": float(candidate["alpha_max"]),
        "tau": float(candidate["tau"]),
        "gamma": float(candidate["gamma"]),
    }
    try:
        adaptive_alpha_from_effective_count(
            np.asarray([1.0], dtype=np.float32),
            alpha_min=parsed["alpha_min"],
            alpha_max=parsed["alpha_max"],
            tau=parsed["tau"],
            gamma=parsed["gamma"],
        )
    except RuntimeError as error:
        raise EvaluationError(f"E3 candidate {candidate_id} 非法：{error}") from error
    return parsed


def _adaptive_run_signature(
    config_sha256: str,
    candidate: dict[str, Any],
    input_hashes: dict[str, str],
) -> str:
    payload = {
        "config_sha256": config_sha256,
        "method": "e3",
        "candidate": candidate,
        "input_hashes": input_hashes,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _run_one_adaptive(
    *,
    candidate: dict[str, Any],
    candidate_order: int,
    aggregate_filename: str,
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    validated: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    effective_query_count = validated.get("effective_query_count")
    reference_ranks = validated.get("reference_ranks")
    reference_metrics = validated.get("reference_metrics")
    if effective_query_count is None:
        raise EvaluationError("E3 缺少冻结的有效 Query 数")
    if reference_ranks is None or reference_metrics is None:
        raise EvaluationError("E3 缺少冻结的 E2/alpha=0.30 对照结果")
    alpha_values = adaptive_alpha_from_effective_count(
        effective_query_count,
        alpha_min=candidate["alpha_min"],
        alpha_max=candidate["alpha_max"],
        tau=candidate["tau"],
        gamma=candidate["gamma"],
    )
    alpha_distribution = _distribution(alpha_values)
    candidate_id = candidate["id"]
    run_dir = output_root / "e3" / candidate_id
    signature = _adaptive_run_signature(
        config_sha256, candidate, validated["hashes"]
    )
    if run_dir.is_dir():
        manifest = _load_json(run_dir / "run_manifest.json", "已有 E3 manifest")
        if manifest.get("status") != "completed":
            raise EvaluationError(f"已有 E3 运行未完成：{run_dir}")
        if manifest.get("signature") != signature:
            raise EvaluationError(f"已有 E3 运行 signature 不一致：{run_dir}")
        return {
            "method": "e3",
            "candidate_id": candidate_id,
            "candidate_order": candidate_order,
            "candidate": candidate,
            "alpha_distribution": manifest["fusion"]["alpha_distribution"],
            "metrics": _load_json(run_dir / "metrics.json", "已有 E3 metrics"),
            "run_dir": str(run_dir),
            "reused": True,
        }

    staging_dir = run_dir.with_name(f".{run_dir.name}.building")
    if staging_dir.exists():
        failed_manifest = _load_json(
            staging_dir / "run_manifest.json", "未完成 E3 manifest"
        )
        if (
            failed_manifest.get("status") != "failed"
            or failed_manifest.get("signature") != signature
        ):
            raise EvaluationError(f"存在不可自动恢复的 E3 staging：{staging_dir}")
        archived_dir = staging_dir.with_name(
            f"{staging_dir.name}.failed-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        )
        os.replace(staging_dir, archived_dir)
    staging_dir.mkdir(parents=True)
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": _utc_now(),
        "experiment_id": config["experiment_id"],
        "method": "e3",
        "candidate": candidate,
        "candidate_order": candidate_order,
        "signature": signature,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "git": _git_state(),
        "input_hashes": validated["hashes"],
    }
    _write_json_atomic(staging_dir / "run_manifest.json", manifest)
    try:
        aggregate_path = validated["paths"]["aggregates_dir"] / aggregate_filename
        query_aggregates = np.load(
            aggregate_path, mmap_mode="r", allow_pickle=False
        )
        fusion_config = _mapping(config.get("fusion"), "fusion")
        faiss_config = _mapping(config.get("faiss"), "faiss")
        topk_indices, topk_scores, faiss_metrics = _exact_search(
            validated["poi_embeddings"],
            query_aggregates,
            validated["position_by_poi_row"],
            validated["query_embeddings"],
            alpha=alpha_values,
            run_label=candidate_id,
            fusion_chunk_rows=int(fusion_config["fusion_chunk_rows"]),
            faiss_config=faiss_config,
        )
        target_ranks = target_ranks_from_topk(
            topk_indices, validated["target_rows"]
        )
        metrics = rank_metrics(target_ranks)
        metric_names = (
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "hit_at_10",
            "hit_at_20",
            "mrr_at_10",
            "ndcg_at_10",
        )
        metrics.update(
            {
                "query_length_buckets": metrics_by_masks(
                    target_ranks, validated["query_length_masks"]
                ),
                "target_unique_query_count_buckets": metrics_by_masks(
                    target_ranks, validated["unique_query_masks"]
                ),
                "target_train_order_count_buckets": metrics_by_masks(
                    target_ranks, validated["train_order_masks"]
                ),
                "paired_vs_e0": paired_rank_comparison(
                    validated["baseline_ranks"],
                    target_ranks,
                    top_k=int(faiss_config["top_k"]),
                ),
                "paired_vs_e2_fixed": paired_rank_comparison(
                    reference_ranks,
                    target_ranks,
                    top_k=int(faiss_config["top_k"]),
                ),
                "deltas_vs_e2_fixed": {
                    name: float(metrics[name] - reference_metrics[name])
                    for name in metric_names
                },
            }
        )
        metrics_path = staging_dir / "metrics.json"
        results_path = staging_dir / "retrieval_results.npz"
        _write_json_atomic(metrics_path, metrics)
        _save_npz_atomic(
            results_path,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            target_ranks=target_ranks,
            alpha_values=alpha_values,
        )
        manifest.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "fusion": {
                    "formula": (
                        "alpha_i=alpha_min+(alpha_max-alpha_min)/"
                        "(1+(n_eff_i/tau)^gamma); "
                        "normalize((1-alpha_i)*c_i+alpha_i*q_i)"
                    ),
                    "effective_query_count": "(sum w)^2/sum(w^2)",
                    "uncovered_policy": "keep frozen c_i unchanged",
                    "aggregate_file": str(aggregate_path),
                    "alpha_distribution": alpha_distribution,
                },
                "faiss": faiss_metrics,
                "metrics": metrics,
                "timing_seconds": {"total": time.perf_counter() - started},
                "outputs": {
                    "metrics": "metrics.json",
                    "retrieval_results": "retrieval_results.npz",
                    "retrieval_results_sha256": _sha256_file(results_path),
                },
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                },
            }
        )
        _write_json_atomic(staging_dir / "run_manifest.json", manifest)
        os.replace(staging_dir, run_dir)
        return {
            "method": "e3",
            "candidate_id": candidate_id,
            "candidate_order": candidate_order,
            "candidate": candidate,
            "alpha_distribution": alpha_distribution,
            "metrics": metrics,
            "run_dir": str(run_dir),
            "reused": False,
        }
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "timing_seconds": {"total": time.perf_counter() - started},
            }
        )
        _write_json_atomic(staging_dir / "run_manifest.json", manifest)
        raise


def _adaptive_selection_key(
    result: dict[str, Any],
    selection_metric: str,
    tie_break_metrics: list[str],
) -> tuple[float, ...]:
    metrics = result["metrics"]
    return (
        float(metrics[selection_metric]),
        *(float(metrics[name]) for name in tie_break_metrics),
        -float(result["candidate_order"]),
    )


def _selection_key(
    result: dict[str, Any],
    selection_metric: str,
    tie_break_metrics: list[str],
) -> tuple[float, ...]:
    metrics = result["metrics"]
    return (
        float(metrics[selection_metric]),
        *(float(metrics[name]) for name in tie_break_metrics),
        -float(result["alpha"]),
        -float(result["method_order"]),
    )


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    config = _load_yaml(config_path)
    config_sha256 = _sha256_file(config_path)
    validated = _validate_inputs(config)
    validation_summary = {
        "status": "passed",
        "experiment_id": config.get("experiment_id"),
        "eval_rows": len(validated["records"]),
        "unique_eval_target_pois": validated["unique_eval_target_pois"],
        "eval_rows_with_train_query_aggregate": validated["covered_eval_rows"],
        "poi_embedding_shape": list(validated["poi_embeddings"].shape),
        "query_embedding_shape": list(validated["query_embeddings"].shape),
        "aggregate_covered_pois": int(
            validated["aggregate_manifest"]["aggregation"]["covered_pois"]
        ),
        "baseline_metrics": validated["baseline_metrics"],
        "input_hashes": validated["hashes"],
    }
    if validated["reference_metrics"] is not None:
        validation_summary["reference_metrics"] = validated["reference_metrics"]
    if validated["effective_query_count"] is not None:
        validation_summary["effective_query_count_distribution"] = _distribution(
            validated["effective_query_count"]
        )
    if args.validate_only:
        print(json.dumps(validation_summary, ensure_ascii=False, indent=2))
        return 0

    output_root = _resolve(_mapping(config.get("output"), "output")["dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_root / "validation.json", validation_summary)
    fusion_config = _mapping(config.get("fusion"), "fusion")
    methods = _mapping(fusion_config.get("methods"), "fusion.methods")
    adaptive_candidates_value = fusion_config.get("adaptive_candidates")
    if adaptive_candidates_value is not None:
        if not isinstance(adaptive_candidates_value, list) or not (
            adaptive_candidates_value
        ):
            raise EvaluationError("fusion.adaptive_candidates 必须是非空列表")
        if set(methods) != {"e3"}:
            raise EvaluationError("自适应评测的 fusion.methods 必须只包含 e3")
        method_config = _mapping(methods["e3"], "fusion.methods.e3")
        aggregate_filename = method_config.get("aggregate_file")
        if not isinstance(aggregate_filename, str):
            raise EvaluationError("e3 aggregate_file 非法")
        candidates = [
            _adaptive_candidate(value, order)
            for order, value in enumerate(adaptive_candidates_value)
        ]
        candidate_ids = [candidate["id"] for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise EvaluationError("E3 candidate id 重复")
        results: list[dict[str, Any]] = []
        for candidate_order, candidate in enumerate(candidates):
            print(
                f"[run] method=e3 candidate={candidate['id']}", flush=True
            )
            result = _run_one_adaptive(
                candidate=candidate,
                candidate_order=candidate_order,
                aggregate_filename=aggregate_filename,
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                validated=validated,
                output_root=output_root,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "method": "e3",
                        "candidate_id": candidate["id"],
                        "ndcg_at_10": result["metrics"]["ndcg_at_10"],
                        "hit_at_10": result["metrics"]["hit_at_10"],
                        "mrr_at_10": result["metrics"]["mrr_at_10"],
                        "deltas_vs_e2_fixed": result["metrics"][
                            "deltas_vs_e2_fixed"
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        selection_metric = str(fusion_config["selection_metric"])
        tie_break_metrics = [
            str(value) for value in fusion_config.get("tie_break_metrics", [])
        ]
        selected = max(
            results,
            key=lambda result: _adaptive_selection_key(
                result, selection_metric, tie_break_metrics
            ),
        )
        summary = {
            "status": "completed",
            "finished_at": _utc_now(),
            "experiment_id": config["experiment_id"],
            "config": str(config_path),
            "config_sha256": config_sha256,
            "selection": {
                "primary_metric": selection_metric,
                "tie_break_metrics": tie_break_metrics,
                "final_tie_break": "frozen candidate order",
            },
            "baseline": validated["baseline_metrics"],
            "reference_e2_fixed": validated["reference_metrics"],
            "runs": results,
            "selected_overall": selected,
            "validation": validation_summary,
            "git": _git_state(),
        }
        _write_json_atomic(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    alphas = [float(value) for value in fusion_config.get("alphas", [])]
    if not alphas or any(not 0.0 < value <= 1.0 for value in alphas):
        raise EvaluationError("fusion.alphas 非法")
    results: list[dict[str, Any]] = []
    for method_order, (method, method_config_value) in enumerate(
        methods.items()
    ):
        method_config = _mapping(
            method_config_value, f"fusion.methods.{method}"
        )
        aggregate_filename = method_config.get("aggregate_file")
        if not isinstance(aggregate_filename, str):
            raise EvaluationError(f"{method} aggregate_file 非法")
        residual_beta_value = method_config.get("residual_beta")
        residual_beta = (
            None
            if residual_beta_value is None
            else float(residual_beta_value)
        )
        if residual_beta is not None and (
            not np.isfinite(residual_beta) or not 0.0 < residual_beta <= 1.0
        ):
            raise EvaluationError(f"{method} residual_beta 非法")
        if residual_beta is not None and (
            validated["category_by_aggregate"] is None
            or validated["category_query_means"] is None
        ):
            raise EvaluationError(f"{method} 缺少 E4 类别残差输入")
        for alpha in alphas:
            print(
                f"[run] method={method} alpha={alpha:.2f} "
                f"beta={residual_beta}",
                flush=True,
            )
            result = _run_one(
                method=method,
                method_order=method_order,
                aggregate_dir=(
                    _resolve(method_config["aggregate_dir"])
                    if method_config.get("aggregate_dir") is not None
                    else validated["paths"]["aggregates_dir"]
                ),
                aggregate_filename=aggregate_filename,
                alpha=alpha,
                residual_beta=residual_beta,
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                validated=validated,
                output_root=output_root,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "method": method,
                        "alpha": alpha,
                        "residual_beta": residual_beta,
                        "ndcg_at_10": result["metrics"]["ndcg_at_10"],
                        "hit_at_10": result["metrics"]["hit_at_10"],
                        "mrr_at_10": result["metrics"]["mrr_at_10"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    selection_metric = str(fusion_config["selection_metric"])
    tie_break_metrics = [
        str(value) for value in fusion_config.get("tie_break_metrics", [])
    ]
    selected_by_method = {
        method: max(
            (result for result in results if result["method"] == method),
            key=lambda result: _selection_key(
                result, selection_metric, tie_break_metrics
            ),
        )
        for method in methods
    }
    selected_overall = max(
        selected_by_method.values(),
        key=lambda result: _selection_key(
            result, selection_metric, tie_break_metrics
        ),
    )
    summary = {
        "status": "completed",
        "finished_at": _utc_now(),
        "experiment_id": config["experiment_id"],
        "config": str(config_path),
        "config_sha256": config_sha256,
        "selection": {
            "primary_metric": selection_metric,
            "tie_break_metrics": tie_break_metrics,
            "final_tie_break": "lower_alpha_then_frozen_method_order",
        },
        "baseline": validated["baseline_metrics"],
        "reference": validated["reference_metrics"],
        "runs": results,
        "selected_by_method": selected_by_method,
        "selected_overall": selected_overall,
        "validation": validation_summary,
        "git": _git_state(),
    }
    _write_json_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate exact POI embedding retrieval with an aligned query encoder."""

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
from dataclasses import asdict, replace
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

from poi_gr.embedding import (  # noqa: E402
    _load_encoder,
    _read_record,
    discover_input_files,
    load_job_config,
)


class EvaluationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用与 POI 完全一致的编码器执行精确向量召回评测。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/embedding/embedding_retrieval_eval.yaml"),
        help="评测配置路径；相对路径相对于仓库根目录。",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("qwen3_0.6b", "qwen3_4b", "bge_m3", "mmbert_recall_128"),
        help="评测模型。",
    )
    parser.add_argument(
        "--instruction",
        required=True,
        choices=("none", "poi_en"),
        help="Query Instruction 模式。",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvaluationError(f"配置根节点必须是 mapping：{path}")
    return payload


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{name} 必须是 mapping")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _non_empty_string(record: dict[str, Any], field: str, row: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"评测数据第 {row} 行的 {field} 为空或不是字符串")
    return value


def load_eval_records(
    path: Path,
    expected_rows: int,
    expected_sha256: str | None,
) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise EvaluationError(f"评测数据不存在：{path}")
    records: list[dict[str, Any]] = []
    order_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle, start=1):
            if not line.strip():
                raise EvaluationError(f"评测数据第 {row} 行为空行")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"评测数据第 {row} 行 JSON 解析失败") from error
            if not isinstance(record, dict):
                raise EvaluationError(f"评测数据第 {row} 行不是 JSON object")
            _non_empty_string(record, "query", row)
            _non_empty_string(record, "poi_id", row)
            order_id = _non_empty_string(record, "order_id", row)
            if order_id in order_ids:
                raise EvaluationError(f"评测数据存在重复 order_id：{order_id}")
            order_ids.add(order_id)
            records.append(record)
    if len(records) != expected_rows:
        raise EvaluationError(
            f"评测数据行数 {len(records)} != 期望 {expected_rows}"
        )
    digest = _sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise EvaluationError(
            f"评测数据 SHA256 {digest} != 期望 {expected_sha256}"
        )
    if len(order_ids) != expected_rows:
        raise EvaluationError("评测数据唯一 order_id 数不等于总行数")
    return records, digest


def validate_reference_eval_order(
    records: list[dict[str, Any]],
    eval_sha256: str,
    reference_manifest_path: Path,
) -> tuple[dict[str, Any], Path]:
    if not reference_manifest_path.is_file():
        raise EvaluationError(f"E1 reference manifest 不存在：{reference_manifest_path}")
    with reference_manifest_path.open("r", encoding="utf-8") as handle:
        reference = json.load(handle)
    if reference.get("status") != "completed":
        raise EvaluationError("E1 reference run 状态不是 completed")
    reference_eval_sha256 = reference.get("gate_zero", {}).get("eval_sha256")
    if reference_eval_sha256 != eval_sha256:
        raise EvaluationError("当前评测数据 SHA256 与 E1 不一致")

    mapping_value = reference.get("outputs", {}).get("query_mapping")
    if not isinstance(mapping_value, str):
        raise EvaluationError("E1 reference manifest 缺少 Query 行映射")
    mapping_path = _resolve(mapping_value)
    if not mapping_path.is_file():
        raise EvaluationError(f"E1 Query 行映射不存在：{mapping_path}")
    mapping_sha256 = _sha256_file(mapping_path)
    expected_mapping_sha256 = reference.get("outputs", {}).get(
        "query_mapping_sha256"
    )
    if mapping_sha256 != expected_mapping_sha256:
        raise EvaluationError("E1 Query 行映射 SHA256 与 manifest 不一致")

    row_count = 0
    with mapping_path.open("r", encoding="utf-8") as handle:
        for row_count, (record, line) in enumerate(
            zip(records, handle, strict=True),
            start=1,
        ):
            try:
                mapping = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"E1 Query 行映射第 {row_count} 行解析失败") from error
            if mapping.get("row_index") != row_count - 1:
                raise EvaluationError(f"E1 Query 行映射第 {row_count} 行 row_index 错误")
            if mapping.get("order_id") != record["order_id"]:
                raise EvaluationError(f"当前 order_id 顺序与 E1 在第 {row_count} 行不一致")
            if mapping.get("target_poi_id") != record["poi_id"]:
                raise EvaluationError(f"当前目标 poi_id 与 E1 在第 {row_count} 行不一致")
    if row_count != len(records):
        raise EvaluationError(
            f"E1 Query 行映射行数 {row_count} != 当前评测行数 {len(records)}"
        )
    return (
        {
            "status": "passed",
            "reference_model": reference.get("model"),
            "reference_instruction": reference.get("instruction"),
            "eval_sha256_match": True,
            "order_id_rows_match": row_count,
            "target_poi_id_rows_match": row_count,
            "query_mapping_sha256": mapping_sha256,
        },
        mapping_path,
    )


def _validate_manifest(
    manifest: dict[str, Any],
    expected_rows: int,
    embedding_shape: tuple[int, ...],
    poi_config_path: Path,
) -> None:
    if manifest.get("status") != "completed":
        raise EvaluationError("POI Embedding manifest 状态不是 completed")
    if int(manifest.get("input", {}).get("total_rows", -1)) != expected_rows:
        raise EvaluationError("manifest 中的 POI 行数不正确")
    if tuple(manifest.get("output", {}).get("shape", ())) != embedding_shape:
        raise EvaluationError("manifest shape 与 embeddings.npy 不一致")
    job_config = load_job_config(poi_config_path, PROJECT_ROOT)
    manifest_model = manifest.get("model", {})
    try:
        expected_model_path = str(job_config.model.path.relative_to(PROJECT_ROOT))
    except ValueError:
        expected_model_path = str(job_config.model.path)
    expected_model = {
        "path": expected_model_path,
        "padding_side": job_config.model.padding_side,
        "normalize_embeddings": job_config.model.normalize_embeddings,
        "torch_dtype": job_config.model.torch_dtype,
        "attention": job_config.model.attention,
        "backend": job_config.model.backend,
    }
    for key, value in expected_model.items():
        actual = manifest_model.get(key)
        if key == "backend" and actual is None:
            actual = "sentence_transformers"
        if actual != value:
            raise EvaluationError(
                f"manifest model.{key}={actual!r} != {value!r}"
            )


def validate_poi_ids_and_alignment(
    poi_config_path: Path,
    ids_path: Path,
    manifest: dict[str, Any],
    target_ids: set[str],
    expected_rows: int,
) -> tuple[dict[str, int], int, str]:
    job_config = load_job_config(poi_config_path, PROJECT_ROOT)
    files = discover_input_files(job_config.data)
    manifest_sources = manifest.get("input", {}).get("sources", [])
    if [path.name for path in files] != [item.get("name") for item in manifest_sources]:
        raise EvaluationError("POI 原始分片列表与 manifest 不一致")
    if not ids_path.is_file():
        raise EvaluationError(f"POI ID 文件不存在：{ids_path}")

    target_indices: dict[str, int] = {}
    unique_ids: set[str] = set()
    rebuilt_sources: list[dict[str, Any]] = []
    total_rows = 0
    with ids_path.open("r", encoding="utf-8") as ids_handle:
        progress = tqdm(total=expected_rows, desc="Gate 0 POI ID alignment", unit="row")
        for path in files:
            file_hash = hashlib.sha256()
            file_rows = 0
            file_bytes = 0
            with path.open("rb") as input_handle:
                for line_number, raw_line in enumerate(input_handle, start=1):
                    poi_id, _ = _read_record(
                        raw_line,
                        path=path,
                        line_number=line_number,
                        id_field=job_config.data.id_field,
                        text_field=job_config.data.text_field,
                    )
                    ids_line = ids_handle.readline()
                    if not ids_line:
                        raise EvaluationError("POI ID 文件比 POI 原始数据短")
                    try:
                        aligned_id = json.loads(ids_line)
                    except json.JSONDecodeError as error:
                        raise EvaluationError(
                            f"POI ID 文件第 {total_rows + 1} 行解析失败"
                        ) from error
                    if aligned_id != poi_id:
                        raise EvaluationError(
                            f"POI ID 第 {total_rows + 1} 行与原始数据顺序不一致"
                        )
                    if poi_id in unique_ids:
                        raise EvaluationError(f"POI ID 重复：{poi_id}")
                    unique_ids.add(poi_id)
                    if poi_id in target_ids:
                        target_indices[poi_id] = total_rows
                    file_hash.update(raw_line)
                    file_rows += 1
                    file_bytes += len(raw_line)
                    total_rows += 1
                    progress.update(1)
            rebuilt_sources.append(
                {
                    "name": path.name,
                    "rows_scanned": file_rows,
                    "bytes_scanned": file_bytes,
                    "sha256_scanned": file_hash.hexdigest(),
                }
            )
        progress.close()
        if ids_handle.readline():
            raise EvaluationError("POI ID 文件比 POI 原始数据长")

    if total_rows != expected_rows:
        raise EvaluationError(f"POI ID 行数 {total_rows} != 期望 {expected_rows}")
    if len(unique_ids) != expected_rows:
        raise EvaluationError(
            f"POI ID 唯一数 {len(unique_ids)} != 期望 {expected_rows}"
        )
    if rebuilt_sources != manifest_sources:
        raise EvaluationError("POI 原始分片内容指纹与 manifest 不一致")
    fingerprint_payload = {
        "id_field": job_config.data.id_field,
        "text_field": job_config.data.text_field,
        "total_rows": total_rows,
        "sources": rebuilt_sources,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if fingerprint != manifest.get("input", {}).get("fingerprint"):
        raise EvaluationError("重建的 POI 输入 fingerprint 与 manifest 不一致")
    missing = target_ids.difference(target_indices)
    if missing:
        example = next(iter(missing))
        raise EvaluationError(
            f"有 {len(missing)} 个目标 poi_id 不在候选库，例如 {example}"
        )
    return target_indices, len(unique_ids), fingerprint


def validate_embeddings(
    embeddings_path: Path,
    expected_shape: tuple[int, int],
    finite_chunk_rows: int,
    norm_sample_rows: int,
) -> tuple[np.memmap, dict[str, float]]:
    if not embeddings_path.is_file():
        raise EvaluationError(f"POI Embedding 文件不存在：{embeddings_path}")
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if tuple(embeddings.shape) != expected_shape:
        raise EvaluationError(
            f"POI Embedding shape {embeddings.shape} != {expected_shape}"
        )
    progress = tqdm(total=expected_shape[0], desc="Gate 0 finite vectors", unit="row")
    for start in range(0, expected_shape[0], finite_chunk_rows):
        stop = min(start + finite_chunk_rows, expected_shape[0])
        if not np.isfinite(embeddings[start:stop]).all():
            raise EvaluationError(f"POI Embedding 在行区间 [{start}, {stop}) 存在 NaN/Inf")
        progress.update(stop - start)
    progress.close()

    sample_count = min(norm_sample_rows, expected_shape[0])
    sample_indices = np.linspace(
        0,
        expected_shape[0] - 1,
        num=sample_count,
        dtype=np.int64,
    )
    sample = np.asarray(embeddings[sample_indices], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)
    if not np.isfinite(norms).all():
        raise EvaluationError("POI Embedding 抽样 L2 范数存在 NaN/Inf")
    return embeddings, {
        "sample_rows": int(sample_count),
        "min": float(norms.min()),
        "mean": float(norms.mean()),
        "max": float(norms.max()),
    }


def run_gate_zero(
    config: dict[str, Any],
    model_config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    np.memmap,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    evaluation = _require_mapping(config.get("evaluation"), "evaluation")
    expected_eval_rows = int(evaluation["expected_eval_rows"])
    expected_poi_rows = int(evaluation["expected_poi_rows"])
    eval_path = _resolve(evaluation["eval_data"])
    records, eval_sha256 = load_eval_records(
        eval_path,
        expected_eval_rows,
        evaluation.get("expected_eval_sha256"),
    )
    reference_manifest_value = evaluation.get("reference_run_manifest")
    reference_manifest_path = (
        _resolve(reference_manifest_value) if reference_manifest_value else None
    )
    if reference_manifest_path is None:
        reference_order = {
            "status": "not_configured",
            "reason": "the frozen eval file and SHA256 define row alignment",
        }
        reference_mapping_path = None
    else:
        reference_order, reference_mapping_path = validate_reference_eval_order(
            records,
            eval_sha256,
            reference_manifest_path,
        )
    target_ids = {record["poi_id"] for record in records}

    poi_config_path = _resolve(model_config["poi_embedding_config"])
    poi_job_config = load_job_config(poi_config_path, PROJECT_ROOT)
    output_dir = poi_job_config.output.dir
    embeddings_path = output_dir / "embeddings.npy"
    ids_path = output_dir / "poi_ids.jsonl"
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise EvaluationError(f"POI manifest 不存在：{manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    embedding_dim = int(manifest.get("model", {}).get("embedding_dim", -1))
    expected_shape = (expected_poi_rows, embedding_dim)
    embeddings, norm_stats = validate_embeddings(
        embeddings_path,
        expected_shape,
        int(evaluation["finite_check_chunk_rows"]),
        int(evaluation["norm_sample_rows"]),
    )
    _validate_manifest(manifest, expected_poi_rows, tuple(embeddings.shape), poi_config_path)
    target_indices_by_id, unique_poi_ids, fingerprint = validate_poi_ids_and_alignment(
        poi_config_path,
        ids_path,
        manifest,
        target_ids,
        expected_poi_rows,
    )
    target_indices = np.fromiter(
        (target_indices_by_id[record["poi_id"]] for record in records),
        dtype=np.int64,
        count=len(records),
    )
    gate = {
        "status": "passed",
        "eval_rows": len(records),
        "eval_unique_order_ids": len({record["order_id"] for record in records}),
        "eval_non_empty_query_rows": sum(bool(record["query"].strip()) for record in records),
        "eval_non_empty_poi_id_rows": sum(bool(record["poi_id"].strip()) for record in records),
        "eval_sha256": eval_sha256,
        "e1_reference_order": reference_order,
        "poi_embedding_rows": int(embeddings.shape[0]),
        "poi_embedding_shape": list(embeddings.shape),
        "poi_embedding_dtype": str(embeddings.dtype),
        "poi_id_rows": expected_poi_rows,
        "poi_id_unique": unique_poi_ids,
        "poi_alignment": "all rows match sorted raw POI shards and manifest",
        "poi_input_fingerprint": fingerprint,
        "target_rows_present": int((target_indices >= 0).sum()),
        "target_unique_poi_ids_present": len(target_indices_by_id),
        "vectors_all_finite": True,
        "poi_l2_norm": norm_stats,
    }
    paths: dict[str, Path] = {
        "eval_data": eval_path,
        "poi_embedding_config": poi_config_path,
        "poi_embeddings": embeddings_path,
        "poi_ids": ids_path,
        "poi_manifest": manifest_path,
    }
    if reference_manifest_path is not None:
        paths["reference_manifest"] = reference_manifest_path
    if reference_mapping_path is not None:
        paths["reference_query_mapping"] = reference_mapping_path
    return records, embeddings, target_indices, gate, paths


def _instruction_text(
    query: str,
    instruction: str,
    instructions: dict[str, Any],
) -> str:
    if instruction == "none":
        return query
    template = instructions.get(instruction)
    if not isinstance(template, str) or "{query}" not in template:
        raise EvaluationError(f"Instruction 模板无效：{instruction}")
    return template.format(query=query)


def _pooling_description(encoder: Any) -> str:
    explicit = getattr(encoder, "pooling_description", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    pooling_modes = (
        ("pooling_mode_cls_token", "cls"),
        ("pooling_mode_mean_tokens", "mean"),
        ("pooling_mode_max_tokens", "max"),
        ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len"),
        ("pooling_mode_weightedmean_tokens", "weighted_mean"),
        ("pooling_mode_lasttoken", "last_token"),
    )
    for module in encoder:
        active_modes = [
            label
            for attribute, label in pooling_modes
            if bool(getattr(module, attribute, False))
        ]
        if active_modes:
            return "+".join(active_modes)
    return "unknown"


def encode_queries(
    records: list[dict[str, Any]],
    model_config: dict[str, Any],
    instruction: str,
    instructions: dict[str, Any],
    output_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    poi_config_path = _resolve(model_config["poi_embedding_config"])
    poi_job_config = load_job_config(poi_config_path, PROJECT_ROOT)
    query_model = replace(
        poi_job_config.model,
        batch_size=int(model_config["query_batch_size"]),
        encode_buffer_size=int(model_config["query_encode_buffer_size"]),
        max_seq_length=int(model_config["query_max_seq_length"]),
        prompt_name=None,
    )
    if query_model.device != "cuda":
        raise EvaluationError("Query 编码必须配置为 CUDA")
    if query_model.torch_dtype != "bfloat16":
        raise EvaluationError("Query 编码 dtype 必须为 bfloat16")
    if not query_model.normalize_embeddings:
        raise EvaluationError("Query 向量必须启用 L2 normalize")
    if not torch.cuda.is_available():
        raise EvaluationError("当前进程无法访问 CUDA，停止 Query 编码")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_load_started = time.perf_counter()
    encoder, device, model_load_seconds = _load_encoder(query_model)
    encoder.eval()
    if device != "cuda":
        raise EvaluationError(f"Query 编码实际设备不是 cuda：{device}")
    embedding_dim = encoder.get_sentence_embedding_dimension()
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise EvaluationError("无法确定 Query embedding 维度")
    pooling = _pooling_description(encoder)
    if pooling == "unknown":
        raise EvaluationError("无法确定 Query encoder 的 pooling 方法")
    model_load_wall_seconds = time.perf_counter() - model_load_started

    queries = [
        _instruction_text(record["query"], instruction, instructions)
        for record in records
    ]
    vectors = np.empty((len(queries), embedding_dim), dtype=np.float16)
    buffer_size = query_model.encode_buffer_size
    encode_started = time.perf_counter()
    progress = tqdm(total=len(queries), desc="Query embedding", unit="query")
    for start in range(0, len(queries), buffer_size):
        stop = min(start + buffer_size, len(queries))
        encoded = np.asarray(
            encoder.encode(
                queries[start:stop],
                batch_size=query_model.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=query_model.normalize_embeddings,
                prompt_name=None,
            )
        )
        if encoded.shape != (stop - start, embedding_dim):
            raise EvaluationError(f"Query encoder 返回异常 shape：{encoded.shape}")
        vectors[start:stop] = encoded.astype(np.float16, copy=False)
        progress.update(stop - start)
    progress.close()
    encoding_seconds = time.perf_counter() - encode_started
    if not np.isfinite(vectors).all():
        raise EvaluationError("Query embedding 存在 NaN/Inf")
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    _save_npy_atomic(output_path, vectors)
    metrics = {
        "shape": list(vectors.shape),
        "saved_dtype": str(vectors.dtype),
        "model_dtype": query_model.torch_dtype,
        "device": device,
        "batch_size": query_model.batch_size,
        "encode_buffer_size": query_model.encode_buffer_size,
        "max_seq_length": query_model.max_seq_length,
        "padding_side": query_model.padding_side,
        "pooling": pooling,
        "normalize_embeddings": query_model.normalize_embeddings,
        "instruction": instruction,
        "model_load_seconds": model_load_seconds,
        "model_load_wall_seconds": model_load_wall_seconds,
        "encoding_seconds": encoding_seconds,
        "rows_per_second": len(queries) / encoding_seconds,
        "l2_norm_min": float(norms.min()),
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_max": float(norms.max()),
        "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "cuda_device": torch.cuda.get_device_name(),
        "model_config": {
            **asdict(query_model),
            "path": str(query_model.path),
        },
    }
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    del encoder
    del queries
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_after_release = int(torch.cuda.memory_allocated())
    reserved_after_release = int(torch.cuda.memory_reserved())
    free_after_release, total_memory = torch.cuda.mem_get_info()
    if allocated_after_release > 512 * 1024**2:
        raise EvaluationError(
            "Query 模型卸载后仍有超过 512 MiB 的 PyTorch CUDA 内存未释放"
        )
    metrics.update(
        {
            "cuda_peak_memory_allocated_bytes": peak_allocated,
            "cuda_peak_memory_reserved_bytes": peak_reserved,
            "cuda_memory_allocated_after_release_bytes": allocated_after_release,
            "cuda_memory_reserved_after_release_bytes": reserved_after_release,
            "cuda_memory_free_after_release_bytes": int(free_after_release),
            "cuda_memory_total_bytes": int(total_memory),
            "model_and_tokenizer_released": True,
        }
    )
    return vectors, metrics


def write_query_mapping(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row_index, record in enumerate(records):
            mapping = {
                "row_index": row_index,
                "order_id": record["order_id"],
                "searchid": record.get("searchid"),
                "target_poi_id": record["poi_id"],
            }
            handle.write(json.dumps(mapping, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def exact_faiss_search(
    poi_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    faiss_config: dict[str, Any],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import faiss
    import torch

    if faiss_config.get("type") != "IndexFlatIP":
        raise EvaluationError("本轮只允许 Faiss IndexFlatIP")
    if faiss_config.get("device") != "gpu":
        raise EvaluationError("当前配置必须使用 Faiss GPU")
    gpu_id = int(faiss_config.get("gpu_id", 0))
    gpu_count = faiss.get_num_gpus()
    if gpu_id < 0 or gpu_id >= gpu_count:
        raise EvaluationError(
            f"Faiss GPU {gpu_id} 不可用，当前仅检测到 {gpu_count} 张 GPU"
        )
    if not hasattr(faiss, "StandardGpuResources"):
        raise EvaluationError("当前 Faiss 未提供 GPU 接口")
    use_float16 = bool(faiss_config.get("use_float16", False))
    if use_float16:
        raise EvaluationError("精确召回要求 Faiss GPU 使用 float32 存储")

    dimension = int(poi_embeddings.shape[1])
    index_storage_estimated_bytes = int(
        len(poi_embeddings) * dimension * np.dtype(np.float32).itemsize
    )
    memory_headroom_bytes = int(
        float(faiss_config.get("memory_headroom_gib", 4.0)) * 1024**3
    )
    build_strategy = faiss_config.get("build_strategy")
    if build_strategy != "gpu_two_batch_add":
        raise EvaluationError(
            "float32 索引必须使用 gpu_two_batch_add，避免末次扩容超过显存"
        )
    initial_add_rows = min(
        int(faiss_config.get("initial_add_rows", 1_000_000)),
        len(poi_embeddings),
    )
    if initial_add_rows <= 0:
        raise EvaluationError("faiss.initial_add_rows 必须大于 0")
    expansion_peak_estimated_bytes = int(
        index_storage_estimated_bytes
        + initial_add_rows * dimension * np.dtype(np.float32).itemsize
    )
    free_before, total_memory = torch.cuda.mem_get_info(gpu_id)
    required_memory = expansion_peak_estimated_bytes + memory_headroom_bytes
    if free_before < required_memory:
        raise EvaluationError(
            "Faiss GPU float32 精确索引显存不足："
            f"可用 {free_before / 1024**3:.2f} GiB，"
            f"索引加安全余量需要 {required_memory / 1024**3:.2f} GiB"
        )
    resources = faiss.StandardGpuResources()
    index_config = faiss.GpuIndexFlatConfig()
    index_config.device = gpu_id
    index_config.useFloat16 = False
    index = faiss.GpuIndexFlatIP(resources, dimension, index_config)
    minimum_free_memory = free_before
    build_started = time.perf_counter()
    add_ranges = [(0, initial_add_rows)]
    if initial_add_rows < len(poi_embeddings):
        add_ranges.append((initial_add_rows, len(poi_embeddings)))
    progress = tqdm(total=len(poi_embeddings), desc="Faiss GPU add", unit="vector")
    add_batch_rows: list[int] = []
    for start, stop in add_ranges:
        batch = np.ascontiguousarray(poi_embeddings[start:stop], dtype=np.float32)
        index.add(batch)
        del batch
        current_free, _ = torch.cuda.mem_get_info(gpu_id)
        minimum_free_memory = min(minimum_free_memory, current_free)
        add_batch_rows.append(stop - start)
        progress.update(stop - start)
    progress.close()
    resources.syncDefaultStreamCurrentDevice()
    build_seconds = time.perf_counter() - build_started
    if type(index).__name__ != "GpuIndexFlatIP":
        raise EvaluationError(
            f"Faiss 索引类型异常：{type(index).__name__}，预期 GpuIndexFlatIP"
        )
    if index.ntotal != len(poi_embeddings):
        raise EvaluationError(
            f"Faiss GPU 索引向量数 {index.ntotal} != {len(poi_embeddings)}"
        )

    query_batch_size = int(faiss_config["query_batch_size"])
    topk_scores = np.empty((len(query_embeddings), top_k), dtype=np.float32)
    topk_indices = np.empty((len(query_embeddings), top_k), dtype=np.int64)
    search_started = time.perf_counter()
    progress = tqdm(total=len(query_embeddings), desc="Faiss GPU search", unit="query")
    for start in range(0, len(query_embeddings), query_batch_size):
        stop = min(start + query_batch_size, len(query_embeddings))
        batch = np.ascontiguousarray(query_embeddings[start:stop], dtype=np.float32)
        scores, indices = index.search(batch, top_k)
        topk_scores[start:stop] = scores
        topk_indices[start:stop] = indices
        current_free, _ = torch.cuda.mem_get_info(gpu_id)
        minimum_free_memory = min(minimum_free_memory, current_free)
        progress.update(stop - start)
    progress.close()
    search_seconds = time.perf_counter() - search_started
    return topk_indices, topk_scores, {
        "type": type(index).__name__,
        "device": "gpu",
        "gpu_id": gpu_id,
        "gpu_name": torch.cuda.get_device_name(gpu_id),
        "gpu_count": gpu_count,
        "use_float16": False,
        "ntotal": int(index.ntotal),
        "dimension": dimension,
        "top_k": top_k,
        "add_batch_rows": add_batch_rows,
        "query_batch_size": query_batch_size,
        "build_strategy": build_strategy,
        "initial_add_rows": initial_add_rows,
        "build_seconds": build_seconds,
        "search_seconds": search_seconds,
        "index_storage_estimated_bytes": index_storage_estimated_bytes,
        "index_expansion_peak_estimated_bytes": expansion_peak_estimated_bytes,
        "memory_headroom_bytes": memory_headroom_bytes,
        "memory_required_bytes": required_memory,
        "memory_check_status": "passed",
        "gpu_memory_total_bytes": int(total_memory),
        "gpu_memory_free_before_bytes": int(free_before),
        "gpu_memory_min_free_bytes": int(minimum_free_memory),
        "gpu_memory_peak_delta_bytes": int(
            max(0, free_before - minimum_free_memory)
        ),
        "faiss_version": faiss.__version__,
    }


def _metric_subset(target_ranks: np.ndarray) -> dict[str, Any]:
    sample_count = len(target_ranks)
    if sample_count == 0:
        return {
            "samples": 0,
            "hit_at_1": None,
            "hit_at_5": None,
            "hit_at_10": None,
            "mrr_at_10": None,
        }
    valid_at_10 = (target_ranks >= 1) & (target_ranks <= 10)
    return {
        "samples": sample_count,
        "hit_at_1": float(np.mean((target_ranks >= 1) & (target_ranks <= 1))),
        "hit_at_5": float(np.mean((target_ranks >= 1) & (target_ranks <= 5))),
        "hit_at_10": float(np.mean(valid_at_10)),
        "mrr_at_10": float(
            np.mean(np.where(valid_at_10, 1.0 / target_ranks.clip(min=1), 0.0))
        ),
    }


def compute_metrics(
    records: list[dict[str, Any]],
    topk_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    matches = topk_indices == target_indices[:, None]
    matched = matches.any(axis=1)
    target_ranks = np.full(len(records), -1, dtype=np.int16)
    target_ranks[matched] = matches[matched].argmax(axis=1).astype(np.int16) + 1
    metrics = {
        "samples": len(records),
        "hit_at_1": float(np.mean((target_ranks >= 1) & (target_ranks <= 1))),
        "hit_at_3": float(np.mean((target_ranks >= 1) & (target_ranks <= 3))),
        "hit_at_5": float(np.mean((target_ranks >= 1) & (target_ranks <= 5))),
        "hit_at_10": float(np.mean((target_ranks >= 1) & (target_ranks <= 10))),
        "hit_at_20": float(np.mean((target_ranks >= 1) & (target_ranks <= 20))),
        "mrr_at_10": float(
            np.mean(
                np.where(
                    (target_ranks >= 1) & (target_ranks <= 10),
                    1.0 / target_ranks.clip(min=1),
                    0.0,
                )
            )
        ),
    }

    lengths = np.fromiter(
        (len(record["query"].strip()) for record in records),
        dtype=np.int32,
        count=len(records),
    )
    buckets = {
        "1-2": (lengths >= 1) & (lengths <= 2),
        "3-5": (lengths >= 3) & (lengths <= 5),
        "6-10": (lengths >= 6) & (lengths <= 10),
        ">10": lengths > 10,
    }
    metrics["query_length_buckets"] = {
        name: _metric_subset(target_ranks[mask]) for name, mask in buckets.items()
    }
    return target_ranks, metrics


def build_reference_comparison(
    reference_manifest_path: Path,
    current_run: str,
    current_metrics: dict[str, Any],
    current_query_metrics: dict[str, Any],
    current_faiss_metrics: dict[str, Any],
    current_total_seconds: float,
) -> dict[str, Any]:
    with reference_manifest_path.open("r", encoding="utf-8") as handle:
        reference_manifest = json.load(handle)
    reference_metrics_path = _resolve(
        reference_manifest["outputs"]["metrics"]
    )
    with reference_metrics_path.open("r", encoding="utf-8") as handle:
        reference_metrics = json.load(handle)

    metric_names = (
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "hit_at_20",
        "mrr_at_10",
    )
    bucket_metric_names = ("samples", "hit_at_1", "hit_at_5", "hit_at_10", "mrr_at_10")
    overall = {
        name: {
            "e1": reference_metrics[name],
            "e2": current_metrics[name],
            "delta": current_metrics[name] - reference_metrics[name],
        }
        for name in metric_names
    }
    buckets: dict[str, Any] = {}
    for bucket_name in ("1-2", "3-5", "6-10", ">10"):
        e1_bucket = reference_metrics["query_length_buckets"][bucket_name]
        e2_bucket = current_metrics["query_length_buckets"][bucket_name]
        buckets[bucket_name] = {
            name: {
                "e1": e1_bucket[name],
                "e2": e2_bucket[name],
                "delta": e2_bucket[name] - e1_bucket[name],
            }
            for name in bucket_metric_names
        }

    reference_timing = reference_manifest["timing_seconds"]
    e1_peak_memory = max(
        int(reference_manifest["cuda_memory"]["query_torch_peak_allocated_bytes"]),
        int(reference_manifest["cuda_memory"]["faiss_observed_peak_delta_bytes"]),
    )
    e2_peak_memory = max(
        int(current_query_metrics["cuda_peak_memory_allocated_bytes"]),
        int(current_faiss_metrics["gpu_memory_peak_delta_bytes"]),
    )
    return {
        "reference_run": "qwen3_0.6b_no_instruction",
        "current_run": current_run,
        "model_selection": "not_performed",
        "overall_metrics": overall,
        "query_length_buckets": buckets,
        "timing_seconds": {
            "query_encoding": {
                "e1": reference_timing["query_encoding"],
                "e2": current_query_metrics["encoding_seconds"],
            },
            "index_build": {
                "e1": reference_timing["index_build"],
                "e2": current_faiss_metrics["build_seconds"],
            },
            "retrieval": {
                "e1": reference_timing["retrieval"],
                "e2": current_faiss_metrics["search_seconds"],
            },
            "total": {
                "e1": reference_timing["total"],
                "e2": current_total_seconds,
            },
        },
        "peak_cuda_memory_bytes": {"e1": e1_peak_memory, "e2": e2_peak_memory},
    }


def _git_state() -> dict[str, Any]:
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


def resolve_output_paths(
    evaluation: dict[str, Any],
    model_name: str,
    instruction: str,
) -> tuple[Path, Path, Path]:
    suffix = "no_instruction" if instruction == "none" else instruction
    run_name = f"{model_name}_{suffix}"
    output_root = _resolve(evaluation["output_root"])
    query_root = output_root / "query_embeddings"
    return (
        query_root / f"{run_name}.npy",
        query_root / f"{run_name}_rows.jsonl",
        output_root / run_name,
    )


def main() -> int:
    args = parse_args()
    started_at = _utc_now()
    total_started = time.perf_counter()
    config_path = _resolve(args.config)
    config = _load_yaml(config_path)
    models = _require_mapping(config.get("models"), "models")
    model_config = _require_mapping(models.get(args.model), f"models.{args.model}")
    instructions = _require_mapping(config.get("instructions"), "instructions")
    if args.instruction not in instructions:
        raise EvaluationError(f"配置不支持 instruction={args.instruction}")
    faiss_config = _require_mapping(config.get("faiss"), "faiss")
    evaluation = _require_mapping(config.get("evaluation"), "evaluation")

    query_embeddings_path, query_mapping_path, output_dir = resolve_output_paths(
        evaluation,
        args.model,
        args.instruction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "model": args.model,
        "instruction": args.instruction,
        "config": str(config_path),
        "git": _git_state(),
    }
    _write_json_atomic(manifest_path, manifest)

    try:
        gate_started = time.perf_counter()
        records, poi_embeddings, target_indices, gate, paths = run_gate_zero(
            config,
            model_config,
        )
        gate_seconds = time.perf_counter() - gate_started
        manifest["gate_zero"] = gate
        manifest["gate_zero_seconds"] = gate_seconds
        manifest["inputs"] = {name: str(path) for name, path in paths.items()}
        _write_json_atomic(manifest_path, manifest)

        query_embeddings, query_metrics = encode_queries(
            records,
            model_config,
            args.instruction,
            instructions,
            query_embeddings_path,
        )
        write_query_mapping(query_mapping_path, records)

        top_k = int(evaluation["top_k"])
        topk_indices, topk_scores, faiss_metrics = exact_faiss_search(
            poi_embeddings,
            query_embeddings,
            faiss_config,
            top_k,
        )
        target_ranks, metrics = compute_metrics(
            records,
            topk_indices,
            target_indices,
        )
        metrics_path = output_dir / "metrics.json"
        results_path = output_dir / "retrieval_results.npz"
        _write_json_atomic(metrics_path, metrics)
        _save_npz_atomic(
            results_path,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            target_ranks=target_ranks,
        )

        total_seconds = time.perf_counter() - total_started
        reference_manifest_value = evaluation.get("reference_run_manifest")
        comparison = None
        if reference_manifest_value:
            comparison = build_reference_comparison(
                _resolve(reference_manifest_value),
                output_dir.name,
                metrics,
                query_metrics,
                faiss_metrics,
                total_seconds,
            )
        manifest.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "query_encoding": query_metrics,
                "faiss": faiss_metrics,
                "comparison_with_e1": comparison,
                "timing_seconds": {
                    "gate_zero": gate_seconds,
                    "query_model_load": query_metrics["model_load_seconds"],
                    "query_encoding": query_metrics["encoding_seconds"],
                    "index_build": faiss_metrics["build_seconds"],
                    "retrieval": faiss_metrics["search_seconds"],
                    "total": total_seconds,
                },
                "cuda_peak_memory_allocated_bytes": query_metrics[
                    "cuda_peak_memory_allocated_bytes"
                ],
                "cuda_peak_memory_reserved_bytes": query_metrics[
                    "cuda_peak_memory_reserved_bytes"
                ],
                "cuda_memory": {
                    "query_torch_peak_allocated_bytes": query_metrics[
                        "cuda_peak_memory_allocated_bytes"
                    ],
                    "query_torch_peak_reserved_bytes": query_metrics[
                        "cuda_peak_memory_reserved_bytes"
                    ],
                    "faiss_observed_peak_delta_bytes": faiss_metrics[
                        "gpu_memory_peak_delta_bytes"
                    ],
                },
                "outputs": {
                    "query_embeddings": str(query_embeddings_path),
                    "query_mapping": str(query_mapping_path),
                    "metrics": str(metrics_path),
                    "retrieval_results": str(results_path),
                    "query_embeddings_sha256": _sha256_file(query_embeddings_path),
                    "query_mapping_sha256": _sha256_file(query_mapping_path),
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
        _write_json_atomic(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "metrics": metrics,
                    "timing_seconds": manifest["timing_seconds"],
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "total_seconds": time.perf_counter() - total_started,
            }
        )
        _write_json_atomic(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

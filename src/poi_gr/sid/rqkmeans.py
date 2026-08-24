"""Streaming residual K-Means construction for aligned POI embeddings."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from tqdm import tqdm

from .evaluation import (
    compute_basic_metrics,
    compute_layer_metrics,
    evaluate_sid,
    load_aligned_category_codes,
    write_evaluation_outputs,
)
from .training import create_fixed_indices


class RQKMeansError(RuntimeError):
    pass


@dataclass(frozen=True)
class RQKMeansConfig:
    source_config: Path
    experiment_id: str
    embeddings_path: Path
    poi_ids_path: Path
    embedding_manifest_path: Path
    poi_data_path: Path
    output_dir: Path
    input_dim: int
    codebook_sizes: tuple[int, ...]
    first_level_mode: str
    implementation: str
    backend: str
    sample_size: int
    iterations: int
    nredo: int
    max_points_per_centroid: int
    max_beam_size: int
    progressive_dim: bool
    refine_codebook: bool
    codebook_refine_iterations: int
    use_beam_lut: bool
    max_memory_mib: int
    omp_threads: int
    seed: int
    validation_ratio: float
    chunk_rows: int
    sample_chunk_rows: int
    sample_storage: str
    gpu_temp_memory_mib: int
    finite_check_chunk_rows: int
    resume: bool
    show_progress: bool
    evaluation_max_cases: int
    evaluation_max_pois_per_case: int
    expected_rows: int | None
    expected_embedding_fingerprint: str | None
    expected_poi_ids_sha256: str | None
    expected_validation_indices_sha256: str | None
    expected_sample_indices_sha256: str | None
    expected_category_count: int | None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RQKMeansError(f"{name} 必须是 YAML mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RQKMeansError(f"{name} 必须是正整数")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RQKMeansError(f"{name} 必须是非空字符串")
    return value


def _path(value: Any, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RQKMeansError(f"{name} 必须是非空路径")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_rqkmeans_config(
    config_path: Path,
    project_root: Path,
    *,
    output_dir: Path | None = None,
    resume: bool | None = None,
    show_progress: bool | None = None,
    implementation: str | None = None,
    backend: str | None = None,
    sample_size: int | None = None,
    iterations: int | None = None,
    nredo: int | None = None,
    max_beam_size: int | None = None,
    codebook_sizes: Sequence[int] | None = None,
) -> RQKMeansConfig:
    """Load and validate one fixed residual K-Means experiment."""

    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = _mapping(yaml.safe_load(handle), "配置根节点")
    data = _mapping(raw.get("data"), "data")
    quantizer = _mapping(raw.get("quantizer"), "quantizer")
    runtime = _mapping(raw.get("runtime"), "runtime")
    evaluation = _mapping(raw.get("evaluation"), "evaluation")
    expected = _mapping(raw.get("expected", {}), "expected")

    experiment_id = raw.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise RQKMeansError("experiment_id 必须是非空字符串")
    declared_codebook_sizes = quantizer.get("codebook_sizes")
    if not isinstance(declared_codebook_sizes, list) or not declared_codebook_sizes:
        raise RQKMeansError("quantizer.codebook_sizes 必须是非空正整数数组")
    sizes = tuple(
        _positive_int(value, "quantizer.codebook_sizes")
        for value in declared_codebook_sizes
    )
    declared_implementation = quantizer.get(
        "implementation", "sequential_kmeans"
    )
    if declared_implementation not in {
        "sequential_kmeans",
        "faiss_residual_quantizer",
    }:
        raise RQKMeansError(
            "quantizer.implementation 必须是 sequential_kmeans/"
            "faiss_residual_quantizer"
        )
    declared_backend = quantizer.get("backend")
    if declared_backend not in {"faiss_gpu", "faiss_cpu", "sklearn"}:
        raise RQKMeansError(
            "quantizer.backend 必须是 faiss_gpu/faiss_cpu/sklearn"
        )
    validation_ratio = quantizer.get("validation_ratio")
    if (
        not isinstance(validation_ratio, (int, float))
        or isinstance(validation_ratio, bool)
        or not 0.0 < float(validation_ratio) < 1.0
    ):
        raise RQKMeansError("quantizer.validation_ratio 必须位于 (0, 1)")
    first_level_mode = str(quantizer.get("first_level_mode", "kmeans"))
    if first_level_mode not in {"kmeans", "category_code"}:
        raise RQKMeansError(
            "quantizer.first_level_mode 必须是 kmeans/category_code"
        )
    if (
        first_level_mode == "category_code"
        and declared_implementation != "sequential_kmeans"
    ):
        raise RQKMeansError(
            "category_code 第一层只支持 sequential_kmeans"
        )
    resolved_output = (
        output_dir.resolve()
        if output_dir is not None
        else _path(raw.get("output_dir"), project_root, "output_dir")
    )
    config = RQKMeansConfig(
        source_config=config_path,
        experiment_id=experiment_id,
        embeddings_path=_path(data.get("embeddings"), project_root, "data.embeddings"),
        poi_ids_path=_path(data.get("poi_ids"), project_root, "data.poi_ids"),
        embedding_manifest_path=_path(
            data.get("embedding_manifest"), project_root, "data.embedding_manifest"
        ),
        poi_data_path=_path(data.get("poi_data"), project_root, "data.poi_data"),
        output_dir=resolved_output,
        input_dim=_positive_int(data.get("input_dim"), "data.input_dim"),
        codebook_sizes=sizes,
        first_level_mode=first_level_mode,
        implementation=declared_implementation,
        backend=declared_backend,
        sample_size=_positive_int(quantizer.get("sample_size"), "quantizer.sample_size"),
        iterations=_positive_int(quantizer.get("iterations"), "quantizer.iterations"),
        nredo=_positive_int(quantizer.get("nredo", 1), "quantizer.nredo"),
        max_points_per_centroid=_positive_int(
            quantizer.get("max_points_per_centroid"),
            "quantizer.max_points_per_centroid",
        ),
        max_beam_size=_positive_int(
            quantizer.get("max_beam_size", 1), "quantizer.max_beam_size"
        ),
        progressive_dim=bool(quantizer.get("progressive_dim", False)),
        refine_codebook=bool(quantizer.get("refine_codebook", False)),
        codebook_refine_iterations=_positive_int(
            quantizer.get("codebook_refine_iterations", 5),
            "quantizer.codebook_refine_iterations",
        ),
        use_beam_lut=bool(quantizer.get("use_beam_lut", False)),
        max_memory_mib=_positive_int(
            quantizer.get("max_memory_mib", 4096),
            "quantizer.max_memory_mib",
        ),
        omp_threads=_positive_int(
            quantizer.get("omp_threads", 64), "quantizer.omp_threads"
        ),
        seed=_positive_int(quantizer.get("seed"), "quantizer.seed"),
        validation_ratio=float(validation_ratio),
        chunk_rows=_positive_int(runtime.get("chunk_rows"), "runtime.chunk_rows"),
        sample_chunk_rows=_positive_int(
            runtime.get("sample_chunk_rows"), "runtime.sample_chunk_rows"
        ),
        sample_storage=str(runtime.get("sample_storage", "memory")),
        gpu_temp_memory_mib=_positive_int(
            runtime.get("gpu_temp_memory_mib"), "runtime.gpu_temp_memory_mib"
        ),
        finite_check_chunk_rows=_positive_int(
            runtime.get("finite_check_chunk_rows"),
            "runtime.finite_check_chunk_rows",
        ),
        resume=bool(runtime.get("resume")),
        show_progress=bool(runtime.get("show_progress")),
        evaluation_max_cases=_positive_int(
            evaluation.get("max_cases"), "evaluation.max_cases"
        ),
        evaluation_max_pois_per_case=_positive_int(
            evaluation.get("max_pois_per_case"),
            "evaluation.max_pois_per_case",
        ),
        expected_rows=_optional_positive_int(expected.get("rows"), "expected.rows"),
        expected_embedding_fingerprint=_optional_string(
            expected.get("embedding_fingerprint"),
            "expected.embedding_fingerprint",
        ),
        expected_poi_ids_sha256=_optional_string(
            expected.get("poi_ids_sha256"), "expected.poi_ids_sha256"
        ),
        expected_validation_indices_sha256=_optional_string(
            expected.get("validation_indices_sha256"),
            "expected.validation_indices_sha256",
        ),
        expected_sample_indices_sha256=_optional_string(
            expected.get("sample_indices_sha256"),
            "expected.sample_indices_sha256",
        ),
        expected_category_count=_optional_positive_int(
            expected.get("category_count"), "expected.category_count"
        ),
    )
    if config.sample_storage not in {"memory", "mmap"}:
        raise RQKMeansError("runtime.sample_storage 必须是 memory/mmap")
    overridden_sizes = (
        config.codebook_sizes
        if codebook_sizes is None
        else tuple(
            _positive_int(value, "codebook_sizes") for value in codebook_sizes
        )
    )
    resolved_implementation = implementation or config.implementation
    resolved_backend = backend or config.backend
    if (
        config.first_level_mode == "category_code"
        and resolved_implementation != "sequential_kmeans"
    ):
        raise RQKMeansError(
            "category_code 第一层只支持 sequential_kmeans"
        )
    if resolved_implementation == "faiss_residual_quantizer":
        if resolved_backend != "faiss_cpu":
            raise RQKMeansError(
                "faiss_residual_quantizer 当前使用 Faiss CPU codec，backend 必须为 faiss_cpu"
            )
        if any(size & (size - 1) for size in overridden_sizes):
            raise RQKMeansError(
                "faiss_residual_quantizer 的各层码本必须是 2 的幂"
            )
    return replace(
        config,
        resume=config.resume if resume is None else resume,
        show_progress=config.show_progress if show_progress is None else show_progress,
        implementation=resolved_implementation,
        backend=resolved_backend,
        sample_size=(
            config.sample_size
            if sample_size is None
            else _positive_int(sample_size, "sample_size")
        ),
        iterations=(
            config.iterations
            if iterations is None
            else _positive_int(iterations, "iterations")
        ),
        nredo=config.nredo if nredo is None else _positive_int(nredo, "nredo"),
        max_beam_size=(
            config.max_beam_size
            if max_beam_size is None
            else _positive_int(max_beam_size, "max_beam_size")
        ),
        codebook_sizes=overridden_sizes,
        expected_sample_indices_sha256=(
            config.expected_sample_indices_sha256
            if sample_size is None or sample_size == config.sample_size
            else None
        ),
    )


def _json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _npy_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _npy_atomic_with_sha256(path: Path, values: np.ndarray) -> str:
    """Write one NPY atomically and hash the exact bytes without a cold reread."""

    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    payload = buffer.getbuffer()
    digest = hashlib.sha256(payload).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        payload.release()
        buffer.close()
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes(order="C")
    ).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_payload(config: RQKMeansConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else list(value)
        if isinstance(value, tuple)
        else value
        for key, value in payload.items()
    }


def _config_signature(config: RQKMeansConfig) -> str:
    payload = _config_payload(config)
    for key in ("resume", "show_progress"):
        payload.pop(key)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
        return {"revision": revision, "worktree_status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "worktree_status": None}


def validate_rqkmeans_inputs(config: RQKMeansConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate aligned artifacts while keeping the embedding array memory-mapped."""

    for path, name in (
        (config.embeddings_path, "Embedding"),
        (config.poi_ids_path, "POI ID"),
        (config.embedding_manifest_path, "Embedding manifest"),
    ):
        if not path.is_file():
            raise RQKMeansError(f"{name} 不存在：{path}")
    if not config.poi_data_path.exists():
        raise RQKMeansError(f"POI 数据不存在：{config.poi_data_path}")
    try:
        embeddings = np.load(
            config.embeddings_path, mmap_mode="r", allow_pickle=False
        )
    except (OSError, ValueError) as error:
        raise RQKMeansError("Embedding NPY 读取失败") from error
    if embeddings.ndim != 2 or embeddings.shape[1] != config.input_dim:
        raise RQKMeansError(
            f"Embedding shape {embeddings.shape} 与输入维度 {config.input_dim} 不一致"
        )
    row_count = int(embeddings.shape[0])
    if config.expected_rows is not None and row_count != config.expected_rows:
        raise RQKMeansError(
            f"Embedding 行数 {row_count} != expected.rows {config.expected_rows}"
        )
    with config.embedding_manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_rows = int(manifest.get("input", {}).get("total_rows", -1))
    manifest_shape = manifest.get("output", {}).get("shape")
    fingerprint = manifest.get("input", {}).get("fingerprint")
    if manifest.get("status") != "completed":
        raise RQKMeansError("Embedding manifest 状态不是 completed")
    if manifest_rows != row_count or manifest_shape != list(embeddings.shape):
        raise RQKMeansError("Embedding manifest 的行数或 shape 与 NPY 不一致")
    if (
        config.expected_embedding_fingerprint is not None
        and fingerprint != config.expected_embedding_fingerprint
    ):
        raise RQKMeansError("Embedding fingerprint 与冻结配置不一致")

    for start in tqdm(
        range(0, row_count, config.finite_check_chunk_rows),
        desc="Embedding finite check",
        unit="poi",
        disable=not config.show_progress,
    ):
        stop = min(start + config.finite_check_chunk_rows, row_count)
        if not np.isfinite(embeddings[start:stop]).all():
            raise RQKMeansError(f"Embedding 行区间 [{start}, {stop}) 存在 NaN/Inf")

    poi_digest = hashlib.sha256()
    poi_rows = 0
    with config.poi_ids_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            poi_digest.update(raw_line)
            try:
                poi_id = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RQKMeansError(f"POI ID 第 {line_number} 行无效") from error
            if not isinstance(poi_id, str) or not poi_id.strip():
                raise RQKMeansError(f"POI ID 第 {line_number} 行不是非空字符串")
            poi_rows += 1
    poi_sha256 = poi_digest.hexdigest()
    if poi_rows != row_count:
        raise RQKMeansError(f"POI ID 行数 {poi_rows} 与 Embedding 行数 {row_count} 不一致")
    if (
        config.expected_poi_ids_sha256 is not None
        and poi_sha256 != config.expected_poi_ids_sha256
    ):
        raise RQKMeansError("POI ID SHA256 与冻结配置不一致")
    return embeddings, {
        "status": "passed",
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_fingerprint": fingerprint,
        "embedding_manifest": str(config.embedding_manifest_path),
        "poi_id_rows": poi_rows,
        "poi_ids_sha256": poi_sha256,
        "vectors_all_finite": True,
    }


def _fit_kmeans(
    values: np.ndarray,
    cluster_count: int,
    *,
    iterations: int,
    nredo: int,
    seed: int,
    backend: str,
    max_points_per_centroid: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if backend == "sklearn":
        from sklearn.cluster import KMeans

        estimator = KMeans(
            n_clusters=cluster_count,
            init="k-means++",
            n_init=nredo,
            max_iter=iterations,
            random_state=seed,
            algorithm="lloyd",
        )
        labels = estimator.fit_predict(np.asarray(values, dtype=np.float32))
        centroids = np.asarray(estimator.cluster_centers_, dtype=np.float32)
        residual = np.asarray(values, dtype=np.float32) - centroids[labels]
        return centroids, labels.astype(np.int32), float(
            np.mean(np.sum(residual * residual, axis=1))
        )
    try:
        import faiss
    except ImportError as error:
        raise RQKMeansError("Faiss KMeans backend 不可用") from error
    gpu = backend == "faiss_gpu"
    if gpu and faiss.get_num_gpus() < 1:
        raise RQKMeansError("faiss_gpu 要求当前进程可见至少一张 GPU")
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    kmeans = faiss.Kmeans(
        contiguous.shape[1],
        cluster_count,
        niter=iterations,
        nredo=nredo,
        verbose=True,
        seed=seed,
        gpu=gpu,
        min_points_per_centroid=1,
        max_points_per_centroid=max_points_per_centroid,
    )
    kmeans.train(contiguous)
    distances, labels = kmeans.index.search(contiguous, 1)
    return (
        np.asarray(kmeans.centroids, dtype=np.float32),
        labels[:, 0].astype(np.int32, copy=False),
        float(distances[:, 0].mean()),
    )


def _assign_numpy(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(values * values, axis=1, keepdims=True)
        + np.sum(centroids * centroids, axis=1)[None, :]
        - 2.0 * values @ centroids.T
    )
    return np.argmin(squared, axis=1).astype(np.int32, copy=False)


def _build_assignment_indices(
    codebooks: Sequence[np.ndarray], config: RQKMeansConfig
) -> tuple[list[Any] | None, Any | None]:
    if config.backend == "sklearn":
        return None, None
    try:
        import faiss
    except ImportError as error:
        raise RQKMeansError("Faiss assignment backend 不可用") from error
    resources = None
    if config.backend == "faiss_gpu":
        if faiss.get_num_gpus() < 1:
            raise RQKMeansError("faiss_gpu 编码要求当前进程可见至少一张 GPU")
        resources = faiss.StandardGpuResources()
        resources.setTempMemory(config.gpu_temp_memory_mib * 1024 * 1024)
    indices: list[Any] = []
    for centroids in codebooks:
        cpu_index = faiss.IndexFlatL2(config.input_dim)
        cpu_index.add(np.ascontiguousarray(centroids, dtype=np.float32))
        indices.append(
            faiss.index_cpu_to_gpu(resources, 0, cpu_index)
            if resources is not None
            else cpu_index
        )
    return indices, resources


def _assign_codes(
    values: np.ndarray,
    centroids: np.ndarray,
    index: Any | None,
) -> np.ndarray:
    if index is None:
        return _assign_numpy(values, centroids)
    _, labels = index.search(np.ascontiguousarray(values, dtype=np.float32), 1)
    return labels[:, 0].astype(np.int32, copy=False)


def _codebook_nbits(codebook_sizes: Sequence[int]) -> list[int]:
    return [int(size).bit_length() - 1 for size in codebook_sizes]


def _new_faiss_residual_index(config: RQKMeansConfig) -> Any:
    try:
        import faiss
    except ImportError as error:
        raise RQKMeansError("Faiss ResidualQuantizer 不可用") from error
    nbits = faiss.UInt64Vector()
    for value in _codebook_nbits(config.codebook_sizes):
        nbits.push_back(value)
    index = faiss.IndexResidualQuantizer(config.input_dim, nbits)
    quantizer = index.rq
    quantizer.cp.niter = config.iterations
    quantizer.cp.nredo = config.nredo
    quantizer.cp.seed = config.seed
    quantizer.cp.min_points_per_centroid = 1
    quantizer.cp.max_points_per_centroid = config.max_points_per_centroid
    quantizer.max_beam_size = config.max_beam_size
    quantizer.use_beam_LUT = int(config.use_beam_lut)
    quantizer.max_mem_distances = config.max_memory_mib * 1024 * 1024
    quantizer.niter_codebook_refine = config.codebook_refine_iterations
    train_type = (
        faiss.ResidualQuantizer.Train_progressive_dim
        if config.progressive_dim
        else faiss.ResidualQuantizer.Train_default
    )
    if config.refine_codebook:
        train_type |= faiss.ResidualQuantizer.Train_refine_codebook
    quantizer.train_type = train_type
    quantizer.verbose = config.show_progress
    faiss.omp_set_num_threads(config.omp_threads)
    return index


def _load_faiss_residual_index(path: Path, config: RQKMeansConfig) -> Any:
    try:
        import faiss
    except ImportError as error:
        raise RQKMeansError("Faiss ResidualQuantizer 不可用") from error
    index = faiss.read_index(str(path))
    if not isinstance(index, faiss.IndexResidualQuantizer):
        raise RQKMeansError("rq_index.faiss 类型不是 IndexResidualQuantizer")
    if index.d != config.input_dim or index.rq.M != len(config.codebook_sizes):
        raise RQKMeansError("rq_index.faiss 维度或层数与配置不一致")
    if list(faiss.vector_to_array(index.rq.nbits)) != _codebook_nbits(
        config.codebook_sizes
    ):
        raise RQKMeansError("rq_index.faiss 各层 bit 数与配置不一致")
    index.rq.max_beam_size = config.max_beam_size
    index.rq.use_beam_LUT = int(config.use_beam_lut)
    index.rq.max_mem_distances = config.max_memory_mib * 1024 * 1024
    faiss.omp_set_num_threads(config.omp_threads)
    return index


def _faiss_residual_codes(index: Any, values: np.ndarray) -> np.ndarray:
    try:
        import faiss
    except ImportError as error:
        raise RQKMeansError("Faiss ResidualQuantizer 不可用") from error
    packed = index.sa_encode(np.ascontiguousarray(values, dtype=np.float32))
    codes = faiss.unpack_bitstrings(
        packed, list(faiss.vector_to_array(index.rq.nbits))
    )
    return np.asarray(codes, dtype=np.int32)


def _extract_faiss_codebooks(index: Any, config: RQKMeansConfig) -> list[np.ndarray]:
    try:
        import faiss
    except ImportError as error:
        raise RQKMeansError("Faiss ResidualQuantizer 不可用") from error
    offsets = faiss.vector_to_array(index.rq.codebook_offsets).astype(np.int64)
    flat = faiss.vector_to_array(index.rq.codebooks).reshape(-1, config.input_dim)
    return [
        np.asarray(flat[offsets[level] : offsets[level + 1]], dtype=np.float32)
        for level in range(len(config.codebook_sizes))
    ]


def _prepare_sample_residual(
    embeddings: np.ndarray,
    sample_indices: np.ndarray,
    sample_codes: np.ndarray,
    completed_codebooks: Sequence[np.ndarray],
    path: Path,
    chunk_rows: int,
    show_progress: bool,
    storage: str,
) -> np.ndarray:
    path.unlink(missing_ok=True)
    residual = (
        np.empty((len(sample_indices), embeddings.shape[1]), dtype=np.float32)
        if storage == "memory"
        else np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(len(sample_indices), embeddings.shape[1]),
        )
    )
    for start in tqdm(
        range(0, len(sample_indices), chunk_rows),
        desc="Prepare KMeans sample",
        unit="poi",
        disable=not show_progress,
    ):
        stop = min(start + chunk_rows, len(sample_indices))
        block = np.asarray(embeddings[sample_indices[start:stop]], dtype=np.float32)
        for level, centroids in enumerate(completed_codebooks):
            labels = np.asarray(sample_codes[start:stop, level], dtype=np.int64)
            if np.any(labels < 0):
                raise RQKMeansError("已完成层的 sample code 存在负值")
            block -= centroids[labels]
        residual[start:stop] = block
    if hasattr(residual, "flush"):
        residual.flush()
    return residual


def _flush_array(values: np.ndarray) -> None:
    flush = getattr(values, "flush", None)
    if flush is not None:
        flush()


def _train_faiss_residual_quantizer(
    config: RQKMeansConfig,
    embeddings: np.ndarray,
    sample_indices: np.ndarray,
    resolved: dict[str, Any],
    resolved_path: Path,
) -> tuple[list[np.ndarray], Any]:
    index_path = config.output_dir / "rq_index.faiss"
    if (
        index_path.is_file()
        and len(resolved["quantizer"]["levels"]) == len(config.codebook_sizes)
    ):
        index = _load_faiss_residual_index(index_path, config)
        return _extract_faiss_codebooks(index, config), index

    work_dir = config.output_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    sample_path = work_dir / "rq_training_sample.npy"
    sample_codes_path = config.output_dir / "sample_codes.npy"
    sample = _prepare_sample_residual(
        embeddings,
        sample_indices,
        np.empty((len(sample_indices), 0), dtype=np.int32),
        [],
        sample_path,
        config.sample_chunk_rows,
        config.show_progress,
        config.sample_storage,
    )
    started = time.perf_counter()
    try:
        index = _new_faiss_residual_index(config)
        index.train(sample)
        codes = _faiss_residual_codes(index, sample)
        codebooks = _extract_faiss_codebooks(index, config)
        _npy_atomic(sample_codes_path, codes)
        _npy_atomic(
            config.output_dir / "codebooks.npy",
            np.concatenate(codebooks, axis=0).astype(np.float32, copy=False),
        )
        level_squared_l2 = np.zeros(len(codebooks), dtype=np.float64)
        for start in range(0, len(sample), config.sample_chunk_rows):
            stop = min(start + config.sample_chunk_rows, len(sample))
            residual = np.asarray(sample[start:stop], dtype=np.float32).copy()
            for level, centroids in enumerate(codebooks):
                residual -= centroids[codes[start:stop, level]]
                level_squared_l2[level] += float(
                    np.sum(residual * residual, dtype=np.float64)
                )
        levels: list[dict[str, Any]] = []
        for level, centroids in enumerate(codebooks):
            counts = np.bincount(
                codes[:, level], minlength=config.codebook_sizes[level]
            )
            codebook_path = config.output_dir / f"codebook_level_{level + 1}.npy"
            codebook_sha256 = _npy_atomic_with_sha256(codebook_path, centroids)
            levels.append(
                {
                    "level": level + 1,
                    "codebook_size": config.codebook_sizes[level],
                    "used_codes_on_sample": int(np.count_nonzero(counts)),
                    "dead_codes_on_sample": int(np.count_nonzero(counts == 0)),
                    "mean_squared_l2_residual_on_sample": float(
                        level_squared_l2[level] / len(sample)
                    ),
                    "codebook_path": codebook_path.name,
                    "codebook_sha256": codebook_sha256,
                }
            )
        temporary_index = index_path.with_name(f".{index_path.name}.tmp")
        try:
            import faiss

            faiss.write_index(index, str(temporary_index))
            os.replace(temporary_index, index_path)
        finally:
            temporary_index.unlink(missing_ok=True)
        resolved["quantizer"]["levels"] = levels
        resolved["quantizer"]["training_seconds"] = time.perf_counter() - started
        resolved["quantizer"]["index_path"] = index_path.name
        resolved["quantizer"]["index_sha256"] = _sha256_file(index_path)
        resolved["status"] = "training_codebooks"
        _json_atomic(resolved_path, resolved)
    finally:
        _flush_array(sample)
        del sample
    sample_path.unlink(missing_ok=True)
    try:
        work_dir.rmdir()
    except OSError:
        pass
    return codebooks, index


def _train_codebooks(
    config: RQKMeansConfig,
    embeddings: np.ndarray,
    sample_indices: np.ndarray,
    resolved: dict[str, Any],
    resolved_path: Path,
    *,
    category_ids: np.ndarray | None = None,
    category_codes: Sequence[str] | None = None,
    validation_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    work_dir = config.output_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    sample_codes_path = config.output_dir / "sample_codes.npy"
    completed_levels = len(resolved["quantizer"]["levels"])
    if sample_codes_path.is_file():
        existing_sample_codes = np.load(
            sample_codes_path,
            mmap_mode="r+" if config.sample_storage == "mmap" else "r",
        )
        if config.sample_storage == "mmap":
            sample_codes = existing_sample_codes
        else:
            sample_codes = np.array(existing_sample_codes, dtype=np.int32, copy=True)
            del existing_sample_codes
        if sample_codes.shape != (len(sample_indices), len(config.codebook_sizes)):
            raise RQKMeansError("现有 sample_codes.npy shape 与配置不一致")
    else:
        if config.sample_storage == "memory":
            sample_codes = np.full(
                (len(sample_indices), len(config.codebook_sizes)),
                -1,
                dtype=np.int32,
            )
            _npy_atomic(sample_codes_path, sample_codes)
        else:
            sample_codes = np.lib.format.open_memmap(
                sample_codes_path,
                mode="w+",
                dtype=np.int32,
                shape=(len(sample_indices), len(config.codebook_sizes)),
            )
            sample_codes[:] = -1
            sample_codes.flush()
    codebooks: list[np.ndarray] = []
    for level in range(completed_levels):
        path = config.output_dir / f"codebook_level_{level + 1}.npy"
        if not path.is_file():
            raise RQKMeansError(f"已完成层缺少码本：{path}")
        codebook = np.load(path, allow_pickle=False)
        expected_shape = (config.codebook_sizes[level], config.input_dim)
        if codebook.shape != expected_shape or codebook.dtype != np.float32:
            raise RQKMeansError(f"第 {level + 1} 层现有码本 shape/dtype 无效")
        codebooks.append(codebook)

    if config.first_level_mode == "category_code" and completed_levels == 0:
        if category_ids is None or category_codes is None or validation_mask is None:
            raise RQKMeansError("category_code 第一层缺少对齐类别或训练划分")
        if len(category_codes) != config.codebook_sizes[0]:
            raise RQKMeansError("category_code 数量与第一层码本不一致")
        started = time.perf_counter()
        category_sums = np.zeros(
            (len(category_codes), config.input_dim), dtype=np.float64
        )
        category_counts = np.zeros(len(category_codes), dtype=np.int64)
        for start in tqdm(
            range(0, len(embeddings), config.chunk_rows),
            desc="Build category centroids",
            unit="poi",
            disable=not config.show_progress,
        ):
            stop = min(start + config.chunk_rows, len(embeddings))
            train_rows = ~validation_mask[start:stop]
            if not np.any(train_rows):
                continue
            labels = np.asarray(category_ids[start:stop], dtype=np.int32)[train_rows]
            values = np.asarray(embeddings[start:stop], dtype=np.float32)[train_rows]
            for label in np.unique(labels):
                selected = labels == label
                category_sums[label] += values[selected].sum(axis=0, dtype=np.float64)
                category_counts[label] += int(np.count_nonzero(selected))
        if np.any(category_counts == 0):
            missing = np.flatnonzero(category_counts == 0).tolist()
            raise RQKMeansError(f"训练划分中存在空 category_code：{missing[:20]}")
        centroids = np.asarray(
            category_sums / category_counts[:, None], dtype=np.float32
        )
        labels = np.asarray(category_ids[sample_indices], dtype=np.int32)
        sample_codes[:, 0] = labels
        if config.sample_storage == "memory":
            _npy_atomic(sample_codes_path, sample_codes)
        else:
            sample_codes.flush()
        squared_l2 = 0.0
        for start in range(0, len(sample_indices), config.sample_chunk_rows):
            stop = min(start + config.sample_chunk_rows, len(sample_indices))
            values = np.asarray(
                embeddings[sample_indices[start:stop]], dtype=np.float32
            )
            block_labels = labels[start:stop]
            residual = values - centroids[block_labels]
            squared_l2 += float(np.sum(residual * residual, dtype=np.float64))
        codebook_path = config.output_dir / "codebook_level_1.npy"
        codebook_sha256 = _npy_atomic_with_sha256(codebook_path, centroids)
        counts_path = config.output_dir / "category_training_counts.npy"
        counts_sha256 = _npy_atomic_with_sha256(counts_path, category_counts)
        level_metrics = {
            "level": 1,
            "codebook_size": len(category_codes),
            "assignment": "category_code",
            "centroid_source": "all_non_validation_pois",
            "used_codes_on_sample": int(np.unique(labels).size),
            "dead_codes_on_sample": int(len(category_codes) - np.unique(labels).size),
            "used_codes_on_train": int(np.count_nonzero(category_counts)),
            "dead_codes_on_train": int(np.count_nonzero(category_counts == 0)),
            "mean_squared_l2_residual_on_sample": squared_l2 / len(sample_indices),
            "seconds": time.perf_counter() - started,
            "codebook_path": codebook_path.name,
            "codebook_sha256": codebook_sha256,
            "category_training_counts_path": counts_path.name,
            "category_training_counts_sha256": counts_sha256,
        }
        codebooks.append(centroids)
        resolved["quantizer"]["levels"].append(level_metrics)
        resolved["status"] = "training_codebooks"
        _json_atomic(resolved_path, resolved)
        print(
            json.dumps({"rqkmeans_level": level_metrics}, ensure_ascii=False),
            flush=True,
        )
        completed_levels = 1

    if config.first_level_mode == "category_code" and completed_levels > 0:
        if category_ids is None:
            raise RQKMeansError("category_code 第一层缺少对齐类别")
        expected_labels = np.asarray(category_ids[sample_indices], dtype=np.int32)
        if not np.array_equal(np.asarray(sample_codes[:, 0]), expected_labels):
            raise RQKMeansError("现有第一层 sample code 与 category_code 不一致")

    if completed_levels == len(config.codebook_sizes):
        del sample_codes
        flattened = np.concatenate(codebooks, axis=0).astype(np.float32, copy=False)
        _npy_atomic(config.output_dir / "codebooks.npy", flattened)
        return codebooks

    residual_path = work_dir / "sample_residual.npy"
    residual = _prepare_sample_residual(
        embeddings,
        sample_indices,
        sample_codes,
        codebooks,
        residual_path,
        config.sample_chunk_rows,
        config.show_progress,
        config.sample_storage,
    )
    try:
        for level in range(completed_levels, len(config.codebook_sizes)):
            started = time.perf_counter()
            centroids, labels, mean_distance = _fit_kmeans(
                residual,
                config.codebook_sizes[level],
                iterations=config.iterations,
                nredo=config.nredo,
                seed=config.seed + level,
                backend=config.backend,
                max_points_per_centroid=config.max_points_per_centroid,
            )
            codebook_path = config.output_dir / f"codebook_level_{level + 1}.npy"
            codebook_sha256 = _npy_atomic_with_sha256(codebook_path, centroids)
            sample_codes[:, level] = labels
            if config.sample_storage == "memory":
                _npy_atomic(sample_codes_path, sample_codes)
            else:
                sample_codes.flush()
            for start in tqdm(
                range(0, len(sample_indices), config.sample_chunk_rows),
                desc=f"Update sample residual L{level + 1}",
                unit="poi",
                disable=not config.show_progress,
            ):
                stop = min(start + config.sample_chunk_rows, len(sample_indices))
                residual[start:stop] -= centroids[labels[start:stop]]
            _flush_array(residual)
            codebooks.append(centroids)
            counts = np.bincount(labels, minlength=config.codebook_sizes[level])
            level_metrics = {
                "level": level + 1,
                "codebook_size": config.codebook_sizes[level],
                "seed": config.seed + level,
                "used_codes_on_sample": int(np.count_nonzero(counts)),
                "dead_codes_on_sample": int(np.count_nonzero(counts == 0)),
                "mean_squared_l2_residual_on_sample": mean_distance,
                "seconds": time.perf_counter() - started,
                "codebook_path": codebook_path.name,
                "codebook_sha256": codebook_sha256,
            }
            resolved["quantizer"]["levels"].append(level_metrics)
            resolved["status"] = "training_codebooks"
            _json_atomic(resolved_path, resolved)
            print(
                json.dumps({"rqkmeans_level": level_metrics}, ensure_ascii=False),
                flush=True,
            )
    finally:
        _flush_array(residual)
        del residual
    residual_path.unlink(missing_ok=True)
    try:
        work_dir.rmdir()
    except OSError:
        pass
    del sample_codes
    flattened = np.concatenate(codebooks, axis=0).astype(np.float32, copy=False)
    _npy_atomic(config.output_dir / "codebooks.npy", flattened)
    return codebooks


def _encode_full_sid(
    config: RQKMeansConfig,
    embeddings: np.ndarray,
    codebooks: Sequence[np.ndarray],
    faiss_residual_index: Any | None = None,
    category_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    sid_path = config.output_dir / "sid_codes.npy"
    codes = np.empty(
        (len(embeddings), len(codebooks)),
        dtype=np.int32,
    )
    if faiss_residual_index is None:
        indices, resources = _build_assignment_indices(codebooks, config)
    else:
        indices, resources = None, None
    level_squared_l2 = np.zeros(len(codebooks), dtype=np.float64)
    reconstruction_cosine_sum = 0.0
    started = time.perf_counter()
    try:
        for start in tqdm(
            range(0, len(embeddings), config.chunk_rows),
            desc="Encode full RQ-KMeans SID",
            unit="poi",
            disable=not config.show_progress,
        ):
            stop = min(start + config.chunk_rows, len(embeddings))
            source = np.asarray(embeddings[start:stop], dtype=np.float32)
            residual = source.copy()
            joint_codes = (
                _faiss_residual_codes(faiss_residual_index, source)
                if faiss_residual_index is not None
                else None
            )
            for level, centroids in enumerate(codebooks):
                if level == 0 and config.first_level_mode == "category_code":
                    if category_ids is None:
                        raise RQKMeansError("全量编码缺少 category_code 对齐数组")
                    labels = np.asarray(
                        category_ids[start:stop], dtype=np.int32
                    )
                elif joint_codes is None:
                    index = None if indices is None else indices[level]
                    labels = _assign_codes(residual, centroids, index)
                else:
                    labels = joint_codes[:, level]
                codes[start:stop, level] = labels
                residual -= centroids[labels]
                level_squared_l2[level] += float(
                    np.sum(residual * residual, dtype=np.float64)
                )
            reconstruction = source - residual
            numerator = np.sum(source * reconstruction, axis=1)
            denominator = np.linalg.norm(source, axis=1) * np.linalg.norm(
                reconstruction, axis=1
            )
            reconstruction_cosine_sum += float(
                np.sum(
                    np.divide(
                        numerator,
                        denominator,
                        out=np.zeros_like(numerator),
                        where=denominator > 0,
                    ),
                    dtype=np.float64,
                )
            )
        sid_codes_sha256 = _npy_atomic_with_sha256(sid_path, codes)
        del codes
    finally:
        if "codes" in locals():
            del codes
        del indices, resources
    row_count = len(embeddings)
    return {
        "rows": row_count,
        "dimension": config.input_dim,
        "mean_squared_l2_residual_by_level": [
            float(value / row_count) for value in level_squared_l2
        ],
        "final_mean_squared_l2_error": float(level_squared_l2[-1] / row_count),
        "final_mean_squared_error_per_dimension": float(
            level_squared_l2[-1] / (row_count * config.input_dim)
        ),
        "mean_reconstruction_cosine": reconstruction_cosine_sum / row_count,
        "encoding_seconds": time.perf_counter() - started,
        "sid_codes_sha256": sid_codes_sha256,
    }


def _screen_on_validation(
    config: RQKMeansConfig,
    embeddings: np.ndarray,
    validation_indices: np.ndarray,
    codebooks: Sequence[np.ndarray],
    faiss_residual_index: Any | None,
    category_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    if faiss_residual_index is None:
        assignment_indices, resources = _build_assignment_indices(codebooks, config)
    else:
        assignment_indices, resources = None, None
    codes = np.empty(
        (len(validation_indices), len(codebooks)), dtype=np.int32
    )
    level_squared_l2 = np.zeros(len(codebooks), dtype=np.float64)
    cosine_sum = 0.0
    started = time.perf_counter()
    try:
        for start in range(0, len(validation_indices), config.chunk_rows):
            stop = min(start + config.chunk_rows, len(validation_indices))
            source = np.asarray(
                embeddings[validation_indices[start:stop]], dtype=np.float32
            )
            residual = source.copy()
            joint_codes = (
                _faiss_residual_codes(faiss_residual_index, source)
                if faiss_residual_index is not None
                else None
            )
            for level, centroids in enumerate(codebooks):
                if level == 0 and config.first_level_mode == "category_code":
                    if category_ids is None:
                        raise RQKMeansError("Validation 编码缺少 category_code 对齐数组")
                    labels = np.asarray(
                        category_ids[validation_indices[start:stop]],
                        dtype=np.int32,
                    )
                elif joint_codes is None:
                    index = (
                        None
                        if assignment_indices is None
                        else assignment_indices[level]
                    )
                    labels = _assign_codes(residual, centroids, index)
                else:
                    labels = joint_codes[:, level]
                codes[start:stop, level] = labels
                residual -= centroids[labels]
                level_squared_l2[level] += float(
                    np.sum(residual * residual, dtype=np.float64)
                )
            reconstruction = source - residual
            numerator = np.sum(source * reconstruction, axis=1)
            denominator = np.linalg.norm(source, axis=1) * np.linalg.norm(
                reconstruction, axis=1
            )
            cosine_sum += float(
                np.sum(
                    np.divide(
                        numerator,
                        denominator,
                        out=np.zeros_like(numerator),
                        where=denominator > 0,
                    ),
                    dtype=np.float64,
                )
            )
    finally:
        del assignment_indices, resources
    basic, _, _, _ = compute_basic_metrics(codes)
    row_count = len(validation_indices)
    return {
        "schema_version": "rqkmeans-screen-v1",
        "status": "completed",
        "rows": row_count,
        "indices_sha256": _sha256_indices(validation_indices),
        "basic": basic,
        "layers": compute_layer_metrics(codes, config.codebook_sizes),
        "quantization": {
            "mean_squared_l2_residual_by_level": [
                float(value / row_count) for value in level_squared_l2
            ],
            "final_mean_squared_l2_error": float(
                level_squared_l2[-1] / row_count
            ),
            "final_mean_squared_error_per_dimension": float(
                level_squared_l2[-1] / (row_count * config.input_dim)
            ),
            "mean_reconstruction_cosine": cosine_sum / row_count,
        },
        "seconds": time.perf_counter() - started,
    }


def run_rqkmeans(
    config: RQKMeansConfig,
    *,
    project_root: Path,
    validate_only: bool = False,
    screen_only: bool = False,
) -> dict[str, Any]:
    """Train residual codebooks, stream full SID export, and run shared evaluation."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = config.output_dir / "resolved_config.json"
    signature = _config_signature(config)
    existing: dict[str, Any] | None = None
    if resolved_path.is_file():
        with resolved_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("config_signature") != signature:
            raise RQKMeansError("现有 resolved_config 与当前配置不一致")
        if existing.get("status") == "completed":
            return existing
        if not config.resume:
            raise RQKMeansError("输出目录已有未完成任务且 resume=false")

    if config.backend == "faiss_gpu":
        try:
            import faiss
        except ImportError as error:
            raise RQKMeansError("配置要求 faiss_gpu，但 Faiss 不可用") from error
        if faiss.get_num_gpus() < 1:
            raise RQKMeansError("配置要求 faiss_gpu，但当前进程未检测到 GPU")

    embeddings, input_validation = validate_rqkmeans_inputs(config)
    validation_mask, validation_indices, sample_indices, split = create_fixed_indices(
        len(embeddings),
        config.validation_ratio,
        config.sample_size,
        config.seed,
    )
    validation_sha = _sha256_indices(validation_indices)
    sample_sha = _sha256_indices(sample_indices)
    if (
        config.expected_validation_indices_sha256 is not None
        and validation_sha != config.expected_validation_indices_sha256
    ):
        raise RQKMeansError("Validation 索引 SHA256 与 TIGER 冻结划分不一致")
    if (
        config.expected_sample_indices_sha256 is not None
        and sample_sha != config.expected_sample_indices_sha256
    ):
        raise RQKMeansError("KMeans 采样索引 SHA256 与 TIGER 冻结采样不一致")
    category_ids: np.ndarray | None = None
    category_codes: tuple[str, ...] | None = None
    if config.first_level_mode == "category_code":
        category_ids, category_codes = load_aligned_category_codes(
            config.poi_data_path,
            config.poi_ids_path,
            len(embeddings),
        )
        if np.any(category_ids < 0):
            raise RQKMeansError("category_code 第一层不允许缺失类别")
        if len(category_codes) != config.codebook_sizes[0]:
            raise RQKMeansError(
                f"第一层码本 {config.codebook_sizes[0]} != 实际 category_code "
                f"数量 {len(category_codes)}"
            )
        if (
            config.expected_category_count is not None
            and len(category_codes) != config.expected_category_count
        ):
            raise RQKMeansError(
                "实际 category_code 数量与 expected.category_count 不一致"
            )
        input_validation["category_first_level"] = {
            "field": "category_code",
            "category_count": len(category_codes),
            "missing_rows": 0,
            "assignment_sha256": hashlib.sha256(
                np.ascontiguousarray(category_ids).tobytes()
            ).hexdigest(),
        }
    split["comparison_role"] = "same_indices_as_tiger_rqvae"
    if validate_only:
        return {
            "status": "validated",
            "config_signature": signature,
            "input_validation": input_validation,
            "split": split,
        }

    sample_indices_path = config.output_dir / "sample_indices.npy"
    if sample_indices_path.is_file():
        persisted = np.load(sample_indices_path, allow_pickle=False)
        if _sha256_indices(persisted) != sample_sha:
            raise RQKMeansError("现有 sample_indices.npy 与冻结采样不一致")
    else:
        _npy_atomic(sample_indices_path, sample_indices)
    resolved = existing or {
        "schema_version": "rqkmeans-training-v1",
        "status": "initialized",
        "experiment_id": config.experiment_id,
        "config_signature": signature,
        "config": _config_payload(config),
        "input_validation": input_validation,
        "split": split,
        "comparison_protocol": {
            "reference": "TIGER-BGE-M3-1024x3",
            "same_embedding_rows": True,
            "same_validation_indices": True,
            "same_kmeans_sample_indices": (
                sample_sha
                == "0f7f22d7a7543965d581bcc9098a833533dbbdbb701b867f54c76d0bf66d6eb7"
            ),
            "same_seed": True,
            "same_layers_and_codebook_sizes": (
                config.codebook_sizes == (1024, 1024, 1024)
            ),
            "same_kmeans_iterations": config.iterations == 20,
            "first_level_mode": config.first_level_mode,
            "first_level_supervised": config.first_level_mode == "category_code",
            "direct_input_dimension": config.input_dim,
            "latent_projection": None,
        },
        "quantizer": {
            "method": (
                "category_residual_kmeans"
                if config.first_level_mode == "category_code"
                else "residual_kmeans"
            ),
            "implementation": config.implementation,
            "backend": config.backend,
            "levels": [],
        },
        "git": _git_state(project_root),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "started_at": _utc_now(),
    }
    _json_atomic(resolved_path, resolved)
    if category_ids is not None and category_codes is not None:
        category_ids_path = config.output_dir / "category_ids.npy"
        if category_ids_path.is_file():
            persisted_category_ids = np.load(
                category_ids_path, mmap_mode="r", allow_pickle=False
            )
            if not np.array_equal(persisted_category_ids, category_ids):
                raise RQKMeansError(
                    "现有 category_ids.npy 与当前 category_code 对齐不一致"
                )
            category_ids_sha256 = _sha256_file(category_ids_path)
        else:
            category_ids_sha256 = _npy_atomic_with_sha256(
                category_ids_path, category_ids
            )
        category_vocab_path = config.output_dir / "category_vocab.json"
        _json_atomic(
            category_vocab_path,
            {
                "schema_version": "category-first-level-v1",
                "field": "category_code",
                "category_count": len(category_codes),
                "category_codes": list(category_codes),
                "category_ids_path": category_ids_path.name,
                "category_ids_sha256": category_ids_sha256,
                "assignment": "one_category_code_one_s1_token",
            },
        )
        resolved["category_first_level"] = {
            "field": "category_code",
            "category_count": len(category_codes),
            "category_ids_path": category_ids_path.name,
            "category_ids_sha256": category_ids_sha256,
            "category_vocab_path": category_vocab_path.name,
            "category_vocab_sha256": _sha256_file(category_vocab_path),
        }
        _json_atomic(resolved_path, resolved)
    if config.implementation == "faiss_residual_quantizer":
        codebooks, faiss_residual_index = _train_faiss_residual_quantizer(
            config, embeddings, sample_indices, resolved, resolved_path
        )
    else:
        codebooks = _train_codebooks(
            config,
            embeddings,
            sample_indices,
            resolved,
            resolved_path,
            category_ids=category_ids,
            category_codes=category_codes,
            validation_mask=validation_mask,
        )
        faiss_residual_index = None
    if "screen_metrics" not in resolved:
        screen_metrics = _screen_on_validation(
            config,
            embeddings,
            validation_indices,
            codebooks,
            faiss_residual_index,
            category_ids,
        )
        _json_atomic(config.output_dir / "screen_metrics.json", screen_metrics)
        resolved["screen_metrics"] = screen_metrics
        resolved["status"] = "screened"
        _json_atomic(resolved_path, resolved)
    if screen_only:
        return resolved
    resolved["status"] = "encoding"
    _json_atomic(resolved_path, resolved)
    sid_path = config.output_dir / "sid_codes.npy"
    if "full_quantization" not in resolved:
        resolved["full_quantization"] = _encode_full_sid(
            config,
            embeddings,
            codebooks,
            faiss_residual_index,
            category_ids,
        )
        _json_atomic(resolved_path, resolved)
    elif not sid_path.is_file():
        raise RQKMeansError("resolved_config 已记录全量编码，但 sid_codes.npy 缺失")

    manifest = {
        "schema_version": "sid-input-v1",
        "experiment_id": config.experiment_id,
        "method": (
            "category_residual_kmeans"
            if config.first_level_mode == "category_code"
            else "residual_kmeans"
        ),
        "embedding": {
            "manifest": os.path.relpath(
                config.embedding_manifest_path, config.output_dir
            ),
            "fingerprint": input_validation["embedding_fingerprint"],
            "shape": input_validation["embedding_shape"],
            "dtype": input_validation["embedding_dtype"],
        },
        "sid_codes": {
            "path": sid_path.name,
            "shape": [len(embeddings), len(config.codebook_sizes)],
            "dtype": "int32",
        },
        "poi_ids": {
            "path": os.path.relpath(config.poi_ids_path, config.output_dir),
            "sha256": input_validation["poi_ids_sha256"],
        },
        "sid_layers": len(config.codebook_sizes),
        "codebook_sizes": list(config.codebook_sizes),
        "codebooks": {
            "path": "codebooks.npy",
            "shape": [
                sum(config.codebook_sizes),
                config.input_dim,
            ],
            "dtype": "float32",
            "sha256": _sha256_file(config.output_dir / "codebooks.npy"),
            "level_offsets": [
                int(value)
                for value in np.cumsum(
                    [0, *config.codebook_sizes], dtype=np.int64
                )
            ],
        },
        "seed": config.seed,
        "sample_indices_sha256": sample_sha,
        "exported_at": _utc_now(),
    }
    if category_ids is not None:
        category_metadata = resolved.get("category_first_level")
        if not isinstance(category_metadata, dict):
            raise RQKMeansError("resolved config 缺少 category-first 元数据")
        manifest["first_level"] = {
            "mode": "category_code",
            **category_metadata,
            "exact_assignment": True,
        }
    manifest_path = config.output_dir / "sid_manifest.json"
    _json_atomic(manifest_path, manifest)
    resolved["status"] = "evaluating"
    _json_atomic(resolved_path, resolved)
    metrics, cases = evaluate_sid(
        manifest_path,
        config.poi_data_path,
        max_cases=config.evaluation_max_cases,
        max_pois_per_case=config.evaluation_max_pois_per_case,
    )
    metrics["quantization"] = resolved["full_quantization"]
    metrics["comparison_protocol"] = resolved["comparison_protocol"]
    if category_ids is not None:
        exported_codes = np.load(sid_path, mmap_mode="r", allow_pickle=False)
        exact_assignment = True
        for start in range(0, len(category_ids), config.chunk_rows):
            stop = min(start + config.chunk_rows, len(category_ids))
            if not np.array_equal(
                exported_codes[start:stop, 0], category_ids[start:stop]
            ):
                exact_assignment = False
                break
        first_purity = metrics["prefixes"][0]["category_purity"]
        if (
            not exact_assignment
            or first_purity["micro_purity"] != 1.0
            or first_purity["macro_purity"] != 1.0
        ):
            raise RQKMeansError(
                "category_code 第一层未通过精确赋值或 100% purity 校验"
            )
        metrics["validation"]["category_first_level_exact_assignment"] = True
        metrics["validation"]["category_first_level_micro_purity"] = 1.0
        metrics["validation"]["category_first_level_macro_purity"] = 1.0
    write_evaluation_outputs(config.output_dir, metrics, cases)
    resolved.update(
        {
            "status": "completed",
            "finished_at": _utc_now(),
            "outputs": {
                "codebooks": str(config.output_dir / "codebooks.npy"),
                "sid_codes": str(sid_path),
                "sid_manifest": str(manifest_path),
                "metrics": str(config.output_dir / "metrics.json"),
                "collision_cases": str(
                    config.output_dir / "collision_cases.jsonl"
                ),
            },
        }
    )
    _json_atomic(resolved_path, resolved)
    (config.output_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")
    return resolved

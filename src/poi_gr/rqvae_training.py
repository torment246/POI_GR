"""Training, checkpointing, and export helpers for Vanilla RQ-VAE."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .rqvae import RQVAE, RQVAEOutput


class RQVAETrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class RQVAETrainingConfig:
    source_config: Path
    experiment_id: str
    embeddings_path: Path
    poi_ids_path: Path
    embedding_manifest_path: Path
    poi_data_path: Path
    output_dir: Path
    input_dim: int
    hidden_dim: int
    latent_dim: int
    codebook_sizes: tuple[int, ...]
    codebook_loss_weight: float
    commitment_loss_weight: float
    seed: int
    device: str
    batch_size: int
    block_rows: int
    max_epochs: int
    checkpoint_epochs: tuple[int, ...]
    validation_ratio: float
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    kmeans_backend: str
    kmeans_sample_size: int
    kmeans_iterations: int
    kmeans_batch_size: int
    kmeans_max_points_per_centroid: int
    finite_check_chunk_rows: int
    collapse_utilization_threshold: float
    resume: bool
    show_progress: bool
    max_rows: int | None
    evaluation_max_cases: int
    evaluation_max_pois_per_case: int


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RQVAETrainingError(f"{name} 必须是 YAML mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RQVAETrainingError(f"{name} 必须是正整数")
    return value


def _positive_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise RQVAETrainingError(f"{name} 必须大于 0")
    return float(value)


def _resolve_path(value: Any, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RQVAETrainingError(f"{name} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_training_config(
    config_path: Path,
    project_root: Path,
    experiment_id: str,
    *,
    output_dir: Path | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    max_epochs: int | None = None,
    checkpoint_epochs: Sequence[int] | None = None,
    max_rows: int | None = None,
    codebook_sizes: Sequence[int] | None = None,
    kmeans_sample_size: int | None = None,
    kmeans_backend: str | None = None,
    resume: bool | None = None,
    show_progress: bool | None = None,
) -> RQVAETrainingConfig:
    """Load one experiment from the shared Beijing RQ-VAE configuration."""

    with config_path.open("r", encoding="utf-8") as handle:
        raw = _require_mapping(yaml.safe_load(handle), "配置根节点")
    data = _require_mapping(raw.get("data"), "data")
    model = _require_mapping(raw.get("model"), "model")
    initialization = _require_mapping(raw.get("initialization"), "initialization")
    training = _require_mapping(raw.get("training"), "training")
    evaluation = _require_mapping(raw.get("evaluation"), "evaluation")
    experiments = _require_mapping(raw.get("experiments"), "experiments")
    experiment = _require_mapping(
        experiments.get(experiment_id), f"experiments.{experiment_id}"
    )

    declared_codebooks = experiment.get("codebook_sizes")
    if codebook_sizes is None:
        if not isinstance(declared_codebooks, list):
            raise RQVAETrainingError("experiment.codebook_sizes 必须是数组")
        codebook_sizes = declared_codebooks
    codebook_tuple = tuple(
        _positive_int(value, "codebook_sizes") for value in codebook_sizes
    )
    rq_layers = _positive_int(model.get("rq_layers"), "model.rq_layers")
    if len(codebook_tuple) != rq_layers:
        raise RQVAETrainingError("codebook_sizes 长度必须等于 model.rq_layers")
    if rq_layers != 3:
        raise RQVAETrainingError("本轮正式配置固定使用 3 层 RQ")

    output_root = _resolve_path(raw.get("output_root"), project_root, "output_root")
    resolved_output = output_dir or output_root / experiment_id
    resolved_device = device or str(training.get("device", "cuda"))
    resolved_batch_size = batch_size or _positive_int(
        training.get("batch_size"), "training.batch_size"
    )
    resolved_max_epochs = max_epochs or _positive_int(
        training.get("max_epochs"), "training.max_epochs"
    )
    if bool(training.get("early_stopping", False)):
        raise RQVAETrainingError("本轮协议要求 training.early_stopping=false")
    raw_checkpoint_epochs = (
        training.get("checkpoint_epochs")
        if checkpoint_epochs is None
        else list(checkpoint_epochs)
    )
    if not isinstance(raw_checkpoint_epochs, list) or not raw_checkpoint_epochs:
        raise RQVAETrainingError("training.checkpoint_epochs 必须是非空数组")
    checkpoint_epochs = tuple(
        _positive_int(value, "training.checkpoint_epochs")
        for value in raw_checkpoint_epochs
    )
    if tuple(sorted(set(checkpoint_epochs))) != checkpoint_epochs:
        raise RQVAETrainingError("training.checkpoint_epochs 必须严格递增且不重复")
    if checkpoint_epochs[-1] > resolved_max_epochs:
        raise RQVAETrainingError(
            "training.checkpoint_epochs 不能超过本次 max_epochs"
        )
    resolved_sample_size = kmeans_sample_size or _positive_int(
        initialization.get("sample_size"), "initialization.sample_size"
    )
    resolved_backend = kmeans_backend or str(
        initialization.get("backend", "faiss_gpu")
    )
    if resolved_backend not in {"faiss_gpu", "faiss_cpu", "sklearn"}:
        raise RQVAETrainingError(
            "initialization.backend 只能是 faiss_gpu、faiss_cpu 或 sklearn"
        )
    if max_rows is not None and max_rows <= 1:
        raise RQVAETrainingError("max_rows 必须大于 1")

    validation_ratio = float(training.get("validation_ratio", 0.01))
    if not 0.0 < validation_ratio < 1.0:
        raise RQVAETrainingError("training.validation_ratio 必须在 (0, 1) 内")
    collapse_threshold = float(
        training.get("collapse_utilization_threshold", 0.01)
    )
    if not 0.0 <= collapse_threshold <= 1.0:
        raise RQVAETrainingError(
            "training.collapse_utilization_threshold 必须在 [0, 1] 内"
        )

    return RQVAETrainingConfig(
        source_config=config_path.resolve(),
        experiment_id=experiment_id,
        embeddings_path=_resolve_path(
            data.get("embeddings"), project_root, "data.embeddings"
        ),
        poi_ids_path=_resolve_path(data.get("poi_ids"), project_root, "data.poi_ids"),
        embedding_manifest_path=_resolve_path(
            data.get("embedding_manifest"), project_root, "data.embedding_manifest"
        ),
        poi_data_path=_resolve_path(
            data.get("poi_data"), project_root, "data.poi_data"
        ),
        output_dir=resolved_output.resolve(),
        input_dim=_positive_int(model.get("input_dim"), "model.input_dim"),
        hidden_dim=_positive_int(model.get("hidden_dim"), "model.hidden_dim"),
        latent_dim=_positive_int(model.get("latent_dim"), "model.latent_dim"),
        codebook_sizes=codebook_tuple,
        codebook_loss_weight=float(model.get("codebook_loss_weight", 1.0)),
        commitment_loss_weight=float(model.get("commitment_loss_weight", 0.25)),
        seed=int(training.get("seed", 42)),
        device=resolved_device,
        batch_size=resolved_batch_size,
        block_rows=_positive_int(training.get("block_rows"), "training.block_rows"),
        max_epochs=resolved_max_epochs,
        checkpoint_epochs=checkpoint_epochs,
        validation_ratio=validation_ratio,
        learning_rate=_positive_float(
            training.get("learning_rate"), "training.learning_rate"
        ),
        weight_decay=float(training.get("weight_decay", 0.0)),
        gradient_clip_norm=_positive_float(
            training.get("gradient_clip_norm", 1.0),
            "training.gradient_clip_norm",
        ),
        kmeans_backend=resolved_backend,
        kmeans_sample_size=resolved_sample_size,
        kmeans_iterations=_positive_int(
            initialization.get("iterations"), "initialization.iterations"
        ),
        kmeans_batch_size=_positive_int(
            initialization.get("batch_size"), "initialization.batch_size"
        ),
        kmeans_max_points_per_centroid=_positive_int(
            initialization.get("max_points_per_centroid"),
            "initialization.max_points_per_centroid",
        ),
        finite_check_chunk_rows=_positive_int(
            data.get("finite_check_chunk_rows", 8192),
            "data.finite_check_chunk_rows",
        ),
        collapse_utilization_threshold=collapse_threshold,
        resume=bool(training.get("resume", True)) if resume is None else resume,
        show_progress=(
            bool(training.get("show_progress", True))
            if show_progress is None
            else show_progress
        ),
        max_rows=max_rows,
        evaluation_max_cases=_positive_int(
            evaluation.get("max_cases", 20), "evaluation.max_cases"
        ),
        evaluation_max_pois_per_case=_positive_int(
            evaluation.get("max_pois_per_case", 20),
            "evaluation.max_pois_per_case",
        ),
    )


def config_payload(config: RQVAETrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["codebook_sizes"] = list(config.codebook_sizes)
    payload["checkpoint_epochs"] = list(config.checkpoint_epochs)
    return payload


def config_signature(config: RQVAETrainingConfig) -> str:
    payload = config_payload(config)
    payload.pop("resume", None)
    payload.pop("show_progress", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def shared_protocol_signature(config: RQVAETrainingConfig) -> str:
    """Fingerprint settings that must match across capacity experiments."""

    payload = config_payload(config)
    for key in (
        "source_config",
        "experiment_id",
        "output_dir",
        "codebook_sizes",
        "resume",
        "show_progress",
    ):
        payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


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


def validate_embedding_artifacts(
    config: RQVAETrainingConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate the source manifest, mmap array, finite values, and aligned IDs."""

    for path, name in (
        (config.embedding_manifest_path, "Embedding manifest"),
        (config.embeddings_path, "Embedding NPY"),
        (config.poi_ids_path, "POI ID mapping"),
    ):
        if not path.is_file():
            raise RQVAETrainingError(f"{name} 不存在：{path}")
    with config.embedding_manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise RQVAETrainingError("Embedding manifest 状态不是 completed")
    declared_shape = tuple(manifest.get("output", {}).get("shape", ()))
    declared_dtype = manifest.get("output", {}).get("dtype")
    embeddings = np.load(config.embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2:
        raise RQVAETrainingError("Embedding 必须是二维数组")
    if tuple(embeddings.shape) != declared_shape:
        raise RQVAETrainingError(
            f"Embedding shape {embeddings.shape} 与 manifest {declared_shape} 不一致"
        )
    if str(embeddings.dtype) != declared_dtype:
        raise RQVAETrainingError(
            f"Embedding dtype {embeddings.dtype} 与 manifest {declared_dtype} 不一致"
        )
    if embeddings.shape[1] != config.input_dim:
        raise RQVAETrainingError(
            f"Embedding dim {embeddings.shape[1]} != model.input_dim {config.input_dim}"
        )
    full_rows = int(embeddings.shape[0])
    effective_rows = full_rows if config.max_rows is None else min(config.max_rows, full_rows)
    if effective_rows <= 1:
        raise RQVAETrainingError("有效 Embedding 行数必须大于 1")

    seen_ids: set[str] = set()
    scanned_id_rows = 0
    with config.poi_ids_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if config.max_rows is not None and scanned_id_rows >= effective_rows:
                break
            if not line.strip():
                raise RQVAETrainingError(f"POI ID 第 {line_number} 行为空")
            try:
                poi_id = json.loads(line)
            except json.JSONDecodeError as error:
                raise RQVAETrainingError(
                    f"POI ID 第 {line_number} 行 JSON 解析失败"
                ) from error
            if not isinstance(poi_id, str) or not poi_id.strip():
                raise RQVAETrainingError(f"POI ID 第 {line_number} 行无效")
            if poi_id in seen_ids:
                raise RQVAETrainingError(f"POI ID 重复：{poi_id}")
            seen_ids.add(poi_id)
            scanned_id_rows += 1
    if scanned_id_rows != effective_rows:
        raise RQVAETrainingError(
            f"POI ID 行数 {scanned_id_rows} 与有效 Embedding 行数 {effective_rows} 不一致"
        )
    if config.max_rows is None and scanned_id_rows != full_rows:
        raise RQVAETrainingError(
            f"POI ID 行数 {scanned_id_rows} 与 Embedding 行数 {full_rows} 不一致"
        )

    for start in tqdm(
        range(0, effective_rows, config.finite_check_chunk_rows),
        desc="Embedding finite check",
        disable=not config.show_progress,
    ):
        stop = min(start + config.finite_check_chunk_rows, effective_rows)
        if not np.isfinite(embeddings[start:stop]).all():
            raise RQVAETrainingError(
                f"Embedding 行区间 [{start}, {stop}) 存在 NaN 或 Inf"
            )

    input_fingerprint = manifest.get("input", {}).get("fingerprint")
    manifest_rows = int(manifest.get("input", {}).get("total_rows", -1))
    if manifest_rows != full_rows:
        raise RQVAETrainingError("manifest input.total_rows 与 Embedding 不一致")
    validation = {
        "status": "passed",
        "full_rows": full_rows,
        "effective_rows": effective_rows,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_manifest": str(config.embedding_manifest_path),
        "embedding_fingerprint": input_fingerprint,
        "poi_ids_path": str(config.poi_ids_path),
        "poi_ids_sha256": _sha256_file(config.poi_ids_path),
        "poi_id_rows_scanned": scanned_id_rows,
        "poi_ids_unique": True,
        "vectors_all_finite": True,
        "validation_scope": "full" if config.max_rows is None else "prefix_smoke",
    }
    return embeddings, validation


def create_fixed_indices(
    row_count: int,
    validation_ratio: float,
    kmeans_sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Create deterministic validation and KMeans row sets shared by all runs."""

    validation_rows = max(1, int(math.ceil(row_count * validation_ratio)))
    if validation_rows >= row_count:
        validation_rows = row_count - 1
    split_rng = np.random.default_rng(seed)
    validation_indices = np.sort(
        split_rng.choice(row_count, size=validation_rows, replace=False)
    ).astype(np.int64, copy=False)
    validation_mask = np.zeros(row_count, dtype=np.bool_)
    validation_mask[validation_indices] = True
    train_indices = np.flatnonzero(~validation_mask).astype(np.int64, copy=False)
    sample_rows = min(kmeans_sample_size, len(train_indices))
    initialization_rng = np.random.default_rng(seed)
    initialization_indices = np.sort(
        initialization_rng.choice(train_indices, size=sample_rows, replace=False)
    ).astype(np.int64, copy=False)
    metadata = {
        "algorithm": "numpy.default_rng(seed).choice_without_replacement",
        "seed": seed,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "validation_indices_sha256": _sha256_indices(validation_indices),
        "kmeans_sample_rows": int(len(initialization_indices)),
        "kmeans_sample_indices_sha256": _sha256_indices(initialization_indices),
    }
    return validation_mask, validation_indices, initialization_indices, metadata


def build_model(config: RQVAETrainingConfig) -> RQVAE:
    return RQVAE(
        config.input_dim,
        config.hidden_dim,
        config.latent_dim,
        config.codebook_sizes,
        codebook_loss_weight=config.codebook_loss_weight,
        commitment_loss_weight=config.commitment_loss_weight,
    )


def _encode_initialization_latents(
    model: RQVAE,
    embeddings: np.ndarray,
    row_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    show_progress: bool,
) -> np.ndarray:
    latent = np.empty((len(row_indices), model.latent_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, len(row_indices), batch_size),
            desc="Encode KMeans sample",
            disable=not show_progress,
        ):
            stop = min(start + batch_size, len(row_indices))
            batch = np.asarray(embeddings[row_indices[start:stop]], dtype=np.float32)
            inputs = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            latent[start:stop] = model.encode(inputs).cpu().numpy()
    return latent


def _faiss_kmeans(
    values: np.ndarray,
    cluster_count: int,
    *,
    iterations: int,
    seed: int,
    gpu: bool,
    max_points_per_centroid: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        import faiss
    except ImportError as error:
        raise RQVAETrainingError("Faiss KMeans backend 不可用") from error
    if gpu and faiss.get_num_gpus() < 1:
        raise RQVAETrainingError("faiss_gpu 初始化要求至少一张可用 GPU")
    values = np.ascontiguousarray(values, dtype=np.float32)
    kmeans = faiss.Kmeans(
        values.shape[1],
        cluster_count,
        niter=iterations,
        nredo=1,
        verbose=True,
        seed=seed,
        gpu=gpu,
        min_points_per_centroid=1,
        max_points_per_centroid=max_points_per_centroid,
    )
    kmeans.train(values)
    distances, labels = kmeans.index.search(values, 1)
    return (
        np.asarray(kmeans.centroids, dtype=np.float32),
        labels[:, 0].astype(np.int64, copy=False),
        float(distances[:, 0].mean()),
    )


def _sklearn_kmeans(
    values: np.ndarray,
    cluster_count: int,
    *,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    from sklearn.cluster import KMeans

    estimator = KMeans(
        n_clusters=cluster_count,
        init="k-means++",
        n_init=1,
        max_iter=iterations,
        random_state=seed,
        algorithm="lloyd",
    )
    labels = estimator.fit_predict(np.asarray(values, dtype=np.float32))
    distances = values - estimator.cluster_centers_[labels]
    return (
        np.asarray(estimator.cluster_centers_, dtype=np.float32),
        labels.astype(np.int64, copy=False),
        float(np.mean(np.sum(distances * distances, axis=1))),
    )


def initialize_codebooks_kmeans(
    model: RQVAE,
    embeddings: np.ndarray,
    row_indices: np.ndarray,
    config: RQVAETrainingConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Initialize every residual codebook with sequential KMeans."""

    latent = _encode_initialization_latents(
        model,
        embeddings,
        row_indices,
        config.kmeans_batch_size,
        device,
        config.show_progress,
    )
    residual = latent.copy()
    results: list[dict[str, Any]] = []
    for level_index, codebook_size in enumerate(config.codebook_sizes):
        started = time.perf_counter()
        if config.kmeans_backend == "sklearn":
            centroids, labels, mean_distance = _sklearn_kmeans(
                residual,
                codebook_size,
                iterations=config.kmeans_iterations,
                seed=config.seed + level_index,
            )
        else:
            centroids, labels, mean_distance = _faiss_kmeans(
                residual,
                codebook_size,
                iterations=config.kmeans_iterations,
                seed=config.seed + level_index,
                gpu=config.kmeans_backend == "faiss_gpu",
                max_points_per_centroid=config.kmeans_max_points_per_centroid,
            )
        model.quantizer.set_codebook(
            level_index, torch.from_numpy(np.ascontiguousarray(centroids))
        )
        residual -= centroids[labels]
        used_codes = int(np.unique(labels).size)
        result = {
            "level": level_index + 1,
            "codebook_size": codebook_size,
            "used_codes_on_initialization_sample": used_codes,
            "dead_codes_on_initialization_sample": codebook_size - used_codes,
            "mean_squared_distance": mean_distance,
            "seconds": time.perf_counter() - started,
        }
        results.append(result)
        print(json.dumps({"kmeans_initialization": result}, ensure_ascii=False), flush=True)
    return results


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")


def _metric_accumulator(
    codebook_sizes: Sequence[int], *, collect_sid_codes: bool = False
) -> dict[str, Any]:
    return {
        "samples": 0,
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "codebook_loss": 0.0,
        "commitment_loss": 0.0,
        "reconstruction_cosine": 0.0,
        "code_counts": [np.zeros(size, dtype=np.int64) for size in codebook_sizes],
        "residual_norm_sums": np.zeros(len(codebook_sizes), dtype=np.float64),
        "sid_code_chunks": [] if collect_sid_codes else None,
    }


def _update_metrics(
    accumulator: dict[str, Any], output: RQVAEOutput, sample_count: int
) -> None:
    for name in (
        "total_loss",
        "reconstruction_loss",
        "codebook_loss",
        "commitment_loss",
        "reconstruction_cosine",
    ):
        value = float(getattr(output, name).detach().item())
        if not math.isfinite(value):
            raise RQVAETrainingError(f"训练指标 {name} 出现 NaN 或 Inf")
        accumulator[name] += value * sample_count
    accumulator["samples"] += sample_count
    residual_norms = output.residual_norms.detach()
    if not torch.isfinite(residual_norms).all():
        raise RQVAETrainingError("residual norm 出现 NaN 或 Inf")
    accumulator["residual_norm_sums"] += residual_norms.sum(dim=0).cpu().numpy()
    codes = output.codes.detach().cpu().numpy()
    for level_index, counts in enumerate(accumulator["code_counts"]):
        counts += np.bincount(codes[:, level_index], minlength=len(counts))
    if accumulator["sid_code_chunks"] is not None:
        accumulator["sid_code_chunks"].append(codes.astype(np.int32, copy=False))


def _finalize_metrics(accumulator: dict[str, Any]) -> dict[str, Any]:
    samples = int(accumulator["samples"])
    if samples <= 0:
        raise RQVAETrainingError("指标阶段没有样本")
    result = {
        name: float(accumulator[name] / samples)
        for name in (
            "total_loss",
            "reconstruction_loss",
            "codebook_loss",
            "commitment_loss",
            "reconstruction_cosine",
        )
    }
    result["samples"] = samples
    layers: list[dict[str, Any]] = []
    for level_index, counts in enumerate(accumulator["code_counts"]):
        used = int(np.count_nonzero(counts))
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
        layers.append(
            {
                "level": level_index + 1,
                "codebook_size": len(counts),
                "used_code_count": used,
                "dead_code_count": len(counts) - used,
                "codebook_utilization_ratio": used / len(counts),
                "normalized_entropy": normalized_entropy,
                "residual_norm": float(
                    accumulator["residual_norm_sums"][level_index] / samples
                ),
            }
        )
    result["layers"] = layers
    return result


def _monitor_collision_metrics(code_chunks: list[np.ndarray]) -> dict[str, Any]:
    if not code_chunks:
        raise RQVAETrainingError("固定监控子集没有 SID code")
    codes = np.concatenate(code_chunks, axis=0)
    _, bucket_sizes = np.unique(codes, axis=0, return_counts=True)
    poi_count = int(len(codes))
    distinct_sid_count = int(len(bucket_sizes))
    collision_excess_count = poi_count - distinct_sid_count
    colliding_poi_count = int(bucket_sizes[bucket_sizes > 1].sum())
    ordered_bucket_sizes = np.sort(bucket_sizes)
    p99_index = max(0, math.ceil(0.99 * len(ordered_bucket_sizes)) - 1)
    return {
        "poi_count": poi_count,
        "distinct_sid_count": distinct_sid_count,
        "distinct_sid_ratio": distinct_sid_count / poi_count,
        "collision_excess_count": collision_excess_count,
        "collision_excess_ratio": collision_excess_count / poi_count,
        "colliding_poi_count": colliding_poi_count,
        "colliding_poi_ratio": colliding_poi_count / poi_count,
        "bucket_size_p99": int(ordered_bucket_sizes[p99_index]),
        "bucket_size_max": int(ordered_bucket_sizes[-1]),
    }


def _batch_tensor(data: np.ndarray, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    batch = np.ascontiguousarray(data[rows], dtype=np.float32)
    return torch.from_numpy(batch).to(device)


def train_one_epoch(
    model: RQVAE,
    optimizer: torch.optim.Optimizer,
    embeddings: np.ndarray,
    row_count: int,
    validation_mask: np.ndarray,
    config: RQVAETrainingConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    model.train()
    accumulator = _metric_accumulator(config.codebook_sizes)
    block_starts = list(range(0, row_count, config.block_rows))
    epoch_rng = np.random.default_rng(config.seed + epoch)
    epoch_rng.shuffle(block_starts)
    progress = tqdm(
        total=int((~validation_mask).sum()),
        desc=f"Train epoch {epoch}",
        unit="poi",
        disable=not config.show_progress,
    )
    for block_start in block_starts:
        block_stop = min(block_start + config.block_rows, row_count)
        block = np.asarray(embeddings[block_start:block_stop], dtype=np.float32)
        local_rows = np.flatnonzero(~validation_mask[block_start:block_stop])
        epoch_rng.shuffle(local_rows)
        for start in range(0, len(local_rows), config.batch_size):
            rows = local_rows[start : start + config.batch_size]
            inputs = _batch_tensor(block, rows, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            if not torch.isfinite(output.total_loss):
                raise RQVAETrainingError("train total loss 出现 NaN 或 Inf")
            output.total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise RQVAETrainingError("梯度范数出现 NaN 或 Inf")
            optimizer.step()
            _update_metrics(accumulator, output, len(rows))
            progress.update(len(rows))
    progress.close()
    return _finalize_metrics(accumulator)


def validate_one_epoch(
    model: RQVAE,
    embeddings: np.ndarray,
    row_count: int,
    validation_mask: np.ndarray,
    config: RQVAETrainingConfig,
    device: torch.device,
    epoch: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    accumulator = _metric_accumulator(
        config.codebook_sizes, collect_sid_codes=True
    )
    progress = tqdm(
        total=int(validation_mask.sum()),
        desc=f"Validation epoch {epoch}",
        unit="poi",
        disable=not config.show_progress,
    )
    with torch.no_grad():
        for block_start in range(0, row_count, config.block_rows):
            block_stop = min(block_start + config.block_rows, row_count)
            local_rows = np.flatnonzero(validation_mask[block_start:block_stop])
            if not len(local_rows):
                continue
            block = np.asarray(embeddings[block_start:block_stop], dtype=np.float32)
            for start in range(0, len(local_rows), config.batch_size):
                rows = local_rows[start : start + config.batch_size]
                inputs = _batch_tensor(block, rows, device)
                output = model(inputs)
                _update_metrics(accumulator, output, len(rows))
                progress.update(len(rows))
    progress.close()
    metrics = _finalize_metrics(accumulator)
    monitor_metrics = _monitor_collision_metrics(accumulator["sid_code_chunks"])
    return metrics, monitor_metrics


def _save_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [device_state.cpu() for device_state in state["torch_cuda"]]
        )


def _truncate_metrics_after_epoch(path: Path, completed_epoch: int) -> None:
    if not path.is_file():
        if completed_epoch:
            raise RQVAETrainingError("恢复 checkpoint 存在但 train_metrics.jsonl 缺失")
        return
    kept_lines: list[str] = []
    seen_epochs: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                epoch = int(record["epoch"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise RQVAETrainingError(
                    f"train_metrics.jsonl 第 {line_number} 行无效"
                ) from error
            if epoch <= completed_epoch:
                kept_lines.append(line if line.endswith("\n") else line + "\n")
                seen_epochs.append(epoch)
    if seen_epochs != list(range(1, completed_epoch + 1)):
        raise RQVAETrainingError("训练指标 epoch 与恢复 checkpoint 不连续")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.writelines(kept_lines)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _allowed_training_outputs(config: RQVAETrainingConfig) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {
        "resolved_config.json",
        "last_checkpoint.pt",
        "train_metrics.jsonl",
        "evaluations",
        "sid_codes.npy",
        "sid_manifest.json",
        "metrics.json",
        "collision_cases.jsonl",
    }
    allowed.update(
        f"checkpoint_epoch_{epoch}.pt" for epoch in config.checkpoint_epochs
    )
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in allowed)
    if unexpected:
        raise RQVAETrainingError(
            f"实验目录包含未声明产物，拒绝继续：{unexpected[0]}"
        )


def _runtime_info(device: torch.device) -> dict[str, Any]:
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        payload["gpu"] = torch.cuda.get_device_name(device)
    return payload


def run_training(
    config: RQVAETrainingConfig,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Run a fixed-length, resumable RQ-VAE capacity experiment."""

    _allowed_training_outputs(config)
    signature = config_signature(config)
    resolved_path = config.output_dir / "resolved_config.json"
    last_path = config.output_dir / "last_checkpoint.pt"
    metrics_path = config.output_dir / "train_metrics.jsonl"

    existing_resolved: dict[str, Any] | None = None
    if resolved_path.is_file():
        with resolved_path.open("r", encoding="utf-8") as handle:
            existing_resolved = json.load(handle)
        if existing_resolved.get("config_signature") != signature:
            raise RQVAETrainingError("现有 resolved_config 与当前训练配置不一致")
        if existing_resolved.get("status") == "completed":
            return existing_resolved
        if not config.resume:
            raise RQVAETrainingError("实验目录已有未完成任务且 resume=false")

    _seed_everything(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RQVAETrainingError("配置要求 CUDA，但当前进程未检测到可用 GPU")
    embeddings, input_validation = validate_embedding_artifacts(config)
    row_count = int(input_validation["effective_rows"])
    validation_mask, _, initialization_indices, split_metadata = create_fixed_indices(
        row_count,
        config.validation_ratio,
        config.kmeans_sample_size,
        config.seed,
    )
    split_metadata["monitor_subset"] = {
        "source": "validation_split",
        "rows": split_metadata["validation_rows"],
        "indices_sha256": split_metadata["validation_indices_sha256"],
    }
    resolved: dict[str, Any] = {
        "schema_version": "rqvae-training-v2",
        "status": "initializing",
        "config_signature": signature,
        "shared_protocol_signature": shared_protocol_signature(config),
        "config": config_payload(config),
        "input_validation": input_validation,
        "split": split_metadata,
        "model": {
            "method": "vanilla_rqvae",
            "encoder": [config.input_dim, config.hidden_dim, config.latent_dim],
            "decoder": [config.latent_dim, config.hidden_dim, config.input_dim],
            "rq_layers": len(config.codebook_sizes),
            "codebook_sizes": list(config.codebook_sizes),
            "latent_dim": config.latent_dim,
            "codebook_loss_weight": config.codebook_loss_weight,
            "commitment_loss_weight": config.commitment_loss_weight,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
        },
        "scheduler": {"name": None},
        "checkpoint_protocol": {
            "fixed_epochs": list(config.checkpoint_epochs),
            "recovery_checkpoint": last_path.name,
            "early_stopping": False,
        },
        "runtime": _runtime_info(device),
        "git": _git_state(project_root),
        "started_at": (
            existing_resolved.get("started_at", _utc_now())
            if existing_resolved is not None
            else _utc_now()
        ),
    }
    if existing_resolved is not None:
        resolved["resumed_at"] = _utc_now()
    _write_json_atomic(resolved_path, resolved)

    model = build_model(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 1
    elapsed_before_resume = 0.0
    initialization_metrics: list[dict[str, Any]] = []

    if config.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("config_signature") != signature:
            raise RQVAETrainingError("last checkpoint 配置签名不一致")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        elapsed_before_resume = float(checkpoint.get("training_seconds", 0.0))
        initialization_metrics = checkpoint.get("initialization_metrics", [])
        if checkpoint.get("scheduler_state_dict") is not None:
            raise RQVAETrainingError("当前协议未配置 scheduler，但恢复状态包含 scheduler")
        rng_state = checkpoint.get("rng_state")
        if not isinstance(rng_state, dict):
            raise RQVAETrainingError("last checkpoint 缺少完整随机状态")
        _restore_rng_state(rng_state)
        _truncate_metrics_after_epoch(metrics_path, int(checkpoint["epoch"]))
        resolved["initialization"] = {
            "backend": config.kmeans_backend,
            "sample_rows": len(initialization_indices),
            "sample_indices_sha256": split_metadata[
                "kmeans_sample_indices_sha256"
            ],
            "iterations": config.kmeans_iterations,
            "levels": initialization_metrics,
        }
        resolved["status"] = "training"
        _write_json_atomic(resolved_path, resolved)
    else:
        init_started = time.perf_counter()
        initialization_metrics = initialize_codebooks_kmeans(
            model,
            embeddings,
            initialization_indices,
            config,
            device,
        )
        resolved["initialization"] = {
            "backend": config.kmeans_backend,
            "sample_rows": len(initialization_indices),
            "sample_indices_sha256": split_metadata[
                "kmeans_sample_indices_sha256"
            ],
            "iterations": config.kmeans_iterations,
            "levels": initialization_metrics,
            "seconds": time.perf_counter() - init_started,
        }
        resolved["status"] = "training"
        _write_json_atomic(resolved_path, resolved)
        metrics_path.unlink(missing_ok=True)

    training_started = time.perf_counter()
    last_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, config.max_epochs + 1):
            epoch_started = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                optimizer,
                embeddings,
                row_count,
                validation_mask,
                config,
                device,
                epoch,
            )
            validation_metrics, monitor_metrics = validate_one_epoch(
                model,
                embeddings,
                row_count,
                validation_mask,
                config,
                device,
                epoch,
            )
            collapse_levels = [
                layer["level"]
                for layer in validation_metrics["layers"]
                if layer["codebook_utilization_ratio"]
                < config.collapse_utilization_threshold
            ]
            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "monitor_sid": monitor_metrics,
                "collapse_warning_levels": collapse_levels,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "recorded_at": _utc_now(),
            }
            if epoch in config.checkpoint_epochs:
                fixed_path = config.output_dir / f"checkpoint_epoch_{epoch}.pt"
                _save_torch_atomic(
                    fixed_path,
                    {
                        "schema_version": "rqvae-fixed-checkpoint-v2",
                        "experiment_id": config.experiment_id,
                        "config_signature": signature,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "train_metrics": train_metrics,
                        "validation_metrics": validation_metrics,
                        "monitor_sid_metrics": monitor_metrics,
                        "codebook_sizes": list(config.codebook_sizes),
                        "saved_at": _utc_now(),
                    },
                )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            cumulative_seconds = (
                elapsed_before_resume + time.perf_counter() - training_started
            )
            _save_torch_atomic(
                last_path,
                {
                    "schema_version": "rqvae-recovery-v2",
                    "experiment_id": config.experiment_id,
                    "config_signature": signature,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": None,
                    "rng_state": _rng_state(),
                    "initialization_metrics": initialization_metrics,
                    "epoch_record": epoch_record,
                    "training_seconds": cumulative_seconds,
                    "saved_at": _utc_now(),
                },
            )
            print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
            last_epoch = epoch
    except BaseException as error:
        resolved["status"] = "failed"
        resolved["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "failed_at": _utc_now(),
        }
        _write_json_atomic(resolved_path, resolved)
        raise

    if last_epoch != config.max_epochs:
        raise RQVAETrainingError(
            f"训练仅完成到 epoch {last_epoch}，目标为 {config.max_epochs}"
        )
    missing_checkpoints = [
        epoch
        for epoch in config.checkpoint_epochs
        if not (config.output_dir / f"checkpoint_epoch_{epoch}.pt").is_file()
    ]
    if missing_checkpoints:
        raise RQVAETrainingError(f"固定 checkpoint 缺失：{missing_checkpoints}")
    training_seconds = elapsed_before_resume + time.perf_counter() - training_started
    resolved.update(
        {
            "status": "completed",
            "initialization": resolved.get(
                "initialization",
                {
                    "backend": config.kmeans_backend,
                    "sample_rows": len(initialization_indices),
                    "sample_indices_sha256": split_metadata[
                        "kmeans_sample_indices_sha256"
                    ],
                    "iterations": config.kmeans_iterations,
                    "levels": initialization_metrics,
                },
            ),
            "training_result": {
                "actual_batch_size": config.batch_size,
                "last_epoch": last_epoch,
                "fixed_checkpoint_epochs": list(config.checkpoint_epochs),
                "stop_reason": "max_epochs",
                "training_seconds": training_seconds,
            },
            "finished_at": _utc_now(),
        }
    )
    _write_json_atomic(resolved_path, resolved)
    return resolved


def _config_from_payload(payload: dict[str, Any]) -> RQVAETrainingConfig:
    converted = dict(payload)
    for key in (
        "source_config",
        "embeddings_path",
        "poi_ids_path",
        "embedding_manifest_path",
        "poi_data_path",
        "output_dir",
    ):
        converted[key] = Path(converted[key])
    converted["codebook_sizes"] = tuple(converted["codebook_sizes"])
    converted["checkpoint_epochs"] = tuple(converted["checkpoint_epochs"])
    return RQVAETrainingConfig(**converted)


def _write_prefix_poi_ids(source: Path, destination: Path, row_count: int) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    rows = 0
    try:
        with (
            source.open("r", encoding="utf-8") as source_handle,
            temporary.open("w", encoding="utf-8") as destination_handle,
        ):
            for rows, line in enumerate(source_handle, start=1):
                if rows > row_count:
                    break
                destination_handle.write(line)
        if min(rows, row_count) != row_count:
            raise RQVAETrainingError(
                f"POI ID 文件不足 {row_count} 行，无法导出 smoke SID"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def export_checkpoint_sid(
    run_dir: Path,
    checkpoint_path: Path,
    *,
    output_dir: Path | None = None,
    device_name: str | None = None,
    batch_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Export SID codes from one user-selected fixed checkpoint and evaluate them."""

    from .sid_evaluation import evaluate_sid, write_evaluation_outputs

    run_dir = run_dir.resolve()
    evaluation_dir = (output_dir or run_dir).resolve()
    if evaluation_dir != run_dir and run_dir not in evaluation_dir.parents:
        raise RQVAETrainingError("SID 评估输出目录必须位于对应实验目录内")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    if evaluation_dir != run_dir:
        allowed_evaluation_outputs = {
            "sid_codes.npy",
            "sid_manifest.json",
            "metrics.json",
            "collision_cases.jsonl",
            "_smoke_poi_ids.jsonl",
        }
        unexpected = sorted(
            path.name
            for path in evaluation_dir.iterdir()
            if path.name not in allowed_evaluation_outputs
        )
        if unexpected:
            raise RQVAETrainingError(
                f"SID 评估目录包含未声明产物，拒绝覆盖：{unexpected[0]}"
            )
    resolved_path = run_dir / "resolved_config.json"
    selected_path = (
        checkpoint_path
        if checkpoint_path.is_absolute()
        else run_dir / checkpoint_path
    ).resolve()
    if selected_path.parent != run_dir:
        raise RQVAETrainingError("checkpoint 必须位于对应实验目录内")
    if not resolved_path.is_file() or not selected_path.is_file():
        raise RQVAETrainingError("导出要求 completed resolved_config 和固定 checkpoint")
    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved = json.load(handle)
    if resolved.get("status") != "completed":
        raise RQVAETrainingError("RQ-VAE 训练尚未完成，不能导出 SID")
    raw_config = _require_mapping(resolved.get("config"), "resolved_config.config")
    config = _config_from_payload(raw_config)
    device = torch.device(device_name or config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RQVAETrainingError("SID 导出要求 CUDA，但当前进程未检测到可用 GPU")
    export_batch_size = batch_size or config.batch_size
    if export_batch_size <= 0:
        raise RQVAETrainingError("export batch_size 必须大于 0")
    embeddings = np.load(config.embeddings_path, mmap_mode="r", allow_pickle=False)
    row_count = int(resolved["input_validation"]["effective_rows"])
    if row_count > len(embeddings):
        raise RQVAETrainingError("resolved effective_rows 超出 Embedding 行数")

    checkpoint = torch.load(selected_path, map_location=device, weights_only=False)
    if checkpoint.get("config_signature") != resolved.get("config_signature"):
        raise RQVAETrainingError("固定 checkpoint 配置签名不一致")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch not in config.checkpoint_epochs:
        raise RQVAETrainingError("checkpoint epoch 不在固定候选列表中")
    model = build_model(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sid_path = evaluation_dir / "sid_codes.npy"
    temporary_sid_path = evaluation_dir / ".sid_codes.npy.tmp"
    temporary_sid_path.unlink(missing_ok=True)
    sid_codes = np.lib.format.open_memmap(
        temporary_sid_path,
        mode="w+",
        dtype=np.int32,
        shape=(row_count, len(config.codebook_sizes)),
    )
    export_started = time.perf_counter()
    try:
        with torch.no_grad():
            for start in tqdm(
                range(0, row_count, export_batch_size),
                desc="Export full SID",
                unit="poi",
                disable=not config.show_progress,
            ):
                stop = min(start + export_batch_size, row_count)
                batch = np.asarray(embeddings[start:stop], dtype=np.float32)
                inputs = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
                codes = model.encode_codes(inputs).cpu().numpy().astype(np.int32, copy=False)
                sid_codes[start:stop] = codes
        sid_codes.flush()
        del sid_codes
        os.replace(temporary_sid_path, sid_path)
    finally:
        if "sid_codes" in locals():
            del sid_codes
        temporary_sid_path.unlink(missing_ok=True)

    full_rows = int(resolved["input_validation"]["full_rows"])
    if row_count == full_rows:
        evaluation_poi_ids = config.poi_ids_path
        poi_ids_sha256 = resolved["input_validation"]["poi_ids_sha256"]
    else:
        evaluation_poi_ids = evaluation_dir / "_smoke_poi_ids.jsonl"
        _write_prefix_poi_ids(config.poi_ids_path, evaluation_poi_ids, row_count)
        poi_ids_sha256 = _sha256_file(evaluation_poi_ids)

    exported_at = _utc_now()
    manifest = {
        "schema_version": "sid-input-v1",
        "experiment_id": config.experiment_id,
        "method": "vanilla_rqvae",
        "embedding": {
            "manifest": os.path.relpath(
                config.embedding_manifest_path, evaluation_dir
            ),
            "fingerprint": resolved["input_validation"]["embedding_fingerprint"],
            "shape": resolved["input_validation"]["embedding_shape"],
            "dtype": resolved["input_validation"]["embedding_dtype"],
        },
        "sid_codes": {
            "path": sid_path.name,
            "shape": [row_count, len(config.codebook_sizes)],
            "dtype": "int32",
        },
        "poi_ids": {
            "path": os.path.relpath(evaluation_poi_ids, evaluation_dir),
            "sha256": poi_ids_sha256,
        },
        "sid_layers": len(config.codebook_sizes),
        "codebook_sizes": list(config.codebook_sizes),
        "checkpoint": {
            "path": os.path.relpath(selected_path, evaluation_dir),
            "sha256": _sha256_file(selected_path),
            "epoch": checkpoint_epoch,
        },
        "seed": config.seed,
        "exported_at": exported_at,
        "export_seconds": time.perf_counter() - export_started,
    }
    manifest_path = evaluation_dir / "sid_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    metrics, cases = evaluate_sid(
        manifest_path,
        config.poi_data_path,
        max_cases=config.evaluation_max_cases,
        max_pois_per_case=config.evaluation_max_pois_per_case,
    )
    write_evaluation_outputs(evaluation_dir, metrics, cases)
    evaluation_key = f"epoch_{checkpoint_epoch}"
    sid_evaluations = resolved.setdefault("sid_evaluations", {})
    sid_evaluations[evaluation_key] = {
        "status": "completed",
        "rows": row_count,
        "sid_codes": str(sid_path),
        "sid_manifest": str(manifest_path),
        "metrics": str(evaluation_dir / "metrics.json"),
        "collision_cases": str(evaluation_dir / "collision_cases.jsonl"),
        "exported_at": exported_at,
        "export_seconds": manifest["export_seconds"],
        "checkpoint": selected_path.name,
        "checkpoint_epoch": checkpoint_epoch,
    }
    _write_json_atomic(resolved_path, resolved)
    return manifest, metrics

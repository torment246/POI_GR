"""Streaming training loop for the Beijing-adapted GNPR-SID RQ-VAE."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from poi_gr.sid_evaluation import compute_prefix_metrics

from .gnpr_rqvae import GnprRQVAE, GnprRQVAEOutput, dense_gnpr_batch
from .gnpr_sid_input import (
    GnprFeatureDimensions,
    GnprSidInput,
    iter_gnpr_sid_input_batches,
    load_gnpr_sid_input_manifest,
)


class GnprTrainingError(RuntimeError):
    """Raised when the GNPR training contract or runtime state is invalid."""


@dataclass(frozen=True)
class GnprTrainingConfig:
    source_config: Path
    experiment_id: str
    input_dir: Path
    output_dir: Path
    expected_rows: int
    input_dim: int
    hidden_dims: tuple[int, ...]
    latent_dim: int
    codebook_sizes: tuple[int, ...]
    dropout: float
    quantization_loss_weight: float
    commitment_beta: float
    diversity_loss_weight: float
    diversity_scale: float
    diversity_temperature: float
    reconstruction_loss: str
    kmeans_backend: str
    kmeans_sample_size: int
    kmeans_iterations: int
    seed: int
    device: str
    batch_size: int
    max_epochs: int
    checkpoint_epochs: tuple[int, ...]
    validation_basis_points: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    resume: bool
    max_rows: int | None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GnprTrainingError(f"{name} 必须是 mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GnprTrainingError(f"{name} 必须是正整数")
    return value


def _resolve(value: Any, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise GnprTrainingError(f"{name} 必须是非空路径")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_gnpr_training_config(
    config_path: Path,
    project_root: Path,
    experiment_id: str,
    *,
    output_dir: Path | None = None,
    device: str | None = None,
    max_epochs: int | None = None,
    checkpoint_epochs: Sequence[int] | None = None,
    batch_size: int | None = None,
    max_rows: int | None = None,
    codebook_sizes: Sequence[int] | None = None,
    kmeans_sample_size: int | None = None,
    kmeans_backend: str | None = None,
    resume: bool | None = None,
    validation_basis_points: int | None = None,
) -> GnprTrainingConfig:
    """Load one GNPR capacity experiment with optional smoke overrides."""

    raw = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "配置")
    data = _mapping(raw.get("data"), "data")
    model = _mapping(raw.get("model"), "model")
    loss = _mapping(raw.get("loss"), "loss")
    initialization = _mapping(raw.get("initialization"), "initialization")
    training = _mapping(raw.get("training"), "training")
    experiments = _mapping(raw.get("experiments"), "experiments")
    experiment = _mapping(experiments.get(experiment_id), f"experiments.{experiment_id}")
    declared_sizes = experiment.get("codebook_sizes") if codebook_sizes is None else codebook_sizes
    if not isinstance(declared_sizes, (list, tuple)):
        raise GnprTrainingError("codebook_sizes 必须是数组")
    resolved_sizes = tuple(_positive_int(value, "codebook_sizes") for value in declared_sizes)
    if len(resolved_sizes) != 3:
        raise GnprTrainingError("GNPR 固定使用三层码本")
    hidden = model.get("hidden_dims")
    if not isinstance(hidden, list):
        raise GnprTrainingError("model.hidden_dims 必须是数组")
    resolved_epochs = max_epochs or _positive_int(training.get("max_epochs"), "training.max_epochs")
    raw_checkpoints = (
        experiment.get("checkpoint_epochs", training.get("checkpoint_epochs"))
        if checkpoint_epochs is None
        else checkpoint_epochs
    )
    if not isinstance(raw_checkpoints, (list, tuple)):
        raise GnprTrainingError("checkpoint_epochs 必须是数组")
    resolved_checkpoints = tuple(_positive_int(value, "checkpoint_epochs") for value in raw_checkpoints)
    if tuple(sorted(set(resolved_checkpoints))) != resolved_checkpoints:
        raise GnprTrainingError("checkpoint_epochs 必须严格递增")
    if resolved_checkpoints[-1] > resolved_epochs:
        raise GnprTrainingError("checkpoint epoch 不能超过 max_epochs")
    resolved_validation_basis_points = _positive_int(
        (
            training.get("validation_basis_points")
            if validation_basis_points is None
            else validation_basis_points
        ),
        "training.validation_basis_points",
    )
    if resolved_validation_basis_points >= 10000:
        raise GnprTrainingError("validation_basis_points 必须小于 10000")
    resolved_backend = kmeans_backend or str(initialization.get("backend"))
    if resolved_backend not in {"faiss_gpu", "faiss_cpu", "sklearn"}:
        raise GnprTrainingError("K-Means backend 无效")
    resolved_max_rows = max_rows
    if resolved_max_rows is not None and resolved_max_rows <= 1:
        raise GnprTrainingError("max_rows 必须大于 1")
    reconstruction_loss = str(loss.get("reconstruction", "mse"))
    if reconstruction_loss not in {"mse", "balanced_mse"}:
        raise GnprTrainingError("loss.reconstruction 只能是 mse 或 balanced_mse")
    output_root = _resolve(raw.get("output_root"), project_root, "output_root")
    resolved_output = (
        output_dir.resolve()
        if output_dir is not None and output_dir.is_absolute()
        else (project_root / output_dir).resolve()
        if output_dir is not None
        else output_root / experiment_id
    )
    return GnprTrainingConfig(
        source_config=config_path.resolve(),
        experiment_id=experiment_id,
        input_dir=_resolve(data.get("input_dir"), project_root, "data.input_dir"),
        output_dir=resolved_output,
        expected_rows=_positive_int(data.get("expected_rows"), "data.expected_rows"),
        input_dim=_positive_int(model.get("input_dim"), "model.input_dim"),
        hidden_dims=tuple(_positive_int(value, "model.hidden_dims") for value in hidden),
        latent_dim=_positive_int(model.get("latent_dim"), "model.latent_dim"),
        codebook_sizes=resolved_sizes,
        dropout=float(model.get("dropout", 0.1)),
        quantization_loss_weight=float(loss.get("quantization_weight", 1.0)),
        commitment_beta=float(loss.get("commitment_beta", 0.25)),
        diversity_loss_weight=float(loss.get("diversity_weight", 0.25)),
        diversity_scale=float(loss.get("released_diversity_scale", 0.05)),
        diversity_temperature=float(loss.get("diversity_temperature", 0.5)),
        reconstruction_loss=reconstruction_loss,
        kmeans_backend=resolved_backend,
        kmeans_sample_size=(
            kmeans_sample_size
            or _positive_int(initialization.get("sample_size"), "initialization.sample_size")
        ),
        kmeans_iterations=_positive_int(initialization.get("iterations"), "initialization.iterations"),
        seed=_positive_int(training.get("seed"), "training.seed"),
        device=device or str(training.get("device", "cuda")),
        batch_size=batch_size or _positive_int(training.get("batch_size"), "training.batch_size"),
        max_epochs=resolved_epochs,
        checkpoint_epochs=resolved_checkpoints,
        validation_basis_points=resolved_validation_basis_points,
        learning_rate=float(training.get("learning_rate")),
        weight_decay=float(training.get("weight_decay")),
        gradient_clip_norm=float(training.get("gradient_clip_norm")),
        resume=bool(training.get("resume", True) if resume is None else resume),
        max_rows=resolved_max_rows,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _config_payload(config: GnprTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("source_config", "input_dir", "output_dir"):
        payload[key] = str(payload[key])
    return payload


def _signature(config: GnprTrainingConfig) -> str:
    payload = _config_payload(replace(config, resume=False))
    for key in ("output_dir", "resume", "max_epochs", "checkpoint_epochs"):
        payload.pop(key)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _split_value(poi_id: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{poi_id}".encode("utf-8"), digest_size=8, person=b"gnpr-split-v1"
    ).digest()
    return int.from_bytes(digest, "little") % 10000


def _build_model(config: GnprTrainingConfig) -> GnprRQVAE:
    return GnprRQVAE(
        config.input_dim,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        codebook_sizes=config.codebook_sizes,
        dropout=config.dropout,
        quantization_loss_weight=config.quantization_loss_weight,
        diversity_loss_weight=config.diversity_loss_weight,
        commitment_beta=config.commitment_beta,
        diversity_scale=config.diversity_scale,
        diversity_temperature=config.diversity_temperature,
        reconstruction_loss=config.reconstruction_loss,
    )


def _iter_rows(
    config: GnprTrainingConfig,
    *,
    shuffle_seed: int | None,
):
    dimensions, batches = iter_gnpr_sid_input_batches(
        config.input_dir,
        batch_size=config.batch_size,
        max_rows=config.max_rows,
        shuffle_seed=shuffle_seed,
    )
    return dimensions, batches


def _collect_initialization_latents(
    model: GnprRQVAE,
    config: GnprTrainingConfig,
    device: torch.device,
) -> np.ndarray:
    dimensions, batches = _iter_rows(config, shuffle_seed=config.seed)
    chunks: list[np.ndarray] = []
    collected = 0
    model.eval()
    with torch.no_grad():
        for rows in batches:
            train_rows = tuple(
                row
                for row in rows
                if _split_value(row.poi_id, config.seed) >= config.validation_basis_points
            )
            if not train_rows:
                continue
            remaining = config.kmeans_sample_size - collected
            train_rows = train_rows[:remaining]
            inputs = dense_gnpr_batch(train_rows, dimensions=dimensions, device=device)
            chunks.append(model.encode(inputs).cpu().numpy().astype(np.float32, copy=False))
            collected += len(train_rows)
            if collected >= config.kmeans_sample_size:
                break
    if collected < max(config.codebook_sizes):
        raise GnprTrainingError(
            f"K-Means 样本 {collected} 少于最大码本 {max(config.codebook_sizes)}"
        )
    return np.concatenate(chunks, axis=0)


def _run_kmeans(
    values: np.ndarray,
    clusters: int,
    *,
    backend: str,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.ascontiguousarray(values, dtype=np.float32)
    if backend == "sklearn":
        from sklearn.cluster import KMeans

        estimator = KMeans(
            n_clusters=clusters,
            n_init=1,
            max_iter=iterations,
            random_state=seed,
            algorithm="lloyd",
        )
        labels = estimator.fit_predict(values)
        return estimator.cluster_centers_.astype(np.float32), labels.astype(np.int64)
    try:
        import faiss
    except ImportError as error:
        raise GnprTrainingError("Faiss K-Means 不可用") from error
    gpu = backend == "faiss_gpu"
    if gpu and faiss.get_num_gpus() < 1:
        raise GnprTrainingError("faiss_gpu 未检测到 GPU")
    kmeans = faiss.Kmeans(
        values.shape[1],
        clusters,
        niter=iterations,
        nredo=1,
        seed=seed,
        gpu=gpu,
        verbose=True,
        min_points_per_centroid=1,
        max_points_per_centroid=2048,
    )
    kmeans.train(values)
    _, labels = kmeans.index.search(values, 1)
    return np.asarray(kmeans.centroids, dtype=np.float32), labels[:, 0].astype(np.int64)


def initialize_codebooks(
    model: GnprRQVAE,
    config: GnprTrainingConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Initialize all residual levels with sequential K-Means."""

    latent = _collect_initialization_latents(model, config, device)
    residual = latent.copy()
    results: list[dict[str, Any]] = []
    for level, size in enumerate(config.codebook_sizes):
        started = time.perf_counter()
        centers, labels = _run_kmeans(
            residual,
            size,
            backend=config.kmeans_backend,
            iterations=config.kmeans_iterations,
            seed=config.seed + level,
        )
        model.quantizer.set_codebook(level, torch.from_numpy(centers))
        residual -= centers[labels]
        results.append(
            {
                "level": level + 1,
                "codebook_size": size,
                "used_codes": int(np.unique(labels).size),
                "mean_residual_norm": float(np.linalg.norm(residual, axis=1).mean()),
                "seconds": time.perf_counter() - started,
            }
        )
    return results


def _empty_metrics(config: GnprTrainingConfig) -> dict[str, Any]:
    return {
        "samples": 0,
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "reconstruction_cosine": 0.0,
        "quantization_loss": 0.0,
        "diversity_loss": 0.0,
        "positive_squared_error": 0.0,
        "positive_count": 0,
        "top_active_hits": 0,
        "topk_evaluated": False,
        "code_counts": [np.zeros(size, dtype=np.int64) for size in config.codebook_sizes],
        "codes": [],
        "category_indices": [],
        "region_indices": [],
    }


def _update_metrics(
    metrics: dict[str, Any],
    output: GnprRQVAEOutput,
    inputs: torch.Tensor,
    rows: Sequence[GnprSidInput],
    *,
    collect_codes: bool,
    compute_topk: bool,
) -> None:
    count = inputs.shape[0]
    metrics["samples"] += count
    for name in (
        "total_loss",
        "reconstruction_loss",
        "reconstruction_cosine",
        "quantization_loss",
        "diversity_loss",
    ):
        metrics[name] += float(getattr(output, name).detach()) * count
    positive = inputs > 0
    metrics["positive_squared_error"] += float(
        ((output.reconstruction.detach() - inputs).square() * positive).sum()
    )
    metrics["positive_count"] += int(positive.sum())
    if compute_topk:
        metrics["topk_evaluated"] = True
        max_active = int(positive.sum(dim=1).max())
        top_indices = output.reconstruction.detach().topk(max_active, dim=1).indices
        for row_index in range(count):
            active_count = int(positive[row_index].sum())
            metrics["top_active_hits"] += int(
                positive[row_index, top_indices[row_index, :active_count]].sum()
            )
    codes = output.codes.detach().cpu().numpy()
    for level, counts in enumerate(metrics["code_counts"]):
        counts += np.bincount(codes[:, level], minlength=len(counts))
    if collect_codes:
        metrics["codes"].append(codes.astype(np.int32, copy=False))
        metrics["category_indices"].append(
            np.asarray([row.category_index for row in rows], dtype=np.int32)
        )
        metrics["region_indices"].append(
            np.asarray([row.region_index for row in rows], dtype=np.int32)
        )


def _finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    samples = int(metrics["samples"])
    if samples <= 0:
        raise GnprTrainingError("指标没有样本")
    positive_count = int(metrics["positive_count"])
    result = {
        name: metrics[name] / samples
        for name in (
            "total_loss",
            "reconstruction_loss",
            "reconstruction_cosine",
            "quantization_loss",
            "diversity_loss",
        )
    }
    result.update(
        {
            "samples": samples,
            "positive_mse": metrics["positive_squared_error"] / positive_count,
            "active_topk_recall": (
                metrics["top_active_hits"] / positive_count
                if metrics["topk_evaluated"]
                else None
            ),
            "layers": [],
        }
    )
    for level, counts in enumerate(metrics["code_counts"]):
        used = int(np.count_nonzero(counts))
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        result["layers"].append(
            {
                "level": level + 1,
                "used_codes": used,
                "utilization": used / len(counts),
                "normalized_entropy": entropy / math.log(len(counts)),
            }
        )
    if metrics["codes"]:
        codes = np.concatenate(metrics["codes"])
        _, full_inverse, bucket_sizes = np.unique(
            codes,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        distinct = len(bucket_sizes)
        result["sid"] = {
            "distinct": distinct,
            "unique_ratio": distinct / len(codes),
            "collision_excess_ratio": 1.0 - distinct / len(codes),
        }
        category_prefixes = compute_prefix_metrics(
            codes,
            np.concatenate(metrics["category_indices"]),
            full_inverse=full_inverse,
            full_bucket_sizes=bucket_sizes,
        )
        region_prefixes = compute_prefix_metrics(
            codes,
            np.concatenate(metrics["region_indices"]),
            full_inverse=full_inverse,
            full_bucket_sizes=bucket_sizes,
        )
        result["prefixes"] = []
        for category, region in zip(
            category_prefixes,
            region_prefixes,
            strict=True,
        ):
            category_purity = dict(category["category_purity"])
            region_purity = dict(region["category_purity"])
            category_purity["label_field"] = "category_index"
            region_purity["label_field"] = "region_index"
            result["prefixes"].append(
                {
                    "depth": category["depth"],
                    "prefix_bucket_count": category["prefix_bucket_count"],
                    "category_purity": category_purity,
                    "region_purity": region_purity,
                }
            )
    return result


def _run_epoch(
    model: GnprRQVAE,
    config: GnprTrainingConfig,
    device: torch.device,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    metrics = _empty_metrics(config)
    dimensions, batches = _iter_rows(
        config,
        shuffle_seed=(config.seed + epoch if training and config.max_rows is None else None),
    )
    for rows in batches:
        selected = tuple(
            row
            for row in rows
            if (
                _split_value(row.poi_id, config.seed) >= config.validation_basis_points
            )
            == training
        )
        if not selected:
            continue
        inputs = dense_gnpr_batch(selected, dimensions=dimensions, device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            if not torch.isfinite(output.total_loss):
                raise GnprTrainingError("训练 loss 出现 NaN/Inf")
            output.total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise GnprTrainingError("梯度出现 NaN/Inf")
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(inputs)
        _update_metrics(
            metrics,
            output,
            inputs,
            selected,
            collect_codes=not training,
            compute_topk=not training,
        )
    return _finalize_metrics(metrics)


def _save_checkpoint(
    path: Path,
    *,
    model: GnprRQVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    signature: str,
    initialization: list[dict[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "epoch": epoch,
            "config_signature": signature,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "initialization": initialization,
        },
        temporary,
    )
    os.replace(temporary, path)

def export_full_sid(
    model: GnprRQVAE,
    config: GnprTrainingConfig,
    device: torch.device,
    *,
    output_dir: Path,
    write_poi_ids: bool,
) -> dict[str, Any]:
    """Stream all effective POIs and evaluate one checkpoint on the full catalog."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dimensions, batches = _iter_rows(config, shuffle_seed=None)
    expected = config.max_rows or config.expected_rows
    codes_path = output_dir / "sid_codes.npy"
    ids_path = config.output_dir / "poi_ids.jsonl"
    temporary_ids = ids_path.with_name(f".{ids_path.name}.tmp")
    codes = np.lib.format.open_memmap(
        codes_path,
        mode="w+",
        dtype=np.int32,
        shape=(expected, 3),
    )
    category_indices = np.empty(expected, dtype=np.int32)
    region_indices = np.empty(expected, dtype=np.int32)
    reconstruction_cosine_sum = 0.0
    offset = 0
    model.eval()
    id_handle = (
        temporary_ids.open("w", encoding="utf-8") if write_poi_ids else None
    )
    try:
        context = torch.no_grad()
        context.__enter__()
        for rows in batches:
            inputs = dense_gnpr_batch(rows, dimensions=dimensions, device=device)
            output = model(inputs)
            batch_codes = output.codes.cpu().numpy().astype(np.int32)
            reconstruction_cosine_sum += (
                float(output.reconstruction_cosine) * len(rows)
            )
            stop = offset + len(rows)
            codes[offset:stop] = batch_codes
            category_indices[offset:stop] = [row.category_index for row in rows]
            region_indices[offset:stop] = [row.region_index for row in rows]
            if id_handle is not None:
                for row in rows:
                    id_handle.write(json.dumps(row.poi_id) + "\n")
            offset = stop
    finally:
        context.__exit__(None, None, None)
        if id_handle is not None:
            id_handle.close()
    if offset != expected:
        raise GnprTrainingError(f"SID 导出行数 {offset} != {expected}")
    codes.flush()
    if write_poi_ids:
        os.replace(temporary_ids, ids_path)
    unique_codes, full_inverse, bucket_sizes = np.unique(
        codes,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    layer_metrics = []
    for level, size in enumerate(config.codebook_sizes):
        counts = np.bincount(codes[:, level], minlength=size)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        used = int(np.count_nonzero(counts))
        layer_metrics.append(
            {
                "level": level + 1,
                "used_codes": used,
                "utilization": float(used / size),
                "normalized_entropy": entropy / math.log(size),
            }
        )
    category_prefixes = compute_prefix_metrics(
        codes,
        category_indices,
        full_inverse=full_inverse,
        full_bucket_sizes=bucket_sizes,
    )
    region_prefixes = compute_prefix_metrics(
        codes,
        region_indices,
        full_inverse=full_inverse,
        full_bucket_sizes=bucket_sizes,
    )
    prefix_metrics = []
    for category, region in zip(category_prefixes, region_prefixes, strict=True):
        category_purity = dict(category["category_purity"])
        region_purity = dict(region["category_purity"])
        category_purity["label_field"] = "category_index"
        region_purity["label_field"] = "region_index"
        prefix_metrics.append(
            {
                "depth": category["depth"],
                "prefix_bucket_count": category["prefix_bucket_count"],
                "category_purity": category_purity,
                "region_purity": region_purity,
            }
        )
    metrics = {
        "poi_count": expected,
        "distinct_sid_count": int(len(unique_codes)),
        "unique_sid_ratio": float(len(unique_codes) / expected),
        "collision_excess_count": int(expected - len(unique_codes)),
        "collision_excess_ratio": float(1.0 - len(unique_codes) / expected),
        "colliding_poi_count": int(bucket_sizes[bucket_sizes > 1].sum()),
        "p99_bucket_size": float(np.quantile(bucket_sizes, 0.99, method="higher")),
        "max_bucket_size": int(bucket_sizes.max()),
        "reconstruction_cosine": reconstruction_cosine_sum / expected,
        "layers": layer_metrics,
        "prefixes": prefix_metrics,
        "artifacts": {
            "sid_codes": str(codes_path),
            "poi_ids": str(ids_path),
        },
    }
    _write_json(output_dir / "sid_metrics.json", metrics)
    return metrics


def run_gnpr_training(config: GnprTrainingConfig) -> dict[str, Any]:
    """Train, resume, checkpoint, and export one GNPR RQ-VAE capacity."""

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GnprTrainingError("配置要求 CUDA，但当前环境不可用")
    dimensions, manifest = load_gnpr_sid_input_manifest(config.input_dir)
    source_rows = int(manifest.get("stats", {}).get("output_behavior_poi_count", -1))
    if source_rows != config.expected_rows:
        raise GnprTrainingError(f"输入行数 {source_rows} != {config.expected_rows}")
    if dimensions.total_dim != config.input_dim:
        raise GnprTrainingError("输入维度与配置不一致")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    signature = _signature(config)
    resolved_path = config.output_dir / "resolved_config.json"
    last_path = config.output_dir / "last_checkpoint.pt"
    metrics_path = config.output_dir / "train_metrics.jsonl"
    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start_epoch = 1
    initialization: list[dict[str, Any]] = []
    if config.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("config_signature") != signature:
            raise GnprTrainingError("恢复 checkpoint 配置签名不一致")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        initialization = checkpoint["initialization"]
    else:
        initialization = initialize_codebooks(model, config, device)
        metrics_path.unlink(missing_ok=True)
    resolved = {
        "schema_version": "gnpr-rqvae-training-v1",
        "status": "training",
        "config_signature": signature,
        "config": _config_payload(config),
        "initialization": initialization,
        "model": {
            "encoder": [config.input_dim, *config.hidden_dims, config.latent_dim],
            "decoder": [config.latent_dim, *config.hidden_dims, config.input_dim],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
    }
    _write_json(resolved_path, resolved)
    started = time.perf_counter()
    if start_epoch == 1:
        _save_checkpoint(
            config.output_dir / "initialization_checkpoint.pt",
            model=model,
            optimizer=optimizer,
            epoch=0,
            signature=signature,
            initialization=initialization,
        )
    for epoch in range(start_epoch, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = _run_epoch(
            model, config, device, epoch=epoch, optimizer=optimizer
        )
        validation_metrics = (
            _run_epoch(model, config, device, epoch=epoch, optimizer=None)
            if epoch in config.checkpoint_epochs
            else None
        )
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_started,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            signature=signature,
            initialization=initialization,
        )
        if epoch in config.checkpoint_epochs:
            _save_checkpoint(
                config.output_dir / f"checkpoint_epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                signature=signature,
                initialization=initialization,
            )
    validation_by_epoch: dict[int, dict[str, Any]] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["validation"] is not None:
                validation_by_epoch[int(record["epoch"])] = record["validation"]
    evaluations: dict[str, Any] = {}
    for index, epoch in enumerate(config.checkpoint_epochs):
        checkpoint_path = config.output_dir / f"checkpoint_epoch_{epoch}.pt"
        if not checkpoint_path.is_file():
            raise GnprTrainingError(f"缺少待评估 checkpoint：{checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("config_signature") != signature:
            raise GnprTrainingError(f"{checkpoint_path.name} 配置签名不一致")
        model.load_state_dict(checkpoint["model_state_dict"])
        full_metrics = export_full_sid(
            model,
            config,
            device,
            output_dir=config.output_dir / "evaluations" / f"epoch_{epoch}",
            write_poi_ids=index == 0,
        )
        evaluations[str(epoch)] = {
            "validation": validation_by_epoch.get(epoch),
            "full_catalog": full_metrics,
        }
    resolved.update(
        {
            "status": "completed",
            "completed_epoch": config.max_epochs,
            "training_seconds": time.perf_counter() - started,
            "checkpoint_evaluations": evaluations,
        }
    )
    _write_json(resolved_path, resolved)
    (config.output_dir / "_SUCCESS").touch()
    return resolved

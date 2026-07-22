"""Train and evaluate a MiniOneRec-style RQ-VAE for POI embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


class RQVAEConfigError(ValueError):
    pass


class RQVAEDataError(ValueError):
    pass


@dataclass(frozen=True)
class RQVAEDataConfig:
    embeddings_path: Path
    ids_path: Path
    manifest_path: Path
    expected_rows: int
    expected_dim: int


@dataclass(frozen=True)
class RQVAEModelConfig:
    input_dim: int
    hidden_dims: tuple[int, ...]
    latent_dim: int
    codebook_sizes: tuple[int, ...]
    dropout: float
    commitment_weight: float
    quantization_weight: float
    kmeans_init: bool
    kmeans_iterations: int
    sinkhorn_epsilons: tuple[float, ...]
    sinkhorn_iterations: int

    @property
    def num_codebooks(self) -> int:
        return len(self.codebook_sizes)


@dataclass(frozen=True)
class RQVAETrainingConfig:
    device: str
    batch_size: int
    eval_batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    collision_eval_interval: int
    num_workers: int
    block_size: int
    seed: int
    use_bf16: bool
    gradient_clip_norm: float | None


@dataclass(frozen=True)
class RQVAEExportConfig:
    resolve_collisions: bool
    sinkhorn_epsilon: float
    sinkhorn_iterations: int
    max_collision_iterations: int
    sinkhorn_batch_elements: int


@dataclass(frozen=True)
class RQVAEOutputConfig:
    dir: Path
    resume: bool


@dataclass(frozen=True)
class RQVAEJobConfig:
    job_name: str
    data: RQVAEDataConfig
    model: RQVAEModelConfig
    training: RQVAETrainingConfig
    export: RQVAEExportConfig
    output: RQVAEOutputConfig


@dataclass(frozen=True)
class EmbeddingSource:
    total_rows: int
    selected_rows: int
    embedding_dim: int
    dtype: str
    embedding_fingerprint: str


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RQVAEConfigError(f"{name} 必须是 YAML mapping")
    return value


def _resolve_path(value: Any, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RQVAEConfigError(f"{name} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RQVAEConfigError(f"{name} 必须是正整数")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RQVAEConfigError(f"{name} 必须是非负整数")
    return value


def _positive_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise RQVAEConfigError(f"{name} 必须是正数")
    return float(value)


def _non_negative_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise RQVAEConfigError(f"{name} 必须是非负数")
    return float(value)


def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RQVAEConfigError(f"{name} 必须是非空正整数列表")
    return tuple(_positive_int(item, f"{name}[{index}]") for index, item in enumerate(value))


def _non_negative_float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise RQVAEConfigError(f"{name} 必须是非空非负数列表")
    return tuple(
        _non_negative_float(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def load_rqvae_config(config_path: Path, project_root: Path) -> RQVAEJobConfig:
    """Load an RQ-VAE job configuration."""

    with config_path.open("r", encoding="utf-8") as handle:
        raw = _require_mapping(yaml.safe_load(handle), "配置根节点")
    data_raw = _require_mapping(raw.get("data"), "data")
    model_raw = _require_mapping(raw.get("model"), "model")
    training_raw = _require_mapping(raw.get("training"), "training")
    export_raw = _require_mapping(raw.get("export"), "export")
    output_raw = _require_mapping(raw.get("output"), "output")

    job_name = raw.get("job_name")
    if not isinstance(job_name, str) or not job_name.strip():
        raise RQVAEConfigError("job_name 必须是非空字符串")

    gradient_clip_raw = training_raw.get("gradient_clip_norm", 1.0)
    gradient_clip_norm = (
        None
        if gradient_clip_raw is None
        else _positive_float(gradient_clip_raw, "training.gradient_clip_norm")
    )
    data = RQVAEDataConfig(
        embeddings_path=_resolve_path(
            data_raw.get("embeddings_path"), project_root, "data.embeddings_path"
        ),
        ids_path=_resolve_path(data_raw.get("ids_path"), project_root, "data.ids_path"),
        manifest_path=_resolve_path(
            data_raw.get("manifest_path"), project_root, "data.manifest_path"
        ),
        expected_rows=_positive_int(data_raw.get("expected_rows"), "data.expected_rows"),
        expected_dim=_positive_int(data_raw.get("expected_dim"), "data.expected_dim"),
    )
    model = RQVAEModelConfig(
        input_dim=_positive_int(model_raw.get("input_dim"), "model.input_dim"),
        hidden_dims=_positive_int_tuple(
            model_raw.get("hidden_dims"), "model.hidden_dims"
        ),
        latent_dim=_positive_int(model_raw.get("latent_dim"), "model.latent_dim"),
        codebook_sizes=_positive_int_tuple(
            model_raw.get("codebook_sizes"), "model.codebook_sizes"
        ),
        dropout=_non_negative_float(model_raw.get("dropout", 0.0), "model.dropout"),
        commitment_weight=_positive_float(
            model_raw.get("commitment_weight", 0.25),
            "model.commitment_weight",
        ),
        quantization_weight=_positive_float(
            model_raw.get("quantization_weight", 1.0),
            "model.quantization_weight",
        ),
        kmeans_init=bool(model_raw.get("kmeans_init", True)),
        kmeans_iterations=_positive_int(
            model_raw.get("kmeans_iterations", 100),
            "model.kmeans_iterations",
        ),
        sinkhorn_epsilons=_non_negative_float_tuple(
            model_raw.get("sinkhorn_epsilons", [0.0, 0.0, 0.0]),
            "model.sinkhorn_epsilons",
        ),
        sinkhorn_iterations=_positive_int(
            model_raw.get("sinkhorn_iterations", 50),
            "model.sinkhorn_iterations",
        ),
    )
    training = RQVAETrainingConfig(
        device=str(training_raw.get("device", "auto")),
        batch_size=_positive_int(
            training_raw.get("batch_size"), "training.batch_size"
        ),
        eval_batch_size=_positive_int(
            training_raw.get("eval_batch_size"), "training.eval_batch_size"
        ),
        epochs=_positive_int(training_raw.get("epochs"), "training.epochs"),
        learning_rate=_positive_float(
            training_raw.get("learning_rate"), "training.learning_rate"
        ),
        weight_decay=_non_negative_float(
            training_raw.get("weight_decay", 0.0),
            "training.weight_decay",
        ),
        warmup_epochs=_non_negative_int(
            training_raw.get("warmup_epochs", 0),
            "training.warmup_epochs",
        ),
        collision_eval_interval=_positive_int(
            training_raw.get("collision_eval_interval", 1),
            "training.collision_eval_interval",
        ),
        num_workers=_non_negative_int(
            training_raw.get("num_workers", 0),
            "training.num_workers",
        ),
        block_size=_positive_int(
            training_raw.get("block_size"), "training.block_size"
        ),
        seed=int(training_raw.get("seed", 20260719)),
        use_bf16=bool(training_raw.get("use_bf16", True)),
        gradient_clip_norm=gradient_clip_norm,
    )
    export = RQVAEExportConfig(
        resolve_collisions=bool(export_raw.get("resolve_collisions", True)),
        sinkhorn_epsilon=_positive_float(
            export_raw.get("sinkhorn_epsilon", 0.003),
            "export.sinkhorn_epsilon",
        ),
        sinkhorn_iterations=_positive_int(
            export_raw.get("sinkhorn_iterations", 50),
            "export.sinkhorn_iterations",
        ),
        max_collision_iterations=_positive_int(
            export_raw.get("max_collision_iterations", 20),
            "export.max_collision_iterations",
        ),
        sinkhorn_batch_elements=_positive_int(
            export_raw.get("sinkhorn_batch_elements", 4_000_000),
            "export.sinkhorn_batch_elements",
        ),
    )
    output = RQVAEOutputConfig(
        dir=_resolve_path(output_raw.get("dir"), project_root, "output.dir"),
        resume=bool(output_raw.get("resume", True)),
    )
    config = RQVAEJobConfig(
        job_name=job_name,
        data=data,
        model=model,
        training=training,
        export=export,
        output=output,
    )
    _validate_config(config)
    return config


def _validate_config(config: RQVAEJobConfig) -> None:
    if config.model.input_dim != config.data.expected_dim:
        raise RQVAEConfigError("model.input_dim 必须等于 data.expected_dim")
    if len(config.model.sinkhorn_epsilons) != config.model.num_codebooks:
        raise RQVAEConfigError(
            "model.sinkhorn_epsilons 数量必须等于 codebook_sizes 数量"
        )
    if config.model.dropout >= 1:
        raise RQVAEConfigError("model.dropout 必须小于 1")
    if max(config.model.codebook_sizes) > np.iinfo(np.uint16).max:
        raise RQVAEConfigError("当前 SID 导出只支持 codebook size <= 65535")
    if config.model.kmeans_init and config.training.batch_size < max(
        config.model.codebook_sizes
    ):
        raise RQVAEConfigError("启用 K-Means 初始化时 batch_size 不能小于最大码本")


def apply_rqvae_overrides(
    config: RQVAEJobConfig,
    *,
    output_dir: Path | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    num_workers: int | None = None,
    resume: bool | None = None,
) -> RQVAEJobConfig:
    """Apply command-line overrides to an RQ-VAE job."""

    training = replace(
        config.training,
        device=device or config.training.device,
        batch_size=config.training.batch_size if batch_size is None else batch_size,
        epochs=config.training.epochs if epochs is None else epochs,
        num_workers=(
            config.training.num_workers if num_workers is None else num_workers
        ),
    )
    output = replace(
        config.output,
        dir=output_dir or config.output.dir,
        resume=config.output.resume if resume is None else resume,
    )
    updated = replace(config, training=training, output=output)
    _validate_config(updated)
    return updated


class EmbeddingMemmapDataset(Dataset[torch.Tensor]):
    """Read individual embedding rows without loading the full NPY array."""

    def __init__(self, path: Path, rows: int | None = None) -> None:
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2:
            raise RQVAEDataError(f"Embedding 必须是二维数组，实际 shape={array.shape}")
        self.path = path
        self.total_rows = int(array.shape[0])
        self.embedding_dim = int(array.shape[1])
        self.rows = self.total_rows if rows is None else rows
        if self.rows <= 0 or self.rows > self.total_rows:
            raise RQVAEDataError(
                f"请求读取 {self.rows} 行，但 Embedding 只有 {self.total_rows} 行"
            )
        self._array: np.ndarray | None = None

    def __len__(self) -> int:
        return self.rows

    def _get_array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        return self._array

    def __getitem__(self, index: int) -> torch.Tensor:
        row = np.array(self._get_array()[index], dtype=np.float32, copy=True)
        return torch.from_numpy(row)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_array"] = None
        return state


class IndexSampler(Sampler[int]):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return (int(index) for index in self.indices)

    def __len__(self) -> int:
        return int(self.indices.size)


class BlockShuffleSampler(Sampler[int]):
    """Shuffle large row blocks while keeping reads sequential inside each block."""

    def __init__(self, row_count: int, block_size: int, seed: int) -> None:
        self.row_count = row_count
        self.block_size = block_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        block_count = math.ceil(self.row_count / self.block_size)
        generator = np.random.default_rng(self.seed + self.epoch)
        for block_index in generator.permutation(block_count):
            start = int(block_index) * self.block_size
            stop = min(start + self.block_size, self.row_count)
            for index in range(start, stop):
                yield int(index)

    def __len__(self) -> int:
        return self.row_count


def _kmeans_centers(
    samples: torch.Tensor,
    num_clusters: int,
    iterations: int,
    seed: int,
) -> torch.Tensor:
    from sklearn.cluster import KMeans

    values = samples.detach().float().cpu().numpy()
    if values.shape[0] < num_clusters:
        raise RuntimeError(
            f"K-Means 样本数 {values.shape[0]} 小于码本大小 {num_clusters}"
        )
    estimator = KMeans(
        n_clusters=num_clusters,
        max_iter=iterations,
        n_init=1,
        random_state=seed,
    )
    centers = estimator.fit(values).cluster_centers_
    return torch.from_numpy(centers).to(
        device=samples.device,
        dtype=torch.float32,
    )


def _squared_distances(
    samples: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    samples_float = samples.float()
    codebook_float = codebook.float()
    return (
        samples_float.square().sum(dim=-1, keepdim=True)
        + codebook_float.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * samples_float @ codebook_float.t()
    )


def _sinkhorn_assign(
    distances: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    max_distance = distances.amax()
    min_distance = distances.amin()
    middle = (max_distance + min_distance) / 2
    amplitude = (max_distance - middle).clamp_min(1e-5)
    centered = (distances - middle) / amplitude
    assignment = torch.exp(-centered.double() / epsilon)
    assignment /= assignment.sum().clamp_min(1e-300)
    sample_count, code_count = assignment.shape
    for _ in range(iterations):
        assignment /= assignment.sum(dim=1, keepdim=True).clamp_min(1e-300)
        assignment /= sample_count
        assignment /= assignment.sum(dim=0, keepdim=True).clamp_min(1e-300)
        assignment /= code_count
    assignment *= sample_count
    return assignment.argmax(dim=-1)


class VectorQuantizer(nn.Module):
    """Learnable VQ codebook with optional K-Means and Sinkhorn assignment."""

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        *,
        commitment_weight: float,
        kmeans_init: bool,
        kmeans_iterations: int,
        kmeans_seed: int,
        sinkhorn_epsilon: float,
        sinkhorn_iterations: int,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iterations = kmeans_iterations
        self.kmeans_seed = kmeans_seed
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.register_buffer(
            "initialized",
            torch.tensor(not kmeans_init, dtype=torch.bool),
        )
        if kmeans_init:
            nn.init.zeros_(self.embedding.weight)
        else:
            nn.init.uniform_(
                self.embedding.weight,
                -1.0 / codebook_size,
                1.0 / codebook_size,
            )

    @torch.no_grad()
    def initialize(self, samples: torch.Tensor) -> None:
        centers = _kmeans_centers(
            samples,
            self.codebook_size,
            self.kmeans_iterations,
            self.kmeans_seed,
        )
        self.embedding.weight.copy_(centers)
        self.initialized.fill_(True)

    def forward(
        self,
        residual: torch.Tensor,
        *,
        use_sinkhorn: bool,
    ) -> dict[str, torch.Tensor]:
        flat_residual = residual.reshape(-1, self.embedding_dim).float()
        if not bool(self.initialized.item()):
            if not self.training:
                raise RuntimeError("码本尚未完成 K-Means 初始化")
            self.initialize(flat_residual)

        distances = _squared_distances(flat_residual, self.embedding.weight)
        if use_sinkhorn and self.sinkhorn_epsilon > 0:
            codes = _sinkhorn_assign(
                distances,
                self.sinkhorn_epsilon,
                self.sinkhorn_iterations,
            )
        else:
            codes = distances.argmin(dim=-1)
        selected = self.embedding(codes).view_as(residual).float()
        residual_float = residual.float()
        commitment_loss = F.mse_loss(selected.detach(), residual_float)
        codebook_loss = F.mse_loss(selected, residual_float.detach())
        quantization_loss = (
            codebook_loss + self.commitment_weight * commitment_loss
        )
        straight_through = residual_float + (
            selected - residual_float
        ).detach()
        return {
            "quantized": straight_through,
            "codes": codes.view(residual.shape[:-1]),
            "quantization_loss": quantization_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }


class ResidualVectorQuantizer(nn.Module):
    """Apply learnable vector quantizers to successive residuals."""

    def __init__(self, config: RQVAEModelConfig, seed: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                VectorQuantizer(
                    codebook_size,
                    config.latent_dim,
                    commitment_weight=config.commitment_weight,
                    kmeans_init=config.kmeans_init,
                    kmeans_iterations=config.kmeans_iterations,
                    kmeans_seed=seed + level,
                    sinkhorn_epsilon=config.sinkhorn_epsilons[level],
                    sinkhorn_iterations=config.sinkhorn_iterations,
                )
                for level, codebook_size in enumerate(config.codebook_sizes)
            ]
        )

    def forward(
        self,
        latent: torch.Tensor,
        *,
        use_sinkhorn: bool,
    ) -> dict[str, torch.Tensor]:
        residual = latent.float()
        quantized_sum = torch.zeros_like(residual)
        codes: list[torch.Tensor] = []
        quantization_losses: list[torch.Tensor] = []
        codebook_losses: list[torch.Tensor] = []
        commitment_losses: list[torch.Tensor] = []
        last_residual = residual
        for level, quantizer in enumerate(self.layers):
            if level == len(self.layers) - 1:
                last_residual = residual
            result = quantizer(residual, use_sinkhorn=use_sinkhorn)
            quantized = result["quantized"]
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            codes.append(result["codes"])
            quantization_losses.append(result["quantization_loss"])
            codebook_losses.append(result["codebook_loss"])
            commitment_losses.append(result["commitment_loss"])
        return {
            "quantized": quantized_sum,
            "codes": torch.stack(codes, dim=-1),
            "quantization_loss": torch.stack(quantization_losses).mean(),
            "codebook_loss": torch.stack(codebook_losses).mean(),
            "commitment_loss": torch.stack(commitment_losses).mean(),
            "last_residual": last_residual,
        }


def _build_mlp(
    dimensions: Sequence[int],
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:])
    ):
        layers.append(nn.Dropout(dropout))
        linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if index != len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    """MLP autoencoder with MiniOneRec-style residual quantization."""

    def __init__(self, config: RQVAEModelConfig, *, seed: int = 20260719) -> None:
        super().__init__()
        self.config = config
        encoder_dims = (
            config.input_dim,
            *config.hidden_dims,
            config.latent_dim,
        )
        self.encoder = _build_mlp(encoder_dims, config.dropout)
        self.quantizer = ResidualVectorQuantizer(config, seed)
        self.decoder = _build_mlp(tuple(reversed(encoder_dims)), config.dropout)

    def encode(
        self,
        embeddings: torch.Tensor,
        *,
        use_sinkhorn: bool = False,
    ) -> dict[str, torch.Tensor]:
        latent = self.encoder(embeddings)
        return self.quantizer(latent, use_sinkhorn=use_sinkhorn)

    def forward(
        self,
        embeddings: torch.Tensor,
        *,
        use_sinkhorn: bool = False,
    ) -> dict[str, torch.Tensor]:
        quantized = self.encode(embeddings, use_sinkhorn=use_sinkhorn)
        reconstruction = self.decoder(quantized["quantized"])
        return {
            "reconstruction": reconstruction,
            **quantized,
        }


def compute_rqvae_losses(
    embeddings: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    quantization_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute MiniOneRec reconstruction and quantization objectives."""

    reconstruction = outputs["reconstruction"].float()
    target = embeddings.float()
    squared_error = (reconstruction - target).square()
    reconstruction_mse = squared_error.mean()
    reconstruction_l2 = squared_error.sum(dim=1).mean()
    cosine_similarity = F.cosine_similarity(
        reconstruction,
        target,
        dim=1,
        eps=1e-8,
    ).mean()
    quantization_loss = outputs["quantization_loss"].float()
    total_loss = reconstruction_mse + quantization_weight * quantization_loss
    return {
        "loss": total_loss,
        "reconstruction_mse": reconstruction_mse,
        "reconstruction_l2": reconstruction_l2,
        "cosine_similarity": cosine_similarity,
        "quantization_loss": quantization_loss,
        "codebook_loss": outputs["codebook_loss"].float(),
        "commitment_loss": outputs["commitment_loss"].float(),
    }


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            count += chunk.count(b"\n")
    return count


def inspect_embedding_source(
    config: RQVAEDataConfig,
    *,
    max_rows: int | None,
) -> EmbeddingSource:
    """Validate the completed embedding artifact used by RQ-VAE."""

    for path in (
        config.embeddings_path,
        config.ids_path,
        config.manifest_path,
    ):
        if not path.is_file():
            raise RQVAEDataError(f"缺少输入产物：{path}")
    with config.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "completed":
        raise RQVAEDataError("Embedding manifest 状态不是 completed")

    array = np.load(config.embeddings_path, mmap_mode="r")
    if array.ndim != 2:
        raise RQVAEDataError(f"Embedding shape 非法：{array.shape}")
    total_rows, embedding_dim = map(int, array.shape)
    if total_rows != config.expected_rows:
        raise RQVAEDataError(
            f"Embedding 行数 {total_rows} != expected_rows {config.expected_rows}"
        )
    if embedding_dim != config.expected_dim:
        raise RQVAEDataError(
            f"Embedding 维度 {embedding_dim} != expected_dim {config.expected_dim}"
        )
    manifest_shape = manifest.get("output", {}).get("shape")
    if manifest_shape != [total_rows, embedding_dim]:
        raise RQVAEDataError("Embedding manifest shape 与 NPY 不一致")
    id_rows = _count_lines(config.ids_path)
    if id_rows != total_rows:
        raise RQVAEDataError(f"POI ID 行数 {id_rows} != Embedding 行数 {total_rows}")

    selected_rows = total_rows if max_rows is None else max_rows
    if selected_rows <= 1 or selected_rows > total_rows:
        raise RQVAEDataError(
            f"max_rows 必须在 2 和 {total_rows} 之间，实际为 {selected_rows}"
        )
    embedding_fingerprint = str(
        manifest.get("input", {}).get("fingerprint")
        or manifest.get("signature")
        or ""
    )
    if not embedding_fingerprint:
        raise RQVAEDataError("Embedding manifest 缺少可复现指纹")
    return EmbeddingSource(
        total_rows=total_rows,
        selected_rows=selected_rows,
        embedding_dim=embedding_dim,
        dtype=str(array.dtype),
        embedding_fingerprint=embedding_fingerprint,
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前进程无法访问 GPU")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _serializable_config(config: RQVAEJobConfig, project_root: Path) -> dict[str, Any]:
    payload = asdict(config)
    payload["data"]["embeddings_path"] = _display_path(
        config.data.embeddings_path, project_root
    )
    payload["data"]["ids_path"] = _display_path(config.data.ids_path, project_root)
    payload["data"]["manifest_path"] = _display_path(
        config.data.manifest_path, project_root
    )
    payload["output"]["dir"] = _display_path(config.output.dir, project_root)
    return payload


def _job_signature(
    config: RQVAEJobConfig,
    source: EmbeddingSource,
    max_rows: int | None,
) -> str:
    payload = {
        "config": _serializable_config(config, Path("/")),
        "embedding_fingerprint": source.embedding_fingerprint,
        "embedding_shape": [source.total_rows, source.embedding_dim],
        "embedding_dtype": source.dtype,
        "max_rows": max_rows,
    }
    payload["config"]["output"].pop("dir", None)
    payload["config"]["output"].pop("resume", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _make_loader(
    dataset: EmbeddingMemmapDataset,
    *,
    sampler: Sampler[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[torch.Tensor]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )
    return DataLoader(**kwargs)


def _usage_metrics(
    counts: Sequence[torch.Tensor],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for level, level_counts in enumerate(counts):
        total = float(level_counts.sum().item())
        used = int((level_counts > 0).sum().item())
        if total:
            probabilities = level_counts[level_counts > 0].float() / total
            perplexity = float(torch.exp(-(probabilities * probabilities.log()).sum()))
        else:
            perplexity = 0.0
        metrics.append(
            {
                "level": level + 1,
                "used_codes": used,
                "total_codes": int(level_counts.numel()),
                "usage_rate": used / int(level_counts.numel()),
                "perplexity": perplexity,
            }
        )
    return metrics


def _run_train_epoch(
    model: RQVAE,
    loader: DataLoader[torch.Tensor],
    *,
    device: torch.device,
    quantization_weight: float,
    use_bf16: bool,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    gradient_clip_norm: float | None,
    description: str,
) -> dict[str, Any]:
    model.train()
    metric_names = (
        "loss",
        "reconstruction_mse",
        "reconstruction_l2",
        "cosine_similarity",
        "quantization_loss",
        "codebook_loss",
        "commitment_loss",
    )
    totals = {name: 0.0 for name in metric_names}
    row_count = 0
    code_counts = [
        torch.zeros(size, dtype=torch.long, device=device)
        for size in model.config.codebook_sizes
    ]
    progress = tqdm(loader, desc=description, unit="batch")
    for embeddings in progress:
        embeddings = embeddings.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
        ):
            outputs = model(embeddings, use_sinkhorn=True)
            losses = compute_rqvae_losses(
                embeddings,
                outputs,
                quantization_weight,
            )
        losses["loss"].backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        scheduler.step()

        batch_rows = int(embeddings.shape[0])
        row_count += batch_rows
        for name in metric_names:
            totals[name] += float(losses[name].detach().item()) * batch_rows
        codes = outputs["codes"].detach()
        for level, size in enumerate(model.config.codebook_sizes):
            code_counts[level] += torch.bincount(
                codes[:, level],
                minlength=size,
            )
        progress.set_postfix(
            loss=f"{totals['loss'] / row_count:.5f}",
            recon_l2=f"{totals['reconstruction_l2'] / row_count:.4f}",
            cosine=f"{totals['cosine_similarity'] / row_count:.4f}",
            quant=f"{totals['quantization_loss'] / row_count:.6f}",
        )
    if row_count == 0:
        raise RuntimeError("DataLoader 没有返回任何训练数据")
    result = {name: value / row_count for name, value in totals.items()}
    result["rows"] = row_count
    result["codebooks"] = _usage_metrics(
        [counts.cpu() for counts in code_counts]
    )
    result["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    return result


def _collision_metrics(sids: np.ndarray) -> dict[str, Any]:
    _, counts = np.unique(sids, axis=0, return_counts=True)
    collision_counts = counts[counts > 1]
    collision_pois = int(collision_counts.sum()) if collision_counts.size else 0
    duplicate_assignments = int(sids.shape[0] - counts.size)
    return {
        "total_rows": int(sids.shape[0]),
        "unique_sids": int(counts.size),
        "unique_sid_rate": float(counts.size / sids.shape[0]),
        "collision_rate": float(duplicate_assignments / sids.shape[0]),
        "duplicate_assignments": duplicate_assignments,
        "collision_groups": int(collision_counts.size),
        "collision_pois": collision_pois,
        "collision_poi_rate": float(collision_pois / sids.shape[0]),
        "max_collision_group_size": (
            int(collision_counts.max()) if collision_counts.size else 1
        ),
        "collision_group_p95_size": (
            float(np.percentile(collision_counts, 95))
            if collision_counts.size
            else 1.0
        ),
    }


@torch.inference_mode()
def _evaluate_collision(
    model: RQVAE,
    loader: DataLoader[torch.Tensor],
    *,
    row_count: int,
    device: torch.device,
    use_bf16: bool,
    description: str,
) -> dict[str, Any]:
    model.eval()
    sids = np.empty(
        (row_count, model.config.num_codebooks),
        dtype=np.uint16,
    )
    next_row = 0
    progress = tqdm(loader, desc=description, unit="batch")
    for embeddings in progress:
        embeddings = embeddings.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
        ):
            outputs = model.encode(embeddings, use_sinkhorn=False)
        batch_rows = int(embeddings.shape[0])
        batch_end = next_row + batch_rows
        sids[next_row:batch_end] = (
            outputs["codes"].cpu().numpy().astype(np.uint16, copy=False)
        )
        next_row = batch_end
    if next_row != row_count:
        raise RuntimeError(f"碰撞评估行数 {next_row} != 预期 {row_count}")
    return _collision_metrics(sids)


def _save_checkpoint(
    path: Path,
    *,
    model: RQVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_train_loss: float,
    best_collision_rate: float,
    signature: str,
) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_train_loss": best_train_loss,
            "best_collision_rate": best_collision_rate,
            "signature": signature,
        },
        temp_path,
    )
    os.replace(temp_path, path)


def _load_checkpoint(
    path: Path,
    *,
    model: RQVAE,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    signature: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("signature") != signature:
        raise RuntimeError("已有 RQ-VAE checkpoint 与当前数据或配置不一致")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def _batched_sinkhorn_assign(
    residuals: torch.Tensor,
    codebook: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    group_count, group_size, embedding_dim = residuals.shape
    flat = residuals.reshape(-1, embedding_dim)
    distances = _squared_distances(flat, codebook).view(
        group_count,
        group_size,
        codebook.shape[0],
    )
    max_distance = distances.amax(dim=(1, 2), keepdim=True)
    min_distance = distances.amin(dim=(1, 2), keepdim=True)
    middle = (max_distance + min_distance) / 2
    amplitude = (max_distance - middle).clamp_min(1e-5)
    centered = (distances - middle) / amplitude
    assignment = torch.exp(-centered.double() / epsilon)
    assignment /= assignment.sum(dim=(1, 2), keepdim=True).clamp_min(1e-300)
    for _ in range(iterations):
        assignment /= assignment.sum(dim=2, keepdim=True).clamp_min(1e-300)
        assignment /= group_size
        assignment /= assignment.sum(dim=1, keepdim=True).clamp_min(1e-300)
        assignment /= codebook.shape[0]
    assignment *= group_size
    return assignment.argmax(dim=2)


@torch.inference_mode()
def resolve_last_level_collisions(
    raw_sids: np.ndarray,
    last_residuals: np.ndarray,
    codebook: torch.Tensor,
    *,
    device: torch.device,
    epsilon: float,
    sinkhorn_iterations: int,
    max_collision_iterations: int,
    sinkhorn_batch_elements: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply MiniOneRec Sinkhorn reassignment to colliding SID groups."""

    resolved = np.array(raw_sids, copy=True)
    best = np.array(resolved, copy=True)
    best_metrics = _collision_metrics(best)
    history: list[dict[str, Any]] = []
    codebook = codebook.detach().float().to(device)
    code_count = int(codebook.shape[0])

    for iteration in range(1, max_collision_iterations + 1):
        _, inverse, counts = np.unique(
            resolved,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        collision_group_ids = np.flatnonzero(counts > 1)
        if collision_group_ids.size == 0:
            break
        order = np.argsort(inverse, kind="stable")
        starts = np.empty(counts.size, dtype=np.int64)
        starts[0] = 0
        if counts.size > 1:
            np.cumsum(counts[:-1], out=starts[1:])

        for group_size in np.unique(counts[collision_group_ids]):
            group_ids = collision_group_ids[counts[collision_group_ids] == group_size]
            groups_per_batch = max(
                1,
                sinkhorn_batch_elements // (int(group_size) * code_count),
            )
            for group_start in range(0, group_ids.size, groups_per_batch):
                batch_group_ids = group_ids[
                    group_start : group_start + groups_per_batch
                ]
                row_indices = np.stack(
                    [
                        order[
                            starts[group_id] : starts[group_id] + int(group_size)
                        ]
                        for group_id in batch_group_ids
                    ]
                )
                residual_batch = torch.from_numpy(
                    np.asarray(last_residuals[row_indices], dtype=np.float32)
                ).to(device)
                assignments = _batched_sinkhorn_assign(
                    residual_batch,
                    codebook,
                    epsilon=epsilon,
                    iterations=sinkhorn_iterations,
                )
                resolved[row_indices, -1] = assignments.cpu().numpy().astype(
                    np.uint16,
                    copy=False,
                )

        metrics = _collision_metrics(resolved)
        record = {"iteration": iteration, **metrics}
        history.append(record)
        print(json.dumps({"collision_resolution": record}, ensure_ascii=False))
        if metrics["collision_rate"] < best_metrics["collision_rate"]:
            best = np.array(resolved, copy=True)
            best_metrics = metrics
        if metrics["duplicate_assignments"] == 0:
            break
    return best, history


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    output = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=array.dtype,
        shape=array.shape,
    )
    output[:] = array
    output.flush()
    del output
    os.replace(temp_path, path)


@torch.inference_mode()
def export_sids_and_evaluate(
    model: RQVAE,
    dataset: EmbeddingMemmapDataset,
    *,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_bf16: bool,
    quantization_weight: float,
    export_config: RQVAEExportConfig | None = None,
) -> dict[str, Any]:
    """Export final SIDs and compute reconstruction, codebook, and collision metrics."""

    model.eval()
    indices = np.arange(len(dataset), dtype=np.int64)
    loader = _make_loader(
        dataset,
        sampler=IndexSampler(indices),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    raw_sids = np.empty(
        (len(dataset), model.config.num_codebooks),
        dtype=np.uint16,
    )
    last_residuals = np.empty(
        (len(dataset), model.config.latent_dim),
        dtype=np.float16,
    )
    metric_names = (
        "loss",
        "reconstruction_mse",
        "reconstruction_l2",
        "cosine_similarity",
        "quantization_loss",
        "codebook_loss",
        "commitment_loss",
    )
    totals = {name: 0.0 for name in metric_names}
    code_counts = [
        torch.zeros(size, dtype=torch.long)
        for size in model.config.codebook_sizes
    ]
    next_row = 0
    progress = tqdm(loader, desc="RQ-VAE final evaluate", unit="batch")
    for embeddings in progress:
        embeddings = embeddings.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
        ):
            outputs = model(embeddings, use_sinkhorn=False)
            losses = compute_rqvae_losses(
                embeddings,
                outputs,
                quantization_weight,
            )
        batch_rows = int(embeddings.shape[0])
        batch_end = next_row + batch_rows
        codes = outputs["codes"].cpu()
        raw_sids[next_row:batch_end] = codes.numpy().astype(
            np.uint16,
            copy=False,
        )
        last_residuals[next_row:batch_end] = (
            outputs["last_residual"].cpu().numpy().astype(np.float16, copy=False)
        )
        for name in metric_names:
            totals[name] += float(losses[name].item()) * batch_rows
        for level, size in enumerate(model.config.codebook_sizes):
            code_counts[level] += torch.bincount(
                codes[:, level],
                minlength=size,
            )
        next_row = batch_end

    nearest_metrics = _collision_metrics(raw_sids)
    nearest_code_counts = code_counts
    final_sids = raw_sids
    resolution_history: list[dict[str, Any]] = []
    if export_config is not None and export_config.resolve_collisions:
        final_sids, resolution_history = resolve_last_level_collisions(
            raw_sids,
            last_residuals,
            model.quantizer.layers[-1].embedding.weight,
            device=device,
            epsilon=export_config.sinkhorn_epsilon,
            sinkhorn_iterations=export_config.sinkhorn_iterations,
            max_collision_iterations=export_config.max_collision_iterations,
            sinkhorn_batch_elements=export_config.sinkhorn_batch_elements,
        )
    del last_residuals
    _write_npy_atomic(output_path, final_sids)
    final_metrics = _collision_metrics(final_sids)
    final_code_counts = [
        torch.from_numpy(
            np.bincount(
                final_sids[:, level].astype(np.int64),
                minlength=size,
            )
        )
        for level, size in enumerate(model.config.codebook_sizes)
    ]
    result = {name: value / len(dataset) for name, value in totals.items()}
    result.update(
        {
            "rows": len(dataset),
            "nearest_assignment_codebooks": _usage_metrics(nearest_code_counts),
            "final_codebooks": _usage_metrics(final_code_counts),
            "nearest_assignment_collisions": nearest_metrics,
            "final_collisions": final_metrics,
            "collision_resolution": {
                "enabled": bool(
                    export_config is not None
                    and export_config.resolve_collisions
                ),
                "iterations": resolution_history,
            },
            "sid_shape": list(final_sids.shape),
            "sid_dtype": str(final_sids.dtype),
            "sid_file_bytes": output_path.stat().st_size,
        }
    )
    return result


def _constant_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(float(step + 1) / float(warmup_steps), 1.0)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_multiplier,
    )


def run_rqvae_job(
    config: RQVAEJobConfig,
    *,
    project_root: Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Train RQ-VAE, select the collision checkpoint, and export SIDs."""

    output_dir = config.output.dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    history_path = output_dir / "training_history.jsonl"
    metrics_path = output_dir / "metrics.json"
    last_checkpoint_path = output_dir / "checkpoint_last.pt"
    best_loss_checkpoint_path = output_dir / "checkpoint_best_loss.pt"
    best_collision_checkpoint_path = output_dir / "checkpoint_best_collision.pt"
    sids_path = output_dir / "sids.npy"

    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        if previous_manifest.get("status") == "completed":
            raise RuntimeError("该输出目录中的 RQ-VAE 任务已经完成")

    source = inspect_embedding_source(config.data, max_rows=max_rows)
    signature = _job_signature(config, source, max_rows)
    device = _resolve_device(config.training.device)
    _set_seed(config.training.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset = EmbeddingMemmapDataset(
        config.data.embeddings_path,
        rows=source.selected_rows,
    )
    train_sampler = BlockShuffleSampler(
        source.selected_rows,
        config.training.block_size,
        config.training.seed,
    )
    train_loader = _make_loader(
        dataset,
        sampler=train_sampler,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    evaluation_loader = _make_loader(
        dataset,
        sampler=IndexSampler(np.arange(source.selected_rows, dtype=np.int64)),
        batch_size=config.training.eval_batch_size,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = RQVAE(config.model, seed=config.training.seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    warmup_steps = config.training.warmup_epochs * len(train_loader)
    scheduler = _constant_warmup_scheduler(optimizer, warmup_steps)
    start_epoch = 0
    best_train_loss = math.inf
    best_collision_rate = math.inf
    best_loss_epoch: int | None = None
    best_collision_epoch: int | None = None
    history: list[dict[str, Any]] = []

    if last_checkpoint_path.is_file():
        if not config.output.resume:
            raise RuntimeError("输出目录存在 checkpoint，但 output.resume=false")
        checkpoint = _load_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            signature=signature,
            device=device,
        )
        start_epoch = int(checkpoint["epoch"])
        best_train_loss = float(checkpoint["best_train_loss"])
        best_collision_rate = float(checkpoint["best_collision_rate"])
        if not history_path.is_file():
            raise RuntimeError("存在 checkpoint，但缺少 training_history.jsonl")
        with history_path.open("r", encoding="utf-8") as handle:
            history = [json.loads(line) for line in handle if line.strip()]
        if not history or int(history[-1]["epoch"]) != start_epoch:
            raise RuntimeError("checkpoint 与 training_history.jsonl 的 epoch 不一致")
        best_loss_records = [
            record for record in history if record.get("is_best_loss")
        ]
        best_collision_records = [
            record for record in history if record.get("is_best_collision")
        ]
        if best_loss_records:
            best_loss_epoch = int(best_loss_records[-1]["epoch"])
        if best_collision_records:
            best_collision_epoch = int(best_collision_records[-1]["epoch"])
    elif (
        history_path.exists()
        or best_loss_checkpoint_path.exists()
        or best_collision_checkpoint_path.exists()
        or sids_path.exists()
    ):
        raise RuntimeError("输出目录存在不完整产物但缺少 checkpoint_last.pt")

    manifest: dict[str, Any] = {
        "job_name": config.job_name,
        "status": "running",
        "started_at": _utc_now(),
        "signature": signature,
        "source": {
            "embeddings_path": _display_path(
                config.data.embeddings_path, project_root
            ),
            "ids_path": _display_path(config.data.ids_path, project_root),
            "embedding_fingerprint": source.embedding_fingerprint,
            "total_rows": source.total_rows,
            "selected_rows": source.selected_rows,
            "embedding_dim": source.embedding_dim,
            "dtype": source.dtype,
        },
        "config": _serializable_config(config, project_root),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    _write_json_atomic(manifest_path, manifest)

    job_started = time.perf_counter()
    try:
        for epoch_index in range(start_epoch, config.training.epochs):
            epoch = epoch_index + 1
            train_sampler.set_epoch(epoch_index)
            epoch_started = time.perf_counter()
            train_metrics = _run_train_epoch(
                model,
                train_loader,
                device=device,
                quantization_weight=config.model.quantization_weight,
                use_bf16=config.training.use_bf16,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_clip_norm=config.training.gradient_clip_norm,
                description=f"RQ-VAE train {epoch}/{config.training.epochs}",
            )
            current_train_loss = float(train_metrics["loss"])
            is_best_loss = current_train_loss < best_train_loss
            if is_best_loss:
                best_train_loss = current_train_loss
                best_loss_epoch = epoch

            collision_metrics: dict[str, Any] | None = None
            should_evaluate_collision = (
                epoch % config.training.collision_eval_interval == 0
                or epoch == config.training.epochs
            )
            if should_evaluate_collision:
                collision_metrics = _evaluate_collision(
                    model,
                    evaluation_loader,
                    row_count=source.selected_rows,
                    device=device,
                    use_bf16=config.training.use_bf16,
                    description=f"RQ-VAE collision {epoch}/{config.training.epochs}",
                )
            is_best_collision = bool(
                collision_metrics is not None
                and collision_metrics["collision_rate"] < best_collision_rate
            )
            if is_best_collision:
                best_collision_rate = float(collision_metrics["collision_rate"])
                best_collision_epoch = epoch

            epoch_record = {
                "epoch": epoch,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "train": train_metrics,
                "collision": collision_metrics,
                "is_best_loss": is_best_loss,
                "is_best_collision": is_best_collision,
            }
            history.append(epoch_record)
            _append_jsonl(history_path, epoch_record)

            if is_best_loss:
                _save_checkpoint(
                    best_loss_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_train_loss=best_train_loss,
                    best_collision_rate=best_collision_rate,
                    signature=signature,
                )
            if is_best_collision:
                _save_checkpoint(
                    best_collision_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_train_loss=best_train_loss,
                    best_collision_rate=best_collision_rate,
                    signature=signature,
                )
            _save_checkpoint(
                last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_train_loss=best_train_loss,
                best_collision_rate=best_collision_rate,
                signature=signature,
            )
            print(json.dumps(epoch_record, ensure_ascii=False, sort_keys=True))

        if not best_loss_checkpoint_path.is_file():
            raise RuntimeError("训练结束但没有生成最低损失 checkpoint")
        if not best_collision_checkpoint_path.is_file():
            raise RuntimeError("训练结束但没有生成最低碰撞 checkpoint")

        selected_checkpoint = _load_checkpoint(
            best_collision_checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            signature=signature,
            device=device,
        )
        evaluation = export_sids_and_evaluate(
            model,
            dataset,
            output_path=sids_path,
            batch_size=config.training.eval_batch_size,
            num_workers=config.training.num_workers,
            device=device,
            use_bf16=config.training.use_bf16,
            quantization_weight=config.model.quantization_weight,
            export_config=config.export,
        )
        result: dict[str, Any] = {
            "best_loss_epoch": best_loss_epoch,
            "best_train_loss": best_train_loss,
            "best_collision_epoch": best_collision_epoch,
            "best_collision_rate": best_collision_rate,
            "selected_checkpoint": best_collision_checkpoint_path.name,
            "selected_checkpoint_epoch": int(selected_checkpoint["epoch"]),
            "history": history,
            "evaluation": evaluation,
            "total_seconds": time.perf_counter() - job_started,
        }
        if device.type == "cuda":
            allocated = int(torch.cuda.max_memory_allocated(device))
            reserved = int(torch.cuda.max_memory_reserved(device))
            result["cuda_peak_memory_allocated_gib"] = allocated / 1024**3
            result["cuda_peak_memory_reserved_gib"] = reserved / 1024**3
        _write_json_atomic(metrics_path, result)
        manifest.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "artifacts": {
                    "best_loss_checkpoint": best_loss_checkpoint_path.name,
                    "best_collision_checkpoint": best_collision_checkpoint_path.name,
                    "last_checkpoint": last_checkpoint_path.name,
                    "selected_checkpoint": best_collision_checkpoint_path.name,
                    "sids": sids_path.name,
                    "metrics": metrics_path.name,
                    "training_history": history_path.name,
                },
                "result": result,
            }
        )
        _write_json_atomic(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _write_json_atomic(manifest_path, manifest)
        raise

"""Method-owned fresh RQ-VAE initialization for TIGER-Joint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poi_gr.methods.tiger_joint.catalog import PoiEmbeddingStore
from poi_gr.methods.tiger_joint.data import TigerJointDataError
from poi_gr.pid.dedup import sha256_file
from poi_gr.sid.rqvae import RQVAE
from poi_gr.sid.training import create_fixed_indices, initialize_codebooks_kmeans


INITIALIZATION_SCHEMA_VERSION = "tiger-joint-rqvae-kmeans-initialization-v1"
INITIALIZATION_CHECKPOINT_NAME = "rqvae_initialization.pt"
INITIALIZATION_MANIFEST_NAME = "manifest.json"
INITIALIZATION_SAMPLE_NAME = "sample_indices.npy"


class TigerJointInitializationError(TigerJointDataError):
    """Raised when a fresh TIGER-Joint initialization is incompatible."""


@dataclass(frozen=True)
class FreshRqKMeansConfig:
    """Frozen first-version KMeans initialization settings."""

    seed: int = 42
    validation_ratio: float = 0.01
    kmeans_backend: str = "faiss_gpu"
    kmeans_sample_size: int = 500_000
    kmeans_iterations: int = 20
    kmeans_batch_size: int = 8_192
    kmeans_max_points_per_centroid: int = 2_048
    show_progress: bool = True

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise TigerJointInitializationError("KMeans seed 不能为负数")
        if not 0.0 < self.validation_ratio < 1.0:
            raise TigerJointInitializationError(
                "KMeans validation_ratio 必须位于 (0, 1)"
            )
        if self.kmeans_backend not in {"faiss_gpu", "faiss_cpu", "sklearn"}:
            raise TigerJointInitializationError("不支持的 KMeans backend")
        for name in (
            "kmeans_sample_size",
            "kmeans_iterations",
            "kmeans_batch_size",
            "kmeans_max_points_per_centroid",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TigerJointInitializationError(f"{name} 必须为正整数")


def fresh_rqvae_model_config() -> dict[str, Any]:
    """Return the exact first-version architecture and loss configuration."""

    return {
        "input_dim": 1024,
        "hidden_dim": 512,
        "latent_dim": 256,
        "codebook_sizes": [1024, 1024, 1024],
        "codebook_loss_weight": 1.0,
        "commitment_loss_weight": 0.25,
        "reconstruction_loss_weight": 1.0,
        "diversity_loss_weight": 0.0,
    }


def build_fresh_rqvae() -> RQVAE:
    """Construct a new RQ-VAE without reading any SID checkpoint."""

    config = fresh_rqvae_model_config()
    return RQVAE(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        codebook_sizes=tuple(int(value) for value in config["codebook_sizes"]),
        codebook_loss_weight=float(config["codebook_loss_weight"]),
        commitment_loss_weight=float(config["commitment_loss_weight"]),
        reconstruction_loss_weight=float(config["reconstruction_loss_weight"]),
        diversity_loss_weight=float(config["diversity_loss_weight"]),
    )


def create_method_owned_kmeans_indices(
    row_count: int,
    config: FreshRqKMeansConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recreate the frozen sample directly from the BGE catalog row count."""

    if row_count <= 1:
        raise TigerJointInitializationError("BGE 目录行数必须大于 1")
    _, _, sample_indices, metadata = create_fixed_indices(
        row_count=row_count,
        validation_ratio=config.validation_ratio,
        kmeans_sample_size=config.kmeans_sample_size,
        seed=config.seed,
    )
    return sample_indices, metadata


def initialize_fresh_rqvae_codebooks(
    model: RQVAE,
    embedding_store: PoiEmbeddingStore,
    sample_indices: np.ndarray,
    config: FreshRqKMeansConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run sequential residual KMeans on the fresh encoder's latent space."""

    if model.input_dim != embedding_store.shape[1]:
        raise TigerJointInitializationError(
            "fresh RQ-VAE 输入维度与 BGE Embedding 不一致"
        )
    if sample_indices.dtype != np.int64 or sample_indices.ndim != 1:
        raise TigerJointInitializationError("KMeans 样本索引必须是一维 int64")
    if len(sample_indices) != min(config.kmeans_sample_size, embedding_store.poi_count):
        raise TigerJointInitializationError("KMeans 样本行数与冻结配置不一致")
    return initialize_codebooks_kmeans(
        model,
        embedding_store.embeddings,
        sample_indices,
        config,
        device,
    )


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a tensor by its contiguous CPU bytes."""

    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def module_sha256(module: torch.nn.Module) -> str:
    """Hash a module state deterministically by sorted parameter name."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _load_manifest(initialization_dir: Path) -> dict[str, Any]:
    manifest_path = initialization_dir / INITIALIZATION_MANIFEST_NAME
    if not manifest_path.is_file():
        raise TigerJointInitializationError(
            f"RQ-VAE 初始化 manifest 不存在：{manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerJointInitializationError(
            "RQ-VAE 初始化 manifest 不是合法 JSON"
        ) from error
    if not isinstance(manifest, dict):
        raise TigerJointInitializationError("RQ-VAE 初始化 manifest 必须是 object")
    return manifest


def load_fresh_rqvae_initialization(
    initialization_dir: Path,
    *,
    embedding_store: PoiEmbeddingStore,
    preflight_state_sha256: str,
    device: torch.device,
) -> tuple[RQVAE, dict[str, Any]]:
    """Load and strictly bind a method-owned KMeans initialization artifact."""

    initialization_dir = initialization_dir.resolve()
    manifest = _load_manifest(initialization_dir)
    if (
        manifest.get("schema_version") != INITIALIZATION_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("formal_initialization_passed") is not True
    ):
        raise TigerJointInitializationError("RQ-VAE 初始化未通过正式门禁")
    if manifest.get("old_sid_artifacts_loaded") != []:
        raise TigerJointInitializationError("RQ-VAE 初始化曾加载旧 SID 产物")

    expected_source = {
        "embedding_manifest_signature": embedding_store.manifest_signature,
        "embedding_poi_ids_sha256": embedding_store.poi_ids_sha256,
        "embedding_shape": list(embedding_store.shape),
        "preflight_state_sha256": preflight_state_sha256,
    }
    source = manifest.get("source")
    if not isinstance(source, dict) or any(
        source.get(name) != value for name, value in expected_source.items()
    ):
        raise TigerJointInitializationError("RQ-VAE 初始化与 BGE 目录或正式预检不一致")
    if manifest.get("model") != fresh_rqvae_model_config():
        raise TigerJointInitializationError("RQ-VAE 初始化模型配置不一致")

    checkpoint_path = initialization_dir / INITIALIZATION_CHECKPOINT_NAME
    checkpoint = manifest.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("path") != INITIALIZATION_CHECKPOINT_NAME
        or not checkpoint_path.is_file()
        or checkpoint.get("sha256") != sha256_file(checkpoint_path)
    ):
        raise TigerJointInitializationError("RQ-VAE 初始化 checkpoint 校验失败")
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise TigerJointInitializationError(
            "RQ-VAE 初始化 checkpoint 无法读取"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != INITIALIZATION_SCHEMA_VERSION
        or payload.get("model") != fresh_rqvae_model_config()
        or not isinstance(payload.get("state_dict"), dict)
    ):
        raise TigerJointInitializationError("RQ-VAE 初始化 checkpoint 契约无效")

    model = build_fresh_rqvae()
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as error:
        raise TigerJointInitializationError(
            "RQ-VAE 初始化 checkpoint 参数不匹配"
        ) from error
    if any(
        not bool(torch.isfinite(value).all().item())
        for value in model.state_dict().values()
    ):
        raise TigerJointInitializationError("RQ-VAE 初始化包含 NaN 或 Inf")
    expected_model_hash = checkpoint.get("model_sha256")
    if expected_model_hash != module_sha256(model):
        raise TigerJointInitializationError("RQ-VAE 初始化参数 SHA256 不一致")
    return model.to(device), manifest

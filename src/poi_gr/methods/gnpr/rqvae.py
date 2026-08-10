"""GNPR-SID RQ-VAE matching the paper's discrete-feature formulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ...sid.rqvae import hard_utilization_loss

from .sid_input import (
    GnprFeatureDimensions,
    GnprSidInput,
    active_feature_indices,
)


def _positive_dimensions(values: Sequence[int], name: str) -> tuple[int, ...]:
    resolved = tuple(int(value) for value in values)
    if not resolved or any(value <= 0 for value in resolved):
        raise ValueError(f"{name} must contain positive integers")
    return resolved


def _build_paper_mlp(
    dimensions: Sequence[int],
    *,
    dropout: float,
) -> nn.Sequential:
    modules: list[nn.Module] = []
    last_index = len(dimensions) - 2
    for index, (input_dim, output_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        modules.append(nn.Dropout(dropout))
        linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        modules.append(linear)
        if index != last_index:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def dense_gnpr_batch(
    rows: Sequence[GnprSidInput],
    *,
    dimensions: GnprFeatureDimensions,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Materialize only one bounded multi-hot batch for the paper MLP."""

    if not rows:
        raise ValueError("GNPR batch cannot be empty")
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row_index, row in enumerate(rows):
        active = active_feature_indices(row, dimensions=dimensions)
        row_indices.extend([row_index] * len(active))
        column_indices.extend(active)
    result = torch.zeros(
        (len(rows), dimensions.total_dim),
        dtype=torch.float32,
        device=device,
    )
    result[
        torch.tensor(row_indices, dtype=torch.long, device=result.device),
        torch.tensor(column_indices, dtype=torch.long, device=result.device),
    ] = 1.0
    return result


@dataclass(frozen=True)
class GnprQuantizerOutput:
    quantized_st: torch.Tensor
    quantized: torch.Tensor
    codes: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    diversity_loss: torch.Tensor
    residual_norms: torch.Tensor


@dataclass(frozen=True)
class GnprRQVAEOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    quantized: torch.Tensor
    codes: torch.Tensor
    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    reconstruction_cosine: torch.Tensor
    quantization_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    diversity_loss: torch.Tensor
    residual_norms: torch.Tensor


def _squared_distances(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return (
        values.square().sum(dim=1, keepdim=True)
        + codebook.square().sum(dim=1).unsqueeze(0)
        - 2.0 * values @ codebook.t()
    )


class GnprResidualQuantizer(nn.Module):
    """Three-level residual quantizer with GNPR utilization regularization."""

    def __init__(
        self,
        latent_dim: int,
        codebook_sizes: Sequence[int],
        *,
        commitment_beta: float = 0.25,
        diversity_scale: float = 0.05,
        diversity_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        sizes = _positive_dimensions(codebook_sizes, "codebook_sizes")
        if len(sizes) != 3:
            raise ValueError("GNPR-SID requires exactly three codebooks")
        if commitment_beta < 0 or diversity_scale < 0 or diversity_temperature <= 0:
            raise ValueError("loss scales must be non-negative")
        self.latent_dim = latent_dim
        self.codebook_sizes = sizes
        self.commitment_beta = float(commitment_beta)
        self.diversity_scale = float(diversity_scale)
        self.diversity_temperature = float(diversity_temperature)
        self.codebooks = nn.ModuleList(
            nn.Embedding(size, latent_dim) for size in self.codebook_sizes
        )
        for codebook, size in zip(self.codebooks, self.codebook_sizes, strict=True):
            nn.init.uniform_(codebook.weight, -1.0 / size, 1.0 / size)

    def set_codebook(self, level_index: int, centroids: torch.Tensor) -> None:
        """Set one level's K-Means centroids before optimization."""

        target = self.codebooks[level_index].weight
        if centroids.shape != target.shape:
            raise ValueError(
                f"centroid shape {tuple(centroids.shape)} != {tuple(target.shape)}"
            )
        with torch.no_grad():
            target.copy_(centroids.to(device=target.device, dtype=target.dtype))

    def forward(self, latent: torch.Tensor) -> GnprQuantizerOutput:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent shape must be [batch, {self.latent_dim}]"
            )
        residual = latent
        quantized_parts: list[torch.Tensor] = []
        codes: list[torch.Tensor] = []
        codebook_losses: list[torch.Tensor] = []
        commitment_losses: list[torch.Tensor] = []
        diversity_losses: list[torch.Tensor] = []
        residual_norms: list[torch.Tensor] = []
        for codebook in self.codebooks:
            distances = _squared_distances(residual, codebook.weight)
            indices = distances.detach().argmin(dim=1)
            quantized = codebook(indices)
            codebook_losses.append(F.mse_loss(quantized, residual.detach()))
            commitment_losses.append(F.mse_loss(residual, quantized.detach()))
            diversity_distances = _squared_distances(
                residual,
                codebook.weight.detach(),
            )
            diversity_losses.append(
                hard_utilization_loss(
                    diversity_distances,
                    temperature=self.diversity_temperature,
                )
            )
            quantized_parts.append(quantized)
            codes.append(indices)
            residual = residual - quantized.detach()
            residual_norms.append(torch.linalg.vector_norm(residual, dim=1))

        quantized_sum = torch.stack(quantized_parts).sum(dim=0)
        return GnprQuantizerOutput(
            quantized_st=latent + (quantized_sum - latent).detach(),
            quantized=quantized_sum,
            codes=torch.stack(codes, dim=1),
            codebook_loss=torch.stack(codebook_losses).mean(),
            commitment_loss=torch.stack(commitment_losses).mean(),
            diversity_loss=self.diversity_scale
            * torch.stack(diversity_losses).mean(),
            residual_norms=torch.stack(residual_norms, dim=1),
        )


class GnprRQVAE(nn.Module):
    """Paper-style GNPR RQ-VAE for bounded sparse Beijing POI vectors."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = (512, 256, 128),
        latent_dim: int = 64,
        codebook_sizes: Sequence[int] = (256, 256, 256),
        dropout: float = 0.1,
        quantization_loss_weight: float = 1.0,
        diversity_loss_weight: float = 0.25,
        commitment_beta: float = 0.25,
        diversity_scale: float = 0.05,
        diversity_temperature: float = 0.5,
        reconstruction_loss: str = "mse",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if quantization_loss_weight < 0 or diversity_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if reconstruction_loss not in {"mse", "balanced_mse"}:
            raise ValueError("reconstruction_loss must be mse or balanced_mse")
        hidden = _positive_dimensions(hidden_dims, "hidden_dims")
        self.input_dim = input_dim
        self.hidden_dims = hidden
        self.latent_dim = latent_dim
        self.codebook_sizes = tuple(int(value) for value in codebook_sizes)
        self.quantization_loss_weight = float(quantization_loss_weight)
        self.diversity_loss_weight = float(diversity_loss_weight)
        self.reconstruction_loss_name = reconstruction_loss
        self.encoder = _build_paper_mlp(
            (input_dim, *hidden, latent_dim),
            dropout=dropout,
        )
        # The released GNPR code uses the same hidden order in the decoder.
        self.decoder = _build_paper_mlp(
            (latent_dim, *hidden, input_dim),
            dropout=dropout,
        )
        self.quantizer = GnprResidualQuantizer(
            latent_dim,
            self.codebook_sizes,
            commitment_beta=commitment_beta,
            diversity_scale=diversity_scale,
            diversity_temperature=diversity_temperature,
        )
        self.commitment_beta = float(commitment_beta)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs.float())

    def encode_codes(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quantizer(self.encode(inputs)).codes

    def forward(self, inputs: torch.Tensor) -> GnprRQVAEOutput:
        inputs_float = inputs.float()
        latent = self.encoder(inputs_float)
        quantizer = self.quantizer(latent)
        reconstruction = torch.sigmoid(self.decoder(quantizer.quantized_st))
        if self.reconstruction_loss_name == "mse":
            reconstruction_loss = F.mse_loss(reconstruction, inputs_float)
        else:
            squared_error = (reconstruction - inputs_float).square()
            positive = inputs_float > 0
            negative = ~positive
            positive_loss = squared_error[positive].mean()
            negative_loss = squared_error[negative].mean()
            reconstruction_loss = 0.5 * (positive_loss + negative_loss)
        quantization_loss = (
            quantizer.codebook_loss
            + self.commitment_beta * quantizer.commitment_loss
        )
        total_loss = (
            reconstruction_loss
            + self.quantization_loss_weight * quantization_loss
            + self.diversity_loss_weight * quantizer.diversity_loss
        )
        reconstruction_cosine = F.cosine_similarity(
            reconstruction,
            inputs_float,
            dim=1,
            eps=1e-8,
        ).mean()
        return GnprRQVAEOutput(
            reconstruction=reconstruction,
            latent=latent,
            quantized=quantizer.quantized,
            codes=quantizer.codes,
            total_loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            reconstruction_cosine=reconstruction_cosine,
            quantization_loss=quantization_loss,
            codebook_loss=quantizer.codebook_loss,
            commitment_loss=quantizer.commitment_loss,
            diversity_loss=quantizer.diversity_loss,
            residual_norms=quantizer.residual_norms,
        )

"""Vanilla residual-quantized variational autoencoder components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QuantizerOutput:
    quantized_st: torch.Tensor
    quantized: torch.Tensor
    codes: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    residual_norms: torch.Tensor


@dataclass(frozen=True)
class RQVAEOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    quantized: torch.Tensor
    codes: torch.Tensor
    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    reconstruction_cosine: torch.Tensor
    residual_norms: torch.Tensor


def _nearest_code_indices(residual: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    distances = (
        residual.square().sum(dim=1, keepdim=True)
        + codebook.square().sum(dim=1).unsqueeze(0)
        - 2.0 * residual @ codebook.t()
    )
    return distances.argmin(dim=1)


class ResidualQuantizer(nn.Module):
    """Apply a sequence of codebooks to successive latent residuals."""

    def __init__(self, latent_dim: int, codebook_sizes: Sequence[int]) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not codebook_sizes or any(size <= 0 for size in codebook_sizes):
            raise ValueError("codebook_sizes must contain positive integers")
        self.latent_dim = latent_dim
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        self.codebooks = nn.ModuleList(
            nn.Embedding(size, latent_dim) for size in self.codebook_sizes
        )
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0 / latent_dim, 1.0 / latent_dim)

    def set_codebook(self, level_index: int, centroids: torch.Tensor) -> None:
        """Replace one codebook with externally initialized centroids."""

        expected = self.codebooks[level_index].weight.shape
        if centroids.shape != expected:
            raise ValueError(
                f"centroid shape {tuple(centroids.shape)} != {tuple(expected)}"
            )
        with torch.no_grad():
            self.codebooks[level_index].weight.copy_(
                centroids.to(
                    device=self.codebooks[level_index].weight.device,
                    dtype=self.codebooks[level_index].weight.dtype,
                )
            )

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent shape must be [batch, {self.latent_dim}], got {tuple(latent.shape)}"
            )
        if latent.dtype not in {torch.float32, torch.float64}:
            raise ValueError("residual distance computation requires float32 or float64")

        residual = latent
        quantized_parts: list[torch.Tensor] = []
        code_indices: list[torch.Tensor] = []
        residual_norms: list[torch.Tensor] = []
        codebook_loss = latent.new_zeros(())
        commitment_loss = latent.new_zeros(())
        for codebook in self.codebooks:
            indices = _nearest_code_indices(residual.detach(), codebook.weight)
            quantized = codebook(indices)
            codebook_loss = codebook_loss + F.mse_loss(
                quantized, residual.detach()
            )
            commitment_loss = commitment_loss + F.mse_loss(
                residual, quantized.detach()
            )
            quantized_parts.append(quantized)
            code_indices.append(indices)
            residual = residual - quantized.detach()
            residual_norms.append(torch.linalg.vector_norm(residual, dim=1))

        quantized_sum = torch.stack(quantized_parts, dim=0).sum(dim=0)
        quantized_st = latent + (quantized_sum - latent).detach()
        return QuantizerOutput(
            quantized_st=quantized_st,
            quantized=quantized_sum,
            codes=torch.stack(code_indices, dim=1),
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            residual_norms=torch.stack(residual_norms, dim=1),
        )


class RQVAE(nn.Module):
    """Encode embeddings, residual-quantize the latent, and reconstruct inputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        codebook_sizes: Sequence[int],
        *,
        codebook_loss_weight: float = 1.0,
        commitment_loss_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, latent_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if codebook_loss_weight < 0 or commitment_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        self.codebook_loss_weight = codebook_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Construct both MLPs before the size-dependent codebooks so the three
        # capacity experiments receive identical encoder/decoder initialization.
        self.quantizer = ResidualQuantizer(latent_dim, self.codebook_sizes)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs.float())

    def encode_codes(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quantizer(self.encode(inputs)).codes

    def forward(self, inputs: torch.Tensor) -> RQVAEOutput:
        inputs_float = inputs.float()
        latent = self.encoder(inputs_float)
        quantizer_output = self.quantizer(latent)
        reconstruction = self.decoder(quantizer_output.quantized_st)
        reconstruction_loss = F.mse_loss(reconstruction, inputs_float)
        total_loss = (
            reconstruction_loss
            + self.codebook_loss_weight * quantizer_output.codebook_loss
            + self.commitment_loss_weight * quantizer_output.commitment_loss
        )
        reconstruction_cosine = F.cosine_similarity(
            reconstruction, inputs_float, dim=1, eps=1e-8
        ).mean()
        return RQVAEOutput(
            reconstruction=reconstruction,
            latent=latent,
            quantized=quantizer_output.quantized,
            codes=quantizer_output.codes,
            total_loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            codebook_loss=quantizer_output.codebook_loss,
            commitment_loss=quantizer_output.commitment_loss,
            reconstruction_cosine=reconstruction_cosine,
            residual_norms=quantizer_output.residual_norms,
        )

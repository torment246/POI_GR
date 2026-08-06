"""Vanilla residual-quantized variational autoencoder components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _normalize_hidden_dims(hidden_dim: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden_dim, bool):
        raise ValueError("hidden dimensions must contain positive integers")
    if isinstance(hidden_dim, int):
        hidden_dims = (hidden_dim,)
    elif isinstance(hidden_dim, Sequence) and not isinstance(hidden_dim, (str, bytes)):
        hidden_dims = tuple(hidden_dim)
    else:
        raise ValueError("hidden dimensions must be an integer or a sequence")
    if (
        not hidden_dims
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in hidden_dims
        )
    ):
        raise ValueError("hidden dimensions must contain positive integers")
    return hidden_dims


def _build_mlp(dimensions: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    final_layer_index = len(dimensions) - 2
    for layer_index, (input_dim, output_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        layers.append(nn.Linear(input_dim, output_dim))
        if layer_index != final_layer_index:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    squared_error = (prediction - target).square()
    if reduction == "element_mean":
        return squared_error.mean()
    if reduction == "vector_sum":
        return squared_error.sum(dim=1).mean()
    raise ValueError(
        "squared_error_reduction must be 'element_mean' or 'vector_sum'"
    )


@dataclass(frozen=True)
class QuantizerOutput:
    quantized_st: torch.Tensor
    quantized: torch.Tensor
    codes: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    diversity_loss: torch.Tensor
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
    diversity_loss: torch.Tensor
    reconstruction_cosine: torch.Tensor
    residual_norms: torch.Tensor


def _squared_distances(
    values: torch.Tensor, codebook: torch.Tensor
) -> torch.Tensor:
    return (
        values.square().sum(dim=1, keepdim=True)
        + codebook.square().sum(dim=1).unsqueeze(0)
        - 2.0 * values @ codebook.t()
    )


def hard_utilization_loss(
    distances: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Match hard code counts in the forward pass with bounded soft gradients."""

    if distances.ndim != 2 or not distances.shape[1]:
        raise ValueError("distances must be a non-empty [batch, codebook] tensor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    centered = distances - distances.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-3)
    soft = F.softmax(-(centered / scale) / temperature, dim=1)
    indices = distances.detach().argmin(dim=1)
    hard = F.one_hot(indices, num_classes=distances.shape[1]).to(soft.dtype)
    assignments = hard + soft - soft.detach()
    counts = assignments.sum(dim=0)
    expected = counts.mean()
    return ((counts - expected).square().mean()) / (expected.square() + 1e-5)


class ResidualQuantizer(nn.Module):
    """Apply a sequence of codebooks to successive latent residuals."""

    def __init__(
        self,
        latent_dim: int,
        codebook_sizes: Sequence[int],
        *,
        squared_error_reduction: str = "element_mean",
        diversity_scale: float = 0.0,
        diversity_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not codebook_sizes or any(size <= 0 for size in codebook_sizes):
            raise ValueError("codebook_sizes must contain positive integers")
        if squared_error_reduction not in {"element_mean", "vector_sum"}:
            raise ValueError(
                "squared_error_reduction must be 'element_mean' or 'vector_sum'"
            )
        if diversity_scale < 0 or diversity_temperature <= 0:
            raise ValueError("diversity settings must be non-negative")
        self.latent_dim = latent_dim
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        self.squared_error_reduction = squared_error_reduction
        self.diversity_scale = float(diversity_scale)
        self.diversity_temperature = float(diversity_temperature)
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

    def forward(
        self, latent: torch.Tensor, *, compute_diversity: bool = True
    ) -> QuantizerOutput:
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
        diversity_losses: list[torch.Tensor] = []
        for codebook in self.codebooks:
            assignment_distances = _squared_distances(
                residual.detach(), codebook.weight.detach()
            )
            indices = assignment_distances.argmin(dim=1)
            quantized = codebook(indices)
            codebook_loss = codebook_loss + _squared_error(
                quantized,
                residual.detach(),
                self.squared_error_reduction,
            )
            commitment_loss = commitment_loss + _squared_error(
                residual,
                quantized.detach(),
                self.squared_error_reduction,
            )
            if compute_diversity and self.diversity_scale > 0:
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
            code_indices.append(indices)
            residual = residual - quantized.detach()
            residual_norms.append(torch.linalg.vector_norm(residual, dim=1))

        quantized_sum = torch.stack(quantized_parts, dim=0).sum(dim=0)
        quantized_st = latent + (quantized_sum - latent).detach()
        diversity_loss = (
            self.diversity_scale * torch.stack(diversity_losses).mean()
            if diversity_losses
            else latent.new_zeros(())
        )
        return QuantizerOutput(
            quantized_st=quantized_st,
            quantized=quantized_sum,
            codes=torch.stack(code_indices, dim=1),
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            diversity_loss=diversity_loss,
            residual_norms=torch.stack(residual_norms, dim=1),
        )


class RQVAE(nn.Module):
    """Encode embeddings, residual-quantize the latent, and reconstruct inputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | Sequence[int],
        latent_dim: int,
        codebook_sizes: Sequence[int],
        *,
        codebook_loss_weight: float = 1.0,
        commitment_loss_weight: float = 0.25,
        reconstruction_loss_weight: float = 1.0,
        diversity_loss_weight: float = 0.0,
        diversity_scale: float = 0.0,
        diversity_temperature: float = 0.5,
        reconstruction_normalization: str = "none",
        squared_error_reduction: str = "element_mean",
    ) -> None:
        super().__init__()
        hidden_dims = _normalize_hidden_dims(hidden_dim)
        if min(input_dim, latent_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if (
            codebook_loss_weight < 0
            or commitment_loss_weight < 0
            or diversity_loss_weight < 0
        ):
            raise ValueError("loss weights must be non-negative")
        if reconstruction_loss_weight <= 0:
            raise ValueError("reconstruction_loss_weight must be positive")
        if reconstruction_normalization not in {"none", "l2"}:
            raise ValueError(
                "reconstruction_normalization must be 'none' or 'l2'"
            )
        if squared_error_reduction not in {"element_mean", "vector_sum"}:
            raise ValueError(
                "squared_error_reduction must be 'element_mean' or 'vector_sum'"
            )
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.hidden_dim = hidden_dims[0] if len(hidden_dims) == 1 else None
        self.latent_dim = latent_dim
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        self.codebook_loss_weight = codebook_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.diversity_loss_weight = float(diversity_loss_weight)
        self.reconstruction_normalization = reconstruction_normalization
        self.squared_error_reduction = squared_error_reduction
        self.encoder = _build_mlp(
            (input_dim, *hidden_dims, latent_dim)
        )
        self.decoder = _build_mlp(
            (latent_dim, *reversed(hidden_dims), input_dim)
        )
        # Construct both MLPs before the size-dependent codebooks so the three
        # capacity experiments receive identical encoder/decoder initialization.
        self.quantizer = ResidualQuantizer(
            latent_dim,
            self.codebook_sizes,
            squared_error_reduction=squared_error_reduction,
            diversity_scale=diversity_scale,
            diversity_temperature=diversity_temperature,
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs.float())

    def encode_codes(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quantizer(
            self.encode(inputs), compute_diversity=False
        ).codes

    def forward(
        self, inputs: torch.Tensor, *, diversity_active: bool = True
    ) -> RQVAEOutput:
        inputs_float = inputs.float()
        latent = self.encoder(inputs_float)
        quantizer_output = self.quantizer(
            latent,
            compute_diversity=(
                diversity_active and self.diversity_loss_weight > 0
            ),
        )
        reconstruction = self.decoder(quantizer_output.quantized_st)
        if self.reconstruction_normalization == "l2":
            reconstruction = F.normalize(reconstruction, p=2, dim=1, eps=1e-8)
        reconstruction_loss = _squared_error(
            reconstruction,
            inputs_float,
            self.squared_error_reduction,
        )
        total_loss = (
            self.reconstruction_loss_weight * reconstruction_loss
            + self.codebook_loss_weight * quantizer_output.codebook_loss
            + self.commitment_loss_weight * quantizer_output.commitment_loss
            + self.diversity_loss_weight * quantizer_output.diversity_loss
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
            diversity_loss=quantizer_output.diversity_loss,
            reconstruction_cosine=reconstruction_cosine,
            residual_norms=quantizer_output.residual_norms,
        )

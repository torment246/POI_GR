"""PyTorch Residual-Quantized VAE for fixed POI feature vectors."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    dropout: float = 0.05,
    final_activation: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [input_dim, *hidden_dims, output_dim]
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        is_last = idx == len(dims) - 2
        if not is_last or final_activation:
            layers.append(nn.LayerNorm(dims[idx + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with straight-through reconstruction path."""

    def __init__(self, num_codebooks: int, codebook_size: int, code_dim: int) -> None:
        super().__init__()
        if num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be positive, got {num_codebooks}")
        if codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {codebook_size}")
        if code_dim <= 0:
            raise ValueError(f"code_dim must be positive, got {code_dim}")
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.codebooks = nn.Parameter(torch.empty(num_codebooks, codebook_size, code_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.codebooks, mean=0.0, std=0.02)

    @staticmethod
    def nearest_code(residual: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual_norm = residual.pow(2).sum(dim=1, keepdim=True)
        code_norm = codebook.pow(2).sum(dim=1).unsqueeze(0)
        distances = residual_norm + code_norm - 2.0 * residual @ codebook.t()
        indices = torch.argmin(distances, dim=1)
        codes = F.embedding(indices, codebook)
        return indices, codes

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = h
        quantized_sum = torch.zeros_like(h)
        indices: list[torch.Tensor] = []

        for level in range(self.num_codebooks):
            codebook = self.codebooks[level]
            nearest_idx, nearest_code = self.nearest_code(residual, codebook)
            quantized_sum = quantized_sum + nearest_code
            # The next codebook receives the remaining encoder residual. Detaching
            # the selected code keeps later residual assignments from backpropagating
            # through earlier codebook choices.
            residual = residual - nearest_code.detach()
            indices.append(nearest_idx)

        stacked_indices = torch.stack(indices, dim=1)
        quantized_st = h + (quantized_sum - h).detach()
        return {
            "quantized_st": quantized_st,
            "quantized": quantized_sum,
            "indices": stacked_indices,
            "residual": residual,
        }


class RQVAE(nn.Module):
    """Residual-Quantized VAE for deterministic semantic or geo-fused inputs."""

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list[int],
        latent_dim: int,
        num_codebooks: int,
        codebook_size: int,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoder_hidden_dims = [int(x) for x in encoder_hidden_dims]
        self.latent_dim = int(latent_dim)
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.commitment_beta = float(commitment_beta)
        self.codebook_loss_weight = float(codebook_loss_weight)
        self.dropout = float(dropout)

        self.encoder = build_mlp(self.input_dim, self.encoder_hidden_dims, self.latent_dim, dropout=dropout)
        self.quantizer = ResidualVectorQuantizer(self.num_codebooks, self.codebook_size, self.latent_dim)
        decoder_hidden_dims = list(reversed(self.encoder_hidden_dims))
        self.decoder = build_mlp(self.latent_dim, decoder_hidden_dims, self.input_dim, dropout=dropout)

    def model_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "encoder_hidden_dims": self.encoder_hidden_dims,
            "latent_dim": self.latent_dim,
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "commitment_beta": self.commitment_beta,
            "codebook_loss_weight": self.codebook_loss_weight,
            "dropout": self.dropout,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        q = self.quantizer(h)
        x_hat = self.decoder(q["quantized_st"])
        recon_loss = F.mse_loss(x_hat, x)

        # These stop-gradient directions intentionally follow the project spec:
        # commitment_loss updates codebook vectors toward the detached encoder
        # output, while codebook_loss updates the encoder toward the detached
        # quantized sum. Reconstruction gradients reach the encoder through the
        # straight-through quantized path.
        commitment_loss = F.mse_loss(h.detach(), q["quantized"])
        codebook_loss = F.mse_loss(h, q["quantized"].detach())
        total_loss = recon_loss + self.commitment_beta * commitment_loss + self.codebook_loss_weight * codebook_loss
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "indices": q["indices"],
            "x_hat": x_hat,
            "h": h,
            "quantized": q["quantized"],
        }

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        q = self.quantizer(h)
        return q["indices"], q["quantized"]

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized)


class CAURQVAE(RQVAE):
    """RQ-VAE with an auxiliary coarse-category classifier for CAU experiments."""

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list[int],
        latent_dim: int,
        num_codebooks: int,
        codebook_size: int,
        num_labels: int,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        dropout: float = 0.05,
        classifier_dropout: float = 0.10,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            commitment_beta=commitment_beta,
            codebook_loss_weight=codebook_loss_weight,
            dropout=dropout,
        )
        if num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {num_labels}")
        self.num_labels = int(num_labels)
        self.classifier_dropout = float(classifier_dropout)
        self.tag_classifier = nn.Sequential(
            nn.Dropout(self.classifier_dropout),
            nn.Linear(self.latent_dim, self.num_labels),
        )

    def model_config(self) -> dict[str, Any]:
        config = super().model_config()
        config.update(
            {
                "architecture": "cau_rqvae",
                "num_labels": self.num_labels,
                "classifier_dropout": self.classifier_dropout,
            }
        )
        return config

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(x)
        outputs["tag_logits"] = self.tag_classifier(outputs["h"])
        return outputs

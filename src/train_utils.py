"""Training helpers for RQ-VAE experiments."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IndexedNumpyDataset(Dataset[torch.Tensor]):
    def __init__(self, array: np.ndarray, indices: np.ndarray) -> None:
        self.array = array
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> torch.Tensor:
        idx = int(self.indices[item])
        return torch.from_numpy(np.array(self.array[idx], dtype=np.float32, copy=True))


def make_loader(
    array: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        IndexedNumpyDataset(array, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None,
    )


def empty_epoch_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_total_loss": 0.0,
        f"{prefix}_recon_loss": 0.0,
        f"{prefix}_commitment_loss": 0.0,
        f"{prefix}_codebook_loss": 0.0,
    }


def aggregate_loss_sums(sums: dict[str, float], outputs: dict[str, torch.Tensor], batch_size: int) -> None:
    for key in ["total_loss", "recon_loss", "commitment_loss", "codebook_loss"]:
        sums[key] += float(outputs[key].detach().item()) * batch_size


def loss_sums_to_metrics(prefix: str, sums: dict[str, float], n_rows: int) -> dict[str, float]:
    denom = max(1, int(n_rows))
    return {
        f"{prefix}_total_loss": sums["total_loss"] / denom,
        f"{prefix}_recon_loss": sums["recon_loss"] / denom,
        f"{prefix}_commitment_loss": sums["commitment_loss"] / denom,
        f"{prefix}_codebook_loss": sums["codebook_loss"] / denom,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None,
    epoch: int,
    log_every_steps: int,
) -> dict[str, float]:
    model.train()
    sums = {"total_loss": 0.0, "recon_loss": 0.0, "commitment_loss": 0.0, "codebook_loss": 0.0}
    seen = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        x = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        outputs["total_loss"].backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()

        batch_size = int(x.shape[0])
        seen += batch_size
        aggregate_loss_sums(sums, outputs, batch_size)
        if log_every_steps > 0 and step % log_every_steps == 0:
            progress.set_postfix(loss=f"{sums['total_loss'] / max(1, seen):.5f}")
    return loss_sums_to_metrics("train", sums, seen)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_codebooks: int,
    codebook_size: int,
) -> dict[str, float]:
    model.eval()
    sums = {"total_loss": 0.0, "recon_loss": 0.0, "commitment_loss": 0.0, "codebook_loss": 0.0}
    cosine_sum = 0.0
    seen = 0
    used: list[set[int]] = [set() for _ in range(num_codebooks)]
    for batch in tqdm(loader, desc="validate", leave=False):
        x = batch.to(device, non_blocking=True)
        outputs = model(x)
        batch_size = int(x.shape[0])
        seen += batch_size
        aggregate_loss_sums(sums, outputs, batch_size)
        cosine = F.cosine_similarity(outputs["x_hat"], x, dim=1, eps=1e-8)
        cosine_sum += float(cosine.sum().item())
        indices = outputs["indices"].detach().cpu().numpy()
        for level in range(num_codebooks):
            used[level].update(int(v) for v in np.unique(indices[:, level]))

    metrics = loss_sums_to_metrics("val", sums, seen)
    metrics["val_recon_cosine"] = cosine_sum / max(1, seen)
    for level in range(num_codebooks):
        count = len(used[level])
        metrics[f"codebook{level}_used"] = float(count)
        metrics[f"codebook{level}_usage_rate"] = float(count / codebook_size)
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mode: str,
    input_dim: int,
    best_epoch: int,
    best_val_loss: float,
    train_log_path: Path,
    preprocess_path: Path,
    rqvae_input_path: Path,
    random_seed: int,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.model_config() if hasattr(model, "model_config") else {},
        "mode": mode,
        "input_dim": int(input_dim),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "train_log_path": str(train_log_path),
        "preprocess_path": str(preprocess_path),
        "rqvae_input_path": str(rqvae_input_path),
        "random_seed": int(random_seed),
        "epoch": int(epoch),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def write_train_log(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def append_debug_suffix(path: Path, debug_max_rows: int | None) -> Path:
    if not debug_max_rows:
        return path
    return path.with_name(path.stem + "_debug" + path.suffix)


def cpu_thread_summary() -> str:
    return f"pid={os.getpid()} torch_threads={torch.get_num_threads()}"

"""Train and persist a query adapter from a prepared tensor bundle."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.config import QGPRQKConfig
from qg_prqk.adapters.model import (
    ResidualQueryAdapter,
    build_negative_valid_mask,
    fit_query_adapter,
    stable_train_dev_split,
)


SCHEMA_VERSION = "qg-prqk-query-adapter-v1"
TRAINING_DATA_SCHEMA_VERSION = "qg-prqk-adapter-training-data-v1"


class AdapterTrainingError(RuntimeError):
    """Raised when prepared Adapter data or output artifacts are invalid."""


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_training_bundle(
    path: Path, *, limit: int | None = None
) -> dict[str, np.ndarray]:
    """Load the documented numeric NPZ contract without pickle objects."""

    if limit is not None and limit <= 1:
        raise AdapterTrainingError("Adapter limit 必须大于 1")
    required = {
        "schema_version",
        "raw_queries",
        "positive_embeddings",
        "negative_embeddings",
        "candidate_poi_rows",
        "target_poi_rows",
        "false_negative_offsets",
        "false_negative_rows",
        "query_weights",
        "query_ids",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise AdapterTrainingError(
                    f"Adapter training NPZ 缺少数组：{missing}"
                )
            schema = np.asarray(archive["schema_version"])
            if schema.shape != () or str(schema.item()) != TRAINING_DATA_SCHEMA_VERSION:
                raise AdapterTrainingError(
                    "Adapter training NPZ schema_version 不匹配"
                )
            arrays = {
                name: np.asarray(archive[name])
                for name in required - {"schema_version"}
            }
    except AdapterTrainingError:
        raise
    except (OSError, ValueError) as error:
        raise AdapterTrainingError(f"Adapter training NPZ 无法读取：{path}") from error
    rows = len(arrays["query_ids"])
    selected = rows if limit is None else min(rows, limit)
    offsets = arrays["false_negative_offsets"]
    if offsets.shape != (rows + 1,) or offsets.dtype.kind not in "iu":
        raise AdapterTrainingError("false_negative_offsets 必须是 int[rows+1]")
    false_stop = int(offsets[selected])
    result = {
        name: value[:selected]
        for name, value in arrays.items()
        if name not in {"false_negative_offsets", "false_negative_rows"}
    }
    result["false_negative_offsets"] = offsets[: selected + 1]
    result["false_negative_rows"] = arrays["false_negative_rows"][:false_stop]
    raw = result["raw_queries"]
    positives = result["positive_embeddings"]
    negatives = result["negative_embeddings"]
    candidates = result["candidate_poi_rows"]
    if raw.ndim != 2 or positives.shape != raw.shape:
        raise AdapterTrainingError("raw/positive embeddings shape 非法")
    if negatives.ndim != 3 or negatives.shape[0] != selected or negatives.shape[2] != raw.shape[1]:
        raise AdapterTrainingError("negative_embeddings shape 非法")
    if candidates.shape != negatives.shape[:2]:
        raise AdapterTrainingError("candidate_poi_rows shape 非法")
    for name in ("target_poi_rows", "query_weights", "query_ids"):
        if result[name].shape != (selected,):
            raise AdapterTrainingError(f"{name} shape 必须是 [rows]")
    if not np.isfinite(raw).all() or not np.isfinite(positives).all() or not np.isfinite(negatives).all():
        raise AdapterTrainingError("Adapter training embeddings 包含 NaN/Inf")
    if not np.isfinite(result["query_weights"]).all() or np.any(result["query_weights"] < 0):
        raise AdapterTrainingError("query_weights 必须有限且非负")
    return result


def _reasonable_positive_rows(bundle: dict[str, np.ndarray]) -> list[np.ndarray]:
    offsets = bundle["false_negative_offsets"]
    values = bundle["false_negative_rows"]
    return [values[int(offsets[index]) : int(offsets[index + 1])] for index in range(len(offsets) - 1)]


def train_adapter_bundle(
    config: QGPRQKConfig,
    *,
    training_data: Path,
    output_dir: Path,
    limit: int | None = None,
    device: str = "cpu",
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Train one sample Adapter artifact; real scale is selected by the caller."""

    training_data = training_data.resolve()
    output_dir = output_dir.resolve()
    if not _is_within(training_data, config.paths.output_root.resolve()):
        raise AdapterTrainingError("Adapter training data 必须位于 qg_prqk/outputs")
    if output_dir.parent != config.paths.output_dir.resolve() or output_dir.name != "query_adapter":
        raise AdapterTrainingError("Adapter 输出必须是配置 output_dir/query_adapter")
    if (output_dir / "_SUCCESS").is_file():
        if not config.runtime.resume:
            raise AdapterTrainingError(f"Adapter 输出已完成且 resume=false：{output_dir}")
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("training_data", {}).get("sha256") != sha256_file(training_data):
            raise AdapterTrainingError("已有 Adapter 与当前 training data 不一致")
        return manifest
    if output_dir.exists():
        if not config.runtime.overwrite:
            raise AdapterTrainingError(f"Adapter 输出已存在且 overwrite=false：{output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    bundle = load_training_bundle(training_data, limit=limit)
    reasonable = _reasonable_positive_rows(bundle)
    candidate_rows = torch.from_numpy(bundle["candidate_poi_rows"].astype(np.int64))
    target_rows = torch.from_numpy(bundle["target_poi_rows"].astype(np.int64))
    valid_mask = build_negative_valid_mask(candidate_rows, target_rows, reasonable)
    if not bool(valid_mask.any(dim=1).all().item()):
        raise AdapterTrainingError("至少一个 Query 在 false-negative mask 后没有负样本")
    query_ids = [int(value) for value in bundle["query_ids"]]
    train_indices, dev_indices = stable_train_dev_split(
        query_ids,
        dev_fraction=config.query_adapter.dev_fraction,
        seed=config.project.seed,
    )
    raw = torch.from_numpy(bundle["raw_queries"].astype(np.float32))
    positives = torch.from_numpy(bundle["positive_embeddings"].astype(np.float32))
    negatives = torch.from_numpy(bundle["negative_embeddings"].astype(np.float32))
    weights = torch.from_numpy(bundle["query_weights"].astype(np.float32))
    model = ResidualQueryAdapter(
        raw.shape[1],
        config.query_adapter.bottleneck,
        residual_scale=config.query_adapter.residual_scale,
        dropout=config.query_adapter.dropout,
    )
    result = fit_query_adapter(
        model,
        raw,
        positives,
        negatives,
        valid_mask,
        weights,
        target_rows,
        reasonable,
        train_indices,
        dev_indices,
        config.query_adapter,
        device=torch.device(device),
        batch_size=batch_size,
        seed=config.project.seed,
    )
    checkpoint_path = output_dir / "adapter.pt"
    temporary = output_dir / ".adapter.pt.writing"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "embedding_dim": raw.shape[1],
            "adapter_config": asdict(config.query_adapter),
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "built_at": utc_now(),
        "config_path": str(config.source_path),
        "config_sha256": config.source_sha256,
        "config_signature": config.signature(),
        "training_data": {
            "schema_version": TRAINING_DATA_SCHEMA_VERSION,
            "path": str(training_data),
            "sha256": sha256_file(training_data),
            "selected_rows": len(raw),
            "embedding_dim": raw.shape[1],
            "negatives_per_query": negatives.shape[1],
            "valid_and_test_read": False,
        },
        "split": {
            "source": "Train-only stable Query ID hash",
            "train_rows": len(train_indices),
            "dev_rows": len(dev_indices),
            "dev_fraction": config.query_adapter.dev_fraction,
            "seed": config.project.seed,
        },
        "training": {
            "device": device,
            "batch_size": batch_size or config.query_adapter.batch_size,
            "epoch_losses": list(result.epoch_losses),
        },
        "evaluation": {
            "raw": result.raw_metrics,
            "adapted": result.adapted_metrics,
            "identity_fallback_decision": "pending_hard_subset_review",
        },
        "output": {
            "checkpoint": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
        },
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").touch()
    return manifest

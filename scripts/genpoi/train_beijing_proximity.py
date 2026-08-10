#!/usr/bin/env python3
"""Train and calibrate the Beijing ordinal safe-prefix head for GenPOI."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding import ModelConfig, _load_encoder  # noqa: E402
from poi_gr.methods.genpoi.beijing_ssp import (  # noqa: E402
    USEFUL_PREFIX_DEPTHS,
    calibrate_safe_prefix_thresholds,
    evaluate_safe_prefixes,
    select_safe_prefix_depth,
)
from poi_gr.pid.trie import sha256_file  # noqa: E402


class BeijingProximityTrainingError(RuntimeError):
    """Raised when Beijing ordinal-head training cannot satisfy its contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "冻结 BGE-M3，训练北京场景的四输出序分类头，并在独立 Calibration "
            "上按误剪预算校准安全 GID 前缀阈值。"
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/bge-m3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--encode-buffer-size", type=int, default=8192)
    parser.add_argument("--head-batch-size", type=int, default=2048)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-unsafe-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--calibration-limit", type=int)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_data_contract(
    data_dir: Path,
) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BeijingProximityTrainingError(f"数据 manifest 不存在：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BeijingProximityTrainingError("数据 manifest JSON 非法") from error
    if manifest.get("schema_version") != "genpoi-beijing-proximity-data-v1":
        raise BeijingProximityTrainingError("数据 schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise BeijingProximityTrainingError("数据状态不是 completed")
    if tuple(manifest.get("useful_prefix_depths", ())) != USEFUL_PREFIX_DEPTHS:
        raise BeijingProximityTrainingError("数据 useful_prefix_depths 不兼容")
    paths: list[Path] = []
    for key in ("train", "calibration"):
        spec = manifest.get(key)
        if not isinstance(spec, dict):
            raise BeijingProximityTrainingError(f"数据 manifest 缺少 {key}")
        path = Path(spec.get("output_file", "")).resolve()
        if not path.is_file() or sha256_file(path) != spec.get("output_sha256"):
            raise BeijingProximityTrainingError(f"{key} Parquet 不存在或 SHA256 错误")
        if pq.ParquetFile(path).metadata.num_rows != spec.get("rows"):
            raise BeijingProximityTrainingError(f"{key} Parquet 行数不一致")
        paths.append(path)
    reserved = manifest.get("reserved_evaluation", {})
    if reserved.get("calibration_overlap_count") != 0:
        raise BeijingProximityTrainingError("Calibration 与固定评测集存在重合")
    return manifest, paths[0], paths[1]


def iter_batches(
    path: Path,
    *,
    batch_rows: int,
    limit: int | None,
) -> Iterator[tuple[list[str], np.ndarray]]:
    emitted = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_rows,
        columns=["query", "label"],
        use_threads=True,
    ):
        remaining = None if limit is None else limit - emitted
        if remaining is not None and remaining <= 0:
            break
        take = batch.num_rows if remaining is None else min(batch.num_rows, remaining)
        queries = batch.column("query").slice(0, take).to_pylist()
        labels = np.asarray(
            batch.column("label").slice(0, take).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        if any(not isinstance(query, str) or not query for query in queries):
            raise BeijingProximityTrainingError("Query Parquet 包含空值")
        if labels.size != take or np.any((labels < 0) | (labels > 6)):
            raise BeijingProximityTrainingError("Label 必须位于 [0, 6]")
        emitted += take
        yield queries, labels


def encode(encoder: Any, queries: list[str], batch_size: int) -> np.ndarray:
    values = np.asarray(
        encoder.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name=None,
        ),
        dtype=np.float32,
    )
    if values.ndim != 2 or values.shape[0] != len(queries):
        raise BeijingProximityTrainingError(f"BGE-M3 返回异常 shape：{values.shape}")
    if not np.isfinite(values).all():
        raise BeijingProximityTrainingError("BGE-M3 embedding 包含 NaN/Inf")
    return values


def make_targets(labels: np.ndarray) -> np.ndarray:
    return np.stack(
        [labels >= depth for depth in USEFUL_PREFIX_DEPTHS],
        axis=1,
    ).astype(np.float32)


def build_positive_weights(data_manifest: dict[str, Any]) -> np.ndarray:
    counts = data_manifest["train"].get("label_counts", {})
    label_counts = np.asarray(
        [int(counts.get(str(index), 0)) for index in range(7)],
        dtype=np.int64,
    )
    rows = int(label_counts.sum())
    if rows != int(data_manifest["train"].get("rows", -1)):
        raise BeijingProximityTrainingError("Train label_counts 行数不守恒")
    weights: list[float] = []
    for depth in USEFUL_PREFIX_DEPTHS:
        positive = int(label_counts[depth:].sum())
        negative = rows - positive
        if positive == 0 or negative == 0:
            raise BeijingProximityTrainingError(f"depth={depth} 无法计算类别权重")
        weights.append(math.sqrt(negative / positive))
    return np.asarray(weights, dtype=np.float32)


def train_epoch(
    *,
    encoder: Any,
    head: Any,
    optimizer: Any,
    train_file: Path,
    positive_weights: Any,
    encode_batch_size: int,
    encode_buffer_size: int,
    head_batch_size: int,
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    import torch

    head.train()
    rng = np.random.default_rng(seed)
    total_rows = 0
    loss_sum = 0.0
    binary_correct = np.zeros(len(USEFUL_PREFIX_DEPTHS), dtype=np.int64)
    started = time.perf_counter()
    for queries, labels in iter_batches(
        train_file,
        batch_rows=encode_buffer_size,
        limit=limit,
    ):
        embeddings = encode(encoder, queries, encode_batch_size)
        targets = make_targets(labels)
        permutation = rng.permutation(len(labels))
        embeddings = embeddings[permutation]
        targets = targets[permutation]
        for start in range(0, len(labels), head_batch_size):
            stop = min(start + head_batch_size, len(labels))
            features = torch.from_numpy(embeddings[start:stop]).to("cuda")
            target_tensor = torch.from_numpy(targets[start:stop]).to("cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                target_tensor,
                pos_weight=positive_weights,
            )
            if not torch.isfinite(loss):
                raise BeijingProximityTrainingError("分类头 Loss 出现 NaN/Inf")
            loss.backward()
            optimizer.step()
            size = stop - start
            loss_sum += float(loss.detach()) * size
            predictions = (logits.detach() > 0).cpu().numpy()
            binary_correct += (predictions == targets[start:stop]).sum(axis=0)
        total_rows += len(labels)
        if total_rows % 100_000 < len(labels):
            print(
                f"[train] {total_rows:,} rows loss={loss_sum / total_rows:.6f}",
                file=sys.stderr,
                flush=True,
            )
    seconds = time.perf_counter() - started
    return {
        "rows": total_rows,
        "loss": loss_sum / total_rows,
        "per_depth_binary_accuracy_at_0_5": {
            str(depth): binary_correct[index] / total_rows
            for index, depth in enumerate(USEFUL_PREFIX_DEPTHS)
        },
        "seconds": seconds,
        "rows_per_second": total_rows / seconds,
    }


def calibrate_epoch(
    *,
    encoder: Any,
    head: Any,
    calibration_file: Path,
    positive_weights: Any,
    encode_batch_size: int,
    encode_buffer_size: int,
    limit: int | None,
    max_unsafe_rate: float,
) -> dict[str, Any]:
    import torch

    head.eval()
    probability_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    loss_sum = 0.0
    total_rows = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for queries, labels in iter_batches(
            calibration_file,
            batch_rows=encode_buffer_size,
            limit=limit,
        ):
            features = torch.from_numpy(
                encode(encoder, queries, encode_batch_size)
            ).to("cuda")
            targets = torch.from_numpy(make_targets(labels)).to("cuda")
            logits = head(features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=positive_weights,
            )
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            probability_chunks.append(probabilities)
            label_chunks.append(labels.copy())
            loss_sum += float(loss) * len(labels)
            total_rows += len(labels)
            if total_rows % 100_000 < len(labels):
                print(
                    f"[calibration] {total_rows:,} rows",
                    file=sys.stderr,
                    flush=True,
                )
    probabilities = np.concatenate(probability_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    thresholds = calibrate_safe_prefix_thresholds(
        probabilities,
        labels,
        max_unsafe_rate=max_unsafe_rate,
    )
    selected = [
        select_safe_prefix_depth(row, thresholds) for row in probabilities
    ]
    safety = asdict(evaluate_safe_prefixes(labels.tolist(), selected))
    per_depth: dict[str, Any] = {}
    for index, depth in enumerate(USEFUL_PREFIX_DEPTHS):
        decisions = probabilities[:, index] > thresholds[depth]
        truth = labels >= depth
        tp = int(np.count_nonzero(decisions & truth))
        fp = int(np.count_nonzero(decisions & ~truth))
        fn = int(np.count_nonzero(~decisions & truth))
        per_depth[str(depth)] = {
            "threshold": thresholds[depth],
            "support": int(np.count_nonzero(truth)),
            "predicted": int(np.count_nonzero(decisions)),
            "true_positive": tp,
            "false_positive": fp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    seconds = time.perf_counter() - started
    return {
        "rows": total_rows,
        "loss": loss_sum / total_rows,
        "max_unsafe_rate": max_unsafe_rate,
        "calibrated_thresholds": {
            str(depth): thresholds[depth] for depth in USEFUL_PREFIX_DEPTHS
        },
        "safe_prefix_metrics": safety,
        "per_depth": per_depth,
        "seconds": seconds,
        "rows_per_second": total_rows / seconds,
    }


def main() -> int:
    args = parse_args()
    try:
        for name in (
            "epochs",
            "encode_batch_size",
            "encode_buffer_size",
            "head_batch_size",
            "max_seq_length",
            "learning_rate",
        ):
            if getattr(args, name) <= 0:
                raise BeijingProximityTrainingError(
                    f"--{name.replace('_', '-')} 必须大于 0"
                )
        if args.weight_decay < 0:
            raise BeijingProximityTrainingError("--weight-decay 不得为负数")
        if not 0.0 <= args.max_unsafe_rate < 1.0:
            raise BeijingProximityTrainingError("--max-unsafe-rate 必须位于 [0, 1)")
        if args.head_batch_size > args.encode_buffer_size:
            raise BeijingProximityTrainingError(
                "--head-batch-size 不得大于 --encode-buffer-size"
            )
        for name in ("train_limit", "calibration_limit"):
            value = getattr(args, name)
            if value is not None and value <= 0:
                raise BeijingProximityTrainingError(
                    f"--{name.replace('_', '-')} 必须大于 0"
                )

        import torch
        from safetensors.torch import save_file

        if not torch.cuda.is_available():
            raise BeijingProximityTrainingError("训练需要 CUDA GPU")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        data_dir = resolve(args.data_dir)
        model_path = resolve(args.model)
        output_dir = resolve(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "training_manifest.json"
        head_path = output_dir / "ordinal_head.safetensors"
        if manifest_path.exists() or head_path.exists():
            raise BeijingProximityTrainingError("输出已存在，请使用新的输出目录")
        data_manifest, train_file, calibration_file = load_data_contract(data_dir)
        positive_weights_array = build_positive_weights(data_manifest)

        model_config = ModelConfig(
            path=model_path,
            device="cuda",
            batch_size=args.encode_batch_size,
            encode_buffer_size=args.encode_buffer_size,
            max_seq_length=args.max_seq_length,
            torch_dtype="bfloat16",
            attention="sdpa",
            padding_side="right",
            normalize_embeddings=True,
            truncate_dim=None,
            prompt_name=None,
        )
        encoder, device, load_seconds = _load_encoder(model_config)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        embedding_dim = encoder.get_sentence_embedding_dimension()
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise BeijingProximityTrainingError("无法确定 BGE-M3 embedding_dim")
        head = torch.nn.Linear(
            embedding_dim,
            len(USEFUL_PREFIX_DEPTHS),
        ).to("cuda", dtype=torch.float32)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        positive_weights = torch.from_numpy(positive_weights_array).to("cuda")

        history: list[dict[str, Any]] = []
        best_state: dict[str, Any] | None = None
        best_rank: tuple[float, float, float] | None = None
        selected_epoch = 0
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                encoder=encoder,
                head=head,
                optimizer=optimizer,
                train_file=train_file,
                positive_weights=positive_weights,
                encode_batch_size=args.encode_batch_size,
                encode_buffer_size=args.encode_buffer_size,
                head_batch_size=args.head_batch_size,
                limit=args.train_limit,
                seed=args.seed + epoch,
            )
            calibration_metrics = calibrate_epoch(
                encoder=encoder,
                head=head,
                calibration_file=calibration_file,
                positive_weights=positive_weights,
                encode_batch_size=args.encode_batch_size,
                encode_buffer_size=args.encode_buffer_size,
                limit=args.calibration_limit,
                max_unsafe_rate=args.max_unsafe_rate,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "calibration": calibration_metrics,
                }
            )
            safety = calibration_metrics["safe_prefix_metrics"]
            print(
                f"[epoch {epoch}] calibration_loss={calibration_metrics['loss']:.6f} "
                f"retention={safety['target_retention_rate']:.4%} "
                f"strong_prefix={safety['strong_prefix_rate']:.4%}",
                file=sys.stderr,
                flush=True,
            )
            rank = (
                float(safety["strong_prefix_rate"]),
                float(safety["useful_prefix_rate"]),
                -float(calibration_metrics["loss"]),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                selected_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().contiguous()
                    for name, tensor in head.state_dict().items()
                }
        assert best_state is not None and selected_epoch > 0
        save_file(best_state, str(head_path))
        selected = history[selected_epoch - 1]["calibration"]
        manifest = {
            "schema_version": "genpoi-beijing-proximity-head-v1",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "method": {
                "backbone": "BGE-M3",
                "backbone_frozen": True,
                "head": "linear ordinal cumulative binary classifier",
                "useful_prefix_depths": list(USEFUL_PREFIX_DEPTHS),
                "selection": "deepest depth whose cumulative decisions pass",
            },
            "safety_contract": {
                "max_unsafe_rate": args.max_unsafe_rate,
                "calibrated_on_reserved_eval": False,
                "calibrated_thresholds": selected["calibrated_thresholds"],
            },
            "training": {
                "epochs": args.epochs,
                "optimizer": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "positive_weighting": "sqrt(negative_count / positive_count)",
                "positive_weights": {
                    str(depth): float(positive_weights_array[index])
                    for index, depth in enumerate(USEFUL_PREFIX_DEPTHS)
                },
                "seed": args.seed,
                "train_limit": args.train_limit,
                "calibration_limit": args.calibration_limit,
            },
            "data_manifest": str(data_dir / "manifest.json"),
            "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
            "data": data_manifest,
            "model": {
                **asdict(model_config),
                "path": str(model_path),
                "embedding_dim": embedding_dim,
                "load_seconds": load_seconds,
                "device": device,
            },
            "history": history,
            "selected_epoch": selected_epoch,
            "selected_calibration": selected,
            "head_file": str(head_path),
            "head_sha256": sha256_file(head_path),
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        atomic_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except (BeijingProximityTrainingError, OSError, ValueError, KeyError) as error:
        print(f"训练失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

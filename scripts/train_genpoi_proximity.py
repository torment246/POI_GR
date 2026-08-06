#!/usr/bin/env python3
"""Train the frozen-BGE-M3 classification head used by GenPOI SSP."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding import ModelConfig, _load_encoder  # noqa: E402
from poi_gr.methods.genpoi_proximity import effective_prefix_length  # noqa: E402
from poi_gr.pid_trie import sha256_file  # noqa: E402


class ProximityTrainingError(RuntimeError):
    """Raised when proximity-head training inputs violate the contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "冻结本地 BGE-M3，仅训练七分类线性头预测 GenPOI 地理邻近层级 λ。"
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
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
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


def validate_positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ProximityTrainingError(f"{name} 必须大于 0")


def load_data_contract(data_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ProximityTrainingError(f"数据 manifest 不存在：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProximityTrainingError("数据 manifest JSON 非法") from error
    if manifest.get("schema_version") != "genpoi-proximity-data-v1":
        raise ProximityTrainingError("数据 schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise ProximityTrainingError("数据 manifest 状态不是 completed")
    paths: list[Path] = []
    for split in ("train", "valid"):
        spec = manifest.get("outputs", {}).get(split)
        if not isinstance(spec, dict):
            raise ProximityTrainingError(f"数据 manifest 缺少 {split}")
        path = Path(spec.get("output_file", "")).resolve()
        expected = (data_dir / f"{split}.parquet").resolve()
        if path != expected or not path.is_file():
            raise ProximityTrainingError(f"{split} Parquet 路径与 manifest 不一致")
        if sha256_file(path) != spec.get("output_sha256"):
            raise ProximityTrainingError(f"{split} Parquet SHA256 不一致")
        if pq.ParquetFile(path).metadata.num_rows != spec.get("rows"):
            raise ProximityTrainingError(f"{split} Parquet 行数不一致")
        paths.append(path)
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
            raise ProximityTrainingError("Query Parquet 包含空值")
        if labels.size != take or np.any((labels < 0) | (labels > 6)):
            raise ProximityTrainingError("Label 必须位于 [0, 6]")
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
        raise ProximityTrainingError(f"BGE-M3 返回异常 shape：{values.shape}")
    if not np.isfinite(values).all():
        raise ProximityTrainingError("BGE-M3 embedding 包含 NaN/Inf")
    return values


def train_epoch(
    *,
    encoder: Any,
    head: Any,
    optimizer: Any,
    train_file: Path,
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
    correct = 0
    label_counts: Counter[int] = Counter()
    started = time.perf_counter()
    for queries, labels in iter_batches(
        train_file,
        batch_rows=encode_buffer_size,
        limit=limit,
    ):
        embeddings = encode(encoder, queries, encode_batch_size)
        permutation = rng.permutation(len(labels))
        embeddings = embeddings[permutation]
        labels = labels[permutation]
        for start in range(0, len(labels), head_batch_size):
            stop = min(start + head_batch_size, len(labels))
            features = torch.from_numpy(embeddings[start:stop]).to("cuda")
            targets = torch.from_numpy(labels[start:stop]).to("cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ProximityTrainingError("分类头 Loss 出现 NaN/Inf")
            loss.backward()
            optimizer.step()
            size = stop - start
            loss_sum += float(loss.detach()) * size
            correct += int((logits.argmax(dim=-1) == targets).sum())
        label_counts.update(int(value) for value in labels)
        total_rows += len(labels)
        if total_rows % 100_000 < len(labels):
            print(
                f"[train] {total_rows:,} rows loss={loss_sum / total_rows:.6f} "
                f"acc={correct / total_rows:.4%}",
                file=sys.stderr,
                flush=True,
            )
    seconds = time.perf_counter() - started
    return {
        "rows": total_rows,
        "loss": loss_sum / total_rows,
        "accuracy": correct / total_rows,
        "label_counts": {str(index): label_counts[index] for index in range(7)},
        "seconds": seconds,
        "rows_per_second": total_rows / seconds,
    }


def evaluate(
    *,
    encoder: Any,
    head: Any,
    valid_file: Path,
    encode_batch_size: int,
    encode_buffer_size: int,
    limit: int | None,
) -> dict[str, Any]:
    import torch

    head.eval()
    confusion = np.zeros((7, 7), dtype=np.int64)
    loss_sum = 0.0
    total_rows = 0
    absolute_error = 0
    retained = 0
    predicted_prefix_total = 0
    oracle_prefix_total = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for queries, labels in iter_batches(
            valid_file,
            batch_rows=encode_buffer_size,
            limit=limit,
        ):
            features = torch.from_numpy(
                encode(encoder, queries, encode_batch_size)
            ).to("cuda")
            targets = torch.from_numpy(labels).to("cuda")
            logits = head(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            predictions = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
            size = len(labels)
            loss_sum += float(loss) * size
            total_rows += size
            absolute_error += int(np.abs(predictions - labels).sum())
            for target, prediction in zip(labels, predictions):
                confusion[int(target), int(prediction)] += 1
                predicted_prefix = effective_prefix_length(int(prediction), gamma=2)
                oracle_prefix = effective_prefix_length(int(target), gamma=2)
                predicted_prefix_total += predicted_prefix
                oracle_prefix_total += oracle_prefix
                retained += int(predicted_prefix <= int(target))
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in range(7):
        tp = int(confusion[label, label])
        support = int(confusion[label].sum())
        predicted = int(confusion[:, label].sum())
        recall = tp / support if support else None
        precision = tp / predicted if predicted else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        if support and f1 is not None:
            f1_values.append(f1)
        per_class[str(label)] = {
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    seconds = time.perf_counter() - started
    correct = int(np.trace(confusion))
    return {
        "rows": total_rows,
        "loss": loss_sum / total_rows,
        "accuracy": correct / total_rows,
        "macro_f1_present_classes": sum(f1_values) / len(f1_values),
        "mean_absolute_class_error": absolute_error / total_rows,
        "target_retention_rate_after_ssp": retained / total_rows,
        "unsafe_pruning_rate": 1 - retained / total_rows,
        "mean_predicted_prefill_tokens": predicted_prefix_total / total_rows,
        "mean_oracle_prefill_tokens": oracle_prefix_total / total_rows,
        "confusion_matrix_true_by_pred": confusion.tolist(),
        "per_class": per_class,
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
            validate_positive(getattr(args, name), f"--{name.replace('_', '-')}")
        if args.weight_decay < 0:
            raise ProximityTrainingError("--weight-decay 不得为负数")
        if args.head_batch_size > args.encode_buffer_size:
            raise ProximityTrainingError(
                "--head-batch-size 不得大于 --encode-buffer-size"
            )
        for name in ("train_limit", "valid_limit"):
            value = getattr(args, name)
            if value is not None:
                validate_positive(value, f"--{name.replace('_', '-')}")

        import torch
        from safetensors.torch import save_file

        if not torch.cuda.is_available():
            raise ProximityTrainingError("训练需要 CUDA GPU")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        data_dir = resolve(args.data_dir)
        model_path = resolve(args.model)
        output_dir = resolve(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "training_manifest.json"
        head_path = output_dir / "proximity_head.safetensors"
        if manifest_path.exists() or head_path.exists():
            raise ProximityTrainingError("输出已存在，请使用新的输出目录")
        data_manifest, train_file, valid_file = load_data_contract(data_dir)

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
            raise ProximityTrainingError("无法确定 BGE-M3 embedding_dim")
        head = torch.nn.Linear(embedding_dim, 7).to("cuda", dtype=torch.float32)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        history: list[dict[str, Any]] = []
        best_state: dict[str, Any] | None = None
        best_loss = math.inf
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                encoder=encoder,
                head=head,
                optimizer=optimizer,
                train_file=train_file,
                encode_batch_size=args.encode_batch_size,
                encode_buffer_size=args.encode_buffer_size,
                head_batch_size=args.head_batch_size,
                limit=args.train_limit,
                seed=args.seed + epoch,
            )
            valid_metrics = evaluate(
                encoder=encoder,
                head=head,
                valid_file=valid_file,
                encode_batch_size=args.encode_batch_size,
                encode_buffer_size=args.encode_buffer_size,
                limit=args.valid_limit,
            )
            history.append(
                {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
            )
            print(
                f"[epoch {epoch}] valid_loss={valid_metrics['loss']:.6f} "
                f"accuracy={valid_metrics['accuracy']:.4%} "
                f"retention={valid_metrics['target_retention_rate_after_ssp']:.4%}",
                file=sys.stderr,
                flush=True,
            )
            if valid_metrics["loss"] < best_loss:
                best_loss = float(valid_metrics["loss"])
                best_state = {
                    name: tensor.detach().cpu().contiguous()
                    for name, tensor in head.state_dict().items()
                }
        assert best_state is not None
        save_file(best_state, str(head_path))

        weight_path = model_path / "pytorch_model.bin"
        manifest = {
            "schema_version": "genpoi-proximity-head-v1",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "paper_method": {
                "backbone": "BGE-M3",
                "backbone_frozen": True,
                "trained_parameters": "linear classification head only",
                "classes": list(range(7)),
                "gamma": 2,
            },
            "local_reproduction_choices_not_specified_by_paper": {
                "epochs": args.epochs,
                "optimizer": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "class_weighting": "none",
                "max_seq_length": args.max_seq_length,
                "seed": args.seed,
            },
            "data_manifest": str(data_dir / "manifest.json"),
            "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
            "data": data_manifest["outputs"],
            "train_limit": args.train_limit,
            "valid_limit": args.valid_limit,
            "model": {
                **asdict(model_config),
                "path": str(model_path),
                "embedding_dim": embedding_dim,
                "load_seconds": load_seconds,
                "device": device,
                "backbone_weight_sha256": (
                    sha256_file(weight_path) if weight_path.is_file() else None
                ),
            },
            "history": history,
            "selected_epoch": min(
                history,
                key=lambda item: item["valid"]["loss"],
            )["epoch"],
            "head_file": str(head_path),
            "head_sha256": sha256_file(head_path),
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        atomic_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except (ProximityTrainingError, OSError, ValueError, KeyError) as error:
        print(f"训练失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

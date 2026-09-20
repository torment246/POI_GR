#!/usr/bin/env python3
"""Train fresh TIGER SID and vanilla Qwen in one optimizer-step pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from poi_gr.methods.tiger_joint import (  # noqa: E402
    HuggingFaceCausalLMJointAdapter,
    JointTigerLossOutput,
    JointTigerTrainingModule,
    PoiEmbeddingStore,
    build_sid_token_layout,
    collate_dynamic_examples,
    load_dynamic_token_ids,
    load_fresh_tokenizer_and_template,
    parse_sid_free_record,
    tokenize_dynamic_record,
)
from poi_gr.methods.tiger_joint.initialization import (  # noqa: E402
    INITIALIZATION_MANIFEST_NAME,
    load_fresh_rqvae_initialization,
)
from poi_gr.methods.tiger_joint.preparation import (  # noqa: E402
    DynamicTigerExample,
)
from poi_gr.pid.dedup import sha256_file  # noqa: E402
from poi_gr.sid.rqvae import RQVAE  # noqa: E402
from scripts.tiger_joint.overfit_joint import (  # noqa: E402
    load_formal_preflight,
    module_sha256,
)
from scripts.tiger_joint.smoke_joint_step import (  # noqa: E402
    CODEBOOK_SIZES,
    CUTOFF_LEN,
    atomic_json,
    gradient_norm,
    load_data_manifest,
    seed_everything,
)


SCHEMA_VERSION = "tiger-joint-training-v3"
CONFIG_SCHEMA_VERSION = "tiger-joint-training-config-v1"
CHECKPOINT_SCHEMA_VERSION = "tiger-joint-checkpoint-v3"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "tiger-joint-checkpoint-manifest-v3"


class TigerJointTrainingError(RuntimeError):
    """Raised when the joint training protocol or runtime is invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    source_path: Path
    data_dir: Path
    preflight_state: Path
    embedding_dir: Path
    rq_initialization_dir: Path
    qwen_dir: Path
    cutoff_len: int
    gradient_checkpointing: bool
    epochs: int
    behavior_batch_size: int
    gradient_accumulation_steps: int
    catalog_batch_size: int
    shuffle_buffer_rows: int
    qwen_learning_rate: float
    rq_learning_rate: float
    qwen_weight_decay: float
    alignment_weight: float
    rq_weight: float
    temperature: float
    warmup_ratio: float
    gradient_clip_norm: float
    log_every_optimizer_steps: int
    probe_every_optimizer_steps: int
    probe_rows: int
    expected_world_size: int | None
    expected_global_behavior_batch_size: int | None
    expected_global_catalog_rows_per_step: int | None
    seed: int


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


@dataclass(frozen=True)
class LineShard:
    rank: int
    start_row: int
    end_row: int
    start_offset: int
    end_offset: int

    @property
    def rows(self) -> int:
        return self.end_row - self.start_row


@dataclass
class MetricAccumulator:
    samples: int = 0
    catalog_samples: int = 0
    microbatches: int = 0
    totals: torch.Tensor | None = None
    rq_total: torch.Tensor | None = None
    target_row_batches: list[torch.Tensor] = field(default_factory=list)
    target_code_batches: list[torch.Tensor] = field(default_factory=list)

    def add(
        self,
        output: JointTigerLossOutput,
        target_poi_rows: torch.Tensor,
        *,
        catalog_samples: int,
    ) -> None:
        batch_size = int(output.target_codes.shape[0])
        if target_poi_rows.shape != (batch_size,):
            raise TigerJointTrainingError(
                "target_poi_rows shape 必须与当前行为 batch 一致"
            )
        if catalog_samples <= 0:
            raise TigerJointTrainingError("catalog_samples 必须为正数")
        metrics = (
            torch.stack(
                (
                    output.generation_loss.detach().float(),
                    output.alignment_loss.detach().float(),
                    output.teacher_forced_token_accuracy.detach().float(),
                    output.teacher_forced_sid_token_accuracy.detach().float(),
                    output.teacher_forced_static_token_accuracy.detach().float(),
                    output.teacher_forced_exact_match.detach().float(),
                )
            )
            * batch_size
        )
        rq_total = output.rq_loss.detach().float() * catalog_samples
        self.samples += batch_size
        self.catalog_samples += catalog_samples
        self.microbatches += 1
        self.totals = metrics if self.totals is None else self.totals + metrics
        self.rq_total = rq_total if self.rq_total is None else self.rq_total + rq_total
        self.target_row_batches.append(target_poi_rows.detach())
        self.target_code_batches.append(output.target_codes.detach())

    def summary(
        self,
        *,
        alignment_weight: float,
        rq_weight: float,
    ) -> dict[str, float | int]:
        if (
            not self.samples
            or not self.catalog_samples
            or not self.microbatches
            or self.totals is None
            or self.rq_total is None
        ):
            raise TigerJointTrainingError("当前 optimizer step 没有有效 microbatch")
        device = self.totals.device
        distinct_target_rows = sum(
            torch.unique(rows).numel() for rows in self.target_row_batches
        )
        distinct_target_sids = sum(
            torch.unique(codes, dim=0).shape[0] for codes in self.target_code_batches
        )
        packed = torch.cat(
            (
                self.totals,
                self.rq_total.reshape(1),
                torch.tensor(
                    (
                        self.samples,
                        self.catalog_samples,
                        self.microbatches,
                        distinct_target_rows,
                        distinct_target_sids,
                    ),
                    device=device,
                    dtype=self.totals.dtype,
                ),
            )
        )
        if dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        values = packed.cpu().tolist()
        samples = int(values[7])
        catalog_samples = int(values[8])
        microbatches = int(values[9])
        generation_loss = values[0] / samples
        alignment_loss = values[1] / samples
        rq_loss = values[6] / catalog_samples
        return {
            "samples": samples,
            "catalog_samples": catalog_samples,
            "microbatches": microbatches,
            "total_loss": (
                generation_loss
                + alignment_weight * alignment_loss
                + rq_weight * rq_loss
            ),
            "generation_loss": generation_loss,
            "alignment_loss": alignment_loss,
            "rq_loss": rq_loss,
            "teacher_forced_token_accuracy": values[2] / samples,
            "teacher_forced_sid_token_accuracy": values[3] / samples,
            "teacher_forced_static_token_accuracy": values[4] / samples,
            "teacher_forced_exact_match": values[5] / samples,
            "mean_distinct_target_rows_per_microbatch": (values[10] / microbatches),
            "mean_distinct_target_sids_per_microbatch": (values[11] / microbatches),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按行为 microbatch 动态生成当前三级 SID，同时反向更新 vanilla Qwen、"
            "Query 投影和 fresh RQ-VAE；支持有界服务器训练探测。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tiger_joint/v1_server.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--local-rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        help="仅运行指定 optimizer step；省略时执行配置中的完整三轮。",
    )
    parser.add_argument("--behavior-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--catalog-batch-size", type=int)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="只允许从完整 epoch 边界且包含 optimizer state 的联合 checkpoint 恢复。",
    )
    parser.add_argument(
        "--save-optimizer-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="checkpoint 是否保存两个 optimizer 和 scheduler 状态。",
    )
    parser.add_argument(
        "--save-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在有界训练结束或每轮结束时保存模型 checkpoint。",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def initialize_distributed(args: argparse.Namespace) -> DistributedRuntime:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        environment_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise TigerJointTrainingError("torchrun rank 环境变量必须是整数") from error
    if world_size <= 0 or not 0 <= rank < world_size:
        raise TigerJointTrainingError("torchrun WORLD_SIZE/RANK 无效")
    if world_size == 1:
        if rank != 0:
            raise TigerJointTrainingError("单进程训练的 RANK 必须为 0")
        return DistributedRuntime(
            rank=0,
            local_rank=args.device_index,
            world_size=1,
        )
    local_rank = (
        args.local_rank if args.local_rank is not None else environment_local_rank
    )
    if not torch.cuda.is_available():
        raise TigerJointTrainingError("NCCL DDP 要求宿主 CUDA")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise TigerJointTrainingError("LOCAL_RANK 超出可见 GPU 数量")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedRuntime(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def build_line_shards(
    path: Path,
    *,
    expected_rows: int,
    world_size: int,
) -> list[LineShard]:
    """Scan JSONL once and return exact row-aligned byte ranges for all ranks."""

    if expected_rows <= 0 or world_size <= 0:
        raise TigerJointTrainingError("行分片要求正的 rows 与 world_size")
    target_rows = [expected_rows * rank // world_size for rank in range(world_size + 1)]
    offsets: list[int | None] = [None] * (world_size + 1)
    target_index = 0
    row = 0
    with path.open("rb") as stream:
        while True:
            position = stream.tell()
            while target_index < len(target_rows) and row == target_rows[target_index]:
                offsets[target_index] = position
                target_index += 1
            line = stream.readline()
            if not line:
                break
            row += 1
    if row != expected_rows:
        raise TigerJointTrainingError(
            f"Train 实际 {row} 行，与 manifest {expected_rows} 不一致"
        )
    if any(offset is None for offset in offsets):
        raise TigerJointTrainingError("未能解析全部 Train 行分片 offset")
    concrete_offsets = [int(offset) for offset in offsets]
    return [
        LineShard(
            rank=rank,
            start_row=target_rows[rank],
            end_row=target_rows[rank + 1],
            start_offset=concrete_offsets[rank],
            end_offset=concrete_offsets[rank + 1],
        )
        for rank in range(world_size)
    ]


def distribute_line_shards(
    path: Path,
    *,
    expected_rows: int,
    runtime: DistributedRuntime,
) -> list[LineShard]:
    if runtime.world_size == 1:
        return [
            LineShard(
                rank=0,
                start_row=0,
                end_row=expected_rows,
                start_offset=0,
                end_offset=path.stat().st_size,
            )
        ]
    payload: list[Any] = [None]
    if runtime.is_main:
        payload[0] = build_line_shards(
            path,
            expected_rows=expected_rows,
            world_size=runtime.world_size,
        )
    if runtime.enabled:
        dist.broadcast_object_list(payload, src=0)
    shards = payload[0]
    if not isinstance(shards, list) or len(shards) != runtime.world_size:
        raise TigerJointTrainingError("DDP Train 行分片广播失败")
    return shards


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TigerJointTrainingError(f"{name} 必须是 YAML mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TigerJointTrainingError(f"{name} 必须是正整数")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise TigerJointTrainingError(f"{name} 必须大于 0")
    return float(value)


def _optional_positive_int(value: Any, name: str) -> int | None:
    return None if value is None else _positive_int(value, name)


def load_config(path: Path, args: argparse.Namespace) -> TrainingConfig:
    path = resolve(path)
    if not path.is_file():
        raise TigerJointTrainingError(f"训练配置不存在：{path}")
    with path.open("r", encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), "配置根节点")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise TigerJointTrainingError("训练配置 schema_version 无效")
    data = _mapping(raw.get("data"), "data")
    model = _mapping(raw.get("model"), "model")
    training = _mapping(raw.get("training"), "training")
    behavior_batch_size = (
        args.behavior_batch_size
        if args.behavior_batch_size is not None
        else training.get("behavior_batch_size")
    )
    accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else training.get("gradient_accumulation_steps")
    )
    catalog_batch_size = (
        args.catalog_batch_size
        if args.catalog_batch_size is not None
        else training.get("catalog_batch_size")
    )
    cutoff_len = _positive_int(model.get("cutoff_len"), "model.cutoff_len")
    if cutoff_len != CUTOFF_LEN:
        raise TigerJointTrainingError("TIGER-Joint cutoff_len 必须为 1024")
    warmup_ratio = float(training.get("warmup_ratio"))
    if not 0.0 <= warmup_ratio < 1.0:
        raise TigerJointTrainingError("warmup_ratio 必须位于 [0, 1)")
    qwen_weight_decay = float(training.get("qwen_weight_decay"))
    if qwen_weight_decay < 0:
        raise TigerJointTrainingError("qwen_weight_decay 不能为负数")
    seed = int(training.get("seed"))
    if seed < 0:
        raise TigerJointTrainingError("seed 不能为负数")
    return TrainingConfig(
        source_path=path,
        data_dir=resolve(Path(data["data_dir"])),
        preflight_state=resolve(Path(data["preflight_state"])),
        embedding_dir=resolve(Path(data["embedding_dir"])),
        rq_initialization_dir=resolve(Path(data["rq_initialization_dir"])),
        qwen_dir=resolve(Path(model["qwen_dir"])),
        cutoff_len=cutoff_len,
        gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
        epochs=_positive_int(training.get("epochs"), "training.epochs"),
        behavior_batch_size=_positive_int(
            behavior_batch_size, "training.behavior_batch_size"
        ),
        gradient_accumulation_steps=_positive_int(
            accumulation_steps, "training.gradient_accumulation_steps"
        ),
        catalog_batch_size=_positive_int(
            catalog_batch_size, "training.catalog_batch_size"
        ),
        shuffle_buffer_rows=_positive_int(
            training.get("shuffle_buffer_rows"),
            "training.shuffle_buffer_rows",
        ),
        qwen_learning_rate=_positive_float(
            training.get("qwen_learning_rate"),
            "training.qwen_learning_rate",
        ),
        rq_learning_rate=_positive_float(
            training.get("rq_learning_rate"),
            "training.rq_learning_rate",
        ),
        qwen_weight_decay=qwen_weight_decay,
        alignment_weight=_positive_float(
            training.get("alignment_weight"),
            "training.alignment_weight",
        ),
        rq_weight=_positive_float(training.get("rq_weight"), "training.rq_weight"),
        temperature=_positive_float(
            training.get("temperature"), "training.temperature"
        ),
        warmup_ratio=warmup_ratio,
        gradient_clip_norm=_positive_float(
            training.get("gradient_clip_norm"),
            "training.gradient_clip_norm",
        ),
        log_every_optimizer_steps=_positive_int(
            training.get("log_every_optimizer_steps"),
            "training.log_every_optimizer_steps",
        ),
        probe_every_optimizer_steps=_positive_int(
            training.get("probe_every_optimizer_steps"),
            "training.probe_every_optimizer_steps",
        ),
        probe_rows=_positive_int(training.get("probe_rows"), "training.probe_rows"),
        expected_world_size=_optional_positive_int(
            training.get("expected_world_size"),
            "training.expected_world_size",
        ),
        expected_global_behavior_batch_size=_optional_positive_int(
            training.get("expected_global_behavior_batch_size"),
            "training.expected_global_behavior_batch_size",
        ),
        expected_global_catalog_rows_per_step=_optional_positive_int(
            training.get("expected_global_catalog_rows_per_step"),
            "training.expected_global_catalog_rows_per_step",
        ),
        seed=seed,
    )


def git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"revision": revision, "status_short": status, "dirty": bool(status)}


def config_payload(config: TrainingConfig) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(config).items()
    }


def config_signature(
    config: TrainingConfig,
    *,
    max_optimizer_steps: int | None,
    world_size: int,
    save_checkpoint: bool,
    save_optimizer_state: bool,
) -> str:
    payload = {
        "config": config_payload(config),
        "max_optimizer_steps": max_optimizer_steps,
        "world_size": world_size,
        "save_checkpoint": save_checkpoint,
        "save_optimizer_state": save_optimizer_state,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_joint_schedulers(
    qwen_optimizer: torch.optim.Optimizer,
    rq_optimizer: torch.optim.Optimizer,
    *,
    planned_optimizer_steps: int,
    warmup_ratio: float,
) -> tuple[Any, Any, int]:
    """Build matching warmup-cosine schedules for Qwen and RQ optimizers."""
    if planned_optimizer_steps <= 0:
        raise TigerJointTrainingError("planned_optimizer_steps 必须大于 0")
    if not 0.0 <= warmup_ratio < 1.0:
        raise TigerJointTrainingError("warmup_ratio 必须位于 [0, 1)")
    from transformers import get_cosine_schedule_with_warmup

    warmup_steps = int(math.floor(planned_optimizer_steps * warmup_ratio))
    qwen_scheduler = get_cosine_schedule_with_warmup(
        qwen_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_optimizer_steps,
    )
    rq_scheduler = get_cosine_schedule_with_warmup(
        rq_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_optimizer_steps,
    )
    return qwen_scheduler, rq_scheduler, warmup_steps


def iter_tokenized_examples(
    path: Path,
    *,
    shard: LineShard,
    tokenizer: Any,
    template: Any,
    token_ids: Any,
    cutoff_len: int,
) -> Iterator[DynamicTigerExample]:
    if path.name != "train.jsonl":
        raise TigerJointTrainingError("训练入口只允许读取 train.jsonl")
    rows = 0
    with path.open("rb") as stream:
        stream.seek(shard.start_offset)
        while rows < shard.rows:
            line = stream.readline()
            if not line:
                raise TigerJointTrainingError(f"Rank {shard.rank} Train 分片提前结束")
            line_number = shard.start_row + rows + 1
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise TigerJointTrainingError(
                    f"Train 第 {line_number} 行 JSON 无效"
                ) from error
            if not isinstance(record, dict):
                raise TigerJointTrainingError(f"Train 第 {line_number} 行必须是 object")
            rows += 1
            yield tokenize_dynamic_record(
                parse_sid_free_record(record, expected_split="train"),
                tokenizer=tokenizer,
                template=template,
                token_ids=token_ids,
                cutoff_len=cutoff_len,
            )
        if stream.tell() != shard.end_offset:
            raise TigerJointTrainingError(
                f"Rank {shard.rank} Train 分片 byte offset 不闭合"
            )
    if rows != shard.rows:
        raise TigerJointTrainingError(
            f"Rank {shard.rank} 实际读取 {rows} 行，与分片 {shard.rows} 不一致"
        )


def buffered_shuffle(
    examples: Iterator[DynamicTigerExample],
    *,
    buffer_rows: int,
    rng: np.random.Generator,
) -> Iterator[DynamicTigerExample]:
    buffer: list[DynamicTigerExample] = []
    for _ in range(buffer_rows):
        try:
            buffer.append(next(examples))
        except StopIteration:
            break
    while buffer:
        selected = int(rng.integers(0, len(buffer)))
        result = buffer[selected]
        try:
            buffer[selected] = next(examples)
        except StopIteration:
            buffer.pop(selected)
        yield result


def iter_behavior_batches(
    path: Path,
    *,
    shard: LineShard,
    tokenizer: Any,
    template: Any,
    token_ids: Any,
    config: TrainingConfig,
    epoch: int,
) -> Iterator[list[DynamicTigerExample]]:
    examples = iter_tokenized_examples(
        path,
        shard=shard,
        tokenizer=tokenizer,
        template=template,
        token_ids=token_ids,
        cutoff_len=config.cutoff_len,
    )
    shuffled = buffered_shuffle(
        examples,
        buffer_rows=config.shuffle_buffer_rows,
        rng=np.random.default_rng(config.seed + epoch + 1_000 * shard.rank),
    )
    batch: list[DynamicTigerExample] = []
    for example in shuffled:
        batch.append(example)
        if len(batch) == config.behavior_batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def global_behavior_batch_size(
    shards: Sequence[LineShard],
    *,
    microbatch_index: int,
    per_rank_batch_size: int,
) -> int:
    """Return the exact cross-rank sample count for one aligned microbatch."""

    if microbatch_index < 0 or per_rank_batch_size <= 0:
        raise TigerJointTrainingError("microbatch index 或 batch size 无效")
    start = microbatch_index * per_rank_batch_size
    return sum(max(0, min(per_rank_batch_size, shard.rows - start)) for shard in shards)


def accumulation_group_sizes(
    shards: Sequence[LineShard],
    *,
    first_microbatch_index: int,
    microbatches_per_epoch: int,
    per_rank_behavior_batch_size: int,
    per_rank_catalog_batch_size: int,
    accumulation_steps: int,
) -> tuple[int, int]:
    """Return exact behavior and catalog rows represented by one DDP step."""

    if (
        first_microbatch_index < 0
        or first_microbatch_index >= microbatches_per_epoch
        or first_microbatch_index % accumulation_steps != 0
        or per_rank_catalog_batch_size <= 0
        or accumulation_steps <= 0
    ):
        raise TigerJointTrainingError("optimizer accumulation group 边界无效")
    end = min(
        first_microbatch_index + accumulation_steps,
        microbatches_per_epoch,
    )
    behavior_rows = sum(
        global_behavior_batch_size(
            shards,
            microbatch_index=index,
            per_rank_batch_size=per_rank_behavior_batch_size,
        )
        for index in range(first_microbatch_index, end)
    )
    catalog_rows = (
        (end - first_microbatch_index) * per_rank_catalog_batch_size * len(shards)
    )
    if behavior_rows <= 0 or catalog_rows <= 0:
        raise TigerJointTrainingError("optimizer accumulation group 为空")
    return behavior_rows, catalog_rows


def encode_probe_codes(
    rqvae: RQVAE,
    embedding_store: PoiEmbeddingStore,
    probe_rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    was_training = rqvae.training
    rqvae.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(probe_rows), batch_size):
            rows = probe_rows[start : start + batch_size]
            values = torch.from_numpy(embedding_store.gather(rows)).to(device)
            chunks.append(rqvae.encode_codes(values).cpu())
    if was_training:
        rqvae.train()
    return torch.cat(chunks, dim=0)


def probe_metrics(
    codes: torch.Tensor,
    previous_codes: torch.Tensor | None,
    *,
    codebook_sizes: Sequence[int],
) -> dict[str, Any]:
    if codes.ndim != 2 or codes.shape[1] != len(codebook_sizes):
        raise TigerJointTrainingError("probe codes shape 无效")
    result: dict[str, Any] = {
        "rows": int(codes.shape[0]),
        "distinct_full_sids": int(torch.unique(codes, dim=0).shape[0]),
        "used_codes_by_level": [
            int(torch.unique(codes[:, level]).numel())
            for level in range(codes.shape[1])
        ],
        "utilization_by_level": [
            float(torch.unique(codes[:, level]).numel() / codebook_sizes[level])
            for level in range(codes.shape[1])
        ],
    }
    if previous_codes is None:
        result["full_sid_churn_rate"] = None
        result["churn_rate_by_level"] = None
    else:
        if previous_codes.shape != codes.shape:
            raise TigerJointTrainingError("probe 前后 codes shape 不一致")
        changed = codes.ne(previous_codes)
        result["full_sid_churn_rate"] = float(changed.any(dim=1).float().mean().item())
        result["churn_rate_by_level"] = [
            float(changed[:, level].float().mean().item())
            for level in range(changed.shape[1])
        ]
    return result


def _hash_checkpoint_files(checkpoint_dir: Path) -> dict[str, str]:
    names = [
        path.name
        for path in checkpoint_dir.iterdir()
        if path.is_file() and path.name != "checkpoint_manifest.json"
    ]
    return {name: sha256_file(checkpoint_dir / name) for name in sorted(names)}


def capture_rank_rng_state(device: torch.device) -> dict[str, Any]:
    """Capture the process-local random streams needed for exact DDP resume."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def gather_rank_rng_states(
    runtime: DistributedRuntime,
    device: torch.device,
) -> list[dict[str, Any]]:
    local_state = capture_rank_rng_state(device)
    if not runtime.enabled:
        return [local_state]
    gathered: list[Any] = [None] * runtime.world_size
    dist.all_gather_object(gathered, local_state)
    if not all(isinstance(state, dict) for state in gathered):
        raise TigerJointTrainingError("DDP rank RNG state 汇总失败")
    return gathered


def validate_rank_rng_states(
    states: Any,
    *,
    world_size: int,
) -> list[dict[str, Any]]:
    required_keys = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(states, list) or len(states) != world_size:
        raise TigerJointTrainingError("checkpoint rank RNG state 数量无效")
    for state in states:
        if not isinstance(state, dict) or not required_keys.issubset(state):
            raise TigerJointTrainingError("checkpoint rank RNG state 内容无效")
        if not isinstance(state["torch_cpu"], torch.Tensor) or not isinstance(
            state["torch_cuda"], torch.Tensor
        ):
            raise TigerJointTrainingError("checkpoint Torch RNG state 无效")
    return states


def restore_rank_rng_state(
    states: Any,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    validated = validate_rank_rng_states(states, world_size=world_size)
    state = validated[rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device=device)


def save_checkpoint(
    *,
    output_dir: Path,
    optimizer_step: int,
    epoch: int,
    examples_consumed: int,
    causal_lm: nn.Module,
    tokenizer: Any,
    rqvae: RQVAE,
    query_projection: nn.Module,
    qwen_optimizer: torch.optim.Optimizer,
    rq_optimizer: torch.optim.Optimizer,
    qwen_scheduler: Any,
    rq_scheduler: Any,
    signature: str,
    save_optimizer_state: bool,
    rank_rng_states: list[dict[str, Any]] | None,
    world_size: int,
    train_rows: int,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-step-{optimizer_step}"
    temporary = output_dir / f".checkpoint-step-{optimizer_step}.tmp"
    if checkpoint_dir.exists() or temporary.exists():
        raise TigerJointTrainingError("checkpoint 目标已存在，拒绝覆盖")
    temporary.mkdir()
    causal_lm.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    joint_state: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_signature": signature,
        "optimizer_step": optimizer_step,
        "epoch": epoch,
        "examples_consumed": examples_consumed,
        "world_size": world_size,
        "epoch_complete": examples_consumed == epoch * train_rows,
        "rqvae_state_dict": {
            name: value.detach().cpu() for name, value in rqvae.state_dict().items()
        },
        "query_projection_state_dict": {
            name: value.detach().cpu()
            for name, value in query_projection.state_dict().items()
        },
        "qwen_scheduler_state_dict": qwen_scheduler.state_dict(),
        "rq_scheduler_state_dict": rq_scheduler.state_dict(),
        "optimizer_state_saved": save_optimizer_state,
    }
    if save_optimizer_state:
        validated_rng_states = validate_rank_rng_states(
            rank_rng_states,
            world_size=world_size,
        )
        joint_state["qwen_optimizer_state_dict"] = qwen_optimizer.state_dict()
        joint_state["rq_optimizer_state_dict"] = rq_optimizer.state_dict()
        joint_state["rank_rng_states"] = validated_rng_states
    torch.save(joint_state, temporary / "joint_state.pt")
    files = _hash_checkpoint_files(temporary)
    atomic_json(
        temporary / "checkpoint_manifest.json",
        {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "status": "completed",
            "config_signature": signature,
            "optimizer_step": optimizer_step,
            "epoch": epoch,
            "examples_consumed": examples_consumed,
            "world_size": world_size,
            "epoch_complete": examples_consumed == epoch * train_rows,
            "optimizer_state_saved": save_optimizer_state,
            "scheduler_states_saved": True,
            "rank_rng_state_saved": save_optimizer_state,
            "files": files,
        },
    )
    os.replace(temporary, checkpoint_dir)
    return checkpoint_dir


def load_resume_checkpoint(
    checkpoint_dir: Path,
    *,
    signature: str,
    train_rows: int,
    world_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_dir = resolve(checkpoint_dir)
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if not checkpoint_dir.is_relative_to(outputs_root):
        raise TigerJointTrainingError("resume checkpoint 必须位于 outputs/ 下")
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    state_path = checkpoint_dir / "joint_state.pt"
    if not manifest_path.is_file() or not state_path.is_file():
        raise TigerJointTrainingError("resume checkpoint 缺少 manifest 或 joint_state")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerJointTrainingError("resume checkpoint manifest 无效") from error
    if not isinstance(manifest, dict) or (
        manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("config_signature") != signature
        or manifest.get("optimizer_state_saved") is not True
        or manifest.get("scheduler_states_saved") is not True
        or manifest.get("rank_rng_state_saved") is not True
        or manifest.get("epoch_complete") is not True
        or manifest.get("world_size") != world_size
    ):
        raise TigerJointTrainingError("resume checkpoint 协议与当前训练不一致")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise TigerJointTrainingError("resume checkpoint 缺少文件哈希")
    for name, expected_sha256 in files.items():
        if not isinstance(name, str) or not isinstance(expected_sha256, str):
            raise TigerJointTrainingError("resume checkpoint 文件哈希格式无效")
        path = checkpoint_dir / name
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise TigerJointTrainingError(f"resume checkpoint 文件损坏：{name}")
    state = torch.load(
        state_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(state, dict) or (
        state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or state.get("config_signature") != signature
        or state.get("optimizer_state_saved") is not True
        or state.get("epoch_complete") is not True
        or state.get("world_size") != world_size
    ):
        raise TigerJointTrainingError("resume joint_state 协议无效")
    required_resume_keys = {
        "qwen_optimizer_state_dict",
        "rq_optimizer_state_dict",
        "qwen_scheduler_state_dict",
        "rq_scheduler_state_dict",
        "rank_rng_states",
    }
    if not required_resume_keys.issubset(state):
        raise TigerJointTrainingError(
            "resume joint_state 缺少 optimizer/scheduler 状态"
        )
    validate_rank_rng_states(
        state.get("rank_rng_states"),
        world_size=world_size,
    )
    epoch = state.get("epoch")
    optimizer_step = state.get("optimizer_step")
    examples_consumed = state.get("examples_consumed")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch <= 0
        or isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step <= 0
        or examples_consumed != epoch * train_rows
    ):
        raise TigerJointTrainingError("resume checkpoint 不是完整 epoch 边界")
    return state, manifest


def run(
    args: argparse.Namespace,
    runtime: DistributedRuntime,
) -> dict[str, Any] | None:
    config = load_config(args.config, args)
    if args.max_optimizer_steps is not None and args.max_optimizer_steps <= 0:
        raise TigerJointTrainingError("max_optimizer_steps 必须大于 0")
    if args.resume_from_checkpoint is not None and (
        args.max_optimizer_steps is not None or not args.save_optimizer_state
    ):
        raise TigerJointTrainingError(
            "resume 只支持完整三轮协议，并要求继续保存 optimizer state"
        )
    if runtime.local_rank < 0:
        raise TigerJointTrainingError("local device index 不能为负数")
    effective_global_behavior_batch_size = (
        config.behavior_batch_size
        * config.gradient_accumulation_steps
        * runtime.world_size
    )
    global_catalog_rows_per_step = (
        config.catalog_batch_size
        * config.gradient_accumulation_steps
        * runtime.world_size
    )
    expected_values = (
        ("world_size", config.expected_world_size, runtime.world_size),
        (
            "global behavior batch",
            config.expected_global_behavior_batch_size,
            effective_global_behavior_batch_size,
        ),
        (
            "global catalog rows per step",
            config.expected_global_catalog_rows_per_step,
            global_catalog_rows_per_step,
        ),
    )
    for name, expected, observed in expected_values:
        if expected is not None and observed != expected:
            raise TigerJointTrainingError(f"{name} 要求 {expected}，当前为 {observed}")
    output_dir = resolve(args.output_dir)
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if not output_dir.is_relative_to(outputs_root):
        raise TigerJointTrainingError("output_dir 必须位于仓库 outputs/ 下")
    output_error: list[str | None] = [None]
    if runtime.is_main:
        if output_dir.exists():
            output_error[0] = f"output_dir 已存在：{output_dir}"
        else:
            try:
                output_dir.mkdir(parents=True)
            except OSError as error:
                output_error[0] = str(error)
    if runtime.enabled:
        dist.broadcast_object_list(output_error, src=0)
    if output_error[0] is not None:
        raise TigerJointTrainingError(output_error[0])
    if runtime.enabled:
        dist.barrier()
    signature = config_signature(
        config,
        max_optimizer_steps=args.max_optimizer_steps,
        world_size=runtime.world_size,
        save_checkpoint=args.save_checkpoint,
        save_optimizer_state=args.save_optimizer_state,
    )
    started_at = datetime.now(timezone.utc)
    running_state = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": started_at.isoformat(),
        "config_signature": signature,
        "config": config_payload(config),
        "scope": {
            "max_optimizer_steps": args.max_optimizer_steps,
            "full_three_epoch_training": args.max_optimizer_steps is None,
            "checkpoint_saving_enabled": args.save_checkpoint,
            "sample_data_splits_read": ["train"],
            "valid_samples_read": False,
            "test_samples_read": False,
            "old_sid_artifacts_loaded": [],
            "distributed": runtime.enabled,
            "world_size": runtime.world_size,
            "resume_from_checkpoint": (
                str(resolve(args.resume_from_checkpoint))
                if args.resume_from_checkpoint is not None
                else None
            ),
        },
        "progress": {"optimizer_steps_completed": 0, "examples_consumed": 0},
    }
    if runtime.is_main:
        atomic_json(output_dir / "run_state.json", running_state)

    if not torch.cuda.is_available():
        raise TigerJointTrainingError("联合训练要求宿主 CUDA，不允许回退 CPU")
    if runtime.local_rank >= torch.cuda.device_count():
        raise TigerJointTrainingError("local rank 超出可见 GPU 数量")
    torch.cuda.set_device(runtime.local_rank)
    device = torch.device("cuda", runtime.local_rank)
    seed_everything(config.seed)

    data_manifest, train_path, _valid_path = load_data_manifest(config.data_dir)
    train_rows = int(data_manifest["outputs"]["train.jsonl"]["rows"])
    shards = distribute_line_shards(
        train_path,
        expected_rows=train_rows,
        runtime=runtime,
    )
    shard = shards[runtime.rank]
    microbatches_by_rank = [
        math.ceil(item.rows / config.behavior_batch_size) for item in shards
    ]
    if len(set(microbatches_by_rank)) != 1:
        raise TigerJointTrainingError(
            "各 DDP rank 的行为 microbatch 数不同，无法安全同步"
        )
    optimizer_steps_per_epoch = math.ceil(
        microbatches_by_rank[0] / config.gradient_accumulation_steps
    )
    embedding_store = PoiEmbeddingStore.from_directory(
        config.embedding_dir, load_poi_index=False
    )
    catalog_contract = data_manifest.get("embedding_catalog")
    if not isinstance(catalog_contract, dict) or (
        catalog_contract.get("manifest_signature") != embedding_store.manifest_signature
        or catalog_contract.get("poi_ids_sha256") != embedding_store.poi_ids_sha256
        or catalog_contract.get("shape") != list(embedding_store.shape)
    ):
        raise TigerJointTrainingError("训练数据与冻结 BGE 行序不一致")
    _, preflight_sha256 = load_formal_preflight(
        config.preflight_state,
        model_dir=config.qwen_dir,
        data_dir=config.data_dir,
        embedding_dir=config.embedding_dir,
        data_manifest=data_manifest,
        embedding_store=embedding_store,
    )
    rqvae, initialization_manifest = load_fresh_rqvae_initialization(
        config.rq_initialization_dir,
        embedding_store=embedding_store,
        preflight_state_sha256=preflight_sha256,
        device=device,
    )
    resume_checkpoint_dir = (
        resolve(args.resume_from_checkpoint)
        if args.resume_from_checkpoint is not None
        else None
    )
    resume_state: dict[str, Any] | None = None
    resume_manifest: dict[str, Any] | None = None
    if resume_checkpoint_dir is not None:
        resume_state, resume_manifest = load_resume_checkpoint(
            resume_checkpoint_dir,
            signature=signature,
            train_rows=train_rows,
            world_size=runtime.world_size,
            device=device,
        )
        rqvae.load_state_dict(resume_state["rqvae_state_dict"], strict=True)

    tokenizer, template = load_fresh_tokenizer_and_template(
        config.qwen_dir, project_root=PROJECT_ROOT
    )
    if tokenizer.pad_token_id is None:
        raise TigerJointTrainingError("Tokenizer 缺少 pad_token_id")
    token_ids = load_dynamic_token_ids(tokenizer)
    token_layout = build_sid_token_layout(tokenizer, CODEBOOK_SIZES)

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    base_vocab_size = int(
        AutoConfig.from_pretrained(
            config.qwen_dir,
            local_files_only=True,
            trust_remote_code=False,
        ).vocab_size
    )
    model_source = resume_checkpoint_dir or config.qwen_dir
    if resume_checkpoint_dir is not None:
        checkpoint_tokenizer = AutoTokenizer.from_pretrained(
            resume_checkpoint_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        if checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise TigerJointTrainingError("resume tokenizer 与 fresh 词表不一致")

    causal_lm = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    if resume_checkpoint_dir is None:
        causal_lm.resize_token_embeddings(len(tokenizer))
    if int(causal_lm.config.vocab_size) != len(tokenizer):
        raise TigerJointTrainingError("Qwen fresh 词表尺寸不一致")
    if not bool(causal_lm.config.tie_word_embeddings) or (
        causal_lm.get_input_embeddings().weight.data_ptr()
        != causal_lm.get_output_embeddings().weight.data_ptr()
    ):
        raise TigerJointTrainingError("Qwen 输入输出 Embedding 必须绑定")
    causal_lm.config.use_cache = False
    if config.gradient_checkpointing:
        causal_lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    generator = HuggingFaceCausalLMJointAdapter(causal_lm)
    query_projection = nn.Linear(
        int(causal_lm.config.hidden_size), rqvae.latent_dim
    ).to(device=device, dtype=torch.float32)
    if resume_state is not None:
        query_projection.load_state_dict(
            resume_state["query_projection_state_dict"], strict=True
        )
    generator.train()
    rqvae.train()
    query_projection.train()
    joint_module = JointTigerTrainingModule(
        generator=generator,
        rqvae=rqvae,
        query_projection=query_projection,
        token_layout=token_layout,
        alignment_weight=config.alignment_weight,
        rq_weight=config.rq_weight,
        temperature=config.temperature,
    )
    if runtime.enabled:
        from torch.nn.parallel import DistributedDataParallel

        training_model: nn.Module = DistributedDataParallel(
            joint_module,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    else:
        training_model = joint_module

    qwen_parameters = [
        parameter for parameter in generator.parameters() if parameter.requires_grad
    ]
    projection_parameters = list(query_projection.parameters())
    rq_parameters = list(rqvae.parameters())
    qwen_optimizer = torch.optim.AdamW(
        [
            {"params": qwen_parameters},
            {"params": projection_parameters},
        ],
        lr=config.qwen_learning_rate,
        weight_decay=config.qwen_weight_decay,
    )
    rq_optimizer = torch.optim.Adam(
        rq_parameters,
        lr=config.rq_learning_rate,
        weight_decay=0.0,
    )
    planned_optimizer_steps = (
        args.max_optimizer_steps
        if args.max_optimizer_steps is not None
        else optimizer_steps_per_epoch * config.epochs
    )
    qwen_scheduler, rq_scheduler, warmup_steps = build_joint_schedulers(
        qwen_optimizer,
        rq_optimizer,
        planned_optimizer_steps=planned_optimizer_steps,
        warmup_ratio=config.warmup_ratio,
    )
    start_epoch = 1
    optimizer_step = 0
    local_examples_consumed = 0
    global_examples_consumed = 0
    if resume_state is not None:
        completed_epochs = int(resume_state["epoch"])
        if completed_epochs >= config.epochs or (
            int(resume_state["optimizer_step"])
            != completed_epochs * optimizer_steps_per_epoch
        ):
            raise TigerJointTrainingError(
                "resume checkpoint 的 epoch/optimizer step 与当前数据不一致"
            )
        qwen_optimizer.load_state_dict(resume_state["qwen_optimizer_state_dict"])
        rq_optimizer.load_state_dict(resume_state["rq_optimizer_state_dict"])
        qwen_scheduler.load_state_dict(resume_state["qwen_scheduler_state_dict"])
        rq_scheduler.load_state_dict(resume_state["rq_scheduler_state_dict"])
        restore_rank_rng_state(
            resume_state.get("rank_rng_states"),
            rank=runtime.rank,
            world_size=runtime.world_size,
            device=device,
        )
        start_epoch = completed_epochs + 1
        optimizer_step = int(resume_state["optimizer_step"])
        global_examples_consumed = int(resume_state["examples_consumed"])
        local_examples_consumed = sum(
            shards[runtime.rank].rows for _ in range(completed_epochs)
        )

    probe_rng = np.random.default_rng(config.seed + 10_000)
    probe_rows = np.sort(
        probe_rng.choice(
            embedding_store.poi_count,
            size=min(config.probe_rows, embedding_store.poi_count),
            replace=False,
        )
    ).astype(np.int64, copy=False)
    previous_probe_codes: torch.Tensor | None = None
    initial_probe: dict[str, Any] | None = None
    if runtime.is_main:
        previous_probe_codes = encode_probe_codes(
            rqvae, embedding_store, probe_rows, device=device
        )
        initial_probe = probe_metrics(
            previous_probe_codes,
            None,
            codebook_sizes=CODEBOOK_SIZES,
        )
    if runtime.enabled:
        dist.barrier()

    qwen_optimizer.zero_grad(set_to_none=True)
    rq_optimizer.zero_grad(set_to_none=True)
    step_trace: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []
    stop_requested = False
    training_started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, config.epochs + 1):
        catalog_rng = np.random.default_rng(
            config.seed + 20_000 + 10_000 * epoch + runtime.rank
        )
        accumulator = MetricAccumulator()
        accumulation_count = 0
        group_behavior_rows = 0
        group_catalog_rows = 0
        local_epoch_examples = 0
        global_epoch_examples = 0
        for microbatch_index, examples in enumerate(
            iter_behavior_batches(
                train_path,
                shard=shard,
                tokenizer=tokenizer,
                template=template,
                token_ids=token_ids,
                config=config,
                epoch=epoch,
            )
        ):
            local_batch_examples = len(examples)
            if accumulation_count == 0:
                group_behavior_rows, group_catalog_rows = accumulation_group_sizes(
                    shards,
                    first_microbatch_index=microbatch_index,
                    microbatches_per_epoch=microbatches_by_rank[0],
                    per_rank_behavior_batch_size=(config.behavior_batch_size),
                    per_rank_catalog_batch_size=config.catalog_batch_size,
                    accumulation_steps=(config.gradient_accumulation_steps),
                )
            global_batch_examples = global_behavior_batch_size(
                shards,
                microbatch_index=microbatch_index,
                per_rank_batch_size=config.behavior_batch_size,
            )
            next_local_epoch_examples = local_epoch_examples + local_batch_examples
            is_epoch_end = next_local_epoch_examples == shard.rows
            will_step = (
                accumulation_count + 1 >= config.gradient_accumulation_steps
                or is_epoch_end
            )
            catalog_rows = catalog_rng.choice(
                embedding_store.poi_count,
                size=config.catalog_batch_size,
                replace=False,
            ).astype(np.int64, copy=False)
            batch = collate_dynamic_examples(
                examples,
                embedding_store=embedding_store,
                catalog_rows=catalog_rows,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
                cutoff_len=config.cutoff_len,
            )
            synchronization = (
                training_model.no_sync()
                if runtime.enabled and not will_step
                else nullcontext()
            )
            with synchronization:
                output = training_model(batch)
                if not isinstance(output, JointTigerLossOutput):
                    raise TigerJointTrainingError(
                        "联合训练模块必须返回 JointTigerLossOutput"
                    )
                behavior_scale = (
                    local_batch_examples * runtime.world_size / group_behavior_rows
                )
                catalog_scale = (
                    config.catalog_batch_size * runtime.world_size / group_catalog_rows
                )
                behavior_objective = (
                    output.generation_loss
                    + config.alignment_weight * output.alignment_loss
                )
                backward_objective = (
                    behavior_scale * behavior_objective
                    + catalog_scale * config.rq_weight * output.rq_loss
                )
                backward_objective.backward()
            accumulator.add(
                output,
                batch.template.target_poi_rows,
                catalog_samples=int(batch.catalog_embeddings.shape[0]),
            )
            accumulation_count += 1
            local_examples_consumed += local_batch_examples
            global_examples_consumed += global_batch_examples
            local_epoch_examples = next_local_epoch_examples
            global_epoch_examples += global_batch_examples
            del output, batch

            if not will_step:
                continue
            gradient_metrics = {
                "qwen": gradient_norm(qwen_parameters),
                "query_projection": gradient_norm(projection_parameters),
                "rqvae": gradient_norm(rq_parameters),
                "rq_encoder": gradient_norm(rqvae.encoder.parameters()),
                "rq_decoder": gradient_norm(rqvae.decoder.parameters()),
                "rq_codebooks": gradient_norm(rqvae.quantizer.codebooks.parameters()),
            }
            if not all(np.isfinite(value) for value in gradient_metrics.values()):
                raise TigerJointTrainingError("联合训练梯度出现 NaN 或 Inf")
            torch.nn.utils.clip_grad_norm_(qwen_parameters, config.gradient_clip_norm)
            torch.nn.utils.clip_grad_norm_(
                projection_parameters, config.gradient_clip_norm
            )
            torch.nn.utils.clip_grad_norm_(rq_parameters, config.gradient_clip_norm)
            qwen_optimizer.step()
            rq_optimizer.step()
            qwen_scheduler.step()
            rq_scheduler.step()
            qwen_optimizer.zero_grad(set_to_none=True)
            rq_optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            metric_summary = accumulator.summary(
                alignment_weight=config.alignment_weight,
                rq_weight=config.rq_weight,
            )
            step_result: dict[str, Any] = {
                "optimizer_step": optimizer_step,
                "epoch": epoch,
                "examples_consumed": global_examples_consumed,
                "epoch_examples_consumed": global_epoch_examples,
                "rank_examples_consumed": local_examples_consumed,
                "losses_and_accuracy": metric_summary,
                "gradient_norms_before_clipping": gradient_metrics,
                "qwen_learning_rate": float(qwen_scheduler.get_last_lr()[0]),
                "rq_learning_rate": float(rq_scheduler.get_last_lr()[0]),
            }
            if (
                optimizer_step == 1
                or optimizer_step % config.probe_every_optimizer_steps == 0
                or optimizer_step == planned_optimizer_steps
            ):
                if runtime.is_main:
                    if previous_probe_codes is None:
                        raise TigerJointTrainingError("主进程缺少初始 probe codes")
                    current_probe_codes = encode_probe_codes(
                        rqvae, embedding_store, probe_rows, device=device
                    )
                    step_result["probe"] = probe_metrics(
                        current_probe_codes,
                        previous_probe_codes,
                        codebook_sizes=CODEBOOK_SIZES,
                    )
                    previous_probe_codes = current_probe_codes
                if runtime.enabled:
                    dist.barrier()
            if runtime.is_main:
                step_trace.append(step_result)
                running_state["progress"] = {
                    "optimizer_steps_completed": optimizer_step,
                    "examples_consumed": global_examples_consumed,
                    "epoch": epoch,
                    "latest": step_result,
                }
                atomic_json(output_dir / "run_state.json", running_state)
            if runtime.is_main and (
                optimizer_step == 1
                or optimizer_step % config.log_every_optimizer_steps == 0
            ):
                summary = step_result["losses_and_accuracy"]
                print(
                    " ".join(
                        (
                            f"step={optimizer_step}/{planned_optimizer_steps}",
                            f"epoch={epoch}",
                            f"examples={global_examples_consumed}",
                            f"gen={summary['generation_loss']:.6f}",
                            f"align={summary['alignment_loss']:.6f}",
                            f"rq={summary['rq_loss']:.6f}",
                            f"sid_acc={summary['teacher_forced_sid_token_accuracy']:.4f}",
                        )
                    ),
                    flush=True,
                )
            accumulator = MetricAccumulator()
            accumulation_count = 0
            if optimizer_step >= planned_optimizer_steps:
                stop_requested = True
                break

        if args.save_checkpoint and (
            stop_requested or local_epoch_examples == shard.rows
        ):
            rank_rng_states = (
                gather_rank_rng_states(runtime, device)
                if args.save_optimizer_state
                else None
            )
            if runtime.enabled:
                dist.barrier()
            if runtime.is_main:
                checkpoint_path = save_checkpoint(
                    output_dir=output_dir,
                    optimizer_step=optimizer_step,
                    epoch=epoch,
                    examples_consumed=global_examples_consumed,
                    causal_lm=causal_lm,
                    tokenizer=tokenizer,
                    rqvae=rqvae,
                    query_projection=query_projection,
                    qwen_optimizer=qwen_optimizer,
                    rq_optimizer=rq_optimizer,
                    qwen_scheduler=qwen_scheduler,
                    rq_scheduler=rq_scheduler,
                    signature=signature,
                    save_optimizer_state=args.save_optimizer_state,
                    rank_rng_states=rank_rng_states,
                    world_size=runtime.world_size,
                    train_rows=train_rows,
                )
                checkpoint_paths.append(str(checkpoint_path))
            if runtime.enabled:
                dist.barrier()
        if stop_requested:
            break

    if runtime.enabled:
        dist.barrier()
    final_probe: dict[str, Any] | None = None
    if runtime.is_main:
        if previous_probe_codes is None:
            raise TigerJointTrainingError("主进程缺少最终 probe 基线")
        final_probe_codes = encode_probe_codes(
            rqvae, embedding_store, probe_rows, device=device
        )
        final_probe = probe_metrics(
            final_probe_codes,
            previous_probe_codes,
            codebook_sizes=CODEBOOK_SIZES,
        )
    local_runtime = {
        "rank": runtime.rank,
        "local_rank": runtime.local_rank,
        "gpu": torch.cuda.get_device_name(device),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    rank_runtimes: list[Any] = [None] * runtime.world_size
    if runtime.enabled:
        dist.all_gather_object(rank_runtimes, local_runtime)
    else:
        rank_runtimes[0] = local_runtime
    completed_full_training = (
        args.max_optimizer_steps is None
        and optimizer_step == optimizer_steps_per_epoch * config.epochs
        and global_examples_consumed == train_rows * config.epochs
    )
    if not runtime.is_main:
        return None
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config_signature": signature,
        "config": config_payload(config),
        "scope": {
            "max_optimizer_steps": args.max_optimizer_steps,
            "full_three_epoch_training": args.max_optimizer_steps is None,
            "completed_full_three_epoch_training": completed_full_training,
            "checkpoint_saving_enabled": args.save_checkpoint,
            "sample_data_splits_read": ["train"],
            "valid_samples_read": False,
            "test_samples_read": False,
            "old_sid_artifacts_loaded": [],
            "distributed": runtime.enabled,
            "world_size": runtime.world_size,
            "resume_from_checkpoint": (
                str(resume_checkpoint_dir)
                if resume_checkpoint_dir is not None
                else None
            ),
        },
        "inputs": {
            "data_manifest_sha256": sha256_file(config.data_dir / "manifest.json"),
            "data_build_fingerprint": data_manifest.get("build_fingerprint"),
            "train_rows": train_rows,
            "train_line_shards": [vars(item) for item in shards],
            "preflight_state_sha256": preflight_sha256,
            "embedding_manifest_signature": embedding_store.manifest_signature,
            "embedding_poi_ids_sha256": embedding_store.poi_ids_sha256,
            "rq_initialization_manifest_sha256": sha256_file(
                config.rq_initialization_dir / INITIALIZATION_MANIFEST_NAME
            ),
            "resume_checkpoint_manifest_sha256": (
                sha256_file(resume_checkpoint_dir / "checkpoint_manifest.json")
                if resume_checkpoint_dir is not None
                else None
            ),
            "resume_checkpoint_epoch": (
                resume_manifest.get("epoch") if resume_manifest is not None else None
            ),
            "old_sid_artifacts_loaded": [],
        },
        "model": {
            "qwen_architecture": causal_lm.config.architectures,
            "qwen_base_vocab_size": base_vocab_size,
            "qwen_fresh_vocab_size": int(causal_lm.config.vocab_size),
            "qwen_dtype": str(next(causal_lm.parameters()).dtype),
            "qwen_gradient_checkpointing": config.gradient_checkpointing,
            "rqvae_initial_sha256": initialization_manifest["checkpoint"][
                "model_sha256"
            ],
            "rqvae_final_sha256": module_sha256(rqvae),
        },
        "optimization": {
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "planned_optimizer_steps": planned_optimizer_steps,
            "optimizer_steps_completed": optimizer_step,
            "examples_consumed": global_examples_consumed,
            "effective_behavior_batch_size": (
                config.behavior_batch_size
                * config.gradient_accumulation_steps
                * runtime.world_size
            ),
            "catalog_rows_per_full_optimizer_step": (
                config.catalog_batch_size
                * config.gradient_accumulation_steps
                * runtime.world_size
            ),
            "learning_rate_schedule": "warmup_then_cosine_decay",
            "warmup_steps": warmup_steps,
            "qwen_peak_learning_rate": config.qwen_learning_rate,
            "rq_peak_learning_rate": config.rq_learning_rate,
            "update_schedule": "qwen_projection_and_rqvae_step_together_at_each_accumulation_boundary",
        },
        "probe": {
            "rows_sha256": hashlib.sha256(probe_rows.tobytes()).hexdigest(),
            "initial": initial_probe,
            "final": final_probe,
        },
        "step_trace": step_trace,
        "checkpoints": checkpoint_paths,
        "git": git_state(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "world_size": runtime.world_size,
            "ranks": rank_runtimes,
            "training_seconds": time.perf_counter() - training_started,
        },
    }
    atomic_json(output_dir / "run_state.json", result)
    return result


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    runtime: DistributedRuntime | None = None
    try:
        runtime = initialize_distributed(args)
        result = run(args, runtime)
    except Exception as error:
        state_path = output_dir / "run_state.json"
        is_main = runtime is None or runtime.is_main
        if is_main and state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            if isinstance(state, dict) and state.get("status") == "running":
                state.update(
                    {
                        "status": "failed",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "valid_samples_read": False,
                        "test_samples_read": False,
                    }
                )
                atomic_json(state_path, state)
        if is_main:
            print(f"TIGER-Joint 训练失败：{error}", file=sys.stderr)
        return 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if runtime is not None and runtime.is_main:
        if result is None:
            raise TigerJointTrainingError("主进程没有生成训练结果")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

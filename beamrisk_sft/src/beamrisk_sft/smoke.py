"""End-to-end trainer contract smoke for the BeamRisk auxiliary gradient."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset, disable_progress_bars
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

from llamafactory.hparams import FinetuningArguments

from .config import BeamRiskConfig, RiskConfig
from .errors import BeamRiskError
from .io import atomic_write_json, read_json
from .risk_store import RiskPairStore
from .schema import RISK_PAIR_SCHEMA_VERSION
from .trainer import BeamRiskTrainer


class _SyntheticProcessor:
    """The minimal processing contract exercised by CustomSeq2SeqTrainer."""

    pad_token_id = 0
    model_input_names = ["input_ids"]

    def save_pretrained(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def _risk_record(index: int) -> dict[str, Any]:
    positive_first = 10 + index % 8
    positive_second = 20 + (index * 3) % 8
    negative_first = 30 + (index * 5) % 8
    negative_second = 40 + (index * 7) % 8
    first_prune = index % 2 == 0
    return {
        "schema_version": RISK_PAIR_SCHEMA_VERSION,
        "sample_id": f"{index:064x}",
        "order_id": f"smoke-order-{index}",
        "searchid": f"smoke-search-{index}",
        "target_poi_id": f"smoke-poi-{index}",
        "risk_type": "first_prune" if first_prune else "final_rank",
        "first_prune_depth": 2 if first_prune else None,
        "prompt_token_ids": [1, 2 + index % 5, 3],
        "positive_token_ids": [positive_first, positive_second],
        "negative_token_ids": [negative_first, negative_second],
        "reference_positive_score": -3.0,
        "reference_negative_score": -2.0,
        "reference_margin": 1.0,
        "negative_structure_valid": True,
        "negative_catalog_expandable": False,
        "negative_poi_id": None,
        "strict_duplicate_filtered": False,
    }


def _main_record(index: int) -> dict[str, list[int]]:
    target_first = 10 + index % 8
    target_second = 20 + (index * 3) % 8
    return {
        "input_ids": [1, 2 + index % 5, 3, target_first, target_second],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [-100, -100, -100, target_first, target_second],
    }


def _probe(model: torch.nn.Module) -> torch.Tensor:
    values = [parameter.detach().float().reshape(-1)[:8] for parameter in model.parameters()]
    return torch.cat(values[:8]).contiguous()


def _distributed_reports(local: dict[str, Any]) -> list[dict[str, Any]]:
    if not dist.is_initialized():
        return [local]
    reports: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(reports, local)
    if any(report is None for report in reports):
        raise BeamRiskError("Trainer smoke 未收齐所有 rank 报告")
    return [report for report in reports if report is not None]


def run_trainer_smoke(
    work_dir: Path,
    *,
    expected_world_size: int,
    device: str,
) -> dict[str, Any] | None:
    """Run CE + BeamRisk through the real Trainer and optimizer for four steps."""

    if device not in ("cpu", "cuda"):
        raise BeamRiskError(f"Trainer smoke device 无效：{device}")
    try:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise BeamRiskError("Trainer smoke 的 torchrun rank 环境无效") from error
    if world_size != expected_world_size:
        raise BeamRiskError(
            f"Trainer smoke 要求 {expected_world_size} 进程，实际 {world_size}"
        )
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
            raise BeamRiskError("CUDA Trainer smoke 要求每个进程各绑定一张可见 GPU")
        torch.cuda.set_device(local_rank)

    disable_progress_bars()
    work_dir = work_dir.resolve()
    rank_dir = work_dir / f"rank_{rank:03d}"
    if rank_dir.exists():
        shutil.rmtree(rank_dir)
    rank_dir.mkdir(parents=True, exist_ok=False)
    report_path = work_dir / "report.json"
    if rank == 0:
        report_path.unlink(missing_ok=True)

    risk_dataset = rank_dir / "risk_dataset"
    Dataset.from_list([_risk_record(index) for index in range(37)]).save_to_disk(
        str(risk_dataset)
    )
    # Thirty-two micro-batches per rank cover four optimizer steps with the
    # formal gradient_accumulation_steps=8.  Different rows across ranks make the
    # final parameter-equality check meaningful for DDP synchronization.
    main_rows = max(32 * world_size, 32)
    train_dataset = Dataset.from_list(
        [_main_record(index) for index in range(main_rows)]
    )
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            pad_token_id=0,
            use_sliding_window=False,
        )
    )
    before = _probe(model).cpu()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(rank_dir / "trainer_output"),
        do_train=True,
        use_cpu=device == "cpu",
        bf16=device == "cuda",
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=4,
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        seed=31_415,
        data_seed=31_415,
    )
    trainer: BeamRiskTrainer | None = None
    try:
        trainer = BeamRiskTrainer(
            model=model,
            args=training_args,
            finetuning_args=FinetuningArguments(finetuning_type="full"),
            processor=None,
            gen_kwargs={},
            tokenizer=_SyntheticProcessor(),
            data_collator=default_data_collator,
            train_dataset=train_dataset,
            risk_dataset_path=risk_dataset,
            risk_dataset_rows=37,
            risk_config=RiskConfig(
                weight=0.2,
                temperature=1.0,
                margin=0.0,
                interval_optimizer_steps=4,
                global_batch_size=2 * world_size,
                per_device_batch_size=2,
                per_device_micro_batch_size=1,
            ),
            risk_seed=42,
        )
        result = trainer.train()
        summary = trainer.beamrisk_summary()
        after = _probe(model)
        local = {
            "rank": rank,
            "global_step": int(trainer.state.global_step),
            "accelerator_gradient_accumulation_steps": int(
                trainer.accelerator.gradient_accumulation_steps
            ),
            "risk_events": int(summary["events"]),
            "local_risk_pairs": int(summary["local_pairs"]),
            "weights_changed": bool(torch.any(after.cpu() != before).item()),
            "train_loss": float(result.metrics["train_loss"]),
        }
        gathered_parameters = [torch.empty_like(after) for _ in range(world_size)]
        if dist.is_initialized():
            dist.all_gather(gathered_parameters, after)
        else:
            gathered_parameters[0].copy_(after)
        max_rank_parameter_diff = max(
            float((value - gathered_parameters[0]).abs().max().item())
            for value in gathered_parameters
        )
        reports = _distributed_reports(local)
        failures: list[str] = []
        expected_ranks = list(range(world_size))
        if sorted(int(report["rank"]) for report in reports) != expected_ranks:
            failures.append("rank 报告不完整")
        for report in reports:
            if report["global_step"] != 4:
                failures.append(f"rank{report['rank']} global_step != 4")
            if report["accelerator_gradient_accumulation_steps"] != 1:
                failures.append(f"rank{report['rank']} Accelerator 累积契约变化")
            if report["risk_events"] != 1 or report["local_risk_pairs"] != 2:
                failures.append(f"rank{report['rank']} risk 注入次数/样本数错误")
            if not report["weights_changed"]:
                failures.append(f"rank{report['rank']} 参数未更新")
        if max_rank_parameter_diff != 0.0:
            failures.append(f"DDP rank 参数不同步：max_diff={max_rank_parameter_diff}")
        if failures:
            raise BeamRiskError("Trainer smoke 失败：" + "；".join(failures))
        payload = {
            "status": "passed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "precision": "bf16" if device == "cuda" else "fp32",
            "world_size": world_size,
            "gradient_accumulation_steps": 8,
            "optimizer_steps": 4,
            "risk_interval_optimizer_steps": 4,
            "expected_risk_events_per_rank": 1,
            "max_rank_parameter_diff": max_rank_parameter_diff,
            "rank_reports": reports,
        }
        if rank == 0:
            atomic_write_json(report_path, payload)
            return payload
        return None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_formal_risk_batch_smoke(
    config: BeamRiskConfig,
    *,
    reference_epoch: int,
    work_dir: Path,
    device: str,
) -> dict[str, Any] | None:
    """Score and backpropagate the next stage's exact first formal risk batch."""

    if device not in ("cpu", "cuda"):
        raise BeamRiskError(f"Formal risk smoke device 无效：{device}")
    try:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise BeamRiskError("Formal risk smoke 的 torchrun rank 环境无效") from error
    if world_size != config.training.world_size:
        raise BeamRiskError(
            f"Formal risk smoke 要求 {config.training.world_size} 进程，实际 {world_size}"
        )
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
            raise BeamRiskError("CUDA formal risk smoke 要求每进程各绑定一张可见 GPU")
        torch.cuda.set_device(local_rank)
        torch_device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        torch_device = torch.device("cpu")
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    output_dir = config.mining_dir(reference_epoch)
    manifest = read_json(output_dir / "manifest.json", name="Mining manifest")
    rows = manifest.get("pairs")
    if not isinstance(rows, int) or rows <= 0:
        raise BeamRiskError("Formal risk smoke 的 mining manifest 行数无效")
    target_epoch = reference_epoch + 1
    first_epoch_step = (
        config.training.expected_optimizer_steps_per_epoch * reference_epoch + 1
    )
    interval = config.risk.interval_optimizer_steps
    first_risk_step = (first_epoch_step + interval - 1) // interval * interval
    store = RiskPairStore(
        output_dir / "dataset",
        seed=config.training.seed + target_epoch * 10_000,
        expected_rows=rows,
    )
    batch = store.batch_for_optimizer_step(
        optimizer_step=first_risk_step,
        interval=interval,
        global_batch_size=config.risk.global_batch_size,
        rank=rank,
        world_size=world_size,
    )
    if len(batch) != config.risk.per_device_batch_size:
        raise BeamRiskError("Formal risk smoke 的 per-rank batch 大小不一致")

    model_config = read_json(config.paths.model / "config.json", name="Model config")
    vocab_size = model_config.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise BeamRiskError("Formal risk smoke 的模型词表大小无效")
    generation_config = read_json(
        config.paths.model / "generation_config.json",
        name="Generation config",
    )
    pad_token_id = generation_config.get("pad_token_id")
    if (
        not isinstance(pad_token_id, int)
        or isinstance(pad_token_id, bool)
        or not 0 <= pad_token_id < vocab_size
    ):
        raise BeamRiskError("Formal risk smoke 的 pad_token_id 无效")
    torch.manual_seed(27_182)
    if device == "cuda":
        torch.cuda.manual_seed_all(27_182)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=config.training.cutoff_len,
            pad_token_id=pad_token_id,
            use_sliding_window=False,
        )
    ).to(torch_device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if device == "cuda" else None,
        output_device=local_rank if device == "cuda" else None,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = 0.0
    from .loss import score_and_compute_beam_risk_loss

    for start in range(0, len(batch), config.risk.per_device_micro_batch_size):
        end = min(start + config.risk.per_device_micro_batch_size, len(batch))
        chunk = batch.slice(start, end)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )
        with autocast_context:
            output = score_and_compute_beam_risk_loss(
                ddp_model,
                chunk.prompts,
                chunk.positives,
                chunk.negatives,
                pad_token_id=pad_token_id,
                temperature=config.risk.temperature,
                margin=config.risk.margin,
                device=torch_device,
            )
            weighted_loss = (
                config.risk.weight * len(chunk) / len(batch) * output.loss
            )
        weighted_loss.backward()
        local_loss_sum += float(output.per_pair_loss.detach().float().sum().item())
    gradients = [parameter.grad for parameter in ddp_model.parameters()]
    if any(gradient is None for gradient in gradients):
        raise BeamRiskError("Formal risk smoke 存在未参与风险反向的模型参数")
    if any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
        raise BeamRiskError("Formal risk smoke 梯度出现 NaN/Inf")
    optimizer.step()

    after = _probe(ddp_model)
    gathered_parameters = [torch.empty_like(after) for _ in range(world_size)]
    dist.all_gather(gathered_parameters, after)
    max_rank_parameter_diff = max(
        float((value - gathered_parameters[0]).abs().max().item())
        for value in gathered_parameters
    )
    local = {
        "rank": rank,
        "pairs": len(batch),
        "risk_type_counts": dict(sorted(Counter(batch.risk_types).items())),
        "max_prompt_plus_path": max(
            len(prompt) + len(positive)
            for prompt, positive in zip(batch.prompts, batch.positives)
        ),
        "mean_pair_loss": local_loss_sum / len(batch),
    }
    reports = _distributed_reports(local)
    if sum(int(report["pairs"]) for report in reports) != config.risk.global_batch_size:
        raise BeamRiskError("Formal risk smoke 未覆盖完整 global risk batch")
    if max_rank_parameter_diff != 0.0:
        raise BeamRiskError(
            "Formal risk smoke 的 DDP 参数不同步："
            f"max_diff={max_rank_parameter_diff}"
        )
    payload = {
        "status": "passed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "precision": "bf16" if device == "cuda" else "fp32",
        "reference_epoch": reference_epoch,
        "target_epoch": target_epoch,
        "optimizer_step": first_risk_step,
        "world_size": world_size,
        "global_risk_batch": config.risk.global_batch_size,
        "per_device_risk_batch": config.risk.per_device_batch_size,
        "per_device_micro_batch": config.risk.per_device_micro_batch_size,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "risk_pairs_sha256": manifest.get("risk_pairs_sha256"),
        "max_rank_parameter_diff": max_rank_parameter_diff,
        "rank_reports": reports,
    }
    if rank == 0:
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(work_dir / f"epoch_{reference_epoch}.json", payload)
        return payload
    return None


def run_formal_risk_batch_smoke(
    config: BeamRiskConfig,
    *,
    reference_epoch: int,
    work_dir: Path,
    device: str,
) -> dict[str, Any] | None:
    """Run formal-batch smoke and always release its dedicated process group."""

    process_group_already_initialized = dist.is_initialized()
    try:
        return _run_formal_risk_batch_smoke(
            config,
            reference_epoch=reference_epoch,
            work_dir=work_dir,
            device=device,
        )
    finally:
        if not process_group_already_initialized and dist.is_initialized():
            dist.destroy_process_group()

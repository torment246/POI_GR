"""Three-stage full-SFT workflow with one continuous optimizer schedule."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch.distributed as dist
import yaml
from transformers import TrainerCallback

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.callbacks import LogCallback, ReporterCallback

from .checkpoints import STAGE_SCHEMA_VERSION, validate_training_checkpoint
from .config import BeamRiskConfig
from .errors import BeamRiskError
from .io import (
    atomic_write_json,
    atomic_write_text,
    implementation_sha256,
    read_json,
    sha256_file,
)
from .trainer import BeamRiskTrainer
from .validation import validate_risk_pairs


class StopAfterEpochCallback(TrainerCallback):
    def __init__(self, target_epoch: int) -> None:
        self.target_epoch = target_epoch

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if state.epoch is not None and float(state.epoch) >= self.target_epoch - 1e-6:
            control.should_training_stop = True
        return control


def _load_baseline(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BeamRiskError(f"TIGER baseline config 读取失败：{path}") from error
    if not isinstance(value, dict):
        raise BeamRiskError("TIGER baseline config 必须是 mapping")
    return value


def validate_frozen_baseline(config: BeamRiskConfig, baseline: Mapping[str, Any]) -> None:
    expected = {
        "stage": "sft",
        "finetuning_type": "full",
        "template": "qwen3_nothink",
        "enable_thinking": False,
        "cutoff_len": 512,
        "packing": True,
        "train_on_prompt": False,
        "num_train_epochs": 3.0,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": 5.0e-5,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "seed": config.training.seed,
        "data_seed": config.training.data_seed,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "save_total_limit": 3,
        "load_best_model_at_end": False,
        "predict_with_generate": False,
    }
    mismatches = {
        key: (baseline.get(key), expected_value)
        for key, expected_value in expected.items()
        if baseline.get(key) != expected_value
    }
    if mismatches:
        raise BeamRiskError(f"TIGER frozen baseline 配置漂移：{mismatches}")
    if Path(str(baseline.get("model_name_or_path"))).name != config.paths.model.name:
        raise BeamRiskError("TIGER baseline 初始模型与 BeamRisk 配置不一致")


def build_lf_stage_config(
    config: BeamRiskConfig,
    *,
    target_epoch: int,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    if target_epoch not in (1, 2, 3):
        raise BeamRiskError("训练 target_epoch 只允许 1/2/3")
    if (target_epoch == 1) != (resume_checkpoint is None):
        raise BeamRiskError("Epoch 1 不得恢复；Epoch 2/3 必须恢复前一轮完整状态")
    baseline = _load_baseline(config.paths.baseline_sft_config)
    validate_frozen_baseline(config, baseline)
    result = dict(baseline)
    result.update(
        {
            "model_name_or_path": str(config.paths.model),
            "dataset_dir": str((config.root / "configs" / "sft").resolve()),
            "tokenized_path": str(config.paths.tokenized_cache),
            "output_dir": str(config.paths.sft_output_dir),
            "logging_dir": str(config.paths.sft_output_dir / "tensorboard"),
            "overwrite_output_dir": False,
            "num_train_epochs": float(config.training.epochs),
            "save_total_limit": 3,
        }
    )
    result.pop("resume_from_checkpoint", None)
    if resume_checkpoint is not None:
        result["resume_from_checkpoint"] = str(resume_checkpoint.resolve())
    return result


def _checkpoint_for_epoch(
    output_dir: Path,
    epoch: int,
    *,
    expected_step: int,
    expected_total_steps: int,
    world_size: int,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{expected_step}"
    validate_training_checkpoint(
        checkpoint,
        output_dir=output_dir,
        expected_epoch=epoch,
        expected_step=expected_step,
        expected_total_steps=expected_total_steps,
        world_size=world_size,
        deep_archive_check=False,
    )
    return checkpoint.resolve()


def _validate_resume_checkpoint(
    config: BeamRiskConfig,
    path: Path,
    expected_epoch: int,
) -> None:
    expected_step = (
        config.training.expected_optimizer_steps_per_epoch * expected_epoch
    )
    validate_training_checkpoint(
        path,
        output_dir=config.paths.sft_output_dir,
        expected_epoch=expected_epoch,
        expected_step=expected_step,
        expected_total_steps=(
            config.training.expected_optimizer_steps_per_epoch
            * config.training.epochs
        ),
        world_size=config.training.world_size,
        deep_archive_check=False,
    )


def run_training_stage(
    config: BeamRiskConfig,
    *,
    target_epoch: int,
    resume_checkpoint: Path | None,
) -> Path:
    if target_epoch == 1:
        existing = list(config.paths.sft_output_dir.glob("checkpoint-*"))
        if existing:
            raise BeamRiskError("Epoch 1 必须从初始模型开始，输出目录不得已有 checkpoint")
    else:
        if resume_checkpoint is None:
            raise BeamRiskError("Epoch 2/3 缺少 resume checkpoint")
        _validate_resume_checkpoint(
            config,
            resume_checkpoint,
            expected_epoch=target_epoch - 1,
        )

    risk_dataset_path: Path | None = None
    risk_dataset_rows = 0
    risk_manifest: dict[str, Any] | None = None
    if target_epoch >= 2:
        validation = validate_risk_pairs(
            config,
            reference_epoch=target_epoch - 1,
        )
        risk_dataset_path = config.mining_dir(target_epoch - 1) / "dataset"
        risk_dataset_rows = int(validation["rows"])
        risk_manifest = read_json(
            config.mining_dir(target_epoch - 1) / "manifest.json",
            name="Risk mining manifest",
        )
        reference_checkpoint = Path(
            str(risk_manifest.get("reference_checkpoint", {}).get("path", ""))
        ).resolve()
        if reference_checkpoint != resume_checkpoint.resolve():
            raise BeamRiskError("风险 pair 的参考 checkpoint 与训练恢复点不一致")

    lf_config = build_lf_stage_config(
        config,
        target_epoch=target_epoch,
        resume_checkpoint=resume_checkpoint,
    )
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(lf_config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="sft",
        **tokenizer_module,
    )
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=(
            IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        ),
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    callbacks = [
        LogCallback(),
        StopAfterEpochCallback(target_epoch),
        ReporterCallback(model_args, data_args, finetuning_args, generating_args),
    ]
    trainer = BeamRiskTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs={},
        **dataset_module,
        **tokenizer_module,
        risk_dataset_path=risk_dataset_path,
        risk_dataset_rows=risk_dataset_rows,
        risk_config=config.risk if target_epoch >= 2 else None,
        risk_seed=config.training.seed + target_epoch * 10_000,
    )
    train_result = trainer.train(
        resume_from_checkpoint=(str(resume_checkpoint) if resume_checkpoint else None)
    )
    trainer.save_model()
    trainer.log_metrics(f"stage_epoch_{target_epoch}", train_result.metrics)
    trainer.save_metrics(f"stage_epoch_{target_epoch}", train_result.metrics)
    trainer.save_state()
    # Every rank writes its own rng_state_<rank>.pth.  Wait until all four files
    # are durable before any rank validates the epoch boundary.
    if dist.is_initialized():
        dist.barrier()
    checkpoint = _checkpoint_for_epoch(
        config.paths.sft_output_dir,
        target_epoch,
        expected_step=(
            config.training.expected_optimizer_steps_per_epoch * target_epoch
        ),
        expected_total_steps=(
            config.training.expected_optimizer_steps_per_epoch
            * config.training.epochs
        ),
        world_size=config.training.world_size,
    )
    if trainer.is_world_process_zero():
        stages_dir = config.paths.sft_output_dir / "stages"
        stages_dir.mkdir(parents=True, exist_ok=True)
        marker = stages_dir / f"epoch_{target_epoch}_checkpoint.txt"
        state = read_json(checkpoint / "trainer_state.json", name="Epoch trainer state")
        stage_manifest = {
            "schema_version": STAGE_SCHEMA_VERSION,
            "status": "completed",
            "target_epoch": target_epoch,
            "initial_model": str(config.paths.model),
            "resume_checkpoint": str(resume_checkpoint.resolve()) if resume_checkpoint else None,
            "checkpoint": str(checkpoint),
            "global_step": state.get("global_step"),
            "trainer_epoch": state.get("epoch"),
            "risk_enabled": target_epoch >= 2,
            "risk_reference_epoch": target_epoch - 1 if target_epoch >= 2 else None,
            "risk_pairs_sha256": risk_manifest.get("risk_pairs_sha256") if risk_manifest else None,
            "beamrisk_summary": trainer.beamrisk_summary(),
            "lf_stage_config": lf_config,
            "baseline_config_sha256": sha256_file(config.paths.baseline_sft_config),
            "implementation_sha256": implementation_sha256(config.root),
        }
        atomic_write_json(stages_dir / f"epoch_{target_epoch}.json", stage_manifest)
        # Commit marker is last so --resume never treats a half-written stage
        # record as completed.  The recovery utility also handles old ordering.
        atomic_write_text(marker, str(checkpoint) + "\n")
    if dist.is_initialized():
        dist.barrier()
    return checkpoint


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

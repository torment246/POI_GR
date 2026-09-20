#!/usr/bin/env python3
"""Fail-fast contract checks before allocating a formal four-GPU job."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401

import transformers
from transformers import Trainer

from beamrisk_sft.checkpoints import (
    audit_checkpoint_chain,
    expected_rng_state_names,
)
from beamrisk_sft.config import load_config
from beamrisk_sft.errors import BeamRiskError
from beamrisk_sft.io import read_json
from beamrisk_sft.workflow import build_lf_stage_config, validate_frozen_baseline


def _validate_transformers_rng_contract() -> dict[str, object]:
    try:
        save_source = inspect.getsource(Trainer._save_rng_state)
        load_source = inspect.getsource(Trainer._load_rng_state)
    except (OSError, TypeError) as error:
        raise BeamRiskError("无法检查 Transformers Trainer RNG 保存契约") from error
    expected_fragments = (
        'f"rng_state_{self.args.process_index}.pth"',
        'f"rng_state_{process_index}.pth"',
    )
    if expected_fragments[0] not in save_source or expected_fragments[1] not in load_source:
        raise BeamRiskError(
            "当前 Transformers 的分布式 RNG 文件命名与 BeamRisk 恢复器不兼容"
        )
    return {
        "transformers_version": transformers.__version__,
        "distributed_save_pattern": "rng_state_<process_index>.pth",
        "distributed_load_pattern": "rng_state_<process_index>.pth",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    required = {
        "baseline_sft_config": config.paths.baseline_sft_config,
        "model_config": config.paths.model / "config.json",
        "tokenizer": config.paths.model / "tokenizer.json",
        "tokenized_cache_manifest": config.paths.tokenized_cache / "cache_manifest.json",
        "raw_train": config.paths.raw_train,
        "raw_valid": config.paths.raw_valid,
        "raw_train_manifest": config.paths.raw_train_manifest,
        "fixed_validation_subset": config.paths.fixed_validation_subset,
        "tiger_identifier_manifest": config.paths.tiger_identifier_dir
        / "tiger_id_manifest.json",
        "tiger_mapping": config.paths.tiger_mapping,
        "poi_catalog_dir": config.paths.poi_catalog_dir,
        "llamafactory": config.root / "third_party" / "LLaMA-Factory" / "src" / "llamafactory",
        "tiger_evaluator": config.root / "scripts" / "tiger" / "evaluate_retrieval.py",
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise BeamRiskError("正式训练输入缺失：" + "；".join(missing))
    forbidden_initial_state = (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    )
    unexpected = [
        name for name in forbidden_initial_state if (config.paths.model / name).exists()
    ]
    if unexpected:
        raise BeamRiskError(
            "初始模型目录含 SFT/优化器状态，禁止作为从头训练基座："
            + ", ".join(unexpected)
        )
    baseline = __import__("yaml").safe_load(
        config.paths.baseline_sft_config.read_text(encoding="utf-8")
    )
    validate_frozen_baseline(config, baseline)
    lf_config = build_lf_stage_config(
        config,
        target_epoch=1,
        resume_checkpoint=None,
    )
    cache_manifest = read_json(
        config.paths.tokenized_cache / "cache_manifest.json",
        name="Tokenized cache manifest",
    )
    cutoff = cache_manifest.get("inputs", {}).get("cutoff_len")
    if cutoff != config.training.cutoff_len:
        raise BeamRiskError(
            f"Tokenized cache cutoff {cutoff} != {config.training.cutoff_len}"
        )
    packed_rows = cache_manifest.get("packed_rows", {}).get("train")
    if not isinstance(packed_rows, int) or packed_rows <= 0:
        raise BeamRiskError("Tokenized cache manifest 缺少 packed Train 行数")
    per_rank_rows = math.ceil(packed_rows / config.training.world_size)
    per_rank_micro_batches = math.ceil(
        per_rank_rows / config.training.per_device_train_batch_size
    )
    steps_per_epoch = math.ceil(
        per_rank_micro_batches / config.training.gradient_accumulation_steps
    )
    if steps_per_epoch != config.training.expected_optimizer_steps_per_epoch:
        raise BeamRiskError(
            f"Packed Train 推导为 {steps_per_epoch} steps/epoch，配置要求 "
            f"{config.training.expected_optimizer_steps_per_epoch}"
        )
    existing_checkpoints = list(config.paths.sft_output_dir.glob("checkpoint-*"))
    if existing_checkpoints and not args.resume:
        raise BeamRiskError(
            "BeamRisk SFT 输出已有 checkpoint；默认禁止隐式续训，请显式 --resume"
        )
    rng_contract = _validate_transformers_rng_contract()
    checkpoint_audit = audit_checkpoint_chain(
        config,
        deep_archive_check=bool(existing_checkpoints and args.resume),
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "config": str(config.source_path),
                "initial_model": str(config.paths.model),
                "sft_output_dir": str(config.paths.sft_output_dir),
                "candidate_pool_dir": str(config.paths.candidate_pool_dir),
                "mining_root": str(config.paths.mining_root),
                "global_ce_batch": config.training.expected_global_batch_size,
                "global_risk_batch": config.risk.global_batch_size,
                "optimizer_steps_per_epoch": steps_per_epoch,
                "lf_initial_model": lf_config["model_name_or_path"],
                "loads_tiger_sft_checkpoint": False,
                "resume_requested": args.resume,
                "existing_checkpoints": [str(path) for path in existing_checkpoints],
                "expected_rng_state_files": list(
                    expected_rng_state_names(config.training.world_size)
                ),
                "transformers_rng_contract": rng_contract,
                "checkpoint_audit": checkpoint_audit,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

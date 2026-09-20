from __future__ import annotations

import json
import random
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from beamrisk_sft.checkpoints import (
    audit_checkpoint_chain,
    expected_rng_state_names,
    recover_stage_records,
    validate_training_checkpoint,
)
from beamrisk_sft.config import load_config
from beamrisk_sft.errors import BeamRiskError


ROOT = Path(__file__).resolve().parents[2]


def _torch_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{path.stem}/data.pkl", b"test")
        archive.writestr(f"{path.stem}/version", b"3\n")


def _checkpoint(output_dir: Path, *, epoch: int, world_size: int = 4) -> Path:
    step = epoch * 5_571
    path = output_dir / f"checkpoint-{step}"
    path.mkdir(parents=True)
    save_file({"weight": np.asarray([1.0], dtype=np.float32)}, path / "model.safetensors")
    for name in ("optimizer.pt", "training_args.bin"):
        _torch_archive(path / name)
    torch.save({"last_epoch": step}, path / "scheduler.pt")
    for name in expected_rng_state_names(world_size):
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
                "cuda": [],
            },
            path / name,
        )
    (path / "trainer_state.json").write_text(
        json.dumps(
            {
                "epoch": float(epoch),
                "global_step": step,
                "max_steps": 16_713,
                "log_history": [
                    {"epoch": float(epoch), "step": step, "eval_loss": 0.4}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_four_rank_checkpoint_requires_and_accepts_rank_rng_files(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path, epoch=1)
    result = validate_training_checkpoint(
        checkpoint,
        output_dir=tmp_path,
        expected_epoch=1,
        expected_step=5_571,
        expected_total_steps=16_713,
        world_size=4,
        deep_archive_check=True,
    )
    assert result["rng_mode"] == "distributed"
    assert result["rng_files"] == [
        "rng_state_0.pth",
        "rng_state_1.pth",
        "rng_state_2.pth",
        "rng_state_3.pth",
    ]
    assert result["tensor_count"] == 1

    (checkpoint / "rng_state_3.pth").unlink()
    with pytest.raises(BeamRiskError, match="rng_state_3.pth"):
        validate_training_checkpoint(
            checkpoint,
            output_dir=tmp_path,
            expected_epoch=1,
            expected_step=5_571,
            expected_total_steps=16_713,
            world_size=4,
        )


def test_single_rank_checkpoint_uses_unsuffixed_rng_file(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, epoch=1, world_size=1)
    result = validate_training_checkpoint(
        checkpoint,
        output_dir=tmp_path,
        expected_epoch=1,
        expected_step=5_571,
        expected_total_steps=16_713,
        world_size=1,
    )
    assert result["rng_files"] == ["rng_state.pth"]


def test_complete_checkpoint_recovers_missing_stage_records(tmp_path: Path) -> None:
    original = load_config(ROOT / "beamrisk_sft" / "configs" / "main_v1.yaml")
    output_dir = tmp_path / "sft"
    config = replace(
        original,
        paths=replace(original.paths, sft_output_dir=output_dir),
    )
    checkpoint = _checkpoint(output_dir, epoch=1)
    (output_dir / "stage_epoch_1_results.json").write_text(
        json.dumps({"train_loss": 0.8}),
        encoding="utf-8",
    )

    before = audit_checkpoint_chain(config, deep_archive_check=True)
    assert before["recoverable_stage_epochs"] == [1]
    result = recover_stage_records(config, deep_archive_check=True)
    assert result["recovered_epochs"] == [1]

    stages = output_dir / "stages"
    assert (stages / "epoch_1_checkpoint.txt").read_text().strip() == str(
        checkpoint.resolve()
    )
    manifest = json.loads((stages / "epoch_1.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["target_epoch"] == 1
    assert manifest["recovery"]["checkpoint_validation"]["rng_mode"] == "distributed"
    assert manifest["beamrisk_summary"]["global_pairs"] == 0

    second = recover_stage_records(config, deep_archive_check=True)
    assert second["recovered_epochs"] == []


def test_audit_rejects_stage_marker_without_checkpoint(tmp_path: Path) -> None:
    original = load_config(ROOT / "beamrisk_sft" / "configs" / "main_v1.yaml")
    output_dir = tmp_path / "sft"
    config = replace(
        original,
        paths=replace(original.paths, sft_output_dir=output_dir),
    )
    stages = output_dir / "stages"
    stages.mkdir(parents=True)
    (stages / "epoch_1_checkpoint.txt").write_text(
        str(output_dir / "checkpoint-5571") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BeamRiskError, match="残留阶段记录"):
        audit_checkpoint_chain(config)

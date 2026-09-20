"""Strict, rank-aware checkpoint validation and epoch-boundary recovery."""

from __future__ import annotations

import math
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .config import BeamRiskConfig
from .errors import BeamRiskError
from .io import (
    atomic_write_json,
    atomic_write_text,
    implementation_sha256,
    read_json,
    sha256_file,
)


STAGE_SCHEMA_VERSION = "beamrisk-training-stage-v1"


def expected_rng_state_names(world_size: int) -> tuple[str, ...]:
    """Return the exact filenames written/read by Transformers Trainer."""

    if world_size < 1:
        raise BeamRiskError("Checkpoint world_size 必须 >= 1")
    if world_size == 1:
        return ("rng_state.pth",)
    return tuple(f"rng_state_{rank}.pth" for rank in range(world_size))


def _require_nonempty_files(checkpoint: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (checkpoint / name).is_file()]
    empty = [
        name
        for name in names
        if (checkpoint / name).is_file() and (checkpoint / name).stat().st_size <= 0
    ]
    if missing or empty:
        details: list[str] = []
        if missing:
            details.append(f"缺失={missing}")
        if empty:
            details.append(f"空文件={empty}")
        raise BeamRiskError(f"Checkpoint 不完整（{'；'.join(details)}）：{checkpoint}")


def _validate_torch_archive(path: Path) -> None:
    """Read the ZIP central directory without materializing tensor payloads."""

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as error:
        raise BeamRiskError(f"PyTorch 状态文件容器损坏：{path}") from error
    if not names or not any(name.endswith("/data.pkl") for name in names):
        raise BeamRiskError(f"PyTorch 状态文件缺少 data.pkl：{path}")
    if not any(name.endswith("/version") for name in names):
        raise BeamRiskError(f"PyTorch 状态文件缺少 version：{path}")


def _validate_safetensors_headers(paths: list[Path]) -> int:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - formal env always has it
        raise BeamRiskError("正式环境缺少 safetensors，无法校验模型 checkpoint") from error

    tensor_count = 0
    for path in paths:
        try:
            with safe_open(path, framework="pt", device="cpu") as stream:
                tensor_count += len(stream.keys())
        except Exception as error:
            raise BeamRiskError(f"Safetensors header 损坏：{path}") from error
    if tensor_count <= 0:
        raise BeamRiskError("Checkpoint safetensors 中没有 tensor")
    return tensor_count


def _validate_scheduler_and_rng_payloads(
    checkpoint: Path,
    *,
    expected_step: int,
    rng_names: tuple[str, ...],
) -> None:
    """Load only the small state files; the multi-GB optimizer stays unmapped."""

    try:
        import torch

        scheduler = torch.load(
            checkpoint / "scheduler.pt",
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise BeamRiskError(f"Scheduler 状态无法反序列化：{checkpoint}") from error
    if not isinstance(scheduler, Mapping) or scheduler.get("last_epoch") != expected_step:
        raise BeamRiskError(
            f"Scheduler last_epoch {getattr(scheduler, 'get', lambda *_: None)('last_epoch')} "
            f"!= checkpoint step {expected_step}"
        )

    required_rng_keys = {"python", "numpy", "cpu", "cuda"}
    for name in rng_names:
        try:
            state = torch.load(
                checkpoint / name,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as error:
            raise BeamRiskError(f"RNG 状态无法反序列化：{checkpoint / name}") from error
        if not isinstance(state, Mapping) or not required_rng_keys.issubset(state):
            observed = sorted(state) if isinstance(state, Mapping) else type(state).__name__
            raise BeamRiskError(
                f"RNG 状态字段不完整：{checkpoint / name}，实际 {observed}"
            )


def _epoch_eval_loss(state: Mapping[str, Any], *, expected_step: int) -> float:
    matches = [
        row
        for row in state.get("log_history", [])
        if isinstance(row, Mapping)
        and row.get("step") == expected_step
        and isinstance(row.get("eval_loss"), (int, float))
        and math.isfinite(float(row["eval_loss"]))
    ]
    if len(matches) != 1:
        raise BeamRiskError(
            f"Checkpoint step {expected_step} 必须且只能有一条 epoch-end eval_loss，"
            f"实际 {len(matches)}"
        )
    return float(matches[0]["eval_loss"])


def validate_training_checkpoint(
    checkpoint: Path,
    *,
    output_dir: Path,
    expected_epoch: int,
    expected_step: int,
    expected_total_steps: int,
    world_size: int,
    deep_archive_check: bool = False,
) -> dict[str, Any]:
    """Validate everything required for exact optimizer/scheduler/RNG resume."""

    checkpoint = checkpoint.resolve()
    output_dir = output_dir.resolve()
    try:
        checkpoint.relative_to(output_dir)
    except ValueError as error:
        raise BeamRiskError("只允许使用当前 BeamRisk-SFT 输出目录内的 checkpoint") from error
    if checkpoint.name != f"checkpoint-{expected_step}" or not checkpoint.is_dir():
        raise BeamRiskError(
            f"Epoch {expected_epoch} checkpoint 路径应为 checkpoint-{expected_step}："
            f"{checkpoint}"
        )

    core_names = (
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    )
    rng_names = expected_rng_state_names(world_size)
    _require_nonempty_files(checkpoint, core_names + rng_names)
    model_files = sorted(checkpoint.glob("*.safetensors"))
    if not model_files or any(path.stat().st_size <= 0 for path in model_files):
        raise BeamRiskError(f"Checkpoint 缺少非空 safetensors：{checkpoint}")

    state = read_json(checkpoint / "trainer_state.json", name="Trainer state")
    observed_epoch = state.get("epoch")
    if not isinstance(observed_epoch, (int, float)) or not math.isclose(
        float(observed_epoch), float(expected_epoch), abs_tol=1e-4
    ):
        raise BeamRiskError(
            f"Checkpoint epoch {observed_epoch} != {expected_epoch}：{checkpoint}"
        )
    if state.get("global_step") != expected_step:
        raise BeamRiskError(
            f"Checkpoint global_step {state.get('global_step')} != {expected_step}："
            f"{checkpoint}"
        )
    if state.get("max_steps") != expected_total_steps:
        raise BeamRiskError(
            f"Checkpoint max_steps {state.get('max_steps')} != {expected_total_steps}；"
            "无法保证三轮连续 scheduler"
        )
    eval_loss = _epoch_eval_loss(state, expected_step=expected_step)

    if deep_archive_check:
        for name in core_names:
            if name != "trainer_state.json":
                _validate_torch_archive(checkpoint / name)
        for name in rng_names:
            _validate_torch_archive(checkpoint / name)
        tensor_count = _validate_safetensors_headers(model_files)
        _validate_scheduler_and_rng_payloads(
            checkpoint,
            expected_step=expected_step,
            rng_names=rng_names,
        )
    else:
        tensor_count = -1

    return {
        "epoch": expected_epoch,
        "global_step": expected_step,
        "max_steps": expected_total_steps,
        "checkpoint": str(checkpoint),
        "eval_loss": eval_loss,
        "model_files": [path.name for path in model_files],
        "model_bytes": sum(path.stat().st_size for path in model_files),
        "optimizer_bytes": (checkpoint / "optimizer.pt").stat().st_size,
        "rng_mode": "single" if world_size == 1 else "distributed",
        "rng_files": list(rng_names),
        "tensor_count": tensor_count,
        "deep_archive_check": deep_archive_check,
    }


def checkpoint_for_epoch(
    config: BeamRiskConfig,
    epoch: int,
    *,
    deep_archive_check: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if epoch not in range(1, config.training.epochs + 1):
        raise BeamRiskError(f"Checkpoint epoch 超出配置范围：{epoch}")
    step = config.training.expected_optimizer_steps_per_epoch * epoch
    checkpoint = config.paths.sft_output_dir / f"checkpoint-{step}"
    validation = validate_training_checkpoint(
        checkpoint,
        output_dir=config.paths.sft_output_dir,
        expected_epoch=epoch,
        expected_step=step,
        expected_total_steps=(
            config.training.expected_optimizer_steps_per_epoch
            * config.training.epochs
        ),
        world_size=config.training.world_size,
        deep_archive_check=deep_archive_check,
    )
    return checkpoint.resolve(), validation


def _stage_record_status(
    config: BeamRiskConfig,
    *,
    epoch: int,
    checkpoint: Path,
) -> str:
    stages_dir = config.paths.sft_output_dir / "stages"
    marker_path = stages_dir / f"epoch_{epoch}_checkpoint.txt"
    manifest_path = stages_dir / f"epoch_{epoch}.json"
    marker_exists = marker_path.is_file()
    manifest_exists = manifest_path.is_file()

    if marker_exists:
        marker_value = marker_path.read_text(encoding="utf-8").strip()
        if not marker_value or Path(marker_value).resolve() != checkpoint.resolve():
            raise BeamRiskError(f"Epoch {epoch} checkpoint marker 与实际 checkpoint 不一致")
    if manifest_exists:
        manifest = read_json(manifest_path, name=f"Epoch {epoch} stage manifest")
        if (
            manifest.get("schema_version") != STAGE_SCHEMA_VERSION
            or manifest.get("status") != "completed"
            or manifest.get("target_epoch") != epoch
            or Path(str(manifest.get("checkpoint", ""))).resolve()
            != checkpoint.resolve()
        ):
            raise BeamRiskError(f"Epoch {epoch} stage manifest 与实际 checkpoint 不一致")
    if marker_exists and manifest_exists:
        return "complete"
    if marker_exists or manifest_exists:
        return "partial_recoverable"
    return "missing_recoverable"


def audit_checkpoint_chain(
    config: BeamRiskConfig,
    *,
    deep_archive_check: bool = False,
) -> dict[str, Any]:
    """Audit all existing checkpoints and require a contiguous epoch chain."""

    output_dir = config.paths.sft_output_dir
    expected_steps = {
        config.training.expected_optimizer_steps_per_epoch * epoch: epoch
        for epoch in range(1, config.training.epochs + 1)
    }
    existing: dict[int, Path] = {}
    unexpected: list[str] = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            try:
                step = int(path.name.removeprefix("checkpoint-"))
            except ValueError:
                unexpected.append(str(path))
                continue
            epoch = expected_steps.get(step)
            if epoch is None:
                unexpected.append(str(path))
            elif epoch in existing:
                raise BeamRiskError(f"Epoch {epoch} 出现重复 checkpoint")
            else:
                existing[epoch] = path
    if unexpected:
        raise BeamRiskError(f"发现非预注册 epoch 边界 checkpoint：{unexpected}")

    rows: list[dict[str, Any]] = []
    missing_seen = False
    for epoch in range(1, config.training.epochs + 1):
        path = existing.get(epoch)
        if path is None:
            stages_dir = config.paths.sft_output_dir / "stages"
            stale_records = [
                record
                for record in (
                    stages_dir / f"epoch_{epoch}.json",
                    stages_dir / f"epoch_{epoch}_checkpoint.txt",
                )
                if record.exists()
            ]
            if stale_records:
                raise BeamRiskError(
                    f"Epoch {epoch} 不存在 checkpoint，却残留阶段记录：{stale_records}"
                )
            missing_seen = True
            continue
        if missing_seen:
            raise BeamRiskError(f"Checkpoint 链不连续：缺少更早 epoch，却存在 epoch {epoch}")
        checkpoint, validation = checkpoint_for_epoch(
            config,
            epoch,
            deep_archive_check=deep_archive_check,
        )
        rows.append(
            {
                **validation,
                "stage_records": _stage_record_status(
                    config,
                    epoch=epoch,
                    checkpoint=checkpoint,
                ),
            }
        )
    return {
        "status": "passed",
        "deep_archive_check": deep_archive_check,
        "completed_checkpoint_epochs": [row["epoch"] for row in rows],
        "recoverable_stage_epochs": [
            row["epoch"] for row in rows if row["stage_records"] != "complete"
        ],
        "checkpoints": rows,
    }


def _recovered_beamrisk_summary(
    state: Mapping[str, Any],
    *,
    epoch: int,
    steps_per_epoch: int,
    world_size: int,
) -> dict[str, Any]:
    if epoch == 1:
        return {
            "enabled": False,
            "local_pairs": 0,
            "global_pairs": 0,
            "events": 0,
            "recovered_from_trainer_state": True,
        }
    first_step = steps_per_epoch * (epoch - 1)
    last_step = steps_per_epoch * epoch
    rows = [
        row
        for row in state.get("log_history", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("step"), int)
        and first_step < int(row["step"]) <= last_step
    ]
    global_pairs = int(
        sum(float(row.get("beamrisk_pairs", 0.0)) for row in rows)
    )
    events = int(sum(float(row.get("beamrisk_events", 0.0)) for row in rows))
    return {
        "enabled": True,
        "local_pairs": global_pairs // world_size,
        "global_pairs": global_pairs,
        "events": events,
        "recovered_from_trainer_state": True,
    }


def _recovery_manifest(
    config: BeamRiskConfig,
    *,
    epoch: int,
    checkpoint: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    state = read_json(checkpoint / "trainer_state.json", name="Epoch trainer state")
    prior_checkpoint: Path | None = None
    risk_pairs_sha256: str | None = None
    if epoch >= 2:
        prior_checkpoint, _ = checkpoint_for_epoch(
            config,
            epoch - 1,
            deep_archive_check=False,
        )
        risk_manifest = read_json(
            config.mining_dir(epoch - 1) / "manifest.json",
            name=f"Epoch {epoch - 1} risk mining manifest",
        )
        reference = risk_manifest.get("reference_checkpoint")
        if not isinstance(reference, Mapping) or Path(
            str(reference.get("path", ""))
        ).resolve() != prior_checkpoint:
            raise BeamRiskError(
                f"无法恢复 Epoch {epoch} 标记：风险 pair 参考 checkpoint 不一致"
            )
        risk_pairs_sha256 = risk_manifest.get("risk_pairs_sha256")
        if not isinstance(risk_pairs_sha256, str) or len(risk_pairs_sha256) != 64:
            raise BeamRiskError(f"无法恢复 Epoch {epoch} 标记：风险 pair SHA256 无效")

    metrics_path = config.paths.sft_output_dir / f"stage_epoch_{epoch}_results.json"
    training_metrics = (
        read_json(metrics_path, name=f"Epoch {epoch} training metrics")
        if metrics_path.is_file()
        else None
    )
    code_sha256 = implementation_sha256(config.root)
    return {
        "schema_version": STAGE_SCHEMA_VERSION,
        "status": "completed",
        "target_epoch": epoch,
        "initial_model": str(config.paths.model),
        "resume_checkpoint": str(prior_checkpoint) if prior_checkpoint else None,
        "checkpoint": str(checkpoint),
        "global_step": state.get("global_step"),
        "trainer_epoch": state.get("epoch"),
        "validation_loss": validation.get("eval_loss"),
        "risk_enabled": epoch >= 2,
        "risk_reference_epoch": epoch - 1 if epoch >= 2 else None,
        "risk_pairs_sha256": risk_pairs_sha256,
        "beamrisk_summary": _recovered_beamrisk_summary(
            state,
            epoch=epoch,
            steps_per_epoch=config.training.expected_optimizer_steps_per_epoch,
            world_size=config.training.world_size,
        ),
        "training_metrics": training_metrics,
        "lf_stage_config": None,
        "baseline_config_sha256": sha256_file(config.paths.baseline_sft_config),
        "implementation_sha256": code_sha256,
        "recovery": {
            "recovered_at": datetime.now().astimezone().isoformat(),
            "reason": "checkpoint_saved_but_stage_record_commit_did_not_complete",
            "training_implementation_sha256": None,
            "validator_implementation_sha256": code_sha256,
            "checkpoint_validation": dict(validation),
        },
    }


def recover_stage_records(
    config: BeamRiskConfig,
    *,
    deep_archive_check: bool = True,
) -> dict[str, Any]:
    """Commit missing stage records for already-complete epoch checkpoints."""

    audit = audit_checkpoint_chain(
        config,
        deep_archive_check=deep_archive_check,
    )
    recovered: list[int] = []
    for row in audit["checkpoints"]:
        epoch = int(row["epoch"])
        if row["stage_records"] == "complete":
            continue
        checkpoint = Path(str(row["checkpoint"])).resolve()
        stages_dir = config.paths.sft_output_dir / "stages"
        manifest_path = stages_dir / f"epoch_{epoch}.json"
        marker_path = stages_dir / f"epoch_{epoch}_checkpoint.txt"
        if not manifest_path.is_file():
            atomic_write_json(
                manifest_path,
                _recovery_manifest(
                    config,
                    epoch=epoch,
                    checkpoint=checkpoint,
                    validation=row,
                ),
            )
        if not marker_path.is_file():
            # Marker is the final commit record and is intentionally written last.
            atomic_write_text(marker_path, str(checkpoint) + "\n")
        recovered.append(epoch)

    final_audit = audit_checkpoint_chain(
        config,
        deep_archive_check=False,
    )
    return {
        "status": "passed",
        "recovered_epochs": recovered,
        "checkpoint_audit": final_audit,
    }

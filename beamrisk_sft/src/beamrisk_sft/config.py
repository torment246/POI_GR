"""Strict configuration contract for the single pre-registered experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import BeamRiskError


CONFIG_SCHEMA_VERSION = "beamrisk-sft-config-v1"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BeamRiskError(f"配置 {name} 必须是非空路径字符串")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise BeamRiskError(f"配置 {name} 必须是 >= {minimum} 的整数")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BeamRiskError(f"配置 {name} 必须是数值")
    result = float(value)
    if minimum is not None and result < minimum:
        raise BeamRiskError(f"配置 {name} 必须 >= {minimum}")
    return result


@dataclass(frozen=True)
class PathConfig:
    baseline_sft_config: Path
    model: Path
    tokenized_cache: Path
    raw_train: Path
    raw_valid: Path
    raw_train_manifest: Path
    fixed_validation_subset: Path
    tiger_identifier_dir: Path
    tiger_mapping: Path
    poi_catalog_dir: Path
    candidate_pool_dir: Path
    sft_output_dir: Path
    mining_root: Path
    eval_output_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    world_size: int
    epochs: int
    cutoff_len: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    expected_global_batch_size: int
    expected_optimizer_steps_per_epoch: int
    seed: int
    data_seed: int


@dataclass(frozen=True)
class MiningConfig:
    candidate_pool_size: int
    target_pair_count: int
    minimum_pair_count: int
    per_rank_pair_cap: int
    batch_size: int
    beam_size: int
    max_new_tokens: int


@dataclass(frozen=True)
class RiskConfig:
    weight: float
    temperature: float
    margin: float
    interval_optimizer_steps: int
    global_batch_size: int
    per_device_batch_size: int
    per_device_micro_batch_size: int


@dataclass(frozen=True)
class BeamRiskConfig:
    source_path: Path
    paths: PathConfig
    training: TrainingConfig
    mining: MiningConfig
    risk: RiskConfig

    @property
    def root(self) -> Path:
        return project_root()

    def mining_dir(self, reference_epoch: int) -> Path:
        if reference_epoch not in (1, 2):
            raise BeamRiskError("风险挖掘参考 epoch 只允许 1 或 2")
        return self.paths.mining_root / f"epoch_{reference_epoch}"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BeamRiskError(f"配置 {name} 必须是 mapping")
    return value


def load_config(path: Path | str) -> BeamRiskConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise BeamRiskError(f"BeamRisk 配置不存在：{source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BeamRiskError(f"BeamRisk 配置读取失败：{source}") from error
    root_mapping = _mapping(raw, "root")
    if root_mapping.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise BeamRiskError(
            f"配置 schema_version 必须是 {CONFIG_SCHEMA_VERSION}"
        )
    root = project_root()
    path_values = _mapping(root_mapping.get("paths"), "paths")
    paths = PathConfig(
        **{
            name: _resolve(root, path_values.get(name), f"paths.{name}")
            for name in PathConfig.__dataclass_fields__
        }
    )
    train_values = _mapping(root_mapping.get("training"), "training")
    training = TrainingConfig(
        world_size=_integer(train_values.get("world_size"), "training.world_size"),
        epochs=_integer(train_values.get("epochs"), "training.epochs"),
        cutoff_len=_integer(train_values.get("cutoff_len"), "training.cutoff_len"),
        per_device_train_batch_size=_integer(
            train_values.get("per_device_train_batch_size"),
            "training.per_device_train_batch_size",
        ),
        gradient_accumulation_steps=_integer(
            train_values.get("gradient_accumulation_steps"),
            "training.gradient_accumulation_steps",
        ),
        expected_global_batch_size=_integer(
            train_values.get("expected_global_batch_size"),
            "training.expected_global_batch_size",
        ),
        expected_optimizer_steps_per_epoch=_integer(
            train_values.get("expected_optimizer_steps_per_epoch"),
            "training.expected_optimizer_steps_per_epoch",
        ),
        seed=_integer(train_values.get("seed"), "training.seed", minimum=0),
        data_seed=_integer(
            train_values.get("data_seed"), "training.data_seed", minimum=0
        ),
    )
    mining_values = _mapping(root_mapping.get("mining"), "mining")
    mining = MiningConfig(
        candidate_pool_size=_integer(
            mining_values.get("candidate_pool_size"),
            "mining.candidate_pool_size",
        ),
        target_pair_count=_integer(
            mining_values.get("target_pair_count"), "mining.target_pair_count"
        ),
        minimum_pair_count=_integer(
            mining_values.get("minimum_pair_count"),
            "mining.minimum_pair_count",
        ),
        per_rank_pair_cap=_integer(
            mining_values.get("per_rank_pair_cap"),
            "mining.per_rank_pair_cap",
        ),
        batch_size=_integer(mining_values.get("batch_size"), "mining.batch_size"),
        beam_size=_integer(mining_values.get("beam_size"), "mining.beam_size"),
        max_new_tokens=_integer(
            mining_values.get("max_new_tokens"), "mining.max_new_tokens"
        ),
    )
    risk_values = _mapping(root_mapping.get("risk"), "risk")
    risk = RiskConfig(
        weight=_number(risk_values.get("weight"), "risk.weight", minimum=0.0),
        temperature=_number(
            risk_values.get("temperature"), "risk.temperature", minimum=1e-12
        ),
        margin=_number(risk_values.get("margin"), "risk.margin"),
        interval_optimizer_steps=_integer(
            risk_values.get("interval_optimizer_steps"),
            "risk.interval_optimizer_steps",
        ),
        global_batch_size=_integer(
            risk_values.get("global_batch_size"), "risk.global_batch_size"
        ),
        per_device_batch_size=_integer(
            risk_values.get("per_device_batch_size"),
            "risk.per_device_batch_size",
        ),
        per_device_micro_batch_size=_integer(
            risk_values.get("per_device_micro_batch_size"),
            "risk.per_device_micro_batch_size",
        ),
    )
    config = BeamRiskConfig(
        source_path=source,
        paths=paths,
        training=training,
        mining=mining,
        risk=risk,
    )
    validate_config(config)
    return config


def validate_config(config: BeamRiskConfig) -> None:
    if config.training.world_size != 4:
        raise BeamRiskError("主实验固定使用 4 卡")
    if config.training.epochs != 3:
        raise BeamRiskError("主实验固定从初始模型训练 3 个 epoch")
    if config.training.cutoff_len != 512:
        raise BeamRiskError("主实验必须复用 TIGER cutoff_len=512")
    global_batch = (
        config.training.per_device_train_batch_size
        * config.training.gradient_accumulation_steps
        * config.training.world_size
    )
    if global_batch != config.training.expected_global_batch_size:
        raise BeamRiskError(
            f"主 CE 全局 batch 不一致：{global_batch} != "
            f"{config.training.expected_global_batch_size}"
        )
    if config.training.expected_optimizer_steps_per_epoch != 5_571:
        raise BeamRiskError("TIGER packed Train 每个 epoch 必须固定为 5,571 steps")
    if config.risk.global_batch_size != (
        config.risk.per_device_batch_size * config.training.world_size
    ):
        raise BeamRiskError("风险 global batch 与四卡 per-device batch 不一致")
    if (
        config.risk.per_device_batch_size
        % config.risk.per_device_micro_batch_size
        != 0
    ):
        raise BeamRiskError("风险 per-device batch 必须整除 micro batch")
    if config.mining.beam_size != 10:
        raise BeamRiskError("风险挖掘 Beam 必须与主评测一致为 10")
    if config.mining.minimum_pair_count > config.mining.target_pair_count:
        raise BeamRiskError("minimum_pair_count 不得大于 target_pair_count")
    if config.paths.tiger_mapping.parent != config.paths.tiger_identifier_dir:
        raise BeamRiskError("TIGER mapping 必须位于冻结 identifier 目录")
    if config.paths.raw_train.parent != config.paths.raw_valid.parent:
        raise BeamRiskError("Train/Validation 必须来自同一个 TIGER 数据版本")
    managed_root = project_root() / "beamrisk_sft"
    writable_paths = (
        config.paths.candidate_pool_dir,
        config.paths.sft_output_dir,
        config.paths.mining_root,
        config.paths.eval_output_dir,
    )
    for path in writable_paths:
        try:
            path.relative_to(managed_root)
        except ValueError as error:
            raise BeamRiskError(
                f"BeamRisk 新产物必须写入 {managed_root}：{path}"
            ) from error

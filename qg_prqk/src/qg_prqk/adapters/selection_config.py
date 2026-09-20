"""Full-data adapter selection protocol layered on the frozen gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.adapters.config import QueryAdapterGateConfig, load_query_adapter_config


SCHEMA_VERSION = "qg-prqk-p3a-full-config-v1"
EXPECTED_SELECTION_METRICS = (
    "recall_at_10",
    "recall_at_1",
    "difficult_recall_at_10",
)
EXPECTED_EXACT_METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "mrr_at_10",
    "difficult_recall_at_10",
    "query_embedding_cosine_drift",
)
EXPECTED_QUERY_VIEW_POLICY = {
    "D1": "raw_bge",
    "D2": "raw_bge",
    "D3": "final_adapter",
}


class AdapterSelectionConfigError(ValueError):
    """Raised when P3A-FULL settings diverge from the approved protocol."""


@dataclass(frozen=True)
class AdapterSelectionConfig:
    source_path: Path
    source_sha256: str
    base: QueryAdapterGateConfig
    gate: QueryAdapterGateConfig
    base_p3a_config_sha256: str
    gate_p3a_config_sha256: str
    gate_manifest_sha256: str
    output_dir_name: str
    d3_rows: int
    gate_exclusion_rows: int
    internal_holdout_rows: int
    holdout_seed: int
    holdout_selection: str
    max_epochs: int
    checkpoint_selection: tuple[str, ...]
    exact_metrics: tuple[str, ...]
    query_view_policy: Mapping[str, str]

    @property
    def output_dir(self) -> Path:
        return self.base.output_dir / self.output_dir_name

    @property
    def gate_dir(self) -> Path:
        return self.gate.output_dir / "query_adapter_exact"

    def resolved_payload(self) -> dict[str, Any]:
        """Return the complete semantic protocol used for output signatures."""

        return {
            "schema_version": SCHEMA_VERSION,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "base_p3a_config_sha256": self.base_p3a_config_sha256,
            "base_p3a_signature": self.base.signature(),
            "gate_p3a_config_sha256": self.gate_p3a_config_sha256,
            "gate_p3a_signature": self.gate.signature(),
            "gate_manifest_sha256": self.gate_manifest_sha256,
            "output_dir_name": self.output_dir_name,
            "d3_rows": self.d3_rows,
            "gate_exclusion_rows": self.gate_exclusion_rows,
            "internal_holdout_rows": self.internal_holdout_rows,
            "holdout_seed": self.holdout_seed,
            "holdout_selection": self.holdout_selection,
            "max_epochs": self.max_epochs,
            "checkpoint_selection": list(self.checkpoint_selection),
            "exact_metrics": list(self.exact_metrics),
            "query_view_policy": dict(self.query_view_policy),
            "inherited_p3a": self.base.resolved_payload(),
        }

    def signature(self) -> str:
        """Hash the full protocol deterministically."""

        payload = json.dumps(
            self.resolved_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterSelectionConfigError(f"{label} 必须是 mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise AdapterSelectionConfigError(f"{label} 缺少 {key}")
    return mapping[key]


def _string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _required(mapping, key, label)
    if not isinstance(value, str) or not value.strip():
        raise AdapterSelectionConfigError(f"{label}.{key} 必须是非空字符串")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = _required(mapping, key, label)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdapterSelectionConfigError(f"{label}.{key} 必须是正整数")
    return value


def _sha256(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _string(mapping, key, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise AdapterSelectionConfigError(f"{label}.{key} 必须是小写 SHA256")
    return value


def _string_tuple(mapping: Mapping[str, Any], key: str, label: str) -> tuple[str, ...]:
    value = _required(mapping, key, label)
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise AdapterSelectionConfigError(f"{label}.{key} 必须是非空字符串列表")
    return tuple(value)


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AdapterSelectionConfigError(
            f"{label} 必须为 {expected!r}，实际 {actual!r}"
        )


def load_adapter_selection_config(path: Path) -> AdapterSelectionConfig:
    """Load and freeze the user-approved P3A-FULL-only settings."""

    source_path = path.resolve()
    try:
        raw_bytes = source_path.read_bytes()
        root = _mapping(yaml.safe_load(raw_bytes), "配置根")
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise AdapterSelectionConfigError(f"配置读取失败：{source_path}") from error
    _assert_equal(root.get("schema_version"), SCHEMA_VERSION, "schema_version")
    base_raw = Path(_string(root, "base_p3a_config", "配置根"))
    base_path = (
        base_raw.resolve()
        if base_raw.is_absolute()
        else (source_path.parent / base_raw).resolve()
    )
    base = load_query_adapter_config(base_path)
    gate_raw_value = root.get("gate_p3a_config")
    if gate_raw_value is None:
        gate = base
        gate_sha256 = _sha256(root, "base_p3a_config_sha256", "配置根")
    else:
        if not isinstance(gate_raw_value, str) or not gate_raw_value.strip():
            raise AdapterSelectionConfigError("配置根.gate_p3a_config 必须是非空字符串")
        gate_raw = Path(gate_raw_value)
        gate_path = (
            gate_raw.resolve()
            if gate_raw.is_absolute()
            else (source_path.parent / gate_raw).resolve()
        )
        gate = load_query_adapter_config(gate_path)
        gate_sha256 = _sha256(root, "gate_p3a_config_sha256", "配置根")
    full = _mapping(_required(root, "p3a_full", "配置根"), "p3a_full")
    policy = _mapping(
        _required(full, "query_view_policy", "p3a_full"),
        "p3a_full.query_view_policy",
    )
    config = AdapterSelectionConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        base=base,
        gate=gate,
        base_p3a_config_sha256=_sha256(
            root, "base_p3a_config_sha256", "配置根"
        ),
        gate_p3a_config_sha256=gate_sha256,
        gate_manifest_sha256=_sha256(
            root, "gate_manifest_sha256", "配置根"
        ),
        output_dir_name=_string(full, "output_dir_name", "p3a_full"),
        d3_rows=_positive_int(full, "d3_rows", "p3a_full"),
        gate_exclusion_rows=_positive_int(
            full, "gate_exclusion_rows", "p3a_full"
        ),
        internal_holdout_rows=_positive_int(
            full, "internal_holdout_rows", "p3a_full"
        ),
        holdout_seed=_positive_int(full, "holdout_seed", "p3a_full"),
        holdout_selection=_string(
            full, "holdout_selection", "p3a_full"
        ),
        max_epochs=_positive_int(full, "max_epochs", "p3a_full"),
        checkpoint_selection=_string_tuple(
            full, "checkpoint_selection", "p3a_full"
        ),
        exact_metrics=_string_tuple(full, "exact_metrics", "p3a_full"),
        query_view_policy={str(key): str(value) for key, value in policy.items()},
    )
    _assert_equal(
        config.base.source_sha256,
        config.base_p3a_config_sha256,
        "base_p3a_config_sha256",
    )
    _assert_equal(
        config.gate.source_sha256,
        config.gate_p3a_config_sha256,
        "gate_p3a_config_sha256",
    )
    for name in (
        "seed",
        "gate_query_limit",
        "sample_query_limit",
        "selection",
        "query_embedding",
        "ann",
        "adapter",
        "evaluation",
    ):
        _assert_equal(
            getattr(config.base, name),
            getattr(config.gate, name),
            f"active FULL 与历史 Gate 的 {name}",
        )
    _assert_equal(config.output_dir_name, "query_adapter_exact_full", "output_dir")
    _assert_equal(config.d3_rows, 291_590, "p3a_full.d3_rows")
    _assert_equal(
        config.gate_exclusion_rows,
        config.base.gate_query_limit,
        "p3a_full.gate_exclusion_rows",
    )
    _assert_equal(config.internal_holdout_rows, 20_000, "internal holdout")
    _assert_equal(config.holdout_seed, config.base.seed, "holdout seed")
    _assert_equal(
        config.holdout_selection,
        "blake2b_query_id_smallest_after_gate_exclusion",
        "holdout selection",
    )
    _assert_equal(config.max_epochs, 3, "p3a_full.max_epochs")
    _assert_equal(
        config.max_epochs, config.base.adapter.epochs, "Gate/FULL max_epoch"
    )
    _assert_equal(
        config.checkpoint_selection,
        EXPECTED_SELECTION_METRICS,
        "checkpoint selection",
    )
    _assert_equal(config.exact_metrics, EXPECTED_EXACT_METRICS, "exact metrics")
    _assert_equal(
        dict(config.query_view_policy),
        EXPECTED_QUERY_VIEW_POLICY,
        "query view policy",
    )
    if config.gate_exclusion_rows + config.internal_holdout_rows >= config.d3_rows:
        raise AdapterSelectionConfigError("Gate 排除集与 holdout 必须小于 D3 总数")
    return config

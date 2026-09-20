"""Configuration for the POI-only base codebook."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.artifacts import sha256_file
from qg_prqk.sid.pipeline_config import DownstreamConfig, load_downstream_config


SCHEMA_VERSION = "qg-prqk-p5-canonical-config-v1"
PROTOCOL_ID = "hard60_topk5_v1"
OUTPUT_SUBDIR = "poi_prqk_a0_hard60_topk5_v1"
EVIDENCE_PATHS = {
    "medium100k_manifest": (
        "qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/"
        "poi_prqk_a0/medium_100000/manifest.json"
    ),
    "hard_endpoint_diagnostic_manifest": (
        "qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/"
        "diagnostics/medium_100000_hard_endpoint_v1/manifest.json"
    ),
    "hard60_topk5_diagnostic_manifest": (
        "qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/poi_prqk_a0/"
        "diagnostics/medium_100000_hard60_topk5_v1/manifest.json"
    ),
}
EVIDENCE_CONTRACTS = {
    "medium100k_manifest": (
        "qg-prqk-p5-poi-prqk-a0-v1",
        "A0_POI_ONLY_PRQK_INITIALIZATION",
    ),
    "hard_endpoint_diagnostic_manifest": (
        "qg-prqk-p5-medium100k-hard-diagnostic-v1",
        "P5_A0_CONTROLLED_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT",
    ),
    "hard60_topk5_diagnostic_manifest": (
        "qg-prqk-p5-medium100k-hard60-topk5-diagnostic-v1",
        "P5_A0_CONTROLLED_HARD60_TOPK5_DIAGNOSTIC_NOT_CANONICAL_CHECKPOINT",
    ),
}


class BaseCodebookConfigError(ValueError):
    """Raised when the selected P5 protocol diverges from reviewed evidence."""


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BaseCodebookConfigError(f"{label} 字段必须恰好为 {sorted(fields)}")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BaseCodebookConfigError(f"{label} 必须是小写 SHA256")
    return value


@dataclass(frozen=True)
class BaseCodebookExecutionConfig:
    """Expose one P5-only overlay through the downstream configuration interface."""

    source_path: Path
    source_sha256: str
    base: DownstreamConfig
    protocol_id: str
    output_subdir: str
    prqk: Mapping[str, Any]
    selection_evidence: Mapping[str, Mapping[str, str]]

    @property
    def upstream(self):
        return self.base.upstream

    @property
    def output_dir(self) -> Path:
        return self.base.output_dir

    @property
    def codebook_sizes(self) -> tuple[int, int, int]:
        return self.base.codebook_sizes

    @property
    def p5_output_dir(self) -> Path:
        return self.output_dir / self.output_subdir

    def resolved_payload(self) -> dict[str, Any]:
        """Return the frozen downstream payload with only the reviewed P5 overlay."""
        payload = self.base.resolved_payload()
        payload["prqk"] = copy.deepcopy(dict(self.prqk))
        payload["p5_protocol"] = {
            "protocol_id": self.protocol_id,
            "scope": "P5-CAT",
            "output_subdir": self.output_subdir,
            "selection_evidence": copy.deepcopy(dict(self.selection_evidence)),
            "base_downstream_signature": self.base.signature(),
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
        }
        return payload

    def signature(self) -> str:
        encoded = json.dumps(
            self.resolved_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_base_codebook_config(
    path: Path, *, payload: Mapping[str, Any] | None = None
) -> BaseCodebookExecutionConfig:
    """Load the reviewed hard60+Top-k5 overlay without mutating its base config."""
    source_path = path.resolve()
    try:
        source_bytes = source_path.read_bytes()
        raw = (
            yaml.safe_load(source_bytes) if payload is None else copy.deepcopy(payload)
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise BaseCodebookConfigError(f"P5 配置读取失败：{source_path}") from error
    root = _exact(
        raw,
        {
            "schema_version",
            "protocol_id",
            "scope",
            "base_downstream_config",
            "base_downstream_config_sha256",
            "base_downstream_signature",
            "output_subdir",
            "prqk",
            "selection_evidence",
        },
        "P5 配置根",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise BaseCodebookConfigError(f"schema_version 必须为 {SCHEMA_VERSION}")
    if root["protocol_id"] != PROTOCOL_ID or root["scope"] != "P5-CAT":
        raise BaseCodebookConfigError("P5 protocol_id/scope 与已确认参数不匹配")
    if root["output_subdir"] != OUTPUT_SUBDIR:
        raise BaseCodebookConfigError(f"P5 output_subdir 必须为 {OUTPUT_SUBDIR}")
    base_path = (source_path.parent / str(root["base_downstream_config"])).resolve()
    if base_path.parent != source_path.parent:
        raise BaseCodebookConfigError("P5 base downstream 配置必须位于同一 configs 目录")
    if sha256_file(base_path) != _sha256(
        root["base_downstream_config_sha256"], "base_downstream_config_sha256"
    ):
        raise BaseCodebookConfigError("P5 base downstream 配置 SHA256 不匹配")
    base = load_downstream_config(base_path)
    if base.signature() != _sha256(
        root["base_downstream_signature"], "base_downstream_signature"
    ):
        raise BaseCodebookConfigError("P5 base downstream signature 不匹配")

    prqk = _exact(
        root["prqk"],
        {
            "metric",
            "init",
            "seed",
            "remove_global_direction",
            "residual",
            "min_iter",
            "max_iter",
            "objective_rel_tol",
            "poi_assignment_change_tol",
            "query_assignment_change_tol",
            "patience",
            "topk_refinement",
        },
        "prqk",
    )
    expected = copy.deepcopy(base.resolved_payload()["prqk"])
    expected["max_iter"] = 60
    if prqk != expected:
        raise BaseCodebookConfigError("P5 prqk 必须只把 max_iter 改为 60 并保留 Top-k5")

    evidence = _exact(
        root["selection_evidence"], set(EVIDENCE_PATHS), "selection_evidence"
    )
    resolved_evidence: dict[str, dict[str, str]] = {}
    project_root = base.upstream.base.project_root
    for name, expected_relative in EVIDENCE_PATHS.items():
        entry = _exact(evidence[name], {"path", "sha256"}, name)
        if entry["path"] != expected_relative:
            raise BaseCodebookConfigError(f"{name}.path 与已审核证据路径不匹配")
        resolved_evidence[name] = {
            "path": str((project_root / expected_relative).resolve()),
            "sha256": _sha256(entry["sha256"], f"{name}.sha256"),
        }
    return BaseCodebookExecutionConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        base=base,
        protocol_id=PROTOCOL_ID,
        output_subdir=OUTPUT_SUBDIR,
        prqk=copy.deepcopy(dict(prqk)),
        selection_evidence=resolved_evidence,
    )


def validate_selection_evidence(config: BaseCodebookExecutionConfig) -> None:
    """Verify all parameter-selection manifests before a new P5 gate runs."""
    for name, entry in config.selection_evidence.items():
        path = Path(entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise BaseCodebookConfigError(f"P5 参数选择证据缺失或哈希不匹配：{name}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            marker = json.loads((path.parent / "_SUCCESS").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BaseCodebookConfigError(f"P5 参数选择证据不可解析：{name}") from error
        expected_schema, expected_role = EVIDENCE_CONTRACTS[name]
        if (
            manifest.get("schema_version") != expected_schema
            or manifest.get("status") != "completed"
            or manifest.get("role") != expected_role
            or marker.get("manifest_sha256") != entry["sha256"]
        ):
            raise BaseCodebookConfigError(f"P5 参数选择证据合同不匹配：{name}")

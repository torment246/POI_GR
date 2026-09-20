"""Configuration for relational S1/S2 codebook refinement."""

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


SCHEMA_VERSION = "qg-prqk-p6-p7-canonical-config-v1"
PROTOCOL_ID = "hard60_topk5_v1"
AUTHORIZATION_ID = "USER_CONFIRMED_20260909_P6_P7_MAX_ITER_60"
SAMPLE_AUTHORIZATION_ID = "USER_CONFIRMED_20260909_P6_GRAPH_CLOSED_SAMPLE_MEDIUM"
GATE_SELECTION = "deterministic_p5_seeded_poi_plus_p4_query_prefix_graph_closure"
P6_GATES = {
    "sample": {
        "base_poi_rows": 10_000,
        "query_rows": 1_000,
        "expected_closed_poi_rows": 11_459,
        "expected_s1_s2_edge_rows": 2_610,
    },
    "medium": {
        "base_poi_rows": 100_000,
        "query_rows": 50_000,
        "expected_closed_poi_rows": 148_860,
        "expected_s1_s2_edge_rows": 134_265,
    },
}
DIRECT_FULL_SCHEMA_VERSION = "qg-prqk-p6-p7-direct-full-config-v2"
DIRECT_FULL_PROTOCOL_ID = "hard60_topk5_direct_full_v2"
DIRECT_FULL_AUTHORIZATION_ID = "USER_CONFIRMED_20260909_P6_DIRECT_FULL_AFTER_SMOKE"
DIRECT_FULL_SELECTION = "sample_graph_closure_then_full_identity_skip_medium"
DIRECT_FULL_GATES = {
    "sample": P6_GATES["sample"],
    "full": {
        "base_poi_rows": 716_245,
        "query_rows": 342_879,
        "expected_closed_poi_rows": 716_245,
        "expected_s1_s2_edge_rows": 912_980,
    },
}
OUTPUT_SUBDIRS = {
    "p6": "poi_query_category_prqk_s1_s2_hard60_topk5_v1",
    "p7": "poi_query_category_geo_prqk_s3_hard60_topk5_v1",
}
FROZEN_INPUTS = {
    "p4_full_manifest": (
        "qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/"
        "query_graph/full_342879/manifest.json",
        "qg-prqk-p4-query-graph-v3",
        "P4-CAT-FULL",
    ),
    "p5_full_manifest": (
        "qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/"
        "poi_prqk_a0_hard60_topk5_v1/full_716245/manifest.json",
        "qg-prqk-p5-poi-prqk-a0-v1",
        "P5-CAT-FULL",
    ),
}
DIRECT_FULL_FROZEN_INPUTS = {
    **FROZEN_INPUTS,
    "p6_sample_manifest": (
        "qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/"
        "poi_query_category_prqk_s1_s2_hard60_topk5_v1/"
        "sample_001000q_011459p/manifest.json",
        "qg-prqk-p6-dual-view-category-prqk-v1",
        "P6-CAT-SAMPLE",
    ),
}


class RelationalConfigError(ValueError):
    """Raised when the P6/P7 protocol diverges from reviewed settings."""


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RelationalConfigError(f"{label} 字段必须恰好为 {sorted(fields)}")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RelationalConfigError(f"{label} 必须是小写 SHA256")
    return value


@dataclass(frozen=True)
class RelationalCodebookConfig:
    """Expose the reviewed P6/P7 overlay without mutating historical configs."""

    source_path: Path
    source_sha256: str
    schema_version: str
    base: DownstreamConfig
    protocol_id: str
    output_subdirs: Mapping[str, str]
    p6_gates: Mapping[str, Any]
    prqk: Mapping[str, Any]
    frozen_inputs: Mapping[str, Mapping[str, str]]
    authorization: Mapping[str, str]

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
    def p6_output_dir(self) -> Path:
        return self.output_dir / self.output_subdirs["p6"]

    @property
    def p7_output_dir(self) -> Path:
        return self.output_dir / self.output_subdirs["p7"]

    def resolved_payload(self) -> dict[str, Any]:
        """Return the downstream payload with the reviewed P6 execution overlay."""
        payload = self.base.resolved_payload()
        payload["prqk"] = copy.deepcopy(dict(self.prqk))
        payload["p6_p7_protocol"] = {
            "protocol_id": self.protocol_id,
            "scope": "P6-CAT_P7-CAT",
            "output_subdirs": copy.deepcopy(dict(self.output_subdirs)),
            "p6_gates": copy.deepcopy(dict(self.p6_gates)),
            "frozen_inputs": copy.deepcopy(dict(self.frozen_inputs)),
            "authorization": copy.deepcopy(dict(self.authorization)),
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


def load_relational_codebook_config(
    path: Path, *, payload: Mapping[str, Any] | None = None
) -> RelationalCodebookConfig:
    """Load the user-confirmed hard60 overlay and preserve the base signature."""
    source_path = path.resolve()
    try:
        source_bytes = source_path.read_bytes()
        raw = yaml.safe_load(source_bytes) if payload is None else copy.deepcopy(payload)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RelationalConfigError(f"P6/P7 配置读取失败：{source_path}") from error
    root = _exact(
        raw,
        {
            "schema_version",
            "protocol_id",
            "scope",
            "base_downstream_config",
            "base_downstream_config_sha256",
            "base_downstream_signature",
            "output_subdirs",
            "p6_gates",
            "prqk",
            "frozen_inputs",
            "authorization",
        },
        "P6/P7 配置根",
    )
    schema_version = str(root["schema_version"])
    if schema_version == SCHEMA_VERSION:
        expected_protocol_id = PROTOCOL_ID
        expected_selection = GATE_SELECTION
        expected_gates = P6_GATES
        expected_inputs = FROZEN_INPUTS
        expected_authorization = {
            "id": AUTHORIZATION_ID,
            "change": "hard_alternating_fit.max_iter_30_to_60",
            "unchanged": "weights_warmup_topk_residual_capacity_and_stopping_thresholds",
            "sample_protocol_id": SAMPLE_AUTHORIZATION_ID,
        }
    elif schema_version == DIRECT_FULL_SCHEMA_VERSION:
        expected_protocol_id = DIRECT_FULL_PROTOCOL_ID
        expected_selection = DIRECT_FULL_SELECTION
        expected_gates = DIRECT_FULL_GATES
        expected_inputs = DIRECT_FULL_FROZEN_INPUTS
        expected_authorization = {
            "id": DIRECT_FULL_AUTHORIZATION_ID,
            "change": "p6_gate_sequence.sample_to_full_skip_medium",
            "unchanged": "algorithm_weights_warmup_topk_residual_capacity_and_stopping_thresholds",
            "sample_protocol_id": SAMPLE_AUTHORIZATION_ID,
        }
    else:
        raise RelationalConfigError(
            "schema_version 必须为已冻结 canonical-v1 或 direct-full-v2"
        )
    if root["protocol_id"] != expected_protocol_id or root["scope"] != "P6-CAT_P7-CAT":
        raise RelationalConfigError("P6/P7 protocol_id/scope 与已确认参数不匹配")
    base_path = (source_path.parent / str(root["base_downstream_config"])).resolve()
    if base_path.parent != source_path.parent:
        raise RelationalConfigError("P6/P7 base downstream 配置必须位于同一 configs 目录")
    if sha256_file(base_path) != _sha256(
        root["base_downstream_config_sha256"], "base_downstream_config_sha256"
    ):
        raise RelationalConfigError("P6/P7 base downstream 配置 SHA256 不匹配")
    base = load_downstream_config(base_path)
    if base.signature() != _sha256(
        root["base_downstream_signature"], "base_downstream_signature"
    ):
        raise RelationalConfigError("P6/P7 base downstream signature 不匹配")

    output_subdirs = _exact(root["output_subdirs"], set(OUTPUT_SUBDIRS), "output_subdirs")
    if dict(output_subdirs) != OUTPUT_SUBDIRS:
        raise RelationalConfigError("P6/P7 输出子目录与冻结 namespace 不匹配")

    gates = _exact(
        root["p6_gates"], {"selection", "seed", *expected_gates}, "p6_gates"
    )
    if gates["selection"] != expected_selection or gates["seed"] != 42:
        raise RelationalConfigError("P6 Gate 选样算法或 seed 与用户确认口径不匹配")
    resolved_gates: dict[str, Any] = {
        "selection": expected_selection,
        "seed": 42,
    }
    for gate, expected in expected_gates.items():
        entry = _exact(gates[gate], set(expected), f"p6_gates.{gate}")
        if dict(entry) != expected:
            raise RelationalConfigError(f"P6 {gate} 规模与用户确认口径不匹配")
        resolved_gates[gate] = copy.deepcopy(dict(entry))

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
    expected_prqk = copy.deepcopy(base.resolved_payload()["prqk"])
    expected_prqk["max_iter"] = 60
    if prqk != expected_prqk:
        raise RelationalConfigError("P6/P7 prqk 只能把 max_iter 改为 60")

    project_root = base.upstream.base.project_root
    inputs = _exact(root["frozen_inputs"], set(expected_inputs), "frozen_inputs")
    resolved_inputs: dict[str, dict[str, str]] = {}
    for name, (expected_relative, _, _) in expected_inputs.items():
        entry = _exact(inputs[name], {"path", "sha256"}, name)
        if entry["path"] != expected_relative:
            raise RelationalConfigError(f"{name}.path 与冻结输入路径不匹配")
        resolved_inputs[name] = {
            "path": str((project_root / expected_relative).resolve()),
            "sha256": _sha256(entry["sha256"], f"{name}.sha256"),
        }

    authorization = _exact(
        root["authorization"],
        {"id", "change", "unchanged", "sample_protocol_id"},
        "authorization",
    )
    if dict(authorization) != expected_authorization:
        raise RelationalConfigError("P6/P7 max_iter=60 用户授权记录不匹配")
    return RelationalCodebookConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        schema_version=schema_version,
        base=base,
        protocol_id=expected_protocol_id,
        output_subdirs=dict(output_subdirs),
        p6_gates=resolved_gates,
        prqk=copy.deepcopy(dict(prqk)),
        frozen_inputs=resolved_inputs,
        authorization=dict(authorization),
    )


def validate_relational_frozen_inputs(config: RelationalCodebookConfig) -> None:
    """Verify the complete P4 graph and P5 A0 manifests before P6 runs."""
    for name, entry in config.frozen_inputs.items():
        path = Path(entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise RelationalConfigError(f"P6/P7 冻结输入缺失或哈希不匹配：{name}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            marker = json.loads((path.parent / "_SUCCESS").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RelationalConfigError(f"P6/P7 冻结输入不可解析：{name}") from error
        frozen_contract = (
            DIRECT_FULL_FROZEN_INPUTS
            if config.schema_version == DIRECT_FULL_SCHEMA_VERSION
            else FROZEN_INPUTS
        )
        _, expected_schema, expected_phase = frozen_contract[name]
        if (
            manifest.get("schema_version") != expected_schema
            or manifest.get("status") != "completed"
            or manifest.get("phase") != expected_phase
            or marker.get("manifest_sha256") != entry["sha256"]
        ):
            raise RelationalConfigError(f"P6/P7 冻结输入合同不匹配：{name}")

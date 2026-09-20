"""Configuration for the S1/S2-parent S3 variant without GID."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.artifacts import sha256_file
from qg_prqk.sid.relational_config import RelationalCodebookConfig, load_relational_codebook_config


SCHEMA_VERSION = "qg-prqk-p7-nogid-direct-full-config-v1"
PROTOCOL_ID = "p7_nogid_s1s2_parent_hard60_topk5_direct_full_v1"
AUTHORIZATION_ID = "USER_CONFIRMED_20260911_BUILD_S3_FROM_S1S2_WITHOUT_GID"
EXPECTED_WEIGHTS = {
    "bge": 0.35,
    "name_alias": 0.30,
    "category": 0.15,
    "address": 0.10,
}


class NoGIDCodebookConfigError(ValueError):
    """Raised when the reviewed no-GID protocol is changed."""


@dataclass(frozen=True)
class NoGIDCodebookConfig:
    """Resolved no-GID S3 configuration and frozen source fingerprints."""

    source_path: Path
    source_sha256: str
    base: RelationalCodebookConfig
    output_subdir: str
    gates: Mapping[str, Mapping[str, Any]]
    poi_catalog: Mapping[str, Any]
    hard_graph: Mapping[str, Any]
    continuous_geo: Mapping[str, Any]
    local_refinement: Mapping[str, Any]
    final_sid: Mapping[str, Any]
    authorization: Mapping[str, Any]

    @property
    def project_root(self) -> Path:
        return self.base.base.upstream.base.project_root

    @property
    def output_dir(self) -> Path:
        return self.base.base.output_dir / self.output_subdir

    def signature(self) -> str:
        payload = self.resolved_payload()
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def resolved_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "base_execution_signature": self.base.signature(),
            "output_dir": str(self.output_dir),
            "gates": {name: dict(value) for name, value in self.gates.items()},
            "poi_catalog": dict(self.poi_catalog),
            "hard_graph": dict(self.hard_graph),
            "continuous_geo": dict(self.continuous_geo),
            "local_refinement": dict(self.local_refinement),
            "final_sid": dict(self.final_sid),
            "authorization": dict(self.authorization),
            "prqk": dict(self.base.prqk),
            "s3": dict(self.base.base.resolved_payload()["s3"]),
            "warmup": dict(self.base.base.resolved_payload()["warmup"]),
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NoGIDCodebookConfigError(f"{label} 必须是 mapping")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise NoGIDCodebookConfigError(f"{label} 必须是小写 SHA256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NoGIDCodebookConfigError(f"{label} 必须是正整数")
    return value


def _relative_file(config_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise NoGIDCodebookConfigError(f"{label} 必须是同目录文件名")
    return (config_dir / value).resolve()


def load_nogid_codebook_config(path: Path) -> NoGIDCodebookConfig:
    """Load the no-GID variant and reject unreviewed changes."""
    source = path.resolve()
    try:
        raw_bytes = source.read_bytes()
        root = _mapping(yaml.safe_load(raw_bytes), "no-GID 配置根")
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise NoGIDCodebookConfigError(f"no-GID 配置读取失败：{source}") from error
    if root.get("schema_version") != SCHEMA_VERSION or root.get("protocol_id") != PROTOCOL_ID:
        raise NoGIDCodebookConfigError("no-GID schema_version/protocol_id 不匹配")
    if root.get("scope") != "P7-CAT_S3_S1S2_PARENT_NO_GID":
        raise NoGIDCodebookConfigError("no-GID scope 不匹配")

    base_path = _relative_file(source.parent, root.get("base_execution_config"), "base_execution_config")
    if sha256_file(base_path) != _sha(root.get("base_execution_config_sha256"), "base_execution_config_sha256"):
        raise NoGIDCodebookConfigError("no-GID base YAML SHA256 不匹配")
    base = load_relational_codebook_config(base_path)
    if base.signature() != _sha(root.get("base_execution_signature"), "base_execution_signature"):
        raise NoGIDCodebookConfigError("no-GID base execution signature 不匹配")
    output_subdir = root.get("output_subdir")
    if output_subdir != "poi_query_category_geo_nogid_prqk_s3_s1s2_parent_hard60_topk5_v1":
        raise NoGIDCodebookConfigError("no-GID 输出 namespace 不匹配")

    gates = _mapping(root.get("gates"), "gates")
    expected = {
        "sample": (11_459, 1_000, 832, 832),
        "full": (716_245, 342_879, 291_590, 291_590),
    }
    if set(gates) != set(expected):
        raise NoGIDCodebookConfigError("no-GID 只开放 sample/full")
    for name, expected_rows in expected.items():
        gate = _mapping(gates[name], f"gates.{name}")
        actual = tuple(
            _positive_int(gate.get(key), f"gates.{name}.{key}")
            for key in ("poi_rows", "p6_query_rows", "d3_query_rows", "s3_edge_rows")
        )
        if actual != expected_rows:
            raise NoGIDCodebookConfigError(f"no-GID {name} 行数不匹配")
        frozen = _mapping(gate.get("p6_manifest"), f"gates.{name}.p6_manifest")
        manifest_path = (base.base.upstream.base.project_root / str(frozen.get("path"))).resolve()
        if not manifest_path.is_relative_to(base.base.output_dir.resolve()):
            raise NoGIDCodebookConfigError("no-GID P6 manifest 越出 QG 输出目录")
        if sha256_file(manifest_path) != _sha(frozen.get("sha256"), f"gates.{name}.p6_manifest.sha256"):
            raise NoGIDCodebookConfigError(f"no-GID {name} P6 manifest SHA256 不匹配")

    catalog = _mapping(root.get("poi_catalog"), "poi_catalog")
    catalog_path = (base.base.upstream.base.project_root / str(catalog.get("path"))).resolve()
    if sha256_file(catalog_path / "manifest.json") != _sha(catalog.get("manifest_sha256"), "catalog manifest"):
        raise NoGIDCodebookConfigError("no-GID active catalog manifest SHA256 不匹配")
    if sha256_file(catalog_path / "poi_ids.jsonl") != _sha(catalog.get("poi_ids_sha256"), "catalog IDs"):
        raise NoGIDCodebookConfigError("no-GID active POI IDs SHA256 不匹配")
    if catalog.get("allowed_fields") != [
        "poi_id", "displayname", "alias", "category", "category_code", "address", "lat", "lng"
    ] or catalog.get("excluded_fields") != ["area", "layer", "click_score"]:
        raise NoGIDCodebookConfigError("no-GID POI 字段合同不匹配")

    hard = _mapping(root.get("hard_graph"), "hard_graph")
    weights = _mapping(hard.get("score_weights"), "hard_graph.score_weights")
    if set(weights) != set(EXPECTED_WEIGHTS) or any(
        abs(float(weights[name]) - expected_weight) > 1.0e-12
        for name, expected_weight in EXPECTED_WEIGHTS.items()
    ):
        raise NoGIDCodebookConfigError("no-GID hard graph 语义权重不匹配")
    if (
        float(hard.get("score_threshold", -1)) != 0.50
        or hard.get("threshold_equivalence") != "original_0.60_minus_same_gid6_constant_0.10"
        or hard.get("max_neighbors") != 20
        or hard.get("parent_key") != ["s1", "s2"]
        or hard.get("require_same_fine_category") is not True
        or hard.get("exclude_self_poi") is not True
        or hard.get("exclude_false_negative_protected") is not True
        or hard.get("canonical_duplicate_mapping") is not None
    ):
        raise NoGIDCodebookConfigError("no-GID hard graph 合同不匹配")

    geo = _mapping(root.get("continuous_geo"), "continuous_geo")
    if (
        geo.get("gid_or_geohash_used") is not False
        or geo.get("parent_key") != ["s1", "s2"]
        or geo.get("coordinate_system") != "beijing_local_equirectangular_meters"
        or geo.get("standardization") != "median_iqr_with_mean_std_fallback"
        or geo.get("singleton_zero") is not True
        or geo.get("features") != [
            "parent_dx", "parent_dy", "log1p_parent_distance", "parent_bearing_sin", "parent_bearing_cos"
        ]
    ):
        raise NoGIDCodebookConfigError("no-GID continuous Geo 合同不匹配")
    local = _mapping(root.get("local_refinement"), "local_refinement")
    if local.get("candidate_codes") != 32 or local.get("sweeps") != 3:
        raise NoGIDCodebookConfigError("no-GID 局部细化必须为 32 candidates / 3 sweeps")
    final_sid = _mapping(root.get("final_sid"), "final_sid")
    if final_sid != {"positions": ["s1", "s2", "s3"], "gid_in_identifier": False, "force_unique": False}:
        raise NoGIDCodebookConfigError("no-GID Final SID 合同不匹配")
    authorization = _mapping(root.get("authorization"), "authorization")
    if (
        authorization.get("id") != AUTHORIZATION_ID
        or authorization.get("gate_sequence") != "sample_smoke_then_full_skip_medium"
        or authorization.get("stop_after") != "HOLD_FOR_NOGID_S3_REVIEW"
        or authorization.get("forbidden_downstream")
        != ["FINAL_PID", "DEDUP_ID", "TOKENIZER", "TRIE", "QWEN_SFT", "EXTERNAL_BASELINE"]
    ):
        raise NoGIDCodebookConfigError("no-GID 用户授权/停止边界不匹配")
    return NoGIDCodebookConfig(
        source_path=source,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        base=base,
        output_subdir=str(output_subdir),
        gates=gates,
        poi_catalog=catalog,
        hard_graph=hard,
        continuous_geo=geo,
        local_refinement=local,
        final_sid=final_sid,
        authorization=authorization,
    )

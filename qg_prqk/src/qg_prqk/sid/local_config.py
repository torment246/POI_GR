"""Configuration for local S3 refinement over frozen S1/S2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.artifacts import sha256_file
from qg_prqk.sid.relational_config import RelationalCodebookConfig, load_relational_codebook_config


SCHEMA_VERSION = "qg-prqk-p7-direct-full-config-v1"
PROTOCOL_ID = "p7_hard60_topk5_direct_full_v1"
AUTHORIZATION_ID = "USER_CONFIRMED_20260911_P7_SAMPLE_THEN_FULL"
EXPECTED_WEIGHTS = {
    "bge": 0.35,
    "name_alias": 0.30,
    "category": 0.15,
    "address": 0.10,
    "geo": 0.10,
}


class LocalCodebookConfigError(ValueError):
    """Raised when the P7 protocol differs from the reviewed method."""


@dataclass(frozen=True)
class LocalCodebookConfig:
    """Resolved immutable P7 configuration and source fingerprints."""

    source_path: Path
    source_sha256: str
    base: RelationalCodebookConfig
    output_subdir: str
    gates: Mapping[str, Mapping[str, Any]]
    poi_catalog: Mapping[str, Any]
    hard_graph: Mapping[str, Any]
    geo: Mapping[str, Any]
    local_refinement: Mapping[str, Any]
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
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def resolved_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "base_execution_signature": self.base.signature(),
            "output_dir": str(self.output_dir),
            "gates": {name: dict(values) for name, values in self.gates.items()},
            "poi_catalog": dict(self.poi_catalog),
            "hard_graph": dict(self.hard_graph),
            "geo": dict(self.geo),
            "local_refinement": dict(self.local_refinement),
            "authorization": dict(self.authorization),
            "prqk": dict(self.base.prqk),
            "s3": dict(self.base.base.resolved_payload()["s3"]),
            "warmup": dict(self.base.base.resolved_payload()["warmup"]),
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LocalCodebookConfigError(f"{label} 必须是 mapping")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise LocalCodebookConfigError(f"{label} 必须是小写 SHA256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LocalCodebookConfigError(f"{label} 必须是正整数")
    return value


def _relative_file(config_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LocalCodebookConfigError(f"{label} 必须是非空路径")
    path = Path(value)
    if path.is_absolute() or path.name != value:
        raise LocalCodebookConfigError(f"{label} 必须是同目录文件名")
    return (config_dir / path).resolve()


def load_local_codebook_config(path: Path) -> LocalCodebookConfig:
    """Load P7 and reject every unreviewed algorithm or input change."""
    source = path.resolve()
    try:
        raw_bytes = source.read_bytes()
        root = _mapping(yaml.safe_load(raw_bytes), "P7 配置根")
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise LocalCodebookConfigError(f"P7 配置读取失败：{source}") from error
    if root.get("schema_version") != SCHEMA_VERSION or root.get("protocol_id") != PROTOCOL_ID:
        raise LocalCodebookConfigError("P7 schema_version/protocol_id 不匹配")
    if root.get("scope") != "P7-CAT_S3_GEO":
        raise LocalCodebookConfigError("P7 scope 不匹配")

    base_path = _relative_file(source.parent, root.get("base_execution_config"), "base_execution_config")
    if sha256_file(base_path) != _sha(root.get("base_execution_config_sha256"), "base_execution_config_sha256"):
        raise LocalCodebookConfigError("P7 base execution YAML SHA256 不匹配")
    base = load_relational_codebook_config(base_path)
    if base.signature() != _sha(root.get("base_execution_signature"), "base_execution_signature"):
        raise LocalCodebookConfigError("P7 base execution signature 不匹配")
    output_subdir = root.get("output_subdir")
    if output_subdir != base.output_subdirs["p7"]:
        raise LocalCodebookConfigError("P7 output_subdir 与冻结 namespace 不一致")

    gates = _mapping(root.get("gates"), "gates")
    if set(gates) != {"sample", "full"}:
        raise LocalCodebookConfigError("P7 只开放 sample 与 full")
    expected = {
        "sample": (11_459, 1_000, 832, 832),
        "full": (716_245, 342_879, 291_590, 291_590),
    }
    for name, values in gates.items():
        gate = _mapping(values, f"gates.{name}")
        actual = tuple(
            _positive_int(gate.get(key), f"gates.{name}.{key}")
            for key in ("poi_rows", "p6_query_rows", "d3_query_rows", "s3_edge_rows")
        )
        if actual != expected[name]:
            raise LocalCodebookConfigError(f"P7 {name} 行数与冻结 P6/P4 不一致")
        manifest = _mapping(gate.get("p6_manifest"), f"gates.{name}.p6_manifest")
        manifest_path = (
            base.base.upstream.base.project_root / str(manifest.get("path"))
        ).resolve()
        if not manifest_path.is_relative_to(base.base.output_dir.resolve()):
            raise LocalCodebookConfigError("P7 P6 manifest 必须位于 QG 输出目录")
        if sha256_file(manifest_path) != _sha(manifest.get("sha256"), f"gates.{name}.p6_manifest.sha256"):
            raise LocalCodebookConfigError(f"P7 {name} P6 manifest SHA256 不匹配")

    catalog = _mapping(root.get("poi_catalog"), "poi_catalog")
    catalog_path = (
        base.base.upstream.base.project_root / str(catalog.get("path"))
    ).resolve()
    if not catalog_path.is_dir():
        raise LocalCodebookConfigError("P7 active POI catalog 不存在")
    if sha256_file(catalog_path / "manifest.json") != _sha(catalog.get("manifest_sha256"), "poi_catalog.manifest_sha256"):
        raise LocalCodebookConfigError("P7 active POI catalog manifest SHA256 不匹配")
    if sha256_file(catalog_path / "poi_ids.jsonl") != _sha(catalog.get("poi_ids_sha256"), "poi_catalog.poi_ids_sha256"):
        raise LocalCodebookConfigError("P7 active POI IDs SHA256 不匹配")
    if catalog.get("allowed_fields") != [
        "poi_id", "displayname", "alias", "category", "category_code", "address", "lat", "lng"
    ] or catalog.get("excluded_fields") != ["area", "layer", "click_score"]:
        raise LocalCodebookConfigError("P7 POI 字段白名单/排除项不匹配")

    hard = _mapping(root.get("hard_graph"), "hard_graph")
    weights = _mapping(hard.get("score_weights"), "hard_graph.score_weights")
    if set(weights) != set(EXPECTED_WEIGHTS) or any(
        abs(float(weights[name]) - value) > 1.0e-12 for name, value in EXPECTED_WEIGHTS.items()
    ):
        raise LocalCodebookConfigError("P7 困难图权重不匹配")
    if (
        float(hard.get("score_threshold", -1)) != 0.60
        or hard.get("max_neighbors") != 20
        or hard.get("parent_key") != ["gid6", "s1", "s2"]
        or hard.get("require_same_fine_category") is not True
        or hard.get("exclude_self_poi") is not True
        or hard.get("exclude_false_negative_protected") is not True
        or hard.get("canonical_duplicate_mapping") is not None
    ):
        raise LocalCodebookConfigError("P7 困难图 Top-20/阈值/parent/filter 不匹配")

    geo = _mapping(root.get("geo"), "geo")
    if (
        geo.get("geohash_length") != 6
        or geo.get("longitude_first") is not True
        or geo.get("coordinate_system") != "beijing_local_equirectangular_meters"
        or geo.get("standardization") != "median_iqr_with_mean_std_fallback"
        or geo.get("singleton_zero") is not True
        or len(geo.get("features", [])) != 5
    ):
        raise LocalCodebookConfigError("P7 Geo 合同不匹配")
    local = _mapping(root.get("local_refinement"), "local_refinement")
    if local.get("candidate_codes") != 32 or local.get("sweeps") != 3:
        raise LocalCodebookConfigError("P7 局部细化必须为 32 candidates / 3 sweeps")
    authorization = _mapping(root.get("authorization"), "authorization")
    if (
        authorization.get("id") != AUTHORIZATION_ID
        or authorization.get("gate_sequence") != "sample_smoke_then_full_skip_medium"
        or authorization.get("stop_after") != "HOLD_FOR_P7_FULL_REVIEW"
    ):
        raise LocalCodebookConfigError("P7 用户授权记录不匹配")
    return LocalCodebookConfig(
        source_path=source,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        base=base,
        output_subdir=str(output_subdir),
        gates=gates,
        poi_catalog=catalog,
        hard_graph=hard,
        geo=geo,
        local_refinement=local,
        authorization=authorization,
    )

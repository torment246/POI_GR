"""Configuration for static Semantic-ID evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.artifacts import sha256_file
from qg_prqk.sid.local_config import LocalCodebookConfig, load_local_codebook_config


SCHEMA_VERSION = "qg-prqk-p8-static-evaluation-config-v1"
PROTOCOL_ID = "p8_a0_vs_a4_static_full_v1"
AUTHORIZATION_ID = "USER_CONFIRMED_20260911_ENTER_P8_CAT"
METHODS = ["A0_POI_ONLY_PRQK_INITIALIZATION", "A4_QUERY_CATEGORY_GEO_FULL"]


class SidEvaluationConfigError(ValueError):
    """Raised when P8 inputs or metric semantics differ from the reviewed protocol."""


@dataclass(frozen=True)
class SidEvaluationConfig:
    """Resolved immutable P8 contract and its frozen source fingerprints."""

    source_path: Path
    source_sha256: str
    base: LocalCodebookConfig
    output_subdir: str
    gates: Mapping[str, Mapping[str, Any]]
    comparison: Mapping[str, Any]
    prefix_probe: Mapping[str, Any]
    publication: Mapping[str, Any]
    authorization: Mapping[str, Any]

    @property
    def project_root(self) -> Path:
        return self.base.project_root

    @property
    def output_dir(self) -> Path:
        return self.base.base.base.output_dir / self.output_subdir

    def resolved_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "base_p7_signature": self.base.signature(),
            "output_dir": str(self.output_dir),
            "gates": {name: dict(values) for name, values in self.gates.items()},
            "comparison": dict(self.comparison),
            "prefix_probe": dict(self.prefix_probe),
            "publication": dict(self.publication),
            "authorization": dict(self.authorization),
        }

    def signature(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.resolved_payload(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SidEvaluationConfigError(f"{label} 必须是 mapping")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SidEvaluationConfigError(f"{label} 必须是小写 SHA256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SidEvaluationConfigError(f"{label} 必须是正整数")
    return value


def _relative_file(config_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SidEvaluationConfigError(f"{label} 必须是非空路径")
    path = Path(value)
    if path.is_absolute() or path.name != value:
        raise SidEvaluationConfigError(f"{label} 必须是同目录文件名")
    return (config_dir / path).resolve()


def load_sid_evaluation_config(path: Path) -> SidEvaluationConfig:
    """Load P8 and reject unreviewed sources, comparisons, or downstream scope."""
    source = path.resolve()
    try:
        raw_bytes = source.read_bytes()
        root = _mapping(yaml.safe_load(raw_bytes), "P8 配置根")
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise SidEvaluationConfigError(f"P8 配置读取失败：{source}") from error
    if root.get("schema_version") != SCHEMA_VERSION or root.get("protocol_id") != PROTOCOL_ID:
        raise SidEvaluationConfigError("P8 schema_version/protocol_id 不匹配")
    if root.get("scope") != "P8-CAT_STATIC_EVALUATION_AND_SID_CANDIDATE":
        raise SidEvaluationConfigError("P8 scope 不匹配")

    base_path = _relative_file(source.parent, root.get("base_p7_config"), "base_p7_config")
    if sha256_file(base_path) != _sha(root.get("base_p7_config_sha256"), "base_p7_config_sha256"):
        raise SidEvaluationConfigError("P8 base P7 YAML SHA256 不匹配")
    base = load_local_codebook_config(base_path)
    if base.signature() != _sha(root.get("base_p7_signature"), "base_p7_signature"):
        raise SidEvaluationConfigError("P8 base P7 signature 不匹配")

    output_subdir = root.get("output_subdir")
    if output_subdir != "p8_static_evaluation_a0_vs_a4_v1":
        raise SidEvaluationConfigError("P8 输出 namespace 不匹配")
    output_dir = base.base.base.output_dir / str(output_subdir)
    if not output_dir.resolve().is_relative_to(base.base.base.output_dir.resolve()):
        raise SidEvaluationConfigError("P8 输出越出 QG namespace")

    gates = _mapping(root.get("gates"), "gates")
    if set(gates) != {"sample", "full"}:
        raise SidEvaluationConfigError("P8 只开放 sample/full")
    expected = {"sample": (11_459, 1_000, 832), "full": (716_245, 342_879, 291_590)}
    for name, values in gates.items():
        gate = _mapping(values, f"gates.{name}")
        actual = tuple(
            _positive_int(gate.get(key), f"gates.{name}.{key}")
            for key in ("poi_rows", "query_rows", "d3_query_rows")
        )
        if actual != expected[name]:
            raise SidEvaluationConfigError(f"P8 {name} 行数与冻结 P7 不一致")
        manifest = _mapping(gate.get("p7_manifest"), f"gates.{name}.p7_manifest")
        manifest_path = (base.project_root / str(manifest.get("path"))).resolve()
        if not manifest_path.is_relative_to(base.base.base.output_dir.resolve()):
            raise SidEvaluationConfigError("P8 P7 manifest 必须位于 QG 输出目录")
        if sha256_file(manifest_path) != _sha(
            manifest.get("sha256"), f"gates.{name}.p7_manifest.sha256"
        ):
            raise SidEvaluationConfigError(f"P8 {name} P7 manifest SHA256 不匹配")

    comparison = _mapping(root.get("comparison"), "comparison")
    if (
        comparison.get("methods") != METHODS
        or comparison.get("external_baseline") is not False
        or comparison.get("codebook_sizes") != [512, 512, 512]
    ):
        raise SidEvaluationConfigError("P8 只允许 A0/A4 512x3 内部比较")

    probe = _mapping(root.get("prefix_probe"), "prefix_probe")
    expected_probe = {
        "similarity": "cosine",
        "query_transform": "remove_frozen_poi_global_direction_then_l2_normalize",
        "residual": "selected_centroid_projection_then_l2_normalize",
        "a0_codebook_view": "poi",
        "a4_codebook_view": "query",
        "s1_candidates": "all_codes",
        "s2_candidates": "codes_observed_under_correct_s1",
        "s3_candidates": "codes_observed_under_correct_gid6_s1_s2",
        "cumulative_mode": "autoregressive_semantic_prefix_with_correct_gid6",
        "aggregation": ["edge_weighted", "unweighted"],
        "tie_break": "lowest_token_id",
    }
    if dict(probe) != expected_probe:
        raise SidEvaluationConfigError("P8 Prefix Probe 口径不匹配")

    publication = _mapping(root.get("publication"), "publication")
    if publication != {
        "sid_source": "A4_QUERY_CATEGORY_GEO_FULL",
        "include_query_assignments": True,
        "include_category_distributions": True,
        "include_bucket_index": True,
        "force_unique_sid": False,
    }:
        raise SidEvaluationConfigError("P8 SID 发布合同不匹配")
    authorization = _mapping(root.get("authorization"), "authorization")
    if (
        authorization.get("id") != AUTHORIZATION_ID
        or authorization.get("sample_role") != "CODE_DATA_GPU_ARTIFACT_SMOKE_ONLY"
        or authorization.get("formal_metric_basis") != "full"
        or authorization.get("stop_after") != "HOLD_FOR_REVIEW"
        or authorization.get("forbidden_downstream")
        != [
            "FINAL_PID",
            "DEDUP_ID",
            "TOKENIZER",
            "TRIE",
            "QWEN_SFT",
            "EXTERNAL_BASELINE",
            "OTHER_ABLATIONS",
        ]
    ):
        raise SidEvaluationConfigError("P8 用户授权或停止边界不匹配")
    return SidEvaluationConfig(
        source_path=source,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        base=base,
        output_subdir=str(output_subdir),
        gates=gates,
        comparison=comparison,
        prefix_probe=probe,
        publication=publication,
        authorization=authorization,
    )

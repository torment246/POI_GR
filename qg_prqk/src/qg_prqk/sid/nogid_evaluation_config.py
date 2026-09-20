"""Configuration for the full no-GID S3 comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.artifacts import sha256_file


SCHEMA_VERSION = "qg-prqk-p8-nogid-comparison-config-v1"
PROTOCOL_ID = "p8_nogid_s1s2_parent_comparison_full_v1"
AUTHORIZATION_ID = "USER_CONFIRMED_20260911_BUILD_S3_FROM_S1S2_WITHOUT_GID"


class NoGIDEvaluationConfigError(ValueError):
    """Raised when the frozen no-GID comparison contract changes."""


@dataclass(frozen=True)
class NoGIDEvaluationConfig:
    """Resolved immutable A0/A4-GID/A4-NoGID comparison configuration."""

    source_path: Path
    source_sha256: str
    project_root: Path
    output_dir: Path
    rows: Mapping[str, int]
    frozen_manifests: Mapping[str, Mapping[str, str]]
    category_mapping: Path
    comparison: Mapping[str, Any]
    publication: Mapping[str, Any]
    authorization: Mapping[str, Any]

    def signature(self) -> str:
        return hashlib.sha256(
            json.dumps(self.resolved_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def resolved_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "output_dir": str(self.output_dir),
            "rows": dict(self.rows),
            "frozen_manifests": {name: dict(value) for name, value in self.frozen_manifests.items()},
            "category_mapping": str(self.category_mapping),
            "comparison": dict(self.comparison),
            "publication": dict(self.publication),
            "authorization": dict(self.authorization),
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NoGIDEvaluationConfigError(f"{label} 必须是 mapping")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise NoGIDEvaluationConfigError(f"{label} 必须是小写 SHA256")
    return value


def load_nogid_evaluation_config(path: Path) -> NoGIDEvaluationConfig:
    """Load and validate the full no-GID comparison protocol."""
    source = path.resolve()
    project_root = source.parents[2]
    try:
        raw_bytes = source.read_bytes()
        root = _mapping(yaml.safe_load(raw_bytes), "P8 no-GID 配置根")
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise NoGIDEvaluationConfigError(f"P8 no-GID 配置读取失败：{source}") from error
    if root.get("schema_version") != SCHEMA_VERSION or root.get("protocol_id") != PROTOCOL_ID:
        raise NoGIDEvaluationConfigError("P8 no-GID schema/protocol 不匹配")
    if root.get("scope") != "P8_COMPARE_A0_A4_GID_A4_NOGID_WITH_THREE_TOKEN_SID":
        raise NoGIDEvaluationConfigError("P8 no-GID scope 不匹配")
    output = (project_root / str(root.get("output_dir"))).resolve()
    qg_output = (project_root / "qg_prqk/outputs").resolve()
    if not output.is_relative_to(qg_output) or output.name != "full_342879q_716245p":
        raise NoGIDEvaluationConfigError("P8 no-GID 输出目录不匹配")
    rows = _mapping(root.get("rows"), "rows")
    if rows != {"poi": 716_245, "query": 342_879, "d3_query": 291_590, "embedding_dim": 1024}:
        raise NoGIDEvaluationConfigError("P8 no-GID 行数合同不匹配")
    frozen = _mapping(root.get("frozen_manifests"), "frozen_manifests")
    if set(frozen) != {"p4", "p5_a0", "p6", "p7_gid", "p7_nogid"}:
        raise NoGIDEvaluationConfigError("P8 no-GID frozen manifest 集合不匹配")
    for name, item in frozen.items():
        entry = _mapping(item, f"frozen_manifests.{name}")
        source_manifest = (project_root / str(entry.get("path"))).resolve()
        if not source_manifest.is_relative_to(qg_output):
            raise NoGIDEvaluationConfigError(f"P8 no-GID {name} manifest 越出 QG 输出")
        if sha256_file(source_manifest) != _sha(entry.get("sha256"), f"{name}.sha256"):
            raise NoGIDEvaluationConfigError(f"P8 no-GID {name} manifest SHA256 不匹配")
    category_mapping = (project_root / str(root.get("category_mapping"))).resolve()
    if not category_mapping.exists() or not category_mapping.is_relative_to(qg_output):
        raise NoGIDEvaluationConfigError("P8 no-GID category mapping 不存在或越界")
    comparison = _mapping(root.get("comparison"), "comparison")
    if (
        comparison.get("methods") != ["A0_POI_ONLY", "A4_GID_PARENT", "A4_NOGID_S1S2_PARENT"]
        or comparison.get("final_sid_positions") != ["s1", "s2", "s3"]
        or comparison.get("s3_candidate_parent") != ["s1", "s2"]
        or comparison.get("similarity") != "cosine"
        or comparison.get("residual") != "selected_centroid_projection_then_l2_normalize"
        or comparison.get("aggregation") != ["edge_weighted", "unweighted"]
        or comparison.get("tie_break") != "lowest_token_id"
        or comparison.get("external_baseline") is not False
    ):
        raise NoGIDEvaluationConfigError("P8 no-GID comparison 合同不匹配")
    publication = _mapping(root.get("publication"), "publication")
    if publication != {
        "sid_source": "A4_NOGID_S1S2_PARENT",
        "include_query_assignments": True,
        "include_bucket_index": True,
        "force_unique_sid": False,
        "gid_artifact": False,
    }:
        raise NoGIDEvaluationConfigError("P8 no-GID publication 合同不匹配")
    authorization = _mapping(root.get("authorization"), "authorization")
    if (
        authorization.get("id") != AUTHORIZATION_ID
        or authorization.get("stop_after") != "HOLD_FOR_NOGID_S3_REVIEW"
        or authorization.get("forbidden_downstream")
        != ["FINAL_PID", "DEDUP_ID", "TOKENIZER", "TRIE", "QWEN_SFT", "EXTERNAL_BASELINE"]
    ):
        raise NoGIDEvaluationConfigError("P8 no-GID 授权或停止边界不匹配")
    return NoGIDEvaluationConfig(
        source_path=source,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        project_root=project_root,
        output_dir=output,
        rows={name: int(value) for name, value in rows.items()},
        frozen_manifests=frozen,
        category_mapping=category_mapping,
        comparison=comparison,
        publication=publication,
        authorization=authorization,
    )

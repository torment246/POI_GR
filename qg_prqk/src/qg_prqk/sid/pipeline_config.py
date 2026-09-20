"""Current 512x3 SID configuration over immutable adapter inputs.

当前 A0 / A4-GID / A4-NoGID 的基础配置入口；历史 config.py 的 1024×3
合同不决定这里的容量。BGE 向量维度仍为 1024，与每层 512 个码字无关。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.adapters.selection_config import AdapterSelectionConfig, load_adapter_selection_config


SCHEMA_VERSION = "qg-prqk-downstream-config-v2.1"
CURRENT_SID_CODEBOOK_SIZES = (512, 512, 512)
OUTPUT_DIR = "qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active"


class DownstreamConfigError(ValueError):
    """Raised when settings diverge from the capacity-only migration."""


def _fields(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DownstreamConfigError(f"{label} 字段必须恰好为 {sorted(expected)}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise DownstreamConfigError(f"{label} 必须是小写 SHA256 哈希")
    return value


def _path(value: Any, parent: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DownstreamConfigError(f"{label} 必须是非空路径")
    return (parent / value).resolve()


@dataclass(frozen=True)
class DownstreamConfig:
    source_path: Path
    source_sha256: str
    upstream: AdapterSelectionConfig
    output_dir: Path
    upstream_artifacts: Mapping[str, Mapping[str, str]]
    _resolved: Mapping[str, Any]
    codebook_sizes: tuple[int, int, int] = CURRENT_SID_CODEBOOK_SIZES

    @property
    def query_view_policy(self) -> dict[str, str]:
        return dict(self.upstream.query_view_policy)

    @property
    def query_depth_dir(self) -> Path:
        return self.upstream.base.query_depth_dir

    @property
    def final_adapter_path(self) -> Path:
        return Path(self.upstream_artifacts["final_adapter_checkpoint"]["path"])

    def resolved_payload(self) -> dict[str, Any]:
        """Return inherited algorithm settings with explicit frozen inputs."""
        return copy.deepcopy(dict(self._resolved))

    def signature(self) -> str:
        payload = json.dumps(
            self.resolved_payload(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_downstream_config(
    path: Path, *, payload: Mapping[str, Any] | None = None,
) -> DownstreamConfig:
    """Validate configuration only; artifact contents need a separate stage gate."""
    source_path = path.resolve()
    try:
        source_bytes = source_path.read_bytes()
        raw = yaml.safe_load(source_bytes) if payload is None else copy.deepcopy(payload)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise DownstreamConfigError(f"配置读取失败：{source_path}") from error
    root = _fields(raw, {
        "schema_version", "base_p3a_full_config", "base_p3a_full_config_sha256",
        "base_category_config_sha256", "downstream", "upstream_artifacts",
    }, "配置根")
    if root["schema_version"] != SCHEMA_VERSION:
        raise DownstreamConfigError(f"schema_version 必须为 {SCHEMA_VERSION}")
    base_path = _path(root["base_p3a_full_config"], source_path.parent, "base_p3a_full_config")
    if base_path.parent != source_path.parent:
        raise DownstreamConfigError("上游配置必须位于同一 configs 目录")
    try:
        upstream = load_adapter_selection_config(base_path)
    except (ValueError, OSError) as error:
        raise DownstreamConfigError(f"冻结上游配置验证失败：{error}") from error
    category = upstream.base.category_config
    for key, actual in (
        ("base_p3a_full_config_sha256", upstream.source_sha256),
        ("base_category_config_sha256", category.source_sha256),
    ):
        if _sha256(root[key], key) != actual:
            raise DownstreamConfigError(f"{key} 哈希不匹配；禁止覆盖冻结上游配置")
    downstream = _fields(root["downstream"], {
        "codebook_sizes", "output_dir", "stop_after_phase", "query_view_policy",
    }, "downstream")
    if downstream["codebook_sizes"] != list(CURRENT_SID_CODEBOOK_SIZES) or any(
        type(value) is not int for value in downstream["codebook_sizes"]
    ):
        raise DownstreamConfigError("本轮 codebook_sizes 必须为 [512, 512, 512]")
    if downstream["query_view_policy"] != dict(upstream.query_view_policy):
        raise DownstreamConfigError("Query view 必须保持 D1/D2 Raw BGE、D3 FINAL")
    if downstream["stop_after_phase"] != "P8-CAT":
        raise DownstreamConfigError("停止边界必须为 P8-CAT，不自动进入 Final PID/SFT")
    project_root = upstream.base.project_root
    output_dir = _path(downstream["output_dir"], project_root, "downstream.output_dir")
    if output_dir != (project_root / OUTPUT_DIR).resolve():
        raise DownstreamConfigError(f"本轮 output_dir 必须为 {OUTPUT_DIR}")
    expected_paths = {
        "p2_5_manifest": upstream.base.query_depth_dir / "manifest.json",
        "d3_query_cache_manifest": upstream.base.output_dir / "query_cache_exact_full/manifest.json",
        "p3a_full_manifest": upstream.output_dir / "manifest.json",
        "final_adapter_manifest": upstream.output_dir / "final/manifest.json",
        "final_adapter_checkpoint": upstream.output_dir / "final/query_adapter_exact_final.pt",
    }
    artifacts = _fields(root["upstream_artifacts"], set(expected_paths), "upstream_artifacts")
    resolved_artifacts = {}
    for name, expected_path in expected_paths.items():
        entry = _fields(artifacts[name], {"path", "sha256"}, name)
        artifact_path = _path(entry["path"], project_root, name)
        if artifact_path != expected_path.resolve():
            raise DownstreamConfigError(f"{name} 必须引用冻结上游路径：{expected_path}")
        resolved_artifacts[name] = {
            "path": str(artifact_path), "sha256": _sha256(entry["sha256"], name),
        }
    if resolved_artifacts["p2_5_manifest"]["sha256"] != upstream.base.p2_5_manifest_sha256:
        raise DownstreamConfigError("P2.5 manifest 哈希与冻结 P3A 配置不一致")

    # The original YAML remains byte-identical for historical manifest validation.
    category_bytes = category.source_path.read_bytes()
    if hashlib.sha256(category_bytes).hexdigest() != category.source_sha256:
        raise DownstreamConfigError("加载期间上游类别配置哈希发生变化")
    resolved = yaml.safe_load(category_bytes)
    resolved["schema_version"] = SCHEMA_VERSION
    resolved["method"]["name"] = "qg_prqk_v2_1_category_active_512x3"
    resolved["paths"]["project_root"] = str(project_root)
    resolved["paths"]["output_dir"] = str(output_dir)
    resolved["identifier"]["codebooks"] = list(downstream["codebook_sizes"])
    resolved["query_embedding"] = {
        "current_status": "P3A_FULL_COMPLETED",
        "query_view_policy": dict(upstream.query_view_policy),
        "encoding": upstream.base.resolved_payload()["query_embedding"],
    }
    resolved["upstream_artifacts"] = resolved_artifacts
    resolved["provenance"] = {
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "base_p3a_full_config_sha256": upstream.source_sha256,
        "base_p3a_full_signature": upstream.signature(),
        "base_category_config_sha256": category.source_sha256,
    }
    return DownstreamConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        upstream=upstream, output_dir=output_dir,
        upstream_artifacts=resolved_artifacts, _resolved=resolved,
    )

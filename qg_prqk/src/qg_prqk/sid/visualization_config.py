"""Configuration for read-only SID visualization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.artifacts import sha256_file


SCHEMA_VERSION = "qg-prqk-sid-visualization-config-v1"
PROTOCOL_ID = "sid_visualization_a0_a4_nogid_full_v1"
METHOD_ORDER = ("A0_POI_ONLY", "A4_GID_PARENT", "A4_NOGID_S1S2_PARENT")
SOURCE_NAMES = {
    "active_embedding",
    "p5_a0",
    "p6_a4_s1_s2",
    "p7_a4_gid",
    "p7_a4_nogid",
    "p8_a4_gid",
    "p8_a4_nogid",
}


class SIDVisualizationConfigError(ValueError):
    """Raised when the visualization protocol or a frozen source changed."""


@dataclass(frozen=True)
class SIDVisualizationMethod:
    """Paths for one frozen SID method."""

    name: str
    display_name: str
    sid_path: Path
    codebook_paths: tuple[Path, Path, Path]
    residual_input_paths: tuple[Path, Path, Path]


@dataclass(frozen=True)
class SIDVisualizationConfig:
    """Resolved three-method visualization contract."""

    source_path: Path
    source_sha256: str
    project_root: Path
    output_dir: Path
    rows: Mapping[str, int]
    frozen_manifests: Mapping[str, Mapping[str, str]]
    poi_embedding_path: Path
    poi_metadata_path: Path
    methods: tuple[SIDVisualizationMethod, ...]
    sampling: Mapping[str, int]
    projection: Mapping[str, Any]
    figures: Mapping[str, Any]

    def signature(self) -> str:
        """Return a stable signature for all semantic settings."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "source_sha256": self.source_sha256,
            "rows": dict(self.rows),
            "frozen_manifests": {
                name: dict(value) for name, value in self.frozen_manifests.items()
            },
            "poi_embedding_path": str(self.poi_embedding_path),
            "poi_metadata_path": str(self.poi_metadata_path),
            "methods": [
                {
                    "name": method.name,
                    "display_name": method.display_name,
                    "sid_path": str(method.sid_path),
                    "codebook_paths": [str(path) for path in method.codebook_paths],
                    "residual_input_paths": [
                        str(path) for path in method.residual_input_paths
                    ],
                }
                for method in self.methods
            ],
            "sampling": dict(self.sampling),
            "projection": dict(self.projection),
            "figures": dict(self.figures),
            "output_dir": str(self.output_dir),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SIDVisualizationConfigError(f"{label} 必须是 mapping")
    return value


def _resolve_qg_output(project_root: Path, value: Any, label: str) -> Path:
    path = (project_root / str(value)).resolve()
    output_root = (project_root / "qg_prqk/outputs").resolve()
    if not path.is_relative_to(output_root):
        raise SIDVisualizationConfigError(f"{label} 必须位于 qg_prqk/outputs")
    return path


def _path_tuple(
    project_root: Path, value: Any, label: str
) -> tuple[Path, Path, Path]:
    if not isinstance(value, list) or len(value) != 3:
        raise SIDVisualizationConfigError(f"{label} 必须有 3 个路径")
    paths = tuple(
        _resolve_qg_output(project_root, item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return paths  # type: ignore[return-value]


def load_sid_visualization_config(path: Path) -> SIDVisualizationConfig:
    """Load and validate the frozen A0/A4/NoGID visualization protocol."""
    source_path = path.resolve()
    project_root = source_path.parents[2]
    try:
        source_bytes = source_path.read_bytes()
        root = _mapping(yaml.safe_load(source_bytes), "SID 可视化配置根")
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise SIDVisualizationConfigError(f"配置读取失败：{source_path}") from error

    if root.get("schema_version") != SCHEMA_VERSION:
        raise SIDVisualizationConfigError("SID 可视化 schema_version 不匹配")
    if root.get("protocol_id") != PROTOCOL_ID:
        raise SIDVisualizationConfigError("SID 可视化 protocol_id 不匹配")
    if root.get("scope") != "READ_ONLY_COMPARE_A0_A4_GID_A4_NOGID":
        raise SIDVisualizationConfigError("SID 可视化 scope 不匹配")

    rows = _mapping(root.get("rows"), "rows")
    expected_rows = {
        "poi": 716_245,
        "embedding_dim": 1024,
        "levels": 3,
        "codebook_size": 512,
    }
    if dict(rows) != expected_rows:
        raise SIDVisualizationConfigError("SID 可视化行数/维数合同不匹配")

    frozen = _mapping(root.get("frozen_manifests"), "frozen_manifests")
    if set(frozen) != SOURCE_NAMES:
        raise SIDVisualizationConfigError("冻结 manifest 集合不匹配")
    resolved_frozen: dict[str, dict[str, str]] = {}
    for name, raw_entry in frozen.items():
        entry = _mapping(raw_entry, f"frozen_manifests.{name}")
        manifest_path = _resolve_qg_output(
            project_root, entry.get("path"), f"frozen_manifests.{name}.path"
        )
        expected_sha = str(entry.get("sha256"))
        if len(expected_sha) != 64 or sha256_file(manifest_path) != expected_sha:
            raise SIDVisualizationConfigError(f"冻结 manifest 哈希不匹配：{name}")
        resolved_frozen[name] = {
            "path": str(manifest_path),
            "sha256": expected_sha,
        }

    embedding_path = _resolve_qg_output(
        project_root, root.get("poi_embedding_path"), "poi_embedding_path"
    )
    metadata_path = _resolve_qg_output(
        project_root, root.get("poi_metadata_path"), "poi_metadata_path"
    )
    if not embedding_path.is_file() or not metadata_path.is_file():
        raise SIDVisualizationConfigError("POI embedding 或 metadata 不存在")

    raw_methods = _mapping(root.get("methods"), "methods")
    if tuple(raw_methods) != METHOD_ORDER:
        raise SIDVisualizationConfigError("方法顺序必须固定为 A0/原 A4/NoGID A4")
    methods: list[SIDVisualizationMethod] = []
    for method_name in METHOD_ORDER:
        entry = _mapping(raw_methods[method_name], f"methods.{method_name}")
        methods.append(
            SIDVisualizationMethod(
                name=method_name,
                display_name=str(entry.get("display_name")),
                sid_path=_resolve_qg_output(
                    project_root,
                    entry.get("sid_path"),
                    f"methods.{method_name}.sid_path",
                ),
                codebook_paths=_path_tuple(
                    project_root,
                    entry.get("codebook_paths"),
                    f"methods.{method_name}.codebook_paths",
                ),
                residual_input_paths=_path_tuple(
                    project_root,
                    entry.get("residual_input_paths"),
                    f"methods.{method_name}.residual_input_paths",
                ),
            )
        )

    sampling = _mapping(root.get("sampling"), "sampling")
    expected_sampling = {
        "seed": 42,
        "prefix_pairs_per_level": 20_000,
        "representative_codes_per_level": 5,
        "rows_per_representative_code": 100,
        "semantic_examples_per_prefix": 3,
        "top_prefixes_per_level": 3,
    }
    if dict(sampling) != expected_sampling:
        raise SIDVisualizationConfigError("抽样合同不匹配")

    projection = _mapping(root.get("projection"), "projection")
    expected_projection = {
        "method": "pca50_then_tsne",
        "cpu_threads": 8,
        "pca_components": 50,
        "tsne_perplexity": 30.0,
        "tsne_max_iter": 1000,
        "tsne_learning_rate": "auto",
        "tsne_metric": "euclidean",
    }
    if dict(projection) != expected_projection:
        raise SIDVisualizationConfigError("降维合同不匹配")

    figures = _mapping(root.get("figures"), "figures")
    expected_figure_names = {
        "code_distribution": "sid_code_distribution.png",
        "residual_semantic_map": "sid_residual_semantic_map.png",
        "prefix_semantics": "sid_prefix_semantics.png",
    }
    if (
        figures.get("dpi") != 180
        or figures.get("format") != "png"
        or {name: figures.get(name) for name in expected_figure_names}
        != expected_figure_names
    ):
        raise SIDVisualizationConfigError("图片输出合同不匹配")
    if root.get("overwrite") is not False or root.get("downstream_started") is not False:
        raise SIDVisualizationConfigError("可视化必须只读且 overwrite=false")

    output_dir = _resolve_qg_output(project_root, root.get("output_dir"), "output_dir")
    figures_root = (project_root / "qg_prqk/outputs/figures").resolve()
    if output_dir.parent != figures_root:
        raise SIDVisualizationConfigError("输出必须是 qg_prqk/outputs/figures 的直接子目录")

    return SIDVisualizationConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        project_root=project_root,
        output_dir=output_dir,
        rows={name: int(value) for name, value in rows.items()},
        frozen_manifests=resolved_frozen,
        poi_embedding_path=embedding_path,
        poi_metadata_path=metadata_path,
        methods=tuple(methods),
        sampling={name: int(value) for name, value in sampling.items()},
        projection=dict(projection),
        figures=dict(figures),
    )

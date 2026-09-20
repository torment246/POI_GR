"""Configuration for category-aware query supervision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "qg-prqk-category-config-v2.1"
SUPPORT_TRANSFORM = "normalized_log1p_min"


class CategoryConfigError(ValueError):
    """Raised when the P2.5-CAT configuration violates its frozen contract."""


@dataclass(frozen=True)
class DepthThreshold:
    min_query_count: int
    min_concentration: float
    max_normalized_entropy: float


@dataclass(frozen=True)
class SensitivitySetting:
    name: str
    d2_fine: DepthThreshold
    d1_coarse: DepthThreshold


@dataclass(frozen=True)
class CategoryPaths:
    project_root: Path
    poi_catalog: Path
    poi_embeddings: Path
    poi_ids: Path
    embedding_manifest: Path
    p2_query_stats: Path
    output_root: Path
    output_dir: Path
    category_indices: Path
    category_vocab: Path
    category_source_manifest: Path


@dataclass(frozen=True)
class FrozenCategoryInputs:
    poi_rows: int
    poi_embedding_dim: int
    poi_embedding_dtype: str
    poi_embedding_normalized: bool
    poi_embedding_manifest_sha256: str
    poi_ids_sha256: str
    p2_manifest_sha256: str
    category_indices_sha256: str
    category_vocab_data_sha256: str
    category_source_manifest_sha256: str


@dataclass(frozen=True)
class CategoryContract:
    fine_source_column: str
    fine_transform: str
    fine_expected_count: int
    coarse_source_column: str
    coarse_transform: str
    coarse_expected_count: int
    readable_path_column: str
    require_full_coverage: bool
    require_fine_to_one_coarse: bool
    unknown_or_other_is_legal_category: bool
    use_as_embedding_input: bool
    mapping_row_order: str
    output_mapping_name: str


@dataclass(frozen=True)
class QueryDepthContract:
    use_existing_p2_exact_core: bool
    d2_fine: DepthThreshold
    d1_coarse: DepthThreshold
    support_cap: int
    support_transform: str
    reliability_gamma: float
    sensitivity: tuple[SensitivitySetting, ...]


@dataclass(frozen=True)
class CategoryRuntime:
    overwrite: bool
    require_sample_and_medium_gate_before_full: bool
    sample_query_limit: int
    medium_query_limit: int


@dataclass(frozen=True)
class CategoryBuildConfig:
    source_path: Path
    source_sha256: str
    method_name: str
    method_version: str
    city: str
    paths: CategoryPaths
    frozen: FrozenCategoryInputs
    category: CategoryContract
    query_depth: QueryDepthContract
    runtime: CategoryRuntime

    @property
    def query_depth_output_dir(self) -> Path:
        return self.paths.output_dir / "query_depth"

    @property
    def gate_dir(self) -> Path:
        return self.paths.output_dir / "p2_5_cat_gates"

    def resolved_payload(self) -> dict[str, Any]:
        """Return the exact P2.5-relevant configuration payload."""

        threshold = lambda value: {  # noqa: E731 - compact serialization helper
            "min_query_count": value.min_query_count,
            "min_concentration": value.min_concentration,
            "max_normalized_entropy": value.max_normalized_entropy,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "method": {
                "name": self.method_name,
                "version": self.method_version,
                "city": self.city,
            },
            "paths": {
                key: str(value)
                for key, value in vars(self.paths).items()
            },
            "frozen_inputs": vars(self.frozen),
            "category": vars(self.category),
            "query_depth": {
                "d3_use_existing_p2_exact_core": (
                    self.query_depth.use_existing_p2_exact_core
                ),
                "d2_fine_category": threshold(self.query_depth.d2_fine),
                "d1_coarse_category": threshold(self.query_depth.d1_coarse),
                "support_cap": self.query_depth.support_cap,
                "support_transform": self.query_depth.support_transform,
                "reliability_gamma": self.query_depth.reliability_gamma,
                "sensitivity": [
                    {
                        "name": setting.name,
                        "d2_fine_category": threshold(setting.d2_fine),
                        "d1_coarse_category": threshold(setting.d1_coarse),
                    }
                    for setting in self.query_depth.sensitivity
                ],
            },
            "runtime": vars(self.runtime),
        }

    def signature(self) -> str:
        payload = json.dumps(
            self.resolved_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CategoryConfigError(f"{label} 必须是 mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise CategoryConfigError(f"{label} 缺少 {key}")
    return mapping[key]


def _string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _required(mapping, key, label)
    if not isinstance(value, str) or not value.strip():
        raise CategoryConfigError(f"{label}.{key} 必须是非空字符串")
    return value


def _boolean(mapping: Mapping[str, Any], key: str, label: str) -> bool:
    value = _required(mapping, key, label)
    if not isinstance(value, bool):
        raise CategoryConfigError(f"{label}.{key} 必须是 bool")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = _required(mapping, key, label)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CategoryConfigError(f"{label}.{key} 必须是正整数")
    return value


def _probability(mapping: Mapping[str, Any], key: str, label: str) -> float:
    value = _required(mapping, key, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CategoryConfigError(f"{label}.{key} 必须是数值")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise CategoryConfigError(f"{label}.{key} 必须位于 [0,1]")
    return result


def _sha256(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _string(mapping, key, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CategoryConfigError(f"{label}.{key} 必须是小写 SHA256")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CategoryConfigError(f"{label} 必须位于 {root} 下：{path}") from error


def _threshold(value: Any, label: str) -> DepthThreshold:
    mapping = _mapping(value, label)
    return DepthThreshold(
        min_query_count=_positive_int(mapping, "min_query_count", label),
        min_concentration=_probability(mapping, "min_concentration", label),
        max_normalized_entropy=_probability(
            mapping, "max_normalized_entropy", label
        ),
    )


def _sensitivity(value: Any) -> tuple[SensitivitySetting, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CategoryConfigError("query_depth.sensitivity 必须是 list")
    settings: list[SensitivitySetting] = []
    for index, raw in enumerate(value):
        mapping = _mapping(raw, f"query_depth.sensitivity[{index}]")
        name = _string(mapping, "name", f"query_depth.sensitivity[{index}]")
        settings.append(
            SensitivitySetting(
                name=name,
                d2_fine=_threshold(
                    _required(mapping, "d2_fine_category", name),
                    f"sensitivity.{name}.d2_fine_category",
                ),
                d1_coarse=_threshold(
                    _required(mapping, "d1_coarse_category", name),
                    f"sensitivity.{name}.d1_coarse_category",
                ),
            )
        )
    names = [setting.name for setting in settings]
    if len(settings) < 3 or len(set(names)) != len(names) or "baseline" not in names:
        raise CategoryConfigError("sensitivity 至少包含唯一命名的 relaxed/baseline/strict")
    return tuple(settings)


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise CategoryConfigError(f"{label} 必须为 {expected!r}，实际 {actual!r}")


def load_category_config(path: Path) -> CategoryBuildConfig:
    """Load only the frozen v2.1 fields needed by P2.5-CAT."""

    source_path = path.resolve()
    try:
        raw_bytes = source_path.read_bytes()
        raw = yaml.safe_load(raw_bytes)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CategoryConfigError(f"配置读取失败：{source_path}") from error
    root = _mapping(raw, "配置根")
    _assert_equal(root.get("schema_version"), SCHEMA_VERSION, "schema_version")

    method = _mapping(_required(root, "method", "配置根"), "method")
    paths_raw = _mapping(_required(root, "paths", "配置根"), "paths")
    frozen_raw = _mapping(
        _required(root, "frozen_inputs", "配置根"), "frozen_inputs"
    )
    category_raw = _mapping(_required(root, "category", "配置根"), "category")
    query_raw = _mapping(_required(root, "query_depth", "配置根"), "query_depth")
    runtime_raw = _mapping(_required(root, "runtime", "配置根"), "runtime")

    _assert_equal(_string(method, "version", "method"), "v2.1-CAT", "method.version")
    _assert_equal(_string(method, "task", "method"), "exact_single_poi_retrieval", "method.task")
    _assert_equal(_string(method, "city", "method"), "北京", "method.city")

    project_root = _resolve(
        source_path.parent, _string(paths_raw, "project_root", "paths")
    )
    existing = _mapping(
        _required(category_raw, "existing_mapping", "category"),
        "category.existing_mapping",
    )
    paths = CategoryPaths(
        project_root=project_root,
        poi_catalog=_resolve(project_root, _string(paths_raw, "poi_catalog", "paths")),
        poi_embeddings=_resolve(
            project_root, _string(paths_raw, "poi_embeddings", "paths")
        ),
        poi_ids=_resolve(project_root, _string(paths_raw, "poi_ids", "paths")),
        embedding_manifest=_resolve(
            project_root, _string(paths_raw, "embedding_manifest", "paths")
        ),
        p2_query_stats=_resolve(
            project_root, _string(paths_raw, "p2_query_stats", "paths")
        ),
        output_root=_resolve(
            project_root, _string(paths_raw, "output_root", "paths")
        ),
        output_dir=_resolve(project_root, _string(paths_raw, "output_dir", "paths")),
        category_indices=_resolve(
            project_root,
            _string(existing, "category_indices", "category.existing_mapping"),
        ),
        category_vocab=_resolve(
            project_root,
            _string(existing, "category_vocab", "category.existing_mapping"),
        ),
        category_source_manifest=_resolve(
            project_root,
            _string(existing, "source_manifest", "category.existing_mapping"),
        ),
    )
    qg_output_root = (project_root / "qg_prqk/outputs").resolve()
    _inside(paths.output_root, qg_output_root, "paths.output_root")
    _inside(paths.output_dir, qg_output_root, "paths.output_dir")
    if paths.output_root != qg_output_root:
        raise CategoryConfigError("paths.output_root 必须精确指向 qg_prqk/outputs")

    fine = _mapping(_required(category_raw, "fine_category_id", "category"), "category.fine_category_id")
    coarse = _mapping(_required(category_raw, "coarse_category_id", "category"), "category.coarse_category_id")
    category = CategoryContract(
        fine_source_column=_string(fine, "source_column", "category.fine_category_id"),
        fine_transform=_string(fine, "transform", "category.fine_category_id"),
        fine_expected_count=_positive_int(fine, "expected_count", "category.fine_category_id"),
        coarse_source_column=_string(coarse, "source_column", "category.coarse_category_id"),
        coarse_transform=_string(coarse, "transform", "category.coarse_category_id"),
        coarse_expected_count=_positive_int(coarse, "expected_count", "category.coarse_category_id"),
        readable_path_column=_string(category_raw, "readable_path_column", "category"),
        require_full_coverage=_boolean(category_raw, "require_full_coverage", "category"),
        require_fine_to_one_coarse=_boolean(category_raw, "require_fine_to_one_coarse", "category"),
        unknown_or_other_is_legal_category=_boolean(
            category_raw, "unknown_or_other_is_legal_category", "category"
        ),
        use_as_embedding_input=_boolean(category_raw, "use_as_embedding_input", "category"),
        mapping_row_order=_string(
            existing, "category_indices_row_order", "category.existing_mapping"
        ),
        output_mapping_name=_string(category_raw, "p2_5_output_mapping", "category"),
    )
    expected_contract = {
        "fine_source_column": "category_code",
        "fine_transform": "identity",
        "coarse_source_column": "category_code",
        "coarse_transform": "first_2_digits",
        "readable_path_column": "category",
        "require_full_coverage": True,
        "require_fine_to_one_coarse": True,
        "use_as_embedding_input": False,
        "mapping_row_order": "BGE-M3 poi_ids.jsonl order",
        "output_mapping_name": "category_mapping.parquet",
    }
    for key, expected in expected_contract.items():
        _assert_equal(getattr(category, key), expected, f"category.{key}")

    d3 = _mapping(_required(query_raw, "d3", "query_depth"), "query_depth.d3")
    query_depth = QueryDepthContract(
        use_existing_p2_exact_core=_boolean(
            d3, "use_existing_p2_exact_core", "query_depth.d3"
        ),
        d2_fine=_threshold(
            _required(query_raw, "d2_fine_category", "query_depth"),
            "query_depth.d2_fine_category",
        ),
        d1_coarse=_threshold(
            _required(query_raw, "d1_coarse_category", "query_depth"),
            "query_depth.d1_coarse_category",
        ),
        support_cap=_positive_int(query_raw, "support_cap", "query_depth"),
        support_transform=_string(query_raw, "support_transform", "query_depth"),
        reliability_gamma=float(
            _required(query_raw, "reliability_gamma", "query_depth")
        ),
        sensitivity=_sensitivity(_required(query_raw, "sensitivity", "query_depth")),
    )
    _assert_equal(
        query_depth.use_existing_p2_exact_core,
        True,
        "query_depth.d3.use_existing_p2_exact_core",
    )
    _assert_equal(
        query_depth.support_transform, SUPPORT_TRANSFORM, "query_depth.support_transform"
    )
    if query_depth.reliability_gamma <= 0:
        raise CategoryConfigError("query_depth.reliability_gamma 必须大于 0")
    baseline = next(
        setting for setting in query_depth.sensitivity if setting.name == "baseline"
    )
    if baseline.d2_fine != query_depth.d2_fine or baseline.d1_coarse != query_depth.d1_coarse:
        raise CategoryConfigError("sensitivity.baseline 必须与正式 D2/D1 阈值一致")

    frozen = FrozenCategoryInputs(
        poi_rows=_positive_int(frozen_raw, "poi_rows", "frozen_inputs"),
        poi_embedding_dim=_positive_int(
            frozen_raw, "poi_embedding_dim", "frozen_inputs"
        ),
        poi_embedding_dtype=_string(
            frozen_raw, "poi_embedding_dtype", "frozen_inputs"
        ),
        poi_embedding_normalized=_boolean(
            frozen_raw, "poi_embedding_normalized", "frozen_inputs"
        ),
        poi_embedding_manifest_sha256=_sha256(
            frozen_raw, "poi_embedding_manifest_sha256", "frozen_inputs"
        ),
        poi_ids_sha256=_sha256(frozen_raw, "poi_ids_sha256", "frozen_inputs"),
        p2_manifest_sha256=_sha256(
            frozen_raw, "p2_manifest_sha256", "frozen_inputs"
        ),
        category_indices_sha256=_sha256(
            existing, "category_indices_sha256", "category.existing_mapping"
        ),
        category_vocab_data_sha256=_sha256(
            existing, "category_vocab_data_sha256", "category.existing_mapping"
        ),
        category_source_manifest_sha256=_sha256(
            existing, "source_manifest_sha256", "category.existing_mapping"
        ),
    )
    _assert_equal(
        _boolean(frozen_raw, "reencode_poi_text", "frozen_inputs"),
        False,
        "frozen_inputs.reencode_poi_text",
    )
    _assert_equal(
        _boolean(frozen_raw, "append_category_to_poi_text", "frozen_inputs"),
        False,
        "frozen_inputs.append_category_to_poi_text",
    )

    runtime = CategoryRuntime(
        overwrite=_boolean(runtime_raw, "overwrite", "runtime"),
        require_sample_and_medium_gate_before_full=_boolean(
            runtime_raw, "require_sample_and_medium_gate_before_full", "runtime"
        ),
        sample_query_limit=_positive_int(
            runtime_raw, "p2_5_sample_query_limit", "runtime"
        ),
        medium_query_limit=_positive_int(
            runtime_raw, "p2_5_medium_query_limit", "runtime"
        ),
    )
    if runtime.overwrite:
        raise CategoryConfigError("P2.5-CAT canonical runtime 必须保持 overwrite=false")
    if runtime.sample_query_limit >= runtime.medium_query_limit:
        raise CategoryConfigError("sample_query_limit 必须小于 medium_query_limit")

    return CategoryBuildConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        method_name=_string(method, "name", "method"),
        method_version=_string(method, "version", "method"),
        city=_string(method, "city", "method"),
        paths=paths,
        frozen=frozen,
        category=category,
        query_depth=query_depth,
        runtime=runtime,
    )

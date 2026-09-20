"""Historical v1.1 input contract; not the current SID configuration.

本模块保留早期 1024×3 配置及上游数据类，不决定当前 SID 码本容量。
当前 512×3 SID：sid/pipeline_config.py 的 load_downstream_config，配置为
qg_prqk/configs/qg_prqk_v2_1_category_active_512x3.yaml。
历史 YAML 和签名保持不变；load_config 仅为 load_legacy_config 的兼容名称。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CONFIG_SCHEMA_VERSION = "qg-prqk-config-v1"
LEGACY_V1_CODEBOOK_SIZES = (1024, 1024, 1024)
EXPECTED_HARD_GRAPH_WEIGHTS = {
    "bge": 0.35,
    "name_alias": 0.30,
    "category": 0.15,
    "address": 0.10,
    "geo": 0.10,
}
FORBIDDEN_POI_FIELDS = {
    "brand",
    "canonical_poi_id",
    "landmark",
    "category_l1",
    "category_l2",
    "category_l3",
}
AUDIT_ONLY_POI_FIELDS = {"area", "layer", "click_score"}


class QGPRQKConfigError(ValueError):
    """Raised when the QG-PRQK configuration violates the frozen contract."""


@dataclass(frozen=True)
class ProjectConfig:
    method: str
    version: str
    city: str
    seed: int
    codebook_sizes: tuple[int, int, int]


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    poi_catalog: Path
    poi_embeddings: Path
    poi_ids: Path
    embedding_manifest: Path
    sft_data_dir: Path
    sft_manifest: Path
    output_root: Path
    output_dir: Path


@dataclass(frozen=True)
class DataContractConfig:
    expected_poi_rows: int
    embedding_dim: int
    embedding_dtype: str
    embedding_normalized: bool
    pca: str | None
    poi_catalog_fields: tuple[str, ...]
    poi_required_fields: tuple[str, ...]
    poi_method_fields: tuple[str, ...]
    poi_audit_only_fields: tuple[str, ...]
    forbidden_poi_fields: tuple[str, ...]
    train_required_fields: tuple[str, ...]
    train_split: str
    train_date_start: str
    train_date_end: str
    valid_date: str
    test_date: str


@dataclass(frozen=True)
class IdentifierConfig:
    gid_tokens: int
    sid_tokens: int
    gid_codebook_size: int
    sid_codebook_sizes: tuple[int, int, int]
    base_token_order: tuple[str, ...]
    singleton_length: int
    collision_length: int
    dedup_capacity: int
    dedup_assignment: str


@dataclass(frozen=True)
class HardGraphConfig:
    weights: dict[str, float]
    top_k: int
    threshold: float
    excluded_fields: tuple[str, ...]


@dataclass(frozen=True)
class MethodConfig:
    poi_embedding_source: str
    metric: str
    residual: str
    remove_common_direction: bool
    gid_order: str
    final_pid: str
    query_normalization: str
    query_weights: tuple[float, float, float]
    hard_graph: HardGraphConfig


@dataclass(frozen=True)
class QueryStatsConfig:
    num_shards: int
    buffer_rows_per_shard: int
    read_batch_rows: int
    min_query_count: int
    min_pair_count: int
    min_top1_share: float
    min_margin: float
    max_normalized_entropy: float
    false_negative_min_share: float
    false_negative_min_count: int


@dataclass(frozen=True)
class QueryEmbeddingConfig:
    model_path: Path
    backend: str
    device: str
    batch_size: int
    encode_buffer_size: int
    max_seq_length: int
    torch_dtype: str
    attention: str | None
    padding_side: str
    normalize_embeddings: bool
    prompt_name: str | None
    output_dtype: str
    checkpoint_interval_batches: int


@dataclass(frozen=True)
class QueryAdapterConfig:
    enabled: bool
    bottleneck: int
    residual_scale: float
    dropout: float
    optimizer: str
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    precision: str
    max_grad_norm: float
    temperature: float
    dev_fraction: float
    ann_top_k: int
    semantic_ann_negatives: int
    lexical_metadata_negatives: int
    local_geo_negatives: int


@dataclass(frozen=True)
class RuntimeConfig:
    sample_limit: int
    overwrite: bool
    resume: bool
    show_progress: bool
    chunk_rows: int


@dataclass(frozen=True)
class FrozenInputs:
    embedding_manifest_sha256: str
    embedding_fingerprint: str
    poi_ids_sha256: str
    sft_manifest_sha256: str
    train_sha256: str
    valid_sha256: str
    test_sha256: str


@dataclass(frozen=True)
class QGPRQKConfig:
    source_path: Path
    source_sha256: str
    project: ProjectConfig
    paths: PathConfig
    data_contracts: DataContractConfig
    identifier: IdentifierConfig
    method: MethodConfig
    query_stats: QueryStatsConfig
    query_embedding: QueryEmbeddingConfig
    query_adapter: QueryAdapterConfig
    runtime: RuntimeConfig
    frozen_inputs: FrozenInputs
    deferred_parameters: dict[str, Any]

    def resolved_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot used by artifact manifests."""

        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["paths"] = {
            key: str(value) for key, value in asdict(self.paths).items()
        }
        payload["project"]["codebook_sizes"] = list(
            self.project.codebook_sizes
        )
        payload["identifier"]["sid_codebook_sizes"] = list(
            self.identifier.sid_codebook_sizes
        )
        payload["identifier"]["base_token_order"] = list(
            self.identifier.base_token_order
        )
        payload["method"]["query_weights"] = list(
            self.method.query_weights
        )
        payload["method"]["hard_graph"]["excluded_fields"] = list(
            self.method.hard_graph.excluded_fields
        )
        payload["query_embedding"]["model_path"] = str(
            self.query_embedding.model_path
        )
        for key in (
            "poi_catalog_fields",
            "poi_required_fields",
            "poi_method_fields",
            "poi_audit_only_fields",
            "forbidden_poi_fields",
            "train_required_fields",
        ):
            payload["data_contracts"][key] = list(
                getattr(self.data_contracts, key)
            )
        return payload

    def signature(self) -> str:
        """Hash the resolved semantic configuration deterministically."""

        payload = self.resolved_payload()
        for runtime_control in (
            "show_progress",
            "sample_limit",
            "resume",
            "overwrite",
        ):
            payload["runtime"].pop(runtime_control, None)
        return hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QGPRQKConfigError(f"{name} 必须是 YAML mapping")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QGPRQKConfigError(f"{name} 必须是正整数")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise QGPRQKConfigError(f"{name} 必须是布尔值")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QGPRQKConfigError(f"{name} 必须是非空字符串")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise QGPRQKConfigError(f"{name} 必须是字符串数组")
    values = tuple(_string(item, name) for item in value)
    if len(set(values)) != len(values):
        raise QGPRQKConfigError(f"{name} 不得包含重复字段")
    return values


def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise QGPRQKConfigError(f"{name} 必须是正整数数组")
    return tuple(_positive_int(item, name) for item in value)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QGPRQKConfigError(f"{name} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise QGPRQKConfigError(f"{name} 必须是有限数值")
    return result


def _float_tuple(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise QGPRQKConfigError(f"{name} 必须是数值数组")
    values = tuple(_number(item, name) for item in value)
    if len(values) != length:
        raise QGPRQKConfigError(f"{name} 必须包含 {length} 项")
    return values


def _sha256(value: Any, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise QGPRQKConfigError(f"{name} 必须是 64 位 SHA256")
    return text


def _resolve_path(value: Any, base: Path, name: str) -> Path:
    raw = os.fspath(value) if isinstance(value, os.PathLike) else _string(value, name)
    if not raw.strip():
        raise QGPRQKConfigError(f"{name} 必须是非空路径")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((path, parent)) == str(parent)
    except ValueError:
        return False


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise QGPRQKConfigError(f"配置不存在：{path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise QGPRQKConfigError(f"配置读取失败：{path}") from error
    return _mapping(value, "配置根节点")


def _load_project(raw: dict[str, Any]) -> ProjectConfig:
    sizes = _positive_int_tuple(
        raw.get("codebook_sizes"), "project.codebook_sizes"
    )
    if sizes != LEGACY_V1_CODEBOOK_SIZES:
        raise QGPRQKConfigError(
            "历史 v1.1 的 project.codebook_sizes 必须固定为 [1024, 1024, 1024]；"
            "当前 512×3 SID 请使用 sid.pipeline_config.load_downstream_config"
        )
    project = ProjectConfig(
        method=_string(raw.get("method"), "project.method"),
        version=_string(raw.get("version"), "project.version"),
        city=_string(raw.get("city"), "project.city"),
        seed=_positive_int(raw.get("seed"), "project.seed"),
        codebook_sizes=sizes,
    )
    if project.method != "qg_prqk" or project.version != "v1.1":
        raise QGPRQKConfigError("方法和版本必须固定为 qg_prqk/v1.1")
    if project.city not in {"北京", "北京市"}:
        raise QGPRQKConfigError("P1 只允许北京单城配置")
    return project


def _load_paths(
    raw: dict[str, Any], config_path: Path, output_dir: Path | None
) -> PathConfig:
    project_root = _resolve_path(
        raw.get("project_root"), config_path.parent, "paths.project_root"
    )

    def resolve(key: str) -> Path:
        return _resolve_path(raw.get(key), project_root, f"paths.{key}")

    output_root = resolve("output_root")
    resolved_output = (
        _resolve_path(output_dir, project_root, "output_dir override")
        if output_dir is not None
        else resolve("output_dir")
    )
    expected_output_root = (project_root / "qg_prqk/outputs").resolve()
    if output_root != expected_output_root:
        raise QGPRQKConfigError("paths.output_root 必须固定为 qg_prqk/outputs")
    if not _is_within(resolved_output, output_root) or resolved_output == output_root:
        raise QGPRQKConfigError("output_dir 必须是 paths.output_root 下的独立子目录")
    return PathConfig(
        project_root=project_root,
        poi_catalog=resolve("poi_catalog"),
        poi_embeddings=resolve("poi_embeddings"),
        poi_ids=resolve("poi_ids"),
        embedding_manifest=resolve("embedding_manifest"),
        sft_data_dir=resolve("sft_data_dir"),
        sft_manifest=resolve("sft_manifest"),
        output_root=output_root,
        output_dir=resolved_output,
    )


def _load_data_contracts(raw: dict[str, Any]) -> DataContractConfig:
    config = DataContractConfig(
        expected_poi_rows=_positive_int(
            raw.get("expected_poi_rows"), "data_contracts.expected_poi_rows"
        ),
        embedding_dim=_positive_int(
            raw.get("embedding_dim"), "data_contracts.embedding_dim"
        ),
        embedding_dtype=_string(
            raw.get("embedding_dtype"), "data_contracts.embedding_dtype"
        ),
        embedding_normalized=_boolean(
            raw.get("embedding_normalized"),
            "data_contracts.embedding_normalized",
        ),
        pca=raw.get("pca"),
        poi_catalog_fields=_string_tuple(
            raw.get("poi_catalog_fields"), "data_contracts.poi_catalog_fields"
        ),
        poi_required_fields=_string_tuple(
            raw.get("poi_required_fields"), "data_contracts.poi_required_fields"
        ),
        poi_method_fields=_string_tuple(
            raw.get("poi_method_fields"), "data_contracts.poi_method_fields"
        ),
        poi_audit_only_fields=_string_tuple(
            raw.get("poi_audit_only_fields"),
            "data_contracts.poi_audit_only_fields",
        ),
        forbidden_poi_fields=_string_tuple(
            raw.get("forbidden_poi_fields"),
            "data_contracts.forbidden_poi_fields",
        ),
        train_required_fields=_string_tuple(
            raw.get("train_required_fields"),
            "data_contracts.train_required_fields",
        ),
        train_split=_string(raw.get("train_split"), "data_contracts.train_split"),
        train_date_start=_string(
            raw.get("train_date_start"), "data_contracts.train_date_start"
        ),
        train_date_end=_string(
            raw.get("train_date_end"), "data_contracts.train_date_end"
        ),
        valid_date=_string(raw.get("valid_date"), "data_contracts.valid_date"),
        test_date=_string(raw.get("test_date"), "data_contracts.test_date"),
    )
    catalog = set(config.poi_catalog_fields)
    required = set(config.poi_required_fields)
    method_fields = set(config.poi_method_fields)
    audit_only = set(config.poi_audit_only_fields)
    forbidden = set(config.forbidden_poi_fields)
    if forbidden != FORBIDDEN_POI_FIELDS:
        raise QGPRQKConfigError("forbidden_poi_fields 与冻结的不存在字段不一致")
    if method_fields & forbidden:
        raise QGPRQKConfigError("方法字段不得包含冻结目录中不存在的字段")
    if not required <= catalog or not method_fields <= catalog:
        raise QGPRQKConfigError("POI required/method fields 必须来自真实 catalog 字段")
    if method_fields & audit_only:
        raise QGPRQKConfigError("方法字段不得包含不存在字段或 area/layer/click_score")
    if audit_only != AUDIT_ONLY_POI_FIELDS:
        raise QGPRQKConfigError("审计字段必须固定为 area/layer/click_score")
    if config.embedding_dim != 1024 or config.embedding_dtype != "float16":
        raise QGPRQKConfigError("冻结 BGE 必须为 1024 维 float16")
    if not config.embedding_normalized or config.pca is not None:
        raise QGPRQKConfigError("冻结 BGE 必须已归一化且不使用 PCA")
    if config.train_split != "train":
        raise QGPRQKConfigError("Query 构建只允许使用 train split")
    return config


def _load_identifier(raw: dict[str, Any]) -> IdentifierConfig:
    sizes = _positive_int_tuple(
        raw.get("sid_codebook_sizes"), "identifier.sid_codebook_sizes"
    )
    config = IdentifierConfig(
        gid_tokens=_positive_int(raw.get("gid_tokens"), "identifier.gid_tokens"),
        sid_tokens=_positive_int(raw.get("sid_tokens"), "identifier.sid_tokens"),
        gid_codebook_size=_positive_int(
            raw.get("gid_codebook_size"), "identifier.gid_codebook_size"
        ),
        sid_codebook_sizes=sizes,
        base_token_order=_string_tuple(
            raw.get("base_token_order"), "identifier.base_token_order"
        ),
        singleton_length=_positive_int(
            raw.get("singleton_length"), "identifier.singleton_length"
        ),
        collision_length=_positive_int(
            raw.get("collision_length"), "identifier.collision_length"
        ),
        dedup_capacity=_positive_int(
            raw.get("dedup_capacity"), "identifier.dedup_capacity"
        ),
        dedup_assignment=_string(
            raw.get("dedup_assignment"), "identifier.dedup_assignment"
        ),
    )
    expected_order = tuple(
        [*(f"G{i}" for i in range(1, 7)), *(f"S{i}" for i in range(1, 4))]
    )
    if (
        config.gid_tokens != 6
        or config.sid_tokens != 3
        or config.gid_codebook_size != 32
        or config.sid_codebook_sizes != LEGACY_V1_CODEBOOK_SIZES
        or config.base_token_order != expected_order
        or config.singleton_length != 9
        or config.collision_length != 10
        or config.dedup_capacity != 512
        or config.dedup_assignment != "poi_id_lexicographic_zero_based"
    ):
        raise QGPRQKConfigError("identifier 必须保持冻结的 GID6+SID3+optional D 合同")
    return config


def _load_method(raw: dict[str, Any]) -> MethodConfig:
    hard_raw = _mapping(raw.get("hard_graph"), "method.hard_graph")
    weight_raw = _mapping(hard_raw.get("weights"), "method.hard_graph.weights")
    weights = {
        key: _number(value, f"method.hard_graph.weights.{key}")
        for key, value in weight_raw.items()
    }
    if set(weights) != set(EXPECTED_HARD_GRAPH_WEIGHTS) or any(
        abs(weights[key] - expected) > 1e-12
        for key, expected in EXPECTED_HARD_GRAPH_WEIGHTS.items()
    ):
        raise QGPRQKConfigError("困难图权重与用户确认的冻结值不一致")
    hard_graph = HardGraphConfig(
        weights=weights,
        top_k=_positive_int(hard_raw.get("top_k"), "method.hard_graph.top_k"),
        threshold=_number(
            hard_raw.get("threshold"), "method.hard_graph.threshold"
        ),
        excluded_fields=_string_tuple(
            hard_raw.get("excluded_fields"), "method.hard_graph.excluded_fields"
        ),
    )
    if (
        hard_graph.top_k != 20
        or abs(hard_graph.threshold - 0.60) > 1e-12
        or not {"area", "layer"} <= set(hard_graph.excluded_fields)
    ):
        raise QGPRQKConfigError("困难图必须使用 Top-20、阈值 0.60 且排除 area/layer")
    config = MethodConfig(
        poi_embedding_source=_string(
            raw.get("poi_embedding_source"), "method.poi_embedding_source"
        ),
        metric=_string(raw.get("metric"), "method.metric"),
        residual=_string(raw.get("residual"), "method.residual"),
        remove_common_direction=_boolean(
            raw.get("remove_common_direction"), "method.remove_common_direction"
        ),
        gid_order=_string(raw.get("gid_order"), "method.gid_order"),
        final_pid=_string(raw.get("final_pid"), "method.final_pid"),
        query_normalization=_string(
            raw.get("query_normalization"), "method.query_normalization"
        ),
        query_weights=_float_tuple(
            raw.get("query_weights"), "method.query_weights", 3
        ),
        hard_graph=hard_graph,
    )
    expected = (
        "frozen_beijing_bge_m3",
        "cosine",
        "projection",
        True,
        "before_sid",
        "gid6+sid3+optional_existing_dedup",
        "conservative_qg_v1",
        (0.15, 0.25, 0.35),
    )
    actual = (
        config.poi_embedding_source,
        config.metric,
        config.residual,
        config.remove_common_direction,
        config.gid_order,
        config.final_pid,
        config.query_normalization,
        config.query_weights,
    )
    if actual != expected:
        raise QGPRQKConfigError("method 与 QG-PRQK v1.1 冻结口径不一致")
    return config


def _load_runtime(raw: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        sample_limit=_positive_int(raw.get("sample_limit"), "runtime.sample_limit"),
        overwrite=_boolean(raw.get("overwrite"), "runtime.overwrite"),
        resume=_boolean(raw.get("resume"), "runtime.resume"),
        show_progress=_boolean(raw.get("show_progress"), "runtime.show_progress"),
        chunk_rows=_positive_int(raw.get("chunk_rows"), "runtime.chunk_rows"),
    )


def _load_query_stats(raw: dict[str, Any]) -> QueryStatsConfig:
    config = QueryStatsConfig(
        num_shards=_positive_int(raw.get("num_shards"), "query_stats.num_shards"),
        buffer_rows_per_shard=_positive_int(
            raw.get("buffer_rows_per_shard"),
            "query_stats.buffer_rows_per_shard",
        ),
        read_batch_rows=_positive_int(
            raw.get("read_batch_rows"), "query_stats.read_batch_rows"
        ),
        min_query_count=_positive_int(
            raw.get("min_query_count"), "query_stats.min_query_count"
        ),
        min_pair_count=_positive_int(
            raw.get("min_pair_count"), "query_stats.min_pair_count"
        ),
        min_top1_share=_number(
            raw.get("min_top1_share"), "query_stats.min_top1_share"
        ),
        min_margin=_number(raw.get("min_margin"), "query_stats.min_margin"),
        max_normalized_entropy=_number(
            raw.get("max_normalized_entropy"),
            "query_stats.max_normalized_entropy",
        ),
        false_negative_min_share=_number(
            raw.get("false_negative_min_share"),
            "query_stats.false_negative_min_share",
        ),
        false_negative_min_count=_positive_int(
            raw.get("false_negative_min_count"),
            "query_stats.false_negative_min_count",
        ),
    )
    for name, value in (
        ("min_top1_share", config.min_top1_share),
        ("min_margin", config.min_margin),
        ("max_normalized_entropy", config.max_normalized_entropy),
        ("false_negative_min_share", config.false_negative_min_share),
    ):
        if not 0.0 <= value <= 1.0:
            raise QGPRQKConfigError(f"query_stats.{name} 必须位于 [0,1]")
    expected = (2, 2, 0.75, 0.25, 0.55, 0.10, 2)
    actual = (
        config.min_query_count,
        config.min_pair_count,
        config.min_top1_share,
        config.min_margin,
        config.max_normalized_entropy,
        config.false_negative_min_share,
        config.false_negative_min_count,
    )
    if actual != expected:
        raise QGPRQKConfigError("query_stats 过滤阈值与 QG v1.1 冻结口径不一致")
    if config.num_shards > 65_536:
        raise QGPRQKConfigError("query_stats.num_shards 必须不超过 65536")
    return config


def _load_query_embedding(
    raw: dict[str, Any], project_root: Path
) -> QueryEmbeddingConfig:
    attention = raw.get("attention")
    prompt_name = raw.get("prompt_name")
    if attention is not None and not isinstance(attention, str):
        raise QGPRQKConfigError("query_embedding.attention 必须是字符串或 null")
    if prompt_name is not None and not isinstance(prompt_name, str):
        raise QGPRQKConfigError("query_embedding.prompt_name 必须是字符串或 null")
    config = QueryEmbeddingConfig(
        model_path=_resolve_path(
            raw.get("model_path"), project_root, "query_embedding.model_path"
        ),
        backend=_string(raw.get("backend"), "query_embedding.backend"),
        device=_string(raw.get("device"), "query_embedding.device"),
        batch_size=_positive_int(
            raw.get("batch_size"), "query_embedding.batch_size"
        ),
        encode_buffer_size=_positive_int(
            raw.get("encode_buffer_size"), "query_embedding.encode_buffer_size"
        ),
        max_seq_length=_positive_int(
            raw.get("max_seq_length"), "query_embedding.max_seq_length"
        ),
        torch_dtype=_string(
            raw.get("torch_dtype"), "query_embedding.torch_dtype"
        ),
        attention=attention,
        padding_side=_string(
            raw.get("padding_side"), "query_embedding.padding_side"
        ),
        normalize_embeddings=_boolean(
            raw.get("normalize_embeddings"),
            "query_embedding.normalize_embeddings",
        ),
        prompt_name=prompt_name,
        output_dtype=_string(
            raw.get("output_dtype"), "query_embedding.output_dtype"
        ),
        checkpoint_interval_batches=_positive_int(
            raw.get("checkpoint_interval_batches"),
            "query_embedding.checkpoint_interval_batches",
        ),
    )
    if config.backend != "sentence_transformers":
        raise QGPRQKConfigError("P3 Query encoder 必须使用 sentence_transformers")
    if config.encode_buffer_size < config.batch_size:
        raise QGPRQKConfigError("Query encode_buffer_size 不能小于 batch_size")
    if (
        config.max_seq_length != 128
        or config.torch_dtype != "bfloat16"
        or config.padding_side != "right"
        or not config.normalize_embeddings
        or config.prompt_name is not None
        or config.output_dtype != "float16"
    ):
        raise QGPRQKConfigError("Query embedding 必须复用冻结 BGE 无 instruction 编码口径")
    return config


def _load_query_adapter(raw: dict[str, Any]) -> QueryAdapterConfig:
    config = QueryAdapterConfig(
        enabled=_boolean(raw.get("enabled"), "query_adapter.enabled"),
        bottleneck=_positive_int(raw.get("bottleneck"), "query_adapter.bottleneck"),
        residual_scale=_number(
            raw.get("residual_scale"), "query_adapter.residual_scale"
        ),
        dropout=_number(raw.get("dropout"), "query_adapter.dropout"),
        optimizer=_string(raw.get("optimizer"), "query_adapter.optimizer"),
        learning_rate=_number(
            raw.get("learning_rate"), "query_adapter.learning_rate"
        ),
        weight_decay=_number(
            raw.get("weight_decay"), "query_adapter.weight_decay"
        ),
        epochs=_positive_int(raw.get("epochs"), "query_adapter.epochs"),
        batch_size=_positive_int(
            raw.get("batch_size"), "query_adapter.batch_size"
        ),
        precision=_string(raw.get("precision"), "query_adapter.precision"),
        max_grad_norm=_number(
            raw.get("max_grad_norm"), "query_adapter.max_grad_norm"
        ),
        temperature=_number(
            raw.get("temperature"), "query_adapter.temperature"
        ),
        dev_fraction=_number(
            raw.get("dev_fraction"), "query_adapter.dev_fraction"
        ),
        ann_top_k=_positive_int(raw.get("ann_top_k"), "query_adapter.ann_top_k"),
        semantic_ann_negatives=_positive_int(
            raw.get("semantic_ann_negatives"),
            "query_adapter.semantic_ann_negatives",
        ),
        lexical_metadata_negatives=_positive_int(
            raw.get("lexical_metadata_negatives"),
            "query_adapter.lexical_metadata_negatives",
        ),
        local_geo_negatives=_positive_int(
            raw.get("local_geo_negatives"),
            "query_adapter.local_geo_negatives",
        ),
    )
    if not config.enabled or config.optimizer.lower() != "adamw":
        raise QGPRQKConfigError("P3 Query Adapter 必须启用并使用 AdamW")
    if not 0.0 <= config.dropout < 1.0:
        raise QGPRQKConfigError("query_adapter.dropout 必须位于 [0,1)")
    if config.residual_scale < 0.0 or config.weight_decay < 0.0:
        raise QGPRQKConfigError("Query Adapter scale/weight_decay 不得为负")
    if config.learning_rate <= 0.0 or config.temperature <= 0.0:
        raise QGPRQKConfigError("Query Adapter learning_rate/temperature 必须为正")
    if not 0.0 < config.dev_fraction < 1.0:
        raise QGPRQKConfigError("query_adapter.dev_fraction 必须位于 (0,1)")
    if config.max_grad_norm <= 0.0 or config.ann_top_k < (
        config.semantic_ann_negatives + 1
    ):
        raise QGPRQKConfigError("Query Adapter max_grad_norm/ann_top_k 非法")
    return config


def _load_frozen_inputs(raw: dict[str, Any]) -> FrozenInputs:
    return FrozenInputs(
        embedding_manifest_sha256=_sha256(
            raw.get("embedding_manifest_sha256"),
            "frozen_inputs.embedding_manifest_sha256",
        ),
        embedding_fingerprint=_sha256(
            raw.get("embedding_fingerprint"),
            "frozen_inputs.embedding_fingerprint",
        ),
        poi_ids_sha256=_sha256(
            raw.get("poi_ids_sha256"), "frozen_inputs.poi_ids_sha256"
        ),
        sft_manifest_sha256=_sha256(
            raw.get("sft_manifest_sha256"),
            "frozen_inputs.sft_manifest_sha256",
        ),
        train_sha256=_sha256(raw.get("train_sha256"), "frozen_inputs.train_sha256"),
        valid_sha256=_sha256(raw.get("valid_sha256"), "frozen_inputs.valid_sha256"),
        test_sha256=_sha256(raw.get("test_sha256"), "frozen_inputs.test_sha256"),
    )


def load_legacy_config(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    seed: int | None = None,
    sample_limit: int | None = None,
    resume: bool | None = None,
    overwrite: bool | None = None,
) -> QGPRQKConfig:
    """Load the historical v1.1 contract and apply safe CLI overrides."""

    config_path = config_path.resolve()
    raw = _load_yaml(config_path)
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise QGPRQKConfigError(
            f"历史 v1.1 schema_version 必须是 {CONFIG_SCHEMA_VERSION}；"
            "当前 512×3 SID 请使用 sid.pipeline_config.load_downstream_config"
        )
    project = _load_project(_mapping(raw.get("project"), "project"))
    paths = _load_paths(
        _mapping(raw.get("paths"), "paths"), config_path, output_dir
    )
    data_contracts = _load_data_contracts(
        _mapping(raw.get("data_contracts"), "data_contracts")
    )
    identifier = _load_identifier(_mapping(raw.get("identifier"), "identifier"))
    method = _load_method(_mapping(raw.get("method"), "method"))
    query_stats = _load_query_stats(_mapping(raw.get("query_stats"), "query_stats"))
    query_embedding = _load_query_embedding(
        _mapping(raw.get("query_embedding"), "query_embedding"),
        paths.project_root,
    )
    query_adapter = _load_query_adapter(
        _mapping(raw.get("query_adapter"), "query_adapter")
    )
    runtime = _load_runtime(_mapping(raw.get("runtime"), "runtime"))
    runtime = replace(
        runtime,
        sample_limit=(
            runtime.sample_limit
            if sample_limit is None
            else _positive_int(sample_limit, "sample_limit")
        ),
        resume=runtime.resume if resume is None else bool(resume),
        overwrite=runtime.overwrite if overwrite is None else bool(overwrite),
    )
    if runtime.resume and runtime.overwrite:
        raise QGPRQKConfigError("resume 与 overwrite 不能同时启用")
    if seed is not None:
        project = replace(project, seed=_positive_int(seed, "seed"))
    deferred_keys = (
        "query_prototypes",
        "rqkmeans",
        "geo_s3",
    )
    deferred = {key: _mapping(raw.get(key), key) for key in deferred_keys}
    return QGPRQKConfig(
        source_path=config_path,
        source_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        project=project,
        paths=paths,
        data_contracts=data_contracts,
        identifier=identifier,
        method=method,
        query_stats=query_stats,
        query_embedding=query_embedding,
        query_adapter=query_adapter,
        runtime=runtime,
        frozen_inputs=_load_frozen_inputs(
            _mapping(raw.get("frozen_inputs"), "frozen_inputs")
        ),
        deferred_parameters=deferred,
    )


# Preserve existing upstream imports without silently changing their schema.
load_config = load_legacy_config

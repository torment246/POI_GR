"""Runtime contract for the query-adapter engineering gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.data.query_supervision_config import CategoryBuildConfig, load_category_config
from qg_prqk.config import QueryAdapterConfig, QueryEmbeddingConfig


SCHEMA_VERSION = "qg-prqk-p3a-config-v1"


class QueryAdapterConfigError(ValueError):
    """Raised when P3A settings diverge from the reviewed gate contract."""


@dataclass(frozen=True)
class ExactRetrievalConfig:
    backend: str
    gpu_id: int
    top_k: int
    add_batch_size: int
    search_batch_size: int
    use_float16_storage: bool


@dataclass(frozen=True)
class AdapterEvaluationConfig:
    head_min_query_count: int
    tail_max_query_count: int
    difficult_definition: str
    primary_metric: str


@dataclass(frozen=True)
class QueryAdapterGateConfig:
    source_path: Path
    source_sha256: str
    category_config: CategoryBuildConfig
    seed: int
    gate_query_limit: int
    sample_query_limit: int
    selection: str
    p2_5_manifest_sha256: str
    query_embedding: QueryEmbeddingConfig
    ann: ExactRetrievalConfig
    adapter: QueryAdapterConfig
    evaluation: AdapterEvaluationConfig

    @property
    def project_root(self) -> Path:
        return self.category_config.paths.project_root

    @property
    def p2_query_stats(self) -> Path:
        return self.category_config.paths.p2_query_stats

    @property
    def query_depth_dir(self) -> Path:
        return self.category_config.query_depth_output_dir

    @property
    def poi_embeddings(self) -> Path:
        return self.category_config.paths.poi_embeddings

    @property
    def poi_ids(self) -> Path:
        return self.category_config.paths.poi_ids

    @property
    def poi_catalog(self) -> Path:
        return self.category_config.paths.poi_catalog

    @property
    def output_dir(self) -> Path:
        return self.category_config.paths.output_dir

    def resolved_payload(self) -> dict[str, Any]:
        """Return the P3A-only semantic configuration used by manifests."""

        embedding = asdict(self.query_embedding)
        embedding["model_path"] = str(self.query_embedding.model_path)
        return {
            "schema_version": SCHEMA_VERSION,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "category_config_signature": self.category_config.signature(),
            "seed": self.seed,
            "gate_query_limit": self.gate_query_limit,
            "sample_query_limit": self.sample_query_limit,
            "selection": self.selection,
            "p2_5_manifest_sha256": self.p2_5_manifest_sha256,
            "query_embedding": embedding,
            "ann": asdict(self.ann),
            "adapter": asdict(self.adapter),
            "evaluation": asdict(self.evaluation),
        }

    def signature(self) -> str:
        """Hash the P3A semantic contract deterministically."""

        payload = json.dumps(
            self.resolved_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QueryAdapterConfigError(f"{label} 必须是 mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise QueryAdapterConfigError(f"{label} 缺少 {key}")
    return mapping[key]


def _string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _required(mapping, key, label)
    if not isinstance(value, str) or not value.strip():
        raise QueryAdapterConfigError(f"{label}.{key} 必须是非空字符串")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = _required(mapping, key, label)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QueryAdapterConfigError(f"{label}.{key} 必须是正整数")
    return value


def _nonnegative_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = _required(mapping, key, label)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QueryAdapterConfigError(f"{label}.{key} 必须是非负整数")
    return value


def _number(mapping: Mapping[str, Any], key: str, label: str) -> float:
    value = _required(mapping, key, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QueryAdapterConfigError(f"{label}.{key} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise QueryAdapterConfigError(f"{label}.{key} 必须是有限数值")
    return result


def _boolean(mapping: Mapping[str, Any], key: str, label: str) -> bool:
    value = _required(mapping, key, label)
    if not isinstance(value, bool):
        raise QueryAdapterConfigError(f"{label}.{key} 必须是 bool")
    return value


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise QueryAdapterConfigError(f"{label} 必须为 {expected!r}，实际 {actual!r}")


def load_query_adapter_config(path: Path) -> QueryAdapterGateConfig:
    """Load v2.1-CAT and validate only the reviewed P3A fields."""

    source_path = path.resolve()
    try:
        raw_bytes = source_path.read_bytes()
        root = _mapping(yaml.safe_load(raw_bytes), "配置根")
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise QueryAdapterConfigError(f"配置读取失败：{source_path}") from error
    _assert_equal(root.get("schema_version"), SCHEMA_VERSION, "schema_version")
    base_raw = Path(_string(root, "base_v2_1_config", "配置根"))
    base_path = (
        base_raw.resolve()
        if base_raw.is_absolute()
        else (source_path.parent / base_raw).resolve()
    )
    category_config = load_category_config(base_path)
    p3a = _mapping(_required(root, "p3a", "配置根"), "p3a")
    embedding_raw = _mapping(
        _required(p3a, "query_embedding", "p3a"), "p3a.query_embedding"
    )
    ann_raw = _mapping(_required(p3a, "ann", "p3a"), "p3a.ann")
    adapter_raw = _mapping(_required(p3a, "adapter", "p3a"), "p3a.adapter")
    evaluation_raw = _mapping(
        _required(p3a, "evaluation", "p3a"), "p3a.evaluation"
    )

    model_raw = Path(_string(embedding_raw, "model_path", "p3a.query_embedding"))
    model_path = (
        model_raw.resolve()
        if model_raw.is_absolute()
        else (category_config.paths.project_root / model_raw).resolve()
    )
    embedding = QueryEmbeddingConfig(
        model_path=model_path,
        backend=_string(embedding_raw, "backend", "p3a.query_embedding"),
        device=_string(embedding_raw, "device", "p3a.query_embedding"),
        batch_size=_positive_int(embedding_raw, "batch_size", "p3a.query_embedding"),
        encode_buffer_size=_positive_int(
            embedding_raw, "encode_buffer_size", "p3a.query_embedding"
        ),
        max_seq_length=_positive_int(
            embedding_raw, "max_seq_length", "p3a.query_embedding"
        ),
        torch_dtype=_string(embedding_raw, "torch_dtype", "p3a.query_embedding"),
        attention=embedding_raw.get("attention"),
        padding_side=_string(embedding_raw, "padding_side", "p3a.query_embedding"),
        normalize_embeddings=_boolean(
            embedding_raw, "normalize_embeddings", "p3a.query_embedding"
        ),
        prompt_name=embedding_raw.get("prompt_name"),
        output_dtype=_string(embedding_raw, "output_dtype", "p3a.query_embedding"),
        checkpoint_interval_batches=20,
    )
    ann = ExactRetrievalConfig(
        backend=_string(ann_raw, "backend", "p3a.ann"),
        gpu_id=_nonnegative_int(ann_raw, "gpu_id", "p3a.ann"),
        top_k=_positive_int(ann_raw, "top_k", "p3a.ann"),
        add_batch_size=_positive_int(ann_raw, "add_batch_size", "p3a.ann"),
        search_batch_size=_positive_int(
            ann_raw, "search_batch_size", "p3a.ann"
        ),
        use_float16_storage=_boolean(
            ann_raw, "use_float16_storage", "p3a.ann"
        ),
    )
    adapter = QueryAdapterConfig(
        enabled=True,
        bottleneck=_positive_int(adapter_raw, "bottleneck", "p3a.adapter"),
        residual_scale=_number(adapter_raw, "residual_scale", "p3a.adapter"),
        dropout=_number(adapter_raw, "dropout", "p3a.adapter"),
        optimizer=_string(adapter_raw, "optimizer", "p3a.adapter"),
        learning_rate=_number(adapter_raw, "learning_rate", "p3a.adapter"),
        weight_decay=_number(adapter_raw, "weight_decay", "p3a.adapter"),
        epochs=_positive_int(adapter_raw, "epochs", "p3a.adapter"),
        batch_size=_positive_int(adapter_raw, "batch_size", "p3a.adapter"),
        precision=_string(adapter_raw, "precision", "p3a.adapter"),
        max_grad_norm=_number(adapter_raw, "max_grad_norm", "p3a.adapter"),
        temperature=_number(adapter_raw, "temperature", "p3a.adapter"),
        dev_fraction=_number(adapter_raw, "dev_fraction", "p3a.adapter"),
        ann_top_k=ann.top_k,
        semantic_ann_negatives=_positive_int(
            adapter_raw, "semantic_ann_negatives", "p3a.adapter"
        ),
        lexical_metadata_negatives=_positive_int(
            adapter_raw, "lexical_metadata_negatives", "p3a.adapter"
        ),
        local_geo_negatives=_positive_int(
            adapter_raw, "local_geo_negatives", "p3a.adapter"
        ),
    )
    evaluation = AdapterEvaluationConfig(
        head_min_query_count=_positive_int(
            evaluation_raw, "head_min_query_count", "p3a.evaluation"
        ),
        tail_max_query_count=_positive_int(
            evaluation_raw, "tail_max_query_count", "p3a.evaluation"
        ),
        difficult_definition=_string(
            evaluation_raw, "difficult_definition", "p3a.evaluation"
        ),
        primary_metric=_string(evaluation_raw, "primary_metric", "p3a.evaluation"),
    )

    _assert_equal(_positive_int(p3a, "seed", "p3a"), 42, "p3a.seed")
    _assert_equal(
        _string(p3a, "selection", "p3a"),
        "blake2b_query_id_smallest",
        "p3a.selection",
    )
    _assert_equal(embedding.backend, "sentence_transformers", "query backend")
    _assert_equal(embedding.device, "cuda", "query device")
    _assert_equal(embedding.max_seq_length, 128, "query max_seq_length")
    _assert_equal(embedding.padding_side, "right", "query padding_side")
    _assert_equal(embedding.normalize_embeddings, True, "query normalize")
    _assert_equal(embedding.prompt_name, None, "query prompt_name")
    _assert_equal(embedding.output_dtype, "float16", "query output_dtype")
    _assert_equal(ann.backend, "faiss_gpu_flat_ip", "ann.backend")
    _assert_equal(ann.top_k, 100, "ann.top_k")
    _assert_equal(adapter.batch_size, 2048, "adapter.batch_size")
    _assert_equal(adapter.bottleneck, 64, "adapter.bottleneck")
    _assert_equal(adapter.optimizer, "adamw", "adapter.optimizer")
    _assert_equal(evaluation.head_min_query_count, 5, "head threshold")
    _assert_equal(evaluation.tail_max_query_count, 4, "tail threshold")
    _assert_equal(
        evaluation.difficult_definition,
        "raw_recall_at_1_miss",
        "difficult_definition",
    )
    _assert_equal(
        evaluation.primary_metric,
        "recall_at_10_or_mean_hard_margin",
        "primary_metric",
    )
    if not 0 <= adapter.dropout < 1 or not 0 < adapter.dev_fraction < 1:
        raise QueryAdapterConfigError("Adapter dropout/dev_fraction 非法")
    if adapter.semantic_ann_negatives + adapter.lexical_metadata_negatives + adapter.local_geo_negatives >= ann.top_k:
        raise QueryAdapterConfigError("固定负例总数必须小于 ANN top_k")
    gate_limit = _positive_int(p3a, "gate_query_limit", "p3a")
    sample_limit = _positive_int(p3a, "sample_query_limit", "p3a")
    if gate_limit > 50_000 or sample_limit >= gate_limit:
        raise QueryAdapterConfigError("P3A sample/gate Query 数不符合最多 50,000 条约束")
    p2_5_sha = _string(root, "p2_5_manifest_sha256", "配置根")
    if len(p2_5_sha) != 64 or any(char not in "0123456789abcdef" for char in p2_5_sha):
        raise QueryAdapterConfigError("frozen_inputs.p2_5_manifest_sha256 非法")
    return QueryAdapterGateConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        category_config=category_config,
        seed=42,
        gate_query_limit=gate_limit,
        sample_query_limit=sample_limit,
        selection="blake2b_query_id_smallest",
        p2_5_manifest_sha256=p2_5_sha,
        query_embedding=embedding,
        ann=ann,
        adapter=adapter,
        evaluation=evaluation,
    )

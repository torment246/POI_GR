"""Shared POI embedding pipeline and evaluation-set contracts."""

from .core import (
    ConfigError,
    DataConfig,
    DataValidationError,
    EmbeddingJobConfig,
    ModelConfig,
    OutputConfig,
    PreparedInput,
    TextEncoder,
    _load_encoder,
    _read_record,
    apply_overrides,
    discover_input_files,
    encode_texts_to_npy,
    iter_texts,
    load_job_config,
    run_embedding_job,
    scan_and_prepare_input,
)

__all__ = [
    "ConfigError",
    "DataConfig",
    "DataValidationError",
    "EmbeddingJobConfig",
    "ModelConfig",
    "OutputConfig",
    "PreparedInput",
    "TextEncoder",
    "apply_overrides",
    "discover_input_files",
    "encode_texts_to_npy",
    "iter_texts",
    "load_job_config",
    "run_embedding_job",
    "scan_and_prepare_input",
]

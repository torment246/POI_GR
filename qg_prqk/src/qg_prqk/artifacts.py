"""Artifact manifest helpers for the staged QG-PRQK pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from qg_prqk.config import QGPRQKConfig


ARTIFACT_MANIFEST_SCHEMA_VERSION = "qg-prqk-artifact-manifest-v1"


class QGPRQKArtifactError(RuntimeError):
    """Raised when artifact isolation or safe writing fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool = False
) -> None:
    """Write JSON atomically while keeping overwrite disabled by default."""

    if path.exists() and not overwrite:
        raise QGPRQKArtifactError(f"输出已存在且 overwrite=false：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_manifest_template(
    config: QGPRQKConfig,
    *,
    sample_limit: int,
    dry_run_checks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the P1 manifest skeleton without claiming algorithm outputs."""

    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "status": "planned",
        "method": config.project.method,
        "version": config.project.version,
        "city": config.project.city,
        "created_at": utc_now(),
        "config": {
            "path": str(config.source_path),
            "sha256": config.source_sha256,
            "resolved_signature": config.signature(),
        },
        "runtime": {
            "seed": config.project.seed,
            "sample_limit": sample_limit,
            "resume": config.runtime.resume,
            "overwrite": config.runtime.overwrite,
        },
        "identifier": {
            "codebook_sizes": list(config.project.codebook_sizes),
            "base_token_order": list(config.identifier.base_token_order),
            "singleton_length": config.identifier.singleton_length,
            "collision_length": config.identifier.collision_length,
            "dedup_assignment": config.identifier.dedup_assignment,
        },
        "inputs": {
            "poi_catalog": str(config.paths.poi_catalog),
            "poi_embeddings": str(config.paths.poi_embeddings),
            "poi_ids": str(config.paths.poi_ids),
            "embedding_manifest": {
                "path": str(config.paths.embedding_manifest),
                "sha256": config.frozen_inputs.embedding_manifest_sha256,
            },
            "sft_manifest": {
                "path": str(config.paths.sft_manifest),
                "sha256": config.frozen_inputs.sft_manifest_sha256,
            },
            "train_date_range": [
                config.data_contracts.train_date_start,
                config.data_contracts.train_date_end,
            ],
        },
        "output_dir": str(config.paths.output_dir),
        "dry_run_checks": dict(dry_run_checks or {}),
        "artifacts": {},
    }

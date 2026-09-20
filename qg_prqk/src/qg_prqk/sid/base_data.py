"""Frozen active-POI inputs for the POI-only base codebook."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qg_prqk.artifacts import sha256_file
from qg_prqk.sid.pipeline_config import DownstreamConfig


SAMPLE_DEFAULT_ROWS = 10_000
SAMPLE_MAX_ROWS = 10_000
MEDIUM_100K_ROWS = 100_000
MEDIUM_500K_ROWS = 500_000
GATE_ORDER = ("sample", "medium100k", "medium500k", "full")
PREVIOUS_GATE = {
    "medium100k": "sample",
    "medium500k": "medium100k",
    "full": "medium500k",
}


class BaseCodebookDataError(ValueError):
    """Raised when P5 data, configuration, or gate contracts diverge."""


@dataclass(frozen=True)
class TopKRefinementConfig:
    enabled: bool
    topk: int
    beta: float
    max_iter: int


@dataclass(frozen=True)
class BaseCodebookAlgorithmConfig:
    metric: str
    init: str
    seed: int
    remove_global_direction: bool
    residual: str
    min_iter: int
    max_iter: int
    objective_rel_tol: float
    assignment_change_tol: float
    patience: int
    topk_refinement: TopKRefinementConfig


@dataclass(frozen=True)
class BaseCodebookInputs:
    embeddings: np.ndarray
    embedding_path: Path
    poi_ids_path: Path
    embedding_manifest_path: Path
    total_rows: int
    embedding_dim: int
    selected_rows: np.ndarray
    source_hashes: Mapping[str, str]


def _exact_mapping(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BaseCodebookDataError(f"{label} 字段必须恰好为 {sorted(fields)}")
    return value


def load_base_codebook_algorithm(config: DownstreamConfig) -> BaseCodebookAlgorithmConfig:
    """Read and strictly validate the frozen P5 portion of the resolved YAML."""
    payload = _exact_mapping(
        config.resolved_payload()["prqk"],
        {
            "metric",
            "init",
            "seed",
            "remove_global_direction",
            "residual",
            "min_iter",
            "max_iter",
            "objective_rel_tol",
            "poi_assignment_change_tol",
            "query_assignment_change_tol",
            "patience",
            "topk_refinement",
        },
        "prqk",
    )
    refinement = _exact_mapping(
        payload["topk_refinement"],
        {"enabled", "topk", "beta", "max_iter"},
        "prqk.topk_refinement",
    )
    expected_max_iter = (
        60 if getattr(config, "protocol_id", None) == "hard60_topk5_v1" else 30
    )
    expected = {
        "metric": "cosine",
        "init": "kmeans++",
        "seed": 42,
        "remove_global_direction": True,
        "residual": "projection",
        "min_iter": 8,
        "max_iter": expected_max_iter,
        "objective_rel_tol": 1.0e-4,
        "poi_assignment_change_tol": 1.0e-3,
        "query_assignment_change_tol": 2.0e-3,
        "patience": 2,
    }
    for key, expected_value in expected.items():
        if payload[key] != expected_value:
            raise BaseCodebookDataError(f"prqk.{key} 必须保持冻结值 {expected_value!r}")
    if refinement != {"enabled": True, "topk": 5, "beta": 15, "max_iter": 5}:
        raise BaseCodebookDataError("Top-k refinement 必须保持 enabled/k/beta/iter=1/5/15/5")
    if list(config.codebook_sizes) != [512, 512, 512]:
        raise BaseCodebookDataError("P5-CAT 只允许 512×3")
    return BaseCodebookAlgorithmConfig(
        metric=str(payload["metric"]),
        init=str(payload["init"]),
        seed=int(payload["seed"]),
        remove_global_direction=bool(payload["remove_global_direction"]),
        residual=str(payload["residual"]),
        min_iter=int(payload["min_iter"]),
        max_iter=int(payload["max_iter"]),
        objective_rel_tol=float(payload["objective_rel_tol"]),
        assignment_change_tol=float(payload["poi_assignment_change_tol"]),
        patience=int(payload["patience"]),
        topk_refinement=TopKRefinementConfig(
            enabled=bool(refinement["enabled"]),
            topk=int(refinement["topk"]),
            beta=float(refinement["beta"]),
            max_iter=int(refinement["max_iter"]),
        ),
    )


def rows_for_gate(
    gate: str,
    total_rows: int,
    sample_rows: int | None = None,
) -> int:
    """Resolve one explicit engineering gate without accepting fake full runs."""
    if gate not in GATE_ORDER:
        raise BaseCodebookDataError(f"未知 P5 gate：{gate}")
    if gate == "sample":
        rows = SAMPLE_DEFAULT_ROWS if sample_rows is None else sample_rows
        if type(rows) is not int or not 512 <= rows <= min(SAMPLE_MAX_ROWS, total_rows):
            raise BaseCodebookDataError("sample rows 必须位于 [512, 10000]")
        return rows
    if sample_rows is not None:
        raise BaseCodebookDataError("只有 sample gate 允许指定 --sample-rows")
    rows = {
        "medium100k": MEDIUM_100K_ROWS,
        "medium500k": MEDIUM_500K_ROWS,
        "full": total_rows,
    }[gate]
    if rows > total_rows:
        raise BaseCodebookDataError(f"{gate} 要求 {rows} 行，但 active POI 只有 {total_rows} 行")
    return rows


def deterministic_selected_rows(total_rows: int, rows: int, seed: int) -> np.ndarray:
    """Use one seeded permutation so all non-full gates are nested POI sets."""
    if rows == total_rows:
        return np.arange(total_rows, dtype=np.int64)
    generator = np.random.Generator(np.random.PCG64(seed))
    selected = np.sort(generator.permutation(total_rows)[:rows]).astype(np.int64)
    if len(np.unique(selected)) != rows:
        raise BaseCodebookDataError("P5 确定性 POI 选样出现重复行")
    return selected


def selected_rows_sha256(values: np.ndarray) -> str:
    rows = np.ascontiguousarray(values, dtype="<i8")
    return hashlib.sha256(rows.tobytes()).hexdigest()


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_base_codebook_inputs(
    config: DownstreamConfig,
    *,
    gate: str,
    sample_rows: int | None = None,
    verify_embedding_hash: bool = True,
) -> BaseCodebookInputs:
    """Validate the active BGE asset and return a read-only mmap plus frozen rows."""
    resolved = config.resolved_payload()
    project_root = config.upstream.base.project_root
    paths = resolved["paths"]
    frozen = resolved["frozen_inputs"]
    embedding_path = _project_path(project_root, paths["poi_embeddings"])
    poi_ids_path = _project_path(project_root, paths["poi_ids"])
    manifest_path = _project_path(project_root, paths["embedding_manifest"])
    for label, path in (
        ("active POI embedding", embedding_path),
        ("active POI row map", poi_ids_path),
        ("active POI embedding manifest", manifest_path),
    ):
        if not path.is_file():
            raise BaseCodebookDataError(f"{label} 不存在：{path}")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != frozen["poi_embedding_manifest_sha256"]:
        raise BaseCodebookDataError("active POI embedding manifest SHA256 不匹配")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_shape = [int(frozen["poi_rows"]), int(frozen["poi_embedding_dim"])]
    output = manifest.get("output", {})
    if (
        manifest.get("status") != "completed"
        or output.get("shape") != expected_shape
        or output.get("dtype") != frozen["poi_embedding_dtype"]
        or output.get("poi_ids_sha256") != frozen["poi_ids_sha256"]
        or manifest.get("model", {}).get("normalize_embeddings") is not True
    ):
        raise BaseCodebookDataError("active POI embedding manifest 内容不符合冻结合同")
    poi_ids_sha = sha256_file(poi_ids_path)
    if poi_ids_sha != frozen["poi_ids_sha256"]:
        raise BaseCodebookDataError("active POI row map SHA256 不匹配")
    embedding_sha = output.get("embeddings_sha256")
    if not isinstance(embedding_sha, str) or len(embedding_sha) != 64:
        raise BaseCodebookDataError("embedding manifest 缺少 embeddings SHA256")
    if verify_embedding_hash and sha256_file(embedding_path) != embedding_sha:
        raise BaseCodebookDataError("active POI embedding SHA256 不匹配")
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if list(embeddings.shape) != expected_shape or embeddings.dtype != np.float16:
        raise BaseCodebookDataError("active POI embedding NPY shape/dtype 不匹配")
    rows = rows_for_gate(gate, len(embeddings), sample_rows)
    settings = load_base_codebook_algorithm(config)
    selected = deterministic_selected_rows(len(embeddings), rows, settings.seed)
    return BaseCodebookInputs(
        embeddings=embeddings,
        embedding_path=embedding_path,
        poi_ids_path=poi_ids_path,
        embedding_manifest_path=manifest_path,
        total_rows=len(embeddings),
        embedding_dim=embeddings.shape[1],
        selected_rows=selected,
        source_hashes={
            "embedding_manifest_sha256": manifest_sha,
            "embedding_sha256": embedding_sha,
            "poi_ids_sha256": poi_ids_sha,
            "selected_rows_sha256": selected_rows_sha256(selected),
        },
    )


def compute_global_mean(
    embeddings: np.ndarray,
    *,
    chunk_rows: int = 8192,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Compute the corpus mean of normalized POI embeddings in stable row order."""
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise BaseCodebookDataError("POI embedding 必须是非空二维矩阵")
    accumulator = np.zeros(embeddings.shape[1], dtype=np.float64)
    min_norm = float("inf")
    max_norm = 0.0
    for start in range(0, len(embeddings), chunk_rows):
        block = np.asarray(embeddings[start : start + chunk_rows], dtype=np.float32)
        if not np.isfinite(block).all():
            raise BaseCodebookDataError("POI embedding 含 NaN/Inf")
        norms = np.linalg.norm(block, axis=1)
        if np.any(norms <= 1.0e-12):
            raise BaseCodebookDataError("POI embedding 含零向量")
        min_norm = min(min_norm, float(norms.min()))
        max_norm = max(max_norm, float(norms.max()))
        block /= norms[:, None]
        accumulator += block.sum(axis=0, dtype=np.float64)
    mean = (accumulator / len(embeddings)).astype(np.float32)
    mean_norm = float(np.linalg.norm(mean))
    if not np.isfinite(mean).all() or mean_norm <= 1.0e-12:
        raise BaseCodebookDataError("POI global mean 非法或为零")
    return mean, {
        "rows": len(embeddings),
        "input_norm_min": min_norm,
        "input_norm_max": max_norm,
        "global_mean_norm": mean_norm,
    }


def output_directory(config: DownstreamConfig, gate: str, rows: int) -> Path:
    labels = {
        "sample": f"sample_{rows:06d}",
        "medium100k": "medium_100000",
        "medium500k": "medium_500000",
        "full": f"full_{rows:06d}",
    }
    if gate not in labels:
        raise BaseCodebookDataError(f"未知 P5 gate：{gate}")
    root = getattr(config, "p5_output_dir", config.output_dir / "poi_prqk_a0")
    path = root / labels[gate]
    if not path.resolve().is_relative_to(config.output_dir.resolve()):
        raise BaseCodebookDataError("P5 输出越出 qg_prqk 512 namespace")
    return path


def gate_next_status(config: DownstreamConfig, gate: str) -> str:
    """Return a protocol-specific review state without changing legacy manifests."""
    prefix = (
        "HOLD_FOR_P5_HARD60_TOPK5"
        if getattr(config, "protocol_id", None) == "hard60_topk5_v1"
        else "HOLD_FOR_P5"
    )
    return f"{prefix}_{gate.upper()}_REVIEW"

"""Build Train-only category-common residual Query aggregates."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from .query_augmentation import validate_query_poi_aggregates


SCHEMA_VERSION = "query-category-residual-v1"


class QueryCategoryResidualError(RuntimeError):
    """Raised when E4 inputs or outputs violate their frozen contract."""


@dataclass(frozen=True)
class QueryCategoryResidualConfig:
    experiment_id: str
    aggregates_dir: Path
    expected_aggregates_manifest_sha256: str
    category_indices: Path
    category_manifest: Path
    expected_category_indices_sha256: str
    expected_category_manifest_sha256: str
    expected_poi_rows: int
    expected_covered_pois: int
    category_count: int
    min_category_support: int
    betas: tuple[float, ...]
    output_dir: Path
    chunk_rows: int


@dataclass(frozen=True)
class QueryCategoryResidualResult:
    output_dir: Path
    manifest_path: Path
    covered_pois: int
    candidates: int
    reused: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise QueryCategoryResidualError(f"{name} 不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise QueryCategoryResidualError(f"{name} 必须是 JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueryCategoryResidualError(f"{name} 必须是 mapping")
    return value


def _path(value: Any, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise QueryCategoryResidualError(f"{name} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise QueryCategoryResidualError(f"{name} 必须是 64 位 SHA256")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QueryCategoryResidualError(f"{name} 必须是正整数")
    return value


def _beta_label(beta: float) -> str:
    return f"{beta:.2f}".replace(".", "p")


def residual_filename(beta: float) -> str:
    """Return the deterministic E4 aggregate filename for one beta."""

    return f"e4_category_residual_beta_{_beta_label(beta)}.npy"


def _distribution(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(
        np.asarray(values, dtype=np.float64),
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
    )
    return {
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p90": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(quantiles[7]),
        "mean": float(np.mean(values, dtype=np.float64)),
    }


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def load_query_category_residual_config(
    config_path: Path,
    project_root: Path,
) -> QueryCategoryResidualConfig:
    """Load and validate the frozen E4 aggregate-build configuration."""

    with config_path.open("r", encoding="utf-8") as handle:
        root = _mapping(yaml.safe_load(handle), "配置根节点")
    input_config = _mapping(root.get("input"), "input")
    residual_config = _mapping(root.get("residual"), "residual")
    output_config = _mapping(root.get("output"), "output")
    runtime_config = _mapping(root.get("runtime"), "runtime")
    experiment_id = root.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise QueryCategoryResidualError("experiment_id 必须是非空字符串")
    beta_values = residual_config.get("betas")
    if not isinstance(beta_values, list) or not beta_values:
        raise QueryCategoryResidualError("residual.betas 必须是非空列表")
    betas = tuple(float(value) for value in beta_values)
    if (
        len(set(betas)) != len(betas)
        or any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in betas)
    ):
        raise QueryCategoryResidualError("residual.betas 必须唯一且位于 (0,1]")
    return QueryCategoryResidualConfig(
        experiment_id=experiment_id,
        aggregates_dir=_path(
            input_config.get("aggregates_dir"), project_root, "input.aggregates_dir"
        ),
        expected_aggregates_manifest_sha256=_sha(
            input_config.get("expected_aggregates_manifest_sha256"),
            "input.expected_aggregates_manifest_sha256",
        ),
        category_indices=_path(
            input_config.get("category_indices"),
            project_root,
            "input.category_indices",
        ),
        category_manifest=_path(
            input_config.get("category_manifest"),
            project_root,
            "input.category_manifest",
        ),
        expected_category_indices_sha256=_sha(
            input_config.get("expected_category_indices_sha256"),
            "input.expected_category_indices_sha256",
        ),
        expected_category_manifest_sha256=_sha(
            input_config.get("expected_category_manifest_sha256"),
            "input.expected_category_manifest_sha256",
        ),
        expected_poi_rows=_positive_int(
            input_config.get("expected_poi_rows"), "input.expected_poi_rows"
        ),
        expected_covered_pois=_positive_int(
            input_config.get("expected_covered_pois"),
            "input.expected_covered_pois",
        ),
        category_count=_positive_int(
            residual_config.get("category_count"), "residual.category_count"
        ),
        min_category_support=_positive_int(
            residual_config.get("min_category_support"),
            "residual.min_category_support",
        ),
        betas=betas,
        output_dir=_path(output_config.get("dir"), project_root, "output.dir"),
        chunk_rows=_positive_int(
            runtime_config.get("chunk_rows"), "runtime.chunk_rows"
        ),
    )


def validate_query_category_residual(
    output_dir: Path,
    *,
    expected_aggregates_manifest_sha256: str | None = None,
    expected_category_indices_sha256: str | None = None,
) -> QueryCategoryResidualResult:
    """Validate E4 category statistics and every residual aggregate."""

    manifest = _load_json(output_dir / "manifest.json", "E4 manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise QueryCategoryResidualError("E4 schema 不兼容")
    if manifest.get("status") != "completed" or not (
        output_dir / "_SUCCESS"
    ).is_file():
        raise QueryCategoryResidualError("E4 产物未完成")
    inputs = _mapping(manifest.get("inputs"), "manifest.inputs")
    if (
        expected_aggregates_manifest_sha256 is not None
        and inputs.get("aggregates_manifest_sha256")
        != expected_aggregates_manifest_sha256
    ):
        raise QueryCategoryResidualError("E4 不是冻结 E2 聚合版本")
    if (
        expected_category_indices_sha256 is not None
        and inputs.get("category_indices_sha256")
        != expected_category_indices_sha256
    ):
        raise QueryCategoryResidualError("E4 不是冻结类别版本")
    stats = _mapping(manifest.get("stats"), "manifest.stats")
    covered_pois = int(stats.get("covered_pois", -1))
    category_count = int(stats.get("category_count", -1))
    dimension = int(stats.get("embedding_dim", -1))
    if covered_pois <= 0 or category_count <= 0 or dimension <= 0:
        raise QueryCategoryResidualError("E4 manifest 维度字段非法")
    outputs = _mapping(manifest.get("outputs"), "manifest.outputs")
    fixed_specs = {
        "category_counts": ((category_count,), "int64"),
        "category_query_means": ((category_count, dimension), "float32"),
    }
    for name, (shape, dtype) in fixed_specs.items():
        spec = _mapping(outputs.get(name), f"outputs.{name}")
        path = output_dir / str(spec.get("file"))
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != shape or str(values.dtype) != dtype:
            raise QueryCategoryResidualError(f"{name} shape/dtype 非法")
        if not np.isfinite(values).all():
            raise QueryCategoryResidualError(f"{name} 存在 NaN/Inf")
        if _sha256_file(path) != spec.get("sha256"):
            raise QueryCategoryResidualError(f"{name} SHA256 不一致")
    candidates = outputs.get("residual_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise QueryCategoryResidualError("E4 residual_candidates 非法")
    beta_values: list[float] = []
    for spec_value in candidates:
        spec = _mapping(spec_value, "residual candidate")
        beta = float(spec.get("beta", -1))
        if not np.isfinite(beta) or not 0.0 < beta <= 1.0:
            raise QueryCategoryResidualError("E4 residual beta 非法")
        beta_values.append(beta)
    if len(beta_values) != len(set(beta_values)):
        raise QueryCategoryResidualError("E4 residual beta 重复")
    return QueryCategoryResidualResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        covered_pois=covered_pois,
        candidates=len(candidates),
        reused=True,
    )


def build_query_category_residual(
    config: QueryCategoryResidualConfig,
    *,
    project_root: Path,
    show_progress: bool = True,
) -> QueryCategoryResidualResult:
    """Build category-common Query residuals with bounded streaming memory."""

    output_dir = config.output_dir.resolve()
    if (output_dir / "_SUCCESS").is_file():
        return validate_query_category_residual(
            output_dir,
            expected_aggregates_manifest_sha256=(
                config.expected_aggregates_manifest_sha256
            ),
            expected_category_indices_sha256=(
                config.expected_category_indices_sha256
            ),
        )
    if output_dir.exists():
        raise QueryCategoryResidualError(f"正式输出目录已存在但未完成：{output_dir}")
    staging_dir = output_dir.with_name(f".{output_dir.name}.building")
    if staging_dir.exists():
        raise QueryCategoryResidualError(f"存在未完成 staging：{staging_dir}")

    aggregates_manifest_path = config.aggregates_dir / "manifest.json"
    if (
        _sha256_file(aggregates_manifest_path)
        != config.expected_aggregates_manifest_sha256
    ):
        raise QueryCategoryResidualError("E1/E2 聚合 manifest SHA256 不一致")
    validate_query_poi_aggregates(config.aggregates_dir)
    if (
        _sha256_file(config.category_indices)
        != config.expected_category_indices_sha256
    ):
        raise QueryCategoryResidualError("类别行号 SHA256 不一致")
    if (
        _sha256_file(config.category_manifest)
        != config.expected_category_manifest_sha256
    ):
        raise QueryCategoryResidualError("类别 manifest SHA256 不一致")

    covered_rows = np.load(
        config.aggregates_dir / "covered_poi_rows.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    query_aggregates = np.load(
        config.aggregates_dir / "e2_query_weighted.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    category_indices = np.load(
        config.category_indices, mmap_mode="r", allow_pickle=False
    )
    if len(covered_rows) != config.expected_covered_pois or (
        query_aggregates.shape[0] != config.expected_covered_pois
    ):
        raise QueryCategoryResidualError("E2 覆盖 POI 数不一致")
    if category_indices.shape != (config.expected_poi_rows,):
        raise QueryCategoryResidualError("类别行数与全量 POI 不一致")
    covered_categories = np.asarray(
        category_indices[covered_rows], dtype=np.int32
    )
    if np.any(covered_categories < 0) or np.any(
        covered_categories >= config.category_count
    ):
        raise QueryCategoryResidualError("类别行号越界")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    started = time.perf_counter()
    started_at = _utc_now()
    try:
        counts = np.bincount(
            covered_categories, minlength=config.category_count
        ).astype(np.int64)
        dimension = query_aggregates.shape[1]
        sums = np.zeros((config.category_count, dimension), dtype=np.float64)
        progress = tqdm(
            total=config.expected_covered_pois,
            desc="E4 category means",
            unit="poi",
            disable=not show_progress,
        )
        for start in range(0, config.expected_covered_pois, config.chunk_rows):
            stop = min(start + config.chunk_rows, config.expected_covered_pois)
            chunk = np.asarray(query_aggregates[start:stop], dtype=np.float32)
            chunk_categories = covered_categories[start:stop]
            for category in np.unique(chunk_categories):
                sums[category] += chunk[chunk_categories == category].sum(
                    axis=0, dtype=np.float64
                )
            progress.update(stop - start)
        progress.close()
        supported = counts >= config.min_category_support
        means = np.zeros((config.category_count, dimension), dtype=np.float32)
        means[supported] = (
            sums[supported] / counts[supported, None]
        ).astype(np.float32)
        mean_norms = np.linalg.norm(means, axis=1)
        if np.any(mean_norms[supported] <= 0) or not np.isfinite(means).all():
            raise QueryCategoryResidualError("类别 Query 中心非法")

        counts_path = staging_dir / "category_counts.npy"
        means_path = staging_dir / "category_query_means.npy"
        with counts_path.open("wb") as handle:
            np.save(handle, counts, allow_pickle=False)
        with means_path.open("wb") as handle:
            np.save(handle, means, allow_pickle=False)

        outputs = {
            "category_counts": {
                "file": counts_path.name,
                "shape": list(counts.shape),
                "dtype": "int64",
                "sha256": _sha256_file(counts_path),
            },
            "category_query_means": {
                "file": means_path.name,
                "shape": list(means.shape),
                "dtype": "float32",
                "sha256": _sha256_file(means_path),
            },
            "residual_candidates": [
                {"beta": beta} for beta in config.betas
            ],
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config.experiment_id,
            "status": "completed",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "inputs": {
                "aggregates_dir": str(config.aggregates_dir.resolve()),
                "aggregates_manifest_sha256": (
                    config.expected_aggregates_manifest_sha256
                ),
                "category_indices": str(config.category_indices.resolve()),
                "category_indices_sha256": (
                    config.expected_category_indices_sha256
                ),
                "category_manifest": str(config.category_manifest.resolve()),
                "category_manifest_sha256": (
                    config.expected_category_manifest_sha256
                ),
            },
            "method": {
                "category_mean": (
                    "equal-POI raw mean of normalized E2 Query aggregates"
                ),
                "residual": "normalize(q_i - beta * mean_category(i))",
                "rare_category_policy": (
                    "mean=0 when covered support < min_category_support"
                ),
                "min_category_support": config.min_category_support,
                "betas": list(config.betas),
                "statistics_split": "Train Query aggregates + static POI category",
            },
            "stats": {
                "poi_rows": config.expected_poi_rows,
                "covered_pois": config.expected_covered_pois,
                "embedding_dim": dimension,
                "category_count": config.category_count,
                "nonempty_categories": int(np.sum(counts > 0)),
                "supported_categories": int(np.sum(supported)),
                "supported_pois": int(counts[supported].sum()),
                "rare_pois_unchanged": int(counts[~supported].sum()),
                "category_support_distribution": _distribution(counts[counts > 0]),
                "category_mean_norm_distribution": _distribution(
                    mean_norms[supported]
                ),
            },
            "outputs": outputs,
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "chunk_rows": config.chunk_rows,
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "git": _git_state(project_root),
        }
        _write_json_atomic(staging_dir / "manifest.json", manifest)
        (staging_dir / "_SUCCESS").touch()
        os.replace(staging_dir, output_dir)
        validated = validate_query_category_residual(
            output_dir,
            expected_aggregates_manifest_sha256=(
                config.expected_aggregates_manifest_sha256
            ),
            expected_category_indices_sha256=(
                config.expected_category_indices_sha256
            ),
        )
        return QueryCategoryResidualResult(
            output_dir=validated.output_dir,
            manifest_path=validated.manifest_path,
            covered_pois=validated.covered_pois,
            candidates=validated.candidates,
            reused=False,
        )
    except Exception as error:
        if staging_dir.exists():
            _write_json_atomic(
                staging_dir / "failure.json",
                {
                    "status": "failed",
                    "failed_at": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise

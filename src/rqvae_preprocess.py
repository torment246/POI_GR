"""Deterministic preprocessing for MobilityBench RQ-VAE training inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle
import sys

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RQVAEPreprocessResult:
    input_path: Path
    preprocess_path: Path
    train_indices: np.ndarray
    val_indices: np.ndarray
    input_shape: tuple[int, int]
    text_pca_dim: int
    geo_alpha: float | None
    rebuilt: bool
    summary: dict[str, Any]


def mode_suffix(mode: str, debug_max_rows: int | None = None) -> str:
    if mode not in {"semantic", "geo_fused"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return f"{mode}_debug" if debug_max_rows else mode


def output_path_for_mode(config_path: str | Path, mode: str, debug_max_rows: int | None = None) -> Path:
    path = Path(config_path)
    if debug_max_rows:
        return path.with_name(path.stem + "_debug" + path.suffix)
    return path


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise SystemExit("Failed to read parquet. Please install pyarrow: pip install pyarrow") from exc


def as_clean_text_list(series: pd.Series) -> list[str]:
    return series.fillna("").astype(str).str.strip().tolist()


def check_meta_alignment(embedding_meta_path: Path, geo_meta_path: Path) -> pd.DataFrame:
    embedding_meta = read_parquet(embedding_meta_path)
    geo_meta = read_parquet(geo_meta_path)
    for label, df in [("embedding_meta", embedding_meta), ("geo_meta", geo_meta)]:
        missing = [column for column in ["row_id", "poi_id"] if column not in df.columns]
        if missing:
            raise ValueError(f"{label} missing required columns: {missing}")
    if len(embedding_meta) != len(geo_meta):
        raise ValueError(f"embedding_meta rows ({len(embedding_meta)}) != geo_meta rows ({len(geo_meta)})")
    if as_clean_text_list(embedding_meta["poi_id"]) != as_clean_text_list(geo_meta["poi_id"]):
        raise ValueError("poi_id order differs between embedding meta and geo meta")
    if as_clean_text_list(embedding_meta["row_id"]) != as_clean_text_list(geo_meta["row_id"]):
        raise ValueError("row_id order differs between embedding meta and geo meta")
    if embedding_meta["poi_id"].nunique() != len(embedding_meta):
        raise ValueError("poi_id must be unique in embedding meta")
    return embedding_meta


def validate_input_arrays(
    text_embeddings: np.ndarray,
    geo_features: np.ndarray,
    embedding_meta: pd.DataFrame,
    mode: str,
) -> None:
    if text_embeddings.ndim != 2:
        raise ValueError(f"text embeddings must be 2D, got shape={text_embeddings.shape}")
    if text_embeddings.shape[0] != len(embedding_meta):
        raise ValueError(f"text embedding rows ({text_embeddings.shape[0]}) != embedding meta rows ({len(embedding_meta)})")
    if np.isnan(text_embeddings).any():
        raise ValueError("text embeddings contain NaN")
    if np.isinf(text_embeddings).any():
        raise ValueError("text embeddings contain Inf")

    if mode == "geo_fused":
        if geo_features.ndim != 2:
            raise ValueError(f"geo features must be 2D, got shape={geo_features.shape}")
        if geo_features.shape[0] != len(embedding_meta):
            raise ValueError(f"geo feature rows ({geo_features.shape[0]}) != embedding meta rows ({len(embedding_meta)})")
        if np.isnan(geo_features).any():
            raise ValueError("geo features contain NaN")
        if np.isinf(geo_features).any():
            raise ValueError("geo features contain Inf")


def make_train_val_indices(n_rows: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_rows <= 1:
        raise ValueError(f"Need at least two rows for train/val split, got {n_rows}")
    val_size = max(1, int(round(n_rows * float(val_ratio))))
    if val_size >= n_rows:
        val_size = 1
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_rows)
    val_indices = np.sort(permutation[:val_size]).astype(np.int64)
    train_indices = np.sort(permutation[val_size:]).astype(np.int64)
    return train_indices, val_indices


def fit_text_pca_and_scaler(
    text_embeddings: np.ndarray,
    train_indices: np.ndarray,
    text_pca_dim: int,
    random_state: int,
) -> tuple[Any, Any, np.ndarray]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("scikit-learn is required. Please install scikit-learn: pip install scikit-learn") from exc

    train_rows, input_dim = len(train_indices), text_embeddings.shape[1]
    if text_pca_dim <= 0:
        raise ValueError(f"text_pca_dim must be positive, got {text_pca_dim}")
    if text_pca_dim > min(train_rows, input_dim):
        raise ValueError(
            f"text_pca_dim ({text_pca_dim}) cannot exceed min(train_rows, input_dim)={min(train_rows, input_dim)}"
        )

    pca = PCA(n_components=text_pca_dim, svd_solver="randomized", random_state=random_state)
    train_text = np.asarray(text_embeddings[train_indices], dtype=np.float32)
    pca.fit(train_text)
    train_pca = pca.transform(train_text).astype(np.float32, copy=False)
    scaler = StandardScaler()
    scaler.fit(train_pca)
    transformed = scaler.transform(pca.transform(np.asarray(text_embeddings, dtype=np.float32))).astype(np.float32)
    return pca, scaler, transformed


def input_numeric_summary(x: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(x, dtype=np.float32)
    means = arr.astype(np.float64).mean(axis=0)
    stds = arr.astype(np.float64).std(axis=0)
    return {
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "input_mean_abs_max": float(np.max(np.abs(means))) if arr.size else 0.0,
        "input_std_min": float(stds.min()) if arr.size else 0.0,
        "input_std_max": float(stds.max()) if arr.size else 0.0,
    }


def geo_numeric_summary(geo_features: np.ndarray) -> dict[str, Any]:
    geo = np.asarray(geo_features, dtype=np.float32)
    means = geo.astype(np.float64).mean(axis=0)
    stds = geo.astype(np.float64).std(axis=0)
    return {
        "geo_mean_abs_max": float(np.max(np.abs(means))) if geo.size else 0.0,
        "geo_std_min": float(stds.min()) if geo.size else 0.0,
        "geo_std_max": float(stds.max()) if geo.size else 0.0,
    }


def save_preprocess(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def install_numpy_pickle_compat_aliases() -> None:
    """Allow preprocess pickles written by NumPy 1.x/2.x to load across envs."""
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_core_multiarray
        import numpy.core.numeric as numpy_core_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_core_multiarray)
        sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
    except Exception:
        pass


def load_preprocess(path: Path) -> dict[str, Any]:
    install_numpy_pickle_compat_aliases()
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid preprocess payload: {path}")
    return payload


def prepare_rqvae_input(
    mode: str,
    text_embeddings_path: Path,
    embedding_meta_path: Path,
    geo_features_path: Path,
    geo_meta_path: Path,
    output_input_path: Path,
    preprocess_path: Path,
    text_pca_dim: int,
    geo_alpha: float,
    val_ratio: float,
    seed: int,
    pca_random_state: int,
    debug_max_rows: int | None = None,
    force_rebuild: bool = False,
) -> RQVAEPreprocessResult:
    if mode not in {"semantic", "geo_fused"}:
        raise ValueError(f"Unsupported mode: {mode}")

    embedding_meta = check_meta_alignment(embedding_meta_path, geo_meta_path)
    text_embeddings_for_check = np.load(text_embeddings_path, mmap_mode="r")
    geo_features_for_check = np.load(geo_features_path, mmap_mode="r")
    validate_input_arrays(text_embeddings_for_check, geo_features_for_check, embedding_meta, mode)
    expected_rows = min(int(debug_max_rows), len(embedding_meta)) if debug_max_rows is not None else len(embedding_meta)

    if output_input_path.exists() and preprocess_path.exists() and not force_rebuild:
        payload = load_preprocess(preprocess_path)
        x = np.load(output_input_path, mmap_mode="r")
        payload_rows = int(payload.get("input_shape", [0, 0])[0])
        if x.shape[0] != payload_rows:
            raise ValueError(f"Existing input rows ({x.shape[0]}) do not match preprocess rows ({payload_rows})")
        if x.shape[0] != expected_rows:
            raise ValueError(f"Existing input rows ({x.shape[0]}) do not match current source rows ({expected_rows})")
        return RQVAEPreprocessResult(
            input_path=output_input_path,
            preprocess_path=preprocess_path,
            train_indices=np.asarray(payload["train_indices"], dtype=np.int64),
            val_indices=np.asarray(payload["val_indices"], dtype=np.int64),
            input_shape=tuple(x.shape),
            text_pca_dim=int(payload["text_pca_dim"]),
            geo_alpha=payload.get("geo_alpha"),
            rebuilt=False,
            summary=dict(payload.get("summary", {})),
        )

    text_embeddings = np.load(text_embeddings_path)
    geo_features = np.load(geo_features_path)

    if debug_max_rows is not None:
        if debug_max_rows <= 0:
            raise ValueError(f"debug_max_rows must be positive, got {debug_max_rows}")
        keep = min(int(debug_max_rows), text_embeddings.shape[0])
        text_embeddings = text_embeddings[:keep]
        geo_features = geo_features[:keep]
        embedding_meta = embedding_meta.iloc[:keep].reset_index(drop=True)

    train_indices, val_indices = make_train_val_indices(len(embedding_meta), val_ratio, seed)
    pca, scaler, text_ready = fit_text_pca_and_scaler(text_embeddings, train_indices, text_pca_dim, pca_random_state)

    summary: dict[str, Any] = {
        "mode": mode,
        "source_rows": int(len(embedding_meta)),
        "text_embeddings_shape": tuple(int(x) for x in text_embeddings.shape),
        "text_pca_dim": int(text_pca_dim),
        "train_rows": int(len(train_indices)),
        "val_rows": int(len(val_indices)),
        "val_ratio": float(val_ratio),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }

    if mode == "semantic":
        x = text_ready
        actual_geo_alpha: float | None = None
    else:
        geo = np.asarray(geo_features, dtype=np.float32)
        summary.update(geo_numeric_summary(geo))
        x = np.concatenate([text_ready, float(geo_alpha) * geo], axis=1).astype(np.float32, copy=False)
        actual_geo_alpha = float(geo_alpha)
        summary["geo_features_shape"] = tuple(int(v) for v in geo.shape)
        summary["geo_alpha"] = actual_geo_alpha

    x = np.asarray(x, dtype=np.float32)
    summary.update(input_numeric_summary(x))
    ensure_parent_dir(output_input_path)
    np.save(output_input_path, x)

    payload = {
        "mode": mode,
        "pca": pca,
        "scaler": scaler,
        "text_pca_dim": int(text_pca_dim),
        "geo_alpha": actual_geo_alpha,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "input_shape": tuple(int(v) for v in x.shape),
        "seed": int(seed),
        "pca_random_state": int(pca_random_state),
        "debug_max_rows": debug_max_rows,
        "summary": summary,
        "embedding_meta_columns": embedding_meta.columns.tolist(),
    }
    save_preprocess(preprocess_path, payload)
    return RQVAEPreprocessResult(
        input_path=output_input_path,
        preprocess_path=preprocess_path,
        train_indices=train_indices,
        val_indices=val_indices,
        input_shape=tuple(int(v) for v in x.shape),
        text_pca_dim=int(text_pca_dim),
        geo_alpha=actual_geo_alpha,
        rebuilt=True,
        summary=summary,
    )

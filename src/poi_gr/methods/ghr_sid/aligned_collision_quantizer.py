"""Build globally aligned two-token suffixes inside frozen TIGER buckets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from poi_gr.methods.tiger.identifier import sha256_file


SCHEMA_VERSION = "ghr-aligned-collision-quantizer-v1"
METRICS_SCHEMA_VERSION = "ghr-aligned-collision-quantizer-metrics-v1"


class AlignedCollisionQuantizerError(ValueError):
    """Raised when aligned collision quantization violates its contract."""


@dataclass(frozen=True)
class AlignedCollisionConfig:
    """Frozen inputs and hyperparameters for one suffix build."""

    experiment_id: str
    base_sid_codes: Path
    base_sid_manifest: Path
    embeddings: Path
    poi_ids: Path
    poi_dir: Path
    rqvae_checkpoint: Path
    rqvae_resolved_config: Path
    output_dir: Path
    codebook_size: int
    geo_weight: float
    geo_scales_km: tuple[float, ...]
    kmeans_sample_size: int
    kmeans_batch_size: int
    kmeans_max_iter: int
    kmeans_n_init: int
    candidate_first: int
    candidate_second: int
    encode_batch_size: int
    search_batch_size: int
    seed: int
    device: str


@dataclass(frozen=True)
class AlignedCollisionResult:
    """Completed static identifier build."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlignedCollisionQuantizerError(
            f"无法读取{name}：{path}：{error}"
        ) from error
    if not isinstance(payload, dict):
        raise AlignedCollisionQuantizerError(f"{name}必须是 JSON object")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlignedCollisionQuantizerError(f"{name}必须是 mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AlignedCollisionQuantizerError(f"{name}必须是正整数")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignedCollisionQuantizerError(f"{name}必须是正数")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise AlignedCollisionQuantizerError(f"{name}必须是有限正数")
    return result


def _rooted(path: Any, project_root: Path, name: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise AlignedCollisionQuantizerError(f"{name}必须是非空路径")
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else project_root / candidate).resolve()


def load_aligned_collision_config(
    path: Path, project_root: Path
) -> AlignedCollisionConfig:
    """Load one YAML experiment config and resolve repository-relative paths."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AlignedCollisionQuantizerError(
            f"无法读取配置：{path}：{error}"
        ) from error
    root = _mapping(raw, "config")
    data = _mapping(root.get("data"), "data")
    quantizer = _mapping(root.get("quantizer"), "quantizer")
    runtime = _mapping(root.get("runtime"), "runtime")
    experiment_id = root.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise AlignedCollisionQuantizerError("experiment_id 必须是非空字符串")
    scales_raw = quantizer.get("geo_scales_km")
    if not isinstance(scales_raw, list) or not scales_raw:
        raise AlignedCollisionQuantizerError("quantizer.geo_scales_km 必须是非空数组")
    scales = tuple(
        _positive_float(value, "quantizer.geo_scales_km") for value in scales_raw
    )
    device = runtime.get("device", "auto")
    if device not in {"auto", "cpu", "cuda"}:
        raise AlignedCollisionQuantizerError("runtime.device 只能是 auto/cpu/cuda")
    return AlignedCollisionConfig(
        experiment_id=experiment_id,
        base_sid_codes=_rooted(
            data.get("base_sid_codes"), project_root, "data.base_sid_codes"
        ),
        base_sid_manifest=_rooted(
            data.get("base_sid_manifest"), project_root, "data.base_sid_manifest"
        ),
        embeddings=_rooted(data.get("embeddings"), project_root, "data.embeddings"),
        poi_ids=_rooted(data.get("poi_ids"), project_root, "data.poi_ids"),
        poi_dir=_rooted(data.get("poi_dir"), project_root, "data.poi_dir"),
        rqvae_checkpoint=_rooted(
            data.get("rqvae_checkpoint"), project_root, "data.rqvae_checkpoint"
        ),
        rqvae_resolved_config=_rooted(
            data.get("rqvae_resolved_config"),
            project_root,
            "data.rqvae_resolved_config",
        ),
        output_dir=_rooted(root.get("output_dir"), project_root, "output_dir"),
        codebook_size=_positive_int(
            quantizer.get("codebook_size"), "quantizer.codebook_size"
        ),
        geo_weight=_positive_float(
            quantizer.get("geo_weight"), "quantizer.geo_weight"
        ),
        geo_scales_km=scales,
        kmeans_sample_size=_positive_int(
            quantizer.get("kmeans_sample_size"), "quantizer.kmeans_sample_size"
        ),
        kmeans_batch_size=_positive_int(
            quantizer.get("kmeans_batch_size"), "quantizer.kmeans_batch_size"
        ),
        kmeans_max_iter=_positive_int(
            quantizer.get("kmeans_max_iter"), "quantizer.kmeans_max_iter"
        ),
        kmeans_n_init=_positive_int(
            quantizer.get("kmeans_n_init"), "quantizer.kmeans_n_init"
        ),
        candidate_first=_positive_int(
            quantizer.get("candidate_first"), "quantizer.candidate_first"
        ),
        candidate_second=_positive_int(
            quantizer.get("candidate_second"), "quantizer.candidate_second"
        ),
        encode_batch_size=_positive_int(
            runtime.get("encode_batch_size"), "runtime.encode_batch_size"
        ),
        search_batch_size=_positive_int(
            runtime.get("search_batch_size"), "runtime.search_batch_size"
        ),
        seed=int(runtime.get("seed", 42)),
        device=str(device),
    )


def multiscale_geo_features(
    coordinates: np.ndarray,
    scales_km: Sequence[float],
    *,
    origin_lng_lat: Sequence[float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Encode absolute POI coordinates with deterministic multi-scale Fourier bands."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise AlignedCollisionQuantizerError("coordinates 必须是 [N,2] lng/lat")
    if not np.isfinite(coordinates).all():
        raise AlignedCollisionQuantizerError("coordinates 包含 NaN/Inf")
    scales = np.asarray(scales_km, dtype=np.float64)
    if scales.ndim != 1 or not len(scales) or np.any(scales <= 0):
        raise AlignedCollisionQuantizerError("scales_km 必须是一维有限正数")
    if origin_lng_lat is None:
        origin = coordinates.mean(axis=0, dtype=np.float64)
    else:
        origin = np.asarray(origin_lng_lat, dtype=np.float64)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise AlignedCollisionQuantizerError("origin_lng_lat 必须是两个有限数")
    longitude_scale = 111.320 * math.cos(math.radians(float(origin[1])))
    x_km = (coordinates[:, 0] - origin[0]) * longitude_scale
    y_km = (coordinates[:, 1] - origin[1]) * 110.574
    phases_x = 2.0 * math.pi * x_km[:, None] / scales[None, :]
    phases_y = 2.0 * math.pi * y_km[:, None] / scales[None, :]
    features = np.concatenate(
        [np.sin(phases_x), np.cos(phases_x), np.sin(phases_y), np.cos(phases_y)],
        axis=1,
    ).astype(np.float32, copy=False)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features /= np.maximum(norms, 1e-12)
    return features, (float(origin[0]), float(origin[1]))


def compose_collision_features(
    latent_residuals: np.ndarray,
    geo_features: np.ndarray,
    bucket_ids: np.ndarray,
    *,
    geo_weight: float,
) -> np.ndarray:
    """Center semantic residuals by TIGER bucket and concatenate weighted geography."""

    latent = np.asarray(latent_residuals, dtype=np.float32)
    geo = np.asarray(geo_features, dtype=np.float32)
    buckets = np.asarray(bucket_ids)
    if latent.ndim != 2 or geo.ndim != 2 or len(latent) != len(geo):
        raise AlignedCollisionQuantizerError("latent/geo 必须是同长度二维矩阵")
    if buckets.shape != (len(latent),):
        raise AlignedCollisionQuantizerError("bucket_ids shape 必须是 [N]")
    if geo_weight <= 0 or not math.isfinite(geo_weight):
        raise AlignedCollisionQuantizerError("geo_weight 必须是有限正数")
    centered = latent.copy()
    order = np.argsort(buckets, kind="stable")
    ordered_buckets = buckets[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_buckets[1:] != ordered_buckets[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends, strict=True):
        rows = order[start:end]
        centered[rows] -= centered[rows].mean(axis=0, dtype=np.float64)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    centered /= np.maximum(norms, 1e-12)
    return np.ascontiguousarray(
        np.concatenate([centered, geo * float(geo_weight)], axis=1),
        dtype=np.float32,
    )


def _write_collision_features(
    latent_residuals: np.ndarray,
    geo_features: np.ndarray,
    bucket_ids: np.ndarray,
    output_path: Path,
    *,
    geo_weight: float,
    progress: Callable[[str], None] | None,
) -> np.memmap:
    """Write collision features without materializing the full matrix in RAM."""

    latent = np.asarray(latent_residuals)
    geo = np.asarray(geo_features, dtype=np.float32)
    buckets = np.asarray(bucket_ids)
    if latent.ndim != 2 or geo.ndim != 2 or len(latent) != len(geo):
        raise AlignedCollisionQuantizerError("latent/geo 必须是同长度二维矩阵")
    if buckets.shape != (len(latent),):
        raise AlignedCollisionQuantizerError("bucket_ids shape 必须是 [N]")
    features = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(latent), latent.shape[1] + geo.shape[1]),
    )
    order = np.argsort(buckets, kind="stable")
    ordered_buckets = buckets[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_buckets[1:] != ordered_buckets[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    for group_index, (start, end) in enumerate(
        zip(starts, ends, strict=True), start=1
    ):
        rows = order[start:end]
        values = np.asarray(latent[rows], dtype=np.float32)
        values -= values.mean(axis=0, dtype=np.float64)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values /= np.maximum(norms, 1e-12)
        features[rows, : latent.shape[1]] = values
        if progress is not None and group_index % 50_000 == 0:
            progress(
                f"桶内残差中心化 {group_index:,}/{len(starts):,} 个碰撞桶"
            )
    for start in range(0, len(geo), 65_536):
        end = min(start + 65_536, len(geo))
        features[start:end, latent.shape[1] :] = geo[start:end] * float(
            geo_weight
        )
    features.flush()
    return features


def _fit_shared_residual_anchors(
    features: np.ndarray,
    bucket_sizes: np.ndarray,
    config: AlignedCollisionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.cluster import MiniBatchKMeans

    row_count = len(features)
    sample_rows = min(config.kmeans_sample_size, row_count)
    rng = np.random.default_rng(config.seed)
    sample_indices = np.sort(
        rng.choice(row_count, size=sample_rows, replace=False)
    ).astype(np.int64, copy=False)
    sample = np.ascontiguousarray(features[sample_indices], dtype=np.float32)
    weights = np.reciprocal(
        np.asarray(bucket_sizes[sample_indices], dtype=np.float64)
    )
    first = MiniBatchKMeans(
        n_clusters=config.codebook_size,
        init="k-means++",
        n_init=config.kmeans_n_init,
        max_iter=config.kmeans_max_iter,
        batch_size=config.kmeans_batch_size,
        random_state=config.seed,
        reassignment_ratio=0.0,
    )
    first_labels = first.fit_predict(sample, sample_weight=weights)
    first_centers = np.asarray(first.cluster_centers_, dtype=np.float32)
    residual = sample - first_centers[first_labels]
    second = MiniBatchKMeans(
        n_clusters=config.codebook_size,
        init="k-means++",
        n_init=config.kmeans_n_init,
        max_iter=config.kmeans_max_iter,
        batch_size=config.kmeans_batch_size,
        random_state=config.seed + 1,
        reassignment_ratio=0.0,
    )
    second.fit(residual, sample_weight=weights)
    second_centers = np.asarray(second.cluster_centers_, dtype=np.float32)
    metadata = {
        "algorithm": "bucket-balanced shared MiniBatchKMeans residual anchors",
        "sample_rows": sample_rows,
        "sample_indices_sha256": hashlib.sha256(
            sample_indices.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "first_inertia": float(first.inertia_),
        "second_inertia": float(second.inertia_),
    }
    return first_centers, second_centers, metadata


def _pair_candidates(
    features: np.ndarray,
    first_centers: np.ndarray,
    second_centers: np.ndarray,
    *,
    first_count: int,
    second_count: int,
    batch_size: int,
    device_name: str,
    candidate_path: Path,
    distance_path: Path,
    progress: Callable[[str], None] | None,
) -> tuple[np.memmap, np.memmap]:
    import torch

    codebook_size = len(first_centers)
    first_count = min(first_count, codebook_size)
    second_count = min(second_count, len(second_centers))
    candidate_count = first_count * second_count
    candidates = np.lib.format.open_memmap(
        candidate_path,
        mode="w+",
        dtype=np.int16 if codebook_size * codebook_size <= 32767 else np.int32,
        shape=(len(features), candidate_count),
    )
    distances = np.lib.format.open_memmap(
        distance_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(features), candidate_count),
    )
    device = torch.device(device_name)
    first_tensor = torch.from_numpy(
        np.ascontiguousarray(first_centers)
    ).to(device)
    second_tensor = torch.from_numpy(
        np.ascontiguousarray(second_centers)
    ).to(device)

    def squared_distances(
        values: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        return (
            values.square().sum(dim=1, keepdim=True)
            + centers.square().sum(dim=1).unsqueeze(0)
            - 2.0 * values @ centers.t()
        ).clamp_min_(0.0)

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        batch = np.ascontiguousarray(features[start:end], dtype=np.float32)
        batch_tensor = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            _, first_ids = torch.topk(
                squared_distances(batch_tensor, first_tensor),
                k=first_count,
                dim=1,
                largest=False,
                sorted=True,
            )
            pair_parts: list[torch.Tensor] = []
            distance_parts: list[torch.Tensor] = []
            for rank in range(first_count):
                first_id = first_ids[:, rank]
                residual = batch_tensor - first_tensor[first_id]
                _, second_ids = torch.topk(
                    squared_distances(residual, second_tensor),
                    k=second_count,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                pair_parts.append(
                    first_id[:, None] * codebook_size + second_ids
                )
                prototypes = (
                    first_tensor[first_id, None, :]
                    + second_tensor[second_ids]
                )
                distance_parts.append(
                    (batch_tensor[:, None, :] - prototypes).square().sum(dim=2)
                )
            pair_tensor = torch.cat(pair_parts, dim=1)
            distance_tensor = torch.cat(distance_parts, dim=1)
            sorted_distances, rank_order = torch.sort(
                distance_tensor, dim=1, stable=True
            )
            sorted_pairs = torch.gather(pair_tensor, 1, rank_order)
        candidates[start:end] = sorted_pairs.cpu().numpy().astype(
            candidates.dtype, copy=False
        )
        distances[start:end] = sorted_distances.cpu().numpy()
        if progress is not None:
            progress(f"全局后缀候选检索 {end:,}/{len(features):,}")
    candidates.flush()
    distances.flush()
    return candidates, distances


def assign_unique_pairs(
    features: np.ndarray,
    bucket_ids: np.ndarray,
    candidates: np.ndarray,
    candidate_distances: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign one unique global pair per frozen TIGER bucket."""

    from scipy.optimize import linear_sum_assignment

    features = np.asarray(features, dtype=np.float32)
    bucket_ids = np.asarray(bucket_ids)
    candidates = np.asarray(candidates)
    candidate_distances = np.asarray(candidate_distances, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if candidates.shape != candidate_distances.shape or len(candidates) != len(features):
        raise AlignedCollisionQuantizerError("候选 ID/距离 shape 不一致")
    if bucket_ids.shape != (len(features),):
        raise AlignedCollisionQuantizerError("bucket_ids shape 必须是 [N]")
    pair_count = len(prototypes)
    if pair_count < 1 or np.any(candidates < 0) or np.any(candidates >= pair_count):
        raise AlignedCollisionQuantizerError("候选 pair 超出全局原型范围")

    order = np.argsort(bucket_ids, kind="stable")
    ordered_buckets = bucket_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_buckets[1:] != ordered_buckets[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    assigned = np.full(len(features), -1, dtype=np.int32)
    direct_bucket_count = 0
    constrained_bucket_count = 0
    full_fallback_bucket_count = 0
    forced_row_count = 0
    nearest_error = 0.0
    assigned_error = 0.0

    for start, end in zip(starts, ends, strict=True):
        rows = order[start:end]
        nearest = candidates[rows, 0].astype(np.int32, copy=False)
        nearest_error += float(candidate_distances[rows, 0].sum(dtype=np.float64))
        if len(np.unique(nearest)) == len(rows):
            chosen = nearest.copy()
            direct_bucket_count += 1
        else:
            constrained_bucket_count += 1
            union = np.unique(candidates[rows].reshape(-1)).astype(np.int32)
            chosen = np.empty(len(rows), dtype=np.int32)
            solved = False
            if len(union) >= len(rows):
                lookup = {int(pair): index for index, pair in enumerate(union)}
                cost = np.full((len(rows), len(union)), np.inf, dtype=np.float64)
                for local_row, source_row in enumerate(rows):
                    for pair, distance in zip(
                        candidates[source_row],
                        candidate_distances[source_row],
                        strict=True,
                    ):
                        column = lookup[int(pair)]
                        cost[local_row, column] = min(
                            cost[local_row, column], float(distance)
                        )
                try:
                    row_ids, column_ids = linear_sum_assignment(cost)
                    if len(row_ids) == len(rows) and np.isfinite(
                        cost[row_ids, column_ids]
                    ).all():
                        chosen[row_ids] = union[column_ids]
                        solved = True
                except ValueError:
                    solved = False
            if not solved:
                full_fallback_bucket_count += 1
                batch = features[rows]
                cost = (
                    np.square(batch[:, None, :] - prototypes[None, :, :])
                    .sum(axis=2, dtype=np.float64)
                )
                row_ids, column_ids = linear_sum_assignment(cost)
                if len(row_ids) != len(rows):
                    raise AlignedCollisionQuantizerError("全量唯一分配未覆盖桶内所有 POI")
                chosen[row_ids] = column_ids.astype(np.int32)
        if len(np.unique(chosen)) != len(rows):
            raise AlignedCollisionQuantizerError("桶内全局 pair 分配仍有重复")
        assigned[rows] = chosen
        forced_row_count += int(np.count_nonzero(chosen != nearest))
        assigned_error += float(
            np.square(features[rows] - prototypes[chosen])
            .sum(axis=1, dtype=np.float64)
            .sum(dtype=np.float64)
        )

    if np.any(assigned < 0):
        raise AlignedCollisionQuantizerError("存在未分配的碰撞 POI")
    row_count = len(features)
    return assigned, {
        "collision_bucket_count": int(len(starts)),
        "direct_nearest_unique_bucket_count": direct_bucket_count,
        "constrained_bucket_count": constrained_bucket_count,
        "full_candidate_fallback_bucket_count": full_fallback_bucket_count,
        "forced_reassignment_row_count": forced_row_count,
        "forced_reassignment_row_ratio": forced_row_count / row_count,
        "nearest_mse_per_feature": nearest_error / (row_count * features.shape[1]),
        "assigned_mse_per_feature": assigned_error / (row_count * features.shape[1]),
        "assignment_mse_penalty_ratio": (
            0.0 if nearest_error == 0 else assigned_error / nearest_error - 1.0
        ),
    }


def compose_five_layer_identifiers(
    base_sid_codes: np.ndarray,
    collision_rows: np.ndarray,
    collision_pairs: np.ndarray,
    codebook_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Append two fixed suffix tokens; singleton suffixes remain zero."""

    base = np.asarray(base_sid_codes)
    rows = np.asarray(collision_rows, dtype=np.int64)
    pairs = np.asarray(collision_pairs)
    if base.ndim != 2 or base.shape[1] != 3:
        raise AlignedCollisionQuantizerError("base_sid_codes 必须是 [N,3]")
    if rows.ndim != 1 or pairs.shape != (len(rows),):
        raise AlignedCollisionQuantizerError("collision_rows/pairs shape 不一致")
    if np.any(pairs < 0) or np.any(pairs >= codebook_size * codebook_size):
        raise AlignedCollisionQuantizerError("collision_pairs 超出两层容量")
    suffix = np.zeros((len(base), 2), dtype=np.int32)
    suffix[rows, 0] = pairs // codebook_size
    suffix[rows, 1] = pairs % codebook_size
    identifiers = np.empty((len(base), 5), dtype=np.int32)
    identifiers[:, :3] = base
    identifiers[:, 3:] = suffix
    return identifiers, suffix


def _read_poi_ids(path: Path, expected_rows: int) -> list[str]:
    result: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                poi_id = json.loads(line)
            except json.JSONDecodeError as error:
                raise AlignedCollisionQuantizerError(
                    f"POI ID 第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(poi_id, str) or not poi_id:
                raise AlignedCollisionQuantizerError("POI ID 必须是非空字符串")
            result.append(poi_id)
    if len(result) != expected_rows:
        raise AlignedCollisionQuantizerError(
            f"POI ID 行数 {len(result)} != {expected_rows}"
        )
    return result


def _load_collision_coordinates(
    poi_dir: Path,
    poi_ids: Sequence[str],
    collision_rows: np.ndarray,
    *,
    progress: Callable[[str], None] | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    files = tuple(sorted(poi_dir.glob("part-*.json")))
    if not files:
        raise AlignedCollisionQuantizerError(f"POI 目录没有 part-*.json：{poi_dir}")
    output = np.empty((len(collision_rows), 2), dtype=np.float64)
    compact = 0
    full_row = 0
    sources: list[dict[str, Any]] = []
    for path in files:
        digest = hashlib.sha256()
        source_rows = 0
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                try:
                    record = json.loads(raw_line)
                    raw_poi_id = str(record["poi_id"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise AlignedCollisionQuantizerError(
                        f"POI JSON 非法：{path.name}:{line_number}"
                    ) from error
                if full_row >= len(poi_ids) or raw_poi_id != poi_ids[full_row]:
                    raise AlignedCollisionQuantizerError(
                        f"POI 与 Embedding ID 行错位：full_row={full_row}"
                    )
                if compact < len(collision_rows) and full_row == int(
                    collision_rows[compact]
                ):
                    try:
                        longitude = float(record["lng"])
                        latitude = float(record["lat"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise AlignedCollisionQuantizerError("POI 坐标非法") from error
                    if not (
                        math.isfinite(longitude)
                        and math.isfinite(latitude)
                        and -180 <= longitude <= 180
                        and -90 <= latitude <= 90
                    ):
                        raise AlignedCollisionQuantizerError("POI 坐标越界或非 finite")
                    output[compact] = (longitude, latitude)
                    compact += 1
                full_row += 1
                source_rows += 1
        sources.append(
            {
                "path": str(path.resolve()),
                "rows": source_rows,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
        if progress is not None:
            progress(
                f"POI 坐标对齐 {full_row:,}/{len(poi_ids):,}；碰撞 POI {compact:,}/"
                f"{len(collision_rows):,}"
            )
    if full_row != len(poi_ids) or compact != len(collision_rows):
        raise AlignedCollisionQuantizerError("POI 坐标回查没有完整覆盖输入")
    return output, sources


def _encode_tiger_latent_residuals(
    config: AlignedCollisionConfig,
    base_sid_codes: np.ndarray,
    collision_rows: np.ndarray,
    output_path: Path,
    *,
    progress: Callable[[str], None] | None,
) -> tuple[np.memmap, dict[str, Any]]:
    import torch

    from poi_gr.sid.training import _config_from_payload, build_model

    resolved = _load_json(config.rqvae_resolved_config, "RQ-VAE resolved config")
    if resolved.get("status") != "completed":
        raise AlignedCollisionQuantizerError("TIGER RQ-VAE 训练状态不是 completed")
    training_config = _config_from_payload(
        _mapping(resolved.get("config"), "resolved.config")
    )
    if tuple(training_config.codebook_sizes) != (1024, 1024, 1024):
        raise AlignedCollisionQuantizerError("冻结 TIGER 必须是 1024×1024×1024")
    requested = config.device
    if requested == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = requested
    if device_name == "cuda" and not torch.cuda.is_available():
        raise AlignedCollisionQuantizerError("请求 CUDA，但当前进程无可用 GPU")
    device = torch.device(device_name)
    checkpoint = torch.load(
        config.rqvae_checkpoint, map_location=device, weights_only=False
    )
    if checkpoint.get("config_signature") != resolved.get("config_signature"):
        raise AlignedCollisionQuantizerError("RQ-VAE checkpoint 配置签名不一致")
    model = build_model(training_config).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    embeddings = np.load(config.embeddings, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != (len(base_sid_codes), training_config.input_dim):
        raise AlignedCollisionQuantizerError("Embedding shape 与 TIGER 模型不一致")
    residuals = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(collision_rows), training_config.latent_dim),
    )
    codebooks = [
        codebook.weight.detach() for codebook in model.quantizer.codebooks
    ]
    # Collision rows are sorted, but NumPy advanced indexing on an OFS-backed
    # mmap still triggers many small page reads. Scan the source NPY once in
    # contiguous blocks, then select collision rows from the resident block.
    source_block_rows = max(32_768, config.encode_batch_size * 4)
    compact_start = 0
    with torch.no_grad():
        for full_start in range(0, len(embeddings), source_block_rows):
            full_end = min(full_start + source_block_rows, len(embeddings))
            compact_end = int(
                np.searchsorted(collision_rows, full_end, side="left")
            )
            if compact_end == compact_start:
                continue
            source_rows = collision_rows[compact_start:compact_end]
            source_block = np.asarray(
                embeddings[full_start:full_end], dtype=np.float32
            )
            selected = np.ascontiguousarray(
                source_block[source_rows - full_start], dtype=np.float32
            )
            for local_start in range(0, len(selected), config.encode_batch_size):
                local_end = min(local_start + config.encode_batch_size, len(selected))
                output_start = compact_start + local_start
                output_end = compact_start + local_end
                inputs = torch.from_numpy(selected[local_start:local_end]).to(device)
                latent = model.encode(inputs)
                codes = np.asarray(
                    base_sid_codes[source_rows[local_start:local_end]],
                    dtype=np.int64,
                )
                quantized = torch.zeros_like(latent)
                for level, codebook in enumerate(codebooks):
                    indices = torch.from_numpy(codes[:, level]).to(device)
                    quantized += codebook[indices]
                residuals[output_start:output_end] = (
                    latent - quantized
                ).cpu().numpy()
            compact_start = compact_end
            if progress is not None:
                progress(
                    f"TIGER 末端残差编码 {compact_end:,}/{len(collision_rows):,}；"
                    f"连续源行 {full_end:,}/{len(embeddings):,}"
                )
    if compact_start != len(collision_rows):
        raise AlignedCollisionQuantizerError("连续扫描未覆盖全部碰撞行")
    residuals.flush()
    return residuals, {
        "device": device_name,
        "latent_dim": training_config.latent_dim,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "torch": torch.__version__,
    }


def _write_mapping(
    path: Path,
    poi_ids: Sequence[str],
    identifiers: np.ndarray,
    bucket_sizes_by_row: np.ndarray,
    *,
    chunk_rows: int = 100_000,
) -> None:
    schema = pa.schema(
        [
            pa.field("poi_id", pa.string(), nullable=False),
            *(pa.field(f"s{level}", pa.int32(), nullable=False) for level in range(1, 4)),
            pa.field("r1", pa.int32(), nullable=False),
            pa.field("r2", pa.int32(), nullable=False),
            pa.field("base_sid_bucket_size", pa.int32(), nullable=False),
            pa.field("has_semantic_collision", pa.bool_(), nullable=False),
        ]
    )
    writer = pq.ParquetWriter(path, schema, compression="zstd", version="2.6")
    try:
        for start in range(0, len(poi_ids), chunk_rows):
            end = min(start + chunk_rows, len(poi_ids))
            codes = identifiers[start:end]
            sizes = bucket_sizes_by_row[start:end]
            table = pa.Table.from_arrays(
                [
                    pa.array(poi_ids[start:end], type=pa.string()),
                    *(pa.array(codes[:, column], type=pa.int32()) for column in range(5)),
                    pa.array(sizes, type=pa.int32()),
                    pa.array(sizes > 1, type=pa.bool_()),
                ],
                schema=schema,
            )
            writer.write_table(table)
    finally:
        writer.close()


def _git_state(project_root: Path) -> dict[str, Any]:
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


def _validate_inputs(config: AlignedCollisionConfig) -> None:
    for name, path in (
        ("base SID", config.base_sid_codes),
        ("base SID manifest", config.base_sid_manifest),
        ("Embedding", config.embeddings),
        ("POI IDs", config.poi_ids),
        ("RQ-VAE checkpoint", config.rqvae_checkpoint),
        ("RQ-VAE resolved config", config.rqvae_resolved_config),
    ):
        if not path.is_file():
            raise AlignedCollisionQuantizerError(f"{name}不存在：{path}")
    if not config.poi_dir.is_dir():
        raise AlignedCollisionQuantizerError(f"POI 目录不存在：{config.poi_dir}")
    if config.codebook_size * config.codebook_size < 306:
        raise AlignedCollisionQuantizerError(
            "两层 suffix 容量小于冻结 TIGER 最大碰撞桶 306"
        )
    if config.candidate_first > config.codebook_size or config.candidate_second > config.codebook_size:
        raise AlignedCollisionQuantizerError("候选宽度不能超过 codebook_size")


def validate_aligned_collision_output(output_dir: Path) -> dict[str, Any]:
    """Validate hashes and invariants of one completed suffix build."""

    manifest = _load_json(output_dir / "manifest.json", "manifest")
    metrics = _load_json(output_dir / "metrics.json", "metrics")
    if manifest.get("status") != "completed" or metrics.get("status") != "completed":
        raise AlignedCollisionQuantizerError("输出状态不是 completed")
    for name, contract in _mapping(manifest.get("artifacts"), "artifacts").items():
        item = _mapping(contract, f"artifacts.{name}")
        path = output_dir / str(item.get("path"))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise AlignedCollisionQuantizerError(f"产物缺失或 SHA256 不一致：{name}")
    if metrics.get("identifier", {}).get("distinct_ratio") != 1.0:
        raise AlignedCollisionQuantizerError("最终五层 identifier 不是全局唯一")
    return {"manifest": manifest, "metrics": metrics}


def build_aligned_collision_identifiers(
    *,
    project_root: Path,
    config: AlignedCollisionConfig,
    progress: Callable[[str], None] | None = None,
) -> AlignedCollisionResult:
    """Run the complete query-free TIGER collision suffix experiment."""

    _validate_inputs(config)
    if config.output_dir.exists():
        validated = validate_aligned_collision_output(config.output_dir)
        return AlignedCollisionResult(
            metrics=validated["metrics"],
            manifest=validated["manifest"],
            output_dir=config.output_dir,
        )
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started = time.perf_counter()
    base_sid = np.load(config.base_sid_codes, mmap_mode="r", allow_pickle=False)
    if base_sid.ndim != 2 or base_sid.shape[1] != 3 or base_sid.dtype.kind not in {"i", "u"}:
        raise AlignedCollisionQuantizerError("冻结 TIGER SID 必须是整数 [N,3]")
    row_count = len(base_sid)
    _, inverse, bucket_sizes = np.unique(
        base_sid, axis=0, return_inverse=True, return_counts=True
    )
    bucket_sizes_by_row = bucket_sizes[inverse].astype(np.int32, copy=False)
    collision_rows = np.flatnonzero(bucket_sizes_by_row > 1).astype(np.int64)
    collision_bucket_ids = inverse[collision_rows].astype(np.int64, copy=False)
    collision_bucket_sizes = bucket_sizes_by_row[collision_rows]
    if int(bucket_sizes.max()) > config.codebook_size * config.codebook_size:
        raise AlignedCollisionQuantizerError("真实最大碰撞桶超过两层 suffix 容量")
    poi_ids = _read_poi_ids(config.poi_ids, row_count)

    temporary_parent = config.output_dir.parent
    # Large, reproducible intermediates stay on local scratch. Only the final
    # compact artifacts are written to an atomic temporary directory on OFS.
    with tempfile.TemporaryDirectory(
        prefix=f"{config.output_dir.name}-scratch-"
    ) as scratch_name, tempfile.TemporaryDirectory(
        prefix=f".{config.output_dir.name}.tmp-", dir=temporary_parent
    ) as output_temporary_name:
        scratch_dir = Path(scratch_name)
        work_dir = Path(output_temporary_name)
        residuals, runtime = _encode_tiger_latent_residuals(
            config,
            base_sid,
            collision_rows,
            scratch_dir / "latent_residuals.npy",
            progress=progress,
        )
        coordinates, poi_sources = _load_collision_coordinates(
            config.poi_dir,
            poi_ids,
            collision_rows,
            progress=progress,
        )
        geo, origin = multiscale_geo_features(coordinates, config.geo_scales_km)
        features = _write_collision_features(
            residuals,
            geo,
            collision_bucket_ids,
            scratch_dir / "collision_features.npy",
            geo_weight=config.geo_weight,
            progress=progress,
        )
        semantic_dim = int(residuals.shape[1])
        del residuals
        (scratch_dir / "latent_residuals.npy").unlink()
        first_centers, second_centers, fit_metadata = _fit_shared_residual_anchors(
            features, collision_bucket_sizes, config
        )
        np.save(work_dir / "anchor_level_1.npy", first_centers, allow_pickle=False)
        np.save(work_dir / "anchor_level_2.npy", second_centers, allow_pickle=False)
        candidates, candidate_distances = _pair_candidates(
            features,
            first_centers,
            second_centers,
            first_count=config.candidate_first,
            second_count=config.candidate_second,
            batch_size=config.search_batch_size,
            device_name=str(runtime["device"]),
            candidate_path=scratch_dir / "pair_candidates.npy",
            distance_path=scratch_dir / "pair_candidate_distances.npy",
            progress=progress,
        )
        prototypes = (
            first_centers[:, None, :] + second_centers[None, :, :]
        ).reshape(config.codebook_size**2, -1)
        pairs, assignment_metrics = assign_unique_pairs(
            features,
            collision_bucket_ids,
            candidates,
            candidate_distances,
            prototypes,
        )
        identifiers, suffix_codes = compose_five_layer_identifiers(
            base_sid,
            collision_rows,
            pairs,
            config.codebook_size,
        )
        order = np.argsort(collision_bucket_ids, kind="stable")
        ordered_buckets = collision_bucket_ids[order]
        starts = np.flatnonzero(
            np.r_[True, ordered_buckets[1:] != ordered_buckets[:-1]]
        )
        ends = np.r_[starts[1:], len(order)]
        for group_start, group_end in zip(starts, ends, strict=True):
            if len(np.unique(pairs[order[group_start:group_end]])) != group_end - group_start:
                raise AlignedCollisionQuantizerError("最终桶内 pair 唯一性复核失败")
        distinct_count = int(np.count_nonzero(bucket_sizes == 1) + len(collision_rows))
        if distinct_count != row_count:
            raise AlignedCollisionQuantizerError("最终 identifier distinct 守恒失败")

        np.save(work_dir / "suffix_codes.npy", suffix_codes, allow_pickle=False)
        np.save(work_dir / "identifier_codes.npy", identifiers, allow_pickle=False)
        _write_mapping(
            work_dir / "poi_identifier_mapping.parquet",
            poi_ids,
            identifiers,
            bucket_sizes_by_row,
        )
        first_counts = np.bincount(
            suffix_codes[collision_rows, 0], minlength=config.codebook_size
        )
        second_counts = np.bincount(
            suffix_codes[collision_rows, 1], minlength=config.codebook_size
        )
        artifacts: dict[str, dict[str, Any]] = {}
        for name in (
            "anchor_level_1.npy",
            "anchor_level_2.npy",
            "suffix_codes.npy",
            "identifier_codes.npy",
            "poi_identifier_mapping.parquet",
        ):
            path = work_dir / name
            artifacts[name] = {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        metrics = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "status": "completed",
            "experiment_id": config.experiment_id,
            "base_tiger": {
                "poi_count": row_count,
                "base_sid_distinct_count": int(len(bucket_sizes)),
                "collision_bucket_count": int(np.count_nonzero(bucket_sizes > 1)),
                "collision_poi_count": int(len(collision_rows)),
                "collision_excess_count": int(row_count - len(bucket_sizes)),
                "max_bucket_size": int(bucket_sizes.max()),
            },
            "suffix": {
                "codebook_size": config.codebook_size,
                "pair_capacity": config.codebook_size**2,
                "level_1_used": int(np.count_nonzero(first_counts)),
                "level_2_used": int(np.count_nonzero(second_counts)),
                "level_1_counts": first_counts.tolist(),
                "level_2_counts": second_counts.tolist(),
                **assignment_metrics,
            },
            "identifier": {
                "layers": 5,
                "distinct_count": distinct_count,
                "distinct_ratio": distinct_count / row_count,
                "fixed_suffix_tokens": 2,
                "within_bucket_duplicate_pair_count": 0,
            },
            "feature": {
                "semantic": "frozen TIGER RQ-VAE latent residual centered per base SID bucket",
                "semantic_dim": semantic_dim,
                "geo": "absolute local-km multi-scale Fourier encoding",
                "geo_dim": int(geo.shape[1]),
                "geo_weight": config.geo_weight,
                "geo_scales_km": list(config.geo_scales_km),
                "origin_lng_lat": list(origin),
            },
            "fit": fit_metadata,
            "runtime": {
                **runtime,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "seconds": time.perf_counter() - started,
            },
        }
        _json_dump(work_dir / "metrics.json", metrics)
        artifacts["metrics.json"] = {
            "path": "metrics.json",
            "bytes": (work_dir / "metrics.json").stat().st_size,
            "sha256": sha256_file(work_dir / "metrics.json"),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "experiment_id": config.experiment_id,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "query_usage": {
                "query_used_to_build_identifier": False,
                "orders_used_to_build_identifier": False,
            },
            "protocol": {
                "base_sid_frozen": True,
                "base_layers": 3,
                "suffix_layers": 2,
                "shared_global_anchors": True,
                "bucket_balanced_fit_weight": "1/base_sid_bucket_size",
                "within_bucket_unique_assignment": "linear_sum_assignment",
                "singleton_suffix": [0, 0],
            },
            "inputs": {
                "base_sid_codes": {
                    "path": str(config.base_sid_codes),
                    "sha256": sha256_file(config.base_sid_codes),
                },
                "base_sid_manifest": {
                    "path": str(config.base_sid_manifest),
                    "sha256": sha256_file(config.base_sid_manifest),
                },
                "embeddings": {
                    "path": str(config.embeddings),
                    "sha256": sha256_file(config.embeddings),
                },
                "poi_ids": {
                    "path": str(config.poi_ids),
                    "sha256": sha256_file(config.poi_ids),
                },
                "rqvae_checkpoint": {
                    "path": str(config.rqvae_checkpoint),
                    "sha256": sha256_file(config.rqvae_checkpoint),
                },
                "rqvae_resolved_config": {
                    "path": str(config.rqvae_resolved_config),
                    "sha256": sha256_file(config.rqvae_resolved_config),
                },
                "poi_sources": poi_sources,
            },
            "git": _git_state(project_root),
            "artifacts": artifacts,
        }
        _json_dump(work_dir / "manifest.json", manifest)
        os.replace(work_dir, config.output_dir)

    validated = validate_aligned_collision_output(config.output_dir)
    return AlignedCollisionResult(
        metrics=validated["metrics"],
        manifest=validated["manifest"],
        output_dir=config.output_dir,
    )

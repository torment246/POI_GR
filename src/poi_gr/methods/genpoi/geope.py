"""Build GenPOI geographic position embedding vectors."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from poi_gr.pid.dedup import sha256_file


SCHEMA_VERSION = "genpoi-geope-embedding-v3"
LOCAL_PROJECTION = "local_equirectangular_km"
RAW_DEGREE_PROJECTION = "raw_longitude_latitude_degrees"
NO_EMBEDDING_PREPROCESSING = "none"
GLOBAL_MEAN_CENTER_L2 = "global_mean_center_l2"
VALID_COORDINATE_PROJECTIONS = {
    LOCAL_PROJECTION,
    RAW_DEGREE_PROJECTION,
}
VALID_EMBEDDING_PREPROCESSING = {
    NO_EMBEDDING_PREPROCESSING,
    GLOBAL_MEAN_CENTER_L2,
}
OUTPUT_FILENAMES = (
    "embeddings.npy",
    "poi_ids.jsonl",
    "reference_points.npy",
    "manifest.json",
)


class GenpoiGeoPEError(ValueError):
    """Raised when GenPOI GeoPE inputs or outputs violate the contract."""


@dataclass(frozen=True)
class GenpoiGeoPEResult:
    """Completed GenPOI GeoPE artifacts."""

    manifest: dict[str, Any]
    output_hashes: dict[str, str]


def fit_reference_points(
    coordinates: np.ndarray,
    *,
    reference_count: int = 32,
    seed: int = 42,
    n_init: int = 3,
    max_iter: int = 100,
    coordinate_projection: str = LOCAL_PROJECTION,
) -> np.ndarray:
    """Fit deterministic mini-batch K-Means anchors in a local metric space."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise GenpoiGeoPEError("coordinates shape 必须是 [N,2]，顺序为 lng/lat")
    if not np.isfinite(coordinates).all():
        raise GenpoiGeoPEError("coordinates 包含 NaN 或 Inf")
    if reference_count <= 0 or reference_count > len(coordinates):
        raise GenpoiGeoPEError("reference_count 必须位于 [1,N]")
    if n_init <= 0 or max_iter <= 0:
        raise GenpoiGeoPEError("K-Means n_init 和 max_iter 必须大于 0")
    if coordinate_projection not in VALID_COORDINATE_PROJECTIONS:
        raise GenpoiGeoPEError("coordinate_projection 配置无效")

    projected, projection = _project_coordinates(
        coordinates,
        coordinate_projection,
    )

    estimator = MiniBatchKMeans(
        n_clusters=reference_count,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        batch_size=min(8192, len(projected)),
        max_no_improvement=20,
        reassignment_ratio=0.01,
        random_state=seed,
    )
    estimator.fit(projected)
    centers = _inverse_project_coordinates(
        np.asarray(estimator.cluster_centers_, dtype=np.float64),
        projection,
    )
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return np.ascontiguousarray(centers[order])


def _project_coordinates(
    coordinates: np.ndarray,
    coordinate_projection: str,
) -> tuple[np.ndarray, dict[str, float | str | list[float]]]:
    if coordinate_projection == RAW_DEGREE_PROJECTION:
        return np.asarray(coordinates, dtype=np.float64), {
            "method": RAW_DEGREE_PROJECTION,
        }
    if coordinate_projection != LOCAL_PROJECTION:
        raise GenpoiGeoPEError("coordinate_projection 配置无效")

    coordinates = np.asarray(coordinates, dtype=np.float64)
    origin = coordinates.mean(axis=0)
    longitude_km_per_degree = 111.320 * math.cos(math.radians(origin[1]))
    latitude_km_per_degree = 110.574
    projected = np.column_stack(
        (
            (coordinates[:, 0] - origin[0]) * longitude_km_per_degree,
            (coordinates[:, 1] - origin[1]) * latitude_km_per_degree,
        )
    )
    return projected, {
        "method": LOCAL_PROJECTION,
        "origin_lng_lat": [float(origin[0]), float(origin[1])],
        "longitude_km_per_degree": longitude_km_per_degree,
        "latitude_km_per_degree": latitude_km_per_degree,
    }


def _inverse_project_coordinates(
    projected: np.ndarray,
    projection: Mapping[str, Any],
) -> np.ndarray:
    if projection["method"] == RAW_DEGREE_PROJECTION:
        return np.asarray(projected, dtype=np.float64)
    origin = projection["origin_lng_lat"]
    return np.column_stack(
        (
            projected[:, 0] / projection["longitude_km_per_degree"] + origin[0],
            projected[:, 1] / projection["latitude_km_per_degree"] + origin[1],
        )
    )


def compute_bearing_angles(
    coordinates: np.ndarray,
    reference_points: np.ndarray,
) -> np.ndarray:
    """Compute GenPOI bearing angles from every anchor to every POI."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    reference_points = np.asarray(reference_points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise GenpoiGeoPEError("coordinates shape 必须是 [N,2]")
    if reference_points.ndim != 2 or reference_points.shape[1] != 2:
        raise GenpoiGeoPEError("reference_points shape 必须是 [Omega,2]")
    if not np.isfinite(coordinates).all() or not np.isfinite(
        reference_points
    ).all():
        raise GenpoiGeoPEError("坐标或 reference points 包含 NaN/Inf")

    delta_longitude = (
        coordinates[:, None, 0] - reference_points[None, :, 0]
    )
    delta_latitude = (
        coordinates[:, None, 1] - reference_points[None, :, 1]
    )
    longitude_scale = np.cos(
        np.deg2rad(reference_points[None, :, 1])
    )
    angles = np.arctan2(
        delta_latitude,
        delta_longitude * longitude_scale,
    )
    return np.mod(angles, 2.0 * np.pi)


def apply_geope_rotation(
    embeddings: np.ndarray,
    coordinates: np.ndarray,
    reference_points: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Rotate equal embedding segments using GenPOI bearing angles."""

    embeddings = np.asarray(embeddings)
    coordinates = np.asarray(coordinates)
    reference_points = np.asarray(reference_points)
    if embeddings.ndim != 2:
        raise GenpoiGeoPEError("embeddings 必须是二维数组")
    if len(embeddings) != len(coordinates):
        raise GenpoiGeoPEError("Embedding 与 coordinates 行数不一致")
    reference_count = len(reference_points)
    if reference_count <= 0:
        raise GenpoiGeoPEError("reference_points 不能为空")
    embedding_dim = embeddings.shape[1]
    if embedding_dim % reference_count != 0:
        raise GenpoiGeoPEError(
            "Embedding 维度必须能被 reference_count 整除"
        )
    segment_dim = embedding_dim // reference_count
    if segment_dim % 2 != 0:
        raise GenpoiGeoPEError("每个 GeoPE 分段维度必须是偶数")

    values = np.asarray(embeddings, dtype=np.float32)
    angles = compute_bearing_angles(coordinates, reference_points).astype(
        np.float32,
        copy=False,
    )
    if device != "cpu":
        try:
            import torch
        except ImportError as error:
            raise GenpoiGeoPEError("GPU GeoPE 旋转要求安装 PyTorch") from error
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise GenpoiGeoPEError("GPU GeoPE 旋转未检测到可用 CUDA")
        values_tensor = torch.from_numpy(
            np.ascontiguousarray(values)
        ).to(torch_device)
        angles_tensor = torch.from_numpy(
            np.ascontiguousarray(angles)
        ).to(torch_device)
        paired_tensor = values_tensor.reshape(
            len(values),
            reference_count,
            segment_dim // 2,
            2,
        )
        cosine_tensor = torch.cos(angles_tensor)[:, :, None]
        sine_tensor = torch.sin(angles_tensor)[:, :, None]
        even_tensor = paired_tensor[..., 0]
        odd_tensor = paired_tensor[..., 1]
        rotated_tensor = torch.empty_like(paired_tensor)
        rotated_tensor[..., 0] = (
            cosine_tensor * even_tensor - sine_tensor * odd_tensor
        )
        rotated_tensor[..., 1] = (
            sine_tensor * even_tensor + cosine_tensor * odd_tensor
        )
        return rotated_tensor.reshape(values.shape).cpu().numpy()

    paired = values.reshape(
        len(values),
        reference_count,
        segment_dim // 2,
        2,
    )
    cosine = np.cos(angles)[:, :, None]
    sine = np.sin(angles)[:, :, None]
    even = paired[..., 0]
    odd = paired[..., 1]
    rotated = np.empty_like(paired)
    rotated[..., 0] = cosine * even - sine * odd
    rotated[..., 1] = sine * even + cosine * odd
    return rotated.reshape(values.shape)


def center_and_l2_normalize_embeddings(
    embeddings: np.ndarray,
    source_mean: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Subtract a fixed global mean and L2-normalize each embedding."""

    values = np.asarray(embeddings, dtype=np.float32)
    mean = np.asarray(source_mean, dtype=np.float32)
    if values.ndim != 2:
        raise GenpoiGeoPEError("embeddings 必须是二维数组")
    if mean.shape != (values.shape[1],):
        raise GenpoiGeoPEError("source_mean shape 必须等于 Embedding 维度")
    if not np.isfinite(values).all() or not np.isfinite(mean).all():
        raise GenpoiGeoPEError("Embedding 或 source_mean 包含 NaN/Inf")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise GenpoiGeoPEError("L2 normalization epsilon 必须大于 0")

    centered = values - mean[None, :]
    norms = np.linalg.norm(centered, axis=1)
    invalid_rows = np.flatnonzero(norms <= epsilon)
    if len(invalid_rows):
        raise GenpoiGeoPEError(
            "全局均值中心化后存在近零范数向量："
            f"row={int(invalid_rows[0])}，count={len(invalid_rows)}"
        )
    centered /= norms[:, None]
    return centered


def _compute_source_mean(
    source_embeddings: np.ndarray,
    row_count: int,
    chunk_rows: int,
    progress: Callable[[str], None] | None,
) -> np.ndarray:
    total = np.zeros(source_embeddings.shape[1], dtype=np.float64)
    for start in range(0, row_count, chunk_rows):
        stop = min(start + chunk_rows, row_count)
        chunk = np.asarray(source_embeddings[start:stop], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise GenpoiGeoPEError(
                f"Embedding 在行区间 [{start},{stop}) 包含 NaN/Inf"
            )
        total += chunk.sum(axis=0, dtype=np.float64)
        if progress is not None and (
            stop == row_count or stop % (chunk_rows * 32) == 0
        ):
            progress(f"GenPOI GeoPE 全局均值：{stop:,}/{row_count:,}")
    return np.asarray(total / row_count, dtype=np.float32)


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise GenpoiGeoPEError(f"{name} 不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GenpoiGeoPEError(f"{name} JSON 解析失败：{path}") from error
    if not isinstance(payload, dict):
        raise GenpoiGeoPEError(f"{name} 必须是 JSON object")
    return payload


def _discover_poi_files(poi_data_dir: Path) -> tuple[Path, ...]:
    if not poi_data_dir.is_dir():
        raise GenpoiGeoPEError(f"POI 数据目录不存在：{poi_data_dir}")
    if not (poi_data_dir / "_SUCCESS").is_file():
        raise GenpoiGeoPEError("POI 数据目录缺少 _SUCCESS")
    files = tuple(
        sorted(
            path
            for path in poi_data_dir.iterdir()
            if path.is_file() and path.name.startswith("part-")
        )
    )
    if not files:
        raise GenpoiGeoPEError("POI 数据目录没有 part-* 分片")
    return files


def _coordinate(
    record: Mapping[str, Any],
    field: str,
    minimum: float,
    maximum: float,
    location: str,
) -> float:
    value = record.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise GenpoiGeoPEError(
            f"{location} 的 {field} 必须 finite 且位于 [{minimum},{maximum}]"
        )
    return float(value)


def _load_aligned_coordinates(
    poi_files: Sequence[Path],
    source_ids_path: Path,
    output_ids_path: Path,
    row_count: int,
    *,
    full_build: bool,
    progress: Callable[[str], None] | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    coordinates = np.empty((row_count, 2), dtype=np.float64)
    row = 0
    sources: list[dict[str, Any]] = []
    stopped_early = False

    with source_ids_path.open("rb") as ids_handle, output_ids_path.open(
        "wb"
    ) as output_ids:
        for path in poi_files:
            digest = hashlib.sha256()
            scanned_rows = 0
            scanned_bytes = 0
            complete_file = True
            with path.open("rb") as poi_handle:
                for line_number, raw_line in enumerate(poi_handle, start=1):
                    if row >= row_count:
                        complete_file = False
                        stopped_early = True
                        break
                    location = f"{path.name}:{line_number}"
                    if not raw_line.strip():
                        raise GenpoiGeoPEError(f"{location} 是空行")
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as error:
                        raise GenpoiGeoPEError(
                            f"{location} JSON 解析失败"
                        ) from error
                    if not isinstance(record, dict):
                        raise GenpoiGeoPEError(f"{location} 不是 JSON object")
                    poi_id = record.get("poi_id")
                    if not isinstance(poi_id, str) or not poi_id:
                        raise GenpoiGeoPEError(f"{location} poi_id 无效")

                    id_line = ids_handle.readline()
                    if not id_line:
                        raise GenpoiGeoPEError("POI ID 文件行数少于 Embedding")
                    try:
                        expected_poi_id = json.loads(id_line)
                    except json.JSONDecodeError as error:
                        raise GenpoiGeoPEError(
                            f"POI ID 第 {row + 1} 行 JSON 解析失败"
                        ) from error
                    if poi_id != expected_poi_id:
                        raise GenpoiGeoPEError(
                            f"POI 行映射不一致：row={row}，"
                            f"POI 数据={poi_id}，Embedding ID={expected_poi_id}"
                        )

                    coordinates[row, 0] = _coordinate(
                        record,
                        "lng",
                        -180.0,
                        180.0,
                        location,
                    )
                    coordinates[row, 1] = _coordinate(
                        record,
                        "lat",
                        -90.0,
                        90.0,
                        location,
                    )
                    output_ids.write(id_line)
                    digest.update(raw_line)
                    scanned_rows += 1
                    scanned_bytes += len(raw_line)
                    row += 1

            if scanned_rows:
                sources.append(
                    {
                        "name": path.name,
                        "rows_scanned": scanned_rows,
                        "bytes_scanned": scanned_bytes,
                        "sha256_scanned": digest.hexdigest(),
                        "complete_file": complete_file,
                    }
                )
            if progress is not None:
                progress(
                    f"GenPOI GeoPE 坐标对齐：{row:,}/{row_count:,}"
                    f"（{path.name}）"
                )
            if stopped_early:
                break

        if row != row_count:
            raise GenpoiGeoPEError(
                f"POI 数据仅对齐 {row} 行，期望 {row_count} 行"
            )
        if full_build and ids_handle.readline():
            raise GenpoiGeoPEError("POI ID 文件行数多于 Embedding")

    return coordinates, sources


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_genpoi_geope_embeddings(
    source_embedding_dir: Path,
    poi_data_dir: Path,
    output_dir: Path,
    *,
    reference_count: int = 32,
    seed: int = 42,
    kmeans_n_init: int = 3,
    kmeans_max_iter: int = 100,
    coordinate_projection: str = LOCAL_PROJECTION,
    reference_fit_sample_size: int | None = 200_000,
    chunk_rows: int = 8192,
    rotation_device: str = "cpu",
    metric_sample_rows: int = 32_768,
    output_dtype: str = "float16",
    embedding_preprocessing: str = NO_EMBEDDING_PREPROCESSING,
    reference_points_path: Path | None = None,
    max_rows: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> GenpoiGeoPEResult:
    """Build row-aligned GeoPE vectors from an existing embedding artifact."""

    if reference_count <= 0:
        raise GenpoiGeoPEError("reference_count 必须大于 0")
    if chunk_rows <= 0:
        raise GenpoiGeoPEError("chunk_rows 必须大于 0")
    if not isinstance(rotation_device, str) or not rotation_device:
        raise GenpoiGeoPEError("rotation_device 必须是非空字符串")
    if metric_sample_rows <= 0:
        raise GenpoiGeoPEError("metric_sample_rows 必须大于 0")
    if output_dtype not in {"float16", "float32"}:
        raise GenpoiGeoPEError("output_dtype 只能是 float16 或 float32")
    if embedding_preprocessing not in VALID_EMBEDDING_PREPROCESSING:
        raise GenpoiGeoPEError("embedding_preprocessing 配置无效")
    if max_rows is not None and max_rows <= reference_count:
        raise GenpoiGeoPEError("max_rows 必须大于 reference_count")
    if coordinate_projection not in VALID_COORDINATE_PROJECTIONS:
        raise GenpoiGeoPEError("coordinate_projection 配置无效")
    if reference_fit_sample_size is not None and reference_fit_sample_size <= 0:
        raise GenpoiGeoPEError("reference_fit_sample_size 必须大于 0")

    source_embedding_dir = source_embedding_dir.resolve()
    poi_data_dir = poi_data_dir.resolve()
    output_dir = output_dir.resolve()
    if reference_points_path is not None:
        reference_points_path = reference_points_path.resolve()
    if output_dir.exists():
        raise GenpoiGeoPEError(f"输出目录已存在：{output_dir}")

    source_manifest_path = source_embedding_dir / "manifest.json"
    source_embeddings_path = source_embedding_dir / "embeddings.npy"
    source_ids_path = source_embedding_dir / "poi_ids.jsonl"
    source_manifest = _load_json_object(
        source_manifest_path,
        "BGE Embedding manifest",
    )
    if source_manifest.get("status") != "completed":
        raise GenpoiGeoPEError("BGE Embedding manifest 状态不是 completed")
    for path, name in (
        (source_embeddings_path, "BGE embeddings.npy"),
        (source_ids_path, "BGE poi_ids.jsonl"),
    ):
        if not path.is_file():
            raise GenpoiGeoPEError(f"{name} 不存在：{path}")

    source_embeddings = np.load(
        source_embeddings_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    declared_shape = tuple(
        source_manifest.get("output", {}).get("shape", ())
    )
    declared_dtype = source_manifest.get("output", {}).get("dtype")
    if (
        source_embeddings.ndim != 2
        or tuple(source_embeddings.shape) != declared_shape
        or str(source_embeddings.dtype) != declared_dtype
    ):
        raise GenpoiGeoPEError("BGE Embedding 与 manifest shape/dtype 不一致")
    full_rows, embedding_dim = source_embeddings.shape
    effective_rows = (
        full_rows if max_rows is None else min(max_rows, full_rows)
    )
    if embedding_dim % reference_count != 0:
        raise GenpoiGeoPEError(
            "BGE Embedding 维度不能被 reference_count 整除"
        )
    if (embedding_dim // reference_count) % 2 != 0:
        raise GenpoiGeoPEError("GeoPE 每个分段维度不是偶数")

    poi_files = _discover_poi_files(poi_data_dir)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    source_embeddings_sha256 = sha256_file(source_embeddings_path)
    source_ids_sha256 = sha256_file(source_ids_path)

    provided_reference_points: np.ndarray | None = None
    provided_reference_points_sha256: str | None = None
    if reference_points_path is not None:
        if not reference_points_path.is_file():
            raise GenpoiGeoPEError(
                f"reference_points 文件不存在：{reference_points_path}"
            )
        provided_reference_points = np.load(
            reference_points_path,
            allow_pickle=False,
        )
        if (
            provided_reference_points.ndim != 2
            or provided_reference_points.shape != (reference_count, 2)
            or not np.isfinite(provided_reference_points).all()
        ):
            raise GenpoiGeoPEError(
                "reference_points 必须是 finite 的 [reference_count,2] 数组"
            )
        provided_reference_points_sha256 = sha256_file(
            reference_points_path
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.building-",
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        output_ids_path = temporary_dir / "poi_ids.jsonl"
        coordinates, coordinate_sources = _load_aligned_coordinates(
            poi_files,
            source_ids_path,
            output_ids_path,
            effective_rows,
            full_build=max_rows is None,
            progress=progress,
        )

        if provided_reference_points is None:
            if progress is not None:
                progress(
                    f"对 {effective_rows:,} 个 POI 坐标拟合 "
                    f"{reference_count} 个 K-Means reference points"
                )
            fit_rows: int | None = (
                effective_rows
                if reference_fit_sample_size is None
                else min(reference_fit_sample_size, effective_rows)
            )
            if fit_rows == effective_rows:
                fit_indices = np.arange(effective_rows, dtype=np.int64)
            else:
                fit_indices = np.sort(
                    np.random.default_rng(seed).choice(
                        effective_rows,
                        size=fit_rows,
                        replace=False,
                    )
                ).astype(np.int64, copy=False)
            fit_coordinates = coordinates[fit_indices]
            _, projection_metadata = _project_coordinates(
                fit_coordinates,
                coordinate_projection,
            )
            reference_points = fit_reference_points(
                fit_coordinates,
                reference_count=reference_count,
                seed=seed,
                n_init=kmeans_n_init,
                max_iter=kmeans_max_iter,
                coordinate_projection=coordinate_projection,
            )
            with (temporary_dir / "reference_points.npy").open("wb") as handle:
                np.save(handle, reference_points, allow_pickle=False)
            reference_point_method = "sklearn_minibatch_kmeans"
            reference_point_order = "longitude_then_latitude_ascending"
            reference_fit_sampling = (
                "all_rows"
                if fit_rows == effective_rows
                else "numpy_default_rng_choice_without_replacement"
            )
        else:
            if progress is not None:
                progress(
                    f"复用 {reference_count} 个 GeoPE reference points："
                    f"{reference_points_path}"
                )
            reference_points = np.asarray(
                provided_reference_points,
                dtype=np.float64,
            )
            assert reference_points_path is not None
            shutil.copyfile(
                reference_points_path,
                temporary_dir / "reference_points.npy",
            )
            fit_rows = None
            projection_metadata = None
            reference_point_method = "provided"
            reference_point_order = "as_provided"
            reference_fit_sampling = "not_applicable"

        source_mean: np.ndarray | None = None
        if embedding_preprocessing == GLOBAL_MEAN_CENTER_L2:
            if progress is not None:
                progress(
                    f"计算 {effective_rows:,} 行 Embedding 的全局均值"
                )
            source_mean = _compute_source_mean(
                source_embeddings,
                effective_rows,
                chunk_rows,
                progress,
            )
            with (temporary_dir / "source_mean.npy").open("wb") as handle:
                np.save(handle, source_mean, allow_pickle=False)

        output_embeddings_path = temporary_dir / "embeddings.npy"
        # Sequential writes avoid dirty mmap page stalls on the shared filesystem.
        with output_embeddings_path.open("wb", buffering=8 * 1024 * 1024) as output_handle:
            np.lib.format.write_array_header_2_0(output_handle, {
                "descr": np.dtype(output_dtype).str,
                "fortran_order": False,
                "shape": (effective_rows, embedding_dim),
            })
            norm_delta_sum = 0.0
            norm_delta_max = 0.0
            vector_delta_sum = 0.0
            preprocessing_vector_delta_sum = 0.0
            rotation_norm_delta_sum = 0.0
            rotation_norm_delta_max = 0.0
            metric_rows = 0
            for start in range(0, effective_rows, chunk_rows):
                stop = min(start + chunk_rows, effective_rows)
                source_chunk = np.asarray(
                    source_embeddings[start:stop],
                    dtype=np.float32,
                )
                if source_mean is None:
                    preprocessed_chunk = source_chunk
                else:
                    preprocessed_chunk = center_and_l2_normalize_embeddings(
                        source_chunk,
                        source_mean,
                    )
                rotated = apply_geope_rotation(
                    preprocessed_chunk,
                    coordinates[start:stop],
                    reference_points,
                    device=rotation_device,
                )
                if metric_rows < metric_sample_rows:
                    sample_stop = min(
                        len(source_chunk),
                        metric_sample_rows - metric_rows,
                    )
                    source_sample = source_chunk[:sample_stop]
                    preprocessed_sample = preprocessed_chunk[:sample_stop]
                    rotated_sample = rotated[:sample_stop]
                    source_norms = np.linalg.norm(source_sample, axis=1)
                    rotated_norms = np.linalg.norm(rotated_sample, axis=1)
                    norm_delta = np.abs(source_norms - rotated_norms)
                    rotation_norm_delta = np.abs(
                        np.linalg.norm(preprocessed_sample, axis=1)
                        - rotated_norms
                    )
                    norm_delta_sum += float(norm_delta.sum())
                    norm_delta_max = max(
                        norm_delta_max,
                        float(norm_delta.max(initial=0.0)),
                    )
                    vector_delta_sum += float(
                        np.linalg.norm(
                            source_sample - rotated_sample,
                            axis=1,
                        ).sum()
                    )
                    preprocessing_vector_delta_sum += float(
                        np.linalg.norm(
                            source_sample - preprocessed_sample,
                            axis=1,
                        ).sum()
                    )
                    rotation_norm_delta_sum += float(rotation_norm_delta.sum())
                    rotation_norm_delta_max = max(
                        rotation_norm_delta_max,
                        float(rotation_norm_delta.max(initial=0.0)),
                    )
                    metric_rows += sample_stop
                output_handle.write(np.asarray(rotated, dtype=output_dtype).tobytes(order="C"))
                if progress is not None and (
                    stop == effective_rows or stop % (chunk_rows * 32) == 0
                ):
                    progress(
                        f"GenPOI GeoPE 旋转：{stop:,}/{effective_rows:,}"
                    )

        output_filenames = OUTPUT_FILENAMES + (
            ("source_mean.npy",)
            if source_mean is not None
            else ()
        )
        output_hashes_without_manifest = {
            name: sha256_file(temporary_dir / name)
            for name in output_filenames
            if name != "manifest.json"
        }
        fingerprint_payload = {
            "schema_version": SCHEMA_VERSION,
            "source_manifest_sha256": source_manifest_sha256,
            "source_embeddings_sha256": source_embeddings_sha256,
            "source_ids_sha256": source_ids_sha256,
            "coordinate_sources": coordinate_sources,
            "effective_rows": effective_rows,
            "reference_count": reference_count,
            "seed": seed,
            "kmeans_n_init": kmeans_n_init,
            "kmeans_max_iter": kmeans_max_iter,
            "coordinate_projection": coordinate_projection,
            "reference_fit_sample_size": reference_fit_sample_size,
            "provided_reference_points_sha256": (
                provided_reference_points_sha256
            ),
            "rotation_device": rotation_device,
            "metric_sample_rows": metric_sample_rows,
            "output_dtype": output_dtype,
            "embedding_preprocessing": embedding_preprocessing,
        }
        build_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "method": "GenPOI_GeoPE",
            "build_fingerprint": build_fingerprint,
            "input": {
                "total_rows": effective_rows,
                "fingerprint": build_fingerprint,
                "scope": (
                    "full" if max_rows is None else "prefix_smoke"
                ),
                "source_embedding": {
                    "directory": source_embedding_dir.name,
                    "shape": list(source_embeddings.shape),
                    "dtype": str(source_embeddings.dtype),
                    "manifest_sha256": source_manifest_sha256,
                    "embeddings_sha256": source_embeddings_sha256,
                    "poi_ids_sha256": source_ids_sha256,
                },
                "poi_data": {
                    "directory": poi_data_dir.name,
                    "coordinate_fields": {
                        "longitude": "lng",
                        "latitude": "lat",
                    },
                    "sources": coordinate_sources,
                    "row_order_matches_embedding_ids": True,
                },
            },
            "geope": {
                "reference_count": reference_count,
                "reference_point_method": reference_point_method,
                "reference_point_order": reference_point_order,
                "reference_point_source_sha256": (
                    provided_reference_points_sha256
                ),
                "kmeans_input": coordinate_projection,
                "kmeans_projection": projection_metadata,
                "reference_fit_rows": fit_rows,
                "reference_fit_sampling": reference_fit_sampling,
                "kmeans_seed": seed,
                "kmeans_n_init": kmeans_n_init,
                "kmeans_max_iter": kmeans_max_iter,
                "angle_formula": (
                    "atan2(delta_lat,delta_lng*cos(rad(reference_lat)))"
                ),
                "angle_range": "[0,2pi)",
                "embedding_segments": reference_count,
                "segment_dim": embedding_dim // reference_count,
                "rotation_pair_order": "consecutive_even_odd_dimensions",
                "embedding_preprocessing": {
                    "mode": embedding_preprocessing,
                    "mean_scope": (
                        "effective_rows" if source_mean is not None else None
                    ),
                    "mean_rows": (
                        effective_rows if source_mean is not None else None
                    ),
                    "source_mean": (
                        "source_mean.npy" if source_mean is not None else None
                    ),
                    "source_mean_l2_norm": (
                        float(np.linalg.norm(source_mean))
                        if source_mean is not None
                        else None
                    ),
                    "l2_epsilon": (
                        1e-12 if source_mean is not None else None
                    ),
                },
                "normalization_before_rotation": (
                    "l2" if source_mean is not None else "none"
                ),
                "normalization_after_rotation": "none",
                "rotation_device": rotation_device,
            },
            "metrics": {
                "sample_rows": metric_rows,
                "sampling": "prefix",
                "mean_absolute_norm_delta_float32": (
                    norm_delta_sum / metric_rows
                ),
                "max_absolute_norm_delta_float32": norm_delta_max,
                "mean_vector_l2_delta_float32": (
                    vector_delta_sum / metric_rows
                ),
                "mean_preprocessing_vector_l2_delta_float32": (
                    preprocessing_vector_delta_sum / metric_rows
                ),
                "mean_absolute_rotation_norm_delta_float32": (
                    rotation_norm_delta_sum / metric_rows
                ),
                "max_absolute_rotation_norm_delta_float32": (
                    rotation_norm_delta_max
                ),
            },
            "output": {
                "shape": [effective_rows, embedding_dim],
                "dtype": output_dtype,
                "embeddings": "embeddings.npy",
                "poi_ids": "poi_ids.jsonl",
                "reference_points": "reference_points.npy",
                "source_mean": (
                    "source_mean.npy" if source_mean is not None else None
                ),
                "hashes": output_hashes_without_manifest,
            },
        }
        _write_json(temporary_dir / "manifest.json", manifest)
        output_hashes = dict(output_hashes_without_manifest)
        output_hashes["manifest.json"] = sha256_file(
            temporary_dir / "manifest.json"
        )
        os.replace(temporary_dir, output_dir)

    return GenpoiGeoPEResult(
        manifest=manifest,
        output_hashes=output_hashes,
    )

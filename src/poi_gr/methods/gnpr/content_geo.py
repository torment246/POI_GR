"""Content and geographic inputs for the GNPR-SID map-search adaptation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


SCHEMA_VERSION = "gnpr-content-geo-input-v1"


class GnprContentGeoError(ValueError):
    """Raised when a content-geographic input artifact is invalid."""


@dataclass(frozen=True)
class GnprContentGeoDimensions:
    """Dimensions of the three independently normalized feature blocks."""

    text_dim: int
    category_dim: int
    region_dim: int

    @property
    def total_dim(self) -> int:
        return self.text_dim + self.category_dim + self.region_dim


@dataclass(frozen=True)
class GnprContentGeoWeights:
    """Relative L2 weights applied before normalizing the fused vector."""

    text: float = 1.0
    category: float = 1.0
    region: float = 1.0

    def __post_init__(self) -> None:
        values = (self.text, self.category, self.region)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise GnprContentGeoError("三个特征块权重必须是有限正数")

    @property
    def normalization_scale(self) -> float:
        return math.sqrt(
            self.text**2 + self.category**2 + self.region**2
        )


class GnprContentGeoArray:
    """Array-like batch materializer backed by BGE and compact index memmaps."""

    def __init__(
        self,
        text_embeddings: np.ndarray,
        category_indices: np.ndarray,
        region_indices: np.ndarray,
        *,
        dimensions: GnprContentGeoDimensions,
        weights: GnprContentGeoWeights | None = None,
    ) -> None:
        if text_embeddings.ndim != 2:
            raise GnprContentGeoError("BGE 文本向量必须是二维数组")
        row_count = int(text_embeddings.shape[0])
        if text_embeddings.shape[1] != dimensions.text_dim:
            raise GnprContentGeoError("BGE 文本向量维度与输入契约不一致")
        if category_indices.shape != (row_count,) or region_indices.shape != (
            row_count,
        ):
            raise GnprContentGeoError("类别或区域索引与 BGE 行数不一致")
        self.text_embeddings = text_embeddings
        self.category_indices = category_indices
        self.region_indices = region_indices
        self.dimensions = dimensions
        self.weights = weights or GnprContentGeoWeights()
        self.shape = (row_count, dimensions.total_dim)
        self.dtype = np.dtype(np.float32)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows: Any) -> np.ndarray:
        text = np.asarray(self.text_embeddings[rows], dtype=np.float32)
        categories = np.asarray(self.category_indices[rows], dtype=np.int64)
        regions = np.asarray(self.region_indices[rows], dtype=np.int64)
        squeeze = text.ndim == 1
        if squeeze:
            text = text[None, :]
            categories = categories.reshape(1)
            regions = regions.reshape(1)
        norms = np.linalg.norm(text, axis=1, keepdims=True)
        if not np.isfinite(text).all() or np.any(norms <= 0):
            raise GnprContentGeoError("BGE batch 包含 NaN、Inf 或零向量")
        output = np.zeros((len(text), self.dimensions.total_dim), dtype=np.float32)
        output[:, : self.dimensions.text_dim] = (
            text / norms * self.weights.text
        )
        batch_rows = np.arange(len(text))
        output[
            batch_rows,
            self.dimensions.text_dim + categories,
        ] = self.weights.category
        output[
            batch_rows,
            self.dimensions.text_dim + self.dimensions.category_dim + regions,
        ] = self.weights.region
        output /= self.weights.normalization_scale
        return output[0] if squeeze else output


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GnprContentGeoError(f"{name} 必须是正整数")
    return value


def fuse_content_geo_batch(
    text_embeddings: np.ndarray | torch.Tensor,
    category_indices: np.ndarray | torch.Tensor,
    region_indices: np.ndarray | torch.Tensor,
    *,
    dimensions: GnprContentGeoDimensions,
    weights: GnprContentGeoWeights | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build one bounded dense batch without materializing the full dataset."""

    text = (
        text_embeddings.to(device=device, dtype=torch.float32)
        if isinstance(text_embeddings, torch.Tensor)
        else torch.tensor(text_embeddings, dtype=torch.float32, device=device)
    )
    categories = (
        category_indices.to(device=text.device, dtype=torch.long)
        if isinstance(category_indices, torch.Tensor)
        else torch.tensor(category_indices, dtype=torch.long, device=text.device)
    )
    regions = (
        region_indices.to(device=text.device, dtype=torch.long)
        if isinstance(region_indices, torch.Tensor)
        else torch.tensor(region_indices, dtype=torch.long, device=text.device)
    )
    if text.ndim != 2 or text.shape[1] != dimensions.text_dim:
        raise GnprContentGeoError(
            f"文本向量必须为 [batch, {dimensions.text_dim}]"
        )
    if categories.ndim != 1 or regions.ndim != 1:
        raise GnprContentGeoError("类别和区域索引必须是一维数组")
    if text.shape[0] != categories.shape[0] or text.shape[0] != regions.shape[0]:
        raise GnprContentGeoError("文本、类别和区域的 batch 行数必须一致")
    if not torch.isfinite(text).all():
        raise GnprContentGeoError("文本向量包含 NaN 或 Inf")
    if torch.any(torch.linalg.vector_norm(text, dim=1) <= 0):
        raise GnprContentGeoError("文本向量不能是零向量")
    if categories.numel() and (
        int(categories.min()) < 0 or int(categories.max()) >= dimensions.category_dim
    ):
        raise GnprContentGeoError("category_index 超出词表范围")
    if regions.numel() and (
        int(regions.min()) < 0 or int(regions.max()) >= dimensions.region_dim
    ):
        raise GnprContentGeoError("region_index 超出词表范围")

    resolved_weights = weights or GnprContentGeoWeights()
    text = (
        F.normalize(text, p=2, dim=1, eps=1e-8)
        * resolved_weights.text
    )
    category_block = text.new_zeros((text.shape[0], dimensions.category_dim))
    region_block = text.new_zeros((text.shape[0], dimensions.region_dim))
    row_indices = torch.arange(text.shape[0], device=text.device)
    category_block[row_indices, categories] = resolved_weights.category
    region_block[row_indices, regions] = resolved_weights.region
    return torch.cat((text, category_block, region_block), dim=1) / (
        resolved_weights.normalization_scale
    )


def _parquet_files(path: Path) -> tuple[Path, ...]:
    files = tuple(sorted(item for item in path.rglob("*.parquet") if item.is_file()))
    if not files:
        raise GnprContentGeoError(f"没有找到 Parquet 分片：{path}")
    return files


def _parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(item).metadata.num_rows for item in _parquet_files(path))


def _read_embedding_ids(path: Path):
    import pyarrow as pa
    import pyarrow.csv as csv

    if not path.is_file():
        raise GnprContentGeoError(f"缺少 BGE POI ID 文件：{path}")
    table = csv.read_csv(
        path,
        read_options=csv.ReadOptions(column_names=["poi_id"]),
        parse_options=csv.ParseOptions(delimiter=",", quote_char='"'),
        convert_options=csv.ConvertOptions(column_types={"poi_id": pa.string()}),
    )
    return table.column("poi_id")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GnprContentGeoError(f"{name} 无法解析：{path}") from error
    if not isinstance(payload, dict):
        raise GnprContentGeoError(f"{name} 必须是 JSON object")
    return payload


def prepare_content_geo_metadata(
    *,
    embedding_dir: Path,
    feature_dir: Path,
    output_dir: Path,
    expected_rows: int | None = None,
    max_rows: int | None = None,
    allow_feature_superset: bool = False,
) -> dict[str, Any]:
    """Align compact category and region indices to the existing BGE row order."""

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    embedding_dir = embedding_dir.resolve()
    feature_dir = feature_dir.resolve()
    output_dir = output_dir.resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists():
        raise GnprContentGeoError(f"输出目录已存在：{output_dir}")
    if staging_dir.exists():
        raise GnprContentGeoError(f"临时输出目录已存在：{staging_dir}")
    if max_rows is not None and max_rows <= 0:
        raise GnprContentGeoError("max_rows 必须大于 0")

    embedding_manifest = _load_json(
        embedding_dir / "manifest.json", "BGE manifest"
    )
    embedding_path = embedding_dir / "embeddings.npy"
    embedding_ids_path = embedding_dir / "poi_ids.jsonl"
    if not embedding_path.is_file():
        raise GnprContentGeoError(f"缺少 BGE 向量：{embedding_path}")
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise GnprContentGeoError("BGE embeddings.npy 必须是二维数组")
    source_rows, text_dim = (int(value) for value in embeddings.shape)
    if expected_rows is not None and source_rows != expected_rows:
        raise GnprContentGeoError(
            f"BGE 行数不一致：{source_rows} != {expected_rows}"
        )
    embedding_ids = _read_embedding_ids(embedding_ids_path)
    if len(embedding_ids) != source_rows:
        raise GnprContentGeoError("BGE POI ID 行数与向量行数不一致")
    if int(pc.count_distinct(embedding_ids).as_py()) != source_rows:
        raise GnprContentGeoError("BGE POI ID 存在重复")

    category_vocab_count = _parquet_row_count(feature_dir / "category_vocab.parquet")
    region_vocab_count = _parquet_row_count(feature_dir / "region_vocab.parquet")
    feature_table = ds.dataset(
        feature_dir / "poi_features.parquet", format="parquet"
    ).to_table(columns=["poi_id", "category_index", "region_index"])
    feature_rows = feature_table.num_rows
    if feature_rows != source_rows and not allow_feature_superset:
        raise GnprContentGeoError(
            f"静态特征行数与 BGE 行数不一致：{feature_rows} != {source_rows}"
        )
    feature_ids = feature_table.column("poi_id")
    if int(pc.count_distinct(feature_ids).as_py()) != feature_rows:
        raise GnprContentGeoError("静态特征 POI ID 存在重复")

    output_rows = source_rows if max_rows is None else min(source_rows, max_rows)
    selected_ids = embedding_ids.slice(0, output_rows)
    positions = pc.index_in(selected_ids, value_set=feature_ids)
    if positions.null_count:
        missing_ids = selected_ids.filter(pc.is_null(positions)).slice(0, 5).to_pylist()
        raise GnprContentGeoError(
            f"{positions.null_count} 条 BGE POI 缺少静态特征，样例：{missing_ids}"
        )

    raw_categories = pc.take(feature_table.column("category_index"), positions)
    raw_regions = pc.take(feature_table.column("region_index"), positions)
    category_valid = pc.fill_null(
        pc.and_(
            pc.greater_equal(raw_categories, pa.scalar(0, raw_categories.type)),
            pc.less(
                raw_categories,
                pa.scalar(category_vocab_count, raw_categories.type),
            ),
        ),
        False,
    )
    region_valid = pc.fill_null(
        pc.and_(
            pc.greater_equal(raw_regions, pa.scalar(0, raw_regions.type)),
            pc.less(raw_regions, pa.scalar(region_vocab_count, raw_regions.type)),
        ),
        False,
    )
    invalid_category_count = int(pc.sum(pc.invert(category_valid)).as_py() or 0)
    invalid_region_count = int(pc.sum(pc.invert(region_valid)).as_py() or 0)
    if invalid_category_count or invalid_region_count:
        raise GnprContentGeoError(
            "静态特征包含无效索引："
            f"category={invalid_category_count}, region={invalid_region_count}"
        )
    categories = raw_categories.to_numpy(zero_copy_only=False).astype(
        np.int32, copy=False
    )
    regions = raw_regions.to_numpy(zero_copy_only=False).astype(
        np.int32, copy=False
    )

    staging_dir.mkdir(parents=True)
    category_path = staging_dir / "category_indices.npy"
    region_path = staging_dir / "region_indices.npy"
    np.save(category_path, categories, allow_pickle=False)
    np.save(region_path, regions, allow_pickle=False)
    dimensions = GnprContentGeoDimensions(
        text_dim=text_dim,
        category_dim=category_vocab_count,
        region_dim=region_vocab_count,
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "method": "GNPR-SID content-geo map-search adaptation",
        "catalog_filter": "embedding_poi_ids" if allow_feature_superset else None,
        "row_order": "BGE-M3 poi_ids.jsonl order",
        "feature_blocks": ["bge_m3_text", "category_code", "plus_code6"],
        "excluded_feature_blocks": ["visit_time", "visitor_identity"],
        "normalization": {
            "per_block_l2": True,
            "concatenated_scale": "1/sqrt(3)",
        },
        "dimensions": {
            "text": dimensions.text_dim,
            "category": dimensions.category_dim,
            "region": dimensions.region_dim,
            "total": dimensions.total_dim,
        },
        "inputs": {
            "embedding_dir": str(embedding_dir),
            "embedding_signature": embedding_manifest.get("signature"),
            "embedding_poi_ids_sha256": _sha256(embedding_ids_path),
            "feature_dir": str(feature_dir),
        },
        "outputs": {
            "category_indices": "category_indices.npy",
            "region_indices": "region_indices.npy",
        },
        "stats": {
            "source_embedding_rows": source_rows,
            "source_feature_rows": feature_rows,
            "output_rows": output_rows,
            "category_vocab_count": category_vocab_count,
            "region_vocab_count": region_vocab_count,
            "invalid_category_count": invalid_category_count,
            "invalid_region_count": invalid_region_count,
        },
        "sha256": {
            "category_indices": _sha256(category_path),
            "region_indices": _sha256(region_path),
        },
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging_dir / "_SUCCESS").touch()
    staging_dir.rename(output_dir)
    return manifest


def load_content_geo_artifact(
    input_dir: Path,
) -> tuple[GnprContentGeoDimensions, dict[str, Any], np.ndarray, np.ndarray]:
    """Validate and memory-map one completed compact input artifact."""

    input_dir = input_dir.resolve()
    if not (input_dir / "_SUCCESS").is_file():
        raise GnprContentGeoError("GNPR content-geo 输入缺少 _SUCCESS")
    manifest = _load_json(input_dir / "manifest.json", "GNPR content-geo manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise GnprContentGeoError("GNPR content-geo schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise GnprContentGeoError("GNPR content-geo 输入尚未完成")
    payload = manifest.get("dimensions")
    if not isinstance(payload, dict):
        raise GnprContentGeoError("manifest 缺少 dimensions")
    dimensions = GnprContentGeoDimensions(
        text_dim=_positive_int(payload.get("text"), "dimensions.text"),
        category_dim=_positive_int(payload.get("category"), "dimensions.category"),
        region_dim=_positive_int(payload.get("region"), "dimensions.region"),
    )
    if int(payload.get("total", -1)) != dimensions.total_dim:
        raise GnprContentGeoError("manifest 总维度与三个特征块不一致")
    outputs = manifest.get("outputs")
    stats = manifest.get("stats")
    if not isinstance(outputs, dict) or not isinstance(stats, dict):
        raise GnprContentGeoError("manifest 缺少 outputs 或 stats")
    categories = np.load(input_dir / str(outputs["category_indices"]), mmap_mode="r")
    regions = np.load(input_dir / str(outputs["region_indices"]), mmap_mode="r")
    expected_rows = int(stats.get("output_rows", -1))
    if categories.shape != (expected_rows,) or regions.shape != (expected_rows,):
        raise GnprContentGeoError("类别或区域索引 shape 与 manifest 不一致")
    if (
        int(categories.min()) < 0
        or int(categories.max()) >= dimensions.category_dim
        or int(regions.min()) < 0
        or int(regions.max()) >= dimensions.region_dim
    ):
        raise GnprContentGeoError("类别或区域索引超出 manifest 词表范围")
    return dimensions, manifest, categories, regions


def open_content_geo_array(
    input_dir: Path,
    text_embeddings: np.ndarray,
    *,
    weights: GnprContentGeoWeights | None = None,
) -> tuple[GnprContentGeoArray, dict[str, Any]]:
    """Open one completed artifact as a lazy array-like training source."""

    dimensions, manifest, categories, regions = load_content_geo_artifact(input_dir)
    return (
        GnprContentGeoArray(
            text_embeddings,
            categories,
            regions,
            dimensions=dimensions,
            weights=weights,
        ),
        manifest,
    )

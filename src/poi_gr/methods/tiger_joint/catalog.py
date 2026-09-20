"""Frozen BGE catalog access for SID-free TIGER joint training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from poi_gr.methods.tiger_joint.data import TigerJointDataError


class PoiCatalogError(TigerJointDataError):
    """Raised when the frozen BGE catalog violates its row contract."""


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise PoiCatalogError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PoiCatalogError(f"{name} 不是合法 JSON：{path}") from error
    if not isinstance(value, dict):
        raise PoiCatalogError(f"{name} 必须是 JSON object：{path}")
    return value


@dataclass(frozen=True)
class PoiEmbeddingStore:
    """Read-only BGE mmap plus its direct POI-ID-to-row lookup."""

    embeddings: np.ndarray
    row_by_poi_id: dict[str, int]
    poi_ids_path: Path
    poi_ids_sha256: str
    manifest_path: Path
    manifest_signature: str

    @classmethod
    def from_directory(
        cls,
        embedding_dir: Path,
        *,
        load_poi_index: bool = True,
    ) -> "PoiEmbeddingStore":
        """Load only the frozen BGE matrix and its own row-order file."""

        embedding_dir = embedding_dir.resolve()
        manifest_path = embedding_dir / "manifest.json"
        manifest = _load_json_object(manifest_path, "BGE Embedding manifest")
        if manifest.get("status") != "completed":
            raise PoiCatalogError("BGE Embedding manifest 状态不是 completed")
        output = manifest.get("output")
        if not isinstance(output, dict):
            raise PoiCatalogError("BGE Embedding manifest 缺少 output")
        shape = output.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
        ):
            raise PoiCatalogError("BGE Embedding shape 无效")
        dtype_name = output.get("dtype")
        if dtype_name not in {"float16", "float32"}:
            raise PoiCatalogError("BGE Embedding dtype 只允许 float16/float32")
        embedding_name = output.get("embeddings")
        poi_ids_name = output.get("poi_ids")
        if not isinstance(embedding_name, str) or not embedding_name:
            raise PoiCatalogError("BGE Embedding 文件名无效")
        if not isinstance(poi_ids_name, str) or not poi_ids_name:
            raise PoiCatalogError("BGE POI 行序文件名无效")

        embedding_path = embedding_dir / embedding_name
        poi_ids_path = embedding_dir / poi_ids_name
        if not poi_ids_path.is_file():
            raise PoiCatalogError(f"BGE POI 行序文件不存在：{poi_ids_path}")
        embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        if embeddings.shape != tuple(shape) or embeddings.dtype != np.dtype(dtype_name):
            raise PoiCatalogError("BGE Embedding NPY 与 manifest 不一致")

        row_by_poi_id: dict[str, int] = {}
        digest = hashlib.sha256()
        row_count = 0
        with poi_ids_path.open("rb") as stream:
            for row, raw_line in enumerate(stream):
                digest.update(raw_line)
                try:
                    poi_id = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise PoiCatalogError(
                        f"BGE poi_ids.jsonl 第 {row + 1} 行无效"
                    ) from error
                if not isinstance(poi_id, str) or not poi_id:
                    raise PoiCatalogError(
                        f"BGE poi_ids.jsonl 第 {row + 1} 行必须是非空字符串"
                    )
                if load_poi_index:
                    if poi_id in row_by_poi_id:
                        raise PoiCatalogError("BGE poi_ids.jsonl 包含重复 POI ID")
                    row_by_poi_id[poi_id] = row
                row_count += 1
        if row_count != int(shape[0]):
            raise PoiCatalogError(
                f"BGE POI 行数 {row_count} 与 Embedding 行数 {shape[0]} 不一致"
            )

        signature = manifest.get("signature")
        if not isinstance(signature, str) or len(signature) != 64:
            raise PoiCatalogError("BGE Embedding manifest signature 无效")
        return cls(
            embeddings=embeddings,
            row_by_poi_id=row_by_poi_id,
            poi_ids_path=poi_ids_path.resolve(),
            poi_ids_sha256=digest.hexdigest(),
            manifest_path=manifest_path.resolve(),
            manifest_signature=signature,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.embeddings.shape[0]), int(self.embeddings.shape[1])

    @property
    def poi_count(self) -> int:
        return self.shape[0]

    def row_for_poi_id(self, poi_id: str) -> int:
        """Return the BGE row for a raw POI identity."""

        if not self.row_by_poi_id:
            raise PoiCatalogError("当前 BGE store 未加载 POI 行索引")
        if not isinstance(poi_id, str) or not poi_id:
            raise PoiCatalogError("poi_id 必须是非空字符串")
        row = self.row_by_poi_id.get(poi_id)
        if row is None:
            raise PoiCatalogError("POI 不在冻结 BGE 目录中")
        return row

    def rows_for_poi_ids(self, poi_ids: Sequence[str]) -> np.ndarray:
        """Resolve a bounded raw POI identity batch in input order."""

        return np.asarray(
            [self.row_for_poi_id(poi_id) for poi_id in poi_ids],
            dtype=np.int64,
        )

    def gather(self, rows: Sequence[int] | np.ndarray) -> np.ndarray:
        """Gather a bounded row batch as contiguous float32 values."""

        indices = np.asarray(rows, dtype=np.int64)
        if indices.ndim != 1:
            raise PoiCatalogError("Embedding rows 必须是一维")
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= self.shape[0]
        ):
            raise PoiCatalogError("Embedding row 超出范围")
        if not indices.size:
            return np.empty((0, self.shape[1]), dtype=np.float32)
        return np.ascontiguousarray(self.embeddings[indices], dtype=np.float32)

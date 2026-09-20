"""Exact cosine retrieval for query-adapter training and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

from qg_prqk.adapters.config import ExactRetrievalConfig


class QueryAdapterRetrievalError(RuntimeError):
    """Raised when exact ANN construction or search fails."""


def _validate_vectors(values: np.ndarray, label: str) -> None:
    if values.ndim != 2 or not min(values.shape):
        raise QueryAdapterRetrievalError(f"{label} 必须是非空二维数组")
    indices = np.linspace(0, len(values) - 1, min(2048, len(values)), dtype=np.int64)
    sample = np.asarray(values[indices], dtype=np.float32)
    if not np.isfinite(sample).all():
        raise QueryAdapterRetrievalError(f"{label} 含 NaN/Inf")
    norms = np.linalg.norm(sample, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 5e-3:
        raise QueryAdapterRetrievalError(f"{label} 未保持 L2 normalize")


def _normalize_float32(values: np.ndarray, label: str) -> np.ndarray:
    block = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(block).all():
        raise QueryAdapterRetrievalError(f"{label} 含 NaN/Inf")
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise QueryAdapterRetrievalError(f"{label} 含零范数向量")
    return block / norms


def _stable_row_order(
    scores: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    for index in range(len(scores)):
        order = np.lexsort((rows[index], -scores[index]))
        scores[index] = scores[index, order]
        rows[index] = rows[index, order]
    return scores, rows


@dataclass
class ExactPoiIndex:
    """Own one exact FAISS GPU index and expose batched deterministic search."""

    poi_embeddings: np.ndarray
    config: ExactRetrievalConfig
    index: Any
    resources: Any

    @classmethod
    def build(
        cls, poi_embeddings: np.ndarray, config: ExactRetrievalConfig
    ) -> "ExactPoiIndex":
        _validate_vectors(poi_embeddings, "POI embedding")
        if config.backend == "numpy_flat_ip":
            return cls(poi_embeddings, config, None, None)
        if config.backend != "faiss_gpu_flat_ip":
            raise QueryAdapterRetrievalError(f"不支持 ANN backend={config.backend}")
        if faiss.get_num_gpus() <= config.gpu_id:
            raise QueryAdapterRetrievalError(f"FAISS 看不到配置 GPU {config.gpu_id}")
        resources = faiss.StandardGpuResources()
        gpu_config = faiss.GpuIndexFlatConfig()
        gpu_config.device = config.gpu_id
        gpu_config.useFloat16 = config.use_float16_storage
        index = faiss.GpuIndexFlatIP(
            resources, int(poi_embeddings.shape[1]), gpu_config
        )
        for start in range(0, len(poi_embeddings), config.add_batch_size):
            stop = min(start + config.add_batch_size, len(poi_embeddings))
            block = _normalize_float32(
                poi_embeddings[start:stop], "POI embedding"
            )
            index.add(block)
        if index.ntotal != len(poi_embeddings):
            raise QueryAdapterRetrievalError("FAISS index 行数不守恒")
        return cls(poi_embeddings, config, index, resources)

    def search(
        self, queries: np.ndarray, *, top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return float32 scores and int64 POI rows sorted by score/row."""

        _validate_vectors(queries, "Query embedding")
        if queries.shape[1] != self.poi_embeddings.shape[1]:
            raise QueryAdapterRetrievalError("Query/POI embedding 维度不一致")
        k = self.config.top_k if top_k is None else int(top_k)
        if k <= 0 or k > len(self.poi_embeddings):
            raise QueryAdapterRetrievalError("ANN top_k 非法")
        scores = np.empty((len(queries), k), dtype=np.float32)
        rows = np.empty((len(queries), k), dtype=np.int64)
        for start in range(0, len(queries), self.config.search_batch_size):
            stop = min(start + self.config.search_batch_size, len(queries))
            block = _normalize_float32(queries[start:stop], "Query embedding")
            if self.config.backend == "numpy_flat_ip":
                database = _normalize_float32(
                    self.poi_embeddings, "POI embedding"
                )
                all_scores = block @ database.T
                candidate_rows = np.broadcast_to(
                    np.arange(len(self.poi_embeddings), dtype=np.int64),
                    all_scores.shape,
                )
                local_rows = np.empty((len(block), k), dtype=np.int64)
                local_scores = np.empty((len(block), k), dtype=np.float32)
                for row_index in range(len(block)):
                    order = np.lexsort(
                        (candidate_rows[row_index], -all_scores[row_index])
                    )[:k]
                    local_rows[row_index] = order
                    local_scores[row_index] = all_scores[row_index, order]
            else:
                local_scores, local_rows = self.index.search(block, k)
            scores[start:stop] = local_scores
            rows[start:stop] = local_rows
        if np.any(rows < 0) or not np.isfinite(scores).all():
            raise QueryAdapterRetrievalError("ANN 返回非法行号或分数")
        return _stable_row_order(scores, rows)

    def search_poi_rows(
        self, poi_rows: np.ndarray, *, top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search neighbors for positive POI rows without copying the full matrix."""

        rows = np.asarray(poi_rows, dtype=np.int64)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(self.poi_embeddings)):
            raise QueryAdapterRetrievalError("target POI rows 非法")
        queries = np.asarray(self.poi_embeddings[rows], dtype=np.float32)
        return self.search(queries, top_k=top_k)

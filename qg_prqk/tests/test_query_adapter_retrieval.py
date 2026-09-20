from __future__ import annotations

import unittest

import numpy as np

from qg_prqk.adapters.retrieval import ExactPoiIndex
from qg_prqk.adapters.config import ExactRetrievalConfig


class QueryAdapterRetrievalTest(unittest.TestCase):
    def test_index_explicitly_normalizes_quantized_like_inputs(self) -> None:
        poi = np.asarray([[1.003, 0.0], [0.0, 0.997]], dtype=np.float32)
        config = ExactRetrievalConfig(
            backend="numpy_flat_ip",
            gpu_id=0,
            top_k=2,
            add_batch_size=2,
            search_batch_size=2,
            use_float16_storage=False,
        )
        index = ExactPoiIndex.build(poi, config)
        scores, rows = index.search(np.asarray([[0.999, 0.0]], dtype=np.float32))
        self.assertEqual(rows.tolist(), [[0, 1]])
        self.assertTrue(np.allclose(scores, [[1.0, 0.0]], atol=1e-6))

    def test_numpy_exact_search_orders_ties_by_row(self) -> None:
        poi = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        config = ExactRetrievalConfig(
            backend="numpy_flat_ip",
            gpu_id=0,
            top_k=3,
            add_batch_size=2,
            search_batch_size=2,
            use_float16_storage=False,
        )
        index = ExactPoiIndex.build(poi, config)
        scores, rows = index.search(np.asarray([[1.0, 0.0]], dtype=np.float32))
        self.assertEqual(rows.tolist(), [[0, 1, 2]])
        self.assertTrue(np.allclose(scores, [[1.0, 1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()

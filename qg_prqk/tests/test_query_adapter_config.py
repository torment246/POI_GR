from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from qg_prqk.adapters.config import QueryAdapterConfigError, load_query_adapter_config


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = QG_ROOT / "configs/qg_prqk_p3a_1024x3_v1.yaml"


class QueryAdapterConfigTest(unittest.TestCase):
    def test_canonical_config_loads_reviewed_parameters(self) -> None:
        config = load_query_adapter_config(CONFIG_PATH)
        self.assertEqual(config.gate_query_limit, 50_000)
        self.assertEqual(config.sample_query_limit, 1_000)
        self.assertEqual(config.ann.backend, "faiss_gpu_flat_ip")
        self.assertEqual(config.ann.top_k, 100)
        self.assertEqual(config.adapter.batch_size, 2048)
        self.assertEqual(
            (
                config.adapter.semantic_ann_negatives,
                config.adapter.lexical_metadata_negatives,
                config.adapter.local_geo_negatives,
            ),
            (6, 4, 6),
        )

    def test_rejects_unreviewed_ann_backend(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        raw = copy.deepcopy(raw)
        raw["base_v2_1_config"] = str(
            QG_ROOT / "configs/qg_prqk_v2_1_category_1024x3.yaml"
        )
        raw["p3a"]["ann"]["backend"] = "faiss_cpu_flat_ip"
        with tempfile.TemporaryDirectory(dir=QG_ROOT / "configs") as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(
                yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(QueryAdapterConfigError, "ann.backend"):
                load_query_adapter_config(path)


if __name__ == "__main__":
    unittest.main()

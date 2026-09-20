from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.config import load_config
from qg_prqk.data.query_embeddings import (
    build_query_embeddings,
    iter_normalized_queries,
    validate_query_embeddings,
)


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = QG_ROOT / "configs/qg_prqk_1024x3.yaml"


class FakeEncoder:
    def get_sentence_embedding_dimension(self) -> int:
        return 1024

    def encode(self, sentences: list[str], **_: object) -> np.ndarray:
        values = np.zeros((len(sentences), 1024), dtype=np.float32)
        for row, sentence in enumerate(sentences):
            values[row, sum(ord(char) for char in sentence) % 1024] = 1.0
        return values


def write_query_parts(root: Path) -> dict:
    directory = root / "query_stats"
    directory.mkdir(parents=True)
    rows = [
        [(0, "北京南站"), (1, "朝阳大悦城")],
        [(2, "首都机场t3")],
    ]
    files = []
    for index, values in enumerate(rows):
        name = f"part-{index:05d}.parquet"
        pq.write_table(
            pa.table(
                {
                    "query_id": [item[0] for item in values],
                    "normalized_query": [item[1] for item in values],
                }
            ),
            directory / name,
        )
        files.append({"file": name, "rows": len(values), "shard_id": index})
    return {
        "source": {"is_prefix_sample": True},
        "outputs": {"query_stats": {"dir": "query_stats", "files": files}},
    }


class QueryEmbeddingsTest(unittest.TestCase):
    def setUp(self) -> None:
        outputs = QG_ROOT / "outputs"
        outputs.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=outputs)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.p2_dir = self.root / "p2"
        self.manifest = write_query_parts(self.p2_dir)

    def contract(self, _: Path) -> tuple[dict, int, str]:
        return self.manifest, 3, "a" * 64

    def test_iter_normalized_queries_preserves_query_id_order(self) -> None:
        with patch(
            "qg_prqk.data.query_embeddings._query_stats_contract",
            side_effect=self.contract,
        ):
            values = list(
                iter_normalized_queries(self.p2_dir, start_row=1, stop_row=3)
            )
        self.assertEqual(values, ["朝阳大悦城", "首都机场t3"])

    def test_fake_encoder_build_and_validation(self) -> None:
        experiment = self.root / "experiment"
        experiment.mkdir()
        config = load_config(CONFIG_PATH)
        config = replace(
            config,
            paths=replace(config.paths, output_dir=experiment),
            runtime=replace(config.runtime, show_progress=False),
        )
        output_dir = experiment / "query_embeddings"
        with patch(
            "qg_prqk.data.query_embeddings._query_stats_contract",
            side_effect=self.contract,
        ):
            result = build_query_embeddings(
                config,
                query_stats_dir=self.p2_dir,
                output_dir=output_dir,
                encoder_loader=lambda _: (FakeEncoder(), "cpu", 0.0),
            )
        self.assertEqual((result.rows, result.embedding_dim), (3, 1024))
        values = np.load(result.embeddings_path)
        self.assertEqual(values.shape, (3, 1024))
        self.assertTrue(np.allclose(np.linalg.norm(values, axis=1), 1.0))
        validated = validate_query_embeddings(output_dir)
        self.assertTrue(validated.reused)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.adapters.data import D3Query
from qg_prqk.adapters.selection_config import load_adapter_selection_config
from qg_prqk.adapters.cache import QueryEmbeddingCacheError, build_d3_query_cache


QG_ROOT = Path(__file__).resolve().parents[1]


class Encoder:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(texts)
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("synthetic interruption")
        values = np.asarray(
            [[int(text) + 1, 1, 2, 3] for text in texts], dtype=np.float32
        )
        return values / np.linalg.norm(values, axis=1, keepdims=True)


class QueryAdapterCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        (QG_ROOT / "outputs").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=QG_ROOT / "outputs")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = load_adapter_selection_config(
            QG_ROOT / "configs/qg_prqk_p3a_full_active_1024x3_v1.yaml"
        )
        category = replace(
            config.base.category_config,
            paths=replace(config.base.category_config.paths, output_dir=self.root),
            frozen=replace(config.base.category_config.frozen, poi_embedding_dim=4),
        )
        self.config = replace(
            config,
            base=replace(
                config.base,
                category_config=category,
                query_embedding=replace(
                    config.base.query_embedding, encode_buffer_size=2
                ),
            ),
        )
        self.queries = [
            D3Query(
                query_id=i,
                selection_hash=i,
                normalized_query=str(i),
                representative_raw_query=str(i),
                query_count=2,
                distinct_poi_count=1,
                target_poi_id=str(i),
                top1_count=2,
                top1_share=1.0,
                normalized_entropy=0.0,
                fine_category_id="fine",
                coarse_category_id="coarse",
                query_weight=1.0,
            )
            for i in range(8)
        ]
        self.source = self.root / "interrupted"
        (self.source / "internal_holdout").mkdir(parents=True)
        selection = self.source / "d3_full_selection.parquet"
        pq.write_table(
            pa.table(
                {
                    "query_id": list(range(8)),
                    "normalized_query": [str(i) for i in range(8)],
                }
            ),
            selection,
        )
        write_json_atomic(
            self.source / "internal_holdout/manifest.json",
            {
                "source_hashes": {
                    "p2_manifest_sha256": self.config.base.category_config.frozen.p2_manifest_sha256,
                    "p2_5_manifest_sha256": self.config.base.p2_5_manifest_sha256,
                    "gate_manifest_sha256": self.config.gate_manifest_sha256,
                    "full_d3_selection_sha256": sha256_file(selection),
                }
            },
        )
        (self.source / "run_log.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "stage": "query_encoding_progress",
                        "next_row": stop,
                        "rows": 8,
                    }
                )
                + "\n"
                for stop in (2, 4)
            ),
            encoding="utf-8",
        )
        self.expected_prefix = Encoder().encode(["0", "1", "2", "3"]).astype(np.float16)
        legacy = np.zeros((8, 4), dtype=np.float16)
        legacy[:4] = self.expected_prefix
        # Norm-valid but unconfirmed garbage must never be reused.
        legacy[4:] = [1, 0, 0, 0]
        np.save(self.source / "raw_query_embeddings_d3.npy", legacy)

    def test_recovery_preserves_prefix_and_resume_skips_committed_chunks(self) -> None:
        source_hash = sha256_file(self.source / "raw_query_embeddings_d3.npy")
        first = Encoder(fail_on_call=2)
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            build_d3_query_cache(
                self.config,
                self.queries,
                source_dir=self.source,
                encoder_loader=lambda _: (first, "cpu", 0.0),
            )
        cache = self.root / "query_cache_exact_full"
        self.assertFalse((cache / "embeddings.npy").exists())
        self.assertFalse((cache / "_SUCCESS").exists())
        second = Encoder()
        path, manifest = build_d3_query_cache(
            self.config,
            self.queries,
            source_dir=self.source,
            encoder_loader=lambda _: (second, "cpu", 0.0),
        )
        self.assertEqual(second.calls, [["6", "7"]])
        actual = np.load(path)
        np.testing.assert_array_equal(actual[:4], self.expected_prefix)
        expected = Encoder().encode([str(i) for i in range(8)])
        np.testing.assert_allclose(actual, expected, atol=0.0005)
        self.assertEqual(manifest["recovery"]["confirmed_rows"], 4)
        self.assertEqual(
            source_hash, sha256_file(self.source / "raw_query_embeddings_d3.npy")
        )

        def forbidden_loader(_: object) -> object:
            raise AssertionError("completed cache must not load BGE")

        self.assertEqual(
            build_d3_query_cache(
                self.config, self.queries, encoder_loader=forbidden_loader
            )[0],
            path,
        )

    def test_rejects_zero_in_logged_prefix(self) -> None:
        values = np.load(self.source / "raw_query_embeddings_d3.npy")
        values[1] = 0
        np.save(self.source / "raw_query_embeddings_d3.npy", values)
        with self.assertRaisesRegex(QueryEmbeddingCacheError, "零向量"):
            build_d3_query_cache(self.config, self.queries, source_dir=self.source)

    def test_rejects_query_order_change(self) -> None:
        with self.assertRaisesRegex(QueryEmbeddingCacheError, "行序"):
            build_d3_query_cache(
                self.config, list(reversed(self.queries)), source_dir=self.source
            )

    def test_rejects_corrupted_committed_chunk(self) -> None:
        encoder = Encoder(fail_on_call=1)
        with self.assertRaises(RuntimeError):
            build_d3_query_cache(
                self.config,
                self.queries,
                source_dir=self.source,
                encoder_loader=lambda _: (encoder, "cpu", 0.0),
            )
        chunk = self.root / "query_cache_exact_full/chunk_0000000_0000002.npy"
        values = np.load(chunk)
        values[0] = [1, 0, 0, 0]
        np.save(chunk, values)
        with self.assertRaisesRegex(QueryEmbeddingCacheError, "hash"):
            build_d3_query_cache(self.config, self.queries, source_dir=self.source)


if __name__ == "__main__":
    unittest.main()

"""Tests for sparse E1/E2 Query-to-POI aggregation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poi_gr.embedding.query_augmentation import (
    QueryAugmentationConfig,
    aggregate_query_poi_partition,
    run_query_poi_aggregation,
    validate_query_poi_aggregates,
)
from poi_gr.embedding.query_encoding import iter_query_catalog
from poi_gr.embedding.query_shards import build_train_query_shards
from poi_gr.embedding.query_stats import build_train_query_stats


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(query: str, poi_id: str) -> dict[str, object]:
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"<QUERY>{query}</QUERY>\n"
                    "<USER_GID><G_1></USER_GID>"
                ),
            },
            {"role": "assistant", "content": "<S1_1><S2_2><S3_3>"},
        ],
        "target_poi_id": poi_id,
        "split": "train",
    }


def _write_train(sft_dir: Path) -> None:
    records = [
        _sample("q1", "101"),
        _sample("q1", "101"),
        _sample("q2", "101"),
        _sample("q1", "102"),
        _sample("q3", "103"),
    ]
    train_path = sft_dir / "train.jsonl"
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (sft_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sft-main-data-v1",
                "status": "completed",
                "outputs": {
                    "train.jsonl": {
                        "rows": len(records),
                        "sha256": _sha256(train_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


class QueryAugmentationTest(unittest.TestCase):
    def test_partition_implements_e1_and_e2_weights(self) -> None:
        query_embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]],
            dtype=np.float32,
        )
        result = aggregate_query_poi_partition(
            query_ids=np.asarray([0, 1, 0, 2], dtype=np.int64),
            poi_rows=np.asarray([4, 4, 7, 9], dtype=np.int64),
            order_counts=np.asarray([2, 1, 1, 1], dtype=np.int64),
            query_df=np.asarray([2, 1, 1], dtype=np.int32),
            query_embeddings=query_embeddings,
            covered_poi_count=3,
        )
        np.testing.assert_array_equal(result["covered_poi_rows"], [4, 7, 9])
        np.testing.assert_allclose(
            result["e1_query_mean"][0].astype(np.float32),
            [2**-0.5, 2**-0.5],
            atol=1e-3,
        )
        self.assertGreater(
            float(result["e2_query_weighted"][0, 1]),
            float(result["e2_query_weighted"][0, 0]),
        )
        np.testing.assert_allclose(
            result["e2_query_weighted"][1].astype(np.float32),
            [1.0, 0.0],
            atol=1e-3,
        )
        np.testing.assert_array_equal(result["unique_query_count"], [2, 1, 1])
        np.testing.assert_array_equal(result["train_order_count"], [3, 1, 1])

    def test_full_build_aligns_sparse_rows_and_reuses_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sft_dir = root / "sft"
            sft_dir.mkdir()
            _write_train(sft_dir)
            shards_dir = root / "query_shards"
            build_train_query_shards(
                sft_dir,
                shards_dir,
                num_shards=4,
                buffer_rows_per_shard=2,
                show_progress=False,
            )
            stats_dir = root / "query_stats"
            stats = build_train_query_stats(
                shards_dir,
                stats_dir,
                poi_partitions=4,
                poi_buffer_rows=2,
                read_batch_rows=2,
                show_progress=False,
            )

            vectors_by_query = {
                "q1": np.asarray([1.0, 0.0], dtype=np.float32),
                "q2": np.asarray([0.0, 1.0], dtype=np.float32),
                "q3": np.asarray([2**-0.5, 2**-0.5], dtype=np.float32),
            }
            query_vectors = np.stack(
                [
                    vectors_by_query[query]
                    for query in iter_query_catalog(stats_dir)
                ]
            ).astype(np.float16)
            query_dir = root / "query_embeddings"
            query_dir.mkdir()
            query_path = query_dir / "embeddings.npy"
            np.save(query_path, query_vectors)
            (query_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "output": {
                            "embeddings": query_path.name,
                            "shape": list(query_vectors.shape),
                        },
                    }
                ),
                encoding="utf-8",
            )

            poi_dir = root / "poi_embeddings"
            poi_dir.mkdir()
            poi_ids_path = poi_dir / "poi_ids.jsonl"
            poi_ids_path.write_text('"103"\n"101"\n"102"\n', encoding="utf-8")
            (poi_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "output": {"shape": [3, 2]},
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "aggregates"
            config = QueryAugmentationConfig(
                job_name="query-augmentation-test",
                query_stats_dir=stats_dir,
                expected_stats_manifest_sha256=_sha256(stats.manifest_path),
                query_embeddings_dir=query_dir,
                expected_query_embeddings_sha256=_sha256(query_path),
                poi_embedding_dir=poi_dir,
                expected_poi_ids_sha256=_sha256(poi_ids_path),
                expected_poi_rows=3,
                expected_covered_pois=3,
                output_dir=output_dir,
                read_batch_rows=2,
                resume=False,
            )
            result = run_query_poi_aggregation(
                config, project_root=root, show_progress=False
            )
            self.assertFalse(result.reused)
            self.assertEqual(result.covered_pois, 3)
            rows = np.load(output_dir / "covered_poi_rows.npy")
            e1 = np.load(output_dir / "e1_query_mean.npy")
            by_row = {int(row): e1[index] for index, row in enumerate(rows)}
            np.testing.assert_allclose(by_row[0], vectors_by_query["q3"], atol=1e-3)
            np.testing.assert_allclose(by_row[2], vectors_by_query["q1"], atol=1e-3)
            np.testing.assert_allclose(
                by_row[1], [2**-0.5, 2**-0.5], atol=1e-3
            )

            validated = validate_query_poi_aggregates(
                output_dir,
                expected_stats_manifest_sha256=config.expected_stats_manifest_sha256,
            )
            self.assertTrue(validated.reused)
            reused = run_query_poi_aggregation(
                config, project_root=root, show_progress=False
            )
            self.assertTrue(reused.reused)


if __name__ == "__main__":
    unittest.main()

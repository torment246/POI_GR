from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from qg_prqk.data.query_supervision_config import DepthThreshold, load_category_config
from qg_prqk.data.query_supervision import (
    DEPTH_LABELS,
    QUERY_POI_LAYER_EDGE_SCHEMA,
    build_category_depth,
    capped_log_support,
    classify_depth,
    layer_reliability,
    normalized_entropy,
    set_progress_enabled,
    summarize_distribution,
    validate_category_depth,
)
from qg_prqk.data.query_statistics import QUERY_POI_SCHEMA, QUERY_STATS_SCHEMA


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parquet_info(path: Path) -> dict[str, object]:
    return {
        "file": path.name,
        "rows": pq.ParquetFile(path).metadata.num_rows,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "shard_id": 0,
    }


def exact_stats(counts: list[int]) -> tuple[float, float]:
    entropy, normalized = normalized_entropy(counts)
    return entropy, normalized


class QuerySupervisionTest(unittest.TestCase):
    def test_entropy_depth_priority_and_reliability(self) -> None:
        fine = summarize_distribution({"100001": 4, "100002": 1})
        coarse = summarize_distribution({"10": 5})
        self.assertAlmostEqual(fine.concentration, 0.8)
        self.assertEqual(coarse.normalized_entropy, 0.0)
        d2 = DepthThreshold(3, 0.8, 0.8)
        d1 = DepthThreshold(3, 0.85, 0.35)
        self.assertEqual(
            classify_depth(
                is_p2_exact_core=False,
                query_count=5,
                fine=fine,
                coarse=coarse,
                d2_fine=d2,
                d1_coarse=d1,
            ),
            2,
        )
        self.assertEqual(
            classify_depth(
                is_p2_exact_core=True,
                query_count=1,
                fine=fine,
                coarse=coarse,
                d2_fine=d2,
                d1_coarse=d1,
            ),
            3,
        )
        self.assertAlmostEqual(capped_log_support(20, 20), 1.0)
        self.assertAlmostEqual(capped_log_support(200, 20), 1.0)
        self.assertLessEqual(layer_reliability(1.0, 0.8, 0.2, 1.0), 1.0)

    def test_synthetic_full_build_and_validation(self) -> None:
        set_progress_enabled(False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._build_fixture(root)
            config = load_category_config(config_path)
            result = build_category_depth(config)
            self.assertEqual(result.category_rows, 4)
            self.assertEqual(result.unique_queries, 4)
            self.assertEqual(result.layer_edges, 9)
            self.assertEqual(
                result.depth_counts,
                {label: 1 for label in DEPTH_LABELS},
            )
            validated = validate_category_depth(result.output_dir, config)
            self.assertTrue(validated.reused)
            metrics = json.loads(
                (result.output_dir / "p2_5_cat_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                metrics["category_mapping"]["fine_category_count"], 4
            )
            self.assertEqual(
                metrics["category_mapping"]["observed_fine_category_count"],
                3,
            )
            self.assertEqual(
                metrics["category_mapping"]["missing_fine_category_codes"],
                ["100003"],
            )
            depth_rows = pq.read_table(
                result.output_dir / "query_category_depth.parquet"
            ).to_pylist()
            self.assertEqual(
                [row["supervision_depth"] for row in depth_rows], [3, 2, 1, 0]
            )
            edges = pq.read_table(
                result.output_dir / "query_poi_layer_edges.parquet"
            )
            self.assertTrue(
                edges.schema.equals(
                    QUERY_POI_LAYER_EDGE_SCHEMA, check_metadata=False
                )
            )
            probabilities: dict[tuple[int, int], float] = {}
            for row in edges.to_pylist():
                key = (row["query_id"], row["layer"])
                probabilities[key] = probabilities.get(key, 0.0) + row[
                    "edge_probability"
                ]
            self.assertTrue(
                all(math.isclose(value, 1.0) for value in probabilities.values())
            )

    def _build_fixture(self, root: Path) -> Path:
        qg_configs = root / "qg_prqk/configs"
        qg_configs.mkdir(parents=True)
        p2_dir = root / "qg_prqk/outputs/p2/query_stats"
        stats_dir = p2_dir / "query_stats"
        pairs_dir = p2_dir / "query_poi_pairs"
        stats_dir.mkdir(parents=True)
        pairs_dir.mkdir()

        query_counts = [10, 5, 4, 2]
        pair_specs = [
            [("p0", 9), ("p3", 1)],
            [("p0", 4), ("p1", 1)],
            [("p0", 2), ("p2", 2)],
            [("p0", 1), ("p3", 1)],
        ]
        stats_rows = []
        pair_rows = []
        for query_id, (query_count, pairs) in enumerate(
            zip(query_counts, pair_specs, strict=True)
        ):
            ranked = sorted(pairs, key=lambda item: (-item[1], item[0]))
            entropy, normalized = exact_stats([count for _, count in ranked])
            top1_id, top1_count = ranked[0]
            top1_share = top1_count / query_count
            top2_share = ranked[1][1] / query_count
            stats_rows.append(
                {
                    "query_id": query_id,
                    "query_shard_id": 0,
                    "normalized_query": f"query-{query_id}",
                    "representative_raw_query": f"Query {query_id}",
                    "query_count": query_count,
                    "distinct_poi_count": len(pairs),
                    "top1_poi_id": top1_id,
                    "top1_count": top1_count,
                    "top1_share": top1_share,
                    "top2_share": top2_share,
                    "margin": top1_share - top2_share,
                    "entropy": entropy,
                    "normalized_entropy": normalized,
                    "is_high_confidence": query_id == 0,
                }
            )
            for poi_id, count in sorted(pairs):
                pair_rows.append(
                    {
                        "query_id": query_id,
                        "normalized_query": f"query-{query_id}",
                        "target_poi_id": poi_id,
                        "pair_count": count,
                        "pair_share": count / query_count,
                        "is_top1": poi_id == top1_id,
                        "retained_for_sid": query_id == 0 and poi_id == top1_id,
                        "sid_weight": 1.0 if query_id == 0 and poi_id == top1_id else 0.0,
                    }
                )
        stats_part = stats_dir / "part-00000-of-00001.parquet"
        pairs_part = pairs_dir / "part-00000-of-00001.parquet"
        pq.write_table(pa.Table.from_pylist(stats_rows, schema=QUERY_STATS_SCHEMA), stats_part)
        pq.write_table(pa.Table.from_pylist(pair_rows, schema=QUERY_POI_SCHEMA), pairs_part)
        p2_manifest = {
            "schema_version": "qg-prqk-query-stats-v1",
            "status": "completed",
            "source": {
                "split": "train",
                "limit": None,
                "is_prefix_sample": False,
                "valid_and_test_read": False,
                "scanned_rows": sum(query_counts),
            },
            "settings": {"num_shards": 1},
            "stats": {
                "source_rows": sum(query_counts),
                "unique_queries": 4,
                "unique_query_poi_pairs": len(pair_rows),
                "covered_pois": 4,
                "retained_queries": 1,
            },
            "outputs": {
                "query_stats": {
                    "dir": "query_stats",
                    "rows": 4,
                    "files": [parquet_info(stats_part)],
                },
                "query_poi_pairs": {
                    "dir": "query_poi_pairs",
                    "rows": len(pair_rows),
                    "files": [parquet_info(pairs_part)],
                },
            },
        }
        p2_manifest_path = p2_dir / "manifest.json"
        p2_manifest_path.write_text(json.dumps(p2_manifest), encoding="utf-8")
        (p2_dir / "_SUCCESS").touch()

        catalog_dir = root / "data/catalog"
        catalog_dir.mkdir(parents=True)
        catalog_path = catalog_dir / "part-00000.json"
        poi_ids = ["p0", "p1", "p2", "p3"]
        codes = ["100001", "100001", "100002", "200001"]
        catalog_lines = [
            json.dumps(
                {
                    "poi_id": poi_id,
                    "category": f"category:{code}",
                    "category_code": code,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for poi_id, code in zip(poi_ids, codes, strict=True)
        ]
        catalog_path.write_text("".join(catalog_lines), encoding="utf-8")

        bge_dir = root / "outputs/bge"
        bge_dir.mkdir(parents=True)
        embeddings_path = bge_dir / "embeddings.npy"
        np.save(embeddings_path, np.zeros((4, 4), dtype=np.float16))
        poi_ids_path = bge_dir / "poi_ids.jsonl"
        poi_ids_path.write_text(
            "".join(json.dumps(value) + "\n" for value in poi_ids),
            encoding="utf-8",
        )
        bge_manifest = {
            "status": "completed",
            "output": {"shape": [4, 4], "dtype": "float16"},
            "model": {"normalize_embeddings": True},
            "input": {
                "total_rows": 4,
                "id_field": "poi_id",
                "sources": [
                    {
                        "name": catalog_path.name,
                        "rows_scanned": 4,
                        "bytes_scanned": catalog_path.stat().st_size,
                        "sha256_scanned": sha256(catalog_path),
                    }
                ],
            },
        }
        bge_manifest_path = bge_dir / "manifest.json"
        bge_manifest_path.write_text(json.dumps(bge_manifest), encoding="utf-8")

        category_dir = root / "outputs/category"
        category_dir.mkdir(parents=True)
        category_indices = category_dir / "category_indices.npy"
        np.save(category_indices, np.array([0, 0, 1, 3], dtype=np.int32))
        vocab_dir = root / "outputs/vocab/category_vocab.parquet"
        vocab_dir.mkdir(parents=True)
        vocab_path = vocab_dir / "part.parquet"
        pq.write_table(
            pa.table(
                {
                    "category_code": pa.array(
                        ["100001", "100002", "100003", "200001"],
                        type=pa.string(),
                    ),
                    "category_index": pa.array([0, 1, 2, 3], type=pa.int64()),
                }
            ),
            vocab_path,
        )
        source_manifest = {
            "status": "completed",
            "row_order": "BGE-M3 poi_ids.jsonl order",
            "stats": {
                "output_rows": 4,
                "source_embedding_rows": 4,
                "source_feature_rows": 4,
                "category_vocab_count": 4,
                "invalid_category_count": 0,
            },
            "sha256": {"category_indices": sha256(category_indices)},
        }
        source_manifest_path = category_dir / "manifest.json"
        source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

        threshold = {
            "min_query_count": 3,
            "min_concentration": 0.8,
            "max_normalized_entropy": 0.4,
        }
        coarse_threshold = {
            "min_query_count": 3,
            "min_concentration": 0.85,
            "max_normalized_entropy": 0.35,
        }
        config = {
            "schema_version": "qg-prqk-category-config-v2.1",
            "method": {
                "name": "synthetic",
                "version": "v2.1-CAT",
                "task": "exact_single_poi_retrieval",
                "city": "北京",
            },
            "paths": {
                "project_root": "../..",
                "poi_catalog": "data/catalog",
                "poi_embeddings": "outputs/bge/embeddings.npy",
                "poi_ids": "outputs/bge/poi_ids.jsonl",
                "embedding_manifest": "outputs/bge/manifest.json",
                "p2_query_stats": "qg_prqk/outputs/p2/query_stats",
                "output_root": "qg_prqk/outputs",
                "output_dir": "qg_prqk/outputs/synthetic_v2_1",
            },
            "frozen_inputs": {
                "poi_rows": 4,
                "poi_embedding_dim": 4,
                "poi_embedding_dtype": "float16",
                "poi_embedding_normalized": True,
                "poi_embedding_manifest_sha256": sha256(bge_manifest_path),
                "poi_ids_sha256": sha256(poi_ids_path),
                "p2_manifest_sha256": sha256(p2_manifest_path),
                "reencode_poi_text": False,
                "append_category_to_poi_text": False,
            },
            "category": {
                "fine_category_id": {
                    "source_column": "category_code",
                    "transform": "identity",
                    "expected_count": 4,
                },
                "coarse_category_id": {
                    "source_column": "category_code",
                    "transform": "first_2_digits",
                    "expected_count": 2,
                },
                "readable_path_column": "category",
                "require_full_coverage": True,
                "require_fine_to_one_coarse": True,
                "unknown_or_other_is_legal_category": True,
                "use_as_embedding_input": False,
                "existing_mapping": {
                    "category_indices": "outputs/category/category_indices.npy",
                    "category_indices_sha256": sha256(category_indices),
                    "category_indices_row_order": "BGE-M3 poi_ids.jsonl order",
                    "category_vocab": "outputs/vocab/category_vocab.parquet",
                    "category_vocab_data_sha256": sha256(vocab_path),
                    "source_manifest": "outputs/category/manifest.json",
                    "source_manifest_sha256": sha256(source_manifest_path),
                },
                "p2_5_output_mapping": "category_mapping.parquet",
            },
            "query_depth": {
                "d3": {"use_existing_p2_exact_core": True},
                "d2_fine_category": threshold,
                "d1_coarse_category": coarse_threshold,
                "support_cap": 20,
                "support_transform": "normalized_log1p_min",
                "reliability_gamma": 1.0,
                "sensitivity": [
                    {
                        "name": "relaxed",
                        "d2_fine_category": threshold,
                        "d1_coarse_category": coarse_threshold,
                    },
                    {
                        "name": "baseline",
                        "d2_fine_category": threshold,
                        "d1_coarse_category": coarse_threshold,
                    },
                    {
                        "name": "strict",
                        "d2_fine_category": threshold,
                        "d1_coarse_category": coarse_threshold,
                    },
                ],
            },
            "runtime": {
                "overwrite": False,
                "require_sample_and_medium_gate_before_full": False,
                "p2_5_sample_query_limit": 1,
                "p2_5_medium_query_limit": 3,
            },
        }
        config_path = qg_configs / "synthetic.yaml"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return config_path


if __name__ == "__main__":
    unittest.main()

"""Synthetic tests for full-catalog GNPR content-geographic inputs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.gnpr_content_geo import (  # noqa: E402
    GnprContentGeoWeights,
    fuse_content_geo_batch,
    load_content_geo_artifact,
    prepare_content_geo_metadata,
)
from poi_gr.rqvae import RQVAE  # noqa: E402
from poi_gr.rqvae_training import (  # noqa: E402
    export_checkpoint_sid,
    load_training_config,
    run_training,
)


class GnprContentGeoTest(unittest.TestCase):
    def test_full_catalog_alignment_streaming_fusion_and_rqvae(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            embedding_dir = root / "bge"
            feature_dir = root / "features"
            output_dir = root / "prepared"
            embedding_dir.mkdir()
            (feature_dir / "poi_features.parquet").mkdir(parents=True)
            (feature_dir / "category_vocab.parquet").mkdir()
            (feature_dir / "region_vocab.parquet").mkdir()

            embeddings = np.asarray(
                [[3.0, 4.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
                dtype=np.float16,
            )
            np.save(embedding_dir / "embeddings.npy", embeddings)
            (embedding_dir / "poi_ids.jsonl").write_text(
                '"poi-b"\n"poi-cold"\n"poi-a"\n', encoding="utf-8"
            )
            (embedding_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "signature": "synthetic-bge",
                        "input": {"total_rows": 3, "fingerprint": "synthetic"},
                        "output": {"shape": [3, 4], "dtype": "float16"},
                    }
                ),
                encoding="utf-8",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"poi_id": "poi-a", "category_index": 0, "region_index": 1},
                        {"poi_id": "poi-b", "category_index": 1, "region_index": 0},
                        {
                            "poi_id": "poi-cold",
                            "category_index": 1,
                            "region_index": 1,
                        },
                    ]
                ),
                feature_dir / "poi_features.parquet" / "part-00000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"category_code": "food", "category_index": 0},
                        {"category_code": "office", "category_index": 1},
                    ]
                ),
                feature_dir / "category_vocab.parquet" / "part-00000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"plus_code_region": "r0", "region_index": 0},
                        {"plus_code_region": "r1", "region_index": 1},
                    ]
                ),
                feature_dir / "region_vocab.parquet" / "part-00000.parquet",
            )

            manifest = prepare_content_geo_metadata(
                embedding_dir=embedding_dir,
                feature_dir=feature_dir,
                output_dir=output_dir,
                expected_rows=3,
            )
            self.assertIsNone(manifest["catalog_filter"])
            self.assertEqual(manifest["stats"]["output_rows"], 3)
            self.assertEqual(manifest["stats"]["invalid_category_count"], 0)
            dimensions, _, categories, regions = load_content_geo_artifact(output_dir)
            self.assertEqual(dimensions.total_dim, 8)
            self.assertEqual(categories.tolist(), [1, 1, 0])
            self.assertEqual(regions.tolist(), [0, 1, 1])

            fused = fuse_content_geo_batch(
                embeddings,
                categories,
                regions,
                dimensions=dimensions,
            )
            self.assertEqual(tuple(fused.shape), (3, 8))
            torch.testing.assert_close(
                torch.linalg.vector_norm(fused, dim=1),
                torch.ones(3),
                rtol=1e-5,
                atol=1e-5,
            )
            weighted = fuse_content_geo_batch(
                embeddings,
                categories,
                regions,
                dimensions=dimensions,
                weights=GnprContentGeoWeights(1.0, 0.25, 0.25),
            )
            block_energy = weighted.square().sum(dim=0)
            self.assertAlmostEqual(
                float(block_energy[:4].sum()), 8.0 / 3.0, places=6
            )
            self.assertAlmostEqual(
                float(block_energy[4:6].sum()), 1.0 / 6.0, places=6
            )
            self.assertAlmostEqual(
                float(block_energy[6:].sum()), 1.0 / 6.0, places=6
            )
            model = RQVAE(
                input_dim=dimensions.total_dim,
                hidden_dim=8,
                latent_dim=4,
                codebook_sizes=(2, 2, 2),
                squared_error_reduction="element_mean",
            )
            result = model(fused)
            self.assertTrue(torch.isfinite(result.total_loss))
            self.assertEqual(tuple(result.codes.shape), (3, 3))

    def test_generic_training_and_export_use_lazy_fused_batches(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            embedding_dir = root / "bge"
            feature_dir = root / "features"
            prepared_dir = root / "prepared"
            poi_data_path = root / "pois.jsonl"
            embedding_dir.mkdir()
            (feature_dir / "poi_features.parquet").mkdir(parents=True)
            (feature_dir / "category_vocab.parquet").mkdir()
            (feature_dir / "region_vocab.parquet").mkdir()

            row_count = 32
            rng = np.random.default_rng(7)
            embeddings = rng.normal(size=(row_count, 4)).astype(np.float32)
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.save(
                embedding_dir / "embeddings.npy",
                embeddings.astype(np.float16),
                allow_pickle=False,
            )
            (embedding_dir / "poi_ids.jsonl").write_text(
                "".join(json.dumps(f"p{index}") + "\n" for index in range(row_count)),
                encoding="utf-8",
            )
            (embedding_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "signature": "synthetic-training-bge",
                        "input": {
                            "total_rows": row_count,
                            "fingerprint": "synthetic-training",
                        },
                        "output": {
                            "shape": [row_count, 4],
                            "dtype": "float16",
                        },
                    }
                ),
                encoding="utf-8",
            )
            feature_rows = [
                {
                    "poi_id": f"p{index}",
                    "category_index": index % 2,
                    "region_index": (index // 2) % 2,
                }
                for index in reversed(range(row_count))
            ]
            pq.write_table(
                pa.Table.from_pylist(feature_rows),
                feature_dir / "poi_features.parquet" / "part-00000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"category_code": "a", "category_index": 0},
                        {"category_code": "b", "category_index": 1},
                    ]
                ),
                feature_dir / "category_vocab.parquet" / "part-00000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"plus_code_region": "r0", "region_index": 0},
                        {"plus_code_region": "r1", "region_index": 1},
                    ]
                ),
                feature_dir / "region_vocab.parquet" / "part-00000.parquet",
            )
            prepare_content_geo_metadata(
                embedding_dir=embedding_dir,
                feature_dir=feature_dir,
                output_dir=prepared_dir,
                expected_rows=row_count,
            )
            poi_data_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "poi_id": f"p{index}",
                            "displayname": f"poi-{index}",
                            "address": "synthetic",
                            "alias": "",
                            "category": "synthetic",
                            "category_code": str(index % 2),
                            "lng": 116.0,
                            "lat": 39.0,
                            "layer": 1,
                            "click_score": float(index),
                        }
                    )
                    + "\n"
                    for index in range(row_count)
                ),
                encoding="utf-8",
            )
            config_path = root / "rqvae.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "embeddings": str(embedding_dir / "embeddings.npy"),
                            "poi_ids": str(embedding_dir / "poi_ids.jsonl"),
                            "embedding_manifest": str(
                                embedding_dir / "manifest.json"
                            ),
                            "feature_input_dir": str(prepared_dir),
                            "feature_block_weights": {
                                "text": 1.0,
                                "category": 0.25,
                                "region": 0.25,
                            },
                            "poi_data": str(poi_data_path),
                            "finite_check_chunk_rows": 8,
                        },
                        "model": {
                            "input_dim": 8,
                            "hidden_dim": 8,
                            "latent_dim": 4,
                            "rq_layers": 3,
                            "codebook_loss_weight": 1.0,
                            "commitment_loss_weight": 0.25,
                            "reconstruction_loss_weight": 2.0,
                            "diversity_loss_weight": 0.25,
                            "diversity_scale": 0.05,
                            "diversity_temperature": 0.5,
                        },
                        "initialization": {
                            "backend": "sklearn",
                            "sample_size": 16,
                            "iterations": 2,
                            "batch_size": 8,
                            "max_points_per_centroid": 32,
                        },
                        "training": {
                            "seed": 7,
                            "device": "cpu",
                            "batch_size": 8,
                            "block_rows": 16,
                            "max_epochs": 1,
                            "diversity_start_epoch": 1,
                            "early_stopping": False,
                            "checkpoint_epochs": [1],
                            "validation_ratio": 0.25,
                            "learning_rate": 0.001,
                            "weight_decay": 0.0,
                            "gradient_clip_norm": 1.0,
                            "collapse_utilization_threshold": 0.01,
                            "resume": True,
                            "show_progress": False,
                        },
                        "evaluation": {
                            "max_cases": 2,
                            "max_pois_per_case": 2,
                        },
                        "output_root": str(root / "runs"),
                        "experiments": {
                            "CONTENT-GEO-SMOKE": {
                                "codebook_sizes": [2, 2, 2]
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            config = load_training_config(
                config_path,
                PROJECT_ROOT,
                "CONTENT-GEO-SMOKE",
            )
            resolved = run_training(config, project_root=PROJECT_ROOT)
            self.assertEqual(resolved["status"], "completed")
            self.assertEqual(
                resolved["input_validation"]["input_shape"],
                [row_count, 8],
            )
            self.assertEqual(resolved["model"]["method"], "gnpr_content_geo_rqvae")
            self.assertEqual(resolved["model"]["reconstruction_loss_weight"], 2.0)
            self.assertEqual(resolved["model"]["diversity_loss_weight"], 0.25)
            self.assertEqual(
                resolved["model"]["diversity_implementation"],
                "straight_through_hard_utilization_encoder_only",
            )
            training_record = json.loads(
                (config.output_dir / "train_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertTrue(training_record["diversity_active"])
            self.assertGreaterEqual(
                training_record["train"]["diversity_loss"], 0.0
            )
            energy_ratio = resolved["input_validation"][
                "feature_block_squared_energy_ratio"
            ]
            self.assertAlmostEqual(energy_ratio["text"], 8.0 / 9.0)
            self.assertAlmostEqual(energy_ratio["category"], 1.0 / 18.0)
            self.assertAlmostEqual(energy_ratio["region"], 1.0 / 18.0)
            evaluation_dir = config.output_dir / "evaluations" / "epoch_1"
            manifest, metrics = export_checkpoint_sid(
                config.output_dir,
                Path("checkpoint_epoch_1.pt"),
                output_dir=evaluation_dir,
                device_name="cpu",
                batch_size=8,
            )
            self.assertEqual(manifest["method"], "gnpr_content_geo_rqvae")
            self.assertEqual(manifest["feature_input"]["shape"], [row_count, 8])
            self.assertEqual(metrics["basic"]["poi_count"], row_count)

    def test_formal_config_enables_paper_diversity_loss_after_one_third(self) -> None:
        config_path = PROJECT_ROOT / "configs/rqvae_gnpr_content_geo.yaml"
        for capacity in (256, 512, 1024):
            config = load_training_config(
                config_path,
                PROJECT_ROOT,
                f"GNPR-ContentGeo-BGE-M3-{capacity}x3",
            )
            self.assertEqual(config.codebook_sizes, (capacity,) * 3)
            self.assertEqual(config.max_epochs, 20)
            self.assertEqual(config.diversity_loss_weight, 0.25)
            self.assertEqual(config.diversity_scale, 0.05)
            self.assertEqual(config.diversity_temperature, 0.5)
            self.assertEqual(config.diversity_start_epoch, 7)


if __name__ == "__main__":
    unittest.main()

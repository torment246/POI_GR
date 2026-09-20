"""Tests for method-owned TIGER-Joint RQ-VAE initialization artifacts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger_joint.catalog import PoiEmbeddingStore  # noqa: E402
from poi_gr.methods.tiger_joint.initialization import (  # noqa: E402
    INITIALIZATION_CHECKPOINT_NAME,
    INITIALIZATION_MANIFEST_NAME,
    INITIALIZATION_SCHEMA_VERSION,
    FreshRqKMeansConfig,
    TigerJointInitializationError,
    build_fresh_rqvae,
    create_method_owned_kmeans_indices,
    fresh_rqvae_model_config,
    load_fresh_rqvae_initialization,
    module_sha256,
)
from poi_gr.pid.dedup import sha256_file  # noqa: E402


class TigerJointInitializationTest(unittest.TestCase):
    def test_method_owned_indices_are_deterministic(self) -> None:
        config = FreshRqKMeansConfig(
            kmeans_backend="sklearn",
            kmeans_sample_size=100,
            kmeans_iterations=2,
            kmeans_batch_size=16,
            kmeans_max_points_per_centroid=32,
            show_progress=False,
        )

        first, first_metadata = create_method_owned_kmeans_indices(1_000, config)
        second, second_metadata = create_method_owned_kmeans_indices(1_000, config)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first.shape, (100,))
        self.assertEqual(first.dtype, np.int64)
        self.assertEqual(first_metadata["kmeans_sample_rows"], 100)

    def test_load_initialization_binds_catalog_and_preflight(self) -> None:
        torch.manual_seed(42)
        model = build_fresh_rqvae()
        model_hash = module_sha256(model)
        embeddings = np.zeros((8, 1024), dtype=np.float16)
        store = PoiEmbeddingStore(
            embeddings=embeddings,
            row_by_poi_id={},
            poi_ids_path=Path("poi_ids.jsonl"),
            poi_ids_sha256="a" * 64,
            manifest_path=Path("manifest.json"),
            manifest_signature="b" * 64,
        )
        preflight_hash = "c" * 64

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            checkpoint_path = output_dir / INITIALIZATION_CHECKPOINT_NAME
            torch.save(
                {
                    "schema_version": INITIALIZATION_SCHEMA_VERSION,
                    "model": fresh_rqvae_model_config(),
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            manifest = {
                "schema_version": INITIALIZATION_SCHEMA_VERSION,
                "status": "completed",
                "formal_initialization_passed": True,
                "old_sid_artifacts_loaded": [],
                "source": {
                    "embedding_manifest_signature": store.manifest_signature,
                    "embedding_poi_ids_sha256": store.poi_ids_sha256,
                    "embedding_shape": list(store.shape),
                    "preflight_state_sha256": preflight_hash,
                },
                "model": fresh_rqvae_model_config(),
                "checkpoint": {
                    "path": INITIALIZATION_CHECKPOINT_NAME,
                    "sha256": sha256_file(checkpoint_path),
                    "model_sha256": model_hash,
                },
            }
            (output_dir / INITIALIZATION_MANIFEST_NAME).write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            loaded, loaded_manifest = load_fresh_rqvae_initialization(
                output_dir,
                embedding_store=store,
                preflight_state_sha256=preflight_hash,
                device=torch.device("cpu"),
            )

            self.assertEqual(module_sha256(loaded), model_hash)
            self.assertEqual(loaded_manifest, manifest)
            with self.assertRaisesRegex(TigerJointInitializationError, "正式预检"):
                load_fresh_rqvae_initialization(
                    output_dir,
                    embedding_store=store,
                    preflight_state_sha256="d" * 64,
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()

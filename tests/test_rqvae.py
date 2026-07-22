"""Lightweight tests for the MiniOneRec-style RQ-VAE pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.rqvae import (
    BlockShuffleSampler,
    EmbeddingMemmapDataset,
    RQVAE,
    RQVAEDataConfig,
    RQVAEModelConfig,
    RQVAEExportConfig,
    _collision_metrics,
    compute_rqvae_losses,
    export_sids_and_evaluate,
    inspect_embedding_source,
    resolve_last_level_collisions,
)


def make_model_config() -> RQVAEModelConfig:
    return RQVAEModelConfig(
        input_dim=8,
        hidden_dims=(6,),
        latent_dim=4,
        codebook_sizes=(4, 4, 4),
        dropout=0.0,
        commitment_weight=0.25,
        quantization_weight=1.0,
        kmeans_init=True,
        kmeans_iterations=5,
        sinkhorn_epsilons=(0.0, 0.0, 0.0),
        sinkhorn_iterations=5,
    )


class RQVAEPipelineTest(unittest.TestCase):
    def test_block_shuffle_keeps_rows_sequential_inside_each_block(self) -> None:
        sampler = BlockShuffleSampler(row_count=12, block_size=4, seed=9)

        indices = list(sampler)
        blocks = [
            indices[start : start + 4]
            for start in range(0, len(indices), 4)
        ]

        self.assertEqual(sorted(indices), list(range(12)))
        self.assertTrue(
            all(
                block == list(range(block[0], block[0] + len(block)))
                for block in blocks
            )
        )

    def test_model_uses_learnable_kmeans_initialized_codebooks(self) -> None:
        torch.manual_seed(7)
        model = RQVAE(make_model_config(), seed=7)
        embeddings = torch.randn(32, 8)
        outputs = model(embeddings)
        losses = compute_rqvae_losses(embeddings, outputs, 1.0)
        losses["loss"].backward()

        self.assertEqual(outputs["reconstruction"].shape, (32, 8))
        self.assertEqual(outputs["codes"].shape, (32, 3))
        self.assertTrue(torch.all(outputs["codes"] >= 0))
        self.assertTrue(torch.all(outputs["codes"] < 4))
        self.assertTrue(
            all(
                bool(layer.initialized.item())
                for layer in model.quantizer.layers
            )
        )
        self.assertTrue(
            all(
                layer.embedding.weight.grad is not None
                for layer in model.quantizer.layers
            )
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertAlmostEqual(
            float(losses["loss"].detach()),
            float(
                (
                    losses["reconstruction_mse"]
                    + losses["quantization_loss"]
                ).detach()
            ),
            places=6,
        )

    def test_collision_rate_matches_minionerec_definition(self) -> None:
        sids = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
            ],
            dtype=np.uint16,
        )

        metrics = _collision_metrics(sids)

        self.assertEqual(metrics["unique_sids"], 3)
        self.assertEqual(metrics["duplicate_assignments"], 1)
        self.assertAlmostEqual(metrics["collision_rate"], 0.25)
        self.assertAlmostEqual(metrics["collision_poi_rate"], 0.5)

    def test_sinkhorn_reassignment_reduces_collisions(self) -> None:
        raw_sids = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
            ],
            dtype=np.uint16,
        )
        last_residuals = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        codebook = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )

        resolved, history = resolve_last_level_collisions(
            raw_sids,
            last_residuals,
            codebook,
            device=torch.device("cpu"),
            epsilon=0.1,
            sinkhorn_iterations=20,
            max_collision_iterations=3,
            sinkhorn_batch_elements=1024,
        )

        self.assertTrue(history)
        self.assertLess(
            _collision_metrics(resolved)["collision_rate"],
            _collision_metrics(raw_sids)["collision_rate"],
        )

    def test_source_validation_and_sid_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embeddings_path = root / "embeddings.npy"
            ids_path = root / "poi_ids.jsonl"
            manifest_path = root / "manifest.json"
            sids_path = root / "sids.npy"
            generator = np.random.default_rng(3)
            embeddings = generator.normal(size=(64, 8)).astype(np.float16)
            np.save(embeddings_path, embeddings)
            with ids_path.open("w", encoding="utf-8") as handle:
                for index in range(64):
                    handle.write(json.dumps(f"poi-{index}") + "\n")
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "input": {"fingerprint": "synthetic"},
                        "output": {"shape": [64, 8]},
                    }
                ),
                encoding="utf-8",
            )
            data_config = RQVAEDataConfig(
                embeddings_path=embeddings_path,
                ids_path=ids_path,
                manifest_path=manifest_path,
                expected_rows=64,
                expected_dim=8,
            )
            source = inspect_embedding_source(data_config, max_rows=None)
            self.assertEqual(source.selected_rows, 64)

            dataset = EmbeddingMemmapDataset(embeddings_path)
            model = RQVAE(make_model_config(), seed=3)
            model(torch.from_numpy(embeddings.astype(np.float32)))
            metrics = export_sids_and_evaluate(
                model,
                dataset,
                output_path=sids_path,
                batch_size=16,
                num_workers=0,
                device=torch.device("cpu"),
                use_bf16=False,
                quantization_weight=1.0,
                export_config=RQVAEExportConfig(
                    resolve_collisions=False,
                    sinkhorn_epsilon=0.003,
                    sinkhorn_iterations=5,
                    max_collision_iterations=2,
                    sinkhorn_batch_elements=1024,
                ),
            )
            sids = np.load(sids_path)
            self.assertEqual(sids.shape, (64, 3))
            self.assertEqual(sids.dtype, np.dtype("uint16"))
            self.assertEqual(metrics["final_collisions"]["total_rows"], 64)
            self.assertEqual(len(metrics["nearest_assignment_codebooks"]), 3)
            self.assertEqual(len(metrics["final_codebooks"]), 3)


if __name__ == "__main__":
    unittest.main()

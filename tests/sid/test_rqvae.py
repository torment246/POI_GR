"""Synthetic tests for Vanilla RQ-VAE training and SID export."""

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

from poi_gr.sid.rqvae import RQVAE, hard_utilization_loss
from poi_gr.sid.training import (
    build_model,
    config_payload,
    create_fixed_indices,
    export_checkpoint_sid,
    initialize_codebooks_kmeans,
    load_training_config,
    run_training,
    shared_protocol_signature,
)
from poi_gr.sid.evaluation import load_sid_input


class RQVAETest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_forward_three_level_residual_quantization_and_backward(self) -> None:
        torch.manual_seed(42)
        model = RQVAE(
            input_dim=8,
            hidden_dim=6,
            latent_dim=4,
            codebook_sizes=[3, 4, 5],
        )
        inputs = torch.randn(7, 8, dtype=torch.float32)
        output = model(inputs)

        self.assertEqual(output.reconstruction.shape, (7, 8))
        self.assertEqual(output.latent.shape, (7, 4))
        self.assertEqual(output.quantized.shape, (7, 4))
        self.assertEqual(output.codes.shape, (7, 3))
        self.assertEqual(output.residual_norms.shape, (7, 3))
        self.assertTrue(torch.isfinite(output.residual_norms).all())
        for level, codebook_size in enumerate((3, 4, 5)):
            self.assertGreaterEqual(int(output.codes[:, level].min()), 0)
            self.assertLess(int(output.codes[:, level].max()), codebook_size)
        output.total_loss.backward()
        self.assertIsNotNone(model.encoder[0].weight.grad)
        self.assertIsNotNone(model.decoder[0].weight.grad)
        self.assertTrue(
            all(codebook.weight.grad is not None for codebook in model.quantizer.codebooks)
        )

    def test_encoder_decoder_initialization_is_capacity_independent(self) -> None:
        torch.manual_seed(42)
        smaller = RQVAE(8, 6, 4, [3, 3, 3])
        torch.manual_seed(42)
        larger = RQVAE(8, 6, 4, [7, 7, 7])
        for left, right in zip(
            smaller.encoder.parameters(), larger.encoder.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(
            smaller.decoder.parameters(), larger.decoder.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(left, right))

    def test_full_catalog_kmeans_scope_includes_validation_rows(self) -> None:
        train_mask, validation, initialization, metadata = create_fixed_indices(
            100,
            validation_ratio=0.1,
            kmeans_sample_size=100,
            seed=42,
            kmeans_sample_scope="all",
        )
        self.assertEqual(int(train_mask.sum()), 10)
        self.assertEqual(len(validation), 10)
        np.testing.assert_array_equal(initialization, np.arange(100))
        self.assertEqual(metadata["kmeans_sample_scope"], "all")
        self.assertEqual(metadata["kmeans_sample_pool_rows"], 100)
        self.assertEqual(metadata["kmeans_sample_rows"], 100)
        self.assertEqual(
            metadata["algorithm"], "full_initialization_pool_in_row_order"
        )

    def test_default_kmeans_scope_excludes_validation_rows(self) -> None:
        mask, validation, initialization, metadata = create_fixed_indices(
            100,
            validation_ratio=0.1,
            kmeans_sample_size=100,
            seed=42,
        )
        self.assertEqual(len(initialization), 90)
        self.assertFalse(np.any(mask[initialization]))
        self.assertTrue(set(validation).isdisjoint(set(initialization)))
        self.assertEqual(metadata["kmeans_sample_scope"], "train")

    def test_multi_hidden_layer_encoder_decoder(self) -> None:
        model = RQVAE(8, [7, 6, 5], 4, [3, 4, 5])
        encoder_shapes = [
            (module.in_features, module.out_features)
            for module in model.encoder
            if isinstance(module, torch.nn.Linear)
        ]
        decoder_shapes = [
            (module.in_features, module.out_features)
            for module in model.decoder
            if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(encoder_shapes, [(8, 7), (7, 6), (6, 5), (5, 4)])
        self.assertEqual(decoder_shapes, [(4, 5), (5, 6), (6, 7), (7, 8)])
        output = model(torch.randn(9, 8))
        self.assertEqual(output.latent.shape, (9, 4))
        self.assertEqual(output.reconstruction.shape, (9, 8))

    def test_vector_sum_loss_and_l2_reconstruction(self) -> None:
        torch.manual_seed(42)
        model = RQVAE(
            8,
            [7, 6],
            4,
            [3, 4, 5],
            reconstruction_normalization="l2",
            squared_error_reduction="vector_sum",
        )
        inputs = torch.randn(9, 8)
        inputs = torch.nn.functional.normalize(inputs, dim=1)
        output = model(inputs)

        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(output.reconstruction, dim=1),
                torch.ones(9),
                atol=1e-6,
            )
        )
        expected_reconstruction_loss = (
            (output.reconstruction - inputs).square().sum(dim=1).mean()
        )
        self.assertTrue(
            torch.allclose(
                output.reconstruction_loss,
                expected_reconstruction_loss,
            )
        )

    def test_reconstruction_loss_weight_only_scales_total_objective(self) -> None:
        torch.manual_seed(42)
        model = RQVAE(
            8,
            6,
            4,
            [3, 4, 5],
            reconstruction_loss_weight=2.5,
        )
        output = model(torch.randn(9, 8))
        expected_total = (
            2.5 * output.reconstruction_loss
            + output.codebook_loss
            + 0.25 * output.commitment_loss
        )
        torch.testing.assert_close(output.total_loss, expected_total)

    def test_hard_utilization_loss_penalizes_collapsed_codes_and_backpropagates(
        self,
    ) -> None:
        collapsed_distances = torch.tensor(
            [[0.0, 2.0, 4.0, 6.0]] * 8,
            requires_grad=True,
        )
        balanced_distances = torch.full((8, 4), 6.0)
        balanced_distances[
            torch.arange(8), torch.arange(8) % 4
        ] = 0.0

        collapsed_loss = hard_utilization_loss(
            collapsed_distances,
            temperature=0.5,
        )
        balanced_loss = hard_utilization_loss(
            balanced_distances,
            temperature=0.5,
        )

        self.assertGreater(float(collapsed_loss.detach()), 0.0)
        self.assertAlmostEqual(float(balanced_loss.detach()), 0.0, places=12)
        collapsed_loss.backward()
        self.assertIsNotNone(collapsed_distances.grad)
        self.assertTrue(torch.isfinite(collapsed_distances.grad).all())
        self.assertGreater(float(collapsed_distances.grad.abs().sum()), 0.0)

    def test_diversity_schedule_only_changes_total_objective(self) -> None:
        torch.manual_seed(42)
        model = RQVAE(
            8,
            6,
            4,
            [4, 4, 4],
            diversity_loss_weight=0.25,
            diversity_scale=0.05,
            diversity_temperature=0.5,
        )
        inputs = torch.randn(16, 8)
        inactive = model(inputs, diversity_active=False)
        active = model(inputs, diversity_active=True)

        self.assertGreater(float(active.diversity_loss.detach()), 0.0)
        self.assertEqual(float(inactive.diversity_loss.detach()), 0.0)
        torch.testing.assert_close(active.codes, inactive.codes)
        torch.testing.assert_close(
            active.total_loss,
            inactive.total_loss + 0.25 * active.diversity_loss,
        )

    def test_tiger_bge_config_uses_data_adapted_learning_rate(self) -> None:
        base_path = PROJECT_ROOT / "configs/sid/rqvae_beijing.yaml"
        tiger_path = PROJECT_ROOT / "configs/sid/rqvae_tiger_bge_m3.yaml"
        shared_protocol_fields = (
            "input_dim",
            "hidden_dim",
            "latent_dim",
            "codebook_sizes",
            "codebook_loss_weight",
            "commitment_loss_weight",
            "seed",
            "batch_size",
            "block_rows",
            "max_epochs",
            "checkpoint_epochs",
            "validation_ratio",
            "weight_decay",
            "gradient_clip_norm",
            "kmeans_backend",
            "kmeans_sample_size",
            "kmeans_iterations",
            "kmeans_batch_size",
            "kmeans_max_points_per_centroid",
        )

        for capacity in (256, 512, 1024):
            base = load_training_config(
                base_path,
                PROJECT_ROOT,
                f"BJ-RQVAE-{capacity}x3",
                max_epochs=20,
                checkpoint_epochs=(20,),
            )
            tiger = load_training_config(
                tiger_path,
                PROJECT_ROOT,
                f"TIGER-BGE-M3-{capacity}x3",
            )
            self.assertEqual(
                {
                    field: getattr(tiger, field)
                    for field in shared_protocol_fields
                },
                {
                    field: getattr(base, field)
                    for field in shared_protocol_fields
                },
            )
            self.assertEqual(tiger.learning_rate, 3e-4)
            self.assertEqual(base.learning_rate, 1e-3)

    def test_genpoi_geope_config_declares_three_capacity_experiments(self) -> None:
        config_path = (
            PROJECT_ROOT / "configs/sid/rqvae_genpoi_bge_m3_geope.yaml"
        )
        for capacity in (256, 512, 1024):
            config = load_training_config(
                config_path,
                PROJECT_ROOT,
                f"GenPOI-BGE-M3-GeoPE-{capacity}x3",
            )
            self.assertEqual(
                config.codebook_sizes,
                (capacity, capacity, capacity),
            )
            self.assertEqual(config.input_dim, 1024)
            self.assertIsNone(config.hidden_dim)
            self.assertEqual(config.hidden_dims, (512, 256, 128))
            self.assertEqual(config.latent_dim, 32)
            self.assertEqual(config.learning_rate, 5e-4)
            self.assertEqual(config.reconstruction_normalization, "l2")
            self.assertEqual(config.squared_error_reduction, "vector_sum")
            self.assertEqual(config.max_epochs, 20)
            self.assertEqual(config.checkpoint_epochs, (20,))
            self.assertEqual(
                config.embedding_manifest_path.name,
                "manifest.json",
            )
            self.assertIn(
                "beijing_poi_bge_m3_genpoi_geope",
                str(config.embeddings_path),
            )
            payload = config_payload(config)
            self.assertNotIn("hidden_dim", payload)
            self.assertEqual(payload["hidden_dims"], [512, 256, 128])

    def test_legacy_single_hidden_layer_payload_is_unchanged(self) -> None:
        config = load_training_config(
            PROJECT_ROOT / "configs/sid/rqvae_tiger_bge_m3.yaml",
            PROJECT_ROOT,
            "TIGER-BGE-M3-256x3",
        )
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.hidden_dims, (512,))
        payload = config_payload(config)
        self.assertEqual(payload["hidden_dim"], 512)
        self.assertNotIn("hidden_dims", payload)
        self.assertNotIn("reconstruction_normalization", payload)
        self.assertNotIn("squared_error_reduction", payload)
        self.assertNotIn("reconstruction_loss_weight", payload)
        self.assertNotIn("feature_block_weights", payload)
        self.assertNotIn("diversity_loss_weight", payload)
        self.assertNotIn("diversity_scale", payload)
        self.assertNotIn("diversity_temperature", payload)
        self.assertNotIn("diversity_start_epoch", payload)

    def write_training_fixture(self) -> Path:
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(64, 8)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings.astype(np.float16)
        embeddings_path = self.root / "embeddings.npy"
        ids_path = self.root / "poi_ids.jsonl"
        manifest_path = self.root / "embedding_manifest.json"
        poi_path = self.root / "pois.jsonl"
        config_path = self.root / "rqvae.yaml"
        np.save(embeddings_path, embeddings, allow_pickle=False)
        ids_path.write_text(
            "".join(json.dumps(f"p{index}") + "\n" for index in range(64)),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "input": {"total_rows": 64, "fingerprint": "synthetic-v1"},
                    "output": {"shape": [64, 8], "dtype": "float16"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        poi_path.write_text(
            "".join(
                json.dumps(
                    {
                        "poi_id": f"p{index}",
                        "displayname": f"poi-{index}",
                        "address": f"address-{index}",
                        "alias": "",
                        "category": "synthetic",
                        "category_code": None if index == 0 else str(index % 4),
                        "lng": 116.0,
                        "lat": 39.0,
                        "layer": 1,
                        "click_score": float(index),
                    }
                )
                + "\n"
                for index in range(64)
            ),
            encoding="utf-8",
        )
        config_path.write_text(
            f"""data:
  embeddings: {embeddings_path}
  poi_ids: {ids_path}
  embedding_manifest: {manifest_path}
  poi_data: {poi_path}
  finite_check_chunk_rows: 16
model:
  input_dim: 8
  hidden_dim: 6
  latent_dim: 4
  rq_layers: 3
  codebook_loss_weight: 1.0
  commitment_loss_weight: 0.25
initialization:
  backend: sklearn
  sample_size: 32
  iterations: 3
  batch_size: 16
  max_points_per_centroid: 64
training:
  seed: 42
  device: cpu
  batch_size: 16
  block_rows: 32
  max_epochs: 2
  early_stopping: false
  checkpoint_epochs: [1, 2]
  validation_ratio: 0.2
  learning_rate: 0.001
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  collapse_utilization_threshold: 0.01
  resume: true
  show_progress: false
evaluation:
  max_cases: 3
  max_pois_per_case: 2
output_root: {self.root / 'runs'}
experiments:
  SMALL-A:
    codebook_sizes: [4, 5, 6]
  SMALL-B:
    codebook_sizes: [3, 3, 3]
""",
            encoding="utf-8",
        )
        return config_path

    def test_kmeans_initialization_updates_all_codebooks(self) -> None:
        config_path = self.write_training_fixture()
        config = load_training_config(config_path, PROJECT_ROOT, "SMALL-A")
        model = build_model(config)
        before = [codebook.weight.detach().clone() for codebook in model.quantizer.codebooks]
        embeddings = np.load(config.embeddings_path, mmap_mode="r")
        results = initialize_codebooks_kmeans(
            model,
            embeddings,
            np.arange(32, dtype=np.int64),
            config,
            torch.device("cpu"),
        )
        self.assertEqual(len(results), 3)
        for previous, codebook in zip(before, model.quantizer.codebooks, strict=True):
            self.assertFalse(torch.equal(previous, codebook.weight))
        self.assertEqual([item["codebook_size"] for item in results], [4, 5, 6])

    def test_training_checkpoint_export_manifest_and_shared_entry(self) -> None:
        config_path = self.write_training_fixture()
        config_a = load_training_config(config_path, PROJECT_ROOT, "SMALL-A")
        config_b = load_training_config(config_path, PROJECT_ROOT, "SMALL-B")
        self.assertEqual(config_a.codebook_sizes, (4, 5, 6))
        self.assertEqual(config_b.codebook_sizes, (3, 3, 3))
        self.assertEqual(
            shared_protocol_signature(config_a),
            shared_protocol_signature(config_b),
        )

        resolved = run_training(config_a, project_root=PROJECT_ROOT)
        self.assertEqual(resolved["status"], "completed")
        self.assertEqual(resolved["training_result"]["last_epoch"], 2)
        self.assertEqual(resolved["training_result"]["stop_reason"], "max_epochs")
        self.assertFalse((config_a.output_dir / "best_model.pt").exists())
        for epoch in (1, 2):
            self.assertTrue(
                (config_a.output_dir / f"checkpoint_epoch_{epoch}.pt").is_file()
            )
        recovery = torch.load(
            config_a.output_dir / "last_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(recovery["epoch"], 2)
        self.assertIn("optimizer_state_dict", recovery)
        self.assertIsNone(recovery["scheduler_state_dict"])
        self.assertEqual(
            set(recovery["rng_state"]),
            {"python", "numpy", "torch_cpu", "torch_cuda"},
        )
        records = [
            json.loads(line)
            for line in (config_a.output_dir / "train_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual([record["epoch"] for record in records], [1, 2])
        for record in records:
            self.assertIn("monitor_sid", record)
            self.assertEqual(record["monitor_sid"]["poi_count"], 13)
            self.assertIn("distinct_sid_ratio", record["monitor_sid"])
            self.assertIn("collision_excess_ratio", record["monitor_sid"])
            self.assertIn("colliding_poi_ratio", record["monitor_sid"])
            for split in ("train", "validation"):
                self.assertTrue(
                    all("residual_norm" in layer for layer in record[split]["layers"])
                )

        resolved["status"] = "training"
        (config_a.output_dir / "resolved_config.json").write_text(
            json.dumps(resolved) + "\n", encoding="utf-8"
        )
        resumed = run_training(config_a, project_root=PROJECT_ROOT)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["training_result"]["last_epoch"], 2)

        evaluation_dir = config_a.output_dir / "evaluations" / "epoch_2"
        manifest, metrics = export_checkpoint_sid(
            config_a.output_dir,
            Path("checkpoint_epoch_2.pt"),
            output_dir=evaluation_dir,
            device_name="cpu",
            batch_size=16,
        )
        self.assertEqual(manifest["sid_codes"]["shape"], [64, 3])
        self.assertEqual(manifest["codebook_sizes"], [4, 5, 6])
        self.assertEqual(manifest["checkpoint"]["epoch"], 2)
        sid_input = load_sid_input(evaluation_dir / "sid_manifest.json")
        self.assertEqual(sid_input.codes.shape, (64, 3))
        self.assertEqual(metrics["basic"]["poi_count"], 64)
        self.assertEqual(len(metrics["layers"]), 3)
        self.assertTrue(
            all("normalized_entropy" in layer for layer in metrics["layers"])
        )
        self.assertEqual(
            {path.name for path in evaluation_dir.iterdir()},
            {
                "sid_codes.npy",
                "sid_manifest.json",
                "metrics.json",
                "collision_cases.jsonl",
            },
        )
        updated_resolved = json.loads(
            (config_a.output_dir / "resolved_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            updated_resolved["sid_evaluations"]["epoch_2"]["checkpoint_epoch"],
            2,
        )
        self.assertEqual(
            {path.name for path in config_a.output_dir.iterdir()},
            {
                "resolved_config.json",
                "checkpoint_epoch_1.pt",
                "checkpoint_epoch_2.pt",
                "last_checkpoint.pt",
                "train_metrics.jsonl",
                "evaluations",
            },
        )


if __name__ == "__main__":
    unittest.main()

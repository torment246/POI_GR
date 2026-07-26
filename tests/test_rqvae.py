"""Synthetic tests for Vanilla RQ-VAE training and SID export."""

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

from poi_gr.rqvae import RQVAE
from poi_gr.rqvae_training import (
    build_model,
    export_checkpoint_sid,
    initialize_codebooks_kmeans,
    load_training_config,
    run_training,
    shared_protocol_signature,
)
from poi_gr.sid_evaluation import load_sid_input


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

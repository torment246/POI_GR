from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qg_prqk.adapters.training import (
    TRAINING_DATA_SCHEMA_VERSION,
    AdapterTrainingError,
    load_training_bundle,
    train_adapter_bundle,
)
from qg_prqk.config import load_config
from qg_prqk.adapters.model import AdapterTrainingResult


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = QG_ROOT / "configs/qg_prqk_1024x3.yaml"
OUTPUT_ROOT = QG_ROOT / "outputs"


def write_bundle(path: Path, *, schema_version: str = TRAINING_DATA_SCHEMA_VERSION) -> None:
    rng = np.random.default_rng(7)
    rows, negatives, dimension = 20, 3, 8
    raw = rng.normal(size=(rows, dimension)).astype(np.float32)
    positive = np.roll(raw, shift=1, axis=1)
    negative = rng.normal(size=(rows, negatives, dimension)).astype(np.float32)
    targets = np.arange(rows, dtype=np.int64)
    candidates = np.stack((targets, targets + 20, targets + 40), axis=1)
    np.savez(
        path,
        schema_version=np.asarray(schema_version),
        raw_queries=raw,
        positive_embeddings=positive,
        negative_embeddings=negative,
        candidate_poi_rows=candidates,
        target_poi_rows=targets,
        false_negative_offsets=np.arange(rows + 1, dtype=np.int64),
        false_negative_rows=targets,
        query_weights=np.ones(rows, dtype=np.float32),
        query_ids=np.arange(rows, dtype=np.int64),
    )


class AdapterTrainingTest(unittest.TestCase):
    def test_load_bundle_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as temporary:
            path = Path(temporary) / "wrong.npz"
            write_bundle(path, schema_version="wrong")
            with self.assertRaisesRegex(AdapterTrainingError, "schema_version"):
                load_training_bundle(path)

    def test_train_bundle_writes_checkpoint_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as temporary:
            root = Path(temporary)
            training_data = root / "training.npz"
            experiment = root / "experiment"
            write_bundle(training_data)
            config = load_config(CONFIG_PATH, output_dir=experiment)
            metrics = {
                "recall_at_1": 0.5,
                "recall_at_10": 1.0,
                "recall_at_50": 1.0,
                "mean_positive_score": 0.4,
                "mean_hardest_negative_score": 0.3,
                "mean_hard_margin": 0.1,
            }
            with patch(
                "qg_prqk.adapters.training.fit_query_adapter",
                return_value=AdapterTrainingResult(
                    raw_metrics=metrics,
                    adapted_metrics=metrics,
                    epoch_losses=(0.25,),
                ),
            ):
                manifest = train_adapter_bundle(
                    config,
                    training_data=training_data,
                    output_dir=experiment / "query_adapter",
                    device="cpu",
                    batch_size=8,
                )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["training_data"]["selected_rows"], 20)
            self.assertFalse(manifest["training_data"]["valid_and_test_read"])
            self.assertEqual(manifest["split"]["dev_rows"], 1)
            self.assertEqual(
                manifest["evaluation"]["identity_fallback_decision"],
                "pending_hard_subset_review",
            )
            self.assertTrue((experiment / "query_adapter/adapter.pt").is_file())
            self.assertTrue((experiment / "query_adapter/manifest.json").is_file())
            self.assertTrue((experiment / "query_adapter/_SUCCESS").is_file())


if __name__ == "__main__":
    unittest.main()

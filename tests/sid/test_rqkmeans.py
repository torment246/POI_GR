"""Synthetic tests for residual K-Means SID construction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.sid.evaluation import load_sid_input
from poi_gr.sid.rqkmeans import load_rqkmeans_config, run_rqkmeans


class RQKMeansTest(unittest.TestCase):
    def setUp(self) -> None:
        output_root = PROJECT_ROOT / "outputs"
        output_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix="rqkmeans_test_", dir=output_root
        )
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_fixture(self) -> Path:
        rng = np.random.default_rng(42)
        centers = np.eye(4, 8, dtype=np.float32)
        embeddings = np.concatenate(
            [center + 0.02 * rng.normal(size=(16, 8)) for center in centers]
        ).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings.astype(np.float16)
        embeddings_path = self.root / "embeddings.npy"
        poi_ids_path = self.root / "poi_ids.jsonl"
        manifest_path = self.root / "embedding_manifest.json"
        poi_data_path = self.root / "pois.jsonl"
        output_dir = self.root / "run"
        np.save(embeddings_path, embeddings, allow_pickle=False)
        poi_ids_path.write_text(
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
        poi_data_path.write_text(
            "".join(
                json.dumps(
                    {
                        "poi_id": f"p{index}",
                        "displayname": f"poi-{index}",
                        "address": f"address-{index}",
                        "alias": "",
                        "category_code": str(index // 16),
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
        config_path = self.root / "rqkmeans.yaml"
        config_path.write_text(
            f"""experiment_id: SYNTHETIC-RQKMEANS
data:
  embeddings: {embeddings_path}
  poi_ids: {poi_ids_path}
  embedding_manifest: {manifest_path}
  poi_data: {poi_data_path}
  input_dim: 8
quantizer:
  codebook_sizes: [4, 4, 4]
  backend: sklearn
  sample_size: 32
  iterations: 3
  max_points_per_centroid: 64
  seed: 42
  validation_ratio: 0.2
runtime:
  chunk_rows: 16
  sample_chunk_rows: 16
  gpu_temp_memory_mib: 64
  finite_check_chunk_rows: 16
  resume: true
  show_progress: false
evaluation:
  max_cases: 3
  max_pois_per_case: 2
expected:
  rows: 64
  embedding_fingerprint: synthetic-v1
output_dir: {output_dir}
""",
            encoding="utf-8",
        )
        return config_path

    def test_full_streaming_build_and_shared_evaluation(self) -> None:
        config = load_rqkmeans_config(
            self._write_fixture(), PROJECT_ROOT
        )
        result = run_rqkmeans(config, project_root=PROJECT_ROOT)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["split"]["kmeans_sample_rows"], 32)
        self.assertEqual(result["split"]["validation_rows"], 13)
        sid_input = load_sid_input(config.output_dir / "sid_manifest.json")
        self.assertEqual(sid_input.codes.shape, (64, 3))
        self.assertEqual(sid_input.codebook_sizes, (4, 4, 4))
        metrics = json.loads(
            (config.output_dir / "metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metrics["basic"]["poi_count"], 64)
        self.assertEqual(len(metrics["layers"]), 3)
        self.assertGreater(metrics["quantization"]["mean_reconstruction_cosine"], 0.9)
        self.assertTrue((config.output_dir / "_SUCCESS").is_file())
        self.assertFalse((config.output_dir / ".work").exists())

    def test_faiss_residual_quantizer_screen(self) -> None:
        config = load_rqkmeans_config(
            self._write_fixture(),
            PROJECT_ROOT,
            output_dir=self.root / "faiss_run",
            implementation="faiss_residual_quantizer",
            backend="faiss_cpu",
            iterations=2,
            max_beam_size=2,
        )
        result = run_rqkmeans(
            config, project_root=PROJECT_ROOT, screen_only=True
        )
        self.assertEqual(result["status"], "screened")
        self.assertEqual(
            result["quantizer"]["implementation"],
            "faiss_residual_quantizer",
        )
        self.assertEqual(len(result["quantizer"]["levels"]), 3)
        self.assertEqual(result["screen_metrics"]["rows"], 13)
        self.assertTrue((config.output_dir / "rq_index.faiss").is_file())
        self.assertFalse((config.output_dir / "sid_codes.npy").exists())


if __name__ == "__main__":
    unittest.main()

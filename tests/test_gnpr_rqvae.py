"""Synthetic tests for the paper-style GNPR-SID RQ-VAE."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.gnpr_rqvae import (  # noqa: E402
    GnprRQVAE,
    dense_gnpr_batch,
)
from poi_gr.methods.gnpr_sid_input import (  # noqa: E402
    GnprSidInput,
    build_feature_dimensions,
    iter_gnpr_sid_input_batches,
)


class GnprRqvaeTest(unittest.TestCase):
    @staticmethod
    def rows(count: int) -> tuple[GnprSidInput, ...]:
        return tuple(
            GnprSidInput(
                poi_id=f"poi-{index}",
                category_index=index % 3,
                region_index=(index // 3) % 4,
                top_visit_hours=tuple(sorted({index % 24, (index * 5 + 3) % 24})),
                user_hash_indices=tuple(
                    sorted({index % 16, (index * 7 + 1) % 16})
                ),
                interaction_count=index + 1,
            )
            for index in range(count)
        )

    def test_dense_batch_offsets_and_paper_model_backward(self) -> None:
        dimensions = build_feature_dimensions(
            category_count=3,
            region_count=4,
            user_hash_buckets=16,
        )
        rows = self.rows(12)
        inputs = dense_gnpr_batch(rows, dimensions=dimensions)
        self.assertEqual(inputs.shape, (12, 47))
        self.assertTrue(torch.all((inputs == 0) | (inputs == 1)))
        model = GnprRQVAE(
            dimensions.total_dim,
            hidden_dims=(32, 16),
            latent_dim=8,
            codebook_sizes=(8, 8, 8),
            dropout=0.0,
        )
        output = model(inputs)
        self.assertEqual(output.reconstruction.shape, inputs.shape)
        self.assertEqual(output.codes.shape, (12, 3))
        self.assertEqual(output.residual_norms.shape, (12, 3))
        self.assertTrue(torch.isfinite(output.total_loss))
        self.assertTrue(torch.isfinite(output.reconstruction_cosine))
        reconstruction_cosine = float(output.reconstruction_cosine.detach())
        self.assertGreaterEqual(reconstruction_cosine, -1.0)
        self.assertLessEqual(reconstruction_cosine, 1.0)
        output.total_loss.backward()
        self.assertTrue(
            all(
                codebook.weight.grad is not None
                for codebook in model.quantizer.codebooks
            )
        )

    def test_small_dataset_loss_decreases_and_all_levels_are_used(self) -> None:
        torch.manual_seed(2024)
        previous_thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_thread_count)
        dimensions = build_feature_dimensions(
            category_count=3,
            region_count=4,
            user_hash_buckets=16,
        )
        inputs = dense_gnpr_batch(self.rows(48), dimensions=dimensions)
        model = GnprRQVAE(
            dimensions.total_dim,
            hidden_dims=(32, 16),
            latent_dim=8,
            codebook_sizes=(4, 4, 4),
            dropout=0.0,
            diversity_loss_weight=0.25,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        model.eval()
        with torch.no_grad():
            initial_loss = float(model(inputs).total_loss)
        model.train()
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            output.total_loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = model(inputs)
        self.assertLess(float(final.total_loss), initial_loss * 0.45)
        for level in range(3):
            self.assertGreaterEqual(torch.unique(final.codes[:, level]).numel(), 2)

    def test_streaming_reader_rejects_cold_rows_by_contract(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parquet_dir = root / "poi_sid_inputs.parquet"
            parquet_dir.mkdir()
            rows = self.rows(3)
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "poi_id": row.poi_id,
                            "category_index": row.category_index,
                            "region_index": row.region_index,
                            "top_visit_hours": list(row.top_visit_hours),
                            "user_hash_indices": list(row.user_hash_indices),
                            "interaction_count": row.interaction_count,
                        }
                        for row in rows
                    ]
                ),
                parquet_dir / "part-00000.parquet",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "gnpr-sid-input-v1",
                        "status": "completed",
                        "catalog_filter": "interaction_count > 0",
                        "dimensions": {
                            "category": 3,
                            "region": 4,
                            "time": 24,
                            "user_hash": 16,
                            "total": 47,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "_SUCCESS").touch()
            dimensions, batches = iter_gnpr_sid_input_batches(
                root,
                batch_size=2,
            )
            loaded = tuple(row for batch in batches for row in batch)
            self.assertEqual(dimensions.total_dim, 47)
            self.assertEqual(loaded, rows)

if __name__ == "__main__":
    unittest.main()

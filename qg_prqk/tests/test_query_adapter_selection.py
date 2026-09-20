from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from qg_prqk.adapters.data import D3Query
from qg_prqk.adapters.selection import (
    FULL_BUNDLE_SCHEMA_VERSION,
    TrainingTensors,
    _fresh_adapter,
    _load_full_bundle,
    _save_initial_state,
    _train_epochs,
    full_retrieval_metrics,
    partition_full_d3,
    select_best_epoch,
)
from qg_prqk.adapters.selection_config import load_adapter_selection_config
from qg_prqk.adapters.model import ResidualQueryAdapter


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = QG_ROOT / "configs/qg_prqk_p3a_full_1024x3_v1.yaml"


def _query(query_id: int) -> D3Query:
    return D3Query(
        query_id=query_id,
        selection_hash=query_id,
        normalized_query=f"q{query_id}",
        representative_raw_query=f"Q{query_id}",
        query_count=2,
        distinct_poi_count=1,
        target_poi_id=f"p{query_id}",
        top1_count=2,
        top1_share=1.0,
        normalized_entropy=0.0,
        fine_category_id="fine",
        coarse_category_id="coarse",
        query_weight=1.0,
    )


class QueryAdapterSelectionTest(unittest.TestCase):
    def test_train_epochs_streams_numpy_batches_to_device(self) -> None:
        config = load_adapter_selection_config(CONFIG_PATH)
        raw = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float16,
        )
        tensors = TrainingTensors(
            raw_queries=raw,
            positives=raw.copy(),
            negatives=np.asarray(
                [
                    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                ],
                dtype=np.float16,
            ),
            negative_valid_mask=torch.ones((3, 2), dtype=torch.bool),
            query_weights=np.ones(3, dtype=np.float32),
            target_rows_cpu=torch.arange(3, dtype=torch.long),
            reasonable_positive_rows=[
                np.asarray([index], dtype=np.int64) for index in range(3)
            ],
            query_ids=(0, 1, 2),
        )
        model = ResidualQueryAdapter(
            4,
            config.base.adapter.bottleneck,
            residual_scale=config.base.adapter.residual_scale,
            dropout=config.base.adapter.dropout,
        )
        with tempfile.TemporaryDirectory(dir=QG_ROOT / "outputs") as temporary:
            losses = _train_epochs(
                model,
                tensors,
                np.arange(3, dtype=np.int64),
                config,
                epochs=1,
                device=torch.device("cpu"),
                log_path=Path(temporary) / "run_log.jsonl",
                run_role="SMOKE",
            )
        self.assertEqual(len(losses), 1)
        self.assertTrue(np.isfinite(losses[0]))

    def test_full_bundle_loader_keeps_large_embeddings_memory_mapped(self) -> None:
        rows = 3
        dimension = 4
        negatives = 2
        with tempfile.TemporaryDirectory(dir=QG_ROOT / "outputs") as temporary:
            root = Path(temporary)
            raw_path = root / "raw.npy"
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()
            np.save(raw_path, np.ones((rows, dimension), dtype=np.float16))
            arrays = {
                "positive_embeddings": np.ones(
                    (rows, dimension), dtype=np.float16
                ),
                "negative_embeddings": np.ones(
                    (rows, negatives, dimension), dtype=np.float16
                ),
                "candidate_poi_rows": np.arange(
                    rows * negatives, dtype=np.int64
                ).reshape(rows, negatives),
                "negative_source_codes": np.ones(
                    (rows, negatives), dtype=np.int8
                ),
                "target_poi_rows": np.arange(rows, dtype=np.int64),
                "false_negative_offsets": np.arange(rows + 1, dtype=np.int64),
                "false_negative_rows": np.arange(rows, dtype=np.int64),
                "query_weights": np.ones(rows, dtype=np.float32),
                "query_ids": np.arange(rows, dtype=np.int64),
            }
            for name, values in arrays.items():
                np.save(bundle_dir / f"{name}.npy", values)
            (bundle_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": FULL_BUNDLE_SCHEMA_VERSION,
                        "status": "completed",
                        "rows": rows,
                        "embedding_dim": dimension,
                        "fixed_negatives_per_query": negatives,
                    }
                ),
                encoding="utf-8",
            )
            (bundle_dir / "_SUCCESS").touch()

            bundle = _load_full_bundle(bundle_dir, raw_path)

            self.assertIsInstance(bundle["raw_queries"], np.memmap)
            self.assertIsInstance(bundle["negative_embeddings"], np.memmap)
            self.assertEqual(
                bundle["negative_embeddings"].shape,
                (rows, negatives, dimension),
            )

    def test_partition_uses_rows_immediately_after_gate_for_holdout(self) -> None:
        queries = [_query(index) for index in range(10)]
        holdout, train = partition_full_d3(
            queries,
            [0, 1, 2],
            gate_exclusion_rows=3,
            holdout_rows=2,
        )
        self.assertEqual(holdout.tolist(), [3, 4])
        self.assertEqual(train.tolist(), [0, 1, 2, 5, 6, 7, 8, 9])
        self.assertEqual(len(np.intersect1d(holdout, train)), 0)

    def test_full_metrics_include_requested_cutoffs_and_drift(self) -> None:
        ranks = np.asarray([1, 5, 10, 21], dtype=np.int32)
        difficult = np.asarray([False, True, True, True])
        raw = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        view = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        metrics = full_retrieval_metrics(ranks, difficult, raw, view)
        self.assertEqual(metrics["recall_at_1"], 0.25)
        self.assertEqual(metrics["recall_at_5"], 0.5)
        self.assertEqual(metrics["recall_at_10"], 0.75)
        self.assertEqual(metrics["recall_at_20"], 0.75)
        self.assertAlmostEqual(metrics["mrr_at_10"], 0.325)
        self.assertAlmostEqual(metrics["difficult_recall_at_10"], 2.0 / 3.0)
        self.assertEqual(
            metrics["query_embedding_cosine_drift"]["mean"],
            0.25,
        )

    def test_best_epoch_applies_both_tie_breaks_then_earliest(self) -> None:
        records = [
            {
                "epoch": 1,
                "metrics": {
                    "recall_at_10": 0.7,
                    "recall_at_1": 0.4,
                    "difficult_recall_at_10": 0.5,
                },
            },
            {
                "epoch": 2,
                "metrics": {
                    "recall_at_10": 0.7,
                    "recall_at_1": 0.5,
                    "difficult_recall_at_10": 0.4,
                },
            },
            {
                "epoch": 3,
                "metrics": {
                    "recall_at_10": 0.7,
                    "recall_at_1": 0.5,
                    "difficult_recall_at_10": 0.6,
                },
            },
        ]
        self.assertEqual(select_best_epoch(records), 3)
        records.append(copy_record := dict(records[-1]))
        copy_record["epoch"] = 4
        self.assertEqual(select_best_epoch(records), 3)

    def test_select_and_final_can_reload_bit_identical_initial_state(self) -> None:
        config = load_adapter_selection_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory(dir=QG_ROOT / "outputs") as temporary:
            path = Path(temporary) / "initial.pt"
            _save_initial_state(path, config, embedding_dim=4)
            first = _fresh_adapter(path, config, device=torch.device("cpu"))
            second = _fresh_adapter(path, config, device=torch.device("cpu"))
            for first_value, second_value in zip(
                first.state_dict().values(),
                second.state_dict().values(),
                strict=True,
            ):
                self.assertTrue(torch.equal(first_value, second_value))

    def test_initial_state_is_seeded_before_random_weights_are_created(self) -> None:
        config = load_adapter_selection_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory(dir=QG_ROOT / "outputs") as temporary:
            first_path = Path(temporary) / "first.pt"
            second_path = Path(temporary) / "second.pt"
            torch.manual_seed(7)
            _save_initial_state(first_path, config, embedding_dim=4)
            torch.manual_seed(999)
            _save_initial_state(second_path, config, embedding_dim=4)
            first = torch.load(first_path, weights_only=True)
            second = torch.load(second_path, weights_only=True)
            self.assertFalse(first["historical_gate_initial_weights_reproduced"])
            self.assertIn("User-confirmed 2026-09-05", first["initialization"])
            for name, value in first["state_dict"].items():
                self.assertTrue(torch.equal(value, second["state_dict"][name]))
            self.assertFalse(bool(first["state_dict"]["up.weight"].any()))
            self.assertFalse(bool(first["state_dict"]["up.bias"].any()))


if __name__ == "__main__":
    unittest.main()

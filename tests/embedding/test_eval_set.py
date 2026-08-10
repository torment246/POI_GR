"""Tests for the frozen embedding evaluation-set contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from poi_gr.embedding.eval_set import (
    EmbeddingEvalSetError,
    freeze_embedding_eval_set,
)


def _sample(
    sample_id: str,
    order_id: str,
    searchid: str,
    query: str,
    poi_id: str,
    split: str,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"<QUERY>{query}</QUERY>\n"
                    "<USER_GID><G_1><G_2><G_3><G_4><G_5><G_6></USER_GID>"
                ),
            },
            {"role": "assistant", "content": "<S1_1><S2_2><S3_3>"},
        ],
        "order_id": order_id,
        "searchid": searchid,
        "target_poi_id": poi_id,
        "split": split,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> str:
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EmbeddingEvalSetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sft_dir = self.root / "sft"
        self.sft_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _prepare(
        self,
        *,
        train: list[dict[str, object]],
        valid: list[dict[str, object]],
        reference_indices: list[int],
    ) -> tuple[Path, Path]:
        train_sha256 = _write_jsonl(self.sft_dir / "train.jsonl", train)
        valid_sha256 = _write_jsonl(self.sft_dir / "valid.jsonl", valid)
        manifest = {
            "schema_version": "sft-main-data-v1",
            "status": "completed",
            "build_fingerprint": "synthetic-build",
            "time_split": {
                "field": "create_time",
                "train": {"start": "2026-07-01", "end": "2026-07-12"},
                "valid": "2026-07-13",
                "test": "2026-07-14",
            },
            "outputs": {
                "train.jsonl": {"rows": len(train), "sha256": train_sha256},
                "valid.jsonl": {"rows": len(valid), "sha256": valid_sha256},
            },
        }
        (self.sft_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        reference = self.root / "validation_subset_2.jsonl"
        selected = [valid[index] for index in reference_indices]
        reference_sha256 = _write_jsonl(reference, selected)
        reference_manifest = {
            "schema_version": "fixed-validation-subset-v1",
            "status": "completed",
            "split": "valid",
            "date": "2026-07-13",
            "source_file": str((self.sft_dir / "valid.jsonl").resolve()),
            "source_rows": len(valid),
            "source_sha256": valid_sha256,
            "subset_size": 2,
            "output_rows": 2,
            "output_sha256": reference_sha256,
            "selection_method": "lowest_sample_id_lexicographic",
            "output_order": "source_row_ascending",
        }
        reference_manifest_path = self.root / "validation_subset_2_manifest.json"
        reference_manifest_path.write_text(
            json.dumps(reference_manifest), encoding="utf-8"
        )
        return reference, self.root / "frozen"

    def test_freeze_is_deterministic_and_extracts_minimal_contract(self) -> None:
        train = [
            _sample("t1", "to1", "ts1", "训练一", "p1", "train"),
            _sample("t2", "to2", "ts2", "训练二", "p2", "train"),
        ]
        valid = [
            _sample("v1", "vo1", "vs1", "北京南站", "p3", "valid"),
            _sample("v2", "vo2", "vs2", "百子湾东里", "p4", "valid"),
            _sample("v3", "vo3", "vs3", "合生汇", "p5", "valid"),
        ]
        reference, output_dir = self._prepare(
            train=train,
            valid=valid,
            reference_indices=[2, 0],
        )
        first = freeze_embedding_eval_set(
            self.sft_dir,
            reference,
            output_dir,
            expected_rows=2,
            expected_train_rows=2,
            expected_valid_rows=3,
        )
        rows = [
            json.loads(line)
            for line in first.data_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            list(rows[0]),
            ["sample_id", "order_id", "searchid", "query", "poi_id", "split"],
        )
        self.assertEqual([row["sample_id"] for row in rows], ["v3", "v1"])
        self.assertEqual([row["query"] for row in rows], ["合生汇", "北京南站"])
        self.assertTrue(first.manifest["leakage_check"]["all_zero"])

        rebuilt = freeze_embedding_eval_set(
            self.sft_dir,
            reference,
            output_dir,
            expected_rows=2,
            expected_train_rows=2,
            expected_valid_rows=3,
        )
        self.assertEqual(first.sha256, rebuilt.sha256)
        self.assertEqual(first.manifest["built_at"], rebuilt.manifest["built_at"])

    def test_training_business_key_overlap_is_rejected(self) -> None:
        train = [
            _sample("t1", "vo1", "ts1", "训练重叠", "p1", "train"),
            _sample("t2", "to2", "ts2", "训练二", "p2", "train"),
        ]
        valid = [
            _sample("v1", "vo1", "vs1", "评测一", "p3", "valid"),
            _sample("v2", "vo2", "vs2", "评测二", "p4", "valid"),
        ]
        reference, output_dir = self._prepare(
            train=train,
            valid=valid,
            reference_indices=[0, 1],
        )
        with self.assertRaisesRegex(EmbeddingEvalSetError, "训练集存在业务键重叠"):
            freeze_embedding_eval_set(
                self.sft_dir,
                reference,
                output_dir,
                expected_rows=2,
                expected_train_rows=2,
                expected_valid_rows=2,
            )


if __name__ == "__main__":
    unittest.main()

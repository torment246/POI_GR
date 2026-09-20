from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from qg_prqk.sft.full_test_data import (
    VARIANTS, JsonlRecordSequence, SftEvaluationError, file_stat, scan_paired_test,
)


def record(index: int, variant: str) -> dict:
    return dict(order_id=f"o{index}", searchid=f"s{index}", target_poi_id=f"p{index}",
                sample_id=f"sample{index}", history_length=0, user_token="u", split="test",
                identifier_variant=variant, requires_dedup=False,
                messages=[dict(role="user", content=f"<CURRENT>合成地点{index}</CURRENT>"),
                          dict(role="assistant", content="synthetic")])


class FullTestDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sources = {}
        for variant in VARIANTS:
            self.save(variant, [record(i, variant) for i in range(3)])

    def save(self, variant: str, rows: list[dict]) -> None:
        path = Path(self.tmp.name) / f"{variant}.jsonl"
        raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        with path.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # OFS publishes final write metadata on the first close-to-open readback.
        self.assertEqual(path.read_bytes(), raw)
        self.sources[variant] = dict(file=str(path), rows=len(rows),
                                    sha256=hashlib.sha256(raw).hexdigest(), **file_stat(path))

    def test_full_pair_and_lazy_slicing_without_copy(self) -> None:
        result = scan_paired_test(self.sources, expected_rows=3)
        self.assertEqual(result["rows"], 3)
        self.assertFalse(result["copy_materialized"])
        source = self.sources[VARIANTS[0]]
        view = JsonlRecordSequence(Path(source["file"]), variant=VARIANTS[0], expected_rows=3,
                                   expected_sha256=source["sha256"])
        self.assertEqual(view.offsets.itemsize, 8)
        self.assertEqual(view[-1]["order_id"], "o2")
        self.assertEqual([r["order_id"] for r in view[1:]], ["o1", "o2"])
        self.assertEqual(len(view[::2]), 2)
        self.assertEqual(view[3:3], [])
        with self.assertRaises(IndexError):
            _ = view[3]

    def test_wrong_split_or_malformed_messages_rejected(self) -> None:
        for patch in (dict(split="valid"), dict(split="train"), dict(messages=[])):
            rows = [record(i, VARIANTS[0]) for i in range(3)]
            rows[1].update(patch)
            self.save(VARIANTS[0], rows)
            with self.assertRaises(SftEvaluationError):
                scan_paired_test(self.sources, expected_rows=3)

    def test_pair_order_target_and_current_mismatch_rejected(self) -> None:
        for patch in (dict(order_id="wrong"), dict(target_poi_id="wrong"),
                      dict(messages=[dict(role="user", content="<CURRENT>变更</CURRENT>"),
                                     dict(role="assistant", content="synthetic")])):
            rows = [record(i, VARIANTS[1]) for i in range(3)]
            rows[1].update(patch)
            self.save(VARIANTS[1], rows)
            with self.assertRaisesRegex(SftEvaluationError, "不一致"):
                scan_paired_test(self.sources, expected_rows=3)

    def test_missing_extra_and_duplicate_rows_rejected(self) -> None:
        for indices in ((0, 1), (0, 1, 2, 3), (0, 1, 1)):
            for variant in VARIANTS:
                self.save(variant, [record(i, variant) for i in indices])
            with self.assertRaises(SftEvaluationError):
                scan_paired_test(self.sources, expected_rows=3)

    def test_hash_mismatch_rejected_by_parent_and_worker(self) -> None:
        self.sources[VARIANTS[0]]["sha256"] = "wrong"
        with self.assertRaisesRegex(SftEvaluationError, "SHA256"):
            scan_paired_test(self.sources, expected_rows=3)
        with self.assertRaisesRegex(SftEvaluationError, "SHA256"):
            JsonlRecordSequence(Path(self.sources[VARIANTS[0]]["file"]), variant=VARIANTS[0],
                                expected_rows=3, expected_sha256="wrong")


if __name__ == "__main__":
    unittest.main()

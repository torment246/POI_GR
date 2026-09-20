from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qg_prqk.artifacts import sha256_file
from qg_prqk.sft.evaluation_data import SftEvaluationError, align_subsets, keys_sha256


def record(index: int) -> dict:
    return dict(order_id=f"order{index}", searchid=f"search{index}", target_poi_id=f"poi{index}",
                sample_id=f"sample{index}", split="valid", identifier_variant="a4_nogid",
                user_token="<USER_0>", history_length=0, requires_dedup=False,
                messages=[dict(role="user", content="<CURRENT><QUERY>合成地点</QUERY></CURRENT>"),
                          dict(role="assistant", content="<TARGET_POI><S1_0><S2_0><S3_0></TARGET_POI>")])


class EvaluationDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "valid.jsonl"
        self.records = [record(i) for i in range(5)]
        self.source.write_text("".join(json.dumps(r) + "\n" for r in self.records))
        self.references = {name: dict(path="synthetic", sha256="synthetic", records=[self.records[i]],
                                     business_keys_sha256=keys_sha256([(f"order{i}", f"search{i}")]))
                           for name, i in (("fixed10k", 3), ("cold_target", 1))}

    def run_alignment(self) -> dict:
        return align_subsets(self.source, self.references, self.root / "aligned", variant="a4_nogid",
                             source_rows=5, source_sha256=sha256_file(self.source), source_manifest_sha256="synthetic")

    def test_reference_order_and_cache_are_stable(self) -> None:
        result = self.run_alignment()
        self.assertEqual(result, self.run_alignment())
        self.assertEqual(json.loads((self.root / "aligned/fixed10k.jsonl").read_text())["order_id"], "order3")

    def test_target_or_current_context_mismatch_fails(self) -> None:
        self.references["fixed10k"]["records"] = [dict(self.records[3], target_poi_id="wrong")]
        with self.assertRaisesRegex(SftEvaluationError, "目标 POI"):
            self.run_alignment()

    def test_optional_legacy_user_token_is_not_a_required_join_field(self) -> None:
        self.references["fixed10k"]["records"] = [dict(self.records[3])]
        self.references["fixed10k"]["records"][0].pop("user_token")
        self.assertEqual(self.run_alignment()["outputs"]["fixed10k"]["rows"], 1)

    def test_changed_current_query_is_rejected(self) -> None:
        changed = dict(self.records[3], messages=[dict(role="user", content="<CURRENT>不同请求</CURRENT>")])
        self.references["fixed10k"]["records"] = [changed]
        with self.assertRaisesRegex(SftEvaluationError, "Query"):
            self.run_alignment()

    def test_test_split_is_rejected(self) -> None:
        self.source.write_text(self.source.read_text().replace('"valid"', '"test"'))
        with self.assertRaises(SftEvaluationError):
            self.run_alignment()

    def test_hash_mismatch_or_partial_source_fails(self) -> None:
        with self.assertRaisesRegex(SftEvaluationError, "SHA256"):
            align_subsets(self.source, self.references, self.root / "aligned", variant="a4_nogid",
                          source_rows=5, source_sha256="wrong", source_manifest_sha256="synthetic")

    def test_cached_subset_tampering_fails(self) -> None:
        self.run_alignment()
        (self.root / "aligned/fixed10k.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(SftEvaluationError, "SHA256"):
            self.run_alignment()

    def test_missing_or_duplicate_reference_is_not_resampled(self) -> None:
        self.references["fixed10k"]["records"] = [record(100)]
        with self.assertRaisesRegex(SftEvaluationError, "缺少"):
            self.run_alignment()


if __name__ == "__main__":
    unittest.main()

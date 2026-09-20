"""Synthetic checks for active GenPOI's paired user/hash/history contract."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from poi_gr.methods.genpoi.aligned_data import aligned_special_tokens, build_aligned_sft_data, transform_record
from poi_gr.methods.genpoi.data import GenpoiDataError
from poi_gr.sft.data import PidLookup
from poi_gr.methods.tiger.data import TigerIdLookup
from poi_gr.pid.dedup import sha256_file

SOURCE_ID = "<S1_1><S2_2><S3_3><C_0>"
GenPOI_ID = "<G_w><G_x><G_4><G_g><G_0><G_0><S1_9><S2_8><S3_7>"


def record(split="train", history=True):
    gid = "<G_w><G_x><G_4><G_g><G_0><G_0>"
    event = f"<EVENT>\n<USER_GID>{gid}</USER_GID>\n<QUERY>历史 A  B</QUERY>\n<POI_TIGER_ID>{SOURCE_ID}</POI_TIGER_ID>\n</EVENT>\n" if history else ""
    return {"split": split, "order_id": "synthetic-order", "searchid": "synthetic-search",
            "target_poi_id": "100", "target_tiger_id_key": "1-2-3|c0", "user_token": "<U_0199>",
            "history_length": int(history), "messages": [
                {"role": "user", "content": f"<USER_ID><U_0199></USER_ID>\n<HISTORY>\n{event}</HISTORY>\n<CURRENT>\n<USER_GID>{gid}</USER_GID>\n<QUERY> 当前 原样 </QUERY>\n</CURRENT>"},
                {"role": "assistant", "content": f"<TARGET_POI>{SOURCE_ID}</TARGET_POI>"}]}


class AlignedDataTest(unittest.TestCase):
    def test_only_identifier_changes_for_empty_and_nonempty_history(self):
        lookup = {SOURCE_ID: ("100", GenPOI_ID, "9-8-7")}
        for history in (True, False):
            source = record(history=history)
            frozen = copy.deepcopy(source)
            output = transform_record(source, "train", lookup)
            self.assertEqual(source, frozen)
            expected = source["messages"][0]["content"].replace(SOURCE_ID, GenPOI_ID).replace("POI_TIGER_ID", "POI_PID")
            self.assertEqual(output["messages"][0]["content"], expected)
            self.assertEqual(output["user_token"], "<U_0199>")
            self.assertEqual(output["target_pid_key"], "9-8-7")

    def test_bad_uid_target_history_or_split_fail_closed(self):
        lookup = {SOURCE_ID: ("100", GenPOI_ID, "9-8-7")}
        for key, value in (("user_token", "<U_0001>"), ("target_poi_id", "200"),
                           ("history_length", 2), ("split", "test")):
            source = record()
            source[key] = value
            with self.assertRaises(GenpoiDataError):
                transform_record(source, "train", lookup)

    def test_full_streaming_builder_checks_hashes_and_preserves_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            outputs = {}
            for split in ("train", "valid", "test"):
                path = source / f"{split}.jsonl"
                path.write_text(json.dumps(record(split), ensure_ascii=False) + "\n")
                outputs[path.name] = {"rows": 1, "sha256": sha256_file(path)}
            token_path = source / "special_tokens.json"
            token_path.write_text(json.dumps({"user_tokens": [f"<U_{i:04d}>" for i in range(2000)]}))
            outputs[token_path.name] = {"sha256": sha256_file(token_path)}
            manifest = {"schema_version": "tiger-map-search-sft-data-v1", "status": "completed",
                        "tiger_identifier": {"mapping_sha256": "tmap", "manifest_sha256": "tmanifest"},
                        "outputs": outputs, "time_split": {"frozen": True}, "history": {"max_events": 10},
                        "user_identifier": {"bucket_count": 2000, "hash": "sha256"}}
            (source / "manifest.json").write_text(json.dumps(manifest))
            tiger = TigerIdLookup({"100": 0}, np.array([[1, 2, 3, 0]], dtype=np.int32), 1, (512, 512, 512, 1))
            pid = PidLookup({"100": 0}, np.array([[28,29,4,15,0,0,9,8,7]], dtype=np.int32), np.array([-1]), np.array([False]), 1)
            with patch("poi_gr.methods.genpoi.aligned_data.load_tiger_id_lookup", return_value=(tiger, {}, "tmap", "tmanifest")), \
                 patch("poi_gr.methods.genpoi.aligned_data.load_pid_lookup", return_value=(pid, {"schema_version": "dedup-pid-v1"}, "gmap", "gmanifest")):
                result = build_aligned_sft_data(source_dir=source, tiger_id_dir=root,
                    pid_dir=root, output_dir=root / "output", progress=lambda _: None)
                self.assertEqual(result["stats"]["retained_sample_count"], 3)
                self.assertEqual(result["user_identifier"], manifest["user_identifier"])
                tokens = json.loads((root / "output/special_tokens.json").read_text())
                self.assertEqual(len(tokens["user_tokens"]), 2000)
                self.assertEqual(len(tokens["additional_special_tokens"]), len(set(tokens["additional_special_tokens"])))
                self.assertNotIn("<d_-1>", tokens["additional_special_tokens"])
                with self.assertRaisesRegex(GenpoiDataError, "拒绝覆盖"):
                    build_aligned_sft_data(source_dir=source, tiger_id_dir=root,
                        pid_dir=root, output_dir=root / "output")
                outputs["train.jsonl"]["sha256"] = "wrong"
                (source / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(GenpoiDataError, "SHA256"):
                    build_aligned_sft_data(source_dir=source, tiger_id_dir=root,
                        pid_dir=root, output_dir=root / "bad-output", progress=lambda _: None)
                self.assertFalse((root / "bad-output").exists())

    def test_missing_user_vocab_is_not_silently_dropped(self):
        with self.assertRaises(GenpoiDataError):
            aligned_special_tokens(512, {"user_tokens": []})


if __name__ == "__main__":
    unittest.main()

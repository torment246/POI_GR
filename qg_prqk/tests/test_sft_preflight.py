from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from qg_prqk.sft.preflight import (
    SftPreflightError,
    scan_train_valid_lengths,
)


class _Slot:
    def __init__(self, prefix: bool = False) -> None:
        self.prefix = prefix

    def apply(self, content: str = "", idx: str = "") -> list[str]:
        del idx
        return [] if self.prefix else [content]


class _Template:
    format_prefix = _Slot(prefix=True)
    format_user = _Slot()
    format_assistant = _Slot()


class _Tokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, list[int]]:
        return {"length": [len(text) for text in texts]}


def _write_split(root: Path, split: str, target: str) -> dict[str, object]:
    path = root / f"{split}.jsonl"
    record = {
        "split": split,
        "identifier_variant": "a4_nogid",
        "messages": [
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": target},
        ],
    }
    raw = (json.dumps(record) + "\n").encode()
    path.write_bytes(raw)
    return {"rows": 1, "sha256": hashlib.sha256(raw).hexdigest()}


class SftPreflightTest(unittest.TestCase):
    def test_scans_only_train_valid_and_requires_nogid_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {
                "train.jsonl": _write_split(
                    root, "train", "<TARGET_POI><S1_1><S2_2><S3_3></TARGET_POI>"
                ),
                "valid.jsonl": _write_split(
                    root, "valid", "<TARGET_POI><S1_4><S2_5><S3_6><D_0></TARGET_POI>"
                ),
            }
            (root / "test.jsonl").write_text("not read\n", encoding="utf-8")
            stats = scan_train_valid_lengths(
                data_dir=root,
                variant="a4_nogid",
                manifest={"outputs": outputs},
                tokenizer=_Tokenizer(),
                template=_Template(),
                cutoff_len=1024,
                batch_size=2,
            )
            self.assertEqual(stats["row_count"], 2)
            self.assertEqual(stats["over_cutoff_count"], 0)
            self.assertEqual(stats["target_truncated_count"], 0)

    def test_rejects_gid_in_nogid_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {
                split + ".jsonl": _write_split(
                    root,
                    split,
                    "<TARGET_POI><G_w><S1_1><S2_2><S3_3></TARGET_POI>",
                )
                for split in ("train", "valid")
            }
            with self.assertRaises(SftPreflightError):
                scan_train_valid_lengths(
                    data_dir=root,
                    variant="a4_nogid",
                    manifest={"outputs": outputs},
                    tokenizer=_Tokenizer(),
                    template=_Template(),
                    cutoff_len=1024,
                    batch_size=2,
                )


if __name__ == "__main__":
    unittest.main()

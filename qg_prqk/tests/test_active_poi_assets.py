from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qg_prqk.data.active_poi import (
    ActivePoiAssetsError,
    _copy_active_arrays_sequentially,
    resolve_active_source_rows,
)


def _write_ids(path: Path, values: list[str]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


class ActivePoiAssetsTest(unittest.TestCase):
    def test_writes_standard_npy_in_active_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "active.npy"
            source_embeddings = np.arange(24, dtype=np.float16).reshape(6, 4)
            source_categories = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32)
            source_rows = np.asarray([0, 2, 5], dtype=np.int64)
            active_categories = np.empty(3, dtype=np.int32)
            _copy_active_arrays_sequentially(
                source_embeddings=source_embeddings,
                source_categories=source_categories,
                source_rows=source_rows,
                embeddings_path=output_path,
                active_categories=active_categories,
                chunk_rows=2,
            )
            np.testing.assert_array_equal(
                np.load(output_path), source_embeddings[source_rows]
            )
            np.testing.assert_array_equal(active_categories, [0, 2, 5])

    def test_resolves_strict_subsequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            active = root / "active.jsonl"
            _write_ids(source, ["p0", "p1", "p2", "p3", "p4"])
            _write_ids(active, ["p0", "p2", "p4"])
            rows = resolve_active_source_rows(
                source,
                active,
                source_rows=5,
                active_rows=3,
            )
            np.testing.assert_array_equal(rows, np.asarray([0, 2, 4]))

    def test_rejects_changed_active_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            active = root / "active.jsonl"
            _write_ids(source, ["p0", "p1", "p2"])
            _write_ids(active, ["p2", "p1"])
            with self.assertRaisesRegex(ActivePoiAssetsError, "顺序错误"):
                resolve_active_source_rows(
                    source,
                    active,
                    source_rows=3,
                    active_rows=2,
                )


if __name__ == "__main__":
    unittest.main()

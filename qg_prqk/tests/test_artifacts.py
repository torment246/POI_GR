from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qg_prqk.artifacts import (
    QGPRQKArtifactError,
    build_manifest_template,
    write_json_atomic,
)
from qg_prqk.config import load_config


QG_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(QG_ROOT / "configs/qg_prqk_1024x3.yaml")


class ArtifactTest(unittest.TestCase):
    def test_manifest_is_planned_not_completed(self) -> None:
        manifest = build_manifest_template(CONFIG, sample_limit=2)
        self.assertEqual(manifest["status"], "planned")
        self.assertEqual(manifest["identifier"]["codebook_sizes"], [1024] * 3)
        self.assertEqual(manifest["artifacts"], {})

    def test_atomic_write_preserves_default_overwrite_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_json_atomic(path, {"status": "planned"})
            self.assertEqual(json.loads(path.read_text())["status"], "planned")
            with self.assertRaisesRegex(QGPRQKArtifactError, "overwrite=false"):
                write_json_atomic(path, {"status": "changed"})
            write_json_atomic(path, {"status": "changed"}, overwrite=True)
            self.assertEqual(json.loads(path.read_text())["status"], "changed")


if __name__ == "__main__":
    unittest.main()

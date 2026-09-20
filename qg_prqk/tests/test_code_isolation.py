from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

import yaml


QG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = QG_ROOT.parent


def source_files() -> list[Path]:
    # Runtime outputs may contain immutable source snapshots from old attempts.
    return sorted(path for name in ("src", "scripts", "tests")
                  for path in (QG_ROOT / name).rglob("*.py"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CodeIsolationTest(unittest.TestCase):
    def test_python_imports_do_not_use_project_modules(self) -> None:
        violations: list[str] = []
        for path in source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if any(module == "poi_gr" or module.startswith("poi_gr.") for module in modules):
                    violations.append(f"{path}: {modules}")
        self.assertEqual(violations, [])

    def test_python_does_not_call_root_project_scripts(self) -> None:
        violations: list[str] = []
        forbidden_fragments = (
            "src/poi_gr/",
            str(PROJECT_ROOT / "src/poi_gr"),
            str(PROJECT_ROOT / "scripts"),
        )
        for path in source_files():
            if path == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            for fragment in forbidden_fragments:
                if fragment in text:
                    violations.append(f"{path}: {fragment}")
        self.assertEqual(violations, [])

    def test_provenance_sources_and_targets_match_hashes(self) -> None:
        path = QG_ROOT / "configs/code_provenance.yaml"
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"], "qg-prqk-code-provenance-v1"
        )
        self.assertTrue(manifest["entries"])
        for entry in manifest["entries"]:
            source = PROJECT_ROOT / entry["source"]
            target = PROJECT_ROOT / entry["target"]
            self.assertTrue(source.is_file(), source)
            self.assertTrue(target.is_file(), target)
            self.assertEqual(sha256(source), entry["source_sha256"])
            self.assertEqual(sha256(target), entry["target_sha256"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.data.query_graph_data import QueryGraphDataError, prepare_query_graph_inputs, read_d3_raw, validate_limit
from qg_prqk.data.query_graph import (
    build_query_graph,
    load_sample_prefix,
    output_directory,
    validate_query_graph,
)
from test_query_graph import FakeEncoder, build_fixture


class QueryGraphScalingTest(unittest.TestCase):
    def setUp(self) -> None:
        parent = Path(__file__).resolve().parents[1] / "outputs/tmp/p4t"
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = build_fixture(self.root)
        self.encoder = FakeEncoder()
        self.loaders = {
            "encoder_loader": lambda c: (self.encoder, "synthetic", 0),
            "adapter_loader": lambda c: lambda v: np.roll(v, 1, axis=1),
        }

    def sample(self, rows: int = 1) -> tuple[Path, str]:
        build_query_graph(
            self.config, prepare_query_graph_inputs(self.config, limit=rows), **self.loaders
        )
        path = output_directory(self.config, rows) / "manifest.json"
        return path, sha256_file(path)

    def test_medium_reuses_prefix_without_overwriting_sample(self) -> None:
        path, digest = self.sample()
        inputs = prepare_query_graph_inputs(self.config, limit=3, gate="medium")
        result = build_query_graph(
            self.config,
            inputs,
            gate="medium",
            reuse_sample=path,
            reuse_sample_sha256=digest,
            **self.loaders,
        )
        self.assertEqual(result["next_status"], "HOLD_FOR_P4_MEDIUM_REVIEW")
        self.assertEqual(self.encoder.queries, ["query3", "query8"])
        manifest = json.loads(
            (output_directory(self.config, 3, "medium") / "manifest.json").read_text()
        )
        self.assertEqual(manifest["runtime"]["newly_encoded_d1_d2_rows"], 1)
        self.assertEqual(manifest["runtime"]["reused_prefix_rows"], 1)
        self.assertFalse(manifest["is_full"])
        self.assertEqual(sha256_file(path), digest)
        self.assertEqual(
            validate_query_graph(self.config, inputs, gate="medium")["status"], "validated"
        )

        def forbidden(*args):
            self.fail("完成 medium resume 不能加载模型")

        build_query_graph(
            self.config,
            inputs,
            gate="medium",
            reuse_sample=path,
            reuse_sample_sha256=digest,
            resume=True,
            encoder_loader=forbidden,
            adapter_loader=forbidden,
        )

    def test_interrupted_medium_skips_committed_prefix_and_new_raw(self) -> None:
        path, digest = self.sample()
        inputs = prepare_query_graph_inputs(self.config, limit=3, gate="medium")

        def interrupted(config):
            raise RuntimeError("synthetic interruption")

        with self.assertRaisesRegex(RuntimeError, "interruption"):
            build_query_graph(
                self.config,
                inputs,
                gate="medium",
                reuse_sample=path,
                reuse_sample_sha256=digest,
                encoder_loader=self.loaders["encoder_loader"],
                adapter_loader=interrupted,
            )
        self.assertEqual(self.encoder.queries, ["query3", "query8"])
        directory = output_directory(self.config, 3, "medium")
        self.assertFalse((directory / "_SUCCESS").exists())
        build_query_graph(
            self.config,
            inputs,
            gate="medium",
            reuse_sample=path,
            reuse_sample_sha256=digest,
            resume=True,
            **self.loaders,
        )
        self.assertEqual(self.encoder.queries, ["query3", "query8"])
        self.assertEqual(sha256_file(path), digest)

    def test_rejects_missing_reuse_and_wrong_pinned_hash(self) -> None:
        path, digest = self.sample()
        inputs = prepare_query_graph_inputs(self.config, limit=3, gate="medium")
        with self.assertRaisesRegex(QueryGraphDataError, "sample"):
            build_query_graph(self.config, inputs, gate="medium")
        with self.assertRaisesRegex(QueryGraphDataError, "SHA256"):
            build_query_graph(
                self.config,
                inputs,
                gate="medium",
                reuse_sample=path,
                reuse_sample_sha256="0" * 64,
                **self.loaders,
            )
        self.assertEqual(sha256_file(path), digest)
        self.assertFalse(output_directory(self.config, 3, "medium").exists())

    def test_rejects_damaged_prefix_and_nonprefix_nodes(self) -> None:
        path, digest = self.sample()
        inputs = prepare_query_graph_inputs(self.config, limit=3, gate="medium")
        from dataclasses import replace

        bad = replace(inputs, nodes=inputs.nodes.slice(1))
        with self.assertRaisesRegex(QueryGraphDataError, "前缀"):
            load_sample_prefix(self.config, bad, read_d3_raw(inputs), path, digest)
        raw = path.parent / "raw_query_embeddings.npy"
        raw.write_bytes(raw.read_bytes() + b"damage")
        with self.assertRaisesRegex(QueryGraphDataError, "哈希"):
            load_sample_prefix(self.config, inputs, read_d3_raw(inputs), path, digest)

    def test_v1_sample_contract_is_accepted_without_ignoring_pinned_hash(self) -> None:
        path, _ = self.sample()
        manifest = json.loads(path.read_text())
        manifest["schema_version"] = "qg-prqk-p4-query-graph-sample-v1"
        manifest["contract"]["schema_version"] = manifest["schema_version"]
        del manifest["contract"]["gate"]
        del manifest["contract"]["reused_prefix"]
        write_json_atomic(path, manifest, overwrite=True)
        progress = path.parent / "progress.json"
        state = json.loads(progress.read_text())
        state["contract"] = manifest["contract"]
        write_json_atomic(progress, state, overwrite=True)
        digest = sha256_file(path)
        write_json_atomic(
            path.parent / "_SUCCESS", {"manifest_sha256": digest}, overwrite=True
        )
        inputs = prepare_query_graph_inputs(self.config, limit=3, gate="medium")
        result = build_query_graph(
            self.config,
            inputs,
            gate="medium",
            reuse_sample=path,
            reuse_sample_sha256=digest,
            **self.loaders,
        )
        self.assertEqual(result["status"], "validated")

    def test_medium_above_1000_runs_bounded_blocks(self) -> None:
        root = self.root / "larger"
        root.mkdir()
        self.config = build_fixture(root, extra_d1=1001, buffer_rows=128)
        path, digest = self.sample(3)
        inputs = prepare_query_graph_inputs(self.config, limit=1004, gate="medium")
        result = build_query_graph(
            self.config,
            inputs,
            gate="medium",
            reuse_sample=path,
            reuse_sample_sha256=digest,
            **self.loaders,
        )
        self.assertEqual(result["metrics"]["query_rows"], 1004)
        self.assertEqual(len(self.encoder.queries), 1003)
        manifest = json.loads(
            (
                output_directory(self.config, 1004, "medium") / "manifest.json"
            ).read_text()
        )
        self.assertEqual(len(manifest["chunks"]), 8)
        self.assertEqual(manifest["chunks"][-1]["stop"], 1004)
        self.assertEqual(manifest["runtime"]["newly_encoded_d1_d2_rows"], 1001)

    def test_limits_and_streamed_signature_keep_v1_bytes(self) -> None:
        for gate, limit in (
            ("full", 3),
            ("medium", 50001),
            ("sample", 1001),
            ("medium", 0),
            ("medium", True),
        ):
            with self.subTest(gate=gate, limit=limit), self.assertRaises(QueryGraphDataError):
                validate_limit(limit, gate)
        validate_limit(50000, "medium")
        validate_limit(342879, "full")
        inputs = prepare_query_graph_inputs(self.config, limit=3)
        payload = {
            k: getattr(inputs, k).to_pylist()
            for k in ("nodes", "edges", "false_negatives")
        }
        payload.update(
            source_files=dict(inputs.source_files),
            poi_rows=inputs.poi_rows,
            embedding_dim=inputs.embedding_dim,
        )
        expected = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False, allow_nan=False
            ).encode()
        ).hexdigest()
        self.assertEqual(inputs.signature(), expected)
        self.assertEqual(inputs.streaming_signature(), inputs.streaming_signature())

    def test_full_reuses_pinned_medium_and_only_processes_new_rows(self) -> None:
        self.config = build_fixture(self.root / "full", extra_d1=1)
        sample_inputs = prepare_query_graph_inputs(self.config, limit=1)
        build_query_graph(self.config, sample_inputs, **self.loaders)
        sample_path = output_directory(self.config, 1) / "manifest.json"
        sample_sha = sha256_file(sample_path)
        medium_inputs = prepare_query_graph_inputs(self.config, limit=2, gate="medium")
        build_query_graph(
            self.config,
            medium_inputs,
            gate="medium",
            reuse_sample=sample_path,
            reuse_sample_sha256=sample_sha,
            **self.loaders,
        )
        medium_path = output_directory(self.config, 2, "medium") / "manifest.json"
        medium_sha = sha256_file(medium_path)

        with (
            patch("qg_prqk.data.query_graph_data.FULL_LIMIT", 4),
            patch("qg_prqk.data.query_graph.MEDIUM_LIMIT", 2),
        ):
            inputs = prepare_query_graph_inputs(self.config, limit=4, gate="full")
            result = build_query_graph(
                self.config,
                inputs,
                gate="full",
                reuse_medium=medium_path,
                reuse_medium_sha256=medium_sha,
                **self.loaders,
            )
            directory = output_directory(self.config, 4, "full")
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(result["next_status"], "HOLD_FOR_P4_FULL_REVIEW")
            self.assertTrue(manifest["is_full"])
            self.assertFalse(manifest["is_sample"])
            self.assertEqual(manifest["runtime"]["reused_prefix_rows"], 2)
            self.assertEqual(manifest["runtime"]["newly_encoded_d1_d2_rows"], 1)
            self.assertEqual(self.encoder.queries, ["query3", "query8", "query21"])
            self.assertEqual(
                validate_query_graph(self.config, inputs, gate="full")["status"],
                "validated",
            )
            before = sha256_file(medium_path)

            def forbidden(*args):
                self.fail("完成 full resume 不得加载模型")

            build_query_graph(
                self.config,
                inputs,
                gate="full",
                reuse_medium=medium_path,
                reuse_medium_sha256=medium_sha,
                resume=True,
                encoder_loader=forbidden,
                adapter_loader=forbidden,
            )
            self.assertEqual(sha256_file(medium_path), before)


if __name__ == "__main__":
    unittest.main()

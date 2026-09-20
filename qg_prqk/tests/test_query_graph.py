from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.data.query_graph_data import QueryGraphDataError, prepare_query_graph_inputs, read_d3_raw
from qg_prqk.data.query_graph import (
    build_query_graph,
    output_directory,
    validate_final_payload,
    validate_query_graph,
)
from test_query_graph_data import synthetic_graph


class FakeEncoder:
    def __init__(self) -> None:
        self.queries = []

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, sentences, **kwargs):
        self.queries.extend(sentences)
        return np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(sentences), 1))


@dataclass
class SyntheticAdapterConfig:
    bottleneck: int = 2


def build_fixture(root: Path, *, extra_d1: int = 0, buffer_rows: int = 2):
    depths, edges, masks, pois, _ = synthetic_graph()
    template_edges = [e for e in edges if e["query_id"] == 3]
    for qid in range(21, 21 + extra_d1):
        depths.append(dict(depths[0], query_id=qid, normalized_query=f"query{qid}"))
        edges.extend(
            dict(e, query_id=qid, normalized_query=f"query{qid}")
            for e in template_edges
        )
    by_id = {row["query_id"]: row for row in depths}
    all_depths = []
    for qid in range(21 + extra_d1):
        row = copy.deepcopy(by_id.get(qid, depths[0]))
        row["query_id"] = qid
        if qid not in by_id:
            row["supervision_depth"] = 0
        all_depths.append(row)
    p25_root, p2_root, full_root, cache_root = [
        root / name for name in ("p25", "p2", "full", "cache")
    ]

    def group(parent, name, rows):
        directory = parent / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "part-00000.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        return {
            "dir": name,
            "rows": len(rows),
            "files": [
                {
                    "file": path.name,
                    "rows": len(rows),
                    "shard_id": 0,
                    "sha256": sha256_file(path),
                },
            ],
        }

    p2 = {
        "status": "completed",
        "source": {"split": "train", "valid_and_test_read": False},
        "outputs": {
            "false_negative_mask": group(p2_root, "false_negative_mask", masks)
        },
    }
    write_json_atomic(p2_root / "manifest.json", p2)
    p2_sha = sha256_file(p2_root / "manifest.json")
    p25 = {
        "status": "completed",
        "source_access": {
            "raw_train_read": False,
            "validation_read": False,
            "test_read": False,
        },
        "inputs": {"p2_manifest": {"sha256": p2_sha}},
        "outputs": {
            "query_category_depth": group(p25_root, "query_category_depth", all_depths),
            "query_poi_layer_edges": group(p25_root, "query_poi_layer_edges", edges),
            "category_mapping": group(
                p25_root,
                "category_mapping",
                [dict(v, poi_id=k) for k, v in pois.items()],
            ),
        },
    }
    write_json_atomic(p25_root / "manifest.json", p25)
    full_root.mkdir()
    selection = [
        {
            "gate_row": 0,
            "query_id": 100,
            "normalized_query": "unused",
            "target_poi_id": "p1",
        },
        {
            "gate_row": 1,
            "query_id": 20,
            "normalized_query": "query20",
            "target_poi_id": "p0",
        },
    ]
    selection_path = full_root / "d3_full_selection.parquet"
    pq.write_table(pa.Table.from_pylist(selection), selection_path)
    selection_sha = sha256_file(selection_path)
    write_json_atomic(
        full_root / "manifest.json",
        {
            "status": "completed",
            "artifacts": {
                "d3_full_selection.parquet": {"sha256": selection_sha},
            },
        },
    )
    cache_root.mkdir()
    values = np.array([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float16)
    np.save(cache_root / "chunk.npy", values)
    query_hash = hashlib.sha256()
    for row in selection:
        query_hash.update(
            json.dumps(
                [row["query_id"], row["normalized_query"]], ensure_ascii=False
            ).encode()
            + b"\n"
        )
    cache = {
        "status": "completed",
        "contract": {
            "schema_version": "qg-prqk-d3-query-chunk-cache-v1",
            "config_signature": "full-test",
            "rows": 2,
            "embedding_dim": 4,
            "dtype": "float16",
            "chunk_rows": buffer_rows,
            "query_id_text_sha256": query_hash.hexdigest(),
        },
        "chunks": [
            {
                "file": "chunk.npy",
                "start": 0,
                "stop": 2,
                "sha256": sha256_file(cache_root / "chunk.npy"),
            }
        ],
    }
    write_json_atomic(cache_root / "manifest.json", cache)
    checkpoint = full_root / "final.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint-not-torch")
    policy = {"D1": "raw_bge", "D2": "raw_bge", "D3": "final_adapter"}
    final = {
        "status": "completed",
        "schema_version": "qg-prqk-p3a-full-final-v1",
        "config_signature": "full-test",
        "query_view_policy": policy,
        "training": {"training_rows": 2, "fixed_epochs": 2},
        "artifacts": {
            "query_adapter_exact_final.pt": {"sha256": sha256_file(checkpoint)}
        },
        "source_hashes": {
            "full_d3_selection_sha256": selection_sha,
            "initial_state_sha256": "synthetic-init",
        },
    }
    write_json_atomic(full_root / "final_manifest.json", final)
    upstream_artifacts = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in (
            ("p2_5_manifest", p25_root / "manifest.json"),
            ("p3a_full_manifest", full_root / "manifest.json"),
            ("final_adapter_manifest", full_root / "final_manifest.json"),
            ("final_adapter_checkpoint", checkpoint),
            ("d3_query_cache_manifest", cache_root / "manifest.json"),
        )
    }
    base = SimpleNamespace(
        signature=lambda: "base-test",
        adapter=SyntheticAdapterConfig(),
        query_embedding=SimpleNamespace(
            batch_size=2, encode_buffer_size=buffer_rows, prompt_name=None
        ),
        category_config=SimpleNamespace(
            paths=SimpleNamespace(p2_query_stats=p2_root),
            frozen=SimpleNamespace(
                p2_manifest_sha256=p2_sha, poi_rows=2, poi_embedding_dim=4
            ),
        ),
    )
    config = SimpleNamespace(
        query_depth_dir=p25_root,
        upstream_artifacts=upstream_artifacts,
        query_view_policy=policy,
        upstream=SimpleNamespace(
            base=base, output_dir=full_root, d3_rows=2, signature=lambda: "full-test"
        ),
        output_dir=root / "qg_prqk/outputs/qg_prqk_512x3_test",
        codebook_sizes=(512, 512, 512),
        signature=lambda: "config-test",
        resolved_payload=lambda: {"synthetic": True},
    )
    return config


class QueryGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        temp_parent = Path(__file__).resolve().parents[1] / "outputs/tmp/p4t"
        temp_parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temp_parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = build_fixture(self.root)
        self.encoder = FakeEncoder()

    def encoder_loader(self, config):
        return self.encoder, "synthetic", 0

    def adapter_loader(self, config):
        return lambda values: np.roll(values, 1, axis=1)

    def test_real_contract_fixture_build_validate_and_idempotent_resume(self) -> None:
        inputs = prepare_query_graph_inputs(self.config, limit=3)
        self.assertEqual(inputs.nodes["query_id"].to_pylist(), [3, 8, 20])
        self.assertFalse(self.config.output_dir.exists())
        result = build_query_graph(
            self.config,
            inputs,
            encoder_loader=self.encoder_loader,
            adapter_loader=self.adapter_loader,
        )
        self.assertEqual(result["status"], "validated")
        self.assertEqual(self.encoder.queries, ["query3", "query8"])
        self.assertEqual(result["metrics"]["depth_counts"], {"D1": 1, "D2": 1, "D3": 1})
        saved = output_directory(self.config, 3) / "query_embeddings.npy"
        before = sha256_file(saved)

        def forbidden(*args):
            self.fail("完成 resume 不得加载模型")

        build_query_graph(
            self.config,
            inputs,
            resume=True,
            encoder_loader=forbidden,
            adapter_loader=forbidden,
        )
        self.assertEqual(sha256_file(saved), before)
        self.assertEqual(validate_query_graph(self.config, inputs)["status"], "validated")
        with self.assertRaises(QueryGraphDataError):
            build_query_graph(self.config, inputs)

    def test_resume_skips_committed_raw_encoding_after_interruption(self) -> None:
        inputs = prepare_query_graph_inputs(self.config, limit=3)

        def interrupted(config):
            raise RuntimeError("synthetic interruption")

        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            build_query_graph(
                self.config,
                inputs,
                encoder_loader=self.encoder_loader,
                adapter_loader=interrupted,
            )
        directory = output_directory(self.config, 3)
        self.assertFalse((directory / "_SUCCESS").exists())
        self.assertEqual(
            len(json.loads((directory / "progress.json").read_text())["chunks"]), 1
        )

        def forbidden(config):
            self.fail("已提交 D1/D2 不得重复编码")

        build_query_graph(
            self.config,
            inputs,
            resume=True,
            encoder_loader=forbidden,
            adapter_loader=self.adapter_loader,
        )
        self.assertEqual(self.encoder.queries, ["query3", "query8"])

    def test_rejects_corrupted_source_and_cache_chunks(self) -> None:
        inputs = prepare_query_graph_inputs(self.config, limit=3)
        chunk = inputs.cache_dir / "chunk.npy"
        chunk.write_bytes(chunk.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(QueryGraphDataError, "SHA256"):
            read_d3_raw(inputs)
        source = (
            self.config.query_depth_dir / "query_poi_layer_edges/part-00000.parquet"
        )
        source.write_bytes(source.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(QueryGraphDataError, "SHA256"):
            prepare_query_graph_inputs(self.config, limit=3)

    def test_rejects_corrupted_output_and_changed_resume_contract(self) -> None:
        inputs = prepare_query_graph_inputs(self.config, limit=3)
        build_query_graph(
            self.config,
            inputs,
            encoder_loader=self.encoder_loader,
            adapter_loader=self.adapter_loader,
        )
        path = output_directory(self.config, 3) / "query_embeddings.npy"
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(QueryGraphDataError, "哈希"):
            validate_query_graph(self.config, inputs)
        self.config.signature = lambda: "different"
        with self.assertRaisesRegex(QueryGraphDataError, "签名"):
            build_query_graph(self.config, inputs, resume=True)

    def test_rejects_full_and_changed_d3_text_contract(self) -> None:
        with self.assertRaisesRegex(QueryGraphDataError, "full"):
            prepare_query_graph_inputs(self.config, limit=1001)
        path = Path(self.config.upstream_artifacts["d3_query_cache_manifest"]["path"])
        cache = json.loads(path.read_text())
        cache["contract"]["query_id_text_sha256"] = "0" * 64
        write_json_atomic(path, cache, overwrite=True)
        self.config.upstream_artifacts["d3_query_cache_manifest"]["sha256"] = (
            sha256_file(path)
        )
        with self.assertRaisesRegex(QueryGraphDataError, "文本/行序"):
            prepare_query_graph_inputs(self.config, limit=3)

    def test_only_final_checkpoint_role_and_frozen_training_are_accepted(self) -> None:
        payload = {
            "schema_version": "qg-prqk-query-adapter-v1",
            "role": "P3A-FULL-FINAL",
            "p3a_full_config_signature": "full-test",
            "p3a_config_signature": "base-test",
            "adapter_config": {"bottleneck": 2},
            "embedding_dim": 4,
            "trained_epochs": 2,
            "initial_state_sha256": "synthetic-init",
        }
        validate_final_payload(payload, self.config)
        for key, value in (
            ("role", "P3A-FULL-SELECT"),
            ("role", "Gate"),
            ("trained_epochs", 3),
            ("initial_state_sha256", "other"),
        ):
            with self.subTest(key=key, value=value):
                changed = dict(payload, **{key: value})
                with self.assertRaisesRegex(QueryGraphDataError, "FINAL"):
                    validate_final_payload(changed, self.config)


if __name__ == "__main__":
    unittest.main()

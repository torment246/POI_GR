"""Synthetic A0-GID remapping, resume, config and preparation-boundary checks."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from qg_prqk.artifacts import sha256_file
from qg_prqk.sid.identifiers import identifier_content, identifier_key
from qg_prqk.sft.a0_gid_data import (
    VARIANT,
    WIRE_FORMAT,
    build_a0_identifiers,
    build_a0_messages,
    completed_stage,
    publish_stage,
    remap_record,
)
from qg_prqk.sft.a0_gid_pipeline import (
    prepare,
    project_root,
    resolve_qg,
    stage_lock,
    validate_matched_config,
    verify_cache_receipt,
)


class A0GidSftTest(unittest.TestCase):
    def setUp(self):
        self.old_base = np.array(
            [[0] * 6 + [1, 2, 3], [0] * 6 + [4, 5, 6]], dtype=np.int32
        )
        self.new_base = np.array(
            [[0] * 6 + [7, 8, 9], [0] * 6 + [7, 8, 9]], dtype=np.int32
        )
        self.old_contents = [
            identifier_content(row, -1, variant=WIRE_FORMAT) for row in self.old_base
        ]
        self.new_contents = [
            identifier_content(row, i, variant=WIRE_FORMAT)
            for i, row in enumerate(self.new_base)
        ]
        self.old = SimpleNamespace(
            poi_ids=["a", "b"],
            base_codes=self.old_base,
            dedup_codes=np.array([-1, -1]),
            row_by_poi_id={"a": 0, "b": 1},
            content=lambda row: self.old_contents[row],
        )

    def source(self, split="train"):
        history = "".join(
            f"<EVENT><QUERY>原查询{i}</QUERY><POI_QGPRQK_ID>{value}</POI_QGPRQK_ID></EVENT>"
            for i, value in enumerate(self.old_contents)
        )
        return {
            "sample_id": "sample",
            "order_id": "order",
            "target_poi_id": "a",
            "split": split,
            "history_length": 2,
            "identifier_variant": WIRE_FORMAT,
            "messages": [
                {
                    "role": "user",
                    "content": f"<HISTORY>{history}</HISTORY><CURRENT><QUERY>当前查询</QUERY><USER_GID>wx4g0</USER_GID></CURRENT>",
                },
                {
                    "role": "assistant",
                    "content": f"<TARGET_POI>{self.old_contents[0]}</TARGET_POI>",
                },
            ],
            "target_qg_prqk_id_key": "old",
            "requires_dedup": False,
        }

    def remap(self, source):
        return remap_record(
            source,
            split="train",
            old_row_by_content={v: i for i, v in enumerate(self.old_contents)},
            row_by_poi_id={"a": 0, "b": 1},
            new_content=lambda row: self.new_contents[row],
            new_key=lambda row: f"key{row}",
            requires_dedup=np.ones(2, dtype=bool),
        )

    def test_remaps_every_history_and_target_without_changing_context(self):
        source = self.source()
        original = copy.deepcopy(source)
        result = self.remap(source)
        self.assertEqual(source, original)
        self.assertEqual(result["identifier_variant"], VARIANT)
        self.assertEqual(
            result["messages"][1]["content"],
            f"<TARGET_POI>{self.new_contents[0]}</TARGET_POI>",
        )
        for new in self.new_contents:
            self.assertIn(new, result["messages"][0]["content"])
        self.assertEqual(
            source["messages"][0]["content"].split("<CURRENT>")[1],
            result["messages"][0]["content"].split("<CURRENT>")[1],
        )
        for key in (
            "sample_id",
            "order_id",
            "target_poi_id",
            "split",
            "history_length",
        ):
            self.assertEqual(source[key], result[key])

    def test_target_id_disagreement_fails(self):
        source = self.source()
        source["target_poi_id"] = "b"
        with self.assertRaises(ValueError):
            self.remap(source)

    def test_missing_or_malformed_history_fails(self):
        source = self.source()
        source["history_length"] = 1
        with self.assertRaises(ValueError):
            self.remap(source)
        source = self.source()
        source["messages"][0]["content"] = source["messages"][0]["content"].replace(
            "<S1_1>", "<S1_999>"
        )
        with self.assertRaises(ValueError):
            self.remap(source)

    def test_test_split_and_wrong_variant_fail(self):
        with self.assertRaises(ValueError):
            self.remap(self.source("test"))
        source = self.source()
        source["identifier_variant"] = "a4_nogid"
        with self.assertRaises(ValueError):
            self.remap(source)

    def test_empty_history_omits_dedup_for_singleton(self):
        source = self.source()
        source["history_length"] = 0
        source["messages"][0]["content"] = "<HISTORY></HISTORY><CURRENT>保留</CURRENT>"
        result = self.remap(source)
        self.assertEqual(
            result["messages"][0]["content"], source["messages"][0]["content"]
        )
        content = identifier_content(self.old_base[0], -1, variant=WIRE_FORMAT)
        self.assertNotIn("<D_", content)
        self.assertTrue(self.new_contents[0].endswith("<D_0>"))
        self.assertTrue(self.new_contents[0].startswith("<G_0>" * 6))

    def test_reuse_requires_matching_contract_and_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / "building"
            temporary.mkdir()
            (temporary / "file").write_text("payload")
            output = root / "published"
            publish_stage(temporary, output, {"contract": {"seed": 42}})
            self.assertIsNotNone(completed_stage(output, {"seed": 42}))
            with self.assertRaises(ValueError):
                completed_stage(output, {"seed": 43})
            (output / "file").write_text("modified")
            with self.assertRaises(ValueError):
                completed_stage(output, {"seed": 42})

    def test_same_model_and_training_hyperparameters_as_a4(self):
        root = project_root() / "qg_prqk/configs/sft"
        config = yaml.safe_load((root / "a0_gid_history10_v1.yaml").read_text())
        reference = yaml.safe_load(
            (root / "a4_gid_parent_history10_v1.yaml").read_text()
        )
        validate_matched_config(config, reference)
        config["model_name_or_path"] = "other-model"
        with self.assertRaises(ValueError):
            validate_matched_config(config, reference)

    def test_four_card_and_no_external_output(self):
        root = project_root() / "qg_prqk/configs/sft"
        config = yaml.safe_load((root / "a0_gid_history10_v1.yaml").read_text())
        reference = yaml.safe_load(
            (root / "a4_gid_parent_history10_v1.yaml").read_text()
        )
        config["nproc_per_node"] = 2
        with self.assertRaises(ValueError):
            validate_matched_config(config, reference)
        with self.assertRaises(ValueError):
            resolve_qg(project_root(), "/tmp/outside")

    def test_prepare_rejects_gpu_or_distributed_driver_before_any_writes(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
            with self.assertRaises(ValueError):
                prepare({}, {}, {}, {})
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "", "WORLD_SIZE": "4"}):
            with self.assertRaises(ValueError):
                prepare({}, {}, {}, {})

    def test_concurrent_preparation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {"run_control_dir": Path(directory)}
            with stage_lock(paths):
                with self.assertRaises(ValueError):
                    with stage_lock(paths):
                        pass

    def test_identifier_build_keeps_a0_sid_and_assigns_fresh_lexical_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a0 = root / "a0"
            a0.mkdir()
            np.save(a0 / "poi_assignments_s1_s2_s3.npy", self.new_base[:, 6:])
            np.save(a0 / "selected_poi_rows.npy", np.arange(2))
            ids = root / "poi_ids.jsonl"
            ids.write_text('"a"\n"b"\n')
            manifest = {
                "status": "completed",
                "contract": {
                    "gate": "full",
                    "codebook_sizes": [512] * 3,
                    "source_hashes": {"poi_ids_sha256": sha256_file(ids)},
                },
                "artifacts": {
                    name: {"sha256": sha256_file(a0 / name)}
                    for name in (
                        "poi_assignments_s1_s2_s3.npy",
                        "selected_poi_rows.npy",
                    )
                },
            }
            (a0 / "manifest.json").write_text(json.dumps(manifest))
            (a0 / "_SUCCESS").touch()
            tokens = {f"<S{level}_{i}>": i for level in (1, 2, 3) for i in range(512)}
            tokens.update({"<D_0>": 1, "<D_1>": 2})
            with patch(
                "qg_prqk.sft.a0_gid_data.load_final_identifier_lookup",
                return_value=self.old,
            ):
                result = build_a0_identifiers(
                    a0_dir=a0,
                    a4_id_dir=root / "a4",
                    poi_ids_path=ids,
                    token_mapping={"tokens": tokens},
                    output=root / "ids",
                    contract={},
                    expected_rows=2,
                )
                self.assertEqual(result["variant"], VARIANT)
                np.testing.assert_array_equal(
                    np.load(root / "ids/base_identifier_codes.npy"), self.new_base
                )
                np.testing.assert_array_equal(
                    np.load(root / "ids/dedup_codes.npy"), [0, 1]
                )
                tokens.pop("<D_1>")
                with self.assertRaises(ValueError):
                    build_a0_identifiers(
                        a0_dir=a0,
                        a4_id_dir=root / "a4",
                        poi_ids_path=ids,
                        token_mapping={"tokens": tokens},
                        output=root / "missing_vocab",
                        contract={},
                        expected_rows=2,
                    )

    def test_message_builder_streams_only_train_valid_and_reuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            a0 = root / "a0"
            a0.mkdir()
            np.save(a0 / "base_identifier_codes.npy", self.new_base)
            np.save(a0 / "dedup_codes.npy", np.array([0, 1], dtype=np.int32))
            (a0 / "manifest.json").write_text("{}")
            outputs = {}
            for split in ("train", "valid"):
                path = source / f"{split}.jsonl"
                path.write_text(json.dumps(self.source(split)) + "\n")
                outputs[path.name] = {"rows": 1, "sha256": sha256_file(path)}
            (source / "manifest.json").write_text(
                json.dumps(
                    {"status": "completed", "variant": WIRE_FORMAT, "outputs": outputs}
                )
            )
            (source / "_SUCCESS").touch()
            (source / "test.jsonl").write_text("must not be read")
            with patch(
                "qg_prqk.sft.a0_gid_data.load_final_identifier_lookup",
                return_value=self.old,
            ):
                kwargs = dict(
                    source_dir=source,
                    a4_id_dir=root / "old",
                    a0_id_dir=a0,
                    output=root / "out",
                    contract={},
                    datasets={"train": "a0_train", "valid": "a0_valid"},
                    expected_rows={"train": 1, "valid": 1},
                )
                result = build_a0_messages(**kwargs)
                self.assertFalse(result["business_test_read"])
                self.assertFalse((root / "out/test.jsonl").exists())
                target = json.loads((root / "out/train.jsonl").read_text())
                self.assertEqual(
                    target["target_qg_prqk_id_key"],
                    identifier_key(self.new_base[0], 0, variant=WIRE_FORMAT),
                )
                with patch(
                    "qg_prqk.sft.a0_gid_data.load_final_identifier_lookup",
                    side_effect=AssertionError("should reuse"),
                ):
                    self.assertEqual(build_a0_messages(**kwargs), result)

    def test_missing_cache_receipt_is_not_training_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                verify_cache_receipt({"run_control_dir": Path(directory)}, {})
            )

    @unittest.skipUnless(
        os.environ.get("QG_A0_CACHE_SMOKE") == "1",
        "按需运行真实 tokenizer 的微型合成缓存检查",
    )
    def test_real_llamafactory_cache_with_synthetic_messages(self):
        from qg_prqk.sft.preflight import (
            _build_cache,
            _cache_inputs,
            _load_tokenizer_and_template,
            _validate_zero_truncation,
            scan_train_valid_lengths,
        )
        from qg_prqk.sft.vocabulary import MAPPING_FILENAME

        model = project_root() / "qg_prqk/outputs/models/Qwen3-0.6B-QGPRQK-A4-Vocab-v1"
        tokenizer, template, module = _load_tokenizer_and_template(model)
        mapping = json.loads((model / MAPPING_FILENAME).read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs, registry = {}, {}
            for split in ("train", "valid"):
                record = self.remap(self.source())
                record["split"] = split
                path = root / f"{split}.jsonl"
                path.write_text((json.dumps(record, ensure_ascii=False) + "\n") * 4)
                outputs[path.name] = {"rows": 4, "sha256": sha256_file(path)}
                registry[split] = {
                    "file_name": str(path),
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages"},
                    "tags": {
                        "role_tag": "role",
                        "content_tag": "content",
                        "user_tag": "user",
                        "assistant_tag": "assistant",
                    },
                }
            (root / "dataset_info.json").write_text(json.dumps(registry))
            stats = scan_train_valid_lengths(
                data_dir=root,
                variant=VARIANT,
                manifest={"outputs": outputs},
                tokenizer=tokenizer,
                template=template,
                cutoff_len=1024,
                batch_size=4,
            )
            _validate_zero_truncation(stats)
            inputs = _cache_inputs(
                variant=VARIANT,
                model_dir=model,
                mapping=mapping,
                stats=stats,
                train_dataset="train",
                valid_dataset="valid",
            )
            kwargs = dict(
                model_dir=model,
                dataset_dir=root,
                output_dir=root / "cache",
                inputs=inputs,
                stats=stats,
                tokenizer=tokenizer,
                template=template,
                tokenizer_module=module,
                workers=1,
                preprocessing_batch_size=4,
            )
            result = _build_cache(**kwargs)
            self.assertEqual(result["status"], "completed")
            self.assertGreater(result["packed_rows"]["train"], 0)
            self.assertEqual(result, _build_cache(**kwargs))


if __name__ == "__main__":
    unittest.main()

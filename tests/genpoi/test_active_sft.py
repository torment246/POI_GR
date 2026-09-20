"""Platform preparation must finish and match the frozen inputs before training."""

import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from poi_gr.methods.genpoi import active_sft as pipeline


class ActiveSftTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(pipeline, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_receipt_rejects_changed_contract_and_changed_output(self):
        self.write("data/manifest.json", {"status": "completed"})
        directory = self.root / "data"
        contract = {"source_hash": "frozen"}
        receipt = self.write("ready.json", {"status": "completed", "contract": contract,
                                           "files": pipeline.tree_state(directory)})
        self.assertTrue(pipeline.verify_receipt(receipt, contract, directory))
        with self.assertRaisesRegex(ValueError, "来源或代码"):
            pipeline.verify_receipt(receipt, {"source_hash": "changed"}, directory)
        self.write("data/manifest.json", {"status": "corrupt"})
        with self.assertRaisesRegex(ValueError, "文件发生变化"):
            pipeline.verify_receipt(receipt, contract, directory)

    def test_prepare_refuses_gpu_or_distributed_worker(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(ValueError, "单 CPU"):
                pipeline.prepare({}, {}, {})
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "WORLD_SIZE": "4"}):
            with self.assertRaisesRegex(ValueError, "torchrun"):
                pipeline.prepare({}, {}, {})

    def test_incomplete_preparation_cannot_launch_training(self):
        protocol = {"tmp_dir": "tmp", "run_control_dir": "control"}
        with patch.object(pipeline, "hardware_check") as hardware:
            with self.assertRaises(FileNotFoundError):
                pipeline.train(protocol, {"output_dir": str(self.root / "train")}, {})
            hardware.assert_not_called()

    def test_preparation_failure_never_writes_ready_receipt(self):
        import subprocess
        protocol = {"tmp_dir": "tmp", "run_control_dir": "control",
                    "stages": [{"name": "messages", "output_dir": "data", "args": ["builder.py"]}]}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}), \
             patch.object(pipeline, "validate_pid"), \
             patch.object(pipeline.subprocess, "run", side_effect=subprocess.CalledProcessError(2, "builder")):
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.prepare(protocol, {"output_dir": str(self.root / "train")}, {})
        self.assertFalse((self.root / "control/prepare_ready.json").exists())
        self.assertFalse((self.root / "control/messages_ready.json").exists())

    def test_successful_launcher_system_exit_is_recorded(self):
        protocol = {"tmp_dir": "tmp", "run_control_dir": "control", "stages": []}
        contract = {"frozen": True}
        self.write("control/prepare_ready.json", {"status": "completed", "contract": contract})
        config = {"output_dir": str(self.root / "train"), "num_train_epochs": 3}

        def launch():
            self.write("train/trainer_state.json", {"epoch": 3, "global_step": 6})
            raise SystemExit(0)

        backend = types.SimpleNamespace(launcher=types.SimpleNamespace(launch=launch))
        with patch.dict(sys.modules, {"llamafactory": backend}), \
             patch.object(pipeline, "validate_pid"), patch.object(pipeline, "validate_cache"), \
             patch.object(pipeline, "hardware_check", return_value=[]), patch.dict(os.environ):
            pipeline.train(protocol, config, contract)
        self.assertEqual(pipeline.read_json(self.root / "train/genpoi_training_manifest.json")["status"], "completed")
        with self.assertRaisesRegex(ValueError, "输出非空"):
            pipeline.require_empty_training(config)

    def test_preparation_reuses_completed_stages_on_second_run(self):
        stages = [{"name": n, "output_dir": n, "args": [n + ".py"]}
                  for n in ("messages", "vocab", "cache")]
        protocol = {"tmp_dir": "tmp", "run_control_dir": "control", "stages": stages}
        config = {"output_dir": str(self.root / "train")}

        def command(args, **kwargs):
            name = Path(args[1]).stem
            self.write(f"{name}/manifest.json", {"status": "completed"})
            if name == "vocab":
                self.write("vocab/poi_token_mapping.json", {"new_vocab_size": 10000,
                           "added_token_count": 4094, "schema_version": "genpoi-vocab-v1"})
                self.write("vocab/config.json", {"vocab_size": 10000})
                (self.root / "vocab/model.safetensors").touch()

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}), \
             patch.object(pipeline, "validate_pid"), patch.object(pipeline, "validate_messages"), \
             patch.object(pipeline, "validate_cache", return_value={"train": 3, "validation": 2}), \
             patch.object(pipeline.subprocess, "run", side_effect=command) as run:
            pipeline.prepare(protocol, config, {"source": "frozen"})
            self.assertEqual(run.call_count, 3)
            run.reset_mock()
            pipeline.prepare(protocol, config, {"source": "frozen"})
            run.assert_not_called()
        self.assertEqual(pipeline.read_json(self.root / "control/prepare_ready.json")["status"], "completed")

    def test_cache_reloads_and_rejects_truncation_or_missing_labels(self):
        from datasets import Dataset, DatasetDict
        cache = self.root / "cache"
        rows = {"input_ids": [[1] * 1024] * 3, "labels": [[-100] * 1023 + [1]] * 3}
        DatasetDict({"train": Dataset.from_dict(rows), "validation": Dataset.from_dict(rows)}).save_to_disk(str(cache))
        protocol = {"stages": [{"output_dir": "data"}], "rows": {"train": 5, "valid": 4}}
        config = {"tokenized_path": str(cache), "model_name_or_path": str(self.root / "model"),
                  "template": "qwen3_nothink", "dataset": "train", "eval_dataset": "valid"}
        self.write("data/manifest.json", {"outputs": {f"{s}.jsonl": {"sha256": s} for s in ("train", "valid")}})
        self.write("model/poi_token_mapping.json", {"extended_tokenizer_sha256": "tokenizer"})
        inputs = {"cutoff_len": 1024, "packing": True, "train_on_prompt": False, "template": "qwen3_nothink",
                  "train_dataset": "train", "valid_dataset": "valid", "model_dir": config["model_name_or_path"],
                  "extended_tokenizer_sha256": "tokenizer", "train_sha256": "train", "valid_sha256": "valid"}
        self.write("cache/cache_manifest.json", {"inputs": inputs, "packed_rows": {"train": 3, "validation": 3}})
        stats = {"effective_cutoff_len": 1024, "over_requested_cutoff_count": 0,
                 "target_truncated_at_requested_cutoff_count": 0, "target_truncated_at_effective_cutoff_count": 0,
                 "splits": {s: {"rows": n} for s, n in protocol["rows"].items()}}
        self.write("cache/length_stats.json", stats)
        self.assertEqual(pipeline.validate_cache(protocol, config), {"train": 3, "validation": 3})
        bad = copy.deepcopy(stats)
        bad["over_requested_cutoff_count"] = 1
        self.write("cache/length_stats.json", bad)
        with self.assertRaisesRegex(ValueError, "零截断"):
            pipeline.validate_cache(protocol, config)
        self.write("cache/length_stats.json", stats)
        bad_rows = {"train": [{"input_ids": [1] * 1024, "labels": [-100] * 1024}] * 3,
                    "validation": [{"input_ids": [1] * 1024, "labels": [1] * 1024}] * 3}
        with patch("datasets.load_from_disk", return_value=bad_rows):
            with self.assertRaisesRegex(ValueError, "监督标签"):
                pipeline.validate_cache(protocol, config)


if __name__ == "__main__":
    unittest.main()

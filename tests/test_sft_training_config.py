"""Contract tests for the first Qwen3 order-main-task SFT run."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from scripts.validate_sft_tokenization import LengthHistogram, _infer_target_length


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "sft"


class SftTrainingConfigTest(unittest.TestCase):
    def load_yaml(self, name: str) -> dict:
        return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))

    def test_dataset_registration_contains_train_and_valid_only(self) -> None:
        expected_path = CONFIG_DIR / "llamafactory_dataset_info.json"
        actual_path = CONFIG_DIR / "dataset_info.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
        self.assertEqual(expected, actual)
        self.assertEqual(
            set(actual),
            {"beijing_order_main_v1_train", "beijing_order_main_v1_valid"},
        )
        serialized = json.dumps(actual)
        self.assertNotIn("test.jsonl", serialized)
        for item in actual.values():
            self.assertEqual(item["formatting"], "sharegpt")
            self.assertEqual(item["columns"]["messages"], "messages")

    def test_formal_training_protocol(self) -> None:
        config = self.load_yaml("qwen3_0.6b_main_v1.yaml")
        self.assertEqual(config["stage"], "sft")
        self.assertEqual(config["finetuning_type"], "full")
        self.assertEqual(config["template"], "qwen3_nothink")
        self.assertFalse(config["enable_thinking"])
        self.assertFalse(config["train_on_prompt"])
        self.assertTrue(config["packing"])
        self.assertEqual(config["num_train_epochs"], 1.0)
        self.assertEqual(config["learning_rate"], 5e-5)
        self.assertEqual(config["optim"], "adamw_torch")
        self.assertEqual(config["save_steps"], 0.5)
        self.assertEqual(config["eval_steps"], 0.5)
        self.assertEqual(config["save_total_limit"], 2)
        self.assertEqual(config["report_to"], "tensorboard")
        self.assertEqual(config["logging_strategy"], "steps")
        self.assertEqual(config["logging_steps"], 20)
        self.assertTrue(config["logging_first_step"])
        self.assertTrue(config["logging_dir"].endswith("/tensorboard"))
        self.assertFalse(config["load_best_model_at_end"])
        self.assertFalse(config["predict_with_generate"])
        self.assertFalse(config["compute_accuracy"])
        global_batch = (
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
        )
        self.assertEqual(global_batch, 512)
        serialized = json.dumps(config)
        self.assertNotIn("test.jsonl", serialized)
        self.assertNotIn("early_stopping", serialized)

    def test_smoke_is_twenty_steps_on_fixed_cache(self) -> None:
        config = self.load_yaml("qwen3_0.6b_main_v1_smoke.yaml")
        self.assertEqual(config["max_steps"], 20)
        self.assertEqual(config["save_steps"], 10)
        self.assertEqual(config["eval_steps"], 10)
        self.assertTrue(config["tokenized_path"].endswith("/smoke"))
        self.assertEqual(
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"],
            512,
        )

    def test_length_histogram_and_target_truncation_detection(self) -> None:
        histogram = LengthHistogram()
        histogram.update([1, 2, 3, 4])
        self.assertEqual(histogram.summary(include_p999=False)["p50"], 2.5)
        source = __import__("numpy").asarray([50, 500])
        target = __import__("numpy").asarray([11, 200])
        retained = _infer_target_length(source, target, 128)
        self.assertEqual(int(retained[0]), 11)
        self.assertLess(int(retained[1]), 200)


if __name__ == "__main__":
    unittest.main()

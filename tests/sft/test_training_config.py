"""Contract tests for the first Qwen3 order-main-task SFT run."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from scripts.sft.validate_tokenization import LengthHistogram, _infer_target_length


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
        self.assertGreaterEqual(
            set(actual),
            {
                "beijing_order_main_v1_train",
                "beijing_order_main_v1_valid",
                "tiger_bge_m3_1024x3_history10_query_gid_v1_train",
                "tiger_bge_m3_1024x3_history10_query_gid_v1_valid",
                "tiger_bge_m3_1024x3_history10_query_gid_v1_smoke_train",
                "tiger_bge_m3_1024x3_history10_query_gid_v1_smoke_valid",
                "genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_train",
                "genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_valid",
                "gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_train",
                "gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_valid",
                "genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_train",
                "genpoi_centered_geope32_tiger_rqvae_1024x3_history10_query_gid_v2_valid",
                "rqkmeans_e4_bge_m3_512x1024x2048_history10_query_gid_tiger_collision_v1_train",
                "rqkmeans_e4_bge_m3_512x1024x2048_history10_query_gid_tiger_collision_v1_valid",
                "rqkmeans_e4_bge_m3_1024x1024x1024_history10_query_gid_tiger_collision_v1_train",
                "rqkmeans_e4_bge_m3_1024x1024x1024_history10_query_gid_tiger_collision_v1_valid",
                "rqkmeans_e4_bge_m3_2048x1024x512_history10_query_gid_tiger_collision_v1_train",
                "rqkmeans_e4_bge_m3_2048x1024x512_history10_query_gid_tiger_collision_v1_valid",
                "ghr_exp09_g6_minimum_entity_dedup_history10_query_gid_v1_train",
                "ghr_exp09_g6_minimum_entity_dedup_history10_query_gid_v1_valid",
                "ghr_exp14_g6_relation_tree_dedup_history10_query_gid_v1_train",
                "ghr_exp14_g6_relation_tree_dedup_history10_query_gid_v1_valid",
                "rqkmeans_bge_m3_1024x1024x1024_history10_query_gid_tiger_collision_v1_train",
                "rqkmeans_bge_m3_1024x1024x1024_history10_query_gid_tiger_collision_v1_valid",
                "rqvae_e4_bge_m3_1024x1024x1024_history10_query_gid_tiger_collision_v1_train",
                "rqvae_e4_bge_m3_1024x1024x1024_history10_query_gid_tiger_collision_v1_valid",
                "ghr_tiger_aligned_collision_32x32_history10_query_gid_v1_train",
                "ghr_tiger_aligned_collision_32x32_history10_query_gid_v1_valid",
                "tiger_mmbert_recall_128_1024x3_history10_query_gid_v1_train",
                "tiger_mmbert_recall_128_1024x3_history10_query_gid_v1_valid",
            },
        )
        serialized = json.dumps(actual)
        self.assertNotIn("test.jsonl", serialized)
        for name in actual:
            self.assertTrue(name.endswith(("_train", "_valid")))
            stem = name.rsplit("_", 1)[0]
            self.assertIn(f"{stem}_train", actual)
            self.assertIn(f"{stem}_valid", actual)
        for item in actual.values():
            self.assertEqual(item["formatting"], "sharegpt")
            self.assertEqual(item["columns"]["messages"], "messages")

    def test_active_gnpr_matches_tiger_except_artifact_paths(self) -> None:
        tiger = self.load_yaml("tiger_active716k_bge_m3_512x3_history10_query_gid_v1.yaml")
        gnpr = self.load_yaml("gnpr_active716k_bge_m3_category_pluscode6_512x3_history10_query_gid_v1.yaml")
        path_keys = {"model_name_or_path", "dataset", "eval_dataset", "tokenized_path", "output_dir", "logging_dir"}
        self.assertEqual({k: v for k, v in tiger.items() if k not in path_keys},
                         {k: v for k, v in gnpr.items() if k not in path_keys})
        self.assertEqual(gnpr["cutoff_len"], 1024)
        self.assertIn("GNPR-Active716K", gnpr["model_name_or_path"])

    def test_tiger_formal_training_protocol(self) -> None:
        config = self.load_yaml(
            "tiger_bge_m3_1024x3_history10_query_gid_v1.yaml"
        )
        self.assertEqual(config["model_name_or_path"], "models/Qwen3-0.6B-TIGER-Vocab-v1")
        self.assertEqual(config["template"], "qwen3_nothink")
        self.assertFalse(config["enable_thinking"])
        self.assertFalse(config["train_on_prompt"])
        self.assertTrue(config["packing"])
        self.assertEqual(config["cutoff_len"], 512)
        self.assertEqual(config["num_train_epochs"], 3.0)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertEqual(config["save_total_limit"], 3)
        self.assertNotIn("save_steps", config)
        self.assertNotIn("eval_steps", config)
        global_batch = (
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
            * 4
        )
        self.assertEqual(global_batch, 512)
        serialized = json.dumps(config)
        self.assertNotIn("test.jsonl", serialized)

    def test_genpoi_single_gpu_formal_training_protocol(self) -> None:
        config = self.load_yaml(
            "genpoi_bge_m3_geope_1024x3_history10_query_gid_v1.yaml"
        )
        self.assertEqual(
            config["model_name_or_path"],
            "models/Qwen3-0.6B-GenPOI-Vocab-v1",
        )
        self.assertEqual(config["template"], "qwen3_nothink")
        self.assertFalse(config["enable_thinking"])
        self.assertFalse(config["train_on_prompt"])
        self.assertTrue(config["packing"])
        self.assertEqual(config["cutoff_len"], 512)
        self.assertEqual(config["num_train_epochs"], 3.0)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertEqual(config["save_total_limit"], 3)
        self.assertNotIn("save_steps", config)
        self.assertNotIn("eval_steps", config)
        global_batch = (
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
        )
        self.assertEqual(global_batch, 512)
        serialized = json.dumps(config)
        self.assertNotIn("test.jsonl", serialized)

    def test_genpoi_smoke_uses_fixed_cache_and_twenty_steps(self) -> None:
        config = self.load_yaml(
            "genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_smoke.yaml"
        )
        self.assertEqual(config["max_steps"], 20)
        self.assertEqual(config["save_steps"], 10)
        self.assertEqual(config["eval_steps"], 10)
        self.assertTrue(config["tokenized_path"].endswith("/smoke"))
        self.assertEqual(config["per_device_train_batch_size"], 1)
        self.assertEqual(config["gradient_accumulation_steps"], 1)
        self.assertEqual(config["cutoff_len"], 512)
        self.assertFalse(config["train_on_prompt"])
        self.assertEqual(config["report_to"], "none")
        serialized = json.dumps(config)
        self.assertNotIn("test.jsonl", serialized)

    def test_centered_genpoi_formal_and_smoke_training_protocol(self) -> None:
        stem = (
            "genpoi_centered_geope32_tiger_rqvae_1024x3_"
            "history10_query_gid_v2"
        )
        config = self.load_yaml(f"{stem}.yaml")
        self.assertEqual(
            config["model_name_or_path"],
            "models/Qwen3-0.6B-GenPOI-Vocab-v1",
        )
        self.assertEqual(config["dataset"], f"{stem}_train")
        self.assertEqual(config["eval_dataset"], f"{stem}_valid")
        self.assertEqual(config["num_train_epochs"], 3.0)
        self.assertEqual(config["cutoff_len"], 512)
        self.assertTrue(config["packing"])
        self.assertFalse(config["train_on_prompt"])
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertEqual(config["save_total_limit"], 3)

        smoke = self.load_yaml(f"{stem}_smoke.yaml")
        self.assertEqual(smoke["max_steps"], 20)
        self.assertTrue(smoke["tokenized_path"].endswith("/smoke"))
        self.assertNotIn("test.jsonl", json.dumps(smoke))

    def test_gnpr_formal_and_smoke_training_protocol(self) -> None:
        config = self.load_yaml(
            "gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1.yaml"
        )
        self.assertEqual(
            config["model_name_or_path"],
            "models/Qwen3-0.6B-GNPR-Vocab-v1",
        )
        self.assertEqual(config["template"], "qwen3_nothink")
        self.assertFalse(config["enable_thinking"])
        self.assertFalse(config["train_on_prompt"])
        self.assertTrue(config["packing"])
        self.assertEqual(config["cutoff_len"], 512)
        self.assertEqual(config["num_train_epochs"], 3.0)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertEqual(config["save_total_limit"], 3)
        self.assertEqual(
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"],
            512,
        )
        serialized = json.dumps(config)
        self.assertNotIn("test.jsonl", serialized)
        self.assertNotIn("passenger_id", serialized)

        smoke = self.load_yaml(
            "gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_smoke.yaml"
        )
        self.assertEqual(smoke["max_steps"], 20)
        self.assertEqual(smoke["save_steps"], 10)
        self.assertEqual(smoke["eval_steps"], 10)
        self.assertTrue(smoke["tokenized_path"].endswith("/smoke"))
        self.assertEqual(smoke["cutoff_len"], 512)
        self.assertFalse(smoke["train_on_prompt"])
        self.assertEqual(smoke["report_to"], "none")

    def test_rqkmeans_e4_formal_training_protocol(self) -> None:
        layouts = {
            "512x1024x2048": (
                "models/Qwen3-0.6B-RQKMeans-E4-512x1024x2048-Vocab-v1"
            ),
            "1024x1024x1024": (
                "models/Qwen3-0.6B-RQKMeans-E4-1024x1024x1024-Vocab-v1"
            ),
            "2048x1024x512": (
                "models/Qwen3-0.6B-RQKMeans-E4-2048x1024x512-Vocab-v1"
            ),
        }
        for layout, model_path in layouts.items():
            stem = (
                f"rqkmeans_e4_bge_m3_{layout}_history10_query_gid_"
                "tiger_collision_v1"
            )
            config = self.load_yaml(f"{stem}.yaml")
            self.assertEqual(config["model_name_or_path"], model_path)
            self.assertEqual(config["dataset"], f"{stem}_train")
            self.assertEqual(config["eval_dataset"], f"{stem}_valid")
            self.assertEqual(
                config["tokenized_path"],
                f"data/sft/tokenized/{stem}",
            )
            self.assertEqual(config["template"], "qwen3_nothink")
            self.assertFalse(config["enable_thinking"])
            self.assertFalse(config["train_on_prompt"])
            self.assertTrue(config["packing"])
            self.assertEqual(config["cutoff_len"], 512)
            self.assertEqual(config["num_train_epochs"], 3.0)
            self.assertEqual(config["save_strategy"], "epoch")
            self.assertEqual(config["eval_strategy"], "epoch")
            self.assertEqual(config["save_total_limit"], 3)
            self.assertEqual(
                config["per_device_train_batch_size"]
                * config["gradient_accumulation_steps"]
                * 4,
                512,
            )
            self.assertNotIn("test.jsonl", json.dumps(config))

    def test_ghr_formal_training_protocol(self) -> None:
        candidates = {
            "ghr_exp09_g6_minimum_entity_dedup_history10_query_gid_v1": (
                "models/Qwen3-0.6B-GHR-EXP09-Vocab-v1",
                512,
            ),
            "ghr_exp14_g6_relation_tree_dedup_history10_query_gid_v1": (
                "models/Qwen3-0.6B-GHR-EXP14-Vocab-v1",
                512,
            ),
            "ghr_tiger_aligned_collision_32x32_history10_query_gid_v1": (
                "models/Qwen3-0.6B-GHR-TIGER-Aligned-Collision-32x32-Vocab-v1",
                1024,
            ),
        }
        for stem, (model_path, cutoff_len) in candidates.items():
            config = self.load_yaml(f"{stem}.yaml")
            self.assertEqual(config["model_name_or_path"], model_path)
            self.assertEqual(config["dataset"], f"{stem}_train")
            self.assertEqual(config["eval_dataset"], f"{stem}_valid")
            self.assertEqual(config["tokenized_path"], f"data/sft/tokenized/{stem}")
            self.assertEqual(config["template"], "qwen3_nothink")
            self.assertFalse(config["enable_thinking"])
            self.assertFalse(config["train_on_prompt"])
            self.assertTrue(config["packing"])
            self.assertEqual(config["cutoff_len"], cutoff_len)
            self.assertEqual(config["num_train_epochs"], 3.0)
            self.assertEqual(config["save_strategy"], "epoch")
            self.assertEqual(config["eval_strategy"], "epoch")
            self.assertEqual(config["save_total_limit"], 3)
            self.assertEqual(
                config["per_device_train_batch_size"]
                * config["gradient_accumulation_steps"]
                * 4,
                512,
            )
            self.assertNotIn("test.jsonl", json.dumps(config))

    def test_rqkmeans_bge_formal_training_protocol(self) -> None:
        stem = (
            "rqkmeans_bge_m3_1024x1024x1024_history10_query_gid_"
            "tiger_collision_v1"
        )
        config = self.load_yaml(f"{stem}.yaml")
        self.assertEqual(
            config["model_name_or_path"],
            "models/Qwen3-0.6B-RQKMeans-BGE-1024x1024x1024-Vocab-v1",
        )
        self.assertEqual(config["dataset"], f"{stem}_train")
        self.assertEqual(config["eval_dataset"], f"{stem}_valid")
        self.assertEqual(config["tokenized_path"], f"data/sft/tokenized/{stem}")
        self.assertEqual(config["template"], "qwen3_nothink")
        self.assertFalse(config["enable_thinking"])
        self.assertFalse(config["train_on_prompt"])
        self.assertTrue(config["packing"])
        self.assertEqual(config["cutoff_len"], 512)
        self.assertEqual(config["num_train_epochs"], 3.0)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertEqual(config["save_total_limit"], 3)
        self.assertEqual(
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
            * 4,
            512,
        )
        self.assertNotIn("test.jsonl", json.dumps(config))

    def test_rqvae_e4_formal_training_protocol(self) -> None:
        stem = (
            "rqvae_e4_bge_m3_1024x1024x1024_history10_query_gid_"
            "tiger_collision_v1"
        )
        config = self.load_yaml(f"{stem}.yaml")
        self.assertEqual(
            config["model_name_or_path"],
            "models/Qwen3-0.6B-RQVAE-E4-1024x1024x1024-Vocab-v1",
        )
        self.assertEqual(config["dataset"], f"{stem}_train")
        self.assertEqual(config["eval_dataset"], f"{stem}_valid")
        self.assertEqual(config["tokenized_path"], f"data/sft/tokenized/{stem}")
        self.assertEqual(config["template"], "qwen3_nothink")
        self.assertFalse(config["enable_thinking"])
        self.assertFalse(config["train_on_prompt"])
        self.assertTrue(config["packing"])
        self.assertEqual(config["cutoff_len"], 512)
        self.assertEqual(config["num_train_epochs"], 3.0)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertEqual(config["save_total_limit"], 3)
        self.assertEqual(
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
            * 4,
            512,
        )
        self.assertNotIn("test.jsonl", json.dumps(config))

    def test_tiger_mmbert_recall_128_formal_training_protocol(self) -> None:
        stem = "tiger_mmbert_recall_128_1024x3_history10_query_gid_v1"
        config = self.load_yaml(f"{stem}.yaml")
        self.assertEqual(
            config["model_name_or_path"],
            "models/Qwen3-0.6B-TIGER-MMBERT-Recall-128-1024x3-Vocab-v1",
        )
        self.assertEqual(config["dataset"], f"{stem}_train")
        self.assertEqual(config["eval_dataset"], f"{stem}_valid")
        self.assertEqual(config["tokenized_path"], f"data/sft/tokenized/{stem}")
        self.assertEqual(config["template"], "qwen3_nothink")
        self.assertFalse(config["enable_thinking"])
        self.assertFalse(config["train_on_prompt"])
        self.assertTrue(config["packing"])
        self.assertEqual(config["cutoff_len"], 1024)
        self.assertEqual(config["num_train_epochs"], 3.0)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertEqual(config["save_total_limit"], 3)
        self.assertEqual(
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
            * 4,
            512,
        )
        self.assertNotIn("test.jsonl", json.dumps(config))

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

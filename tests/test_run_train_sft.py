import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_train_sft import build_training_config, validate_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_args(**overrides):
    values = {
        "output_dir": str(PROJECT_ROOT / "outputs/sft/test"),
        "experiment_name": "test",
        "model_name_or_path": "models/Qwen3-0.6B-POI-Vocab-v1",
        "dataset_dir": "configs/sft",
        "train_dataset": "beijing_order_main_v1_train",
        "eval_dataset": "beijing_order_main_v1_valid",
        "tokenized_path": "data/sft/tokenized/beijing_order_main_v1_qwen3_0.6b",
        "template": "qwen3_nothink",
        "num_epochs": 1.0,
        "batch_size": 64,
        "eval_batch_size": 64,
        "gradient_accumulation_steps": 2,
        "nproc_per_node": 4,
        "expected_global_batch_size": 512,
        "cutoff_len": 128,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "precision": "bf16",
        "packing": 1,
        "train_on_prompt": 0,
        "gradient_checkpointing": 1,
        "logging_steps": 20,
        "save_steps": 0.5,
        "eval_steps": 0.5,
        "save_total_limit": 2,
        "report_to": "tensorboard",
        "seed": 42,
        "data_seed": 42,
        "dataloader_num_workers": 4,
        "preprocessing_num_workers": 16,
        "include_tokens_per_second": 1,
        "master_port": 29500,
        "resume": 0,
        "resume_path": "",
        "llamafactory_path": "third_party/LLaMA-Factory",
        "dry_run": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RunTrainSftTest(unittest.TestCase):
    def test_cluster_shell_keeps_training_in_foreground(self):
        shell_text = (PROJECT_ROOT / "run_train_sft.sh").read_text(encoding="utf-8")
        self.assertTrue(shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertNotIn("nohup", shell_text)
        self.assertNotIn("setsid", shell_text)
        self.assertNotIn("disown", shell_text)
        self.assertFalse(
            any(line.rstrip().endswith(" &") for line in shell_text.splitlines())
        )
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
        self.assertIn('name="qwen3_0.6b_main_v1_gpu4_e2"', shell_text)
        self.assertIn("--num_epochs 2.0", shell_text)
        self.assertIn("--gradient_accumulation_steps 2", shell_text)
        self.assertIn("--nproc_per_node 4", shell_text)
        self.assertIn("--expected_global_batch_size 512", shell_text)
        self.assertIn("--save_steps 0.25", shell_text)
        self.assertIn("--eval_steps 0.25", shell_text)
        self.assertIn("--save_total_limit 4", shell_text)
        self.assertIn("--gradient_checkpointing 0", shell_text)

    def test_four_gpu_arguments_keep_global_batch_512(self):
        args = make_args(
            num_epochs=2.0,
            save_steps=0.25,
            eval_steps=0.25,
            save_total_limit=4,
        )
        resolved = validate_args(args, PROJECT_ROOT)
        self.assertEqual(resolved["global_batch_size"], 512)
        config = build_training_config(args, resolved)
        self.assertEqual(config["per_device_train_batch_size"], 64)
        self.assertEqual(config["gradient_accumulation_steps"], 2)
        self.assertEqual(config["num_train_epochs"], 2.0)
        self.assertEqual(config["save_steps"], 0.25)
        self.assertEqual(config["eval_steps"], 0.25)
        self.assertEqual(config["save_total_limit"], 4)

    def test_single_a100_two_epoch_shell(self):
        shell_path = PROJECT_ROOT / "run_train_sft_single_a100_2epoch.sh"
        shell_text = shell_path.read_text(encoding="utf-8")
        self.assertTrue(shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertNotIn("nohup", shell_text)
        self.assertNotIn("setsid", shell_text)
        self.assertNotIn("disown", shell_text)
        self.assertFalse(
            any(line.rstrip().endswith(" &") for line in shell_text.splitlines())
        )
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", shell_text)
        self.assertIn('name="qwen3_0.6b_main_v1_a100_e2"', shell_text)
        self.assertIn("--num_epochs 2.0", shell_text)
        self.assertIn("--batch_size 64", shell_text)
        self.assertIn("--gradient_accumulation_steps 8", shell_text)
        self.assertIn("--nproc_per_node 1", shell_text)
        self.assertIn("--expected_global_batch_size 512", shell_text)
        self.assertIn("--save_steps 0.25", shell_text)
        self.assertIn("--eval_steps 0.25", shell_text)
        self.assertIn("--save_total_limit 4", shell_text)
        self.assertIn('2>&1 | tee "${output_dir}/train_console.log"', shell_text)

        args = make_args(
            output_dir=str(PROJECT_ROOT / "outputs/sft/test_single_a100_e2"),
            experiment_name="test_single_a100_e2",
            num_epochs=2.0,
            gradient_accumulation_steps=8,
            nproc_per_node=1,
            save_steps=0.25,
            eval_steps=0.25,
            save_total_limit=4,
        )
        resolved = validate_args(args, PROJECT_ROOT)
        config = build_training_config(args, resolved)
        self.assertEqual(resolved["global_batch_size"], 512)
        self.assertEqual(config["num_train_epochs"], 2.0)
        self.assertEqual(config["save_steps"], 0.25)
        self.assertEqual(config["eval_steps"], 0.25)
        self.assertEqual(config["save_total_limit"], 4)

    def test_rejects_wrong_global_batch(self):
        args = make_args(gradient_accumulation_steps=1)
        with self.assertRaises(ValueError):
            validate_args(args, PROJECT_ROOT)

    def test_only_tensorboard_and_no_test_dataset(self):
        args = make_args()
        resolved = validate_args(args, PROJECT_ROOT)
        config = build_training_config(args, resolved)
        self.assertEqual(config["report_to"], "tensorboard")
        self.assertNotIn("test", config["dataset"])
        self.assertNotIn("test", config["eval_dataset"])
        self.assertFalse(config["predict_with_generate"])

    def test_resume_requires_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_args(resume=1, resume_path=temporary)
            with self.assertRaises(FileNotFoundError):
                validate_args(args, PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()

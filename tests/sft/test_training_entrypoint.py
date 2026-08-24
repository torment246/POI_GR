import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.sft.train import (
    build_training_config,
    configure_distributed_environment,
    validate_args,
    validate_tokenized_cache_cutoff,
    write_epoch_checkpoint_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        "save_strategy": "steps",
        "save_steps": 0.5,
        "eval_strategy": "steps",
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
        shell_text = (
            PROJECT_ROOT / "launchers/run_train_sft.sh"
        ).read_text(encoding="utf-8")
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
        shell_path = (
            PROJECT_ROOT
            / "launchers/run_train_sft_single_a100_2epoch.sh"
        )
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

    def test_rejects_tokenized_cache_cutoff_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            tokenized_path = Path(temporary)
            (tokenized_path / "cache_manifest.json").write_text(
                json.dumps({"inputs": {"cutoff_len": 512}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Tokenized Cache 不一致"):
                validate_tokenized_cache_cutoff(tokenized_path, 1024)

    def test_auto_master_port_removes_inherited_fixed_port(self):
        args = make_args(master_port=0)
        validate_args(args, PROJECT_ROOT)
        original_master_port = os.environ.get("MASTER_PORT")
        try:
            os.environ["MASTER_PORT"] = "29734"
            configure_distributed_environment(args)
            self.assertNotIn("MASTER_PORT", os.environ)
            self.assertEqual(os.environ["MASTER_ADDR"], "127.0.0.1")
            self.assertEqual(os.environ["NPROC_PER_NODE"], "4")
        finally:
            if original_master_port is None:
                os.environ.pop("MASTER_PORT", None)
            else:
                os.environ["MASTER_PORT"] = original_master_port

    def test_fixed_master_port_is_preserved(self):
        args = make_args(master_port=29734)
        validate_args(args, PROJECT_ROOT)
        original_master_port = os.environ.get("MASTER_PORT")
        try:
            configure_distributed_environment(args)
            self.assertEqual(os.environ["MASTER_PORT"], "29734")
        finally:
            if original_master_port is None:
                os.environ.pop("MASTER_PORT", None)
            else:
                os.environ["MASTER_PORT"] = original_master_port

    def test_rejects_invalid_master_port(self):
        for port in (-1, 65536):
            with self.subTest(port=port):
                with self.assertRaises(ValueError):
                    validate_args(make_args(master_port=port), PROJECT_ROOT)

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

    def test_tiger_four_gpu_epoch_config(self):
        args = make_args(
            model_name_or_path="models/Qwen3-0.6B-TIGER-Vocab-v1",
            train_dataset="tiger_bge_m3_1024x3_history10_query_gid_v1_train",
            eval_dataset="tiger_bge_m3_1024x3_history10_query_gid_v1_valid",
            tokenized_path=(
                "data/sft/tokenized/"
                "tiger_bge_m3_1024x3_history10_query_gid_v1"
            ),
            num_epochs=3.0,
            batch_size=16,
            eval_batch_size=16,
            gradient_accumulation_steps=8,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=3,
            cutoff_len=512,
        )
        resolved = validate_args(args, PROJECT_ROOT)
        config = build_training_config(args, resolved)
        self.assertEqual(resolved["global_batch_size"], 512)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertNotIn("save_steps", config)
        self.assertNotIn("eval_steps", config)

    def test_genpoi_single_gpu_epoch_config(self):
        args = make_args(
            model_name_or_path="models/Qwen3-0.6B-GenPOI-Vocab-v1",
            train_dataset="genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_train",
            eval_dataset="genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_valid",
            tokenized_path=(
                "data/sft/tokenized/"
                "genpoi_bge_m3_geope_1024x3_history10_query_gid_v1"
            ),
            num_epochs=3.0,
            batch_size=16,
            eval_batch_size=16,
            gradient_accumulation_steps=32,
            nproc_per_node=1,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=3,
            cutoff_len=512,
        )
        resolved = validate_args(args, PROJECT_ROOT)
        config = build_training_config(args, resolved)
        self.assertEqual(resolved["global_batch_size"], 512)
        self.assertEqual(config["num_train_epochs"], 3.0)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertNotIn("save_steps", config)
        self.assertNotIn("eval_steps", config)

    def test_genpoi_four_6000d_epoch_checkpoint_protocol(self):
        shell_path = (
            PROJECT_ROOT
            / "launchers/run_train_genpoi_sft_4x6000d_3epoch.sh"
        )
        shell_text = shell_path.read_text(encoding="utf-8")
        self.assertTrue(shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
        self.assertNotIn("nohup", shell_text)
        self.assertNotIn("setsid", shell_text)
        self.assertNotIn("disown", shell_text)
        self.assertFalse(
            any(line.rstrip().endswith(" &") for line in shell_text.splitlines())
        )
        self.assertIn("torch.cuda.device_count()", shell_text)
        self.assertIn('--num_epochs 3.0', shell_text)
        self.assertIn('--batch_size 16', shell_text)
        self.assertIn('--gradient_accumulation_steps 8', shell_text)
        self.assertIn('--nproc_per_node 4', shell_text)
        self.assertIn('--expected_global_batch_size 512', shell_text)
        self.assertIn('--cutoff_len 512', shell_text)
        self.assertIn('--save_strategy "epoch"', shell_text)
        self.assertIn('--eval_strategy "epoch"', shell_text)
        self.assertIn('--save_total_limit 3', shell_text)
        self.assertIn(
            'name="genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_gpu4_6000d_e3"',
            shell_text,
        )

        args = make_args(
            model_name_or_path="models/Qwen3-0.6B-GenPOI-Vocab-v1",
            train_dataset="genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_train",
            eval_dataset="genpoi_bge_m3_geope_1024x3_history10_query_gid_v1_valid",
            tokenized_path=(
                "data/sft/tokenized/"
                "genpoi_bge_m3_geope_1024x3_history10_query_gid_v1"
            ),
            num_epochs=3.0,
            batch_size=16,
            eval_batch_size=16,
            gradient_accumulation_steps=8,
            nproc_per_node=4,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=3,
            cutoff_len=512,
        )
        resolved = validate_args(args, PROJECT_ROOT)
        config = build_training_config(args, resolved)
        self.assertEqual(resolved["global_batch_size"], 512)
        self.assertEqual(config["per_device_train_batch_size"], 16)
        self.assertEqual(config["gradient_accumulation_steps"], 8)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertNotIn("save_steps", config)
        self.assertNotIn("eval_steps", config)

    def test_centered_genpoi_four_gpu_scripts(self):
        stem = (
            "genpoi_centered_geope32_tiger_rqvae_1024x3_"
            "history10_query_gid_v2"
        )
        scripts = {
            "run_train_genpoi_centered_sft_4x6000d_3epoch.sh": "gpu4_6000d_e3",
            "run_train_genpoi_centered_sft_4a100_3epoch.sh": "gpu4_a100_e3",
            "run_train_genpoi_centered_sft_4xa6000_3epoch.sh": "gpu4_a6000_e3",
        }
        for filename, output_suffix in scripts.items():
            shell_text = (
                PROJECT_ROOT / "launchers" / filename
            ).read_text(encoding="utf-8")
            self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
            self.assertIn("--num_epochs 3.0", shell_text)
            self.assertIn("--batch_size 16", shell_text)
            self.assertIn("--gradient_accumulation_steps 8", shell_text)
            self.assertIn("--nproc_per_node 4", shell_text)
            self.assertIn("--expected_global_batch_size 512", shell_text)
            self.assertIn('--save_strategy "epoch"', shell_text)
            self.assertIn('--eval_strategy "epoch"', shell_text)
            self.assertIn("--save_total_limit 3", shell_text)
            self.assertIn(f'name="{stem}_{output_suffix}"', shell_text)
            self.assertNotIn("nohup", shell_text)
            self.assertFalse(
                any(
                    line.rstrip().endswith(" &")
                    for line in shell_text.splitlines()
                )
            )

    def test_tiger_four_6000d_epoch_checkpoint_protocol(self):
        shell_path = (
            PROJECT_ROOT
            / "launchers/run_train_tiger_sft_4x6000d_3epoch.sh"
        )
        shell_text = shell_path.read_text(encoding="utf-8")
        self.assertTrue(shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
        self.assertNotIn("nohup", shell_text)
        self.assertNotIn("setsid", shell_text)
        self.assertNotIn("disown", shell_text)
        self.assertFalse(
            any(line.rstrip().endswith(" &") for line in shell_text.splitlines())
        )
        self.assertIn("torch.cuda.device_count()", shell_text)
        self.assertIn('--num_epochs 3.0', shell_text)
        self.assertIn('--batch_size 16', shell_text)
        self.assertIn('--gradient_accumulation_steps 8', shell_text)
        self.assertIn('--nproc_per_node 4', shell_text)
        self.assertIn('--expected_global_batch_size 512', shell_text)
        self.assertIn('--cutoff_len 512', shell_text)
        self.assertIn('--save_strategy "epoch"', shell_text)
        self.assertIn('--eval_strategy "epoch"', shell_text)
        self.assertIn('--save_total_limit 3', shell_text)
        self.assertIn(
            'name="tiger_bge_m3_1024x3_history10_query_gid_v1_gpu4_6000d_e3"',
            shell_text,
        )

        args = make_args(
            model_name_or_path="models/Qwen3-0.6B-TIGER-Vocab-v1",
            train_dataset="tiger_bge_m3_1024x3_history10_query_gid_v1_train",
            eval_dataset="tiger_bge_m3_1024x3_history10_query_gid_v1_valid",
            tokenized_path=(
                "data/sft/tokenized/"
                "tiger_bge_m3_1024x3_history10_query_gid_v1"
            ),
            num_epochs=3.0,
            batch_size=16,
            eval_batch_size=16,
            gradient_accumulation_steps=8,
            nproc_per_node=4,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=3,
            cutoff_len=512,
        )
        resolved = validate_args(args, PROJECT_ROOT)
        config = build_training_config(args, resolved)
        self.assertEqual(resolved["global_batch_size"], 512)
        self.assertEqual(config["per_device_train_batch_size"], 16)
        self.assertEqual(config["gradient_accumulation_steps"], 8)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertNotIn("save_steps", config)
        self.assertNotIn("eval_steps", config)

    def test_rqkmeans_e4_four_gpu_launchers(self):
        layouts = (
            "512x1024x2048",
            "1024x1024x1024",
            "2048x1024x512",
        )
        runtime_tags = {
            ("512x1024x2048", "4a100"): "rk512a",
            ("512x1024x2048", "4x6000d"): "rk512d",
            ("1024x1024x1024", "4a100"): "rk1024a",
            ("1024x1024x1024", "4x6000d"): "rk1024d",
            ("2048x1024x512", "4a100"): "rk2048a",
            ("2048x1024x512", "4x6000d"): "rk2048d",
        }
        for layout in layouts:
            stem = (
                f"rqkmeans_e4_bge_m3_{layout}_history10_query_gid_"
                "tiger_collision_v1"
            )
            for hardware, suffix in (
                ("4a100", "gpu4_a100_e3"),
                ("4x6000d", "gpu4_6000d_e3"),
            ):
                shell_path = (
                    PROJECT_ROOT
                    / "launchers"
                    / (
                        f"run_train_rqkmeans_e4_{layout}_sft_"
                        f"{hardware}_3epoch.sh"
                    )
                )
                shell_text = shell_path.read_text(encoding="utf-8")
                self.assertTrue(
                    shell_text.startswith(
                        "#!/usr/bin/env bash\nset -euo pipefail\n"
                    )
                )
                self.assertIn("export HADOOP_USER_NAME=map_search", shell_text)
                self.assertIn(
                    "conda activate /ofs/map_search/hudan/envs/poi-gr",
                    shell_text,
                )
                self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
                self.assertIn(f'stem="{stem}"', shell_text)
                self.assertIn(f'name="${{stem}}_{suffix}"', shell_text)
                self.assertIn("--num_epochs 3.0", shell_text)
                self.assertIn("--batch_size 16", shell_text)
                self.assertIn("--gradient_accumulation_steps 8", shell_text)
                self.assertIn("--nproc_per_node 4", shell_text)
                self.assertIn("--expected_global_batch_size 512", shell_text)
                self.assertIn("--cutoff_len 512", shell_text)
                self.assertIn('--save_strategy "epoch"', shell_text)
                self.assertIn('--eval_strategy "epoch"', shell_text)
                self.assertIn("--save_total_limit 3", shell_text)
                self.assertIn(
                    'test -f "${tokenized_path}/cache_manifest.json"',
                    shell_text,
                )
                self.assertIn(
                    f'runtime_tmp="./outputs/tmp/{runtime_tags[(layout, hardware)]}"',
                    shell_text,
                )
                self.assertIn(
                    'export TMPDIR="$(pwd)/${runtime_tmp#./}"',
                    shell_text,
                )
                self.assertIn('test "${#TMPDIR}" -le 64', shell_text)
                self.assertIn(
                    '"tokenizer_hash": inputs.get("extended_tokenizer_sha256")',
                    shell_text,
                )
                self.assertIn(
                    '"target_truncation": lengths.get(',
                    shell_text,
                )
                self.assertNotIn("TMPDIR=/tmp", shell_text)
                self.assertNotIn("nohup", shell_text)
                self.assertNotIn("setsid", shell_text)
                self.assertNotIn("disown", shell_text)
                self.assertFalse(
                    any(
                        line.rstrip().endswith(" &")
                        for line in shell_text.splitlines()
                    )
                )
                if hardware == "4a100":
                    self.assertIn('all("A100" in name', shell_text)
                else:
                    self.assertIn('"6000D" in name', shell_text)

    def test_ghr_four_6000d_launchers(self):
        candidates = {
            "exp09": (
                "ghr_exp09_g6_minimum_entity_dedup_history10_query_gid_v1",
                "ghr09d",
                "29731",
            ),
            "exp14": (
                "ghr_exp14_g6_relation_tree_dedup_history10_query_gid_v1",
                "ghr14d",
                "29732",
            ),
        }
        for variant, (stem, runtime_tag, port) in candidates.items():
            shell_path = (
                PROJECT_ROOT
                / "launchers"
                / f"run_train_ghr_{variant}_sft_4x6000d_3epoch.sh"
            )
            shell_text = shell_path.read_text(encoding="utf-8")
            def require(fragment: str) -> None:
                self.assertTrue(
                    fragment in shell_text,
                    f"{shell_path.name} missing required launcher fragment",
                )

            def forbid(fragment: str) -> None:
                self.assertFalse(
                    fragment in shell_text,
                    f"{shell_path.name} contains forbidden launcher fragment",
                )

            self.assertTrue(
                shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
            )
            for assignment in (
                "export HADOOP_USER_NAME=",
                "export HADOOP_USER_PASSWORD=",
            ):
                matches = [
                    line
                    for line in shell_text.splitlines()
                    if line.startswith(assignment)
                ]
                self.assertEqual(len(matches), 1)
                literal_value = matches[0][len(assignment) :]
                self.assertTrue(
                    literal_value
                    and "$" not in literal_value
                    and "REDACTED" not in literal_value,
                    f"{shell_path.name} requires a literal platform assignment",
                )
            require('conda_base="${CONDA_BASE:-$(conda info --base)}"')
            require('source "${conda_base}/etc/profile.d/conda.sh"')
            require("conda activate /ofs/map_search/hudan/envs/poi-gr")
            require("export CUDA_VISIBLE_DEVICES=0,1,2,3")
            require(f'stem="{stem}"')
            require('name="${stem}_gpu4_6000d_e3"')
            require(f'runtime_tmp="./outputs/tmp/{runtime_tag}"')
            require('export TMPDIR="$(pwd)/${runtime_tmp#./}"')
            require('test "${#TMPDIR}" -le 64')
            require('"6000D" in name')
            require("python3 scripts/sft/train.py")
            require("--num_epochs 3.0")
            require("--batch_size 16")
            require("--gradient_accumulation_steps 8")
            require("--nproc_per_node 4")
            require("--expected_global_batch_size 512")
            require("--cutoff_len 512")
            require(f"--master_port {port}")
            forbid("python_bin=")
            forbid("A100")
            forbid("nohup")
            forbid("setsid")
            forbid("disown")

    def test_rqkmeans_bge_1024_four_6000d_launcher(self):
        shell_path = (
            PROJECT_ROOT
            / "launchers"
            / "run_train_rqkmeans_bge_1024x1024x1024_sft_4x6000d_3epoch.sh"
        )
        shell_text = shell_path.read_text(encoding="utf-8")
        stem = (
            "rqkmeans_bge_m3_1024x1024x1024_history10_query_gid_"
            "tiger_collision_v1"
        )
        self.assertTrue(
            shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        )
        self.assertIn("conda activate /ofs/map_search/hudan/envs/poi-gr", shell_text)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
        self.assertIn('"6000D" in name', shell_text)
        self.assertIn(f'stem="{stem}"', shell_text)
        self.assertIn('name="${stem}_gpu4_6000d_e3"', shell_text)
        self.assertIn(
            'model_path="./models/'
            'Qwen3-0.6B-RQKMeans-BGE-1024x1024x1024-Vocab-v1"',
            shell_text,
        )
        self.assertIn('runtime_tmp="./outputs/tmp/rkb1024d"', shell_text)
        self.assertIn('export TMPDIR="$(pwd)/${runtime_tmp#./}"', shell_text)
        self.assertIn('test "${#TMPDIR}" -le 64', shell_text)
        self.assertIn("--num_epochs 3.0", shell_text)
        self.assertIn("--batch_size 16", shell_text)
        self.assertIn("--gradient_accumulation_steps 8", shell_text)
        self.assertIn("--nproc_per_node 4", shell_text)
        self.assertIn("--expected_global_batch_size 512", shell_text)
        self.assertIn("--master_port 0", shell_text)
        self.assertNotIn("A100", shell_text)
        self.assertNotIn("nohup", shell_text)
        self.assertNotIn("setsid", shell_text)
        self.assertNotIn("disown", shell_text)

    def test_rqvae_e4_1024_four_6000d_launcher(self):
        shell_path = (
            PROJECT_ROOT
            / "launchers"
            / "run_train_rqvae_e4_1024x1024x1024_sft_4x6000d_3epoch.sh"
        )
        shell_text = shell_path.read_text(encoding="utf-8")
        stem = (
            "rqvae_e4_bge_m3_1024x1024x1024_history10_query_gid_"
            "tiger_collision_v1"
        )
        self.assertTrue(
            shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        )
        self.assertIn("conda activate /ofs/map_search/hudan/envs/poi-gr", shell_text)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
        self.assertIn('"6000D" in name', shell_text)
        self.assertIn(f'stem="{stem}"', shell_text)
        self.assertIn('name="${stem}_gpu4_6000d_e3"', shell_text)
        self.assertIn(
            'model_path="./models/'
            'Qwen3-0.6B-RQVAE-E4-1024x1024x1024-Vocab-v1"',
            shell_text,
        )
        self.assertIn('runtime_tmp="./outputs/tmp/rqe41024d"', shell_text)
        self.assertIn('export TMPDIR="$(pwd)/${runtime_tmp#./}"', shell_text)
        self.assertIn('test "${#TMPDIR}" -le 64', shell_text)
        self.assertIn("--num_epochs 3.0", shell_text)
        self.assertIn("--batch_size 16", shell_text)
        self.assertIn("--gradient_accumulation_steps 8", shell_text)
        self.assertIn("--nproc_per_node 4", shell_text)
        self.assertIn("--expected_global_batch_size 512", shell_text)
        self.assertIn("--master_port 0", shell_text)
        self.assertNotIn("A100", shell_text)
        self.assertNotIn("nohup", shell_text)
        self.assertNotIn("setsid", shell_text)
        self.assertNotIn("disown", shell_text)

    def test_gnpr_four_6000d_epoch_checkpoint_protocol(self):
        shell_path = (
            PROJECT_ROOT
            / "launchers/run_train_gnpr_sft_4x6000d_3epoch.sh"
        )
        shell_text = shell_path.read_text(encoding="utf-8")
        self.assertTrue(shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
        self.assertNotIn("nohup", shell_text)
        self.assertNotIn("setsid", shell_text)
        self.assertNotIn("disown", shell_text)
        self.assertFalse(
            any(line.rstrip().endswith(" &") for line in shell_text.splitlines())
        )
        self.assertIn("torch.cuda.device_count()", shell_text)
        self.assertIn('--num_epochs 3.0', shell_text)
        self.assertIn('--batch_size 16', shell_text)
        self.assertIn('--gradient_accumulation_steps 8', shell_text)
        self.assertIn('--nproc_per_node 4', shell_text)
        self.assertIn('--expected_global_batch_size 512', shell_text)
        self.assertIn('--cutoff_len 512', shell_text)
        self.assertIn('--save_strategy "epoch"', shell_text)
        self.assertIn('--eval_strategy "epoch"', shell_text)
        self.assertIn('--save_total_limit 3', shell_text)
        self.assertIn(
            'name="gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_6000d_e3"',
            shell_text,
        )

        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model"
            tokenized_path = Path(temporary) / "tokenized"
            model_path.mkdir()
            tokenized_path.mkdir()
            (tokenized_path / "cache_manifest.json").write_text(
                json.dumps({"inputs": {"cutoff_len": 512}}),
                encoding="utf-8",
            )
            args = make_args(
                model_name_or_path=str(model_path),
                train_dataset=(
                    "gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_train"
                ),
                eval_dataset=(
                    "gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_valid"
                ),
                tokenized_path=str(tokenized_path),
                num_epochs=3.0,
                batch_size=16,
                eval_batch_size=16,
                gradient_accumulation_steps=8,
                nproc_per_node=4,
                save_strategy="epoch",
                eval_strategy="epoch",
                save_total_limit=3,
                cutoff_len=512,
            )
            resolved = validate_args(args, PROJECT_ROOT)
            config = build_training_config(args, resolved)
        self.assertEqual(resolved["global_batch_size"], 512)
        self.assertEqual(config["save_strategy"], "epoch")
        self.assertEqual(config["eval_strategy"], "epoch")
        self.assertNotIn("save_steps", config)
        self.assertNotIn("eval_steps", config)

    def test_tiger_mmbert_recall_128_four_6000d_launcher(self):
        stem = "tiger_mmbert_recall_128_1024x3_history10_query_gid_v1"
        shell_path = (
            PROJECT_ROOT
            / "launchers"
            / "run_train_tiger_mmbert_recall_128_1024x3_sft_4x6000d_3epoch.sh"
        )
        shell_text = shell_path.read_text(encoding="utf-8")
        self.assertTrue(shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
        self.assertNotIn("nohup", shell_text)
        self.assertNotIn("setsid", shell_text)
        self.assertNotIn("disown", shell_text)
        self.assertFalse(
            any(line.rstrip().endswith(" &") for line in shell_text.splitlines())
        )
        self.assertIn("torch.cuda.device_count()", shell_text)
        self.assertIn('--master_port 0', shell_text)
        self.assertIn('--num_epochs 3.0', shell_text)
        self.assertIn('--batch_size 8', shell_text)
        self.assertIn('--eval_batch_size 8', shell_text)
        self.assertIn('--gradient_accumulation_steps 16', shell_text)
        self.assertIn('--nproc_per_node 4', shell_text)
        self.assertIn('--expected_global_batch_size 512', shell_text)
        self.assertIn('--cutoff_len 1024', shell_text)
        self.assertIn('--save_strategy "epoch"', shell_text)
        self.assertIn('--eval_strategy "epoch"', shell_text)
        self.assertIn('--save_total_limit 3', shell_text)
        self.assertIn(f'stem="{stem}"', shell_text)
        self.assertIn('name="${stem}_gpu4_6000d_e3"', shell_text)

    def test_tiger_mmbert_recall_128_four_a100_launcher(self):
        stem = "tiger_mmbert_recall_128_1024x3_history10_query_gid_v1"
        shell_path = (
            PROJECT_ROOT
            / "launchers"
            / "run_train_tiger_mmbert_recall_128_1024x3_sft_4a100_3epoch.sh"
        )
        shell_text = shell_path.read_text(encoding="utf-8")
        self.assertTrue(shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
        self.assertIn('all("A100" in name.upper() for name in gpu_names)', shell_text)
        self.assertNotIn("6000D", shell_text)
        self.assertNotIn("nohup", shell_text)
        self.assertNotIn("setsid", shell_text)
        self.assertNotIn("disown", shell_text)
        self.assertFalse(
            any(line.rstrip().endswith(" &") for line in shell_text.splitlines())
        )
        self.assertIn("torch.cuda.device_count()", shell_text)
        self.assertIn('--master_port 0', shell_text)
        self.assertIn('--num_epochs 3.0', shell_text)
        self.assertIn('--batch_size 8', shell_text)
        self.assertIn('--eval_batch_size 8', shell_text)
        self.assertIn('--gradient_accumulation_steps 16', shell_text)
        self.assertIn('--nproc_per_node 4', shell_text)
        self.assertIn('--expected_global_batch_size 512', shell_text)
        self.assertIn('--cutoff_len 1024', shell_text)
        self.assertIn('--save_strategy "epoch"', shell_text)
        self.assertIn('--eval_strategy "epoch"', shell_text)
        self.assertIn('--save_total_limit 3', shell_text)
        self.assertIn(f'stem="{stem}"', shell_text)
        self.assertIn('name="${stem}_gpu4_a100_e3"', shell_text)

    def test_gnpr_four_a100_epoch_checkpoint_protocol(self):
        shell_path = (
            PROJECT_ROOT
            / "launchers/run_train_gnpr_sft_4a100_3epoch.sh"
        )
        shell_text = shell_path.read_text(encoding="utf-8")
        self.assertTrue(shell_text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", shell_text)
        self.assertNotIn("nohup", shell_text)
        self.assertNotIn("setsid", shell_text)
        self.assertNotIn("disown", shell_text)
        self.assertFalse(
            any(line.rstrip().endswith(" &") for line in shell_text.splitlines())
        )
        self.assertIn('all("A100" in name for name in gpu_names)', shell_text)
        self.assertIn('--num_epochs 3.0', shell_text)
        self.assertIn('--batch_size 16', shell_text)
        self.assertIn('--gradient_accumulation_steps 8', shell_text)
        self.assertIn('--nproc_per_node 4', shell_text)
        self.assertIn('--expected_global_batch_size 512', shell_text)
        self.assertIn('--cutoff_len 512', shell_text)
        self.assertIn('--save_strategy "epoch"', shell_text)
        self.assertIn('--eval_strategy "epoch"', shell_text)
        self.assertIn('--save_total_limit 3', shell_text)
        self.assertIn(
            'name="gnpr_bge_m3_category_pluscode6_512x3_history10_query_gid_v1_gpu4_a100_e3"',
            shell_text,
        )

    def test_epoch_checkpoint_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            for epoch, step in ((1, 100), (2, 200), (3, 300)):
                checkpoint = output_dir / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    __import__("json").dumps(
                        {"epoch": float(epoch), "global_step": step}
                    ),
                    encoding="utf-8",
                )
            path = write_epoch_checkpoint_index(output_dir, 3.0)
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["epoch"] for item in payload["checkpoints"]],
                [1, 2, 3],
            )


if __name__ == "__main__":
    unittest.main()

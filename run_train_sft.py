import os
import json
import sys
import argparse
import tempfile
from pathlib import Path


def set_args(argv=None):
    parser = argparse.ArgumentParser(description="Qwen3-0.6B 订单主任务全参数 SFT")
    parser.add_argument("--output_dir", type=str, required=True, help="模型保存路径")
    parser.add_argument("--experiment_name", type=str, default="qwen3_0.6b_main_v1", help="实验名字")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="扩词表后的模型路径")
    parser.add_argument("--dataset_dir", type=str, default="configs/sft", help="dataset_info.json 目录")
    parser.add_argument("--train_dataset", type=str, default="beijing_order_main_v1_train", help="训练集注册名")
    parser.add_argument("--eval_dataset", type=str, default="beijing_order_main_v1_valid", help="验证集注册名")
    parser.add_argument("--tokenized_path", type=str, required=True, help="Tokenized Cache 路径")
    parser.add_argument("--template", type=str, default="qwen3_nothink", help="LLaMA-Factory 模板")
    parser.add_argument("--num_epochs", type=float, default=1.0, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="单卡训练 Batch Size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="单卡验证 Batch Size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="当前节点使用的 GPU 数")
    parser.add_argument("--expected_global_batch_size", type=int, default=512, help="预期全局 Batch Size")
    parser.add_argument("--cutoff_len", type=int, default=128, help="最大序列长度")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight Decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup 比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16", help="训练精度")
    parser.add_argument("--packing", type=int, choices=[0, 1], default=1, help="是否启用 Packing")
    parser.add_argument("--train_on_prompt", type=int, choices=[0, 1], default=0, help="是否训练 Prompt")
    parser.add_argument("--gradient_checkpointing", type=int, choices=[0, 1], default=1, help="梯度检查点")
    parser.add_argument("--logging_steps", type=int, default=20, help="日志间隔")
    parser.add_argument("--save_steps", type=float, default=0.5, help="Checkpoint 保存间隔")
    parser.add_argument("--eval_steps", type=float, default=0.5, help="完整验证间隔")
    parser.add_argument("--save_total_limit", type=int, default=2, help="最多保留的 Checkpoint 数")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="本地监控后端")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--data_seed", type=int, default=42, help="数据随机种子")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="DataLoader 进程数")
    parser.add_argument("--preprocessing_num_workers", type=int, default=16, help="预处理进程数")
    parser.add_argument("--include_tokens_per_second", type=int, choices=[0, 1], default=1, help="统计 Tokens/s")
    parser.add_argument("--master_port", type=int, default=29500, help="单节点多卡通信端口")
    parser.add_argument("--resume", type=int, choices=[0, 1], default=0, help="是否恢复训练")
    parser.add_argument("--resume_path", type=str, default="", help="恢复用 Checkpoint 路径")
    parser.add_argument(
        "--llamafactory_path",
        type=str,
        default="./third_party/LLaMA-Factory",
        help="LLaMA-Factory 源码目录",
    )
    parser.add_argument("--dry_run", type=int, choices=[0, 1], default=0, help="只打印配置，不启动训练")
    return parser.parse_args(argv)


def resolve_path(project_root, raw_path):
    path = Path(raw_path).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def validate_args(args, project_root):
    if args.nproc_per_node <= 0:
        raise ValueError("nproc_per_node 必须大于 0")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch_size 和 gradient_accumulation_steps 必须大于 0")

    global_batch_size = (
        args.batch_size
        * args.gradient_accumulation_steps
        * args.nproc_per_node
    )
    if global_batch_size != args.expected_global_batch_size:
        raise ValueError(
            f"全局 Batch 不一致：{args.batch_size} × "
            f"{args.gradient_accumulation_steps} × {args.nproc_per_node} "
            f"= {global_batch_size}，预期 {args.expected_global_batch_size}"
        )
    if args.report_to != "tensorboard":
        raise ValueError("本任务只允许 report_to=tensorboard")
    if args.resume and not args.resume_path:
        raise ValueError("resume=1 时必须传入 resume_path")
    if not 1 <= args.master_port <= 65535:
        raise ValueError("master_port 必须位于 1～65535")

    model_path = resolve_path(project_root, args.model_name_or_path)
    dataset_dir = resolve_path(project_root, args.dataset_dir)
    tokenized_path = resolve_path(project_root, args.tokenized_path)
    llamafactory_path = resolve_path(project_root, args.llamafactory_path)
    required_paths = {
        "model_name_or_path": model_path,
        "dataset_dir": dataset_dir,
        "dataset_info.json": dataset_dir / "dataset_info.json",
        "tokenized_path": tokenized_path,
        "LLaMA-Factory": llamafactory_path / "src/llamafactory",
    }
    missing = [f"{name}={path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少正式训练输入：" + "；".join(missing))

    resume_path = None
    if args.resume:
        resume_path = resolve_path(project_root, args.resume_path)
        checkpoint_files = (
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            "rng_state.pth",
        )
        missing_checkpoint_files = [
            name for name in checkpoint_files if not (resume_path / name).is_file()
        ]
        if missing_checkpoint_files:
            raise FileNotFoundError(
                "恢复 Checkpoint 缺少完整训练状态："
                + "、".join(missing_checkpoint_files)
            )

    return {
        "global_batch_size": global_batch_size,
        "model_path": model_path,
        "dataset_dir": dataset_dir,
        "tokenized_path": tokenized_path,
        "llamafactory_path": llamafactory_path,
        "resume_path": resume_path,
    }


def build_training_config(args, resolved):
    output_dir = Path(args.output_dir).expanduser().resolve()
    config = {
        "model_name_or_path": str(resolved["model_path"]),
        "trust_remote_code": False,
        "resize_vocab": False,
        "stage": "sft",
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "full",
        "dataset": args.train_dataset,
        "eval_dataset": args.eval_dataset,
        "dataset_dir": str(resolved["dataset_dir"]),
        "tokenized_path": str(resolved["tokenized_path"]),
        "template": args.template,
        "enable_thinking": False,
        "cutoff_len": args.cutoff_len,
        "packing": bool(args.packing),
        "train_on_prompt": bool(args.train_on_prompt),
        "preprocessing_num_workers": args.preprocessing_num_workers,
        "dataloader_num_workers": args.dataloader_num_workers,
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "logging_dir": str(output_dir / "tensorboard"),
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_only_model": False,
        "save_safetensors": True,
        "plot_loss": True,
        "report_to": args.report_to,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "optim": "adamw_torch",
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "bf16": args.precision == "bf16",
        "fp16": args.precision == "fp16",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "disable_gradient_checkpointing": not bool(args.gradient_checkpointing),
        "include_tokens_per_second": bool(args.include_tokens_per_second),
        "include_num_input_tokens_seen": True,
        "skip_memory_metrics": False,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "load_best_model_at_end": False,
        "predict_with_generate": False,
        "compute_accuracy": False,
        "ddp_timeout": 7200,
    }
    if resolved["resume_path"] is not None:
        config["resume_from_checkpoint"] = str(resolved["resume_path"])
    return config


def save_resolved_config(args, config, resolved):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_name": args.experiment_name,
        "nproc_per_node": args.nproc_per_node,
        "global_batch_size": resolved["global_batch_size"],
        "master_port": args.master_port,
        "training_config": config,
    }
    path = output_dir / "resolved_config.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def launch_training(args):
    project_root = Path(__file__).resolve().parent
    resolved = validate_args(args, project_root)
    config = build_training_config(args, resolved)
    resolved_config_path = save_resolved_config(args, config, resolved)

    os.environ["NPROC_PER_NODE"] = str(args.nproc_per_node)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.nproc_per_node > 1:
        os.environ["FORCE_TORCHRUN"] = "1"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="poi_sft_",
        encoding="utf-8",
        delete=False,
    ) as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        runtime_config_path = Path(stream.name)

    print(f"Resolved config: {resolved_config_path}")
    print(
        f"Global Batch Size: {args.batch_size} × "
        f"{args.gradient_accumulation_steps} × {args.nproc_per_node} "
        f"= {resolved['global_batch_size']}"
    )
    print(
        "Training entry: "
        f"{resolved['llamafactory_path'] / 'src/llamafactory/launcher.py'} "
        f"{runtime_config_path}"
    )
    print(
        "TensorBoard: "
        f"tensorboard --logdir {config['logging_dir']} "
        "--host 127.0.0.1 --port 6006"
    )
    if args.dry_run:
        runtime_config_path.unlink(missing_ok=True)
        return 0

    source_path = str(resolved["llamafactory_path"] / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    current_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        source_path
        if not current_pythonpath
        else source_path + os.pathsep + current_pythonpath
    )

    from llamafactory import launcher

    original_argv = sys.argv
    sys.argv = [str(Path(__file__).resolve()), "train", str(runtime_config_path)]
    try:
        launcher.launch()
        return 0
    finally:
        sys.argv = original_argv
        runtime_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    args = set_args()
    raise SystemExit(launch_training(args))

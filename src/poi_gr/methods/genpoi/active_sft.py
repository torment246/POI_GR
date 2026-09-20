"""Platform CPU preparation and four-GPU training for the frozen active SID."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = "configs/sft/genpoi_active716k_pipeline_v1.yaml"
TRAIN_PATHS = ("model_name_or_path", "dataset_dir", "tokenized_path", "output_dir", "logging_dir")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def resolve(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"路径必须位于 poi_genret：{path}")
    return path


def file_state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def tree_state(directory: Path) -> dict[str, Any]:
    return {str(p.relative_to(directory)): file_state(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_protocol(path: Path) -> tuple[dict, dict, dict]:
    """Read small manifests and file metadata; plan never builds or scans JSONL."""
    protocol = yaml.safe_load(path.read_text())
    require(protocol["schema_version"] == "genpoi-active-platform-sft-v1", "平台协议版本不匹配")
    config_path = resolve(protocol["training_config"])
    config = yaml.safe_load(config_path.read_text())
    expected = {"cutoff_len": 1024, "packing": True, "train_on_prompt": False,
                "template": "qwen3_nothink", "bf16": True, "num_train_epochs": 3.0,
                "per_device_train_batch_size": 8, "gradient_accumulation_steps": 16,
                "finetuning_type": "full", "save_strategy": "epoch", "eval_strategy": "epoch"}
    require(all(config.get(k) == v for k, v in expected.items()), "必须保持四卡 3 轮、batch 512、1024 协议")
    for key in TRAIN_PATHS:
        config[key] = str(resolve(config[key]))
    require(len(str(resolve(protocol["tmp_dir"])).encode()) <= 64, "TMPDIR 超过 64 字节")
    stages = protocol["stages"]
    require([s["name"] for s in stages] == ["messages", "vocab", "cache"], "准备阶段顺序错误")
    require(resolve(stages[1]["output_dir"]) == Path(config["model_name_or_path"]), "扩词模型路径不一致")
    require(resolve(stages[2]["output_dir"]) == Path(config["tokenized_path"]), "缓存路径不一致")
    for stage in stages:
        require(resolve(stage["args"][0]).is_file(), "准备入口不存在")
        require(stage["args"][stage["args"].index("--output-dir") + 1] == stage["output_dir"], "阶段输出路径不一致")
    outputs = [resolve(s["output_dir"]) for s in stages] + [Path(config["output_dir"])]
    require(len(set(outputs)) == len(outputs), "数据、模型、缓存和训练输出必须隔离")
    require(all("genpoi" in str(p).lower() for p in outputs), "输出必须位于隔离的 GenPOI 路径")
    for name, digest in protocol["sources"].items():
        require(sha256(resolve(name)) == digest, f"冻结来源 manifest 发生变化：{name}")
    source = read_json(resolve(protocol["source_dir"]) / "manifest.json")
    require(source["status"] == "completed", "源 TIGER 数据未完成")
    watched = []
    for split in ("train", "valid", "test"):
        require(source["outputs"][f"{split}.jsonl"]["rows"] == protocol["rows"][split], "冻结划分行数不匹配")
        watched.append(resolve(protocol["source_dir"]) / f"{split}.jsonl")
    base = resolve(protocol["base_model_dir"])
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        watched.append(base / name)
    registry = read_json(Path(config["dataset_dir"]) / "dataset_info.json")
    data_dir = outputs[0]
    for split, key in (("train", "dataset"), ("valid", "eval_dataset")):
        entry = registry[config[key]]
        actual = (Path(config["dataset_dir"]) / entry["file_name"]).resolve()
        require(actual == data_dir / f"{split}.jsonl", "训练注册表没有指向本次 GenPOI 数据")
    code = [path, config_path, ROOT / "scripts/genpoi/active_sft.py", Path(__file__),
            ROOT / "src/poi_gr/methods/genpoi/aligned_data.py",
            ROOT / "src/poi_gr/methods/genpoi/data.py", ROOT / "src/poi_gr/sft/data.py",
            ROOT / "src/poi_gr/methods/tiger/data.py"]
    code += [resolve(s["args"][0]) for s in stages]
    contract = {"protocol": protocol, "training_config": config,
                "code_sha256": {str(p): sha256(p) for p in code},
                "source_file_state": {str(p): file_state(p) for p in watched},
                "dataset_registry": {config[k]: registry[config[k]] for k in ("dataset", "eval_dataset")}}
    return protocol, config, contract


def environment(protocol: dict) -> None:
    temporary = resolve(protocol["tmp_dir"])
    temporary.mkdir(parents=True, exist_ok=True)
    require(os.access(temporary, os.W_OK), "TMPDIR 不可写")
    os.environ.update(TMPDIR=str(temporary), HF_HOME=str(temporary / "hf"),
                      HF_DATASETS_CACHE=str(temporary / "datasets"),
                      XDG_CACHE_HOME=str(temporary / "cache"), MPLCONFIGDIR=str(temporary / "mpl"),
                      TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")


@contextmanager
def pipeline_lock(protocol: dict) -> Iterator[Path]:
    control = resolve(protocol["run_control_dir"])
    control.mkdir(parents=True, exist_ok=True)
    with (control / ".pipeline.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("另一进程正在准备或训练同一 GenPOI 任务") from error
        try:
            yield control
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def require_empty_training(config: dict) -> None:
    output = Path(config["output_dir"])
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            f"训练输出非空，拒绝覆盖或隐式续训：{output}")


def validate_pid(protocol: dict) -> None:
    import numpy as np
    directory = resolve(protocol["pid_dir"])
    manifest = read_json(directory / "final_pid_manifest.json")
    require(manifest["status"] == "completed" and manifest["poi_count"] == protocol["rows"]["poi"], "唯一 PID 未完成")
    for name, spec in manifest["outputs"].items():
        require(sha256(directory / name) == spec["sha256"], f"PID 产物哈希变化：{name}")
    codes = np.load(directory / "final_pid_codes.npy", mmap_mode="r")
    require(codes.shape == (protocol["rows"]["poi"], 10) and codes.dtype == np.int32, "PID 形状或类型错误")
    require(bool(np.all((codes[:, :6] >= 0) & (codes[:, :6] < 32))), "GID 码值越界")
    require(bool(np.all((codes[:, 6:9] >= 0) & (codes[:, 6:9] < 512))), "SID 码值越界")
    require(bool(np.all((codes[:, 9] >= -1) & (codes[:, 9] < 512))), "Dedup 码值越界")
    require(len(np.unique(codes, axis=0)) == len(codes), "最终 PID 不唯一")


def validate_messages(protocol: dict, directory: Path) -> None:
    data = read_json(directory / "manifest.json")
    source = read_json(resolve(protocol["source_dir"]) / "manifest.json")
    pid_path = resolve(protocol["pid_dir"]) / "final_pid_manifest.json"
    pid = read_json(pid_path)
    require((directory / "_SUCCESS").is_file() and data["status"] == "completed", "配对数据未完成")
    require(data["input"]["all_files_fully_scanned"] is True, "禁止用 smoke 数据训练")
    require(data["pid"]["manifest_sha256"] == sha256(pid_path)
            and data["pid"]["mapping_sha256"] == pid["outputs"]["poi_pid_mapping.parquet"]["sha256"], "数据与 PID 绑定错误")
    require(data["source_sft"]["manifest_sha256"] == sha256(resolve(protocol["source_dir"]) / "manifest.json"), "配对来源错误")
    require(all(data[k] == source[k] for k in ("user_identifier", "history", "time_split")), "用户哈希/历史/划分协议变化")
    require(data["stats"]["non_identifier_mismatch_count"] == data["stats"]["user_hash_mismatch_count"] == 0, "非标识输入不一致")
    for split in ("train", "valid", "test"):
        name = f"{split}.jsonl"
        require(data["outputs"][name]["rows"] == protocol["rows"][split], "配对数据行数错误")
        require(data["source_sft"]["files"][name]["sha256"] == source["outputs"][name]["sha256"], "源数据哈希不一致")
    for name, spec in data["outputs"].items():
        require(sha256(directory / name) == spec["sha256"], f"配对数据文件损坏：{name}")


def validate_cache(protocol: dict, config: dict) -> dict:
    """Reopen packed Arrow datasets and enforce full-input zero-truncation gates."""
    from datasets import load_from_disk
    directory = Path(config["tokenized_path"])
    manifest = read_json(directory / "cache_manifest.json")
    stats = read_json(directory / "length_stats.json")
    data = read_json(resolve(protocol["stages"][0]["output_dir"]) / "manifest.json")
    mapping = read_json(Path(config["model_name_or_path"]) / "poi_token_mapping.json")
    inputs = manifest["inputs"]
    expected = {"cutoff_len": 1024, "packing": True, "train_on_prompt": False,
                "template": config["template"], "train_dataset": config["dataset"],
                "valid_dataset": config["eval_dataset"],
                "model_dir": config["model_name_or_path"],
                "extended_tokenizer_sha256": mapping["extended_tokenizer_sha256"]}
    require(all(inputs.get(k) == v for k, v in expected.items()), "Cache 与训练/词表不一致")
    require(stats["effective_cutoff_len"] == 1024 and stats["over_requested_cutoff_count"] == 0
            and stats["target_truncated_at_requested_cutoff_count"] == 0
            and stats["target_truncated_at_effective_cutoff_count"] == 0, "未通过 1024 零截断门禁")
    for split in ("train", "valid"):
        require(stats["splits"][split]["rows"] == protocol["rows"][split], "长度预检不是全量数据")
        require(inputs[f"{split}_sha256"] == data["outputs"][f"{split}.jsonl"]["sha256"], "Cache 数据来源变化")
    cached = load_from_disk(str(directory))
    require(set(cached) == {"train", "validation"}, "Cache 只能包含 Train/Validation")
    for split in cached:
        rows = len(cached[split])
        require(rows == manifest["packed_rows"][split] and rows > 0, "Cache 重载行数不一致")
        for index in sorted({0, rows // 2, rows - 1}):
            sample = cached[split][index]
            require(len(sample["input_ids"]) == len(sample["labels"]) == 1024
                    and any(label != -100 for label in sample["labels"]), "Cache 长度或监督标签错误")
    return manifest["packed_rows"]


def verify_receipt(path: Path, contract: dict, directory: Path) -> bool:
    if not path.exists():
        return False
    receipt = read_json(path)
    require(receipt["contract"] == contract, "已完成阶段的来源或代码发生变化")
    require(receipt["status"] == "completed" and receipt["files"] == tree_state(directory),
            f"已完成阶段文件发生变化：{directory}")
    return True


def prepare(protocol: dict, config: dict, contract: dict) -> None:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "准备阶段须设置 CUDA_VISIBLE_DEVICES=''，由单 CPU 主进程执行")
    require(all(int(os.environ.get(k, "0")) <= 1 for k in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")), "不能在 torchrun 内准备数据")
    environment(protocol)
    with pipeline_lock(protocol) as control:
        require_empty_training(config)
        validate_pid(protocol)
        for stage in protocol["stages"]:
            name = stage["name"]
            directory = resolve(stage["output_dir"])
            receipt_path = control / f"{name}_ready.json"
            if verify_receipt(receipt_path, contract, directory):
                print(f"{name}：来源及文件状态一致，复用已完成产物。", flush=True)
                continue
            print(f"准备 {name}：{directory}", flush=True)
            # Messages publish atomically; a completed directory can be adopted after a full hash audit.
            if name != "messages" or not directory.exists():
                subprocess.run([sys.executable, *stage["args"]], cwd=ROOT, check=True)
            if name == "messages":
                validate_messages(protocol, directory)
            elif name == "vocab":
                mapping = read_json(directory / "poi_token_mapping.json")
                model = read_json(directory / "config.json")
                require((directory / "model.safetensors").is_file()
                        and model["vocab_size"] == mapping["new_vocab_size"]
                        and mapping["added_token_count"] == 4094
                        and mapping["schema_version"] == "genpoi-vocab-v1", "扩词模型不完整")
            elif name == "cache":
                validate_cache(protocol, config)
            require(bool(tree_state(directory)), f"阶段没有生成产物：{name}")
            write_json(receipt_path, {"status": "completed", "contract": contract,
                                     "files": tree_state(directory)})
        rows = validate_cache(protocol, config)
        write_json(control / "prepare_ready.json", {"status": "completed", "contract": contract, "packed_rows": rows})
    print("GENPOI_ACTIVE_PREPARATION_COMPLETED：配对数据、扩词模型、零截断预检和缓存已完成。", flush=True)


def hardware_check() -> list[dict]:
    import torch
    require(torch.cuda.device_count() == 4, "要求恰好四张可见 CUDA GPU")
    hardware = []
    for index in range(4):
        with torch.cuda.device(index):
            free, total = torch.cuda.mem_get_info(index)
            require(torch.cuda.is_bf16_supported() and free >= 36 * 1024**3,
                    f"GPU {index} 须支持 BF16 且至少 36 GiB 可用显存")
            hardware.append({"index": index, "name": torch.cuda.get_device_name(index),
                             "free_gib": free / 1024**3, "total_gib": total / 1024**3})
    return hardware


def train(protocol: dict, config: dict, contract: dict) -> None:
    environment(protocol)
    with pipeline_lock(protocol) as control:
        require_empty_training(config)
        ready = read_json(control / "prepare_ready.json")
        require(ready["status"] == "completed" and ready["contract"] == contract, "准备尚未验收")
        for stage in protocol["stages"]:
            require(verify_receipt(control / f'{stage["name"]}_ready.json', contract,
                                   resolve(stage["output_dir"])), "准备阶段缺少完成凭证")
        validate_pid(protocol)
        validate_cache(protocol, config)
        hardware = hardware_check()
        runtime = resolve(protocol["tmp_dir"]) / "genpoi_active_runtime.json"
        write_json(runtime, config)
        os.environ.update(FORCE_TORCHRUN="1", NPROC_PER_NODE="4", MASTER_ADDR="127.0.0.1")
        os.environ.pop("MASTER_PORT", None)
        sys.path.insert(0, str(ROOT / "third_party/LLaMA-Factory/src"))
        from llamafactory import launcher
        previous = sys.argv
        try:
            sys.argv = [str(Path(__file__)), "train", str(runtime)]
            print("准备已验收，开始 GenPOI 四卡 SFT：3 epoch，global batch=512。", flush=True)
            try:
                launcher.launch()
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
            state = read_json(Path(config["output_dir"]) / "trainer_state.json")
            require(float(state.get("epoch", 0)) >= float(config["num_train_epochs"])
                    and int(state.get("global_step", 0)) > 0, "训练退出但未完成指定轮数")
            write_json(Path(config["output_dir"]) / "genpoi_training_manifest.json",
                       {"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                        "contract": contract, "hardware": hardware, "global_batch_size": 512,
                        "epoch": state["epoch"], "global_step": state["global_step"]})
        finally:
            sys.argv = previous


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Active GenPOI：平台单进程数据准备，再启动四卡 SFT")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG), help="平台准备配置")
    parser.add_argument("--stage", choices=("plan", "prepare", "hardware", "train"), default="plan")
    args = parser.parse_args(argv)
    try:
        protocol, config, contract = load_protocol(resolve(args.config))
        if args.stage == "plan":
            print(json.dumps({"status": "planned_not_prepared", "training_gpus": 4,
                              "global_batch_size": 512, "epochs": 3, "cutoff_len": 1024,
                              "preparation_location": "training_platform_CPU",
                              "rows": protocol["rows"], "stages": [s["name"] for s in protocol["stages"]],
                              "existing_stages": {s["name"]: resolve(s["output_dir"]).exists() for s in protocol["stages"]},
                              "full_data_scanned": False, "training_output": config["output_dir"]},
                             ensure_ascii=False, indent=2))
        elif args.stage == "hardware":
            print(json.dumps(hardware_check(), ensure_ascii=False))
        elif args.stage == "prepare":
            prepare(protocol, config, contract)
        else:
            train(protocol, config, contract)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"GenPOI 平台流水线停止：{error}\n")
    return 0

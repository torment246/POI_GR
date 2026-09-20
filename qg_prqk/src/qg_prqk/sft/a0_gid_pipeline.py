"""Platform-side A0-GID preparation followed by the frozen four-GPU SFT protocol."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sft.a0_gid_data import (
    VARIANT,
    build_a0_identifiers,
    build_a0_messages,
    completed_stage,
    file_state,
    read_json,
)
from qg_prqk.sft.preflight import (
    _build_cache,
    _cache_inputs,
    _load_tokenizer_and_template,
    _validate_zero_truncation,
    scan_train_valid_lengths,
)
from qg_prqk.sft.training import _validate_algorithm_config, run_llamafactory_launcher
from qg_prqk.sft.vocabulary import (
    TOKENIZER_FILES,
    _hash_named_files,
    _verify_atomic_tokens,
)


DEFAULT_CONFIG = "qg_prqk/configs/sft/a0_gid_pipeline_v1.yaml"
PATH_KEYS = (
    "model_name_or_path",
    "dataset_dir",
    "tokenized_path",
    "output_dir",
    "logging_dir",
)
ALLOWED_CONFIG_DIFFERENCES = {
    "variant",
    "dataset",
    "eval_dataset",
    "dataset_dir",
    "tokenized_path",
    "output_dir",
    "logging_dir",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_qg(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root / "qg_prqk"):
        raise ValueError(f"路径必须位于 qg_prqk：{path}")
    return path


def validate_matched_config(
    config: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    """Change identifier/data paths only; keep A4's entire optimization protocol."""
    _validate_algorithm_config(config)
    if config.get("variant") != VARIANT or config.get("nproc_per_node") != 4:
        raise ValueError("本入口只支持 A0-GID 单机四卡")
    for key in set(config) | set(reference):
        if key not in ALLOWED_CONFIG_DIFFERENCES and config.get(key) != reference.get(
            key
        ):
            raise ValueError(f"A0/A4 训练协议不匹配：{key}")


def load_protocol(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], dict[str, Any]]:
    """Check small manifests and file headers only; never scan full data in plan mode."""
    root = project_root()
    path = resolve_qg(root, str(path))
    protocol = yaml.safe_load(path.read_text())
    if (
        protocol.get("schema_version") != "qg-prqk-a0-gid-platform-pipeline-v1"
        or protocol.get("variant") != VARIANT
    ):
        raise ValueError("不是 A0-GID 平台协议")
    training_path = resolve_qg(root, protocol["training_config"])
    reference_path = resolve_qg(root, protocol["reference_training_config"])
    config = yaml.safe_load(training_path.read_text())
    validate_matched_config(config, yaml.safe_load(reference_path.read_text()))
    paths = {key: resolve_qg(root, config[key]) for key in PATH_KEYS}
    paths.update(
        {
            key: resolve_qg(root, protocol[key])
            for key in ("final_id_dir", "run_control_dir", "tmp_dir", "poi_ids")
        }
    )
    if len(str(paths["tmp_dir"]).encode()) > 64:
        raise ValueError("TMPDIR 超过 64 字节")
    for key in (
        "dataset_dir",
        "tokenized_path",
        "output_dir",
        "logging_dir",
        "final_id_dir",
        "run_control_dir",
        "tmp_dir",
    ):
        if not paths[key].is_relative_to(root / "qg_prqk/outputs") or "a0" not in str(
            paths[key]
        ):
            raise ValueError(f"{key} 必须是隔离的 A0 输出路径")
    output_paths = [
        paths[key]
        for key in (
            "dataset_dir",
            "tokenized_path",
            "output_dir",
            "final_id_dir",
            "run_control_dir",
            "tmp_dir",
        )
    ]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("准备/缓存/训练输出不能使用相同目录")
    sources = {}
    for key, spec in protocol["sources"].items():
        source = resolve_qg(root, spec["path"])
        if sha256_file(source) != spec["sha256"]:
            raise ValueError(f"冻结来源 manifest 变化：{source}")
        paths[key] = source
        sources[key] = {"path": str(source), "sha256": spec["sha256"]}
    if paths["shared_token_mapping"].parent != paths["model_name_or_path"]:
        raise ValueError("必须复用 A4 共同扩词初始模型")
    if protocol["rows"] != {"poi": 716245, "train": 7586410, "valid": 597421}:
        raise ValueError("A0 对照必须使用冻结全库和相同 Train/Valid 行数")
    source_data = read_json(paths["a4_data_manifest"])
    watched = [paths["poi_ids"], paths["model_name_or_path"] / "model.safetensors"]
    for split in ("train", "valid"):
        file = paths["a4_data_manifest"].parent / f"{split}.jsonl"
        if file.stat().st_size != source_data["outputs"][file.name]["bytes"]:
            raise ValueError(f"A4 {split} 大小不匹配")
        watched.append(file)
    for key in ("a0_manifest", "a4_identifier_manifest"):
        manifest = read_json(paths[key])
        names = (
            ["poi_assignments_s1_s2_s3.npy", "selected_poi_rows.npy"]
            if key == "a0_manifest"
            else [
                "base_identifier_codes.npy",
                "dedup_codes.npy",
                "requires_dedup.npy",
                "poi_final_id_mapping.parquet",
            ]
        )
        for name in names:
            file = paths[key].parent / name
            if file.stat().st_size != manifest["artifacts"][name]["bytes"]:
                raise ValueError(f"冻结产物大小不匹配：{file}")
            watched.append(file)
    watched.extend(
        file
        for name in TOKENIZER_FILES
        if (file := paths["model_name_or_path"] / name).is_file()
    )
    source_files = [
        path,
        training_path,
        reference_path,
        root / "qg_prqk/scripts/a0_gid_sft.py",
    ]
    source_files.extend(
        root / f"qg_prqk/src/qg_prqk/{name}"
        for name in (
            "sft/a0_gid_data.py",
            "sft/a0_gid_pipeline.py",
            "sft/preflight.py",
            "sft/training.py",
            "sft/vocabulary.py",
            "sft/data.py",
            "sid/identifiers.py",
            "artifacts.py",
        )
    )
    contract = {
        "variant": VARIANT,
        "sources": sources,
        "code_and_config_sha256": {
            str(file): sha256_file(file) for file in source_files
        },
        "immutable_source_files": {str(file): file_state(file) for file in watched},
        "rows": protocol["rows"],
        "business_test_read": False,
    }
    return protocol, config, paths, contract


def check_environment(paths: Mapping[str, Path]) -> None:
    """Use short project-local caches, never a worker's home or system temp."""
    temporary = paths["tmp_dir"]
    temporary.mkdir(parents=True, exist_ok=True)
    if not os.access(temporary, os.W_OK):
        raise ValueError("TMPDIR 不可写")
    os.environ.update(
        TMPDIR=str(temporary),
        TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(temporary / "hf"),
        HF_DATASETS_CACHE=str(temporary / "datasets"),
        XDG_CACHE_HOME=str(temporary / "cache"),
        MPLCONFIGDIR=str(temporary / "mpl"),
    )


@contextmanager
def stage_lock(paths: Mapping[str, Path]) -> Iterator[None]:
    """Prevent concurrent launchers from preparing or training the same control."""
    paths["run_control_dir"].mkdir(parents=True, exist_ok=True)
    with (paths["run_control_dir"] / ".pipeline.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("另一进程正在准备或训练同一 A0-GID 任务") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_cache_receipt(
    paths: Mapping[str, Path], expected: Mapping[str, Any]
) -> dict[str, Any] | None:
    receipt_path = paths["run_control_dir"] / "cache_ready.json"
    if not receipt_path.exists():
        return None
    receipt = read_json(receipt_path)
    if receipt.get("inputs") != dict(expected):
        raise ValueError("既有缓存凭证与当前输入不一致")
    cache = paths["tokenized_path"]
    actual_names = {
        str(file.relative_to(cache)) for file in cache.rglob("*") if file.is_file()
    }
    if actual_names != set(receipt["files"]):
        raise ValueError("Tokenized Cache 文件集合变化")
    for name, spec in receipt["files"].items():
        file = cache / name
        if not file.resolve().is_relative_to(cache) or file_state(file) != spec["stat"]:
            raise ValueError(f"缓存文件发生变化：{file}")
    if not (cache / "_SUCCESS").is_file():
        raise ValueError("Tokenized Cache 缺少成功标记")
    return receipt


def cache_expected(
    paths: Mapping[str, Path], contract: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "contract": dict(contract),
        "data_manifest_sha256": sha256_file(paths["dataset_dir"] / "manifest.json"),
        "identifier_manifest_sha256": sha256_file(
            paths["final_id_dir"] / "manifest.json"
        ),
    }


def prepare_cache(
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected = cache_expected(paths, contract)
    existing = verify_cache_receipt(paths, expected)
    if existing is not None:
        print(
            "A0-GID：缓存已完成且来源/文件状态一致，跳过预检与 Token 化。", flush=True
        )
        return existing
    model = paths["model_name_or_path"]
    mapping = read_json(paths["shared_token_mapping"])
    if (
        _hash_named_files(model, TOKENIZER_FILES)
        != mapping["extended_tokenizer_sha256"]
    ):
        raise ValueError("共同 tokenizer 实际文件哈希变化")
    tokenizer, template, module = _load_tokenizer_and_template(model)
    _verify_atomic_tokens(tokenizer, list(mapping["tokens"]), mapping["tokens"])
    preflight_path = paths["run_control_dir"] / "token_preflight.json"
    if preflight_path.exists():
        state = read_json(preflight_path)
        if state.get("inputs") != expected:
            raise ValueError("已保存长度预检与当前输入不匹配")
        stats = state["stats"]
    else:
        print(
            "A0-GID：在平台 CPU 上执行 Train/Valid 全量 1024 零截断预检。", flush=True
        )
        stats = scan_train_valid_lengths(
            data_dir=paths["dataset_dir"],
            variant=VARIANT,
            manifest=read_json(paths["dataset_dir"] / "manifest.json"),
            tokenizer=tokenizer,
            template=template,
            cutoff_len=1024,
            batch_size=int(protocol["cache"]["scan_batch_size"]),
        )
        write_json_atomic(preflight_path, {"inputs": expected, "stats": stats})
    _validate_zero_truncation(stats)
    inputs = _cache_inputs(
        variant=VARIANT,
        model_dir=model,
        mapping=mapping,
        stats=stats,
        train_dataset=config["dataset"],
        valid_dataset=config["eval_dataset"],
    )
    print("A0-GID：在平台 CPU 上生成 LLaMA-Factory packed Cache。", flush=True)
    manifest = _build_cache(
        model_dir=model,
        dataset_dir=paths["dataset_dir"],
        output_dir=paths["tokenized_path"],
        inputs=inputs,
        stats=stats,
        tokenizer=tokenizer,
        template=template,
        tokenizer_module=module,
        workers=int(protocol["cache"]["workers"]),
        preprocessing_batch_size=int(protocol["cache"]["preprocessing_batch_size"]),
    )
    if manifest.get("status") != "completed" or manifest.get("inputs") != inputs:
        raise ValueError("缓存没有按同一配置完成")
    _validate_zero_truncation(manifest["preflight"])
    receipt = {
        "status": "completed",
        "built_at": utc_now(),
        "inputs": expected,
        "packed_rows": manifest["packed_rows"],
        "initial_model_sha256": sha256_file(model / "model.safetensors"),
        "files": {
            str(file.relative_to(paths["tokenized_path"])): {
                "sha256": sha256_file(file),
                "stat": file_state(file),
            }
            for file in sorted(paths["tokenized_path"].rglob("*"))
            if file.is_file()
        },
    }
    write_json_atomic(paths["run_control_dir"] / "cache_ready.json", receipt)
    return receipt


def prepare(
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    contract: Mapping[str, Any],
) -> None:
    """Prepare on one CPU driver before any distributed trainer is launched."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError(
            "准备阶段请设置 CUDA_VISIBLE_DEVICES=''，由一个 CPU 主进程执行"
        )
    if any(
        int(os.environ.get(key, "0")) > 1 for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")
    ):
        raise ValueError("不能在 torchrun 多进程内构造数据")
    check_environment(paths)
    with stage_lock(paths):
        print("[1/3] A0-GID：构造或复用 GID6+A0-SID3+[D] 全目录。", flush=True)
        build_a0_identifiers(
            a0_dir=paths["a0_manifest"].parent,
            a4_id_dir=paths["a4_identifier_manifest"].parent,
            poi_ids_path=paths["poi_ids"],
            token_mapping=read_json(paths["shared_token_mapping"]),
            output=paths["final_id_dir"],
            contract=contract,
            expected_rows=int(protocol["rows"]["poi"]),
        )
        print("[2/3] A0-GID：构造或复用 identifier-only Train/Valid。", flush=True)
        build_a0_messages(
            source_dir=paths["a4_data_manifest"].parent,
            a4_id_dir=paths["a4_identifier_manifest"].parent,
            a0_id_dir=paths["final_id_dir"],
            output=paths["dataset_dir"],
            contract=contract,
            datasets={"train": config["dataset"], "valid": config["eval_dataset"]},
            expected_rows=protocol["rows"],
        )
        print("[3/3] A0-GID：零截断预检与 LLaMA-Factory Cache。", flush=True)
        prepare_cache(protocol, config, paths, contract)
    print("A0_GID_PREPARATION_COMPLETED：准备完成，可启动四卡训练。", flush=True)


def hardware_check() -> list[dict[str, Any]]:
    """Require four usable BF16 GPUs without vendor-model string matching."""
    import torch

    if torch.cuda.device_count() != 4:
        raise ValueError("要求恰好四张可见 CUDA GPU")
    result = []
    for index in range(4):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(f"GPU {index} 不支持当前冻结 BF16 协议")
            free, total = torch.cuda.mem_get_info(index)
            if free < 28 * 1024**3:
                raise ValueError(f"GPU {index} 可用显存少于 28 GiB，不自动改变 batch")
            result.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "free_gib": free / 1024**3,
                    "total_gib": total / 1024**3,
                }
            )
    return result


def train(
    config: Mapping[str, Any], paths: Mapping[str, Path], contract: Mapping[str, Any]
) -> None:
    """Launch the unchanged LLaMA-Factory SFT backend after all preparation gates."""
    check_environment(paths)
    with stage_lock(paths):
        for key in ("final_id_dir", "dataset_dir"):
            if completed_stage(paths[key], contract) is None:
                raise ValueError("请先在平台执行 prepare")
        receipt = verify_cache_receipt(paths, cache_expected(paths, contract))
        if receipt is None:
            raise ValueError("缓存未验收，禁止直接让四个 worker 构建缓存")
        output = paths["output_dir"]
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"训练输出非空，拒绝覆盖或隐式续训：{output}")
        hardware = hardware_check()
        runtime = {
            key: value
            for key, value in config.items()
            if key not in ("variant", "nproc_per_node")
        }
        runtime.update({key: str(paths[key]) for key in PATH_KEYS})
        runtime_path = paths["tmp_dir"] / "a0_gid_runtime.json"
        write_json_atomic(runtime_path, runtime, overwrite=True)
        os.environ.update(
            FORCE_TORCHRUN="1",
            NPROC_PER_NODE="4",
            MASTER_ADDR="127.0.0.1",
            PYTHONUNBUFFERED="1",
        )
        os.environ.pop("MASTER_PORT", None)
        sys.path.insert(0, str(project_root() / "third_party/LLaMA-Factory/src"))
        from llamafactory import launcher

        old_argv = sys.argv
        try:
            sys.argv = [str(Path(__file__)), "train", str(runtime_path)]
            print(
                "A0-GID：准备已验收，开始四卡 SFT，global batch=512，epoch=3。",
                flush=True,
            )
            run_llamafactory_launcher(launcher.launch)
            write_json_atomic(
                output / "qg_prqk_training_manifest.json",
                {
                    "schema_version": "qg-prqk-sft-training-v1",
                    "status": "completed",
                    "variant": VARIANT,
                    "completed_at": utc_now(),
                    "contract": dict(contract),
                    "runtime_config": runtime,
                    "hardware": hardware,
                    "global_batch_size": 512,
                    "initial_model_sha256": receipt["initial_model_sha256"],
                    "cache_manifest_sha256": sha256_file(
                        paths["tokenized_path"] / "cache_manifest.json"
                    ),
                    "data_manifest_sha256": sha256_file(
                        paths["dataset_dir"] / "manifest.json"
                    ),
                },
            )
        finally:
            sys.argv = old_argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A0-GID 平台流水线：单进程准备后四卡训练；开发机只运行 plan。"
    )
    parser.add_argument(
        "--config", type=Path, default=Path(DEFAULT_CONFIG), help="冻结平台准备协议"
    )
    parser.add_argument(
        "--stage",
        choices=("plan", "prepare", "train", "hardware"),
        default="plan",
        help="默认只查合同，不构造数据",
    )
    args = parser.parse_args(argv)
    try:
        protocol, config, paths, contract = load_protocol(args.config)
        if args.stage == "plan":
            print(
                json.dumps(
                    {
                        "status": "planned_not_prepared",
                        "variant": VARIANT,
                        "preparation_location": "training_platform_CPU",
                        "training_gpus": 4,
                        "rows": protocol["rows"],
                        "outputs": {
                            key: str(paths[key])
                            for key in (
                                "final_id_dir",
                                "dataset_dir",
                                "tokenized_path",
                                "output_dir",
                            )
                        },
                        "existing_stages": {
                            key: paths[key].exists()
                            for key in ("final_id_dir", "dataset_dir", "tokenized_path")
                        },
                        "full_data_scanned": False,
                        "business_test_read": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.stage == "hardware":
            print(json.dumps(hardware_check(), ensure_ascii=False))
        elif args.stage == "prepare":
            prepare(protocol, config, paths, contract)
        else:
            train(config, paths, contract)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"A0-GID 流水线停止：{error}\n")
    return 0

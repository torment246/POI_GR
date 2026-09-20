"""Validated LLaMA-Factory launch for both SFT variants."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from qg_prqk.artifacts import sha256_file, utc_now
from qg_prqk.sft.data import SFT_DATA_SCHEMA_VERSION
from qg_prqk.sft.preflight import CACHE_SCHEMA_VERSION, REQUIRED_CUTOFF_LEN
from qg_prqk.sft.vocabulary import MAPPING_FILENAME, VOCAB_SCHEMA_VERSION


VARIANT_CONFIGS = {
    "a4_gid_parent": "a4_gid_parent_history10_v1.yaml",
    "a4_nogid": "a4_nogid_history10_v1.yaml",
}
VARIANT_DATA_DIRS = {
    "a4_gid_parent": "a4_gid_parent_order_a_history10_v1",
    "a4_nogid": "a4_nogid_sid3_history10_v1",
}
VARIANT_TMP_TAGS = {
    "a4_gid_parent": "qgsgid",
    "a4_nogid": "qgsngd",
}
TRAINING_MANIFEST_SCHEMA_VERSION = "qg-prqk-sft-training-v1"


class SftTrainingError(ValueError):
    """Raised when a requested SFT run is not reproducible or isolated."""


def run_llamafactory_launcher(launch) -> None:
    """Continue postprocessing after a successful distributed SystemExit."""

    try:
        launch()
    except SystemExit as error:
        if error.code not in (None, 0):
            raise


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SftTrainingError(f"{name} 不是合法 JSON：{path}") from error
    if not isinstance(payload, dict):
        raise SftTrainingError(f"{name} 必须是 JSON object")
    return payload


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_training_config(variant: str, root: Path | None = None) -> dict[str, Any]:
    """Load the fixed per-variant YAML and reject unexpected algorithm drift."""

    if variant not in VARIANT_CONFIGS:
        raise SftTrainingError(f"未知 variant：{variant}")
    root = (root or project_root()).resolve()
    path = root / "qg_prqk/configs/sft" / VARIANT_CONFIGS[variant]
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SftTrainingError(f"训练配置无效：{path}") from error
    if not isinstance(config, dict) or config.get("variant") != variant:
        raise SftTrainingError("训练配置 variant 不一致")
    _validate_algorithm_config(config)
    return config


def _validate_algorithm_config(config: Mapping[str, Any]) -> None:
    expected = {
        "stage": "sft",
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "full",
        "trust_remote_code": False,
        "resize_vocab": False,
        "template": "qwen3_nothink",
        "enable_thinking": False,
        "cutoff_len": REQUIRED_CUTOFF_LEN,
        "packing": True,
        "train_on_prompt": False,
        "num_train_epochs": 3.0,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5.0e-5,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "seed": 42,
        "data_seed": 42,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "report_to": "tensorboard",
        "overwrite_output_dir": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SftTrainingError(
                f"配置 {key}={config.get(key)!r}，冻结值应为 {value!r}"
            )
    nproc = config.get("nproc_per_node")
    if isinstance(nproc, bool) or not isinstance(nproc, int) or nproc <= 0:
        raise SftTrainingError("nproc_per_node 必须为正整数")
    global_batch = (
        int(config["per_device_train_batch_size"])
        * int(config["gradient_accumulation_steps"])
        * nproc
    )
    if global_batch != 512:
        raise SftTrainingError(f"Global Batch 必须为 512，实际为 {global_batch}")
    if "test" in str(config.get("dataset", "")).lower() or "test" in str(
        config.get("eval_dataset", "")
    ).lower():
        raise SftTrainingError("训练配置禁止注册 Test")


def validate_training_inputs(
    variant: str, config: Mapping[str, Any], root: Path | None = None
) -> dict[str, Any]:
    """Validate frozen raw data, shared vocabulary, cache and path isolation."""

    root = (root or project_root()).resolve()
    qg_root = (root / "qg_prqk").resolve()
    data_dir = qg_root / "outputs/sft_data" / VARIANT_DATA_DIRS[variant]
    data_manifest_path = data_dir / "manifest.json"
    data_manifest = _load_json(data_manifest_path, "SFT 数据 manifest")
    if (
        data_manifest.get("schema_version") != SFT_DATA_SCHEMA_VERSION
        or data_manifest.get("status") != "completed"
        or data_manifest.get("scan_mode") != "full"
        or data_manifest.get("variant") != variant
        or not (data_dir / "_SUCCESS").is_file()
    ):
        raise SftTrainingError("SFT 数据不是对应 variant 的已完成全量产物")

    model_dir = _resolve(root, str(config["model_name_or_path"]))
    cache_dir = _resolve(root, str(config["tokenized_path"]))
    dataset_dir = _resolve(root, str(config["dataset_dir"]))
    output_dir = _resolve(root, str(config["output_dir"]))
    logging_dir = _resolve(root, str(config["logging_dir"]))
    for name, path in {
        "model": model_dir,
        "cache": cache_dir,
        "dataset_dir": dataset_dir,
        "output": output_dir,
        "logging": logging_dir,
    }.items():
        if not _inside(path, qg_root):
            raise SftTrainingError(f"{name} 必须位于 qg_prqk：{path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SftTrainingError(f"输出目录非空，拒绝覆盖或隐式续训：{output_dir}")

    registry = _load_json(dataset_dir / "dataset_info.json", "dataset_info")
    for key in (str(config["dataset"]), str(config["eval_dataset"])):
        if key not in registry:
            raise SftTrainingError(f"dataset_info 缺少注册项：{key}")
    if any("test" in key.lower() for key in registry):
        raise SftTrainingError("QG SFT dataset_info 不得注册 Test")

    vocab = _load_json(model_dir / MAPPING_FILENAME, "扩词表映射")
    if vocab.get("schema_version") != VOCAB_SCHEMA_VERSION:
        raise SftTrainingError("扩词表不符合 QG-PRQK 共同词表契约")
    token_paths = [Path(value) for value in vocab.get("token_source_paths", [])]
    token_hashes = vocab.get("token_source_sha256")
    if len(token_paths) != 2 or not isinstance(token_hashes, list):
        raise SftTrainingError("共同词表缺少两套 special_tokens 来源")
    for path, expected_hash in zip(token_paths, token_hashes, strict=True):
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise SftTrainingError(f"共同词表 Token 来源已变化：{path}")

    cache = _load_json(cache_dir / "cache_manifest.json", "Cache manifest")
    inputs = cache.get("inputs")
    outputs = data_manifest.get("outputs")
    if (
        cache.get("schema_version") != CACHE_SCHEMA_VERSION
        or cache.get("status") != "completed"
        or not (cache_dir / "_SUCCESS").is_file()
        or not isinstance(inputs, dict)
        or inputs.get("variant") != variant
        or inputs.get("cutoff_len") != REQUIRED_CUTOFF_LEN
        or inputs.get("train_sha256") != outputs["train.jsonl"]["sha256"]
        or inputs.get("valid_sha256") != outputs["valid.jsonl"]["sha256"]
        or inputs.get("extended_tokenizer_sha256")
        != vocab.get("extended_tokenizer_sha256")
        or inputs.get("train_dataset") != config["dataset"]
        or inputs.get("valid_dataset") != config["eval_dataset"]
    ):
        raise SftTrainingError("Tokenized Cache 与当前 variant、数据、词表或配置不一致")
    preflight = cache.get("preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("over_cutoff_count") != 0
        or preflight.get("target_truncated_count") != 0
    ):
        raise SftTrainingError("全量 Train/Valid 未通过零截断门禁")
    return {
        "qg_root": qg_root,
        "data_dir": data_dir,
        "data_manifest_path": data_manifest_path,
        "model_dir": model_dir,
        "cache_dir": cache_dir,
        "cache_manifest_path": cache_dir / "cache_manifest.json",
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "nproc_per_node": int(config["nproc_per_node"]),
        "global_batch_size": 512,
    }


def _runtime_config(config: Mapping[str, Any], resolved: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(config)
    runtime.pop("variant", None)
    runtime.pop("nproc_per_node", None)
    for key in (
        "model_name_or_path",
        "tokenized_path",
        "dataset_dir",
        "output_dir",
        "logging_dir",
    ):
        runtime[key] = str(resolved[
            {
                "model_name_or_path": "model_dir",
                "tokenized_path": "cache_dir",
                "dataset_dir": "dataset_dir",
                "output_dir": "output_dir",
                "logging_dir": "logging_dir",
            }[key]
        ])
    return runtime


def run_sft_variant(variant: str, *, dry_run: bool = False) -> int:
    """Validate and launch one fixed variant; never reads business Test."""

    root = project_root()
    config = load_training_config(variant, root)
    resolved = validate_training_inputs(variant, config, root)
    runtime = _runtime_config(config, resolved)
    summary = {
        "variant": variant,
        "global_batch_size": resolved["global_batch_size"],
        "nproc_per_node": resolved["nproc_per_node"],
        "cutoff_len": runtime["cutoff_len"],
        "data_manifest_sha256": sha256_file(resolved["data_manifest_path"]),
        "cache_manifest_sha256": sha256_file(resolved["cache_manifest_path"]),
        "runtime_config": runtime,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if dry_run:
        return 0

    temporary_dir = (
        resolved["qg_root"] / "outputs/tmp" / VARIANT_TMP_TAGS[variant]
    ).resolve()
    if len(str(temporary_dir)) > 64:
        raise SftTrainingError(
            f"TMPDIR 路径超过 64 字节门禁：{temporary_dir}"
        )
    temporary_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "TMPDIR": str(temporary_dir),
            "NPROC_PER_NODE": str(resolved["nproc_per_node"]),
            "MASTER_ADDR": "127.0.0.1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "FORCE_TORCHRUN": "1",
        }
    )
    os.environ.pop("MASTER_PORT", None)
    third_party_src = root / "third_party/LLaMA-Factory/src"
    if not (third_party_src / "llamafactory").is_dir():
        raise SftTrainingError(f"缺少本地 LLaMA-Factory：{third_party_src}")
    sys.path.insert(0, str(third_party_src))

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{variant}_",
        dir=temporary_dir,
        encoding="utf-8",
        delete=False,
    ) as stream:
        json.dump(runtime, stream, ensure_ascii=False, indent=2)
        runtime_path = Path(stream.name)
    original_argv = sys.argv
    try:
        from llamafactory import launcher

        sys.argv = [str(Path(__file__).resolve()), "train", str(runtime_path)]
        run_llamafactory_launcher(launcher.launch)
        manifest = {
            "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
            "status": "completed",
            "completed_at": utc_now(),
            **summary,
        }
        (resolved["output_dir"] / "qg_prqk_training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    finally:
        sys.argv = original_argv
        runtime_path.unlink(missing_ok=True)

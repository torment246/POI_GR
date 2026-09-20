#!/usr/bin/env python3
"""Validate full Train/Valid dynamic templates before TIGER-Joint training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger_joint import (  # noqa: E402
    DynamicPreflightTemplate,
    PoiEmbeddingStore,
    TigerJointPreparationError,
    TigerJointPreflightError,
    combine_dynamic_preflight_splits,
    build_fresh_initialization_contract,
    load_dynamic_token_ids,
    load_fresh_tokenizer_and_template,
    scan_dynamic_preflight_split,
)
from poi_gr.pid.dedup import sha256_file  # noqa: E402


SCHEMA_VERSION = "tiger-joint-sid-free-preflight-v2"
CUTOFF_LEN = 1024


class DynamicPreflightCliError(TigerJointPreflightError):
    """Raised when CLI artifacts do not satisfy the J1 contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "流式扫描 SID-free TIGER-Joint Train/Valid，验证显式 BGE 行号、"
            "动态三级 SID 槽位和 1024 Token 零截断门槛；不会读取 Test。"
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/Qwen3-0.6B"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/tiger_joint/data/sid_free_v1"),
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("outputs/embeddings/beijing_poi_bge_m3"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="仅调试用 Train 前缀行数；正式预检不要设置。",
    )
    parser.add_argument(
        "--max-valid-rows",
        type=int,
        default=None,
        help="仅调试用 Valid 前缀行数；必须与 --max-train-rows 同时设置。",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise DynamicPreflightCliError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DynamicPreflightCliError(f"{name} 不是合法 JSON") from error
    if not isinstance(value, dict):
        raise DynamicPreflightCliError(f"{name} 必须是 JSON object")
    return value


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.progress_every <= 0:
        raise DynamicPreflightCliError("batch_size/progress_every 必须为正数")
    prefix_limits = (args.max_train_rows, args.max_valid_rows)
    if (prefix_limits[0] is None) != (prefix_limits[1] is None):
        raise DynamicPreflightCliError(
            "max_train_rows 与 max_valid_rows 必须同时设置或同时省略"
        )
    if any(value is not None and value <= 0 for value in prefix_limits):
        raise DynamicPreflightCliError("前缀行数必须为正数")


def load_data_contract(
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = data_dir / "manifest.json"
    manifest = load_json_object(manifest_path, "数据 manifest")
    if manifest.get("schema_version") != "tiger-joint-sid-free-data-v1":
        raise DynamicPreflightCliError("数据 schema_version 不兼容")
    isolation = manifest.get("isolation")
    if not isinstance(isolation, dict) or any(
        isolation.get(name) is not False
        for name in (
            "old_tiger_jsonl_loaded",
            "old_sid_mapping_loaded",
            "old_rqvae_checkpoint_loaded",
            "old_qwen_tiger_checkpoint_loaded",
        )
    ):
        raise DynamicPreflightCliError("数据 manifest 未证明与旧 SID 隔离")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise DynamicPreflightCliError("数据 manifest 缺少 outputs")
    if "test.jsonl" in outputs or (data_dir / "test.jsonl").exists():
        raise DynamicPreflightCliError("联合训练数据目录不得包含 test.jsonl")
    contracts: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        file_name = f"{split}.jsonl"
        spec = outputs.get(file_name)
        path = data_dir / file_name
        if (
            not isinstance(spec, dict)
            or isinstance(spec.get("rows"), bool)
            or not isinstance(spec.get("rows"), int)
            or spec["rows"] <= 0
            or not isinstance(spec.get("sha256"), str)
            or len(spec["sha256"]) != 64
            or not path.is_file()
        ):
            raise DynamicPreflightCliError(f"{file_name} 契约无效")
        contracts[split] = {
            "path": path,
            "rows": spec["rows"],
            "sha256": spec["sha256"],
        }
    return manifest, contracts


def git_state() -> dict[str, Any]:
    def run_git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        revision = run_git("rev-parse", "HEAD")
        status = run_git("status", "--short")
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}
    return {
        "revision": revision,
        "worktree_status": status.splitlines(),
    }


def print_progress(
    split: str,
    rows: int,
    elapsed: float,
    rows_per_second: float,
    max_length: int,
    over_cutoff: int,
) -> None:
    print(
        f"[{split}] rows={rows:,} elapsed={elapsed:.1f}s "
        f"rate={rows_per_second:,.0f}/s max_len={max_length} "
        f"over_1024={over_cutoff}",
        flush=True,
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    model_dir = resolve(args.model_dir)
    data_dir = resolve(args.data_dir)
    embedding_dir = resolve(args.embedding_dir)
    output_dir = resolve(args.output_dir)
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if output_dir == outputs_root or not output_dir.is_relative_to(outputs_root):
        raise DynamicPreflightCliError("output_dir 必须是 outputs/ 下的子目录")
    if output_dir.exists():
        raise DynamicPreflightCliError(f"output_dir 已存在，拒绝覆盖：{output_dir}")
    if not model_dir.is_dir():
        raise DynamicPreflightCliError(f"模型目录不存在：{model_dir}")
    output_dir.mkdir(parents=True)
    state_path = output_dir / "run_state.json"
    started_at = datetime.now(timezone.utc)
    running_state = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": started_at.isoformat(),
        "scope": (
            "full_train_valid" if args.max_train_rows is None else "prefix_smoke"
        ),
        "test_samples_read": False,
    }
    atomic_json(state_path, running_state)

    data_manifest, split_contracts = load_data_contract(data_dir)
    data_scan_mode = data_manifest.get("input", {}).get("scan_mode")
    if args.max_train_rows is None and data_scan_mode != "full":
        raise DynamicPreflightCliError("非全量数据必须显式设置 Train/Valid prefix 行数")
    embedding_store = PoiEmbeddingStore.from_directory(
        embedding_dir, load_poi_index=False
    )
    if embedding_store.shape != (2_337_178, 1024):
        raise DynamicPreflightCliError("第一版只接受冻结 BGE-M3 [2,337,178, 1024]")
    catalog_contract = data_manifest.get("embedding_catalog")
    if not isinstance(catalog_contract, dict) or (
        catalog_contract.get("manifest_signature") != embedding_store.manifest_signature
        or catalog_contract.get("poi_ids_sha256") != embedding_store.poi_ids_sha256
        or catalog_contract.get("shape") != list(embedding_store.shape)
    ):
        raise DynamicPreflightCliError("SID-free 数据与当前 BGE 行序不一致")
    initialization_contract = build_fresh_initialization_contract(
        row_count=embedding_store.shape[0]
    )
    tokenizer, template = load_fresh_tokenizer_and_template(
        model_dir,
        project_root=PROJECT_ROOT,
    )
    token_ids = load_dynamic_token_ids(tokenizer)
    template_contract = DynamicPreflightTemplate.from_runtime(
        tokenizer=tokenizer,
        template=template,
        token_ids=token_ids,
    )

    scan_started = time.perf_counter()
    split_results = []
    for split, max_rows in (
        ("train", args.max_train_rows),
        ("valid", args.max_valid_rows),
    ):
        contract = split_contracts[split]
        split_results.append(
            scan_dynamic_preflight_split(
                name=split,
                path=contract["path"],
                expected_rows=contract["rows"],
                expected_sha256=contract["sha256"],
                tokenizer=tokenizer,
                template_contract=template_contract,
                cutoff_len=CUTOFF_LEN,
                batch_size=args.batch_size,
                max_rows=max_rows,
                progress_every=args.progress_every,
                progress_callback=print_progress,
            )
        )
        print(
            f"[{split}] completed rows={split_results[-1].rows:,} "
            f"seconds={split_results[-1].seconds:.1f}",
            flush=True,
        )
    combined = combine_dynamic_preflight_splits(split_results)
    full_scope = args.max_train_rows is None
    formal_gate_passed = bool(full_scope and combined["formal_zero_gate_passed"])
    if full_scope and not formal_gate_passed:
        raise DynamicPreflightCliError("全量 Train/Valid 未通过 1024 Token 零截断门槛")

    finished_at = datetime.now(timezone.utc)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "scope": "full_train_valid" if full_scope else "prefix_smoke",
        "formal_gate_eligible": full_scope,
        "formal_gate_passed": formal_gate_passed,
        "cutoff_len": CUTOFF_LEN,
        "inputs": {
            "model_dir": str(model_dir),
            "model_config_sha256": sha256_file(model_dir / "config.json"),
            "data_dir": str(data_dir),
            "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
            "data_build_fingerprint": data_manifest.get("build_fingerprint"),
            "embedding_dir": str(embedding_dir),
            "embedding_manifest_signature": (embedding_store.manifest_signature),
            "embedding_shape": list(embedding_store.shape),
            "embedding_poi_ids_sha256": embedding_store.poi_ids_sha256,
            "old_sid_artifacts_loaded": [],
            "sample_data_splits_read": ["train", "valid"],
            "test_samples_read": False,
        },
        "dynamic_template": {
            "template": "qwen3_nothink",
            "target_token_length": len(template_contract.target_token_ids),
            "history_identifier": ("<POI_SID><S1_0><S2_0><S3_0></POI_SID>"),
            "target_identifier": ("<TARGET_POI><S1_0><S2_0><S3_0></TARGET_POI>"),
            "old_collision_token_used": False,
        },
        "splits": {split.name: split.to_dict() for split in split_results},
        "combined": combined,
        "initialization_contract": initialization_contract,
        "git": git_state(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "batch_size": args.batch_size,
            "progress_every": args.progress_every,
            "scan_seconds": time.perf_counter() - scan_started,
            "total_seconds": (finished_at - started_at).total_seconds(),
        },
    }
    atomic_json(state_path, result)
    return result


def main() -> int:
    args = parse_args()
    try:
        result = run_preflight(args)
    except (OSError, ValueError, TigerJointPreparationError) as error:
        output_dir = resolve(args.output_dir)
        state_path = output_dir / "run_state.json"
        previous: dict[str, Any] = {}
        if state_path.is_file():
            try:
                previous = load_json_object(state_path, "run_state")
            except TigerJointPreparationError:
                previous = {}
        if (
            previous.get("schema_version") == SCHEMA_VERSION
            and previous.get("status") == "running"
        ):
            previous.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(error),
                    "test_samples_read": False,
                }
            )
            atomic_json(state_path, previous)
        print(f"TIGER-Joint 动态预检失败：{error}", file=sys.stderr)
        return 2
    print(
        "TIGER-Joint 动态预检完成："
        f"rows={result['combined']['rows']:,}, "
        f"max_len={result['combined']['total_length']['max']}, "
        f"over_1024={result['combined']['over_cutoff_count']}, "
        f"target_truncated={result['combined']['target_truncated_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the method-owned KMeans initialization for TIGER-Joint RQ-VAE."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from poi_gr.methods.tiger_joint import PoiEmbeddingStore  # noqa: E402
from poi_gr.methods.tiger_joint.initialization import (  # noqa: E402
    INITIALIZATION_CHECKPOINT_NAME,
    INITIALIZATION_MANIFEST_NAME,
    INITIALIZATION_SAMPLE_NAME,
    INITIALIZATION_SCHEMA_VERSION,
    FreshRqKMeansConfig,
    TigerJointInitializationError,
    build_fresh_rqvae,
    create_method_owned_kmeans_indices,
    fresh_rqvae_model_config,
    initialize_fresh_rqvae_codebooks,
    module_sha256,
)
from poi_gr.pid.dedup import sha256_file  # noqa: E402


class TigerJointInitializationRunError(TigerJointInitializationError):
    """Raised when the formal initialization run violates its frozen inputs."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 fresh RQ-VAE 随机 encoder 和冻结 BGE 全目录中，按 TIGER-Joint "
            "自有固定索引执行 50 万行三级顺序 KMeans，并保存可恢复初始化。"
        )
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("outputs/embeddings/beijing_poi_bge_m3"),
    )
    parser.add_argument("--preflight-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")


def git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"revision": revision, "status_short": status, "dirty": bool(status)}


def load_and_validate_preflight(
    path: Path,
    *,
    embedding_dir: Path,
    embedding_store: PoiEmbeddingStore,
    config: FreshRqKMeansConfig,
    split_metadata: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise TigerJointInitializationRunError(f"正式 SID-free 预检状态不存在：{path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerJointInitializationRunError("正式预检状态不是合法 JSON") from error
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != "tiger-joint-sid-free-preflight-v2"
        or state.get("status") != "completed"
        or state.get("scope") != "full_train_valid"
        or state.get("formal_gate_passed") is not True
    ):
        raise TigerJointInitializationRunError("正式 SID-free 全量预检未通过")

    inputs = state.get("inputs")
    expected_inputs = {
        "embedding_dir": str(embedding_dir),
        "embedding_manifest_signature": embedding_store.manifest_signature,
        "embedding_poi_ids_sha256": embedding_store.poi_ids_sha256,
        "embedding_shape": list(embedding_store.shape),
        "old_sid_artifacts_loaded": [],
        "test_samples_read": False,
    }
    if not isinstance(inputs, dict) or any(
        inputs.get(name) != value for name, value in expected_inputs.items()
    ):
        raise TigerJointInitializationRunError("正式预检与冻结 BGE 目录不一致")

    contract = state.get("initialization_contract")
    expected_kmeans = {
        "backend": config.kmeans_backend,
        "sample_rows": config.kmeans_sample_size,
        "sample_indices_sha256": split_metadata["kmeans_sample_indices_sha256"],
        "residual_mode": "sequential",
        "iterations": config.kmeans_iterations,
        "batch_size": config.kmeans_batch_size,
        "max_points_per_centroid": config.kmeans_max_points_per_centroid,
    }
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != "tiger-joint-fresh-initialization-v1"
        or contract.get("legacy_artifacts_loaded") != []
        or contract.get("random_seed") != config.seed
        or contract.get("kmeans") != expected_kmeans
        or contract.get("fixed_split") != split_metadata
    ):
        raise TigerJointInitializationRunError(
            "正式预检中的 fresh KMeans 契约与当前实现不一致"
        )
    return state, sha256_file(path)


def save_indices(path: Path, indices: np.ndarray) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.npy")
    np.save(temporary, indices, allow_pickle=False)
    os.replace(temporary, path)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    embedding_dir = resolve(args.embedding_dir)
    preflight_path = resolve(args.preflight_state)
    output_dir = resolve(args.output_dir)
    allowed_root = (PROJECT_ROOT / "outputs").resolve()
    if allowed_root not in output_dir.parents:
        raise TigerJointInitializationRunError("output_dir 必须位于仓库 outputs/ 下")
    if output_dir.exists():
        raise TigerJointInitializationRunError(f"output_dir 已存在：{output_dir}")
    if args.device_index < 0:
        raise TigerJointInitializationRunError("device_index 不能为负数")

    output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    running_state = {
        "schema_version": INITIALIZATION_SCHEMA_VERSION,
        "status": "running",
        "started_at": started_at.isoformat(),
        "old_sid_artifacts_loaded": [],
        "formal_initialization_passed": False,
    }
    atomic_json(output_dir / INITIALIZATION_MANIFEST_NAME, running_state)

    if not torch.cuda.is_available():
        raise TigerJointInitializationRunError(
            "正式 faiss_gpu 初始化要求宿主 CUDA，不允许回退 CPU"
        )
    if args.device_index >= torch.cuda.device_count():
        raise TigerJointInitializationRunError("device_index 超出可见 GPU 数量")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    config = FreshRqKMeansConfig()
    seed_everything(config.seed)

    embedding_store = PoiEmbeddingStore.from_directory(
        embedding_dir, load_poi_index=False
    )
    if embedding_store.shape != (2_337_178, 1024):
        raise TigerJointInitializationRunError(
            "第一版冻结 BGE shape 必须为 [2,337,178, 1024]"
        )
    sample_indices, split_metadata = create_method_owned_kmeans_indices(
        embedding_store.poi_count, config
    )
    _, preflight_sha256 = load_and_validate_preflight(
        preflight_path,
        embedding_dir=embedding_dir,
        embedding_store=embedding_store,
        config=config,
        split_metadata=split_metadata,
    )
    save_indices(output_dir / INITIALIZATION_SAMPLE_NAME, sample_indices)

    model = build_fresh_rqvae().to(device)
    random_model_sha256 = module_sha256(model)
    torch.cuda.reset_peak_memory_stats(device)
    initialization_metrics = initialize_fresh_rqvae_codebooks(
        model,
        embedding_store,
        sample_indices,
        config,
        device,
    )
    initialized_model_sha256 = module_sha256(model)
    codebooks_finite = all(
        bool(torch.isfinite(codebook.weight).all().item())
        for codebook in model.quantizer.codebooks
    )
    all_codes_used = all(
        metric["used_codes_on_initialization_sample"] == metric["codebook_size"]
        for metric in initialization_metrics
    )
    formal_checks = {
        "preflight_contract_matched": True,
        "sample_indices_matched": True,
        "three_levels_initialized": len(initialization_metrics) == 3,
        "all_codebooks_finite": codebooks_finite,
        "all_codes_used_on_initialization_sample": all_codes_used,
        "initialization_changed_model": (
            initialized_model_sha256 != random_model_sha256
        ),
        "old_sid_artifacts_loaded_is_empty": True,
    }
    formal_passed = all(formal_checks.values())
    if not formal_passed:
        raise TigerJointInitializationRunError(
            "fresh RQ-VAE KMeans 初始化未通过完整性门禁"
        )

    checkpoint_path = output_dir / INITIALIZATION_CHECKPOINT_NAME
    save_checkpoint(
        checkpoint_path,
        {
            "schema_version": INITIALIZATION_SCHEMA_VERSION,
            "model": fresh_rqvae_model_config(),
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
    )
    finished_at = datetime.now(timezone.utc)
    result = {
        "schema_version": INITIALIZATION_SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "source": {
            "embedding_dir": str(embedding_dir),
            "embedding_manifest_signature": embedding_store.manifest_signature,
            "embedding_poi_ids_sha256": embedding_store.poi_ids_sha256,
            "embedding_shape": list(embedding_store.shape),
            "preflight_state": str(preflight_path),
            "preflight_state_sha256": preflight_sha256,
        },
        "model": fresh_rqvae_model_config(),
        "kmeans": {
            **asdict(config),
            "sample_indices_sha256": split_metadata["kmeans_sample_indices_sha256"],
            "sample_indices_file_sha256": sha256_file(
                output_dir / INITIALIZATION_SAMPLE_NAME
            ),
            "split_metadata": split_metadata,
            "levels": initialization_metrics,
        },
        "checkpoint": {
            "path": INITIALIZATION_CHECKPOINT_NAME,
            "sha256": sha256_file(checkpoint_path),
            "random_model_sha256": random_model_sha256,
            "model_sha256": initialized_model_sha256,
        },
        "old_sid_artifacts_loaded": [],
        "formal_checks": formal_checks,
        "formal_initialization_passed": True,
        "git": git_state(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    atomic_json(output_dir / INITIALIZATION_MANIFEST_NAME, result)
    return result


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    try:
        result = run(args)
    except Exception as error:
        manifest_path = output_dir / INITIALIZATION_MANIFEST_NAME
        if manifest_path.is_file():
            try:
                state = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            if isinstance(state, dict) and state.get("status") == "running":
                state.update(
                    {
                        "status": "failed",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "formal_initialization_passed": False,
                    }
                )
                atomic_json(manifest_path, state)
        print(f"TIGER-Joint RQ-VAE 初始化失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

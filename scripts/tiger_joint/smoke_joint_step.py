#!/usr/bin/env python3
"""Run one real-data TIGER-Joint optimizer step on a server GPU."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger_joint import (  # noqa: E402
    DynamicTigerExample,
    HuggingFaceCausalLMJointAdapter,
    PoiEmbeddingStore,
    TigerJointPreparationError,
    build_sid_token_layout,
    collate_dynamic_examples,
    compute_joint_tiger_loss,
    load_dynamic_examples,
    load_dynamic_token_ids,
    load_fresh_tokenizer_and_template,
)
from poi_gr.pid.dedup import sha256_file  # noqa: E402
from poi_gr.sid.rqvae import RQVAE  # noqa: E402


SCHEMA_VERSION = "tiger-joint-server-smoke-v2"
CODEBOOK_SIZES = (1024, 1024, 1024)
CUTOFF_LEN = 1024


class TigerJointSmokeError(RuntimeError):
    """Raised when a real server smoke cannot prove the joint gradient path."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在服务器单卡 CUDA 上，用真实 Train/Valid 前缀、冻结 BGE mmap、"
            "vanilla Qwen fresh 扩词表和 fresh RQ-VAE 执行一次联合 step。"
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
    parser.add_argument("--train-scan-rows", type=int, default=32)
    parser.add_argument("--valid-scan-rows", type=int, default=2)
    parser.add_argument("--behavior-batch-size", type=int, default=2)
    parser.add_argument("--catalog-batch-size", type=int, default=64)
    parser.add_argument("--qwen-learning-rate", type=float, default=5e-5)
    parser.add_argument("--rq-learning-rate", type=float, default=3e-4)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--rq-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
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


def require_positive(value: int | float, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise TigerJointSmokeError(f"{name} 必须大于 0")


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "train_scan_rows",
        "valid_scan_rows",
        "behavior_batch_size",
        "catalog_batch_size",
        "qwen_learning_rate",
        "rq_learning_rate",
        "alignment_weight",
        "rq_weight",
        "temperature",
    ):
        require_positive(getattr(args, name), name)
    if args.behavior_batch_size < 2:
        raise TigerJointSmokeError(
            "behavior_batch_size 至少为 2，单样本 InfoNCE 恒为 0"
        )
    if args.train_scan_rows < args.behavior_batch_size:
        raise TigerJointSmokeError("train_scan_rows 不能小于 behavior_batch_size")
    if args.catalog_batch_size < 2:
        raise TigerJointSmokeError("catalog_batch_size 至少为 2")
    if args.seed < 0:
        raise TigerJointSmokeError("seed 不能为负数")


def load_data_manifest(data_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise TigerJointSmokeError(f"数据 manifest 不存在：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerJointSmokeError("数据 manifest 不是合法 JSON") from error
    if manifest.get("schema_version") != "tiger-joint-sid-free-data-v1":
        raise TigerJointSmokeError("数据 schema_version 不兼容")
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
        raise TigerJointSmokeError("数据 manifest 未证明与旧 TIGER SID 隔离")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise TigerJointSmokeError("数据 manifest 缺少 outputs")
    if "test.jsonl" in outputs or (data_dir / "test.jsonl").exists():
        raise TigerJointSmokeError("联合训练数据目录不得包含 test.jsonl")
    paths: list[Path] = []
    for split in ("train", "valid"):
        name = f"{split}.jsonl"
        spec = outputs.get(name)
        path = data_dir / name
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("rows"), int)
            or spec["rows"] <= 0
            or not isinstance(spec.get("sha256"), str)
            or len(spec["sha256"]) != 64
            or not path.is_file()
        ):
            raise TigerJointSmokeError(f"{name} 契约无效或文件不存在")
        paths.append(path)
    return manifest, paths[0], paths[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def select_distinct_sid_examples(
    examples: Sequence[DynamicTigerExample],
    *,
    rqvae: RQVAE,
    embedding_store: PoiEmbeddingStore,
    batch_size: int,
    device: torch.device,
) -> tuple[list[DynamicTigerExample], dict[str, Any]]:
    """Select distinct initial SIDs so smoke validates non-zero InfoNCE gradients."""

    target_rows = [example.target_poi_row for example in examples]
    target_embeddings = torch.from_numpy(embedding_store.gather(target_rows)).to(device)
    with torch.no_grad():
        codes = rqvae.encode_codes(target_embeddings).cpu().numpy()
    selected: list[DynamicTigerExample] = []
    seen_rows: set[int] = set()
    seen_codes: set[tuple[int, int, int]] = set()
    for example, raw_codes in zip(examples, codes, strict=True):
        code_key = tuple(int(value) for value in raw_codes)
        if example.target_poi_row in seen_rows or code_key in seen_codes:
            continue
        selected.append(example)
        seen_rows.add(example.target_poi_row)
        seen_codes.add(code_key)
        if len(selected) == batch_size:
            break
    if len(selected) != batch_size:
        raise TigerJointSmokeError(
            "fresh RQ-VAE 在候选前缀中没有足够的不同三级 SID；"
            "增加 --train-scan-rows 后重试"
        )
    return selected, {
        "selection": "first_distinct_target_row_and_fresh_sid_in_prefix",
        "candidate_rows": len(examples),
        "candidate_distinct_target_rows": len(set(target_rows)),
        "candidate_distinct_initial_sids": len(
            {tuple(int(value) for value in row) for row in codes}
        ),
        "selected_rows": len(selected),
    }


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared_norm = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        value = float(parameter.grad.detach().float().norm().cpu())
        squared_norm += value * value
    return squared_norm**0.5 if found else 0.0


def require_finite_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise TigerJointSmokeError(f"{name} 必须是有限正数，实际为 {value}")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    model_dir = resolve(args.model_dir)
    data_dir = resolve(args.data_dir)
    embedding_dir = resolve(args.embedding_dir)
    output_dir = resolve(args.output_dir)
    if not output_dir.is_relative_to((PROJECT_ROOT / "outputs").resolve()):
        raise TigerJointSmokeError("output_dir 必须位于仓库 outputs/ 下")
    if output_dir.exists():
        raise TigerJointSmokeError(f"output_dir 已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    atomic_json(
        output_dir / "run_state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at": started_at.isoformat(),
        },
    )

    if not torch.cuda.is_available():
        raise TigerJointSmokeError("服务器 smoke 要求宿主 CUDA；不允许回退 CPU")
    device = torch.device("cuda", torch.cuda.current_device())
    seed_everything(args.seed)
    data_manifest, train_path, valid_path = load_data_manifest(data_dir)
    embedding_store = PoiEmbeddingStore.from_directory(
        embedding_dir, load_poi_index=False
    )
    if embedding_store.shape != (2_337_178, 1024):
        raise TigerJointSmokeError("第一版要求冻结 BGE-M3 shape 为 [2,337,178, 1024]")
    catalog_contract = data_manifest.get("embedding_catalog")
    if not isinstance(catalog_contract, dict) or (
        catalog_contract.get("manifest_signature") != embedding_store.manifest_signature
        or catalog_contract.get("poi_ids_sha256") != embedding_store.poi_ids_sha256
        or catalog_contract.get("shape") != list(embedding_store.shape)
    ):
        raise TigerJointSmokeError("SID-free 数据与当前 BGE 行序不一致")
    tokenizer, template = load_fresh_tokenizer_and_template(
        model_dir,
        project_root=PROJECT_ROOT,
    )
    if tokenizer.pad_token_id is None:
        raise TigerJointSmokeError("Tokenizer 缺少 pad_token_id")
    token_ids = load_dynamic_token_ids(tokenizer)
    token_layout = build_sid_token_layout(tokenizer, CODEBOOK_SIZES)
    train_examples = load_dynamic_examples(
        train_path,
        split="train",
        max_rows=args.train_scan_rows,
        tokenizer=tokenizer,
        template=template,
        token_ids=token_ids,
        cutoff_len=CUTOFF_LEN,
    )
    valid_examples = load_dynamic_examples(
        valid_path,
        split="valid",
        max_rows=args.valid_scan_rows,
        tokenizer=tokenizer,
        template=template,
        token_ids=token_ids,
        cutoff_len=CUTOFF_LEN,
    )

    rqvae = RQVAE(
        input_dim=1024,
        hidden_dim=512,
        latent_dim=256,
        codebook_sizes=CODEBOOK_SIZES,
        codebook_loss_weight=1.0,
        commitment_loss_weight=0.25,
    ).to(device)
    selected_examples, selection_metrics = select_distinct_sid_examples(
        train_examples,
        rqvae=rqvae,
        embedding_store=embedding_store,
        batch_size=args.behavior_batch_size,
        device=device,
    )
    rng = np.random.default_rng(args.seed)
    catalog_rows = rng.choice(
        embedding_store.poi_count,
        size=args.catalog_batch_size,
        replace=False,
    ).astype(np.int64, copy=False)
    batch = collate_dynamic_examples(
        selected_examples,
        embedding_store=embedding_store,
        catalog_rows=catalog_rows,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        cutoff_len=CUTOFF_LEN,
    )

    from transformers import AutoModelForCausalLM

    causal_lm = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    causal_lm.resize_token_embeddings(len(tokenizer))
    if int(causal_lm.config.vocab_size) != len(tokenizer):
        raise TigerJointSmokeError("Qwen vocab_size 与冻结 tokenizer 不一致")
    if not bool(causal_lm.config.tie_word_embeddings):
        raise TigerJointSmokeError("第一版要求 Qwen 输入输出 Embedding 绑定")
    if (
        causal_lm.get_input_embeddings().weight.data_ptr()
        != causal_lm.get_output_embeddings().weight.data_ptr()
    ):
        raise TigerJointSmokeError("Qwen 输入输出 Embedding 实际未绑定")
    causal_lm.config.use_cache = False
    causal_lm.train()
    rqvae.train()
    generator = HuggingFaceCausalLMJointAdapter(causal_lm)
    query_projection = nn.Linear(
        int(causal_lm.config.hidden_size), rqvae.latent_dim
    ).to(device=device, dtype=torch.float32)

    qwen_parameters = [
        parameter for parameter in generator.parameters() if parameter.requires_grad
    ]
    projection_parameters = list(query_projection.parameters())
    rq_parameters = list(rqvae.parameters())
    qwen_optimizer = torch.optim.AdamW(
        [
            {"params": qwen_parameters},
            {"params": projection_parameters},
        ],
        lr=args.qwen_learning_rate,
        weight_decay=0.01,
    )
    rq_optimizer = torch.optim.Adam(
        rq_parameters,
        lr=args.rq_learning_rate,
        weight_decay=0.0,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_step = torch.cuda.memory_allocated(device)
    qwen_optimizer.zero_grad(set_to_none=True)
    rq_optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    step_started = time.perf_counter()
    output = compute_joint_tiger_loss(
        generator=generator,
        rqvae=rqvae,
        query_projection=query_projection,
        batch=batch,
        token_layout=token_layout,
        alignment_weight=args.alignment_weight,
        rq_weight=args.rq_weight,
        temperature=args.temperature,
    )
    output.total_loss.backward()

    gradient_metrics = {
        "qwen": gradient_norm(qwen_parameters),
        "query_projection": gradient_norm(projection_parameters),
        "rqvae": gradient_norm(rq_parameters),
        "rq_encoder": gradient_norm(rqvae.encoder.parameters()),
        "rq_decoder": gradient_norm(rqvae.decoder.parameters()),
        "rq_codebooks": gradient_norm(rqvae.quantizer.codebooks.parameters()),
    }
    for name, value in gradient_metrics.items():
        require_finite_positive(value, f"{name} gradient norm")
    clipped_gradient_metrics = {
        "qwen": float(
            torch.nn.utils.clip_grad_norm_(qwen_parameters, max_norm=1.0).cpu()
        ),
        "query_projection": float(
            torch.nn.utils.clip_grad_norm_(projection_parameters, max_norm=1.0).cpu()
        ),
        "rqvae": float(
            torch.nn.utils.clip_grad_norm_(rq_parameters, max_norm=1.0).cpu()
        ),
    }
    qwen_optimizer.step()
    rq_optimizer.step()
    torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - step_started

    losses = {
        "total": float(output.total_loss.detach().cpu()),
        "generation": float(output.generation_loss.detach().cpu()),
        "alignment": float(output.alignment_loss.detach().cpu()),
        "rq": float(output.rq_loss.detach().cpu()),
    }
    for name, value in losses.items():
        require_finite_positive(value, f"{name} loss")
    target_codes = output.target_codes.cpu().numpy()
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "optimizer_steps": 1,
            "full_training": False,
            "checkpoint_written": False,
            "sample_data_splits_read": ["train", "valid"],
            "test_samples_read": False,
            "rqvae_initialization": "fresh_random_smoke_only",
        },
        "inputs": {
            "model_dir": str(model_dir),
            "model_config_sha256": sha256_file(model_dir / "config.json"),
            "data_dir": str(data_dir),
            "data_build_fingerprint": data_manifest.get("build_fingerprint"),
            "declared_train_rows": data_manifest["outputs"]["train.jsonl"]["rows"],
            "declared_valid_rows": data_manifest["outputs"]["valid.jsonl"]["rows"],
            "embedding_dir": str(embedding_dir),
            "embedding_manifest_signature": embedding_store.manifest_signature,
            "embedding_poi_ids_sha256": embedding_store.poi_ids_sha256,
            "embedding_shape": list(embedding_store.shape),
            "old_sid_artifacts_loaded": [],
        },
        "data_smoke": {
            "train_rows_tokenized": len(train_examples),
            "valid_rows_tokenized": len(valid_examples),
            "train_sequence_length_min": min(
                example.sequence_length for example in train_examples
            ),
            "train_sequence_length_max": max(
                example.sequence_length for example in train_examples
            ),
            "valid_sequence_length_min": min(
                example.sequence_length for example in valid_examples
            ),
            "valid_sequence_length_max": max(
                example.sequence_length for example in valid_examples
            ),
            "cutoff_len": CUTOFF_LEN,
            "old_collision_token_used_in_dynamic_template": False,
            **selection_metrics,
        },
        "model": {
            "qwen_architecture": causal_lm.config.architectures,
            "qwen_dtype": str(next(causal_lm.parameters()).dtype),
            "qwen_vocab_size": int(causal_lm.config.vocab_size),
            "qwen_hidden_size": int(causal_lm.config.hidden_size),
            "rqvae_encoder": [1024, 512, 256],
            "rqvae_codebook_sizes": list(CODEBOOK_SIZES),
            "query_projection": [int(causal_lm.config.hidden_size), 256],
        },
        "optimization": {
            "qwen_learning_rate": args.qwen_learning_rate,
            "rq_learning_rate": args.rq_learning_rate,
            "alignment_weight": args.alignment_weight,
            "rq_weight": args.rq_weight,
            "temperature": args.temperature,
            "behavior_batch_size": args.behavior_batch_size,
            "catalog_batch_size": args.catalog_batch_size,
            "seed": args.seed,
        },
        "step": {
            "losses": losses,
            "gradient_norms_before_clipping": gradient_metrics,
            "clip_grad_norm_return_values": clipped_gradient_metrics,
            "target_sid_distinct_by_level": [
                int(np.unique(target_codes[:, level]).size)
                for level in range(target_codes.shape[1])
            ],
            "target_sid_distinct_full": int(np.unique(target_codes, axis=0).shape[0]),
            "seconds": step_seconds,
            "optimizer_step_completed": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                device
            ).total_memory,
            "memory_allocated_before_step_bytes": memory_before_step,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    atomic_json(output_dir / "run_state.json", result)
    return result


def main() -> int:
    args = parse_args()
    try:
        result = run_smoke(args)
    except (
        OSError,
        ValueError,
        TigerJointPreparationError,
        TigerJointSmokeError,
    ) as error:
        state_path = resolve(args.output_dir) / "run_state.json"
        if state_path.is_file():
            try:
                previous = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
            if (
                isinstance(previous, dict)
                and previous.get("schema_version") == SCHEMA_VERSION
                and previous.get("status") == "running"
            ):
                previous.update(
                    {
                        "status": "failed",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "error": str(error),
                        "test_samples_read": False,
                    }
                )
                atomic_json(state_path, previous)
        print(f"TIGER-Joint server smoke 失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a bounded fixed-batch TIGER-Joint overfit experiment on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from poi_gr.methods.tiger_joint import (  # noqa: E402
    HuggingFaceCausalLMJointAdapter,
    JointTigerBatch,
    JointTigerLossOutput,
    PoiEmbeddingStore,
    SidTokenLayout,
    build_sid_token_layout,
    collate_dynamic_examples,
    compute_joint_tiger_loss,
    load_dynamic_examples,
    load_dynamic_token_ids,
    load_fresh_tokenizer_and_template,
)
from poi_gr.methods.tiger_joint.initialization import (  # noqa: E402
    INITIALIZATION_MANIFEST_NAME,
    build_fresh_rqvae,
    load_fresh_rqvae_initialization,
)
from poi_gr.pid.dedup import sha256_file  # noqa: E402
from poi_gr.sid.rqvae import RQVAE  # noqa: E402
from scripts.tiger_joint.smoke_joint_step import (  # noqa: E402
    CODEBOOK_SIZES,
    CUTOFF_LEN,
    atomic_json,
    gradient_norm,
    load_data_manifest,
    seed_everything,
    select_distinct_sid_examples,
)


SCHEMA_VERSION = "tiger-joint-bounded-overfit-v2"
MIN_GENERATION_LOSS_REDUCTION = 0.50
MIN_TEACHER_TOKEN_ACCURACY = 0.80
MIN_FREE_TARGET_EXACT_RATE = 1.0


class TigerJointOverfitError(RuntimeError):
    """Raised when the bounded overfit protocol or its inputs are invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在严格 SID-free 全量门禁产物上固定少量 Train 样本，逐 step 同时更新 "
            "vanilla Qwen、Query 投影和 fresh RQ-VAE，并验证动态 SID 与监督一致。"
        )
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/Qwen3-0.6B"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("outputs/embeddings/beijing_poi_bge_m3"),
    )
    parser.add_argument("--preflight-state", type=Path, required=True)
    parser.add_argument(
        "--rq-initialization-dir",
        type=Path,
        help=(
            "方法自有 fresh RQ-VAE KMeans 初始化目录；不传时仅保留历史随机初始化诊断。"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=30)
    parser.add_argument("--train-scan-rows", type=int, default=64)
    parser.add_argument("--behavior-batch-size", type=int, default=2)
    parser.add_argument("--catalog-batch-size", type=int, default=64)
    parser.add_argument("--qwen-learning-rate", type=float, default=5e-5)
    parser.add_argument("--rq-learning-rate", type=float, default=3e-4)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--rq-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _require_positive(value: int | float, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise TigerJointOverfitError(f"{name} 必须大于 0")


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "optimizer_steps",
        "train_scan_rows",
        "behavior_batch_size",
        "catalog_batch_size",
        "qwen_learning_rate",
        "rq_learning_rate",
        "alignment_weight",
        "rq_weight",
        "temperature",
    ):
        _require_positive(getattr(args, name), name)
    if args.behavior_batch_size < 2:
        raise TigerJointOverfitError(
            "behavior_batch_size 至少为 2，单样本 InfoNCE 恒为 0"
        )
    if args.train_scan_rows < args.behavior_batch_size:
        raise TigerJointOverfitError("train_scan_rows 不能小于 behavior_batch_size")
    if args.catalog_batch_size < 2:
        raise TigerJointOverfitError("catalog_batch_size 至少为 2")
    if args.seed < 0 or args.device_index < 0:
        raise TigerJointOverfitError("seed 与 device_index 不能为负数")


def load_formal_preflight(
    preflight_path: Path,
    *,
    model_dir: Path,
    data_dir: Path,
    embedding_dir: Path,
    data_manifest: dict[str, Any],
    embedding_store: PoiEmbeddingStore,
) -> tuple[dict[str, Any], str]:
    if not preflight_path.is_file():
        raise TigerJointOverfitError(f"正式预检状态不存在：{preflight_path}")
    try:
        state = json.loads(preflight_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerJointOverfitError("正式预检状态不是合法 JSON") from error
    if (
        state.get("schema_version") != "tiger-joint-sid-free-preflight-v2"
        or state.get("status") != "completed"
        or state.get("scope") != "full_train_valid"
        or state.get("formal_gate_passed") is not True
    ):
        raise TigerJointOverfitError("正式预检未通过 full_train_valid 门禁")
    inputs = state.get("inputs")
    if not isinstance(inputs, dict):
        raise TigerJointOverfitError("正式预检状态缺少 inputs")
    manifest_path = data_dir / "manifest.json"
    expected = {
        "model_dir": str(model_dir),
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "data_dir": str(data_dir),
        "data_build_fingerprint": data_manifest.get("build_fingerprint"),
        "data_manifest_sha256": sha256_file(manifest_path),
        "embedding_dir": str(embedding_dir),
        "embedding_manifest_signature": embedding_store.manifest_signature,
        "embedding_poi_ids_sha256": embedding_store.poi_ids_sha256,
        "embedding_shape": list(embedding_store.shape),
    }
    mismatches = [name for name, value in expected.items() if inputs.get(name) != value]
    if mismatches:
        raise TigerJointOverfitError(
            "正式预检与本次输入不一致：" + ", ".join(mismatches)
        )
    if inputs.get("old_sid_artifacts_loaded") != []:
        raise TigerJointOverfitError("正式预检曾加载旧 SID 产物")
    if inputs.get("test_samples_read") is not False:
        raise TigerJointOverfitError("正式预检未证明 Test 隔离")
    return state, sha256_file(preflight_path)


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def stable_sequence_sha256(values: Sequence[str | int]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def loss_metrics(output: JointTigerLossOutput) -> dict[str, float]:
    metrics = {
        "total": float(output.total_loss.detach().cpu()),
        "generation": float(output.generation_loss.detach().cpu()),
        "alignment": float(output.alignment_loss.detach().cpu()),
        "rq": float(output.rq_loss.detach().cpu()),
        "teacher_forced_token_accuracy": float(
            output.teacher_forced_token_accuracy.cpu()
        ),
        "teacher_forced_exact_match": float(output.teacher_forced_exact_match.cpu()),
        "teacher_forced_sid_token_accuracy": float(
            output.teacher_forced_sid_token_accuracy.cpu()
        ),
        "teacher_forced_static_token_accuracy": float(
            output.teacher_forced_static_token_accuracy.cpu()
        ),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise TigerJointOverfitError("loss 或教师强制准确率出现 NaN/Inf")
    return metrics


def assert_same_step_sid_supervision(
    output: JointTigerLossOutput,
    *,
    batch: JointTigerBatch,
    token_layout: SidTokenLayout,
) -> None:
    target_tokens = token_layout.tokens_for_codes(output.target_codes)
    target_samples = torch.arange(
        output.target_codes.shape[0], device=output.target_codes.device
    ).unsqueeze(1)
    target_positions = batch.template.target_sid_positions
    for name, values in (
        (
            "target input_ids",
            output.materialized.input_ids[target_samples, target_positions],
        ),
        (
            "target labels",
            output.materialized.labels[target_samples, target_positions],
        ),
    ):
        if not torch.equal(values, target_tokens):
            raise TigerJointOverfitError(f"当步动态 SID 与 {name} 不一致")
    if output.history_codes.numel():
        history_tokens = token_layout.tokens_for_codes(output.history_codes)
        materialized_history = output.materialized.input_ids[
            batch.template.history_sample_indices.unsqueeze(1),
            batch.template.history_sid_positions,
        ]
        if not torch.equal(materialized_history, history_tokens):
            raise TigerJointOverfitError("当步历史 SID 与 input_ids 不一致")


def sid_churn(
    previous: torch.Tensor,
    current: torch.Tensor,
) -> dict[str, Any]:
    if previous.shape != current.shape:
        raise TigerJointOverfitError("相邻 step 的目标 SID shape 不一致")
    changed = previous.ne(current)
    changed_rows = changed.any(dim=1)
    return {
        "full_sid_changed_rows": int(changed_rows.sum().item()),
        "full_sid_churn_rate": float(changed_rows.float().mean().item()),
        "changed_rows_by_level": [
            int(changed[:, level].sum().item()) for level in range(changed.shape[1])
        ],
        "churn_rate_by_level": [
            float(changed[:, level].float().mean().item())
            for level in range(changed.shape[1])
        ],
    }


def free_target_generation_metrics(
    *,
    causal_lm: nn.Module,
    output: JointTigerLossOutput,
    batch: JointTigerBatch,
    token_layout: SidTokenLayout,
    target_open_token_id: int,
    target_close_token_id: int,
    pad_token_id: int,
) -> dict[str, Any]:
    exact_count = 0
    legal_count = 0
    generated_lengths: list[int] = []
    for sample_index in range(output.target_codes.shape[0]):
        sid_positions = batch.template.target_sid_positions[sample_index]
        start = int(sid_positions[0].item()) - 1
        stop = int(sid_positions[-1].item()) + 2
        expected = output.materialized.input_ids[sample_index, start:stop]
        expected_sid = token_layout.tokens_for_codes(
            output.target_codes[sample_index : sample_index + 1]
        )[0]
        if (
            expected.numel() != 5
            or int(expected[0].item()) != target_open_token_id
            or int(expected[-1].item()) != target_close_token_id
            or not torch.equal(expected[1:4], expected_sid)
        ):
            raise TigerJointOverfitError("动态目标 wrapper 或三级 SID 位置无效")
        prompt = output.materialized.input_ids[sample_index, :start].unsqueeze(0)
        attention_mask = torch.ones_like(prompt)
        generated = causal_lm.generate(
            input_ids=prompt,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=5,
            use_cache=True,
            pad_token_id=pad_token_id,
        )[0, prompt.shape[1] :]
        generated_lengths.append(int(generated.numel()))
        is_exact = torch.equal(generated, expected)
        is_legal = bool(
            generated.numel() == 5
            and int(generated[0].item()) == target_open_token_id
            and int(generated[-1].item()) == target_close_token_id
            and torch.equal(generated[1:4], expected_sid)
        )
        exact_count += int(is_exact)
        legal_count += int(is_legal)
    sample_count = output.target_codes.shape[0]
    return {
        "sample_count": sample_count,
        "exact_count": exact_count,
        "exact_rate": exact_count / sample_count,
        "legal_current_sid_count": legal_count,
        "legal_current_sid_rate": legal_count / sample_count,
        "generated_lengths": generated_lengths,
        "target_token_count": 5,
        "raw_generated_token_ids_saved": False,
    }


def evaluate_fixed_batch(
    *,
    generator: HuggingFaceCausalLMJointAdapter,
    rqvae: RQVAE,
    query_projection: nn.Module,
    batch: JointTigerBatch,
    token_layout: SidTokenLayout,
    alignment_weight: float,
    rq_weight: float,
    temperature: float,
    target_open_token_id: int,
    target_close_token_id: int,
    pad_token_id: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    generator.eval()
    rqvae.eval()
    query_projection.eval()
    with torch.no_grad():
        output = compute_joint_tiger_loss(
            generator=generator,
            rqvae=rqvae,
            query_projection=query_projection,
            batch=batch,
            token_layout=token_layout,
            alignment_weight=alignment_weight,
            rq_weight=rq_weight,
            temperature=temperature,
        )
        assert_same_step_sid_supervision(output, batch=batch, token_layout=token_layout)
        free_generation = free_target_generation_metrics(
            causal_lm=generator.causal_lm,
            output=output,
            batch=batch,
            token_layout=token_layout,
            target_open_token_id=target_open_token_id,
            target_close_token_id=target_close_token_id,
            pad_token_id=pad_token_id,
        )
        codes = output.target_codes.cpu().clone()
        metrics = {
            "losses_and_teacher_forcing": loss_metrics(output),
            "free_dynamic_target_generation": free_generation,
            "target_sid_digest": tensor_sha256(codes),
            "target_sid_distinct_full": int(torch.unique(codes, dim=0).shape[0]),
            "same_step_sid_supervision_consistent": True,
        }
    generator.train()
    rqvae.train()
    query_projection.train()
    return metrics, codes


def _base_running_state(
    *,
    started_at: datetime,
    optimizer_steps: int,
    data_dir: Path,
    preflight_path: Path,
    rq_initialization_dir: Path | None,
) -> dict[str, Any]:
    initialization_mode = (
        "fresh_method_owned_kmeans"
        if rq_initialization_dir is not None
        else "fresh_random_bounded_overfit_only"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": started_at.isoformat(),
        "scope": {
            "optimizer_steps_planned": optimizer_steps,
            "fixed_train_behavior_batch": True,
            "fixed_catalog_batch": True,
            "full_training": False,
            "checkpoint_written": False,
            "sample_data_splits_read": ["train"],
            "valid_samples_read": False,
            "test_samples_read": False,
            "rqvae_initialization": initialization_mode,
        },
        "inputs": {
            "data_dir": str(data_dir),
            "preflight_state": str(preflight_path),
            "rq_initialization_dir": (
                None if rq_initialization_dir is None else str(rq_initialization_dir)
            ),
            "old_sid_artifacts_loaded": [],
        },
        "progress": {"optimizer_steps_completed": 0},
    }


def run_overfit(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    model_dir = resolve(args.model_dir)
    data_dir = resolve(args.data_dir)
    embedding_dir = resolve(args.embedding_dir)
    preflight_path = resolve(args.preflight_state)
    rq_initialization_dir = (
        None
        if args.rq_initialization_dir is None
        else resolve(args.rq_initialization_dir)
    )
    output_dir = resolve(args.output_dir)
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    if not output_dir.is_relative_to(outputs_root):
        raise TigerJointOverfitError("output_dir 必须位于仓库 outputs/ 下")
    if output_dir.exists():
        raise TigerJointOverfitError(f"output_dir 已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    running_state = _base_running_state(
        started_at=started_at,
        optimizer_steps=args.optimizer_steps,
        data_dir=data_dir,
        preflight_path=preflight_path,
        rq_initialization_dir=rq_initialization_dir,
    )
    atomic_json(output_dir / "run_state.json", running_state)

    if not torch.cuda.is_available():
        raise TigerJointOverfitError("J2 有界过拟合要求宿主 CUDA，不允许回退 CPU")
    if args.device_index >= torch.cuda.device_count():
        raise TigerJointOverfitError("device_index 超出可见 CUDA 设备数量")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    seed_everything(args.seed)

    data_manifest, train_path, _valid_path = load_data_manifest(data_dir)
    embedding_store = PoiEmbeddingStore.from_directory(
        embedding_dir, load_poi_index=False
    )
    if embedding_store.shape != (2_337_178, 1024):
        raise TigerJointOverfitError("第一版要求冻结 BGE-M3 shape 为 [2,337,178, 1024]")
    catalog_contract = data_manifest.get("embedding_catalog")
    if not isinstance(catalog_contract, dict) or (
        catalog_contract.get("manifest_signature") != embedding_store.manifest_signature
        or catalog_contract.get("poi_ids_sha256") != embedding_store.poi_ids_sha256
        or catalog_contract.get("shape") != list(embedding_store.shape)
    ):
        raise TigerJointOverfitError("SID-free 数据与冻结 BGE 行序不一致")
    _preflight, preflight_sha256 = load_formal_preflight(
        preflight_path,
        model_dir=model_dir,
        data_dir=data_dir,
        embedding_dir=embedding_dir,
        data_manifest=data_manifest,
        embedding_store=embedding_store,
    )

    tokenizer, template = load_fresh_tokenizer_and_template(
        model_dir, project_root=PROJECT_ROOT
    )
    if tokenizer.pad_token_id is None:
        raise TigerJointOverfitError("Tokenizer 缺少 pad_token_id")
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

    initialization_manifest: dict[str, Any] | None = None
    if rq_initialization_dir is None:
        rqvae = build_fresh_rqvae().to(device)
    else:
        rqvae, initialization_manifest = load_fresh_rqvae_initialization(
            rq_initialization_dir,
            embedding_store=embedding_store,
            preflight_state_sha256=preflight_sha256,
            device=device,
        )
    rq_initial_sha256 = module_sha256(rqvae)
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
    base_vocab_size = int(causal_lm.config.vocab_size)
    causal_lm.resize_token_embeddings(len(tokenizer))
    for generation_setting in ("temperature", "top_p", "top_k"):
        setattr(causal_lm.generation_config, generation_setting, None)
    if int(causal_lm.config.vocab_size) != len(tokenizer):
        raise TigerJointOverfitError("Qwen vocab_size 与 fresh tokenizer 不一致")
    if not bool(causal_lm.config.tie_word_embeddings):
        raise TigerJointOverfitError("第一版要求 Qwen 输入输出 Embedding 绑定")
    if (
        causal_lm.get_input_embeddings().weight.data_ptr()
        != causal_lm.get_output_embeddings().weight.data_ptr()
    ):
        raise TigerJointOverfitError("Qwen 输入输出 Embedding 实际未绑定")
    causal_lm.config.use_cache = False
    generator = HuggingFaceCausalLMJointAdapter(causal_lm)
    query_projection = nn.Linear(
        int(causal_lm.config.hidden_size), rqvae.latent_dim
    ).to(device=device, dtype=torch.float32)
    added_embeddings = causal_lm.get_input_embeddings().weight[base_vocab_size:]
    added_embedding_initial_sha256 = tensor_sha256(added_embeddings)
    projection_initial_sha256 = module_sha256(query_projection)

    generator.train()
    rqvae.train()
    query_projection.train()
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
        rq_parameters, lr=args.rq_learning_rate, weight_decay=0.0
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_training = torch.cuda.memory_allocated(device)
    training_started = time.perf_counter()
    initial_evaluation, previous_codes = evaluate_fixed_batch(
        generator=generator,
        rqvae=rqvae,
        query_projection=query_projection,
        batch=batch,
        token_layout=token_layout,
        alignment_weight=args.alignment_weight,
        rq_weight=args.rq_weight,
        temperature=args.temperature,
        target_open_token_id=token_ids.target_open,
        target_close_token_id=token_ids.target_close,
        pad_token_id=int(tokenizer.pad_token_id),
    )

    step_trace: list[dict[str, Any]] = []
    all_gradient_groups_finite_positive = True
    for step_index in range(1, args.optimizer_steps + 1):
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
        assert_same_step_sid_supervision(output, batch=batch, token_layout=token_layout)
        output.total_loss.backward()
        gradient_metrics = {
            "qwen": gradient_norm(qwen_parameters),
            "query_projection": gradient_norm(projection_parameters),
            "rqvae": gradient_norm(rq_parameters),
            "rq_encoder": gradient_norm(rqvae.encoder.parameters()),
            "rq_decoder": gradient_norm(rqvae.decoder.parameters()),
            "rq_codebooks": gradient_norm(rqvae.quantizer.codebooks.parameters()),
        }
        if not all(np.isfinite(value) for value in gradient_metrics.values()):
            raise TigerJointOverfitError(
                f"step {step_index} gradient norm 出现 NaN/Inf"
            )
        gradient_groups_finite_positive = all(
            value > 0 for value in gradient_metrics.values()
        )
        all_gradient_groups_finite_positive &= gradient_groups_finite_positive
        clipped_gradient_metrics = {
            "qwen": float(
                torch.nn.utils.clip_grad_norm_(qwen_parameters, max_norm=1.0).cpu()
            ),
            "query_projection": float(
                torch.nn.utils.clip_grad_norm_(
                    projection_parameters, max_norm=1.0
                ).cpu()
            ),
            "rqvae": float(
                torch.nn.utils.clip_grad_norm_(rq_parameters, max_norm=1.0).cpu()
            ),
        }
        current_codes = output.target_codes.cpu().clone()
        metrics = loss_metrics(output)
        churn = sid_churn(previous_codes, current_codes)
        code_digest = tensor_sha256(current_codes)
        qwen_optimizer.step()
        rq_optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        step_result = {
            "step": step_index,
            "losses_and_teacher_forcing": metrics,
            "gradient_norms_before_clipping": gradient_metrics,
            "all_six_gradient_groups_finite_positive": (
                gradient_groups_finite_positive
            ),
            "clip_grad_norm_return_values": clipped_gradient_metrics,
            "sid_churn_from_previous_observation": churn,
            "target_sid_digest": code_digest,
            "target_sid_distinct_full": int(
                torch.unique(current_codes, dim=0).shape[0]
            ),
            "same_step_sid_supervision_consistent": True,
            "seconds": step_seconds,
            "optimizer_step_completed": True,
        }
        step_trace.append(step_result)
        previous_codes = current_codes
        running_state["progress"] = {
            "optimizer_steps_completed": step_index,
            "latest_generation_loss": metrics["generation"],
            "latest_teacher_forced_token_accuracy": metrics[
                "teacher_forced_token_accuracy"
            ],
            "latest_target_sid_digest": code_digest,
        }
        atomic_json(output_dir / "run_state.json", running_state)
        print(
            " ".join(
                (
                    f"step={step_index}/{args.optimizer_steps}",
                    f"total={metrics['total']:.6f}",
                    f"generation={metrics['generation']:.6f}",
                    f"teacher_token_acc={metrics['teacher_forced_token_accuracy']:.4f}",
                    f"sid_churn={churn['full_sid_churn_rate']:.4f}",
                    f"seconds={step_seconds:.3f}",
                )
            ),
            flush=True,
        )
        del output

    final_evaluation, final_codes = evaluate_fixed_batch(
        generator=generator,
        rqvae=rqvae,
        query_projection=query_projection,
        batch=batch,
        token_layout=token_layout,
        alignment_weight=args.alignment_weight,
        rq_weight=args.rq_weight,
        temperature=args.temperature,
        target_open_token_id=token_ids.target_open,
        target_close_token_id=token_ids.target_close,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    final_evaluation["sid_churn_from_last_pre_update_step"] = sid_churn(
        previous_codes, final_codes
    )
    initial_generation_loss = initial_evaluation["losses_and_teacher_forcing"][
        "generation"
    ]
    final_generation_loss = final_evaluation["losses_and_teacher_forcing"]["generation"]
    generation_loss_reduction = (
        initial_generation_loss - final_generation_loss
    ) / initial_generation_loss
    final_teacher_accuracy = final_evaluation["losses_and_teacher_forcing"][
        "teacher_forced_token_accuracy"
    ]
    final_free = final_evaluation["free_dynamic_target_generation"]
    initial_distinct_sids = initial_evaluation["target_sid_distinct_full"]
    final_distinct_sids = final_evaluation["target_sid_distinct_full"]
    acceptance_checks = {
        "all_optimizer_steps_completed": len(step_trace) == args.optimizer_steps,
        "all_six_gradient_groups_finite_positive_each_step": (
            all_gradient_groups_finite_positive
        ),
        "same_step_sid_supervision_consistent_each_observation": True,
        "generation_loss_reduction_at_least_50_percent": (
            generation_loss_reduction >= MIN_GENERATION_LOSS_REDUCTION
        ),
        "final_teacher_forced_token_accuracy_at_least_80_percent": (
            final_teacher_accuracy >= MIN_TEACHER_TOKEN_ACCURACY
        ),
        "final_free_dynamic_target_exact_rate_is_100_percent": (
            final_free["exact_rate"] >= MIN_FREE_TARGET_EXACT_RATE
        ),
        "final_free_dynamic_target_legal_rate_is_100_percent": (
            final_free["legal_current_sid_rate"] >= 1.0
        ),
        "distinct_target_sids_preserved": (
            initial_distinct_sids == args.behavior_batch_size
            and final_distinct_sids == initial_distinct_sids
        ),
        "no_valid_or_test_samples_read": True,
        "no_checkpoint_written": True,
    }
    j2_gate_passed = all(acceptance_checks.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "optimizer_steps_planned": args.optimizer_steps,
            "optimizer_steps_completed": len(step_trace),
            "fixed_train_behavior_batch": True,
            "fixed_catalog_batch": True,
            "full_training": False,
            "checkpoint_written": False,
            "sample_data_splits_read": ["train"],
            "valid_samples_read": False,
            "test_samples_read": False,
            "rqvae_initialization": (
                "fresh_method_owned_kmeans"
                if initialization_manifest is not None
                else "fresh_random_bounded_overfit_only"
            ),
        },
        "inputs": {
            "model_dir": str(model_dir),
            "model_config_sha256": sha256_file(model_dir / "config.json"),
            "data_dir": str(data_dir),
            "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
            "data_build_fingerprint": data_manifest.get("build_fingerprint"),
            "declared_train_rows": data_manifest["outputs"]["train.jsonl"]["rows"],
            "declared_valid_rows": data_manifest["outputs"]["valid.jsonl"]["rows"],
            "preflight_state": str(preflight_path),
            "preflight_state_sha256": preflight_sha256,
            "embedding_dir": str(embedding_dir),
            "embedding_manifest_signature": embedding_store.manifest_signature,
            "embedding_poi_ids_sha256": embedding_store.poi_ids_sha256,
            "embedding_shape": list(embedding_store.shape),
            "old_sid_artifacts_loaded": [],
            "rq_initialization_dir": (
                None if rq_initialization_dir is None else str(rq_initialization_dir)
            ),
            "rq_initialization_manifest_sha256": (
                None
                if rq_initialization_dir is None
                else sha256_file(rq_initialization_dir / INITIALIZATION_MANIFEST_NAME)
            ),
        },
        "fixed_data": {
            "train_rows_tokenized": len(train_examples),
            "train_sequence_length_min": min(
                example.sequence_length for example in train_examples
            ),
            "train_sequence_length_max": max(
                example.sequence_length for example in train_examples
            ),
            "selected_sample_ids_sha256": stable_sequence_sha256(
                [example.sample_id for example in selected_examples]
            ),
            "selected_target_rows_sha256": stable_sequence_sha256(
                [example.target_poi_row for example in selected_examples]
            ),
            "catalog_rows_sha256": stable_sequence_sha256(
                [int(value) for value in catalog_rows]
            ),
            "real_sample_text_saved": False,
            "raw_sid_mapping_saved": False,
            "cutoff_len": CUTOFF_LEN,
            **selection_metrics,
        },
        "model": {
            "qwen_architecture": causal_lm.config.architectures,
            "qwen_dtype": str(next(causal_lm.parameters()).dtype),
            "qwen_base_vocab_size": base_vocab_size,
            "qwen_fresh_vocab_size": int(causal_lm.config.vocab_size),
            "qwen_hidden_size": int(causal_lm.config.hidden_size),
            "qwen_input_output_embeddings_tied": True,
            "qwen_added_embedding_initial_sha256": (added_embedding_initial_sha256),
            "qwen_added_embedding_final_sha256": tensor_sha256(
                causal_lm.get_input_embeddings().weight[base_vocab_size:]
            ),
            "rqvae_encoder": [1024, 512, 256],
            "rqvae_codebook_sizes": list(CODEBOOK_SIZES),
            "rqvae_initial_sha256": rq_initial_sha256,
            "rqvae_initialization_metrics": (
                None
                if initialization_manifest is None
                else initialization_manifest["kmeans"]["levels"]
            ),
            "rqvae_final_sha256": module_sha256(rqvae),
            "query_projection": [int(causal_lm.config.hidden_size), 256],
            "query_projection_initial_sha256": projection_initial_sha256,
            "query_projection_final_sha256": module_sha256(query_projection),
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
            "gradient_clip_norm_per_group": 1.0,
            "update_schedule": "one_qwen_projection_step_and_one_rqvae_step_per_behavior_batch",
        },
        "initial_evaluation": initial_evaluation,
        "step_trace": step_trace,
        "final_evaluation": final_evaluation,
        "acceptance": {
            "thresholds": {
                "minimum_generation_loss_reduction": MIN_GENERATION_LOSS_REDUCTION,
                "minimum_teacher_forced_token_accuracy": MIN_TEACHER_TOKEN_ACCURACY,
                "minimum_free_dynamic_target_exact_rate": MIN_FREE_TARGET_EXACT_RATE,
            },
            "observed_generation_loss_reduction": generation_loss_reduction,
            "observed_initial_distinct_target_sids": initial_distinct_sids,
            "observed_final_distinct_target_sids": final_distinct_sids,
            "checks": acceptance_checks,
            "j2_bounded_overfit_gate_passed": j2_gate_passed,
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
            "memory_allocated_before_training_bytes": memory_before_training,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "training_and_evaluation_seconds": time.perf_counter() - training_started,
        },
    }
    atomic_json(output_dir / "run_state.json", result)
    return result


def main() -> int:
    args = parse_args()
    try:
        result = run_overfit(args)
    except Exception as error:
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
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "valid_samples_read": False,
                        "test_samples_read": False,
                    }
                )
                atomic_json(state_path, previous)
        print(f"TIGER-Joint 有界过拟合失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

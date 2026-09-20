#!/usr/bin/env python3
"""Evaluate one frozen TIGER-Joint checkpoint on the fixed Validation 10k."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.methods.tiger_joint.catalog import PoiEmbeddingStore  # noqa: E402
from poi_gr.methods.tiger_joint.evaluation import (  # noqa: E402
    JointBucketIndex,
    JointGeneratedCandidate,
    JointLegalPathConstraint,
    JointSidCandidateParser,
    TigerJointEvaluationError,
    compute_joint_sid_metrics,
    empty_bucket_metrics,
    finalize_bucket_metrics,
    materialize_joint_prompt,
    update_bucket_metrics,
)
from poi_gr.methods.tiger_joint.initialization import (  # noqa: E402
    build_fresh_rqvae,
    module_sha256,
)
from poi_gr.methods.tiger_joint.preparation import (  # noqa: E402
    build_sid_token_layout,
    load_dynamic_token_ids,
    parse_sid_free_record,
    tokenize_dynamic_record,
)
from poi_gr.pid.trie import sha256_file  # noqa: E402
from poi_gr.sft.evaluation import (  # noqa: E402
    build_reference_aligned_validation_subset,
    load_lf_tokenizer_and_template,
    validate_split_manifest,
)


CHECKPOINT_SCHEMA_VERSION = "tiger-joint-checkpoint-v3"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "tiger-joint-checkpoint-manifest-v3"
CODEBOOK_SIZES = (1024, 1024, 1024)
TIGER_BUCKET_BASELINE = {
    1: 0.5553,
    3: 0.7964,
    5: 0.8510,
    10: 0.8806,
}
JOINT_UNCONSTRAINED_BUCKET_BASELINE = {
    1: 0.2284,
    3: 0.3290,
    5: 0.3477,
    10: 0.3497,
}


class TigerJointBucketEvalError(TigerJointEvaluationError):
    """Raised when the formal checkpoint evaluation cannot be audited."""


@dataclass(frozen=True)
class CheckpointMetadata:
    path: Path
    epoch: int
    optimizer_step: int
    config_signature: str
    files: Mapping[str, str]


@dataclass(frozen=True)
class EvaluationExample:
    sample_id: str
    order_id: str
    searchid: str
    target_poi_id: str
    target_poi_row: int
    target_codes: tuple[int, int, int]
    prompt_ids: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 TIGER-Joint 终态 checkpoint 重建全量动态 SID，并在固定 Validation "
            "10k 上执行 Beam=10 的三级 Bucket 召回评测；可显式启用仅用于诊断的"
            "终态全目录合法路径约束。"
        )
    )
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--reference-validation-subset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-sid-artifact-dir",
        type=Path,
        help=(
            "复用同一 checkpoint 与 BGE 行序已经完整导出的 SID；会重新核验哈希并"
            "复制到当前 output-dir。"
        ),
    )
    parser.add_argument("--expected-step", type=int, default=44454)
    parser.add_argument("--expected-epoch", type=int, default=3)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        choices=(16, 8, 4, 2, 1),
        default=8,
    )
    parser.add_argument("--sid-batch-size", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument(
        "--legal-path-constraint",
        action="store_true",
        help="Beam 每一步只允许沿 epoch-3 全目录中真实存在的三级 SID Prefix 扩展。",
    )
    parser.add_argument(
        "--skip-data-hash",
        action="store_true",
        help="仅调试时跳过 597,421 行 Validation 文件哈希，正式实验不得使用。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查输入、checkpoint、固定集与宿主 GPU，不创建评测产物。",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise TigerJointBucketEvalError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TigerJointBucketEvalError(f"{name} JSON 非法：{path}") from error
    if not isinstance(value, dict):
        raise TigerJointBucketEvalError(f"{name} 必须是 object")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_checkpoint(
    checkpoint: Path,
    *,
    expected_step: int,
    expected_epoch: int,
) -> CheckpointMetadata:
    checkpoint = checkpoint.resolve()
    if checkpoint.name != f"checkpoint-step-{expected_step}":
        raise TigerJointBucketEvalError("checkpoint 目录名与 expected step 不一致")
    manifest = load_json_object(
        checkpoint / "checkpoint_manifest.json", "checkpoint manifest"
    )
    if (
        manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("optimizer_step") != expected_step
        or manifest.get("epoch") != expected_epoch
        or manifest.get("epoch_complete") is not True
    ):
        raise TigerJointBucketEvalError("checkpoint manifest 不是完整指定 epoch")
    signature = manifest.get("config_signature")
    files = manifest.get("files")
    if not isinstance(signature, str) or len(signature) != 64:
        raise TigerJointBucketEvalError("checkpoint config signature 无效")
    if not isinstance(files, dict) or not files:
        raise TigerJointBucketEvalError("checkpoint manifest 缺少文件哈希")
    required = {
        "model.safetensors",
        "joint_state.pt",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
    }
    if not required.issubset(files):
        raise TigerJointBucketEvalError("checkpoint manifest 缺少评测必要文件")
    for name, expected_hash in files.items():
        path = checkpoint / name
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise TigerJointBucketEvalError(f"checkpoint 文件哈希无效：{name}")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise TigerJointBucketEvalError(f"checkpoint 文件缺失或损坏：{name}")
    return CheckpointMetadata(
        path=checkpoint,
        epoch=expected_epoch,
        optimizer_step=expected_step,
        config_signature=signature,
        files={str(name): str(value) for name, value in files.items()},
    )


def validate_catalog_contract(
    *,
    valid_file: Path,
    embedding_store: PoiEmbeddingStore,
) -> dict[str, Any]:
    manifest = load_json_object(valid_file.parent / "manifest.json", "joint data manifest")
    contract = manifest.get("embedding_catalog")
    if not isinstance(contract, dict):
        raise TigerJointBucketEvalError("joint data manifest 缺少 BGE 目录合同")
    expected = {
        "manifest_signature": embedding_store.manifest_signature,
        "poi_ids_sha256": embedding_store.poi_ids_sha256,
        "shape": list(embedding_store.shape),
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise TigerJointBucketEvalError("联合数据与 BGE 行序/向量 manifest 不一致")
    if manifest.get("isolation", {}).get("old_sid_mapping_loaded") is not False:
        raise TigerJointBucketEvalError("联合数据没有声明旧 SID 隔离")
    return manifest


def read_reference_row_count(path: Path) -> int:
    rows = 0
    with path.open("rb") as stream:
        for raw_line in stream:
            rows += 1
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise TigerJointBucketEvalError("固定 Validation 子集 JSONL 非法") from error
            if not isinstance(value, dict):
                raise TigerJointBucketEvalError("固定 Validation 子集每行必须是 object")
    if rows != 10_000:
        raise TigerJointBucketEvalError(f"固定 Validation 子集必须为 10,000 行，实际 {rows}")
    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise TigerJointBucketEvalError(
                    f"对齐 Validation 第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(value, dict):
                raise TigerJointBucketEvalError("对齐 Validation 每行必须是 object")
            records.append(value)
    return records


def load_rqvae_from_checkpoint(
    metadata: CheckpointMetadata,
    *,
    expected_model_sha256: str | None,
    device: Any,
) -> Any:
    import torch

    state = torch.load(
        metadata.path / "joint_state.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(state, dict) or (
        state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or state.get("config_signature") != metadata.config_signature
        or state.get("optimizer_step") != metadata.optimizer_step
        or state.get("epoch") != metadata.epoch
        or state.get("epoch_complete") is not True
    ):
        raise TigerJointBucketEvalError("joint_state 与 checkpoint manifest 不一致")
    rqvae_state = state.get("rqvae_state_dict")
    if not isinstance(rqvae_state, dict):
        raise TigerJointBucketEvalError("joint_state 缺少 RQ-VAE state_dict")
    rqvae = build_fresh_rqvae()
    rqvae.load_state_dict(rqvae_state, strict=True)
    actual_model_sha256 = module_sha256(rqvae)
    if expected_model_sha256 is not None and actual_model_sha256 != expected_model_sha256:
        raise TigerJointBucketEvalError("终态 RQ-VAE 模块哈希与训练 run_state 不一致")
    del rqvae_state, state
    gc.collect()
    return rqvae.to(device=device, dtype=torch.float32).eval(), actual_model_sha256


def export_full_sid(
    *,
    rqvae: Any,
    embedding_store: PoiEmbeddingStore,
    output_dir: Path,
    checkpoint: CheckpointMetadata,
    model_sha256: str,
    batch_size: int,
    device: Any,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    import torch

    if batch_size <= 0:
        raise TigerJointBucketEvalError("sid-batch-size 必须为正数")
    sid_path = output_dir / "sid_codes.npy"
    temporary_path = output_dir / ".sid_codes.npy.tmp"
    codes = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.int32,
        shape=(embedding_store.poi_count, 3),
    )
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, embedding_store.poi_count, batch_size):
                stop = min(start + batch_size, embedding_store.poi_count)
                values = torch.from_numpy(
                    np.ascontiguousarray(
                        embedding_store.embeddings[start:stop],
                        dtype=np.float32,
                    )
                ).to(device)
                batch_codes = rqvae.encode_codes(values).cpu().numpy()
                codes[start:stop] = batch_codes.astype(np.int32, copy=False)
        codes.flush()
        del codes
        os.replace(temporary_path, sid_path)
    finally:
        if "codes" in locals():
            del codes
        temporary_path.unlink(missing_ok=True)
    sid_codes = np.load(sid_path, mmap_mode="r", allow_pickle=False)
    sid_sha256 = sha256_file(sid_path)
    manifest = {
        "schema_version": "sid-input-v1",
        "experiment_id": output_dir.name,
        "method": "tiger_joint_epoch3_dynamic_rqvae",
        "sid_codes": {
            "path": sid_path.name,
            "shape": [embedding_store.poi_count, 3],
            "dtype": "int32",
            "sha256": sid_sha256,
        },
        "poi_ids": {
            "path": str(embedding_store.poi_ids_path),
            "sha256": embedding_store.poi_ids_sha256,
        },
        "codebook_sizes": list(CODEBOOK_SIZES),
        "checkpoint": {
            "path": str(checkpoint.path),
            "epoch": checkpoint.epoch,
            "optimizer_step": checkpoint.optimizer_step,
            "model_sha256": checkpoint.files["model.safetensors"],
            "joint_state_sha256": checkpoint.files["joint_state.pt"],
            "rqvae_module_sha256": model_sha256,
        },
        "embedding": {
            "manifest": str(embedding_store.manifest_path),
            "manifest_signature": embedding_store.manifest_signature,
            "shape": list(embedding_store.shape),
        },
        "exported_at": utc_now(),
        "export_seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "sid_manifest.json", manifest)
    metrics = compute_joint_sid_metrics(sid_codes, CODEBOOK_SIZES)
    atomic_json(output_dir / "sid_metrics.json", metrics)
    return sid_codes, manifest, metrics


def reuse_full_sid(
    *,
    source_dir: Path,
    output_dir: Path,
    checkpoint: CheckpointMetadata,
    embedding_store: PoiEmbeddingStore,
    expected_rqvae_sha256: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Validate and isolate one deterministic full-catalog SID export."""

    source_dir = source_dir.resolve()
    source_manifest_path = source_dir / "sid_manifest.json"
    source_metrics_path = source_dir / "sid_metrics.json"
    source_manifest = load_json_object(source_manifest_path, "复用 SID manifest")
    metrics = load_json_object(source_metrics_path, "复用 SID metrics")
    codes_spec = source_manifest.get("sid_codes")
    checkpoint_spec = source_manifest.get("checkpoint")
    embedding_spec = source_manifest.get("embedding")
    if (
        source_manifest.get("schema_version") != "sid-input-v1"
        or source_manifest.get("codebook_sizes") != list(CODEBOOK_SIZES)
        or not isinstance(codes_spec, dict)
        or codes_spec.get("shape") != [embedding_store.poi_count, 3]
        or codes_spec.get("dtype") != "int32"
        or not isinstance(checkpoint_spec, dict)
        or checkpoint_spec.get("epoch") != checkpoint.epoch
        or checkpoint_spec.get("optimizer_step") != checkpoint.optimizer_step
        or checkpoint_spec.get("model_sha256")
        != checkpoint.files["model.safetensors"]
        or checkpoint_spec.get("joint_state_sha256")
        != checkpoint.files["joint_state.pt"]
        or checkpoint_spec.get("rqvae_module_sha256") != expected_rqvae_sha256
        or not isinstance(embedding_spec, dict)
        or embedding_spec.get("manifest_signature")
        != embedding_store.manifest_signature
        or metrics.get("status") != "completed"
        or metrics.get("shape") != [embedding_store.poi_count, 3]
        or metrics.get("codebook_sizes") != list(CODEBOOK_SIZES)
    ):
        raise TigerJointBucketEvalError("复用 SID 与正式 checkpoint/BGE 合同不一致")
    raw_path = codes_spec.get("path")
    expected_sha256 = codes_spec.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_sha256, str):
        raise TigerJointBucketEvalError("复用 SID codes 路径或哈希无效")
    source_codes_path = Path(raw_path)
    if not source_codes_path.is_absolute():
        source_codes_path = source_dir / source_codes_path
    if not source_codes_path.is_file() or sha256_file(source_codes_path) != expected_sha256:
        raise TigerJointBucketEvalError("复用 SID codes 文件缺失或哈希不一致")
    source_codes = np.load(source_codes_path, mmap_mode="r", allow_pickle=False)
    if source_codes.shape != (embedding_store.poi_count, 3) or (
        source_codes.dtype != np.int32
    ):
        raise TigerJointBucketEvalError("复用 SID codes NPY shape/dtype 无效")

    destination = output_dir / "sid_codes.npy"
    temporary = output_dir / ".sid_codes.npy.tmp"
    try:
        with source_codes_path.open("rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(destination) != expected_sha256:
        raise TigerJointBucketEvalError("隔离复制后的 SID codes 哈希不一致")
    manifest = {
        **source_manifest,
        "experiment_id": output_dir.name,
        "sid_codes": {**codes_spec, "path": destination.name},
        "reuse": {
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_metrics": str(source_metrics_path),
            "source_metrics_sha256": sha256_file(source_metrics_path),
            "validated_at": utc_now(),
            "reason": "same checkpoint, RQ-VAE hash, BGE manifest, row order and full SID hash",
        },
    }
    atomic_json(output_dir / "sid_manifest.json", manifest)
    atomic_json(output_dir / "sid_metrics.json", metrics)
    return (
        np.load(destination, mmap_mode="r", allow_pickle=False),
        manifest,
        metrics,
    )


def verify_target_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    poi_ids_path: Path,
) -> None:
    expected_by_row: dict[int, str] = {}
    for record in records:
        row = record.get("target_poi_row")
        poi_id = record.get("target_poi_id")
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or row < 0
            or not isinstance(poi_id, str)
            or not poi_id
        ):
            raise TigerJointBucketEvalError("固定子集 target POI 行/ID 无效")
        existing = expected_by_row.get(row)
        if existing is not None and existing != poi_id:
            raise TigerJointBucketEvalError("同一 BGE 行对应多个 target POI ID")
        expected_by_row[row] = poi_id
    remaining = set(expected_by_row)
    with poi_ids_path.open("r", encoding="utf-8") as stream:
        for row, line in enumerate(stream):
            if row not in remaining:
                continue
            try:
                poi_id = json.loads(line)
            except json.JSONDecodeError as error:
                raise TigerJointBucketEvalError("BGE POI 行序 JSON 非法") from error
            if poi_id != expected_by_row[row]:
                raise TigerJointBucketEvalError("固定集 target_poi_row 与 POI ID 不一致")
            remaining.remove(row)
            if not remaining:
                break
    if remaining:
        raise TigerJointBucketEvalError("固定集 target_poi_row 超出 BGE 行序")


def build_evaluation_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    template: Any,
    token_ids: Any,
    token_layout: Any,
    sid_codes: np.ndarray,
    cutoff_len: int,
) -> list[EvaluationExample]:
    examples: list[EvaluationExample] = []
    seen_business_keys: set[tuple[str, str]] = set()
    for record in records:
        text_record = parse_sid_free_record(record, expected_split="valid")
        dynamic = tokenize_dynamic_record(
            text_record,
            tokenizer=tokenizer,
            template=template,
            token_ids=token_ids,
            cutoff_len=cutoff_len,
        )
        order_id = record.get("order_id")
        searchid = record.get("searchid")
        target_poi_id = record.get("target_poi_id")
        if not all(
            isinstance(value, str) and value
            for value in (order_id, searchid, target_poi_id)
        ):
            raise TigerJointBucketEvalError("固定子集业务键或 target POI ID 无效")
        business_key = (str(order_id), str(searchid))
        if business_key in seen_business_keys:
            raise TigerJointBucketEvalError("固定子集业务键重复")
        seen_business_keys.add(business_key)
        target_codes_array = sid_codes[text_record.target_poi_row]
        target_codes = tuple(int(value) for value in target_codes_array)
        prompt_ids = materialize_joint_prompt(
            dynamic,
            sid_codes=sid_codes,
            token_layout=token_layout,
        )
        examples.append(
            EvaluationExample(
                sample_id=text_record.sample_id,
                order_id=str(order_id),
                searchid=str(searchid),
                target_poi_id=str(target_poi_id),
                target_poi_row=text_record.target_poi_row,
                target_codes=(target_codes[0], target_codes[1], target_codes[2]),
                prompt_ids=prompt_ids,
            )
        )
    return examples


def validate_prompts(
    examples: Sequence[EvaluationExample],
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    if not examples:
        raise TigerJointBucketEvalError("固定评测样本不能为空")
    if tokenizer.padding_side != "left":
        raise TigerJointBucketEvalError("批量生成必须使用 left padding")
    lengths = np.asarray([len(example.prompt_ids) for example in examples])
    checks: list[dict[str, Any]] = []
    for example in examples[:100]:
        decoded = tokenizer.decode(example.prompt_ids, skip_special_tokens=False)
        if not decoded.endswith("<|im_start|>assistant\n"):
            raise TigerJointBucketEvalError("Prompt 未停在 Assistant 生成起点")
        if "<think>" in decoded or "</think>" in decoded:
            raise TigerJointBucketEvalError("qwen3_nothink Prompt 含 thinking 标记")
        if "<TARGET_POI>" in decoded:
            raise TigerJointBucketEvalError("Prompt 泄露 target wrapper")
        checks.append(
            {
                "sample_id": example.sample_id,
                "prompt_token_count": len(example.prompt_ids),
                "prompt_ids_sha256": hashlib.sha256(
                    np.asarray(example.prompt_ids, dtype=np.int32).tobytes()
                ).hexdigest(),
            }
        )
    return {
        "schema_version": "tiger-joint-prompt-validation-v1",
        "status": "passed",
        "checked_samples": len(checks),
        "all_samples": len(examples),
        "padding_side": tokenizer.padding_side,
        "prompt_length": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "checks": checks,
    }


def pad_prompts(
    prompts: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full(
        (len(prompts), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompts):
        length = len(prompt)
        input_ids[row, width - length :] = torch.tensor(
            prompt, dtype=torch.long, device=device
        )
        attention_mask[row, width - length :] = 1
    return input_ids, attention_mask


def candidate_trace_row(
    *,
    row_index: int,
    example: EvaluationExample,
    candidates: Sequence[JointGeneratedCandidate],
    ranking: Any,
    index: JointBucketIndex,
) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    for beam_rank, candidate in enumerate(candidates, start=1):
        bucket_size = (
            index.bucket_size(candidate.bucket) if candidate.bucket is not None else 0
        )
        candidate_rows.append(
            {
                "beam_rank": beam_rank,
                "sequence_score": candidate.score,
                "sequence_token_ids": list(candidate.sequence_token_ids),
                "bucket": list(candidate.bucket) if candidate.bucket is not None else None,
                "bucket_size": bucket_size,
                "prefix_error": candidate.prefix_error,
                "target_close_valid": candidate.target_close_valid,
            }
        )
    unique_rows = [
        {
            "codes": list(bucket),
            "first_beam_rank": beam_rank,
            "bucket_size": bucket_size,
        }
        for bucket, beam_rank, bucket_size in zip(
            ranking.unique_buckets,
            ranking.unique_bucket_first_beam_ranks,
            ranking.unique_bucket_sizes,
            strict=True,
        )
    ]
    return {
        "schema_version": "tiger-joint-bucket-candidate-trace-v1",
        "row_index": row_index,
        "sample_id": example.sample_id,
        "order_id": example.order_id,
        "searchid": example.searchid,
        "target_poi_id": example.target_poi_id,
        "target_poi_row": example.target_poi_row,
        "target_bucket": list(example.target_codes),
        "target_bucket_size": index.bucket_size(example.target_codes),
        "raw_slot_bucket_target_rank": ranking.raw_slot_target_rank,
        "unique_bucket_target_rank": ranking.unique_target_rank,
        "unique_expandable_buckets": unique_rows,
        "candidates": candidate_rows,
    }


def evaluate_model(
    *,
    checkpoint: CheckpointMetadata,
    tokenizer: Any,
    examples: Sequence[EvaluationExample],
    parser: JointSidCandidateParser,
    token_layout: Any,
    index: JointBucketIndex,
    output_dir: Path,
    num_beams: int,
    batch_size: int,
    chunk_size: int,
    legal_path_constraint: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint.path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    model.config.use_cache = True
    if int(model.config.vocab_size) != len(tokenizer):
        raise TigerJointBucketEvalError("checkpoint 模型词表与 Tokenizer 不一致")
    parts_dir = output_dir / "candidate_trace_parts"
    parts_dir.mkdir()
    metrics = empty_bucket_metrics()
    parts: list[dict[str, Any]] = []
    legal_path_cache: dict[tuple[int, ...], list[int]] = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for chunk_start in range(0, len(examples), chunk_size):
        chunk = examples[chunk_start : chunk_start + chunk_size]
        traces: list[dict[str, Any]] = []
        for batch_start in range(0, len(chunk), batch_size):
            batch = chunk[batch_start : batch_start + batch_size]
            input_ids, attention_mask = pad_prompts(
                [example.prompt_ids for example in batch],
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            prompt_width = int(input_ids.shape[1])
            generation_kwargs: dict[str, Any] = {}
            if legal_path_constraint:
                if tokenizer.eos_token_id is None:
                    raise TigerJointBucketEvalError("Tokenizer 缺少 EOS Token")
                generation_kwargs["prefix_allowed_tokens_fn"] = (
                    JointLegalPathConstraint(
                        index=index,
                        token_layout=token_layout,
                        target_open_token_id=parser.target_open_token_id,
                        target_close_token_id=parser.target_close_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        prompt_width=prompt_width,
                        next_token_cache=legal_path_cache,
                    )
                )
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    max_new_tokens=5,
                    length_penalty=1.0,
                    early_stopping=True,
                    renormalize_logits=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_kwargs,
                )
            if generated.sequences_scores is None:
                raise TigerJointBucketEvalError("Beam Search 缺少 sequence scores")
            raw_sequences = (
                generated.sequences[:, prompt_width:].detach().cpu().tolist()
            )
            raw_scores = generated.sequences_scores.float().detach().cpu().tolist()
            expected = len(batch) * num_beams
            if len(raw_sequences) != expected or len(raw_scores) != expected:
                raise TigerJointBucketEvalError("Beam Search 返回候选数量不一致")
            for batch_index, example in enumerate(batch):
                begin = batch_index * num_beams
                end = begin + num_beams
                ranked = sorted(
                    zip(raw_sequences[begin:end], raw_scores[begin:end]),
                    key=lambda item: (
                        -float(item[1]),
                        tuple(int(value) for value in item[0]),
                    ),
                )
                candidates = [
                    parser.parse(sequence, score) for sequence, score in ranked
                ]
                ranking = update_bucket_metrics(
                    metrics,
                    target_codes=example.target_codes,
                    candidates=candidates,
                    index=index,
                )
                traces.append(
                    candidate_trace_row(
                        row_index=chunk_start + batch_start + batch_index,
                        example=example,
                        candidates=candidates,
                        ranking=ranking,
                        index=index,
                    )
                )
            del generated, input_ids, attention_mask
        part_path = parts_dir / f"part-{chunk_start // chunk_size:05d}.jsonl"
        atomic_jsonl(part_path, traces)
        parts.append(
            {
                "path": str(part_path.relative_to(output_dir)),
                "row_start": chunk_start,
                "rows": len(traces),
                "sha256": sha256_file(part_path),
            }
        )
        atomic_json(
            output_dir / "progress.json",
            {
                "schema_version": "tiger-joint-bucket-eval-progress-v1",
                "status": "running",
                "completed_rows": chunk_start + len(traces),
                "total_rows": len(examples),
                "parts": parts,
            },
        )
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    final_metrics = finalize_bucket_metrics(
        metrics,
        legal_path_constraint=legal_path_constraint,
    )
    trace_manifest = {
        "schema_version": "tiger-joint-bucket-candidate-trace-manifest-v1",
        "status": "completed",
        "rows": len(examples),
        "candidates_per_row": num_beams,
        "parts": parts,
        "ordered_parts_sha256": hashlib.sha256(
            "".join(part["sha256"] for part in parts).encode("ascii")
        ).hexdigest(),
    }
    atomic_json(output_dir / "candidate_trace_manifest.json", trace_manifest)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return final_metrics, {
        "generation_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
        "effective_batch_size": batch_size,
        "legal_path_cache_entries": len(legal_path_cache),
    }


def system_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise TigerJointBucketEvalError("正式 Bucket 评测要求 CUDA，不允许 CPU 回退")
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "visible_cuda_device_count": torch.cuda.device_count(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_step <= 0 or args.expected_epoch <= 0:
        raise TigerJointBucketEvalError("expected step/epoch 必须为正数")
    if args.num_beams != 10:
        raise TigerJointBucketEvalError("固定 10k 主协议的 Beam 必须为 10")
    if args.cutoff_len != 1024:
        raise TigerJointBucketEvalError("联合训练固定评测 cutoff_len 必须为 1024")
    if args.sid_batch_size <= 0 or args.chunk_size <= 0:
        raise TigerJointBucketEvalError("batch/chunk 参数必须为正数")
    if args.smoke_limit is not None and not 0 < args.smoke_limit <= 10_000:
        raise TigerJointBucketEvalError("smoke-limit 必须位于 1～10,000")
    valid_file = resolve(args.valid_file)
    reference_subset = resolve(args.reference_validation_subset)
    checkpoint_path = resolve(args.checkpoint)
    embedding_dir = resolve(args.embedding_dir)
    output_dir = resolve(args.output_dir)
    reuse_sid_artifact_dir = (
        resolve(args.reuse_sid_artifact_dir)
        if args.reuse_sid_artifact_dir is not None
        else None
    )

    source_manifest, source_rows, source_sha256 = validate_split_manifest(
        valid_file,
        split="valid",
        verify_hash=not args.skip_data_hash,
    )
    del source_manifest
    read_reference_row_count(reference_subset)
    checkpoint = validate_checkpoint(
        checkpoint_path,
        expected_step=args.expected_step,
        expected_epoch=args.expected_epoch,
    )
    embedding_store = PoiEmbeddingStore.from_directory(
        embedding_dir,
        load_poi_index=False,
    )
    data_manifest = validate_catalog_contract(
        valid_file=valid_file,
        embedding_store=embedding_store,
    )
    environment = system_environment()
    preflight = {
        "schema_version": "tiger-joint-bucket-eval-preflight-v1",
        "status": "passed",
        "valid_file": str(valid_file),
        "valid_rows": source_rows,
        "valid_sha256": source_sha256,
        "valid_hash_verified": not args.skip_data_hash,
        "reference_validation_subset": str(reference_subset),
        "reference_rows": 10_000,
        "reference_sha256": sha256_file(reference_subset),
        "checkpoint": str(checkpoint.path),
        "checkpoint_epoch": checkpoint.epoch,
        "checkpoint_step": checkpoint.optimizer_step,
        "checkpoint_model_sha256": checkpoint.files["model.safetensors"],
        "checkpoint_joint_state_sha256": checkpoint.files["joint_state.pt"],
        "embedding_manifest": str(embedding_store.manifest_path),
        "embedding_manifest_signature": embedding_store.manifest_signature,
        "embedding_shape": list(embedding_store.shape),
        "old_sid_artifacts_loaded": [],
        "requested_reuse_sid_artifact_dir": (
            str(reuse_sid_artifact_dir)
            if reuse_sid_artifact_dir is not None
            else None
        ),
        "legal_path_constraint": args.legal_path_constraint,
        "environment": environment,
    }
    if args.preflight_only:
        return preflight
    if args.skip_data_hash:
        raise TigerJointBucketEvalError("正式评测禁止 --skip-data-hash")
    if output_dir.exists():
        raise TigerJointBucketEvalError(f"output-dir 已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "preflight.json", preflight)
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    serialized_args["output_dir"] = str(output_dir)
    atomic_json(
        output_dir / "run_state.json",
        {
            "schema_version": "tiger-joint-bucket-eval-run-v1",
            "status": "running",
            "started_at": utc_now(),
            "config": serialized_args,
        },
    )
    subset = build_reference_aligned_validation_subset(
        valid_file,
        reference_subset,
        output_dir,
        source_rows=source_rows,
        source_sha256=source_sha256,
    )
    records = load_records(subset.data_path)
    if args.smoke_limit is None and len(records) != 10_000:
        raise TigerJointBucketEvalError("正式对齐子集不是 10,000 行")
    if args.smoke_limit is not None:
        records = records[: args.smoke_limit]
    verify_target_rows(records, poi_ids_path=embedding_store.poi_ids_path)

    import torch

    device = torch.device("cuda:0")
    training_state_path = checkpoint.path.parent / "run_state.json"
    training_state = load_json_object(training_state_path, "training run_state")
    if training_state.get("status") != "completed":
        raise TigerJointBucketEvalError("联合训练 run_state 未完成")
    expected_rqvae_sha256 = training_state.get("model", {}).get(
        "rqvae_final_sha256"
    )
    if not isinstance(expected_rqvae_sha256, str):
        raise TigerJointBucketEvalError("training run_state 缺少终态 RQ-VAE 哈希")
    if reuse_sid_artifact_dir is None:
        rqvae, rqvae_sha256 = load_rqvae_from_checkpoint(
            checkpoint,
            expected_model_sha256=expected_rqvae_sha256,
            device=device,
        )
        sid_codes, sid_manifest, sid_metrics = export_full_sid(
            rqvae=rqvae,
            embedding_store=embedding_store,
            output_dir=output_dir,
            checkpoint=checkpoint,
            model_sha256=rqvae_sha256,
            batch_size=args.sid_batch_size,
            device=device,
        )
        del rqvae
        gc.collect()
        torch.cuda.empty_cache()
    else:
        sid_codes, sid_manifest, sid_metrics = reuse_full_sid(
            source_dir=reuse_sid_artifact_dir,
            output_dir=output_dir,
            checkpoint=checkpoint,
            embedding_store=embedding_store,
            expected_rqvae_sha256=expected_rqvae_sha256,
        )

    tokenizer, template = load_lf_tokenizer_and_template(
        checkpoint.path,
        project_root=PROJECT_ROOT,
    )
    token_ids = load_dynamic_token_ids(tokenizer)
    token_layout = build_sid_token_layout(tokenizer, CODEBOOK_SIZES)
    examples = build_evaluation_examples(
        records,
        tokenizer=tokenizer,
        template=template,
        token_ids=token_ids,
        token_layout=token_layout,
        sid_codes=sid_codes,
        cutoff_len=args.cutoff_len,
    )
    prompt_validation = validate_prompts(examples, tokenizer=tokenizer)
    atomic_json(output_dir / "prompt_validation.json", prompt_validation)
    index = JointBucketIndex.from_codes(sid_codes, CODEBOOK_SIZES)
    parser = JointSidCandidateParser(
        token_layout=token_layout,
        target_open_token_id=token_ids.target_open,
        target_close_token_id=token_ids.target_close,
    )
    bucket_metrics, generation_runtime = evaluate_model(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        examples=examples,
        parser=parser,
        token_layout=token_layout,
        index=index,
        output_dir=output_dir,
        num_beams=args.num_beams,
        batch_size=args.per_device_eval_batch_size,
        chunk_size=args.chunk_size,
        legal_path_constraint=args.legal_path_constraint,
    )
    comparison = {
        "baseline": "TIGER epoch 3 fixed Validation 10k unique Bucket HR",
        "baseline_values": {
            f"unique_bucket_hr@{cutoff}": value
            for cutoff, value in TIGER_BUCKET_BASELINE.items()
        },
        "delta_percentage_points": {
            f"unique_bucket_hr@{cutoff}": 100.0
            * (
                bucket_metrics[f"unique_bucket_hr@{cutoff}"]
                - TIGER_BUCKET_BASELINE[cutoff]
            )
            for cutoff in TIGER_BUCKET_BASELINE
        },
        "passes_all_registered_cutoffs": all(
            bucket_metrics[f"unique_bucket_hr@{cutoff}"]
            >= TIGER_BUCKET_BASELINE[cutoff]
            for cutoff in TIGER_BUCKET_BASELINE
        ),
        "decode_protocol_matched": not args.legal_path_constraint,
        "epoch1_epoch2_comparison_available": False,
        "note": (
            "约束模式与 TIGER 无约束基线的解码协议不同，只提供数值背景，不能作"
            "严格同口径优劣结论。"
            if args.legal_path_constraint
            else "本轮只评测 epoch 3，不能据此判断其优于 epoch 1/2。"
        ),
    }
    comparison_to_unconstrained_joint = (
        {
            "baseline": (
                "TIGER-Joint epoch 3 EXP-20260829-01 fixed Validation 10k "
                "unconstrained unique Bucket HR"
            ),
            "same_checkpoint": True,
            "same_validation_business_keys": True,
            "only_decoding_constraint_changed": True,
            "baseline_values": {
                f"unique_bucket_hr@{cutoff}": value
                for cutoff, value in JOINT_UNCONSTRAINED_BUCKET_BASELINE.items()
            },
            "delta_percentage_points": {
                f"unique_bucket_hr@{cutoff}": 100.0
                * (
                    bucket_metrics[f"unique_bucket_hr@{cutoff}"]
                    - JOINT_UNCONSTRAINED_BUCKET_BASELINE[cutoff]
                )
                for cutoff in JOINT_UNCONSTRAINED_BUCKET_BASELINE
            },
        }
        if args.legal_path_constraint
        else None
    )
    result = {
        "schema_version": "tiger-joint-fixed10k-bucket-result-v1",
        "status": "completed",
        "finished_at": utc_now(),
        "scope": {
            "split": "valid",
            "rows": len(examples),
            "test_read": False,
            "checkpoint_epoch": checkpoint.epoch,
            "checkpoint_step": checkpoint.optimizer_step,
            "decoding": (
                "epoch3 full-catalog legal-path constrained Beam=10"
                if args.legal_path_constraint
                else "unconstrained Beam=10"
            ),
            "legal_path_constraint": args.legal_path_constraint,
            "collision_token": False,
            "bucket": "epoch3 full-catalog [S1,S2,S3]",
        },
        "inputs": {
            "valid_file": str(valid_file),
            "valid_sha256": source_sha256,
            "reference_subset": str(reference_subset),
            "reference_subset_sha256": sha256_file(reference_subset),
            "aligned_subset": str(subset.data_path),
            "aligned_subset_sha256": subset.sha256,
            "business_keys_sha256": subset.manifest["business_keys_sha256"],
            "checkpoint": str(checkpoint.path),
            "model_sha256": checkpoint.files["model.safetensors"],
            "joint_state_sha256": checkpoint.files["joint_state.pt"],
            "embedding_manifest_signature": embedding_store.manifest_signature,
            "data_build_fingerprint": data_manifest.get("build_fingerprint"),
        },
        "sid_artifact": sid_manifest,
        "sid_metrics": sid_metrics,
        "training_probe": training_state.get("probe"),
        "prompt_validation": prompt_validation,
        "bucket_metrics": bucket_metrics,
        "comparison_to_tiger": comparison,
        "comparison_to_unconstrained_joint": comparison_to_unconstrained_joint,
        "environment": environment,
        "runtime": generation_runtime,
        "git": training_state.get("git"),
    }
    atomic_json(output_dir / "result.json", result)
    atomic_json(
        output_dir / "run_state.json",
        {
            "schema_version": "tiger-joint-bucket-eval-run-v1",
            "status": "completed",
            "finished_at": result["finished_at"],
            "result": str(output_dir / "result.json"),
            "result_sha256": sha256_file(output_dir / "result.json"),
            "evaluated_rows": len(examples),
        },
    )
    (output_dir / "progress.json").unlink(missing_ok=True)
    return result


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    try:
        result = run(args)
    except Exception as error:
        if output_dir.is_dir() and (output_dir / "run_state.json").is_file():
            try:
                atomic_json(
                    output_dir / "run_state.json",
                    {
                        "schema_version": "tiger-joint-bucket-eval-run-v1",
                        "status": "failed",
                        "finished_at": utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
            except OSError:
                pass
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    summary = {
        "status": result["status"],
        "result": str(output_dir / "result.json"),
        "rows": result["scope"]["rows"],
        "sid_basic": result["sid_metrics"]["basic"],
        "sid_layers": [
            {
                key: layer[key]
                for key in (
                    "level",
                    "used_token_count",
                    "codebook_utilization_ratio",
                    "normalized_entropy",
                    "kish_effective_code_count",
                    "max_code_share",
                )
            }
            for layer in result["sid_metrics"]["layers"]
        ],
        "bucket_metrics": result["bucket_metrics"],
        "comparison_to_tiger": result["comparison_to_tiger"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

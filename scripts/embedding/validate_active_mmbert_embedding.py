#!/usr/bin/env python3
"""Validate a completed active-catalog MMBERT embedding artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CHECKPOINT = Path(
    "/ofs/map_search/xiaolu/multilingual_search/recall/mmBERT-emb/"
    "output_checkpoint/output_4gpus_1pos_30neg_add_mlp_v2_plus/"
    "checkpoint-00612000"
)


class ValidationError(ValueError):
    """Raised when the active MMBERT artifact violates its frozen contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验 active MMBERT 128 维向量、ID 行序、有限值与归一化。"
    )
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--target-poi-ids", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.expected_rows <= 0 or args.chunk_rows <= 0:
        raise ValidationError("expected_rows 和 chunk_rows 必须大于 0")
    embedding_dir = resolve(args.embedding_dir)
    target_ids = resolve(args.target_poi_ids)
    manifest_path = embedding_dir / "manifest.json"
    progress_path = embedding_dir / "progress.json"
    vectors_path = embedding_dir / "embeddings.npy"
    output_ids = embedding_dir / "poi_ids.jsonl"
    for path in (manifest_path, progress_path, vectors_path, output_ids, target_ids):
        if not path.is_file():
            raise ValidationError(f"缺少正式输入：{path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValidationError("Embedding manifest 状态不是 completed")
    if progress.get("status") != "completed":
        raise ValidationError("Embedding progress 状态不是 completed")
    if progress.get("next_row") != args.expected_rows:
        raise ValidationError("Embedding progress 没有覆盖全部 active POI")
    if manifest.get("input", {}).get("total_rows") != args.expected_rows:
        raise ValidationError("Embedding manifest 输入行数错误")

    model = manifest.get("model", {})
    if model.get("backend") != "mmbert_recall":
        raise ValidationError("Embedding backend 不是 mmbert_recall")
    if Path(model.get("path", "")) != EXPECTED_CHECKPOINT:
        raise ValidationError("Embedding checkpoint 不是冻结的线上 checkpoint-00612000")
    if model.get("embedding_dim") != 128:
        raise ValidationError("Embedding 维度不是训练投影头的 128 维")
    if model.get("normalize_embeddings") is not True:
        raise ValidationError("Embedding 未启用 L2 normalize")

    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (args.expected_rows, 128):
        raise ValidationError(f"Embedding shape 错误：{vectors.shape}")
    if vectors.dtype != np.dtype("float16"):
        raise ValidationError(f"Embedding dtype 错误：{vectors.dtype}")
    if manifest.get("output", {}).get("shape") != list(vectors.shape):
        raise ValidationError("manifest shape 与 NPY 不一致")
    if manifest.get("output", {}).get("dtype") != str(vectors.dtype):
        raise ValidationError("manifest dtype 与 NPY 不一致")

    target_ids_sha256 = sha256_file(target_ids)
    output_ids_sha256 = sha256_file(output_ids)
    if output_ids_sha256 != target_ids_sha256:
        raise ValidationError("Embedding POI ID 未逐字节对齐 active 主表")

    norm_sum = 0.0
    norm_min = float("inf")
    norm_max = 0.0
    for start in range(0, args.expected_rows, args.chunk_rows):
        stop = min(start + args.chunk_rows, args.expected_rows)
        block = np.asarray(vectors[start:stop], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValidationError(f"Embedding [{start}, {stop}) 包含 NaN/Inf")
        norms = np.linalg.norm(block, axis=1)
        norm_sum += float(norms.sum(dtype=np.float64))
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
    if norm_min < 0.99 or norm_max > 1.01:
        raise ValidationError(
            f"Embedding L2 norm 超出 float16 归一化范围：{norm_min}..{norm_max}"
        )

    print(
        json.dumps(
            {
                "status": "completed",
                "shape": list(vectors.shape),
                "dtype": str(vectors.dtype),
                "poi_ids_sha256": output_ids_sha256,
                "embeddings_sha256": sha256_file(vectors_path),
                "l2_norm_min": norm_min,
                "l2_norm_mean": norm_sum / args.expected_rows,
                "l2_norm_max": norm_max,
                "vectors_all_finite": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"active MMBERT Embedding 校验失败：{error}", file=sys.stderr)
        raise SystemExit(2)

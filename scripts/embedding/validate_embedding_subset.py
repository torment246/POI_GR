#!/usr/bin/env python3
"""Independently validate an exact ordered embedding subset artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ValidationError(ValueError):
    """Raised when a subset artifact violates its frozen contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "独立校验 Embedding 子集的 ID、shape、manifest，并按固定位置回查"
            "源 Embedding，要求向量逐元素完全相同。"
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--target-poi-ids", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--spot-checks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_ids(path: Path):
    with path.open("rb") as handle:
        for index, raw_line in enumerate(handle):
            try:
                value = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValidationError(f"非法 POI ID JSONL：{path}:{index + 1}") from error
            if not isinstance(value, str) or not value:
                raise ValidationError(f"非法 POI ID：{path}:{index + 1}")
            yield index, value


def fixed_indices(rows: int, count: int, seed: int) -> np.ndarray:
    if rows <= 0 or count <= 0:
        raise ValidationError("expected_rows 和 spot_checks 必须大于 0")
    anchors = np.asarray([0, rows // 2, rows - 1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    random_count = min(count, rows)
    random_indices = rng.choice(rows, size=random_count, replace=False)
    return np.unique(np.concatenate((anchors, random_indices))).astype(np.int64)


def main() -> int:
    args = parse_args()
    source_dir = resolve(args.source_dir)
    subset_dir = resolve(args.subset_dir)
    target_ids_path = resolve(args.target_poi_ids)
    manifest_path = subset_dir / "manifest.json"
    source_ids_path = source_dir / "poi_ids.jsonl"
    source_vectors_path = source_dir / "embeddings.npy"
    subset_ids_path = subset_dir / "poi_ids.jsonl"
    subset_vectors_path = subset_dir / "embeddings.npy"
    for path in (
        manifest_path,
        source_ids_path,
        source_vectors_path,
        target_ids_path,
        subset_ids_path,
        subset_vectors_path,
    ):
        if not path.is_file():
            raise ValidationError(f"缺少输入：{path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValidationError("subset manifest 状态不是 completed")
    derivation = manifest.get("derivation", {})
    if derivation.get("method") != "strict_ordered_poi_id_subsequence":
        raise ValidationError("subset derivation method 不符合契约")
    if derivation.get("exact_vector_copy") is not True:
        raise ValidationError("subset 未声明 exact_vector_copy=true")

    source_vectors = np.load(source_vectors_path, mmap_mode="r", allow_pickle=False)
    subset_vectors = np.load(subset_vectors_path, mmap_mode="r", allow_pickle=False)
    if subset_vectors.shape != (args.expected_rows, source_vectors.shape[1]):
        raise ValidationError(
            f"subset shape 错误：{subset_vectors.shape}，预期 "
            f"{(args.expected_rows, source_vectors.shape[1])}"
        )
    if subset_vectors.dtype != source_vectors.dtype:
        raise ValidationError("source/subset dtype 不一致")
    if manifest.get("output", {}).get("shape") != list(subset_vectors.shape):
        raise ValidationError("subset manifest shape 与 NPY 不一致")
    if manifest.get("output", {}).get("dtype") != str(subset_vectors.dtype):
        raise ValidationError("subset manifest dtype 与 NPY 不一致")

    target_ids_sha256 = sha256_file(target_ids_path)
    subset_ids_sha256 = sha256_file(subset_ids_path)
    if subset_ids_sha256 != target_ids_sha256:
        raise ValidationError("subset POI ID 清单未逐字节等于 active 目标清单")
    if manifest.get("output", {}).get("poi_ids_sha256") != subset_ids_sha256:
        raise ValidationError("subset manifest POI ID SHA256 不一致")

    selected = fixed_indices(args.expected_rows, args.spot_checks, args.seed)
    selected_set = set(int(value) for value in selected)
    target_samples: dict[str, int] = {}
    target_count = 0
    for target_index, poi_id in iter_ids(target_ids_path):
        target_count += 1
        if target_index in selected_set:
            target_samples[poi_id] = target_index
    if target_count != args.expected_rows:
        raise ValidationError(
            f"active ID 行数为 {target_count}，预期 {args.expected_rows}"
        )
    if len(target_samples) != len(selected):
        raise ValidationError("固定抽样 POI ID 不唯一")

    source_positions: dict[str, int] = {}
    for source_index, poi_id in iter_ids(source_ids_path):
        if poi_id in target_samples:
            source_positions[poi_id] = source_index
            if len(source_positions) == len(target_samples):
                break
    missing = sorted(set(target_samples) - set(source_positions))
    if missing:
        raise ValidationError(f"源 ID 清单缺少固定抽样 POI：{missing[:3]}")

    for poi_id, target_index in target_samples.items():
        source_index = source_positions[poi_id]
        if not np.array_equal(
            source_vectors[source_index], subset_vectors[target_index]
        ):
            raise ValidationError(
                f"向量逐元素不一致：poi_id={poi_id}, "
                f"source_row={source_index}, subset_row={target_index}"
            )

    print(
        json.dumps(
            {
                "status": "completed",
                "subset_shape": list(subset_vectors.shape),
                "dtype": str(subset_vectors.dtype),
                "poi_ids_sha256": subset_ids_sha256,
                "spot_checks": len(selected),
                "spot_checks_bitwise_equal": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, ValueError) as error:
        print(f"Embedding 子集校验失败：{error}", file=sys.stderr)
        raise SystemExit(2)

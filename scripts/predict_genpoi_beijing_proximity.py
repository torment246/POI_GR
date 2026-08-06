#!/usr/bin/env python3
"""Predict direct safe GID prefixes with a Beijing ordinal SSP head."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poi_gr.embedding import ModelConfig, _load_encoder  # noqa: E402
from poi_gr.methods.genpoi_beijing_ssp import (  # noqa: E402
    USEFUL_PREFIX_DEPTHS,
    select_safe_prefix_depth,
)
from poi_gr.pid_trie import sha256_file  # noqa: E402


OUTPUT_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("user_gid", pa.string()),
        ("predicted_lambda", pa.int8()),
        ("prefill_length", pa.int8()),
    ]
)


class BeijingProximityPredictionError(RuntimeError):
    """Raised when Beijing SSP prediction inputs are inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用北京序分类头为 GenPOI 评测样本预测安全 GID 前缀。"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--encode-buffer-size", type=int, default=8192)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise BeijingProximityPredictionError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BeijingProximityPredictionError(f"{name} JSON 非法") from error
    if not isinstance(value, dict):
        raise BeijingProximityPredictionError(f"{name} 必须是 object")
    return value


def main() -> int:
    args = parse_args()
    try:
        if args.encode_batch_size <= 0 or args.encode_buffer_size <= 0:
            raise BeijingProximityPredictionError("编码 Batch 参数必须为正整数")
        if args.limit is not None and args.limit <= 0:
            raise BeijingProximityPredictionError("--limit 必须为正整数")

        import torch
        from safetensors.torch import load_file

        if not torch.cuda.is_available():
            raise BeijingProximityPredictionError("预测需要 CUDA GPU")
        data_dir = resolve(args.data_dir)
        head_dir = resolve(args.head_dir)
        output_dir = resolve(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "predictions.parquet"
        manifest_path = output_dir / "manifest.json"
        if output_path.exists() or manifest_path.exists():
            raise BeijingProximityPredictionError("输出已存在，请使用新的输出目录")

        data_manifest = load_json(data_dir / "manifest.json", "邻近数据 manifest")
        valid_spec = data_manifest.get("outputs", {}).get("valid")
        if not isinstance(valid_spec, dict):
            raise BeijingProximityPredictionError("邻近数据 manifest 缺少 valid")
        valid_path = Path(valid_spec.get("output_file", "")).resolve()
        if not valid_path.is_file() or sha256_file(valid_path) != valid_spec.get(
            "output_sha256"
        ):
            raise BeijingProximityPredictionError("Valid Parquet 不存在或 SHA256 错误")

        head_manifest_path = head_dir / "training_manifest.json"
        head_manifest = load_json(head_manifest_path, "北京分类头 manifest")
        if head_manifest.get("schema_version") != "genpoi-beijing-proximity-head-v1":
            raise BeijingProximityPredictionError("北京分类头 schema_version 不兼容")
        if head_manifest.get("status") != "completed":
            raise BeijingProximityPredictionError("北京分类头状态不是 completed")
        head_path = Path(head_manifest.get("head_file", "")).resolve()
        if not head_path.is_file() or sha256_file(head_path) != head_manifest.get(
            "head_sha256"
        ):
            raise BeijingProximityPredictionError("北京分类头权重不存在或 SHA256 错误")
        raw_thresholds = head_manifest.get("safety_contract", {}).get(
            "calibrated_thresholds", {}
        )
        thresholds = {
            depth: float(raw_thresholds[str(depth)]) for depth in USEFUL_PREFIX_DEPTHS
        }

        model_info = head_manifest.get("model", {})
        model_path = Path(model_info.get("path", "")).resolve()
        model_config = ModelConfig(
            path=model_path,
            device="cuda",
            batch_size=args.encode_batch_size,
            encode_buffer_size=args.encode_buffer_size,
            max_seq_length=int(model_info.get("max_seq_length", 128)),
            torch_dtype="bfloat16",
            attention="sdpa",
            padding_side="right",
            normalize_embeddings=True,
            truncate_dim=None,
            prompt_name=None,
        )
        encoder, device, load_seconds = _load_encoder(model_config)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        embedding_dim = encoder.get_sentence_embedding_dimension()
        head = torch.nn.Linear(embedding_dim, len(USEFUL_PREFIX_DEPTHS)).to(
            "cuda", dtype=torch.float32
        )
        head.load_state_dict(load_file(str(head_path), device="cuda"), strict=True)
        head.eval()

        temporary = output_path.with_name(f".{output_path.name}.tmp")
        writer = pq.ParquetWriter(temporary, OUTPUT_SCHEMA, compression="zstd")
        parquet = pq.ParquetFile(valid_path)
        rows = 0
        depth_counts: Counter[int] = Counter()
        try:
            with torch.inference_mode():
                for batch in parquet.iter_batches(
                    batch_size=args.encode_buffer_size,
                    columns=["sample_id", "query", "user_gid"],
                    use_threads=True,
                ):
                    remaining = None if args.limit is None else args.limit - rows
                    if remaining is not None and remaining <= 0:
                        break
                    take = batch.num_rows if remaining is None else min(
                        batch.num_rows, remaining
                    )
                    sample_ids = batch.column("sample_id").slice(0, take).to_pylist()
                    queries = batch.column("query").slice(0, take).to_pylist()
                    user_gids = batch.column("user_gid").slice(0, take).to_pylist()
                    embeddings = np.asarray(
                        encoder.encode(
                            queries,
                            batch_size=args.encode_batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            prompt_name=None,
                        ),
                        dtype=np.float32,
                    )
                    probabilities = torch.sigmoid(
                        head(torch.from_numpy(embeddings).to("cuda"))
                    ).cpu().numpy()
                    depths = np.asarray(
                        [
                            select_safe_prefix_depth(row, thresholds)
                            for row in probabilities
                        ],
                        dtype=np.int8,
                    )
                    writer.write_table(
                        pa.Table.from_pydict(
                            {
                                "sample_id": sample_ids,
                                "user_gid": user_gids,
                                "predicted_lambda": depths,
                                "prefill_length": depths,
                            },
                            schema=OUTPUT_SCHEMA,
                        )
                    )
                    depth_counts.update(int(value) for value in depths)
                    rows += take
                    print(f"[predict] {rows:,} rows", file=sys.stderr, flush=True)
        finally:
            writer.close()
        expected_rows = int(valid_spec.get("rows", -1))
        if args.limit is None and rows != expected_rows:
            temporary.unlink(missing_ok=True)
            raise BeijingProximityPredictionError("预测行数与 Validation 不一致")
        os.replace(temporary, output_path)
        manifest = {
            "schema_version": "genpoi-beijing-ssp-predictions-v1",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "selection_strategy": "direct_ordinal_safe_prefix",
            "useful_prefix_depths": list(USEFUL_PREFIX_DEPTHS),
            "calibrated_thresholds": {
                str(depth): thresholds[depth] for depth in USEFUL_PREFIX_DEPTHS
            },
            "rows": rows,
            "limit": args.limit,
            "inference_features": ["current_query"],
            "target_fields_used": [],
            "source_validation_jsonl": valid_spec["source_file"],
            "source_validation_jsonl_sha256": valid_spec["source_sha256"],
            "head_manifest": str(head_manifest_path),
            "head_manifest_sha256": sha256_file(head_manifest_path),
            "head_sha256": head_manifest["head_sha256"],
            "model_path": str(model_path),
            "model_load_seconds": load_seconds,
            "device": device,
            "prefix_depth_counts": {
                str(depth): depth_counts[depth]
                for depth in (0, *USEFUL_PREFIX_DEPTHS)
            },
            "output_file": str(output_path),
            "output_sha256": sha256_file(output_path),
        }
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except (BeijingProximityPredictionError, OSError, ValueError, KeyError) as error:
        print(f"预测失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

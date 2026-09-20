#!/usr/bin/env python3
"""Prepare a frozen active validation subset and run the existing SSP+TCG evaluator."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.genpoi.build_proximity_data import write_split
from poi_gr.pid.trie import sha256_file
from poi_gr.sft.evaluation import (
    build_reference_aligned_validation_subset, load_ssp_predictions, validate_split_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="对齐 active GenPOI Validation 10k，复用冻结 SSP 与 TCG 评测。")
    for name in ("valid-file", "reference-validation-subset", "checkpoint", "tokenizer", "trie-dir", "head-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true", help="打印命令，不加载 GPU 或创建评测数据。")
    parser.add_argument("--prepare-only", action="store_true", help="仅对齐数据并准备 SSP 输入，不运行 GPU。")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, (ROOT / value).resolve())
    output = args.output_dir
    proximity = output / "proximity_data"
    predictions = output / "ssp_predictions"
    prediction_cmd = [sys.executable, str(ROOT / "scripts/genpoi/predict_proximity.py"),
                      "--data-dir", str(proximity), "--head-dir", str(args.head_dir),
                      "--output-dir", str(predictions), "--encode-batch-size", "256"]
    evaluation_cmd = [sys.executable, str(ROOT / "scripts/sft/evaluate_retrieval.py"),
                      "--mode", "valid-checkpoints", "--valid-file", str(args.valid_file),
                      "--reference-validation-subset", str(args.reference_validation_subset),
                      "--checkpoints", str(args.checkpoint), "--expected-checkpoint-steps", str(args.expected_step),
                      "--expected-checkpoint-epochs", "3", "--tokenizer", str(args.tokenizer),
                      "--trie-dir", str(args.trie_dir), "--ssp-predictions-dir", str(predictions),
                      "--output-dir", str(output), "--num-beams", "10", "--top-k", "10",
                      "--per-device-eval-batch-size", "32", "--chunk-size", "1000", "--cutoff-len", "1024"]
    if args.dry_run:
        print(shlex.join(prediction_cmd))
        print(shlex.join(evaluation_cmd))
        return 0
    _, rows, digest = validate_split_manifest(args.valid_file, split="valid", verify_hash=True)
    subset = build_reference_aligned_validation_subset(args.valid_file, args.reference_validation_subset,
                                                      output, source_rows=rows, source_sha256=digest)
    subset_path = subset.data_path
    if subset.row_count != 10000 or not subset_path.is_file():
        raise ValueError("未生成固定 10000 条评测子集")
    subset_hash = sha256_file(subset_path)
    proximity.mkdir(parents=True, exist_ok=True)
    manifest_path = proximity / "manifest.json"
    if manifest_path.exists():
        spec = json.loads(manifest_path.read_text())["outputs"]["valid"]
        if spec["source_sha256"] != subset_hash or sha256_file(proximity / "valid.parquet") != spec["output_sha256"]:
            raise ValueError("已存在的 SSP 输入与当前子集不一致")
    else:
        spec = write_split(subset_path, proximity / "valid.parquet", split="valid", batch_rows=100000, limit=None)
        if spec["rows"] != 10000:
            raise ValueError("SSP 输入必须为 10000 行")
        manifest_path.write_text(json.dumps({"schema_version": "genpoi-proximity-data-v1", "status": "completed",
            "classes": list(range(7)), "gid_length": 6, "query_source": "CURRENT/QUERY",
            "outputs": {"valid": spec}}, ensure_ascii=False, indent=2) + "\n")
    if args.prepare_only:
        print("固定 10k 与 SSP 输入准备完成；未运行生成评测。")
        return 0
    if not (predictions / "manifest.json").exists():
        subprocess.run(prediction_cmd, cwd=ROOT, check=True)
    prediction_manifest = json.loads((predictions / "manifest.json").read_text())
    if (prediction_manifest.get("head_manifest_sha256") != sha256_file(args.head_dir / "training_manifest.json")
            or prediction_manifest.get("head_sha256") != sha256_file(args.head_dir / "proximity_head.safetensors")):
        raise ValueError("已存在的 SSP 预测与当前冻结 head 不一致")
    load_ssp_predictions(predictions, evaluation_data_sha256=subset_hash, expected_rows=10000)
    subprocess.run(evaluation_cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Active GenPOI 评测失败：{error}", file=sys.stderr)
        raise SystemExit(2)

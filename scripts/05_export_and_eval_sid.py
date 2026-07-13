#!/usr/bin/env python3
"""Export RQ-VAE Semantic IDs and evaluate SID quality for MobilityBench POIs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("缺少 torch，请安装：pip install torch") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional but expected
    tqdm = None

from rqvae import CAURQVAE, RQVAE
from sid_eval import (
    SID_MAPPING_COLUMNS,
    build_compare_report,
    build_mode_metrics,
    build_quality_report,
    build_sid_mapping,
    clean_text,
    ensure_dir,
    ensure_parent_dir,
    fmt_float,
    read_parquet,
    validate_meta_alignment,
    validate_mapping,
    write_parquet,
)


DEFAULTS = {
    "embedding_meta": "data/embeddings/poi_embedding_meta.parquet",
    "geo_meta": "data/geo/poi_geo_meta.parquet",
    "qrels": "data/processed/mobilitybench/qrels.csv",
    "candidates": "data/processed/mobilitybench/candidates.csv",
    "semantic_checkpoint": "outputs/rqvae/rqvae_semantic.pt",
    "geo_fused_checkpoint": "outputs/rqvae/rqvae_geo_fused.pt",
    "semantic_input": "data/rqvae/rqvae_input_semantic.npy",
    "geo_fused_input": "data/rqvae/rqvae_input_geo_fused.npy",
    "semantic_mapping": "data/sid/poi_sid_mapping_semantic.parquet",
    "geo_fused_mapping": "data/sid/poi_sid_mapping_geo_fused.parquet",
    "semantic_indices": "data/sid/poi_sid_indices_semantic.npy",
    "geo_fused_indices": "data/sid/poi_sid_indices_geo_fused.npy",
    "semantic_report": "reports/sid_quality_semantic.md",
    "geo_fused_report": "reports/sid_quality_geo_fused.md",
    "compare_report": "reports/sid_quality_compare.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["semantic", "geo_fused", "both"], default="both")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--rqvae-input", default=None)
    parser.add_argument("--semantic-checkpoint", default=DEFAULTS["semantic_checkpoint"])
    parser.add_argument("--geo-fused-checkpoint", default=DEFAULTS["geo_fused_checkpoint"])
    parser.add_argument("--semantic-input", default=DEFAULTS["semantic_input"])
    parser.add_argument("--geo-fused-input", default=DEFAULTS["geo_fused_input"])
    parser.add_argument("--embedding-meta", default=DEFAULTS["embedding_meta"])
    parser.add_argument("--geo-meta", default=DEFAULTS["geo_meta"])
    parser.add_argument("--qrels", default=DEFAULTS["qrels"])
    parser.add_argument("--candidates", default=DEFAULTS["candidates"])
    parser.add_argument("--out-mapping", default=None)
    parser.add_argument("--out-indices", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-report-groups", type=int, default=1000)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("用户指定了 CUDA，但当前 torch.cuda.is_available() 为 False")
    return device


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> tuple[RQVAE, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"RQ-VAE checkpoint 不存在：{path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint 缺少 model_config 或 model_state_dict：{path}")

    model_config = dict(checkpoint["model_config"])
    if "input_dim" not in model_config and "input_dim" in checkpoint:
        model_config["input_dim"] = int(checkpoint["input_dim"])
    architecture = model_config.pop("architecture", "")
    if architecture == "cau_rqvae" or "num_labels" in model_config:
        model = CAURQVAE(**model_config)
    else:
        model = RQVAE(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def encode_sid_indices(
    model: RQVAE,
    rqvae_input: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if rqvae_input.ndim != 2:
        raise ValueError(f"RQ-VAE input must be 2D, got shape {rqvae_input.shape}")
    if rqvae_input.shape[1] != model.input_dim:
        raise ValueError(f"RQ-VAE input dim {rqvae_input.shape[1]} != checkpoint input_dim {model.input_dim}")
    if not np.isfinite(rqvae_input).all():
        raise ValueError("RQ-VAE input contains NaN or Inf")

    n_rows = rqvae_input.shape[0]
    batches = range(0, n_rows, batch_size)
    iterator = tqdm(batches, desc="Encoding SID", unit="batch") if tqdm is not None else batches
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in iterator:
            end = min(start + batch_size, n_rows)
            batch = np.asarray(rqvae_input[start:end], dtype=np.float32).copy()
            x = torch.from_numpy(batch).to(device, non_blocking=True)
            indices, _ = model.encode(x)
            parts.append(indices.detach().cpu().numpy().astype(np.int64))
    out = np.concatenate(parts, axis=0)
    if out.shape != (n_rows, model.num_codebooks):
        raise ValueError(f"Encoded SID shape mismatch: got {out.shape}, expected {(n_rows, model.num_codebooks)}")
    return out


def mode_defaults(mode: str) -> dict[str, str]:
    if mode == "semantic":
        return {
            "checkpoint": DEFAULTS["semantic_checkpoint"],
            "rqvae_input": DEFAULTS["semantic_input"],
            "out_mapping": DEFAULTS["semantic_mapping"],
            "out_indices": DEFAULTS["semantic_indices"],
            "report": DEFAULTS["semantic_report"],
        }
    if mode == "geo_fused":
        return {
            "checkpoint": DEFAULTS["geo_fused_checkpoint"],
            "rqvae_input": DEFAULTS["geo_fused_input"],
            "out_mapping": DEFAULTS["geo_fused_mapping"],
            "out_indices": DEFAULTS["geo_fused_indices"],
            "report": DEFAULTS["geo_fused_report"],
        }
    raise ValueError(f"Unsupported mode: {mode}")


def resolve_mode_paths(args: argparse.Namespace, mode: str) -> dict[str, str]:
    defaults = mode_defaults(mode)
    if args.mode == "both":
        if mode == "semantic":
            defaults["checkpoint"] = args.semantic_checkpoint
            defaults["rqvae_input"] = args.semantic_input
        else:
            defaults["checkpoint"] = args.geo_fused_checkpoint
            defaults["rqvae_input"] = args.geo_fused_input
        return defaults

    return {
        "checkpoint": args.checkpoint or defaults["checkpoint"],
        "rqvae_input": args.rqvae_input or defaults["rqvae_input"],
        "out_mapping": args.out_mapping or defaults["out_mapping"],
        "out_indices": args.out_indices or defaults["out_indices"],
        "report": args.report or defaults["report"],
    }


def load_aligned_inputs(
    embedding_meta_path: str | Path,
    geo_meta_path: str | Path,
    rqvae_input_path: str | Path,
) -> tuple[np.ndarray, Any, Any]:
    embedding_meta = read_parquet(embedding_meta_path)
    geo_meta = read_parquet(geo_meta_path)
    rqvae_input = np.load(rqvae_input_path, mmap_mode="r")
    validate_meta_alignment(embedding_meta, geo_meta, rqvae_input)
    return rqvae_input, embedding_meta, geo_meta


def write_text(path: str | Path, text: str) -> None:
    ensure_parent_dir(path)
    Path(path).write_text(text, encoding="utf-8")


def run_one_mode(args: argparse.Namespace, mode: str, device: torch.device) -> dict[str, Any]:
    paths = resolve_mode_paths(args, mode)
    print(f"[{mode}] loading inputs")
    rqvae_input, embedding_meta, geo_meta = load_aligned_inputs(
        args.embedding_meta,
        args.geo_meta,
        paths["rqvae_input"],
    )

    print(f"[{mode}] loading checkpoint: {paths['checkpoint']}")
    model, checkpoint = load_checkpoint_model(paths["checkpoint"], device)
    if rqvae_input.shape[1] != int(checkpoint.get("input_dim", model.input_dim)):
        raise ValueError(
            f"[{mode}] input dim {rqvae_input.shape[1]} != checkpoint input_dim {checkpoint.get('input_dim')}"
        )

    print(f"[{mode}] encoding {rqvae_input.shape[0]} POIs on {device}")
    indices = encode_sid_indices(model, rqvae_input, args.batch_size, device)
    if indices.shape[1] != 3:
        raise ValueError(f"[{mode}] expected 3 codebooks for sid0/sid1/sid2, got {indices.shape[1]}")

    print(f"[{mode}] building SID mapping")
    mapping = build_sid_mapping(
        embedding_meta=embedding_meta,
        geo_meta=geo_meta,
        indices=indices,
        codebook_size=model.codebook_size,
    )
    validate_mapping(mapping, model.codebook_size)

    ensure_parent_dir(paths["out_indices"])
    np.save(paths["out_indices"], indices.astype(np.int64))
    write_parquet(mapping[SID_MAPPING_COLUMNS], paths["out_mapping"])

    report_dir = Path(paths["report"]).parent
    metrics = build_mode_metrics(
        mode=mode,
        mapping=mapping,
        indices=indices,
        codebook_size=model.codebook_size,
        checkpoint_path=paths["checkpoint"],
        rqvae_input_path=paths["rqvae_input"],
        mapping_path=paths["out_mapping"],
        indices_path=paths["out_indices"],
        report_dir=report_dir,
        qrels_path=args.qrels,
        candidates_path=args.candidates,
        max_report_groups=args.max_report_groups,
    )
    write_text(paths["report"], build_quality_report(metrics))
    metrics["report_path"] = paths["report"]
    metrics["device"] = str(device)

    unique_sid_rate = metrics["sid_stats"]["unique_sid_rate"]
    unique_pid_rate = metrics["pid_stats"]["unique_pid_gid6_sid_rate"]
    p1 = metrics["prefix_stats"]["prefix1"]["weighted_semantic_purity"]
    p2 = metrics["prefix_stats"]["prefix2"]["weighted_semantic_purity"]
    p3 = metrics["prefix_stats"]["prefix3"]["weighted_semantic_purity"]
    print(
        f"[{mode}] done: unique_sid_rate={fmt_float(unique_sid_rate)}, "
        f"unique_pid_rate={fmt_float(unique_pid_rate)}, "
        f"prefix_semantic_purity={fmt_float(p1)}/{fmt_float(p2)}/{fmt_float(p3)}"
    )
    return metrics


def print_single_summary(metrics: dict[str, Any]) -> None:
    prefix = metrics["prefix_stats"]
    print("")
    print("SID export finished")
    print(f"mode: {metrics['mode']}")
    print(f"unique_sid_rate: {fmt_float(metrics['sid_stats']['unique_sid_rate'])}")
    print(f"unique_pid_rate: {fmt_float(metrics['pid_stats']['unique_pid_gid6_sid_rate'])}")
    print(
        "prefix1/2/3 semantic purity: "
        f"{fmt_float(prefix['prefix1']['weighted_semantic_purity'])} / "
        f"{fmt_float(prefix['prefix2']['weighted_semantic_purity'])} / "
        f"{fmt_float(prefix['prefix3']['weighted_semantic_purity'])}"
    )
    print(f"mapping path: {metrics['mapping_path']}")
    print(f"indices path: {metrics['indices_path']}")
    print(f"report path: {metrics['report_path']}")


def print_both_summary(semantic: dict[str, Any], geo_fused: dict[str, Any], recommendation: str, compare_path: str) -> None:
    print("")
    print("SID export finished")
    for metrics in (semantic, geo_fused):
        prefix = metrics["prefix_stats"]
        print(
            f"{metrics['mode']} unique_sid_rate={fmt_float(metrics['sid_stats']['unique_sid_rate'])}, "
            f"unique_pid_rate={fmt_float(metrics['pid_stats']['unique_pid_gid6_sid_rate'])}, "
            "prefix1/2/3 semantic purity="
            f"{fmt_float(prefix['prefix1']['weighted_semantic_purity'])}/"
            f"{fmt_float(prefix['prefix2']['weighted_semantic_purity'])}/"
            f"{fmt_float(prefix['prefix3']['weighted_semantic_purity'])}"
        )
    print(f"推荐使用: {clean_text(recommendation)}")
    print(f"semantic report: {semantic['report_path']}")
    print(f"geo_fused report: {geo_fused['report_path']}")
    print(f"compare report: {compare_path}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    for directory in ("data/sid", "reports", "outputs/sid"):
        ensure_dir(directory)

    device = select_device(args.device)
    print(f"device: {device}")

    if args.mode == "both":
        semantic = run_one_mode(args, "semantic", device)
        geo_fused = run_one_mode(args, "geo_fused", device)
        compare_text, recommendation = build_compare_report(semantic, geo_fused)
        compare_path = DEFAULTS["compare_report"]
        write_text(compare_path, compare_text)
        print_both_summary(semantic, geo_fused, recommendation, compare_path)
    else:
        metrics = run_one_mode(args, args.mode, device)
        print_single_summary(metrics)


if __name__ == "__main__":
    main()

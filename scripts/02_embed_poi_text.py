#!/usr/bin/env python3
"""Embed MobilityBench POI text for later Semantic ID work."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embedding_utils import (  # noqa: E402
    TEXT_FIELDS,
    build_embedding_text,
    embedding_text_has_forbidden_location_tokens,
    format_rate,
    l2_norms,
    load_sentence_transformer_model,
    resolve_device,
    set_model_max_length,
    validate_embeddings,
)
from text_normalize import clean_text, is_empty_text  # noqa: E402


INPUT_COLUMNS = [
    "row_id",
    "poi_id",
    "name",
    "address",
    "city",
    "lat",
    "lon",
    "category_l1",
    "category_l2",
    "brand",
    "tags",
    "parse_warning",
    "poi_json",
    "poi_text",
]

META_COLUMNS = [
    "row_id",
    "poi_id",
    "name",
    "address",
    "city",
    "lat",
    "lon",
    "category_l1",
    "category_l2",
    "brand",
    "tags",
    "parse_warning",
    "embedding_text",
    "embedding_text_len",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sid/poi_sid_train.parquet", help="Input POI SID parquet.")
    parser.add_argument("--out-emb", default="data/embeddings/poi_text_embeddings.npy", help="Output embedding npy.")
    parser.add_argument(
        "--out-meta",
        default="data/embeddings/poi_embedding_meta.parquet",
        help="Output embedding metadata parquet.",
    )
    parser.add_argument(
        "--report",
        default="reports/poi_embedding_quality_report.md",
        help="Output embedding quality report.",
    )
    parser.add_argument("--model", default="/model/Qwen3-Embedding-0.6B", help="Local sentence-transformers model path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--max-length", type=int, default=256, help="SentenceTransformer max sequence length.")
    normalize = parser.add_mutually_exclusive_group()
    normalize.add_argument("--normalize", dest="normalize", action="store_true", help="L2-normalize embeddings.")
    normalize.add_argument("--no-normalize", dest="normalize", action="store_false", help="Do not normalize embeddings.")
    parser.set_defaults(normalize=True)
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = ensure_columns(df, INPUT_COLUMNS)
    for column in ["poi_id", "name", "address", "city", "category_l1", "category_l2", "brand", "tags", "parse_warning", "poi_json", "poi_text"]:
        df[column] = df[column].map(clean_text)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df


def build_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta = df.copy()
    meta["embedding_text"] = meta.apply(build_embedding_text, axis=1)
    meta["embedding_text_len"] = meta["embedding_text"].astype(str).str.len().astype(int)
    return meta.reindex(columns=META_COLUMNS)


def encode_texts(
    texts: list[str],
    model_path: Path,
    device: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
) -> tuple[np.ndarray, str]:
    resolved_device = resolve_device(device)
    model = load_sentence_transformer_model(model_path, resolved_device)
    set_model_max_length(model, max_length)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings, resolved_device


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise SystemExit("Failed to write parquet. Please install pyarrow: pip install pyarrow") from exc
    except ValueError as exc:
        if "pyarrow" in str(exc).lower() or "fastparquet" in str(exc).lower():
            raise SystemExit("Failed to write parquet. Please install pyarrow: pip install pyarrow") from exc
        raise


def field_included_count(meta: pd.DataFrame, label: str) -> int:
    return int(meta["embedding_text"].astype(str).str.contains(f"{label}：", regex=False).sum())


def report_manual_samples(meta: pd.DataFrame, n: int = 20) -> str:
    cols = ["poi_id", "name", "category_l1", "address", "embedding_text"]
    samples = meta.reindex(columns=cols).head(n)
    lines: list[str] = []
    for idx, row in samples.iterrows():
        text = str(row["embedding_text"]).replace("\n", "<br>")
        lines.append(
            f"| {idx} | {row['poi_id']} | {row['name']} | {row['category_l1']} | {row['address']} | {text} |"
        )
    return "\n".join(lines)


def build_report(
    input_rows: int,
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    model_path: Path,
    device: str,
    normalize: bool,
    batch_size: int,
) -> str:
    lengths = meta["embedding_text_len"].astype(int)
    norms = l2_norms(embeddings)
    total = len(meta)
    forbidden = meta["embedding_text"].map(embedding_text_has_forbidden_location_tokens)

    field_labels = {
        "name": "名称",
        "category_l1": "类别",
        "category_l2": "二级类别",
        "brand": "品牌",
        "tags": "标签",
        "city": "城市",
        "address": "地址",
    }
    coverage_lines = []
    for field in TEXT_FIELDS:
        count = field_included_count(meta, field_labels[field])
        coverage_lines.append(f"- {field}_included_count / rate: {count} / {format_rate(count, total)}")
    coverage_lines.append(f"- location_excluded_check: {not bool(forbidden.any())}")

    return "\n".join(
        [
            "# POI Embedding Quality Report",
            "",
            "## Basic Statistics",
            f"- input_rows: {input_rows}",
            f"- output_embeddings_shape: {list(embeddings.shape)}",
            f"- embedding_dim: {embeddings.shape[1] if embeddings.ndim == 2 else ''}",
            f"- meta_rows: {len(meta)}",
            f"- unique_poi_id: {int(meta['poi_id'].nunique())}",
            f"- empty_embedding_text_count: {int(meta['embedding_text'].map(is_empty_text).sum())}",
            f"- min_embedding_text_len: {int(lengths.min()) if len(lengths) else 0}",
            f"- max_embedding_text_len: {int(lengths.max()) if len(lengths) else 0}",
            f"- mean_embedding_text_len: {float(lengths.mean()) if len(lengths) else 0:.4f}",
            f"- p50_embedding_text_len: {float(lengths.quantile(0.50)) if len(lengths) else 0:.4f}",
            f"- p95_embedding_text_len: {float(lengths.quantile(0.95)) if len(lengths) else 0:.4f}",
            "",
            "## Model",
            f"- model_path: {model_path}",
            f"- device: {device}",
            f"- normalize_embeddings: {normalize}",
            f"- batch_size: {batch_size}",
            "",
            "## Field Coverage in embedding_text",
            *coverage_lines,
            "",
            "## Embedding Numeric Checks",
            f"- has_nan: {bool(np.isnan(embeddings).any())}",
            f"- has_inf: {bool(np.isinf(embeddings).any())}",
            f"- mean_l2_norm: {float(norms.mean()):.6f}",
            f"- min_l2_norm: {float(norms.min()):.6f}",
            f"- max_l2_norm: {float(norms.max()):.6f}",
            "",
            "## Manual Samples",
            "| idx | poi_id | name | category_l1 | address | embedding_text |",
            "| --- | --- | --- | --- | --- | --- |",
            report_manual_samples(meta, 20),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_emb_path = Path(args.out_emb)
    out_meta_path = Path(args.out_meta)
    report_path = Path(args.report)
    model_path = Path(args.model)

    if not model_path.exists():
        raise SystemExit(
            f"Embedding model path does not exist: {model_path}. "
            "Use --model to specify a local model directory. No model will be downloaded."
        )

    df = load_input(input_path)
    input_rows = len(df)
    meta = build_meta(df)
    texts = meta["embedding_text"].tolist()

    embeddings, device = encode_texts(
        texts=texts,
        model_path=model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize=args.normalize,
    )
    validate_embeddings(embeddings, meta, input_rows, args.normalize)

    ensure_parent_dir(out_emb_path)
    np.save(out_emb_path, embeddings.astype(np.float32, copy=False))
    write_parquet(meta, out_meta_path)

    ensure_parent_dir(report_path)
    report = build_report(input_rows, embeddings, meta, model_path, device, args.normalize, args.batch_size)
    report_path.write_text(report, encoding="utf-8")

    print(f"input_rows: {input_rows}")
    print(f"embedding_shape: {list(embeddings.shape)}")
    print(f"meta_output: {out_meta_path}")
    print(f"embedding_output: {out_emb_path}")
    print(f"report_output: {report_path}")
    print("next_step: continue building geohash and continuous spatial features")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build high-confidence coarse category labels for CAU-RQ-VAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cau_labels import (  # noqa: E402
    HELDOUT_SPLIT,
    SUPERVISED_SPLIT,
    UNSUPERVISED_SPLIT,
    build_labels,
    label_distribution,
    write_vocab_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sid/poi_sid_train.parquet", help="Input POI SID train parquet.")
    parser.add_argument(
        "--out-labels",
        default="outputs/experiments/cau_rqvae/labels/cau_coarse_category_labels.parquet",
        help="Output label parquet.",
    )
    parser.add_argument(
        "--out-vocab",
        default="outputs/experiments/cau_rqvae/labels/cau_category_vocab.json",
        help="Output label vocabulary JSON.",
    )
    parser.add_argument(
        "--out-distribution",
        default="outputs/experiments/cau_rqvae/labels/cau_label_distribution.csv",
        help="Output label distribution CSV.",
    )
    parser.add_argument(
        "--report",
        default="outputs/reports/cau_rqvae_label_gate1.md",
        help="Output Gate 1 report.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-ratio", type=float, default=0.2)
    parser.add_argument("--min-label-count", type=int, default=20)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_report(
    args: argparse.Namespace,
    summary: dict,
    distribution: pd.DataFrame,
    gate_pass: bool,
    gate_reasons: list[str],
) -> str:
    lines = [
        "# CAU-RQ-VAE Gate 1 Coarse Category Labels",
        "",
        "This gate builds high-confidence coarse category labels for CAU-RQ-VAE. It does not train a model and does not drop any POI.",
        "",
        "## Policy",
        "",
        "- `UNK` / unmatched POIs are retained for reconstruction and final SID evaluation.",
        "- `UNK` / unmatched POIs do not participate in category supervision (`L_tag`).",
        "- Training labels use high-confidence rules with field priority `category_l1 -> tags -> name`.",
        "- Full `sid_eval.py` inferred semantic labels remain the baseline-compatible evaluation metric and are not used directly as training labels.",
        "- High-confidence supervised labels are split by label into 80% tag-train and 20% tag-heldout validation using a fixed seed.",
        "",
        "## Artifacts",
        "",
        f"- Input: `{args.input}`",
        f"- Labels: `{args.out_labels}`",
        f"- Vocabulary: `{args.out_vocab}`",
        f"- Distribution: `{args.out_distribution}`",
        f"- Report: `{args.report}`",
        "",
        "## Summary",
        "",
        f"- rows: {summary['rows']}",
        f"- supervised rows: {summary['supervised_rows']} / {summary['supervised_rate']:.4f}",
        f"- tag train rows: {summary['tag_train_rows']}",
        f"- tag held-out rows: {summary['tag_heldout_rows']}",
        f"- unsupervised rows: {summary['unsupervised_rows']}",
        f"- number of supervised labels: {summary['num_labels']}",
        f"- seed: {summary['seed']}",
        f"- heldout_ratio: {summary['heldout_ratio']}",
        f"- min_label_count: {summary['min_label_count']}",
        f"- low-count labels removed from supervision: {summary['low_count_labels_removed']}",
        "",
        "## Split Names",
        "",
        f"- category loss train split: `{SUPERVISED_SPLIT}`",
        f"- held-out category validation split: `{HELDOUT_SPLIT}`",
        f"- no category supervision split: `{UNSUPERVISED_SPLIT}`",
        "",
        "## Label Distribution",
        "",
        "| label | total | train | heldout | heldout_rate | category_l1_source | tags_source | name_source | example_keywords |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in distribution.iterrows():
        lines.append(
            "| {coarse_label} | {total} | {train} | {heldout} | {heldout_rate:.4f} | {category_l1_source} | {tags_source} | {name_source} | {example_keywords} |".format(
                **row.to_dict()
            )
        )
    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
            f"- Gate 1 status: `{'PASS' if gate_pass else 'FAIL'}`",
        ]
    )
    for reason in gate_reasons:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    poi_df = pd.read_parquet(args.input)
    labels, summary = build_labels(
        poi_df=poi_df,
        seed=args.seed,
        heldout_ratio=args.heldout_ratio,
        min_label_count=args.min_label_count,
    )
    distribution = label_distribution(labels)

    split_counts = labels["split"].value_counts().to_dict()
    gate_reasons: list[str] = []
    gate_pass = True

    if len(labels) != len(poi_df):
        gate_pass = False
        gate_reasons.append(f"row count mismatch: labels={len(labels)} input={len(poi_df)}")
    if not labels["poi_id"].is_unique:
        gate_pass = False
        gate_reasons.append("poi_id is not unique")
    if not labels["row_id"].is_unique:
        gate_pass = False
        gate_reasons.append("row_id is not unique")
    if summary["num_labels"] < 2:
        gate_pass = False
        gate_reasons.append("fewer than two supervised labels")
    if summary["tag_train_rows"] <= 0 or summary["tag_heldout_rows"] <= 0:
        gate_pass = False
        gate_reasons.append("empty tag train or held-out split")
    if not distribution.empty and (distribution["train"].le(0).any() or distribution["heldout"].le(0).any()):
        gate_pass = False
        gate_reasons.append("at least one label lacks train or held-out examples")
    if labels.loc[labels["split"].eq(UNSUPERVISED_SPLIT), "coarse_label_id"].ne(-1).any():
        gate_pass = False
        gate_reasons.append("unsupervised rows should have coarse_label_id=-1")

    if gate_pass:
        gate_reasons.append("all POIs retained; supervised labels have disjoint train/held-out splits; UNK is excluded from L_tag.")
    gate_reasons.append(f"split counts: {json.dumps(split_counts, ensure_ascii=False, sort_keys=True)}")

    write_parquet(labels, Path(args.out_labels))
    write_vocab_json(summary, Path(args.out_vocab))
    write_csv(distribution, Path(args.out_distribution))
    report = build_report(args, summary, distribution, gate_pass, gate_reasons)
    report_path = Path(args.report)
    ensure_parent(report_path)
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps({"gate_pass": gate_pass, **summary, "report": args.report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

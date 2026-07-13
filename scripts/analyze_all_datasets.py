#!/usr/bin/env python3
"""Analyze independently processed LLM4POI, Yelp, and MobilityBench datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poi_genret.process_utils import log  # noqa: E402
from poi_genret.schema import (  # noqa: E402
    INTERACTION_COLUMNS,
    MOBILITY_CANDIDATE_COLUMNS,
    POI_COLUMNS,
    QREL_COLUMNS,
    QUERY_COLUMNS,
    read_csv_if_exists,
    write_csv,
)

warnings.filterwarnings("ignore", message="Glyph .* missing from font")


DATASETS = [
    ("LLM4POI", "Foursquare-NYC", "llm4poi/Foursquare-NYC", "next_poi", "English/Check-in", "NYC"),
    ("LLM4POI", "Foursquare-TKY", "llm4poi/Foursquare-TKY", "next_poi", "English/Check-in", "Tokyo"),
    ("LLM4POI", "Gowalla-CA", "llm4poi/Gowalla-CA", "next_poi", "English/Check-in", "California"),
    ("Yelp", "Yelp", "yelp", "template_poi_search", "English", "North America"),
    ("MobilityBench", "MobilityBench", "mobilitybench", "real_poi_search/real_nearby_search", "Chinese", "China"),
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_dir(output_dir: Path, rel: str) -> Path:
    return output_dir / rel


def missing_summary(dataset_label: str, table_name: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if df.empty:
        return rows
    for col in df.columns:
        missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        rows.append(
            {
                "dataset": dataset_label,
                "table": table_name,
                "field": col,
                "missing_count": int(missing.sum()),
                "missing_rate": float(missing.mean()),
            }
        )
    return rows


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, rotate: int = 45) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    if df.empty:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
    else:
        labels = df[x].astype(str).tolist()
        values = pd.to_numeric(df[y], errors="coerce").fillna(0).tolist()
        plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=rotate, ha="right")
        plt.ylabel(y)
        plt.title(title)
        plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_hist(series: pd.Series, title: str, path: Path, bins: int = 60, log_y: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
    else:
        plt.hist(values, bins=bins)
        if log_y:
            plt.yscale("log")
        plt.ylabel("count")
        plt.title(title)
        plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def count_table(series: pd.Series, value_col: str, count_col: str = "count", top_n: int | None = None) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=[value_col, count_col])
    counts = series.replace("", "missing").fillna("missing").value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    out = counts.reset_index()
    out.columns = [value_col, count_col]
    return out


def make_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No data."
    view = df.head(max_rows).copy()
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        values = [str(row[col]).replace("|", "\\|").replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def collect_dataset(output_dir: Path, source: str, sub_dataset: str, rel: str, task_type: str, language: str, region: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    d = dataset_dir(output_dir, rel)
    pois = read_csv_if_exists(d / "pois.csv", POI_COLUMNS)
    interactions = read_csv_if_exists(d / "interactions.csv", INTERACTION_COLUMNS)
    queries = read_csv_if_exists(d / "queries.csv", QUERY_COLUMNS)
    qrels = read_csv_if_exists(d / "qrels.csv", QREL_COLUMNS)
    candidates = read_csv_if_exists(d / "candidates.csv", MOBILITY_CANDIDATE_COLUMNS)
    train_queries = read_csv_if_exists(d / "train_queries.csv", QUERY_COLUMNS)
    dev_queries = read_csv_if_exists(d / "dev_queries.csv", QUERY_COLUMNS)
    test_queries = read_csv_if_exists(d / "test_queries.csv", QUERY_COLUMNS)
    train_qrels = read_csv_if_exists(d / "train_qrels.csv", QREL_COLUMNS)
    dev_qrels = read_csv_if_exists(d / "dev_qrels.csv", QREL_COLUMNS)
    test_qrels = read_csv_if_exists(d / "test_qrels.csv", QREL_COLUMNS)
    summary_json = load_json(d / "dataset_summary.json")

    notes = {
        "LLM4POI": "Trajectory next-POI data; split by time order inside each trajectory.",
        "Yelp": "Template queries from business attributes; split by POI to avoid label leakage.",
        "MobilityBench": "Real Chinese map queries; no-target samples kept as tool-intent data.",
    }.get(source, "")
    row = {
        "dataset": source,
        "sub_dataset": sub_dataset,
        "scenario": summary_json.get("scenario", ""),
        "language": language,
        "region": region,
        "task_type": task_type,
        "pois": len(pois),
        "queries": len(queries),
        "qrels": len(qrels),
        "interactions": len(interactions),
        "candidates": len(candidates),
        "train_queries": len(train_queries),
        "dev_queries": len(dev_queries),
        "test_queries": len(test_queries),
        "train_qrels": len(train_qrels),
        "dev_qrels": len(dev_qrels),
        "test_qrels": len(test_qrels),
        "notes": notes,
        "path": str(d),
    }
    missing = []
    for table_name, df in [("pois", pois), ("interactions", interactions), ("queries", queries), ("qrels", qrels), ("candidates", candidates)]:
        missing.extend(missing_summary(f"{source}/{sub_dataset}", table_name, df))
    return row, missing


def analyze(output_dir: Path, report_dir: Path) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_rows = []
    for source, sub_dataset, rel, task_type, language, region in DATASETS:
        row, missing = collect_dataset(output_dir, source, sub_dataset, rel, task_type, language, region)
        rows.append(row)
        missing_rows.extend(missing)
    summary_df = pd.DataFrame(rows)
    missing_df = pd.DataFrame(missing_rows)
    write_csv(summary_df, report_dir / "dataset_summary.csv")
    write_csv(missing_df, report_dir / "missing_value_summary.csv")

    plot_bar(summary_df, "sub_dataset", "pois", "POI Count by Dataset", figures / "dataset_poi_count.png")
    plot_bar(summary_df, "sub_dataset", "queries", "Query Count by Dataset", figures / "dataset_query_count.png")
    plot_bar(summary_df, "sub_dataset", "qrels", "Qrels Count by Dataset", figures / "dataset_qrels_count.png")

    split_long = summary_df.melt(id_vars=["sub_dataset"], value_vars=["train_queries", "dev_queries", "test_queries"], var_name="split", value_name="count")
    split_long["label"] = split_long["sub_dataset"] + "/" + split_long["split"].str.replace("_queries", "")
    plot_bar(split_long, "label", "count", "Train Dev Test Query Count", figures / "split_query_count.png")

    llm = summary_df[summary_df["dataset"] == "LLM4POI"]
    plot_bar(llm, "sub_dataset", "interactions", "LLM4POI Check-in Count", figures / "llm4poi_checkin_count.png")

    all_interaction_counts = []
    for _, sub_dataset, rel, *_ in DATASETS[:3]:
        interactions = read_csv_if_exists(output_dir / rel / "interactions.csv", INTERACTION_COLUMNS)
        if not interactions.empty:
            counts = interactions["poi_id"].value_counts()
            all_interaction_counts.append(counts)
    if all_interaction_counts:
        plot_hist(pd.concat(all_interaction_counts), "LLM4POI POI Interaction Long Tail", figures / "llm4poi_poi_interaction_long_tail.png", log_y=True)
    else:
        plot_hist(pd.Series(dtype=float), "LLM4POI POI Interaction Long Tail", figures / "llm4poi_poi_interaction_long_tail.png")

    yelp_pois = read_csv_if_exists(output_dir / "yelp/pois.csv", POI_COLUMNS)
    plot_bar(count_table(yelp_pois.get("category_l1", pd.Series(dtype=str)), "category_l1", top_n=20), "category_l1", "count", "Yelp Top 20 Categories", figures / "yelp_top20_categories.png")
    plot_bar(count_table(yelp_pois.get("city", pd.Series(dtype=str)), "city", top_n=20), "city", "count", "Yelp Top 20 Cities", figures / "yelp_top20_cities.png")
    plot_hist(pd.to_numeric(yelp_pois.get("rating", pd.Series(dtype=str)), errors="coerce"), "Yelp Rating Distribution", figures / "yelp_rating_distribution.png", bins=10)

    mb_dir = output_dir / "mobilitybench"
    mb_tasks = read_csv_if_exists(mb_dir / "task_distribution.csv")
    if not mb_tasks.empty:
        mb_tasks = mb_tasks.sort_values("count", ascending=False).head(20)
    plot_bar(mb_tasks, "task_scenario", "count", "MobilityBench Task Scenario Distribution", figures / "mobilitybench_task_scenario_distribution.png")
    mb_queries = read_csv_if_exists(mb_dir / "queries.csv", QUERY_COLUMNS)
    plot_bar(count_table(mb_queries.get("query_type", pd.Series(dtype=str)), "query_type"), "query_type", "count", "MobilityBench POI Search vs Nearby Search", figures / "mobilitybench_poi_vs_nearby.png")
    plot_hist(mb_queries["query_text"].astype(str).str.len() if not mb_queries.empty else pd.Series(dtype=float), "MobilityBench Query Length Distribution", figures / "mobilitybench_query_length_distribution.png")

    if not missing_df.empty:
        missing_plot = missing_df.groupby(["dataset", "table"], as_index=False)["missing_rate"].mean()
        missing_plot["label"] = missing_plot["dataset"] + "/" + missing_plot["table"]
        plot_bar(missing_plot, "label", "missing_rate", "Average Missing Rate by Dataset Table", figures / "field_missing_rate_comparison.png")
    else:
        plot_bar(pd.DataFrame(columns=["label", "missing_rate"]), "label", "missing_rate", "Average Missing Rate by Dataset Table", figures / "field_missing_rate_comparison.png")

    write_report(report_dir / "dataset_report.md", summary_df, output_dir)
    return {"datasets": rows, "report": str(report_dir / "dataset_report.md")}


def dataset_block(output_dir: Path, rel: str, label: str) -> str:
    d = output_dir / rel
    s = load_json(d / "dataset_summary.json")
    if not s:
        return f"### {label}\n\nNo processed data found.\n"
    split = s.get("split", {})
    lines = [f"### {label}", ""]
    for key in ["pois", "users", "interactions", "queries", "qrels", "categories", "cities", "candidates", "total_episodes", "poi_search", "nearby_search", "tool_intent_queries"]:
        if key in s:
            lines.append(f"- {key}: `{s[key]}`")
    lines.extend(
        [
            f"- train/dev/test queries: `{split.get('train_queries', 0)}` / `{split.get('dev_queries', 0)}` / `{split.get('test_queries', 0)}`",
            f"- train/dev/test qrels: `{split.get('train_qrels', 0)}` / `{split.get('dev_qrels', 0)}` / `{split.get('test_qrels', 0)}`",
        ]
    )
    if "lat_min" in s:
        lines.append(f"- lat range: `{s['lat_min']}` to `{s['lat_max']}`")
        lines.append(f"- lon range: `{s['lon_min']}` to `{s['lon_max']}`")
    if "avg_trajectory_length" in s:
        lines.append(f"- average trajectory length: `{s['avg_trajectory_length']:.4f}`")
    lines.append(f"- original split used: `{s.get('whether_original_split_used', False)}`")
    lines.append("")
    return "\n".join(lines)


def write_report(path: Path, summary_df: pd.DataFrame, output_dir: Path) -> None:
    comparison = summary_df[
        [
            "dataset",
            "sub_dataset",
            "scenario",
            "language",
            "region",
            "task_type",
            "pois",
            "queries",
            "qrels",
            "interactions",
            "candidates",
            "train_queries",
            "dev_queries",
            "test_queries",
            "notes",
        ]
    ]
    mb = load_json(output_dir / "mobilitybench/dataset_summary.json")
    text = f"""# Dataset Report

## 1. Dataset Design

The current data layer uses three data sources as independent datasets, with independent train/dev/test splits and independent evaluation.

Reasons:

- LLM4POI is trajectory and check-in data, closer to next POI recommendation.
- Yelp is English business metadata, and its queries are template-generated.
- MobilityBench provides Chinese real-map queries, closer to POI Search and Nearby Search.
- The three sources differ in language, region, task form, and POI taxonomy, so they are not mixed into one main training set.
- Cross-dataset results should be interpreted as generalization analysis rather than same-distribution evaluation.

## 2. LLM4POI Processing

LLM4POI is used for next POI / trajectory-based retrieval. Each user or trajectory is sorted by time. A query sample is built from the historical POI/category sequence, and the next visited POI is the positive qrel.

{dataset_block(output_dir, "llm4poi/Foursquare-NYC", "Foursquare-NYC")}
{dataset_block(output_dir, "llm4poi/Foursquare-TKY", "Foursquare-TKY")}
{dataset_block(output_dir, "llm4poi/Gowalla-CA", "Gowalla-CA")}

Limitations:

- The preprocessed files do not reliably provide POI names or addresses.
- Query text is constructed from history categories and is not an open-ended search log.

## 3. Yelp Processing

Yelp is used for template-based POI retrieval. Each business becomes a POI, and 3-5 template queries are generated from name, category, city, rating, and open status.

{dataset_block(output_dir, "yelp", "Yelp")}

Limitations:

- Yelp queries are template-generated, not real search logs.
- Yelp language and geographic distribution differ from Chinese map search.
- Split is by POI/business_id, not by query, to avoid leaking the same target POI across train/dev/test.

## 4. MobilityBench Processing

MobilityBench is used for Chinese real-map POI Search / Nearby Search. Route planning, weather, traffic, and other non-POI retrieval tasks are excluded.

{dataset_block(output_dir, "mobilitybench", "MobilityBench")}

Additional MobilityBench notes:

- Total episodes scanned: `{mb.get('total_episodes', 0)}`
- POI Search queries: `{mb.get('poi_search', 0)}`
- Nearby Search queries: `{mb.get('nearby_search', 0)}`
- Query samples with unique qrels: `{mb.get('qrels', 0)}`
- Tool-intent samples without unique qrels: `{mb.get('tool_intent_queries', 0)}`
- Candidate rows: `{mb.get('candidates', 0)}`

Limitations:

- Some samples only expose candidates or tool-call information, not a unique target POI.
- Those samples are kept for tool selection, intent detection, parameter extraction, and candidate parsing evaluation, but not direct Recall/MRR/NDCG evaluation.

## 5. Dataset Comparison

{make_markdown_table(comparison, max_rows=20)}

## 6. Figures

- `figures/dataset_poi_count.png`
- `figures/dataset_query_count.png`
- `figures/dataset_qrels_count.png`
- `figures/split_query_count.png`
- `figures/llm4poi_checkin_count.png`
- `figures/llm4poi_poi_interaction_long_tail.png`
- `figures/yelp_top20_categories.png`
- `figures/yelp_top20_cities.png`
- `figures/yelp_rating_distribution.png`
- `figures/mobilitybench_task_scenario_distribution.png`
- `figures/mobilitybench_poi_vs_nearby.png`
- `figures/mobilitybench_query_length_distribution.png`
- `figures/field_missing_rate_comparison.png`

## 7. Next Steps

- Use LLM4POI for next-POI retrieval baselines.
- Use Yelp for controlled template-based POI retrieval experiments.
- Use MobilityBench as real Chinese map-query evaluation data.
- Keep tool-intent MobilityBench samples for tool routing and parameter extraction evaluation.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/data_report")
    parser.add_argument("--split-ratio", default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(args.output_dir, args.report_dir)
    log("analyze_all_datasets", f"Report: {summary['report']}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

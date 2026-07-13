"""SID export and quality evaluation helpers for MobilityBench POI RQ-VAE."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd


SID_MAPPING_COLUMNS = [
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
    "geohash5",
    "geohash6",
    "geohash7",
    "sid0",
    "sid1",
    "sid2",
    "sid_str",
    "sid_prefix1",
    "sid_prefix2",
    "gid6_str",
    "pid_gid6_sid",
    "pid_gid6_sid_dedup",
    "dedup_id",
    "embedding_text",
]

PREFIX_SUMMARY_COLUMNS = [
    "group_size",
    "top_semantic_label",
    "semantic_purity",
    "top_city",
    "city_purity",
    "top_geohash5",
    "geohash5_purity",
    "example_poi_ids",
    "example_names",
    "example_addresses",
]


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def rate(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_count_rate(count: int | float, total: int | float) -> str:
    return f"{int(count)} / {fmt_float(rate(count, total))}"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return " ".join(text.split())


def is_valid_label(value: Any) -> bool:
    text = clean_text(value)
    return bool(text) and text.upper() != "UNK"


SEMANTIC_INFER_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("酒店住宿", ("酒店", "宾馆", "民宿", "客栈")),
    ("汽车充电站", ("充电站", "充电桩", "超级充电")),
    ("加油站", ("加油站", "中石化", "中石油")),
    ("学校", ("大学", "学院", "中学", "小学", "幼儿园", "学校")),
    ("医疗机构", ("医院", "诊所", "卫生院", "门诊")),
    ("药店", ("药店", "药房")),
    ("银行金融", ("银行", "支行", "ATM")),
    ("景点休闲", ("公园", "景区", "广场")),
    ("停车场", ("停车场",)),
    ("餐饮服务", ("餐厅", "饭店", "小吃", "火锅", "烧烤", "面馆", "咖啡", "奶茶")),
    ("政府机构", ("派出所", "公安局", "政府", "政务", "法院")),
    ("交通设施", ("公交站", "地铁站", "火车站", "机场", "汽车站", "高铁站")),
    ("住宅小区", ("小区", "公寓", "住宅", "花园", "家园")),
    ("公司企业", ("公司", "产业园", "写字楼", "办公楼")),
    ("公共厕所", ("厕所", "卫生间", "公厕")),
]


def infer_semantic_label(row: pd.Series) -> str:
    category_l1 = clean_text(row.get("category_l1", ""))
    if is_valid_label(category_l1):
        return category_l1
    tags = clean_text(row.get("tags", ""))
    if is_valid_label(tags):
        return tags
    name = clean_text(row.get("name", ""))
    name_upper = name.upper()
    for label, keywords in SEMANTIC_INFER_RULES:
        for keyword in keywords:
            if keyword.upper() in name_upper:
                return label
    return "UNK"


def add_analysis_labels(mapping: pd.DataFrame) -> pd.DataFrame:
    df = mapping.copy()
    df["semantic_label"] = df.apply(infer_semantic_label, axis=1)
    df["city_label"] = df["city"].map(clean_text).replace("", "UNK")
    df["geo_label"] = df["geohash5"].map(clean_text).replace("", "UNK")
    return df


def read_parquet(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError("读取 parquet 失败，请安装 pyarrow：pip install pyarrow") from exc


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent_dir(path)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError("保存 parquet 失败，请安装 pyarrow：pip install pyarrow") from exc


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def validate_meta_alignment(embedding_meta: pd.DataFrame, geo_meta: pd.DataFrame, rqvae_input: np.ndarray) -> None:
    if len(embedding_meta) != len(geo_meta):
        raise ValueError(f"embedding meta rows ({len(embedding_meta)}) != geo meta rows ({len(geo_meta)})")
    if rqvae_input.shape[0] != len(embedding_meta):
        raise ValueError(f"RQ-VAE input rows ({rqvae_input.shape[0]}) != meta rows ({len(embedding_meta)})")
    for column in ("row_id", "poi_id"):
        if column not in embedding_meta.columns or column not in geo_meta.columns:
            raise ValueError(f"Missing required alignment column: {column}")
        if not embedding_meta[column].astype(str).equals(geo_meta[column].astype(str)):
            raise ValueError(f"{column} order is not aligned between embedding meta and geo meta")


def _require_or_fill_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = ""
    return out


def format_sid(indices: np.ndarray) -> tuple[list[str], list[str], list[str]]:
    sid_prefix1 = [f"S0_{int(x)}" for x in indices[:, 0]]
    sid_prefix2 = [f"S0_{int(x0)} S1_{int(x1)}" for x0, x1 in indices[:, :2]]
    sid_str = [f"S0_{int(x0)} S1_{int(x1)} S2_{int(x2)}" for x0, x1, x2 in indices[:, :3]]
    return sid_prefix1, sid_prefix2, sid_str


def format_gid6(geohash6: Any) -> str:
    value = clean_text(geohash6)
    if len(value) != 6:
        raise ValueError(f"Invalid geohash6 value for gid6_str: {value!r}")
    return " ".join(f"G{i}_{char}" for i, char in enumerate(value))


def build_sid_mapping(
    embedding_meta: pd.DataFrame,
    geo_meta: pd.DataFrame,
    indices: np.ndarray,
    codebook_size: int,
) -> pd.DataFrame:
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"SID indices must have shape [N, 3], got {indices.shape}")
    if len(embedding_meta) != indices.shape[0]:
        raise ValueError(f"meta rows ({len(embedding_meta)}) != SID index rows ({indices.shape[0]})")
    if indices.min() < 0 or indices.max() >= codebook_size:
        raise ValueError(f"SID index out of range [0, {codebook_size - 1}]")

    meta_columns = [
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
        "embedding_text",
    ]
    geo_columns = ["geohash5", "geohash6", "geohash7"]
    meta = _require_or_fill_columns(embedding_meta, meta_columns)
    geo = _require_or_fill_columns(geo_meta, geo_columns)

    mapping = meta[meta_columns].copy()
    for column in geo_columns:
        mapping[column] = geo[column].map(clean_text)

    for level in range(3):
        mapping[f"sid{level}"] = indices[:, level].astype(np.int64)

    sid_prefix1, sid_prefix2, sid_str = format_sid(indices)
    mapping["sid_str"] = sid_str
    mapping["sid_prefix1"] = sid_prefix1
    mapping["sid_prefix2"] = sid_prefix2
    mapping["gid6_str"] = mapping["geohash6"].map(format_gid6)
    mapping["pid_gid6_sid"] = mapping["gid6_str"] + " " + mapping["sid_str"]

    pid_group_size = mapping.groupby("pid_gid6_sid", sort=False)["pid_gid6_sid"].transform("size")
    mapping["dedup_id"] = mapping.groupby("pid_gid6_sid", sort=False).cumcount().astype(np.int64)
    mapping["pid_gid6_sid_dedup"] = np.where(
        pid_group_size.gt(1),
        mapping["pid_gid6_sid"] + " D_" + mapping["dedup_id"].astype(str),
        mapping["pid_gid6_sid"],
    )

    mapping = mapping[SID_MAPPING_COLUMNS]
    validate_mapping(mapping, codebook_size)
    return mapping


def validate_mapping(mapping: pd.DataFrame, codebook_size: int) -> None:
    if not mapping["poi_id"].is_unique:
        raise ValueError("poi_id is not unique in SID mapping")
    if mapping["sid_str"].map(clean_text).eq("").any():
        raise ValueError("sid_str contains empty values")
    if mapping["pid_gid6_sid"].map(clean_text).eq("").any():
        raise ValueError("pid_gid6_sid contains empty values")
    if not mapping["pid_gid6_sid_dedup"].is_unique:
        raise ValueError("pid_gid6_sid_dedup is not unique")
    for level in range(3):
        column = f"sid{level}"
        if mapping[column].min() < 0 or mapping[column].max() >= codebook_size:
            raise ValueError(f"{column} out of range [0, {codebook_size - 1}]")


def codebook_usage(indices: np.ndarray, codebook_size: int) -> list[dict[str, Any]]:
    usages: list[dict[str, Any]] = []
    total = indices.shape[0]
    for level in range(indices.shape[1]):
        counts = np.bincount(indices[:, level].astype(np.int64), minlength=codebook_size)
        probs = counts.astype(np.float64) / max(total, 1)
        nonzero = probs > 0
        entropy = float(-(probs[nonzero] * np.log(probs[nonzero])).sum())
        used = int(nonzero.sum())
        top_order = np.argsort(-counts)[:10]
        usages.append(
            {
                "level": int(level),
                "used_code_count": used,
                "used_code_rate": rate(used, codebook_size),
                "dead_code_count": int(codebook_size - used),
                "entropy": entropy,
                "perplexity": float(np.exp(entropy)),
                "top_code_distribution": [
                    {"code": int(code), "count": int(counts[code]), "rate": rate(int(counts[code]), total)}
                    for code in top_order
                    if counts[code] > 0
                ],
            }
        )
    return usages


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def collision_stats(df: pd.DataFrame, key_col: str, prefix: str) -> dict[str, Any]:
    sizes = df.groupby(key_col, sort=False).size().astype(np.int64)
    values = sizes.to_numpy(dtype=np.float64)
    collisions = sizes[sizes > 1]
    pct = _percentiles(values)
    return {
        f"unique_{prefix}_count": int(len(sizes)),
        f"unique_{prefix}_rate": rate(len(sizes), len(df)),
        f"{prefix}_collision_group_count": int(len(collisions)),
        f"{prefix}_collision_poi_count": int(collisions.sum()) if len(collisions) else 0,
        f"max_pois_per_{prefix}": int(sizes.max()) if len(sizes) else 0,
        f"mean_pois_per_{prefix}": float(values.mean()) if len(values) else 0.0,
        f"p50_pois_per_{prefix}": pct["p50"],
        f"p95_pois_per_{prefix}": pct["p95"],
        f"p99_pois_per_{prefix}": pct["p99"],
    }


def pid_collision_stats(df: pd.DataFrame) -> dict[str, Any]:
    base = collision_stats(df, "pid_gid6_sid", "pid")
    base["unique_pid_gid6_sid_count"] = base.pop("unique_pid_count")
    base["unique_pid_gid6_sid_rate"] = base.pop("unique_pid_rate")
    base["max_pois_per_pid"] = base.pop("max_pois_per_pid")
    base.pop("mean_pois_per_pid", None)
    base.pop("p50_pois_per_pid", None)
    base.pop("p95_pois_per_pid", None)
    base.pop("p99_pois_per_pid", None)
    dedup_unique = int(df["pid_gid6_sid_dedup"].nunique())
    base["pid_gid6_sid_dedup_unique_count"] = dedup_unique
    base["pid_gid6_sid_dedup_unique_rate"] = rate(dedup_unique, len(df))
    return base


def prefix_group_size_stats(summary: pd.DataFrame) -> dict[str, Any]:
    sizes = summary["group_size"].to_numpy(dtype=np.float64)
    pct = _percentiles(sizes)
    singleton_count = int((summary["group_size"] == 1).sum())
    return {
        "num_groups": int(len(summary)),
        "mean_group_size": float(sizes.mean()) if len(sizes) else 0.0,
        "p50_group_size": pct["p50"],
        "p95_group_size": pct["p95"],
        "p99_group_size": pct["p99"],
        "max_group_size": int(sizes.max()) if len(sizes) else 0,
        "singleton_group_count": singleton_count,
        "singleton_group_rate": rate(singleton_count, len(summary)),
    }


def weighted_purity(summary: pd.DataFrame) -> dict[str, float]:
    total = float(summary["group_size"].sum())
    if total == 0:
        return {
            "weighted_semantic_purity": 0.0,
            "weighted_city_purity": 0.0,
            "weighted_geohash5_purity": 0.0,
        }
    return {
        "weighted_semantic_purity": float((summary["semantic_purity"] * summary["group_size"]).sum() / total),
        "weighted_city_purity": float((summary["city_purity"] * summary["group_size"]).sum() / total),
        "weighted_geohash5_purity": float((summary["geohash5_purity"] * summary["group_size"]).sum() / total),
    }


def _join_examples(values: pd.Series, max_items: int = 5) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_items:
            break
    return " | ".join(out)


def _top_label(df: pd.DataFrame, key_col: str, label_col: str, top_col: str, count_col: str) -> pd.DataFrame:
    counts = df.groupby([key_col, label_col], dropna=False, sort=False).size().reset_index(name=count_col)
    counts = counts.sort_values([key_col, count_col, label_col], ascending=[True, False, True], kind="mergesort")
    top = counts.drop_duplicates(key_col, keep="first")[[key_col, label_col, count_col]]
    top = top.rename(columns={label_col: top_col})
    return top


def build_group_summary(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    grouped = df.groupby(key_col, sort=False)
    summary = grouped.size().rename("group_size").reset_index()
    summary = summary.merge(
        _top_label(df, key_col, "semantic_label", "top_semantic_label", "top_semantic_label_count"),
        on=key_col,
        how="left",
    )
    summary = summary.merge(
        _top_label(df, key_col, "city_label", "top_city", "top_city_count"),
        on=key_col,
        how="left",
    )
    summary = summary.merge(
        _top_label(df, key_col, "geo_label", "top_geohash5", "top_geohash5_count"),
        on=key_col,
        how="left",
    )
    summary["semantic_purity"] = summary["top_semantic_label_count"] / summary["group_size"]
    summary["city_purity"] = summary["top_city_count"] / summary["group_size"]
    summary["geohash5_purity"] = summary["top_geohash5_count"] / summary["group_size"]

    examples = grouped.agg(
        example_poi_ids=("poi_id", _join_examples),
        example_names=("name", _join_examples),
        example_addresses=("address", _join_examples),
    ).reset_index()
    summary = summary.merge(examples, on=key_col, how="left")
    return summary


def summarize_prefixes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "prefix1": build_group_summary(df, "sid_prefix1"),
        "prefix2": build_group_summary(df, "sid_prefix2"),
        "prefix3": build_group_summary(df, "sid_str"),
    }


def summarize_pid_groups(df: pd.DataFrame) -> pd.DataFrame:
    summary = build_group_summary(df, "pid_gid6_sid")
    geohash6 = df.groupby("pid_gid6_sid", sort=False)["geohash6"].first().rename("geohash6").reset_index()
    summary = summary.merge(geohash6, on="pid_gid6_sid", how="left")
    return summary


def _read_reference_csv(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def _require_reference_columns(df: pd.DataFrame, path: str | Path) -> None:
    missing = [column for column in ("query_id", "poi_id") if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def analyze_qrels(mapping: pd.DataFrame, qrels_path: str | Path, out_csv: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    qrels = _read_reference_csv(qrels_path)
    if qrels is None:
        return {"exists": False}, pd.DataFrame()
    _require_reference_columns(qrels, qrels_path)

    lookup = mapping[["poi_id", "sid_str", "pid_gid6_sid", "pid_gid6_sid_dedup"]].copy()
    sid_sizes = mapping.groupby("sid_str", sort=False).size().rename("sid_group_size").reset_index()
    pid_sizes = mapping.groupby("pid_gid6_sid", sort=False).size().rename("pid_group_size").reset_index()
    merged = qrels.merge(lookup, on="poi_id", how="left")
    merged["missing_in_sid"] = merged["sid_str"].isna()
    merged = merged.merge(sid_sizes, on="sid_str", how="left")
    merged = merged.merge(pid_sizes, on="pid_gid6_sid", how="left")
    merged["sid_group_size"] = merged["sid_group_size"].fillna(0).astype(np.int64)
    merged["pid_group_size"] = merged["pid_group_size"].fillna(0).astype(np.int64)

    output_columns = [
        "query_id",
        "poi_id",
        "sid_str",
        "pid_gid6_sid",
        "pid_gid6_sid_dedup",
        "sid_group_size",
        "pid_group_size",
        "missing_in_sid",
    ]
    write_csv(merged[output_columns], out_csv)

    nonmissing = merged[~merged["missing_in_sid"]]
    qrels_unique_poi = int(qrels["poi_id"].nunique())
    stats = {
        "exists": True,
        "qrels_rows": int(len(qrels)),
        "qrels_unique_poi": qrels_unique_poi,
        "qrels_missing_poi_in_sid": int(merged["missing_in_sid"].sum()),
        "qrels_target_unique_sid_count": int(nonmissing["sid_str"].nunique()),
        "qrels_target_unique_sid_rate": rate(nonmissing["sid_str"].nunique(), qrels_unique_poi),
        "qrels_target_unique_pid_count": int(nonmissing["pid_gid6_sid"].nunique()),
        "qrels_target_unique_pid_rate": rate(nonmissing["pid_gid6_sid"].nunique(), qrels_unique_poi),
        "qrels_targets_in_sid_collision_count": int((nonmissing["sid_group_size"] > 1).sum()),
        "qrels_targets_in_sid_collision_rate": rate(int((nonmissing["sid_group_size"] > 1).sum()), len(qrels)),
        "qrels_targets_in_pid_collision_count": int((nonmissing["pid_group_size"] > 1).sum()),
        "qrels_targets_in_pid_collision_rate": rate(int((nonmissing["pid_group_size"] > 1).sum()), len(qrels)),
    }
    return stats, merged


def analyze_candidates(mapping: pd.DataFrame, candidates_path: str | Path) -> dict[str, Any]:
    candidates = _read_reference_csv(candidates_path)
    if candidates is None:
        return {"exists": False}
    _require_reference_columns(candidates, candidates_path)
    lookup = mapping[["poi_id", "sid_str", "pid_gid6_sid"]].copy()
    merged = candidates.merge(lookup, on="poi_id", how="left")
    merged["missing_in_sid"] = merged["sid_str"].isna()
    per_query = (
        merged.groupby("query_id", sort=False)
        .agg(
            candidate_count=("poi_id", "size"),
            unique_sid_count=("sid_str", "nunique"),
            unique_pid_count=("pid_gid6_sid", "nunique"),
            missing_poi_in_sid=("missing_in_sid", "sum"),
        )
        .reset_index()
    )
    per_query["sid_collision_within_query_count"] = per_query["candidate_count"] - per_query["unique_sid_count"]
    per_query["pid_collision_within_query_count"] = per_query["candidate_count"] - per_query["unique_pid_count"]
    query_count = len(per_query)
    return {
        "exists": True,
        "candidate_queries": int(query_count),
        "candidate_rows": int(len(candidates)),
        "candidate_missing_poi_in_sid": int(merged["missing_in_sid"].sum()),
        "mean_candidate_count": float(per_query["candidate_count"].mean()) if query_count else 0.0,
        "mean_unique_sid_per_query": float(per_query["unique_sid_count"].mean()) if query_count else 0.0,
        "mean_unique_pid_per_query": float(per_query["unique_pid_count"].mean()) if query_count else 0.0,
        "queries_with_sid_collision_rate": rate(int((per_query["sid_collision_within_query_count"] > 0).sum()), query_count),
        "queries_with_pid_collision_rate": rate(int((per_query["pid_collision_within_query_count"] > 0).sum()), query_count),
    }


def write_mode_csv_reports(
    mode: str,
    mapping_with_labels: pd.DataFrame,
    prefix_summaries: dict[str, pd.DataFrame],
    qrels_merged: pd.DataFrame,
    report_dir: str | Path,
    max_report_groups: int = 1000,
) -> dict[str, str]:
    report_dir = Path(report_dir)
    paths: dict[str, str] = {}

    sid_collision = prefix_summaries["prefix3"]
    sid_collision = sid_collision[sid_collision["group_size"] > 1].sort_values(
        ["group_size", "semantic_purity"], ascending=[False, True]
    )
    sid_collision_path = report_dir / f"sid_collision_groups_{mode}.csv"
    write_csv(
        sid_collision[
            [
                "sid_str",
                "group_size",
                "top_semantic_label",
                "semantic_purity",
                "top_city",
                "city_purity",
                "top_geohash5",
                "geohash5_purity",
                "example_poi_ids",
                "example_names",
                "example_addresses",
            ]
        ].head(max_report_groups),
        sid_collision_path,
    )
    paths["sid_collision_groups"] = str(sid_collision_path)

    pid_summary = summarize_pid_groups(mapping_with_labels)
    pid_collision = pid_summary[pid_summary["group_size"] > 1].sort_values(
        ["group_size", "semantic_purity"], ascending=[False, True]
    )
    pid_collision_path = report_dir / f"pid_collision_groups_{mode}.csv"
    write_csv(
        pid_collision[
            [
                "pid_gid6_sid",
                "group_size",
                "geohash6",
                "top_semantic_label",
                "semantic_purity",
                "top_city",
                "city_purity",
                "example_poi_ids",
                "example_names",
                "example_addresses",
            ]
        ].head(max_report_groups),
        pid_collision_path,
    )
    paths["pid_collision_groups"] = str(pid_collision_path)

    prefix_specs = [
        ("prefix1", "sid_prefix1", report_dir / f"sid_prefix1_summary_{mode}.csv"),
        ("prefix2", "sid_prefix2", report_dir / f"sid_prefix2_summary_{mode}.csv"),
        ("prefix3", "sid_str", report_dir / f"sid_prefix3_summary_{mode}.csv"),
    ]
    for prefix_key, key_col, path in prefix_specs:
        summary = prefix_summaries[prefix_key].sort_values(["group_size", key_col], ascending=[False, True])
        columns = [
            key_col,
            "group_size",
            "top_semantic_label",
            "semantic_purity",
            "top_city",
            "city_purity",
            "top_geohash5",
            "geohash5_purity",
            "example_names",
            "example_addresses",
        ]
        write_csv(summary[columns], path)
        paths[f"{prefix_key}_summary"] = str(path)

    manual = build_manual_cluster_samples(prefix_summaries, qrels_merged)
    manual_path = report_dir / f"sid_manual_cluster_samples_{mode}.csv"
    write_csv(manual.head(500), manual_path)
    paths["manual_cluster_samples"] = str(manual_path)
    return paths


def _manual_rows(
    sample_group: str,
    df: pd.DataFrame,
    key_col: str,
    prefix_level: str,
    limit: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.head(limit).copy()
    out.insert(0, "sample_group", sample_group)
    out["sid_key"] = out[key_col]
    out["prefix_level"] = prefix_level
    columns = [
        "sample_group",
        "sid_key",
        "prefix_level",
        "group_size",
        "top_semantic_label",
        "semantic_purity",
        "top_city",
        "city_purity",
        "top_geohash5",
        "geohash5_purity",
        "example_poi_ids",
        "example_names",
        "example_addresses",
    ]
    return out[columns]


def build_manual_cluster_samples(prefix_summaries: dict[str, pd.DataFrame], qrels_merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    prefix1 = prefix_summaries["prefix1"]
    prefix2 = prefix_summaries["prefix2"]
    prefix3 = prefix_summaries["prefix3"]

    largest_sid = prefix3[prefix3["group_size"] > 1].sort_values("group_size", ascending=False)
    rows.append(_manual_rows("largest_collision_sid", largest_sid, "sid_str", "sid_str", 100))

    low_semantic = prefix1[prefix1["group_size"] > 1].sort_values(["semantic_purity", "group_size"], ascending=[True, False])
    rows.append(_manual_rows("low_semantic_purity_prefix1", low_semantic, "sid_prefix1", "prefix1", 100))

    high_semantic = prefix1[prefix1["group_size"] >= 20].sort_values(
        ["semantic_purity", "group_size"], ascending=[False, False]
    )
    rows.append(_manual_rows("high_semantic_purity_prefix1", high_semantic, "sid_prefix1", "prefix1", 100))

    high_geo = prefix1[prefix1["group_size"] >= 20].sort_values(
        ["geohash5_purity", "group_size"], ascending=[False, False]
    )
    rows.append(_manual_rows("high_geo_purity_prefix1", high_geo, "sid_prefix1", "prefix1", 100))

    random_prefix2 = prefix2.sample(n=min(80, len(prefix2)), random_state=42) if len(prefix2) else prefix2
    rows.append(_manual_rows("random_prefix2", random_prefix2, "sid_prefix2", "prefix2", 80))

    if not qrels_merged.empty and "sid_group_size" in qrels_merged.columns:
        qrels_sids = qrels_merged.loc[qrels_merged["sid_group_size"] > 1, "sid_str"].dropna().drop_duplicates()
        qrels_collision = prefix3[prefix3["sid_str"].isin(qrels_sids)].sort_values("group_size", ascending=False)
        rows.append(_manual_rows("qrels_collision_sid", qrels_collision, "sid_str", "sid_str", 20))

    if not rows:
        return pd.DataFrame()
    return pd.concat([frame for frame in rows if not frame.empty], ignore_index=True)


def build_mode_metrics(
    mode: str,
    mapping: pd.DataFrame,
    indices: np.ndarray,
    codebook_size: int,
    checkpoint_path: str | Path,
    rqvae_input_path: str | Path,
    mapping_path: str | Path,
    indices_path: str | Path,
    report_dir: str | Path,
    qrels_path: str | Path,
    candidates_path: str | Path,
    max_report_groups: int = 1000,
) -> dict[str, Any]:
    mapping_with_labels = add_analysis_labels(mapping)
    prefix_summaries = summarize_prefixes(mapping_with_labels)
    usages = codebook_usage(indices, codebook_size)
    sid_stats = collision_stats(mapping, "sid_str", "sid")
    pid_stats = pid_collision_stats(mapping)
    prefix_stats = {
        key: {**prefix_group_size_stats(summary), **weighted_purity(summary)}
        for key, summary in prefix_summaries.items()
    }

    qrels_report_path = Path(report_dir) / f"qrels_sid_analysis_{mode}.csv"
    qrels_stats, qrels_merged = analyze_qrels(mapping, qrels_path, qrels_report_path)
    candidates_stats = analyze_candidates(mapping, candidates_path)
    csv_paths = write_mode_csv_reports(
        mode=mode,
        mapping_with_labels=mapping_with_labels,
        prefix_summaries=prefix_summaries,
        qrels_merged=qrels_merged,
        report_dir=report_dir,
        max_report_groups=max_report_groups,
    )
    csv_paths["qrels_sid_analysis"] = str(qrels_report_path)

    return {
        "mode": mode,
        "input_rows": int(len(mapping)),
        "checkpoint_path": str(checkpoint_path),
        "rqvae_input_path": str(rqvae_input_path),
        "mapping_path": str(mapping_path),
        "indices_path": str(indices_path),
        "codebook_size": int(codebook_size),
        "num_codebooks": int(indices.shape[1]),
        "unique_poi_id": int(mapping["poi_id"].nunique()),
        "row_order_aligned_with_embedding_meta": True,
        "row_order_aligned_with_geo_meta": True,
        "sid_example": clean_text(mapping["sid_str"].iloc[0]) if len(mapping) else "",
        "gid6_example": clean_text(mapping["gid6_str"].iloc[0]) if len(mapping) else "",
        "pid_gid6_sid_example": clean_text(mapping["pid_gid6_sid"].iloc[0]) if len(mapping) else "",
        "pid_gid6_sid_dedup_example": clean_text(mapping["pid_gid6_sid_dedup"].iloc[0]) if len(mapping) else "",
        "codebook_usage": usages,
        "sid_stats": sid_stats,
        "pid_stats": pid_stats,
        "prefix_stats": prefix_stats,
        "qrels_stats": qrels_stats,
        "candidates_stats": candidates_stats,
        "csv_paths": csv_paths,
        "prefix_summaries": prefix_summaries,
        "mapping_with_labels": mapping_with_labels,
    }


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:
    if df.empty:
        return "无样本。"
    view = df[columns].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: fmt_float(x))
        else:
            view[column] = view[column].map(lambda x: clean_text(x).replace("\n", " "))
    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy(dtype=object)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        escaped = [cell.replace("|", "/") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def build_quality_report(metrics: dict[str, Any]) -> str:
    sid_stats = metrics["sid_stats"]
    pid_stats = metrics["pid_stats"]
    prefix_stats = metrics["prefix_stats"]
    qrels = metrics["qrels_stats"]
    candidates = metrics["candidates_stats"]
    prefix_summaries = metrics["prefix_summaries"]

    lines: list[str] = []
    lines.append("# SID Quality Report")
    lines.append("")
    lines.append("## Basic Information")
    lines.append(f"- mode: {metrics['mode']}")
    lines.append(f"- input_rows: {metrics['input_rows']}")
    lines.append(f"- checkpoint_path: {metrics['checkpoint_path']}")
    lines.append(f"- rqvae_input_path: {metrics['rqvae_input_path']}")
    lines.append(f"- mapping_path: {metrics['mapping_path']}")
    lines.append(f"- indices_path: {metrics['indices_path']}")
    lines.append(f"- codebook_size: {metrics['codebook_size']}")
    lines.append(f"- num_codebooks: {metrics['num_codebooks']}")
    lines.append(f"- unique_poi_id: {metrics['unique_poi_id']}")
    lines.append(f"- row_order_aligned_with_embedding_meta: {metrics['row_order_aligned_with_embedding_meta']}")
    lines.append(f"- row_order_aligned_with_geo_meta: {metrics['row_order_aligned_with_geo_meta']}")
    lines.append("")

    lines.append("## SID Format")
    lines.append(f"- sid example: {metrics['sid_example']}")
    lines.append(f"- gid6 example: {metrics['gid6_example']}")
    lines.append(f"- pid_gid6_sid example: {metrics['pid_gid6_sid_example']}")
    lines.append(f"- pid_gid6_sid_dedup example: {metrics['pid_gid6_sid_dedup_example']}")
    lines.append("")

    lines.append("## Codebook Usage")
    for usage in metrics["codebook_usage"]:
        top = ", ".join(
            f"{item['code']}:{item['count']}({fmt_float(item['rate'])})"
            for item in usage["top_code_distribution"]
        )
        lines.append(f"### codebook{usage['level']}")
        lines.append(f"- used_code_count / rate: {usage['used_code_count']} / {fmt_float(usage['used_code_rate'])}")
        lines.append(f"- dead_code_count: {usage['dead_code_count']}")
        lines.append(f"- entropy: {fmt_float(usage['entropy'])}")
        lines.append(f"- perplexity: {fmt_float(usage['perplexity'])}")
        lines.append(f"- top 10 code frequency: {top}")
    lines.append("")

    lines.append("## Full SID Collision")
    lines.append(f"- unique_sid_count / rate: {sid_stats['unique_sid_count']} / {fmt_float(sid_stats['unique_sid_rate'])}")
    lines.append(f"- sid_collision_group_count: {sid_stats['sid_collision_group_count']}")
    lines.append(f"- sid_collision_poi_count: {sid_stats['sid_collision_poi_count']}")
    lines.append(f"- max_pois_per_sid: {sid_stats['max_pois_per_sid']}")
    lines.append(f"- mean_pois_per_sid: {fmt_float(sid_stats['mean_pois_per_sid'])}")
    lines.append(
        f"- p50 / p95 / p99 pois_per_sid: {fmt_float(sid_stats['p50_pois_per_sid'])} / "
        f"{fmt_float(sid_stats['p95_pois_per_sid'])} / {fmt_float(sid_stats['p99_pois_per_sid'])}"
    )
    lines.append("")

    lines.append("## PID Collision")
    lines.append(
        f"- unique_pid_gid6_sid_count / rate: {pid_stats['unique_pid_gid6_sid_count']} / "
        f"{fmt_float(pid_stats['unique_pid_gid6_sid_rate'])}"
    )
    lines.append(f"- pid_collision_group_count: {pid_stats['pid_collision_group_count']}")
    lines.append(f"- pid_collision_poi_count: {pid_stats['pid_collision_poi_count']}")
    lines.append(f"- max_pois_per_pid: {pid_stats['max_pois_per_pid']}")
    lines.append(
        f"- pid_gid6_sid_dedup_unique_count / rate: {pid_stats['pid_gid6_sid_dedup_unique_count']} / "
        f"{fmt_float(pid_stats['pid_gid6_sid_dedup_unique_rate'])}"
    )
    lines.append("")

    lines.append("## Prefix Group Size")
    for key in ("prefix1", "prefix2", "prefix3"):
        stats = prefix_stats[key]
        lines.append(f"### {key}")
        lines.append(f"- num_groups: {stats['num_groups']}")
        lines.append(f"- mean_group_size: {fmt_float(stats['mean_group_size'])}")
        lines.append(
            f"- p50 / p95 / p99 / max group_size: {fmt_float(stats['p50_group_size'])} / "
            f"{fmt_float(stats['p95_group_size'])} / {fmt_float(stats['p99_group_size'])} / {stats['max_group_size']}"
        )
        lines.append(
            f"- singleton_group_count / rate: {stats['singleton_group_count']} / "
            f"{fmt_float(stats['singleton_group_rate'])}"
        )
    lines.append("")

    lines.append("## Prefix Semantic Purity")
    for key in ("prefix1", "prefix2", "prefix3"):
        stats = prefix_stats[key]
        lines.append(f"### {key}")
        lines.append(f"- weighted_semantic_purity: {fmt_float(stats['weighted_semantic_purity'])}")
        lines.append(f"- weighted_city_purity: {fmt_float(stats['weighted_city_purity'])}")
        lines.append(f"- weighted_geohash5_purity: {fmt_float(stats['weighted_geohash5_purity'])}")
    lines.append("")

    lines.append("## Qrels SID Analysis")
    if qrels.get("exists"):
        lines.append(f"- qrels_rows: {qrels['qrels_rows']}")
        lines.append(f"- qrels_unique_poi: {qrels['qrels_unique_poi']}")
        lines.append(f"- qrels_missing_poi_in_sid: {qrels['qrels_missing_poi_in_sid']}")
        lines.append(
            f"- qrels_target_unique_sid_count / rate: {qrels['qrels_target_unique_sid_count']} / "
            f"{fmt_float(qrels['qrels_target_unique_sid_rate'])}"
        )
        lines.append(
            f"- qrels_target_unique_pid_count / rate: {qrels['qrels_target_unique_pid_count']} / "
            f"{fmt_float(qrels['qrels_target_unique_pid_rate'])}"
        )
        lines.append(
            f"- qrels_targets_in_sid_collision_count / rate: {qrels['qrels_targets_in_sid_collision_count']} / "
            f"{fmt_float(qrels['qrels_targets_in_sid_collision_rate'])}"
        )
        lines.append(
            f"- qrels_targets_in_pid_collision_count / rate: {qrels['qrels_targets_in_pid_collision_count']} / "
            f"{fmt_float(qrels['qrels_targets_in_pid_collision_rate'])}"
        )
    else:
        lines.append("- qrels file not found, skipped.")
    lines.append("")

    lines.append("## Candidates SID Analysis")
    if candidates.get("exists"):
        lines.append(f"- candidate_queries: {candidates['candidate_queries']}")
        lines.append(f"- candidate_rows: {candidates['candidate_rows']}")
        lines.append(f"- candidate_missing_poi_in_sid: {candidates['candidate_missing_poi_in_sid']}")
        lines.append(f"- mean_candidate_count: {fmt_float(candidates['mean_candidate_count'])}")
        lines.append(f"- mean_unique_sid_per_query: {fmt_float(candidates['mean_unique_sid_per_query'])}")
        lines.append(f"- mean_unique_pid_per_query: {fmt_float(candidates['mean_unique_pid_per_query'])}")
        lines.append(f"- queries_with_sid_collision_rate: {fmt_float(candidates['queries_with_sid_collision_rate'])}")
        lines.append(f"- queries_with_pid_collision_rate: {fmt_float(candidates['queries_with_pid_collision_rate'])}")
    else:
        lines.append("- candidates file not found, skipped.")
    lines.append("")

    lines.append("## Manual Cluster Examples")
    lines.append("### 最大 SID collision group Top 10")
    largest_sid = prefix_summaries["prefix3"][prefix_summaries["prefix3"]["group_size"] > 1].sort_values(
        "group_size", ascending=False
    )
    lines.append(
        _markdown_table(
            largest_sid,
            ["sid_str", "group_size", "top_semantic_label", "top_city", "top_geohash5", "example_names", "example_addresses"],
            10,
        )
    )
    lines.append("")
    lines.append("### semantic_purity 最低的 prefix1 group Top 10")
    low_semantic = prefix_summaries["prefix1"].sort_values(["semantic_purity", "group_size"], ascending=[True, False])
    lines.append(
        _markdown_table(
            low_semantic,
            [
                "sid_prefix1",
                "group_size",
                "top_semantic_label",
                "semantic_purity",
                "top_city",
                "top_geohash5",
                "example_names",
                "example_addresses",
            ],
            10,
        )
    )
    lines.append("")
    lines.append("### semantic_purity 最高且 group_size >= 20 的 prefix1 group Top 10")
    high_semantic = prefix_summaries["prefix1"][prefix_summaries["prefix1"]["group_size"] >= 20].sort_values(
        ["semantic_purity", "group_size"], ascending=[False, False]
    )
    lines.append(
        _markdown_table(
            high_semantic,
            [
                "sid_prefix1",
                "group_size",
                "top_semantic_label",
                "semantic_purity",
                "top_city",
                "top_geohash5",
                "example_names",
                "example_addresses",
            ],
            10,
        )
    )
    lines.append("")
    lines.append("### geohash5_purity 最高且 group_size >= 20 的 prefix1 group Top 10")
    high_geo = prefix_summaries["prefix1"][prefix_summaries["prefix1"]["group_size"] >= 20].sort_values(
        ["geohash5_purity", "group_size"], ascending=[False, False]
    )
    lines.append(
        _markdown_table(
            high_geo,
            [
                "sid_prefix1",
                "group_size",
                "top_semantic_label",
                "top_city",
                "top_geohash5",
                "geohash5_purity",
                "example_names",
                "example_addresses",
            ],
            10,
        )
    )
    lines.append("")
    lines.append("### qrels target 所在 SID collision group 示例 Top 20")
    qrels_path = metrics["csv_paths"].get("qrels_sid_analysis")
    qrels_examples = pd.DataFrame()
    if qrels.get("exists") and qrels_path and Path(qrels_path).exists():
        qrels_df = pd.read_csv(qrels_path, dtype=str, keep_default_na=False)
        qrels_sids = qrels_df.loc[qrels_df["sid_group_size"].astype(int) > 1, "sid_str"].drop_duplicates()
        qrels_examples = prefix_summaries["prefix3"][prefix_summaries["prefix3"]["sid_str"].isin(qrels_sids)].sort_values(
            "group_size", ascending=False
        )
    lines.append(
        _markdown_table(
            qrels_examples,
            ["sid_str", "group_size", "top_semantic_label", "top_city", "top_geohash5", "example_names", "example_addresses"],
            20,
        )
    )
    lines.append("")

    lines.append("## Diagnostics")
    diagnostics = build_diagnostics(metrics)
    for item in diagnostics:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def build_diagnostics(metrics: dict[str, Any]) -> list[str]:
    diagnostics: list[str] = []
    for usage in metrics["codebook_usage"]:
        if usage["used_code_rate"] < 0.9:
            diagnostics.append(f"codebook{usage['level']} usage rate < 0.9，提示 codebook 利用不足。")
    sid_stats = metrics["sid_stats"]
    pid_stats = metrics["pid_stats"]
    prefix1 = metrics["prefix_stats"]["prefix1"]
    if sid_stats["unique_sid_rate"] < 0.7:
        diagnostics.append("unique_sid_rate < 0.7，提示 full SID collision 偏高。")
    if sid_stats["max_pois_per_sid"] > 50:
        diagnostics.append("max_pois_per_sid > 50，提示存在大碰撞组。")
    if pid_stats["pid_gid6_sid_dedup_unique_rate"] < 1.0:
        diagnostics.append("pid_gid6_sid_dedup_unique_rate < 1.0，标记为错误。")
    if prefix1["weighted_semantic_purity"] < 0.5:
        diagnostics.append("prefix1 semantic purity 较低，提示第一层 SID 没有形成稳定粗语义。")
    if not diagnostics:
        diagnostics.append("未发现硬性质量错误；建议结合 manual cluster samples 做人工抽查。")
    return diagnostics


def compare_recommendation(semantic: dict[str, Any], geo_fused: dict[str, Any]) -> str:
    sem_prefix = semantic["prefix_stats"]
    geo_prefix = geo_fused["prefix_stats"]
    sem_pid = semantic["pid_stats"]["unique_pid_gid6_sid_rate"]
    geo_pid = geo_fused["pid_stats"]["unique_pid_gid6_sid_rate"]
    sem_semantic_p3 = sem_prefix["prefix3"]["weighted_semantic_purity"]
    geo_semantic_p3 = geo_prefix["prefix3"]["weighted_semantic_purity"]
    sem_geo_p1 = sem_prefix["prefix1"]["weighted_geohash5_purity"]
    geo_geo_p1 = geo_prefix["prefix1"]["weighted_geohash5_purity"]
    sem_semantic_p1 = sem_prefix["prefix1"]["weighted_semantic_purity"]
    geo_semantic_p1 = geo_prefix["prefix1"]["weighted_semantic_purity"]

    if geo_semantic_p3 >= sem_semantic_p3 - 0.02 and geo_pid >= sem_pid:
        return "推荐使用 geo_fused：semantic purity 没有明显下降，且 PID collision 更低或持平。"
    if geo_geo_p1 >= sem_geo_p1 + 0.10 and geo_semantic_p1 <= sem_semantic_p1 - 0.05:
        return "推荐使用 semantic：geo_fused 的地理纯度明显升高但语义纯度明显下降，可能 geo_alpha 仍偏大。"
    return "两者接近，优先推荐 semantic 作为第一版生成式检索 target，geo_fused 作为消融对照。"


def build_compare_report(semantic: dict[str, Any], geo_fused: dict[str, Any]) -> tuple[str, str]:
    rows = []
    for mode, metrics in [("semantic", semantic), ("geo_fused", geo_fused)]:
        prefix = metrics["prefix_stats"]
        qrels = metrics["qrels_stats"]
        candidates = metrics["candidates_stats"]
        rows.append(
            {
                "mode": mode,
                "unique_sid_rate": metrics["sid_stats"]["unique_sid_rate"],
                "unique_pid_gid6_sid_rate": metrics["pid_stats"]["unique_pid_gid6_sid_rate"],
                "max_pois_per_sid": metrics["sid_stats"]["max_pois_per_sid"],
                "codebook_usage_mean": float(np.mean([u["used_code_rate"] for u in metrics["codebook_usage"]])),
                "prefix1_weighted_semantic_purity": prefix["prefix1"]["weighted_semantic_purity"],
                "prefix2_weighted_semantic_purity": prefix["prefix2"]["weighted_semantic_purity"],
                "prefix3_weighted_semantic_purity": prefix["prefix3"]["weighted_semantic_purity"],
                "prefix1_weighted_city_purity": prefix["prefix1"]["weighted_city_purity"],
                "prefix2_weighted_city_purity": prefix["prefix2"]["weighted_city_purity"],
                "prefix3_weighted_city_purity": prefix["prefix3"]["weighted_city_purity"],
                "prefix1_weighted_geohash5_purity": prefix["prefix1"]["weighted_geohash5_purity"],
                "prefix2_weighted_geohash5_purity": prefix["prefix2"]["weighted_geohash5_purity"],
                "prefix3_weighted_geohash5_purity": prefix["prefix3"]["weighted_geohash5_purity"],
                "qrels_targets_in_sid_collision_rate": qrels.get("qrels_targets_in_sid_collision_rate", 0.0),
                "qrels_targets_in_pid_collision_rate": qrels.get("qrels_targets_in_pid_collision_rate", 0.0),
                "queries_with_sid_collision_rate": candidates.get("queries_with_sid_collision_rate", 0.0),
                "queries_with_pid_collision_rate": candidates.get("queries_with_pid_collision_rate", 0.0),
            }
        )
    table = pd.DataFrame(rows)
    view = table.copy()
    for column in view.columns:
        if column != "mode" and pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: fmt_float(x))
    recommendation = compare_recommendation(semantic, geo_fused)
    lines = [
        "# SID Quality Compare Report",
        "",
        "## Summary Table",
        _markdown_table(view, list(view.columns), max_rows=len(view)),
        "",
        "## Recommendation",
        f"- {recommendation}",
        "",
    ]
    return "\n".join(lines), recommendation

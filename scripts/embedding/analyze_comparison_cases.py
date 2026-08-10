#!/usr/bin/env python3
"""Analyze aligned E1/E2 retrieval cases without running model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EVAL_ROWS = 10_000
CATEGORY_NAMES = (
    "only_0.6b_hit10",
    "only_4b_hit10",
    "both_hit_0.6b_better",
    "both_hit_4b_better",
    "both_miss20",
)
LENGTH_BUCKETS = ("1-2", "3-5", "6-10", ">10")


class AnalysisError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复用现有 E1/E2 Top-20 结果，对比抽样并统计可解释字符串特征。",
    )
    parser.add_argument("--eval-data", type=Path, required=True, help="评测 JSONL。")
    parser.add_argument("--e1-dir", type=Path, required=True, help="E1 结果目录。")
    parser.add_argument("--e2-dir", type=Path, required=True, help="E2 结果目录。")
    parser.add_argument("--poi-data", type=Path, required=True, help="POI JSONL 文件或分片目录。")
    parser.add_argument("--output", type=Path, required=True, help="分析输出目录。")
    parser.add_argument("--seed", type=int, default=20260722, help="Case 抽样随机种子。")
    parser.add_argument(
        "--max-cases-per-category",
        type=int,
        default=30,
        help="每类最多输出的 Case 数。",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisError(f"文件不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AnalysisError(f"JSON 根节点不是 object：{path}")
    return payload


def load_eval_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise AnalysisError(f"评测数据第 {line_number} 行解析失败") from error
            if not isinstance(record, dict):
                raise AnalysisError(f"评测数据第 {line_number} 行不是 object")
            for field in ("order_id", "query", "poi_id"):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise AnalysisError(f"评测数据第 {line_number} 行 {field} 无效")
            records.append(record)
    if len(records) != EXPECTED_EVAL_ROWS:
        raise AnalysisError(
            f"评测数据行数 {len(records)} != {EXPECTED_EVAL_ROWS}"
        )
    if len({record["order_id"] for record in records}) != EXPECTED_EVAL_ROWS:
        raise AnalysisError("评测数据 order_id 不唯一")
    return records


def resolve_manifest_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"run_manifest 缺少 {name}")
    return resolve_path(Path(value))


def validate_query_mapping(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    run_name: str,
) -> Path:
    if manifest.get("status") != "completed":
        raise AnalysisError(f"{run_name} 状态不是 completed")
    mapping_path = resolve_manifest_path(
        manifest.get("outputs", {}).get("query_mapping"),
        f"{run_name}.outputs.query_mapping",
    )
    row_count = 0
    with mapping_path.open("r", encoding="utf-8") as handle:
        for row_count, (record, line) in enumerate(
            zip(records, handle, strict=True),
            start=1,
        ):
            mapping = json.loads(line)
            if mapping.get("row_index") != row_count - 1:
                raise AnalysisError(f"{run_name} 第 {row_count} 行 row_index 错误")
            if mapping.get("order_id") != record["order_id"]:
                raise AnalysisError(f"{run_name} 第 {row_count} 行 order_id 顺序错误")
            if mapping.get("target_poi_id") != record["poi_id"]:
                raise AnalysisError(f"{run_name} 第 {row_count} 行目标 POI 错误")
    if row_count != len(records):
        raise AnalysisError(f"{run_name} 行映射数量错误：{row_count}")
    expected_sha256 = manifest.get("outputs", {}).get("query_mapping_sha256")
    if expected_sha256 and sha256_file(mapping_path) != expected_sha256:
        raise AnalysisError(f"{run_name} 行映射 SHA256 错误")
    return mapping_path


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    return {
        "hit_at_1": float(np.mean((ranks >= 1) & (ranks <= 1))),
        "hit_at_3": float(np.mean((ranks >= 1) & (ranks <= 3))),
        "hit_at_5": float(np.mean((ranks >= 1) & (ranks <= 5))),
        "hit_at_10": float(np.mean((ranks >= 1) & (ranks <= 10))),
        "hit_at_20": float(np.mean((ranks >= 1) & (ranks <= 20))),
        "mrr_at_10": float(
            np.mean(
                np.where(
                    (ranks >= 1) & (ranks <= 10),
                    1.0 / ranks.clip(min=1),
                    0.0,
                )
            )
        ),
    }


def load_run_arrays(
    run_dir: Path,
    records: list[dict[str, Any]],
    run_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, float], Path]:
    manifest = load_json(run_dir / "run_manifest.json")
    validate_query_mapping(manifest, records, run_name)
    eval_sha256 = manifest.get("gate_zero", {}).get("eval_sha256")
    if eval_sha256 != sha256_file(resolve_path(Path(manifest["inputs"]["eval_data"]))):
        raise AnalysisError(f"{run_name} manifest 中评测数据 SHA256 不一致")

    results_path = run_dir / "retrieval_results.npz"
    with np.load(results_path, allow_pickle=False) as payload:
        required = {"topk_indices", "topk_scores", "target_ranks"}
        if set(payload.files) != required:
            raise AnalysisError(f"{run_name} NPZ 字段错误：{payload.files}")
        arrays = {name: payload[name].copy() for name in required}
    if arrays["topk_indices"].shape != (EXPECTED_EVAL_ROWS, 20):
        raise AnalysisError(f"{run_name} topk_indices shape 错误")
    if arrays["topk_scores"].shape != (EXPECTED_EVAL_ROWS, 20):
        raise AnalysisError(f"{run_name} topk_scores shape 错误")
    if arrays["target_ranks"].shape != (EXPECTED_EVAL_ROWS,):
        raise AnalysisError(f"{run_name} target_ranks shape 错误")
    if not np.isfinite(arrays["topk_scores"]).all():
        raise AnalysisError(f"{run_name} Top-K 分数包含 NaN/Inf")
    ranks = arrays["target_ranks"]
    if not np.all((ranks == -1) | ((ranks >= 1) & (ranks <= 20))):
        raise AnalysisError(f"{run_name} target_ranks 范围错误")

    recomputed_metrics = metrics_from_ranks(ranks)
    saved_metrics = load_json(run_dir / "metrics.json")
    for key, value in recomputed_metrics.items():
        if not math.isclose(value, float(saved_metrics[key]), abs_tol=1e-12):
            raise AnalysisError(f"{run_name} {key} 与 metrics.json 不一致")
    poi_ids_path = resolve_manifest_path(
        manifest.get("inputs", {}).get("poi_ids"),
        f"{run_name}.inputs.poi_ids",
    )
    return manifest, arrays, recomputed_metrics, poi_ids_path


def classify_samples(
    ranks_e1: np.ndarray,
    ranks_e2: np.ndarray,
) -> tuple[dict[str, list[int]], list[int]]:
    categories = {name: [] for name in CATEGORY_NAMES}
    unclassified: list[int] = []
    for index, (rank_e1, rank_e2) in enumerate(zip(ranks_e1, ranks_e2, strict=True)):
        if 1 <= rank_e1 <= 10 and rank_e2 == -1:
            categories["only_0.6b_hit10"].append(index)
        elif 1 <= rank_e2 <= 10 and rank_e1 == -1:
            categories["only_4b_hit10"].append(index)
        elif rank_e1 >= 1 and rank_e2 >= 1 and rank_e2 - rank_e1 >= 5:
            categories["both_hit_0.6b_better"].append(index)
        elif rank_e1 >= 1 and rank_e2 >= 1 and rank_e1 - rank_e2 >= 5:
            categories["both_hit_4b_better"].append(index)
        elif rank_e1 == -1 and rank_e2 == -1:
            categories["both_miss20"].append(index)
        else:
            unclassified.append(index)
    return categories, unclassified


def discover_poi_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise AnalysisError(f"POI 数据不存在：{path}")
    files = tuple(
        sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.name.startswith("part-")
        )
    )
    if not files:
        raise AnalysisError(f"POI 目录没有 part-* 分片：{path}")
    return files


def load_required_poi_records(
    poi_data: Path,
    e1_ids_path: Path,
    e2_ids_path: Path,
    required_indices: set[int],
    target_ids: set[str],
    expected_rows: int,
) -> tuple[dict[str, dict[str, Any]], dict[int, str], dict[str, int]]:
    poi_records: dict[str, dict[str, Any]] = {}
    index_to_id: dict[int, str] = {}
    target_indices: dict[str, int] = {}
    row_index = 0
    with (
        e1_ids_path.open("r", encoding="utf-8") as e1_ids,
        e2_ids_path.open("r", encoding="utf-8") as e2_ids,
    ):
        progress = tqdm(total=expected_rows, desc="POI text alignment", unit="row")
        for shard in discover_poi_files(poi_data):
            with shard.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise AnalysisError(
                            f"{shard.name}:{line_number} JSON 解析失败"
                        ) from error
                    if not isinstance(record, dict):
                        raise AnalysisError(f"{shard.name}:{line_number} 不是 object")
                    poi_id = record.get("poi_id")
                    if not isinstance(poi_id, str) or not poi_id:
                        raise AnalysisError(f"{shard.name}:{line_number} poi_id 无效")
                    e1_line = e1_ids.readline()
                    e2_line = e2_ids.readline()
                    if not e1_line or not e2_line:
                        raise AnalysisError("POI ID 顺序文件短于原始 POI 数据")
                    e1_id = json.loads(e1_line)
                    e2_id = json.loads(e2_line)
                    if e1_id != e2_id or e1_id != poi_id:
                        raise AnalysisError(f"POI 第 {row_index} 行索引映射不一致")
                    if row_index in required_indices:
                        index_to_id[row_index] = poi_id
                        poi_records[poi_id] = record
                    if poi_id in target_ids:
                        target_indices[poi_id] = row_index
                        poi_records[poi_id] = record
                    row_index += 1
                    progress.update(1)
        progress.close()
        if e1_ids.readline() or e2_ids.readline():
            raise AnalysisError("POI ID 顺序文件长于原始 POI 数据")
    if row_index != expected_rows:
        raise AnalysisError(f"POI 数据行数 {row_index} != {expected_rows}")
    if len(index_to_id) != len(required_indices):
        raise AnalysisError("部分召回索引未映射到 POI ID")
    if len(target_indices) != len(target_ids):
        raise AnalysisError("部分目标 POI 未在原始 POI 数据中找到")
    return poi_records, index_to_id, target_indices


def normalized_text(value: Any) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def contains_query(query: str, value: Any) -> bool:
    normalized_query = normalized_text(query)
    return bool(normalized_query) and normalized_query in normalized_text(value)


def same_name(first: Any, second: Any) -> bool:
    left = normalized_text(first)
    right = normalized_text(second)
    return bool(left) and left == right


def length_bucket(query: str) -> str:
    length = len(query.strip())
    if length <= 2:
        return "1-2"
    if length <= 5:
        return "3-5"
    if length <= 10:
        return "6-10"
    return ">10"


def ratio_payload(count: int, total: int) -> dict[str, int | float]:
    return {"count": count, "ratio": count / total if total else 0.0}


def poi_item(
    row_index: int,
    score: float,
    index_to_id: dict[int, str],
    poi_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    poi_id = index_to_id[row_index]
    record = poi_records[poi_id]
    return {
        "poi_id": poi_id,
        "displayname": record.get("displayname"),
        "text": record.get("text"),
        "score": float(score),
    }


def case_features(
    record: dict[str, Any],
    target: dict[str, Any],
    top1_e1: dict[str, Any],
    top1_e2: dict[str, Any],
    query_target_count: int,
) -> dict[str, Any]:
    query = record["query"]
    target_name = target.get("displayname")

    def model_features(top1: dict[str, Any]) -> dict[str, bool]:
        wrong_poi = top1.get("poi_id") != record["poi_id"]
        top1_name = top1.get("displayname")
        target_category = normalized_text(target.get("category_code"))
        top1_category = normalized_text(top1.get("category_code"))
        return {
            "query_in_top1_text": contains_query(query, top1.get("text")),
            "target_name_equals_top1_name": same_name(target_name, top1_name),
            "same_name_wrong_poi": wrong_poi and same_name(target_name, top1_name),
            "same_category_different_name": (
                wrong_poi
                and bool(target_category)
                and target_category == top1_category
                and not same_name(target_name, top1_name)
            ),
        }

    return {
        "query_in_target_displayname": contains_query(query, target.get("displayname")),
        "query_in_target_alias": contains_query(query, target.get("alias")),
        "query_in_target_address": contains_query(query, target.get("address")),
        "query_in_target_text": contains_query(query, target.get("text")),
        "query_length_bucket": length_bucket(query),
        "query_target_poi_count": query_target_count,
        "query_has_multiple_targets": query_target_count > 1,
        "is_short_query": len(query.strip()) <= 2,
        "is_short_or_multi_target": len(query.strip()) <= 2 or query_target_count > 1,
        "top1_0.6b": model_features(top1_e1),
        "top1_4b": model_features(top1_e2),
    }


def build_summary_statistics(
    records: list[dict[str, Any]],
    categories: dict[str, list[int]],
    unclassified: list[int],
    arrays_e1: dict[str, np.ndarray],
    arrays_e2: dict[str, np.ndarray],
    poi_records: dict[str, dict[str, Any]],
    index_to_id: dict[int, str],
    query_target_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    all_features: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(records):
        target = poi_records[record["poi_id"]]
        e1_top1_id = index_to_id[int(arrays_e1["topk_indices"][index, 0])]
        e2_top1_id = index_to_id[int(arrays_e2["topk_indices"][index, 0])]
        e1_top1 = {"poi_id": e1_top1_id, **poi_records[e1_top1_id]}
        e2_top1 = {"poi_id": e2_top1_id, **poi_records[e2_top1_id]}
        all_features[index] = case_features(
            record,
            target,
            e1_top1,
            e2_top1,
            query_target_counts[record["query"].strip()],
        )

    def statistics_for(indices: list[int]) -> dict[str, Any]:
        total = len(indices)
        lengths = Counter(all_features[index]["query_length_bucket"] for index in indices)
        target_fields = (
            "query_in_target_displayname",
            "query_in_target_alias",
            "query_in_target_address",
            "query_in_target_text",
        )
        payload: dict[str, Any] = {
            "samples": total,
            "ratio_of_eval": total / len(records),
            "query_length_distribution": {
                bucket: ratio_payload(lengths[bucket], total) for bucket in LENGTH_BUCKETS
            },
            "target_string_matches": {
                field: ratio_payload(
                    sum(bool(all_features[index][field]) for index in indices),
                    total,
                )
                for field in target_fields
            },
            "top1_query_text_matches": {
                model: ratio_payload(
                    sum(
                        bool(all_features[index][f"top1_{model}"]["query_in_top1_text"])
                        for index in indices
                    ),
                    total,
                )
                for model in ("0.6b", "4b")
            },
            "multiple_target_query_samples": ratio_payload(
                sum(all_features[index]["query_has_multiple_targets"] for index in indices),
                total,
            ),
            "short_or_multi_target_samples": ratio_payload(
                sum(all_features[index]["is_short_or_multi_target"] for index in indices),
                total,
            ),
            "top1_error_types": {},
        }
        for model in ("0.6b", "4b"):
            payload["top1_error_types"][model] = {
                field: ratio_payload(
                    sum(bool(all_features[index][f"top1_{model}"][field]) for index in indices),
                    total,
                )
                for field in (
                    "target_name_equals_top1_name",
                    "same_name_wrong_poi",
                    "same_category_different_name",
                )
            }
        return payload

    statistics = {name: statistics_for(indices) for name, indices in categories.items()}
    statistics["unclassified"] = statistics_for(unclassified)
    statistics["all_samples"] = statistics_for(list(range(len(records))))
    return statistics, all_features


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_tmp = output_dir / ".summary.json.tmp"
    cases_tmp = output_dir / ".cases.jsonl.tmp"
    with summary_tmp.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with cases_tmp.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    os.replace(summary_tmp, output_dir / "summary.json")
    os.replace(cases_tmp, output_dir / "cases.jsonl")


def main() -> int:
    args = parse_args()
    if args.max_cases_per_category <= 0:
        raise AnalysisError("--max-cases-per-category 必须大于 0")
    eval_path = resolve_path(args.eval_data)
    e1_dir = resolve_path(args.e1_dir)
    e2_dir = resolve_path(args.e2_dir)
    poi_data = resolve_path(args.poi_data)
    output_dir = resolve_path(args.output)

    records = load_eval_records(eval_path)
    eval_sha256 = sha256_file(eval_path)
    manifest_e1, arrays_e1, metrics_e1, e1_ids_path = load_run_arrays(
        e1_dir, records, "E1"
    )
    manifest_e2, arrays_e2, metrics_e2, e2_ids_path = load_run_arrays(
        e2_dir, records, "E2"
    )
    if manifest_e1.get("gate_zero", {}).get("eval_sha256") != eval_sha256:
        raise AnalysisError("E1 评测数据 SHA256 与当前文件不一致")
    if manifest_e2.get("gate_zero", {}).get("eval_sha256") != eval_sha256:
        raise AnalysisError("E2 评测数据 SHA256 与当前文件不一致")

    categories, unclassified = classify_samples(
        arrays_e1["target_ranks"], arrays_e2["target_ranks"]
    )
    rng = random.Random(args.seed)
    sampled_indices = {
        name: rng.sample(indices, min(len(indices), args.max_cases_per_category))
        for name, indices in categories.items()
    }

    required_indices = {
        int(index)
        for arrays in (arrays_e1, arrays_e2)
        for index in arrays["topk_indices"][:, 0]
    }
    for name in CATEGORY_NAMES:
        for eval_index in sampled_indices[name]:
            required_indices.update(
                int(value) for value in arrays_e1["topk_indices"][eval_index, :5]
            )
            required_indices.update(
                int(value) for value in arrays_e2["topk_indices"][eval_index, :5]
            )
    expected_poi_rows = int(manifest_e1["gate_zero"]["poi_id_rows"])
    if int(manifest_e2["gate_zero"]["poi_id_rows"]) != expected_poi_rows:
        raise AnalysisError("E1/E2 POI 候选库行数不一致")
    if min(required_indices) < 0 or max(required_indices) >= expected_poi_rows:
        raise AnalysisError("Top-K POI 行索引越界")

    target_ids = {record["poi_id"] for record in records}
    poi_records, index_to_id, target_indices = load_required_poi_records(
        poi_data,
        e1_ids_path,
        e2_ids_path,
        required_indices,
        target_ids,
        expected_poi_rows,
    )
    target_index_array = np.fromiter(
        (target_indices[record["poi_id"]] for record in records),
        dtype=np.int64,
        count=len(records),
    )
    for run_name, arrays in (("E1", arrays_e1), ("E2", arrays_e2)):
        matches = arrays["topk_indices"] == target_index_array[:, None]
        rebuilt_ranks = np.full(EXPECTED_EVAL_ROWS, -1, dtype=np.int16)
        hit = matches.any(axis=1)
        rebuilt_ranks[hit] = matches[hit].argmax(axis=1).astype(np.int16) + 1
        if not np.array_equal(rebuilt_ranks, arrays["target_ranks"]):
            raise AnalysisError(f"{run_name} target_ranks 与 POI 索引重算不一致")

    query_targets: dict[str, set[str]] = defaultdict(set)
    for record in records:
        query_targets[record["query"].strip()].add(record["poi_id"])
    query_target_counts = {query: len(targets) for query, targets in query_targets.items()}
    statistics, all_features = build_summary_statistics(
        records,
        categories,
        unclassified,
        arrays_e1,
        arrays_e2,
        poi_records,
        index_to_id,
        query_target_counts,
    )

    cases: list[dict[str, Any]] = []
    for category in CATEGORY_NAMES:
        for eval_index in sampled_indices[category]:
            record = records[eval_index]
            target = poi_records[record["poi_id"]]
            top5_e1 = [
                poi_item(
                    int(arrays_e1["topk_indices"][eval_index, rank]),
                    float(arrays_e1["topk_scores"][eval_index, rank]),
                    index_to_id,
                    poi_records,
                )
                for rank in range(5)
            ]
            top5_e2 = [
                poi_item(
                    int(arrays_e2["topk_indices"][eval_index, rank]),
                    float(arrays_e2["topk_scores"][eval_index, rank]),
                    index_to_id,
                    poi_records,
                )
                for rank in range(5)
            ]
            case = {
                "category": category,
                "eval_row_index": eval_index,
                "order_id": record["order_id"],
                "query": record["query"],
                "query_length": len(record["query"].strip()),
                "target_poi_id": record["poi_id"],
                "target_poi_name": target.get("displayname"),
                "target_poi_text": target.get("text"),
                "target_alias": target.get("alias"),
                "target_address": target.get("address"),
                "target_category": target.get("category"),
                "target_category_code": target.get("category_code"),
                "rank_0.6b": int(arrays_e1["target_ranks"][eval_index]),
                "rank_4b": int(arrays_e2["target_ranks"][eval_index]),
                "features": all_features[eval_index],
                "top5_0.6b": top5_e1,
                "top5_4b": top5_e2,
            }
            cases.append(case)

    unique_multi_queries = sum(count > 1 for count in query_target_counts.values())
    multi_target_samples = sum(
        query_target_counts[record["query"].strip()] > 1 for record in records
    )
    summary = {
        "analysis": {
            "random_seed": args.seed,
            "max_cases_per_category": args.max_cases_per_category,
            "category_definitions_are_exhaustive": False,
            "string_match_method": "query.strip().casefold() 的完整子串匹配",
            "short_or_generalized_method": "Query 长度 <= 2，或同一 strip 后 Query 对应多个目标 POI；不使用人工词表",
            "same_category_method": "目标与 Top-1 的非空 category_code 完全相同且名称不同，仅作为可解释启发式",
        },
        "inputs": {
            "eval_data": str(eval_path),
            "eval_sha256": eval_sha256,
            "e1_results": str(e1_dir / "retrieval_results.npz"),
            "e2_results": str(e2_dir / "retrieval_results.npz"),
            "e1_poi_ids": str(e1_ids_path),
            "e2_poi_ids": str(e2_ids_path),
            "poi_data": str(poi_data),
        },
        "metrics_reference": {"e1_0.6b": metrics_e1, "e2_4b": metrics_e2},
        "categories": {
            name: {
                **statistics[name],
                "sampled_cases": len(sampled_indices[name]),
            }
            for name in CATEGORY_NAMES
        },
        "unclassified": {
            **statistics["unclassified"],
            "reason": "五个指定条件并非 10,000 条样本的穷尽划分",
        },
        "all_samples": statistics["all_samples"],
        "query_multi_target": {
            "unique_queries": len(query_target_counts),
            "unique_queries_with_multiple_targets": ratio_payload(
                unique_multi_queries, len(query_target_counts)
            ),
            "samples_with_multiple_target_query": ratio_payload(
                multi_target_samples, len(records)
            ),
        },
        "correctness": {
            "eval_rows": len(records),
            "e1_rows": len(arrays_e1["target_ranks"]),
            "e2_rows": len(arrays_e2["target_ranks"]),
            "order_id_and_target_order_match": True,
            "target_ranks_match_saved_metrics": True,
            "target_ranks_match_rebuilt_poi_indices": True,
            "poi_rows": expected_poi_rows,
            "e1_e2_poi_id_order_matches_raw_data": True,
            "required_poi_text_records": len(poi_records),
            "case_poi_text_mapping_verified": True,
        },
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "cases": str(output_dir / "cases.jsonl"),
            "case_rows": len(cases),
        },
    }
    write_outputs(output_dir, summary, cases)
    print(
        json.dumps(
            {
                "status": "completed",
                "categories": {
                    name: {
                        "count": len(categories[name]),
                        "ratio": len(categories[name]) / len(records),
                    }
                    for name in CATEGORY_NAMES
                },
                "unclassified": len(unclassified),
                "case_rows": len(cases),
                "output": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reliability-gated lexical relation proxy for QGR-SID M2-D."""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from poi_gr.methods.qgr_sid.proxy import (
    PROXY_SCHEMA_VERSION,
    QgrSidProxyError,
    StrictRelationCatalog,
    TemporalSplit,
    _compile_all_buckets,
    _evaluate_holdout,
    _git_state,
    _json_dump,
    _load_json,
    _ratio,
    _save_array,
    _signature,
    _utc_now,
    _read_order_events,
    validate_proxy_output,
)
from poi_gr.methods.qgr_sid.relations import RELATION_TYPES
from poi_gr.methods.tiger.identifier import sha256_file


RELIABLE_PROXY_SCHEMA_VERSION = "qgr-sid-reliable-lexical-proxy-v1"
RELIABLE_PROXY_METRICS_SCHEMA_VERSION = (
    "qgr-sid-reliable-lexical-proxy-metrics-v1"
)


class QgrSidReliableProxyError(QgrSidProxyError):
    """Raised when M2-D inputs or frozen reliability gates are violated."""


@dataclass(frozen=True)
class ReliableProxyResult:
    """Completed M2-D reliability-gated relation artifacts."""

    metrics: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


def select_reliable_relation_types(
    *,
    early_metrics: dict[str, Any],
    min_support_orders: int,
    min_value_match_ratio: float,
) -> tuple[str, ...]:
    """Freeze relation eligibility using early-only global measurements."""

    if min_support_orders <= 0:
        raise QgrSidReliableProxyError("min_support_orders 必须大于 0")
    if not 0.0 < min_value_match_ratio <= 1.0:
        raise QgrSidReliableProxyError("min_value_match_ratio 必须位于 (0,1]")
    per_relation = early_metrics.get("per_relation")
    if not isinstance(per_relation, dict):
        raise QgrSidReliableProxyError("M2-A early metrics 缺少 per_relation")
    selected = []
    for relation_type in RELATION_TYPES:
        value = per_relation.get(relation_type)
        if not isinstance(value, dict):
            raise QgrSidReliableProxyError(f"M2-A 缺少关系：{relation_type}")
        if int(value["early_support_order_count"]) < min_support_orders:
            continue
        if float(value["early_value_aware_match_ratio"]) < min_value_match_ratio:
            continue
        selected.append(relation_type)
    if not selected:
        raise QgrSidReliableProxyError("可靠性门禁未选中任何关系类型")
    return tuple(selected)


def _load_m2a(
    m2a_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    validate_proxy_output(m2a_dir)
    manifest = _load_json(m2a_dir / "manifest.json", "M2-A manifest")
    metrics = _load_json(m2a_dir / "metrics.json", "M2-A metrics")
    if manifest.get("schema_version") != PROXY_SCHEMA_VERSION:
        raise QgrSidReliableProxyError("M2-A schema_version 不受支持")
    contracts = manifest.get("outputs", {}).get("arrays")
    if not isinstance(contracts, dict):
        raise QgrSidReliableProxyError("M2-A 缺少 arrays")
    names = (
        "collision_poi_ids",
        "base_sid_keys",
        "strict_relations",
        "conflict_masks",
        "early_order_counts",
        "early_relation_match_counts",
        "query_guided_path_types",
        "query_guided_path_values",
        "query_guided_resolved",
    )
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        contract = contracts.get(name)
        if not isinstance(contract, dict):
            raise QgrSidReliableProxyError(f"M2-A 缺少数组：{name}")
        arrays[name] = np.load(
            m2a_dir / str(contract["path"]), mmap_mode="r", allow_pickle=False
        )
    return arrays, manifest, metrics


def run_reliable_lexical_proxy(
    *,
    project_root: Path,
    order_dir: Path,
    sft_manifest_path: Path,
    m2a_dir: Path,
    output_dir: Path,
    min_support_orders: int = 10_000,
    min_value_match_ratio: float = 0.15,
    max_pairs: int = 3,
    prior_orders: float = 20.0,
    examples_per_kind: int = 20,
    progress: Callable[[str], None] | None = None,
) -> ReliableProxyResult:
    """Compile and evaluate one pre-registered reliable lexical relation tree."""

    started_at = _utc_now()
    started = time.monotonic()
    project_root = project_root.resolve()
    order_dir = order_dir.resolve()
    sft_manifest_path = sft_manifest_path.resolve()
    m2a_dir = m2a_dir.resolve()
    output_dir = output_dir.resolve()
    for path in (project_root, order_dir, m2a_dir):
        if not path.is_dir():
            raise QgrSidReliableProxyError(f"目录不存在：{path}")
    if not sft_manifest_path.is_file():
        raise QgrSidReliableProxyError("SFT manifest 不存在")
    if output_dir.exists():
        raise QgrSidReliableProxyError("输出目录已存在，拒绝覆盖")

    arrays, m2a_manifest, m2a_metrics = _load_m2a(m2a_dir)
    split = TemporalSplit(**m2a_metrics["temporal_split"])
    selected_types = select_reliable_relation_types(
        early_metrics=m2a_metrics["early"],
        min_support_orders=min_support_orders,
        min_value_match_ratio=min_value_match_ratio,
    )
    selected_indices = np.asarray(
        [RELATION_TYPES.index(name) for name in selected_types], dtype=np.int64
    )
    selected_mask = np.zeros(len(RELATION_TYPES), dtype=np.bool_)
    selected_mask[selected_indices] = True
    reliable_relations = np.array(arrays["strict_relations"], copy=True)
    reliable_relations[:, ~selected_mask] = -1
    reliable_matches = np.array(arrays["early_relation_match_counts"], copy=True)
    reliable_matches[:, ~selected_mask] = 0
    catalog = StrictRelationCatalog(
        poi_ids=arrays["collision_poi_ids"],
        bucket_keys=arrays["base_sid_keys"],
        relations=reliable_relations,
        conflict_masks=arrays["conflict_masks"],
        input_poi_count=int(m2a_metrics["catalog"]["input_poi_count"]),
        colliding_poi_count=len(arrays["collision_poi_ids"]),
    )
    sft_manifest = _load_json(sft_manifest_path, "SFT manifest")
    if sft_manifest.get("schema_version") != "sft-main-data-v1" or sft_manifest.get(
        "status"
    ) != "completed":
        raise QgrSidReliableProxyError("SFT manifest 非正式 completed")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}"
    if staging_dir.exists():
        raise QgrSidReliableProxyError("临时输出目录已存在")
    staging_dir.mkdir()

    if progress is not None:
        progress(
            "可靠词法关系已冻结：" + ", ".join(selected_types)
        )
        progress("按 M2-A 同一切分重扫订单并验证 early/holdout 统计")
    (
        early_orders,
        recomputed_matches,
        holdout_events,
        order_metrics,
        order_sources,
    ) = _read_order_events(
        order_dir=order_dir,
        sft_manifest=sft_manifest,
        split=split,
        catalog=catalog,
        progress=progress,
    )
    if not np.array_equal(early_orders, arrays["early_order_counts"]):
        raise QgrSidReliableProxyError("M2-D early POI 请求计数与 M2-A 不一致")
    if not np.array_equal(recomputed_matches, reliable_matches):
        raise QgrSidReliableProxyError("M2-D early 关系命中与 M2-A 过滤结果不一致")
    predictability = np.zeros(len(RELATION_TYPES), dtype=np.float64)
    for index, relation_type in enumerate(RELATION_TYPES):
        if selected_mask[index]:
            predictability[index] = float(
                order_metrics["per_relation"][relation_type][
                    "early_value_aware_match_ratio"
                ]
            )
    p99_order_count = float(
        m2a_metrics["early"]["p99_positive_poi_order_count"]
    )
    if progress is not None:
        progress("编译唯一一套预注册可靠词法关系树")
    (
        path_types,
        path_values,
        resolved,
        compilation_metrics,
        bucket_order,
        bucket_starts,
    ) = _compile_all_buckets(
        catalog=catalog,
        early_orders=early_orders,
        early_matches=recomputed_matches,
        global_predictability=predictability,
        query_guided=True,
        max_pairs=max_pairs,
        p99_order_count=p99_order_count,
        prior_orders=prior_orders,
        progress=progress,
    )
    if progress is not None:
        progress("统一评测 full lexical 与 reliability-gated lexical")
    holdout_raw, cases = _evaluate_holdout(
        holdout_events=holdout_events,
        catalog=catalog,
        early_orders=early_orders,
        bucket_order=bucket_order,
        bucket_starts=bucket_starts,
        static_types=arrays["query_guided_path_types"],
        static_values=arrays["query_guided_path_values"],
        static_resolved=arrays["query_guided_resolved"],
        guided_types=path_types,
        guided_values=path_values,
        guided_resolved=resolved,
        examples_per_kind=examples_per_kind,
        progress=progress,
    )
    ranking = {
        "popularity": holdout_raw["ranking"]["popularity"],
        "full_query_guided_lexical": holdout_raw["ranking"]["static"],
        "reliable_lexical": holdout_raw["ranking"]["query_guided"],
    }
    for new_name, old_name in (
        ("popularity", "popularity"),
        ("full_query_guided_lexical", "query_guided"),
    ):
        expected = m2a_metrics["holdout"]["ranking"][old_name]
        actual = ranking[new_name]
        if any(
            abs(float(actual[key]) - float(expected[key])) > 1e-12
            for key in ("hr_at_1", "hr_at_10", "ndcg_at_10", "mean_rank")
        ):
            raise QgrSidReliableProxyError("M2-D 未精确复现 M2-A 基线")
    path_present_orders = sum(
        bool(np.any(path_types[event.collision_index] >= 0))
        for event in holdout_events
    )
    exact_orders = int(
        holdout_raw["query_guided"]["exact_lexical_path_order_count"]
    )
    reliable = ranking["reliable_lexical"]
    full = ranking["full_query_guided_lexical"]
    popularity = ranking["popularity"]
    deltas = {
        "reliable_hr_at_1_vs_popularity": reliable["hr_at_1"]
        - popularity["hr_at_1"],
        "reliable_ndcg_at_10_vs_popularity": reliable["ndcg_at_10"]
        - popularity["ndcg_at_10"],
        "reliable_hr_at_1_vs_full_lexical": reliable["hr_at_1"]
        - full["hr_at_1"],
        "reliable_ndcg_at_10_vs_full_lexical": reliable["ndcg_at_10"]
        - full["ndcg_at_10"],
    }
    selected_relation_metrics = {
        name: {
            "early_support_order_count": int(
                m2a_metrics["early"]["per_relation"][name][
                    "early_support_order_count"
                ]
            ),
            "early_value_aware_match_ratio": float(
                m2a_metrics["early"]["per_relation"][name][
                    "early_value_aware_match_ratio"
                ]
            ),
        }
        for name in selected_types
    }
    metrics: dict[str, Any] = {
        "schema_version": RELIABLE_PROXY_METRICS_SCHEMA_VERSION,
        "status": "completed",
        "warning": (
            "Known-gold-bucket reliability gate only; unresolved targets are "
            "assumed to retain TIGER collision fallback in the future mapping."
        ),
        "temporal_split": asdict(split),
        "gate": {
            "min_early_support_order_count": min_support_orders,
            "min_early_value_aware_match_ratio": min_value_match_ratio,
            "selected_relation_types": list(selected_types),
            "selected_relation_metrics": selected_relation_metrics,
            "selection_uses_holdout": False,
        },
        "orders": order_metrics,
        "compilation": compilation_metrics,
        "holdout": {
            "collision_holdout_order_count": len(holdout_events),
            "ranking": ranking,
            "full_lexical": holdout_raw["static"],
            "reliable_lexical": {
                **holdout_raw["query_guided"],
                "path_present_order_count": path_present_orders,
                "path_present_order_ratio": _ratio(
                    path_present_orders, len(holdout_events)
                ),
                "exact_path_given_present_ratio": _ratio(
                    exact_orders, path_present_orders
                ),
            },
        },
        "fallback": {
            "type": "retain original TIGER collision code for non-unique leaves",
            "early_order_count": int(early_orders[~resolved].sum()),
            "early_order_ratio": _ratio(
                int(early_orders[~resolved].sum()), int(early_orders.sum())
            ),
            "poi_count": int((~resolved).sum()),
            "poi_ratio": _ratio(int((~resolved).sum()), len(resolved)),
        },
        "deltas": deltas,
        "decision": {
            "reliable_proxy_positive_vs_popularity": bool(
                deltas["reliable_hr_at_1_vs_popularity"] > 0
                and deltas["reliable_ndcg_at_10_vs_popularity"] > 0
            ),
            "retains_at_least_95_percent_of_full_hr_gain": bool(
                deltas["reliable_hr_at_1_vs_popularity"]
                >= 0.95 * (full["hr_at_1"] - popularity["hr_at_1"])
            ),
            "next_step_rule": (
                "Build the final variable mapping only if reliable relations "
                "retain positive proxy signal with materially better target "
                "expressibility; otherwise stop before SFT."
            ),
        },
    }
    metrics["signature"] = _signature(
        {
            "schema_version": RELIABLE_PROXY_SCHEMA_VERSION,
            "m2a_manifest_sha256": sha256_file(m2a_dir / "manifest.json"),
            "min_support_orders": min_support_orders,
            "min_value_match_ratio": min_value_match_ratio,
            "max_pairs": max_pairs,
            "prior_orders": prior_orders,
        }
    )

    array_contracts = {
        "reliable_path_types": _save_array(
            staging_dir, "reliable_path_types", path_types
        ),
        "reliable_path_values": _save_array(
            staging_dir, "reliable_path_values", path_values
        ),
        "reliable_resolved": _save_array(
            staging_dir, "reliable_resolved", resolved
        ),
    }
    metrics_path = staging_dir / "metrics.json"
    _json_dump(metrics_path, metrics)
    cases_path = staging_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    manifest: dict[str, Any] = {
        "schema_version": RELIABLE_PROXY_SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "signature": metrics["signature"],
        "configuration": {
            "min_early_support_order_count": min_support_orders,
            "min_early_value_aware_match_ratio": min_value_match_ratio,
            "max_relation_pairs": max_pairs,
            "prior_orders": prior_orders,
            "unresolved_fallback": "original TIGER collision code",
            "evaluation_scope": "known gold TIGER collision bucket",
        },
        "inputs": {
            "m2a_manifest": {
                "path": str(m2a_dir / "manifest.json"),
                "sha256": sha256_file(m2a_dir / "manifest.json"),
            },
            "sft_manifest": {
                "path": str(sft_manifest_path),
                "sha256": sha256_file(sft_manifest_path),
            },
            "order_sources": order_sources,
        },
        "outputs": {
            "metrics": {
                "path": metrics_path.name,
                "bytes": metrics_path.stat().st_size,
                "sha256": sha256_file(metrics_path),
            },
            "cases": {
                "path": cases_path.name,
                "bytes": cases_path.stat().st_size,
                "sha256": sha256_file(cases_path),
                "rows": len(cases),
            },
            "arrays": array_contracts,
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "git": _git_state(project_root),
        "validation": {
            "gate_uses_early_only": True,
            "single_pre_registered_gate_no_grid_search": True,
            "m2a_baselines_exactly_reproduced": True,
            "early_counts_and_matches_equal_m2a_filtered_arrays": True,
            "holdout_not_used_for_compilation": True,
            "max_relation_pairs_respected": bool(
                np.all((path_types >= 0).sum(axis=1) <= max_pairs)
            ),
        },
    }
    _json_dump(staging_dir / "manifest.json", manifest)
    (staging_dir / "_SUCCESS").touch()
    staging_dir.replace(output_dir)
    return ReliableProxyResult(metrics=metrics, manifest=manifest, output_dir=output_dir)


def validate_reliable_proxy_output(output_dir: Path) -> dict[str, Any]:
    """Validate immutable M2-D artifacts."""

    output_dir = output_dir.resolve()
    manifest = _load_json(output_dir / "manifest.json", "M2-D manifest")
    if manifest.get("schema_version") != RELIABLE_PROXY_SCHEMA_VERSION:
        raise QgrSidReliableProxyError("M2-D schema_version 不受支持")
    if manifest.get("status") != "completed" or not (
        output_dir / "_SUCCESS"
    ).is_file():
        raise QgrSidReliableProxyError("M2-D 输出未 completed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise QgrSidReliableProxyError("M2-D 缺少 outputs")
    contracts: dict[str, Any] = {}
    for name in ("metrics", "cases"):
        if not isinstance(outputs.get(name), dict):
            raise QgrSidReliableProxyError(f"M2-D 缺少 {name}")
        contracts[name] = outputs[name]
    arrays = outputs.get("arrays")
    if not isinstance(arrays, dict):
        raise QgrSidReliableProxyError("M2-D 缺少 arrays")
    contracts.update(arrays)
    checked: dict[str, str] = {}
    for name, item in contracts.items():
        path = output_dir / str(item.get("path", ""))
        if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
            raise QgrSidReliableProxyError(f"M2-D 输出文件/字节数非法：{name}")
        digest = sha256_file(path)
        if digest != item.get("sha256"):
            raise QgrSidReliableProxyError(f"M2-D 输出 SHA256 不一致：{name}")
        if name in arrays:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != item.get("shape") or str(
                array.dtype
            ) != item.get("dtype"):
                raise QgrSidReliableProxyError(f"M2-D 数组契约不一致：{name}")
        checked[name] = digest
    return {
        "status": "validated",
        "schema_version": manifest["schema_version"],
        "signature": manifest["signature"],
        "checked_output_count": len(checked),
        "checked_outputs": checked,
    }

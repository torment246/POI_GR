"""Exact A0-to-A4 prefix membership changes and outcome-stratified examples."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.category_region_visualization import (
    _load_inputs,
    _local_path,
    _sources,
    load_protocol,
)
from qg_prqk.sid.geo import encode_geohash_tokens, geohash_strings


SCHEMA = "qg-prqk-sid-migration-v1"
OUTCOMES = {1: "improved", 0: "tied", -1: "degraded", -2: "not_comparable"}


def prefix_keys(sid: np.ndarray, depth: int) -> np.ndarray:
    """Encode validated prefix tokens without treating their numeric distance as geometry."""
    if (
        sid.ndim != 2
        or sid.shape[1] != 3
        or sid.dtype.kind not in "iu"
        or not len(sid)
        or depth not in (1, 2, 3)
        or np.any(sid < 0)
        or np.any(sid >= 512)
    ):
        raise ValueError("SID 必须是非空三列 [0,512) 整数，depth 为 1/2/3")
    keys = np.zeros(len(sid), dtype=np.int64)
    for column in range(depth):
        keys = keys * 512 + sid[:, column]
    return keys


def _same_label_counts(groups: np.ndarray, labels: np.ndarray) -> np.ndarray:
    _, label_ids = np.unique(labels, return_inverse=True)
    keys = groups.astype(np.int64) * (int(label_ids.max()) + 1) + label_ids
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return counts[inverse]


def compare_peer_labels(
    before_same: np.ndarray,
    after_same: np.ndarray,
    before_size: np.ndarray,
    after_size: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compare same-label fractions among other POIs; singleton sides are undefined."""
    before_peers, after_peers = before_size - 1, after_size - 1
    comparable = (before_peers > 0) & (after_peers > 0)
    before_rate = np.divide(
        before_same - 1,
        before_peers,
        out=np.full(len(before_size), np.nan),
        where=before_peers > 0,
    )
    after_rate = np.divide(
        after_same - 1,
        after_peers,
        out=np.full(len(after_size), np.nan),
        where=after_peers > 0,
    )
    # Exact integer comparison keeps rational ties independent of float rounding.
    numerator = (after_same - 1) * before_peers - (before_same - 1) * after_peers
    outcome = np.full(len(before_size), -2, dtype=np.int8)
    outcome[comparable] = np.sign(numerator[comparable]).astype(np.int8)
    return {"before_rate": before_rate, "after_rate": after_rate, "outcome": outcome}


def compare_prefix_memberships(
    before: np.ndarray,
    after: np.ndarray,
    depth: int,
    categories: np.ndarray,
    regions: np.ndarray,
) -> dict[str, Any]:
    """Count exact intersections, coding changes, peer changes and label outcomes for every POI."""
    if not (len(before) == len(after) == len(categories) == len(regions)):
        raise ValueError("前后 SID、类别、区域的 POI 行数必须一致")
    before_keys, after_keys = prefix_keys(before, depth), prefix_keys(after, depth)
    _, old_groups, old_counts = np.unique(
        before_keys, return_inverse=True, return_counts=True
    )
    _, new_groups, new_counts = np.unique(
        after_keys, return_inverse=True, return_counts=True
    )
    joint = old_groups.astype(np.int64) * len(new_counts) + new_groups
    _, joint_groups, joint_counts = np.unique(
        joint, return_inverse=True, return_counts=True
    )
    before_size, after_size = old_counts[old_groups], new_counts[new_groups]
    common = joint_counts[joint_groups]
    changed = (common != before_size) | (common != after_size)
    peer_union = before_size + after_size - common - 1
    peer_jaccard = np.divide(
        common - 1, peer_union, out=np.ones(len(before)), where=peer_union > 0
    )
    result: dict[str, Any] = {
        "before_size": before_size,
        "after_size": after_size,
        "common_size": common,
        "prefix_changed": before_keys != after_keys,
        "membership_changed": changed,
        "peer_jaccard": peer_jaccard,
    }
    for name, labels in (("category", categories), ("region", regions)):
        result[name] = compare_peer_labels(
            _same_label_counts(old_groups, labels),
            _same_label_counts(new_groups, labels),
            before_size,
            after_size,
        )
    return result


def summarize_migration(comparison: dict[str, Any], depth: int) -> dict[str, Any]:
    """Separate numeric recoding, changed peers and singleton transitions; use POI-weighted counts."""
    old, new = comparison["before_size"], comparison["after_size"]
    prefix_changed, members_changed = (
        comparison["prefix_changed"],
        comparison["membership_changed"],
    )
    summary: dict[str, Any] = {
        "depth": depth,
        "poi_rows": len(old),
        "prefix_changed_count": int(prefix_changed.sum()),
        "membership_changed_count": int(members_changed.sum()),
        "prefix_same_members_changed_count": int(
            (~prefix_changed & members_changed).sum()
        ),
        "prefix_changed_members_same_count": int(
            (prefix_changed & ~members_changed).sum()
        ),
        "mean_peer_jaccard": float(comparison["peer_jaccard"].mean()),
        "singleton_transitions": {
            "singleton_to_singleton": int(((old == 1) & (new == 1)).sum()),
            "singleton_to_collision": int(((old == 1) & (new > 1)).sum()),
            "collision_to_singleton": int(((old > 1) & (new == 1)).sum()),
            "collision_to_collision": int(((old > 1) & (new > 1)).sum()),
        },
    }
    for name in ("category", "region"):
        values = comparison[name]
        valid = values["outcome"] != -2
        summary[name] = {
            "counts": {
                label: int((values["outcome"] == code).sum())
                for code, label in OUTCOMES.items()
            },
            "both_non_singleton_poi": int(valid.sum()),
            "before_mean_on_paired_non_singleton": float(
                values["before_rate"][valid].mean()
            )
            if valid.any()
            else None,
            "after_mean_on_paired_non_singleton": float(
                values["after_rate"][valid].mean()
            )
            if valid.any()
            else None,
        }
    return summary


def select_migration_rows(
    comparison: dict[str, Any], depth: int, seed: int
) -> list[dict[str, Any]]:
    """Sample one row per category-outcome stratum, without ranking by improvement magnitude."""
    eligible = comparison["prefix_changed"] & comparison["membership_changed"]
    if depth > 1:
        eligible &= (comparison["before_size"] <= 20) & (comparison["after_size"] <= 20)
    selected = []
    rng = np.random.default_rng(seed + depth)
    for code in (1, 0, -1):
        candidates = np.flatnonzero(
            eligible & (comparison["category"]["outcome"] == code)
        )
        selected.append(
            {
                "outcome": OUTCOMES[code],
                "eligible_poi": len(candidates),
                "anchor_row": int(rng.choice(candidates)) if len(candidates) else None,
            }
        )
    return selected


def describe_example(
    table: pa.Table,
    before: np.ndarray,
    after: np.ndarray,
    depth: int,
    row: int,
    comparison: dict[str, Any],
    regions: np.ndarray,
    case_id: str,
    preview_limit: int = 5,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Trace an anchor's old/new buckets, retaining complete row/ID sets and bounded previews."""
    old_rows = np.flatnonzero(np.all(before[:, :depth] == before[row, :depth], axis=1))
    new_rows = np.flatnonzero(np.all(after[:, :depth] == after[row, :depth], axis=1))
    groups = {
        "retained": np.intersect1d(old_rows, new_rows),
        "removed": np.setdiff1d(old_rows, new_rows),
        "added": np.setdiff1d(new_rows, old_rows),
    }
    example = {
        "case_id": case_id,
        "depth": depth,
        "anchor": table.slice(row, 1).to_pylist()[0],
        "before_prefix": before[row, :depth].tolist(),
        "after_prefix": after[row, :depth].tolist(),
        "before_size": len(old_rows),
        "after_size": len(new_rows),
        "groups": {},
    }
    arrays = {}
    for name, rows in groups.items():
        arrays[f"{case_id}_{name}_rows"] = rows
        arrays[f"{case_id}_{name}_poi_ids"] = np.asarray(
            table.take(pa.array(rows))["poi_id"].to_pylist(), dtype=str
        )
        records = table.take(pa.array(rows[:preview_limit])).to_pylist()
        for record, index in zip(records, rows[:preview_limit]):
            record["old_prefix"] = before[index, :depth].tolist()
            record["new_prefix"] = after[index, :depth].tolist()
            record["geohash5"] = str(regions[index])
        example["groups"][name] = {
            "count": len(rows),
            "preview": records,
            "preview_truncated": len(rows) > preview_limit,
        }
    assert len(groups["retained"]) + len(groups["removed"]) == len(old_rows)
    assert len(groups["retained"]) + len(groups["added"]) == len(new_rows)
    for name in ("category", "region"):
        values = comparison[name]
        example[name] = {
            "before": float(values["before_rate"][row]),
            "after": float(values["after_rate"][row]),
            "outcome": OUTCOMES[int(values["outcome"][row])],
        }
    return example, arrays


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build the read-only migration view from frozen inputs or verify its existing publication."""
    started = time.monotonic()
    settings, config = load_protocol(args.config)
    output = _local_path(config.project_root, str(args.output_dir), outputs_only=True)
    parameters = {
        "seed": args.seed,
        "case_depth2_3_max_bucket_size": 20,
        "preview_limit_per_membership_group": 5,
        "case_selection": "prefix_and_membership_changed_both_non_singleton_uniform_by_category_outcome",
        "category": "coarse_at_s1_full_category_path_at_s1s2_and_s1s2s3",
        "region": "geohash5_not_administrative_region",
        "score": "same_anchor_label_fraction_of_other_bucket_members_excluding_self",
    }
    sources = _sources(settings, config, args.config)
    for relative in (
        "src/qg_prqk/sid/migration_analysis.py",
        "src/qg_prqk/sid/migration_plots.py",
        "scripts/analyze_sid_migration.py",
    ):
        path = config.project_root / "qg_prqk" / relative
        sources[str(path)] = sha256_file(path)
    if args.validate_only:
        manifest = json.loads((output / "manifest.json").read_text())
        marker = json.loads((output / "_SUCCESS").read_text())
        if (
            marker["manifest_sha256"] != sha256_file(output / "manifest.json")
            or manifest["sources"] != sources
            or manifest["parameters"] != parameters
        ):
            raise ValueError("迁移视图的来源、参数或 manifest 哈希已变化")
        for name, digest in manifest["artifacts"].items():
            if Path(name).name != name or sha256_file(output / name) != digest:
                raise ValueError(f"迁移输出 SHA256 不符：{name}")
        return {"status": "validated", "output_dir": str(output), **marker}
    if output.exists():
        raise ValueError(f"输出已存在，拒绝覆盖：{output}")
    print("读取冻结输入，逐层精确统计编码、同伴集合与类别/区域变化……", flush=True)
    table, sids, _ = _load_inputs(config)
    fine = np.asarray(table["category"].to_pylist())
    coarse = np.asarray([value.split(":", 1)[0] for value in fine])
    tokens = encode_geohash_tokens(
        table["lng"].to_numpy(), table["lat"].to_numpy(), length=5
    )
    regions = np.asarray(list(geohash_strings(tokens)))
    summaries, examples, memberships = {}, {}, {}
    for index, method in enumerate(config.methods[1:], start=1):
        summaries[method.name], examples[method.name] = [], []
        for depth in (1, 2, 3):
            comparison = compare_prefix_memberships(
                sids[0], sids[index], depth, coarse if depth == 1 else fine, regions
            )
            summary = summarize_migration(comparison, depth)
            summary["sample_selection"] = select_migration_rows(
                comparison, depth, args.seed
            )
            for selection in summary["sample_selection"]:
                if selection["anchor_row"] is None:
                    continue
                case_id = f"{method.name}_d{depth}_{selection['outcome']}"
                example, arrays = describe_example(
                    table,
                    sids[0],
                    sids[index],
                    depth,
                    selection["anchor_row"],
                    comparison,
                    regions,
                    case_id,
                )
                example["eligible_poi"] = selection["eligible_poi"]
                examples[method.name].append(example)
                memberships.update(arrays)
            summaries[method.name].append(summary)
            print(
                f"{method.name} S1..S{depth}：换前缀 {summary['prefix_changed_count']:,}，换同伴集合 {summary['membership_changed_count']:,}",
                flush=True,
            )
            del comparison
    output.mkdir(parents=True, exist_ok=False)
    data = {
        "schema_version": SCHEMA,
        "parameters": parameters,
        "poi_rows": table.num_rows,
        "scope": "whole_active_catalog_three_token_sid_without_gid_or_dedup_no_business_splits",
        "methods": summaries,
    }
    write_json_atomic(output / "metrics.json", data)
    write_json_atomic(
        output / "examples.json", {"methods": examples, "parameters": parameters}
    )
    np.savez_compressed(output / "memberships.npz", **memberships)
    from qg_prqk.sid.migration_plots import (
        draw_migration_figures,
        migration_examples_html,
    )

    draw_migration_figures(settings, config.project_root, data, examples, output)
    (output / "migration_examples.html").write_text(
        migration_examples_html(examples), encoding="utf-8"
    )
    if any(sha256_file(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError("运行期间输入/代码变化，不发布成功标记")
    names = [
        "metrics.json",
        "examples.json",
        "memberships.npz",
        "migration_examples.html",
    ]
    for stem in (
        "sid_migration_overview",
        "sid_migration_gid_examples",
        "sid_migration_nogid_examples",
    ):
        names.extend([f"{stem}.png", f"{stem}.pdf"])
    write_json_atomic(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "completed",
            "built_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "parameters": parameters,
            "sources": sources,
            "artifacts": {name: sha256_file(output / name) for name in names},
            "source_access": {
                "business_validation_read": False,
                "business_test_read": False,
                "tsne_recomputed": False,
                "sid_or_model_updated": False,
            },
        },
    )
    write_json_atomic(
        output / "_SUCCESS", {"manifest_sha256": sha256_file(output / "manifest.json")}
    )
    return {
        "status": "completed",
        "output_dir": str(output),
        "poi_rows": table.num_rows,
        "examples": sum(len(cases) for cases in examples.values()),
        "elapsed_seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A0→两种 A4 的全库前缀/同伴变化及提高、持平、降低案例；只读，不重训"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qg_prqk/configs/qg_prqk_sid_category_region_v2.yaml"),
        help="冻结来源配置",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qg_prqk/outputs/figures/qg_prqk_sid_migration_v1"),
        help="独立输出目录，拒绝覆盖",
    )
    parser.add_argument("--seed", type=int, default=42, help="分组内确定性抽样 seed")
    parser.add_argument(
        "--validate-only", action="store_true", help="只复核既有输出与来源哈希"
    )
    try:
        result = run(parser.parse_args(argv))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"SID 迁移分析失败：{error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

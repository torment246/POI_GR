#!/usr/bin/env python3
"""Audit paired SFT inputs and frozen S1 content-versus-graph assignments."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "qg_prqk/src"))

import numpy as np
import yaml

from qg_prqk.artifacts import sha256_file, write_json_atomic
from qg_prqk.sid.evaluation import _agreement, _map_edges
from qg_prqk.sid.evaluation_config import load_sid_evaluation_config
from qg_prqk.sid.evaluation_data import load_sid_evaluation_inputs
from qg_prqk.sft.a0_gid_data import HISTORY_ID, checked_artifact, remap_record
from qg_prqk.sft.data import load_final_identifier_lookup
from qg_prqk.sid.identifiers import identifier_content, identifier_key

SUBSETS = ("fixed10k", "seen_query_unseen_pair", "unseen_query_seen_target", "long_tail_target", "cold_target")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def stable_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Rank target codes with the production lowest-token-ID tie break."""
    selected = scores[np.arange(len(scores)), targets, None]
    return 1 + (scores > selected).sum(axis=1) + (
        (scores == selected) & (np.arange(scores.shape[1])[None, :] < targets[:, None])
    ).sum(axis=1)


def request_without_ids(record: dict) -> tuple:
    messages = record["messages"]
    assert record["split"] == "valid" and [m["role"] for m in messages] == ["user", "assistant"]
    masked, count = HISTORY_ID.subn("<HISTORY_ID>", messages[0]["content"])
    assert count == record["history_length"]
    return (record["order_id"], record["searchid"], record["target_poi_id"], count, masked)


def audit_sft() -> dict:
    """Compare every frozen evaluation request and training hyperparameter."""
    paths = [ROOT / "qg_prqk/outputs/eval/a0_gid_epoch3_dual_decode_v1/plan.json",
             ROOT / "qg_prqk/outputs/eval/sft_epoch3_fixed10k_generalization_v1/a4_gid_parent/plan_2x6000d.json"]
    plans = [read_json(path) for path in paths]
    assert plans[0]["config"]["decoding"] == plans[1]["config"]["decoding"]
    assert plans[0]["checkpoint"]["token_mapping_sha256"] == plans[1]["checkpoint"]["token_mapping_sha256"]
    configs = [yaml.safe_load((ROOT / "qg_prqk/configs/sft" / name).read_text()) for name in
               ("a0_gid_history10_v1.yaml", "a4_gid_parent_history10_v1.yaml")]
    allowed = {"variant", "dataset", "eval_dataset", "dataset_dir", "tokenized_path", "output_dir", "logging_dir"}
    differences = {key: [c.get(key) for c in configs] for key in set(configs[0]) | set(configs[1])
                   if configs[0].get(key) != configs[1].get(key)}
    assert not set(differences) - allowed, differences
    old_spec = plans[1]["config"]["variants"]["a4_gid_parent"]
    new_spec = plans[0]["config"]["variants"]["a0_gid"]
    old = load_final_identifier_lookup(ROOT / old_spec["identifier_dir"], "a4_gid_parent")
    directory = ROOT / new_spec["identifier_dir"]
    manifest = read_json(directory / "manifest.json")
    assert sha256_file(directory / "manifest.json") == new_spec["identifier_manifest_sha256"]
    base = np.load(checked_artifact(directory, manifest, "base_identifier_codes.npy"), mmap_mode="r")
    dedup = np.load(checked_artifact(directory, manifest, "dedup_codes.npy"), mmap_mode="r")
    assert base.shape == old.base_codes.shape and dedup.shape == old.dedup_codes.shape
    old_rows = {old.content(row): row for row in range(len(old.poi_ids))}
    row_by_poi = {poi: row for row, poi in enumerate(old.poi_ids)}

    @lru_cache(maxsize=716245)
    def content(row: int) -> str:
        return identifier_content(base[row], int(dedup[row]), variant="a4_gid_parent")

    def key(row: int) -> str:
        return identifier_key(base[row], int(dedup[row]), variant="a4_gid_parent")

    requires_dedup = dedup >= 0
    result = {"plan_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
              "training_config_differences": differences, "subsets": {}}
    for subset in SUBSETS:
        specs = [plan["data"]["outputs"][subset] for plan in plans]
        hashes = [hashlib.sha256(), hashlib.sha256()]
        count = histories = 0
        with Path(specs[0]["file"]).open("rb") as left, Path(specs[1]["file"]).open("rb") as right:
            while True:
                lines = [left.readline(), right.readline()]
                if not any(lines):
                    break
                assert all(lines), "配对文件长度不同"
                records = [json.loads(line) for line in lines]
                assert request_without_ids(records[0]) == request_without_ids(records[1]), (subset, count)
                expected = remap_record(records[1], split="valid", old_row_by_content=old_rows,
                                        row_by_poi_id=row_by_poi, new_content=content, new_key=key,
                                        requires_dedup=requires_dedup)
                assert expected["messages"] == records[0]["messages"], (subset, count, "标识回映射错误")
                assert expected["requires_dedup"] == records[0]["requires_dedup"]
                for digest, line in zip(hashes, lines):
                    digest.update(line)
                count += 1
                histories += records[0]["history_length"]
        assert count == specs[0]["rows"] == specs[1]["rows"] == 10000
        assert all(d.hexdigest() == s["sha256"] for d, s in zip(hashes, specs))
        result["subsets"][subset] = dict(rows=count, history_events=histories, non_identifier_mismatches=0,
                                         history_and_target_identifier_mismatches=0,
                                         complete_source_hashes_verified=True)
        print(f"[配对核验] {subset}: {count} 条一致", flush=True)
    return result


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def diagnose_s1(chunk_rows: int) -> dict:
    """Recompute full S1 ranks in bounded CPU chunks, without retraining."""
    config = load_sid_evaluation_config(ROOT / "qg_prqk/configs/qg_prqk_p8_a0_vs_a4_static_full_v1.yaml")
    inputs = load_sid_evaluation_inputs(config, gate="full")
    # The loader checks manifests and headers; bind all actual arrays consumed here.
    verified = {}
    for phase, names in {
        "p5_manifest": ("poi_codebook_s1.npy", "poi_assignments_s1_s2_s3.npy"),
        "p6_manifest": ("config_resolved.json", "query_residual_s0.npy", "query_codebook_s1.npy", "query_nodes.parquet", "query_poi_edges_s1_s2.parquet"),
        "p7_manifest": ("selected_poi_rows.npy", "poi_sid_s1_s2_s3.npy", "query_sid_s1_s2_s3.npy"),
    }.items():
        path = inputs.source_paths[phase]
        manifest = read_json(path)
        for name in names:
            digest = sha256_file(path.parent / name)
            assert digest == manifest["artifacts"][name]["sha256"], name
            verified[str((path.parent / name).relative_to(ROOT))] = digest
    edge = _map_edges(inputs, inputs.edges_s1_s2, 1)
    n = len(inputs.query_residual_s0)
    order = np.argsort(edge.query_rows, kind="stable")
    qr, pr, weights, depths = (getattr(edge, field)[order] for field in ("query_rows", "poi_rows", "weights", "depths"))
    methods = {}
    settings = read_json(inputs.source_paths["p6_manifest"].parent / "config_resolved.json")["s1"]
    for name, method in (("A0", inputs.a0), ("A4", inputs.a4)):
        centroids = normalize(method.query_codebooks[0])
        ranks = np.zeros(len(qr), dtype=np.int32)
        coupled_ranks = np.zeros(n, dtype=np.int32)
        gaps = np.zeros(n, dtype=np.float32)
        recomputed = np.zeros(n, dtype=np.int32)
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            scores = normalize(inputs.query_residual_s0[start:stop]) @ centroids.T
            lo, hi = np.searchsorted(qr, [start, stop])
            targets = method.poi_sid[pr[lo:hi], 0]
            ranks[lo:hi] = stable_ranks(scores[qr[lo:hi] - start], targets)
            if method.query_sid is not None:
                labels = method.query_sid[start:stop, 0]
                coupled_ranks[start:stop] = stable_ranks(scores, labels)
                gaps[start:stop] = scores.max(axis=1) - scores[np.arange(stop-start), labels]
                agreement = np.zeros_like(scores)
                np.add.at(agreement, (qr[lo:hi] - start, targets), weights[lo:hi].astype(np.float32))
                objective = settings["query_distortion_weight"] * scores + settings["graph_alignment_weight"] * agreement
                recomputed[start:stop] = np.argmax(objective, axis=1)
            if start % (chunk_rows * 16) == 0:
                print(f"[S1/{name}] {stop:,}/{n:,}", flush=True)
        payload = {"edge_rows": len(qr), "query_rows": n,
                   "content_target_topk": {str(k): _agreement(ranks <= k, weights) for k in (1, 5, 10, 50)},
                   "by_depth": {str(d): {str(k): _agreement(ranks[depths == d] <= k, weights[depths == d])
                                         for k in (1, 5, 10, 50)} for d in (1, 2, 3)}}
        if method.query_sid is not None:
            coupled_hit = method.query_sid[qr, 0] == method.poi_sid[pr, 0]
            content_hit = ranks == 1
            payload.update(
                coupled_graph_accuracy=_agreement(coupled_hit, weights),
                assigned_code_content_topk={str(k): float(np.mean(coupled_ranks <= k)) for k in (1, 5, 10, 50)},
                assigned_cosine_gap_mean=float(gaps.mean()),
                assigned_cosine_gap_quantiles=np.quantile(gaps, [.5, .9, .99]).tolist(),
                frozen_objective_reproduction_match=float(np.mean(recomputed == method.query_sid[:, 0])),
                frozen_objective_settings=settings,
                joint_edge_mass={f"coupled_{int(a)}_content_{int(b)}": _agreement((coupled_hit == a) & (content_hit == b), weights)
                                 for a in (False, True) for b in (False, True)},
            )
        methods[name] = payload
    reference = read_json(ROOT / "qg_prqk/outputs/qg_prqk_512x3_v2_1_cat_active/p8_static_evaluation_a0_vs_a4_v1/full_342879q_716245p/prefix_probe_metrics.json")
    for name in methods:
        previous = reference["methods"][name]["teacher_forced"]["query_to_s1"]["overall"]["weighted_accuracy"]
        assert abs(methods[name]["content_target_topk"]["1"]["weighted_accuracy"] - previous) < 1e-4
    return {"methods": methods, "verified_inputs": verified, "scope": "full_train_derived_s1_static_not_qwen_inference"}


def main() -> None:
    parser = argparse.ArgumentParser(description="核验 A0/A4 配对数据和全量 S1 内容/图分配差距；不训练、不读取 Test。")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    args = parser.parse_args()
    if not 1 <= args.chunk_rows <= 8192:
        parser.error("chunk-rows 必须在 1—8192")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "qg_prqk/outputs") or output.exists():
        parser.error("输出必须是 qg_prqk/outputs 下尚不存在的 JSON")
    result = dict(schema_version="qg-a0-a4-causal-audit-v1", status="completed", sft_pairing=audit_sft(),
                  s1=diagnose_s1(args.chunk_rows), source_sha256=sha256_file(Path(__file__)))
    write_json_atomic(output, result)
    print(f"核验完成：{output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"核验失败：{error}", file=sys.stderr)
        raise SystemExit(2)

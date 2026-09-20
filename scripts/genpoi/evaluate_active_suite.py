#!/usr/bin/env python3
"""Coordinate the five frozen active GenPOI evaluation subsets."""
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.pid.prepare_vocab import TOKENIZER_FILES, sha256_file, sha256_named_files

VERSION = "genpoi_active716k_centered_geope32_512x3_history10_query_gid_v1"
TRAIN_CONFIG = ROOT / "configs/sft" / f"{VERSION}.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/eval" / f"{VERSION}_gpu4_6000d_e3"
SUITE = ROOT / "outputs/eval/generalization_validation_suite_10k_v1/suite_manifest.json"
PID_DIR = ROOT / "outputs/pid/genpoi/GenPOI-ACTIVE716K-CenteredGeoPE32-512x3-e20-G6-Dedup"
HEAD_DIR = ROOT / "outputs/genpoi/proximity_head_bge_m3_v1"
GENERALIZATION = ("seen_query_unseen_pair", "unseen_query_seen_target", "long_tail_target", "cold_target")
SUBSETS = (*GENERALIZATION, "fixed10k")
METRICS = tuple(f"{name}@{k}" for name in ("hr", "ndcg") for k in (1, 3, 5, 10))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(output: Path) -> dict[str, Any]:
    """Validate small manifests and frozen references without loading models."""
    training = yaml.safe_load(TRAIN_CONFIG.read_text())
    if training["cutoff_len"] != 1024:
        raise ValueError("active GenPOI 训练长度必须为 1024")
    checkpoint = ROOT / training["output_dir"] / "checkpoint-8868"
    tokenizer = ROOT / training["model_name_or_path"]
    state = read_json(checkpoint / "trainer_state.json")
    if state.get("global_step") != 8868 or state.get("epoch") != 3.0:
        raise ValueError("要求已完成 epoch 3 的 checkpoint-8868")
    cache = read_json(ROOT / training["tokenized_path"] / "cache_manifest.json")
    if cache["inputs"]["cutoff_len"] != 1024:
        raise ValueError("训练 Cache 长度不一致")
    suite = read_json(SUITE)
    if suite["status"] != "completed":
        raise ValueError("冻结泛化集未完成")
    references = {name: {"path": Path(suite["subsets"][name]["output_file"]),
                         "sha256": suite["subsets"][name]["output_sha256"]}
                  for name in GENERALIZATION}
    fixed = suite["references"]["random_traffic"]
    references["fixed10k"] = {"path": Path(fixed["data_file"]), "sha256": fixed["data_sha256"]}
    for name, spec in references.items():
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise ValueError(f"冻结参考集发生变化：{name}")
        with spec["path"].open("rb") as handle:
            if sum(1 for _ in handle) != 10000:
                raise ValueError(f"参考集必须为 10000 行：{name}")
    valid = ROOT / "data/sft" / VERSION / "valid.jsonl"
    data_manifest = read_json(valid.parent / "manifest.json")
    if (data_manifest["status"] != "completed"
            or data_manifest["outputs"]["valid.jsonl"]["sha256"] != cache["inputs"]["valid_sha256"]):
        raise ValueError("评测 Validation 与训练 Cache 来源不一致")
    pipeline = yaml.safe_load((ROOT / "configs/sft/genpoi_active716k_pipeline_v1.yaml").read_text())
    pid_manifest = PID_DIR / "final_pid_manifest.json"
    if sha256_file(pid_manifest) != pipeline["sources"][str(pid_manifest.relative_to(ROOT))]:
        raise ValueError("active PID manifest 与训练冻结版本不一致")
    head = read_json(HEAD_DIR / "training_manifest.json")
    if (head["status"] != "completed" or not head["paper_method"]["backbone_frozen"]
            or head["head_sha256"] != sha256_file(HEAD_DIR / "proximity_head.safetensors")):
        raise ValueError("冻结 SSP head 校验失败")
    if not Path(head["model"]["path"]).is_dir():
        raise ValueError("SSP BGE-M3 模型目录不存在")
    for path in (valid, checkpoint / "model.safetensors", checkpoint / "tokenizer.json",
                 tokenizer / "tokenizer.json", tokenizer / "poi_token_mapping.json",
                 PID_DIR / "final_pid_manifest.json", PID_DIR / "poi_pid_mapping.parquet",
                 HEAD_DIR / "training_manifest.json", HEAD_DIR / "proximity_head.safetensors"):
        if not path.is_file():
            raise ValueError(f"缺少输入：{path}")
    token_hash = sha256_file(tokenizer / "tokenizer.json")
    if (sha256_named_files(tokenizer, TOKENIZER_FILES) != cache["inputs"]["extended_tokenizer_sha256"]
            or token_hash != sha256_file(checkpoint / "tokenizer.json")):
        raise ValueError("checkpoint、扩词模型与训练 Cache 的 Tokenizer 不一致")
    return dict(output=output, checkpoint=checkpoint, tokenizer=tokenizer, valid=valid,
                references=references, trie=output / "trie")


def subset_command(inputs: dict[str, Any], name: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts/genpoi/evaluate_active_retrieval.py"),
            "--valid-file", str(inputs["valid"]),
            "--reference-validation-subset", str(inputs["references"][name]["path"]),
            "--checkpoint", str(inputs["checkpoint"]), "--expected-step", "8868",
            "--tokenizer", str(inputs["tokenizer"]), "--trie-dir", str(inputs["trie"]),
            "--head-dir", str(HEAD_DIR), "--output-dir", str(inputs["output"] / name)]


def prepare_trie(inputs: dict[str, Any]) -> None:
    """Build once or verify the active catalog Trie before GPU workers start."""
    from poi_gr.pid.trie import CompactPidTrie, build_pid_trie
    from poi_gr.sft.evaluation import validate_checkpoints

    validate_checkpoints([inputs["checkpoint"]], tokenizer_path=inputs["tokenizer"],
                         expected_steps=[8868], expected_epochs=[3.0])
    trie_dir = inputs["trie"]
    manifest_path = trie_dir / "trie_manifest.json"
    if not manifest_path.exists():
        if trie_dir.exists() and any(trie_dir.iterdir()):
            raise ValueError(f"Trie 目录不完整，拒绝覆盖：{trie_dir}")
        build_pid_trie(PID_DIR / "poi_pid_mapping.parquet", PID_DIR / "final_pid_manifest.json",
                       inputs["tokenizer"], trie_dir, progress=lambda value: print(value, flush=True))
    manifest = read_json(manifest_path)
    for key, filename in (("pid_mapping", "poi_pid_mapping.parquet"),
                          ("pid_manifest", "final_pid_manifest.json"),
                          ("final_pid_codes", "final_pid_codes.npy")):
        if manifest["input"][f"{key}_sha256"] != sha256_file(PID_DIR / filename):
            raise ValueError(f"Trie 与 active PID 不一致：{key}")
    if manifest["tokenizer"]["tokenizer_json_sha256"] != sha256_file(inputs["tokenizer"] / "tokenizer.json"):
        raise ValueError("Trie 与 active Tokenizer 不一致")
    if manifest["path_definition"]["pid_order"] != "gid_sid" or CompactPidTrie.load(trie_dir).leaf_count != 716245:
        raise ValueError("Trie 必须包含 716245 条 GID-first PID")


def summarize(inputs: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete or mismatched cells and average only generalization sets."""
    rows = []
    signatures = set()
    for name in SUBSETS:
        payload = read_json(inputs["output"] / name / "valid_checkpoint_results.json")
        if payload.get("status") != "completed" or len(payload.get("results", [])) != 1:
            raise ValueError(f"评测未完整完成：{name}")
        result = payload["results"][0]
        config, metrics = result["config"], result["metrics"]
        if result.get("status") != "completed" or result.get("covered_rows") != 10000 or metrics.get("sample_count") != 10000:
            raise ValueError(f"评测不足 10000 条：{name}")
        expected = {"checkpoint": str(inputs["checkpoint"]), "epoch": 3.0,
                    "cutoff_len": 1024, "constraint_mode": "tcg_ssp", "ssp_gamma": 2,
                    "num_beams": 10, "num_return_sequences": 10, "top_k": 10,
                    "smoke_limit": None, "trie_dir": str(inputs["trie"]),
                    "tokenizer": str(inputs["tokenizer"])}
        if any(key not in config or config[key] != value for key, value in expected.items()):
            raise ValueError(f"评测协议不一致：{name}")
        context = config["dataset_context"]
        if context["reference_sha256"] != inputs["references"][name]["sha256"]:
            raise ValueError(f"评测参考集不一致：{name}")
        signatures.add(tuple(config[key] for key in
                             ("checkpoint_model_sha256", "tokenizer_json_sha256", "trie_manifest_sha256", "ssp_head_sha256")))
        row = {"subset": name, "sample_count": 10000, **{key: metrics[key] for key in METRICS},
               "valid_pid_ratio": metrics["generation_validity"]["valid_pid_ratio"]}
        if any(not math.isfinite(row[key]) or not 0 <= row[key] <= 1 for key in (*METRICS, "valid_pid_ratio")):
            raise ValueError(f"指标范围不合法：{name}")
        rows.append(row)
    if len(signatures) != 1:
        raise ValueError("五集使用的模型、Tokenizer、Trie 或 SSP head 不一致")
    macro = {key: sum(row[key] for row in rows if row["subset"] in GENERALIZATION) / 4
             for key in (*METRICS, "valid_pid_ratio")}
    return {"status": "completed", "checkpoint": str(inputs["checkpoint"]),
            "results": rows, "generalization_macro": macro}


def main() -> int:
    parser = argparse.ArgumentParser(description="Active GenPOI epoch 3：固定 10k 与四类泛化 SSP+TCG 评测。")
    parser.add_argument("--stage", choices=("plan", "prepare", "evaluate", "summarize"), default="plan")
    parser.add_argument("--subset", choices=SUBSETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = load_inputs((ROOT / args.output_dir).resolve())
    if args.stage == "plan":
        print("预检通过：epoch 3 / step 8868，四个泛化集并行，随后固定 10k；SSP+TCG / Beam 10 / cutoff 1024。")
        for name in SUBSETS:
            print(shlex.join(subset_command(inputs, name)))
    elif args.stage == "prepare":
        prepare_trie(inputs)
        print("active Trie 准备完成。")
    elif args.stage == "evaluate":
        if args.subset is None:
            parser.error("evaluate 阶段必须提供 --subset")
        subprocess.run(subset_command(inputs, args.subset), cwd=ROOT, check=True)
    else:
        from poi_gr.sft.evaluation import atomic_write_json, write_results_csv

        payload = summarize(inputs)
        atomic_write_json(inputs["output"] / "suite_summary.json", payload)
        write_results_csv(inputs["output"] / "suite_summary.csv", payload["results"] + [
            {"subset": "generalization_macro", "sample_count": 40000, **payload["generalization_macro"]}])
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Active GenPOI 五集评测失败：{error}", file=sys.stderr)
        raise SystemExit(2)

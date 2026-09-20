#!/usr/bin/env python3
"""Run four frozen active models on one paired full Test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "qg_prqk/src")]
from scripts.pid.prepare_vocab import sha256_file

METHODS = ("gnpr", "genpoi", "qg_hrq_gid", "qg_hrq_nogid")
MODES = dict(gnpr="unconstrained", genpoi="tcg_ssp",
             qg_hrq_gid="full_catalog_constrained", qg_hrq_nogid="full_catalog_constrained")


def resolve(path: str | Path) -> Path:
    return (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def load_config(path: Path) -> dict[str, Any]:
    """Check manifests and checkpoint metadata without scanning the Test files."""
    config = yaml.safe_load(path.read_text())
    if (config["schema_version"] != "active-four-model-full-test-v1"
            or set(config["models"]) != set(METHODS) or config["split"] != "test"
            or (config["date"], config["rows"]) != ("2026-07-14", 606682)
            or config["num_beams"] != 10 or config["cutoff_len"] != 1024):
        raise ValueError("必须显式冻结四个 active 模型、Test、Beam=10 和 cutoff=1024")
    if resolve(config["output_root"]) != resolve("outputs/eval/active_four_models_full_test_20260714_v1"):
        raise ValueError("必须使用本协议独立的输出目录")
    for name in METHODS:
        spec = config["models"][name]
        if spec["decoding"] != MODES[name]:
            raise ValueError(f"解码协议错误：{name}")
        state = read_json(resolve(spec["checkpoint"]) / "trainer_state.json")
        if (state.get("epoch"), state.get("global_step"), state.get("max_steps")) != (3.0, spec["step"], spec["step"]):
            raise ValueError(f"不是已完成的指定 epoch-3 checkpoint：{name}")
        manifest = read_json(resolve(spec["data_dir"]) / "manifest.json")
        time_split = manifest.get("time_split", manifest.get("source_sft", {}).get("time_split"))
        if manifest["status"] != "completed" or time_split["test"] != config["date"]:
            raise ValueError(f"Test 日期或数据状态不一致：{name}")
        source = manifest["outputs"]["test.jsonl"]
        if source["rows"] != config["rows"] or source["sha256"] != spec["test_sha256"]:
            raise ValueError(f"Test 全集行数或版本不一致：{name}")
        for filename in ("model.safetensors", "tokenizer.json", "config.json"):
            if not (resolve(spec["checkpoint"]) / filename).is_file():
                raise ValueError(f"缺少模型输入：{name}/{filename}")
        if not (resolve(spec["data_dir"]) / "test.jsonl").is_file():
            raise ValueError(f"缺少 Test：{name}")
        if sha256_file(resolve(spec["tokenizer"]) / "tokenizer.json") != sha256_file(resolve(spec["checkpoint"]) / "tokenizer.json"):
            raise ValueError(f"Checkpoint 和冻结 Tokenizer 不同：{name}")
    for filename, digest in config["frozen_files"].items():
        if sha256_file(resolve(filename)) != digest:
            raise ValueError(f"冻结输入发生变化：{filename}")
    qg_test = yaml.safe_load(resolve(config["qg_test_config"]).read_text())
    qg_protocol = yaml.safe_load(resolve(qg_test["validation_protocol"]).read_text())
    if (qg_test["date"], qg_test["source_rows"]) != (config["date"], config["rows"]):
        raise ValueError("QG 已有全量 Test 与四模型协议的日期/行数不同")
    for name in ("qg_hrq_gid", "qg_hrq_nogid"):
        spec = config["models"][name]
        variant = spec["variant"]
        original = qg_protocol["variants"][variant]
        if (any(resolve(spec[key]) != resolve(original[key]) for key in ("checkpoint", "data_dir", "identifier_dir"))
                or spec["step"] != original["expected_step"]
                or spec["checkpoint_sha256"] != qg_test["checkpoint_sha256"][variant]
                or resolve(spec["tokenizer"]) != resolve(qg_protocol["tokenizer"])):
            raise ValueError(f"QG 模型与原冻结全量 Test 配置不一致：{name}")
    with ExitStack() as stack:
        streams = [stack.enter_context((resolve(config["models"][name]["data_dir"]) / "test.jsonl").open()) for name in METHODS]
        for row in range(32):
            identities = [record_identity(json.loads(stream.readline())) for stream in streams]
            if any(value != identities[0] for value in identities[1:]):
                raise ValueError(f"轻量预检发现 Test 第 {row + 1} 行不对齐")
    return config


def record_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Compare request identity and current input without comparing method IDs."""
    if record.get("split") != "test":
        raise ValueError("全量 Test 包含非 Test 行")
    fields = ("order_id", "searchid", "target_poi_id")
    if any(record.get(key) is None or not str(record[key]).strip() for key in fields):
        raise ValueError("Test 缺少业务键或目标 POI")
    messages = record.get("messages", [])
    if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        raise ValueError("Test 必须为完整 user/assistant Messages")
    current = re.findall(r"<CURRENT>(.*?)</CURRENT>", messages[0]["content"], flags=re.S)
    if len(current) != 1:
        raise ValueError("Test 必须有唯一 CURRENT 块")
    return (*(str(record[key]) for key in fields), current[0], record.get("history_length"))


def scan_aligned_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    """Stream all four JSONL files once; reject mismatches rather than drop rows."""
    digests = {name: hashlib.sha256() for name in METHODS}
    keys_digest = hashlib.sha256()
    seen: set[bytes] = set()
    count = 0
    sources = {name: resolve(config["models"][name]["data_dir"]) / "test.jsonl" for name in METHODS}
    before = {name: file_stat(path) for name, path in sources.items()}
    with ExitStack() as stack:
        streams = {name: stack.enter_context(path.open("rb")) for name, path in sources.items()}
        while True:
            lines = {name: stream.readline() for name, stream in streams.items()}
            if not any(lines.values()):
                break
            if not all(lines.values()) or count >= config["rows"]:
                raise ValueError("四模型 Test 行数不一致或超过冻结全集")
            identities = [record_identity(json.loads(lines[name])) for name in METHODS]
            if any(value != identities[0] for value in identities[1:]):
                raise ValueError(f"Test 第 {count + 1} 行业务键、目标、CURRENT 或历史长度不同")
            key = json.dumps(identities[0][:2], ensure_ascii=False, separators=(",", ":")).encode()
            key_hash = hashlib.sha256(key).digest()
            if key_hash in seen:
                raise ValueError("Test 业务键重复，禁止去重后继续")
            seen.add(key_hash)
            keys_digest.update(key + b"\n")
            for name, line in lines.items():
                digests[name].update(line)
            count += 1
            if count % 50000 == 0:
                print(f"[四模型 Test 对齐] {count:,}/{config['rows']:,}", flush=True)
    if count != config["rows"]:
        raise ValueError("Test 行数不足")
    for name, path in sources.items():
        if (digests[name].hexdigest() != config["models"][name]["test_sha256"]
                or file_stat(path) != before[name]):
            raise ValueError(f"Test 哈希变化：{name}")
    return dict(rows=count, business_keys_sha256=keys_digest.hexdigest(),
                sources={name: dict(file=str(path), rows=count, sha256=digests[name].hexdigest(), **before[name])
                         for name, path in sources.items()})


def source_hashes() -> dict[str, str]:
    files = [Path(__file__), ROOT / "scripts/gnpr/evaluate_retrieval.py",
             ROOT / "src/poi_gr/methods/gnpr/eval.py", ROOT / "src/poi_gr/sft/evaluation.py",
             ROOT / "src/poi_gr/pid/trie.py", ROOT / "scripts/genpoi/build_proximity_data.py",
             ROOT / "scripts/genpoi/predict_proximity.py"]
    files += [ROOT / "qg_prqk/src/qg_prqk/sft" / name for name in
              ("evaluation.py", "full_test_data.py", "full_test_evaluation.py", "constrained_decoding.py", "constrained_full_test.py")]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in files}


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    root = resolve(config["output_root"])
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        verify_plan(config, plan)
        print("已完成的四模型 Test 预检来源一致，复用。", flush=True)
        return plan
    data = scan_aligned_sources(config)
    weights = {}
    for name, spec in config["models"].items():
        path = resolve(spec["checkpoint"]) / "model.safetensors"
        digest = sha256_file(path)
        if digest != spec["checkpoint_sha256"]:
            raise ValueError(f"模型不是已冻结权重：{name}")
        weights[name] = dict(file=str(path), sha256=digest, **file_stat(path))
    plan = dict(status="prepared", config=config, data=data, weights=weights, source_hashes=source_hashes())
    atomic_json(plan_path, plan)
    return plan


def verify_plan(config: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if plan.get("status") != "prepared" or plan["config"] != config or plan["source_hashes"] != source_hashes():
        raise ValueError("预检计划与配置/代码不同，禁止混用旧断点")
    for spec in [*plan["data"]["sources"].values(), *plan["weights"].values()]:
        if any(file_stat(Path(spec["file"]))[key] != spec[key] for key in ("bytes", "mtime_ns")):
            raise ValueError(f"完整哈希预检后文件发生变化：{spec['file']}")


def run_gnpr(config: Mapping[str, Any], output: Path, smoke: int | None) -> dict[str, Any]:
    from scripts.gnpr import evaluate_retrieval as ev
    from scripts.tiger.evaluate_retrieval import JsonlRecordSequence

    spec = config["models"]["gnpr"]
    tokenizer_path = resolve(spec["tokenizer"])
    tokenizer, template = ev.load_lf_tokenizer_and_template(tokenizer_path, project_root=ROOT)
    tokens, token_metadata = ev.load_gnpr_token_ids(tokenizer_path, tokenizer)
    index, index_metadata = ev.load_gnpr_id_index(resolve(spec["identifier_dir"]))
    if index_metadata["rows"] != 716245 or len(tokens.dedup) != index.dedup_capacity:
        raise ValueError("GNPR 不是 active 716245 库或词表容量不一致")
    metadata = ev.validate_checkpoints([resolve(spec["checkpoint"])], expected_steps=[spec["step"]],
                                       expected_epochs=[3.0], tokenizer_path=tokenizer_path)[0]
    path = resolve(spec["data_dir"]) / "test.jsonl"
    records = JsonlRecordSequence(path, expected_rows=config["rows"])
    return ev.evaluate_checkpoint(metadata=metadata, records=records, subset_path=path,
                                  subset_sha256=spec["test_sha256"], tokenizer=tokenizer, template=template,
                                  tokens=tokens, token_metadata=token_metadata, index=index, index_metadata=index_metadata,
                                  output_dir=output, num_beams=10, initial_batch_size=32, chunk_size=500,
                                  cutoff_len=1024, smoke_limit=smoke, split="test")


def run_genpoi(config: Mapping[str, Any], output: Path, smoke: int | None) -> dict[str, Any]:
    from scripts.genpoi.build_proximity_data import write_split
    from poi_gr.sft.evaluation import load_ssp_predictions, run_full_evaluation

    spec = config["models"]["genpoi"]
    source = resolve(spec["data_dir"]) / "test.jsonl"
    shared = resolve(config["output_root"]) / "genpoi/ssp_input"
    predictions = resolve(config["output_root"]) / "genpoi/ssp_predictions"
    shared.mkdir(parents=True, exist_ok=True)
    manifest_path = shared / "manifest.json"
    if not manifest_path.exists():
        # The frozen prediction CLI calls its input slot 'valid'; rows remain Test.
        data = write_split(source, shared / "valid.parquet", split="test", batch_rows=50000, limit=None)
        if data["rows"] != config["rows"] or data["source_sha256"] != spec["test_sha256"]:
            raise ValueError("GenPOI SSP Test 数据来源不一致")
        atomic_json(manifest_path, dict(schema_version="genpoi-proximity-data-v1", status="completed",
                                       split="test", classes=list(range(7)), gid_length=6,
                                       query_source="CURRENT/QUERY", outputs={"valid": data}))
    data = read_json(manifest_path)["outputs"]["valid"]
    if data["source_sha256"] != spec["test_sha256"] or sha256_file(shared / "valid.parquet") != data["output_sha256"]:
        raise ValueError("已有 SSP 输入不同，拒绝复用")
    if not (predictions / "manifest.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/genpoi/predict_proximity.py"),
                        "--data-dir", str(shared), "--head-dir", str(resolve(spec["head_dir"])),
                        "--output-dir", str(predictions), "--encode-batch-size", "256"], check=True, cwd=ROOT)
    pred = read_json(predictions / "manifest.json")
    if (pred["head_sha256"] != sha256_file(resolve(spec["head_dir"]) / "proximity_head.safetensors")
            or pred["head_manifest_sha256"] != sha256_file(resolve(spec["head_dir"]) / "training_manifest.json")
            or pred["gamma"] != 2):
        raise ValueError("SSP 预测来自不同的 head")
    load_ssp_predictions(predictions, evaluation_data_sha256=spec["test_sha256"], expected_rows=config["rows"])
    trie = resolve(spec["trie_dir"])
    trie_manifest = read_json(trie / "trie_manifest.json")
    if trie_manifest["leaf_count"] != 716245:
        raise ValueError("GenPOI Trie 不是 active 全目录")
    return run_full_evaluation(data_file=source, split="test", expected_rows=config["rows"],
                               data_sha256=spec["test_sha256"], checkpoint=resolve(spec["checkpoint"]),
                               epoch=3.0, validation_loss=None, tokenizer_path=resolve(spec["tokenizer"]),
                               trie_dir=trie, mapping_path=Path(trie_manifest["input"]["pid_mapping"]),
                               output_root=output, project_root=ROOT, num_beams=10, num_return_sequences=10,
                               top_k=10, initial_batch_size=32, chunk_size=500, cutoff_len=1024,
                               smoke_limit=smoke, ssp_predictions_dir=predictions)


def run_worker(config: dict[str, Any], plan: dict[str, Any], method: str, smoke: int | None) -> None:
    verify_plan(config, plan)
    output = resolve(config["output_root"]) / method / (f"smoke{smoke}" if smoke else "full")
    if method == "gnpr":
        result = run_gnpr(config, output, smoke)
    elif method == "genpoi":
        result = run_genpoi(config, output, smoke)
    else:
        from qg_prqk.sft.full_test_data import load_test_config
        from qg_prqk.sft.constrained_full_test import evaluate_constrained_test

        qg_config = load_test_config(resolve(config["qg_test_config"]))
        variant = config["models"][method]["variant"]
        result = evaluate_constrained_test(qg_config, variant=variant, source=plan["data"]["sources"][method],
                                           output=output, plan_signature=signature(plan), smoke_limit=smoke)
    atomic_json(output / "suite_result.json", dict(status="completed", method=method,
                plan_signature=signature(plan), decoding=MODES[method], split="test", date=config["date"],
                smoke_limit=smoke, result=result))


def summarize(config: dict[str, Any], plan: dict[str, Any], smoke: int | None) -> dict[str, Any]:
    rows = []
    for name in METHODS:
        directory = resolve(config["output_root"]) / name / (f"smoke{smoke}" if smoke else "full")
        payload = read_json(directory / "suite_result.json")
        result = payload["result"]
        expected = dict(status="completed", method=name, plan_signature=signature(plan), decoding=MODES[name],
                        split="test", date=config["date"], smoke_limit=smoke)
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"结果来源或协议不同：{name}")
        metrics = result["metrics"]
        if result["status"] != "completed" or metrics["sample_count"] != (smoke or config["rows"]):
            raise ValueError(f"结果未覆盖全集：{name}")
        validity = (metrics["generation_validity"]["valid_pid_ratio"] if name == "genpoi" else metrics["valid_id_rate"])
        if name != "gnpr" and validity != 1.0:
            raise ValueError(f"约束模型出现非法候选：{name}")
        row = dict(method=name, decoding=MODES[name], sample_count=metrics["sample_count"],
                   **{key: metrics[key] for key in ("hr@1", "hr@3", "hr@5", "hr@10", "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10")},
                   valid_id_rate=validity)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for key, value in row.items() if "@" in key or key == "valid_id_rate"):
            raise ValueError("指标不是有限的 0—1 比例")
        rows.append(row)
    return dict(status="completed", date=config["date"], split="test", smoke_limit=smoke,
                business_keys_sha256=plan["data"]["business_keys_sha256"], results=rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="四卡 active 全量 Test：GNPR 无约束、GenPOI SSP+TCG、QG-HRQ 两版全目录约束。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "prepare", "worker", "summarize"), default="plan")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args()
    if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 100:
        parser.error("--smoke-limit 必须为 1—100")
    config = load_config(resolve(args.config))
    if args.stage == "plan":
        print(f"输入预检通过：{config['date']} Test 共 {config['rows']:,} 条；正式运行时先全量哈希及四模型同行对齐。")
        for gpu, name in enumerate(METHODS):
            print(f"GPU {gpu}: {name}, {MODES[name]}, {config['models'][name]['checkpoint']}")
        return 0
    if args.stage == "prepare":
        prepare(config)
        return 0
    plan = read_json(resolve(config["output_root"]) / "plan.json")
    verify_plan(config, plan)
    if args.stage == "worker":
        if args.method is None:
            parser.error("worker 必须指定 --method")
        run_worker(config, plan, args.method, args.smoke_limit)
    else:
        import csv
        payload = summarize(config, plan, args.smoke_limit)
        output = resolve(config["output_root"])
        if args.smoke_limit:
            output /= f"smoke{args.smoke_limit}"
        atomic_json(output / "summary.json", payload)
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload["results"][0]))
            writer.writeheader()
            writer.writerows(payload["results"])
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"四模型 Test 评测失败：{error}", file=sys.stderr)
        raise SystemExit(2)

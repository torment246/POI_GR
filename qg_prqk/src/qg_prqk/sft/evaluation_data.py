"""Align five frozen Validation sets to each branch without resampling."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sft.data import SFT_DATA_SCHEMA_VERSION
from qg_prqk.sft.training import project_root


SUBSETS = (
    "fixed10k", "seen_query_unseen_pair", "unseen_query_seen_target",
    "long_tail_target", "cold_target",
)
DEFAULT_CONFIG = "qg_prqk/configs/sft/evaluation_epoch3_fixed10k_generalization_v1.yaml"


class SftEvaluationError(ValueError):
    """Raised when evaluation inputs violate the frozen protocol."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SftEvaluationError(f"无法读取 JSON：{path}") from error
    if not isinstance(value, dict):
        raise SftEvaluationError(f"JSON 必须是 object：{path}")
    return value


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root() / path).resolve()


def signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise SftEvaluationError(f"文件缺失或 SHA256 与冻结契约不一致：{path}")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "qg-prqk-sft-evaluation-suite-v1":
        raise SftEvaluationError("评测配置 schema 不兼容")
    if (config.get("split"), config.get("date"), config.get("source_rows"), config.get("subset_size")) != (
        "valid", "2026-07-13", 597421, 10000,
    ):
        raise SftEvaluationError("只允许冻结的 2026-07-13 Validation 五组 10k")
    decoding = config.get("decoding", {})
    expected = dict(mode="unconstrained", num_beams=10, cutoff_len=1024,
                    length_penalty=1.0, early_stopping=True, renormalize_logits=True)
    if any(decoding.get(k) != v for k, v in expected.items()):
        raise SftEvaluationError("不得隐式改变既有无约束 Beam=10、1024 评测协议")
    if any(not isinstance(decoding.get(k), int) or decoding[k] <= 0 for k in ("batch_size", "chunk_size")):
        raise SftEvaluationError("batch_size/chunk_size 必须为正整数")
    if set(config.get("variants", {})) != {"a4_gid_parent", "a4_nogid"}:
        raise SftEvaluationError("必须声明两个 QG SFT 分支")
    output = resolve(config["output_root"])
    if not output.is_relative_to(resolve("qg_prqk/outputs")) or output == resolve("qg_prqk/outputs"):
        raise SftEvaluationError("评测输出必须隔离在 qg_prqk/outputs 子目录")
    return config


def business_key(record: Mapping[str, Any]) -> tuple[str, str]:
    values = tuple(record.get(k) for k in ("order_id", "searchid"))
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise SftEvaluationError("评测样本缺少 order_id/searchid")
    return values


def keys_sha256(keys: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256("".join(f"{a}\t{b}\n" for a, b in keys).encode()).hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("rb") as stream:
        for row, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError) as error:
                raise SftEvaluationError(f"JSONL 第 {row} 行非法：{path}") from error
            if not isinstance(record, dict) or record.get("split") != "valid":
                raise SftEvaluationError("评测只允许 valid object，禁止 Train/Test")
            records.append(record)
    return records


def current_context(record: Mapping[str, Any]) -> str:
    try:
        content = record["messages"][0]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise SftEvaluationError("缺少用户 Messages") from error
    matches = re.findall(r"<CURRENT>.*?</CURRENT>", content, flags=re.DOTALL)
    if len(matches) != 1:
        raise SftEvaluationError("当前请求必须有唯一完整 CURRENT 块")
    return matches[0]


def load_references(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fixed = resolve(config["fixed_reference"])
    fixed_manifest = fixed.with_name(f"{fixed.stem}_manifest.json")
    require_hash(fixed_manifest, config["fixed_manifest_sha256"])
    fixed_spec = load_json(fixed_manifest)
    suite_dir = resolve(config["generalization_dir"])
    require_hash(suite_dir / "suite_manifest.json", config["generalization_manifest_sha256"])
    suite = load_json(suite_dir / "suite_manifest.json")
    for manifest in (fixed_spec, suite):
        if (manifest.get("status"), manifest.get("split"), manifest.get("date")) != (
            "completed", "valid", config["date"],
        ):
            raise SftEvaluationError("参考集不是冻结日期的已完成 Validation")
    references = {}
    union: set[tuple[str, str]] = set()
    for name in SUBSETS:
        spec = fixed_spec if name == "fixed10k" else suite["subsets"][name]
        path = fixed if name == "fixed10k" else suite_dir / f"{name}_10000.jsonl"
        require_hash(path, spec["output_sha256"])
        records = read_records(path)
        keys = [business_key(record) for record in records]
        if len(keys) != config["subset_size"] or len(set(keys)) != len(keys):
            raise SftEvaluationError(f"{name} 行数不等于 10k 或业务主键重复")
        if keys_sha256(keys) != spec["business_keys_sha256"] or union.intersection(keys):
            raise SftEvaluationError(f"{name} 业务主键哈希变化或五组交叠")
        union.update(keys)
        references[name] = dict(path=str(path), sha256=spec["output_sha256"],
                                business_keys_sha256=spec["business_keys_sha256"], records=records)
    return references


def align_subsets(
    valid_file: Path, references: Mapping[str, Mapping[str, Any]], output_dir: Path,
    *, variant: str, source_rows: int, source_sha256: str, source_manifest_sha256: str,
) -> dict[str, Any]:
    """Scan Validation once, hash while reading, and retain only requested rows."""
    reference_meta = {name: {k: v for k, v in ref.items() if k != "records"}
                      for name, ref in references.items()}
    contract = dict(schema_version="qg-prqk-aligned-evaluation-data-v1", variant=variant,
                    source_file=str(valid_file.resolve()), source_rows=source_rows,
                    source_sha256=source_sha256, source_manifest_sha256=source_manifest_sha256,
                    references=reference_meta, split="valid", date="2026-07-13")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if any(manifest.get(k) != v for k, v in contract.items()) or manifest.get("status") != "completed":
            raise SftEvaluationError("已有子集契约不同，拒绝覆盖；请使用新版本输出目录")
        for name, ref in references.items():
            path = output_dir / f"{name}.jsonl"
            require_hash(path, manifest["outputs"][name]["sha256"])
            records = read_records(path)
            if len(records) != len(ref["records"]) or keys_sha256([business_key(r) for r in records]) != ref["business_keys_sha256"]:
                raise SftEvaluationError("已有子集行数或业务主键发生变化")
        return manifest
    wanted = {}
    for ref in references.values():
        for record in ref["records"]:
            key = business_key(record)
            if key in wanted:
                raise SftEvaluationError("五个冻结集合必须互斥")
            wanted[key] = record
    matched: dict[tuple[str, str], bytes] = {}
    digest = hashlib.sha256()
    count = 0
    with valid_file.open("rb") as stream:
        for count, raw in enumerate(stream, 1):
            digest.update(raw)
            record = json.loads(raw)
            if record.get("split") != "valid" or record.get("identifier_variant") != variant:
                raise SftEvaluationError("源数据 split/identifier_variant 错配")
            key = business_key(record)
            if key not in wanted:
                continue
            reference = wanted[key]
            if key in matched:
                raise SftEvaluationError("Validation 业务主键重复")
            if record.get("target_poi_id") != reference.get("target_poi_id") or not record.get("target_poi_id"):
                raise SftEvaluationError("同一业务主键目标 POI 不一致")
            # Some historical GenPOI references omit the auxiliary user_token field.
            # Their CURRENT block and history_length are still present and immutable.
            user_mismatch = reference.get("user_token") is not None and record.get("user_token") != reference["user_token"]
            if current_context(record) != current_context(reference) or record.get("history_length") != reference.get("history_length") or user_mismatch:
                raise SftEvaluationError("同一业务主键的 Query/用户位置/历史长度发生变化")
            matched[key] = raw
    if count != source_rows or digest.hexdigest() != source_sha256:
        raise SftEvaluationError("源 Validation 行数/SHA256 与训练数据 manifest 不一致")
    if len(matched) != len(wanted):
        raise SftEvaluationError(f"Validation 缺少 {len(wanted) - len(matched)} 条冻结样本")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, ref in references.items():
        path = output_dir / f"{name}.jsonl"
        temporary = path.with_suffix(".jsonl.writing")
        with temporary.open("wb") as stream:
            for record in ref["records"]:
                stream.write(matched[business_key(record)])
        digest_value = sha256_file(temporary)
        if path.exists():
            require_hash(path, digest_value)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        outputs[name] = dict(file=str(path.resolve()), rows=len(ref["records"]),
                             sha256=digest_value, business_keys_sha256=ref["business_keys_sha256"])
    manifest = {**contract, "status": "completed", "built_at": utc_now(), "outputs": outputs,
                "selection": "exact_order_id_searchid_match_in_reference_order",
                "target_poi_mismatch_count": 0, "current_context_mismatch_count": 0}
    write_json_atomic(manifest_path, manifest)
    return manifest


def prepare_variant(config: Mapping[str, Any], variant: str) -> dict[str, Any]:
    spec = config["variants"][variant]
    data_dir = resolve(spec["data_dir"])
    require_hash(data_dir / "manifest.json", spec["data_manifest_sha256"])
    require_hash(resolve(spec["identifier_dir"]) / "manifest.json", spec["identifier_manifest_sha256"])
    manifest = load_json(data_dir / "manifest.json")
    if (manifest.get("status"), manifest.get("schema_version"), manifest.get("variant"), manifest.get("scan_mode")) != (
        "completed", SFT_DATA_SCHEMA_VERSION, variant, "full",
    ) or not (data_dir / "_SUCCESS").is_file():
        raise SftEvaluationError("SFT 数据不是对应分支的已完成全量产物")
    if manifest["final_identifier"]["manifest_sha256"] != spec["identifier_manifest_sha256"]:
        raise SftEvaluationError("评测 ID 与训练数据使用的 ID 不一致")
    if manifest["source_sft"]["time_split"]["valid"] != config["date"]:
        raise SftEvaluationError("SFT Validation 日期发生变化")
    valid = manifest["outputs"]["valid.jsonl"]
    if valid["rows"] != config["source_rows"]:
        raise SftEvaluationError("Validation 行数契约发生变化")
    return align_subsets(
        data_dir / "valid.jsonl", load_references(config),
        resolve(config["output_root"]) / variant / "data", variant=variant,
        source_rows=valid["rows"], source_sha256=valid["sha256"],
        source_manifest_sha256=spec["data_manifest_sha256"],
    )

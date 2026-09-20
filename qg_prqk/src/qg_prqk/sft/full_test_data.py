"""Validate paired Test manifests and read full JSONL with bounded memory."""

from __future__ import annotations

import hashlib
import json
from array import array
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from qg_prqk.sft.evaluation_data import (
    SftEvaluationError, business_key, current_context, load_config, load_json,
    require_hash, resolve,
)

VARIANTS = ("a4_gid_parent", "a4_nogid")
DEFAULT_CONFIG = "qg_prqk/configs/sft/evaluation_epoch3_full_test_20260714_v1.yaml"


def load_test_config(path: Path) -> dict[str, Any]:
    """Extend the frozen Validation decoding protocol, never its sample selection."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or (
        spec.get("schema_version"), spec.get("split"), spec.get("date"), spec.get("source_rows")
    ) != ("qg-prqk-full-test-config-v1", "test", "2026-07-14", 606682):
        raise SftEvaluationError("只允许 2026-07-14 全量 606,682 条 Test")
    protocol = resolve(spec["validation_protocol"])
    require_hash(protocol, spec["validation_protocol_sha256"])
    base = load_config(protocol)
    output = resolve(spec["output_root"])
    allowed = resolve("qg_prqk/outputs/eval")
    validation_output = resolve(base["output_root"])
    if (not output.is_relative_to(allowed) or output == allowed
            or output.is_relative_to(validation_output) or validation_output.is_relative_to(output)):
        raise SftEvaluationError("Test 必须使用 qg_prqk/outputs/eval 下独立于 Validation 的目录")
    if set(spec.get("checkpoint_sha256", {})) != set(VARIANTS):
        raise SftEvaluationError("必须冻结两个已评测 epoch-3 模型的 SHA256")
    return {**spec, "tokenizer": base["tokenizer"], "variants": base["variants"],
            "decoding": base["decoding"], "output_root": str(output)}


def parse_test_record(raw: bytes, *, line_number: int, variant: str) -> dict[str, Any]:
    """Reject wrong splits, malformed messages and missing exact-POI labels."""
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise SftEvaluationError(f"Test 第 {line_number} 行 JSON 非法") from error
    if not isinstance(record, dict) or (record.get("split"), record.get("identifier_variant")) != ("test", variant):
        raise SftEvaluationError(f"Test 第 {line_number} 行 split/variant 不一致")
    business_key(record)
    messages = record.get("messages")
    if (not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages)
            or [m.get("role") for m in messages] != ["user", "assistant"]):
        raise SftEvaluationError(f"Test 第 {line_number} 行必须有 user/assistant Messages")
    if not isinstance(record.get("target_poi_id"), str) or not record["target_poi_id"]:
        raise SftEvaluationError("Test 缺少目标 POI，禁止丢行")
    if not isinstance(record.get("requires_dedup"), bool):
        raise SftEvaluationError("Test 缺少末位 Dedup 标志")
    current_context(record)
    return record


def file_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def test_sources(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Bind the date through frozen data manifests; Test JSONL has no date column."""
    sources = {}
    for variant in VARIANTS:
        spec = config["variants"][variant]
        directory = resolve(spec["data_dir"])
        require_hash(directory / "manifest.json", spec["data_manifest_sha256"])
        manifest = load_json(directory / "manifest.json")
        if (manifest.get("schema_version"), manifest.get("status"), manifest.get("variant"), manifest.get("scan_mode")) != (
            "qg-prqk-paired-sft-data-v1", "completed", variant, "full",
        ) or not (directory / "_SUCCESS").is_file():
            raise SftEvaluationError("Test 的原始 QG Messages 构建尚未验收")
        source = manifest["source_sft"]
        if source["time_split"]["test"] != config["date"]:
            raise SftEvaluationError("Test 来源不是最后一天 2026-07-14")
        if manifest["final_identifier"]["manifest_sha256"] != spec["identifier_manifest_sha256"]:
            raise SftEvaluationError("Test 标签与冻结 Final ID 版本不一致")
        require_hash(resolve(spec["identifier_dir"]) / "manifest.json", spec["identifier_manifest_sha256"])
        data = manifest["outputs"]["test.jsonl"]
        path = directory / "test.jsonl"
        if data["rows"] != config["source_rows"] or path.stat().st_size != data["bytes"]:
            raise SftEvaluationError("Test 文件大小/全量行数与契约不一致")
        sources[variant] = dict(file=str(path), rows=data["rows"], sha256=data["sha256"],
                                **file_stat(path), data_manifest_sha256=spec["data_manifest_sha256"],
                                original_test_sha256=source["files"]["test"]["scanned_sha256"])
    if len({s["original_test_sha256"] for s in sources.values()}) != 1:
        raise SftEvaluationError("两个 Test 分支不是同一原始请求全集")
    return sources


def scan_paired_test(sources: Mapping[str, Mapping[str, Any]], *, expected_rows: int) -> dict[str, Any]:
    """Hash and align every row once without materializing full message objects."""
    digests = {v: hashlib.sha256() for v in VARIANTS}
    keys_digest = hashlib.sha256()
    seen: set[bytes] = set()
    rows = 0
    with ExitStack() as stack:
        streams = {v: stack.enter_context(Path(sources[v]["file"]).open("rb")) for v in VARIANTS}
        while True:
            lines = {v: stream.readline() for v, stream in streams.items()}
            if not any(lines.values()):
                break
            if not all(lines.values()) or rows >= expected_rows:
                raise SftEvaluationError("两版 Test 行数不同或超出冻结全集")
            rows += 1
            records = {}
            for variant, line in lines.items():
                digests[variant].update(line)
                records[variant] = parse_test_record(line, line_number=rows, variant=variant)
            left, right = (records[v] for v in VARIANTS)
            keys = business_key(left)
            if keys != business_key(right) or any(left.get(k) != right.get(k) for k in (
                "sample_id", "target_poi_id", "history_length", "user_token",
            )) or current_context(left) != current_context(right):
                raise SftEvaluationError(f"两版 Test 第 {rows} 行业务键/目标/当前请求不一致")
            key_bytes = f"{keys[0]}\t{keys[1]}\n".encode()
            key_hash = hashlib.sha256(key_bytes).digest()
            if key_hash in seen:
                raise SftEvaluationError("Test 存在重复业务键，禁止静默去重")
            seen.add(key_hash)
            keys_digest.update(key_bytes)
            if rows % 50000 == 0:
                print(f"[Test 预检] 两版全量对齐 {rows:,}/{expected_rows:,}", flush=True)
    if rows != expected_rows:
        raise SftEvaluationError("Test 行数不足，禁止将子集冒充全量")
    for variant, source in sources.items():
        if digests[variant].hexdigest() != source["sha256"]:
            raise SftEvaluationError(f"{variant} Test SHA256 不一致")
        if any(file_stat(Path(source["file"]))[k] != source[k] for k in ("bytes", "mtime_ns")):
            raise SftEvaluationError("Test 在预检扫描时被修改")
    return dict(status="completed", split="test", date="2026-07-14", rows=rows,
                business_keys_sha256=keys_digest.hexdigest(), sources=dict(sources),
                selection_method="all_source_rows", copy_materialized=False)


class JsonlRecordSequence(Sequence[Mapping[str, Any]]):
    """Keep only uint64 byte offsets in RAM; decode messages one chunk at a time."""

    def __init__(self, path: Path, *, variant: str, expected_rows: int, expected_sha256: str) -> None:
        self.path, self.variant = path, variant
        self.offsets = array("Q")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            position = 0
            for line in stream:
                self.offsets.append(position)
                position += len(line)
                digest.update(line)
        if len(self.offsets) != expected_rows or digest.hexdigest() != expected_sha256:
            raise SftEvaluationError("全量 Test 行数/SHA256 在 worker 启动时不一致")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, key: int | slice) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            if start >= stop:
                return []
            with self.path.open("rb") as stream:
                stream.seek(self.offsets[start])
                return [parse_test_record(stream.readline(), line_number=i + 1, variant=self.variant)
                        for i in range(start, stop)]
        index = int(key)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self[index:index + 1][0]

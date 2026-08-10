"""Freeze a leakage-free embedding retrieval evaluation set."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class EmbeddingEvalSetError(RuntimeError):
    """Raised when the frozen evaluation-set contract is violated."""


@dataclass(frozen=True)
class FrozenEmbeddingEvalSet:
    """Paths and immutable identifiers of a frozen evaluation set."""

    data_path: Path
    manifest_path: Path
    row_count: int
    sha256: str
    manifest: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise EmbeddingEvalSetError(f"{name} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EmbeddingEvalSetError(f"{name} JSON 非法：{path}") from error
    if not isinstance(value, dict):
        raise EmbeddingEvalSetError(f"{name} 必须是 JSON object：{path}")
    return value


def _non_empty_string(record: Mapping[str, Any], field: str, source: str) -> str:
    value = record.get(field)
    if value is None or not str(value).strip():
        raise EmbeddingEvalSetError(f"{source} 缺少有效 {field}")
    return str(value)


def _business_key(record: Mapping[str, Any], source: str) -> tuple[str, str]:
    return (
        _non_empty_string(record, "order_id", source),
        _non_empty_string(record, "searchid", source),
    )


def _extract_query(record: Mapping[str, Any], source: str) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise EmbeddingEvalSetError(f"{source} messages 必须恰好包含 user/assistant")
    user, assistant = messages
    if not isinstance(user, Mapping) or user.get("role") != "user":
        raise EmbeddingEvalSetError(f"{source} 第一条 message 不是 user")
    if not isinstance(assistant, Mapping) or assistant.get("role") != "assistant":
        raise EmbeddingEvalSetError(f"{source} 第二条 message 不是 assistant")
    content = user.get("content")
    if not isinstance(content, str) or not content.startswith("<QUERY>"):
        raise EmbeddingEvalSetError(f"{source} user content 缺少 <QUERY> 前缀")
    marker = "</QUERY>\n<USER_GID>"
    if marker not in content or not content.endswith("</USER_GID>"):
        raise EmbeddingEvalSetError(f"{source} user content 结构非法")
    query, _ = content[len("<QUERY>") :].rsplit(marker, 1)
    if not query.strip():
        raise EmbeddingEvalSetError(f"{source} Query 为空")
    return query


def _validate_sft_manifest(
    sft_dir: Path,
    *,
    expected_train_rows: int,
    expected_valid_rows: int,
) -> tuple[dict[str, Any], Path, Path, str, str]:
    manifest_path = sft_dir / "manifest.json"
    manifest = _load_json_object(manifest_path, "SFT manifest")
    if manifest.get("schema_version") != "sft-main-data-v1":
        raise EmbeddingEvalSetError("SFT manifest schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise EmbeddingEvalSetError("SFT manifest 状态不是 completed")

    split = manifest.get("time_split")
    expected_split = {
        "field": "create_time",
        "train": {"start": "2026-07-01", "end": "2026-07-12"},
        "valid": "2026-07-13",
        "test": "2026-07-14",
    }
    if not isinstance(split, Mapping) or any(
        split.get(key) != value for key, value in expected_split.items()
    ):
        raise EmbeddingEvalSetError("SFT Train/Validation 时间切分不是冻结的 V1 口径")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise EmbeddingEvalSetError("SFT manifest 缺少 outputs")

    def output_spec(filename: str, expected_rows: int) -> tuple[Path, str]:
        spec = outputs.get(filename)
        if not isinstance(spec, Mapping):
            raise EmbeddingEvalSetError(f"SFT manifest 缺少 {filename}")
        if spec.get("rows") != expected_rows:
            raise EmbeddingEvalSetError(
                f"{filename} 行数声明 {spec.get('rows')} != {expected_rows}"
            )
        digest = spec.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise EmbeddingEvalSetError(f"{filename} 缺少有效 SHA256")
        path = (sft_dir / filename).resolve()
        if not path.is_file():
            raise EmbeddingEvalSetError(f"SFT 数据不存在：{path}")
        return path, digest

    train_path, train_sha256 = output_spec("train.jsonl", expected_train_rows)
    valid_path, valid_sha256 = output_spec("valid.jsonl", expected_valid_rows)
    return manifest, train_path, valid_path, train_sha256, valid_sha256


def _load_reference_records(
    reference_subset: Path,
    reference_manifest_path: Path,
    *,
    expected_rows: int,
    expected_valid_rows: int,
    valid_path: Path,
    valid_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, Any], str, str]:
    reference_manifest = _load_json_object(
        reference_manifest_path,
        "固定 Validation manifest",
    )
    expected_manifest = {
        "schema_version": "fixed-validation-subset-v1",
        "status": "completed",
        "split": "valid",
        "date": "2026-07-13",
        "source_rows": expected_valid_rows,
        "source_sha256": valid_sha256,
        "subset_size": expected_rows,
        "output_rows": expected_rows,
        "selection_method": "lowest_sample_id_lexicographic",
        "output_order": "source_row_ascending",
    }
    if any(reference_manifest.get(key) != value for key, value in expected_manifest.items()):
        raise EmbeddingEvalSetError("固定 Validation manifest 与冻结协议不一致")
    declared_source = Path(str(reference_manifest.get("source_file", ""))).resolve()
    if declared_source != valid_path:
        raise EmbeddingEvalSetError("固定 Validation 的来源不是 V1 valid.jsonl")
    reference_sha256 = sha256_file(reference_subset)
    if reference_manifest.get("output_sha256") != reference_sha256:
        raise EmbeddingEvalSetError("固定 Validation 数据 SHA256 与 manifest 不一致")

    rows: list[dict[str, str]] = []
    sample_ids: set[str] = set()
    business_keys: set[tuple[str, str]] = set()
    order_ids: set[str] = set()
    with reference_subset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            source = f"固定 Validation 第 {line_number} 行"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EmbeddingEvalSetError(f"{source} JSON 非法") from error
            if not isinstance(record, Mapping):
                raise EmbeddingEvalSetError(f"{source} 不是 object")
            if record.get("split") != "valid":
                raise EmbeddingEvalSetError(f"{source} split 不是 valid")
            sample_id = _non_empty_string(record, "sample_id", source)
            key = _business_key(record, source)
            target_poi_id = _non_empty_string(record, "target_poi_id", source)
            query = _extract_query(record, source)
            if sample_id in sample_ids:
                raise EmbeddingEvalSetError(f"固定 Validation sample_id 重复：{sample_id}")
            if key in business_keys:
                raise EmbeddingEvalSetError("固定 Validation (order_id, searchid) 重复")
            if key[0] in order_ids:
                raise EmbeddingEvalSetError(f"固定 Validation order_id 重复：{key[0]}")
            sample_ids.add(sample_id)
            business_keys.add(key)
            order_ids.add(key[0])
            rows.append(
                {
                    "sample_id": sample_id,
                    "order_id": key[0],
                    "searchid": key[1],
                    "query": query,
                    "poi_id": target_poi_id,
                    "split": "valid",
                }
            )
    if len(rows) != expected_rows:
        raise EmbeddingEvalSetError(
            f"固定 Validation 行数 {len(rows):,} != {expected_rows:,}"
        )
    return rows, reference_manifest, reference_sha256, sha256_file(reference_manifest_path)


def _scan_jsonl_source(
    path: Path,
    *,
    expected_rows: int,
    expected_sha256: str,
    expected_split: str,
    reference_by_key: Mapping[tuple[str, str], Mapping[str, str]] | None = None,
    eval_order_ids: set[str] | None = None,
    eval_searchids: set[str] | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    matched_keys: set[tuple[str, str]] = set()
    overlap_keys: set[tuple[str, str]] = set()
    overlap_order_ids: set[str] = set()
    overlap_searchids: set[str] = set()
    reference_keys = set(reference_by_key or {})
    eval_order_ids = eval_order_ids or set()
    eval_searchids = eval_searchids or set()

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            rows += 1
            source = f"{path.name} 第 {line_number} 行"
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise EmbeddingEvalSetError(f"{source} JSON 非法") from error
            if not isinstance(record, Mapping):
                raise EmbeddingEvalSetError(f"{source} 不是 object")
            if record.get("split") != expected_split:
                raise EmbeddingEvalSetError(f"{source} split 不是 {expected_split}")
            key = _business_key(record, source)

            if expected_split == "train":
                if key in reference_keys:
                    overlap_keys.add(key)
                if key[0] in eval_order_ids:
                    overlap_order_ids.add(key[0])
                if key[1] in eval_searchids:
                    overlap_searchids.add(key[1])
                continue

            expected = (reference_by_key or {}).get(key)
            if expected is None:
                continue
            if key in matched_keys:
                raise EmbeddingEvalSetError("完整 Validation 中固定业务键重复")
            sample_id = _non_empty_string(record, "sample_id", source)
            target_poi_id = _non_empty_string(record, "target_poi_id", source)
            query = _extract_query(record, source)
            if sample_id != expected["sample_id"]:
                raise EmbeddingEvalSetError("固定评测集 sample_id 与完整 Validation 不一致")
            if target_poi_id != expected["poi_id"]:
                raise EmbeddingEvalSetError("固定评测集目标 POI 与完整 Validation 不一致")
            if query != expected["query"]:
                raise EmbeddingEvalSetError("固定评测集 Query 与完整 Validation 不一致")
            matched_keys.add(key)

    if rows != expected_rows:
        raise EmbeddingEvalSetError(f"{path.name} 实际行数 {rows:,} != {expected_rows:,}")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise EmbeddingEvalSetError(f"{path.name} SHA256 与 SFT manifest 不一致")
    return {
        "rows": rows,
        "sha256": actual_sha256,
        "matched_reference_rows": len(matched_keys),
        "missing_reference_rows": len(reference_keys - matched_keys),
        "business_key_overlap_count": len(overlap_keys),
        "order_id_overlap_count": len(overlap_order_ids),
        "searchid_overlap_count": len(overlap_searchids),
    }


def _serialized_rows(rows: list[dict[str, str]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, content)


def freeze_embedding_eval_set(
    sft_dir: Path,
    reference_subset: Path,
    output_dir: Path,
    *,
    expected_rows: int = 10_000,
    expected_train_rows: int = 7_586_410,
    expected_valid_rows: int = 597_421,
) -> FrozenEmbeddingEvalSet:
    """Build and verify the frozen V1 Validation embedding evaluation set."""

    sft_dir = sft_dir.resolve()
    reference_subset = reference_subset.resolve()
    output_dir = output_dir.resolve()
    reference_manifest_path = reference_subset.with_name(
        f"{reference_subset.stem}_manifest.json"
    )
    if expected_rows <= 0:
        raise EmbeddingEvalSetError("expected_rows 必须为正数")

    (
        sft_manifest,
        train_path,
        valid_path,
        train_sha256,
        valid_sha256,
    ) = _validate_sft_manifest(
        sft_dir,
        expected_train_rows=expected_train_rows,
        expected_valid_rows=expected_valid_rows,
    )
    (
        rows,
        reference_manifest,
        reference_sha256,
        reference_manifest_sha256,
    ) = _load_reference_records(
        reference_subset,
        reference_manifest_path,
        expected_rows=expected_rows,
        expected_valid_rows=expected_valid_rows,
        valid_path=valid_path,
        valid_sha256=valid_sha256,
    )

    reference_by_key = {
        (row["order_id"], row["searchid"]): row for row in rows
    }
    eval_order_ids = {row["order_id"] for row in rows}
    eval_searchids = {row["searchid"] for row in rows}
    valid_scan = _scan_jsonl_source(
        valid_path,
        expected_rows=expected_valid_rows,
        expected_sha256=valid_sha256,
        expected_split="valid",
        reference_by_key=reference_by_key,
    )
    if valid_scan["matched_reference_rows"] != expected_rows:
        raise EmbeddingEvalSetError(
            "完整 Validation 未覆盖全部固定评测样本："
            f"{valid_scan['matched_reference_rows']:,}/{expected_rows:,}"
        )

    train_scan = _scan_jsonl_source(
        train_path,
        expected_rows=expected_train_rows,
        expected_sha256=train_sha256,
        expected_split="train",
        reference_by_key=reference_by_key,
        eval_order_ids=eval_order_ids,
        eval_searchids=eval_searchids,
    )
    overlap_counts = {
        "business_key_overlap_count": train_scan["business_key_overlap_count"],
        "order_id_overlap_count": train_scan["order_id_overlap_count"],
        "searchid_overlap_count": train_scan["searchid_overlap_count"],
    }
    if any(overlap_counts.values()):
        raise EmbeddingEvalSetError(
            "固定评测集与训练集存在业务键重叠："
            + ", ".join(f"{key}={value}" for key, value in overlap_counts.items())
        )

    output_bytes = _serialized_rows(rows)
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    key_bytes = "".join(
        f"{row['order_id']}\t{row['searchid']}\n" for row in rows
    ).encode("utf-8")
    sample_id_bytes = "".join(f"{row['sample_id']}\n" for row in rows).encode(
        "utf-8"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "eval.jsonl"
    manifest_path = output_dir / "manifest.json"
    stable_manifest = {
        "schema_version": "embedding-retrieval-eval-set-v1",
        "status": "completed",
        "protocol_name": "sft_validation_10k_v1",
        "split": "valid",
        "date": "2026-07-13",
        "selection": {
            "reference_file": str(reference_subset),
            "reference_sha256": reference_sha256,
            "reference_manifest": str(reference_manifest_path),
            "reference_manifest_sha256": reference_manifest_sha256,
            "selection_method": reference_manifest["selection_method"],
            "output_order": reference_manifest["output_order"],
            "business_keys_sha256": hashlib.sha256(key_bytes).hexdigest(),
            "sample_ids_sha256": hashlib.sha256(sample_id_bytes).hexdigest(),
        },
        "source": {
            "sft_dir": str(sft_dir),
            "sft_manifest": str(sft_dir / "manifest.json"),
            "sft_manifest_sha256": sha256_file(sft_dir / "manifest.json"),
            "sft_build_fingerprint": sft_manifest.get("build_fingerprint"),
            "train": {
                "file": str(train_path),
                "rows": train_scan["rows"],
                "sha256": train_scan["sha256"],
                "date_start": "2026-07-01",
                "date_end": "2026-07-12",
            },
            "validation": {
                "file": str(valid_path),
                "rows": valid_scan["rows"],
                "sha256": valid_scan["sha256"],
                "date": "2026-07-13",
            },
        },
        "leakage_check": {
            **overlap_counts,
            "all_zero": True,
            "comparison_scope": [
                "order_id+searchid",
                "order_id",
                "searchid",
            ],
        },
        "contract": {
            "query_source": "messages[role=user].content/<QUERY>",
            "target_source": "target_poi_id",
            "output_fields": [
                "sample_id",
                "order_id",
                "searchid",
                "query",
                "poi_id",
                "split",
            ],
            "query_normalization": "none",
            "target_label": "positive_order_poi",
        },
        "statistics": {
            "rows": expected_rows,
            "unique_sample_ids": len({row["sample_id"] for row in rows}),
            "unique_business_keys": len(reference_by_key),
            "unique_order_ids": len(eval_order_ids),
            "unique_searchids": len(eval_searchids),
            "unique_queries": len({row["query"] for row in rows}),
            "unique_target_pois": len({row["poi_id"] for row in rows}),
        },
        "output": {
            "file": str(data_path),
            "rows": expected_rows,
            "sha256": output_sha256,
        },
    }

    if data_path.exists() or manifest_path.exists():
        if not data_path.is_file() or not manifest_path.is_file():
            raise EmbeddingEvalSetError("冻结评测集与 manifest 必须同时存在")
        existing_manifest = _load_json_object(manifest_path, "Embedding 评测集 manifest")
        comparable_manifest = dict(existing_manifest)
        comparable_manifest.pop("built_at", None)
        if comparable_manifest != stable_manifest:
            raise EmbeddingEvalSetError("已有 Embedding 评测集 manifest 与当前输入不一致")
        if sha256_file(data_path) != output_sha256:
            raise EmbeddingEvalSetError("已有 Embedding 评测集 SHA256 与 manifest 不一致")
        with data_path.open("rb") as handle:
            actual_rows = sum(1 for _ in handle)
        if actual_rows != expected_rows:
            raise EmbeddingEvalSetError("已有 Embedding 评测集行数不一致")
        return FrozenEmbeddingEvalSet(
            data_path=data_path,
            manifest_path=manifest_path,
            row_count=expected_rows,
            sha256=output_sha256,
            manifest=existing_manifest,
        )

    _write_bytes_atomic(data_path, output_bytes)
    manifest = {
        **stable_manifest,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(manifest_path, manifest)
    return FrozenEmbeddingEvalSet(
        data_path=data_path,
        manifest_path=manifest_path,
        row_count=expected_rows,
        sha256=output_sha256,
        manifest=manifest,
    )

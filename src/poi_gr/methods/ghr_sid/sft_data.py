"""Derive paired GHR-SID Messages data from the frozen TIGER SFT rows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from poi_gr.methods.ghr_sid.identifier import validate_ghr_identifier_output
from poi_gr.methods.tiger.data import load_tiger_id_lookup
from poi_gr.pid.dedup import sha256_file
from poi_gr.pid.geohash import GEOHASH_ALPHABET


SCHEMA_VERSION = "ghr-map-search-sft-data-v1"
SPECIAL_TOKENS_SCHEMA_VERSION = "ghr-map-search-special-tokens-v1"
TASK_NAME = "ghr_map_search_history_query_gid_poi_to_target"
SPLITS = ("train", "valid", "test")
STRUCTURE_TOKENS = (
    "<USER_ID>",
    "</USER_ID>",
    "<HISTORY>",
    "</HISTORY>",
    "<EVENT>",
    "</EVENT>",
    "<USER_GID>",
    "</USER_GID>",
    "<QUERY>",
    "</QUERY>",
    "<POI_GHR_ID>",
    "</POI_GHR_ID>",
    "<CURRENT>",
    "</CURRENT>",
    "<TARGET_POI>",
    "</TARGET_POI>",
)
TIGER_CONTENT_PATTERN = r"<S1_(\d+)><S2_(\d+)><S3_(\d+)><C_(\d+)>"
HISTORY_ID_PATTERN = re.compile(
    rf"<POI_TIGER_ID>({TIGER_CONTENT_PATTERN})</POI_TIGER_ID>"
)
TARGET_ID_PATTERN = re.compile(
    rf"\A<TARGET_POI>({TIGER_CONTENT_PATTERN})</TARGET_POI>\Z"
)


class GhrSftDataError(ValueError):
    """Raised when paired GHR SFT derivation violates its contract."""


@dataclass(frozen=True)
class GhrSftCandidate:
    """One identifier variant and its requested SFT output directory."""

    variant: str
    identifier_dir: Path
    output_dir: Path


@dataclass
class GhrIdLookup:
    """Memory-mapped GHR codes plus a lazy row-to-text cache."""

    variant: str
    identifier_dir: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    poi_ids: np.ndarray
    codes: np.ndarray
    lengths: np.ndarray
    tokens: tuple[str, ...]
    content_by_row: list[str | None]

    def content(self, row: int) -> str:
        cached = self.content_by_row[row]
        if cached is not None:
            return cached
        length = int(self.lengths[row])
        value = "".join(self.tokens[int(code)] for code in self.codes[row, :length])
        self.content_by_row[row] = value
        return value


@dataclass(frozen=True)
class PairedGhrSftResult:
    """Completed one-to-one SFT remapping outputs."""

    manifests: dict[str, dict[str, Any]]
    output_hashes: dict[str, dict[str, str]]


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise GhrSftDataError(f"{name} 不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GhrSftDataError(f"{name} JSON 解析失败：{path}") from error
    if not isinstance(payload, dict):
        raise GhrSftDataError(f"{name} 必须是 JSON object：{path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _pack_tiger_codes(values: Sequence[int]) -> int:
    if len(values) != 4:
        raise GhrSftDataError("TIGER identifier 必须是四层")
    s1, s2, s3, collision = (int(value) for value in values)
    if any(value < 0 or value >= 1024 for value in (s1, s2, s3, collision)):
        raise GhrSftDataError("TIGER identifier code 超出十位打包范围")
    return (s1 << 30) | (s2 << 20) | (s3 << 10) | collision


def _row_from_match(match: re.Match[str], row_by_tiger_key: Mapping[int, int]) -> int:
    key = _pack_tiger_codes(tuple(int(match.group(i)) for i in range(2, 6)))
    row = row_by_tiger_key.get(key)
    if row is None:
        raise GhrSftDataError(f"TIGER identifier 未匹配正式目录：{match.group(1)}")
    return row


def _load_ghr_lookup(candidate: GhrSftCandidate) -> GhrIdLookup:
    identifier_dir = candidate.identifier_dir.resolve()
    validation = validate_ghr_identifier_output(identifier_dir)
    manifest_path = identifier_dir / "ghr_id_manifest.json"
    manifest = _load_json(manifest_path, "GHR identifier manifest")
    if (
        validation["variant"] != candidate.variant
        or manifest.get("variant") != candidate.variant
    ):
        raise GhrSftDataError(f"候选名与 identifier variant 不一致：{candidate.variant}")
    outputs = manifest["outputs"]
    arrays = outputs["arrays"]
    poi_ids = np.load(
        identifier_dir / arrays["poi_ids"]["path"], mmap_mode="r", allow_pickle=False,
    )
    codes = np.load(
        identifier_dir / arrays["identifier_token_codes"]["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    lengths = np.load(
        identifier_dir / arrays["identifier_lengths"]["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    vocabulary = _load_json(
        identifier_dir / outputs["vocabulary"]["path"], "GHR identifier vocabulary",
    )
    raw_tokens = vocabulary.get("tokens")
    if (
        not isinstance(raw_tokens, list)
        or not all(isinstance(token, str) for token in raw_tokens)
        or len(raw_tokens) != vocabulary.get("token_count")
    ):
        raise GhrSftDataError("GHR identifier vocabulary token 契约无效")
    return GhrIdLookup(
        variant=candidate.variant,
        identifier_dir=identifier_dir,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        poi_ids=poi_ids,
        codes=codes,
        lengths=lengths,
        tokens=tuple(raw_tokens),
        content_by_row=[None] * len(poi_ids),
    )


def build_ghr_special_tokens(
    identifier_tokens: Sequence[str], *, user_bucket_count: int = 2000,
) -> dict[str, Any]:
    """Build a deduplicated tokenizer inventory shared by prompt and target."""

    if user_bucket_count != 2000:
        raise GhrSftDataError("地图检索 SFT 口径固定为 2000 个用户 Token")
    if not identifier_tokens or len(identifier_tokens) != len(set(identifier_tokens)):
        raise GhrSftDataError("identifier_tokens 必须非空且唯一")
    geohash_tokens = [f"<G_{value}>" for value in GEOHASH_ALPHABET]
    user_tokens = [f"<U_{value:04d}>" for value in range(user_bucket_count)]
    ordered: list[str] = []
    seen: set[str] = set()
    for token in (*STRUCTURE_TOKENS, *geohash_tokens, *user_tokens, *identifier_tokens):
        if token not in seen:
            ordered.append(token)
            seen.add(token)
    return {
        "schema_version": SPECIAL_TOKENS_SCHEMA_VERSION,
        "structure_tokens": list(STRUCTURE_TOKENS),
        "geohash_tokens": geohash_tokens,
        "user_bucket_count": user_bucket_count,
        "user_tokens": user_tokens,
        "identifier_tokens": list(identifier_tokens),
        "additional_special_tokens": ordered,
        "token_count": len(ordered),
    }


def _validate_source_contract(source_dir: Path) -> tuple[dict[str, Any], str]:
    manifest_path = source_dir / "manifest.json"
    manifest = _load_json(manifest_path, "TIGER SFT manifest")
    if manifest.get("schema_version") != "tiger-map-search-sft-data-v1":
        raise GhrSftDataError("源数据不是冻结的 TIGER map-search SFT schema")
    if manifest.get("status") != "completed":
        raise GhrSftDataError("源 TIGER SFT 数据未完成")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise GhrSftDataError("源 TIGER manifest 缺少 outputs")
    for filename in ("special_tokens.json", "stats.json"):
        contract = outputs.get(filename)
        path = source_dir / filename
        if not isinstance(contract, dict) or sha256_file(path) != contract.get(
            "sha256"
        ):
            raise GhrSftDataError(f"源 TIGER 文件 SHA256 不匹配：{filename}")
    for split in SPLITS:
        filename = f"{split}.jsonl"
        contract = outputs.get(filename)
        if (
            not isinstance(contract, dict)
            or not isinstance(contract.get("rows"), int)
            or not isinstance(contract.get("sha256"), str)
            or not (source_dir / filename).is_file()
        ):
            raise GhrSftDataError(f"源 TIGER split 契约无效：{filename}")
    return manifest, sha256_file(manifest_path)


def _build_tiger_row_index(
    tiger_identifier_dir: Path,
) -> tuple[dict[int, int], np.ndarray, str, str]:
    lookup, _, mapping_hash, manifest_hash = load_tiger_id_lookup(
        tiger_identifier_dir.resolve()
    )
    numeric_poi_ids = np.empty(lookup.poi_count, dtype=np.int64)
    for poi_id, row in lookup.row_by_poi_id.items():
        if not poi_id.isdecimal():
            raise GhrSftDataError(f"TIGER mapping POI ID 不是数字：{poi_id}")
        numeric_poi_ids[row] = int(poi_id)
    row_by_tiger_key: dict[int, int] = {}
    for row, codes in enumerate(lookup.codes):
        key = _pack_tiger_codes(codes)
        if key in row_by_tiger_key:
            raise GhrSftDataError("正式 TIGER identifier 不是全局唯一")
        row_by_tiger_key[key] = row
    return row_by_tiger_key, numeric_poi_ids, mapping_hash, manifest_hash


def _transform_record(
    record: dict[str, Any],
    *,
    split: str,
    line_number: int,
    row_by_tiger_key: Mapping[int, int],
    tiger_poi_ids: np.ndarray,
    lookups: Mapping[str, GhrIdLookup],
) -> tuple[dict[str, dict[str, Any]], int]:
    if record.get("split") != split:
        raise GhrSftDataError(f"{split}:{line_number} split 字段不一致")
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
        or not isinstance(messages[0].get("content"), str)
        or not isinstance(messages[1].get("content"), str)
    ):
        raise GhrSftDataError(f"{split}:{line_number} Messages 格式非法")
    history_length = record.get("history_length")
    if isinstance(history_length, bool) or not isinstance(history_length, int):
        raise GhrSftDataError(f"{split}:{line_number} history_length 非法")

    user_content = messages[0]["content"]
    history_matches = list(HISTORY_ID_PATTERN.finditer(user_content))
    if len(history_matches) != history_length:
        raise GhrSftDataError(f"{split}:{line_number} 历史 SID 数与 history_length 不一致")
    history_rows = [
        _row_from_match(match, row_by_tiger_key) for match in history_matches
    ]
    target_match = TARGET_ID_PATTERN.fullmatch(messages[1]["content"])
    if target_match is None:
        raise GhrSftDataError(f"{split}:{line_number} TIGER target 格式非法")
    target_row = _row_from_match(target_match, row_by_tiger_key)
    target_poi_id = record.get("target_poi_id")
    if str(int(tiger_poi_ids[target_row])) != target_poi_id:
        raise GhrSftDataError(f"{split}:{line_number} target POI 与 TIGER ID 不一致")

    transformed: dict[str, dict[str, Any]] = {}
    for variant, lookup in lookups.items():
        row_iterator = iter(history_rows)

        def replace_history(_: re.Match[str]) -> str:
            row = next(row_iterator)
            return f"<POI_GHR_ID>{lookup.content(row)}</POI_GHR_ID>"

        new_user_content = HISTORY_ID_PATTERN.sub(replace_history, user_content)
        if "POI_TIGER_ID" in new_user_content:
            raise GhrSftDataError(f"{split}:{line_number} 存在未替换的 TIGER SID")
        sample = dict(record)
        sample["messages"] = [
            {"role": "user", "content": new_user_content},
            {
                "role": "assistant",
                "content": f"<TARGET_POI>{lookup.content(target_row)}</TARGET_POI>",
            },
        ]
        sample.pop("target_tiger_id_key", None)
        sample["target_ghr_id_key"] = lookup.content(target_row)
        transformed[variant] = sample
    return transformed, len(history_rows)


def build_paired_ghr_sft_data(
    *,
    source_tiger_sft_dir: Path,
    tiger_identifier_dir: Path,
    candidates: Sequence[GhrSftCandidate],
    max_rows_per_split: int | None = None,
    progress_every: int = 250_000,
    progress: Callable[[str], None] | None = None,
) -> PairedGhrSftResult:
    """Remap the same frozen TIGER rows to one or more GHR identifier variants."""

    if not candidates:
        raise GhrSftDataError("至少需要一个 GHR SFT 候选")
    variants = [candidate.variant for candidate in candidates]
    if len(variants) != len(set(variants)):
        raise GhrSftDataError("GHR SFT candidate variant 重复")
    if max_rows_per_split is not None and max_rows_per_split <= 0:
        raise GhrSftDataError("max_rows_per_split 必须大于 0")
    if progress_every <= 0:
        raise GhrSftDataError("progress_every 必须大于 0")

    source_dir = source_tiger_sft_dir.resolve()
    source_manifest, source_manifest_hash = _validate_source_contract(source_dir)
    lookups = {
        candidate.variant: _load_ghr_lookup(candidate) for candidate in candidates
    }
    output_dirs = {
        candidate.variant: candidate.output_dir.resolve() for candidate in candidates
    }
    for variant, output_dir in output_dirs.items():
        if output_dir.exists():
            raise GhrSftDataError(f"{variant} 输出目录已存在，拒绝覆盖：{output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)

    if progress is not None:
        progress("构建正式 TIGER ID 到目录行号的唯一索引")
    (
        row_by_tiger_key,
        tiger_poi_ids,
        tiger_mapping_hash,
        tiger_identifier_manifest_hash,
    ) = _build_tiger_row_index(tiger_identifier_dir)
    for variant, lookup in lookups.items():
        if not np.array_equal(lookup.poi_ids, tiger_poi_ids):
            raise GhrSftDataError(f"{variant} GHR 与 TIGER POI 行序不一致")

    temporary_dirs: dict[str, Path] = {}
    handles: dict[str, dict[str, Any]] = {}
    digests: dict[str, dict[str, Any]] = {}
    row_counts = {split: 0 for split in SPLITS}
    history_occurrences = 0
    source_split_hashes: dict[str, str] = {}
    processing_error: BaseException | None = None
    try:
        for variant, output_dir in output_dirs.items():
            temporary_dirs[variant] = Path(
                tempfile.mkdtemp(
                    dir=output_dir.parent, prefix=f".{output_dir.name}.building-",
                )
            )
            handles[variant] = {
                split: (temporary_dirs[variant] / f"{split}.jsonl").open("wb")
                for split in SPLITS
            }
            digests[variant] = {split: hashlib.sha256() for split in SPLITS}

        for split in SPLITS:
            source_digest = hashlib.sha256()
            source_path = source_dir / f"{split}.jsonl"
            with source_path.open("rb") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    if (
                        max_rows_per_split is not None
                        and line_number > max_rows_per_split
                    ):
                        break
                    source_digest.update(raw_line)
                    try:
                        record = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise GhrSftDataError(
                            f"{split}:{line_number} 不是合法 JSON"
                        ) from error
                    if not isinstance(record, dict):
                        raise GhrSftDataError(f"{split}:{line_number} 必须是 JSON object")
                    transformed, occurrence_count = _transform_record(
                        record,
                        split=split,
                        line_number=line_number,
                        row_by_tiger_key=row_by_tiger_key,
                        tiger_poi_ids=tiger_poi_ids,
                        lookups=lookups,
                    )
                    history_occurrences += occurrence_count
                    for variant, sample in transformed.items():
                        encoded = (
                            json.dumps(
                                sample,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        ).encode("utf-8")
                        handles[variant][split].write(encoded)
                        digests[variant][split].update(encoded)
                    row_counts[split] += 1
                    if progress is not None and line_number % progress_every == 0:
                        progress(
                            f"GHR SFT：{split} 已完成 {line_number:,} 行，"
                            f"同时写入 {len(candidates)} 个候选"
                        )
            source_split_hashes[split] = source_digest.hexdigest()
            expected = source_manifest["outputs"][f"{split}.jsonl"]
            expected_rows = int(expected["rows"])
            if max_rows_per_split is None:
                if row_counts[split] != expected_rows:
                    raise GhrSftDataError(f"{split} 行数与源 TIGER manifest 不一致")
                if source_split_hashes[split] != expected["sha256"]:
                    raise GhrSftDataError(f"{split} 源 TIGER SHA256 不一致")
            elif row_counts[split] != min(max_rows_per_split, expected_rows):
                raise GhrSftDataError(f"{split} smoke 行数不符合上限")
            if progress is not None:
                progress(f"GHR SFT：{split} 完成 {row_counts[split]:,} 行")
    except BaseException as error:
        processing_error = error
    finally:
        for variant_handles in handles.values():
            for handle in variant_handles.values():
                handle.close()
    if processing_error is not None:
        for temporary_dir in temporary_dirs.values():
            if temporary_dir.exists():
                import shutil

                shutil.rmtree(temporary_dir)
        raise processing_error

    manifests: dict[str, dict[str, Any]] = {}
    output_hashes: dict[str, dict[str, str]] = {}
    try:
        source_stats = _load_json(source_dir / "stats.json", "TIGER SFT stats")
        for variant, lookup in lookups.items():
            temporary_dir = temporary_dirs[variant]
            special_tokens = build_ghr_special_tokens(lookup.tokens)
            stats = {
                **source_stats,
                "retained_sample_count": sum(row_counts.values()),
                "train_count": row_counts["train"],
                "valid_count": row_counts["valid"],
                "test_count": row_counts["test"],
                "history_event_occurrence_count": history_occurrences,
                "source_tiger_row_alignment": "exact_one_to_one_in_source_order",
                "identifier_variant": variant,
            }
            _write_json(temporary_dir / "special_tokens.json", special_tokens)
            _write_json(temporary_dir / "stats.json", stats)
            outputs = {
                f"{split}.jsonl": {
                    "rows": row_counts[split],
                    "sha256": digests[variant][split].hexdigest(),
                }
                for split in SPLITS
            }
            outputs["special_tokens.json"] = {
                "sha256": sha256_file(temporary_dir / "special_tokens.json")
            }
            outputs["stats.json"] = {
                "sha256": sha256_file(temporary_dir / "stats.json")
            }
            fingerprint_payload = {
                "schema_version": SCHEMA_VERSION,
                "variant": variant,
                "source_tiger_manifest_sha256": source_manifest_hash,
                "source_split_sha256": source_split_hashes,
                "tiger_identifier_mapping_sha256": tiger_mapping_hash,
                "tiger_identifier_manifest_sha256": tiger_identifier_manifest_hash,
                "ghr_identifier_manifest_sha256": lookup.manifest_sha256,
                "max_rows_per_split": max_rows_per_split,
            }
            build_fingerprint = hashlib.sha256(
                json.dumps(
                    fingerprint_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "task_name": TASK_NAME,
                "method": "GHR-SID",
                "variant": variant,
                "build_fingerprint": build_fingerprint,
                "source_tiger_sft": {
                    "path": str(source_dir),
                    "schema_version": source_manifest["schema_version"],
                    "manifest_sha256": source_manifest_hash,
                    "split_sha256": source_split_hashes,
                    "derivation": "one_to_one_identifier_only_remap",
                },
                "tiger_identifier": {
                    "path": str(tiger_identifier_dir.resolve()),
                    "mapping_sha256": tiger_mapping_hash,
                    "manifest_sha256": tiger_identifier_manifest_hash,
                },
                "ghr_identifier": {
                    "path": str(lookup.identifier_dir),
                    "schema_version": lookup.manifest["schema_version"],
                    "manifest_sha256": lookup.manifest_sha256,
                    "poi_count": len(lookup.poi_ids),
                    "protocol": lookup.manifest.get("protocol"),
                    "variable_length": True,
                    "query_or_order_used_to_build_identifier": False,
                },
                "alignment": {
                    "source_order_preserved": True,
                    "sample_id_preserved": True,
                    "split_preserved": True,
                    "metadata_preserved": True,
                    "raw_query_preserved": True,
                    "request_gid_preserved": True,
                    "history_length_preserved": True,
                    "changed_fields": [
                        "messages.history_poi_identifier",
                        "messages.target_poi_identifier",
                        "target_tiger_id_key->target_ghr_id_key",
                    ],
                    "row_counts": row_counts,
                    "history_identifier_occurrences_replaced": history_occurrences,
                },
                "input": {
                    "scan_mode": "full"
                    if max_rows_per_split is None
                    else "smoke_prefix",
                    "max_rows_per_split": max_rows_per_split,
                },
                "history": source_manifest["history"],
                "user_identifier": source_manifest["user_identifier"],
                "time_split": source_manifest["time_split"],
                "prompt": {
                    **source_manifest["prompt"],
                    "history_event_order": [
                        "request_geohash6_gid",
                        "raw_query",
                        "variable_length_ghr_poi_id",
                    ],
                    "target": "variable_length_unique_ghr_poi_id",
                },
                "processing_rules": source_manifest["processing_rules"],
                "stats": stats,
                "outputs": outputs,
            }
            _write_json(temporary_dir / "manifest.json", manifest)
            hashes = {
                filename: sha256_file(temporary_dir / filename)
                for filename in (
                    "train.jsonl",
                    "valid.jsonl",
                    "test.jsonl",
                    "special_tokens.json",
                    "stats.json",
                    "manifest.json",
                )
            }
            for split in SPLITS:
                if hashes[f"{split}.jsonl"] != outputs[f"{split}.jsonl"]["sha256"]:
                    raise GhrSftDataError(f"{variant} {split} 写入后 SHA256 不一致")
            manifests[variant] = manifest
            output_hashes[variant] = hashes
        for variant, output_dir in output_dirs.items():
            os.replace(temporary_dirs[variant], output_dir)
    except BaseException:
        for temporary_dir in temporary_dirs.values():
            if temporary_dir.exists():
                import shutil

                shutil.rmtree(temporary_dir)
        raise

    return PairedGhrSftResult(manifests=manifests, output_hashes=output_hashes,)

"""Build deterministic V1 order-to-Final-PID SFT data."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ...pid.dedup import sha256_file
from ...pid.geohash import (
    GEOHASH_ALPHABET,
    encode_geohash,
)
from ...sft.data import (
    PidLookup,
    SftDataError,
    TimeSplit,
    assistant_pid_content,
    discover_order_files,
    load_pid_lookup,
    stable_final_pid_key,
    stable_sample_id,
)


SFT_SCHEMA_VERSION = "sft-main-data-v1"
SPECIAL_TOKENS_SCHEMA_VERSION = "sft-special-tokens-v1"
TASK_NAME = "order_query_user_geohash6_to_unique_final_pid"
OUTPUT_FILENAMES = (
    "train.jsonl",
    "valid.jsonl",
    "test.jsonl",
    "special_tokens.json",
    "manifest.json",
    "stats.json",
)
REQUIRED_ORDER_FIELDS = (
    "order_id",
    "searchid",
    "query",
    "disp_lng",
    "disp_lat",
    "create_time",
    "source_dt",
    "poi_id",
)
PID_TOKEN_PATTERN = re.compile(r"<([^<>]+)>")
HEX_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


SftMainDataError = SftDataError


class SftDataValidationError(SftMainDataError):
    def __init__(
        self,
        message: str,
        *,
        anomaly_counts: Mapping[str, int] | None = None,
        examples: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.anomaly_counts = dict(anomaly_counts or {})
        self.examples = tuple(dict(example) for example in (examples or ()))

    def summary(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "anomaly_counts": self.anomaly_counts,
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class InputFile:
    path: Path
    relative_path: str
    sha256: str
    rows: int


@dataclass(frozen=True)
class PreflightResult:
    input_files: tuple[InputFile, ...]
    stats: dict[str, Any]
    build_fingerprint: str


@dataclass(frozen=True)
class SftMainDataResult:
    manifest: dict[str, Any]
    stats: dict[str, Any]
    output_hashes: dict[str, str]


class _AnomalyCollector:
    def __init__(self, max_examples: int = 10) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []
        self.max_examples = max_examples

    def add(self, kind: str, relative_path: str, line_number: int) -> None:
        self.counts[kind] += 1
        if len(self.examples) < self.max_examples:
            self.examples.append(
                {
                    "kind": kind,
                    "source": relative_path,
                    "line_number": line_number,
                }
            )

    def raise_if_any(self) -> None:
        if self.counts:
            raise SftDataValidationError(
                "订单预检发现不允许静默过滤的异常，未生成 SFT 数据",
                anomaly_counts=dict(sorted(self.counts.items())),
                examples=self.examples,
            )


def user_content(query: str, longitude: float, latitude: float) -> str:
    """Format the raw query and user Geohash6 without query normalization."""

    geohash = encode_geohash(longitude, latitude, 6)
    gid = "".join(f"<G_{character}>" for character in geohash)
    return f"<QUERY>{query}</QUERY>\n<USER_GID>{gid}</USER_GID>"


def _valid_coordinate(value: Any, minimum: float, maximum: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and minimum <= float(value) <= maximum
    )


def _parse_create_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    if parsed.strftime("%Y-%m-%d %H:%M:%S") != value:
        return None
    return parsed.date()


def _parse_source_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None
    if parsed.strftime("%Y%m%d") != value:
        return None
    return parsed


def _nearest_rank_from_histogram(
    histogram: Mapping[int, int], count: int, quantile: float
) -> int:
    target = max(1, math.ceil(quantile * count))
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= target:
            return int(value)
    raise SftMainDataError("Query 长度分位数计算失败")


def _fingerprint_payload(
    input_files: Sequence[InputFile],
    pid_mapping_sha256: str,
    pid_manifest_sha256: str,
    split: TimeSplit,
    geohash_length: int,
) -> dict[str, Any]:
    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "input_files": [
            {
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "rows": item.rows,
            }
            for item in input_files
        ],
        "pid_mapping_sha256": pid_mapping_sha256,
        "pid_manifest_sha256": pid_manifest_sha256,
        "time_split": split.to_manifest(),
        "geohash_length": geohash_length,
        "only_filter": "query_is_null_or_query_strip_is_empty",
    }


def preflight_orders(
    orders_dir: Path,
    order_files: Sequence[Path],
    pid_lookup: PidLookup,
    split: TimeSplit,
    *,
    pid_mapping_sha256: str,
    pid_manifest_sha256: str,
    geohash_length: int,
    progress: Callable[[str], None] | None = None,
) -> PreflightResult:
    """Scan all orders, count allowed blanks, and reject every other anomaly."""

    raw_order_count = 0
    blank_query_count = 0
    retained_sample_count = 0
    pid_matched_count = 0
    requires_dedup_count = 0
    unique_queries: set[str] = set()
    unique_target_pois: set[str] = set()
    length_histogram: Counter[int] = Counter()
    daily_counts = {
        current.isoformat(): 0
        for current in (
            date.fromordinal(ordinal)
            for ordinal in range(
                split.allowed_start.toordinal(),
                split.allowed_end.toordinal() + 1,
            )
        )
    }
    anomalies = _AnomalyCollector()
    input_files: list[InputFile] = []

    for file_index, path in enumerate(order_files, start=1):
        relative_path = path.relative_to(orders_dir).as_posix()
        digest = hashlib.sha256()
        file_rows = 0
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                raw_order_count += 1
                file_rows += 1
                try:
                    line = raw_line.decode("utf-8")
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    anomalies.add("invalid_json", relative_path, line_number)
                    continue
                if not isinstance(record, dict):
                    anomalies.add(
                        "record_not_object", relative_path, line_number
                    )
                    continue
                missing_fields = [
                    field
                    for field in REQUIRED_ORDER_FIELDS
                    if field not in record
                ]
                if missing_fields:
                    for field in missing_fields:
                        anomalies.add(
                            f"missing_{field}", relative_path, line_number
                        )
                    continue

                query = record["query"]
                if query is None or (
                    isinstance(query, str) and not query.strip()
                ):
                    blank_query_count += 1
                    continue
                if not isinstance(query, str):
                    anomalies.add(
                        "invalid_query_type", relative_path, line_number
                    )
                    continue
                retained_sample_count += 1
                unique_queries.add(query)
                length_histogram[len(query)] += 1

                row_has_error = False
                for field in ("order_id", "searchid"):
                    value = record[field]
                    if not isinstance(value, str):
                        anomalies.add(
                            f"invalid_{field}", relative_path, line_number
                        )
                        row_has_error = True
                poi_id = record["poi_id"]
                mapping_row: int | None = None
                if not isinstance(poi_id, str) or not poi_id:
                    anomalies.add(
                        "invalid_poi_id", relative_path, line_number
                    )
                    row_has_error = True
                else:
                    mapping_row = pid_lookup.row_by_poi_id.get(poi_id)
                    if mapping_row is None:
                        anomalies.add(
                            "unmatched_poi_id", relative_path, line_number
                        )
                        row_has_error = True
                    else:
                        pid_matched_count += 1

                if not _valid_coordinate(
                    record["disp_lng"], -180.0, 180.0
                ):
                    anomalies.add(
                        "invalid_disp_lng", relative_path, line_number
                    )
                    row_has_error = True
                if not _valid_coordinate(
                    record["disp_lat"], -90.0, 90.0
                ):
                    anomalies.add(
                        "invalid_disp_lat", relative_path, line_number
                    )
                    row_has_error = True

                create_date = _parse_create_date(record["create_time"])
                if create_date is None:
                    anomalies.add(
                        "invalid_create_time", relative_path, line_number
                    )
                    row_has_error = True
                elif not (
                    split.allowed_start <= create_date <= split.allowed_end
                ):
                    anomalies.add(
                        "create_date_out_of_range",
                        relative_path,
                        line_number,
                    )
                    row_has_error = True
                source_date = _parse_source_date(record["source_dt"])
                if source_date is None:
                    anomalies.add(
                        "invalid_source_dt", relative_path, line_number
                    )
                    row_has_error = True
                elif create_date is not None and source_date != create_date:
                    anomalies.add(
                        "source_dt_create_time_mismatch",
                        relative_path,
                        line_number,
                    )
                    row_has_error = True

                if (
                    not row_has_error
                    and create_date is not None
                    and mapping_row is not None
                ):
                    split.split_for(create_date)
                    daily_counts[create_date.isoformat()] += 1
                    unique_target_pois.add(poi_id)
                    requires_dedup_count += int(
                        pid_lookup.requires_dedup[mapping_row]
                    )
        input_files.append(
            InputFile(
                path=path.resolve(),
                relative_path=relative_path,
                sha256=digest.hexdigest(),
                rows=file_rows,
            )
        )
        if progress is not None:
            progress(
                f"订单预检：{file_index}/{len(order_files)} 个分片，"
                f"累计 {raw_order_count:,} 行"
            )

    anomalies.raise_if_any()
    if retained_sample_count <= 0:
        raise SftDataValidationError("过滤空 Query 后没有保留样本")
    if pid_matched_count != retained_sample_count:
        raise SftDataValidationError("PID 匹配数与保留样本数不一致")
    if sum(daily_counts.values()) != retained_sample_count:
        raise SftDataValidationError("按日样本数与保留样本数不守恒")

    stats = {
        "raw_order_count": raw_order_count,
        "blank_query_count": blank_query_count,
        "retained_sample_count": retained_sample_count,
        "pid_matched_count": pid_matched_count,
        "pid_match_ratio": pid_matched_count / retained_sample_count,
        "train_count": sum(
            count
            for value, count in daily_counts.items()
            if split.train_start <= date.fromisoformat(value) <= split.train_end
        ),
        "valid_count": daily_counts[split.valid_date.isoformat()],
        "test_count": daily_counts[split.test_date.isoformat()],
        "unique_query_count": len(unique_queries),
        "unique_target_poi_count": len(unique_target_pois),
        "unique_target_poi_coverage_ratio": (
            len(unique_target_pois) / pid_lookup.poi_count
        ),
        "requires_dedup_sample_count": requires_dedup_count,
        "requires_dedup_sample_ratio": (
            requires_dedup_count / retained_sample_count
        ),
        "query_char_length_p50": _nearest_rank_from_histogram(
            length_histogram, retained_sample_count, 0.50
        ),
        "query_char_length_p90": _nearest_rank_from_histogram(
            length_histogram, retained_sample_count, 0.90
        ),
        "query_char_length_p95": _nearest_rank_from_histogram(
            length_histogram, retained_sample_count, 0.95
        ),
        "query_char_length_p99": _nearest_rank_from_histogram(
            length_histogram, retained_sample_count, 0.99
        ),
        "query_char_length_max": max(length_histogram),
        "daily_sample_counts": daily_counts,
    }
    payload = _fingerprint_payload(
        input_files,
        pid_mapping_sha256,
        pid_manifest_sha256,
        split,
        geohash_length,
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return PreflightResult(
        input_files=tuple(input_files),
        stats=stats,
        build_fingerprint=fingerprint,
    )


def build_special_tokens() -> dict[str, Any]:
    """Build the complete fixed special-token inventory for this task."""

    structure_tokens = ["<QUERY>", "</QUERY>", "<USER_GID>", "</USER_GID>"]
    geohash_tokens = [
        f"<G_{character}>" for character in GEOHASH_ALPHABET
    ]
    sid_tokens = {
        f"s{level}": [
            f"<S{level}_{value}>" for value in range(1024)
        ]
        for level in range(1, 4)
    }
    dedup_tokens = [f"<D_{value}>" for value in range(512)]
    all_tokens = [
        *structure_tokens,
        *geohash_tokens,
        *sid_tokens["s1"],
        *sid_tokens["s2"],
        *sid_tokens["s3"],
        *dedup_tokens,
    ]
    if len(all_tokens) != len(set(all_tokens)):
        raise SftMainDataError("特殊 Token 表包含重复项")
    if "<D_-1>" in all_tokens:
        raise SftMainDataError("特殊 Token 表不得包含 <D_-1>")
    return {
        "schema_version": SPECIAL_TOKENS_SCHEMA_VERSION,
        "structure_tokens": structure_tokens,
        "geohash_tokens": geohash_tokens,
        "sid_tokens": sid_tokens,
        "dedup_tokens": dedup_tokens,
        "additional_special_tokens": all_tokens,
        "token_count": len(all_tokens),
    }


def _sample_from_record(
    record: Mapping[str, Any],
    relative_path: str,
    line_number: int,
    split_name: str,
    pid_lookup: PidLookup,
) -> dict[str, Any]:
    poi_id = record["poi_id"]
    mapping_row = pid_lookup.row_by_poi_id[poi_id]
    codes = pid_lookup.codes[mapping_row]
    dedup_code = int(pid_lookup.dedup_codes[mapping_row])
    requires_dedup = bool(pid_lookup.requires_dedup[mapping_row])
    if requires_dedup != (dedup_code >= 0):
        raise SftMainDataError("PID 映射 Dedup 状态不一致")
    return {
        "sample_id": stable_sample_id(relative_path, line_number),
        "messages": [
            {
                "role": "user",
                "content": user_content(
                    record["query"],
                    float(record["disp_lng"]),
                    float(record["disp_lat"]),
                ),
            },
            {
                "role": "assistant",
                "content": assistant_pid_content(codes, dedup_code),
            },
        ],
        "order_id": record["order_id"],
        "searchid": record["searchid"],
        "target_poi_id": poi_id,
        "target_pid_key": stable_final_pid_key(codes, dedup_code),
        "requires_dedup": requires_dedup,
        "split": split_name,
    }


def write_sft_jsonl(
    orders_dir: Path,
    input_files: Sequence[InputFile],
    output_paths: Mapping[str, Path],
    pid_lookup: PidLookup,
    split: TimeSplit,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Write deterministic split JSONL while preserving every nonblank row."""

    handles = {
        name: output_paths[name].open("w", encoding="utf-8", newline="\n")
        for name in ("train", "valid", "test")
    }
    counts = {"train": 0, "valid": 0, "test": 0}
    try:
        for file_index, item in enumerate(input_files, start=1):
            with item.path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    record = json.loads(line)
                    query = record["query"]
                    if query is None or not query.strip():
                        continue
                    create_date = _parse_create_date(record["create_time"])
                    if create_date is None:
                        raise SftMainDataError(
                            "预检后 create_time 解析结果发生变化"
                        )
                    split_name = split.split_for(create_date)
                    sample = _sample_from_record(
                        record,
                        item.relative_path,
                        line_number,
                        split_name,
                        pid_lookup,
                    )
                    handles[split_name].write(
                        json.dumps(
                            sample,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    counts[split_name] += 1
            if progress is not None:
                progress(
                    f"SFT 写出：{file_index}/{len(input_files)} 个分片，"
                    f"累计 {sum(counts.values()):,} 条"
                )
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def _parse_pid_content(content: Any) -> tuple[list[str], list[int], int]:
    if not isinstance(content, str) or not content:
        raise SftMainDataError("Assistant content 必须是非空字符串")
    matches = PID_TOKEN_PATTERN.findall(content)
    if "".join(f"<{value}>" for value in matches) != content:
        raise SftMainDataError("Assistant content 包含 PID Token 以外的字符")
    if len(matches) not in {9, 10}:
        raise SftMainDataError("Assistant PID 必须包含 9 或 10 个 Token")
    geohash_tokens = matches[:6]
    if any(
        len(token) != 3
        or not token.startswith("G_")
        or token[2] not in GEOHASH_ALPHABET
        for token in geohash_tokens
    ):
        raise SftMainDataError("Assistant GID Token 非法")
    sid_values: list[int] = []
    for level, token in enumerate(matches[6:9], start=1):
        prefix = f"S{level}_"
        if not token.startswith(prefix):
            raise SftMainDataError("Assistant SID Token 层级或顺序非法")
        try:
            value = int(token[len(prefix) :])
        except ValueError as error:
            raise SftMainDataError("Assistant SID Token 数值非法") from error
        if not 0 <= value < 1024:
            raise SftMainDataError("Assistant SID Token 超出范围")
        sid_values.append(value)
    dedup_code = -1
    if len(matches) == 10:
        token = matches[9]
        if not token.startswith("D_"):
            raise SftMainDataError("Assistant 第十个 Token 必须是 Dedup Token")
        try:
            dedup_code = int(token[2:])
        except ValueError as error:
            raise SftMainDataError("Assistant Dedup Token 数值非法") from error
        if not 0 <= dedup_code < 512:
            raise SftMainDataError("Assistant Dedup Token 超出范围")
    return geohash_tokens, sid_values, dedup_code


def validate_sft_outputs(
    output_paths: Mapping[str, Path],
    pid_lookup: PidLookup,
    expected_counts: Mapping[str, int],
) -> None:
    """Read every output line and verify Messages and mapped PID content."""

    observed_counts = {"train": 0, "valid": 0, "test": 0}
    for split_name in ("train", "valid", "test"):
        with output_paths[split_name].open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SftMainDataError(
                        f"{split_name}.jsonl 第 {line_number} 行 JSON 非法"
                    ) from error
                if not isinstance(sample, dict):
                    raise SftMainDataError("SFT 样本必须是 JSON object")
                if sample.get("split") != split_name:
                    raise SftMainDataError("SFT 样本 split 与文件不一致")
                sample_id = sample.get("sample_id")
                if not isinstance(sample_id, str) or not HEX_SHA256_PATTERN.fullmatch(
                    sample_id
                ):
                    raise SftMainDataError("sample_id 不是完整 SHA256")
                messages = sample.get("messages")
                if (
                    not isinstance(messages, list)
                    or len(messages) != 2
                    or messages[0].get("role") != "user"
                    or messages[1].get("role") != "assistant"
                ):
                    raise SftMainDataError(
                        "messages 必须严格包含一个 user 和一个 assistant"
                    )
                user_value = messages[0].get("content")
                if not isinstance(user_value, str):
                    raise SftMainDataError("User content 必须是字符串")
                marker = "</QUERY>\n<USER_GID>"
                if not user_value.startswith("<QUERY>") or marker not in user_value:
                    raise SftMainDataError("User content 模板非法")
                query, gid_suffix = user_value[len("<QUERY>") :].rsplit(
                    marker, 1
                )
                if not query.strip() or not gid_suffix.endswith("</USER_GID>"):
                    raise SftMainDataError("User Query 为空或 USER_GID 未闭合")
                gid_content = gid_suffix[: -len("</USER_GID>")]
                gid_tokens = PID_TOKEN_PATTERN.findall(gid_content)
                if (
                    len(gid_tokens) != 6
                    or "".join(f"<{value}>" for value in gid_tokens)
                    != gid_content
                    or any(
                        len(token) != 3
                        or not token.startswith("G_")
                        or token[2] not in GEOHASH_ALPHABET
                        for token in gid_tokens
                    )
                ):
                    raise SftMainDataError(
                        "用户位置必须是六个合法 Geohash Token"
                    )

                poi_id = sample.get("target_poi_id")
                mapping_row = (
                    pid_lookup.row_by_poi_id.get(poi_id)
                    if isinstance(poi_id, str)
                    else None
                )
                if mapping_row is None:
                    raise SftMainDataError("输出 target_poi_id 不在 PID 映射")
                mapped_codes = pid_lookup.codes[mapping_row]
                mapped_dedup = int(pid_lookup.dedup_codes[mapping_row])
                mapped_requires = bool(
                    pid_lookup.requires_dedup[mapping_row]
                )
                _, _, parsed_dedup = _parse_pid_content(
                    messages[1].get("content")
                )
                if parsed_dedup != mapped_dedup:
                    raise SftMainDataError("Assistant Dedup Token 与映射不一致")
                expected_assistant = assistant_pid_content(
                    mapped_codes, mapped_dedup
                )
                if messages[1].get("content") != expected_assistant:
                    raise SftMainDataError("Assistant PID 与正式映射不一致")
                expected_key = stable_final_pid_key(
                    mapped_codes, mapped_dedup
                )
                if sample.get("target_pid_key") != expected_key:
                    raise SftMainDataError("target_pid_key 与正式映射不一致")
                if sample.get("requires_dedup") is not mapped_requires:
                    raise SftMainDataError("requires_dedup 与正式映射不一致")
                if "<D_-1>" in messages[1].get("content"):
                    raise SftMainDataError("输出包含非法 <D_-1>")
                observed_counts[split_name] += 1
    if observed_counts != {
        name: int(expected_counts[f"{name}_count"])
        for name in ("train", "valid", "test")
    }:
        raise SftMainDataError("输出 JSONL 行数与预检统计不一致")
    if sum(observed_counts.values()) != expected_counts[
        "retained_sample_count"
    ]:
        raise SftMainDataError("三个切分行数之和与 retained_sample_count 不一致")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_output_directory(output_dir: Path) -> bool:
    created = False
    if output_dir.exists():
        if not output_dir.is_dir():
            raise SftMainDataError(f"输出路径不是目录：{output_dir}")
        unexpected = {
            path.name for path in output_dir.iterdir()
        } - set(OUTPUT_FILENAMES)
        if unexpected:
            raise SftMainDataError(
                "输出目录包含 SFT-DATA-001 范围外文件，不会覆盖："
                + ", ".join(sorted(unexpected))
            )
    else:
        output_dir.mkdir(parents=True)
        created = True
    return created


def _previous_built_at(
    output_dir: Path, build_fingerprint: str
) -> str | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if previous.get("build_fingerprint") != build_fingerprint:
        return None
    built_at = previous.get("built_at")
    return built_at if isinstance(built_at, str) and built_at else None


def build_sft_main_data(
    orders_dir: Path,
    pid_mapping_path: Path,
    pid_manifest_path: Path,
    output_dir: Path,
    split: TimeSplit,
    *,
    geohash_length: int = 6,
    progress: Callable[[str], None] | None = None,
) -> SftMainDataResult:
    """Build, validate, and atomically publish six deterministic SFT files."""

    orders_dir = orders_dir.resolve()
    pid_mapping_path = pid_mapping_path.resolve()
    pid_manifest_path = pid_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if geohash_length != 6:
        raise SftMainDataError("SFT-DATA-001 只允许 geohash_length=6")
    order_files = discover_order_files(orders_dir)
    if progress is not None:
        progress(
            f"发现 {len(order_files)} 个订单 JSONL 分片；加载唯一 POI-PID 映射"
        )
    (
        pid_lookup,
        pid_manifest,
        pid_mapping_sha256,
        pid_manifest_sha256,
    ) = load_pid_lookup(pid_mapping_path, pid_manifest_path)
    if progress is not None:
        progress("完整扫描订单并统计所有非空 Query 与异常")
    preflight = preflight_orders(
        orders_dir,
        order_files,
        pid_lookup,
        split,
        pid_mapping_sha256=pid_mapping_sha256,
        pid_manifest_sha256=pid_manifest_sha256,
        geohash_length=geohash_length,
        progress=progress,
    )

    created_output_dir = _validate_output_directory(output_dir)
    prior_built_at = _previous_built_at(
        output_dir, preflight.build_fingerprint
    )
    built_at = prior_built_at or datetime.now(timezone.utc).isoformat()
    suffix = f".{os.getpid()}.tmp"
    temporary = {
        filename: output_dir / f".{filename}{suffix}"
        for filename in OUTPUT_FILENAMES
    }
    split_paths = {
        name: temporary[f"{name}.jsonl"]
        for name in ("train", "valid", "test")
    }
    try:
        written_counts = write_sft_jsonl(
            orders_dir,
            preflight.input_files,
            split_paths,
            pid_lookup,
            split,
            progress=progress,
        )
        expected_counts = {
            name: preflight.stats[f"{name}_count"]
            for name in ("train", "valid", "test")
        }
        if written_counts != expected_counts:
            raise SftMainDataError("SFT 写出行数与预检统计不一致")
        if progress is not None:
            progress("逐行回读验证 Messages、用户 GID 与目标 Final PID")
        validate_sft_outputs(split_paths, pid_lookup, preflight.stats)

        special_tokens = build_special_tokens()
        _write_json(temporary["special_tokens.json"], special_tokens)
        _write_json(temporary["stats.json"], preflight.stats)
        output_hashes = {
            filename: sha256_file(temporary[filename])
            for filename in (
                "train.jsonl",
                "valid.jsonl",
                "test.jsonl",
                "special_tokens.json",
                "stats.json",
            )
        }
        manifest = {
            "schema_version": SFT_SCHEMA_VERSION,
            "task_name": TASK_NAME,
            "status": "completed",
            "built_at": built_at,
            "build_fingerprint": preflight.build_fingerprint,
            "orders": {
                "directory": str(orders_dir),
                "format": "json_lines",
                "file_count": len(preflight.input_files),
                "files": [
                    {
                        "relative_path": item.relative_path,
                        "rows": item.rows,
                        "sha256": item.sha256,
                    }
                    for item in preflight.input_files
                ],
                "actual_required_fields": list(REQUIRED_ORDER_FIELDS),
                "create_time_format": "YYYY-MM-DD HH:MM:SS",
                "source_dt_format": "YYYYMMDD",
            },
            "pid_mapping": {
                "path": str(pid_mapping_path),
                "sha256": pid_mapping_sha256,
                "poi_count": pid_lookup.poi_count,
            },
            "pid_manifest": {
                "path": str(pid_manifest_path),
                "sha256": pid_manifest_sha256,
                "schema_version": pid_manifest.get("schema_version"),
            },
            "geohash": {
                "algorithm": "standard_binary_interval_subdivision",
                "longitude_first": True,
                "alphabet": GEOHASH_ALPHABET,
                "length": geohash_length,
                "coordinate_fields": {
                    "longitude": "disp_lng",
                    "latitude": "disp_lat",
                },
                "coordinate_conversion": "none",
            },
            "templates": {
                "user": (
                    "<QUERY>{raw_query}</QUERY>\\n"
                    "<USER_GID><G_x1><G_x2><G_x3><G_x4><G_x5><G_x6>"
                    "</USER_GID>"
                ),
                "assistant_singleton": (
                    "<G_g1>...<G_g6><S1_x><S2_x><S3_x>"
                ),
                "assistant_dedup": (
                    "<G_g1>...<G_g6><S1_x><S2_x><S3_x><D_x>"
                ),
                "eos_written_to_content": False,
            },
            "time_split": split.to_manifest(),
            "processing_rules": {
                "only_filter": "query_is_null_or_query_strip_is_empty",
                "delete_duplicate_orders": False,
                "query_normalization": False,
                "query_length_filter": False,
                "include_time_features": False,
                "include_user_history": False,
                "include_poi_pid_alignment_task": False,
                "include_negative_samples": False,
                "sampling": False,
            },
            "outputs": {
                f"{name}.jsonl": {
                    "rows": preflight.stats[f"{name}_count"],
                    "sha256": output_hashes[f"{name}.jsonl"],
                }
                for name in ("train", "valid", "test")
            },
            "special_tokens": {
                "path": "special_tokens.json",
                "token_count": special_tokens["token_count"],
                "sha256": output_hashes["special_tokens.json"],
            },
            "stats": {
                "path": "stats.json",
                "sha256": output_hashes["stats.json"],
            },
            "validation": {
                "all_retained_queries_nonblank": True,
                "all_target_pois_in_mapping": True,
                "assistant_pid_matches_mapping": True,
                "singleton_pid_token_count": 9,
                "dedup_pid_token_count": 10,
                "contains_dedup_minus_one_token": False,
                "user_gid_token_count": 6,
                "time_splits_disjoint": True,
                "output_count_conservation": True,
                "all_lines_valid_json": True,
                "messages_roles_exact": True,
                "assistant_uses_only_valid_pid_tokens": True,
                "only_blank_queries_filtered": True,
                "sample_id_rule": (
                    "sha256(relative_source_path + ':' + one_based_line_number)"
                ),
            },
        }
        _write_json(temporary["manifest.json"], manifest)
        output_hashes["manifest.json"] = sha256_file(
            temporary["manifest.json"]
        )
        for filename in OUTPUT_FILENAMES:
            os.replace(temporary[filename], output_dir / filename)
    except Exception:
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        if (
            created_output_dir
            and output_dir.is_dir()
            and not any(output_dir.iterdir())
        ):
            output_dir.rmdir()
    return SftMainDataResult(
        manifest=manifest,
        stats=preflight.stats,
        output_hashes=output_hashes,
    )

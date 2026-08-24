"""Build deterministic Validation subsets for retrieval generalization audits."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..methods.genpoi.proximity import (
    GenPoiProximityError,
    extract_proximity_example,
)
from ..pid.trie import sha256_file
from .evaluation import atomic_write_json
from .geo_query_subset import normalize_query


GENERALIZATION_SUITE_SCHEMA_VERSION = "generalization-validation-suite-v1"
DEFAULT_SUBSET_SIZE = 10_000
DEFAULT_LONG_TAIL_MAX_FREQUENCY = 5
SUBSET_NAMES = (
    "seen_query_unseen_pair",
    "unseen_query_seen_target",
    "long_tail_target",
    "cold_target",
)
SUBSET_DESCRIPTIONS = {
    "seen_query_unseen_pair": (
        "归一化 Query 在 Train 出现、Query-POI 对未出现，且目标 POI "
        "Train 频次高于长尾阈值"
    ),
    "unseen_query_seen_target": (
        "归一化 Query 在 Train 未出现，且目标 POI Train 频次高于长尾阈值"
    ),
    "long_tail_target": "目标 POI Train 频次位于 [1, long_tail_max_frequency]",
    "cold_target": "目标 POI Train 频次为 0",
}


class GeneralizationSubsetError(ValueError):
    """Raised when the generalization-suite contract is violated."""


@dataclass(frozen=True)
class TrainMembershipIndex:
    """Hashed Train membership and exact target-order frequencies."""

    query_digests: set[bytes]
    pair_digests: set[bytes]
    target_frequencies: Counter[str]
    rows: int
    unique_queries: int
    unique_pairs: int
    manifest_sha256: str


@dataclass(frozen=True)
class GeneralizationCandidate:
    """One eligible Validation row and its Train-membership annotations."""

    row_index: int
    sample_id: str
    order_id: str
    searchid: str
    target_poi_id: str
    normalized_query: str
    query_seen: bool
    pair_seen: bool
    target_frequency: int
    selection_hash: str
    raw_line: bytes


@dataclass(frozen=True)
class GeneralizationSuite:
    """Frozen generalization-suite artifacts."""

    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def normalized_query_digest(normalized_query: str) -> bytes:
    """Return a stable SHA256 digest for one already-normalized Query."""

    return hashlib.sha256(normalized_query.encode("utf-8")).digest()


def normalized_pair_digest(normalized_query: str, target_poi_id: str) -> bytes:
    """Return a stable SHA256 digest for one normalized Query-POI pair."""

    value = normalized_query.encode("utf-8") + b"\0" + target_poi_id.encode("utf-8")
    return hashlib.sha256(value).digest()


def classify_generalization_row(
    *,
    query_seen: bool,
    pair_seen: bool,
    target_frequency: int,
    long_tail_max_frequency: int,
) -> str | None:
    """Assign one row to at most one mutually exclusive generalization stratum."""

    if target_frequency < 0:
        raise GeneralizationSubsetError("目标 POI Train 频次不得为负数")
    if long_tail_max_frequency <= 0:
        raise GeneralizationSubsetError("长尾最大频次必须为正整数")
    if pair_seen and not query_seen:
        raise GeneralizationSubsetError("Query-POI 对已见时 Query 不可能未见")
    if target_frequency == 0:
        return "cold_target"
    if target_frequency <= long_tail_max_frequency:
        return "long_tail_target"
    if query_seen and not pair_seen:
        return "seen_query_unseen_pair"
    if not query_seen:
        return "unseen_query_seen_target"
    return None


def stable_subset_selection_hash(
    subset_name: str,
    order_id: str,
    searchid: str,
    target_poi_id: str,
) -> str:
    """Hash immutable keys for deterministic within-stratum sampling."""

    value = (
        f"{subset_name}\0{order_id}\0{searchid}\0{target_poi_id}".encode("utf-8")
    )
    return hashlib.sha256(value).hexdigest()


def _required_text(record: Mapping[str, Any], field: str, source: str) -> str:
    value = record.get(field)
    if value is None or not str(value).strip():
        raise GeneralizationSubsetError(f"{source} 缺少有效 {field}")
    return str(value)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise GeneralizationSubsetError(f"{label} 不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GeneralizationSubsetError(f"{label} JSON 非法：{path}") from error
    if not isinstance(value, dict):
        raise GeneralizationSubsetError(f"{label} 必须是 JSON object：{path}")
    return value


def load_train_membership_index(
    train_query_shards_dir: Path,
    *,
    verify_hash: bool = True,
) -> TrainMembershipIndex:
    """Load normalized Query/pair membership from frozen Train Parquet shards."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise GeneralizationSubsetError("构建泛化集需要安装 pyarrow") from error

    train_query_shards_dir = train_query_shards_dir.resolve()
    manifest_path = train_query_shards_dir / "manifest.json"
    manifest = _load_json_object(manifest_path, "Train Query 分片 manifest")
    if manifest.get("schema_version") != "train-query-hash-shards-v1":
        raise GeneralizationSubsetError("Train Query 分片 schema_version 不兼容")
    if manifest.get("status") != "completed":
        raise GeneralizationSubsetError("Train Query 分片状态不是 completed")
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping) or contract.get("split") != "train":
        raise GeneralizationSubsetError("Train Query 分片未声明 train-only 契约")
    if contract.get("output_fields") != ["query", "target_poi_id"]:
        raise GeneralizationSubsetError("Train Query 分片字段契约不兼容")
    sharding = manifest.get("sharding")
    files = sharding.get("files") if isinstance(sharding, Mapping) else None
    total_rows = sharding.get("total_rows") if isinstance(sharding, Mapping) else None
    if not isinstance(files, list) or not files or not isinstance(total_rows, int):
        raise GeneralizationSubsetError("Train Query 分片 manifest 缺少文件或行数")

    query_digests: set[bytes] = set()
    pair_digests: set[bytes] = set()
    target_frequencies: Counter[str] = Counter()
    rows = 0
    for file_index, file_spec in enumerate(files):
        if not isinstance(file_spec, Mapping):
            raise GeneralizationSubsetError("Train Query 分片文件声明必须是 object")
        filename = file_spec.get("file")
        expected_rows = file_spec.get("rows")
        expected_sha256 = file_spec.get("sha256")
        if (
            not isinstance(filename, str)
            or not isinstance(expected_rows, int)
            or not isinstance(expected_sha256, str)
        ):
            raise GeneralizationSubsetError("Train Query 分片文件声明不完整")
        path = train_query_shards_dir / filename
        if not path.is_file():
            raise GeneralizationSubsetError(f"Train Query 分片不存在：{path}")
        if verify_hash and sha256_file(path) != expected_sha256:
            raise GeneralizationSubsetError(f"Train Query 分片 SHA256 不一致：{path}")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow.names != ["query", "target_poi_id"]:
            raise GeneralizationSubsetError(f"Train Query 分片字段不兼容：{path}")
        file_rows = 0
        for batch in parquet.iter_batches(
            batch_size=65_536,
            columns=["query", "target_poi_id"],
        ):
            queries = batch.column(0).to_pylist()
            target_poi_ids = batch.column(1).to_pylist()
            for query, target_poi_id in zip(queries, target_poi_ids, strict=True):
                if not isinstance(query, str) or not query.strip():
                    raise GeneralizationSubsetError(
                        f"Train Query 分片 {file_index} 包含空 Query"
                    )
                if not isinstance(target_poi_id, str) or not target_poi_id:
                    raise GeneralizationSubsetError(
                        f"Train Query 分片 {file_index} 包含空目标 POI"
                    )
                normalized = normalize_query(query)
                query_digests.add(normalized_query_digest(normalized))
                pair_digests.add(normalized_pair_digest(normalized, target_poi_id))
                target_frequencies[target_poi_id] += 1
            file_rows += len(queries)
        if file_rows != expected_rows:
            raise GeneralizationSubsetError(
                f"Train Query 分片 {filename} 行数 {file_rows:,} != {expected_rows:,}"
            )
        rows += file_rows
    if rows != total_rows:
        raise GeneralizationSubsetError(
            f"Train Query 分片总行数 {rows:,} != manifest {total_rows:,}"
        )
    return TrainMembershipIndex(
        query_digests=query_digests,
        pair_digests=pair_digests,
        target_frequencies=target_frequencies,
        rows=rows,
        unique_queries=len(query_digests),
        unique_pairs=len(pair_digests),
        manifest_sha256=sha256_file(manifest_path),
    )


def _load_reference_keys(
    data_path: Path,
    manifest_path: Path,
    *,
    subset_size: int,
    label: str,
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    data_path = data_path.resolve()
    manifest_path = manifest_path.resolve()
    manifest = _load_json_object(manifest_path, f"{label} manifest")
    if manifest.get("status") != "completed":
        raise GeneralizationSubsetError(f"{label} manifest 状态不是 completed")
    data_sha256 = sha256_file(data_path)
    declared_sha256 = manifest.get("output_sha256")
    if declared_sha256 is None:
        output = manifest.get("output")
        declared_sha256 = output.get("sha256") if isinstance(output, Mapping) else None
    if declared_sha256 != data_sha256:
        raise GeneralizationSubsetError(f"{label} 数据 SHA256 与 manifest 不一致")
    keys: set[tuple[str, str]] = set()
    sample_ids: set[str] = set()
    key_order: list[tuple[str, str]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise GeneralizationSubsetError(
                    f"{label} 第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(record, Mapping):
                raise GeneralizationSubsetError(f"{label} 每行必须是 object")
            source = f"{label} 第 {line_number} 行"
            key = (
                _required_text(record, "order_id", source),
                _required_text(record, "searchid", source),
            )
            sample_id = _required_text(record, "sample_id", source)
            if key in keys or sample_id in sample_ids:
                raise GeneralizationSubsetError(f"{label} 存在重复业务键或 sample_id")
            keys.add(key)
            sample_ids.add(sample_id)
            key_order.append(key)
    if len(keys) != subset_size:
        raise GeneralizationSubsetError(
            f"{label} 行数 {len(keys):,} != 预期 {subset_size:,}"
        )
    key_bytes = "".join(f"{left}\t{right}\n" for left, right in key_order).encode(
        "utf-8"
    )
    return keys, {
        "data_file": str(data_path),
        "data_sha256": data_sha256,
        "manifest_file": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": subset_size,
        "business_keys_sha256": hashlib.sha256(key_bytes).hexdigest(),
    }


def _push_candidate(
    heap: list[tuple[int, int, GeneralizationCandidate]],
    candidate: GeneralizationCandidate,
    subset_size: int,
) -> None:
    priority = int(candidate.selection_hash, 16)
    entry = (-priority, -candidate.row_index, candidate)
    if len(heap) < subset_size:
        heapq.heappush(heap, entry)
        return
    worst_priority = -heap[0][0]
    worst_row_index = -heap[0][1]
    if (priority, candidate.row_index) < (worst_priority, worst_row_index):
        heapq.heapreplace(heap, entry)


def _selected_candidates(
    heap: list[tuple[int, int, GeneralizationCandidate]],
) -> list[GeneralizationCandidate]:
    return sorted((entry[2] for entry in heap), key=lambda item: item.row_index)


def _write_subset(
    output_path: Path,
    selected: list[GeneralizationCandidate],
) -> str:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as destination:
        for candidate in selected:
            destination.write(candidate.raw_line)
            digest.update(candidate.raw_line)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, output_path)
    return digest.hexdigest()


def _validate_existing_suite(
    output_dir: Path,
    manifest_path: Path,
    *,
    subset_size: int,
    source_sha256: str,
    train_manifest_sha256: str,
    random_reference_sha256: str,
    geo_reference_sha256: str,
    long_tail_max_frequency: int,
) -> GeneralizationSuite:
    manifest = _load_json_object(manifest_path, "泛化专项集 suite manifest")
    expected = {
        "schema_version": GENERALIZATION_SUITE_SCHEMA_VERSION,
        "status": "completed",
        "split": "valid",
        "subset_size_each": subset_size,
        "source_sha256": source_sha256,
        "train_query_shards_manifest_sha256": train_manifest_sha256,
        "long_tail_max_frequency": long_tail_max_frequency,
        "normalization": "NFKC_lower_remove_whitespace",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise GeneralizationSubsetError("已有泛化专项集 manifest 与当前协议不一致")
    references = manifest.get("references")
    if not isinstance(references, Mapping):
        raise GeneralizationSubsetError("已有泛化专项集 manifest 缺少 references")
    if references.get("random_traffic", {}).get("data_sha256") != random_reference_sha256:
        raise GeneralizationSubsetError("已有 suite 的随机参考集已经变化")
    if references.get("geo_complex", {}).get("data_sha256") != geo_reference_sha256:
        raise GeneralizationSubsetError("已有 suite 的地理参考集已经变化")
    subsets = manifest.get("subsets")
    if not isinstance(subsets, Mapping):
        raise GeneralizationSubsetError("已有泛化专项集 manifest 缺少 subsets")
    for subset_name in SUBSET_NAMES:
        spec = subsets.get(subset_name)
        if not isinstance(spec, Mapping):
            raise GeneralizationSubsetError(f"已有 suite 缺少 {subset_name}")
        path = Path(str(spec.get("output_file", "")))
        if not path.is_file() or sha256_file(path) != spec.get("output_sha256"):
            raise GeneralizationSubsetError(f"已有 {subset_name} 文件或 SHA256 不一致")
        with path.open("rb") as handle:
            rows = sum(1 for _ in handle)
        if rows != subset_size or spec.get("output_rows") != subset_size:
            raise GeneralizationSubsetError(f"已有 {subset_name} 行数不一致")
    return GeneralizationSuite(
        output_dir=output_dir,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def build_generalization_validation_suite(
    valid_file: Path,
    train_query_shards_dir: Path,
    random_subset_file: Path,
    random_subset_manifest: Path,
    geo_subset_file: Path,
    geo_subset_manifest: Path,
    output_dir: Path,
    *,
    source_rows: int,
    source_sha256: str,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    long_tail_max_frequency: int = DEFAULT_LONG_TAIL_MAX_FREQUENCY,
    verify_train_shard_hashes: bool = True,
) -> GeneralizationSuite:
    """Freeze four disjoint Validation generalization subsets and suite metadata."""

    if subset_size <= 0 or subset_size > source_rows:
        raise GeneralizationSubsetError("每个专项集大小必须位于 (0, source_rows]")
    if long_tail_max_frequency <= 0:
        raise GeneralizationSubsetError("长尾最大频次必须为正整数")
    valid_file = valid_file.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "suite_manifest.json"

    train_index = load_train_membership_index(
        train_query_shards_dir,
        verify_hash=verify_train_shard_hashes,
    )
    random_keys, random_reference = _load_reference_keys(
        random_subset_file,
        random_subset_manifest,
        subset_size=subset_size,
        label="原随机 Validation 集",
    )
    geo_keys, geo_reference = _load_reference_keys(
        geo_subset_file,
        geo_subset_manifest,
        subset_size=subset_size,
        label="复杂地理 Validation 集",
    )
    if manifest_path.exists():
        return _validate_existing_suite(
            output_dir,
            manifest_path,
            subset_size=subset_size,
            source_sha256=source_sha256,
            train_manifest_sha256=train_index.manifest_sha256,
            random_reference_sha256=random_reference["data_sha256"],
            geo_reference_sha256=geo_reference["data_sha256"],
            long_tail_max_frequency=long_tail_max_frequency,
        )
    expected_outputs = [
        output_dir / f"{subset_name}_{subset_size}.jsonl"
        for subset_name in SUBSET_NAMES
    ]
    if any(path.exists() for path in expected_outputs):
        raise GeneralizationSubsetError("输出目录存在无 suite manifest 的专项集文件")

    excluded_keys = random_keys | geo_keys
    heaps: dict[str, list[tuple[int, int, GeneralizationCandidate]]] = {
        name: [] for name in SUBSET_NAMES
    }
    eligible_counts: Counter[str] = Counter()
    excluded_rows = 0
    actual_source_rows = 0
    source_digest = hashlib.sha256()
    with valid_file.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            actual_source_rows += 1
            source_digest.update(raw_line)
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise GeneralizationSubsetError(
                    f"Validation 第 {line_number} 行 JSON 非法"
                ) from error
            if not isinstance(record, Mapping):
                raise GeneralizationSubsetError("Validation 每行必须是 object")
            try:
                example = extract_proximity_example(record, expected_split="valid")
            except GenPoiProximityError as error:
                raise GeneralizationSubsetError(
                    f"Validation 第 {line_number} 行格式非法：{error}"
                ) from error
            source = f"Validation 第 {line_number} 行"
            order_id = _required_text(record, "order_id", source)
            searchid = _required_text(record, "searchid", source)
            target_poi_id = _required_text(record, "target_poi_id", source)
            key = (order_id, searchid)
            if key in excluded_keys:
                excluded_rows += 1
                continue
            normalized = normalize_query(example.query)
            query_seen = normalized_query_digest(normalized) in train_index.query_digests
            pair_seen = (
                normalized_pair_digest(normalized, target_poi_id)
                in train_index.pair_digests
            )
            target_frequency = train_index.target_frequencies[target_poi_id]
            subset_name = classify_generalization_row(
                query_seen=query_seen,
                pair_seen=pair_seen,
                target_frequency=target_frequency,
                long_tail_max_frequency=long_tail_max_frequency,
            )
            if subset_name is None:
                continue
            eligible_counts[subset_name] += 1
            candidate = GeneralizationCandidate(
                row_index=line_number - 1,
                sample_id=example.sample_id,
                order_id=order_id,
                searchid=searchid,
                target_poi_id=target_poi_id,
                normalized_query=normalized,
                query_seen=query_seen,
                pair_seen=pair_seen,
                target_frequency=target_frequency,
                selection_hash=stable_subset_selection_hash(
                    subset_name,
                    order_id,
                    searchid,
                    target_poi_id,
                ),
                raw_line=raw_line,
            )
            _push_candidate(heaps[subset_name], candidate, subset_size)
    if actual_source_rows != source_rows:
        raise GeneralizationSubsetError(
            f"Validation 实际行数 {actual_source_rows:,} != manifest {source_rows:,}"
        )
    if source_digest.hexdigest() != source_sha256:
        raise GeneralizationSubsetError("Validation SHA256 与 manifest 不一致")
    insufficient = {
        name: eligible_counts[name]
        for name in SUBSET_NAMES
        if eligible_counts[name] < subset_size
    }
    if insufficient:
        raise GeneralizationSubsetError(f"专项集候选不足：{insufficient}")

    selected_by_name = {
        name: _selected_candidates(heaps[name]) for name in SUBSET_NAMES
    }
    all_new_keys: set[tuple[str, str]] = set()
    for subset_name, selected in selected_by_name.items():
        sample_ids = {item.sample_id for item in selected}
        keys = {(item.order_id, item.searchid) for item in selected}
        if len(sample_ids) != subset_size or len(keys) != subset_size:
            raise GeneralizationSubsetError(f"{subset_name} 存在重复样本或业务键")
        if keys & excluded_keys or keys & all_new_keys:
            raise GeneralizationSubsetError(f"{subset_name} 与其他 suite 集合发生重叠")
        all_new_keys.update(keys)

    subsets: dict[str, dict[str, Any]] = {}
    key_sets: dict[str, set[tuple[str, str]]] = {
        "random_traffic": random_keys,
        "geo_complex": geo_keys,
    }
    for subset_name, selected in selected_by_name.items():
        output_path = output_dir / f"{subset_name}_{subset_size}.jsonl"
        output_sha256 = _write_subset(output_path, selected)
        key_order = [(item.order_id, item.searchid) for item in selected]
        sample_id_bytes = "".join(f"{item.sample_id}\n" for item in selected).encode(
            "utf-8"
        )
        key_bytes = "".join(
            f"{order_id}\t{searchid}\n" for order_id, searchid in key_order
        ).encode("utf-8")
        target_frequencies = [item.target_frequency for item in selected]
        subsets[subset_name] = {
            "description": SUBSET_DESCRIPTIONS[subset_name],
            "eligible_rows_after_reference_exclusion": eligible_counts[subset_name],
            "sampling_rate": subset_size / eligible_counts[subset_name],
            "selection_method": (
                "lowest_sha256(subset_name\\0order_id\\0searchid\\0target_poi_id)"
            ),
            "output_file": str(output_path),
            "output_rows": subset_size,
            "output_sha256": output_sha256,
            "business_keys_sha256": hashlib.sha256(key_bytes).hexdigest(),
            "selected_sample_ids_sha256": hashlib.sha256(sample_id_bytes).hexdigest(),
            "unique_normalized_queries": len(
                {item.normalized_query for item in selected}
            ),
            "unique_target_pois": len({item.target_poi_id for item in selected}),
            "query_seen_rows": sum(item.query_seen for item in selected),
            "pair_seen_rows": sum(item.pair_seen for item in selected),
            "target_frequency_min": min(target_frequencies),
            "target_frequency_max": max(target_frequencies),
        }
        key_sets[subset_name] = set(key_order)

    overlap_matrix: dict[str, dict[str, int]] = {}
    for left_name, left_keys in key_sets.items():
        overlap_matrix[left_name] = {
            right_name: len(left_keys & right_keys)
            for right_name, right_keys in key_sets.items()
        }
    manifest = {
        "schema_version": GENERALIZATION_SUITE_SCHEMA_VERSION,
        "status": "completed",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "split": "valid",
        "date": "2026-07-13",
        "source_file": str(valid_file),
        "source_rows": source_rows,
        "source_sha256": source_sha256,
        "subset_size_each": subset_size,
        "normalization": "NFKC_lower_remove_whitespace",
        "membership_digest": "SHA256(normalized Query or normalized Query\\0target_poi_id)",
        "target_frequency_unit": "Train positive-order rows per target_poi_id",
        "long_tail_max_frequency": long_tail_max_frequency,
        "established_target_min_frequency": long_tail_max_frequency + 1,
        "train_query_shards_dir": str(train_query_shards_dir.resolve()),
        "train_query_shards_manifest_sha256": train_index.manifest_sha256,
        "train_rows": train_index.rows,
        "train_unique_normalized_queries": train_index.unique_queries,
        "train_unique_normalized_query_target_pairs": train_index.unique_pairs,
        "train_unique_target_pois": len(train_index.target_frequencies),
        "reference_excluded_rows": excluded_rows,
        "reference_union_business_keys": len(excluded_keys),
        "references": {
            "random_traffic": random_reference,
            "geo_complex": geo_reference,
        },
        "subsets": subsets,
        "business_key_overlap_matrix": overlap_matrix,
        "notes": [
            "四个新专项集互斥，并排除原随机与复杂地理参考集。",
            "原随机与复杂地理参考集保持历史冻结版本，二者已有重叠不会被重写。",
            "未进入四个专项集的已见高频 Query-POI 行仍由原随机流量集覆盖。",
        ],
    }
    atomic_write_json(manifest_path, manifest)
    return GeneralizationSuite(
        output_dir=output_dir,
        manifest_path=manifest_path,
        manifest=manifest,
    )

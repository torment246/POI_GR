"""Validate the method's frozen input contracts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from qg_prqk.artifacts import build_manifest_template, sha256_file
from qg_prqk.config import QGPRQKConfig, QGPRQKConfigError, load_config
from qg_prqk.data.contracts import (
    QGPRQKDataContractError,
    load_json_object,
    validate_aligned_row_counts,
    validate_embedding_manifest,
    validate_poi_record,
    validate_sft_manifest,
    validate_train_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QG-PRQK 配置与真实输入契约核验。"
    )
    parser.add_argument("--config", type=Path, required=True, help="QG v1.1 YAML 配置。")
    parser.add_argument("--output-dir", type=Path, help="覆盖隔离的 QG 输出目录。")
    parser.add_argument("--seed", type=int, help="覆盖随机种子。")
    parser.add_argument(
        "--limit",
        "--sample-size",
        dest="sample_limit",
        type=int,
        help="只检查前 N 条 POI、POI ID 和 Train 记录。",
    )
    parser.add_argument("--resume", action="store_true", help="仅解析并记录 resume。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖保护开关，默认关闭。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读检查配置、manifest、NPY header 和有限样例。",
    )
    return parser


def _iter_json_records(paths: list[Path], limit: int) -> Iterator[Mapping[str, Any]]:
    seen = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise QGPRQKDataContractError(f"{path}:{line_number} 是空行")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise QGPRQKDataContractError(
                        f"{path}:{line_number} JSON 非法"
                    ) from error
                if not isinstance(record, Mapping):
                    raise QGPRQKDataContractError(
                        f"{path}:{line_number} 必须是 JSON object"
                    )
                yield record
                seen += 1
                if seen >= limit:
                    return


def _read_poi_ids(path: Path, limit: int) -> list[str]:
    values: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(values) >= limit:
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise QGPRQKDataContractError(
                    f"{path}:{line_number} POI ID JSON 非法"
                ) from error
            if not isinstance(value, str) or not value.strip():
                raise QGPRQKDataContractError(
                    f"{path}:{line_number} POI ID 必须是非空字符串"
                )
            values.append(value)
    if len(values) != limit:
        raise QGPRQKDataContractError(
            f"POI ID 样例不足：期望 {limit}，实际 {len(values)}"
        )
    return values


def run_dry_run(config: QGPRQKConfig) -> dict[str, Any]:
    """Validate frozen manifests, NPY metadata, and bounded real samples."""

    for path, label in (
        (config.paths.poi_catalog, "POI catalog"),
        (config.paths.poi_embeddings, "POI embeddings"),
        (config.paths.poi_ids, "POI IDs"),
        (config.paths.embedding_manifest, "Embedding manifest"),
        (config.paths.sft_data_dir, "SFT data"),
        (config.paths.sft_manifest, "SFT manifest"),
    ):
        if not path.exists():
            raise QGPRQKDataContractError(f"{label} 不存在：{path}")

    embedding_manifest = load_json_object(
        config.paths.embedding_manifest, "Embedding manifest"
    )
    if sha256_file(config.paths.embedding_manifest) != (
        config.frozen_inputs.embedding_manifest_sha256
    ):
        raise QGPRQKDataContractError("Embedding manifest SHA256 与冻结配置不一致")
    declared_rows, declared_dim, declared_dtype = validate_embedding_manifest(
        embedding_manifest, config.data_contracts, config.frozen_inputs
    )
    embeddings = np.load(config.paths.poi_embeddings, mmap_mode="r")
    if tuple(embeddings.shape) != (declared_rows, declared_dim):
        raise QGPRQKDataContractError("Embedding NPY shape 与 manifest 不一致")
    if str(embeddings.dtype) != declared_dtype:
        raise QGPRQKDataContractError("Embedding NPY dtype 与 manifest 不一致")

    sft_manifest = load_json_object(config.paths.sft_manifest, "SFT manifest")
    if sha256_file(config.paths.sft_manifest) != config.frozen_inputs.sft_manifest_sha256:
        raise QGPRQKDataContractError("SFT manifest SHA256 与冻结配置不一致")
    split_rows = validate_sft_manifest(
        sft_manifest, config.data_contracts, config.frozen_inputs
    )

    limit = min(config.runtime.sample_limit, declared_rows, split_rows["train.jsonl"])
    poi_paths = sorted(config.paths.poi_catalog.glob("part-*.json"))
    if not poi_paths:
        raise QGPRQKDataContractError("POI catalog 中没有 part-*.json")
    poi_ids = _read_poi_ids(config.paths.poi_ids, limit)
    sampled_poi_ids: list[str] = []
    for index, record in enumerate(_iter_json_records(poi_paths, limit), start=1):
        sampled_poi_ids.append(
            validate_poi_record(record, config.data_contracts, f"POI sample {index}")
        )
    if sampled_poi_ids != poi_ids:
        raise QGPRQKDataContractError("POI catalog 与 embedding POI ID 样例行序不一致")

    train_path = config.paths.sft_data_dir / "train.jsonl"
    train_samples = list(_iter_json_records([train_path], limit))
    for index, record in enumerate(train_samples, start=1):
        validate_train_record(record, config.data_contracts, f"Train sample {index}")

    validate_aligned_row_counts(
        poi_rows=config.data_contracts.expected_poi_rows,
        embedding_rows=declared_rows,
        poi_id_rows=declared_rows,
    )
    checks = {
        "status": "passed",
        "embedding_shape": [declared_rows, declared_dim],
        "embedding_dtype": declared_dtype,
        "declared_split_rows": split_rows,
        "sampled_poi_rows": len(sampled_poi_ids),
        "sampled_poi_id_rows": len(poi_ids),
        "sampled_train_rows": len(train_samples),
        "full_poi_ids_sha256": "frozen_not_rehashed_in_dry_run",
        "full_data_scan": False,
    }
    return {
        "status": "dry_run_passed",
        "checks": checks,
        "manifest_template": build_manifest_template(
            config, sample_limit=limit, dry_run_checks=checks
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("输入契约核验命令必须显式提供 --dry-run")
    try:
        config = load_config(
            args.config,
            output_dir=args.output_dir,
            seed=args.seed,
            sample_limit=args.sample_limit,
            resume=True if args.resume else None,
            overwrite=True if args.overwrite else None,
        )
        result = run_dry_run(config)
    except (QGPRQKConfigError, QGPRQKDataContractError, OSError, ValueError) as error:
        print(f"QG-PRQK preflight failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

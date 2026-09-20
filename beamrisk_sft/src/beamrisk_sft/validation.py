"""Independent structural and leakage validation for finalized risk pairs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .candidate_pool import load_validation_business_keys
from .config import BeamRiskConfig
from .errors import BeamRiskError
from .io import iter_jsonl, read_json, sha256_file
from .schema import RiskPair


def validate_risk_pairs(
    config: BeamRiskConfig,
    *,
    reference_epoch: int,
    require_minimum: bool = True,
) -> dict[str, Any]:
    output_dir = config.mining_dir(reference_epoch)
    manifest = read_json(output_dir / "manifest.json", name="Mining manifest")
    path = output_dir / "risk_pairs.jsonl"
    if manifest.get("status") != "completed":
        raise BeamRiskError("Mining manifest 状态不是 completed")
    if manifest.get("reference_epoch") != reference_epoch:
        raise BeamRiskError("Mining manifest reference_epoch 不一致")
    if sha256_file(path) != manifest.get("risk_pairs_sha256"):
        raise BeamRiskError("risk_pairs.jsonl SHA256 与 manifest 不一致")
    validation_keys = load_validation_business_keys(
        config.paths.fixed_validation_subset
    )
    sample_ids: set[str] = set()
    request_keys: set[str] = set()
    risk_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    selected_final_rank_catalog_counts: Counter[str] = Counter()
    model_config = read_json(config.paths.model / "config.json", name="Model config")
    vocab_size = model_config.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise BeamRiskError("Model config vocab_size 无效")
    max_token_id = -1
    max_prompt_plus_path = 0
    rows = 0
    for _, value in iter_jsonl(path):
        pair = RiskPair.from_mapping(value)
        rows += 1
        if pair.sample_id in sample_ids:
            raise BeamRiskError(f"风险 pair sample_id 重复：{pair.sample_id}")
        sample_ids.add(pair.sample_id)
        if pair.key in request_keys:
            raise BeamRiskError(f"风险 pair 业务键重复：{pair.key}")
        request_keys.add(pair.key)
        if pair.key in validation_keys:
            raise BeamRiskError(f"风险 pair 与固定 Validation 泄漏：{pair.key}")
        if len(pair.prompt_token_ids) + len(pair.positive_token_ids) > config.training.cutoff_len:
            raise BeamRiskError(f"风险路径超过 cutoff_len：{pair.sample_id}")
        pair_max_token = max(
            *pair.prompt_token_ids,
            *pair.positive_token_ids,
            *pair.negative_token_ids,
        )
        if pair_max_token >= vocab_size:
            raise BeamRiskError(
                f"风险路径 Token {pair_max_token} 超出模型词表 {vocab_size}："
                f"{pair.sample_id}"
            )
        max_token_id = max(max_token_id, pair_max_token)
        max_prompt_plus_path = max(
            max_prompt_plus_path,
            len(pair.prompt_token_ids) + len(pair.positive_token_ids),
        )
        if pair.strict_duplicate_filtered:
            raise BeamRiskError("最终训练集不得保留 strict_duplicate_filtered pair")
        risk_counts[pair.risk_type] += 1
        if pair.risk_type == "final_rank":
            if not pair.negative_structure_valid:
                selected_final_rank_catalog_counts["structure_invalid"] += 1
            elif not pair.negative_catalog_expandable:
                selected_final_rank_catalog_counts["catalog_unexpandable"] += 1
            else:
                selected_final_rank_catalog_counts["catalog_expandable"] += 1
        if pair.first_prune_depth is not None:
            depth_counts[str(pair.first_prune_depth)] += 1
    if rows != manifest.get("pairs"):
        raise BeamRiskError(f"风险 pair 行数 {rows} != manifest {manifest.get('pairs')}")
    if require_minimum and rows < config.mining.minimum_pair_count:
        raise BeamRiskError(
            f"风险 pair {rows:,} 低于门禁 {config.mining.minimum_pair_count:,}"
        )
    observed_catalog_counts = dict(sorted(selected_final_rank_catalog_counts.items()))
    if observed_catalog_counts != manifest.get("selected_final_rank_catalog_counts"):
        raise BeamRiskError("最终风险 pair 的目录可展开分类与 mining manifest 不一致")
    dataset_path = output_dir / "dataset"
    if not dataset_path.is_dir():
        raise BeamRiskError("风险 Arrow dataset 不存在")
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))
    if len(dataset) != rows:
        raise BeamRiskError("风险 Arrow dataset 与 JSONL 行数不一致")
    # Exercise the exact first auxiliary event that the next training epoch
    # will request, for every DDP rank, before a model is loaded.
    from .risk_store import RiskPairStore

    target_epoch = reference_epoch + 1
    first_epoch_step = (
        config.training.expected_optimizer_steps_per_epoch * reference_epoch + 1
    )
    interval = config.risk.interval_optimizer_steps
    first_risk_step = (
        (first_epoch_step + interval - 1) // interval * interval
    )
    store = RiskPairStore(
        dataset_path,
        seed=config.training.seed + target_epoch * 10_000,
        expected_rows=rows,
    )
    first_batches = [
        store.batch_for_optimizer_step(
            optimizer_step=first_risk_step,
            interval=interval,
            global_batch_size=config.risk.global_batch_size,
            rank=rank,
            world_size=config.training.world_size,
        )
        for rank in range(config.training.world_size)
    ]
    if any(len(batch) != config.risk.per_device_batch_size for batch in first_batches):
        raise BeamRiskError("下一训练阶段的首个四卡 risk batch 大小不一致")
    diagnostics_path = Path(str(manifest.get("state_diagnostics", "")))
    try:
        diagnostics_path.resolve().relative_to(output_dir.resolve())
    except ValueError as error:
        raise BeamRiskError("Mining state diagnostics 必须位于当前挖掘目录") from error
    diagnostics = read_json(diagnostics_path, name="Mining state diagnostics")
    if (
        diagnostics.get("status") != "completed"
        or diagnostics.get("reference_epoch") != reference_epoch
        or diagnostics.get("rows") != manifest.get("scanned_state_rows")
        or diagnostics.get("state_counts") != manifest.get("scanned_state_counts")
    ):
        raise BeamRiskError("Mining state diagnostics 与 manifest 不一致")
    catalog_diagnostics_path = Path(
        str(manifest.get("catalog_path_diagnostics", ""))
    )
    try:
        catalog_diagnostics_path.resolve().relative_to(output_dir.resolve())
    except ValueError as error:
        raise BeamRiskError("Catalog path diagnostics 必须位于当前挖掘目录") from error
    catalog_diagnostics = read_json(
        catalog_diagnostics_path,
        name="Catalog path diagnostics",
    )
    if (
        catalog_diagnostics.get("status") != "completed"
        or catalog_diagnostics.get("reference_epoch") != reference_epoch
        or catalog_diagnostics.get("catalog_unexpandable_final_rank_pairs")
        != manifest.get("catalog_unexpandable_final_rank_negatives")
    ):
        raise BeamRiskError("Catalog path diagnostics 与 manifest 不一致")
    return {
        "status": "passed",
        "reference_epoch": reference_epoch,
        "rows": rows,
        "train_validation_overlap": 0,
        "risk_type_counts": dict(sorted(risk_counts.items())),
        "first_prune_depth_counts": dict(sorted(depth_counts.items())),
        "selected_final_rank_catalog_counts": observed_catalog_counts,
        "vocab_size": vocab_size,
        "max_token_id": max_token_id,
        "max_prompt_plus_path": max_prompt_plus_path,
        "first_training_risk_optimizer_step": first_risk_step,
        "first_training_risk_global_batch": sum(len(batch) for batch in first_batches),
        "scanned_state_rows": diagnostics["rows"],
        "scanned_state_counts": diagnostics["state_counts"],
        "risk_pairs_sha256": manifest["risk_pairs_sha256"],
    }

"""Reproducible category-cohort SID geometry and full-catalog prefix statistics."""

from __future__ import annotations

import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.geo import encode_geohash_tokens, geohash_strings
from qg_prqk.sid.visualization_config import load_sid_visualization_config


SCHEMA = "qg-prqk-sid-category-region-v2"
CODE_PATHS = (
    "src/qg_prqk/sid/category_region_visualization.py",
    "src/qg_prqk/sid/category_region_plots.py",
    "src/qg_prqk/sid/visualization_config.py",
    "src/qg_prqk/sid/geo.py",
    "src/qg_prqk/artifacts.py",
    "src/qg_prqk/commands/visualize_sid_categories.py",
    "src/qg_prqk/cli.py",
    "scripts/qg_prqk.py",
)


def balanced_category_sample(
    labels: Sequence[str], categories: Sequence[str], per_category: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select one method-independent cohort; the first draw per class is its anchor."""
    if per_category < 1 or not categories or len(set(categories)) != len(categories):
        raise ValueError("类别须非空且不重复，每类抽样数须为正")
    values = np.asarray(labels)
    rng = np.random.default_rng(seed)
    selected = []
    anchors = []
    for category in categories:
        candidates = np.flatnonzero(values == category)
        if len(candidates) < per_category:
            raise ValueError(f"类别 {category} 只有 {len(candidates)} 行，不足抽样")
        rows = rng.choice(candidates, per_category, replace=False)
        selected.extend(rows.tolist())
        anchors.append(int(rows[0]))
    return np.asarray(selected, dtype=np.int64), np.asarray(anchors, dtype=np.int64)


def sid_codeword_features(sid: np.ndarray, books: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate selected unit POI codewords, with equal weight for all three levels."""
    if (
        sid.ndim != 2
        or sid.shape[1] != 3
        or sid.dtype.kind not in "iu"
        or len(books) != 3
    ):
        raise ValueError("SID 必须是三列整数，且对应三层 POI 码本")
    parts = []
    for level, book in enumerate(books):
        if (
            book.ndim != 2
            or np.any(sid[:, level] < 0)
            or np.any(sid[:, level] >= len(book))
        ):
            raise ValueError("SID code 超出码本范围或码本不是二维")
        vectors = np.asarray(book[sid[:, level]], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms <= 1e-12):
            raise ValueError("选中码向量包含非有限值或零范数")
        parts.append(vectors / norms / np.sqrt(3.0))
    return np.concatenate(parts, axis=1).astype(np.float32)


def label_distribution(labels: Sequence[str], top_k: int) -> dict[str, Any]:
    """Keep exact full-bucket counts and a bounded top-k plus remainder view."""
    counts = Counter(str(label) for label in labels)
    total = sum(counts.values())
    if not total or top_k < 1:
        raise ValueError("前缀桶不能为空，top_k 必须为正")
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top = [
        {"label": label, "count": count, "share": count / total}
        for label, count in ranked[:top_k]
    ]
    other = total - sum(item["count"] for item in top)
    return {
        "total": total,
        "counts": dict(ranked),
        "top": top,
        "other_count": other,
        "other_share": other / total,
    }


def prefix_examples(
    sid: np.ndarray,
    anchors: np.ndarray,
    coarse: np.ndarray,
    fine: np.ndarray,
    regions: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    """Follow the same anchors in each method; count every POI in each exact prefix."""
    if not (len(sid) == len(coarse) == len(fine) == len(regions)):
        raise ValueError("前缀统计的 SID/类别/区域行未对齐")
    result = []
    for level in (1, 2, 3):
        for anchor_index, row in enumerate(anchors):
            prefix = sid[row, :level]
            mask = np.all(sid[:, :level] == prefix, axis=1)
            result.append(
                {
                    "level": level,
                    "anchor_index": anchor_index,
                    "anchor_row": int(row),
                    "prefix": prefix.tolist(),
                    "poi_count": int(mask.sum()),
                    "coarse_category": label_distribution(coarse[mask], top_k),
                    "fine_category": label_distribution(fine[mask], top_k),
                    "region_geohash5": label_distribution(regions[mask], top_k),
                }
            )
    return result


def _weighted_purity(
    groups: np.ndarray, sizes: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    _, label_ids = np.unique(labels, return_inverse=True)
    label_count = int(label_ids.max()) + 1
    pairs, counts = np.unique(
        groups.astype(np.int64) * label_count + label_ids, return_counts=True
    )
    maxima = np.zeros(len(sizes), dtype=np.int64)
    np.maximum.at(maxima, pairs // label_count, counts)
    non_singleton = sizes > 1
    denominator = int(sizes[non_singleton].sum())
    return {
        "poi_weighted_top1_share": float(maxima.sum() / sizes.sum()),
        "non_singleton_poi_weighted_top1_share": (
            float(maxima[non_singleton].sum() / denominator) if denominator else None
        ),
    }


def catalog_prefix_metrics(
    sid: np.ndarray,
    coarse: np.ndarray,
    fine: np.ndarray,
    regions: np.ndarray,
    codebook_size: int = 512,
) -> list[dict[str, Any]]:
    """Report exact population-weighted purity, including a singleton-excluded check."""
    keys = np.zeros(len(sid), dtype=np.int64)
    result = []
    for column in range(3):
        keys = keys * codebook_size + sid[:, column]
        _, groups, sizes = np.unique(keys, return_inverse=True, return_counts=True)
        result.append(
            {
                "level": column + 1,
                "unique_prefixes": len(sizes),
                "singleton_poi_share": float(np.count_nonzero(sizes == 1) / len(sid)),
                "max_bucket_size": int(sizes.max()),
                "coarse_category": _weighted_purity(groups, sizes, coarse),
                "fine_category": _weighted_purity(groups, sizes, fine),
                "region_geohash5": _weighted_purity(groups, sizes, regions),
            }
        )
    return result


def _local_path(root: Path, value: str, *, outputs_only: bool = False) -> Path:
    path = (root / value).resolve()
    allowed = root / "qg_prqk" / "outputs" if outputs_only else root / "qg_prqk"
    if not path.is_relative_to(allowed.resolve()):
        raise ValueError(f"路径必须位于 {allowed}：{path}")
    return path


def load_protocol(path: Path) -> tuple[dict[str, Any], Any]:
    """Bind the new presentation protocol to the previous immutable data contract."""
    path = path.resolve()
    root = path.parents[2]
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if settings.get("schema_version") != SCHEMA:
        raise ValueError("类别/区域可视化配置版本错误")
    source = _local_path(root, settings["source_config"])
    if sha256_file(source) != settings["source_config_sha256"]:
        raise ValueError("冻结来源配置 SHA256 不符")
    config = load_sid_visualization_config(source)
    for key in ("font_path", "output_dir"):
        _local_path(root, settings[key], outputs_only=True)
    if settings["geohash_precision"] != 5 or settings["prefix_top_k"] != 3:
        raise ValueError("当前已确认协议固定 Geohash5 与前三项展示")
    projection = settings["projection"]
    if (
        projection["representation"]
        != "concatenate_unit_poi_codewords_equal_level_weight"
        or projection["fit_scope"] != "joint_three_methods_same_poi"
    ):
        raise ValueError("必须使用三层等权码向量与共同抽样、联合投影")
    if (
        len(settings["categories"]) != 5
        or settings["rows_per_category"] < 1
        or len(set(settings["categories"])) != 5
        or not 1 <= projection["cpu_threads"] <= 8
        or not 1 <= projection["pca_components"] < settings["rows_per_category"] * 15
        or not 0 < projection["perplexity"] < settings["rows_per_category"] * 15
        or projection["max_iter"] < 250
    ):
        raise ValueError("抽样或投影参数非法")
    return settings, config


def _sources(
    settings: dict[str, Any], config: Any, protocol_path: Path
) -> dict[str, str]:
    root = config.project_root
    sources = {
        str(protocol_path.resolve()): sha256_file(protocol_path),
        str(config.source_path): config.source_sha256,
    }
    manifests = {}
    for item in config.frozen_manifests.values():
        path = Path(item["path"])
        sources[str(path)] = item["sha256"]
        manifests[path.parent] = json.loads(path.read_text(encoding="utf-8"))
    paths = [config.poi_metadata_path]
    for method in config.methods:
        paths.extend([method.sid_path, *method.codebook_paths])
    for path in paths:
        expected = manifests[path.parent]["artifacts"][path.name]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"源产物 SHA256 不符：{path}")
        sources[str(path)] = actual
    font = _local_path(root, settings["font_path"], outputs_only=True)
    sources[str(font)] = sha256_file(font)
    for relative in CODE_PATHS:
        path = root / "qg_prqk" / relative
        sources[str(path)] = sha256_file(path)
    return sources


def _load_inputs(
    config: Any,
) -> tuple[pa.Table, list[np.ndarray], list[list[np.ndarray]]]:
    columns = [
        "poi_row_index",
        "poi_id",
        "displayname",
        "category",
        "category_code",
        "lat",
        "lng",
    ]
    table = pq.read_table(config.poi_metadata_path, columns=columns)
    if (
        table.num_rows != config.rows["poi"]
        or any(table[name].null_count for name in columns)
        or not np.array_equal(
            table["poi_row_index"].to_numpy(), np.arange(table.num_rows)
        )
        or len(set(table["poi_id"].to_pylist())) != table.num_rows
    ):
        raise ValueError("metadata 行序/空值/POI 唯一性不符")
    sids, codebooks = [], []
    for method in config.methods:
        sid = np.load(method.sid_path, mmap_mode="r", allow_pickle=False)
        if (
            sid.shape != (table.num_rows, 3)
            or sid.dtype != np.int32
            or np.any(sid < 0)
            or np.any(sid >= 512)
        ):
            raise ValueError(f"{method.name} 的 SID shape/dtype/range 错误")
        books = [
            np.load(path, mmap_mode="r", allow_pickle=False)
            for path in method.codebook_paths
        ]
        if any(
            book.shape != (512, 1024)
            or book.dtype != np.float32
            or not np.isfinite(book).all()
            for book in books
        ):
            raise ValueError(f"{method.name} 的码本合同错误")
        sids.append(sid)
        codebooks.append(books)
    if not np.array_equal(sids[1][:, :2], sids[2][:, :2]) or any(
        not np.array_equal(codebooks[1][i], codebooks[2][i]) for i in (0, 1)
    ):
        raise ValueError("两种 A4 的冻结 S1/S2 应完全一致")
    return table, sids, codebooks


def validate_category_region_visualization(path: Path) -> dict[str, Any]:
    """Verify existing outputs and all consumed input/code hashes without recomputing t-SNE."""
    settings, config = load_protocol(path)
    output = _local_path(config.project_root, settings["output_dir"], outputs_only=True)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    marker = json.loads((output / "_SUCCESS").read_text(encoding="utf-8"))
    if marker["manifest_sha256"] != sha256_file(output / "manifest.json"):
        raise ValueError("输出 manifest SHA256 不符")
    if manifest["sources"] != _sources(settings, config, path):
        raise ValueError("来源或代码已改变，不能混用已有输出")
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or sha256_file(output / name) != digest:
            raise ValueError(f"输出 SHA256 不符：{name}")
    return {
        "status": "validated",
        "output_dir": str(output),
        "manifest_sha256": marker["manifest_sha256"],
    }


def build_category_region_visualization(
    path: Path, *, dry_run: bool = False
) -> dict[str, Any]:
    """Build bounded joint t-SNE and exact prefix views without reading business query splits."""
    started = time.monotonic()
    settings, config = load_protocol(path)
    output = _local_path(config.project_root, settings["output_dir"], outputs_only=True)
    if not dry_run and output.exists():
        raise ValueError(f"输出目录已存在，不覆盖：{output}；可用 --validate-only 复核")
    print("核验冻结 SID、码本、metadata 和来源 SHA256……", flush=True)
    sources = _sources(settings, config, path)
    table, sids, books = _load_inputs(config)
    fine = np.asarray(table["category"].to_pylist())
    coarse = np.asarray([label.split(":", 1)[0] for label in fine])
    codes = table["category_code"].to_pylist()
    bindings = set(zip((str(code)[:2] for code in codes), coarse.tolist()))
    if len(bindings) != len({pair[0] for pair in bindings}) or len(bindings) != len(
        set(coarse)
    ):
        raise ValueError("coarse category_code 与展示名称不是一一对应")
    category_counts = Counter(coarse.tolist())
    top_five = [
        label
        for label, _ in sorted(
            category_counts.items(), key=lambda item: (-item[1], item[0])
        )[:5]
    ]
    if settings["categories"] != top_five:
        raise ValueError("配置五类不是全库计数确定的前五类")
    rows, anchors = balanced_category_sample(
        coarse, settings["categories"], settings["rows_per_category"], settings["seed"]
    )
    tokens = encode_geohash_tokens(
        table["lng"].to_numpy(), table["lat"].to_numpy(), length=5
    )
    regions = np.asarray(list(geohash_strings(tokens)))
    if dry_run:
        return {
            "status": "inputs_validated",
            "poi_rows": table.num_rows,
            "sample_rows": len(rows),
            "categories": category_counts,
            "sources_checked": len(sources),
            "output_created": False,
        }
    output.mkdir(parents=True, exist_ok=False)
    print(f"固定五类共 {len(rows)} 个 POI；统计三种 SID 的全库前缀……", flush=True)
    methods = {}
    for method, sid in zip(config.methods, sids):
        methods[method.name] = {
            "display_name": method.display_name,
            "prefix_examples": prefix_examples(
                sid, anchors, coarse, fine, regions, settings["prefix_top_k"]
            ),
            "catalog_prefix_metrics": catalog_prefix_metrics(
                sid, coarse, fine, regions
            ),
        }
        print(f"{method.name}：全库前缀统计完成", flush=True)

    from sklearn import __version__ as sklearn_version
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from threadpoolctl import threadpool_limits

    projection = settings["projection"]
    with threadpool_limits(limits=projection["cpu_threads"]):
        features = [
            sid_codeword_features(sid[rows], book) for sid, book in zip(sids, books)
        ]
        # Labels never enter PCA/t-SNE; used only for post-hoc diagnostics and plot colors.
        for method, vectors in zip(config.methods, features):
            methods[method.name]["sample_codeword_cosine_silhouette"] = float(
                silhouette_score(vectors, coarse[rows], metric="cosine")
            )
        combined = np.concatenate(features, axis=0)
        pca = PCA(
            n_components=projection["pca_components"],
            svd_solver="randomized",
            random_state=settings["seed"],
        )
        reduced = pca.fit_transform(combined)
        print(f"联合 PCA 完成，开始 t-SNE：{len(combined)} 点……", flush=True)
        tsne = TSNE(
            n_components=2,
            perplexity=projection["perplexity"],
            max_iter=projection["max_iter"],
            learning_rate="auto",
            init="pca",
            metric="euclidean",
            random_state=settings["seed"],
            n_jobs=projection["cpu_threads"],
            verbose=1,
        )
        coordinates = tsne.fit_transform(reduced).reshape(3, len(rows), 2)
    if not np.isfinite(coordinates).all():
        raise ValueError("t-SNE 输出含非有限值")
    sample = table.take(pa.array(rows))
    sample = sample.append_column("coarse_category", pa.array(coarse[rows]))
    sample = sample.append_column("geohash5", pa.array(regions[rows]))
    sample = sample.append_column("is_anchor", pa.array(np.isin(rows, anchors)))
    for index, method in enumerate(config.methods):
        sample = sample.append_column(
            f"{method.name}_sid", pa.array(sids[index][rows].tolist())
        )
        sample = sample.append_column(
            f"{method.name}_tsne_x", pa.array(coordinates[index, :, 0])
        )
        sample = sample.append_column(
            f"{method.name}_tsne_y", pa.array(coordinates[index, :, 1])
        )
    pq.write_table(sample, output / "sample_and_coordinates.parquet")
    metrics = {
        "schema_version": SCHEMA,
        "poi_rows": table.num_rows,
        "sample_rows_per_method": len(rows),
        "categories": settings["categories"],
        "catalog_category_counts": dict(category_counts),
        "anchor_rows": anchors.tolist(),
        "anchor_poi_ids": table.take(pa.array(anchors))["poi_id"].to_pylist(),
        "category_colors_scope": "same_label_same_color_all_tsne_panels",
        "prefix_counts_scope": "all_catalog_poi_matching_exact_sid_prefix_no_gid_or_category_filter",
        "region_definition": "geohash5_grid_not_administrative_district",
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "tsne_kl_divergence": float(tsne.kl_divergence_),
        "methods": methods,
        "limitations": [
            "t-SNE 仅作定性展示，不是检索评测或监督独立验证集",
            "SID 码向量串联是本次适配口径，不是 GenPOI 原文 GeoPE embedding 的严格复现",
            "类别参与过 A4 构建，因此类别聚集只能说明结构，不能证明泛化提升",
            "S3 单例/小桶的高纯度有机械效应；同时报告非单例全库指标",
            "无显式 GID token 不等于无地理特征；此图不含 GID/dedup 或 Query 码本",
        ],
    }
    write_json_atomic(output / "metrics.json", metrics)
    print("绘制同类 t-SNE 与类别/区域前缀分布……", flush=True)
    from qg_prqk.sid.category_region_plots import draw_category_region_figures

    draw_category_region_figures(
        settings, config.project_root, coordinates, coarse[rows], metrics, output
    )
    # Validate once more before publication; source files are never updated by this analysis.
    if sources != _sources(settings, config, path):
        raise ValueError("运行期间输入或代码发生变化，不发布成功标记")
    artifact_names = [
        "sample_and_coordinates.parquet",
        "metrics.json",
        "sid_category_tsne.png",
        "sid_category_tsne.pdf",
        "sid_prefix_category_region.png",
        "sid_prefix_category_region.pdf",
    ]
    write_json_atomic(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "completed",
            "built_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "settings": settings,
            "sources": sources,
            "artifacts": {name: sha256_file(output / name) for name in artifact_names},
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "sklearn": sklearn_version,
                "device": "cpu",
                "threads": projection["cpu_threads"],
            },
            "source_access": {
                "business_validation_read": False,
                "business_test_read": False,
                "raw_bge_embedding_read": False,
                "sid_or_model_updated": False,
            },
            "reference": "https://arxiv.org/html/2605.03397v1",
        },
    )
    write_json_atomic(
        output / "_SUCCESS", {"manifest_sha256": sha256_file(output / "manifest.json")}
    )
    return {
        "status": "completed",
        "output_dir": str(output),
        "poi_rows": table.num_rows,
        "sample_rows_per_method": len(rows),
        "elapsed_seconds": time.monotonic() - started,
    }

"""Paired, non-singleton SID prefix examples with complete collision membership."""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa

from qg_prqk.artifacts import sha256_file, utc_now, write_json_atomic
from qg_prqk.sid.category_region_plots import METHOD_LABELS, _composition_bar, _save
from qg_prqk.sid.category_region_visualization import (
    _load_inputs,
    _local_path,
    _sources,
    load_protocol,
    prefix_examples,
)
from qg_prqk.sid.geo import encode_geohash_tokens, geohash_strings


SCHEMA = "qg-prqk-sid-collision-examples-v1"


def collision_bucket_sizes(sid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each POI's complete SID bucket size and the sizes of all unique buckets."""
    if (
        sid.ndim != 2
        or sid.shape[1] != 3
        or sid.dtype.kind not in "iu"
        or not len(sid)
        or np.any(sid < 0)
        or np.any(sid >= 512)
    ):
        raise ValueError("SID 必须是非空三列整数，且 code 位于 [0,512)")
    keys = (sid[:, 0].astype(np.int64) * 512 + sid[:, 1]) * 512 + sid[:, 2]
    _, groups, sizes = np.unique(keys, return_inverse=True, return_counts=True)
    return sizes[groups], sizes


def select_collision_anchors(
    sids: Sequence[np.ndarray],
    coarse: np.ndarray,
    categories: Sequence[str],
    *,
    seed: int,
    min_size: int,
    max_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Uniformly sample one POI per class from the intersection of collision-eligible rows."""
    if min_size < 2 or max_size < min_size or len(sids) != 3:
        raise ValueError("必须有三版 SID，且 2 <= min_size <= max_size")
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("展示类别须非空且不重复")
    eligible = np.ones(len(coarse), dtype=bool)
    catalog_counts = []
    for sid in sids:
        if len(sid) != len(coarse):
            raise ValueError("三版 SID 与 metadata 行数不一致")
        per_row, sizes = collision_bucket_sizes(sid)
        eligible &= (per_row >= min_size) & (per_row <= max_size)
        catalog_counts.append(
            {
                "collision_poi": int((per_row >= 2).sum()),
                "collision_buckets": int((sizes >= 2).sum()),
                "eligible_buckets": int(
                    ((sizes >= min_size) & (sizes <= max_size)).sum()
                ),
            }
        )
    rng = np.random.default_rng(seed)
    anchors, category_candidates = [], {}
    for category in categories:
        candidates = np.flatnonzero(eligible & (coarse == category))
        if not len(candidates):
            raise ValueError(
                f"类别 {category} 没有三版同时满足碰撞范围的 POI，不退回单例"
            )
        anchors.append(int(rng.choice(candidates)))
        category_candidates[category] = len(candidates)
    return np.asarray(anchors, dtype=np.int64), {
        "common_eligible_poi": int(eligible.sum()),
        "common_eligible_poi_per_category": category_candidates,
        "catalog_counts_by_method": catalog_counts,
        "selection": "uniform_poi_per_category_after_joint_size_filter_no_region_or_purity_filter",
    }


def collision_members(
    table: pa.Table,
    sid: np.ndarray,
    row: int,
    regions: np.ndarray,
) -> list[dict[str, Any]]:
    """Return every distinct POI in an anchor's full SID bucket, without truncation."""
    rows = np.flatnonzero(np.all(sid == sid[row], axis=1))
    if len(rows) < 2:
        raise ValueError("碰撞案例不允许单例")
    members = table.take(pa.array(rows)).to_pylist()
    for member, index in zip(members, rows):
        member["geohash5"] = str(regions[index])
    if len({member["poi_id"] for member in members}) != len(rows):
        raise ValueError("碰撞必须来自不同 POI ID，不能重复计数")
    return members


def draw_collision_prefixes(
    settings: dict[str, Any], root: Path, data: dict[str, Any], output: Path
) -> None:
    """Reuse the existing bar semantics, but render only paired collision-conditioned paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = root / settings["font_path"]
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    with plt.rc_context(
        {"font.family": family, "axes.unicode_minus": False, "pdf.fonttype": 42}
    ):
        figure, axes = plt.subplots(3, 3, figsize=(27, 19))
        titles = ("S1 前缀 · 粗类别", "S1/S2 前缀 · 细类别", "S1/S2/S3 碰撞桶 · 细类别")
        for method_index, method in enumerate(data["methods"].values()):
            for column in range(3):
                axis = axes[method_index, column]
                for entry in method["prefix_examples"]:
                    if entry["level"] != column + 1:
                        continue
                    anchor = entry["anchor_index"]
                    y = (4 - anchor) * 1.6
                    tokens = [str(token) for token in entry["prefix"]] + ["*"] * (
                        2 - column
                    )
                    prefix = "[" + ", ".join(tokens) + "]"
                    label = f"锚点 {anchor + 1} · {settings['categories'][anchor]}  {prefix}  n={entry['poi_count']:,}"
                    axis.text(
                        0, y + 0.61, label, fontsize=10, va="bottom", color="#25384b"
                    )
                    category_key = "coarse_category" if column == 0 else "fine_category"
                    _composition_bar(axis, y + 0.26, entry[category_key], region=False)
                    _composition_bar(
                        axis, y - 0.17, entry["region_geohash5"], region=True
                    )
                    if column == 2:
                        axis.text(
                            0,
                            y - 0.48,
                            f"不同细类别 {len(entry['fine_category']['counts'])} 种 / Geohash5 {len(entry['region_geohash5']['counts'])} 格",
                            fontsize=8,
                            color="#66717e",
                        )
                axis.set_title(
                    f"{METHOD_LABELS[method_index]}\n{titles[column]}",
                    loc="left",
                    fontsize=14,
                    pad=15,
                )
                axis.set_xlim(0, 100)
                axis.set_ylim(-0.73, 7.35)
                axis.set_xticks(
                    [0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"]
                )
                axis.tick_params(axis="x", length=0, labelsize=9, colors="#788491")
                axis.set_yticks([])
                for name, spine in axis.spines.items():
                    spine.set_visible(name == "bottom")
                    spine.set_color("#d7dde4")
        figure.suptitle(
            "只看真实碰撞：三版同五个锚点，末列全部包含多个 POI", fontsize=24, y=0.982
        )
        selection = data["parameters"]
        figure.text(
            0.5,
            0.947,
            f"先要求该 POI 在三版的完整 SID 桶均含 {selection['min_size']}–{selection['max_size']} 个不同 POI，再按类别以 seed={selection['seed']} 抽取；不按区域或类别纯度选例",
            ha="center",
            fontsize=12,
            color="#435365",
        )
        figure.text(
            0.5,
            0.925,
            "蓝＝类别 / 橙＝Geohash5 网格（非行政区）；深到浅为桶内 Top-1/2/3，灰＝其余。各条统计完整桶，不限五类、不按 GID 筛选。",
            ha="center",
            fontsize=12,
            color="#66717e",
        )
        figure.text(
            0.5,
            0.025,
            "这是碰撞条件案例，不能代表全库优劣。100% 仅表示同类/同网格，不代表同一地点；不同 POI 的名称、坐标与完整计数见 collision_members.html / collision_examples.json。",
            ha="center",
            fontsize=10.5,
            color="#66717e",
        )
        figure.subplots_adjust(
            left=0.035, right=0.985, top=0.872, bottom=0.065, wspace=0.13, hspace=0.30
        )
        _save(
            figure,
            output,
            "sid_collision_prefix_category_region",
            settings["projection"]["dpi"],
        )
        plt.close(figure)


def collision_members_html(data: dict[str, Any]) -> str:
    """Render a local-only, escaped table containing all members of all 15 collision buckets."""
    lines = [
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
        "<title>三版 SID 碰撞 POI 明细</title><style>body{font-family:sans-serif;margin:24px;color:#25384b}"
        "table{border-collapse:collapse;width:100%;margin-bottom:24px}td,th{border:1px solid #dce1e7;padding:7px;text-align:left}"
        "th{background:#eef3f8}h2{margin-top:40px}small{color:#66717e}</style>",
        "<h1>三版 SID：碰撞桶完整 POI 明细</h1>",
        "<p>每版沿同五个固定 POI 锚点展开，所有完整 SID 桶至少包含多个不同 POI ID；未按区域纯度选择或删除成员。"
        "区域为 Geohash5 网格，不是行政区；同网格不等于同坐标。本表只按三位 SID 分组，不加入 GID 或 dedup。</p>",
    ]
    for anchor, category in enumerate(data["categories"]):
        lines.append(
            f"<h2>锚点 {anchor + 1}：{html.escape(category)} / POI {html.escape(str(data['anchor_poi_ids'][anchor]))}</h2>"
        )
        for method_index, method in enumerate(data["methods"].values()):
            entry = next(
                x
                for x in method["prefix_examples"]
                if x["level"] == 3 and x["anchor_index"] == anchor
            )
            lines.append(
                f"<h3>{METHOD_LABELS[method_index]} · SID {entry['prefix']} · {entry['poi_count']} 个 POI · {len(entry['region_geohash5']['counts'])} 个网格</h3>"
            )
            lines.append(
                "<table><tr><th>POI ID</th><th>名称</th><th>完整类别</th><th>Geohash5</th><th>经度</th><th>纬度</th></tr>"
            )
            for member in entry["members"]:
                values = [
                    member["poi_id"],
                    member["displayname"],
                    member["category"],
                    member["geohash5"],
                    f"{member['lng']:.6f}",
                    f"{member['lat']:.6f}",
                ]
                lines.append(
                    "<tr>"
                    + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
                    + "</tr>"
                )
            lines.append("</table>")
    return "\n".join(lines) + "</html>\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build or verify the collision-only supplement without changing the frozen V2 implementation."""
    started = time.monotonic()
    settings, config = load_protocol(args.config)
    output = _local_path(config.project_root, str(args.output_dir), outputs_only=True)
    parameters = {
        "seed": args.seed,
        "min_size": args.min_size,
        "max_size": args.max_size,
    }
    sources = _sources(settings, config, args.config)
    for relative in (
        "src/qg_prqk/sid/collision_visualization.py",
        "scripts/visualize_sid_collisions.py",
    ):
        path = config.project_root / "qg_prqk" / relative
        sources[str(path)] = sha256_file(path)
    if args.validate_only:
        manifest = json.loads((output / "manifest.json").read_text())
        marker = json.loads((output / "_SUCCESS").read_text())
        if (
            marker["manifest_sha256"] != sha256_file(output / "manifest.json")
            or manifest["sources"] != sources
            or manifest["parameters"] != parameters
        ):
            raise ValueError("碰撞图的来源、参数或 manifest 哈希已变化")
        for name, digest in manifest["artifacts"].items():
            if Path(name).name != name or sha256_file(output / name) != digest:
                raise ValueError(f"碰撞输出哈希不符：{name}")
        return {"status": "validated", "output_dir": str(output), **marker}
    if output.exists():
        raise ValueError(f"输出已存在，拒绝覆盖：{output}")
    print("读取冻结 SID/metadata，只筛选碰撞锚点；不重算 t-SNE……", flush=True)
    table, sids, _ = _load_inputs(config)
    fine = np.asarray(table["category"].to_pylist())
    coarse = np.asarray([label.split(":", 1)[0] for label in fine])
    anchors, selection = select_collision_anchors(
        sids, coarse, settings["categories"], **parameters
    )
    tokens = encode_geohash_tokens(
        table["lng"].to_numpy(), table["lat"].to_numpy(), length=5
    )
    regions = np.asarray(list(geohash_strings(tokens)))
    methods = {}
    for method, sid in zip(config.methods, sids):
        entries = prefix_examples(sid, anchors, coarse, fine, regions, 3)
        for entry in entries:
            if entry["level"] == 3:
                entry["members"] = collision_members(
                    table, sid, entry["anchor_row"], regions
                )
                if (
                    not args.min_size
                    <= len(entry["members"])
                    == entry["poi_count"]
                    <= args.max_size
                ):
                    raise ValueError("选中桶成员数不满足固定碰撞范围")
        methods[method.name] = {
            "display_name": method.display_name,
            "prefix_examples": entries,
        }
        print(
            method.name,
            "末层桶大小：",
            [x["poi_count"] for x in entries if x["level"] == 3],
            flush=True,
        )
    data = {
        "schema_version": SCHEMA,
        "parameters": parameters,
        "categories": settings["categories"],
        "anchor_rows": anchors.tolist(),
        "anchor_poi_ids": table.take(pa.array(anchors))["poi_id"].to_pylist(),
        "selection_audit": selection,
        "methods": methods,
        "scope": "three_token_sid_only_without_gid_or_dedup_collision_conditioned_examples",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "collision_examples.json", data)
    (output / "collision_members.html").write_text(
        collision_members_html(data), encoding="utf-8"
    )
    draw_collision_prefixes(settings, config.project_root, data, output)
    if any(sha256_file(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError("运行期间输入或代码发生变化，不发布成功标记")
    names = [
        "collision_examples.json",
        "collision_members.html",
        "sid_collision_prefix_category_region.png",
        "sid_collision_prefix_category_region.pdf",
    ]
    write_json_atomic(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "completed",
            "built_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "parameters": parameters,
            "sources": sources,
            "artifacts": {name: sha256_file(output / name) for name in names},
            "source_access": {
                "business_validation_read": False,
                "business_test_read": False,
                "tsne_recomputed": False,
                "sid_or_model_updated": False,
            },
        },
    )
    write_json_atomic(
        output / "_SUCCESS", {"manifest_sha256": sha256_file(output / "manifest.json")}
    )
    return {
        "status": "completed",
        "output_dir": str(output),
        "collision_examples": 15,
        "singleton_examples": 0,
        "elapsed_seconds": time.monotonic() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="三版共同碰撞锚点的前缀分布与完整 POI 明细；保留旧图，不重算 t-SNE"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qg_prqk/configs/qg_prqk_sid_category_region_v2.yaml"),
        help="冻结的 V2 来源配置",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qg_prqk/outputs/figures/qg_prqk_sid_collision_examples_v1"),
        help="新的 QG 输出目录，拒绝覆盖",
    )
    parser.add_argument("--seed", type=int, default=42, help="固定抽样 seed")
    parser.add_argument(
        "--min-size", type=int, default=3, help="三版完整 SID 桶的最少 POI 数，须 >=2"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=20,
        help="三版完整 SID 桶的最多 POI 数，便于完整查看",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="只复核已有碰撞图和来源哈希"
    )
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"碰撞可视化失败：{error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

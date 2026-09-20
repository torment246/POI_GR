"""Matched-category t-SNE and concrete SID prefix composition panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


CATEGORY_COLORS = ("#3977b8", "#e79d30", "#309e88", "#c66594", "#8264b4")
METHOD_LABELS = ("A0 · POI-only", "A4 · GID-parent", "A4 · NoGID")
CATEGORY_RANK_COLORS = ("#3977b8", "#80add6", "#b8d4eb")
REGION_RANK_COLORS = ("#d78a2d", "#e6b46f", "#f2d9b3")
OTHER_COLOR = "#e4e7eb"


def _save(figure: Any, output: Path, stem: str, dpi: int) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"{stem}.{suffix}", dpi=dpi, facecolor="white", bbox_inches="tight"
        )


def _composition_bar(
    axis: Any, y: float, distribution: dict[str, Any], *, region: bool
) -> None:
    colors = REGION_RANK_COLORS if region else CATEGORY_RANK_COLORS
    left = 0.0
    for rank, item in enumerate(distribution["top"]):
        width = 100 * item["share"]
        axis.barh(
            y,
            width,
            left=left,
            height=0.34,
            color=colors[rank],
            edgecolor="white",
            linewidth=0.6,
        )
        # Small slices retain their counts in metrics.json; do not obscure adjacent labels.
        if width >= 15:
            name = item["label"] if region else item["label"].split(":")[-1]
            if len(name) > 10:
                name = name[:9] + "…"
            text = f"{name} {width:.0f}%"
            axis.text(
                left + width / 2,
                y,
                text,
                ha="center",
                va="center",
                fontsize=8.2,
                color="white" if rank == 0 else "#25384b",
            )
        left += width
    other = distribution["other_share"] * 100
    if other > 0:
        axis.barh(
            y,
            other,
            left=left,
            height=0.34,
            color=OTHER_COLOR,
            edgecolor="white",
            linewidth=0.6,
        )
        if other >= 15:
            axis.text(
                left + other / 2,
                y,
                f"其余 {other:.0f}%",
                ha="center",
                va="center",
                fontsize=8.2,
                color="#4d5966",
            )
    axis.text(
        -1.2,
        y,
        "区" if region else "类",
        ha="right",
        va="center",
        fontsize=8.5,
        color="#556372",
    )


def draw_category_region_figures(
    settings: dict[str, Any],
    root: Path,
    coordinates: np.ndarray,
    sampled_categories: np.ndarray,
    metrics: dict[str, Any],
    output: Path,
) -> None:
    """Plot computed coordinates/counts only; never refit or adjust points by their labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    font_path = root / settings["font_path"]
    font_manager.fontManager.addfont(str(font_path))
    font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
    dpi = settings["projection"]["dpi"]
    with plt.rc_context(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
            "font.size": 10,
            "pdf.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(18, 6.9), sharex=True, sharey=True)
        low, high = coordinates.min(axis=(0, 1)), coordinates.max(axis=(0, 1))
        margin = (high - low) * 0.05
        for index, (axis, method) in enumerate(zip(axes, metrics["methods"].values())):
            for category, color in zip(settings["categories"], CATEGORY_COLORS):
                mask = sampled_categories == category
                axis.scatter(
                    coordinates[index, mask, 0],
                    coordinates[index, mask, 1],
                    s=12,
                    color=color,
                    alpha=0.78,
                    linewidths=0,
                    rasterized=True,
                )
            score = method["sample_codeword_cosine_silhouette"]
            axis.set_title(
                f"{METHOD_LABELS[index]}\n高维类别 silhouette（cosine）：{score:.3f}",
                fontsize=13,
                pad=10,
            )
            axis.set_xlim(low[0] - margin[0], high[0] + margin[0])
            axis.set_ylim(low[1] - margin[1], high[1] + margin[1])
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#d7dde4")
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=7,
                color=color,
                label=category,
            )
            for category, color in zip(settings["categories"], CATEGORY_COLORS)
        ]
        figure.suptitle("同一批五类 POI：三版 SID 码向量的 t-SNE", fontsize=19, y=0.99)
        figure.legend(
            handles=handles,
            loc="upper center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.93),
            fontsize=11,
        )
        count = settings["rows_per_category"]
        figure.text(
            0.5,
            0.055,
            f"每类 {count} 个相同 POI · seed={settings['seed']} · 三层归一化 POI 码向量等权串联 · 联合 PCA50 + t-SNE · 同坐标范围",
            ha="center",
            fontsize=10,
            color="#435365",
        )
        figure.text(
            0.5,
            0.022,
            "颜色只用于标注；未输入类别或区域标签。二维聚集不等于检索提升；类别是 A4 已用监督，非独立验证。",
            ha="center",
            fontsize=9,
            color="#66717e",
        )
        figure.subplots_adjust(
            left=0.025, right=0.985, top=0.79, bottom=0.11, wspace=0.09
        )
        _save(figure, output, "sid_category_tsne", dpi)
        plt.close(figure)

        figure, axes = plt.subplots(3, 3, figsize=(27, 18.5))
        level_titles = (
            "S1 前缀 [s1, *, *] · 粗类别",
            "S1/S2 前缀 [s1, s2, *] · 细类别",
            "完整 SID [s1, s2, s3] · 细类别",
        )
        for method_index, method in enumerate(metrics["methods"].values()):
            for column in range(3):
                axis = axes[method_index, column]
                examples = [
                    entry
                    for entry in method["prefix_examples"]
                    if entry["level"] == column + 1
                ]
                for entry in examples:
                    anchor_index = entry["anchor_index"]
                    y = (4 - anchor_index) * 1.6
                    tokens = [str(token) for token in entry["prefix"]] + ["*"] * (
                        2 - column
                    )
                    prefix = "[" + ", ".join(tokens) + "]"
                    label = f"锚点 {anchor_index + 1} · {settings['categories'][anchor_index]}    {prefix}    n={entry['poi_count']:,}"
                    if entry["poi_count"] == 1:
                        label += "（单例）"
                    axis.text(
                        0, y + 0.6, label, fontsize=10, color="#25384b", va="bottom"
                    )
                    category_key = "coarse_category" if column == 0 else "fine_category"
                    _composition_bar(axis, y + 0.26, entry[category_key], region=False)
                    _composition_bar(
                        axis, y - 0.17, entry["region_geohash5"], region=True
                    )
                axis.set_title(
                    f"{METHOD_LABELS[method_index]}\n{level_titles[column]}",
                    fontsize=14,
                    loc="left",
                    pad=15,
                )
                axis.set_xlim(0, 100)
                axis.set_ylim(-0.6, 7.35)
                axis.set_xticks(
                    [0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"]
                )
                axis.tick_params(axis="x", length=0, labelsize=9, colors="#788491")
                axis.set_yticks([])
                for name, spine in axis.spines.items():
                    spine.set_visible(name == "bottom")
                    spine.set_color("#d7dde4")
        figure.suptitle(
            "SID 前缀中的类别与区域：同五个固定 POI 锚点，逐层观察完整桶",
            fontsize=24,
            y=0.982,
        )
        figure.text(
            0.5,
            0.946,
            "类＝类别（蓝色）   区＝Geohash5 空间网格（橙色，非行政区）   各条由深到浅表示占比第 1 / 2 / 3 名；灰色＝其余",
            ha="center",
            fontsize=13,
            color="#435365",
        )
        figure.text(
            0.5,
            0.927,
            "每条统计全库匹配该 SID 前缀的 POI，不限五类、不按 GID 再筛选；横向沿同一锚点展开，纵向比较三种方法。",
            ha="center",
            fontsize=12,
            color="#66717e",
        )
        figure.text(
            0.5,
            0.026,
            "示例由固定随机种子选定，未按纯度挑选；S3 小桶/单例的 100% 不代表更强语义。跨方法相同数字不保证相同语义；两种 A4 的 S1/S2 完全相同。",
            ha="center",
            fontsize=11,
            color="#66717e",
        )
        figure.subplots_adjust(
            left=0.035, right=0.985, top=0.875, bottom=0.06, wspace=0.13, hspace=0.30
        )
        _save(figure, output, "sid_prefix_category_region", dpi)
        plt.close(figure)

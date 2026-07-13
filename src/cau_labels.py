"""High-confidence coarse category labels for CAU-RQ-VAE experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


UNK_LABEL = "UNK"
SUPERVISED_SPLIT = "tag_train"
HELDOUT_SPLIT = "tag_heldout"
UNSUPERVISED_SPLIT = "unsupervised"


@dataclass(frozen=True)
class KeywordRule:
    label: str
    keywords: tuple[str, ...]


# These rules are intentionally narrower than sid_eval.py::SEMANTIC_INFER_RULES.
# They are used for training labels, so each keyword should be a high-confidence
# indicator of the coarse class rather than a broad evaluation fallback.
HIGH_CONFIDENCE_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("parking", ("停车场", "停车位", "停车库", "地下停车", "免费停车场")),
    KeywordRule("ev_charging", ("汽车充电站", "新能源汽车充电站", "充电站", "充电桩", "超级充电", "小桔充电", "特斯拉超级充电", "星星充电", "云快充")),
    KeywordRule("gas_station", ("加油站", "加油加气站", "中石化加油", "中国石化加油", "中石油加油", "中国石油加油", "CNG加气站", "cng加气站", "加气站")),
    KeywordRule("hotel", ("酒店", "宾馆", "民宿", "客栈", "旅馆", "住宿")),
    KeywordRule("public_toilet", ("公共厕所", "公厕", "洗手间", "卫生间")),
    KeywordRule("transit", ("地铁站", "公交站", "火车站", "高铁站", "汽车站", "客运站", "机场", "航站楼", "交通枢纽")),
    KeywordRule("bank", ("银行", "ATM", "atm", "自助银行", "工商银行", "建设银行", "农业银行", "招商银行", "邮政储蓄银行", "中国人民银行")),
    KeywordRule("medical", ("医院", "诊所", "卫生院", "门诊", "社区卫生服务中心", "卫生服务站", "妇幼保健院")),
    KeywordRule("pharmacy", ("药店", "药房", "大药房")),
    KeywordRule("school", ("大学", "学院", "中学", "小学", "幼儿园", "学校")),
    KeywordRule("restaurant", ("餐厅", "饭店", "美食", "小吃", "火锅", "烧烤", "面馆", "拉面", "咖啡", "奶茶", "麻辣烫", "茶饮")),
    KeywordRule("shopping", ("商场", "购物中心", "百货", "超市", "便利店", "菜市场", "农贸市场", "五金店", "建材市场")),
    KeywordRule("scenic", ("景区", "景点", "公园", "博物馆", "纪念馆", "旅游区", "国家地质公园", "风景区")),
    KeywordRule("government", ("人民政府", "政府", "派出所", "公安局", "法院", "税务局", "政务", "村委会", "居委会", "退役军人服务站")),
    KeywordRule("company", ("有限公司", "公司", "企业", "产业园", "科技园", "工业园", "电子厂", "工厂")),
    KeywordRule("residential", ("住宅小区", "小区", "公寓", "花园", "家园", "新村", "社区")),
    KeywordRule("auto_service", ("汽修", "汽车维修", "汽车快修", "补胎", "轮胎", "洗车", "养车", "蓄电池", "汽车服务")),
    KeywordRule("logistics", ("物流", "快递", "菜鸟驿站", "驿站", "配送站")),
    KeywordRule("telecom_electronics", ("中国移动", "中国联通", "中国电信", "手机维修", "电脑维修", "通讯", "vivo", "OPPO", "oppo", "小米之家", "苹果授权")),
    KeywordRule("beauty", ("理发店", "美发", "美容", "美甲")),
    KeywordRule("entertainment", ("网吧", "KTV", "ktv", "电影院", "影城", "台球", "棋牌", "健身房", "健身")),
)


FIELD_PRIORITY = ("category_l1", "tags", "name")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return " ".join(text.split())


def _contains_keyword(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    if re.fullmatch(r"[A-Za-z0-9_]+", keyword):
        return keyword.casefold() in text.casefold()
    return keyword in text


def match_label(value: Any) -> tuple[str, str]:
    text = clean_text(value)
    if not text:
        return UNK_LABEL, ""
    for rule in HIGH_CONFIDENCE_RULES:
        for keyword in rule.keywords:
            if _contains_keyword(text, keyword):
                return rule.label, keyword
    return UNK_LABEL, ""


def assign_row_label(row: pd.Series) -> dict[str, Any]:
    for field in FIELD_PRIORITY:
        label, keyword = match_label(row.get(field, ""))
        if label != UNK_LABEL:
            return {
                "coarse_label": label,
                "label_source_field": field,
                "label_source_value": clean_text(row.get(field, "")),
                "label_rule_keyword": keyword,
                "is_supervised": True,
            }
    return {
        "coarse_label": UNK_LABEL,
        "label_source_field": "",
        "label_source_value": "",
        "label_rule_keyword": "",
        "is_supervised": False,
    }


def build_labels(
    poi_df: pd.DataFrame,
    seed: int = 42,
    heldout_ratio: float = 0.2,
    min_label_count: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"row_id", "poi_id", "name", "category_l1", "tags"}
    missing = sorted(required - set(poi_df.columns))
    if missing:
        raise ValueError(f"Missing required POI columns: {missing}")
    if not poi_df["poi_id"].is_unique:
        raise ValueError("poi_id must be unique")
    if not poi_df["row_id"].is_unique:
        raise ValueError("row_id must be unique")
    if not 0.0 < heldout_ratio < 1.0:
        raise ValueError(f"heldout_ratio must be in (0, 1), got {heldout_ratio}")

    base_cols = ["row_id", "poi_id", "name", "category_l1", "tags"]
    optional_cols = [c for c in ["address", "city", "lat", "lon"] if c in poi_df.columns]
    out = poi_df[base_cols + optional_cols].copy()
    for col in ["poi_id", "name", "category_l1", "tags", *optional_cols]:
        if col in out.columns and out[col].dtype == object:
            out[col] = out[col].map(clean_text)

    assigned = out.apply(assign_row_label, axis=1, result_type="expand")
    out = pd.concat([out, assigned], axis=1)

    raw_supervised = out["is_supervised"].astype(bool)
    raw_counts = out.loc[raw_supervised, "coarse_label"].value_counts().to_dict()
    low_count_labels = {label for label, count in raw_counts.items() if int(count) < int(min_label_count)}
    if low_count_labels:
        low_mask = out["coarse_label"].isin(low_count_labels)
        out.loc[low_mask, "is_supervised"] = False
        out.loc[low_mask, "coarse_label"] = UNK_LABEL
        out.loc[low_mask, "label_source_field"] = ""
        out.loc[low_mask, "label_source_value"] = ""
        out.loc[low_mask, "label_rule_keyword"] = ""

    supervised_labels = out.loc[out["is_supervised"].astype(bool), "coarse_label"]
    label_counts = supervised_labels.value_counts()
    vocab = sorted(label_counts.index.tolist())
    label_to_id = {label: idx for idx, label in enumerate(vocab)}
    out["coarse_label_id"] = out["coarse_label"].map(label_to_id).fillna(-1).astype(np.int64)
    out["split"] = UNSUPERVISED_SPLIT

    rng = np.random.default_rng(seed)
    for label in vocab:
        label_index = out.index[(out["coarse_label"] == label) & out["is_supervised"].astype(bool)].to_numpy()
        label_index = np.sort(label_index)
        shuffled = rng.permutation(label_index)
        heldout_n = max(1, int(round(len(shuffled) * heldout_ratio)))
        heldout_n = min(heldout_n, len(shuffled) - 1)
        heldout_index = shuffled[:heldout_n]
        train_index = shuffled[heldout_n:]
        out.loc[train_index, "split"] = SUPERVISED_SPLIT
        out.loc[heldout_index, "split"] = HELDOUT_SPLIT

    out["is_tag_train"] = out["split"].eq(SUPERVISED_SPLIT)
    out["is_tag_heldout"] = out["split"].eq(HELDOUT_SPLIT)

    summary = {
        "rows": int(len(out)),
        "supervised_rows": int(out["is_supervised"].sum()),
        "supervised_rate": float(out["is_supervised"].mean()),
        "tag_train_rows": int(out["is_tag_train"].sum()),
        "tag_heldout_rows": int(out["is_tag_heldout"].sum()),
        "unsupervised_rows": int(out["split"].eq(UNSUPERVISED_SPLIT).sum()),
        "num_labels": int(len(vocab)),
        "vocab": vocab,
        "label_to_id": label_to_id,
        "raw_matched_label_counts": {str(k): int(v) for k, v in raw_counts.items()},
        "low_count_labels_removed": sorted(low_count_labels),
        "seed": int(seed),
        "heldout_ratio": float(heldout_ratio),
        "min_label_count": int(min_label_count),
    }
    return out, summary


def label_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    supervised = labels[labels["is_supervised"].astype(bool)].copy()
    if supervised.empty:
        return pd.DataFrame()
    grouped = (
        supervised.groupby("coarse_label", dropna=False)
        .agg(
            total=("coarse_label", "size"),
            train=("is_tag_train", "sum"),
            heldout=("is_tag_heldout", "sum"),
            category_l1_source=("label_source_field", lambda s: int((s == "category_l1").sum())),
            tags_source=("label_source_field", lambda s: int((s == "tags").sum())),
            name_source=("label_source_field", lambda s: int((s == "name").sum())),
            example_keywords=("label_rule_keyword", lambda s: " / ".join(s.astype(str).value_counts().head(5).index.tolist())),
            example_names=("name", lambda s: " / ".join(s.astype(str).head(5).tolist())),
        )
        .reset_index()
        .sort_values(["total", "coarse_label"], ascending=[False, True])
    )
    grouped["heldout_rate"] = grouped["heldout"] / grouped["total"]
    return grouped


def write_vocab_json(summary: dict[str, Any], path: str | Path) -> None:
    payload = {
        "vocab": summary["vocab"],
        "label_to_id": summary["label_to_id"],
        "unk_label": UNK_LABEL,
        "unk_label_id": -1,
        "supervised_split": SUPERVISED_SPLIT,
        "heldout_split": HELDOUT_SPLIT,
        "unsupervised_split": UNSUPERVISED_SPLIT,
        "seed": summary["seed"],
        "heldout_ratio": summary["heldout_ratio"],
        "min_label_count": summary["min_label_count"],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

"""Rule-based semantic IDs for POI retrieval MVP."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def stable_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def slug(value: Any, default: str = "unknown", max_len: int = 32) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text).strip("_")
    return (text or default)[:max_len]


def build_sid(city: str, lat: Any, lon: Any, category: str, poi_id: str) -> str:
    region = slug(city)
    if region == "unknown" and lat not in ("", None) and lon not in ("", None):
        try:
            region = f"gh_{float(lat):.2f}_{float(lon):.2f}".replace("-", "m").replace(".", "p")
        except (TypeError, ValueError):
            region = "unknown"
    category_part = slug(category)
    entity = stable_hash(str(poi_id), length=10)
    return f"R_{region}|S_{category_part}|E_{entity}"


def sid_parts(sid: str) -> tuple[str, str, str]:
    parts = dict(part.split("_", 1) for part in sid.split("|") if "_" in part)
    return parts.get("R", "unknown"), parts.get("S", "unknown"), parts.get("E", "unknown")


def write_sid_maps(pois: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    poi_id_to_sid: dict[str, str] = {}
    sid_to_poi_ids: dict[str, list[str]] = defaultdict(list)
    prefix_to_pois: dict[str, set[str]] = defaultdict(set)

    for row in pois.fillna("").itertuples(index=False):
        row_dict = row._asdict()
        poi_id = str(row_dict["poi_id"])
        sid = build_sid(
            city=str(row_dict.get("city", "")),
            lat=row_dict.get("lat", ""),
            lon=row_dict.get("lon", ""),
            category=str(row_dict.get("category_l1", "")),
            poi_id=poi_id,
        )
        poi_id_to_sid[poi_id] = sid
        sid_to_poi_ids[sid].append(poi_id)
        region, semantic, _ = sid_parts(sid)
        prefix_to_pois[f"R_{region}"].add(poi_id)
        prefix_to_pois[f"R_{region}|S_{semantic}"].add(poi_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "poi_id_to_sid.json").write_text(
        json.dumps(poi_id_to_sid, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "sid_to_poi_ids.json").write_text(
        json.dumps(dict(sid_to_poi_ids), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    collisions = {sid: ids for sid, ids in sid_to_poi_ids.items() if len(ids) > 1}
    return {
        "sid_count": len(sid_to_poi_ids),
        "poi_count": len(poi_id_to_sid),
        "collision_sid_count": len(collisions),
        "collision_poi_count": sum(len(ids) for ids in collisions.values()),
        "collision_rate": (len(collisions) / len(sid_to_poi_ids)) if sid_to_poi_ids else 0.0,
        "region_count": len(Counter(sid_parts(sid)[0] for sid in sid_to_poi_ids)),
        "semantic_count": len(Counter(sid_parts(sid)[1] for sid in sid_to_poi_ids)),
        "entity_count": len(Counter(sid_parts(sid)[2] for sid in sid_to_poi_ids)),
        "prefix_bucket_count": len(prefix_to_pois),
    }


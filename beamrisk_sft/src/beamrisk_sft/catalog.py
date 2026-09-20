"""TIGER SID lookup and strict directory-equivalence protection."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .errors import BeamRiskError


ALIAS_SEPARATOR = re.compile(r"[|｜]")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return "".join(
        unicodedata.normalize("NFKC", str(value)).casefold().split()
    )


def normalized_aliases(value: object) -> tuple[str, ...]:
    normalized = {
        normalize_text(item)
        for item in ALIAS_SEPARATOR.split("" if value is None else str(value))
        if normalize_text(item)
    }
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class PoiSignature:
    displayname: str
    address: str
    aliases: tuple[str, ...]
    lng: float
    lat: float


def build_tiger_key_to_poi(
    mapping_path: Path,
    requested_keys: Iterable[str],
    *,
    require_all: bool = True,
) -> dict[str, str]:
    keys = set(requested_keys)
    if not keys:
        return {}
    # PyArrow 19 requires ``columns`` to be a list and ``value_set`` to be an
    # Arrow Array.  Keep both explicit: this lookup runs only after hours of
    # distributed mining, so a permissive version-dependent coercion is unsafe.
    table = pq.read_table(mapping_path, columns=["poi_id", "tiger_id_key"])
    value_set = pa.array(sorted(keys), type=table["tiger_id_key"].type)
    filtered = table.filter(pc.is_in(table["tiger_id_key"], value_set=value_set))
    result = {
        str(key): str(poi_id)
        for poi_id, key in zip(
            filtered["poi_id"].to_pylist(),
            filtered["tiger_id_key"].to_pylist(),
        )
    }
    if filtered.num_rows != len(result):
        raise BeamRiskError("TIGER mapping 的候选路径不是一对一")
    missing_keys = keys - set(result)
    if require_all and missing_keys:
        missing = sorted(missing_keys)[:10]
        raise BeamRiskError(f"TIGER mapping 缺少候选路径：{missing}")
    return result


def load_poi_signatures(
    catalog_dir: Path,
    poi_ids: Iterable[str],
) -> dict[str, PoiSignature]:
    requested = set(poi_ids)
    if not requested:
        return {}
    result: dict[str, PoiSignature] = {}
    paths = sorted(catalog_dir.glob("part-*.json"))
    if not paths:
        raise BeamRiskError(f"POI catalog 没有 part-*.json：{catalog_dir}")
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise BeamRiskError(
                        f"POI catalog JSON 无效：{path}:{line_number}"
                    ) from error
                poi_id = value.get("poi_id")
                if poi_id not in requested:
                    continue
                lng = value.get("lng")
                lat = value.get("lat")
                if not isinstance(lng, (int, float)) or not isinstance(lat, (int, float)):
                    raise BeamRiskError(f"POI 坐标无效：{poi_id}")
                result[str(poi_id)] = PoiSignature(
                    displayname=normalize_text(value.get("displayname")),
                    address=normalize_text(value.get("address")),
                    aliases=normalized_aliases(value.get("alias")),
                    lng=float(lng),
                    lat=float(lat),
                )
        if len(result) == len(requested):
            break
    if len(result) != len(requested):
        missing = sorted(requested - set(result))[:10]
        raise BeamRiskError(f"POI catalog 缺少目标：{missing}")
    return result


def strictly_equivalent(
    target_poi_id: str,
    negative_poi_id: str,
    signatures: Mapping[str, PoiSignature],
) -> bool:
    if target_poi_id == negative_poi_id:
        return True
    try:
        return signatures[target_poi_id] == signatures[negative_poi_id]
    except KeyError as error:
        raise BeamRiskError(f"严格重复检查缺少 POI signature：{error}") from error

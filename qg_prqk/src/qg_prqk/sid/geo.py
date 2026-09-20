"""Geohash and local-geometry helpers for Semantic-ID construction."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
EARTH_RADIUS_METERS = 6_371_008.8
GEO_FEATURE_DIM = 5


class GeoFeatureError(ValueError):
    """Raised when coordinates or local-geometry inputs are invalid."""


def encode_geohash_tokens(longitudes: np.ndarray, latitudes: np.ndarray, *, length: int = 6) -> np.ndarray:
    """Encode longitude-first standard geohash tokens without external packages."""
    lng = np.asarray(longitudes, dtype=np.float64)
    lat = np.asarray(latitudes, dtype=np.float64)
    if lng.shape != lat.shape or lng.ndim != 1 or length <= 0:
        raise GeoFeatureError("P7 经纬度必须是一维同 shape，geohash length 必须为正")
    if not np.isfinite(lng).all() or not np.isfinite(lat).all():
        raise GeoFeatureError("P7 经纬度包含 NaN/Inf")
    if np.any(lng < -180.0) or np.any(lng > 180.0) or np.any(lat < -90.0) or np.any(lat > 90.0):
        raise GeoFeatureError("P7 经纬度超出合法范围")
    result = np.empty((len(lng), length), dtype=np.uint8)
    lng_lo = np.full(len(lng), -180.0)
    lng_hi = np.full(len(lng), 180.0)
    lat_lo = np.full(len(lat), -90.0)
    lat_hi = np.full(len(lat), 90.0)
    longitude_bit = True
    bit = 0
    token = np.zeros(len(lng), dtype=np.uint8)
    for position in range(length * 5):
        if longitude_bit:
            midpoint = (lng_lo + lng_hi) * 0.5
            high = lng >= midpoint
            lng_lo = np.where(high, midpoint, lng_lo)
            lng_hi = np.where(high, lng_hi, midpoint)
        else:
            midpoint = (lat_lo + lat_hi) * 0.5
            high = lat >= midpoint
            lat_lo = np.where(high, midpoint, lat_lo)
            lat_hi = np.where(high, lat_hi, midpoint)
        token = (token << 1) | high.astype(np.uint8)
        bit += 1
        if bit == 5:
            result[:, position // 5] = token
            token.fill(0)
            bit = 0
        longitude_bit = not longitude_bit
    return result


def geohash_cell_centers(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode standard geohash tokens to cell-center longitude and latitude."""
    values = np.asarray(tokens)
    if values.ndim != 2 or not values.shape[1] or values.dtype.kind not in "iu":
        raise GeoFeatureError("P7 GID tokens 必须是二维整数数组")
    if np.any(values < 0) or np.any(values >= len(GEOHASH_ALPHABET)):
        raise GeoFeatureError("P7 GID token 超出 geohash alphabet")
    rows = len(values)
    lng_lo = np.full(rows, -180.0)
    lng_hi = np.full(rows, 180.0)
    lat_lo = np.full(rows, -90.0)
    lat_hi = np.full(rows, 90.0)
    longitude_bit = True
    for column in range(values.shape[1]):
        token = values[:, column].astype(np.uint8, copy=False)
        for shift in range(4, -1, -1):
            high = ((token >> shift) & 1).astype(bool)
            if longitude_bit:
                midpoint = (lng_lo + lng_hi) * 0.5
                lng_lo = np.where(high, midpoint, lng_lo)
                lng_hi = np.where(high, lng_hi, midpoint)
            else:
                midpoint = (lat_lo + lat_hi) * 0.5
                lat_lo = np.where(high, midpoint, lat_lo)
                lat_hi = np.where(high, lat_hi, midpoint)
            longitude_bit = not longitude_bit
    return (lng_lo + lng_hi) * 0.5, (lat_lo + lat_hi) * 0.5


def pack_gid(tokens: np.ndarray, *, precision: int = 6) -> np.ndarray:
    """Pack a geohash prefix into deterministic unsigned integer keys."""
    values = np.asarray(tokens)
    if values.ndim != 2 or not 1 <= precision <= values.shape[1]:
        raise GeoFeatureError("P7 GID precision 非法")
    result = np.zeros(len(values), dtype=np.uint32)
    for column in range(precision):
        result = (result << np.uint32(5)) | values[:, column].astype(np.uint32)
    return result


def parent_group_ids(gid6: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Map `(GID6,S1,S2)` tuples to stable dense group ids."""
    gid = pack_gid(gid6, precision=6).astype(np.uint64)
    first = np.asarray(s1, dtype=np.int64)
    second = np.asarray(s2, dtype=np.int64)
    if first.shape != gid.shape or second.shape != gid.shape:
        raise GeoFeatureError("P7 parent assignment 与 GID 行未对齐")
    if np.any(first < 0) or np.any(first >= 512) or np.any(second < 0) or np.any(second >= 512):
        raise GeoFeatureError("P7 parent assignment 超出 512 code")
    keys = (gid << np.uint64(18)) | (first.astype(np.uint64) << np.uint64(9)) | second.astype(np.uint64)
    _, inverse = np.unique(keys, return_inverse=True)
    return inverse.astype(np.int64)


def _group_means(values: np.ndarray, groups: np.ndarray, group_count: int) -> np.ndarray:
    sums = np.bincount(groups, weights=values, minlength=group_count)
    counts = np.bincount(groups, minlength=group_count)
    return sums / np.maximum(counts, 1)


def local_geo_features(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    gid6: np.ndarray,
    parent_groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    """Build standardized five-dimensional local Geo features per POI."""
    lng = np.asarray(longitudes, dtype=np.float64)
    lat = np.asarray(latitudes, dtype=np.float64)
    groups = np.asarray(parent_groups, dtype=np.int64)
    if lng.shape != lat.shape or lng.shape != groups.shape or lng.ndim != 1 or not len(lng):
        raise GeoFeatureError("P7 local Geo 输入行未对齐")
    if np.any(groups < 0):
        raise GeoFeatureError("P7 parent group 不得为负")
    group_count = int(groups.max()) + 1
    counts = np.bincount(groups, minlength=group_count)
    singleton = counts[groups] == 1
    cell_lng, cell_lat = geohash_cell_centers(gid6)
    lat_radians = np.deg2rad(lat)
    cell_dx = EARTH_RADIUS_METERS * np.deg2rad(lng - cell_lng) * np.cos(lat_radians)
    cell_dy = EARTH_RADIUS_METERS * np.deg2rad(lat - cell_lat)

    reference_lat = float(np.median(lat))
    x = EARTH_RADIUS_METERS * np.deg2rad(lng) * math.cos(math.radians(reference_lat))
    y = EARTH_RADIUS_METERS * np.deg2rad(lat)
    center_x = _group_means(x, groups, group_count)[groups]
    center_y = _group_means(y, groups, group_count)[groups]
    delta_x = x - center_x
    delta_y = y - center_y
    distance = np.hypot(delta_x, delta_y)
    bearing_valid = distance > 1.0e-12
    bearing_sin = np.divide(delta_x, distance, out=np.zeros_like(distance), where=bearing_valid)
    bearing_cos = np.divide(delta_y, distance, out=np.zeros_like(distance), where=bearing_valid)
    raw = np.column_stack((cell_dx, cell_dy, np.log1p(distance), bearing_sin, bearing_cos))
    raw[singleton] = 0.0
    eligible = ~singleton
    if not np.any(eligible):
        center = np.zeros(GEO_FEATURE_DIM, dtype=np.float64)
        scale = np.ones(GEO_FEATURE_DIM, dtype=np.float64)
        fallback = np.ones(GEO_FEATURE_DIM, dtype=bool)
    else:
        active = raw[eligible]
        center = np.median(active, axis=0)
        q75, q25 = np.quantile(active, [0.75, 0.25], axis=0)
        scale = q75 - q25
        fallback = (~np.isfinite(scale)) | (scale <= 1.0e-12)
        if np.any(fallback):
            center[fallback] = active[:, fallback].mean(axis=0)
            scale[fallback] = active[:, fallback].std(axis=0)
        scale[(~np.isfinite(scale)) | (scale <= 1.0e-12)] = 1.0
    standardized = (raw - center) / scale
    standardized[singleton] = 0.0
    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    normalized = np.divide(
        standardized,
        norms,
        out=np.zeros_like(standardized),
        where=norms > 1.0e-12,
    ).astype(np.float32)
    stats: dict[str, object] = {
        "feature_names": [
            "cell_dx",
            "cell_dy",
            "log1p_parent_distance",
            "parent_bearing_sin",
            "parent_bearing_cos",
        ],
        "center": center.tolist(),
        "scale": scale.tolist(),
        "fallback_to_mean_std": fallback.tolist(),
        "reference_latitude": reference_lat,
        "singleton_rows": int(singleton.sum()),
        "non_singleton_rows": int(eligible.sum()),
    }
    return normalized, stats, singleton


def geohash_strings(tokens: np.ndarray) -> Iterable[str]:
    """Yield readable geohashes for bounded diagnostics."""
    for row in np.asarray(tokens):
        yield "".join(GEOHASH_ALPHABET[int(value)] for value in row)

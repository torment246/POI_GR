"""Continuous S1/S2-parent geometry for the no-GID S3 variant."""

from __future__ import annotations

import math

import numpy as np

from qg_prqk.sid.geo import EARTH_RADIUS_METERS, GEO_FEATURE_DIM, GeoFeatureError


def s1_s2_parent_keys(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Return stable 18-bit `(S1,S2)` parent keys without any GID input."""
    first = np.asarray(s1, dtype=np.int64)
    second = np.asarray(s2, dtype=np.int64)
    if first.ndim != 1 or first.shape != second.shape:
        raise GeoFeatureError("no-GID S1/S2 parent 行未对齐")
    if np.any(first < 0) or np.any(first >= 512) or np.any(second < 0) or np.any(second >= 512):
        raise GeoFeatureError("no-GID S1/S2 parent code 超界")
    return first * 512 + second


def _group_means(values: np.ndarray, groups: np.ndarray, group_count: int) -> np.ndarray:
    sums = np.bincount(groups, weights=values, minlength=group_count)
    counts = np.bincount(groups, minlength=group_count)
    return sums / np.maximum(counts, 1)


def s1_s2_parent_geo_features(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    parent_keys: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    """Build normalized continuous geometry relative to each S1/S2 parent center."""
    lng = np.asarray(longitudes, dtype=np.float64)
    lat = np.asarray(latitudes, dtype=np.float64)
    parents = np.asarray(parent_keys, dtype=np.int64)
    if lng.ndim != 1 or lng.shape != lat.shape or lng.shape != parents.shape or not len(lng):
        raise GeoFeatureError("no-GID 经纬度与 S1/S2 parent 未对齐")
    if not np.isfinite(lng).all() or not np.isfinite(lat).all() or np.any(parents < 0):
        raise GeoFeatureError("no-GID Geo 输入包含非法值")
    if np.any(lng < -180.0) or np.any(lng > 180.0) or np.any(lat < -90.0) or np.any(lat > 90.0):
        raise GeoFeatureError("no-GID 经纬度超出合法范围")

    group_count = int(parents.max()) + 1
    counts = np.bincount(parents, minlength=group_count)
    singleton = counts[parents] == 1
    reference_lat = float(np.median(lat))
    x = EARTH_RADIUS_METERS * np.deg2rad(lng) * math.cos(math.radians(reference_lat))
    y = EARTH_RADIUS_METERS * np.deg2rad(lat)
    center_x = _group_means(x, parents, group_count)[parents]
    center_y = _group_means(y, parents, group_count)[parents]
    delta_x = x - center_x
    delta_y = y - center_y
    distance = np.hypot(delta_x, delta_y)
    valid_bearing = distance > 1.0e-12
    bearing_sin = np.divide(delta_x, distance, out=np.zeros_like(distance), where=valid_bearing)
    bearing_cos = np.divide(delta_y, distance, out=np.zeros_like(distance), where=valid_bearing)
    raw = np.column_stack((delta_x, delta_y, np.log1p(distance), bearing_sin, bearing_cos))
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
    features = np.divide(
        standardized,
        norms,
        out=np.zeros_like(standardized),
        where=norms > 1.0e-12,
    ).astype(np.float32)
    return features, {
        "feature_names": [
            "parent_dx", "parent_dy", "log1p_parent_distance", "parent_bearing_sin", "parent_bearing_cos"
        ],
        "center": center.tolist(),
        "scale": scale.tolist(),
        "fallback_to_mean_std": fallback.tolist(),
        "reference_latitude": reference_lat,
        "parent_key": ["s1", "s2"],
        "gid_or_geohash_used": False,
        "singleton_rows": int(singleton.sum()),
        "non_singleton_rows": int(eligible.sum()),
    }, singleton

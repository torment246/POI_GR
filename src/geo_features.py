"""Geographic feature utilities for MobilityBench POIs."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
EARTH_RADIUS_KM = 6371.0088


def validate_coordinate(lat: float, lon: float) -> None:
    if not math.isfinite(float(lat)) or not math.isfinite(float(lon)):
        raise ValueError(f"Latitude/longitude must be finite, got lat={lat}, lon={lon}")
    if not -90.0 <= float(lat) <= 90.0:
        raise ValueError(f"Latitude out of range [-90, 90]: {lat}")
    if not -180.0 <= float(lon) <= 180.0:
        raise ValueError(f"Longitude out of range [-180, 180]: {lon}")


def invalid_latlon_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    return (
        ~np.isfinite(lat_arr)
        | ~np.isfinite(lon_arr)
        | (lat_arr < -90.0)
        | (lat_arr > 90.0)
        | (lon_arr < -180.0)
        | (lon_arr > 180.0)
    )


def encode_geohash(lat: float, lon: float, precision: int = 6) -> str:
    """Encode latitude/longitude with the standard geohash base32 alphabet."""
    if precision <= 0:
        raise ValueError(f"Geohash precision must be positive, got {precision}")
    validate_coordinate(lat, lon)

    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even_bit = True
    geohash: list[str] = []

    while len(geohash) < precision:
        if even_bit:
            mid = (lon_interval[0] + lon_interval[1]) / 2.0
            if lon >= mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2.0
            if lat >= mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid

        even_bit = not even_bit
        if bit < 4:
            bit += 1
        else:
            geohash.append(GEOHASH_BASE32[ch])
            bit = 0
            ch = 0

    return "".join(geohash)


def encode_geohashes(lat: Iterable[float], lon: Iterable[float], precision: int) -> list[str]:
    return [encode_geohash(float(a), float(b), precision=precision) for a, b in zip(lat, lon)]


def feature_names(n_anchors: int) -> list[str]:
    names = ["lat_norm", "lon_norm", "sin_lat", "cos_lat", "sin_lon", "cos_lon"]
    for anchor_id in range(n_anchors):
        names.extend(
            [
                f"anchor_{anchor_id}_distance_km",
                f"anchor_{anchor_id}_bearing_sin",
                f"anchor_{anchor_id}_bearing_cos",
            ]
        )
    return names


def latlon_base_features(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_rad = np.radians(lat_arr)
    lon_rad = np.radians(lon_arr)
    return np.column_stack(
        [
            lat_arr / 90.0,
            lon_arr / 180.0,
            np.sin(lat_rad),
            np.cos(lat_rad),
            np.sin(lon_rad),
            np.cos(lon_rad),
        ]
    )


def anchor_distance_bearing_features(
    poi_lat: np.ndarray,
    poi_lon: np.ndarray,
    anchor_latlon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return distance km, bearing sin, and bearing cos for every POI-anchor pair."""
    lat = np.radians(np.asarray(poi_lat, dtype=np.float64))[:, None]
    lon = np.radians(np.asarray(poi_lon, dtype=np.float64))[:, None]
    anchors = np.asarray(anchor_latlon, dtype=np.float64)
    anchor_lat = np.radians(anchors[:, 0])[None, :]
    anchor_lon = np.radians(anchors[:, 1])[None, :]

    dlat = lat - anchor_lat
    dlon = lon - anchor_lon
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(anchor_lat) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    hav = np.clip(hav, 0.0, 1.0)
    distance_km = EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(hav), np.sqrt(1.0 - hav))

    y = np.sin(dlon) * np.cos(lat)
    x = np.cos(anchor_lat) * np.sin(lat) - np.sin(anchor_lat) * np.cos(lat) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    return distance_km, np.sin(bearing), np.cos(bearing)


def build_raw_geo_features(
    lat: np.ndarray,
    lon: np.ndarray,
    anchor_latlon: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    base = latlon_base_features(lat, lon)
    distances, bearing_sin, bearing_cos = anchor_distance_bearing_features(lat, lon, anchor_latlon)
    columns = [base]
    n_anchors = anchor_latlon.shape[0]
    for anchor_id in range(n_anchors):
        columns.append(distances[:, [anchor_id]])
        columns.append(bearing_sin[:, [anchor_id]])
        columns.append(bearing_cos[:, [anchor_id]])
    return np.hstack(columns), feature_names(n_anchors), distances


def standardize_features(raw_features: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_features, dtype=np.float64)
    mean = raw.mean(axis=0)
    raw_std = raw.std(axis=0)
    std = np.where(raw_std < eps, 1.0, raw_std)
    standardized = (raw - mean) / std
    return standardized.astype(np.float32), mean, std, raw_std

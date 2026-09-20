from __future__ import annotations

import unittest

import numpy as np

from qg_prqk.sid.geo import GEOHASH_ALPHABET, encode_geohash_tokens, geohash_strings, local_geo_features, parent_group_ids


class GeoFeaturesTest(unittest.TestCase):
    def test_standard_longitude_first_geohash(self) -> None:
        tokens = encode_geohash_tokens(np.array([-5.6]), np.array([42.6]), length=6)
        self.assertEqual(list(geohash_strings(tokens)), ["ezs42e"])
        self.assertEqual(tokens.tolist(), [[GEOHASH_ALPHABET.index(value) for value in "ezs42e"]])

    def test_parent_groups_and_singleton_geo_zero(self) -> None:
        lng = np.array([116.3970, 116.3971, 116.45])
        lat = np.array([39.9080, 39.9081, 39.95])
        gid = encode_geohash_tokens(lng, lat)
        gid[1] = gid[0]
        parents = parent_group_ids(gid, np.array([1, 1, 2]), np.array([3, 3, 4]))
        self.assertEqual(parents[0], parents[1])
        self.assertNotEqual(parents[0], parents[2])
        features, stats, singleton = local_geo_features(lng, lat, gid, parents)
        self.assertEqual(features.shape, (3, 5))
        self.assertEqual(singleton.tolist(), [False, False, True])
        np.testing.assert_array_equal(features[2], np.zeros(5, dtype=np.float32))
        self.assertEqual(stats["singleton_rows"], 1)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()

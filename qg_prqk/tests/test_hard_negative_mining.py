from __future__ import annotations

import unittest

import numpy as np

from qg_prqk.hard_negative_mining import (
    PoiMetadata,
    in_batch_candidates,
    lexical_metadata_candidates,
    local_geo_candidates,
    merge_negative_sources,
    semantic_ann_candidates,
)


def poi(
    row: int,
    name: str,
    *,
    alias: str = "",
    category: str = "商场",
    category_code: str = "100",
    address: str = "北京市朝阳区建国路",
    lat: float = 39.90,
    lng: float = 116.40,
) -> PoiMetadata:
    return PoiMetadata(
        row=row,
        poi_id=f"poi-{row}",
        displayname=name,
        alias=alias,
        category=category,
        category_code=category_code,
        address=address,
        lat=lat,
        lng=lng,
    )


class HardNegativeMiningTest(unittest.TestCase):
    def test_semantic_ann_masks_target_and_reasonable_positive(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.9, 0.1], [0.0, 1.0]],
            dtype=np.float32,
        )
        candidates = semantic_ann_candidates(
            0, embeddings, top_k=2, false_negative_rows={1}, chunk_rows=2
        )
        self.assertEqual([item.poi_row for item in candidates], [2, 3])

    def test_lexical_metadata_prefers_name_alias_overlap(self) -> None:
        target = poi(0, "北京大悦城", alias="朝阳大悦城")
        candidates = lexical_metadata_candidates(
            target,
            [
                target,
                poi(1, "朝阳大悦城购物中心"),
                poi(2, "远郊商场"),
                poi(3, "北京大悦城停车场"),
            ],
            top_k=2,
            false_negative_rows={3},
        )
        self.assertEqual(candidates[0].poi_row, 1)
        self.assertNotIn(3, [item.poi_row for item in candidates])

    def test_local_geo_prefers_same_category_then_distance(self) -> None:
        target = poi(0, "目标")
        candidates = local_geo_candidates(
            target,
            [
                target,
                poi(1, "近但异类", category="医院", category_code="200", lat=39.9001),
                poi(2, "较远同类", lat=39.91),
                poi(3, "最近同类", lat=39.901),
            ],
            top_k=3,
        )
        self.assertEqual([item.poi_row for item in candidates], [3, 2, 1])

    def test_merge_keeps_source_priority_and_masks_false_negatives(self) -> None:
        in_batch = in_batch_candidates(0, [0, 1, 2, 2], {2})
        semantic = semantic_ann_candidates(
            0,
            np.eye(5, dtype=np.float32),
            top_k=4,
            false_negative_rows={2},
            chunk_rows=3,
        )
        merged = merge_negative_sources(
            target_row=0,
            false_negative_rows={2},
            in_batch=in_batch,
            semantic_ann=semantic,
            lexical_metadata=[],
            local_geo=[],
            semantic_budget=2,
            lexical_budget=0,
            local_geo_budget=0,
        )
        rows = [item.poi_row for item in merged]
        self.assertEqual(rows[0], 1)
        self.assertNotIn(0, rows)
        self.assertNotIn(2, rows)
        self.assertEqual(len(rows), len(set(rows)))


if __name__ == "__main__":
    unittest.main()

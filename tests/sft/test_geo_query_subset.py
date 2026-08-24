import json
import tempfile
import unittest
from pathlib import Path

from poi_gr.pid.trie import sha256_file
from poi_gr.sft.geo_query_subset import (
    build_geo_query_validation_subset,
    geographic_relation_types,
    normalize_query,
)


def record(
    index: int,
    query: str,
    *,
    user_gid: str = "wx4g0b",
    target_gid: str = "wx4g1c",
) -> dict:
    user = "".join(f"<G_{value}>" for value in user_gid)
    target = "".join(f"<G_{value}>" for value in target_gid) + "<S1_1><S2_2><S3_3>"
    return {
        "sample_id": f"sample-{index}",
        "messages": [
            {
                "role": "user",
                "content": (
                    "<CURRENT>\n"
                    f"<USER_GID>{user}</USER_GID>\n"
                    f"<QUERY>{query}</QUERY>\n"
                    "</CURRENT>"
                ),
            },
            {"role": "assistant", "content": target},
        ],
        "order_id": f"order-{index}",
        "searchid": f"search-{index}",
        "target_poi_id": f"poi-{index}",
        "target_pid_key": target,
        "requires_dedup": False,
        "history_length": 0,
        "split": "valid",
    }


class GeoQuerySubsetTest(unittest.TestCase):
    def test_normalization_and_relation_categories(self) -> None:
        self.assertEqual(normalize_query(" 地铁  站 A口 "), "地铁站a口")
        self.assertEqual(
            geographic_relation_types("北京西站附近酒店"),
            ("nearby",),
        )
        self.assertIn("opposite_or_intersection", geographic_relation_types("商场对面咖啡"))
        self.assertEqual(geographic_relation_types("普通餐厅"), ())

    def test_builds_stable_result_independent_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "valid.jsonl"
            values = [
                record(0, "北京西站附近酒店"),
                record(1, "三里屯路口咖啡店", target_gid="wx4erf"),
                record(2, "地铁国贸站a口餐厅"),
                record(3, "普通餐厅"),
            ]
            source.write_text(
                "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
                encoding="utf-8",
            )
            output = root / "output"
            subset = build_geo_query_validation_subset(
                source,
                output,
                source_rows=4,
                source_sha256=sha256_file(source),
                subset_size=2,
                min_query_length=6,
            )
            self.assertEqual(subset.row_count, 2)
            self.assertEqual(subset.manifest["eligible_rows"], 3)
            self.assertFalse(subset.manifest["selection_uses_model_outputs"])
            self.assertFalse(
                subset.manifest["selection_uses_target_geo_prefix_length"]
            )
            selected = [
                json.loads(line)
                for line in subset.data_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(selected), 2)
            self.assertEqual(
                [value["sample_id"] for value in selected],
                sorted(
                    (value["sample_id"] for value in selected),
                    key=lambda value: int(value.split("-")[1]),
                ),
            )
            reused = build_geo_query_validation_subset(
                source,
                output,
                source_rows=4,
                source_sha256=sha256_file(source),
                subset_size=2,
                min_query_length=6,
            )
            self.assertEqual(reused.sha256, subset.sha256)


if __name__ == "__main__":
    unittest.main()

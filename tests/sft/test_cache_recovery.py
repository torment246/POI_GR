import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from scripts.sft.recover_tokenized_cache import (
    PACKED_COLUMNS,
    _link_split,
    discover_arrow_group,
    inspect_packed_shards,
)


def write_arrow(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


class RecoverSftTokenizedCacheTest(unittest.TestCase):
    def test_discovers_validates_and_links_packed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_dir = root / "raw" / "json" / "default" / "0.0.0" / "hash"
            dataset_info = {
                "splits": {"train": {"num_examples": 4}},
            }
            cache_dir.mkdir(parents=True)
            (cache_dir / "dataset_info.json").write_text(
                json.dumps(dataset_info),
                encoding="utf-8",
            )
            table = pa.table(
                {
                    "input_ids": [[1, 2, 3], [4, 5, 6]],
                    "attention_mask": [[1, 1, 1], [1, 1, 1]],
                    "position_ids": [[0, 1, 2], [0, 1, 2]],
                    "labels": [[-100, 2, 3], [-100, 5, 6]],
                    "images": pa.nulls(2),
                    "videos": pa.nulls(2),
                    "audios": pa.nulls(2),
                }
            )
            files = []
            for index in range(2):
                path = cache_dir / f"cache-abcdef_{index:05d}_of_00002.arrow"
                write_arrow(path, table)
                files.append(path)

            group = discover_arrow_group(
                root,
                source_rows=4,
                required_columns=PACKED_COLUMNS,
            )
            self.assertEqual(group.files, tuple(files))
            counts = inspect_packed_shards(
                group.files,
                cutoff_len=3,
                workers=2,
            )
            self.assertEqual(sum(counts.values()), 4)

            destination = root / "formal" / "train"
            _link_split(destination, group.files, counts)
            state = json.loads(
                (destination / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(state["_data_files"]), 2)
            self.assertEqual(
                (destination / "data-00000-of-00002.arrow").stat().st_ino,
                files[0].stat().st_ino,
            )


if __name__ == "__main__":
    unittest.main()

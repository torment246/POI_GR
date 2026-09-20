from __future__ import annotations

import copy
import unittest

from qg_prqk.sft.training import (
    SftTrainingError,
    VARIANT_TMP_TAGS,
    _validate_algorithm_config,
    load_training_config,
    run_llamafactory_launcher,
)


class SftTrainingConfigTest(unittest.TestCase):
    def test_two_variants_share_algorithm_and_global_batch(self) -> None:
        gid = load_training_config("a4_gid_parent")
        nogid = load_training_config("a4_nogid")
        ignored = {
            "variant",
            "dataset",
            "eval_dataset",
            "tokenized_path",
            "output_dir",
            "logging_dir",
        }
        self.assertEqual(
            {key: value for key, value in gid.items() if key not in ignored},
            {key: value for key, value in nogid.items() if key not in ignored},
        )

    def test_rejects_algorithm_drift(self) -> None:
        config = copy.deepcopy(load_training_config("a4_nogid"))
        config["cutoff_len"] = 512
        with self.assertRaises(SftTrainingError):
            _validate_algorithm_config(config)

    def test_variants_use_distinct_short_runtime_tmp_tags(self) -> None:
        self.assertEqual(set(VARIANT_TMP_TAGS), {"a4_gid_parent", "a4_nogid"})
        self.assertEqual(len(set(VARIANT_TMP_TAGS.values())), 2)
        self.assertTrue(all(len(tag) <= 8 for tag in VARIANT_TMP_TAGS.values()))

    def test_successful_distributed_exit_continues_postprocessing(self) -> None:
        calls = []

        def launch() -> None:
            calls.append("launch")
            raise SystemExit(0)

        run_llamafactory_launcher(launch)
        calls.append("postprocess")
        self.assertEqual(calls, ["launch", "postprocess"])

    def test_failed_distributed_exit_is_preserved(self) -> None:
        with self.assertRaises(SystemExit) as context:
            run_llamafactory_launcher(
                lambda: (_ for _ in ()).throw(SystemExit(9))
            )
        self.assertEqual(context.exception.code, 9)


if __name__ == "__main__":
    unittest.main()

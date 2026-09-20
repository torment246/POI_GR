from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from unittest import mock

from qg_prqk import cli


class CliTest(unittest.TestCase):
    def test_help_groups_commands_by_responsibility(self) -> None:
        output = cli.format_help()

        self.assertIn("数据与 Query 监督", output)
        self.assertIn("Query Adapter", output)
        self.assertIn("Semantic ID", output)
        self.assertIn("SFT", output)
        self.assertIn("build-query-graph", output)
        self.assertIn("build-base-codebook", output)
        self.assertNotIn("build_p5", output)

    def test_dispatch_loads_only_selected_command(self) -> None:
        received: list[str] = []

        def command_main(arguments: list[str] | None) -> int:
            received.extend(arguments or [])
            self.assertEqual(sys.argv[0], "python -m qg_prqk build-query-stats")
            return 7

        module = types.SimpleNamespace(main=command_main)
        original_program = sys.argv[0]
        with mock.patch.object(cli.importlib, "import_module", return_value=module) as load:
            result = cli.main(["build-query-stats", "--config", "example.yaml"])

        self.assertEqual(result, 7)
        self.assertEqual(received, ["--config", "example.yaml"])
        self.assertEqual(sys.argv[0], original_program)
        load.assert_called_once_with("qg_prqk.commands.query_statistics")

    def test_unknown_command_returns_usage_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = cli.main(["p5"])

        self.assertEqual(result, 2)
        self.assertIn("未知命令：p5", stderr.getvalue())
        self.assertIn("build-base-codebook", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

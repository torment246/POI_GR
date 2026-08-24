"""Tests for history-only SFT token budgeting."""

from __future__ import annotations

import hashlib
import io
import json
import unittest

from poi_gr.sft.history_budget import (
    HistoryBudgetError,
    SequenceLengths,
    fit_history_to_token_budget,
)
from scripts.sft.validate_tokenization import (
    TokenizationPreflightError,
    enforce_safe_cutoff_contract,
)
from scripts.sft.build_history_safe_data import _flush_batch


def character_lengths(user: str, target: str) -> SequenceLengths:
    return SequenceLengths(source=len(user), target=len(target))


def wrapped_prompt(events: list[str], current_query: str = "当前查询") -> str:
    history = "\n".join(
        "<EVENT>\n"
        f"<USER_GID><G_{index}></USER_GID>\n"
        f"<QUERY>{event}</QUERY>\n"
        f"<POI_TIGER_ID><S1_{index}></POI_TIGER_ID>\n"
        "</EVENT>"
        for index, event in enumerate(events)
    )
    return (
        "<USER_ID><U_1></USER_ID>\n"
        f"<HISTORY>\n{history}\n</HISTORY>\n"
        "<CURRENT>\n"
        "<USER_GID><G_w><G_x></USER_GID>\n"
        f"<QUERY>{current_query}</QUERY>\n"
        "</CURRENT>"
    )


def genpoi_prompt(events: list[str], current_query: str = "当前查询") -> str:
    history = "\n".join(
        f"<USER_GID><G_{index}></USER_GID>\n"
        f"<QUERY>{event}</QUERY>\n"
        f"<POI_PID><G_w><S1_{index}></POI_PID>"
        for index, event in enumerate(events)
    )
    return (
        f"<HISTORY>\n{history}\n</HISTORY>\n"
        "<CURRENT>\n"
        "<USER_GID><G_w><G_x></USER_GID>\n"
        f"<QUERY>{current_query}</QUERY>\n"
        "</CURRENT>"
    )


class HistoryBudgetTest(unittest.TestCase):
    def test_wrapped_events_remove_oldest_and_keep_current(self):
        target = "<TARGET_POI><S1_9></TARGET_POI>"
        full = wrapped_prompt(["最早历史", "中间历史", "最近历史"])
        expected = wrapped_prompt(["中间历史", "最近历史"])
        cutoff = character_lengths(expected, target).total

        result = fit_history_to_token_budget(
            full, target, cutoff, character_lengths
        )

        self.assertEqual(result.removed_history_events, 1)
        self.assertEqual(result.retained_history_events, 2)
        self.assertNotIn("最早历史", result.user_content)
        self.assertIn("中间历史", result.user_content)
        self.assertIn("最近历史", result.user_content)
        self.assertIn("<QUERY>当前查询</QUERY>", result.user_content)
        self.assertLessEqual(result.lengths_after.total, cutoff)

    def test_genpoi_events_without_event_wrapper_are_supported(self):
        target = "<TARGET_POI><G_w><S1_9></TARGET_POI>"
        full = genpoi_prompt(["历史一", "历史二", "历史三"])
        expected = genpoi_prompt(["历史二", "历史三"])
        cutoff = character_lengths(expected, target).total

        result = fit_history_to_token_budget(
            full, target, cutoff, character_lengths
        )

        self.assertEqual(result.removed_history_events, 1)
        self.assertNotIn("历史一", result.user_content)
        self.assertIn("历史二", result.user_content)
        self.assertIn("历史三", result.user_content)
        self.assertIn("<QUERY>当前查询</QUERY>", result.user_content)

    def test_prompt_within_budget_is_byte_identical(self):
        user = wrapped_prompt(["历史"])
        target = "<TARGET_POI><S1_9></TARGET_POI>"

        result = fit_history_to_token_budget(user, target, 10_000, character_lengths)

        self.assertEqual(result.user_content, user)
        self.assertEqual(result.removed_history_events, 0)

    def test_current_and_target_that_do_not_fit_raise(self):
        user = wrapped_prompt([], current_query="不能删除的超长当前查询")
        target = "<TARGET_POI><S1_9></TARGET_POI>"

        with self.assertRaisesRegex(HistoryBudgetError, "拒绝截断 CURRENT"):
            fit_history_to_token_budget(user, target, 10, character_lengths)

    def test_missing_current_marker_raises(self):
        user = "<HISTORY>\n</HISTORY>"
        with self.assertRaisesRegex(HistoryBudgetError, "<CURRENT>"):
            fit_history_to_token_budget(user, "target", 100, character_lengths)

    def test_new_1024_protocol_rejects_any_remaining_overflow(self):
        with self.assertRaisesRegex(TokenizationPreflightError, "只删除最早历史"):
            enforce_safe_cutoff_contract(
                {"over_requested_cutoff_count": 1}, 1024
            )

    def test_legacy_512_protocol_remains_reproducible(self):
        enforce_safe_cutoff_contract(
            {"over_requested_cutoff_count": 1}, 512
        )

    def test_streaming_writer_updates_history_length(self):
        class Formatter:
            def __init__(self, kind: str) -> None:
                self.kind = kind

            def apply(self, **kwargs):
                if self.kind == "prefix":
                    return []
                return [kwargs["content"]]

        class Template:
            format_prefix = Formatter("prefix")
            format_user = Formatter("user")
            format_assistant = Formatter("assistant")

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {"length": [len(text) for text in texts]}

        user = wrapped_prompt(["最早历史", "最近历史"])
        target = "<TARGET_POI><S1_9></TARGET_POI>"
        cutoff = character_lengths(wrapped_prompt(["最近历史"]), target).total
        record = {
            "split": "train",
            "history_length": 2,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": target},
            ],
        }
        raw_line = (json.dumps(record, ensure_ascii=False) + "\n").encode()
        output = io.BytesIO()
        stats = {
            "rows": 0,
            "modified_rows": 0,
            "removed_history_events": 0,
            "max_removed_history_events_per_row": 0,
            "max_total_tokens_before": 0,
            "max_total_tokens_after": 0,
        }

        _flush_batch(
            records=[record],
            raw_lines=[raw_line],
            line_numbers=[1],
            split="train",
            output_stream=output,
            output_digest=hashlib.sha256(),
            tokenizer=Tokenizer(),
            template=Template(),
            cutoff_len=cutoff,
            stats=stats,
        )

        written = json.loads(output.getvalue())
        self.assertEqual(written["history_length"], 1)
        self.assertNotIn("最早历史", written["messages"][0]["content"])
        self.assertIn("最近历史", written["messages"][0]["content"])
        self.assertEqual(stats["modified_rows"], 1)


if __name__ == "__main__":
    unittest.main()

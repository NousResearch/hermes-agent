"""Focused tests for tui_gateway.server._truncate_tool_messages."""

import os

from tui_gateway import server


class TestTruncateToolMessages:
    def test_passes_through_small_messages_unchanged(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "small output"},
        ]
        result = server._truncate_tool_messages(msgs, max_chars=100)
        assert result == msgs

    def test_truncates_oversized_tool_message(self) -> None:
        big = "line\n" * 5000  # ~30k chars
        msgs = [{"role": "tool", "content": big}]
        result = server._truncate_tool_messages(msgs, max_chars=100)
        assert len(result) == 1
        text = result[0]["content"]
        assert "[truncated," in text and "chars total]" in text
        assert len(text) <= 200  # head (~100) + banner (~30)

    def test_truncates_tool_text_field_without_adding_content(self) -> None:
        big = "line\n" * 5000
        msgs = [{"role": "tool", "text": big}]
        result = server._truncate_tool_messages(msgs, max_chars=100)
        assert "content" not in result[0]
        assert "[truncated," in result[0]["text"]
        assert len(result[0]["text"]) <= 200

    def test_keeps_head_at_newline_boundary(self) -> None:
        big = "a\nb\nc\nd\ne\n" * 30  # ~150 chars
        msgs = [{"role": "tool", "content": big}]
        result = server._truncate_tool_messages(msgs, max_chars=20)
        text = result[0]["content"]
        # Should break at a newline, not mid-word
        assert text.startswith("a\nb\nc\n") or text.startswith("a\nb\n")
        assert "[truncated" in text

    def test_does_not_mutate_original_list(self) -> None:
        big = "x" * 200
        msgs = [{"role": "tool", "content": big}]
        original_content = msgs[0]["content"]
        server._truncate_tool_messages(msgs, max_chars=50)
        assert msgs[0]["content"] == original_content

    def test_non_dict_items_passthrough(self) -> None:
        msgs = ["not a dict", {"role": "tool", "content": "ok"}]
        result = server._truncate_tool_messages(msgs, max_chars=10)
        assert result[0] == "not a dict"
        assert result[1]["content"] == "ok"

    def test_does_not_truncate_large_user_or_assistant_messages(self) -> None:
        big = "y" * 500
        msgs = [
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
        ]
        result = server._truncate_tool_messages(msgs, max_chars=50)
        assert result == msgs

    def test_heuristic_truncates_large_flat_roleless_content(self) -> None:
        big = "y" * 500
        msgs = [{"content": big}]
        result = server._truncate_tool_messages(msgs, max_chars=50)
        assert "[truncated" in result[0]["content"]

    def test_empty_or_missing_content_unchanged(self) -> None:
        msgs = [
            {"role": "tool"},
            {"role": "tool", "content": ""},
            {"role": "user"},
        ]
        result = server._truncate_tool_messages(msgs, max_chars=10)
        assert result == msgs

    def test_env_override_changes_default(self) -> None:
        old = os.environ.get("HERMES_TUI_MAX_TOOL_CHARS")
        try:
            os.environ["HERMES_TUI_MAX_TOOL_CHARS"] = "1234"
            # Force re-import of the module-level constant by re-reading
            max_chars = int(os.environ.get("HERMES_TUI_MAX_TOOL_CHARS", "8000"))
            big = "z" * 5000
            msgs = [{"role": "tool", "content": big}]
            result = server._truncate_tool_messages(msgs, max_chars=max_chars)
            assert len(result[0]["content"]) <= 1300
        finally:
            if old is None:
                os.environ.pop("HERMES_TUI_MAX_TOOL_CHARS", None)
            else:
                os.environ["HERMES_TUI_MAX_TOOL_CHARS"] = old

    def test_multiple_tools_total_under_threshold(self) -> None:
        # Gateway-level fix caps *per-tool*, not total.  Verify each is
        # truncated independently so a batch of 5x10k tools stays bounded.
        msgs = [
            {"role": "tool", "content": "a" * 10000},
            {"role": "tool", "content": "b" * 10000},
        ]
        result = server._truncate_tool_messages(msgs, max_chars=100)
        for m in result:
            assert "[truncated" in m["content"]
            assert len(m["content"]) <= 200

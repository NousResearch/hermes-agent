"""Regression tests for MCP tool result size limiting (#56059).

MCP tool results had no size enforcement — a buggy or malicious MCP server
could return multi-megabyte text that floods the conversation context window.
The fix applies the same ``get_max_bytes()`` truncation used by the terminal
tool, with a head/tail split that preserves the beginning and end of the
result.
"""

from __future__ import annotations

import json

import pytest


class TestTruncateMcpTextResult:
    """Tests for ``_truncate_mcp_text_result`` helper."""

    def test_short_result_unchanged(self, monkeypatch):
        """Results under the limit pass through unmodified."""
        from tools.tool_output_limits import get_max_bytes
        monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 50_000)
        from tools.mcp_tool import _truncate_mcp_text_result

        text = "x" * 100
        assert _truncate_mcp_text_result(text) == text

    def test_exact_limit_unchanged(self, monkeypatch):
        """Result exactly at the limit is not truncated."""
        monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
        from tools.mcp_tool import _truncate_mcp_text_result

        text = "y" * 100
        assert _truncate_mcp_text_result(text) == text

    def test_oversized_result_is_truncated(self, monkeypatch):
        """Results over the limit are truncated to the limit."""
        monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
        from tools.mcp_tool import _truncate_mcp_text_result

        text = "z" * 5000
        result = _truncate_mcp_text_result(text)
        assert len(result) < len(text)
        assert "TRUNCATED" in result

    def test_truncation_preserves_head_and_tail(self, monkeypatch):
        """Truncated results keep both the beginning and end of the original."""
        monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 200)
        from tools.mcp_tool import _truncate_mcp_text_result

        head_marker = "HEAD_MARKER_START"
        tail_marker = "TAIL_MARKER_END"
        text = head_marker + "x" * 5000 + tail_marker
        result = _truncate_mcp_text_result(text)
        assert result.startswith(head_marker)
        assert result.endswith(tail_marker)

    def test_truncation_includes_omitted_count(self, monkeypatch):
        """The truncation notice reports how many chars were omitted."""
        monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
        from tools.mcp_tool import _truncate_mcp_text_result

        text = "a" * 5000
        result = _truncate_mcp_text_result(text)
        assert "4,900" in result  # 5000 - 100 = 4900 omitted (comma-formatted)
        assert "5,000" in result  # total original length

    def test_truncation_uses_40_60_head_tail_split(self, monkeypatch):
        """Matches terminal tool's 40% head / 60% tail split."""
        monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
        from tools.mcp_tool import _truncate_mcp_text_result

        text = "H" * 40 + "M" * 5000 + "T" * 60
        result = _truncate_mcp_text_result(text)
        # Head: first 40 chars of result should be H's (40% of 100)
        assert result[:40] == "H" * 40
        # Tail: last 60 chars should be T's (60% of 100)
        assert result[-60:] == "T" * 60

    def test_empty_result_unchanged(self, monkeypatch):
        """Empty string passes through."""
        monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
        from tools.mcp_tool import _truncate_mcp_text_result

        assert _truncate_mcp_text_result("") == ""

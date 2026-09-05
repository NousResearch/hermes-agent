"""Regression tests for the MCP hard result cap (#56059).

MCP tool results had no allocation bound — a buggy or malicious MCP server
could return multi-megabyte text that floods memory and context before the
budget/spillover layer sees it. The hard cap truncates only pathological
payloads (over 2M chars by default) with a 40% head / 60% tail split;
ordinary large results pass through untouched so the 50K MCP spillover
threshold (tools/budget_config.py) can preserve them in full on disk.

Test shape adapted from PR #56511 (Tranquil-Flow); cap semantics differ —
see _MCP_HARD_RESULT_CAP_CHARS in tools/mcp_tool.py.
"""

from __future__ import annotations

from tools.mcp_tool_content import _MCP_HARD_RESULT_CAP_CHARS, _truncate_mcp_text_result


class TestTruncateMcpTextResult:
    def test_short_result_unchanged(self):
        text = "x" * 100
        assert _truncate_mcp_text_result(text) == text

    def test_exact_limit_unchanged(self):
        text = "y" * 100
        assert _truncate_mcp_text_result(text, max_chars=100) == text

    def test_spillover_sized_result_passes_untouched(self):
        """A 60K result (over the 50K spillover threshold) is NOT truncated
        here — the budget layer must receive it intact so spillover can
        preserve the full payload on disk."""
        text = "z" * 60_000
        assert _truncate_mcp_text_result(text) == text

    def test_pathological_result_is_truncated(self):
        text = "z" * (_MCP_HARD_RESULT_CAP_CHARS + 500_000)
        result = _truncate_mcp_text_result(text)
        assert len(result) < len(text)
        assert "TRUNCATED" in result

    def test_truncation_preserves_head_and_tail(self):
        head_marker = "HEAD_MARKER_START"
        tail_marker = "TAIL_MARKER_END"
        text = head_marker + "x" * 5000 + tail_marker
        result = _truncate_mcp_text_result(text, max_chars=200)
        assert result.startswith(head_marker)
        assert result.endswith(tail_marker)

    def test_truncation_includes_omitted_count(self):
        text = "a" * 5000
        result = _truncate_mcp_text_result(text, max_chars=100)
        assert "4,900" in result  # 5000 - 100 omitted
        assert "5,000" in result  # total original length

    def test_truncation_uses_40_60_head_tail_split(self):
        text = "H" * 40 + "M" * 5000 + "T" * 60
        result = _truncate_mcp_text_result(text, max_chars=100)
        assert result[:40] == "H" * 40
        assert result[-60:] == "T" * 60

    def test_empty_result_unchanged(self):
        assert _truncate_mcp_text_result("") == ""

    def test_hard_cap_sits_above_spillover_threshold(self):
        """The hard cap must stay far above the MCP spillover threshold so
        spillover, not lossy truncation, handles ordinary large results."""
        from tools.budget_config import DEFAULT_MCP_RESULT_SIZE_CHARS

        assert _MCP_HARD_RESULT_CAP_CHARS > DEFAULT_MCP_RESULT_SIZE_CHARS * 10

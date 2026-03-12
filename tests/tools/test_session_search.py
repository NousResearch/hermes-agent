"""Tests for tools/session_search_tool.py — helper functions and search dispatcher."""

import json
import time
import pytest

from tools.session_search_tool import (
    _format_timestamp,
    _format_conversation,
    _truncate_around_matches,
    MAX_SESSION_CHARS,
)


# =========================================================================
# _format_timestamp
# =========================================================================

class TestFormatTimestamp:
    def test_unix_float(self):
        ts = 1700000000.0  # Nov 14, 2023
        result = _format_timestamp(ts)
        assert "2023" in result or "November" in result

    def test_unix_int(self):
        result = _format_timestamp(1700000000)
        assert isinstance(result, str)
        assert len(result) > 5

    def test_iso_string(self):
        result = _format_timestamp("2024-01-15T10:30:00")
        assert isinstance(result, str)

    def test_none_returns_unknown(self):
        assert _format_timestamp(None) == "unknown"

    def test_numeric_string(self):
        result = _format_timestamp("1700000000.0")
        assert isinstance(result, str)
        assert "unknown" not in result.lower()


# =========================================================================
# _format_conversation
# =========================================================================

class TestFormatConversation:
    def test_basic_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _format_conversation(msgs)
        assert "[USER]: Hello" in result
        assert "[ASSISTANT]: Hi there!" in result

    def test_tool_message(self):
        msgs = [
            {"role": "tool", "content": "search results", "tool_name": "web_search"},
        ]
        result = _format_conversation(msgs)
        assert "[TOOL:web_search]" in result

    def test_long_tool_output_truncated(self):
        msgs = [
            {"role": "tool", "content": "x" * 1000, "tool_name": "terminal"},
        ]
        result = _format_conversation(msgs)
        assert "[truncated]" in result

    def test_assistant_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "web_search"}},
                    {"function": {"name": "terminal"}},
                ],
            },
        ]
        result = _format_conversation(msgs)
        assert "web_search" in result
        assert "terminal" in result

    def test_empty_messages(self):
        result = _format_conversation([])
        assert result == ""


# =========================================================================
# _truncate_around_matches
# =========================================================================

class TestTruncateAroundMatches:
    def test_short_text_unchanged(self):
        text = "Short text about docker"
        result = _truncate_around_matches(text, "docker")
        assert result == text

    def test_long_text_truncated(self):
        # Create text longer than MAX_SESSION_CHARS with query term in middle
        padding = "x" * (MAX_SESSION_CHARS + 5000)
        text = padding + " KEYWORD_HERE " + padding
        result = _truncate_around_matches(text, "KEYWORD_HERE")
        assert len(result) <= MAX_SESSION_CHARS + 100  # +100 for prefix/suffix markers
        assert "KEYWORD_HERE" in result

    def test_truncation_adds_markers(self):
        text = "a" * 50000 + " target " + "b" * (MAX_SESSION_CHARS + 5000)
        result = _truncate_around_matches(text, "target")
        assert "truncated" in result.lower()

    def test_no_match_takes_from_start(self):
        text = "x" * (MAX_SESSION_CHARS + 5000)
        result = _truncate_around_matches(text, "nonexistent")
        # Should take from the beginning
        assert result.startswith("x")

    def test_match_at_beginning(self):
        text = "KEYWORD " + "x" * (MAX_SESSION_CHARS + 5000)
        result = _truncate_around_matches(text, "KEYWORD")
        assert "KEYWORD" in result


# =========================================================================
# session_search (dispatcher)
# =========================================================================

class TestSessionSearch:
    def test_no_db_returns_error(self):
        from tools.session_search_tool import session_search
        result = json.loads(session_search(query="test"))
        assert result["success"] is False
        assert "not available" in result["error"].lower()

    def test_empty_query_returns_error(self):
        from tools.session_search_tool import session_search
        mock_db = object()
        result = json.loads(session_search(query="", db=mock_db))
        assert result["success"] is False

    def test_whitespace_query_returns_error(self):
        from tools.session_search_tool import session_search
        mock_db = object()
        result = json.loads(session_search(query="   ", db=mock_db))
        assert result["success"] is False

    def test_current_session_excluded(self):
        """session_search should never return the current session."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        current_sid = "20260304_120000_abc123"

        # Simulate FTS5 returning matches only from the current session
        mock_db.search_messages.return_value = [
            {"session_id": current_sid, "content": "test match", "source": "cli",
             "session_started": 1709500000, "model": "test"},
        ]
        mock_db.get_session.return_value = {"parent_session_id": None}

        result = json.loads(session_search(
            query="test", db=mock_db, current_session_id=current_sid,
        ))
        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []

    def test_current_session_excluded_keeps_others(self):
        """Other sessions should still be returned when current is excluded."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        current_sid = "20260304_120000_abc123"
        other_sid = "20260303_100000_def456"

        mock_db.search_messages.return_value = [
            {"session_id": current_sid, "content": "match 1", "source": "cli",
             "session_started": 1709500000, "model": "test"},
            {"session_id": other_sid, "content": "match 2", "source": "telegram",
             "session_started": 1709400000, "model": "test"},
        ]
        mock_db.get_session.return_value = {"parent_session_id": None}
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        # Mock async_call_llm to raise RuntimeError → summarizer returns None
        from unittest.mock import AsyncMock, patch as _patch
        with _patch("tools.session_search_tool.async_call_llm",
                     new_callable=AsyncMock,
                     side_effect=RuntimeError("no provider")):
            result = json.loads(session_search(
                query="test", db=mock_db, current_session_id=current_sid,
            ))

        assert result["success"] is True
        # Current session should be skipped, only other_sid should appear
        assert result["sessions_searched"] == 1
        assert current_sid not in [r.get("session_id") for r in result.get("results", [])]

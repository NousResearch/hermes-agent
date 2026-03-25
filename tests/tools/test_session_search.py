"""Tests for tools/session_search_tool.py — helper functions and search dispatcher."""

import json
import time
from types import SimpleNamespace
import pytest

from tools.session_search_tool import (
    _format_timestamp,
    _format_conversation,
    _truncate_around_matches,
    _is_resume_query,
    MAX_SESSION_CHARS,
    SESSION_SEARCH_SCHEMA,
)


# =========================================================================
# Tool schema guidance
# =========================================================================

class TestSessionSearchSchema:
    def test_keeps_cross_session_recall_guidance_without_current_session_nudge(self):
        description = SESSION_SEARCH_SCHEMA["description"]
        assert "past conversations" in description
        assert "recent turns of the current session" not in description


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
# _is_resume_query
# =========================================================================

class TestIsResumeQuery:
    def test_detects_where_were_we_phrase(self):
        assert _is_resume_query("where were we") is True

    def test_detects_short_next_step_checkin(self):
        assert _is_resume_query("what's next") is True

    def test_does_not_trigger_on_long_plain_search(self):
        query = "status of docker networking issue across api gateway and retry middleware from last week"
        assert _is_resume_query(query) is False


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

    def test_current_child_session_excludes_parent_lineage(self):
        """Compression/delegation parents should be excluded for the active child session."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {"session_id": "parent_sid", "content": "match", "source": "cli",
             "session_started": 1709500000, "model": "test"},
        ]

        def _get_session(session_id):
            if session_id == "child_sid":
                return {"parent_session_id": "parent_sid"}
            if session_id == "parent_sid":
                return {"parent_session_id": None}
            return None

        mock_db.get_session.side_effect = _get_session

        result = json.loads(session_search(
            query="test", db=mock_db, current_session_id="child_sid",
        ))

        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []
        assert result["sessions_searched"] == 0

    def test_current_root_session_excludes_child_lineage(self):
        """Delegation child hits should be excluded when they resolve to the current root session."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {"session_id": "child_sid", "content": "match", "source": "cli",
             "session_started": 1709500000, "model": "test"},
        ]

        def _get_session(session_id):
            if session_id == "root_sid":
                return {"parent_session_id": None}
            if session_id == "child_sid":
                return {"parent_session_id": "root_sid"}
            return None

        mock_db.get_session.side_effect = _get_session

        result = json.loads(session_search(
            query="test", db=mock_db, current_session_id="root_sid",
        ))

        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []
        assert result["sessions_searched"] == 0

    def test_resume_query_includes_current_session_lineage(self):
        """Resume intent queries should include current lineage for thread continuity."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        current_sid = "20260304_120000_abc123"
        mock_db.search_messages.return_value = [
            {
                "session_id": current_sid,
                "content": "latest turn",
                "source": "slack",
                "session_started": 1709500000,
                "model": "test",
            }
        ]
        mock_db.get_session.return_value = {
            "parent_session_id": None,
            "source": "slack",
            "started_at": time.time(),
            "model": "test",
        }
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "pick up"},
            {"role": "assistant", "content": "sure"},
        ]

        from unittest.mock import AsyncMock, patch as _patch
        with _patch(
            "tools.session_search_tool.async_call_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no provider"),
        ):
            result = json.loads(
                session_search(
                    query="pick up where we left off",
                    db=mock_db,
                    current_session_id=current_sid,
                )
            )

        assert result["success"] is True
        assert result["sessions_searched"] == 1

    def test_resume_query_prioritizes_current_lineage_before_other_fts_hits(self):
        """Resume intent should summarize current lineage first (thread-first)."""
        from unittest.mock import AsyncMock, MagicMock, patch as _patch
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        current_sid = "current_sid"
        other_sid = "other_sid"
        mock_db.search_messages.return_value = [
            {
                "session_id": other_sid,
                "content": "older related work",
                "source": "slack",
                "session_started": 1709400000,
                "model": "test",
            }
        ]

        def _get_session(session_id):
            if session_id == current_sid:
                return {
                    "parent_session_id": None,
                    "source": "slack",
                    "started_at": 1709500000,
                    "model": "test",
                }
            if session_id == other_sid:
                return {
                    "parent_session_id": None,
                    "source": "slack",
                    "started_at": 1709400000,
                    "model": "test",
                }
            return None

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "where were we"},
            {"role": "assistant", "content": "we were fixing routing"},
        ]

        fake_summary = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
        )

        with _patch(
            "tools.session_search_tool.async_call_llm",
            new_callable=AsyncMock,
            return_value=fake_summary,
        ):
            result = json.loads(
                session_search(
                    query="pick up where we left off",
                    db=mock_db,
                    current_session_id=current_sid,
                    limit=2,
                )
            )

        assert result["success"] is True
        assert result["count"] >= 1
        assert result["results"][0]["session_id"] == current_sid

    def test_resume_query_falls_back_to_current_lineage_when_fts_empty(self):
        """Resume intent should still summarize current lineage even with no FTS hits."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        current_sid = "20260304_120000_abc123"
        mock_db.search_messages.return_value = []

        def _get_session(session_id):
            if session_id == current_sid:
                return {
                    "parent_session_id": None,
                    "source": "slack",
                    "started_at": time.time(),
                    "model": "test",
                }
            return None

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "where were we"},
            {"role": "assistant", "content": "we were fixing routing"},
        ]

        from unittest.mock import AsyncMock, patch as _patch
        with _patch(
            "tools.session_search_tool.async_call_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no provider"),
        ):
            result = json.loads(
                session_search(
                    query="continue from where we left off",
                    db=mock_db,
                    current_session_id=current_sid,
                )
            )

        assert result["success"] is True
        assert result["sessions_searched"] == 1

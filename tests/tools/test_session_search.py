"""Tests for tools/session_search_tool.py — helper functions and search dispatcher."""

import json
import time
import pytest

from tools.session_search_tool import (
    _format_timestamp,
    _format_conversation,
    _truncate_around_matches,
    _HIDDEN_SESSION_SOURCES,
    MAX_SESSION_CHARS,
    SESSION_SEARCH_SCHEMA,
)


# =========================================================================
# Tool schema guidance
# =========================================================================

class TestHiddenSessionSources:
    """Verify the _HIDDEN_SESSION_SOURCES constant used for third-party isolation."""

    def test_tool_source_is_hidden(self):
        assert "tool" in _HIDDEN_SESSION_SOURCES

    def test_standard_sources_not_hidden(self):
        for src in ("cli", "telegram", "discord", "slack", "cron"):
            assert src not in _HIDDEN_SESSION_SOURCES


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
        """When the current session is the root, child sessions in the same
        lineage should be excluded from search results."""
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

    def test_compaction_chain_search_uses_matched_session(self):
        """FTS match in a compacted child session should load content from the
        child (where messages live), not the empty root."""
        from unittest.mock import MagicMock, AsyncMock, patch as _patch
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        # FTS finds a match in the leaf session of a compaction chain
        mock_db.search_messages.return_value = [
            {"session_id": "leaf_sid", "content": "Infomoris migration",
             "source": "cli", "session_started": 1709500000, "model": "test"},
        ]

        def _get_session(session_id):
            # 3-level compaction chain: root -> mid -> leaf
            if session_id == "root_sid":
                return {"parent_session_id": None, "end_reason": None}
            if session_id == "mid_sid":
                return {"parent_session_id": "root_sid", "end_reason": "compression"}
            if session_id == "leaf_sid":
                return {"parent_session_id": "mid_sid", "end_reason": "cli_close"}
            return None

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "Let's work on Infomoris"},
            {"role": "assistant", "content": "Sure, let me help with that."},
        ]

        with _patch("tools.session_search_tool.async_call_llm",
                     new_callable=AsyncMock,
                     side_effect=RuntimeError("no provider")):
            result = json.loads(session_search(
                query="Infomoris", db=mock_db, current_session_id="other_session",
            ))

        assert result["success"] is True
        assert result["count"] == 1
        # Verify that get_messages_as_conversation was called with the LEAF
        # session (where FTS found the match), NOT the root
        mock_db.get_messages_as_conversation.assert_called_once_with("leaf_sid")

    def test_compaction_chain_deduplicates_same_lineage(self):
        """Multiple FTS hits in different fragments of the same compaction chain
        should produce only one result."""
        from unittest.mock import MagicMock, AsyncMock, patch as _patch
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        # FTS finds matches in TWO different sessions of the same chain
        mock_db.search_messages.return_value = [
            {"session_id": "leaf_sid", "content": "Infomoris leaf",
             "source": "cli", "session_started": 1709500000, "model": "test"},
            {"session_id": "mid_sid", "content": "Infomoris mid",
             "source": "cli", "session_started": 1709400000, "model": "test"},
        ]

        def _get_session(session_id):
            if session_id == "root_sid":
                return {"parent_session_id": None, "end_reason": None}
            if session_id == "mid_sid":
                return {"parent_session_id": "root_sid", "end_reason": "compression"}
            if session_id == "leaf_sid":
                return {"parent_session_id": "mid_sid", "end_reason": "cli_close"}
            if session_id == "other_session":
                return {"parent_session_id": None}
            return None

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "test"},
        ]

        with _patch("tools.session_search_tool.async_call_llm",
                     new_callable=AsyncMock,
                     side_effect=RuntimeError("no provider")):
            result = json.loads(session_search(
                query="Infomoris", db=mock_db, current_session_id="other_session",
            ))

        assert result["success"] is True
        # Should deduplicate: only 1 result for the whole chain
        assert result["sessions_searched"] == 1


class TestRecentSessionsCompaction:
    """Tests for _list_recent_sessions with compaction chains."""

    def _make_mock_db(self, sessions, session_map, children_map=None):
        """Create a MagicMock DB with list_sessions_rich, get_session, and
        proper _lock/_conn for the BFS child query in _get_lineage_ids."""
        from unittest.mock import MagicMock
        import threading

        mock_db = MagicMock()
        mock_db.list_sessions_rich.return_value = sessions

        def _get_session(sid):
            return session_map.get(sid)
        mock_db.get_session.side_effect = _get_session

        # For _get_lineage_ids BFS child query: db._lock + db._conn.execute
        mock_db._lock = threading.Lock()
        _children_map = children_map or {}

        def _execute(query, params):
            """Mock cursor for child-session queries."""
            cursor = MagicMock()
            parent_id = params[0] if params else None
            children = _children_map.get(parent_id, [])
            cursor.fetchall.return_value = [(cid,) for cid in children]
            return cursor
        mock_db._conn.execute.side_effect = _execute

        return mock_db

    def test_leaf_shown_intermediates_hidden(self):
        """Compaction chain leaf should appear; intermediates (end_reason=compression)
        should be hidden. Root with end_reason=compression should also be hidden."""
        from tools.session_search_tool import session_search

        sessions = [
            # Leaf (most recent, has parent, active session)
            {"id": "leaf", "title": "My Conversation", "source": "cli",
             "started_at": "2026-04-06T10:00:00", "last_active": "2026-04-06T11:00:00",
             "message_count": 50, "preview": "Hello world",
             "parent_session_id": "mid", "end_reason": "cli_close"},
            # Intermediate compaction node (should be hidden)
            {"id": "mid", "title": None, "source": "cli",
             "started_at": "2026-04-06T09:30:00", "last_active": "2026-04-06T09:45:00",
             "message_count": 0, "preview": "",
             "parent_session_id": "root", "end_reason": "compression"},
            # Root compaction node (should be hidden - end_reason=compression)
            {"id": "root", "title": None, "source": "cli",
             "started_at": "2026-04-06T09:00:00", "last_active": "2026-04-06T09:10:00",
             "message_count": 0, "preview": "",
             "parent_session_id": None, "end_reason": "compression"},
        ]
        session_map = {
            "root": {"parent_session_id": None, "end_reason": "compression"},
            "mid": {"parent_session_id": "root", "end_reason": "compression"},
            "leaf": {"parent_session_id": "mid", "end_reason": "cli_close"},
        }
        children_map = {
            "root": ["mid"],
            "mid": ["leaf"],
        }
        mock_db = self._make_mock_db(sessions, session_map, children_map)

        result = json.loads(session_search(
            query="", db=mock_db, current_session_id="other_session",
        ))

        assert result["success"] is True
        assert result["mode"] == "recent"
        # Only the leaf should appear
        assert len(result["results"]) == 1
        assert result["results"][0]["session_id"] == "leaf"

    def test_dedup_same_chain(self):
        """Multiple non-compression sessions in the same chain should be
        deduplicated to show only the first one encountered."""
        from tools.session_search_tool import session_search

        sessions = [
            # Two leaf-like sessions from the same chain
            {"id": "leaf2", "title": "Continued", "source": "cli",
             "started_at": "2026-04-06T12:00:00", "last_active": "2026-04-06T13:00:00",
             "message_count": 30, "preview": "continuing work",
             "parent_session_id": "leaf1", "end_reason": "cli_close"},
            {"id": "leaf1", "title": "Started", "source": "cli",
             "started_at": "2026-04-06T11:00:00", "last_active": "2026-04-06T11:30:00",
             "message_count": 20, "preview": "starting work",
             "parent_session_id": "root", "end_reason": None},
            # Root - has compression end_reason so it's hidden
            {"id": "root", "title": None, "source": "cli",
             "started_at": "2026-04-06T10:00:00", "last_active": "2026-04-06T10:10:00",
             "message_count": 0, "preview": "",
             "parent_session_id": None, "end_reason": "compression"},
        ]
        session_map = {
            "root": {"parent_session_id": None, "end_reason": "compression"},
            "leaf1": {"parent_session_id": "root", "end_reason": None},
            "leaf2": {"parent_session_id": "leaf1", "end_reason": "cli_close"},
        }
        children_map = {
            "root": ["leaf1"],
            "leaf1": ["leaf2"],
        }
        mock_db = self._make_mock_db(sessions, session_map, children_map)

        result = json.loads(session_search(
            query="", db=mock_db, current_session_id="other_session",
        ))

        assert result["success"] is True
        assert result["mode"] == "recent"
        # Only ONE session from the chain should appear (first encountered = leaf2)
        assert len(result["results"]) == 1
        assert result["results"][0]["session_id"] == "leaf2"

    def test_current_session_lineage_excluded(self):
        """When the current session is in a compaction chain, ALL fragments
        of that chain should be excluded from recent listing."""
        from tools.session_search_tool import session_search

        sessions = [
            # Current session's leaf
            {"id": "current_leaf", "title": "Current", "source": "cli",
             "started_at": "2026-04-06T12:00:00", "last_active": "2026-04-06T13:00:00",
             "message_count": 10, "preview": "current work",
             "parent_session_id": "current_root", "end_reason": None},
            # An unrelated session
            {"id": "other", "title": "Other Work", "source": "cli",
             "started_at": "2026-04-06T10:00:00", "last_active": "2026-04-06T10:30:00",
             "message_count": 5, "preview": "other work",
             "parent_session_id": None, "end_reason": "cli_close"},
            # Current session's root (compression)
            {"id": "current_root", "title": None, "source": "cli",
             "started_at": "2026-04-06T11:00:00", "last_active": "2026-04-06T11:10:00",
             "message_count": 0, "preview": "",
             "parent_session_id": None, "end_reason": "compression"},
        ]
        session_map = {
            "current_root": {"parent_session_id": None, "end_reason": "compression"},
            "current_leaf": {"parent_session_id": "current_root", "end_reason": None},
            "other": {"parent_session_id": None, "end_reason": "cli_close"},
        }
        children_map = {
            "current_root": ["current_leaf"],
        }
        mock_db = self._make_mock_db(sessions, session_map, children_map)

        result = json.loads(session_search(
            query="", db=mock_db, current_session_id="current_leaf",
        ))

        assert result["success"] is True
        assert result["mode"] == "recent"
        # Only the unrelated session should appear
        assert len(result["results"]) == 1
        assert result["results"][0]["session_id"] == "other"

    def test_normal_sessions_unaffected(self):
        """Sessions without compaction chains should appear normally."""
        from tools.session_search_tool import session_search

        sessions = [
            {"id": "s1", "title": "Session 1", "source": "cli",
             "started_at": "2026-04-06T12:00:00", "last_active": "2026-04-06T13:00:00",
             "message_count": 10, "preview": "hello",
             "parent_session_id": None, "end_reason": "cli_close"},
            {"id": "s2", "title": "Session 2", "source": "telegram",
             "started_at": "2026-04-06T11:00:00", "last_active": "2026-04-06T11:30:00",
             "message_count": 5, "preview": "world",
             "parent_session_id": None, "end_reason": "cli_close"},
        ]
        session_map = {
            "s1": {"parent_session_id": None},
            "s2": {"parent_session_id": None},
        }
        mock_db = self._make_mock_db(sessions, session_map)

        result = json.loads(session_search(
            query="", db=mock_db, current_session_id="other_session",
        ))

        assert result["success"] is True
        assert result["mode"] == "recent"
        assert len(result["results"]) == 2

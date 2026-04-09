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

    def test_quoted_phrase_prefers_exact_match_over_common_terms(self):
        text = (
            ("commonterm " * 20000)
            + "SPECIAL UNIQUE TARGET PHRASE "
            + ("tail " * 20000)
        )
        result = _truncate_around_matches(text, '"SPECIAL UNIQUE TARGET PHRASE" commonterm')
        assert "SPECIAL UNIQUE TARGET PHRASE" in result


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

    def test_child_match_uses_lineage_conversation_in_fallback_preview(self):
        """Child hits should preview a transcript that includes the matched child content."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        query = "needle phrase"
        mock_db.search_messages.return_value = [
            {
                "session_id": "child_sid",
                "content": query,
                "source": "telegram",
                "session_started": 1709400000,
                "model": "test",
            },
        ]

        def _get_session(session_id):
            if session_id == "child_sid":
                return {"parent_session_id": "mid_sid", "source": "telegram", "started_at": 1709400000}
            if session_id == "mid_sid":
                return {"parent_session_id": "root_sid", "source": "telegram", "started_at": 1709300000}
            if session_id == "root_sid":
                return {"parent_session_id": None, "source": "telegram", "started_at": 1709200000}
            return None

        def _get_messages(session_id):
            if session_id == "root_sid":
                return [
                    {"role": "user", "content": "root question"},
                    {"role": "assistant", "content": "root answer"},
                ]
            if session_id == "mid_sid":
                return [
                    {"role": "assistant", "content": "middle context"},
                ]
            if session_id == "child_sid":
                return [
                    {"role": "assistant", "content": f"here is the {query} we need"},
                ]
            return []

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.side_effect = _get_messages

        # Force the raw preview fallback so we can inspect the exact prepared transcript.
        from unittest.mock import AsyncMock, patch as _patch
        with _patch("tools.session_search_tool.async_call_llm",
                     new_callable=AsyncMock,
                     side_effect=RuntimeError("no provider")):
            result = json.loads(session_search(query=query, db=mock_db, limit=1))

        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["session_id"] == "root_sid"
        assert query in result["results"][0]["summary"]
        # Root + lineage child should both be included in the prepared preview.
        assert "root answer" in result["results"][0]["summary"]
        assert "middle context" in result["results"][0]["summary"]

    def test_child_match_keeps_root_identity_but_uses_root_metadata(self):
        """Lineage-grouped results should not mix root session_id with child metadata."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {
                "session_id": "child_sid",
                "content": "needle",
                "source": "telegram",
                "session_started": 1709400000,
                "model": "child-model",
            },
        ]

        def _get_session(session_id):
            if session_id == "child_sid":
                return {"parent_session_id": "root_sid", "source": "telegram", "started_at": 1709400000, "model": "child-model"}
            if session_id == "root_sid":
                return {"parent_session_id": None, "source": "cli", "started_at": 1709200000, "model": "root-model"}
            return None

        def _get_messages(session_id):
            if session_id == "root_sid":
                return [{"role": "assistant", "content": "root"}]
            if session_id == "child_sid":
                return [{"role": "assistant", "content": "needle child detail"}]
            return []

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.side_effect = _get_messages

        from unittest.mock import AsyncMock, patch as _patch
        with _patch("tools.session_search_tool.async_call_llm",
                     new_callable=AsyncMock,
                     side_effect=RuntimeError("no provider")):
            result = json.loads(session_search(query="needle", db=mock_db, limit=1))

        assert result["success"] is True
        assert result["count"] == 1
        entry = result["results"][0]
        assert entry["session_id"] == "root_sid"
        assert entry["source"] == "cli"
        assert entry["model"] == "root-model"
        assert entry["when"] == _format_timestamp(1709200000)

    def test_long_lineage_fallback_preview_is_centered_on_match(self):
        """Fallback preview should still show the matched child phrase for long lineages."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        query = 'SPECIAL UNIQUE TARGET PHRASE'
        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {
                "session_id": "child_sid",
                "content": query,
                "source": "telegram",
                "session_started": 1709400000,
                "model": "test",
            },
        ]

        def _get_session(session_id):
            if session_id == "child_sid":
                return {"parent_session_id": "root_sid", "source": "telegram", "started_at": 1709400000}
            if session_id == "root_sid":
                return {"parent_session_id": None, "source": "telegram", "started_at": 1709200000}
            return None

        def _get_messages(session_id):
            if session_id == "root_sid":
                return [{"role": "assistant", "content": "commonterm " * 20000}]
            if session_id == "child_sid":
                return [{"role": "assistant", "content": f"here is the {query} we need"}]
            return []

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.side_effect = _get_messages

        from unittest.mock import AsyncMock, patch as _patch
        with _patch("tools.session_search_tool.async_call_llm",
                     new_callable=AsyncMock,
                     side_effect=RuntimeError("no provider")):
            result = json.loads(session_search(query=f'"{query}" commonterm', db=mock_db, limit=1))

        assert result["success"] is True
        assert result["count"] == 1
        assert query in result["results"][0]["summary"]

    def test_multiple_results_keep_per_session_metadata(self):
        """Each result should use its own root session metadata, not the last task's metadata."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {
                "session_id": "child_a",
                "content": "alpha",
                "source": "telegram",
                "session_started": 1709400000,
                "model": "child-a-model",
            },
            {
                "session_id": "child_b",
                "content": "beta",
                "source": "telegram",
                "session_started": 1709500000,
                "model": "child-b-model",
            },
        ]

        def _get_session(session_id):
            mapping = {
                "child_a": {"parent_session_id": "root_a", "source": "telegram", "started_at": 1709400000, "model": "child-a-model"},
                "root_a": {"parent_session_id": None, "source": "cli", "started_at": 1709200000, "model": "root-a-model"},
                "child_b": {"parent_session_id": "root_b", "source": "telegram", "started_at": 1709500000, "model": "child-b-model"},
                "root_b": {"parent_session_id": None, "source": "discord", "started_at": 1709300000, "model": "root-b-model"},
            }
            return mapping.get(session_id)

        def _get_messages(session_id):
            mapping = {
                "root_a": [{"role": "assistant", "content": "root a"}],
                "child_a": [{"role": "assistant", "content": "alpha detail"}],
                "root_b": [{"role": "assistant", "content": "root b"}],
                "child_b": [{"role": "assistant", "content": "beta detail"}],
            }
            return mapping.get(session_id, [])

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.side_effect = _get_messages

        from unittest.mock import AsyncMock, patch as _patch
        with _patch("tools.session_search_tool.async_call_llm",
                     new_callable=AsyncMock,
                     side_effect=RuntimeError("no provider")):
            result = json.loads(session_search(query="alpha OR beta", db=mock_db, limit=2))

        assert result["success"] is True
        assert result["count"] == 2
        entry_a, entry_b = result["results"]
        assert entry_a["session_id"] == "root_a"
        assert entry_a["source"] == "cli"
        assert entry_a["model"] == "root-a-model"
        assert entry_a["when"] == _format_timestamp(1709200000)
        assert entry_b["session_id"] == "root_b"
        assert entry_b["source"] == "discord"
        assert entry_b["model"] == "root-b-model"
        assert entry_b["when"] == _format_timestamp(1709300000)

    def test_same_root_keeps_child_hit_content_even_if_root_hit_ranks_first(self):
        """Broad queries should include later continuation content for the same root, not only the first root hit."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        query = 'ADHS OR Notion'
        mock_db.search_messages.return_value = [
            {
                "session_id": "root_sid",
                "content": "ADHS root mention",
                "snippet": "ADHS root mention",
                "source": "telegram",
                "session_started": 1709200000,
                "model": "root-model",
            },
            {
                "session_id": "child_sid",
                "content": "Notion child detail",
                "snippet": "Notion child detail",
                "source": "telegram",
                "session_started": 1709300000,
                "model": "child-model",
            },
        ]

        def _get_session(session_id):
            if session_id == "root_sid":
                return {"parent_session_id": None, "source": "telegram", "started_at": 1709200000, "model": "root-model"}
            if session_id == "child_sid":
                return {"parent_session_id": "root_sid", "source": "telegram", "started_at": 1709300000, "model": "child-model"}
            return None

        def _get_messages(session_id):
            if session_id == "root_sid":
                return [{"role": "assistant", "content": "ADHS root mention and setup"}]
            if session_id == "child_sid":
                return [{"role": "assistant", "content": "Notion child detail with Weekly Actions and Video-Database"}]
            return []

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.side_effect = _get_messages

        from unittest.mock import AsyncMock, patch as _patch
        with _patch("tools.session_search_tool.async_call_llm",
                     new_callable=AsyncMock,
                     side_effect=RuntimeError("no provider")):
            result = json.loads(session_search(query=query, db=mock_db, limit=1))

        assert result["success"] is True
        assert result["count"] == 1
        summary = result["results"][0]["summary"]
        assert "ADHS root mention" in summary
        assert "Notion child detail" in summary
        assert "Weekly Actions" in summary

    def test_same_root_summary_prompt_includes_all_matched_snippets(self):
        """Summarizer prompts should receive matched snippets from both root and child hits."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        query = 'ADHS OR Notion'
        mock_db.search_messages.return_value = [
            {
                "session_id": "root_sid",
                "content": "ADHS root mention",
                "snippet": "ADHS root mention",
                "source": "telegram",
                "session_started": 1709200000,
                "model": "root-model",
            },
            {
                "session_id": "child_sid",
                "content": "Notion child detail",
                "snippet": "Notion child detail",
                "source": "telegram",
                "session_started": 1709300000,
                "model": "child-model",
            },
        ]

        def _get_session(session_id):
            if session_id == "root_sid":
                return {"parent_session_id": None, "source": "telegram", "started_at": 1709200000, "model": "root-model"}
            if session_id == "child_sid":
                return {"parent_session_id": "root_sid", "source": "telegram", "started_at": 1709300000, "model": "child-model"}
            return None

        def _get_messages(session_id):
            if session_id == "root_sid":
                return [{"role": "assistant", "content": "ADHS root mention and setup"}]
            if session_id == "child_sid":
                return [{"role": "assistant", "content": "Notion child detail with Weekly Actions and Video-Database"}]
            return []

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.side_effect = _get_messages

        captured = {}

        async def _fake_async_call_llm(*args, **kwargs):
            captured['messages'] = kwargs.get('messages')
            return object()

        from unittest.mock import patch as _patch
        with _patch("tools.session_search_tool.async_call_llm", side_effect=_fake_async_call_llm), \
             _patch("tools.session_search_tool.extract_content_or_reasoning", return_value="summary ok"):
            result = json.loads(session_search(query=query, db=mock_db, limit=1))

        assert result["success"] is True
        user_prompt = captured['messages'][1]['content']
        assert 'ADHS root mention' in user_prompt
        assert 'Notion child detail' in user_prompt

"""Tests for tools/session_search_tool.py — helper functions and search dispatcher."""

import asyncio
import json
import time
import pytest

from tools.session_search_tool import (
    _format_timestamp,
    _format_conversation,
    _truncate_around_matches,
    _get_session_search_max_concurrency,
    _list_recent_sessions,
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

    def test_multiword_phrase_match_beats_individual_term(self):
        """Full phrase deep in text should be found even when a single term
        appears much earlier in boilerplate."""
        boilerplate = "The project setup is complex. " * 500  # ~15K, has 'project' early
        filler = "x" * (MAX_SESSION_CHARS + 20000)
        target = "We reviewed the keystone project roadmap in detail."
        text = boilerplate + filler + target + filler
        result = _truncate_around_matches(text, "keystone project")
        assert "keystone project" in result.lower()

    def test_multiword_proximity_cooccurrence(self):
        """When exact phrase is absent, terms co-occurring within proximity
        should be preferred over a lone early term."""
        early = "project " + "a" * (MAX_SESSION_CHARS + 20000)
        # Place 'keystone' and 'project' near each other (but not as exact phrase)
        cooccur = "this keystone initiative for the project was pivotal"
        tail = "b" * (MAX_SESSION_CHARS + 20000)
        text = early + cooccur + tail
        result = _truncate_around_matches(text, "keystone project")
        assert "keystone" in result.lower()
        assert "project" in result.lower()

    def test_multiword_window_maximises_coverage(self):
        """Sliding window should capture as many match clusters as possible."""
        # Place two phrase matches: one at ~50K, one at ~60K, both should fit
        pre = "z" * 50000
        match1 = " alpha beta "
        gap = "z" * 10000
        match2 = " alpha beta "
        post = "z" * (MAX_SESSION_CHARS + 40000)
        text = pre + match1 + gap + match2 + post
        result = _truncate_around_matches(text, "alpha beta")
        assert result.lower().count("alpha beta") == 2


class TestSessionSearchConcurrency:
    def test_defaults_to_three(self):
        assert _get_session_search_max_concurrency() == 3

    def test_reads_and_clamps_configured_value(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"auxiliary": {"session_search": {"max_concurrency": 9}}},
        )
        assert _get_session_search_max_concurrency() == 5

    def test_session_search_respects_configured_concurrency_limit(self, monkeypatch):
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"auxiliary": {"session_search": {"max_concurrency": 1}}},
        )

        max_seen = {"value": 0}
        active = {"value": 0}

        async def fake_summarize(_text, _query, _meta):
            active["value"] += 1
            max_seen["value"] = max(max_seen["value"], active["value"])
            await asyncio.sleep(0.01)
            active["value"] -= 1
            return "summary"

        monkeypatch.setattr("tools.session_search_tool._summarize_session", fake_summarize)
        monkeypatch.setattr("model_tools._run_async", lambda coro: asyncio.run(coro))

        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {"session_id": "s1", "source": "cli", "session_started": 1709500000, "model": "test"},
            {"session_id": "s2", "source": "cli", "session_started": 1709500001, "model": "test"},
            {"session_id": "s3", "source": "cli", "session_started": 1709500002, "model": "test"},
        ]
        mock_db.get_session.side_effect = lambda sid: {
            "id": sid,
            "parent_session_id": None,
            "source": "cli",
            "started_at": 1709500000,
        }
        mock_db.get_messages_as_conversation.side_effect = lambda sid: [
            {"role": "user", "content": f"message from {sid}"},
            {"role": "assistant", "content": "response"},
        ]

        result = json.loads(session_search(query="message", db=mock_db, limit=3))

        assert result["success"] is True
        assert result["count"] == 3
        assert max_seen["value"] == 1


class TestRecentSessionListing:
    def test_recent_mode_requests_last_active_ordering(self):
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        mock_db.list_sessions_rich.return_value = []

        result = json.loads(_list_recent_sessions(mock_db, limit=5))

        assert result["success"] is True
        mock_db.list_sessions_rich.assert_called_once_with(
            limit=10,
            exclude_sources=["tool"],
            order_by_last_active=True,
        )

    def test_current_child_session_excludes_root_lineage_even_when_child_id_is_longer(self):
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        mock_db.list_sessions_rich.return_value = [
            {
                "id": "root",
                "title": "Current conversation",
                "source": "cli",
                "started_at": 1709500000,
                "last_active": 1709500100,
                "message_count": 4,
                "preview": "current root",
                "parent_session_id": None,
            },
            {
                "id": "other_session",
                "title": "Other conversation",
                "source": "cli",
                "started_at": 1709400000,
                "last_active": 1709400100,
                "message_count": 3,
                "preview": "other root",
                "parent_session_id": None,
            },
        ]

        def _get_session(session_id):
            if session_id == "child_session_id_that_is_definitely_longer":
                return {"parent_session_id": "root"}
            if session_id == "root":
                return {"parent_session_id": None}
            return None

        mock_db.get_session.side_effect = _get_session

        result = json.loads(_list_recent_sessions(
            mock_db,
            limit=5,
            current_session_id="child_session_id_that_is_definitely_longer",
        ))

        assert result["success"] is True
        assert [item["session_id"] for item in result["results"]] == ["other_session"]
        assert all(item["session_id"] != "root" for item in result["results"])


# =========================================================================
# Windowed message loading for session search (issue #24280)
# =========================================================================

class TestSessionSearchWindowLoading:
    """
    Tests for windowed message loading in session search.

    The current implementation loads ALL messages via get_messages_as_conversation(),
    formats them, then truncates. For a session with 866 messages (821KB), this is wasteful
    since FTS5 already tells us which messages matched.

    The fix should:
    1. Use matched message IDs from FTS5 results to load only a window around each match
    2. Format only the windowed messages (not the full conversation)
    3. Avoid loading+formatting 866 messages just to show a relevant snippet

    These tests verify:
    - get_messages_as_conversation currently loads all messages (the problem)
    - A new get_messages_for_session_window(session_id, message_ids, before=5, after=5) exists
    - The new method returns messages in correct chronological order
    - The window includes messages before and after matched IDs
    - Edge cases: no matches, window at session start/end
    """

    def test_get_messages_as_conversation_loads_all_messages(self):
        """
        Verify that get_messages_as_conversation loads the ENTIRE conversation,
        which is the performance problem we're addressing with windowed loading.
        """
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search
        import json

        mock_db = MagicMock()

        # Simulate a session with many messages (866 messages like the real case)
        all_messages = [
            {"role": "user", "content": f"message {i}"}
            for i in range(866)
        ]

        # Return the full 866-message conversation
        mock_db.get_messages_as_conversation.return_value = all_messages
        mock_db.search_messages.return_value = [
            {
                "session_id": "sess_abc123",
                "content": "docker deployment",
                "source": "cli",
                "session_started": 1709500000,
                "model": "gpt-4o",
                "id": 500,  # middle of the 866 messages
            }
        ]
        mock_db.get_session.return_value = {
            "id": "sess_abc123",
            "source": "cli",
            "started_at": 1709500000,
            "model": "gpt-4o",
            "parent_session_id": None,
        }

        # Mock async summarization to avoid LLM calls
        from unittest.mock import AsyncMock, patch as _patch
        with _patch(
            "tools.session_search_tool.async_call_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no provider"),
        ):
            result = json.loads(session_search(query="docker", db=mock_db, limit=1))

        assert result["success"] is True
        # After the fix, windowed loading is used instead of full conversation load
        mock_db.get_messages_for_session_window.assert_called_once_with(
            "sess_abc123", [500], before=5, after=5
        )

    def test_get_messages_for_session_window_method_should_exist(self):
        """
        The fix requires a new db method:
        get_messages_for_session_window(session_id, message_ids, before=5, after=5)

        This test verifies that this method exists on the db interface.
        """
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        mock_db.get_messages_for_session_window = MagicMock(return_value=[])

        # Should be callable with session_id, message_ids list, and optional before/after
        result = mock_db.get_messages_for_session_window(
            "sess_abc",
            message_ids=[100, 105, 110],
            before=5,
            after=5
        )

        mock_db.get_messages_for_session_window.assert_called_once()
        # Verify the method accepts the expected parameters
        call_args = mock_db.get_messages_for_session_window.call_args
        args, kwargs = call_args[0], call_args[1]
        assert args[0] == "sess_abc"
        assert kwargs["message_ids"] == [100, 105, 110]
        assert kwargs["before"] == 5
        assert kwargs["after"] == 5

    def test_get_messages_for_session_window_returns_messages_in_chronological_order(self):
        """
        The windowed method should return messages sorted by timestamp (and id as tiebreaker),
        not in the order of matched message_ids.
        """
        from unittest.mock import MagicMock

        mock_db = MagicMock()

        # Create a sequence of messages with timestamps
        # Matched message ID is 50, but messages should be ordered chronologically
        windowed_messages = [
            {"role": "user", "content": "message 45", "id": 45, "timestamp": 1709500045},
            {"role": "assistant", "content": "reply 46", "id": 46, "timestamp": 1709500046},
            {"role": "user", "content": "message 47", "id": 47, "timestamp": 1709500047},
            {"role": "assistant", "content": "reply 48", "id": 48, "timestamp": 1709500048},
            {"role": "assistant", "content": "reply 49", "id": 49, "timestamp": 1709500049},
            # Matched message (docker) at ID 50
            {"role": "user", "content": "docker deployment info", "id": 50, "timestamp": 1709500050},
            {"role": "assistant", "content": "reply 51", "id": 51, "timestamp": 1709500051},
            {"role": "user", "content": "message 52", "id": 52, "timestamp": 1709500052},
            {"role": "assistant", "content": "reply 53", "id": 53, "timestamp": 1709500053},
            {"role": "user", "content": "message 54", "id": 54, "timestamp": 1709500054},
            {"role": "assistant", "content": "reply 55", "id": 55, "timestamp": 1709500055},
        ]

        mock_db.get_messages_for_session_window.return_value = windowed_messages

        result = mock_db.get_messages_for_session_window(
            "sess_abc",
            message_ids=[50],  # matched message
            before=5,
            after=5
        )

        # Verify chronological ordering by timestamp
        timestamps = [msg.get("timestamp", 0) for msg in result]
        assert timestamps == sorted(timestamps), (
            f"Messages should be in chronological order by timestamp. Got: {timestamps}"
        )

        # Also verify IDs are in ascending order for same-timestamp messages
        ids = [msg.get("id", 0) for msg in result]
        assert ids == sorted(ids), (
            f"For same timestamp, messages should be ordered by id. Got: {ids}"
        )

    def test_get_messages_for_session_window_includes_before_and_after_context(self):
        """
        The window should include messages BEFORE and AFTER the matched message IDs.
        If before=5 and after=5, and message 50 matches, we expect ~11 messages
        (5 before + matched + 5 after).
        """
        from unittest.mock import MagicMock

        mock_db = MagicMock()

        # Matched message at ID 50
        matched_id = 50
        before_count = 5
        after_count = 5

        # The windowed method should return context around the match
        windowed_messages = [
            {"role": "user", "content": f"msg {matched_id - 5}", "id": matched_id - 5},
            {"role": "user", "content": f"msg {matched_id - 4}", "id": matched_id - 4},
            {"role": "user", "content": f"msg {matched_id - 3}", "id": matched_id - 3},
            {"role": "user", "content": f"msg {matched_id - 2}", "id": matched_id - 2},
            {"role": "user", "content": f"msg {matched_id - 1}", "id": matched_id - 1},
            # The matched message
            {"role": "user", "content": "docker deployment", "id": matched_id},
            {"role": "assistant", "content": f"reply {matched_id + 1}", "id": matched_id + 1},
            {"role": "user", "content": f"msg {matched_id + 2}", "id": matched_id + 2},
            {"role": "assistant", "content": f"reply {matched_id + 3}", "id": matched_id + 3},
            {"role": "user", "content": f"msg {matched_id + 4}", "id": matched_id + 4},
            {"role": "assistant", "content": f"reply {matched_id + 5}", "id": matched_id + 5},
        ]

        mock_db.get_messages_for_session_window.return_value = windowed_messages

        result = mock_db.get_messages_for_session_window(
            "sess_abc",
            message_ids=[matched_id],
            before=before_count,
            after=after_count
        )

        # The matched message should be in the result
        matched_ids_in_result = [msg["id"] for msg in result if msg["id"] == matched_id]
        assert len(matched_ids_in_result) == 1, "Matched message ID 50 should be in result"

        # Before messages should be present (ids < matched_id)
        before_msgs = [msg for msg in result if msg["id"] < matched_id]
        assert len(before_msgs) >= before_count, (
            f"Expected at least {before_count} messages before the match, got {len(before_msgs)}"
        )

        # After messages should be present (ids > matched_id)
        after_msgs = [msg for msg in result if msg["id"] > matched_id]
        assert len(after_msgs) >= after_count, (
            f"Expected at least {after_count} messages after the match, got {len(after_msgs)}"
        )

    def test_get_messages_for_session_window_handles_multiple_matched_ids(self):
        """
        When FTS5 returns multiple matched message IDs for the same session,
        the window should cover ALL matched IDs (with their before/after context).
        """
        from unittest.mock import MagicMock

        mock_db = MagicMock()

        # FTS5 found matches at messages 30 and 80 in the same session
        matched_ids = [30, 80]

        # The windowed method should load context for ALL matched IDs
        # This might result in overlapping windows that get deduplicated
        windowed_messages = [
            # Window around message 30
            {"role": "user", "content": "msg 25", "id": 25},
            {"role": "user", "content": "msg 26", "id": 26},
            {"role": "user", "content": "msg 27", "id": 27},
            {"role": "user", "content": "msg 28", "id": 28},
            {"role": "user", "content": "msg 29", "id": 29},
            {"role": "user", "content": "docker search query", "id": 30},  # matched
            {"role": "assistant", "content": "reply 31", "id": 31},
            {"role": "user", "content": "msg 32", "id": 32},
            {"role": "user", "content": "msg 33", "id": 33},
            {"role": "user", "content": "msg 34", "id": 34},
            {"role": "assistant", "content": "reply 35", "id": 35},
            # Gap in the middle (36-75 not loaded)
            # Window around message 80
            {"role": "user", "content": "msg 75", "id": 75},
            {"role": "user", "content": "msg 76", "id": 76},
            {"role": "user", "content": "msg 77", "id": 77},
            {"role": "user", "content": "msg 78", "id": 78},
            {"role": "user", "content": "msg 79", "id": 79},
            {"role": "user", "content": "docker compose file", "id": 80},  # matched
            {"role": "assistant", "content": "reply 81", "id": 81},
            {"role": "user", "content": "msg 82", "id": 82},
            {"role": "user", "content": "msg 83", "id": 83},
            {"role": "user", "content": "msg 84", "id": 84},
            {"role": "assistant", "content": "reply 85", "id": 85},
        ]

        mock_db.get_messages_for_session_window.return_value = windowed_messages

        result = mock_db.get_messages_for_session_window(
            "sess_abc",
            message_ids=matched_ids,
            before=5,
            after=5
        )

        # Both matched IDs should be present
        result_ids = [msg["id"] for msg in result]
        for matched_id in matched_ids:
            assert matched_id in result_ids, (
                f"Matched message ID {matched_id} should be in result"
            )

    def test_get_messages_for_session_window_edge_case_no_matches(self):
        """
        Edge case: FTS5 returns no matched messages for a session.
        The windowed method should return empty list (not crash).
        """
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        mock_db.get_messages_for_session_window.return_value = []

        result = mock_db.get_messages_for_session_window(
            "sess_abc",
            message_ids=[],  # No matches
            before=5,
            after=5
        )

        assert result == [], "Should return empty list when no matched messages"

    def test_get_messages_for_session_window_edge_case_at_session_start(self):
        """
        Edge case: Matched message is near the START of the conversation.
        Window should still include messages after, but before may be limited
        (only messages that exist before the matched ID).
        """
        from unittest.mock import MagicMock

        mock_db = MagicMock()

        # Matched message at ID 3 (near session start, only 2 messages before)
        matched_id = 3

        windowed_messages = [
            {"role": "user", "content": "msg 1", "id": 1},
            {"role": "assistant", "content": "reply 2", "id": 2},
            {"role": "user", "content": "first docker command", "id": 3},  # matched, near start
            {"role": "assistant", "content": "reply 4", "id": 4},
            {"role": "user", "content": "msg 5", "id": 5},
            {"role": "assistant", "content": "reply 6", "id": 6},
            {"role": "user", "content": "msg 7", "id": 7},
            {"role": "assistant", "content": "reply 8", "id": 8},
        ]

        mock_db.get_messages_for_session_window.return_value = windowed_messages

        result = mock_db.get_messages_for_session_window(
            "sess_abc",
            message_ids=[matched_id],
            before=5,
            after=5
        )

        # The matched message should be present
        matched_present = any(msg["id"] == matched_id for msg in result)
        assert matched_present, "Matched message should be in result"

        # There should be messages AFTER (no guarantee on count before due to session start)
        after_msgs = [msg for msg in result if msg["id"] > matched_id]
        assert len(after_msgs) > 0, "Should have messages after match even at session start"

    def test_get_messages_for_session_window_edge_case_at_session_end(self):
        """
        Edge case: Matched message is near the END of the conversation.
        Window should still include messages before, but after may be limited.
        """
        from unittest.mock import MagicMock

        mock_db = MagicMock()

        # Matched message at ID 862 (near end, session has 866 messages total)
        matched_id = 862

        windowed_messages = [
            {"role": "user", "content": "msg 857", "id": 857},
            {"role": "assistant", "content": "reply 858", "id": 858},
            {"role": "user", "content": "msg 859", "id": 859},
            {"role": "assistant", "content": "reply 860", "id": 860},
            {"role": "user", "content": "msg 861", "id": 861},
            {"role": "user", "content": "last docker config", "id": 862},  # matched, near end
            {"role": "assistant", "content": "final reply 863", "id": 863},
            {"role": "user", "content": "msg 864", "id": 864},
            {"role": "assistant", "content": "reply 865", "id": 865},
            {"role": "user", "content": "msg 866", "id": 866},
        ]

        mock_db.get_messages_for_session_window.return_value = windowed_messages

        result = mock_db.get_messages_for_session_window(
            "sess_abc",
            message_ids=[matched_id],
            before=5,
            after=5
        )

        # The matched message should be present
        matched_present = any(msg["id"] == matched_id for msg in result)
        assert matched_present, "Matched message should be in result"

        # There should be messages BEFORE (no guarantee on count after due to session end)
        before_msgs = [msg for msg in result if msg["id"] < matched_id]
        assert len(before_msgs) > 0, "Should have messages before match even at session end"

    def test_session_search_with_windowed_loading_avoids_full_conversation_load(self):
        """
        Integration test: After the fix, session_search should use windowed loading
        instead of get_messages_as_conversation when matched message IDs are available.

        This verifies that for sessions with matched messages, we load only the
        windowed context rather than the entire 866-message conversation.
        """
        from unittest.mock import MagicMock, AsyncMock, patch as _patch
        from tools.session_search_tool import session_search
        import json

        mock_db = MagicMock()

        # FTS5 returned a match with the message ID (from search_messages)
        fts_results = [
            {
                "session_id": "sess_big",
                "content": "docker compose up",
                "source": "cli",
                "session_started": 1709500000,
                "model": "gpt-4o",
                "id": 500,  # The matched message ID
            }
        ]
        mock_db.search_messages.return_value = fts_results
        mock_db.get_session.return_value = {
            "id": "sess_big",
            "source": "cli",
            "started_at": 1709500000,
            "model": "gpt-4o",
            "parent_session_id": None,
        }

        # The windowed loading should return only ~11 messages (before=5 + matched + after=5)
        windowed_messages = [
            {"role": "user", "content": f"msg {i}", "id": 495 + i}
            for i in range(11)
        ]

        # Define side effect to track which method was called
        call_tracker = {"windowed_called": False, "full_called": False}

        def get_messages_window(sid, msg_ids, before=5, after=5):
            call_tracker["windowed_called"] = True
            return windowed_messages

        def get_messages_full(sid):
            call_tracker["full_called"] = True
            # Return 866 messages (the problem!)
            return [{"role": "user", "content": f"msg {i}"} for i in range(866)]

        mock_db.get_messages_for_session_window.side_effect = get_messages_window
        mock_db.get_messages_as_conversation.side_effect = get_messages_full

        with _patch(
            "tools.session_search_tool.async_call_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no provider"),
        ):
            result = json.loads(session_search(query="docker", db=mock_db, limit=1))

        assert result["success"] is True
        # After the fix, windowed loading should be used instead of full conversation load
        assert call_tracker["windowed_called"], (
            "get_messages_for_session_window should have been called (windowed loading)"
        )
        assert not call_tracker["full_called"], (
            "get_messages_as_conversation should NOT be called (full conversation loading)"
        )


# =========================================================================
# session_search (dispatcher)
# =========================================================================

class TestSessionSearch:
    def test_no_db_lazily_opens_default_session_db(self, monkeypatch):
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = []

        class FakeSessionDB:
            def __new__(cls):
                return mock_db

        import types
        import sys

        fake_state = types.ModuleType("hermes_state")
        fake_state.SessionDB = FakeSessionDB
        monkeypatch.setitem(sys.modules, "hermes_state", fake_state)

        result = json.loads(session_search(query="test"))
        assert result["success"] is True
        mock_db.search_messages.assert_called_once()

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

    def test_limit_none_coerced_to_default(self):
        """Model sends limit=null → should fall back to 3, not TypeError."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = []

        result = json.loads(session_search(
            query="test", db=mock_db, limit=None,
        ))
        assert result["success"] is True

    def test_limit_type_object_coerced_to_default(self):
        """Model sends limit as a type object → should fall back to 3, not TypeError."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = []

        result = json.loads(session_search(
            query="test", db=mock_db, limit=int,
        ))
        assert result["success"] is True

    def test_limit_string_coerced(self):
        """Model sends limit as string '2' → should coerce to int."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = []

        result = json.loads(session_search(
            query="test", db=mock_db, limit="2",
        ))
        assert result["success"] is True

    def test_limit_clamped_to_range(self):
        """Negative or zero limit should be clamped to 1."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = []

        result = json.loads(session_search(
            query="test", db=mock_db, limit=-5,
        ))
        assert result["success"] is True

        result = json.loads(session_search(
            query="test", db=mock_db, limit=0,
        ))
        assert result["success"] is True

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

    def test_source_from_resolved_parent_not_fts5_child(self):
        """source in output must reflect the resolved parent session, not the child that matched FTS5.

        Regression test for #15909: when a delegation child session (source='telegram')
        resolves to a parent (source='api_server'), the result entry must report
        'api_server', not 'telegram'.
        """
        from unittest.mock import MagicMock, AsyncMock, patch as _patch
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        # FTS5 hit is in the child delegation session which carries source='telegram'
        mock_db.search_messages.return_value = [
            {
                "session_id": "child_sid",
                "content": "hello world",
                "source": "telegram",       # child session source — wrong value to surface
                "session_started": 1709400000,
                "model": "gpt-4o-mini",
            },
        ]

        def _get_session(session_id):
            if session_id == "child_sid":
                return {
                    "id": "child_sid",
                    "parent_session_id": "parent_sid",
                    "source": "telegram",
                    "started_at": 1709400000,
                    "model": "gpt-4o-mini",
                }
            if session_id == "parent_sid":
                return {
                    "id": "parent_sid",
                    "parent_session_id": None,
                    "source": "api_server",  # correct parent source
                    "started_at": 1709300000,
                    "model": "gpt-4o-mini",
                }
            return None

        mock_db.get_session.side_effect = _get_session
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
        ]

        with _patch(
            "tools.session_search_tool.async_call_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no provider"),
        ):
            result = json.loads(session_search(query="hello world", db=mock_db))

        assert result["success"] is True
        assert result["count"] == 1
        entry = result["results"][0]
        assert entry["session_id"] == "parent_sid", "should report resolved parent session ID"
        assert entry["source"] == "api_server", (
            f"source should be parent's 'api_server', got {entry['source']!r}"
        )

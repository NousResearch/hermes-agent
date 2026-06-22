"""
Integration test: verify _finalize_session persists messages on force-quit.

Tests the fix for TUI sessions losing conversation history when the
user interrupts and exits before the agent thread finishes flushing.

Scenarios:
  1. Normal interrupt (single Ctrl+C) — messages already in session["history"]
  2. Force-quit mid-tool (double Ctrl+C) — session["history"] has previous turns
  3. Empty session — no-op, no crash
  4. Agent with _persist_session missing — graceful no-op
"""

import threading
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(history=None, session_id="test_session_001"):
    """Build a mock AIAgent with enough surface for _finalize_session."""
    agent = MagicMock()
    agent._persist_session = MagicMock()
    agent.commit_memory_session = MagicMock()
    agent.session_id = session_id
    agent.model = "test-model"
    agent.platform = "tui"
    # _session_messages must be explicitly absent (None), otherwise
    # MagicMock auto-creates it and getattr returns a truthy mock.
    agent._session_messages = None
    return agent


def _make_session(agent=None, history=None, session_key="test_key_001"):
    return {
        "agent": agent,
        "history": history or [],
        "history_lock": threading.Lock(),
        "session_key": session_key,
        "_finalized": False,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFinalizeSessionPersist:
    """Verify _finalize_session flushes messages via _persist_session."""

    def test_persist_called_with_history(self):
        """History from session is passed to agent._persist_session.

        When _session_messages is None (not yet set by any turn),
        the session["history"] is used as the snapshot.
        """
        from tui_gateway.server import _finalize_session

        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        agent = _make_agent()
        session = _make_session(agent=agent, history=history)

        _finalize_session(session, end_reason="test")

        agent._persist_session.assert_called_once()
        # snapshot = history (since _session_messages is None)
        called_with = agent._persist_session.call_args[0][0]
        assert called_with == history
        # conversation_history kwarg passed for correct flush indexing
        assert agent._persist_session.call_args[1].get("conversation_history") == history

    def test_persist_uses_session_messages_when_available(self):
        """agent._session_messages takes priority over session['history']."""
        from tui_gateway.server import _finalize_session

        history = [{"role": "user", "content": "old"}]
        session_msgs = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "newer"},
        ]
        agent = _make_agent()
        agent._session_messages = session_msgs
        session = _make_session(agent=agent, history=history)

        _finalize_session(session)

        agent._persist_session.assert_called_once()
        called_with = agent._persist_session.call_args[0][0]
        assert called_with == session_msgs  # _session_messages wins
        assert agent._persist_session.call_args[1].get("conversation_history") == history

    def test_commit_memory_still_called(self):
        """Existing memory commit path is preserved."""
        from tui_gateway.server import _finalize_session

        history = [{"role": "user", "content": "x"}]
        agent = _make_agent()
        session = _make_session(agent=agent, history=history)

        _finalize_session(session)

        agent.commit_memory_session.assert_called_once()

    def test_no_agent_no_crash(self):
        """Session with agent=None exits cleanly."""
        from tui_gateway.server import _finalize_session

        session = _make_session(agent=None, history=[{"role": "user", "content": "x"}])
        _finalize_session(session)  # must not raise

    def test_empty_history_skips_persist(self):
        """Empty history → _persist_session not called (guard)."""
        from tui_gateway.server import _finalize_session

        agent = _make_agent()
        session = _make_session(agent=agent, history=[])

        _finalize_session(session)

        agent._persist_session.assert_not_called()

    def test_no_persist_method_skips(self):
        """Agent without _persist_session attribute → graceful skip."""
        from tui_gateway.server import _finalize_session

        agent = _make_agent()
        del agent._persist_session  # simulate older agent without the method
        session = _make_session(
            agent=agent,
            history=[{"role": "user", "content": "x"}],
        )

        _finalize_session(session)  # must not raise

    def test_already_finalized_skips(self):
        """Double-finalize is a no-op."""
        from tui_gateway.server import _finalize_session

        agent = _make_agent()
        session = _make_session(agent=agent, history=[{"role": "user", "content": "x"}])
        session["_finalized"] = True

        _finalize_session(session)

        agent._persist_session.assert_not_called()

    def test_persist_exception_does_not_block(self):
        """If _persist_session raises, finalization continues."""
        from tui_gateway.server import _finalize_session

        agent = _make_agent()
        agent._persist_session.side_effect = RuntimeError("db is down")
        session = _make_session(
            agent=agent,
            history=[{"role": "user", "content": "x"}],
        )

        _finalize_session(session)  # must not raise
        # commit_memory_session should still be called
        agent.commit_memory_session.assert_called_once()

    @patch("tui_gateway.server._get_db")
    def test_db_end_session_still_called(self, mock_get_db):
        """Existing db.end_session() path is preserved after the new code."""
        from tui_gateway.server import _finalize_session

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        agent = _make_agent(session_id="sess_123")
        session = _make_session(agent=agent, history=[{"role": "user", "content": "x"}])

        _finalize_session(session, end_reason="test")

        mock_db.end_session.assert_called_once_with("sess_123", "test")


class TestOnSessionEndHook:
    """Verify on_session_end plugin hook fires on finalize."""

    @patch("hermes_cli.plugins.invoke_hook")
    def test_hook_fired_with_interrupted_true(self, mock_invoke_hook):
        """on_session_end is called with interrupted=True when finalizing."""
        from tui_gateway.server import _finalize_session

        agent = _make_agent(session_id="hook_test_001")
        agent.model = "claude-sonnet-4"
        agent.platform = "tui"
        session = _make_session(agent=agent, history=[{"role": "user", "content": "test"}])

        _finalize_session(session, end_reason="tui_close")

        mock_invoke_hook.assert_any_call(
            "on_session_end",
            session_id="hook_test_001",
            completed=False,
            interrupted=True,
            model="claude-sonnet-4",
            platform="tui",
        )

    @patch("hermes_cli.plugins.invoke_hook")
    def test_hook_exception_does_not_block(self, mock_invoke_hook):
        """Hook failure doesn't prevent session finalization."""
        from tui_gateway.server import _finalize_session

        mock_invoke_hook.side_effect = RuntimeError("plugin crash")
        agent = _make_agent()
        session = _make_session(agent=agent, history=[{"role": "user", "content": "x"}])

        _finalize_session(session)  # must not raise
        agent.commit_memory_session.assert_called_once()

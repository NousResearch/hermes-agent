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

    def test_no_session_messages_skips_persist(self):
        """When _session_messages is empty/None the agent processed nothing
        this session, so there is nothing new to flush. Falling back to
        session["history"] here re-appended already-durable resumed rows as
        duplicates, so finalize must NOT write in that case.
        """
        from tui_gateway.server import _finalize_session

        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        agent = _make_agent()  # _session_messages is None
        session = _make_session(agent=agent, history=history)

        _finalize_session(session, end_reason="test")

        agent._persist_session.assert_not_called()

    def test_persist_uses_session_messages(self):
        """agent._session_messages is flushed via the marker-based dedup path
        (no conversation_history — passing the same list neutered the write)."""
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

        agent._persist_session.assert_called_once_with(session_msgs)
        # conversation_history must NOT be passed — it aliases the snapshot and
        # makes _flush_messages_to_session_db skip every message.
        assert "conversation_history" not in agent._persist_session.call_args[1]

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
        agent._session_messages = [{"role": "user", "content": "x"}]
        agent._persist_session.side_effect = RuntimeError("db is down")
        session = _make_session(
            agent=agent,
            history=[{"role": "user", "content": "x"}],
        )

        _finalize_session(session)  # must not raise
        agent._persist_session.assert_called_once()
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


class TestFinalizeSessionPersistE2E:
    """End-to-end: _finalize_session must actually land unflushed turns in
    state.db on disconnect/restart.

    The mock-based tests above assert that _persist_session is *called*, but a
    call whose arguments neuter the underlying flush persists nothing. These
    tests drive the REAL AIAgent flush against a REAL SessionDB, reproducing
    the "conversation contains many events yet is absent from state.db across
    disconnect/restart" symptom.
    """

    @staticmethod
    def _real_agent(db, session_id, session_messages):
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = session_id
        agent.platform = "tui"
        agent.model = "test-model"
        agent._session_messages = session_messages
        agent._last_flushed_db_idx = 0
        agent._flushed_db_message_ids = set()
        agent._flushed_db_message_session_id = None
        agent._persist_disabled = False
        agent._cached_system_prompt = None
        agent._session_init_model_config = None
        agent._parent_session_id = None
        agent._session_json_enabled = False
        agent.quiet_mode = True
        # commit_memory_session runs heavy machinery we don't exercise here.
        agent.commit_memory_session = lambda *a, **k: None
        return agent

    def test_unflushed_turn_survives_disconnect(self, tmp_path, monkeypatch):
        """A completed turn whose transcript flush did NOT durably persist
        (messages live only in agent._session_messages / session['history'],
        never written to the DB) must be flushed to state.db when the WS
        disconnect tears the session down."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        from hermes_state import SessionDB
        import tui_gateway.server as srv

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "sess-unflushed"
        db.create_session(session_id=session_id, source="tui")
        monkeypatch.setattr(srv, "_get_db", lambda: db)

        # The live turn list that became session["history"] AND
        # agent._session_messages (same object), but was never persisted.
        turn = [
            {"role": "user", "content": "scan the repo and summarise"},
            {"role": "assistant", "content": "Here is the summary…"},
            {"role": "user", "content": "now open a PR"},
            {"role": "assistant", "content": "PR opened."},
        ]
        agent = self._real_agent(db, session_id, turn)
        session = _make_session(agent=agent, history=turn, session_key=session_id)

        assert db.get_messages_as_conversation(session_id) == []

        srv._finalize_session(session, end_reason="ws_disconnect")

        after = db.get_messages_as_conversation(session_id)
        contents = [m.get("content") for m in after]
        assert len(after) == 4, after
        assert any("scan the repo" in (c or "") for c in contents), contents
        assert any("PR opened" in (c or "") for c in contents), contents

    def test_resumed_session_not_reflushed_as_duplicates(self, tmp_path, monkeypatch):
        """A resumed session torn down before any new turn (its transcript is
        already durable in the DB) must NOT re-append duplicate rows."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        from hermes_state import SessionDB
        import tui_gateway.server as srv

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "sess-resumed"
        db.create_session(session_id=session_id, source="tui")
        loaded = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        for m in loaded:
            db.append_message(session_id=session_id, role=m["role"], content=m["content"])
        monkeypatch.setattr(srv, "_get_db", lambda: db)

        # Resumed session: history hydrated from the DB, no turn ran, so the
        # agent processed nothing this session.
        agent = self._real_agent(db, session_id, [])
        session = _make_session(agent=agent, history=loaded, session_key=session_id)

        srv._finalize_session(session, end_reason="ws_disconnect")

        after = db.get_messages_as_conversation(session_id)
        assert len(after) == 2, after

    def test_resumed_then_run_turn_not_duplicated(self, tmp_path, monkeypatch):
        """A resumed session that RUNS a turn must not have its loaded (durable)
        prefix re-appended by finalize.

        This exercises the exact path the ``conversation_history`` argument used
        to guard: the in-turn flush stamps the loaded prefix with
        ``_DB_PERSISTED_MARKER`` (recognising it as durable), so the marker-only
        finalize flush skips it. Without that stamping — or if finalize wrote a
        markerless copy — the durable prefix would double. The
        ``_session_messages``-empty test above skips the flush entirely, so it
        can't catch a duplicate-write regression; this one drives a real flush.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        from hermes_state import SessionDB
        import tui_gateway.server as srv

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "sess-resume-run"
        db.create_session(session_id=session_id, source="tui")
        loaded = [
            {"role": "user", "content": "remember: my cat is Mochi"},
            {"role": "assistant", "content": "Noted — Mochi."},
        ]
        for m in loaded:
            db.append_message(session_id=session_id, role=m["role"], content=m["content"])
        monkeypatch.setattr(srv, "_get_db", lambda: db)

        # Live turn list = loaded prefix (same dicts, as run_conversation copies
        # conversation_history) + the new turn.
        new_turn = [
            {"role": "user", "content": "what's the cat's name?"},
            {"role": "assistant", "content": "Mochi."},
        ]
        messages = list(loaded) + new_turn
        agent = self._real_agent(db, session_id, messages)

        # Drive the in-turn flush the way run_conversation does — the loaded
        # prefix rides in as conversation_history, so it is recognised durable
        # (and marker-stamped) while only the new turn is written.
        agent._flush_messages_to_session_db(messages, conversation_history=loaded)
        assert len(db.get_messages_as_conversation(session_id)) == 4

        # WS disconnect → finalize. Must re-append nothing.
        session = _make_session(agent=agent, history=messages, session_key=session_id)
        srv._finalize_session(session, end_reason="ws_disconnect")

        after = db.get_messages_as_conversation(session_id)
        assert len(after) == 4, after


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

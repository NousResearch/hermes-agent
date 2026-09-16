"""Tests for #60609: the TUI backend must not end gateway-owned sessions.

``_finalize_session`` (and thus the ws-orphan reaper / session.close paths
that funnel into it) marks the session ended in state.db.  For sessions the
messaging gateway owns (telegram, discord, ...), that write creates the
Groundhog Day routing loop described in #60609 — the gateway self-heal
drops the ended-but-routed entry, recovers hours-old parent context, and
loops.  The TUI is only a viewer of those sessions.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from tui_gateway.server import _finalize_session, _is_gateway_owned_source, _teardown_session


class TestIsGatewayOwnedSource:
    def test_builtin_gateway_platforms_are_owned(self):
        for src in ("telegram", "discord", "whatsapp", "slack", "signal",
                    "matrix", "mattermost", "bluebubbles", "sms", "email"):
            assert _is_gateway_owned_source(src) is True, src

    def test_case_and_whitespace_normalized(self):
        assert _is_gateway_owned_source(" Telegram ") is True

    def test_tui_owned_sources_are_not(self):
        for src in ("tui", "cli", "webui", "desktop", "cron", "subagent",
                    "test", "acp", ""):
            assert _is_gateway_owned_source(src) is False, src

    def test_local_and_server_endpoints_are_not(self):
        # Platform enum members, but their sessions aren't owned by a remote
        # chat surface — reaping them keeps /resume clean.
        for src in ("local", "webhook", "api_server", "msgraph_webhook"):
            assert _is_gateway_owned_source(src) is False, src

    def test_arbitrary_strings_are_not(self):
        assert _is_gateway_owned_source("hermesbench-task-xyz") is False
        assert _is_gateway_owned_source(None) is False


def _make_session(session_id="sess_1"):
    agent = MagicMock()
    agent.session_id = session_id
    return {
        "agent": agent,
        "history": [{"role": "user", "content": "x"}],
        "history_lock": None,
        "session_key": session_id,
    }


def _make_real_session(tmp_path, monkeypatch, *, source, session_id):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    db_path = hermes_home / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source=source)

    # Exercise the production close path without constructing a model provider.
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent._session_db = db
    agent._owns_session_db = True
    agent._end_session_on_close = True
    agent._session_messages = []
    agent._active_children_lock = threading.Lock()
    agent._active_children = []

    session = _make_session(session_id)
    session["agent"] = agent
    session["profile_home"] = str(hermes_home)
    session["source"] = source
    return db_path, session, agent


def _read_real_row(db_path, session_id):
    from hermes_state import SessionDB

    db = SessionDB(db_path=db_path)
    try:
        return db.get_session(session_id)
    finally:
        db.close()


class TestFinalizeSkipsGatewaySessions:
    @patch("tui_gateway.server._get_db")
    def test_gateway_session_not_ended(self, mock_get_db):
        db = MagicMock()
        db.get_session.return_value = {"id": "sess_1", "source": "telegram"}
        mock_get_db.return_value = db

        _finalize_session(_make_session(), end_reason="ws_orphan_reap")

        db.end_session.assert_not_called()


    @patch("tui_gateway.server._get_db")
    def test_missing_row_still_ended(self, mock_get_db):
        """A session with no state.db row can't be gateway-owned — keep the
        pre-existing reap behavior."""
        db = MagicMock()
        db.get_session.return_value = None
        mock_get_db.return_value = db

        _finalize_session(_make_session(), end_reason="tui_close")

        db.end_session.assert_called_once_with("sess_1", "tui_close")


class TestGatewayOwnedSessionTeardown:
    def test_viewer_teardown_leaves_gateway_owned_row_open(self, tmp_path, monkeypatch):
        session_id = "gateway-owned-session"
        db_path, session, agent = _make_real_session(
            tmp_path, monkeypatch, source="telegram", session_id=session_id
        )

        _teardown_session(session, end_reason="ws_orphan_reap")

        row = _read_real_row(db_path, session_id)
        assert row["ended_at"] is None
        assert row["end_reason"] is None
        assert agent._end_session_on_close is False


class TestDesktopAutomaticReclaimTeardown:
    """#105588: automatic Desktop cleanup reclaims runtime but leaves the durable row open THROUGH the full
    teardown — ``_finalize_session`` skipping ``end_session`` is not enough, ``agent.close()`` ends the row
    as ``agent_close`` unless the spare decision reaches the agent."""

    @pytest.mark.parametrize("reason", ["ws_orphan_reap", "idle_timeout", "lru_evict"])
    def test_automatic_reclaim_leaves_desktop_row_open(self, tmp_path, monkeypatch, reason):
        db_path, session, agent = _make_real_session(
            tmp_path, monkeypatch, source="desktop", session_id=f"desktop-{reason}"
        )

        _teardown_session(session, end_reason=reason)

        row = _read_real_row(db_path, f"desktop-{reason}")
        assert (row["ended_at"], row["end_reason"]) == (None, None)
        assert agent._end_session_on_close is False

    @pytest.mark.parametrize("source,reason", [("desktop", "tui_close"), ("tui", "ws_orphan_reap")])
    def test_explicit_close_and_non_desktop_reap_still_end_row(self, tmp_path, monkeypatch, source, reason):
        db_path, session, _agent = _make_real_session(
            tmp_path, monkeypatch, source=source, session_id=f"{source}-{reason}"
        )

        _teardown_session(session, end_reason=reason)

        row = _read_real_row(db_path, f"{source}-{reason}")
        assert row["ended_at"] is not None
        assert row["end_reason"] == reason

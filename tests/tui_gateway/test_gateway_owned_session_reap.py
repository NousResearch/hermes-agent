"""Tests for #60609: the TUI backend must not end gateway-owned sessions.

``_finalize_session`` (and thus the ws-orphan reaper / session.close paths
that funnel into it) marks the session ended in state.db.  For sessions the
messaging gateway owns (telegram, discord, ...), that write creates the
Groundhog Day routing loop described in #60609 — the gateway self-heal
drops the ended-but-routed entry, recovers hours-old parent context, and
loops.  The TUI is only a viewer of those sessions.
"""

from unittest.mock import MagicMock, patch

from tui_gateway.server import _finalize_session, _is_gateway_owned_source


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


class TestFinalizeFailsClosedWhenProfileDbUnavailable:
    """Companion to TestFinalizeSkipsGatewaySessions, for the profile-DB half.

    The row of a profile-owned session lives only in that profile's state.db.
    When it cannot be opened, gateway ownership is unprovable — assuming
    TUI ownership ends the gateway's row (#60609 Groundhog Day loop) and
    interrupts its live delegations. Fail closed there; the launch profile
    keeps the "assume ownership" contract.
    """

    def _make_profile_session(self, profile_home, session_key="sess_prof_1"):
        agent = MagicMock()
        agent.session_id = session_key
        sess = _make_session()
        sess["profile_home"] = str(profile_home)
        sess["session_key"] = session_key
        sess["agent"] = agent
        return sess

    @staticmethod
    def _capture_interrupts():
        """Capture interrupt_for_session's keyword args."""
        captured = []

        def _capture(session_key="", origin_ui_session_id="", parent_session_id="", reason="session_end"):
            captured.append({"session_key": session_key, "reason": reason})

        return captured, _capture

    def test_unopenable_profile_db_does_not_interrupt_by_key(self, tmp_path):
        profile_home = tmp_path / "remote-profile"
        profile_home.mkdir()
        sess = self._make_profile_session(profile_home)
        captured, _capture = self._capture_interrupts()
        # A directory where sqlite expects its file: _session_db yields None.
        (profile_home / "state.db").mkdir()
        with patch("tools.async_delegation.interrupt_for_session", _capture):
            _finalize_session(sess, end_reason="ws_orphan_reap")

        # Fail closed: the durable session_key selector is emptied — the TUI is
        # only a viewer when ownership is unprovable.
        assert captured == [{"session_key": "", "reason": "ws_orphan_reap"}]

    def test_openable_profile_db_still_interrupts_by_key(self, tmp_path):
        profile_home = tmp_path / "the-profile"
        profile_home.mkdir()
        sess = self._make_profile_session(profile_home)
        captured, _capture = self._capture_interrupts()
        from hermes_state import SessionDB

        db = SessionDB(db_path=profile_home / "state.db")
        db.create_session("sess_prof_1", source="tui")
        db.close()
        with patch("tools.async_delegation.interrupt_for_session", _capture):
            _finalize_session(sess, end_reason="tui_close")

        # Ownership proven (tui-sourced row in its own db): interrupt by key.
        assert captured == [{"session_key": "sess_prof_1", "reason": "tui_close"}]

    def test_openable_profile_db_gateway_owned_row_is_never_ended(self, tmp_path):
        profile_home = tmp_path / "gateway-owned"
        profile_home.mkdir()
        sess = self._make_profile_session(profile_home, session_key="gateway-owned-key")
        captured, _capture = self._capture_interrupts()
        from hermes_state import SessionDB

        db = SessionDB(db_path=profile_home / "state.db")
        db.create_session("gateway-owned-key", source="telegram")
        db.close()
        with patch("tools.async_delegation.interrupt_for_session", _capture):
            _finalize_session(sess, end_reason="tui_close")

        assert captured == [{"session_key": "", "reason": "tui_close"}]
        reopened = SessionDB(db_path=profile_home / "state.db")
        try:
            row = reopened.get_session("gateway-owned-key")
            assert row is not None
            assert row.get("ended_at") is None
        finally:
            reopened.close()

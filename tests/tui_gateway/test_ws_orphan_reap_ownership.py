"""Regression tests for #60609 — ws_orphan_reap must not end sessions the
TUI does not own.

``_finalize_session`` marks a session ended in state.db so it doesn't linger
as a ghost row in /resume.  But the TUI can also *view* sessions it does not
own — gateway-originated sessions (telegram, bluebubbles, ...) opened via
``session.resume``.  The messaging gateway is the lifecycle owner of those
and still routes inbound messages to them; ending one in state.db triggers
the Groundhog Day routing loop described in #60609 (the gateway's #54878
self-heal drops the "ended" routing entry, recovers to the pre-compression
parent, compression splits back to the reaped child, repeat on every
message).

The guard is a fail-closed ALLOW-list (``_TUI_OWNED_SESSION_SOURCES``):
only sources this process creates (tui/cli/desktop/webui/local/codex) are
ever ended; unknown/plugin/gateway/tool sources are left alone.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from tui_gateway.server import _TUI_OWNED_SESSION_SOURCES, _finalize_session


def _make_agent(session_id="sess_60609"):
    agent = MagicMock()
    agent._persist_session = MagicMock()
    agent.commit_memory_session = MagicMock()
    agent.session_id = session_id
    agent.model = "test-model"
    agent.platform = "tui"
    agent._session_messages = None
    return agent


def _make_session(agent, session_key="key_60609"):
    return {
        "agent": agent,
        "history": [{"role": "user", "content": "x"}],
        "history_lock": threading.Lock(),
        "session_key": session_key,
        "_finalized": False,
    }


def _finalize_with_source(source):
    """Run _finalize_session against a mock DB whose session row has *source*.

    Returns the mock db so callers can assert on end_session.
    """
    mock_db = MagicMock()
    row = {"source": source} if source is not None else None
    mock_db.get_session.return_value = row
    with patch("tui_gateway.server._get_db", return_value=mock_db):
        _finalize_session(_make_session(_make_agent()), end_reason="ws_orphan_reap")
    return mock_db


class TestGatewaySessionsNeverEnded:
    """Sources owned by the messaging gateway must never be end_session()'d."""

    @pytest.mark.parametrize(
        "source",
        [
            "telegram",
            "bluebubbles",
            "discord",
            "signal",
            "whatsapp",
            "slack",
            "sms",
            "matrix",
            "mattermost",
            # Platforms the original deny-list proposal MISSED — the reason
            # the guard is an allow-list:
            "qqbot",
            "dingtalk",
            "wecom",
            "feishu",
            "yuanbao",
            "email",
            "webhook",
            "homeassistant",
            "irc",  # plugin platform
        ],
    )
    def test_gateway_source_not_ended(self, source):
        db = _finalize_with_source(source)
        db.end_session.assert_not_called()

    def test_tool_subagent_row_not_ended(self):
        """Sub-agent rows are owned by their parent run, not the viewer."""
        db = _finalize_with_source("tool")
        db.end_session.assert_not_called()

    def test_unknown_source_not_ended(self):
        """The gateway can stamp source="unknown" — fail closed."""
        db = _finalize_with_source("unknown")
        db.end_session.assert_not_called()

    def test_missing_row_not_ended(self):
        """No DB row at all → cannot prove ownership → don't end."""
        db = _finalize_with_source(None)
        db.end_session.assert_not_called()

    def test_empty_source_not_ended(self):
        db = _finalize_with_source("")
        db.end_session.assert_not_called()


class TestTuiOwnedSessionsStillEnded:
    """TUI-owned sessions keep the #20001 ghost-row cleanup behavior."""

    @pytest.mark.parametrize("source", sorted(_TUI_OWNED_SESSION_SOURCES))
    def test_owned_source_ended(self, source):
        db = _finalize_with_source(source)
        db.end_session.assert_called_once_with("sess_60609", "ws_orphan_reap")

    def test_source_matching_is_case_and_whitespace_insensitive(self):
        db = _finalize_with_source("  TUI ")
        db.end_session.assert_called_once()


class TestAllowListShape:
    def test_no_gateway_platform_in_allow_list(self):
        """Invariant: no gateway Platform enum value may ever be TUI-owned."""
        from gateway.config import Platform

        gateway_values = {p.value for p in Platform}
        assert not (_TUI_OWNED_SESSION_SOURCES & gateway_values)

    def test_tool_not_in_allow_list(self):
        assert "tool" not in _TUI_OWNED_SESSION_SOURCES

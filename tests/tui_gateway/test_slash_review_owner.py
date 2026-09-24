"""/review from slash.exec runs on the RPC pool, outside any turn. The reviewer it dispatches must
still be registered under the parent Desktop/TUI session (HERMES_UI_SESSION_ID + exact steer
authority), or `subagent.list` hides it and the Desktop status stack shows nothing for /review."""

from __future__ import annotations

import threading
from unittest.mock import patch

from tui_gateway import server
from tui_gateway.transport import StdioTransport


def _live_session(agent):
    return {
        "agent": agent, "session_key": "review-key", "history": [{"role": "user", "content": "hi"}],
        "history_lock": threading.Lock(), "running": False, "transport": StdioTransport(lambda: None, threading.Lock()),
        "cwd": "", "source": "desktop",
    }


def test_slash_review_dispatches_under_the_parent_session_identity(monkeypatch):
    sid = "review-sid"
    session = _live_session(object())
    server._sessions[sid] = session
    seen = {}

    def fake_start_review(agent, snapshot, prompt):
        from gateway.session_context import get_session_env
        seen["ui_session_id"] = get_session_env("HERMES_UI_SESSION_ID", "")
        seen["authority"] = server._current_session_steer_authority(sid)
        return {"status": "dispatched", "delegation_id": "deleg_x"}

    token = server.bind_transport(session["transport"])
    try:
        with (
            patch.object(server, "_session_uses_compute_host", return_value=False),
            patch("agent.review_engine.start_review", fake_start_review),
        ):
            out = server._live_slash_command_output(sid, session, "review", "")
        from gateway.session_context import get_session_env
        assert get_session_env("HERMES_UI_SESSION_ID", "") == ""  # scope cleared after dispatch
        assert server._current_runtime_session_record.get() is None
    finally:
        server.reset_transport(token)
        server._sessions.pop(sid, None)

    assert out == "Review started. Results will return here."
    assert seen["ui_session_id"] == sid
    assert seen["authority"] == (session["transport"], session)

"""handoff.request must tolerate non-string ``platform`` on the inline reader path."""

from __future__ import annotations

import threading

from tui_gateway import server


def _install_session(sid: str):
    ready = threading.Event()
    ready.set()
    server._sessions[sid] = {
        "agent": object(),
        "agent_ready": ready,
        "session_key": sid,
        "running": False,
        "history_lock": threading.Lock(),
    }


def test_handoff_request_rejects_null_platform():
    sid = "handoff-null"
    _install_session(sid)
    try:
        resp = server.dispatch(
            {
                "id": "1",
                "method": "handoff.request",
                "params": {"session_id": sid, "platform": None},
            }
        )
        assert resp["error"]["code"] == 4023
    finally:
        server._sessions.pop(sid, None)


def test_handoff_request_list_platform_does_not_crash():
    """List platform must not AttributeError on the inline reader path."""
    sid = "handoff-list"
    _install_session(sid)
    try:
        resp = server.dispatch(
            {
                "id": "2",
                "method": "handoff.request",
                "params": {"session_id": sid, "platform": ["telegram"]},
            }
        )
        # Coerced str(list) is not a valid Platform name — expect 4024, not a crash.
        assert "error" in resp, resp
        assert resp["error"]["code"] == 4024
    finally:
        server._sessions.pop(sid, None)


def test_handoff_request_int_platform_does_not_crash():
    sid = "handoff-int"
    _install_session(sid)
    try:
        resp = server.dispatch(
            {
                "id": "3",
                "method": "handoff.request",
                "params": {"session_id": sid, "platform": 42},
            }
        )
        assert "error" in resp, resp
        assert resp["error"]["code"] == 4024
    finally:
        server._sessions.pop(sid, None)

"""session.steer / session.redirect must tolerate non-string ``text``.

Both RPCs run inline (not in ``_LONG_HANDLERS``). A bare ``.strip()`` on a
list/int ``text`` raises AttributeError and can tear down the stdin/WS
reader loop.
"""

from __future__ import annotations

import threading

from tui_gateway import server


def _install_session(sid: str, *, support_steer: bool = True):
    class _Agent:
        def steer(self, text):
            self.last = text
            return True

        def redirect(self, text):
            self.last = text
            return True

    agent = _Agent() if support_steer else object()
    ready = threading.Event()
    ready.set()
    server._sessions[sid] = {
        "agent": agent,
        "agent_ready": ready,
        "session_key": sid,
        "history_lock": threading.Lock(),
        "last_active": 0,
    }
    return agent


def test_session_steer_rejects_empty_after_coerce():
    sid = "steer-empty"
    _install_session(sid)
    try:
        resp = server.dispatch(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": sid, "text": None},
            }
        )
        assert resp["error"]["code"] == 4002
    finally:
        server._sessions.pop(sid, None)


def test_session_steer_coerces_int_text():
    sid = "steer-int"
    agent = _install_session(sid)
    try:
        resp = server.dispatch(
            {
                "id": "2",
                "method": "session.steer",
                "params": {"session_id": sid, "text": 42},
            }
        )
        assert "error" not in resp, resp
        assert resp["result"]["status"] == "queued"
        assert resp["result"]["text"] == "42"
        assert agent.last == "42"
    finally:
        server._sessions.pop(sid, None)


def test_session_redirect_coerces_list_text_without_crash():
    """List text must not AttributeError on the inline reader path."""
    sid = "redir-list"
    _install_session(sid)
    try:
        resp = server.dispatch(
            {
                "id": "3",
                "method": "session.redirect",
                "params": {"session_id": sid, "text": ["go", "left"]},
            }
        )
        # Coerced to "['go', 'left']" which is non-empty — either queued/rejected
        # or a later agent error, but never an uncaught AttributeError.
        assert isinstance(resp, dict)
        assert "error" in resp or "result" in resp
    finally:
        server._sessions.pop(sid, None)

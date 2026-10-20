"""Pending-steer RPC contract: ``session.steer_clear`` drops the stashed slot and
``session.pending`` snapshots the steer plus every queued prompt.

The TUI renders a pending steer as row 1 of its queue strip, so deleting that row
(Ctrl+X) has to clear the agent slot server-side — otherwise the text still gets
injected on the next tool result — and a reconnect has to be able to rebuild the
strip from ``session.pending``. ``session.steer`` itself keeps upstream's statuses
(``queued`` on the busy path, ``rejected`` when idle); these tests pin only the two
new methods and the rejected-steer passthrough.
"""

from __future__ import annotations

import threading
import types

from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "transport": None,
        "attached_images": [],
        **extra,
    }


def test_session_steer_rejected_when_agent_refuses():
    """When the agent refuses the steer, the RPC must report 'rejected' so the
    client falls back to the queue instead of showing a phantom pending steer.
    """
    server._sessions["sid"] = _session(agent=types.SimpleNamespace(steer=lambda t: False))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "sid", "text": "nope"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "result" in resp, resp
    assert resp["result"]["status"] == "rejected"
    assert resp["result"]["text"] == "nope"


def test_session_steer_clear_drops_pending_steer():
    """session.steer_clear must drop the agent's stashed pending steer and
    report whether there was anything to drop."""
    lock = threading.Lock()
    agent = types.SimpleNamespace(_pending_steer="inject me", _pending_steer_lock=lock)
    server._sessions["sid"] = _session(agent=agent)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer_clear",
                "params": {"session_id": "sid"},
            }
        )
        assert resp["result"]["status"] == "cleared"
        assert resp["result"]["cleared"] is True
        assert agent._pending_steer is None

        resp = server.handle_request(
            {
                "id": "2",
                "method": "session.steer_clear",
                "params": {"session_id": "sid"},
            }
        )
        assert resp["result"]["status"] == "noop"
        assert resp["result"]["cleared"] is False
    finally:
        server._sessions.pop("sid", None)


def test_session_steer_clear_uses_agent_method_when_present():
    """A live AIAgent exposes clear_pending_steer(); the RPC must go through it
    (it owns the lock) instead of reaching into the slot."""
    calls = []

    class _Agent:
        _pending_steer = "text"

        def clear_pending_steer(self):
            calls.append("cleared")
            self._pending_steer = None
            return True

    agent = _Agent()
    server._sessions["sid"] = _session(agent=agent)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.steer_clear", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"] == {"status": "cleared", "cleared": True}
    assert calls == ["cleared"]


def test_session_steer_clear_without_agent_is_4010():
    server._sessions["sid"] = _session(agent=None)
    server._sessions["sid"]["agent"] = None
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.steer_clear", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["error"]["code"] == 4010


def test_session_pending_snapshots_steer_and_queue():
    """session.pending must expose the pending steer plus every queued prompt
    so a reconnect can rebuild the TUI pending strip."""
    agent = types.SimpleNamespace(_pending_steer="wait, use SQLite")
    server._sessions["sid"] = _session(
        agent=agent,
        queued_prompt={"text": "first queued", "transport": object()},
        queued_prompts=[{"text": "second queued", "transport": object()}],
    )
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.pending",
                "params": {"session_id": "sid"},
            }
        )
        assert resp["result"] == {
            "steer": "wait, use SQLite",
            "queue": ["first queued", "second queued"],
        }
    finally:
        server._sessions.pop("sid", None)


def test_session_pending_includes_a_waiting_model_redirect():
    """A redirect that has not rebuilt the request yet is part of the snapshot,
    separate from the steer slot a tool-time redirect uses."""
    agent = types.SimpleNamespace(_pending_steer=None, _pending_redirect="answer with Monday only")
    server._sessions["sid"] = _session(agent=agent)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.pending", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["steer"] is None
    assert resp["result"]["redirect"] == "answer with Monday only"
    assert resp["result"]["queue"] == []


def test_session_pending_empty_when_nothing_pending():
    server._sessions["sid"] = _session(agent=types.SimpleNamespace())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.pending", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"] == {"steer": None, "queue": []}

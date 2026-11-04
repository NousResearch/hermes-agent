"""``session.redirect`` between two model requests of a live turn joins it at the steer boundary.

A redirect cancels only an in-flight model request. Between two iterations (or past the last
one) the agent refuses it, and the Desktop queued the refused correction behind the whole turn
(#141973). While the session is running, the handler now hands the words to the steer drain and
answers ``queued``; an idle session still refuses, so its text becomes a normal next prompt.
"""

from __future__ import annotations

import threading

from agent.interrupt_control import InterruptControlMixin
from tui_gateway import server


class _Agent(InterruptControlMixin):
    _supports_active_turn_redirect = True

    def __init__(self) -> None:
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()
        self._pending_redirect = None
        self._pending_redirect_lock = threading.Lock()
        self._model_request_active = threading.Event()  # never set: between two model requests
        self._executing_tools = False
        self._interrupt_requested = False
        self._interrupt_message = None


def _redirect(running: bool) -> tuple[dict, _Agent]:
    agent = _Agent()
    server._sessions["sid"] = {
        "agent": agent, "session_key": "session-key", "history": [], "history_lock": threading.RLock(),
        "history_version": 0, "running": running, "transport": None, "attached_images": [],
    }
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.redirect", "params": {"session_id": "sid", "text": "use Postgres"}})
    finally:
        server._sessions.pop("sid", None)
    return resp, agent


def test_redirect_between_requests_of_a_running_turn_is_steered_into_it():
    resp, agent = _redirect(running=True)

    assert resp["result"]["status"] == "queued"
    assert agent._drain_pending_steer() == "use Postgres"
    assert agent._pending_redirect is None


def test_redirect_into_an_idle_session_still_refuses():
    resp, agent = _redirect(running=False)

    assert resp["result"]["status"] == "rejected"
    assert agent._pending_steer is None

"""A Desktop/TUI steer reaches the subagents the running turn is waiting on (#141973).

A parent blocked inside ``delegate_task`` drains its own steer buffer only after the tool returns,
i.e. after the child finishes, so a correction aimed at the delegated work sat unread for the whole
delegation. The messaging gateway already fans an accepted steer out to the parent's live children;
``session.steer`` and a ``session.redirect`` that takes the steer boundary now do the same.
"""

from __future__ import annotations

import threading

import pytest

from agent.interrupt_control import InterruptControlMixin
from tui_gateway import server


class _Agent(InterruptControlMixin):
    _supports_active_turn_redirect = True

    def __init__(self, children=()) -> None:
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()
        self._pending_redirect = None
        self._pending_redirect_lock = threading.Lock()
        self._model_request_active = threading.Event()
        self._executing_tools = True  # the parent is inside delegate_task
        self._interrupt_requested = False
        self._interrupt_message = None
        self._active_children = list(children)
        self._active_children_lock = threading.Lock()


# A redirect while the parent runs tools already lands on its steer boundary (answered 'redirected').
@pytest.mark.parametrize(("method", "status"), [("session.steer", "queued"), ("session.redirect", "redirected")])
def test_a_live_steer_also_reaches_the_running_subagent(method, status):
    child = _Agent()
    parent = _Agent(children=[child])
    server._sessions["sid"] = {
        "agent": parent, "session_key": "session-key", "history": [], "history_lock": threading.RLock(),
        "history_version": 0, "running": True, "transport": None, "attached_images": [],
    }
    try:
        resp = server.handle_request(
            {"id": "1", "method": method, "params": {"session_id": "sid", "text": "skip the tests"}})
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["status"] == status
    assert child._drain_pending_steer() == "skip the tests"
    assert parent._drain_pending_steer() == "skip the tests"

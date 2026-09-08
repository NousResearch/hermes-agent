"""`session.active_list` / `session.status` must report a session parked on a
dangerous-command approval as ``waiting``, not ``working``.

Approvals queue in ``tools.approval._gateway_queues`` (keyed by session key),
not in the server's ``_pending`` map that clarify/sudo/secret use. A live
status that only consulted ``_pending`` reported the blocked session as
``working``; the desktop's liveness poll then published ``needsInput: False``
a beat after the ``approval.request`` event set it, and the sidebar's amber
"needs input" cue vanished while the agent sat blocked (the symptom: a chat
that "didn't even start because it needed approval", with nothing in the
sidebar saying so).
"""

import threading

import pytest

from tools import approval as approval_mod
from tools.approval_gateway_wait import _ApprovalEntry
from tui_gateway import server


@pytest.fixture
def parked_approval():
    key = "20990101_000000_approv"
    entry = _ApprovalEntry(
        {"command": "python3 -c 'x'", "pattern_key": "k", "pattern_keys": ["k"], "description": "d"}
    )
    with approval_mod._lock:
        approval_mod._gateway_queues.setdefault(key, []).append(entry)
    try:
        yield key
    finally:
        with approval_mod._lock:
            approval_mod._gateway_queues.pop(key, None)


def _running_session(key: str) -> dict:
    ready = threading.Event()
    ready.set()
    return {"running": True, "agent_ready": ready, "session_key": key}


def test_live_status_is_waiting_while_an_approval_is_queued(monkeypatch, parked_approval):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")

    assert server._session_live_status("sid", _running_session(parked_approval)) == "waiting"


def test_live_status_returns_to_working_once_the_approval_resolves(monkeypatch, parked_approval):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    session = _running_session(parked_approval)
    assert server._session_live_status("sid", session) == "waiting"

    with approval_mod._lock:
        approval_mod._gateway_queues.pop(parked_approval, None)

    assert server._session_live_status("sid", session) == "working"


def test_live_status_ignores_another_sessions_approval(monkeypatch, parked_approval):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")

    assert server._session_live_status("sid", _running_session("20990101_000000_other")) == "working"

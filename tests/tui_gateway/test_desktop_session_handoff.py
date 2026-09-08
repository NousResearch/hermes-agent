"""Desktop session writer handoff contracts for one shared WebSocket gateway.

A resumed session may have several viewers, but only one transport may submit
turns. The tests exercise the JSON-RPC boundary rather than source shape:

* a second live viewer cannot submit merely because prompt.submit rebinding
  changed the session's event transport;
* an explicit handoff transfers an idle writer to that viewer;
* a running turn is never transferred implicitly;
* disconnecting the writer fences its old transport before another viewer can
  reclaim the session.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class _Transport:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self._closed = False

    def write(self, frame: dict) -> bool:
        self.frames.append(frame)
        return not self._closed

    def close(self) -> None:
        self._closed = True


@contextmanager
def _bound(transport):
    token = bind_transport(transport)
    try:
        yield
    finally:
        reset_transport(token)


def _session(owner: _Transport, viewer: _Transport) -> dict:
    return {
        "_writer_lock": threading.RLock(),
        "_writer_transport": owner,
        "active_session_lease": object(),
        "agent": SimpleNamespace(session_id="stored-handoff"),
        "agent_ready": threading.Event(),
        "attached_images": [],
        "close_on_disconnect": False,
        "history": [],
        "history_lock": threading.Lock(),
        "inflight_turn": None,
        "profile_home": None,
        "running": False,
        "session_key": "stored-handoff",
        "source": "desktop",
        "transport": owner,
        "viewers": {owner: 1.0, viewer: 2.0},
    }


@pytest.fixture
def registered_session(monkeypatch):
    owner, viewer = _Transport(), _Transport()
    session = _session(owner, viewer)
    sid = "runtime-handoff"
    with server._sessions_lock:
        server._sessions[sid] = session
    try:
        yield sid, session, owner, viewer
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        server._pending_ws_reaps.pop(sid, None)


def test_second_viewer_cannot_submit_through_transport_rebind(
    registered_session, monkeypatch
):
    """The event sink is not the writer authority.

    Before the fix, prompt.submit assigned ``session["transport"]`` to the
    request transport before checking ownership. With an already-held lease,
    the second viewer therefore passed the cheap lease check and could submit
    on the same runtime as the first viewer.
    """
    sid, session, owner, viewer = registered_session
    monkeypatch.setattr(server, "_legacy_group_fence_error", lambda *_args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda *_args: {})
    monkeypatch.setattr(server, "_lock_in_submit_turn", lambda *_args: (None, {}))
    monkeypatch.setattr(server, "_persist_session_row_for_submit", lambda *_args: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_args: True)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *_args: None)

    with _bound(viewer):
        response = server.handle_request(
            {"id": "submit-1", "method": "prompt.submit", "params": {"session_id": sid, "text": "duplicate?"}}
        )

    assert response["error"]["code"] == 4090
    assert response["error"]["data"] == {
        "handoff_available": True,
        "owner_running": False,
        "reason": "SESSION_NOT_OWNED",
        "session_id": "stored-handoff",
    }
    assert session["transport"] is owner
    assert session["_writer_transport"] is owner


def test_explicit_handoff_transfers_an_idle_writer(registered_session):
    sid, session, owner, viewer = registered_session

    with _bound(viewer):
        response = server.handle_request(
            {"id": "handoff-1", "method": "session.handoff", "params": {"session_id": sid}}
        )

    assert response["result"] == {
        "session_id": sid,
        "status": "transferred",
        "writer": True,
    }
    assert session["_writer_transport"] is viewer
    assert session["transport"] is viewer


def test_handoff_does_not_steal_a_running_turn(registered_session):
    sid, session, _owner, viewer = registered_session
    session["running"] = True

    with _bound(viewer):
        response = server.handle_request(
            {"id": "handoff-2", "method": "session.handoff", "params": {"session_id": sid}}
        )

    assert response["error"]["code"] == 4090
    assert response["error"]["data"] == {
        "handoff_available": False,
        "owner_running": True,
        "reason": "SESSION_NOT_OWNED",
        "session_id": "stored-handoff",
    }
    assert session["_writer_transport"] is not viewer


def test_disconnect_fences_the_old_writer_before_viewer_reclaim(registered_session, monkeypatch):
    sid, session, owner, viewer = registered_session
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda _sid: None)

    reaped, detached = server._close_sessions_for_transport(owner)

    assert (reaped, detached) == (0, 0)
    assert session["_writer_transport"] is server._detached_ws_transport
    assert session["transport"] is viewer

    with _bound(viewer):
        response = server.handle_request(
            {"id": "handoff-3", "method": "session.handoff", "params": {"session_id": sid}}
        )

    assert response["result"]["status"] == "transferred"
    assert session["_writer_transport"] is viewer


def test_disconnect_of_demoted_viewer_does_not_detach_new_writer(registered_session, monkeypatch):
    sid, session, owner, viewer = registered_session
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda _sid: None)

    with _bound(viewer):
        handoff = server.handle_request(
            {"id": "handoff-4", "method": "session.handoff", "params": {"session_id": sid}}
        )
    assert handoff["result"]["status"] == "transferred"

    reaped, detached = server._close_sessions_for_transport(owner)

    assert (reaped, detached) == (0, 0)
    assert session["_writer_transport"] is viewer
    assert session["transport"] is viewer
    assert owner not in session["viewers"]


def test_queued_turn_uses_surviving_viewer_after_writer_disconnect(registered_session, monkeypatch):
    sid, session, owner, viewer = registered_session
    owner.close()
    session["queued_prompt"] = {"text": "continue later", "transport": owner}
    session["queued_prompts"] = []
    session["_writer_transport"] = server._detached_ws_transport
    session["transport"] = viewer

    fired: list[tuple[str, str]] = []
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, runtime_sid, _session, text, **_kwargs: fired.append((runtime_sid, text)),
    )

    assert server._drain_queued_prompt("queued-1", sid, session) is True
    assert fired == [(sid, "continue later")]
    assert session["_writer_transport"] is viewer
    assert session["transport"] is viewer

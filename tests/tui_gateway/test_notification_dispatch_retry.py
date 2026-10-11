"""Rejected notification dispatch must retain delivery work (regression for #90020)."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tui_gateway import server


def _session(**extra):
    return {"agent": SimpleNamespace(), "session_key": "session-key", "history": [],
            "history_lock": threading.RLock(), "running": False, **extra}


@pytest.mark.parametrize("deferred_drain", [False, True])
@pytest.mark.parametrize("event_type", ["async_delegation", "completion"])
def test_notification_poller_releases_claim_when_dispatch_is_rejected(
    monkeypatch, deferred_drain, event_type
):
    """Rebound notification helpers release and requeue every refused delivery."""
    import queue
    from tools import async_delegation
    from tools.process_registry import process_registry
    from tools.process_registry_notifications import format_process_notification

    session = _session(session_key="notification-owner")
    events = [{"type": event_type, "delegation_id": "refused-delegation",
               "session_id": "refused-process-1", "session_key": "notification-owner",
               "goal": "return result", "summary": "done", "status": "completed",
               "command": "echo done", "exit_code": 0, "output": "done"}]
    if event_type == "completion":
        events.append({**events[0], "session_id": "refused-process-2"})
    isolated_queue = queue.Queue()
    completed, released, submits = [], [], []
    deferred = [] if deferred_drain else None
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_background_notifications_off", lambda _session: False)
    def refuse(_rid, _sid, current, text, **_kwargs):
        submits.append(text)
        with current["history_lock"]:
            current["running"] = False  # a refusing _run_prompt_submit releases the turn itself
        return False

    monkeypatch.setattr(server, "_run_prompt_submit", refuse)
    monkeypatch.setattr(async_delegation, "claim_event_delivery", lambda _event, _consumer: "claim")
    monkeypatch.setattr(async_delegation, "complete_event_delivery",
                        lambda event, claim: completed.append((event, claim)))
    monkeypatch.setattr(async_delegation, "release_event_delivery",
                        lambda event, claim: released.append((event, claim)))
    emitted = set()
    server._notif_handle_ready("sid-refused", session, events, emitted, process_registry,
                               format_process_notification, deferred, owned=True)

    assert completed == []
    assert released == [(event, "claim") for event in events]
    assert session["running"] is False
    requeued = deferred if deferred is not None else list(isolated_queue.queue)
    assert requeued == events
    assert len(submits) == 1  # consecutive completions use one turn

    # Persistent ownership refusal must not emit an error on every queue/poller pass.
    assert server._notif_claim_turn(session) is False
    monkeypatch.setattr(server.time, "monotonic", lambda: float("inf"))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_args, **_kwargs: True)
    retry = list(requeued)
    if deferred is not None:
        deferred.clear()
    else:
        while not isolated_queue.empty():
            isolated_queue.get_nowait()
    server._notif_handle_ready("sid-refused", session, retry, emitted, process_registry,
                               format_process_notification, deferred, owned=True)
    assert completed == [(event, "claim") for event in events]
    assert (deferred if deferred is not None else list(isolated_queue.queue)) == []


@pytest.mark.parametrize("unavailable", ["agent", "closing"])
def test_notification_dispatch_guard_keeps_items_pending(monkeypatch, unavailable):
    """No notification submit or error frame while the owner cannot run a turn."""
    import queue
    from tools.process_registry import process_registry
    from tools.process_registry_notifications import format_process_notification

    session = _session(_kanban_pending=["kanban waiting"])
    session["agent" if unavailable == "agent" else "_closing"] = None if unavailable == "agent" else True
    events = [{"type": kind, "session_id": f"guard-{kind}", "delegation_id": "guard-delegation",
               "summary": "done", "status": "completed", "command": "echo done", "exit_code": 0}
              for kind in ("async_delegation", "completion")]
    isolated_queue = queue.Queue()
    frames = []
    submit = Mock(return_value=False)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    monkeypatch.setattr(server, "_emit", lambda *args, **_kwargs: frames.append(args))
    monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
    monkeypatch.setattr(server, "_background_notifications_off", lambda _session: False)
    server._notif_poll_kanban("sid-guard", session)
    server._notif_handle_ready("sid-guard", session, events, set(), process_registry,
                               format_process_notification, None, owned=True)
    submit.assert_not_called()
    assert not any(frame[0] in {"error", "message.start"} for frame in frames)
    assert session["_kanban_pending"] == ["kanban waiting"]
    assert list(isolated_queue.queue) == events
    assert session["running"] is False

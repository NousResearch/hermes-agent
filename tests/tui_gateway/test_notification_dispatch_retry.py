"""Rejected notification dispatch must retain delivery work (regression for #90020)."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tui_gateway import server


def _session(**extra):
    return {"agent": SimpleNamespace(), "session_key": "session-key", "history": [],
            "history_lock": threading.RLock(), "running": False, **extra}


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

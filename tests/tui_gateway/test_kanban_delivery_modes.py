"""The TUI consumer honors the delivery policy stored on real Kanban subscriptions."""

import json
import threading

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from tui_gateway import server


@pytest.mark.parametrize("mode,ping,wake", [("notify", True, False), ("notify+wake", True, True), ("wake", False, True)])
@pytest.mark.parametrize("gate", [None, "running", "_turn_cancel_requested"])
def test_completion_obeys_subscription_mode_once(monkeypatch, mode, ping, wake, gate):
    session = {"session_key": "coordinator", "history_lock": threading.Lock(), "running": False}
    emitted, submitted = [], []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, payload)))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda rid, sid, sess, text, **kw: submitted.append(text) or True)
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="implement fix", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="tui", chat_id=session["session_key"], delivery_mode=mode)
        assert kb.complete_task(conn, tid, summary="Ready for the next step")
    if gate:
        session[gate] = True
    server._notif_poll_kanban("ui-coordinator", session)
    if gate:
        assert submitted == []
        assert bool(session.get("_kanban_pending")) == wake
        session[gate] = False
    server._notif_poll_kanban("ui-coordinator", session)
    session["running"] = False
    server._notif_poll_kanban("ui-coordinator", session)
    assert len([event for event, _ in emitted if event == "status.update"]) == int(ping)
    assert len(submitted) == int(wake)
    if wake:
        assert tid in submitted[0] and "Ready for the next step" in submitted[0]
    assert not session.get("_kanban_pending")
    with kbc.connect_closing() as conn:
        sub = kbn.list_notify_subs(conn, task_id=tid)[0]
        assert sub["delivery_mode"] == mode
        assert sub["last_event_id"] == kb.list_events(conn, tid)[-1].id


@pytest.mark.parametrize("mode,ping", [("notify", True), ("notify+wake", True), ("wake", False)])
def test_bookkeeping_and_progress_never_start_a_turn(monkeypatch, mode, ping):
    from tools import kanban_tools  # registers the worker tools
    from tools.registry import registry

    session = {"session_key": "coordinator", "history_lock": threading.Lock(), "running": False}
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, payload)))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *args, **kw: pytest.fail("bookkeeping started a turn"))
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="implement fix", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="tui", chat_id=session["session_key"], delivery_mode=mode)
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    for tool, args in [("kanban_comment", {"task_id": tid, "body": "Progress: please check the current plan"}),
                       ("kanban_heartbeat", {"task_id": tid})]:
        result = json.loads(registry.dispatch(tool, args))
        assert result.get("ok"), result
    server._notif_poll_kanban("ui-coordinator", session)
    assert emitted == []
    with kbc.connect_closing() as conn, kb.write_txn(conn):
        kb._append_event(conn, tid, "status", {"status": "running"})
    server._notif_poll_kanban("ui-coordinator", session)
    server._notif_poll_kanban("ui-coordinator", session)
    assert len([event for event, _ in emitted if event == "status.update"]) == int(ping)
    assert not session.get("_kanban_pending")

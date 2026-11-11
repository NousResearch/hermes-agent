"""Passive Kanban transitions do not create turns; actionable bursts batch with a deadline."""

import threading
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
# session_notifications bodies are rebound onto server.py's globals (method_ctx.bind_module): patch there.
from tui_gateway import server as sn

SESSION_KEY = "tui-quiet-1"


def _ev(kind, **payload):
    return SimpleNamespace(kind=kind, payload=payload)


@pytest.mark.parametrize("ev,passive", [
    (_ev("status", status="ready"), True),
    (_ev("status", status="todo"), True),
    (_ev("status", status="running"), True),
    (_ev("status", status="blocked"), False),
    (_ev("status", status="triage"), False),
    (_ev("completed", summary="x"), False),
    (_ev("blocked", reason="x"), False),
    (_ev("timed_out", limit_seconds=60), False),
])
def test_only_requeue_transitions_are_passive(ev, passive):
    assert sn._kanban_event_is_passive(ev) is passive


def _subscribed_task():
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="quiet tui", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="tui", chat_id=SESSION_KEY)
        return tid
    finally:
        conn.close()


def _event(tid, kind, payload):
    conn = kbc.connect()
    try:
        with conn:
            kb._append_event(conn, tid, kind, payload)
    finally:
        conn.close()


def test_collected_ready_transition_is_tagged_passive_and_done_is_not():
    tid = _subscribed_task()
    _event(tid, "status", {"status": "ready", "reason": "reopened_done", "previous_status": "done"})
    _event(tid, "completed", {"summary": "closed #10"})
    texts = sn._collect_kanban_notifications({"session_key": SESSION_KEY})
    assert len(texts) == 2
    ready, done = texts
    assert isinstance(ready, sn._PassiveKanbanText) and "→ ready" in ready
    assert not isinstance(done, sn._PassiveKanbanText) and "done" in done


@pytest.fixture
def poller(monkeypatch):
    """Drive _notif_poll_kanban_scoped with canned texts; capture submitted turns."""
    feed, turns, status_lines = [], [], []
    clock = {"now": 1000.0}
    settings = {"passive_status": True, "batch_window_seconds": 10.0, "batch_max_wait_seconds": 60.0}
    monkeypatch.setattr(sn, "_collect_kanban_notifications", lambda session: feed.pop(0) if feed else [])
    monkeypatch.setattr(sn, "_kanban_turn_settings", lambda session: dict(settings))
    monkeypatch.setattr(sn, "_notif_claim_turn", lambda session: True)
    monkeypatch.setattr(sn, "_notif_submit", lambda rid, sid, session, text, *a, **k: turns.append(text))
    monkeypatch.setattr(sn, "_emit", lambda *a, **k: status_lines.append(a))
    monkeypatch.setattr(sn.time, "time", lambda: clock["now"])
    session = {"history_lock": threading.RLock(), "session_key": SESSION_KEY}

    def poll(texts=None, advance=5.0):
        if texts is not None:
            feed.append(texts)
        sn._notif_poll_kanban_scoped("sid", session)
        clock["now"] += advance

    return SimpleNamespace(poll=poll, turns=turns, status=status_lines, settings=settings, session=session)


READY = sn._PassiveKanbanText("🔄 @coding Kanban t_a11feac1 → ready")
DONE = "✔ @coding Kanban t_a11feac1 done — hermes-skills #10"
BLOCKED = "⏸ @default Kanban t_21905319 blocked: needs the operator's decision"


def test_ready_transition_produces_no_parent_turn(poller):
    for _ in range(6):
        poller.poll([READY] if not poller.status else [])
    assert poller.turns == []
    assert poller.status  # still visible on the status line


def test_done_still_wakes_and_carries_the_folded_ready_line(poller):
    poller.poll([READY])
    poller.poll([DONE])
    for _ in range(3):
        poller.poll([])
    assert len(poller.turns) == 1
    assert poller.turns[0].splitlines() == [str(READY), DONE]


def test_blocked_still_wakes(poller):
    poller.poll([BLOCKED])
    poller.poll([], advance=10.0)
    poller.poll([])
    assert poller.turns == [BLOCKED]


def test_burst_is_batched_into_one_turn(poller):
    poller.poll([DONE], advance=3.0)
    poller.poll([BLOCKED], advance=3.0)
    poller.poll([], advance=11.0)
    poller.poll([])
    assert len(poller.turns) == 1 and DONE in poller.turns[0] and BLOCKED in poller.turns[0]


def test_a_steady_trickle_is_not_held_past_max_wait(poller):
    for i in range(14):  # a new event every 5s never goes quiet for 10s
        poller.poll([f"✔ Kanban t_{i:08x} done"])
    assert len(poller.turns) >= 1  # flushed by batch_max_wait_seconds (60s)


def test_flag_off_restores_old_behaviour(poller):
    poller.settings["passive_status"] = False
    poller.settings["batch_window_seconds"] = 0.0
    poller.poll([READY])
    assert poller.turns == [str(READY)]

"""TUI/Desktop attention events must survive triage and review transitions.

Exercise the real DB, cursor claim and notification tick without a model call.
"""

import threading

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from tui_gateway import server


@pytest.fixture
def attention_board():
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="attention handoff", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="tui", chat_id="owner-key",
                           notifier_profile="coordinator")
        assert kb.claim_task(conn, tid) is not None
        yield conn, tid


def _cursor(conn, tid):
    return kbn.list_notify_subs(conn, task_id=tid)[0]["last_event_id"]


def _attention_event(conn, tid, kind):
    if kind == "block_loop_detected":
        assert kb.block_task(conn, tid, reason="owner decision pending", kind="needs_input")
        assert kb.unblock_task(conn, tid)
        assert kb.claim_task(conn, tid) is not None
        assert kb.block_task(conn, tid, reason="owner decision still pending", kind="needs_input")
    else:
        assert kb.request_review(conn, tid, summary="verified implementation", reviewer="reviewer")
        if kind == "changes_requested":
            assert kb.claim_review_task(conn, tid) is not None
            ok, reason = kb.request_changes(conn, tid, reason="fix the missing case")
            assert ok, reason
    events = [ev for ev in kb.list_events(conn, tid) if ev.kind == kind]
    assert len(events) == 1
    return events[0]


@pytest.mark.parametrize("kind,expected", [
    ("block_loop_detected", "triage"),
    ("review_requested", "review requested"),
    ("changes_requested", "changes requested"),
])
def test_attention_reaches_only_owner_once_with_cursor(attention_board, kind, expected):
    conn, tid = attention_board
    before_cursor = _cursor(conn, tid)
    event = _attention_event(conn, tid, kind)
    state = kb.get_task(conn, tid)
    assert state is not None
    assert server._collect_kanban_notifications({"session_key": "foreign-profile-key"}) == []
    assert _cursor(conn, tid) == before_cursor
    assert server._collect_kanban_notifications({"session_key": "owner-key", "_finalized": True}) == []
    assert _cursor(conn, tid) == before_cursor

    texts = server._collect_kanban_notifications({"session_key": "owner-key"})
    selected = [text for text in texts if expected in text]
    assert len(selected) == 1
    assert tid in selected[0]
    assert _cursor(conn, tid) >= event.id
    assert server._collect_kanban_notifications({"session_key": "owner-key"}) == []
    assert kb.get_task(conn, tid) == state
    assert len(kbn.list_notify_subs(conn, task_id=tid)) == 1
    if kind == "block_loop_detected":
        assert state.status == "triage"
        assert state.current_run_id is None
        assert state.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT
        assert "no automatic retry" in selected[0]
    elif kind == "review_requested":
        assert "verified implementation" in selected[0]
    else:
        assert "fix the missing case" in selected[0]


@pytest.mark.parametrize("kind", ["block_loop_detected", "review_requested", "changes_requested"])
@pytest.mark.parametrize("held", ["busy", "stopped"])
def test_attention_buffers_then_runs_one_turn_without_restarting_task(
    attention_board, monkeypatch, kind, held,
):
    conn, tid = attention_board
    event = _attention_event(conn, tid, kind)
    state = kb.get_task(conn, tid)
    emitted, submitted = [], []
    session = {"session_key": "owner-key", "history_lock": threading.RLock(),
               "running": held == "busy", "_turn_cancel_requested": held == "stopped"}
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    def run_prompt(rid, sid, sess, text, **kwargs):
        submitted.append((sid, text, kwargs))
        assert sess["running"]
        sess["running"] = False
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt)
    server._notif_poll_kanban("owner-ui", session)
    pending = list(session["_kanban_pending"])
    assert pending
    assert submitted == []
    assert _cursor(conn, tid) >= event.id
    assert any(args[0] == "status.update" for args in emitted)
    server._notif_poll_kanban("owner-ui", session)
    assert session["_kanban_pending"] == pending
    assert submitted == []
    session["running"] = False
    session["_turn_cancel_requested"] = False
    server._notif_poll_kanban("owner-ui", session)
    assert len(submitted) == 1
    assert submitted[0][0] == "owner-ui"
    assert submitted[0][1] == "\n".join(pending)
    assert session["_kanban_pending"] == []
    for _ in range(3):
        server._notif_poll_kanban("owner-ui", session)
    assert len(submitted) == 1
    assert kb.get_task(conn, tid) == state


def test_silent_bookkeeping_advances_cursor_without_wake(attention_board):
    conn, tid = attention_board
    assert kb.block_task(conn, tid, reason="wait", kind="needs_input")
    assert len(server._collect_kanban_notifications({"session_key": "owner-key"})) == 1
    assert kb.unblock_task(conn, tid)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "commented", {"body": "progress only"})
        kb._append_event(conn, tid, "heartbeat", {})
    assert server._collect_kanban_notifications({"session_key": "owner-key"}) == []
    unblock = [ev for ev in kb.list_events(conn, tid) if ev.kind == "unblocked"][-1]
    assert _cursor(conn, tid) >= unblock.id
    assert kb.archive_task(conn, tid)
    assert server._collect_kanban_notifications({"session_key": "owner-key"}) == []
    assert kbn.list_notify_subs(conn, task_id=tid) == []


@pytest.mark.parametrize("block_kind,is_diagnostic", [("needs_input", False), ("capability", True)])
def test_triage_preserves_attention_category_and_foreign_subscription(
    attention_board, block_kind, is_diagnostic,
):
    from gateway.warning_notifications import DiagnosticText

    conn, tid = attention_board
    kbn.add_notify_sub(conn, task_id=tid, platform="tui", chat_id="other-profile-key",
                       notifier_profile="other-profile")
    foreign_before = next(s for s in kbn.list_notify_subs(conn, task_id=tid)
                          if s["chat_id"] == "other-profile-key")["last_event_id"]
    assert kb.block_task(conn, tid, reason="unresolved prerequisite", kind=block_kind)
    assert kb.unblock_task(conn, tid)
    assert kb.claim_task(conn, tid) is not None
    assert kb.block_task(conn, tid, reason="still unresolved", kind=block_kind)
    texts = server._collect_kanban_notifications({"session_key": "owner-key"})
    triage = [text for text in texts if "triage" in text]
    assert len(triage) == 1
    assert isinstance(triage[0], DiagnosticText) is is_diagnostic
    foreign_after = next(s for s in kbn.list_notify_subs(conn, task_id=tid)
                         if s["chat_id"] == "other-profile-key")["last_event_id"]
    assert foreign_after == foreign_before
    assert server._collect_kanban_notifications({"session_key": "owner-key"}) == []
    foreign = server._collect_kanban_notifications({"session_key": "other-profile-key"})
    assert len([text for text in foreign if "triage" in text]) == 1
    assert server._collect_kanban_notifications({"session_key": "other-profile-key"}) == []

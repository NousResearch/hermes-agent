"""Cursor-claimed kanban notifications remain buffered until a turn starts."""

import pytest

from tui_gateway import server
from tests.tui_gateway.test_kanban_notify_poller import _create_subscribed_task, _complete, _sub_rows
from tests.tui_gateway.test_turn_claim_interrupt import _idle_session, turn_env


@pytest.mark.parametrize("refusal", ["lease", "exception", "stop"])
def test_unstarted_kanban_batch_retries_without_reclaiming_events(monkeypatch, turn_env, refusal):
    session = _idle_session()
    sid = "kanban-delivery"
    task_id = _create_subscribed_task(chat_id=session["session_key"])
    _complete(task_id, summary="ready for review")
    collected, ran = [], []
    collect = server._collect_kanban_notifications

    def collect_once(record):
        texts = collect(record)
        collected.extend(texts)
        return texts

    def refuse(_sid, record):
        if refusal == "exception":
            raise RuntimeError("lease store unavailable")
        if refusal == "stop":
            server._interrupt_session_turn(sid, record)
            return None
        return "session slot unavailable"

    monkeypatch.setattr(server, "_collect_kanban_notifications", collect_once)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_ensure_active_session_slot", refuse)
    monkeypatch.setattr(server, "_prepare_turn_input", lambda _sid, _session, _st, text, _images: ran.append(text))
    try:
        server._notif_poll_kanban_scoped(sid, session)
        pending_after_refusal = list(session.get("_kanban_pending", []))
        cursor_after_refusal = _sub_rows(task_id)[0]["last_event_id"]
        with session["history_lock"]:
            session["_turn_cancel_requested"] = False
        monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
        server._notif_poll_kanban_scoped(sid, session)
        turn_env.join_all()
        outcome = {
            "pending_after_refusal": pending_after_refusal,
            "pending_after_retry": session.get("_kanban_pending", []),
            "ran": ran, "running": session["running"], "replayed": collect(session),
            "cursor_unchanged": _sub_rows(task_id)[0]["last_event_id"] == cursor_after_refusal,
        }
    finally:
        turn_env.finish.set()
        turn_env.join_all()
    assert outcome == {
        "pending_after_refusal": collected, "pending_after_retry": [],
        "ran": ["\n".join(collected)], "running": False, "replayed": [], "cursor_unchanged": True,
    }


def test_started_kanban_batch_consumes_only_its_selected_buffer_entries(tmp_path, monkeypatch):
    from gateway.warning_notifications import DiagnosticText

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    owner = tmp_path / "profiles" / "owner"
    owner.mkdir(parents=True)
    (owner / "config.yaml").write_text("display: {suppress_warning_notifications: true}")
    session = _idle_session()
    session.update(profile_home=str(owner), _kanban_pending=[DiagnosticText("same"), "same"])
    buffered_during_start = []

    def start(*_args, **_kwargs):
        buffered_during_start.extend(session["_kanban_pending"])
        session["_kanban_pending"].append("later")
        return True

    monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    monkeypatch.setattr(server, "_notif_submit", start)
    server._notif_poll_kanban("kanban-delivery", session)
    assert {
        "during_start": [(text, isinstance(text, DiagnosticText)) for text in buffered_during_start],
        "pending": [(text, isinstance(text, DiagnosticText)) for text in session["_kanban_pending"]],
    } == {
        "during_start": [("same", True), ("same", False)],
        "pending": [("same", False), ("later", False)],
    }

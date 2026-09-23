"""The TUI/desktop kanban notice must not promise a retry the live board contradicts.

Twin of ``tests/gateway/test_kanban_timeout_notice_stale.py``: the messaging
renderer and this one read the same ``timed_out`` event, so fixing only one just
moves the false "will retry" alarm to the other surface. Both derive the claim
from the LIVE task row via ``gateway.kanban_watchers_notifier.retry_notice_claim``.
"""

from tui_gateway.session_notifications import _format_kanban_event_text, _kb_timed_out
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


class _Ev:
    def __init__(self, kind, payload=None):
        self.kind = kind
        self.payload = payload or {}


def _board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "tui-stale-timeout.db"))
    kb.init_db()
    conn = kbc.connect()
    tid = kb.create_task(conn, title="temp-ssh drop lab", assignee="hardware-manager")
    return conn, tid


def _render(conn, tid, ev):
    sub = {"task_id": tid, "platform": "tui", "chat_id": "sess-1"}
    return _format_kanban_event_text(sub, kb.get_task(conn, tid), ev, "default")


def _timeout_event(conn, tid, *, limit_seconds=0):
    kb._append_event(
        conn, tid, "timed_out",
        {"pid": 4242, "elapsed_seconds": limit_seconds + 1, "limit_seconds": limit_seconds,
         "sigkill": False, "retry_status": "ready"},
    )


def test_finished_card_gets_no_timeout_notice(tmp_path, monkeypatch):
    """The reported instance: the card is done, so there is nothing to say."""
    conn, tid = _board(tmp_path, monkeypatch)
    try:
        _timeout_event(conn, tid)
        assert kb.complete_task(conn, tid, summary="lab dropped") is True
        task = kb.get_task(conn, tid)
        assert _kb_timed_out(task, {"limit_seconds": 0}, "lab") is None
        assert _render(conn, tid, _Ev("timed_out", {"limit_seconds": 0})) is None
    finally:
        conn.close()


def test_queued_retry_still_says_will_retry(tmp_path, monkeypatch):
    """Companion case: the retry is really queued, so the claim stands unchanged."""
    conn, tid = _board(tmp_path, monkeypatch)
    try:
        _timeout_event(conn, tid, limit_seconds=1800)
        text = _render(conn, tid, _Ev("timed_out", {"limit_seconds": 1800}))
        assert text is not None
        assert text.endswith("Kanban %s timed out (max_runtime=1800s); will retry" % tid), text
    finally:
        conn.close()


def test_no_retry_queued_reports_the_live_status(tmp_path, monkeypatch):
    """A blocked card claims nothing and names where it actually stands."""
    conn, tid = _board(tmp_path, monkeypatch)
    try:
        _timeout_event(conn, tid)
        assert kb.block_task(conn, tid, reason="worker died for good") is True
        text = _render(conn, tid, _Ev("timed_out", {"limit_seconds": 0}))
        assert text is not None
        assert "will retry" not in text, text
        assert "no retry is queued" in text and "blocked" in text, text
    finally:
        conn.close()


def test_other_kinds_are_untouched(tmp_path, monkeypatch):
    """The live check is scoped to the two kinds that predict a retry."""
    conn, tid = _board(tmp_path, monkeypatch)
    try:
        assert kb.complete_task(conn, tid, summary="lab dropped") is True
        text = _render(conn, tid, _Ev("completed", {"summary": "lab dropped"}))
        assert text is not None and "done" in text, text
    finally:
        conn.close()

"""Tests for the TUI-side kanban notification poller (issue #59890).

``kanban_create`` auto-subscribes TUI/desktop sessions with
``platform="tui"`` / ``chat_id=HERMES_SESSION_KEY``, but no component ever
read those rows back: the gateway notifier skips them (no "tui" messaging
adapter) and the TUI notification poller only watched process completions.
``last_event_id`` stayed 0 forever and no notification was ever delivered.

These tests cover the delivery half that now lives in tui_gateway/server.py:
``_collect_kanban_notifications`` (cursor claim + formatting + terminal
unsubscribe) and ``_format_kanban_event_text``.
"""

from types import SimpleNamespace

from hermes_cli import kanban_db as kb
from tui_gateway.server import (
    _collect_kanban_notifications,
    _format_kanban_event_text,
)

SESSION_KEY = "tui-session-key-1"


def _session(key: str = SESSION_KEY) -> dict:
    return {"session_key": key}


def _create_subscribed_task(*, chat_id: str = SESSION_KEY, platform: str = "tui"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify tui", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform=platform, chat_id=chat_id)
        return tid
    finally:
        conn.close()


def _complete(tid: str, summary: str = "all done") -> None:
    conn = kb.connect()
    try:
        kb.complete_task(conn, tid, summary=summary)
    finally:
        conn.close()


def _sub_rows(tid: str) -> list:
    conn = kb.connect()
    try:
        return kb.list_notify_subs(conn, task_id=tid)
    finally:
        conn.close()


class TestCollectKanbanNotifications:
    def test_delivers_completed_event_and_unsubscribes(self):
        tid = _create_subscribed_task()
        _complete(tid, summary="shipped the fix")

        texts = _collect_kanban_notifications(_session())

        assert len(texts) == 1
        assert tid in texts[0]
        assert "done" in texts[0]
        assert "shipped the fix" in texts[0]
        # Task is at a final status -> subscription removed.
        assert _sub_rows(tid) == []

    def test_claim_advances_cursor_so_second_poll_is_empty(self):
        tid = _create_subscribed_task()
        conn = kb.connect()
        try:
            kb.block_task(conn, tid, reason="waiting on review")
        finally:
            conn.close()

        first = _collect_kanban_notifications(_session())
        second = _collect_kanban_notifications(_session())

        assert len(first) == 1
        assert "blocked" in first[0]
        assert "waiting on review" in first[0]
        assert second == []
        # Blocked is not a final status -> subscription stays alive so a
        # respawned task's next terminal event still reaches the user.
        assert len(_sub_rows(tid)) == 1

    def test_ignores_other_sessions_and_platforms(self):
        tid_other_session = _create_subscribed_task(chat_id="some-other-session")
        tid_gateway = _create_subscribed_task(platform="telegram", chat_id="chat-1")
        _complete(tid_other_session)
        _complete(tid_gateway)

        texts = _collect_kanban_notifications(_session())

        assert texts == []
        # Foreign subscriptions untouched: cursors still 0, rows still there.
        for tid in (tid_other_session, tid_gateway):
            rows = _sub_rows(tid)
            assert len(rows) == 1
            assert rows[0]["last_event_id"] == 0

    def test_no_session_key_is_a_noop(self):
        tid = _create_subscribed_task()
        _complete(tid)

        assert _collect_kanban_notifications({"session_key": ""}) == []
        assert _collect_kanban_notifications({"session_key": None}) == []
        assert len(_sub_rows(tid)) == 1


class TestFormatKanbanEventText:
    SUB = {"task_id": "t_abc123"}
    TASK = SimpleNamespace(title="build the thing", assignee="worker", result=None)

    def test_silent_kinds_return_none(self):
        for kind in ("archived", "unblocked"):
            ev = SimpleNamespace(kind=kind, payload={})
            assert _format_kanban_event_text(self.SUB, self.TASK, ev, "main") is None

    def test_blocked_includes_reason(self):
        ev = SimpleNamespace(kind="blocked", payload={"reason": "needs creds"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "main")
        assert "t_abc123" in text
        assert "blocked" in text
        assert "needs creds" in text
        assert "[main]" in text
        assert "@worker" in text

    def test_completed_prefers_payload_summary(self):
        ev = SimpleNamespace(kind="completed", payload={"summary": "first line\nsecond"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "")
        assert "done" in text
        assert "first line" in text
        assert "second" not in text

    def test_timed_out_with_bad_payload_does_not_raise(self):
        ev = SimpleNamespace(kind="timed_out", payload={"limit_seconds": "not-a-number"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "")
        assert "timed out" in text


class TestNotificationPollerLoopKanbanWiring:
    """Drive a real TUI subscription through ``_notification_poller_loop``.

    Covers the wiring above ``_collect_kanban_notifications``: status.update
    emission, agent-turn dispatch when the session is idle, and the
    busy-session pending buffer that flushes once the session goes idle.
    """

    def _start_poller(self, session: dict, monkeypatch):
        import threading
        import tui_gateway.server as server

        emits: list = []
        submits: list = []
        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(
            server, "_emit", lambda event, sid, payload=None: emits.append((event, payload))
        )
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda rid, sid, sess, text: submits.append(text),
        )
        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        thread.start()
        return stop, thread, emits, submits

    @staticmethod
    def _wait_for(predicate, timeout: float = 5.0) -> bool:
        import time as _time

        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if predicate():
                return True
            _time.sleep(0.02)
        return False

    def _poller_session(self, *, running: bool = False) -> dict:
        import threading

        return {
            "session_key": SESSION_KEY,
            "history_lock": threading.Lock(),
            "running": running,
        }

    def test_idle_session_gets_status_update_and_agent_turn(self, monkeypatch):
        tid = _create_subscribed_task()
        _complete(tid, summary="poller e2e done")
        session = self._poller_session(running=False)

        stop, thread, emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert self._wait_for(lambda: submits), "agent turn was never dispatched"
        finally:
            stop.set()
            thread.join(timeout=5)

        status_texts = [p["text"] for e, p in emits if e == "status.update" and p]
        assert any(tid in t for t in status_texts), status_texts
        assert any(e == "message.start" for e, _ in emits)
        assert any(tid in text for text in submits), submits
        assert session["running"] is True  # poller claimed the turn
        assert not session.get("_kanban_pending")

    def test_busy_session_buffers_then_flushes_when_idle(self, monkeypatch):
        tid = _create_subscribed_task()
        _complete(tid, summary="buffered while busy")
        session = self._poller_session(running=True)

        stop, thread, emits, submits = self._start_poller(session, monkeypatch)
        try:
            # Busy: the status line appears and the event is buffered, but no
            # agent turn is dispatched while another turn is running.
            assert self._wait_for(
                lambda: any(e == "status.update" for e, _ in emits)
                and session.get("_kanban_pending")
            )
            assert not submits

            with session["history_lock"]:
                session["running"] = False

            assert self._wait_for(lambda: submits), "pending batch never flushed"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert any(tid in text for text in submits), submits
        assert session["_kanban_pending"] == []
        assert session["running"] is True

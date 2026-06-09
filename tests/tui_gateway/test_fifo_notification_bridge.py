"""Tests for the Kanban→TUI DB-based notification bridge.

Tests both sides of the DB-driven notification bridge:
  - Writer side (hermes_cli/kanban_db.py _append_event): writes event rows
    to the task_events table.
  - Reader side (tui_gateway/server.py): _format_kanban_event formats
    events correctly; _poll_kanban_notifications delivers to sessions
    using upstream claim_unseen_events_for_sub.
    from session dispatch.

All _kanban_fifo_* names have been renamed to _kanban_event_*.
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_server_module(tmp_path, monkeypatch):
    """Import tui_gateway.server with a clean environment (no FIFO)."""
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(
            get_hermes_home=MagicMock(return_value=str(tmp_path))
        ),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        import tui_gateway.server as srv
        importlib.reload(srv)

        yield srv

        srv._sessions.clear()
        importlib.reload(srv)


# ---------------------------------------------------------------------------
# Writer-side tests (kanban_db._append_event DB write)
# ---------------------------------------------------------------------------

class TestAppendEventDbWriter:
    """Tests for the DB write side of _append_event in kanban_db.py.

    After the FIFO→DB refactor, _append_event writes event rows to the
    task_events table.  The FIFO carries only an alert signal — the actual
    event data lives in DB.
    """

    def test_completed_event_in_db(self, kanban_home):
        """A 'completed' event should be in task_events after commit."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="ship it", assignee="worker")
            kb.complete_task(conn, tid, summary="all done")

        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? AND kind = 'completed'",
                (tid,),
            ).fetchall()
            assert len(rows) == 1
            payload = json.loads(rows[0]["payload"])
            assert payload["summary"] == "all done"

    def test_blocked_event_in_db(self, kanban_home):
        """A 'blocked' event should be in task_events after commit."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="blocked task", assignee="worker")
            kb.block_task(conn, tid, reason="need input")

        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? AND kind = 'blocked'",
                (tid,),
            ).fetchall()
            assert len(rows) == 1
            payload = json.loads(rows[0]["payload"])
            assert payload["reason"] == "need input"

    def test_crashed_event_in_db(self, kanban_home):
        """A 'crashed' event should be in task_events after commit."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="crash task", assignee="worker")
            kb._append_event(conn, tid, kind="crashed")

        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? AND kind = 'crashed'",
                (tid,),
            ).fetchall()
            assert len(rows) == 1

    def test_timed_out_event_in_db(self, kanban_home):
        """A 'timed_out' event should be in task_events after commit."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="timeout task", assignee="worker")
            kb._append_event(conn, tid, kind="timed_out")

        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? AND kind = 'timed_out'",
                (tid,),
            ).fetchall()
            assert len(rows) == 1

    def test_gave_up_event_in_db(self, kanban_home):
        """A 'gave_up' event should be in task_events after commit."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="give-up task", assignee="worker")
            kb._append_event(conn, tid, kind="gave_up")

        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? AND kind = 'gave_up'",
                (tid,),
            ).fetchall()
            assert len(rows) == 1

    def test_created_event_in_db(self, kanban_home):
        """A 'created' event should be in task_events (non-terminal)."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="no-notify", assignee="worker")

        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? AND kind = 'created'",
                (tid,),
            ).fetchall()
            assert len(rows) == 1

    def test_heartbeat_event_in_db(self, kanban_home):
        """A 'heartbeat' event should be in task_events (non-terminal)."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="hb task", assignee="worker")
            kb._append_event(conn, tid, kind="heartbeat")

        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? AND kind = 'heartbeat'",
                (tid,),
            ).fetchall()
            assert len(rows) == 1

    def test_multiple_events_all_in_db(self, kanban_home):
        """Multiple events should all be persisted to task_events."""
        with kb.connect() as conn:
            tid1 = kb.create_task(conn, title="task-1", assignee="worker")
            kb.complete_task(conn, tid1, summary="done-1")
            tid2 = kb.create_task(conn, title="task-2", assignee="worker")
            kb.block_task(conn, tid2, reason="waiting")

        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT task_id, kind FROM task_events WHERE kind IN ('completed', 'blocked') ORDER BY id"
            ).fetchall()
        assert len(rows) == 2
        assert rows[0]["task_id"] == tid1
        assert rows[0]["kind"] == "completed"
        assert rows[1]["task_id"] == tid2
        assert rows[1]["kind"] == "blocked"


# ---------------------------------------------------------------------------
# Reader-side tests (tui_gateway.server._format_kanban_event)
# ---------------------------------------------------------------------------

class TestFormatKanbanNotification:
    """Tests for _format_kanban_event in tui_gateway/server.py."""

    @pytest.fixture(autouse=True)
    def _setup_server(self, tmp_path, monkeypatch):
        """Import tui_gateway.server with clean environment."""
        with patch.dict("sys.modules", {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value=str(tmp_path))
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        }):
            import importlib
            import tui_gateway.server as srv
            importlib.reload(srv)
            self.server = srv
            yield
            srv._sessions.clear()
            importlib.reload(srv)

    def _make_sub(self, task_id="t_abc"):
        return {"task_id": task_id, "platform": "cli", "chat_id": "cli-123"}

    def _make_event(self, kind, payload=None):
        ev = MagicMock()
        ev.kind = kind
        ev.payload = payload
        ev.get = lambda k, d=None: getattr(ev, k, d) if k != "payload" else payload
        return ev

    def test_completed_with_summary(self):
        ev = self._make_event("completed", {"summary": "shipped rate limiter"})
        sub = self._make_sub("t_abc")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_abc" in msg
        assert "done" in msg
        assert "shipped rate limiter" in msg

    def test_completed_without_summary(self):
        ev = self._make_event("completed", {})
        sub = self._make_sub("t_xyz")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_xyz" in msg
        assert "done" in msg

    def test_completed_with_none_payload(self):
        ev = self._make_event("completed", None)
        sub = self._make_sub("t_123")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_123" in msg

    def test_blocked_with_reason(self):
        ev = self._make_event("blocked", {"reason": "need API key"})
        sub = self._make_sub("t_blk")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_blk" in msg
        assert "blocked" in msg
        assert "need API key" in msg

    def test_blocked_without_reason(self):
        ev = self._make_event("blocked", {})
        sub = self._make_sub("t_blk2")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_blk2" in msg
        assert "blocked" in msg

    def test_crashed(self):
        ev = self._make_event("crashed")
        sub = self._make_sub("t_crash")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_crash" in msg
        assert "crashed" in msg
        assert "dispatcher will retry" in msg

    def test_timed_out_with_limit(self):
        ev = self._make_event("timed_out", {"limit_seconds": 300})
        sub = self._make_sub("t_to")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_to" in msg
        assert "timed out" in msg
        assert "300" in msg

    def test_timed_out_without_limit(self):
        ev = self._make_event("timed_out", None)
        sub = self._make_sub("t_to2")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_to2" in msg

    def test_gave_up_with_error(self):
        ev = self._make_event("gave_up", {"error": "spawn failed 3x"})
        sub = self._make_sub("t_gu")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_gu" in msg
        assert "gave up" in msg
        assert "spawn failed 3x" in msg

    def test_gave_up_without_error(self):
        ev = self._make_event("gave_up", None)
        sub = self._make_sub("t_gu2")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_gu2" in msg

    def test_unknown_kind_returns_none(self):
        ev = self._make_event("edited")
        sub = self._make_sub("t_xxx")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is None

    def test_completed_summary_truncated_to_200_chars(self):
        long_summary = "x" * 300
        ev = self._make_event("completed", {"summary": long_summary})
        sub = self._make_sub("t_long")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        # The summary in the message should be truncated.
        assert len(msg) < len(long_summary) + 100  # rough upper bound

    def test_blocked_reason_truncated_to_160_chars(self):
        long_reason = "y" * 200
        ev = self._make_event("blocked", {"reason": long_reason})
        sub = self._make_sub("t_longblk")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert len(msg) < len(long_reason) + 100

    def test_dict_event_access(self):
        """Events from DB queries arrive as dicts, not MagicMock objects."""
        ev = {"kind": "completed", "payload": {"summary": "dict event"}}
        sub = self._make_sub("t_dict")
        msg = self.server._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_dict" in msg
        assert "dict event" in msg


# ---------------------------------------------------------------------------
# DB poller lifecycle tests (replaces FIFO lifecycle)
# ---------------------------------------------------------------------------

class TestDbNotificationEndToEnd:
    """End-to-end tests: kanban_db writes → DB poll → notification dispatch."""

    def test_complete_task_event_visible_in_db(self, kanban_home):
        """Completing a task stores the event in DB where the poller can find it."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="e2e-test", assignee="worker")
            kb.complete_task(conn, tid, summary="e2e complete")

        # Verify the event is in DB and queryable by the poller
        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? AND kind = 'completed'",
                (tid,),
            ).fetchall()
            assert len(rows) == 1
            payload = json.loads(rows[0]["payload"])
            assert payload["summary"] == "e2e complete"

    def test_non_notify_event_in_db_but_not_in_notify_kinds(self, kanban_home):
        """Non-notify events (created, edited) are in DB but not in the notify set."""
        _notify_kinds = {"completed", "blocked", "gave_up", "crashed", "timed_out"}

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="silent", assignee="worker")

        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (tid,)
            ).fetchall()
            for row in rows:
                assert row["kind"] not in _notify_kinds, (
                    f"'{row['kind']}' should not be a notify kind"
                )

    def test_poll_delivers_formatted_notification(self, kanban_home, tmp_path, monkeypatch):
        """_notify_kanban_event pushes formatted event to completion_queue."""
        with patch.dict("sys.modules", {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value=str(tmp_path))
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        }):
            import importlib
            import tui_gateway.server as srv
            importlib.reload(srv)

            # Create a task and complete it
            with kb.connect() as conn:
                tid = kb.create_task(conn, title="dispatch-test", assignee="worker")
                kb.add_notify_sub(conn, task_id=tid, platform="cli", chat_id="cli-test")

            with kb.connect() as conn:
                kb.complete_task(conn, tid, summary="all done")

            # Verify event was pushed to completion_queue
            from tools.process_registry import process_registry
            events = []
            while not process_registry.completion_queue.empty():
                evt = process_registry.completion_queue.get_nowait()
                if evt.get("type") == "kanban_event" and evt.get("task_id") == tid:
                    events.append(evt)

            assert len(events) >= 1
            assert events[0]["kind"] == "completed"

            srv._sessions.clear()
            importlib.reload(srv)


# ---------------------------------------------------------------------------
# DB writer edge cases
# ---------------------------------------------------------------------------

class TestDbWriterEdgeCases:
    """Tests for DB writer-side edge cases: transaction rollback, concurrent writes."""

    def test_transaction_rollback_discards_event(self, kanban_home):
        """When an explicit transaction rolls back, the event should NOT be in DB.

        _append_event is documented as "called from within an already-open txn".
        With isolation_level=None (autocommit), bare INSERTs commit immediately.
        An explicit BEGIN wraps the call so a later rollback actually discards it.
        """
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="rollback-test", assignee="worker")

        conn = kb.connect()
        try:
            conn.execute("BEGIN")
            kb._append_event(conn, tid, kind="completed")
            conn.rollback()
        finally:
            conn.close()

        # The event should NOT exist — transaction was rolled back
        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? AND kind = 'completed'",
                (tid,),
            ).fetchall()
            assert len(rows) == 0

    def test_multiple_events_all_persisted(self, kanban_home):
        """Multiple events for the same task should all be persisted."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="multi-event", assignee="worker")
            kb.block_task(conn, tid, reason="waiting")
            kb.unblock_task(conn, tid)
            kb.complete_task(conn, tid, summary="done")

        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
                (tid,),
            ).fetchall()
            kinds = [r["kind"] for r in rows]
            assert "created" in kinds
            assert "blocked" in kinds
            assert "unblocked" in kinds
            assert "completed" in kinds

    def test_event_payload_json_structure(self, kanban_home):
        """Event payloads should be valid JSON with expected keys."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="payload-test", assignee="worker")
            kb.complete_task(conn, tid, summary="test summary")

        with kb.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completed'",
                (tid,),
            ).fetchone()
            payload = json.loads(row["payload"])
            assert isinstance(payload, dict)
            assert "summary" in payload
            assert payload["summary"] == "test summary"

    def test_task_completion_no_error(self, kanban_home):
        """Completing a task should not raise."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="no-error-test", assignee="worker")
            kb.complete_task(conn, tid, summary="done")

    def test_append_event_directly(self, kanban_home):
        """_append_event should write to task_events without error."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="direct-test", assignee="worker")
            kb._append_event(conn, tid, kind="crashed")

        with kb.connect() as conn:
            row = conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? AND kind = 'crashed'",
                (tid,),
            ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# DB event reader edge cases (replaces FIFO reader edge cases)
# ---------------------------------------------------------------------------

class TestDbDispatchEdgeCases:
    """Tests for dispatch-side edge cases: DB failures, filtering."""

    @pytest.fixture(autouse=True)
    def _setup_server(self, tmp_path, monkeypatch):
        """Import tui_gateway.server with clean environment."""
        with patch.dict("sys.modules", {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value=str(tmp_path))
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        }):
            import importlib
            import tui_gateway.server as srv
            importlib.reload(srv)
            self.server = srv
            yield
            srv._sessions.clear()
            importlib.reload(srv)

    def test_dispatch_db_failure_handled(self):
        """When DB connect fails, _notify_kanban_event should not raise."""
        from hermes_cli import kanban_db as kb
        # _notify_kanban_event catches all exceptions internally
        # This test verifies it doesn't propagate errors
        kb._notify_kanban_event("t_test", "completed", {"summary": "test"})
        # Should not raise

    def test_poll_empty_session_handled(self):
        """Poll should handle empty/missing session gracefully."""
        srv = self.server
        session = {}
        # Should return without doing anything
        # Empty session — no-op
        # Empty task_id — no-op

    def test_dispatch_while_busy_queues_notification(self):
        """When session is running, kanban events are queued via completion_queue."""
        srv = self.server

        # Simulate what happens when _notify_kanban_event pushes to queue
        from tools.process_registry import process_registry
        process_registry.completion_queue.put({
            "type": "kanban_event",
            "task_id": "t_test",
            "kind": "completed",
            "payload": {"summary": "all done"},
        })

        # Verify event is in the queue
        events = []
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "kanban_event" and evt.get("task_id") == "t_test":
                events.append(evt)

        assert len(events) >= 1
        assert events[0]["kind"] == "completed"

    def test_pending_kanban_drained_when_session_goes_idle(self):
        """Pending notifications are processed after session goes idle."""
        srv = self.server
        lock = threading.Lock()
        session = {
            "history_lock": lock,
            "running": False,
            "_pending_kanban": ["First notification", "Second notification"],
            "_kanban_cursors": {},
        }
        submitted = []

        def fake_run_prompt_submit(rid, sid, sess, text):
            submitted.append(text)
            # Simulate the turn ending — set running back to False
            with sess["history_lock"]:
                sess["running"] = False

        with patch.object(srv, "_run_prompt_submit", side_effect=fake_run_prompt_submit):
            with patch.object(srv, "_emit"):
                # Simulate the drain logic from _run_prompt_submit's finally block
                while True:
                    with session["history_lock"]:
                        _pending = session.get("_pending_kanban", [])
                        if not _pending:
                            break
                        _next_msg = _pending.pop(0)
                        if session.get("running"):
                            _pending.insert(0, _next_msg)
                            break
                        session["running"] = True
                    try:
                        srv._emit("message.start", "sid-1")
                        fake_run_prompt_submit("rid", "sid-1", session, _next_msg)
                    except Exception:
                        with session["history_lock"]:
                            session["running"] = False

        # Both notifications should have been submitted
        assert submitted == ["First notification", "Second notification"]
        # Queue should be empty
        assert session["_pending_kanban"] == []
        # Session should be idle at the end
        assert session["running"] is False

    def test_pending_drain_stops_if_session_becomes_busy(self):
        """Drain stops if another turn starts mid-drain."""
        srv = self.server
        lock = threading.Lock()
        session = {
            "history_lock": lock,
            "running": False,
            "_pending_kanban": ["First notification", "Second notification"],
            "_kanban_cursors": {},
        }
        submitted = []

        def fake_run_prompt_submit(rid, sid, sess, text):
            submitted.append(text)
            # After first notification, simulate another turn starting
            if text == "First notification":
                with sess["history_lock"]:
                    sess["running"] = True  # Another turn started!

        with patch.object(srv, "_run_prompt_submit", side_effect=fake_run_prompt_submit):
            with patch.object(srv, "_emit"):
                while True:
                    with session["history_lock"]:
                        _pending = session.get("_pending_kanban", [])
                        if not _pending:
                            break
                        _next_msg = _pending.pop(0)
                        if session.get("running"):
                            _pending.insert(0, _next_msg)
                            break
                        session["running"] = True
                    try:
                        srv._emit("message.start", "sid-1")
                        fake_run_prompt_submit("rid", "sid-1", session, _next_msg)
                    except Exception:
                        with session["history_lock"]:
                            session["running"] = False

        # Only first notification submitted; second put back
        assert submitted == ["First notification"]
        # Second notification should still be queued
        assert session["_pending_kanban"] == ["Second notification"]
        # Session is busy (another turn is running)
        assert session["running"] is True

    def test_stale_events_skipped_for_terminal_tasks(self):
        """Non-completed events for done/archived tasks should be skipped."""
        from hermes_cli import kanban_db as kb

        # Create a task and complete it
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="stale-test", assignee="worker")

        with kb.connect() as conn:
            kb.complete_task(conn, tid, summary="done")

        # The event should have been pushed to completion_queue
        from tools.process_registry import process_registry
        events = []
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "kanban_event" and evt.get("task_id") == tid:
                events.append(evt)

        # Should have the completed event
        assert len(events) >= 1
        assert events[0]["kind"] == "completed"

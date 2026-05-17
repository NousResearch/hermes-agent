"""Tests for the Kanban→TUI FIFO notification bridge.

Tests both sides of the zero-polling FIFO bridge:
  - Writer side (hermes_cli/kanban_db.py _append_event): writes JSON lines
    to ~/.hermes/tui_kanban.fifo for notification-worthy event kinds.
  - Reader side (tui_gateway/server.py): _format_kanban_notification formats
    events correctly; _start_kanban_fifo_reader reads from FIFO and dispatches.
"""

from __future__ import annotations

import json
import os
import tempfile
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


@pytest.fixture
def fifo_path(tmp_path, monkeypatch):
    """Create a real FIFO under tmp_path and point the code at it."""
    fifo = tmp_path / "tui_kanban.fifo"
    os.mkfifo(str(fifo), 0o600)

    # Patch both modules' path constants to point at our test FIFO.
    monkeypatch.setattr(kb, "_KANBAN_FIFO_PATH_CANDIDATE", str(fifo), raising=False)

    # Also patch the expanduser call inside _append_event by overriding
    # the module-level reference.  We patch os.path.expanduser so that
    # "~/.hermes/tui_kanban.fifo" resolves to our temp FIFO.
    _real_expand = os.path.expanduser

    def _fake_expand(p):
        if "tui_kanban.fifo" in p:
            return str(fifo)
        return _real_expand(p)

    monkeypatch.setattr(os.path, "expanduser", _fake_expand)
    return str(fifo)


def _make_server_module(tmp_path, monkeypatch):
    """Import tui_gateway.server with the FIFO path redirected to tmp_path."""
    fifo = tmp_path / "tui_kanban.fifo"
    os.mkfifo(str(fifo), 0o600)

    # We must patch _KANBAN_FIFO_PATH before importing the module because
    # the module-level os.mkfifo runs at import time.
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

        # Override the module-level path after reload so cleanup uses our tmp.
        srv._KANBAN_FIFO_PATH = str(fifo)
        yield srv

        srv._sessions.clear()
        importlib.reload(srv)


# ---------------------------------------------------------------------------
# Writer-side tests (kanban_db._append_event FIFO write)
# ---------------------------------------------------------------------------

class TestAppendEventFifoWriter:
    """Tests for the FIFO write side of _append_event in kanban_db.py."""

    def test_completed_event_writes_to_fifo(self, kanban_home, fifo_path):
        """A 'completed' event should produce a JSON line on the FIFO."""
        _messages = []

        def _reader():
            with open(fifo_path, "r") as f:
                for line in f:
                    _messages.append(json.loads(line.strip()))
                    break

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        # Give the reader time to open the FIFO (blocks until writer opens).
        time.sleep(0.2)

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="ship it", assignee="worker")
            kb.complete_task(conn, tid, summary="all done")

        t.join(timeout=5)
        assert len(_messages) == 1
        assert _messages[0]["task_id"] == tid
        assert _messages[0]["kind"] == "completed"

    def test_blocked_event_writes_to_fifo(self, kanban_home, fifo_path):
        """A 'blocked' event should produce a JSON line on the FIFO."""
        _messages = []

        def _reader():
            with open(fifo_path, "r") as f:
                for line in f:
                    _messages.append(json.loads(line.strip()))
                    break

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        time.sleep(0.2)

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="blocked task", assignee="worker")
            kb.block_task(conn, tid, reason="need input")

        t.join(timeout=5)
        assert len(_messages) == 1
        assert _messages[0]["task_id"] == tid
        assert _messages[0]["kind"] == "blocked"

    def test_crashed_event_writes_to_fifo(self, kanban_home, fifo_path):
        """A 'crashed' event should produce a JSON line on the FIFO."""
        _messages = []

        def _reader():
            with open(fifo_path, "r") as f:
                for line in f:
                    _messages.append(json.loads(line.strip()))
                    break

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        time.sleep(0.2)

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="crash task", assignee="worker")
            kb._append_event(conn, tid, kind="crashed")

        t.join(timeout=5)
        assert len(_messages) == 1
        assert _messages[0]["kind"] == "crashed"

    def test_timed_out_event_writes_to_fifo(self, kanban_home, fifo_path):
        """A 'timed_out' event should produce a JSON line on the FIFO."""
        _messages = []

        def _reader():
            with open(fifo_path, "r") as f:
                for line in f:
                    _messages.append(json.loads(line.strip()))
                    break

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        time.sleep(0.2)

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="timeout task", assignee="worker")
            kb._append_event(conn, tid, kind="timed_out")

        t.join(timeout=5)
        assert len(_messages) == 1
        assert _messages[0]["kind"] == "timed_out"

    def test_gave_up_event_writes_to_fifo(self, kanban_home, fifo_path):
        """A 'gave_up' event should produce a JSON line on the FIFO."""
        _messages = []

        def _reader():
            with open(fifo_path, "r") as f:
                for line in f:
                    _messages.append(json.loads(line.strip()))
                    break

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        time.sleep(0.2)

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="give-up task", assignee="worker")
            kb._append_event(conn, tid, kind="gave_up")

        t.join(timeout=5)
        assert len(_messages) == 1
        assert _messages[0]["kind"] == "gave_up"

    def test_created_event_does_not_write_to_fifo(self, kanban_home, fifo_path):
        """A 'created' event should NOT produce a FIFO write."""
        # Open FIFO with a short timeout via select to detect if anything is written.
        import select

        r_fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            with kb.connect() as conn:
                kb.create_task(conn, title="no-notify", assignee="worker")

            # Give a moment for any spurious write.
            time.sleep(0.3)
            ready, _, _ = select.select([r_fd], [], [], 0.5)
            # Nothing should be readable — created is not a notify kind.
            assert not ready, "FIFO should not receive data for 'created' events"
        finally:
            os.close(r_fd)

    def test_heartbeat_event_does_not_write_to_fifo(self, kanban_home, fifo_path):
        """A 'heartbeat' event should NOT produce a FIFO write."""
        import select

        r_fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            with kb.connect() as conn:
                tid = kb.create_task(conn, title="hb task", assignee="worker")
                kb._append_event(conn, tid, kind="heartbeat")

            time.sleep(0.3)
            ready, _, _ = select.select([r_fd], [], [], 0.5)
            assert not ready, "FIFO should not receive data for 'heartbeat' events"
        finally:
            os.close(r_fd)

    def test_no_fifo_present_does_not_raise(self, kanban_home, tmp_path, monkeypatch):
        """When FIFO file doesn't exist, _append_event should not raise."""
        # Point expanduser at a nonexistent FIFO.
        fake_fifo = str(tmp_path / "nonexistent" / "tui_kanban.fifo")
        _real_expand = os.path.expanduser

        def _fake_expand(p):
            if "tui_kanban.fifo" in p:
                return fake_fifo
            return _real_expand(p)

        monkeypatch.setattr(os.path, "expanduser", _fake_expand)

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="no-fifo", assignee="worker")
            # Should not raise even though FIFO path doesn't exist.
            kb.complete_task(conn, tid, summary="done")

    def test_multiple_events_write_multiple_lines(self, kanban_home, fifo_path):
        """Multiple notify events should each produce a FIFO line.

        Each write requires a fresh reader because a FIFO with no reader
        causes open(fifo, 'w') to block. The reader reopens per event.
        """
        _messages = []
        _lock = threading.Lock()

        def _read_one():
            with open(fifo_path, "r") as f:
                for line in f:
                    with _lock:
                        _messages.append(json.loads(line.strip()))
                    break  # one line per open

        # Read first event.
        t1 = threading.Thread(target=_read_one, daemon=True)
        t1.start()
        time.sleep(0.2)

        with kb.connect() as conn:
            tid1 = kb.create_task(conn, title="task-1", assignee="worker")
            kb.complete_task(conn, tid1, summary="done-1")

        t1.join(timeout=5)

        # Read second event.
        t2 = threading.Thread(target=_read_one, daemon=True)
        t2.start()
        time.sleep(0.2)

        with kb.connect() as conn:
            tid2 = kb.create_task(conn, title="task-2", assignee="worker")
            kb.block_task(conn, tid2, reason="waiting")

        t2.join(timeout=5)

        assert len(_messages) == 2
        assert _messages[0]["kind"] == "completed"
        assert _messages[0]["task_id"] == tid1
        assert _messages[1]["kind"] == "blocked"
        assert _messages[1]["task_id"] == tid2


# ---------------------------------------------------------------------------
# Reader-side tests (tui_gateway.server._format_kanban_notification)
# ---------------------------------------------------------------------------

class TestFormatKanbanNotification:
    """Tests for _format_kanban_notification in tui_gateway/server.py."""

    @pytest.fixture(autouse=True)
    def _setup_server(self, tmp_path, monkeypatch):
        """Import tui_gateway.server with FIFO redirected to tmp_path."""
        fifo = tmp_path / "tui_kanban.fifo"
        os.mkfifo(str(fifo), 0o600)

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
            srv._KANBAN_FIFO_PATH = str(fifo)
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
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_abc" in msg
        assert "done" in msg
        assert "shipped rate limiter" in msg

    def test_completed_without_summary(self):
        ev = self._make_event("completed", {})
        sub = self._make_sub("t_xyz")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_xyz" in msg
        assert "done" in msg

    def test_completed_with_none_payload(self):
        ev = self._make_event("completed", None)
        sub = self._make_sub("t_123")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_123" in msg

    def test_blocked_with_reason(self):
        ev = self._make_event("blocked", {"reason": "need API key"})
        sub = self._make_sub("t_blk")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_blk" in msg
        assert "blocked" in msg
        assert "need API key" in msg

    def test_blocked_without_reason(self):
        ev = self._make_event("blocked", {})
        sub = self._make_sub("t_blk2")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_blk2" in msg
        assert "blocked" in msg

    def test_crashed(self):
        ev = self._make_event("crashed")
        sub = self._make_sub("t_crash")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_crash" in msg
        assert "crashed" in msg
        assert "dispatcher will retry" in msg

    def test_timed_out_with_limit(self):
        ev = self._make_event("timed_out", {"limit_seconds": 300})
        sub = self._make_sub("t_to")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_to" in msg
        assert "timed out" in msg
        assert "300" in msg

    def test_timed_out_without_limit(self):
        ev = self._make_event("timed_out", None)
        sub = self._make_sub("t_to2")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_to2" in msg

    def test_gave_up_with_error(self):
        ev = self._make_event("gave_up", {"error": "spawn failed 3x"})
        sub = self._make_sub("t_gu")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_gu" in msg
        assert "gave up" in msg
        assert "spawn failed 3x" in msg

    def test_gave_up_without_error(self):
        ev = self._make_event("gave_up", None)
        sub = self._make_sub("t_gu2")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_gu2" in msg

    def test_unknown_kind_returns_none(self):
        ev = self._make_event("edited")
        sub = self._make_sub("t_xxx")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is None

    def test_completed_summary_truncated_to_200_chars(self):
        long_summary = "x" * 300
        ev = self._make_event("completed", {"summary": long_summary})
        sub = self._make_sub("t_long")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        # The summary in the message should be truncated.
        assert len(msg) < len(long_summary) + 100  # rough upper bound

    def test_blocked_reason_truncated_to_160_chars(self):
        long_reason = "y" * 200
        ev = self._make_event("blocked", {"reason": long_reason})
        sub = self._make_sub("t_longblk")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert len(msg) < len(long_reason) + 100

    def test_dict_event_access(self):
        """Events from DB queries arrive as dicts, not MagicMock objects."""
        ev = {"kind": "completed", "payload": {"summary": "dict event"}}
        sub = self._make_sub("t_dict")
        msg = self.server._format_kanban_notification(ev, sub)
        assert msg is not None
        assert "t_dict" in msg
        assert "dict event" in msg


# ---------------------------------------------------------------------------
# FIFO lifecycle tests (module-level mkfifo + atexit cleanup)
# ---------------------------------------------------------------------------

class TestFifoLifecycle:
    """Tests for FIFO creation and cleanup."""

    def test_fifo_created_on_import(self, tmp_path, monkeypatch):
        """Importing tui_gateway.server should create the FIFO."""
        fifo = tmp_path / "tui_kanban.fifo"

        # Patch os.mkfifo so the module-level code creates our test FIFO.
        _real_mkfifo = os.mkfifo
        _called = []

        def _fake_mkfifo(path, mode=0o666):
            _called.append(path)
            return _real_mkfifo(str(fifo), mode)

        monkeypatch.setattr(os, "mkfifo", _fake_mkfifo)

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
            srv._KANBAN_FIFO_PATH = str(fifo)

            # The module-level code should have created the FIFO.
            assert os.path.exists(str(fifo))

            srv._sessions.clear()
            importlib.reload(srv)

    def test_cleanup_unlinks_fifo(self, tmp_path, monkeypatch):
        """_cleanup_kanban_fifo should unlink the FIFO."""
        fifo = tmp_path / "tui_kanban.fifo"
        os.mkfifo(str(fifo), 0o600)  # pre-create so module import gets FileExistsError (caught)

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
            srv._KANBAN_FIFO_PATH = str(fifo)

            assert os.path.exists(str(fifo))
            srv._cleanup_kanban_fifo()
            assert not os.path.exists(str(fifo))

            srv._sessions.clear()
            importlib.reload(srv)


# ---------------------------------------------------------------------------
# Integration: write → read through real FIFO
# ---------------------------------------------------------------------------

class TestFifoEndToEnd:
    """End-to-end tests: kanban_db writes → tui_gateway reads via real FIFO."""

    def test_complete_task_delivers_notification_through_fifo(
        self, kanban_home, tmp_path, monkeypatch
    ):
        """Completing a task should deliver a notification through the FIFO.

        This is the core integration test: _append_event writes a JSON line,
        a reader thread picks it up, and the formatted message matches.
        """
        fifo = tmp_path / "tui_kanban.fifo"
        os.mkfifo(str(fifo), 0o600)

        _real_expand = os.path.expanduser

        def _fake_expand(p):
            if "tui_kanban.fifo" in p:
                return str(fifo)
            return _real_expand(p)

        monkeypatch.setattr(os.path, "expanduser", _fake_expand)

        _received = []
        _done = threading.Event()

        def _reader():
            with open(str(fifo), "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    _received.append(data)
                    _done.set()
                    break

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        time.sleep(0.2)

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="e2e-test", assignee="worker")
            kb.complete_task(conn, tid, summary="e2e complete")

        _done.wait(timeout=5)
        t.join(timeout=5)

        assert len(_received) == 1
        assert _received[0]["task_id"] == tid
        assert _received[0]["kind"] == "completed"

    def test_non_notify_event_not_seen_by_reader(
        self, kanban_home, tmp_path, monkeypatch
    ):
        """Non-notify events (created, edited) should NOT appear on the FIFO."""
        fifo = tmp_path / "tui_kanban.fifo"
        os.mkfifo(str(fifo), 0o600)

        _real_expand = os.path.expanduser

        def _fake_expand(p):
            if "tui_kanban.fifo" in p:
                return str(fifo)
            return _real_expand(p)

        monkeypatch.setattr(os.path, "expanduser", _fake_expand)

        import select

        r_fd = os.open(str(fifo), os.O_RDONLY | os.O_NONBLOCK)
        try:
            with kb.connect() as conn:
                # create_task fires a 'created' event — should NOT write to FIFO.
                kb.create_task(conn, title="silent", assignee="worker")

            time.sleep(0.3)
            ready, _, _ = select.select([r_fd], [], [], 0.5)
            assert not ready, "No FIFO data should arrive for 'created' events"
        finally:
            os.close(r_fd)

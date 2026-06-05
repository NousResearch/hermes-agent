"""Tests for the unified loop architecture.

Covers:
  - UnifiedLoopTicker singleton, start/stop, heap, optimistic locking
  - UnifiedLoopManager CRUD, pause/resume, status_line
  - parse_loop_command parser
  - execute_loop_command integration
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate HERMES_HOME for each test."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Clear singleton between tests
    from hermes_cli.unified_loop import UnifiedLoopTicker

    UnifiedLoopTicker.reset_instance()
    yield home
    UnifiedLoopTicker.reset_instance()


@pytest.fixture
def db(_hermes_home: Path):
    """Return a fresh SessionDB."""
    from hermes_state import SessionDB

    return SessionDB()


@pytest.fixture
def ticker(_hermes_home: Path):
    """Return a fresh UnifiedLoopTicker (not started)."""
    from hermes_cli.unified_loop import UnifiedLoopTicker

    return UnifiedLoopTicker.get_instance(str(_hermes_home))


@pytest.fixture
def manager(_hermes_home: Path, request):
    """Return a UnifiedLoopManager for a test-specific session."""
    from hermes_cli.unified_loop import UnifiedLoopManager

    # Unique session ID per test to avoid cross-contamination
    sid = f"test-{request.node.nodeid.replace('/', '-').replace('::', '-')}"
    return UnifiedLoopManager(sid, str(_hermes_home))


# ---------------------------------------------------------------------------
# UnifiedLoopTicker tests
# ---------------------------------------------------------------------------


class TestUnifiedLoopTicker:
    def test_singleton(self, _hermes_home: Path):
        """Only one ticker instance per hermes_home."""
        from hermes_cli.unified_loop import UnifiedLoopTicker

        t1 = UnifiedLoopTicker.get_instance(str(_hermes_home))
        t2 = UnifiedLoopTicker.get_instance(str(_hermes_home))
        assert t1 is t2

    def test_start_stop(self, ticker):
        """Start and stop the ticker thread cleanly."""
        ticker.start()
        assert ticker._thread is not None
        assert ticker._thread.is_alive()
        ticker.stop(timeout=2.0)
        assert ticker._stop.is_set()

    def test_start_idempotent(self, ticker):
        """Starting twice doesn't create a second thread."""
        ticker.start()
        thread1 = ticker._thread
        ticker.start()
        assert ticker._thread is thread1
        ticker.stop(timeout=2.0)

    def test_register_dispatcher(self, ticker):
        """Register and unregister dispatchers."""
        handler = MagicMock()
        ticker.register_dispatcher("cli", handler)
        assert ticker._dispatchers["cli"] is handler
        ticker.unregister_dispatcher("cli")
        assert "cli" not in ticker._dispatchers

    def test_mark_heap_dirty(self, ticker):
        """mark_heap_dirty sets the event."""
        ticker._heap_dirty.clear()
        ticker.mark_heap_dirty()
        assert ticker._heap_dirty.is_set()

    def test_compute_sleep_empty_heap(self, ticker):
        """Empty heap returns 5.0s sleep."""
        ticker._heap = []
        assert ticker._compute_sleep() == 5.0

    def test_compute_sleep_with_due(self, ticker):
        """Heap with past entry returns ~0 sleep."""
        ticker._heap = [(time.time() - 10, "s1", "u1")]
        sleep = ticker._compute_sleep()
        assert 0.0 <= sleep <= 0.5

    def test_compute_sleep_with_future(self, ticker):
        """Heap with future entry returns capped sleep."""
        ticker._heap = [(time.time() + 120, "s1", "u1")]
        sleep = ticker._compute_sleep()
        assert 59.0 <= sleep <= 60.0

    def test_try_fire_loop_no_row(self, ticker, db):
        """_try_fire_loop returns None for non-existent loop."""
        result = ticker._try_fire_loop("no-session", "no-uid", time.time())
        assert result is None

    def test_try_fire_loop_fires(self, ticker, manager):
        """_try_fire_loop returns row data when interval elapsed."""
        uid = manager.create(
            body="test prompt",
            interval_seconds=60,
            platform="cli",
            fire_now=True,  # last_fired_at = 0 → fires immediately
        )
        now = time.time()
        result = ticker._try_fire_loop(manager.session_id, uid, now)
        assert result is not None
        assert result["prompt"] == "test prompt"
        assert result["platform"] == "cli"

    def test_try_fire_loop_not_due(self, ticker, manager):
        """_try_fire_loop returns None when interval not elapsed."""
        uid = manager.create(
            body="test prompt",
            interval_seconds=300,
            platform="cli",
            fire_now=False,  # last_fired_at = now
        )
        # Try to fire 10s later (interval is 300s)
        result = ticker._try_fire_loop(manager.session_id, uid, time.time() + 10)
        assert result is None

    def test_try_fire_loop_max_turns(self, ticker, manager):
        """Loop auto-pauses when max_turns reached."""
        uid = manager.create(
            body="limited",
            interval_seconds=60,
            platform="cli",
            max_turns=1,
            fire_now=True,
        )
        # First fire succeeds
        result = ticker._try_fire_loop(manager.session_id, uid, time.time())
        assert result is not None
        # Second fire should mark as done
        result = ticker._try_fire_loop(manager.session_id, uid, time.time() + 61)
        assert result is None

    def test_dispatch_loop(self, ticker):
        """_dispatch_loop calls the registered handler."""
        handler = MagicMock()
        ticker.register_dispatcher("cli", handler)
        ticker._dispatch_loop(
            "test-session",
            "test-uid",
            {"prompt": "hello", "platform": "cli", "source_json": None},
        )
        handler.assert_called_once_with(
            "test-session", "hello", source_json=None
        )

    def test_dispatch_loop_no_handler(self, ticker):
        """_dispatch_loop logs warning when no handler registered."""
        # Should not raise
        ticker._dispatch_loop(
            "test-session",
            "test-uid",
            {"prompt": "hello", "platform": "unknown", "source_json": None},
        )

    def test_dispatch_loop_handler_error(self, ticker):
        """_dispatch_loop catches handler exceptions."""
        handler = MagicMock(side_effect=RuntimeError("boom"))
        ticker.register_dispatcher("cli", handler)
        # Should not raise
        ticker._dispatch_loop(
            "test-session",
            "test-uid",
            {"prompt": "hello", "platform": "cli", "source_json": None},
        )


# ---------------------------------------------------------------------------
# UnifiedLoopManager tests
# ---------------------------------------------------------------------------


class TestUnifiedLoopManager:
    def test_create_delete(self, manager):
        """Create and delete a loop."""
        uid = manager.create(body="ping", interval_seconds=60)
        assert uid
        loops = manager.list()
        assert len(loops) == 1
        assert loops[0]["prompt"] == "ping"
        assert manager.delete(uid) is True
        assert manager.list() == []

    def test_create_empty_body_raises(self, manager):
        """Creating a loop with empty body raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            manager.create(body="", interval_seconds=60)

    def test_create_min_interval(self, manager):
        """Interval is clamped to MIN_INTERVAL_SECONDS."""
        uid = manager.create(body="fast", interval_seconds=10)
        loops = manager.list()
        assert loops[0]["interval_seconds"] >= 60

    def test_delete_all(self, manager):
        """delete_all removes all loops."""
        manager.create(body="a", interval_seconds=60)
        manager.create(body="b", interval_seconds=60)
        count = manager.delete_all()
        assert count == 2
        assert manager.list() == []

    def test_pause_resume(self, manager):
        """Pause and resume a loop."""
        uid = manager.create(body="test", interval_seconds=60)
        assert manager.pause(uid, reason="user paused") is True
        loops = manager.list()
        assert loops[0]["status"] == "paused"
        assert loops[0]["reason"] == "user paused"
        assert manager.resume(uid) is True
        loops = manager.list()
        assert loops[0]["status"] == "active"

    def test_active_loop(self, manager):
        """active_loop returns first active or paused loop."""
        assert manager.active_loop is None
        uid = manager.create(body="test", interval_seconds=60)
        assert manager.active_loop is not None
        assert manager.active_loop["uid"] == uid

    def test_is_active(self, manager):
        """is_active returns True when loops exist."""
        assert manager.is_active() is False
        manager.create(body="test", interval_seconds=60)
        assert manager.is_active() is True

    def test_status_line_empty(self, manager):
        """status_line shows message when no loops."""
        line = manager.status_line()
        assert "No active loops" in line

    def test_status_line_active(self, manager):
        """status_line shows active loop with countdown."""
        manager.create(body="ping", interval_seconds=300)
        line = manager.status_line()
        assert "⊙" in line
        assert "active" in line
        assert "ping" in line

    def test_status_line_paused(self, manager):
        """status_line shows paused loop with reason."""
        uid = manager.create(body="ping", interval_seconds=300)
        manager.pause(uid, reason="debugging")
        line = manager.status_line()
        assert "⏸" in line
        assert "paused" in line
        assert "debugging" in line

    def test_source_json_stored(self, manager):
        """source_json is stored and retrievable."""
        source = json.dumps({"chat_id": "123", "user_id": "456"})
        uid = manager.create(
            body="test", interval_seconds=60, source_json=source, platform="gateway"
        )
        loops = manager.list()
        assert loops[0]["source_json"] == source
        assert loops[0]["platform"] == "gateway"


# ---------------------------------------------------------------------------
# parse_loop_command tests
# ---------------------------------------------------------------------------


class TestParseLoopCommand:
    def test_empty(self):
        from hermes_cli.loop_commands import parse_loop_command

        assert parse_loop_command("") == {"action": "list"}
        assert parse_loop_command("/loop") == {"action": "list"}

    def test_list(self):
        from hermes_cli.loop_commands import parse_loop_command

        assert parse_loop_command("/loop list") == {"action": "list"}
        assert parse_loop_command("/loop status") == {"action": "list"}

    def test_pause(self):
        from hermes_cli.loop_commands import parse_loop_command

        result = parse_loop_command("/loop pause abc123")
        assert result == {"action": "pause", "uid": "abc123"}

    def test_pause_all(self):
        from hermes_cli.loop_commands import parse_loop_command

        assert parse_loop_command("/loop pause") == {"action": "pause_all"}

    def test_resume(self):
        from hermes_cli.loop_commands import parse_loop_command

        result = parse_loop_command("/loop resume abc123")
        assert result == {"action": "resume", "uid": "abc123"}

    def test_delete(self):
        from hermes_cli.loop_commands import parse_loop_command

        for cmd in ("remove", "delete", "rm", "clear", "stop", "done"):
            result = parse_loop_command(f"/loop {cmd} abc123")
            assert result == {"action": "delete", "uid": "abc123"}, cmd

    def test_delete_all(self):
        from hermes_cli.loop_commands import parse_loop_command

        assert parse_loop_command("/loop clear") == {"action": "delete_all"}

    def test_create(self):
        from hermes_cli.loop_commands import parse_loop_command

        result = parse_loop_command("/loop 5m check status")
        assert result["action"] == "create"
        assert result["interval_seconds"] == 300
        assert result["prompt"] == "check status"

    def test_create_every_prefix(self):
        from hermes_cli.loop_commands import parse_loop_command

        result = parse_loop_command("/loop every 5m check status")
        assert result["action"] == "create"
        assert result["interval_seconds"] == 300
        assert result["prompt"] == "check status"

    def test_create_missing_prompt(self):
        from hermes_cli.loop_commands import parse_loop_command

        result = parse_loop_command("/loop 5m")
        assert result["action"] == "error"
        assert "Missing prompt" in result["message"]

    def test_unknown_subcommand(self):
        from hermes_cli.loop_commands import parse_loop_command

        result = parse_loop_command("/loop foobar")
        assert result["action"] == "error"
        assert "Unknown subcommand" in result["message"]

    def test_strips_slash_prefix(self):
        from hermes_cli.loop_commands import parse_loop_command

        result = parse_loop_command("loop 5m test")
        assert result["action"] == "create"

    def test_uid_strips_hash(self):
        from hermes_cli.loop_commands import parse_loop_command

        result = parse_loop_command("/loop pause #abc123")
        assert result["uid"] == "abc123"


# ---------------------------------------------------------------------------
# delete_meta / prune_sessions tests
# ---------------------------------------------------------------------------


class TestDeleteMeta:
    def test_delete_meta(self, db):
        """delete_meta removes a key from state_meta."""
        db.set_meta("test-key", "test-value")
        assert db.get_meta("test-key") == "test-value"
        db.delete_meta("test-key")
        assert db.get_meta("test-key") is None

    def test_delete_meta_nonexistent(self, db):
        """delete_meta is a no-op for non-existent keys."""
        db.delete_meta("no-such-key")  # should not raise

    def test_prune_sessions_cleans_loops(self, db):
        """prune_sessions removes orphaned loop keys."""
        # Create a fake session
        db._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO sessions (id, source, started_at, ended_at) "
                "VALUES (?, ?, ?, ?)",
                ("old-session", "cli", time.time() - 91 * 86400, time.time()),
            )
        )
        # Add loop metadata
        db.set_meta("loop:old-session:abc123", '{"prompt":"test"}')
        db.set_meta("loop:old-session:__ids__", '["abc123"]')
        # Prune
        count = db.prune_sessions(older_than_days=90)
        assert count >= 1
        assert db.get_meta("loop:old-session:abc123") is None

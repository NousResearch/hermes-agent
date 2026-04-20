"""Tests for watch_patterns background process monitoring feature.

Covers:
  - ProcessSession.watch_patterns field
  - ProcessRegistry._check_watch_patterns() matching + notification
  - Rate limiting (WATCH_MAX_PER_WINDOW) and overload kill switch
  - watch_queue population
  - Checkpoint persistence of watch_patterns
  - Terminal tool schema includes watch_patterns
  - Terminal tool handler passes watch_patterns through
"""

import json
import queue
import time
import pytest
from unittest.mock import patch

from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    WATCH_MAX_PER_WINDOW,
    WATCH_WINDOW_SECONDS,
    WATCH_OVERLOAD_KILL_SECONDS,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


def _make_session(
    sid="proc_test_watch",
    command="tail -f app.log",
    task_id="t1",
    watch_patterns=None,
) -> ProcessSession:
    s = ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=time.time(),
        watch_patterns=watch_patterns or [],
    )
    return s


# =========================================================================
# ProcessSession field defaults
# =========================================================================

class TestProcessSessionField:
    def test_default_empty(self):
        s = ProcessSession(id="proc_1", command="echo hi")
        assert s.watch_patterns == []
        assert s._watch_disabled is False
        assert s._watch_hits == 0
        assert s._watch_suppressed == 0

    def test_can_set_patterns(self):
        s = _make_session(watch_patterns=["ERROR", "WARN"])
        assert s.watch_patterns == ["ERROR", "WARN"]


# =========================================================================
# Pattern matching + queue population
# =========================================================================

class TestCheckWatchPatterns:
    def test_no_patterns_no_notification(self, registry):
        """No watch_patterns → no notifications."""
        session = _make_session(watch_patterns=[])
        registry._check_watch_patterns(session, "ERROR: something broke\n")
        assert registry.completion_queue.empty()

    def test_no_match_no_notification(self, registry):
        """Output that doesn't match any pattern → no notification."""
        session = _make_session(watch_patterns=["ERROR", "FAIL"])
        registry._check_watch_patterns(session, "INFO: all good\nDEBUG: fine\n")
        assert registry.completion_queue.empty()

    def test_basic_match(self, registry):
        """Single matching line triggers a notification."""
        session = _make_session(watch_patterns=["ERROR"])
        registry._check_watch_patterns(session, "INFO: ok\nERROR: disk full\n")
        assert not registry.completion_queue.empty()
        evt = registry.completion_queue.get_nowait()
        assert evt["type"] == "watch_match"
        assert evt["pattern"] == "ERROR"
        assert "disk full" in evt["output"]
        assert evt["session_id"] == "proc_test_watch"

    def test_match_carries_session_key_and_watcher_routing_metadata(self, registry):
        session = _make_session(watch_patterns=["ERROR"])
        session.session_key = "agent:main:telegram:group:-100:42"
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "-100"
        session.watcher_user_id = "u123"
        session.watcher_user_name = "alice"
        session.watcher_thread_id = "42"

        registry._check_watch_patterns(session, "ERROR: disk full\n")
        evt = registry.completion_queue.get_nowait()

        assert evt["session_key"] == "agent:main:telegram:group:-100:42"
        assert evt["platform"] == "telegram"
        assert evt["chat_id"] == "-100"
        assert evt["user_id"] == "u123"
        assert evt["user_name"] == "alice"
        assert evt["thread_id"] == "42"

    def test_multiple_patterns(self, registry):
        """First matching pattern is reported."""
        session = _make_session(watch_patterns=["WARN", "ERROR"])
        registry._check_watch_patterns(session, "ERROR: bad\nWARN: hmm\n")
        evt = registry.completion_queue.get_nowait()
        # ERROR appears first in the output, and we check patterns in order
        # so "WARN" won't match "ERROR: bad" but "ERROR" will
        assert evt["pattern"] == "ERROR"
        assert "bad" in evt["output"]

    def test_disabled_skips(self, registry):
        """Disabled watch produces no notifications."""
        session = _make_session(watch_patterns=["ERROR"])
        session._watch_disabled = True
        registry._check_watch_patterns(session, "ERROR: boom\n")
        assert registry.completion_queue.empty()

    def test_hit_counter_increments(self, registry):
        """Each delivered notification increments _watch_hits."""
        session = _make_session(watch_patterns=["X"])
        registry._check_watch_patterns(session, "X\n")
        assert session._watch_hits == 1
        registry._check_watch_patterns(session, "X\n")
        assert session._watch_hits == 2

    def test_output_truncation(self, registry):
        """Very long matched output is truncated."""
        session = _make_session(watch_patterns=["X"])
        # Generate 30 matching lines (more than the 20-line cap)
        text = "\n".join(f"X line {i}" for i in range(30)) + "\n"
        registry._check_watch_patterns(session, text)
        evt = registry.completion_queue.get_nowait()
        # Should only have 20 lines max
        assert evt["output"].count("\n") <= 20


# =========================================================================
# Rate limiting
# =========================================================================

class TestRateLimiting:
    def test_within_window_limit(self, registry):
        """Notifications within the rate limit all get delivered."""
        session = _make_session(watch_patterns=["E"])
        for i in range(WATCH_MAX_PER_WINDOW):
            registry._check_watch_patterns(session, f"E {i}\n")
        assert registry.completion_queue.qsize() == WATCH_MAX_PER_WINDOW

    def test_exceeds_window_limit(self, registry):
        """Notifications beyond the rate limit are suppressed."""
        session = _make_session(watch_patterns=["E"])
        for i in range(WATCH_MAX_PER_WINDOW + 5):
            registry._check_watch_patterns(session, f"E {i}\n")
        # Only WATCH_MAX_PER_WINDOW should be in the queue
        assert registry.completion_queue.qsize() == WATCH_MAX_PER_WINDOW
        assert session._watch_suppressed == 5

    def test_window_resets(self, registry):
        """After the window expires, notifications can flow again."""
        session = _make_session(watch_patterns=["E"])
        # Fill the window
        for i in range(WATCH_MAX_PER_WINDOW):
            registry._check_watch_patterns(session, f"E {i}\n")
        # One more should be suppressed
        registry._check_watch_patterns(session, "E extra\n")
        assert session._watch_suppressed == 1

        # Fast-forward past window
        session._watch_window_start = time.time() - WATCH_WINDOW_SECONDS - 1
        registry._check_watch_patterns(session, "E after reset\n")
        # Should deliver now (window reset)
        assert registry.completion_queue.qsize() == WATCH_MAX_PER_WINDOW + 1

    def test_suppressed_count_in_next_delivery(self, registry):
        """Suppressed count is reported in the next successful delivery."""
        session = _make_session(watch_patterns=["E"])
        for i in range(WATCH_MAX_PER_WINDOW):
            registry._check_watch_patterns(session, f"E {i}\n")
        # Suppress 3 more
        for i in range(3):
            registry._check_watch_patterns(session, f"E suppressed {i}\n")
        assert session._watch_suppressed == 3

        # Fast-forward past window to allow delivery
        session._watch_window_start = time.time() - WATCH_WINDOW_SECONDS - 1
        registry._check_watch_patterns(session, "E back\n")
        # Drain to the last event
        last_evt = None
        while not registry.completion_queue.empty():
            last_evt = registry.completion_queue.get_nowait()
        assert last_evt["suppressed"] == 3
        assert session._watch_suppressed == 0  # reset after delivery


# =========================================================================
# Overload kill switch
# =========================================================================

class TestOverloadKillSwitch:
    def test_sustained_overload_disables(self, registry):
        """Sustained overload beyond threshold permanently disables watching."""
        session = _make_session(watch_patterns=["E"])
        # Fill the window to trigger rate limit
        for i in range(WATCH_MAX_PER_WINDOW):
            registry._check_watch_patterns(session, f"E {i}\n")

        # Simulate sustained overload: set overload_since to past threshold
        session._watch_overload_since = time.time() - WATCH_OVERLOAD_KILL_SECONDS - 1
        # Force another suppressed hit
        registry._check_watch_patterns(session, "E overload\n")
        registry._check_watch_patterns(session, "E overload2\n")

        assert session._watch_disabled is True
        # Should have a watch_disabled event in the queue
        disabled_evts = []
        while not registry.completion_queue.empty():
            evt = registry.completion_queue.get_nowait()
            if evt.get("type") == "watch_disabled":
                disabled_evts.append(evt)
        assert len(disabled_evts) == 1
        assert "too many matches" in disabled_evts[0]["message"]

    def test_overload_resets_on_delivery(self, registry):
        """Overload timer resets when a notification gets through."""
        session = _make_session(watch_patterns=["E"])
        # Start overload tracking
        session._watch_overload_since = time.time() - 10
        # But window allows delivery → overload should reset
        registry._check_watch_patterns(session, "E ok\n")
        assert session._watch_overload_since == 0.0
        assert session._watch_disabled is False


# =========================================================================
# Checkpoint persistence
# =========================================================================

class TestCheckpointPersistence:
    def test_watch_patterns_in_checkpoint(self, registry):
        """watch_patterns is included in checkpoint data."""
        session = _make_session(watch_patterns=["ERROR", "FAIL"])
        with registry._lock:
            registry._running[session.id] = session

        with patch("utils.atomic_json_write") as mock_write:
            registry._write_checkpoint()
            args = mock_write.call_args
            entries = args[0][1]  # second positional arg
            assert len(entries) == 1
            assert entries[0]["watch_patterns"] == ["ERROR", "FAIL"]

    def test_watch_patterns_recovery(self, registry, tmp_path, monkeypatch):
        """watch_patterns survives checkpoint recovery."""
        import tools.process_registry as pr_mod
        checkpoint = tmp_path / "processes.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_recovered",
            "command": "tail -f log",
            "pid": 99999999,  # non-existent
            "pid_scope": "host",
            "started_at": time.time(),
            "task_id": "",
            "session_key": "",
            "watcher_platform": "",
            "watcher_chat_id": "",
            "watcher_thread_id": "",
            "watcher_interval": 0,
            "notify_on_complete": False,
            "watch_patterns": ["PANIC", "OOM"],
        }]))
        monkeypatch.setattr(pr_mod, "CHECKPOINT_PATH", checkpoint)
        # PID doesn't exist, so nothing will be recovered
        count = registry.recover_from_checkpoint()
        # Won't recover since PID is fake, but verify the code path doesn't crash
        assert count == 0


# =========================================================================
# Terminal tool schema + handler
# =========================================================================

class TestTerminalToolSchema:
    def test_schema_includes_watch_patterns(self):
        from tools.terminal_tool import TERMINAL_SCHEMA
        props = TERMINAL_SCHEMA["parameters"]["properties"]
        assert "watch_patterns" in props
        assert props["watch_patterns"]["type"] == "array"
        assert props["watch_patterns"]["items"] == {"type": "string"}

    def test_handler_passes_watch_patterns(self):
        """_handle_terminal passes watch_patterns to terminal_tool."""
        from tools.terminal_tool import _handle_terminal
        with patch("tools.terminal_tool.terminal_tool") as mock_tt:
            mock_tt.return_value = json.dumps({"output": "ok", "exit_code": 0})
            _handle_terminal(
                {"command": "echo hi", "watch_patterns": ["ERR"]},
                task_id="t1",
            )
            _, kwargs = mock_tt.call_args
            assert kwargs.get("watch_patterns") == ["ERR"]


# =========================================================================
# Code execution tool blocked params
# =========================================================================

class TestCodeExecutionBlocked:
    def test_watch_patterns_blocked(self):
        from tools.code_execution_tool import _TERMINAL_BLOCKED_PARAMS
        assert "watch_patterns" in _TERMINAL_BLOCKED_PARAMS

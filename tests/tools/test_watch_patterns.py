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
import time
import pytest
from unittest.mock import patch

from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    WATCH_STRIKE_LIMIT,
    WATCH_GLOBAL_MAX_PER_WINDOW,
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
# Per-session rate limiting: 1 notification per 15s, 3 strikes → disable
# =========================================================================

class TestPerSessionRateLimit:
    def test_first_match_delivers(self, registry):
        """A fresh session with no prior cooldown delivers the first match."""
        session = _make_session(watch_patterns=["E"])
        registry._check_watch_patterns(session, "E first\n")
        assert registry.completion_queue.qsize() == 1
        evt = registry.completion_queue.get_nowait()
        assert evt["type"] == "watch_match"
        assert session._watch_hits == 1
        # Cooldown is now armed.
        assert session._watch_cooldown_until > 0


    def test_suppressed_count_in_next_delivery(self, registry):
        """Suppressed count from a strike window is reported in the next emit."""
        session = _make_session(watch_patterns=["E"])
        registry._check_watch_patterns(session, "E emit\n")
        for _ in range(4):
            registry._check_watch_patterns(session, "E drop\n")
        assert session._watch_suppressed == 4

        # Fast-forward past cooldown.
        session._watch_cooldown_until = time.time() - 0.01
        # Drain the queue so we can inspect the next emission.
        while not registry.completion_queue.empty():
            registry.completion_queue.get_nowait()

        registry._check_watch_patterns(session, "E back\n")
        evt = registry.completion_queue.get_nowait()
        assert evt["type"] == "watch_match"
        assert evt["suppressed"] == 4
        assert session._watch_suppressed == 0  # reset after delivery


# =========================================================================
# Lifetime cap: sparsely-spaced matches never trip the consecutive-strike
# counter, but must still stop eventually (#93513).
# =========================================================================

class TestLifetimeCap:
    def test_sparse_matches_disable_after_lifetime_cap(self, registry):
        """Matches spaced well past the cooldown never strike, but a service
        restarted over and over must still get disabled instead of forcing a
        full-context turn forever."""
        from tools.process_registry import WATCH_LIFETIME_MAX_HITS

        session = _make_session(watch_patterns=["Application started"])
        for i in range(WATCH_LIFETIME_MAX_HITS):
            # Force the cooldown to have already expired, simulating a
            # cleanly-spaced restart (no strikes should ever accumulate).
            session._watch_cooldown_until = 0.0
            registry._check_watch_patterns(session, "Application started\n")

        assert session._watch_hits == WATCH_LIFETIME_MAX_HITS
        assert session._watch_consecutive_strikes == 0  # never struck
        assert session._watch_disabled is True
        assert session.notify_on_complete is True

        events = []
        while not registry.completion_queue.empty():
            events.append(registry.completion_queue.get_nowait())
        match_events = [e for e in events if e["type"] == "watch_match"]
        disabled_events = [e for e in events if e["type"] == "watch_disabled"]
        assert len(match_events) == WATCH_LIFETIME_MAX_HITS
        assert len(disabled_events) == 1
        assert "lifetime cap" in disabled_events[0]["message"]

        # One more restart after the cap: no further turns are forced.
        session._watch_cooldown_until = 0.0
        registry._check_watch_patterns(session, "Application started\n")
        assert registry.completion_queue.empty()
        assert session._watch_hits == WATCH_LIFETIME_MAX_HITS

    def test_suppressed_matches_do_not_count_toward_lifetime_cap(self, registry):
        """Only DELIVERED notifications consume the lifetime budget — matches
        suppressed inside the cooldown window must not increment the counter
        (they never forced an agent turn)."""
        from tools.process_registry import WATCH_LIFETIME_MAX_HITS

        session = _make_session(watch_patterns=["E"])
        registry._check_watch_patterns(session, "E emit\n")
        assert session._watch_hits == 1

        # Flood inside the cooldown: all suppressed, none delivered.
        for _ in range(WATCH_LIFETIME_MAX_HITS * 3):
            registry._check_watch_patterns(session, "E drop\n")
        assert session._watch_hits == 1  # unchanged
        # Strike-limit disable may or may not have tripped depending on
        # WATCH_STRIKE_LIMIT vs the single window here; assert it did NOT,
        # since all drops land in ONE window (one strike max).
        assert session._watch_consecutive_strikes == 1
        assert session._watch_disabled is False

    def test_cap_trips_exactly_at_nth_delivery_and_promotes(self, registry):
        """The Nth delivered match is still delivered, then the session is
        promoted to notify_on_complete in the same call, with the
        watch_disabled summary queued right after the final match."""
        from tools.process_registry import WATCH_LIFETIME_MAX_HITS

        session = _make_session(watch_patterns=["ready"])
        for i in range(WATCH_LIFETIME_MAX_HITS - 1):
            session._watch_cooldown_until = 0.0
            registry._check_watch_patterns(session, "ready\n")
            assert session._watch_disabled is False
            assert session.notify_on_complete is False

        # Nth delivery trips the cap.
        session._watch_cooldown_until = 0.0
        registry._check_watch_patterns(session, "ready\n")
        assert session._watch_disabled is True
        assert session.notify_on_complete is True

        events = []
        while not registry.completion_queue.empty():
            events.append(registry.completion_queue.get_nowait())
        # Last two events: the Nth delivered match, then the summary.
        assert events[-2]["type"] == "watch_match"
        assert events[-1]["type"] == "watch_disabled"
        assert "lifetime cap" in events[-1]["message"]
        assert str(WATCH_LIFETIME_MAX_HITS) in events[-1]["message"]
        assert "notify_on_complete" in events[-1]["message"]


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


# =========================================================================
# Suppress-after-exit (anti-spam fix)
# =========================================================================

class TestSuppressAfterExit:
    def test_match_dropped_once_session_exited(self, registry):
        """watch_patterns notifications stop the moment session.exited is set."""
        session = _make_session(watch_patterns=["ERROR"])
        # Mark the process as exited BEFORE the late chunk arrives.
        session.exited = True
        registry._check_watch_patterns(session, "ERROR: late buffer\n")
        assert registry.completion_queue.empty()
        assert session._watch_hits == 0

    def test_match_still_delivered_while_session_running(self, registry):
        """Sanity: while the process is still running, matches still deliver."""
        session = _make_session(watch_patterns=["ERROR"])
        session.exited = False
        registry._check_watch_patterns(session, "ERROR: oh no\n")
        assert not registry.completion_queue.empty()
        evt = registry.completion_queue.get_nowait()
        assert evt["type"] == "watch_match"


# =========================================================================
# Mutual exclusion: notify_on_complete wins over watch_patterns
# =========================================================================

class TestMutualExclusion:
    def test_resolver_drops_watch_when_notify_set(self):
        """Both flags set → watch_patterns dropped with a note."""
        from tools.terminal_tool import _resolve_notification_flag_conflict

        resolved, note = _resolve_notification_flag_conflict(
            notify_on_complete=True,
            watch_patterns=["ERROR", "DONE"],
            background=True,
        )
        assert resolved is None
        assert "notify_on_complete" in note
        assert "duplicate notifications" in note


    def test_resolver_inert_when_not_background(self):
        """Without background=True, the whole thing is a no-op."""
        from tools.terminal_tool import _resolve_notification_flag_conflict

        resolved, note = _resolve_notification_flag_conflict(
            notify_on_complete=True,
            watch_patterns=["ERROR"],
            background=False,
        )
        assert resolved == ["ERROR"]
        assert note == ""


# =========================================================================
# Global circuit breaker (cross-session overflow blocker)
# =========================================================================

class TestGlobalCircuitBreaker:
    def test_trips_after_global_threshold(self, registry):
        """When >N matches fire across sessions in the window, breaker trips."""
        sessions = [
            _make_session(sid=f"proc_s{i}", watch_patterns=["E"])
            for i in range(WATCH_GLOBAL_MAX_PER_WINDOW + 3)
        ]
        # Each session fires exactly one match — individually well under the
        # per-session cap. But collectively they should trip the global cap.
        for s in sessions:
            registry._check_watch_patterns(s, "E hit\n")

        # Drain the queue and count event types.
        watch_matches = 0
        overflow_tripped = 0
        while not registry.completion_queue.empty():
            evt = registry.completion_queue.get_nowait()
            if evt.get("type") == "watch_match":
                watch_matches += 1
            elif evt.get("type") == "watch_overflow_tripped":
                overflow_tripped += 1
        assert watch_matches == WATCH_GLOBAL_MAX_PER_WINDOW
        assert overflow_tripped == 1
        assert registry._global_watch_tripped_until > 0

    def test_cooldown_suppresses_and_then_releases(self, registry):
        """After trip, further events are suppressed; cooldown expiry emits release."""
        # Spawn enough fresh sessions to trip the global breaker.
        sessions = [
            _make_session(sid=f"proc_t{i}", watch_patterns=["E"])
            for i in range(WATCH_GLOBAL_MAX_PER_WINDOW + 1)
        ]
        for s in sessions:
            registry._check_watch_patterns(s, "E hit\n")
        assert registry._global_watch_tripped_until > 0

        # Further matches from BRAND-NEW sessions during cooldown are dropped.
        q_size_before = registry.completion_queue.qsize()
        extra1 = _make_session(sid="proc_extra1", watch_patterns=["E"])
        extra2 = _make_session(sid="proc_extra2", watch_patterns=["E"])
        registry._check_watch_patterns(extra1, "E hit\n")
        registry._check_watch_patterns(extra2, "E hit\n")
        assert registry.completion_queue.qsize() == q_size_before  # no new events
        assert registry._global_watch_suppressed_during_trip >= 2

        # Simulate cooldown expiry.
        registry._global_watch_tripped_until = time.time() - 1

        # Next call admits AND emits the release summary.
        released_session = _make_session(sid="proc_after", watch_patterns=["E"])
        registry._check_watch_patterns(released_session, "E hit\n")
        released = False
        admitted = False
        while not registry.completion_queue.empty():
            evt = registry.completion_queue.get_nowait()
            if evt.get("type") == "watch_overflow_released":
                released = True
                assert evt["suppressed"] >= 2
            elif evt.get("type") == "watch_match":
                admitted = True
        assert released
        assert admitted


class TestOverflowNotificationFormatting:
    """watch_overflow_* events must surface their summary, not fall through
    to the completion formatter as a phantom 'process exited (exit code ?)'."""

    def test_overflow_tripped_formats_message(self):
        from tools.process_registry import format_process_notification

        evt = {
            "type": "watch_overflow_tripped",
            "message": "watch flood detected: 47 notifications suppressed for pattern 'ERROR'",
            "session_id": "proc_a1b2",
        }
        out = format_process_notification(evt)
        assert "47 notifications suppressed" in out
        assert "exit code" not in out

    def test_overflow_released_formats_message(self):
        from tools.process_registry import format_process_notification

        evt = {
            "type": "watch_overflow_released",
            "message": "watch flood released: notifications resumed for pattern 'ERROR'",
            "session_id": "proc_a1b2",
        }
        out = format_process_notification(evt)
        assert "notifications resumed" in out
        assert "exit code" not in out

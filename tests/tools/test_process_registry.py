"""Tests for tools/process_registry.py — ProcessRegistry query methods, pruning, checkpoint."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.environments.local import _HERMES_PROVIDER_ENV_FORCE_PREFIX
from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    MAX_OUTPUT_CHARS,
    FINISHED_TTL_SECONDS,
    MAX_PROCESSES,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


def _make_session(
    sid="proc_test123",
    command="echo hello",
    task_id="t1",
    exited=False,
    exit_code=None,
    output="",
    started_at=None,
) -> ProcessSession:
    """Helper to create a ProcessSession for testing."""
    s = ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=started_at or time.time(),
        exited=exited,
        exit_code=exit_code,
        output_buffer=output,
    )
    return s


def _spawn_python_sleep(seconds: float) -> subprocess.Popen:
    """Spawn a portable short-lived Python sleep process."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
    )


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll a predicate until it returns truthy or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# =========================================================================
# Output buffering / metadata
# =========================================================================

class TestOutputHardening:
    def test_append_output_rolls_buffer_and_updates_metadata(self, registry):
        s = _make_session()
        s.max_output_chars = 10

        registry._append_output(s, "hello\n")
        registry._append_output(s, "world\nagain")

        assert s.output_buffer == "orld\nagain"
        assert s.output_total_chars == len("hello\nworld\nagain")
        assert s.output_total_lines == 2
        assert s.output_buffer_chars == 10
        assert s.buffer_truncated is True
        assert s.output_dropped_chars == len("hello\nworld\nagain") - 10

    def test_append_output_merges_partial_lines_across_chunks(self, registry):
        s = _make_session()

        registry._append_output(s, "abc")
        registry._append_output(s, "def\n")

        assert s.output_buffer == "abcdef\n"
        assert s.output_total_lines == 1
        assert s.output_total_chars == len("abcdef\n")

    def test_append_output_carriage_return_refresh_is_not_new_line(self, registry):
        s = _make_session()

        registry._append_output(s, "progress 1\rprogress 2\r")

        assert s.output_total_lines == 0
        assert s.output_buffer == "progress 1\rprogress 2\r"

    def test_diff_flood_detection_triggers_across_chunks(self, registry):
        s = _make_session()
        chunks = []
        for file_no in range(3):
            lines = [
                f"diff --git a/file{file_no}.py b/file{file_no}.py\n",
                "index 1111111..2222222 100644\n",
                f"--- a/file{file_no}.py\n",
                f"+++ b/file{file_no}.py\n",
                "@@ -1,20 +1,20 @@\n",
            ]
            for i in range(20):
                lines.append(f"-old line {file_no}-{i}\n")
                lines.append(f"+new line {file_no}-{i}\n")
            chunks.append("".join(lines))

        for chunk in chunks:
            registry._append_output(s, chunk)

        assert s.diff_flood_detected is True
        assert s.diff_flood_score > 0
        assert s.diff_flood_first_seen_at > 0

    def test_diff_flood_detection_ignores_normal_short_output(self, registry):
        s = _make_session()

        registry._append_output(s, "collected 12 items\n...........\n12 passed\n")

        assert s.diff_flood_detected is False
        assert s.diff_flood_score == 0.0

    def test_diff_flood_detection_ignores_markdown_bullet_flood(self, registry):
        s = _make_session()

        registry._append_output(s, "".join(f"- checklist item {i}\n" for i in range(80)))

        assert s.diff_flood_detected is False
        assert s.diff_flood_score > 0

    def test_diff_flood_detection_ignores_markdown_rule_and_bullets(self, registry):
        s = _make_session()

        text = "--- release notes\n" + "".join(f"- checklist item {i}\n" for i in range(80))
        registry._append_output(s, text)

        assert s.diff_flood_detected is False
        assert s.diff_flood_score > 0

    def test_diff_flood_detection_ignores_markdown_plus_minus_sections(self, registry):
        s = _make_session()

        text = (
            "+++ added section\n"
            "-- removed section\n"
            + "".join(f"- bullet item {i}\n" for i in range(80))
        )
        registry._append_output(s, text)

        assert s.diff_flood_detected is False
        assert s.diff_flood_score > 0

    def test_poll_wait_read_log_and_list_include_output_metadata(self, registry, monkeypatch):
        s = _make_session(exited=True, exit_code=0)
        registry._append_output(s, "line 1\nline 2\n")
        registry._finished[s.id] = s
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")

        poll = registry.poll(s.id)
        wait = registry.wait(s.id, timeout=1)
        log = registry.read_log(s.id)
        entry = registry.list_sessions()[0]

        for result in (poll, wait, log, entry):
            assert result["output_total_chars"] == len("line 1\nline 2\n")
            assert result["output_total_lines"] == 2
            assert result["output_buffer_chars"] == len("line 1\nline 2\n")
            assert result["buffer_truncated"] is False
            assert result["output_dropped_chars"] == 0
            assert result["diff_flood_detected"] is False
            assert result["diff_flood_score"] == 0.0
            assert result["diff_flood_first_seen_at"] is None
            assert "returned_chars" in result
        assert log["source"] == "rolling_buffer"

    def test_completion_notification_says_output_tail_only(self, registry):
        s = _make_session(exited=True, exit_code=0)
        s.notify_on_complete = True
        registry._running[s.id] = s
        registry._append_output(s, "done\n")

        registry._move_to_finished(s)
        event, text = registry.drain_notifications()[0]

        assert event["source"] == "rolling_buffer"
        assert "Output tail only (not full output):" in text
        assert "done" in text

    def test_concurrent_append_output_sanity(self, registry):
        s = _make_session()
        chunks = [f"thread-{i}\n" for i in range(20)]

        def worker(chunk):
            for _ in range(50):
                registry._append_output(s, chunk)

        threads = [threading.Thread(target=worker, args=(chunk,)) for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        expected_chars = sum(len(chunk) * 50 for chunk in chunks)
        assert s.output_total_chars == expected_chars
        assert s.output_total_lines == 20 * 50
        assert s.output_buffer_chars == len(s.output_buffer)
        assert s.output_dropped_chars + s.output_buffer_chars == s.output_total_chars


# =========================================================================
# Get / Poll
# =========================================================================

class TestGetAndPoll:
    def test_get_not_found(self, registry):
        assert registry.get("nonexistent") is None

    def test_get_running(self, registry):
        s = _make_session()
        registry._running[s.id] = s
        assert registry.get(s.id) is s

    def test_get_finished(self, registry):
        s = _make_session(exited=True, exit_code=0)
        registry._finished[s.id] = s
        assert registry.get(s.id) is s

    def test_poll_not_found(self, registry):
        result = registry.poll("nonexistent")
        assert result["status"] == "not_found"

    def test_poll_running(self, registry):
        s = _make_session(output="some output here")
        registry._running[s.id] = s
        result = registry.poll(s.id)
        assert result["status"] == "running"
        assert "some output" in result["output_preview"]
        assert result["command"] == "echo hello"

    def test_poll_exited(self, registry):
        s = _make_session(exited=True, exit_code=0, output="done")
        registry._finished[s.id] = s
        result = registry.poll(s.id)
        assert result["status"] == "exited"
        assert result["exit_code"] == 0


# =========================================================================
# Orphaned-pipe reconciliation (issue #17327)
# =========================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: uses setsid/fcntl")
@pytest.mark.live_system_guard_bypass
class TestOrphanedPipeReconciliation:
    """Regression tests for issue #17327.

    `hermes update` in Feishu spawned a background subprocess that restarted
    the gateway; the direct child exited quickly but a descendant daemon
    held the stdout pipe open. `_reader_loop.finally` never ran, so
    `session.exited` stayed False and the agent polled 74 times over 7
    minutes, all returning `status: running`.

    The fix is `_reconcile_local_exit()`: poll() and wait() now check the
    direct `Popen.poll()` before trusting `session.exited`.
    """

    def test_reconcile_flips_exited_when_direct_child_done(self, registry):
        """Direct child exited but reader thread is blocked on orphaned pipe."""
        # Simulate the orphaned-pipe scenario: direct child exited, but a
        # descendant holds stdout open so the reader never sees EOF.
        # Approach: spawn `sh -c 'sleep 10 &'` with setsid — sh forks the
        # sleep into a new session group, exits immediately, but sleep
        # inherits the stdout pipe and keeps it open.
        proc = subprocess.Popen(
            ["sh", "-c", "exec 1>&2; ( sleep 30 ) & disown; exit 0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        s = _make_session(sid="proc_orphan_test")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        # Wait for the direct child to exit. We don't start a reader thread,
        # so session.exited stays False (mimicking the stuck-reader state).
        assert _wait_until(lambda: proc.poll() is not None, timeout=5.0), (
            "Direct child should exit quickly (sh exits, sleep descendant "
            "holds the pipe open)"
        )

        # Before the fix: poll would return "running" forever.
        # After the fix: poll reconciles against proc.poll() and flips.
        assert s.exited is False  # Precondition: reader hasn't updated it.
        result = registry.poll(s.id)
        assert result["status"] == "exited", (
            f"Expected reconciled 'exited' status; got {result!r}. "
            "This is issue #17327 — reader is blocked on orphaned pipe."
        )
        assert result["exit_code"] == 0
        assert s.exited is True
        assert s.id in registry._finished
        assert s.id not in registry._running

        # Clean up the orphaned descendant.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def test_reconcile_noop_when_child_still_running(self, registry):
        """Reconcile must NOT flip exited when the direct child is alive."""
        proc = _spawn_python_sleep(5.0)
        s = _make_session(sid="proc_running_test")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        result = registry.poll(s.id)
        assert result["status"] == "running"
        assert s.exited is False

        proc.kill()
        proc.wait()

    def test_reconcile_noop_on_already_exited(self, registry):
        """Reconcile is a no-op when session.exited is already True."""
        s = _make_session(sid="proc_already_exited", exited=True, exit_code=7)
        s.process = MagicMock()
        s.process.poll = MagicMock(return_value=0)  # Would say exit 0
        registry._finished[s.id] = s

        registry._reconcile_local_exit(s)
        # Must not overwrite the existing exit_code with proc.poll()'s 0.
        assert s.exit_code == 7

    def test_reconcile_noop_on_no_process(self, registry):
        """Reconcile is a no-op for sessions without a local Popen (env/PTY)."""
        s = _make_session(sid="proc_no_popen")
        assert getattr(s, "process", None) is None
        # Must not raise.
        registry._reconcile_local_exit(s)
        assert s.exited is False

    def test_wait_returns_when_reader_blocked(self, registry):
        """wait() must also reconcile — not just poll()."""
        proc = subprocess.Popen(
            ["sh", "-c", "( sleep 30 ) & disown; exit 0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        s = _make_session(sid="proc_wait_orphan")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        assert _wait_until(lambda: proc.poll() is not None, timeout=5.0)

        start = time.monotonic()
        result = registry.wait(s.id, timeout=10)
        elapsed = time.monotonic() - start

        assert result["status"] == "exited", result
        assert elapsed < 5.0, (
            f"wait() should return ~immediately via reconcile; took {elapsed:.1f}s"
        )

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# =========================================================================
# Read log
# =========================================================================

class TestReadLog:
    def test_not_found(self, registry):
        result = registry.read_log("nonexistent")
        assert result["status"] == "not_found"

    def test_read_full_log(self, registry):
        lines = "\n".join([f"line {i}" for i in range(50)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id)
        assert result["total_lines"] == 50

    def test_read_with_limit(self, registry):
        lines = "\n".join([f"line {i}" for i in range(100)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id, limit=10)
        # Default: last 10 lines
        assert "10 lines" in result["showing"]

    def test_read_with_offset(self, registry):
        lines = "\n".join([f"line {i}" for i in range(100)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id, offset=10, limit=5)
        assert "5 lines" in result["showing"]


# =========================================================================
# Stdin helpers
# =========================================================================

class TestStdinHelpers:
    def test_close_stdin_not_found(self, registry):
        result = registry.close_stdin("nonexistent")
        assert result["status"] == "not_found"

    def test_close_stdin_pipe_mode(self, registry):
        proc = MagicMock()
        proc.stdin = MagicMock()
        s = _make_session()
        s.process = proc
        registry._running[s.id] = s

        result = registry.close_stdin(s.id)

        proc.stdin.close.assert_called_once()
        assert result["status"] == "ok"

    def test_close_stdin_pty_mode(self, registry):
        pty = MagicMock()
        s = _make_session()
        s._pty = pty
        registry._running[s.id] = s

        result = registry.close_stdin(s.id)

        pty.sendeof.assert_called_once()
        assert result["status"] == "ok"

    def test_close_stdin_allows_eof_driven_process_to_finish(self, registry, tmp_path):
        """PTY mode: writing data + sending EOF lets an EOF-driven child finish.

        Background non-PTY mode used to expose subprocess stdin via a pipe,
        but PR #214b95392 detached non-PTY stdin to DEVNULL to fix keyboard
        lockout (#17959). For interactive stdin → PTY mode is now the only
        supported path.
        """
        pytest.importorskip("ptyprocess", reason="PTY process tests require ptyprocess")
        session = registry.spawn_local(
            'python3 -c "import sys; print(sys.stdin.read().strip())"',
            cwd=str(tmp_path),
            use_pty=True,
        )

        try:
            time.sleep(0.5)
            assert registry.submit_stdin(session.id, "hello")["status"] == "ok"
            assert registry.close_stdin(session.id)["status"] == "ok"

            deadline = time.time() + 5
            while time.time() < deadline:
                poll = registry.poll(session.id)
                if poll["status"] == "exited":
                    assert poll["exit_code"] == 0
                    assert "hello" in poll["output_preview"]
                    return
                time.sleep(0.2)

            pytest.fail("process did not exit after stdin was closed")
        finally:
            registry.kill_process(session.id)


# =========================================================================
# List sessions
# =========================================================================

class TestListSessions:
    def test_empty(self, registry):
        assert registry.list_sessions() == []

    def test_lists_running_and_finished(self, registry):
        s1 = _make_session(sid="proc_1", task_id="t1")
        s2 = _make_session(sid="proc_2", task_id="t1", exited=True, exit_code=0)
        registry._running[s1.id] = s1
        registry._finished[s2.id] = s2
        result = registry.list_sessions()
        assert len(result) == 2

    def test_filter_by_task_id(self, registry):
        s1 = _make_session(sid="proc_1", task_id="t1")
        s2 = _make_session(sid="proc_2", task_id="t2")
        registry._running[s1.id] = s1
        registry._running[s2.id] = s2
        result = registry.list_sessions(task_id="t1")
        assert len(result) == 1
        assert result[0]["session_id"] == "proc_1"

    def test_list_entry_fields(self, registry):
        s = _make_session(output="preview text")
        registry._running[s.id] = s
        entry = registry.list_sessions()[0]
        assert "session_id" in entry
        assert "command" in entry
        assert "status" in entry
        assert "pid" in entry
        assert "output_preview" in entry

    def test_list_codex_process_uses_context_safe_preview(self, registry):
        raw = "diff --git a/x.py b/x.py\n@@\n+SECRET_SOURCE_LINE\n" * 80
        s = _make_session(command="codex-yuna exec --full-auto 'review'", output=raw)
        registry._running[s.id] = s

        entry = registry.list_sessions()[0]

        assert entry["context_safe_summary"] is True
        assert entry["raw_log_available_via_process_log"] is True
        assert "Codex output suppressed for context safety" in entry["output_preview"]
        assert "SECRET_SOURCE_LINE" not in entry["output_preview"]


# =========================================================================
# Active process queries
# =========================================================================

class TestActiveQueries:
    def test_has_active_processes(self, registry):
        s = _make_session(task_id="t1")
        registry._running[s.id] = s
        assert registry.has_active_processes("t1") is True
        assert registry.has_active_processes("t2") is False

    def test_has_active_for_session(self, registry):
        s = _make_session()
        s.session_key = "gw_session_1"
        registry._running[s.id] = s
        assert registry.has_active_for_session("gw_session_1") is True
        assert registry.has_active_for_session("other") is False

    def test_exited_not_active(self, registry):
        s = _make_session(task_id="t1", exited=True, exit_code=0)
        registry._finished[s.id] = s
        assert registry.has_active_processes("t1") is False


# =========================================================================
# Pruning
# =========================================================================

class TestPruning:
    def test_prune_expired_finished(self, registry):
        old_session = _make_session(
            sid="proc_old",
            exited=True,
            started_at=time.time() - FINISHED_TTL_SECONDS - 100,
        )
        registry._finished[old_session.id] = old_session
        registry._prune_if_needed()
        assert "proc_old" not in registry._finished

    def test_prune_keeps_recent(self, registry):
        recent = _make_session(sid="proc_recent", exited=True)
        registry._finished[recent.id] = recent
        registry._prune_if_needed()
        assert "proc_recent" in registry._finished

    def test_prune_over_max_removes_oldest(self, registry):
        # Fill up to MAX_PROCESSES
        for i in range(MAX_PROCESSES):
            s = _make_session(
                sid=f"proc_{i}",
                exited=True,
                started_at=time.time() - i,  # older as i increases
            )
            registry._finished[s.id] = s

        # Add one more running to trigger prune
        s = _make_session(sid="proc_new")
        registry._running[s.id] = s
        registry._prune_if_needed()

        total = len(registry._running) + len(registry._finished)
        assert total <= MAX_PROCESSES


# =========================================================================
# Spawn env sanitization
# =========================================================================

class TestSpawnEnvSanitization:
    def test_spawn_local_strips_blocked_vars_from_background_env(self, registry):
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["env"] = kwargs["env"]
            proc = MagicMock()
            proc.pid = 4321
            proc.stdout = iter([])
            proc.stdin = MagicMock()
            proc.poll.return_value = None
            return proc

        fake_thread = MagicMock()

        with patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "USER": "tester",
            "TELEGRAM_BOT_TOKEN": "bot-secret",
            "FIRECRAWL_API_KEY": "fc-secret",
        }, clear=True), \
            patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
            patch("subprocess.Popen", side_effect=fake_popen), \
            patch("threading.Thread", return_value=fake_thread), \
            patch.object(registry, "_write_checkpoint"):
            registry.spawn_local(
                "echo hello",
                cwd="/tmp",
                env_vars={
                    "MY_CUSTOM_VAR": "keep-me",
                    "TELEGRAM_BOT_TOKEN": "drop-me",
                    f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}TELEGRAM_BOT_TOKEN": "forced-bot-token",
                },
            )

        env = captured["env"]
        assert env["MY_CUSTOM_VAR"] == "keep-me"
        assert env["TELEGRAM_BOT_TOKEN"] == "forced-bot-token"
        assert "FIRECRAWL_API_KEY" not in env
        assert f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}TELEGRAM_BOT_TOKEN" not in env
        assert env["PYTHONUNBUFFERED"] == "1"

    def test_spawn_via_env_uses_backend_temp_dir_for_artifacts(self, registry):
        class FakeEnv:
            def __init__(self):
                self.commands = []

            def get_temp_dir(self):
                return "/data/data/com.termux/files/usr/tmp"

            def execute(self, command, timeout=None):
                self.commands.append((command, timeout))
                return {"output": "4321\n"}

        env = FakeEnv()
        fake_thread = MagicMock()

        with patch("tools.process_registry.threading.Thread", return_value=fake_thread), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_via_env(env, "echo hello")

        bg_command = env.commands[0][0]
        assert session.pid == 4321
        assert "/data/data/com.termux/files/usr/tmp/hermes_bg_" in bg_command
        assert ".exit" in bg_command
        assert "rc=$?;" in bg_command
        assert " > /tmp/hermes_bg_" not in bg_command
        assert "cat /tmp/hermes_bg_" not in bg_command
        fake_thread.start.assert_called_once()

    def test_env_poller_quotes_temp_paths_with_spaces(self, registry):
        session = _make_session(sid="proc_space")
        session.exited = False

        class FakeEnv:
            def __init__(self):
                self.commands = []
                self._responses = iter([
                    {"output": "hello\n"},
                    {"output": "1\n"},
                    {"output": "0\n"},
                ])

            def execute(self, command, timeout=None):
                self.commands.append((command, timeout))
                return next(self._responses)

        env = FakeEnv()

        with patch("tools.process_registry.time.sleep", return_value=None), \
            patch.object(registry, "_move_to_finished"):
            registry._env_poller_loop(
                session,
                env,
                "/path with spaces/hermes_bg.log",
                "/path with spaces/hermes_bg.pid",
                "/path with spaces/hermes_bg.exit",
            )

        assert env.commands[0][0] == "cat '/path with spaces/hermes_bg.log' 2>/dev/null"
        assert env.commands[1][0] == "kill -0 \"$(cat '/path with spaces/hermes_bg.pid' 2>/dev/null)\" 2>/dev/null; echo $?"
        assert env.commands[2][0] == "cat '/path with spaces/hermes_bg.exit' 2>/dev/null"


# =========================================================================
# Popen leak prevention
# =========================================================================

class TestPopenLeakOnSetupFailure:
    """Regression for issue #2749: subprocess orphaned when post-Popen setup raises."""

    def test_popen_killed_when_thread_creation_fails(self, registry):
        """If Thread() raises after Popen, proc must be killed — not orphaned."""
        killed = []

        proc = MagicMock()
        proc.pid = 9999
        proc.stdout = iter([])
        proc.stdin = MagicMock()
        proc.poll.return_value = None

        def fake_kill():
            killed.append(True)

        proc.kill = fake_kill
        proc.wait = MagicMock()

        def boom(*args, **kwargs):
            raise RuntimeError("Thread creation failed")

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("threading.Thread", side_effect=boom), \
             patch.object(registry, "_write_checkpoint"):
            with pytest.raises(RuntimeError, match="Thread creation failed"):
                registry.spawn_local("echo hello", cwd="/tmp")

        assert killed, "proc.kill() must be called when post-Popen setup raises"

    def test_popen_killed_when_write_checkpoint_fails(self, registry):
        """If _write_checkpoint raises after Popen, proc must still be killed."""
        killed = []

        proc = MagicMock()
        proc.pid = 8888
        proc.stdout = iter([])
        proc.stdin = MagicMock()
        proc.poll.return_value = None

        def fake_kill():
            killed.append(True)

        proc.kill = fake_kill
        proc.wait = MagicMock()

        fake_thread = MagicMock()

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                registry.spawn_local("echo hello", cwd="/tmp")

        assert killed, "proc.kill() must be called when _write_checkpoint raises"

    def test_popen_not_killed_on_success(self, registry):
        """Successful spawn must NOT kill the process."""
        killed = []

        proc = MagicMock()
        proc.pid = 7777
        proc.stdout = iter([])
        proc.stdin = MagicMock()
        proc.poll.return_value = None

        def fake_kill():
            killed.append(True)

        proc.kill = fake_kill
        proc.wait = MagicMock()

        fake_thread = MagicMock()

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_local("echo hello", cwd="/tmp")

        assert not killed, "proc.kill() must NOT be called on successful spawn"
        assert session.pid == 7777


# =========================================================================
# Checkpoint
# =========================================================================

class TestCheckpoint:
    def test_write_checkpoint(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            s = _make_session()
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads((tmp_path / "procs.json").read_text())
            assert len(data) == 1
            assert data[0]["session_id"] == s.id

    def test_recover_no_file(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "missing.json"):
            assert registry.recover_from_checkpoint() == 0

    def test_recover_dead_pid(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_dead",
            "command": "sleep 999",
            "pid": 999999999,  # almost certainly not running
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), \
             patch.object(registry, "_is_host_pid_alive", return_value=False):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0

    def test_write_checkpoint_includes_watcher_metadata(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            s = _make_session()
            s.watcher_platform = "telegram"
            s.watcher_chat_id = "999"
            s.watcher_user_id = "u123"
            s.watcher_user_name = "alice"
            s.watcher_thread_id = "42"
            s.watcher_interval = 60
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads((tmp_path / "procs.json").read_text())
            assert len(data) == 1
            assert data[0]["watcher_platform"] == "telegram"
            assert data[0]["watcher_chat_id"] == "999"
            assert data[0]["watcher_user_id"] == "u123"
            assert data[0]["watcher_user_name"] == "alice"
            assert data[0]["watcher_thread_id"] == "42"
            assert data[0]["watcher_interval"] == 60

    def test_recover_enqueues_watchers(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),  # current process — guaranteed alive
            "task_id": "t1",
            "session_key": "sk1",
            "watcher_platform": "telegram",
            "watcher_chat_id": "123",
            "watcher_user_id": "u123",
            "watcher_user_name": "alice",
            "watcher_thread_id": "42",
            "watcher_interval": 60,
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            assert len(registry.pending_watchers) == 1
            w = registry.pending_watchers[0]
            assert w["session_id"] == "proc_live"
            assert w["platform"] == "telegram"
            assert w["chat_id"] == "123"
            assert w["user_id"] == "u123"
            assert w["user_name"] == "alice"
            assert w["thread_id"] == "42"
            assert w["check_interval"] == 60

    def test_recover_skips_watcher_when_no_interval(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "watcher_interval": 0,
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            assert len(registry.pending_watchers) == 0

    def test_recovery_keeps_live_checkpoint_entries(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "session_key": "sk1",
        }]))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            assert registry.get("proc_live") is not None

            data = json.loads(checkpoint.read_text())
            assert len(data) == 1
            assert data[0]["session_id"] == "proc_live"
            assert data[0]["pid"] == os.getpid()
            assert data != []

    def test_recovery_skips_explicit_sandbox_backed_entries(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        original = [{
            "session_id": "proc_remote",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "pid_scope": "sandbox",
        }]
        checkpoint.write_text(json.dumps(original))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0
            assert registry.get("proc_remote") is None

            data = json.loads(checkpoint.read_text())
            assert data == []

    def test_detached_recovered_process_eventually_exits(self, registry, tmp_path):
        proc = _spawn_python_sleep(0.4)
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "python -c 'import time; time.sleep(0.4)'",
            "pid": proc.pid,
            "task_id": "t1",
            "session_key": "sk1",
        }]))

        try:
            with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), \
                 patch.object(registry, "_is_host_pid_alive", side_effect=lambda _pid: proc.poll() is None):
                recovered = registry.recover_from_checkpoint()
                assert recovered == 1

                session = registry.get("proc_live")
                assert session is not None
                assert session.detached is True

                proc.wait(timeout=5)

                assert _wait_until(
                    lambda: registry.get("proc_live") is not None
                    and registry.get("proc_live").exited,
                    timeout=5,
                )

                poll_result = registry.poll("proc_live")
                assert poll_result["status"] == "exited"

                wait_result = registry.wait("proc_live", timeout=1)
                assert wait_result["status"] == "exited"
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=5)


# =========================================================================
# Kill process
# =========================================================================

class TestKillProcess:
    def test_kill_not_found(self, registry):
        result = registry.kill_process("nonexistent")
        assert result["status"] == "not_found"

    def test_kill_already_exited(self, registry):
        s = _make_session(exited=True, exit_code=0)
        registry._finished[s.id] = s
        result = registry.kill_process(s.id)
        assert result["status"] == "already_exited"

    def test_kill_detached_session_uses_host_pid(self, registry):
        s = _make_session(sid="proc_detached", command="sleep 999")
        s.pid = 424242
        s.detached = True
        registry._running[s.id] = s

        terminate_calls = []

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
            def children(self, recursive=False):
                return []
            def terminate(self):
                terminate_calls.append(("terminate", self.pid))

        try:
            # Post-#21561: liveness probe routes through
            # ``ProcessRegistry._is_host_pid_alive`` (→
            # ``gateway.status._pid_exists``), and the actual kill routes
            # through ``ProcessRegistry._terminate_host_pid``. Mock both seams
            # so the test does not depend on optional psutil being installed.
            with patch("gateway.status._pid_exists", return_value=True), \
                 patch("tools.process_registry.ProcessRegistry._terminate_host_pid", return_value={"method": "psutil", "fallback_used": False}):
                result = registry.kill_process(s.id)

            assert result["status"] == "killed"
            assert result["termination_method"] == "psutil"
        finally:
            registry._running.pop(s.id, None)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group fallback")
    def test_kill_local_process_falls_back_without_psutil(self, registry, monkeypatch):
        """Missing psutil must not make process(action='kill') unusable."""
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        proc.wait.return_value = -15
        s = _make_session(sid="proc_no_psutil", command="sleep 999")
        s.process = proc
        s.pid = proc.pid
        s.pgid = proc.pid
        registry._running[s.id] = s

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("No module named 'psutil'")
            return real_import(name, *args, **kwargs)

        killpg_calls = []

        def fake_killpg(pgid, sig):
            killpg_calls.append((pgid, sig))

        monkeypatch.setattr("builtins.__import__", fake_import)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", fake_killpg)

        result = registry.kill_process(s.id)

        assert result["status"] == "killed"
        assert result["fallback_used"] is True
        assert result["termination_method"] == "os.killpg"
        assert killpg_calls == [(12345, signal.SIGTERM)]
        assert s.kill_requested is True
        assert s.trusted_completion is False

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group fallback")
    def test_kill_pty_process_prefers_process_group(self, registry, monkeypatch):
        """PTY-backed Codex wrappers should be killed by process group, not just parent PID."""
        pty = MagicMock()
        pty.pid = 23456
        pty.isalive.return_value = True
        s = _make_session(sid="proc_pty", command="codex-yuna exec ...")
        s._pty = pty
        s.pid = pty.pid
        s.pgid = pty.pid
        registry._running[s.id] = s

        killpg_calls = []
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

        result = registry.kill_process(s.id)

        assert result["status"] == "killed"
        assert result["termination_method"] == "os.killpg"
        assert killpg_calls == [(23456, signal.SIGTERM)]
        pty.terminate.assert_not_called()

    def test_codex_kill_after_wait_timeout_requires_force(self, registry):
        """A wait-window timeout must not let Hermes kill Codex by mistake."""
        s = _make_session(sid="proc_codex_guard", command="codex-yuna exec --full-auto task")
        s.last_wait_timeout_at = time.time()
        s.last_wait_timeout_seconds = 60
        registry._running[s.id] = s

        result = registry.kill_process(s.id)

        assert result["status"] == "refused"
        assert result["requires_force"] is True
        assert result["codex_process"] is True
        assert s.exited is False
        assert s.kill_attempted is False
        assert s.trusted_completion is True

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group fallback")
    def test_process_tool_force_false_string_does_not_bypass_codex_guard(self, registry, monkeypatch):
        from tools.process_registry import _handle_process
        import tools.process_registry as process_registry_module

        s = _make_session(sid="proc_force_false_string", command="codex-yuna exec --full-auto task")
        s.last_wait_timeout_at = time.time()
        s.last_wait_timeout_seconds = 60
        registry._running[s.id] = s
        monkeypatch.setattr(process_registry_module, "process_registry", registry)

        result = json.loads(_handle_process({"action": "kill", "session_id": s.id, "force": "false"}))

        assert result["status"] == "refused"
        assert result["requires_force"] is True
        assert s.exited is False
        assert s.kill_attempted is False

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group fallback")
    def test_process_tool_force_true_string_bypasses_codex_guard(self, registry, monkeypatch):
        from tools.process_registry import _handle_process
        import tools.process_registry as process_registry_module

        proc = MagicMock()
        proc.pid = 45679
        s = _make_session(sid="proc_force_true_string", command="codex-yuna exec --full-auto task")
        s.process = proc
        s.pid = proc.pid
        s.pgid = proc.pid
        s.last_wait_timeout_at = time.time()
        s.last_wait_timeout_seconds = 60
        registry._running[s.id] = s
        monkeypatch.setattr(process_registry_module, "process_registry", registry)
        monkeypatch.setattr(
            "tools.process_registry.ProcessRegistry._terminate_host_pid",
            staticmethod(lambda *args, **kwargs: {"method": "os.killpg", "fallback_used": True}),
        )

        result = json.loads(_handle_process({"action": "kill", "session_id": s.id, "force": "true"}))

        assert result["status"] == "killed"
        assert s.kill_attempted is True

    def test_codex_kill_after_wait_timeout_allows_explicit_force(self, registry, monkeypatch):
        """Explicit stop/hard-deadline paths can still terminate Codex."""
        proc = MagicMock()
        proc.pid = 45678
        s = _make_session(sid="proc_codex_force", command="codex-yuna exec --full-auto task")
        s.process = proc
        s.pid = proc.pid
        s.pgid = proc.pid
        s.last_wait_timeout_at = time.time()
        s.last_wait_timeout_seconds = 60
        registry._running[s.id] = s

        monkeypatch.setattr(
            "tools.process_registry.ProcessRegistry._terminate_host_pid",
            staticmethod(lambda *args, **kwargs: {"method": "os.killpg", "fallback_used": True}),
        )

        result = registry.kill_process(s.id, force=True, reason="user requested stop")

        assert result["status"] == "killed"
        assert result["trusted_completion"] is False
        assert s.kill_attempted is True

    def test_scoped_kill_all_respects_codex_wait_timeout_guard(self, registry):
        """Agent close/cache cleanup should not force-kill a healthy Codex worker."""
        s = _make_session(
            sid="proc_codex_scoped_cleanup",
            command="codex-yuna exec --full-auto task",
            task_id="agent-session",
        )
        s.last_wait_timeout_at = time.time()
        s.last_wait_timeout_seconds = 60
        registry._running[s.id] = s

        killed = registry.kill_all(task_id="agent-session")

        assert killed == 0
        assert s.exited is False
        assert s.kill_attempted is False

    def test_global_kill_all_forces_codex_wait_timeout_guard(self, registry, monkeypatch):
        """Explicit global stop/shutdown can still kill guarded Codex workers."""
        proc = MagicMock()
        proc.pid = 56789
        s = _make_session(
            sid="proc_codex_global_stop",
            command="codex-yuna exec --full-auto task",
            task_id="agent-session",
        )
        s.process = proc
        s.pid = proc.pid
        s.last_wait_timeout_at = time.time()
        s.last_wait_timeout_seconds = 60
        registry._running[s.id] = s

        monkeypatch.setattr(
            "tools.process_registry.ProcessRegistry._terminate_host_pid",
            staticmethod(lambda *args, **kwargs: {"method": "os.kill", "fallback_used": True}),
        )

        killed = registry.kill_all()

        assert killed == 1
        assert s.exited is True
        assert s.kill_attempted is True

    def test_kill_failed_records_state(self, registry, monkeypatch):
        proc = MagicMock()
        proc.pid = 34567
        proc.poll.return_value = None
        s = _make_session(sid="proc_kill_error", command="sleep 999")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        def boom(*args, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr("tools.process_registry.ProcessRegistry._terminate_host_pid", staticmethod(boom))

        result = registry.kill_process(s.id)

        assert result["status"] == "error"
        assert s.kill_attempted is True
        assert s.kill_failed is True
        assert "denied" in s.kill_error

    def test_no_runtime_handle_branch_returns_kill_failure_metadata(self, registry):
        s = _make_session(sid="proc_no_handle", command="sleep 999")
        registry._running[s.id] = s

        result = registry.kill_process(s.id)

        assert result["status"] == "error"
        assert result["kill_attempted"] is True
        assert result["kill_requested"] is True
        assert result["kill_failed"] is True
        assert result["trusted_completion"] is False
        assert s.kill_failed is True


class TestWaitTimeoutMetadata:
    def test_wait_clamp_returns_structured_metadata(self, registry, monkeypatch):
        s = _make_session(sid="proc_wait_clamp")
        registry._running[s.id] = s
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")

        result = registry.wait(s.id, timeout=5)

        assert result["status"] == "timeout"
        assert result["requested_timeout"] == 5
        assert result["effective_timeout"] == 1
        assert result["max_wait_timeout"] == 1
        assert result["clamped"] is True

    def test_codex_command_detection_handles_quoted_wrapper_path(self):
        assert ProcessRegistry._is_codex_command('"$HOME/.local/bin/codex" exec --full-auto task')

    def test_codex_command_detection_does_not_match_plain_text_mentions(self):
        assert not ProcessRegistry._is_codex_command("echo codex-yuna")
        assert not ProcessRegistry._is_codex_command("python -c 'print(\"codex-yuna\")'")

    def test_codex_command_detection_malformed_quotes_does_not_raise(self):
        assert not ProcessRegistry._is_codex_command("codex-yuna exec 'unterminated")

    def test_poll_and_list_tolerate_malformed_command_quotes(self, registry):
        s = _make_session(
            sid="proc_malformed_command",
            command="codex-yuna exec 'unterminated",
            exited=True,
            exit_code=2,
        )
        registry._finished[s.id] = s

        poll_result = registry.poll(s.id)
        list_result = registry.list_sessions()

        assert poll_result["status"] == "exited"
        assert poll_result["exit_code"] == 2
        assert list_result[0]["session_id"] == s.id
        assert list_result[0]["status"] == "exited"

    def test_wait_returns_exited_if_process_finishes_at_deadline_boundary(self, registry, monkeypatch):
        s = _make_session(sid="proc_wait_boundary", command="codex-yuna exec --full-auto task")
        registry._running[s.id] = s
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")

        monotonic_values = iter([0.0, 0.0, 2.0])
        monkeypatch.setattr(time, "monotonic", lambda: next(monotonic_values))

        def finish_during_sleep(_seconds):
            with s._lock:
                s.output_buffer = "done\n"
                s.exited = True
                s.exit_code = 0

        monkeypatch.setattr(time, "sleep", finish_during_sleep)

        result = registry.wait(s.id, timeout=1)

        assert result["status"] == "exited"
        assert result["exit_code"] == 0
        assert "Codex output suppressed for context safety" in result["output"]
        assert "raw_log_available_via_process_log=True" in result["output"]
        assert "process_still_running" not in result
        assert not s.last_wait_timeout_at

    def test_wait_timeout_is_marked_as_wait_window_not_failure(self, registry, monkeypatch):
        s = _make_session(sid="proc_wait_window", command="codex-yuna exec --full-auto task")
        registry._running[s.id] = s
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")

        result = registry.wait(s.id, timeout=1)

        assert result["status"] == "timeout"
        assert result["timeout_kind"] == "wait_window_expired"
        assert result["process_still_running"] is True
        assert result["is_failure"] is False
        assert result["codex_guard"] is True
        assert result["codex_process"] is True
        assert result["last_wait_timeout_kind"] == "wait_window_expired"
        assert s.last_wait_timeout_seconds == 1

    def test_exited_after_kill_request_is_not_trusted_completion(self, registry):
        s = _make_session(sid="proc_killed_then_zero", exited=True, exit_code=0)
        s.kill_requested = True
        s.termination_method = "os.killpg"
        registry._finished[s.id] = s

        result = registry.poll(s.id)

        assert result["status"] == "exited"
        assert result["exit_code"] == 0
        assert result["kill_requested"] is True
        assert result["trusted_completion"] is False


# =========================================================================
# Tool handler
# =========================================================================

class TestProcessToolHandler:
    def test_list_action(self):
        from tools.process_registry import _handle_process
        result = json.loads(_handle_process({"action": "list"}))
        assert "processes" in result

    def test_poll_missing_session_id(self):
        from tools.process_registry import _handle_process
        result = json.loads(_handle_process({"action": "poll"}))
        assert "error" in result

    def test_unknown_action(self):
        from tools.process_registry import _handle_process
        result = json.loads(_handle_process({"action": "unknown_action"}))
        assert "error" in result


# =========================================================================
# format_process_notification + drain_notifications (shared helpers)
# =========================================================================

from tools.process_registry import format_process_notification


def test_format_completion_event():
    evt = {
        "type": "completion",
        "session_id": "proc_abc",
        "command": "sleep 5",
        "exit_code": 0,
        "output": "done",
    }
    result = format_process_notification(evt)
    assert "[IMPORTANT: Background process proc_abc completed" in result
    assert "exit code 0" in result
    assert "Command: sleep 5" in result
    assert "Output tail only (not full output):\ndone]" in result


def test_format_codex_completion_event_uses_context_safe_summary():
    evt = {
        "type": "completion",
        "session_id": "proc_codex",
        "command": "codex-yuna exec --full-auto 'review'",
        "exit_code": 0,
        "output": "diff --git a/x.py b/x.py\n@@\n+SECRET_SOURCE_LINE\n" * 80,
        "stdout_chars": 9999,
        "stdout_lines": 240,
        "diff_flood_detected": True,
    }

    result = format_process_notification(evt)

    assert "Context-safe Codex summary:" in result
    assert "Codex output suppressed for context safety" in result
    assert "stdout_chars=9999" in result
    assert "raw_log_available_via_process_log=True" in result
    assert "SECRET_SOURCE_LINE" not in result


def test_poll_codex_process_returns_summary_but_log_keeps_raw(registry):
    raw = "diff --git a/x.py b/x.py\n@@\n+SECRET_SOURCE_LINE\n" * 80
    s = _make_session(command="codex-yuna exec --full-auto 'implement'", output=raw)
    registry._running[s.id] = s

    poll = registry.poll(s.id)
    log = registry.read_log(s.id, limit=20)

    assert poll["context_safe_summary"] is True
    assert poll["raw_log_available_via_process_log"] is True
    assert "Codex output suppressed for context safety" in poll["output_preview"]
    assert "SECRET_SOURCE_LINE" not in poll["output_preview"]
    assert "SECRET_SOURCE_LINE" in log["output"]


def test_format_completion_event_after_kill_request_is_not_plain_completed():
    evt = {
        "type": "completion",
        "session_id": "proc_killed",
        "command": "codex-yuna exec ...",
        "exit_code": 0,
        "output": "",
        "kill_requested": True,
        "trusted_completion": False,
        "termination_method": "os.killpg",
    }

    result = format_process_notification(evt)

    assert "exited after a kill/termination request" in result
    assert "trusted_completion=false" in result
    assert "completed (exit code 0)" not in result


def test_format_watch_match_event():
    evt = {
        "type": "watch_match",
        "session_id": "proc_xyz",
        "command": "tail -f log",
        "pattern": "ERROR",
        "output": "ERROR: disk full",
        "suppressed": 0,
    }
    result = format_process_notification(evt)
    assert 'watch pattern "ERROR"' in result
    assert "Matched output:\nERROR: disk full" in result


def test_format_watch_match_with_suppressed():
    evt = {
        "type": "watch_match",
        "session_id": "proc_xyz",
        "command": "tail -f log",
        "pattern": "WARN",
        "output": "WARN: low mem",
        "suppressed": 3,
    }
    result = format_process_notification(evt)
    assert "3 earlier matches were suppressed" in result


def test_format_watch_disabled_event():
    evt = {
        "type": "watch_disabled",
        "message": "Watch disabled for proc_xyz: too many matches",
    }
    result = format_process_notification(evt)
    assert "[IMPORTANT: Watch disabled for proc_xyz" in result


def test_format_returns_none_for_empty_event():
    evt = {}
    result = format_process_notification(evt)
    assert result is not None
    assert "unknown" in result


def test_drain_notifications_returns_pending_events():
    from tools.process_registry import process_registry

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    process_registry.completion_queue.put({
        "type": "completion",
        "session_id": "proc_drain1",
        "command": "echo hi",
        "exit_code": 0,
        "output": "hi",
    })
    process_registry.completion_queue.put({
        "type": "watch_match",
        "session_id": "proc_drain2",
        "command": "tail -f x",
        "pattern": "ERR",
        "output": "ERR found",
        "suppressed": 0,
    })

    try:
        results = process_registry.drain_notifications()
        assert len(results) == 2
        assert results[0][0]["session_id"] == "proc_drain1"
        assert "proc_drain1 completed" in results[0][1]
        assert results[1][0]["session_id"] == "proc_drain2"
        assert "watch pattern" in results[1][1]
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
        process_registry._completion_consumed.discard("proc_drain1")
        process_registry._completion_consumed.discard("proc_drain2")


def test_drain_notifications_skips_consumed():
    from tools.process_registry import process_registry

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    process_registry._completion_consumed.add("proc_consumed")
    process_registry.completion_queue.put({
        "type": "completion",
        "session_id": "proc_consumed",
        "command": "echo done",
        "exit_code": 0,
        "output": "done",
    })

    try:
        results = process_registry.drain_notifications()
        assert len(results) == 0
    finally:
        process_registry._completion_consumed.discard("proc_consumed")
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_drain_notifications_empty_queue():
    from tools.process_registry import process_registry

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    results = process_registry.drain_notifications()
    assert results == []


# ---------------------------------------------------------------------------
# _terminate_host_pid — cross-platform process-tree termination
# ---------------------------------------------------------------------------


class TestTerminateHostPidWindows:
    """Windows branch uses ``taskkill /T /F`` — the documented MS tree-kill
    primitive. We can't use psutil's ``children(recursive=True)`` /
    ``.terminate()`` path on Windows because (1) Windows doesn't maintain
    a Unix-style process tree so the walk is unreliable, and (2)
    ``Process.terminate()`` on Windows is ``TerminateProcess()`` for the
    target handle only, not the tree.
    """

    def test_windows_invokes_taskkill_with_tree_and_force_flags(self, monkeypatch):
        """The Windows branch must shell out to ``taskkill /PID N /T /F``."""
        from tools import process_registry as pr

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(pr, "_IS_WINDOWS", True)
        monkeypatch.setattr(pr.subprocess, "run", fake_run)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert captured["args"][0] == "taskkill"
        assert "/PID" in captured["args"]
        assert "12345" in captured["args"]
        assert "/T" in captured["args"], "Tree flag required to reach descendants"
        assert "/F" in captured["args"], "Force flag required for headless Chromium"

    def test_windows_falls_back_to_os_kill_when_taskkill_missing(self, monkeypatch):
        """If ``taskkill.exe`` is somehow unavailable, fall back to a bare
        ``os.kill(pid, SIGTERM)`` so we at least try to kill the parent."""
        from tools import process_registry as pr

        kill_calls = []

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("taskkill not found")

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))

        monkeypatch.setattr(pr, "_IS_WINDOWS", True)
        monkeypatch.setattr(pr.subprocess, "run", fake_run)
        monkeypatch.setattr(pr.os, "kill", fake_kill)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert kill_calls == [(12345, signal.SIGTERM)]

    def test_windows_falls_back_to_os_kill_when_taskkill_returns_nonzero(self, monkeypatch):
        """taskkill can fail with a non-zero return code instead of raising."""
        from tools import process_registry as pr

        kill_calls = []

        def fake_run(*args, **kwargs):
            return MagicMock(returncode=1, stderr="Access is denied", stdout="")

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))

        monkeypatch.setattr(pr, "_IS_WINDOWS", True)
        monkeypatch.setattr(pr.subprocess, "run", fake_run)
        monkeypatch.setattr(pr.os, "kill", fake_kill)

        result = pr.ProcessRegistry._terminate_host_pid(12345)

        assert kill_calls == [(12345, signal.SIGTERM)]
        assert result["method"] == "os.kill"
        assert result["fallback_used"] is True
        assert "Access is denied" in result["fallback_error"]

    def test_windows_does_not_call_psutil(self, monkeypatch):
        """The Windows branch must NOT exercise the psutil tree-walk
        (it's unreliable on Windows — see the function docstring)."""
        from tools import process_registry as pr

        psutil_calls = []

        class _BoomProcess:
            def __init__(self, pid):
                psutil_calls.append(("Process", pid))

            def children(self, recursive=False):
                psutil_calls.append(("children", recursive))
                return []

            def terminate(self):
                psutil_calls.append(("terminate",))

        fake_psutil = types.SimpleNamespace(Process=_BoomProcess, NoSuchProcess=Exception)

        def fake_run(args, **kwargs):
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(pr, "_IS_WINDOWS", True)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        monkeypatch.setattr(pr.subprocess, "run", fake_run)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert psutil_calls == [], (
            f"Windows branch must not touch psutil, but saw {psutil_calls!r}"
        )


class TestTerminateHostPidPosix:
    """POSIX branch walks the tree via psutil and SIGTERMs children first."""

    def test_posix_walks_tree_and_terminates_children_then_parent(self, monkeypatch):
        from tools import process_registry as pr

        terminate_order = []

        class _FakeChild:
            def __init__(self, pid):
                self.pid = pid

            def terminate(self):
                terminate_order.append(self.pid)

        class _FakeParent:
            def __init__(self, pid):
                self.pid = pid

            def children(self, recursive=False):
                assert recursive is True
                return [_FakeChild(101), _FakeChild(102), _FakeChild(103)]

            def terminate(self):
                terminate_order.append(self.pid)

        class _NoSuchProcess(Exception):
            pass

        fake_psutil = types.SimpleNamespace(Process=_FakeParent, NoSuchProcess=_NoSuchProcess)
        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        result = pr.ProcessRegistry._terminate_host_pid(12345)

        assert result["method"] == "psutil"
        assert terminate_order == [101, 102, 103, 12345], (
            "Children must be terminated before the parent"
        )

    def test_posix_no_such_process_swallowed(self, monkeypatch):
        from tools import process_registry as pr

        class _NoSuchProcess(Exception):
            pass

        def boom(pid):
            raise _NoSuchProcess(pid)

        fake_psutil = types.SimpleNamespace(Process=boom, NoSuchProcess=_NoSuchProcess)
        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        # Must not raise.
        result = pr.ProcessRegistry._terminate_host_pid(999999999)
        assert result["method"] == "psutil.no_such_process"

    def test_posix_oserror_default_fallback_kills_only_pid(self, monkeypatch):
        from tools import process_registry as pr

        class _NoSuchProcess(Exception):
            pass

        def boom(pid):
            raise PermissionError("can't read /proc")

        fake_psutil = types.SimpleNamespace(Process=boom, NoSuchProcess=_NoSuchProcess)
        killpg_calls = []
        kill_calls = []

        def fake_killpg(pgid, sig):
            killpg_calls.append((pgid, sig))

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))

        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        monkeypatch.setattr(pr.os, "getpgid", lambda pid: 99999)
        monkeypatch.setattr(pr.os, "killpg", fake_killpg)
        monkeypatch.setattr(pr.os, "kill", fake_kill)

        result = pr.ProcessRegistry._terminate_host_pid(12345)

        assert result["method"] == "os.kill"
        assert killpg_calls == []
        assert kill_calls == [(12345, signal.SIGTERM)]

    def test_posix_oserror_allows_process_group_only_for_isolated_group_leader(self, monkeypatch):
        from tools import process_registry as pr

        class _NoSuchProcess(Exception):
            pass

        def boom(pid):
            raise PermissionError("can't read /proc")

        fake_psutil = types.SimpleNamespace(Process=boom, NoSuchProcess=_NoSuchProcess)
        killpg_calls = []
        kill_calls = []

        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        monkeypatch.setattr(pr.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(pr.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))
        monkeypatch.setattr(pr.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

        result = pr.ProcessRegistry._terminate_host_pid(
            12345,
            allow_process_group=True,
            pgid=12345,
        )

        assert result["method"] == "os.killpg"
        assert killpg_calls == [(12345, signal.SIGTERM)]
        assert kill_calls == []

    def test_posix_allow_process_group_rejects_non_leader_pgid(self, monkeypatch):
        from tools import process_registry as pr

        killpg_calls = []
        kill_calls = []

        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setattr(pr.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))
        monkeypatch.setattr(pr.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

        result = pr.ProcessRegistry._terminate_posix_process_group_or_pid(
            12345,
            allow_process_group=True,
            pgid=99999,
        )

        assert result["method"] == "os.kill"
        assert killpg_calls == []
        assert kill_calls == [(12345, signal.SIGTERM)]
        assert "refusing killpg for non-leader" in result["fallback_error"]

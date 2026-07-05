"""Tests for tools/process_registry.py — ProcessRegistry query methods, pruning, checkpoint."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from tools.environments.local import _HERMES_PROVIDER_ENV_FORCE_PREFIX
from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    FINISHED_TTL_SECONDS,
    MAX_PROCESSES,
    MAX_ACTIVE_PROCESS_AGE,
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


def test_write_stdin_uses_str_for_windows_pty(monkeypatch, registry):
    """pywinpty expects str input; bytes raises a PyString conversion error."""
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-win")
    session._pty = _FakePty()
    registry._running[session.id] = session
    monkeypatch.setattr("tools.process_registry._IS_WINDOWS", True)

    result = registry.write_stdin(session.id, "hello\n")

    assert result == {"status": "ok", "bytes_written": 6}
    assert written == ["hello\n"]
    assert isinstance(written[0], str)


def test_write_stdin_uses_bytes_for_posix_pty(monkeypatch, registry):
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-posix")
    session._pty = _FakePty()
    registry._running[session.id] = session
    monkeypatch.setattr("tools.process_registry._IS_WINDOWS", False)

    result = registry.write_stdin(session.id, "hello\n")

    assert result == {"status": "ok", "bytes_written": 6}
    assert written == [b"hello\n"]


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


def test_request_close_terminal_without_sink_is_desktop_only_error(registry):
    """No UI close sink wired (CLI/messaging) → clear desktop-only error, no raise."""
    s = _make_session(sid="proc_close_nosink")
    registry._running[s.id] = s

    result = registry.request_close_terminal(s.id)

    assert result["status"] == "error"
    assert "desktop" in result["error"].lower()


def test_request_close_terminal_invokes_sink_without_killing(registry):
    """With a sink wired, close routes (session, process_id) to the UI and leaves
    the process running — close is a view drop, not a kill."""
    s = _make_session(sid="proc_close_live")
    registry._running[s.id] = s
    calls = []
    registry.on_close = lambda session, pid: calls.append((session, pid))

    result = registry.request_close_terminal(s.id)

    assert result["status"] == "ok"
    assert result["closed"] == "proc_close_live"
    assert calls == [(s, "proc_close_live")]
    # Still tracked as running — closing the tab must not reap the process.
    assert s.id in registry._running


def test_close_terminal_tool_requires_process_id():
    """The desktop-gated close_terminal tool rejects a missing process_id."""
    from tools.close_terminal_tool import close_terminal_tool

    assert json.loads(close_terminal_tool(""))["error"]


def test_close_terminal_tool_routes_to_registry(monkeypatch):
    """close_terminal delegates to process_registry.request_close_terminal."""
    import tools.close_terminal_tool as ct

    seen = {}

    def _fake_close(sid):
        seen["sid"] = sid

        return {"status": "ok", "closed": sid}

    monkeypatch.setattr(ct.process_registry, "request_close_terminal", _fake_close)

    out = ct.close_terminal_tool("proc_abc")

    assert json.loads(out)["closed"] == "proc_abc"
    assert seen["sid"] == "proc_abc"


def test_close_terminal_tool_gated_on_desktop(monkeypatch):
    """Hidden unless HERMES_DESKTOP is set (mirrors read_terminal gating)."""
    from tools.close_terminal_tool import check_close_terminal_requirements

    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    assert check_close_terminal_requirements() is False

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    assert check_close_terminal_requirements() is True


def test_reader_loop_streams_incremental_chunks_from_read1(registry, monkeypatch):
    """Local reader must emit live chunks, not one EOF burst.

    Regression for desktop agent terminals: ``stdout.read(4096)`` can buffer
    until process exit for small periodic output. ``buffer.read1(4096)`` should
    surface each chunk as it arrives.
    """

    class _FakeBuffer:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        def read1(self, _n):
            if self._chunks:
                return self._chunks.pop(0)
            return b""

    class _FakeStdout:
        def __init__(self, chunks):
            self.buffer = _FakeBuffer(chunks)

    class _FakeProcess:
        def __init__(self, chunks):
            self.stdout = _FakeStdout(chunks)
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

    session = _make_session(sid="proc_reader_live")
    session.process = _FakeProcess([b"tick 1\n", b"tick 2\n", b"tick 3\n", b""])
    emitted = []
    moved = []

    monkeypatch.setattr(registry, "_check_watch_patterns", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_emit_output", lambda _s, chunk: emitted.append(chunk))
    monkeypatch.setattr(registry, "_move_to_finished", lambda _s: moved.append(_s.id))

    registry._reader_loop(session)

    assert emitted == ["tick 1\n", "tick 2\n", "tick 3\n"]
    assert session.output_buffer == "tick 1\ntick 2\ntick 3\n"
    assert session.exited is True
    assert session.exit_code == 0
    assert moved == ["proc_reader_live"]


# =========================================================================
# Orphaned-pipe reconciliation (issue #17327)
# =========================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: uses setsid/fcntl")
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

    def test_wait_wakes_when_session_moves_to_finished(self, registry):
        """wait() should not sleep for the old 1s polling tick after exit."""
        s = _make_session(sid="proc_wait_event", output="done")
        registry._running[s.id] = s

        def finish_later():
            time.sleep(0.05)
            s.exited = True
            s.exit_code = 0
            with patch.object(registry, "_write_checkpoint"):
                registry._move_to_finished(s)

        t = threading.Thread(target=finish_later)
        t.start()
        start = time.monotonic()
        try:
            result = registry.wait(s.id, timeout=5)
        finally:
            t.join(timeout=1)
        elapsed = time.monotonic() - start

        assert result["status"] == "exited", result
        assert result["exit_code"] == 0
        assert elapsed < 0.3, f"wait() should wake on completion; took {elapsed:.3f}s"


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

    def test_session_key_surfaces_cross_task_processes(self, registry):
        """A bg process under the same gateway session but a DIFFERENT task is
        surfaced when session_key is passed, and flagged session_scoped (#29177).
        """
        # Current turn's task = "t_now"; forgotten preview server = "t_old"
        # but both share gateway session_key "gw1".
        own = _make_session(sid="proc_own", task_id="t_now")
        own.session_key = "gw1"
        forgotten = _make_session(sid="proc_forgotten", task_id="t_old")
        forgotten.session_key = "gw1"
        other = _make_session(sid="proc_other", task_id="t_x")
        other.session_key = "gw_other"
        registry._running[own.id] = own
        registry._running[forgotten.id] = forgotten
        registry._running[other.id] = other

        # Task-only (legacy) view sees just the current task's process.
        legacy = registry.list_sessions(task_id="t_now")
        assert {r["session_id"] for r in legacy} == {"proc_own"}

        # With session_key, the forgotten process under the same gateway
        # session is surfaced and flagged; the unrelated session is not.
        result = registry.list_sessions(task_id="t_now", session_key="gw1")
        by_id = {r["session_id"]: r for r in result}
        assert set(by_id) == {"proc_own", "proc_forgotten"}
        assert by_id["proc_forgotten"].get("session_scoped") is True
        assert "session_scoped" not in by_id["proc_own"]

    def test_list_entry_fields(self, registry):
        s = _make_session(output="preview text")
        registry._running[s.id] = s
        entry = registry.list_sessions()[0]
        assert "session_id" in entry
        assert "command" in entry
        assert "status" in entry
        assert "pid" in entry
        assert "output_preview" in entry


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

    def test_has_active_for_session_with_max_age_recent(self, registry):
        """Recent process is considered active when max_active_age is set."""
        s = _make_session(started_at=time.time() - 100)
        s.session_key = "gw_session_1"
        registry._running[s.id] = s
        assert registry.has_active_for_session("gw_session_1", max_active_age=3600) is True

    def test_has_active_for_session_with_max_age_stale(self, registry):
        """Stale process (older than max_active_age) is ignored."""
        s = _make_session(started_at=time.time() - 90000)  # 25 hours ago
        s.session_key = "gw_session_1"
        registry._running[s.id] = s
        assert registry.has_active_for_session("gw_session_1", max_active_age=86400) is False

    def test_has_active_for_session_max_age_none_preserves_legacy(self, registry):
        """Without max_active_age, any running process blocks (legacy behaviour)."""
        s = _make_session(started_at=time.time() - 90000)  # 25 hours ago
        s.session_key = "gw_session_1"
        registry._running[s.id] = s
        assert registry.has_active_for_session("gw_session_1") is True

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

            def execute(self, command, **kwargs):
                self.commands.append((command, kwargs))
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

    def test_spawn_via_env_checks_returncode_when_wrapper_fails(self, registry):
        class FakeEnv:
            def __init__(self):
                self.commands = []

            def execute(self, command, **kwargs):
                self.commands.append((command, kwargs))
                return {"output": "syntax error", "returncode": 2}

        env = FakeEnv()
        fake_thread = MagicMock()

        with patch("tools.process_registry.threading.Thread", return_value=fake_thread), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_via_env(env, "echo hello")

        assert session.exited is True
        assert session.exit_code == 2
        assert session.pid is None
        assert session.output_buffer == "syntax error"
        fake_thread.start.assert_not_called()
        # A failed launch must not be exposed as a running/tracked session.
        assert session.id not in registry._running

    def test_spawn_via_env_disables_rewrite_for_bg_wrapper(self, registry):
        class FakeEnv:
            def __init__(self):
                self.commands = []

            def get_temp_dir(self):
                return "/tmp"

            def execute(self, command, **kwargs):
                self.commands.append((command, kwargs))
                return {"output": "4321\n", "returncode": 0}

        env = FakeEnv()
        fake_thread = MagicMock()

        with patch("tools.process_registry.threading.Thread", return_value=fake_thread), \
            patch.object(registry, "_write_checkpoint"):
            registry.spawn_via_env(env, "echo hello")

        args, kwargs = env.commands[0]
        assert kwargs.get("rewrite_compound_background") is False

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

            def execute(self, command, **kwargs):
                self.commands.append((command, kwargs))
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

        # proc.pid is a MagicMock-backed fake; os.getpgid(fake_pid) would query
        # the real OS for an arbitrary PID. On a busy host that PID may exist,
        # in which case spawn_local's primary cleanup path
        # (os.killpg(os.getpgid(pid), SIGKILL)) succeeds against an UNRELATED
        # real process group and proc.kill() is never reached — flaky failure,
        # and a real risk of SIGKILLing an innocent process group. Force the
        # ProcessLookupError fallback so the test deterministically exercises
        # proc.kill() and never issues a real killpg.
        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("threading.Thread", side_effect=boom), \
             patch("os.getpgid", side_effect=ProcessLookupError), \
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

        # See note in test_popen_killed_when_thread_creation_fails: force the
        # ProcessLookupError fallback so cleanup deterministically calls
        # proc.kill() instead of issuing a real os.killpg against whatever
        # process group happens to own the fake PID on the host.
        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("threading.Thread", return_value=fake_thread), \
             patch("os.getpgid", side_effect=ProcessLookupError), \
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
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
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
            with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
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

    def test_kill_local_popen_uses_host_tree_terminator(self, registry, monkeypatch):
        s = _make_session(sid="proc_local", command="sleep 999")
        s.process = MagicMock()
        s.process.pid = 12345
        s.host_start_time = 67890
        registry._running[s.id] = s
        terminate_calls = []

        monkeypatch.setattr(
            registry,
            "_terminate_host_pid",
            lambda pid, expected_start=None: terminate_calls.append((pid, expected_start)),
        )
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

        result = registry.kill_process(s.id)

        assert result["status"] == "killed"
        assert terminate_calls == [(12345, 67890)]

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

        import psutil as _psutil

        try:
            # Post-#21561: liveness probe routes through
            # ``ProcessRegistry._is_host_pid_alive`` (→
            # ``gateway.status._pid_exists``), and the actual kill on POSIX
            # routes through ``psutil.Process(pid).terminate()``. Neither
            # touches ``os.kill`` directly. Mock both seams.  Disable the
            # SIGKILL-escalation step (grace=0) so it doesn't call
            # ``psutil.wait_procs`` on the FakeProcess.
            with patch("gateway.status._pid_exists", return_value=True), \
                 patch.object(ProcessRegistry, "_daemon_term_grace_seconds",
                              staticmethod(lambda: 0.0)), \
                 patch.object(_psutil, "Process", side_effect=lambda pid: FakeProcess(pid)):
                result = registry.kill_process(s.id)

            assert result["status"] == "killed"
            assert ("terminate", 424242) in terminate_calls
        finally:
            registry._running.pop(s.id, None)


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
    assert "[IMPORTANT: Background process proc_abc completed normally" in result
    assert "exit code 0" in result
    assert "Command: sleep 5" in result
    assert "Output:\ndone]" in result


def test_format_killed_completion_event_names_source_and_signal():
    evt = {
        "type": "completion",
        "session_id": "proc_killed",
        "command": "sleep 5",
        "exit_code": -15,
        "completion_reason": "killed",
        "termination_source": "process.kill",
        "output": "",
    }
    result = format_process_notification(evt)
    assert "proc_killed terminated by process.kill" in result
    assert "exit code -15, SIGTERM" in result


def test_format_external_sigterm_does_not_claim_agent_kill():
    evt = {
        "type": "completion",
        "session_id": "proc_external",
        "command": "sleep 5",
        "exit_code": 143,
        "output": "",
    }
    result = format_process_notification(evt)
    assert "proc_external exited" in result
    assert "terminated by" not in result
    assert "exit code 143, SIGTERM" in result


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
        assert "proc_drain1 completed normally" in results[0][1]
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


def test_drain_notifications_filters_async_delegation_by_session_key():
    """Async-delegation events should only be consumed by the matching session's drain.

    Regression test for issue #58684: background delegation results delivered
    to the wrong session when the user switches sessions while a subagent runs.
    """
    from tools.process_registry import process_registry

    # Clear the queue first
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    try:
        # Put events for different sessions
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_session_a",
            "session_key": "telegram:dm:111:user_a",
            "goal": "task A",
            "status": "completed",
            "summary": "done A",
            "api_calls": 1,
            "duration_seconds": 0.5,
        })
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_session_b",
            "session_key": "telegram:dm:222:user_b",
            "goal": "task B",
            "status": "completed",
            "summary": "done B",
            "api_calls": 1,
            "duration_seconds": 0.3,
        })

        # Drain for session A — should only get deleg_session_a
        results_a = process_registry.drain_notifications(session_key="telegram:dm:111:user_a")
        assert len(results_a) == 1, (
            f"Expected 1 event for session A, got {len(results_a)}"
        )
        assert results_a[0][0]["delegation_id"] == "deleg_session_a"
        assert "done A" in results_a[0][1]

        # Session B's event should have been re-queued — drain for session B
        results_b = process_registry.drain_notifications(session_key="telegram:dm:222:user_b")
        assert len(results_b) == 1, (
            f"Expected 1 event for session B, got {len(results_b)}"
        )
        assert results_b[0][0]["delegation_id"] == "deleg_session_b"
        assert "done B" in results_b[0][1]

        # No more events should remain
        assert process_registry.completion_queue.empty()
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_drain_notifications_no_filter_passes_all_async_delegation():
    """Without a session_key filter, all async-delegation events are consumed.

    This ensures backward compatibility — the default (session_key="") permits
    all events, matching pre-fix behavior.
    """
    from tools.process_registry import process_registry

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    try:
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_1",
            "session_key": "telegram:dm:111:user_a",
            "goal": "task 1",
            "status": "completed",
            "summary": "done 1",
            "api_calls": 1,
            "duration_seconds": 0.5,
        })
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_2",
            "session_key": "telegram:dm:222:user_b",
            "goal": "task 2",
            "status": "completed",
            "summary": "done 2",
            "api_calls": 1,
            "duration_seconds": 0.3,
        })

        # No filter — both should be consumed
        results = process_registry.drain_notifications()
        assert len(results) == 2, (
            f"Expected 2 events without filter, got {len(results)}"
        )
        ids = {r[0]["delegation_id"] for r in results}
        assert ids == {"deleg_1", "deleg_2"}
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


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

    def test_windows_does_not_call_psutil(self, monkeypatch):
        """The Windows branch must NOT exercise the psutil tree-walk
        (it's unreliable on Windows — see the function docstring)."""
        from tools import process_registry as pr
        import psutil

        psutil_calls = []

        class _BoomProcess:
            def __init__(self, pid):
                psutil_calls.append(("Process", pid))

            def children(self, recursive=False):
                psutil_calls.append(("children", recursive))
                return []

            def terminate(self):
                psutil_calls.append(("terminate",))

        def fake_run(args, **kwargs):
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(pr, "_IS_WINDOWS", True)
        monkeypatch.setattr(pr.subprocess, "run", fake_run)
        monkeypatch.setattr(psutil, "Process", _BoomProcess)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert psutil_calls == [], (
            f"Windows branch must not touch psutil, but saw {psutil_calls!r}"
        )


class TestTerminateHostPidPosix:
    """POSIX branch walks the tree via psutil and SIGTERMs children first."""

    def test_posix_walks_tree_and_terminates_children_then_parent(self, monkeypatch):
        from tools import process_registry as pr
        import psutil

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

        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setattr(psutil, "Process", _FakeParent)
        # This test covers only the SIGTERM tree-walk ordering; disable the
        # SIGKILL-escalation step (which would call psutil.wait_procs on the
        # fakes) by setting the grace to 0.
        monkeypatch.setattr(pr.ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.0))

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert terminate_order == [101, 102, 103, 12345], (
            "Children must be terminated before the parent"
        )

    def test_posix_no_such_process_swallowed(self, monkeypatch):
        from tools import process_registry as pr
        import psutil

        def boom(pid):
            raise psutil.NoSuchProcess(pid)

        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setattr(psutil, "Process", boom)

        # Must not raise.
        pr.ProcessRegistry._terminate_host_pid(999999999)

    def test_posix_oserror_falls_back_to_os_kill(self, monkeypatch):
        from tools import process_registry as pr
        import psutil

        def boom(pid):
            raise PermissionError("can't read /proc")

        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))

        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setattr(psutil, "Process", boom)
        monkeypatch.setattr(pr.os, "kill", fake_kill)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert kill_calls == [(12345, signal.SIGTERM)]


# =========================================================================
# PID-reuse guard — a recycled PID/PGID must never be signalled.
#
# Regression: once a background-session process exits and is reaped, the kernel
# can recycle its PID onto an unrelated process (observed in the wild landing on
# a desktop browser's session leader, whose whole tree we then SIGTERMed —
# Firefox dying at irregular intervals).  Identity is re-validated via the
# kernel start time captured at spawn before any signal is sent.
# =========================================================================

class TestPidReuseGuard:
    def test_terminate_refuses_when_start_time_mismatches(self, registry):
        """A live PID whose start time changed (recycled) is NOT killed."""
        proc = _spawn_python_sleep(30)
        try:
            real_start = ProcessRegistry._safe_host_start_time(proc.pid)
            assert real_start is not None, "no /proc start time on this platform?"
            # Simulate recycling: the recorded baseline no longer matches.
            registry._terminate_host_pid(proc.pid, expected_start=real_start + 1)
            # The process must still be alive — the guard refused to signal it.
            assert not _wait_until(lambda: proc.poll() is not None, timeout=1.0)
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()

    def test_terminate_kills_when_start_time_matches(self, registry):
        """The genuine process (start time matches) IS terminated."""
        proc = _spawn_python_sleep(30)
        try:
            real_start = ProcessRegistry._safe_host_start_time(proc.pid)
            registry._terminate_host_pid(proc.pid, expected_start=real_start)
            assert _wait_until(lambda: proc.poll() is not None, timeout=5.0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_terminate_without_baseline_is_best_effort(self, registry):
        """No baseline (legacy) → degrade to prior unconditional behaviour."""
        proc = _spawn_python_sleep(30)
        try:
            registry._terminate_host_pid(proc.pid)  # expected_start=None
            assert _wait_until(lambda: proc.poll() is not None, timeout=5.0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_recover_skips_recycled_pid(self, registry, tmp_path):
        """Checkpoint PID is alive but its start time changed → not adopted."""
        wrong_start = (ProcessRegistry._safe_host_start_time(os.getpid()) or 0) + 999
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_recycled",
            "command": "sleep 999",
            "pid": os.getpid(),            # alive...
            "pid_scope": "host",
            "host_start_time": wrong_start,  # ...but a different process now
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            assert registry.recover_from_checkpoint() == 0
            assert len(registry._running) == 0

    def test_recover_adopts_when_start_time_matches(self, registry, tmp_path):
        """Checkpoint PID alive AND start time matches → adopted as before."""
        real_start = ProcessRegistry._safe_host_start_time(os.getpid())
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_match",
            "command": "sleep 999",
            "pid": os.getpid(),
            "pid_scope": "host",
            "host_start_time": real_start,
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            assert registry.recover_from_checkpoint() == 1

    def test_legacy_checkpoint_without_start_time_still_recovers(self, registry, tmp_path):
        """Entries written before host_start_time existed degrade to liveness."""
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_legacy",
            "command": "sleep 999",
            "pid": os.getpid(),
            "pid_scope": "host",
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            assert registry.recover_from_checkpoint() == 1

    def test_write_checkpoint_backfills_host_start_time(self, registry, tmp_path):
        """A host session is checkpointed with a kernel start time recorded."""
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            s = _make_session()
            s.pid = os.getpid()
            s.pid_scope = "host"
            registry._running[s.id] = s
            registry._write_checkpoint()
            data = json.loads((tmp_path / "procs.json").read_text())
            assert data[0]["host_start_time"] is not None

    def test_refresh_detached_marks_recycled_pid_exited(self, registry):
        """A detached session whose PID got recycled is moved to finished."""
        wrong_start = (ProcessRegistry._safe_host_start_time(os.getpid()) or 0) + 999
        s = _make_session(sid="proc_detached")
        s.pid = os.getpid()          # alive, but...
        s.pid_scope = "host"
        s.detached = True
        s.host_start_time = wrong_start  # ...identity no longer matches
        registry._running[s.id] = s
        refreshed = registry._refresh_detached_session(s)
        assert refreshed.exited is True
        assert s.id in registry._finished


@pytest.mark.skipif(sys.platform == "win32",
                    reason="POSIX SIGTERM→SIGKILL escalation; Windows uses taskkill /F")
class TestSigkillEscalation:
    """Bounded SIGTERM→SIGKILL escalation in _terminate_host_pid.

    A daemon that ignores/stalls on SIGTERM must be force-killed after the
    configured grace window so it can't leak indefinitely — while well-behaved
    processes still exit cleanly on SIGTERM and the recycled-PID guard is never
    bypassed.
    """

    # A process that traps SIGTERM (ignores it): only SIGKILL stops it.
    # It prints "ready" AFTER installing the handler so the parent never
    # signals it during the startup window (before SIG_IGN is in place).
    _TRAP = (
        "import signal, sys, time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "sys.stdout.write('ready\\n'); sys.stdout.flush();"
        "[time.sleep(0.2) for _ in iter(int, 1)]"
    )

    def _spawn_trap(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", self._TRAP],
            stdout=subprocess.PIPE, text=True,
        )
        # Wait until the handler is installed before returning.
        line = proc.stdout.readline()
        assert line.strip() == "ready", "trap process failed to start"
        return proc

    def test_sigterm_ignoring_daemon_is_sigkilled(self, monkeypatch):
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 1.0))
        proc = self._spawn_trap()
        try:
            ProcessRegistry._terminate_host_pid(proc.pid)
            assert _wait_until(lambda: proc.poll() is not None, timeout=4.0), \
                "SIGTERM-ignoring daemon should be SIGKILLed after grace"
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_grace_zero_disables_escalation(self, monkeypatch):
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.0))
        proc = self._spawn_trap()
        try:
            ProcessRegistry._terminate_host_pid(proc.pid)
            # No escalation → the SIGTERM-ignoring process survives.
            assert not _wait_until(lambda: proc.poll() is not None, timeout=1.0)
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()

    def test_well_behaved_process_dies_on_sigterm(self, monkeypatch):
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 2.0))
        proc = _spawn_python_sleep(60)
        try:
            ProcessRegistry._terminate_host_pid(proc.pid)
            assert _wait_until(lambda: proc.poll() is not None, timeout=3.0)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_escalation_does_not_bypass_recycled_pid_guard(self, monkeypatch):
        """A start-time mismatch must still spare the PID — no SIGTERM, no SIGKILL."""
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 1.0))
        proc = self._spawn_trap()
        try:
            real_start = ProcessRegistry._safe_host_start_time(proc.pid)
            ProcessRegistry._terminate_host_pid(
                proc.pid, expected_start=(real_start or 0) + 1)
            assert not _wait_until(lambda: proc.poll() is not None, timeout=1.5)
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()

    def test_grace_reader_floors_at_zero(self, monkeypatch):
        """A negative configured grace is clamped to 0 (no escalation)."""
        import hermes_cli.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "read_raw_config",
                            lambda: {"terminal": {"daemon_term_grace_seconds": -5}})
        assert ProcessRegistry._daemon_term_grace_seconds() == 0.0

    @pytest.mark.live_system_guard_bypass
    def test_entire_tree_is_sigkilled_not_just_parent(self, monkeypatch):
        """A SIGTERM-ignoring parent + children are ALL force-killed.

        Regression: an earlier implementation trusted psutil.wait_procs's
        gone/alive partition, which mis-partitioned across a parent/child tree
        and left survivors un-killed (flaky — sometimes the parent lived,
        sometimes a child). The escalation now re-probes every target directly.
        """
        import psutil
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 1.0))
        # Parent spawns 2 children; all trap SIGTERM. Parent prints child pids
        # after the handler is installed.
        parent_src = (
            "import signal, subprocess, sys, time;"
            "child='import signal,time\\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
            "[time.sleep(0.2) for _ in iter(int,1)]';"
            "kids=[subprocess.Popen([sys.executable,'-c',child]) for _ in range(2)];"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "sys.stdout.write(' '.join(str(k.pid) for k in kids)+'\\n'); sys.stdout.flush();"
            "[time.sleep(0.2) for _ in iter(int,1)]"
        )
        parent = subprocess.Popen([sys.executable, "-c", parent_src],
                                  stdout=subprocess.PIPE, text=True)
        child_pids = [int(x) for x in parent.stdout.readline().split()]
        all_pids = [parent.pid] + child_pids
        try:
            ProcessRegistry._terminate_host_pid(parent.pid)

            def _pid_dead(p: int) -> bool:
                # A pid is "dead" for our purposes if it no longer exists OR
                # exists only as an unreaped zombie (already terminated, just
                # not reaped by its reparented parent yet). psutil can also
                # raise mid-probe if the pid vanishes between the existence
                # check and the status read — treat any such race as dead.
                try:
                    if not psutil.pid_exists(p):
                        return True
                    return not ProcessRegistry._proc_alive(psutil.Process(p))
                except Exception:
                    return True

            def _all_dead():
                return all(_pid_dead(p) for p in all_pids)

            # _terminate_host_pid SIGKILLs synchronously before returning, so
            # the kill signals are already delivered here. The only remaining
            # wait is the kernel tearing down 3 processes and the reparented
            # children transitioning to zombie — which can lag on a loaded CI
            # runner. Give a generous budget (matches the wait() test's 10s)
            # so this asserts the escalation BEHAVIOR, not the runner's
            # scheduling latency. The assertion itself never weakens: every
            # tree member must end up dead/zombie.
            assert _wait_until(_all_dead, timeout=15.0, interval=0.02), (
                "entire SIGTERM-ignoring tree (parent + children) must be SIGKILLed"
            )
        finally:
            for p in all_pids:
                try:
                    os.kill(p, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            parent.wait()


class TestHandleProcessRedaction:
    """`_handle_process` redacts background-process output before it reaches the
    model / session.db / CLI display — issue #43025.

    Mirrors the foreground `terminal` redaction so the two surfaces can't
    diverge. Env-dump commands (`printenv`/`env`) get the ENV-assignment pass
    so opaque tokens are masked; other commands stay on the code_file path.
    """

    def _setup(self, monkeypatch, command, output):
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", True)
        from tools import process_registry as pr
        reg = ProcessRegistry()
        sess = _make_session(sid="proc_redact1", command=command)
        sess.output_buffer = output
        sess.exited = True
        sess.exit_code = 0
        reg._running.clear()
        reg._finished[sess.id] = sess
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)
        return pr, sess

    def test_log_redacts_env_dump_opaque_token(self, monkeypatch):
        pr, sess = self._setup(
            monkeypatch, "printenv",
            "MY_SERVICE_TOKEN=abc123randomopaquetokenvalue999\nHOME=/home/u",
        )
        out = json.loads(pr._handle_process({"action": "log", "session_id": sess.id}))
        assert "abc123randomopaquetokenvalue999" not in out["output"]
        assert "HOME=/home/u" in out["output"]

    def test_poll_redacts_prefix_key(self, monkeypatch):
        pr, sess = self._setup(
            monkeypatch, "python app.py",
            "leaked OPENAI_API_KEY sk-proj-abc123def456ghi789jkl012 here",
        )
        out = json.loads(pr._handle_process({"action": "poll", "session_id": sess.id}))
        assert "abc123def456" not in out["output_preview"]

    def test_disabled_passes_through(self, monkeypatch):
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", False)
        from tools import process_registry as pr
        reg = ProcessRegistry()
        sess = _make_session(sid="proc_redact2", command="printenv")
        sess.output_buffer = "CUSTOM_TOKEN=zzzopaque1234567890abcdef"
        sess.exited = True
        sess.exit_code = 0
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)
        out = json.loads(pr._handle_process({"action": "log", "session_id": sess.id}))
        assert "zzzopaque1234567890abcdef" in out["output"]

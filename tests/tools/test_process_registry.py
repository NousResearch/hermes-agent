"""Tests for tools/process_registry.py — ProcessRegistry query methods, pruning, checkpoint."""

from tools import process_registry_scope as _process_scope

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_FORCE_PREFIX
from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    FINISHED_TTL_SECONDS,
    MAX_PROCESSES,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


@pytest.fixture(autouse=True)
def _reset_systemd_scope_cache():
    """Reset the cached ``systemd-run --user --scope`` availability flag
    before each test so a probe run on a real systemd host (where
    ``INVOCATION_ID`` is set) doesn't leak into tests that mock
    ``subprocess.Popen``. Tests that exercise the probe directly reset the
    cache themselves."""
    import tools.process_registry as _pr

    original = _process_scope._SYSTEMD_SCOPE_AVAILABLE
    _process_scope._SYSTEMD_SCOPE_AVAILABLE = False
    yield
    _process_scope._SYSTEMD_SCOPE_AVAILABLE = original


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


def test_kill_started_since_preserves_preexisting_and_foreign_processes(registry):
    old = _make_session(sid="proc_old", task_id="session-a")
    finished = _make_session(
        sid="proc_finished", task_id="session-a", exited=True, exit_code=0
    )
    registry._running[old.id] = old
    registry._finished[finished.id] = finished
    baseline = registry.snapshot_running_ids("session-a")

    new = _make_session(sid="proc_new", task_id="session-a")
    foreign = _make_session(sid="proc_foreign", task_id="session-b")
    registry._running[new.id] = new
    registry._running[foreign.id] = foreign

    calls = []

    def fake_kill(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {"status": "killed"}

    registry.kill_process = fake_kill

    assert baseline == frozenset({"proc_old"})
    assert registry.kill_started_since(
        "session-a", baseline, source="gateway_turn_timeout"
    ) == 1
    assert calls == [
        (
            "proc_new",
            {
                "source": "gateway_turn_timeout",
                "consume_output": True,
            },
        )
    ]


def test_kill_all_backward_compat_and_exclude_ids(registry):
    """kill_all keeps its historical default behavior (kill everything for
    the task, consume_output=False, source='kill_all') and honors the new
    exclude_ids kwarg that kill_started_since delegates through (#76188)."""
    a = _make_session(sid="proc_a", task_id="session-a")
    b = _make_session(sid="proc_b", task_id="session-a")
    registry._running[a.id] = a
    registry._running[b.id] = b

    calls = []

    def fake_kill(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {"status": "killed"}

    registry.kill_process = fake_kill

    assert registry.kill_all("session-a", exclude_ids=frozenset({"proc_a"})) == 1
    assert calls == [
        ("proc_b", {"source": "kill_all", "consume_output": False})
    ]

    calls.clear()
    assert registry.kill_all("session-a") == 2
    assert sorted(c[0] for c in calls) == ["proc_a", "proc_b"]


@pytest.mark.windows_only
def test_write_stdin_uses_str_for_windows_pty(registry):
    """pywinpty expects str input; bytes raises a PyString conversion error.

    Windows-only: the str-vs-bytes choice IS the ``_IS_WINDOWS`` branch, and
    the real pty handle it must satisfy (pywinpty) does not exist elsewhere.
    """
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-win")
    session._pty = _FakePty()
    registry._running[session.id] = session

    result = registry.write_stdin(session.id, "hello\n")

    assert result == {"status": "ok", "bytes_written": 6}
    assert written == ["hello\n"]
    assert isinstance(written[0], str)


@pytest.mark.linux_only
def test_write_stdin_uses_bytes_for_posix_pty(registry):
    """The POSIX counterpart: ptyprocess expects bytes, not str."""
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-posix")
    session._pty = _FakePty()
    registry._running[session.id] = session

    result = registry.write_stdin(session.id, "hello\n")

    assert result == {"status": "ok", "bytes_written": 6}
    assert written == [b"hello\n"]


@pytest.mark.windows_only
def test_submit_stdin_uses_crlf_for_windows_pty(registry):
    """Enter on a Windows PTY is a carriage return, not a bare LF.

    ConPTY cooked input only ends a line on ``\\r``; a bare ``\\n`` through
    pywinpty is never delivered to a blocking line read (Python readline,
    Go bufio.Scanner — the exact hang seen live with ``gh auth login``'s
    "Press Enter to open the browser" prompt). submit_stdin must append
    ``\\r\\n`` for Windows PTY sessions.
    """
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-win-submit")
    session._pty = _FakePty()
    registry._running[session.id] = session

    result = registry.submit_stdin(session.id, "Y")

    assert result["status"] == "ok"
    assert written == ["Y\r\n"]


@pytest.mark.windows_only
def test_submit_stdin_keeps_lf_for_windows_pipe(registry):
    """Non-PTY (Popen pipe) sessions keep the plain LF on Windows."""
    session = _make_session(sid="pipe-win-submit")
    fake_stdin = MagicMock()
    session.process = MagicMock()
    session.process.stdin = fake_stdin
    registry._running[session.id] = session

    result = registry.submit_stdin(session.id, "Y")

    assert result["status"] == "ok"
    fake_stdin.write.assert_called_once_with("Y\n")


class TestGetAndPoll:
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


class _FakeChunkBuffer:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read1(self, _n):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeChunkStdout:
    def __init__(self, chunks):
        self.buffer = _FakeChunkBuffer(chunks)


class _FakeChunkProcess:
    def __init__(self, chunks):
        self.stdout = _FakeChunkStdout(chunks)
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


def _run_reader(registry, monkeypatch, chunks, sid="proc_utf8"):
    session = _make_session(sid=sid)
    session.process = _FakeChunkProcess(chunks)
    monkeypatch.setattr(registry, "_check_watch_patterns", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_emit_output", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_move_to_finished", lambda _s: None)
    registry._reader_loop(session)
    return session


def test_reader_loop_reassembles_multibyte_char_split_across_chunks(registry, monkeypatch):
    """A UTF-8 char split across two read1() chunks must not become U+FFFD.

    Before the incremental decoder, each chunk was decoded statelessly with
    ``errors="replace"``, so ``é`` (0xC3 0xA9) straddling a 4096-byte read
    boundary decoded as two replacement characters.
    """
    session = _run_reader(registry, monkeypatch, [b"caf\xc3", b"\xa9 ok\n"])
    assert session.output_buffer == "café ok\n"
    assert "\ufffd" not in session.output_buffer


def test_reader_loop_reassembles_four_byte_char_split_three_ways(registry, monkeypatch):
    """A 4-byte emoji fragmented across three reads reassembles cleanly."""
    session = _run_reader(registry, monkeypatch, [b"\xf0", b"\x9f\x92", b"\xa9\n"])
    assert session.output_buffer == "\U0001f4a9\n"


def test_reader_loop_flushes_truncated_multibyte_tail_at_eof(registry, monkeypatch):
    """A sequence truncated by process exit flushes as a single U+FFFD."""
    session = _run_reader(registry, monkeypatch, [b"ok \xe2\x82"])
    assert session.output_buffer == "ok \ufffd"


def test_reader_loop_still_replaces_genuinely_invalid_bytes(registry, monkeypatch):
    """Truly invalid bytes keep the errors="replace" behavior."""
    session = _run_reader(registry, monkeypatch, [b"ok\xffdone\n"])
    assert session.output_buffer == "ok\ufffddone\n"


def test_pty_reader_loop_reassembles_multibyte_char_split_across_chunks(registry, monkeypatch):
    """The PTY reader gets the same incremental-decode treatment."""

    class _FakePty:
        def __init__(self, chunks):
            self._chunks = list(chunks)
            self.exitstatus = 0

        def isalive(self):
            return bool(self._chunks)

        def read(self, _n):
            if self._chunks:
                return self._chunks.pop(0)
            raise EOFError

        def wait(self):
            return 0

    session = _make_session(sid="proc_pty_utf8")
    session._pty = _FakePty([b"caf\xc3", b"\xa9\n"])
    monkeypatch.setattr(registry, "_check_watch_patterns", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_emit_output", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_move_to_finished", lambda _s: None)

    registry._pty_reader_loop(session)

    assert session.output_buffer == "café\n"
    assert "\ufffd" not in session.output_buffer


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
        assert elapsed < 0.9  # must stay under the old 1s poll tick being regression-tested, f"wait() should wake on completion; took {elapsed:.3f}s"


class TestReadLog:
    def test_read_full_log(self, registry):
        lines = "\n".join([f"line {i}" for i in range(50)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id)
        assert result["total_lines"] == 50

    def test_read_with_offset(self, registry):
        lines = "\n".join([f"line {i}" for i in range(100)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id, offset=10, limit=5)
        assert "5 lines" in result["showing"]


class TestStdinHelpers:
    def test_close_stdin_pipe_mode(self, registry):
        proc = MagicMock()
        proc.stdin = MagicMock()
        s = _make_session()
        s.process = proc
        registry._running[s.id] = s

        result = registry.close_stdin(s.id)

        proc.stdin.close.assert_called_once()
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
            # Wait for the PTY child to be up rather than sleeping blindly.
            assert _wait_until(
                lambda: registry.poll(session.id)["status"] == "running",
                timeout=5.0,
                interval=0.02,
            ), "PTY session never reached running"
            assert registry.submit_stdin(session.id, "hello")["status"] == "ok"
            assert registry.close_stdin(session.id)["status"] == "ok"

            deadline = time.time() + 5
            while time.time() < deadline:
                poll = registry.poll(session.id)
                if poll["status"] == "exited":
                    assert poll["exit_code"] == 0
                    assert "hello" in poll["output_preview"]
                    return
                time.sleep(0.02)

            pytest.fail("process did not exit after stdin was closed")
        finally:
            registry.kill_process(session.id)


class TestListSessions:
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


class TestActiveQueries:
    def test_has_active_processes(self, registry):
        s = _make_session(task_id="t1")
        registry._running[s.id] = s
        assert registry.has_active_processes("t1") is True
        assert registry.has_active_processes("t2") is False

    def test_has_active_for_session_with_max_age_stale(self, registry):
        """Stale process (older than max_active_age) is ignored."""
        s = _make_session(started_at=time.time() - 90000)  # 25 hours ago
        s.session_key = "gw_session_1"
        registry._running[s.id] = s
        assert registry.has_active_for_session("gw_session_1", max_active_age=86400) is False


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

    def test_env_poller_quotes_temp_paths_with_spaces(self, registry):
        session = _make_session(sid="proc_space")
        session.exited = False

        class FakeEnv:
            def __init__(self):
                self.commands = []
                self._responses = iter([
                    {"output": "6 0\nhello\n"},
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

        assert "'/path with spaces/hermes_bg.log'" in env.commands[0][0]
        assert "cat '/path with spaces/hermes_bg.log'" not in env.commands[0][0]
        assert env.commands[1][0] == "kill -0 \"$(cat '/path with spaces/hermes_bg.pid' 2>/dev/null)\" 2>/dev/null; echo $?"
        assert env.commands[2][0] == "cat '/path with spaces/hermes_bg.exit' 2>/dev/null"


class TestEnvPollerIncrementalRead:
    """The sandbox log poller must read only new bytes, not the whole file.

    Reading the whole file every poll made one poll cost grow with the total
    output so far, so a long noisy job re-sent all of its output over the
    docker or SSH channel every two seconds.
    """

    @staticmethod
    def _run_poller(registry, session, responses):
        """Drive one poll cycle and hand back the commands the env saw."""

        class FakeEnv:
            def __init__(self):
                self.commands = []
                self._responses = iter(responses)

            def execute(self, command, **kwargs):
                self.commands.append(command)
                return next(self._responses)

        env = FakeEnv()
        with patch("tools.process_registry.time.sleep", return_value=None), \
            patch.object(registry, "_move_to_finished"):
            registry._env_poller_loop(
                session, env, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit"
            )
        return env.commands

    def test_read_command_asks_only_for_new_bytes(self):
        cmd = ProcessRegistry._log_delta_command("'/tmp/bg.log'", 4096)
        # The offset is carried into the command, and the file is opened with
        # tail rather than cat.
        assert "O=4096" in cmd
        assert "tail -c +$((O+1)) '/tmp/bg.log'" in cmd
        assert "cat '/tmp/bg.log'" not in cmd

    def test_read_command_starts_from_zero_on_first_poll(self):
        cmd = ProcessRegistry._log_delta_command("'/tmp/bg.log'", 0)
        assert "O=0" in cmd

    @pytest.mark.skipif(not shutil.which("sh"), reason="needs a POSIX sh")
    def test_read_command_holds_back_a_split_utf8_sequence(self, tmp_path):
        """A multibyte character straddling two polls must not be split.

        The backend decodes each execute() result on its own, so returning
        the first byte of an 'é' in one poll and the rest in the next would
        yield replacement characters in the transcript (and break watch
        patterns at the seam). Every prefix of a mixed ASCII/2/3/4-byte
        string must come back decodable, with at most 3 bytes held back and
        nothing held back once the trailing character is complete.
        """
        full = "hé😀中a\n€bz🚀".encode()
        log = tmp_path / "bg.log"
        quoted = shlex.quote(str(log))
        for n in range(1, len(full) + 1):
            log.write_bytes(full[:n])
            out = subprocess.run(
                ["sh", "-c", ProcessRegistry._log_delta_command(quoted, 0)],
                capture_output=True, timeout=30,
            ).stdout
            header, _, delta = out.partition(b"\n")
            size, _offset = map(int, header.split())
            delta.decode("utf-8")  # must not raise
            assert delta == full[:size]
            complete = full[:n].decode("utf-8", "ignore").encode() == full[:n]
            assert (n - size) == 0 if complete else 0 < (n - size) <= 3

    def test_first_poll_reads_from_the_start(self, registry):
        session = _make_session(sid="proc_delta")
        session.exited = False
        commands = self._run_poller(
            registry,
            session,
            [
                {"output": "11 0\nfirst chunk"},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert "O=0" in commands[0]
        assert session.output_buffer == "first chunk"

    def test_delta_is_appended_not_replaced(self, registry):
        session = _make_session(sid="proc_append", output="already here ")
        session.exited = False
        self._run_poller(
            registry,
            session,
            [
                {"output": "8 0\nand new"},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "already here and new"

    def test_second_poll_asks_from_where_the_first_one_stopped(self, registry):
        session = _make_session(sid="proc_two_polls")
        session.exited = False
        commands = self._run_poller(
            registry,
            session,
            [
                {"output": "11 0\nfirst chunk"},
                {"output": "0\n"},          # still running, poll again
                {"output": "17 11\n and more"},
                {"output": "1\n"},          # gone now
                {"output": "0\n"},
            ],
        )
        assert "O=0" in commands[0]
        # The second read starts at byte 11, so the first chunk is not sent
        # a second time.
        assert "O=11" in commands[2]
        assert session.output_buffer == "first chunk and more"

    def test_truncated_log_drops_the_stale_buffer(self, registry):
        session = _make_session(sid="proc_rotate")
        session.exited = False
        # The second read reports offset 0 even though the first one left off
        # at byte 11. The file no longer reaches that byte, so it was rotated
        # or truncated and the buffer we hold no longer matches it.
        self._run_poller(
            registry,
            session,
            [
                {"output": "11 0\nfirst chunk"},
                {"output": "0\n"},          # still running, poll again
                {"output": "5 0\nfresh"},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "fresh"

    def test_unreadable_header_leaves_the_buffer_alone(self, registry):
        session = _make_session(sid="proc_bad", output="keep me")
        session.exited = False
        # No header at all, for example when the shell is missing one of the
        # tools the command needs.
        self._run_poller(
            registry,
            session,
            [
                {"output": ""},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "keep me"

    def test_buffer_stays_within_the_cap(self, registry):
        session = _make_session(sid="proc_cap")
        session.exited = False
        session.max_output_chars = 10
        self._run_poller(
            registry,
            session,
            [
                {"output": "20 0\n" + "x" * 20},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "x" * 10


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


class TestSpawnRewriteCompoundBackground:
    """Verify that spawn_local rewrites `A && B &` patterns to avoid subshell deadlocks.

    Issue #68915: when bash parses ``A && B &`` it forks a subshell ``(A && B) &``.
    If B is a long-running server, the subshell never exits and holds the stdout
    pipe open, causing a permanent deadlock. The rewriter wraps the tail to
    ``A && { B & }`` so no subshell fork occurs.
    """

    def test_compound_and_background_gets_rewritten(self, registry):
        """A && B & must be rewritten to A && { B & } before Popen."""
        captured_cmd = []

        def fake_popen(args, **kwargs):
            captured_cmd.append(args)
            proc = MagicMock()
            proc.pid = 1111
            proc.stdout = MagicMock()
            return proc

        fake_thread = MagicMock()
        fake_thread.daemon = False

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            registry.spawn_local("cd /app && node server.js &>/tmp/srv.log &", cwd="/tmp")

        assert len(captured_cmd) == 1
        shell_cmd = captured_cmd[0]
        # The command passed to Popen should be the REWRITTEN version
        assert "&& { node server.js &>/tmp/srv.log & }" in shell_cmd[2]

    def test_simple_background_preserved(self, registry):
        """Simple cmd & (no &&) must NOT be rewritten — no subshell bug."""
        captured_cmd = []

        def fake_popen(args, **kwargs):
            captured_cmd.append(args)
            proc = MagicMock()
            proc.pid = 2222
            proc.stdout = MagicMock()
            return proc

        fake_thread = MagicMock()
        fake_thread.daemon = False

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            registry.spawn_local("sleep 5 &", cwd="/tmp")

        assert len(captured_cmd) == 1
        shell_cmd = captured_cmd[0][2]
        # Simple background must remain as-is
        assert "sleep 5 &" in shell_cmd

    def test_pty_path_uses_rewritten_command(self, registry):
        """PTY spawn path must also use the rewritten command (issue #68915)."""
        mock_pty_proc = MagicMock()
        mock_pty_proc.pid = 5555

        mock_pty_module = MagicMock()
        mock_pty_module.PtyProcess.spawn = MagicMock(return_value=mock_pty_proc)

        fake_thread = MagicMock()
        fake_thread.daemon = False

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch.dict("sys.modules", {"ptyprocess": mock_pty_module}), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_local(
                "cd /app && node server.js &",
                cwd="/tmp",
                use_pty=True,
            )

        assert mock_pty_module.PtyProcess.spawn.called, \
            "PTY spawn should have been attempted"
        pty_args = mock_pty_module.PtyProcess.spawn.call_args[0][0]
        assert "&& { node server.js & }" in pty_args[2], \
            f"PTY path should use rewritten command, got: {pty_args[2]}"
        assert session.command == "cd /app && node server.js &"

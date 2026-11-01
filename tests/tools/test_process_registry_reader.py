"""Capture-pipe EOF and PTY completion status regressions (#86416)."""

import shlex
import subprocess
import sys
import time
from contextlib import suppress
from unittest.mock import MagicMock

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def registry(monkeypatch, tmp_path):
    monkeypatch.setattr("tools.process_registry._SYSTEMD_SCOPE_AVAILABLE", False)
    monkeypatch.setattr("tools.process_registry.CHECKPOINT_PATH", tmp_path / "processes.json")
    return ProcessRegistry()


@pytest.mark.parametrize(
    "exitstatus,signalstatus,expected_code", [(0, None, 0), (None, 9, -9)]
)
def test_pty_reader_loop_reassembles_multibyte_char_split_across_chunks(
    registry, monkeypatch, exitstatus, signalstatus, expected_code
):
    """The PTY reader preserves UTF-8 and finalizes both ordinary and signal
    exits (ptyprocess can report exitstatus=None with a valid signalstatus —
    a bare exitstatus read would finalize an unknown code)."""

    class _FakePty:
        def __init__(self, chunks):
            self._chunks = list(chunks)
            self.exitstatus = exitstatus
            self.signalstatus = signalstatus

        def isalive(self):
            return bool(self._chunks)

        def read(self, _n):
            if self._chunks:
                return self._chunks.pop(0)
            raise EOFError

        def wait(self):
            return self.exitstatus

    session = ProcessSession(id="proc_pty_utf8", command="echo hello", task_id="t1", started_at=time.time())
    session._pty = _FakePty([b"caf\xc3", b"\xa9\n"])
    session.notify_on_complete = True
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "_check_watch_patterns", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_emit_output", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

    registry._pty_reader_loop(session)

    assert session.output_buffer == "café\n"
    assert "\ufffd" not in session.output_buffer
    assert session.exited and session.exit_code == expected_code
    assert session._completion_event.is_set()
    assert registry.completion_queue.get_nowait()["exit_code"] == expected_code
    assert registry.completion_queue.empty()



# =========================================================================
# EOF-while-alive: capture pipe closes before process exits (issue #86416)
# =========================================================================

@pytest.mark.platforms("posix")
class TestReaderLoopEofWhileAlive:
    """Regression tests for issue #86416.

    A child that redirects or closes its stdout/stderr makes the reader reach
    EOF while the process is still alive. The reader must not treat that EOF
    as a real exit (a completion with exit_code=None), and it must not die
    either: it stays parked on the real child exit so the notify_on_complete
    contract still fires autonomously, without any poll()/wait() reconcile.
    """

    def test_eof_on_closed_capture_pipe_does_not_mark_exited(self, registry, tmp_path):
        """EOF while the direct child is alive: no completion; the reader
        stays parked on the real exit (past the old 5s reader bound)."""
        session = registry.spawn_local(
            f"exec {shlex.quote(sys.executable)} -c 'import os, time; os.close(1); os.close(2); time.sleep(30)'",
            cwd=str(tmp_path),
        )
        session.notify_on_complete = True
        try:
            assert session._reader_thread is not None, "spawn must attach a reader thread"
            # Past the historical 5s reader bound: the capture pipe is long at
            # EOF and the direct child is still sleeping.
            assert not session._completion_event.wait(5.5)
            assert session.process.poll() is None
            assert session.exited is False
            assert session.id in registry._running
            assert session.id not in registry._finished
            assert registry.completion_queue.empty()
            # The reader stays alive — parked on the real child exit — so a
            # later exit still has an observer.
            assert session._reader_thread.is_alive()
            result = registry.kill_process(session.id)
            assert result["status"] == "killed"
            assert session.process.poll() is not None
            session._reader_thread.join(timeout=5)
            assert not session._reader_thread.is_alive()
            notification = registry.completion_queue.get(timeout=5)
            assert notification["session_id"] == session.id
            assert notification["exit_code"] is not None
            assert registry.completion_queue.empty()
        finally:
            registry.kill_process(session.id)
            if session._reader_thread is not None:
                session._reader_thread.join(timeout=5)

    def test_real_exit_after_eof_notifies_without_polling(
        self, registry, tmp_path, monkeypatch
    ):
        """After EOF-while-alive, the eventual real exit must notify on its own:
        no poll()/wait() may supply the reconcile (issue #86416, second half)."""
        monkeypatch.setattr(
            "tools.process_registry.CHECKPOINT_PATH", tmp_path / "processes.json"
        )
        release = tmp_path / "release"
        child = (
            "import pathlib,time\n"
            "release=pathlib.Path('release')\n"
            "deadline=time.monotonic()+20\n"
            "while not release.exists() and time.monotonic()<deadline:\n"
            "    time.sleep(0.02)\n"
            "raise SystemExit(7 if release.exists() else 97)"
        )
        # exec + redirection: the shell replaces itself with the worker and the
        # capture pipe's write end closes at exec time — EOF while alive.
        session = registry.spawn_local(
            f"exec {shlex.quote(sys.executable)} -c {shlex.quote(child)} "
            ">worker.out 2>worker.err",
            cwd=str(tmp_path), use_pty=False,
        )
        session.notify_on_complete = True
        try:
            # Longer than the old reader bound: no completion may appear while
            # the child is still waiting on the release file.
            assert not session._completion_event.wait(5.3)
            assert not session.exited
            assert registry.is_session_waiting(session.id)
            assert registry.completion_queue.empty()
            assert session._reader_thread is not None
            assert session._reader_thread.is_alive()

            release.touch()
            notification = registry.completion_queue.get(timeout=5)
            assert notification["session_id"] == session.id
            assert notification["exit_code"] == 7
            assert session.exited and session.exit_code == 7
            assert not registry.is_session_waiting(session.id)
            session._reader_thread.join(timeout=5)
            assert not session._reader_thread.is_alive()
            assert registry.completion_queue.empty()
        finally:
            release.touch()
            if session.process is not None:
                with suppress(subprocess.TimeoutExpired, OSError):
                    session.process.wait(timeout=5)
            if session._reader_thread is not None:
                session._reader_thread.join(timeout=5)
            if session.id in registry._running:
                registry.kill_process(session.id)

    def test_reader_wait_failure_does_not_report_unknown_exit(self, registry):
        """An exceptional wait is not evidence that the child exited — never
        finalize a completion on an unknown code."""
        import codecs

        session = ProcessSession(id="proc_unknown", command="echo hello", task_id="t1", started_at=time.time())
        session.notify_on_complete = True
        registry._running[session.id] = session
        wait = MagicMock(side_effect=OSError("wait failed"))
        registry._finish_reader(
            session,
            codecs.getincrementaldecoder("utf-8")(errors="replace"),
            lambda text: None,
            "Process",
            wait,
            lambda: None,
        )
        assert not session.exited
        assert not session._completion_event.is_set()
        assert registry.completion_queue.empty()

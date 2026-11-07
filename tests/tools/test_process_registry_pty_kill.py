"""Killing a PTY background process must return while a descendant that escapes it holds the PTY.

An escapee (a descendant that ``setsid()``s into its own session) keeps the PTY slave open, so
the reader thread stays blocked in ``read()`` holding the PTY file object's buffer lock. Closing
the PTY from the kill path waited on that lock until the escapee exited (forever, for a
long-lived one), so the kill never returned. Once killed, the session's output stays as the
kill reported it. Real PTY, real processes: a fake PTY cannot hold the lock.
"""

import shutil
import sys
import threading
import time

import pytest

import tools.process_registry as module
from tools.process_registry import ProcessRegistry

_POSIX_PTY = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("setsid") is None, reason="POSIX PTY + setsid")

# The escapee exits on its own (it is reparented to init, outside what a test may signal).
# Before the fix the kill could only return once it had, so the kill deadline is shorter.
_ESCAPEE_LIFETIME_S = 8
_KILL_DEADLINE_S = 5


@pytest.fixture
def unscoped(monkeypatch):
    # No systemd scope: stopping one would reap the escapee and hide the hang.
    monkeypatch.setattr(module, "_is_supervised_gateway_process", lambda: False)


def _wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _terminate_owned(session):
    """Failure-path cleanup: stop the PTY child this test spawned (never the reparented escapee)."""
    if not session.exited and session._pty is not None:
        try:
            session._pty.terminate(force=True)
        except Exception:
            pass


def _spawn_with_escapee(registry, tmp_path, late_output=""):
    pidfile = tmp_path / "escapee.pid"
    # Without ``late_output`` the escapee never writes to the PTY, so nothing ends the reader's
    # blocked read before the kill deadline. ``late_output`` is printed after the test has
    # killed the session.
    late = f"sleep 2; echo {late_output}; " if late_output else ""
    # The escapee's pid is read as the PPID of a grandchild, not as ``$$``: under
    # ``systemd-run --scope`` with environment expansion, ``$$`` reaches the shell as ``$``.
    session = registry.spawn_local(
        f"setsid sh -c 'sh -c \"echo \\$PPID\" > {pidfile}; {late}exec sleep {_ESCAPEE_LIFETIME_S}' & sleep 120",
        cwd=str(tmp_path), use_pty=True)
    try:
        assert _wait_for(lambda: pidfile.exists() and pidfile.read_text().strip(), 5)
    except BaseException:
        _terminate_owned(session)
        raise
    return session


def _kill_within_deadline(registry, session):
    result = {}
    killer = threading.Thread(
        target=lambda: result.update(registry.kill_process(session.id)), daemon=True)
    killer.start()
    killer.join(_KILL_DEADLINE_S)
    assert not killer.is_alive(), "kill_process blocked closing the PTY under a live reader"
    return result


def _wait_for_reader_close(session):
    # The reader outlives the kill and closes the PTY itself once its read ends (when the
    # escapee exits), so the master FD is still released.
    reader = session._reader_thread
    assert reader is not None
    assert _wait_for(lambda: not reader.is_alive(), _ESCAPEE_LIFETIME_S + 7)
    assert session._pty.closed


@_POSIX_PTY
def test_kill_returns_while_escaped_descendant_holds_the_pty(tmp_path, unscoped):
    pytest.importorskip("ptyprocess")
    registry = ProcessRegistry()
    session = _spawn_with_escapee(registry, tmp_path)
    try:
        result = _kill_within_deadline(registry, session)
    finally:
        _terminate_owned(session)
    assert result["status"] == "killed"
    assert session.id in registry._finished
    _wait_for_reader_close(session)


@_POSIX_PTY
def test_output_after_the_kill_is_not_added_to_the_killed_session(tmp_path, monkeypatch, unscoped):
    pytest.importorskip("ptyprocess")
    registry = ProcessRegistry()
    emit, emitted = registry._emit_output, []

    def recording_emit(session, text):
        emitted.append(text)
        emit(session, text)

    registry._emit_output = recording_emit
    session = _spawn_with_escapee(registry, tmp_path, late_output="LATE-ESCAPEE-OUTPUT")
    try:
        result = _kill_within_deadline(registry, session)
    finally:
        _terminate_owned(session)
    assert result["status"] == "killed"
    # The reader drains what the escapee printed after the kill, but does not keep it.
    _wait_for_reader_close(session)
    assert "LATE-ESCAPEE-OUTPUT" not in session.output_buffer
    assert not any("LATE-ESCAPEE-OUTPUT" in text for text in emitted)


@_POSIX_PTY
def test_chunk_read_before_the_kill_is_not_added_after_it(tmp_path, unscoped):
    # The reader has read a chunk but not yet buffered it when the kill snapshots the output
    # and sets ``exited``. The chunk must not land in the killed session afterwards.
    pytest.importorskip("ptyprocess")
    registry = ProcessRegistry()
    entered, release = threading.Event(), threading.Event()
    ingest, emit, emitted = registry._ingest_output, registry._emit_output, []

    def paused_ingest(session, text, **kwargs):
        if "RACE-MARKER" in text:
            entered.set()
            release.wait(10)
        ingest(session, text, **kwargs)

    def recording_emit(session, text):
        emitted.append(text)
        emit(session, text)

    registry._ingest_output = paused_ingest
    registry._emit_output = recording_emit
    session = registry.spawn_local(
        "sleep 0.2; echo RACE-MARKER; exec sleep 15", cwd=str(tmp_path), use_pty=True)
    try:
        assert entered.wait(5), "reader never read the marker"
        result = registry.kill_process(session.id)
    finally:
        release.set()
    assert result["status"] == "killed"
    assert _wait_for(lambda: not session._reader_thread.is_alive(), 5)
    assert "RACE-MARKER" not in result.get("output", "")
    assert "RACE-MARKER" not in session.output_buffer
    assert not any("RACE-MARKER" in text for text in emitted)

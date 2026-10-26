"""Native PTY Stop reaps detached descendants and closes after the final drain.

A detached descendant can keep the PTY slave open after its primary exits.
The Linux helper keeps custody through that exit; Stop must finish within its
native grace and leave no later output after its ACK. Real PTYs and processes
exercise the reader's buffer-lock behavior.
"""

import shutil
import threading
import time

import pytest

import tools.process_registry as module
from tools.process_registry import ProcessRegistry

_POSIX_PTY = pytest.mark.skipif(
    shutil.which("setsid") is None, reason="setsid executable required")

# Natural lifetime exceeds the Stop deadline, so a passing Stop requires cleanup.
_ESCAPEE_LIFETIME_S = 8
_KILL_DEADLINE_S = 5


@pytest.fixture
def native_local(monkeypatch):
    # Exercise the Linux job helper without the optional systemd transport.
    monkeypatch.setattr(module, "_is_supervised_gateway_process", lambda: False)


def _wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _terminate_owned(session):
    """Failure cleanup uses the birth-verified native job helper."""
    if not session.exited:
        from tools.process_registry_job import request_job_stop
        assert request_job_stop(ProcessRegistry(), session)


def _spawn_with_escapee(registry, tmp_path, late_output=""):
    pidfile = tmp_path / "escapee.pid"
    # Without ``late_output`` the escapee never writes to the PTY, so nothing ends the reader's
    # blocked read before the kill deadline. ``late_output`` is printed after the test has
    # received the native Stop ACK.
    gate = tmp_path / 'late-release'
    late = (f'for i in $(seq 1 80); do test -f {gate} && break; sleep .1; done; '
            f'test -f {gate} && echo {late_output}; ') if late_output else ""
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
    # The final reader releases the PTY master after the contained job closes
    # its slave, including a native Stop that crossed a read in progress.
    reader = session._reader_thread
    assert reader is not None
    assert _wait_for(lambda: not reader.is_alive(), _ESCAPEE_LIFETIME_S + 7)
    assert session._pty.closed


@pytest.mark.platforms("linux")
@_POSIX_PTY
def test_kill_reaps_escaped_descendant_without_blocking_on_pty_reader(tmp_path, native_local):
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


@pytest.mark.platforms("linux")
@_POSIX_PTY
def test_output_after_the_kill_is_not_added_to_the_killed_session(tmp_path, monkeypatch, native_local):
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
    # Release output only AFTER the native cleanup ACK. The contained escapee
    # must already be dead, so it cannot produce that later output/effect.
    (tmp_path / 'late-release').touch()
    _wait_for_reader_close(session)
    assert "LATE-ESCAPEE-OUTPUT" not in session.output_buffer
    assert not any("LATE-ESCAPEE-OUTPUT" in text for text in emitted)


@pytest.mark.platforms("linux")
@_POSIX_PTY
def test_chunk_read_before_the_kill_is_not_added_after_it(tmp_path, native_local):
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

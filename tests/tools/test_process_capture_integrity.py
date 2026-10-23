"""Background capture failures must survive completion and durable reload."""

import errno

import pytest

from tests.waiting import wait_until


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_background_capture_fault_survives_all_surfaces(tmp_path, monkeypatch, exit_code):
    from tools.environments import fd_wait
    from tools.process_registry import ProcessRegistry
    from tools.process_registry_notifications import format_process_notification

    registry = ProcessRegistry()
    original = fd_wait.open_fd_waiter

    def broken(fd):
        waiter = original(fd)
        def fail(timeout):
            raise OSError(errno.EIO, "private-token-must-not-leak")
        monkeypatch.setattr(waiter, "wait", fail)
        return waiter

    monkeypatch.setattr(fd_wait, "open_fd_waiter", broken)
    import subprocess
    import sys
    proc = subprocess.Popen([sys.executable, "-c", f"print('expected'); raise SystemExit({exit_code})"],
                            stdout=subprocess.PIPE, text=True)
    proc.wait(timeout=10)
    session = registry.adopt_local(proc, command=f"exited child {exit_code}", cwd=str(tmp_path), task_id="capture")
    wait_until(lambda: session._completion_event.is_set(), timeout=10, what="capture completion")
    result = registry.poll(session.id)
    assert result["exit_code"] == exit_code
    assert result.get("output_incomplete") is True
    assert result.get("capture_error")
    assert registry.read_log(session.id).get("output_incomplete") is True
    assert registry.wait(session.id, timeout=1).get("output_incomplete") is True
    assert "capture" in format_process_notification({
        "session_id": session.id, "command": session.command,
        "exit_code": exit_code, **result,
    }).lower()
    assert "private-token" not in str(result)


def test_capture_receipt_roundtrip_and_legacy_default(tmp_path, monkeypatch):
    import json
    from tools.process_registry import ProcessSession
    from tools.process_registry_results import save_completed_result, load_completed_results

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_ID", "capture-owner")
    session = ProcessSession(id="proc_capture", command="example", parent_session_id="capture-owner",
                             exited=True, exit_code=0, output_buffer="partial")
    session.capture_failed("stream read")
    save_completed_result(session)
    loaded = load_completed_results()[session.id]
    assert loaded.exit_code == 0 and loaded.output_buffer == "partial"
    assert loaded.capture_fields()["output_incomplete"] is True
    path = tmp_path / "logs" / "process-results" / "proc_capture.json"
    record = json.loads(path.read_text())
    del record["capture_error"]
    path.write_text(json.dumps(record))
    legacy = load_completed_results()[session.id]
    assert legacy.capture_fields() == {}
    assert legacy.exit_code == 0


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("sigpipe", [False, True])
def test_capture_failure_releases_full_pipe_and_reaps_child(tmp_path, monkeypatch, sigpipe):
    import subprocess
    import sys
    from tools.environments import fd_wait
    from tools.process_registry import ProcessRegistry

    def fail(fd):
        raise OSError(errno.EIO, "injected setup fault")

    monkeypatch.setattr(fd_wait, "open_fd_waiter", fail)
    script = "import os, signal\n"
    if sigpipe:
        script += "signal.signal(signal.SIGPIPE, signal.SIG_DFL)\n"
    script += ("try:\n while True: os.write(1, b'x' * 65536)\n"
               "except BrokenPipeError: os._exit(23)")
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    registry = ProcessRegistry()
    try:
        session = registry.adopt_local(proc, command="pipe-capacity writer", cwd=str(tmp_path))
        wait_until(session._completion_event.is_set, timeout=10, what="failed capture child reaped")
        session._reader_thread.join(timeout=2)
        assert not session._reader_thread.is_alive()
        result = registry.poll(session.id)
        assert result["output_incomplete"] is True
        assert result["exit_code"] == proc.returncode == (-13 if sigpipe else 23)
        assert proc.stdout.closed
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        proc.stdout.close()


def test_remote_log_capture_fault_does_not_advance_offset(monkeypatch):
    from unittest.mock import Mock
    from tools.process_registry import ProcessRegistry, ProcessSession

    registry = ProcessRegistry()
    session = ProcessSession(id="proc_remote", command="side-effecting job")
    env = Mock()
    env.execute.side_effect = [
        {"output": "6 0\nabc", "returncode": 0, "output_incomplete": True,
         "capture_error": "injected"},
        {"output": "0", "returncode": 0},
        {"output": "6 0\nabcdef", "returncode": 0},
        {"output": "1", "returncode": 0},
        {"output": "7", "returncode": 0},
        {"output": "10 6\ntail", "returncode": 0},
    ]
    offsets = []
    def log_command(path, offset):
        offsets.append(offset)
        return f"read-log {offset}"
    monkeypatch.setattr(registry, "_log_delta_command", log_command)
    monkeypatch.setattr("tools.process_registry.time.sleep", lambda _: None)
    registry._env_poller_loop(session, env, "log", "pid", "exit")
    assert offsets == [0, 0, 6]
    assert session.output_buffer == "abcdeftail"
    assert session.exit_code == 7
    assert session.capture_fields()["output_incomplete"] is True
    assert env.execute.call_count == 6


def test_adopted_log_decode_failure_still_records_exit(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from tools.process_registry import ProcessRegistry, ProcessSession
    from tools import process_registry_output_log as output_log
    log = tmp_path / "child.log"
    log.write_bytes(b"tail")
    log.with_suffix(".exit").write_text("7")
    session = ProcessSession(id="proc_log", command="example", output_log=str(log))
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_detached_host_fate", lambda *args: "gone")
    decoder = Mock()
    decoder.decode.side_effect = ValueError("private-token")
    monkeypatch.setattr(output_log.codecs, "getincrementaldecoder", lambda _: lambda **kw: decoder)
    registry._log_reader_loop(session)
    assert session.exited and session.exit_code == 7
    assert session.capture_fields()["output_incomplete"] is True
    assert session._completion_event.is_set()


@pytest.mark.platforms("posix")
def test_pty_decode_failure_still_reaps_and_records_signal(monkeypatch):
    from unittest.mock import Mock
    from tools.process_registry import ProcessRegistry, ProcessSession
    import tools.process_registry as process_module
    pty = Mock(exitstatus=None, signalstatus=15)
    pty.isalive.return_value = True
    pty.read.return_value = b"text"
    session = ProcessSession(id="proc_pty", command="example", _pty=pty)
    decoder = Mock()
    decoder.decode.side_effect = ValueError("private-token")
    monkeypatch.setattr(process_module.codecs, "getincrementaldecoder", lambda _: lambda **kw: decoder)
    ProcessRegistry()._pty_reader_loop(session)
    pty.wait.assert_called_once()
    assert session.exited and session.exit_code == -15
    assert session._completion_event.is_set()
    assert session.capture_fields()["output_incomplete"] is True

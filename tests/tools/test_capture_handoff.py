"""Lifecycle contracts at capture ownership boundaries."""
from unittest.mock import Mock

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession
from tools.environments.capture import OutputCaptureError


@pytest.mark.platforms("posix")
def test_pty_exit_does_not_discard_buffered_tail():
    pty = Mock()
    pty.isalive.return_value = False
    pty.read.side_effect = [b"last \xe2", b"\x9c\x93", EOFError()]
    pty.exitstatus = 7
    session = ProcessSession(id="proc_tail", command="example")
    session._pty = pty
    registry = ProcessRegistry()
    registry._pty_reader_loop(session)
    assert session.output_buffer == "last ✓"
    assert session.exit_code == 7
    assert not session.capture_error


@pytest.mark.parametrize("recovered", [True, False])
def test_uncertain_remote_launch_reads_pid_without_replay(monkeypatch, recovered):
    registry = ProcessRegistry()
    env = Mock()
    env.execute.side_effect = [
        {"output": "12", "returncode": 0, "output_incomplete": True},
        {"output": "12345" if recovered else "123", "returncode": 0,
         "output_incomplete": not recovered},
    ]
    monkeypatch.setattr(registry, "_track_started", Mock())
    if recovered:
        session = registry.spawn_via_env(env, "side-effecting job")
        assert session.pid == 12345
        assert session.capture_fields()["output_incomplete"] is True
    else:
        with pytest.raises(OutputCaptureError, match="may already"):
            registry.spawn_via_env(env, "side-effecting job")
    assert env.execute.call_count == 2
    assert "nohup" in env.execute.call_args_list[0].args[0]
    assert env.execute.call_args_list[1].args[0].startswith("cat ")


def test_yielded_spill_is_redacted_and_returned(tmp_path, monkeypatch):
    from pathlib import Path
    from tools.environments.base_output import _BoundedOutputCollector
    from tools.terminal_tool_background import yield_to_background_handler
    from tools import process_registry as registry_module

    collector = _BoundedOutputCollector(40, spill_path=tmp_path / "spill.log")
    collector.append("API_KEY=very-private-example-value\n" + "x" * 200)
    proc = Mock()
    proc._hermes_capture_collector = collector
    session = ProcessSession(id="proc_spill", command="env", pid=123)
    def adopt(*args, finalize_capture, **kwargs):
        finalize_capture()
        return session
    monkeypatch.setattr(registry_module.process_registry, "adopt_local", adopt)
    handler = yield_to_background_handler(command="env", env_type="local", cwd=str(tmp_path),
                                          effective_task_id="task", task_id="task", session_key="")
    result = handler(proc, collector.render())
    assert "very-private-example-value" not in Path(result["full_output_path"]).read_text()
    assert result["output_total_chars"] == collector.total_chars
    assert "incomplete" in result["truncation_note"]
    assert collector._spill_fh is None


def test_persistent_remote_capture_failure_still_observes_exit(monkeypatch):
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_remote_failed", command="job")
    env = Mock()
    env.execute.side_effect = [
        {"output": "9 0\npartial", "output_incomplete": True},
        {"output": "1"}, {"output": "7"},
        {"output": "9 0\npartial", "output_incomplete": True},
    ]
    monkeypatch.setattr("tools.process_registry_env_log.time.sleep", lambda _: None)
    checkpoint = Mock()
    monkeypatch.setattr(registry, "_write_checkpoint", checkpoint)
    registry._env_poller_loop(session, env, "log", "pid", "exit")
    assert session.exited and session.exit_code == 7
    assert session.output_buffer == ""
    assert session.capture_error
    assert env.execute.call_count == 4
    # One capture transition plus final lifecycle persistence, not every failed fetch.
    assert checkpoint.call_count == 2


@pytest.mark.platforms("posix")
def test_partial_utf8_decoder_transfers_to_background(monkeypatch):
    import io
    import os
    import threading
    from types import SimpleNamespace
    from tools.environments import base_output

    reader, writer = os.pipe()
    stream = io.TextIOWrapper(os.fdopen(reader, "rb"), encoding="utf-8")
    proc = SimpleNamespace(stdout=stream, stderr=None, stdin=None, returncode=0, wait=lambda: 0)
    collector = base_output._BoundedOutputCollector(100)
    stop = threading.Event()
    def partial(proc, fd, output, decoder, stop):
        output.append(decoder.decode(b"start \xe2"))
        stop.set()
    monkeypatch.setattr(base_output, "_drain_fd_select", partial)
    base_output._drain_stdout(proc, collector, stop)
    assert collector.render() == "start "
    assert not collector.capture_error
    os.write(writer, b"\x9c\x93 end")
    os.close(writer)
    session = ProcessSession(id="proc_utf8", command="job", process=proc)
    session.append_output(collector.render())
    registry = ProcessRegistry()
    registry._reader_loop(session)
    assert session.output_buffer == "start ✓ end"
    assert not session.capture_error


def test_late_read_failure_after_exit_reconciliation_is_reported():
    session = ProcessSession(id="proc_late", command="job")
    proc = Mock()
    proc.stdout.buffer.read1.side_effect = OSError("private")
    proc.returncode = 7
    session.process = proc
    session.mark_exited(7)
    registry = ProcessRegistry()
    registry._reader_loop(session)
    assert session.capture_fields()["output_incomplete"] is True
    assert session.exit_code == 7

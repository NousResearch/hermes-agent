"""Review regressions: capture uncertainty must not erase lifecycle facts."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_transient_remote_result_read_recovers_without_republishing(monkeypatch):
    import json
    from tools import code_kernel_remote as remote
    from tools.environments.capture import OutputCaptureError
    payload = {"status": "ok", "stdout": "complete"}
    kernel = SimpleNamespace(cell_seq=0, kernel_dir="/tmp/test", env=Mock(),
        sh=Mock(side_effect=[OutputCaptureError("incomplete"), json.dumps(payload), ""]))
    publish = Mock()
    monkeypatch.setattr("tools.code_execution_tool._ship_file_to_remote", publish)
    monkeypatch.setattr(remote.time, "sleep", Mock())
    assert remote._run_remote_cell(kernel, "side_effect()", 5) == ("ok", payload)
    publish.assert_called_once()


def test_prepublication_setup_failure_allows_fallback(monkeypatch):
    import json
    from tools import code_execution_tool as tool, code_kernel_remote as remote
    env = Mock()
    env.execute.return_value = {"output": "OK", "returncode": 0}
    monkeypatch.setattr(tool, "_load_config", dict)
    monkeypatch.setattr(tool, "_get_or_create_env", lambda _: (env, "ssh"))
    monkeypatch.setattr(remote, "_acquire_remote_kernel", Mock(side_effect=OSError("setup")))
    fallback = Mock(return_value='{"status":"success"}')
    monkeypatch.setattr(tool, "_run_remote_per_call", fallback)
    assert json.loads(tool._execute_remote("side_effect()", "task", []))["status"] == "success"
    fallback.assert_called_once()


@pytest.mark.platforms("windows")
def test_windows_pty_idle_is_not_eof():
    from tools.process_registry_output_log import iter_pty_output
    pty = Mock()
    pty.read.side_effect = ["", "tail", EOFError()]
    pty.isalive.return_value = True
    assert list(iter_pty_output(pty)) == ["tail"]


def test_remote_result_capture_exhaustion_remains_capture_error(monkeypatch):
    from tools import code_kernel_remote as remote
    from tools.environments.capture import OutputCaptureError
    kernel = SimpleNamespace(cell_seq=0, kernel_dir="/tmp/test", env=Mock(),
                             sh=Mock(side_effect=OutputCaptureError("incomplete")))
    monkeypatch.setattr("tools.code_execution_tool._ship_file_to_remote", Mock())
    monkeypatch.setattr(remote.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    monkeypatch.setattr(remote.time, "sleep", Mock())
    with pytest.raises(OutputCaptureError):
        remote._run_remote_cell(kernel, "side_effect()", 1)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_final_log_transport_failure_keeps_known_exit(monkeypatch, exit_code):
    from tools.process_registry import ProcessRegistry, ProcessSession
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_final", command="job")
    env = Mock()
    env.execute.side_effect = [{"output": "3 0\nabc"}, {"output": "1"},
                               {"output": str(exit_code)}, OSError("private")]
    monkeypatch.setattr("tools.process_registry_env_log.time.sleep", Mock())
    registry._env_poller_loop(session, env, "log", "pid", "exit")
    assert session.exit_code == exit_code
    assert session.capture_fields()["output_incomplete"] is True
    assert session.output_buffer == "abc"


@pytest.mark.parametrize("more_output", [False, True])
def test_failed_adoption_preserves_foreground_spill(tmp_path, monkeypatch, more_output):
    from tools.environments.base_output import _BoundedOutputCollector, _finalize_wait_result
    from tools.terminal_tool_background import yield_to_background_handler
    collector = _BoundedOutputCollector(20, spill_path=tmp_path / "spill.log")
    collector.append("x" * 200)
    proc = SimpleNamespace(_hermes_capture_collector=collector)
    handler = yield_to_background_handler(command="printf", env_type="local", cwd=str(tmp_path),
        effective_task_id="task", task_id="task", session_key="")
    monkeypatch.setattr("tools.process_registry.process_registry.adopt_local",
                        Mock(side_effect=RuntimeError("cannot start reader")))
    with pytest.raises(RuntimeError):
        handler(proc, collector.render())
    if more_output:
        collector.append("tail")
    result = _finalize_wait_result(collector, collector.render(), 0)
    assert result["full_output_path"] == str(tmp_path / "spill.log")
    assert (tmp_path / "spill.log").read_text() == "x" * 200 + ("tail" if more_output else "")
    assert not result.get("output_incomplete")


@pytest.mark.platforms("linux")
def test_natural_pty_exit_does_not_wait_for_descendant_slave(tmp_path, monkeypatch):
    import sys
    from ptyprocess import PtyProcess
    from tools.process_registry import ProcessRegistry, ProcessSession
    # The helper forks a silent, finite-lived descendant retaining the slave.
    # No external process is signalled; the descendant also exits when released.
    release = tmp_path / "release"
    ready = tmp_path / "ready"
    code = (
        "import os,time,signal; from pathlib import Path; "
        "signal.signal(signal.SIGHUP, signal.SIG_IGN); pid=os.fork(); "
        f"release=Path({str(release)!r}); ready=Path({str(ready)!r}); "
        "\nif pid == 0:\n os.setsid(); ready.touch(); deadline=time.monotonic()+10\n"
        " while not release.exists() and time.monotonic()<deadline: time.sleep(.05)\n os._exit(0)\n"
        "while not ready.exists(): time.sleep(.01)\nprint('final-tail', flush=True)\nos._exit(7)"
    )
    pty = PtyProcess.spawn([sys.executable, "-c", code])
    session = ProcessSession(id="proc_pty_final", command="job")
    session._pty = pty
    registry = ProcessRegistry()
    thread = threading.Thread(target=registry._pty_reader_loop, args=(session,), daemon=True)
    thread.start()
    try:
        thread.join(3)
        assert not thread.is_alive(), "PTY completion waited for descendant instead of direct child"
        assert session.exit_code == 7
        assert "final-tail" in session.output_buffer
    finally:
        release.touch()
        thread.join(12)
        pty.close()

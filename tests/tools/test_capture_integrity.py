"""Capture integrity is independent of a real child's exit status.

Faults below are deterministic collector injection, not natural OS failures.
"""

import errno
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest

from tools.environments import base_output
from tools.environments.local import LocalEnvironment


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("fault", ["setup", "wait", "read"])
def test_capture_failure_preserves_child_exit(tmp_path, monkeypatch, exit_code, fault):
    env = LocalEnvironment(cwd=str(tmp_path))
    proc = subprocess.Popen(
        [sys.executable, "-c", f"print('expected'); raise SystemExit({exit_code})"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    proc.wait(timeout=10)
    original_open = base_output.open_fd_waiter

    def fail(*args):
        raise OSError(errno.EIO, "private-token-must-not-leak")

    def broken_waiter(fd):
        waiter = original_open(fd)
        monkeypatch.setattr(waiter, "wait", fail)
        return waiter

    if fault == "setup":
        monkeypatch.setattr(base_output, "open_fd_waiter", fail)
    elif fault == "wait":
        monkeypatch.setattr(base_output, "open_fd_waiter", broken_waiter)
    else:
        monkeypatch.setattr(base_output.os, "read", fail)
    try:
        result = env._wait_for_process(proc)
    finally:
        proc.stdout.close()
        env.cleanup()
    assert result["returncode"] == exit_code
    assert result.get("output_incomplete") is True
    assert result.get("capture_error")
    assert "private-token" not in str(result)


@pytest.mark.platforms("posix")
def test_partial_capture_survives_read_failure(tmp_path, monkeypatch):
    env = LocalEnvironment(cwd=str(tmp_path))
    proc = subprocess.Popen(
        [sys.executable, "-c", "print('partial-✓', flush=True)"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    proc.wait(timeout=10)
    read = base_output.os.read
    calls = 0

    def fail_after_data(fd, size):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError(errno.EIO, "private-token-must-not-leak")
        return read(fd, size)

    monkeypatch.setattr(base_output.os, "read", fail_after_data)
    try:
        result = env._wait_for_process(proc)
    finally:
        proc.stdout.close()
        env.cleanup()
    assert result["returncode"] == 0
    assert result["output"] == "partial-✓\n"
    assert result.get("output_incomplete") is True
    assert result.get("capture_error")


@pytest.mark.platforms("posix")
def test_quiet_success_is_not_capture_failure(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        result = env.execute("true")
    finally:
        env.cleanup()
    assert result["returncode"] == 0
    assert not result.get("output_incomplete")
    assert not result.get("capture_error")


@pytest.mark.parametrize("diagnostic", ["Output capture failed; output may be incomplete.", ""])
def test_foreground_json_preserves_capture_failure(tmp_path, diagnostic):
    from tools.terminal_tool_result import finalize_foreground_result

    result = json.loads(finalize_foreground_result(
        command="printf partial", result={
            "output": "partial", "returncode": 0,
            "capture_error": diagnostic,
            "output_incomplete": True,
        }, env=Mock(cwd=str(tmp_path)), env_type="local", effective_task_id="capture",
        task_id="capture", session_id="capture", session_key="capture",
        workdir=None, command_cwd=str(tmp_path), approval_note=None,
    ))
    assert result["exit_code"] == 0
    assert result["output"] == "partial"
    assert result.get("output_incomplete") is True
    assert result.get("capture_error")
    assert result["error"]


def test_file_consumer_rejects_incomplete_payload_without_replay(tmp_path):
    from tools.file_operations import ShellFileOperations

    env = Mock(cwd=str(tmp_path))
    env.execute.return_value = {
        "output": "plausible partial file", "returncode": 0,
        "output_incomplete": True, "capture_error": "Output capture failed.",
    }
    ops = ShellFileOperations(env)
    with pytest.raises(RuntimeError, match="capture"):
        ops._exec("cat example")
    env.execute.assert_called_once()


def test_spill_finalize_failure_is_reported(tmp_path):
    collector = base_output._BoundedOutputCollector(8, spill_path=tmp_path / "spill")
    collector.append("enough text to spill")
    collector._spill_fh.close()
    collector._spill_fh = Mock(close=Mock(side_effect=OSError("private-token")))
    result = base_output._finalize_wait_result(collector, collector.render(), 0)
    assert result["returncode"] == 0
    assert result.get("output_incomplete") is True
    assert result.get("capture_error")
    assert "private-token" not in str(result)


def test_iterable_failure_preserves_partial_output():
    def broken_stream():
        yield "partial"
        raise OSError("private-token")

    collector = base_output._BoundedOutputCollector(1024)
    base_output._drain_stdout(Mock(stdout=broken_stream()), collector)
    result = base_output._finalize_wait_result(collector, collector.render(), 0)
    assert result["output"] == "partial"
    assert result.get("output_incomplete") is True
    assert result.get("capture_error")


def test_persisted_preview_retains_capture_failure(tmp_path, monkeypatch):
    from tools.tool_result_storage import BudgetConfig, maybe_persist_tool_result

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    content = json.dumps({
        "output": "x" * 5000, "exit_code": 7,
        "capture_error": "Output capture failed.", "output_incomplete": True,
        "error": "Output capture failed.",
    })
    result = maybe_persist_tool_result(
        content, "terminal", "capture-preview", threshold=100,
        config=BudgetConfig(preview_size=200),
    )
    assert '"output_incomplete": true' in result
    assert '"exit_code": 7' in result
    assert "Output capture failed" in result


@pytest.mark.platforms("posix")
def test_foreground_live_writer_capture_failure_reaps_actual_exit(tmp_path, monkeypatch):
    env = LocalEnvironment(cwd=str(tmp_path))
    proc = subprocess.Popen([sys.executable, "-c",
        "import os, signal; signal.signal(signal.SIGPIPE, signal.SIG_DFL)\n"
        "while True: os.write(1, b'x' * 65536)"], stdout=subprocess.PIPE)
    def fail(fd):
        raise OSError(errno.EIO, "injected")
    monkeypatch.setattr(base_output, "open_fd_waiter", fail)
    try:
        result = env._wait_for_process(proc, timeout=5)
        assert result["returncode"] == proc.returncode == -13
        assert result["output_incomplete"] is True
        assert not result.get("hermes_timed_out")
        assert proc.stdout.closed
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        proc.stdout.close()
        env.cleanup()


def test_intentional_reader_stop_is_not_capture_failure():
    import io
    import threading
    stop = threading.Event()
    stop.set()
    collector = base_output._BoundedOutputCollector(1024)
    proc = Mock(stdout=io.BytesIO(b""))
    base_output._drain_stdout(proc, collector, stop)
    assert collector.capture_fields() == {}


def test_redaction_spill_failure_reaches_final_json(tmp_path, monkeypatch):
    from tools.terminal_tool_result import finalize_foreground_result
    spill = tmp_path / "spill.log"
    spill.write_text("middle")
    def fail(*args, **kwargs):
        raise OSError("private-token")
    monkeypatch.setattr("tools.spill_safety.write_text_exclusive", fail)
    result = json.loads(finalize_foreground_result(
        command="example", result={"output": "head tail", "returncode": 0,
            "full_output_path": str(spill), "output_total_chars": 9999},
        env=Mock(cwd=str(tmp_path)), env_type="local", effective_task_id="capture",
        task_id="capture", session_id="capture", session_key="capture", workdir=None,
        command_cwd=str(tmp_path), approval_note=None,
    ))
    assert result["exit_code"] == 0
    assert result["output_incomplete"] is True and result["error"]
    assert "verification_evidence" not in result
    assert "full_output_path" not in result
    assert "private-token" not in str(result)

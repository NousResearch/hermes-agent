"""The shared SDK adapter must not turn producer loss into clean pipe EOF."""
import errno
import os
import threading

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.base_output import _ThreadedProcessHandle
from unittest.mock import Mock


class SDKEnvironment(BaseEnvironment):
    def cleanup(self):
        pass


@pytest.mark.parametrize("sdk_exit", [0, 7])
@pytest.mark.parametrize("fault", ["short", "error", "partial_error", "zero"])
def test_sdk_emission_integrity(tmp_path, monkeypatch, sdk_exit, fault):
    env = SDKEnvironment(cwd=str(tmp_path), timeout=5)
    release = threading.Event()
    executions = []
    text = "prefix-✓-完整"

    def execute():
        executions.append(True)
        assert release.wait(5)
        return text, sdk_exit

    proc = _ThreadedProcessHandle(execute)
    write = os.write
    writes = []

    def emit(fd, data):
        if fd != proc._write_fd:
            return write(fd, data)
        writes.append(len(data))
        if fault == "error" or (fault == "partial_error" and len(writes) > 1):
            raise OSError(errno.EIO, "private-token-must-not-leak")
        if fault == "zero":
            return 0
        return write(fd, data[:2] if fault == "short" else data[:6])

    monkeypatch.setattr("tools.environments.base_output.os.write", emit)
    try:
        release.set()
        result = env._wait_for_process(proc, timeout=5)
        assert result["returncode"] == sdk_exit
        assert proc._done.wait(2)
        assert proc.stdout.closed
        assert len(executions) == 1
        assert "private-token" not in str(result)
        if fault == "short":
            assert result["output"] == text
            assert not result.get("output_incomplete")
        else:
            assert result["output"] == ("prefix" if fault == "partial_error" else "")
            assert result.get("output_incomplete") is True
            assert result.get("capture_error")
    finally:
        release.set()
        proc.wait(timeout=5)
        proc.stdout.close()


def test_drain_start_failure_returns_integrity_not_retry_exception(tmp_path, monkeypatch):
    import subprocess
    import sys
    from tools.environments import base_output
    env = SDKEnvironment(cwd=str(tmp_path), timeout=5)
    proc = subprocess.Popen([sys.executable, "-c", "print('output')"], stdout=subprocess.PIPE)
    proc.wait(timeout=5)
    fake_thread = Mock()
    fake_thread.start.side_effect = RuntimeError("cannot start new thread")
    monkeypatch.setattr(base_output.threading, "Thread", Mock(return_value=fake_thread))
    try:
        result = env._wait_for_process(proc, timeout=5)
        assert result["returncode"] == 0
        assert result["output_incomplete"] is True
        assert proc.stdout.closed
    finally:
        proc.stdout.close()

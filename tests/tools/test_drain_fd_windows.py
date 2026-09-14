"""Regression tests for issue #105865: Windows stdout drain hangs when grandchild inherits write pipe."""

import codecs
import os
import threading
import time
from unittest.mock import MagicMock

import pytest

from tools.environments.base_output import _BoundedOutputCollector, _drain_fd_windows


@pytest.mark.windows_only
def test_drain_fd_windows_returns_promptly_when_writer_remains_open():
    """Windows drain must stop shortly after process exit even if a grandchild holds the pipe write handle open."""
    r, w = os.pipe()
    try:
        os.write(w, b"hello from child process\n")
        proc = MagicMock()
        proc.poll.return_value = 0  # Process has exited, but 'w' is still open

        output = _BoundedOutputCollector(1000)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        t0 = time.monotonic()
        _drain_fd_windows(proc, r, output, decoder)
        elapsed = time.monotonic() - t0

        assert "hello from child process" in output.render()
        assert elapsed < 1.0, f"drain hung for {elapsed:.2f}s instead of bounding after exit"
    finally:
        os.close(r)
        os.close(w)


@pytest.mark.windows_only
def test_drain_fd_windows_observes_stop_event():
    """Windows drain must honor the stop event immediately when handed to another reader."""
    r, w = os.pipe()
    try:
        proc = MagicMock()
        proc.poll.return_value = None  # Process still running

        stop = threading.Event()
        stop.set()

        output = _BoundedOutputCollector(1000)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        t0 = time.monotonic()
        _drain_fd_windows(proc, r, output, decoder, stop=stop)
        elapsed = time.monotonic() - t0

        assert elapsed < 0.2, f"drain took {elapsed:.2f}s despite stop event being set"
    finally:
        os.close(r)
        os.close(w)


@pytest.mark.windows_only
def test_drain_fd_windows_e2e_local_environment_background_child(tmp_path):
    """LocalEnvironment.execute() on Windows must return promptly when a backgrounded
    grandchild inherits the pipe writer (reproducer for issue #105865 / #67362)."""
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        marker = "windows_drain_e2e_marker"
        # In Git Bash on Windows, backgrounding with disown leaves the grandchild alive
        # with an open stdout pipe write handle. The drain must bound to ~300ms.
        cmd = f'python -c "import time; time.sleep(30)" & disown; echo {marker}'
        t0 = time.monotonic()
        result = env.execute(cmd, timeout=15)
        elapsed = time.monotonic() - t0

        assert elapsed < 5.0, f"LocalEnvironment.execute hung for {elapsed:.2f}s on Windows"
        assert result["returncode"] == 0
        assert marker in result["output"]
    finally:
        env.cleanup()

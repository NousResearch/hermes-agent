"""Windows stdout drain: bounded when a grandchild keeps the pipe's write handle open.

``select()`` cannot poll pipe fds on Windows, so the drain used a blocking
``os.read`` loop that waited for true EOF — a backgrounded grandchild that
inherited the write end kept the terminal tool hung for its whole lifetime.
The ``PeekNamedPipe`` drain mirrors the POSIX ``select`` loop's post-exit bound.
"""

import codecs
import os
import time
from unittest.mock import MagicMock

import pytest

from tools.environments.base_output import _BoundedOutputCollector, _drain_fd_windows


@pytest.mark.platforms("windows")
def test_drain_fd_windows_returns_promptly_when_writer_remains_open():
    """Data already in the pipe is captured; the drain stops shortly after the
    process reports exit even though the write end is still open."""
    r, w = os.pipe()
    try:
        os.write(w, b"hello from child process\n")
        proc = MagicMock()
        proc.poll.return_value = 0  # bash exited; ``w`` (the grandchild's copy) is still open

        output = _BoundedOutputCollector(1000)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        t0 = time.monotonic()
        _drain_fd_windows(proc, r, output, decoder)
        elapsed = time.monotonic() - t0

        assert "hello from child process" in output.render()
        assert elapsed < 2.0, f"drain hung for {elapsed:.2f}s instead of bounding after exit"
    finally:
        os.close(r)
        os.close(w)


@pytest.mark.platforms("windows")
def test_local_environment_returns_while_background_grandchild_holds_pipe(tmp_path):
    """End-to-end on Git Bash: ``execute()`` returns promptly with the marker while a
    backgrounded grandchild still holds the stdout pipe (issues #105865 / #67362)."""
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        marker = "windows_drain_e2e_marker"
        cmd = f'python -c "import time; time.sleep(30)" & echo {marker}'
        t0 = time.monotonic()
        result = env.execute(cmd, timeout=15)
        elapsed = time.monotonic() - t0

        assert elapsed < 10.0, f"LocalEnvironment.execute hung for {elapsed:.2f}s on Windows"
        assert result["returncode"] == 0
        assert marker in result["output"]
    finally:
        env.cleanup()

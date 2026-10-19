"""Regression: the POSIX stdout drain must not lose output when a command's
pipe fd is at or above FD_SETSIZE (1024).

A long-lived gateway/desktop process accumulates open fds over uptime. Once a
command's stdout pipe fd reaches 1024, the drain's old ``select.select()`` call
raises ``ValueError: filedescriptor out of range in select()``. That ValueError
was caught and treated as "fd already closed", so ``terminal`` and ``read_file``
returned an empty string with ``exit_code 0``; output silently discarded with
no error signal (issue #94928). The drain now waits with ``select.poll()``,
which has no FD_SETSIZE ceiling.
"""

import codecs
import os
from unittest.mock import MagicMock

import pytest

from tools.environments.base_output import _BoundedOutputCollector, _drain_fd_select

# Enough held fds to push the next pipe fd past FD_SETSIZE (1024).
_TARGET_LOW_FDS = 1100


def _hold_high_fds():
    """Best-effort raise the soft fd limit, then hold fds past FD_SETSIZE.

    Returns the list of held fd ints, or None when the host cannot reach the
    count; the caller skips rather than assert a false repro.
    """
    import resource  # POSIX-only; this file is never imported on the Windows lane

    want = _TARGET_LOW_FDS + 200
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < want:
        new_soft = want if hard == resource.RLIM_INFINITY else min(hard, want)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        except (ValueError, OSError):
            pass
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < want:
            return None
    held = []
    while len(held) < _TARGET_LOW_FDS:
        held.append(os.open(os.devnull, os.O_RDONLY))
    if held[-1] < 1024:
        for fd in held:
            os.close(fd)
        return None
    return held


def _release(held):
    for fd in held:
        try:
            os.close(fd)
        except OSError:
            pass


@pytest.mark.platforms("posix")
def test_drain_fd_select_captures_output_above_fd_setsize():
    """The drain's readiness wait must not drop output when the pipe fd >= 1024."""
    held = _hold_high_fds()
    if held is None:
        pytest.skip("fd limit too low to reproduce")
    r, w = os.pipe()
    try:
        assert r >= 1024, "setup failed to push the pipe fd above FD_SETSIZE"
        os.write(w, b"hello-high-fd\n")
        os.close(w)  # true EOF once drained

        proc = MagicMock()
        proc.poll.return_value = 0
        output = _BoundedOutputCollector(1000)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        _drain_fd_select(proc, r, output, decoder)

        assert "hello-high-fd" in output.render()
    finally:
        try:
            os.close(r)
        except OSError:
            pass
        _release(held)


@pytest.mark.platforms("posix")
def test_terminal_output_survives_pipe_fd_above_fd_setsize(tmp_path):
    """End-to-end: ``terminal`` still captures stdout once the pipe fd is >= 1024."""
    held = _hold_high_fds()
    if held is None:
        pytest.skip("fd limit too low to reproduce")
    try:
        assert held[-1] >= 1024, "setup failed to push fds above FD_SETSIZE"

        from tools.environments.local import LocalEnvironment

        env = LocalEnvironment(cwd=str(tmp_path))
        try:
            result = env.execute("echo hello-high-fd")
        finally:
            env.cleanup()

        assert result["returncode"] == 0
        assert "hello-high-fd" in result["output"]
    finally:
        _release(held)

"""Regression for #94928: no output reader may lose output once its fd is at or
above FD_SETSIZE (1024).

A long-lived gateway/desktop process accumulates open fds over uptime. Once a
command's pipe fd reached 1024, ``select.select()`` raised ``ValueError:
filedescriptor out of range in select()``; the foreground drain, the background
reader and the PTY bridge all caught it as "fd already closed", so ``terminal``
returned an empty string with ``exit_code 0``. Readiness now goes through a
selector (epoll/kqueue). The PTY case also pins that an idle read really waits:
macOS ``poll()`` reports a PTY master as POLLNVAL ("ready" at once), which would
turn the dashboard terminal's read loop into a busy spin.
"""

import os
import time

import pytest

from tests.waiting import wait_until

# Enough held fds to push the next pipe fd past FD_SETSIZE (1024).
_TARGET_LOW_FDS = 1100


@pytest.fixture
def high_fds():
    """Raise the soft fd limit, then hold fds so every new fd lands above FD_SETSIZE."""
    import resource  # POSIX-only; this file is never imported on the Windows lane

    want = _TARGET_LOW_FDS + 200
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < want:
        new_soft = want if hard == resource.RLIM_INFINITY else min(hard, want)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        except (ValueError, OSError):
            pass
        if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < want:
            pytest.skip("fd limit too low to reproduce")
    held = []
    try:
        while len(held) < _TARGET_LOW_FDS:
            held.append(os.open(os.devnull, os.O_RDONLY))
        assert held[-1] >= 1024, "setup failed to push fds above FD_SETSIZE"
        yield
    finally:
        for fd in held:
            os.close(fd)


@pytest.mark.platforms("posix")
def test_command_output_survives_pipe_fd_above_fd_setsize(high_fds, tmp_path, monkeypatch):
    """Foreground ``terminal`` drain and the background-process reader both keep output,
    including when the selector itself cannot be created (EMFILE: epoll/kqueue need an fd)."""
    import errno

    from tools.environments import fd_wait
    from tools.environments.local import LocalEnvironment
    from tools.process_registry import ProcessRegistry

    def _emfile():
        raise OSError(errno.EMFILE, "Too many open files")

    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        result = env.execute("echo hello-high-fd")
        with monkeypatch.context() as m:
            m.setattr(fd_wait.selectors, "DefaultSelector", _emfile)
            at_fd_limit = env.execute("echo still-here")
    finally:
        env.cleanup()
    assert result["returncode"] == 0
    assert "hello-high-fd" in result["output"]
    assert "still-here" in at_fd_limit["output"]

    registry = ProcessRegistry()
    session = registry.spawn_local("echo bg-high-fd", cwd=str(tmp_path), task_id="high-fd")
    wait_until(lambda: session.exited, timeout=10, what="background process exit")
    wait_until(lambda: "bg-high-fd" in session.output_buffer, timeout=5,
               what=f"background output (got {session.output_buffer!r})")


@pytest.mark.platforms("posix")
def _assert_idle_read_waits(bridge):
    started = time.monotonic()
    assert bridge.read(timeout=0.5) == b""
    assert time.monotonic() - started >= 0.3, "idle PTY read returned without waiting"


@pytest.mark.platforms("posix")
def test_pty_bridge_reads_and_waits_above_fd_setsize(high_fds, monkeypatch):
    """The PTY bridge delivers output at a high fd and an idle read blocks for its timeout;
    when the selector cannot be created (EMFILE), the bridge either still waits (poll
    fallback) or refuses to spawn, never a busy-looping bridge (macOS poll on a PTY)."""
    import errno

    pytest.importorskip("ptyprocess")
    from hermes_cli.pty_bridge import PtyBridge
    from tools.environments import fd_wait

    bridge = PtyBridge.spawn(["/bin/sh", "-c", "echo pty-high-fd; exec sleep 30"])
    try:
        assert bridge._fd >= 1024
        buf = bytearray()
        deadline = time.monotonic() + 5
        while b"pty-high-fd" not in buf and time.monotonic() < deadline:
            chunk = bridge.read(timeout=0.2)
            assert chunk is not None, f"PTY reported EOF before output (got {bytes(buf)!r})"
            buf.extend(chunk)
        assert b"pty-high-fd" in buf
        _assert_idle_read_waits(bridge)
    finally:
        bridge.close()

    def _emfile():
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(fd_wait.selectors, "DefaultSelector", _emfile)
    try:
        fallback = PtyBridge.spawn(["/bin/sh", "-c", "exec sleep 30"])
    except OSError:
        return  # no safe wait for this device on this OS: the spawn is refused
    try:
        _assert_idle_read_waits(fallback)
    finally:
        fallback.close()

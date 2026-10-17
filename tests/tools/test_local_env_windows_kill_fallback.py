"""Windows terminate path must not orphan descendants when the tree kill fails.

Regression for a terminal timeout whose ``taskkill /T /F`` failed: the ``proc.kill()``
fallback signalled only the bash wrapper and a child (grep) outlived it. Mock-only, no
real processes.
"""
import subprocess

import pytest

from tools.environments import local as local_env


class _FakeChild:
    def __init__(self, running=True):
        self.running = running
        self.killed = False

    def is_running(self):
        return self.running

    def kill(self):
        self.killed = True


class _FakeProc:
    pid = 4242

    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 1


def _patch_children(monkeypatch, children):
    import psutil

    class _P:
        def __init__(self, pid):
            pass

        def children(self, recursive=False):
            assert recursive is True
            return list(children)

    monkeypatch.setattr(psutil, "Process", _P)


def test_descendants_killed_when_tree_kill_fails(monkeypatch):
    import gateway.status as gs

    alive, gone = _FakeChild(), _FakeChild(running=False)
    _patch_children(monkeypatch, [alive, gone])
    monkeypatch.setattr(gs, "get_process_start_time", lambda pid: 1.0)

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired("taskkill", 10)

    monkeypatch.setattr(gs, "terminate_pid", _boom)
    proc = _FakeProc()

    local_env._kill_process_windows(proc)

    assert proc.killed, "fallback must still kill the wrapper"
    assert alive.killed, "surviving descendant must be killed by PID"
    assert not gone.killed, "already-exited descendant is skipped"


def test_snapshot_failure_does_not_break_kill(monkeypatch):
    import gateway.status as gs
    import psutil

    def _nope(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", _nope)
    monkeypatch.setattr(gs, "get_process_start_time", lambda pid: 1.0)
    monkeypatch.setattr(gs, "terminate_pid", lambda *a, **k: None)
    proc = _FakeProc()

    local_env._kill_process_windows(proc)  # must not raise

    assert not proc.killed, "terminate_pid succeeded; no fallback kill needed"


def test_clean_tree_kill_leaves_nothing_to_sweep(monkeypatch):
    import gateway.status as gs

    child = _FakeChild(running=False)  # taskkill /T already took it
    _patch_children(monkeypatch, [child])
    monkeypatch.setattr(gs, "get_process_start_time", lambda pid: 1.0)
    monkeypatch.setattr(gs, "terminate_pid", lambda *a, **k: None)

    local_env._kill_process_windows(_FakeProc())

    assert not child.killed

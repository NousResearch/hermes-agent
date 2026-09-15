"""Regression tests for bounded kill paths (terminal kill-path deadlock).

A terminal tool child that refuses to die (D-state from a stalled network
syscall, wedged /proc walk under memory pressure) used to wedge the kill path
itself for the full duration of the hang — 61 minutes in the 2026-09-07
production incident — because psutil children walks and the kill helpers had
no wall-clock bound. The timeout had fired at 60s; the kill never returned.
These tests pin the hard budgets so the kill path can never hold the caller
(and through it the gateway event loop) past a few seconds.
"""

import importlib
import subprocess
import sys
import time

import psutil
import pytest

from tools.environments.base import BaseEnvironment, _run_best_effort_bounded


def _block_forever():
    time.sleep(3600)


class _StubEnv(BaseEnvironment):
    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError("use mocks")

    def cleanup(self):
        pass


class TestBestEffortBounded:
    def test_returns_within_budget_when_fn_blocks(self):
        start = time.monotonic()
        _run_best_effort_bounded(_block_forever, 0.5, "test-block")
        assert time.monotonic() - start < 3.0

    def test_returns_promptly_when_fn_finishes(self):
        start = time.monotonic()
        _run_best_effort_bounded(lambda: None, 5.0, "test-fast")
        assert time.monotonic() - start < 2.0

    def test_best_effort_helper_never_raises(self):
        def _boom():
            raise RuntimeError("boom")

        start = time.monotonic()
        _run_best_effort_bounded(_boom, 0.5, "test-boom")  # must not raise
        assert time.monotonic() - start < 3.0


class TestKillSpawnedTreeBounded:
    def test_returns_when_kill_process_wedges(self, monkeypatch):
        env = _StubEnv()
        env._kill_process = lambda proc: _block_forever()  # wedge the kill
        spawned = subprocess.Popen(["sleep", "0.05"])
        try:
            start = time.monotonic()
            env._kill_spawned_tree(spawned)  # budget 3s on _kill_process
            assert time.monotonic() - start < 6.0
        finally:
            if spawned.poll() is None:
                spawned.kill()
            spawned.wait()

    def test_returns_when_tree_kill_wedges(self, monkeypatch):
        env = _StubEnv()
        from agent import deadline as deadline_mod

        monkeypatch.setattr(deadline_mod, "kill_process_tree", lambda pid: _block_forever())
        spawned = subprocess.Popen(["sleep", "0.05"])
        try:
            start = time.monotonic()
            env._kill_spawned_tree(spawned)  # budget 6s on kill_process_tree
            assert time.monotonic() - start < 9.0
        finally:
            if spawned.poll() is None:
                spawned.kill()
            spawned.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group kill only")
class TestKillProcessGroupPosixBounded:
    @pytest.mark.live_system_guard_bypass  # real SIGTERM/SIGKILL to our own isolated-session child only
    def test_returns_when_psutil_walk_blocks(self, monkeypatch):
        local_env = importlib.import_module("tools.environments.local")

        class _BlockingProcess:
            def __init__(self, pid):
                self._pid = pid

            def children(self, recursive=False):
                _block_forever()
                return []

        monkeypatch.setattr(psutil, "Process", _BlockingProcess)

        # Spawn in its OWN session so the group kill can never touch the test runner.
        proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        try:
            start = time.monotonic()
            local_env._kill_process_group_posix(proc)  # psutil budget 1s
            assert time.monotonic() - start < 4.0
        finally:
            proc.wait(timeout=3)
        assert proc.poll() is not None  # the sleep was still killed via killpg
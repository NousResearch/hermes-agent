"""Tests for the Windows tree-kill contract in ``_kill_process_windows`` (#132958).

The contract under test is host-independent: whatever the taskkill channel does —
refused by the identity guard, or a ``/T`` sweep that misses a broken-chain
descendant — the descendants snapshotted before the first signal must not survive.
Only the Windows-only channel (``gateway.status.terminate_pid`` /
``get_process_start_time``) is stubbed; the wrapper process, its child, psutil's
snapshot and every kill are real, per the house rule of mocking dependencies
rather than the host. The sweep kills real descendants whose parent chain is
already broken (the wrapper died first), which the live-system guard would
mistake for a stray host process — hence the bypass mark, same as the POSIX
sibling test (``test_local_setsid_descendant_sweep.py``).
"""

from __future__ import annotations

import subprocess
import time
from unittest import mock

import psutil
import pytest

from tools.environments import local as local_mod

pytestmark = pytest.mark.live_system_guard_bypass


def _children_of(pid):
    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return []


def _spawn_wrapper_with_child(tmp_path):
    """A real wrapper bash holding a real child (``sleep``), as Git Bash holds find.exe."""
    proc = subprocess.Popen(
        ["bash", "-c", "sleep 600 & wait"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        children = _children_of(proc.pid)
        if children:
            return proc, children
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("wrapper never spawned its child")


def _alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_dead(pid, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


class TestKillProcessWindowsTreeContract:
    def test_refused_taskkill_still_kills_descendants(self, tmp_path):
        """A taskkill refused by the identity guard used to degrade to a bare
        ``proc.kill()``: the wrapper died, the child survived as an orphan."""
        proc, children = _spawn_wrapper_with_child(tmp_path)
        with (
            mock.patch("gateway.status.get_process_start_time", return_value=None),
            mock.patch(
                "gateway.status.terminate_pid",
                side_effect=OSError(
                    "refusing to force-kill PID; start time is unavailable"
                ),
            ),
        ):
            local_mod._kill_process_windows(proc)
        assert _wait_dead(proc.pid)
        for child in children:
            assert _wait_dead(child.pid), (
                f"descendant {child.pid} survived a refused taskkill"
            )

    def test_sweep_kills_descendants_the_tree_kill_missed(self, tmp_path):
        """taskkill /T walks the parent-PID chain and can miss a member whose chain
        broke; the snapshot sweep must still catch it."""
        proc, children = _spawn_wrapper_with_child(tmp_path)

        def missed_the_child(pid, *, force=False, expected_start_time=None):
            # Simulates /T enumerating only the root: the wrapper dies, the child survives.
            assert pid == proc.pid
            proc.kill()

        with (
            mock.patch("gateway.status.get_process_start_time", return_value=12345),
            mock.patch("gateway.status.terminate_pid", side_effect=missed_the_child),
        ):
            local_mod._kill_process_windows(proc)
        for child in children:
            assert _wait_dead(child.pid), (
                f"descendant {child.pid} survived a /T sweep that missed it"
            )

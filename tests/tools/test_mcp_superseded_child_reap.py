"""Regression test for #126990: stdio MCP children superseded by a failed
transport attempt must be reaped when the attempt ends — not left for the next
attempt's entry sweep.

``_run_stdio``'s finally hands a still-live child to the orphan ledger
(``_release_spawned_children``); the only consumer of that ledger was the entry
sweep of the NEXT attempt. While the run task parks awaiting a revive — or dies
cancelled — no next attempt comes, so a reconnect/revive loop accumulated one
orphaned bridge process per attempt for the life of the gateway.

The fix reaps the server's orphans at two structural boundaries where "next
attempt" would otherwise be delayed or never arrive: the run loop's
per-iteration finally and the park entry. These tests drive the real reaper
(``_kill_orphaned_mcp_children``) against a real stubborn child process that
ignores SIGTERM, so the TERM -> grace -> KILL escalation itself is under test.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from unittest.mock import patch

import pytest

pytest.importorskip("mcp")

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")


def _spawn_stubborn_child() -> int:
    """Fork a child in its own session that ignores SIGTERM — the SDK's stdio
    spawn shape (start_new_session) in its worst cleanup-defying form."""
    pid = os.fork()
    if pid == 0:  # child: detach, ignore TERM, outlive everything
        try:
            os.setsid()
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            while True:
                time.sleep(3600)
        except BaseException:  # noqa: BLE001 - the child must never leak into pytest
            pass
        finally:
            os._exit(0)
    return pid


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_gone(pid: int, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:  # reap the (forked) corpse so a zombie doesn't read as alive
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return True
        except ChildProcessError:
            return True
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def _force_cleanup(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    try:  # reap the corpse so the test leaves no zombie behind
        os.waitpid(pid, 0)
    except (ChildProcessError, OSError):
        pass


@pytest.fixture()
def ledger():
    """Isolated orphan ledger state; also detaches the death supervisor (its
    unregister handshake would spawn a real supervisor process)."""
    from tools import mcp_tool_lifecycle as _lifecycle
    from tools import mcp_tool as _supervisor_host

    saved = (_lifecycle._orphan_stdio_pids, _lifecycle._orphan_stdio_pid_servers,
             _lifecycle._stdio_pids, _lifecycle._stdio_pgids)
    _lifecycle._orphan_stdio_pids = set()
    _lifecycle._orphan_stdio_pid_servers = {}
    _lifecycle._stdio_pids = {}
    _lifecycle._stdio_pgids = {}
    with patch.object(_supervisor_host, "_update_death_supervisor", lambda *_a, **_k: None):
        yield _lifecycle
    (_lifecycle._orphan_stdio_pids, _lifecycle._orphan_stdio_pid_servers,
     _lifecycle._stdio_pids, _lifecycle._stdio_pgids) = saved


def _mk_task(name: str):
    from tools import mcp_tool
    return mcp_tool.MCPServerTask(name)


class TestSupersededChildReap:
    def test_park_boundary_reaps_the_failed_attempts_child(self, ledger):
        """Entering _park must kill a SIGTERM-ignoring orphan attributed to this
        server: the revive the next attempt's entry sweep depends on can be hours
        away, which is how one bridge per reconnect attempt accumulated (#126990)."""
        pid = _spawn_stubborn_child()
        try:
            assert _alive(pid)
            ledger._stdio_pgids[pid] = pid  # spawn-time pgid, as _track_spawned_children records
            ledger._orphan_stdio_pids.add(pid)
            ledger._orphan_stdio_pid_servers[pid] = "leaky"

            task = _mk_task("leaky")
            task._shutdown_event.set()  # _park returns at once after its entry work
            assert asyncio.run(task._park("after initial connection failures")) is True

            assert _wait_gone(pid), "stubborn child survived the pre-park reap"
            assert pid not in ledger._orphan_stdio_pids
            assert pid not in ledger._stdio_pgids
        finally:
            _force_cleanup(pid)

    def test_other_servers_orphans_are_untouched(self, ledger):
        """The reap is scoped: an orphan belonging to a DIFFERENT server must
        survive — its own task owns it."""
        pid = _spawn_stubborn_child()
        try:
            ledger._stdio_pgids[pid] = pid
            ledger._orphan_stdio_pids.add(pid)
            ledger._orphan_stdio_pid_servers[pid] = "someone-else"

            asyncio.run(_mk_task("leaky")._reap_superseded_stdio_children())

            assert _alive(pid), "foreign orphan was wrongly reaped"
            assert pid in ledger._orphan_stdio_pids
        finally:
            _force_cleanup(pid)

    def test_reap_is_a_noop_with_empty_ledger(self, ledger):
        """Healthy path: nothing orphaned -> the reap costs one lock, raises nothing."""
        asyncio.run(_mk_task("leaky")._reap_superseded_stdio_children())
        assert ledger._orphan_stdio_pids == set()

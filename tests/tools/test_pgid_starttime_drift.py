"""PGID start-time guards tolerate same-host start-time drift (#117505; follow-up to #131910).

The #43044 PID-reuse guards introduced by #131910 compared the spawn-time start-time
baseline with a later reading using exact ``==``/``!=``:
``tools.environments.local._leader_is_ours`` (terminal group teardown) and
``tools.mcp_tool_lifecycle._signal_mcp_process`` (stdio MCP orphan reaper).  Same-host
readings drift ~1 s on macOS (``kern.boottime`` adjustment, #117505) and
``gateway.status`` ships ``start_time_fingerprints_match`` (tolerance 200 ticks ≈ 2 s)
for exactly that; a drifted reading therefore made the guard refuse to kill a live,
legitimately-owned process group — timed-out terminal commands survived their kill path
and live MCP stdio servers leaked past the reaper.  Both comparisons now route through
the shared tolerant comparator, and an unreadable current value for a still-live leader
keeps the legacy best-effort kill instead of refusing.
"""

import os
import signal
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX-only process-group teardown")

RECORDED = 178864182760  # spawn-time baseline (start ticks / psutil centiseconds — same scale)


# --- Local terminal teardown ------------------------------------------------

def _fake_proc():
    """A live wrapper whose spawn-time PGID/start-time baselines were captured."""
    return SimpleNamespace(
        pid=12345,
        _hermes_pgid=67890,
        _hermes_pgid_start=RECORDED,
        poll=lambda: 0,
        wait=lambda timeout=None: 0,
        kill=lambda: None,
    )


@pytest.fixture()
def killpg_calls(monkeypatch):
    calls = []

    def fake_getpgid(_pid):
        return 67890

    def fake_killpg(pgid, sig):
        calls.append((pgid, sig))
        if sig == 0:  # group-gone probe: _wait_for_group_exit returns immediately
            raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", fake_getpgid)
    monkeypatch.setattr(os, "killpg", fake_killpg)
    return calls


def test_local_group_kill_survives_start_time_drift(monkeypatch, killpg_calls):
    """~1 s drift between the recorded baseline and the later read must NOT make the
    guard refuse to kill a live, legitimately-owned process group."""
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: RECORDED + 100)
    from tools.environments.local import _kill_process_group_posix

    _kill_process_group_posix(_fake_proc())

    assert (67890, signal.SIGTERM) in killpg_calls


def test_local_guard_still_refuses_recycled_pgid(monkeypatch, killpg_calls):
    """A start time far outside the drift tolerance means the PID/PGID was recycled:
    never signal the stale number (#43044)."""
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: RECORDED + 500000)
    from tools.environments.local import _kill_process_group_posix

    _kill_process_group_posix(_fake_proc())

    assert killpg_calls == []


def test_local_guard_unreadable_start_time_live_leader_is_best_effort(monkeypatch, killpg_calls):
    """An unreadable current start time for a still-live leader keeps the legacy
    best-effort kill; a gone leader still refuses the possibly-recycled number."""
    from tools.environments.local import _kill_process_group_posix, _leader_is_ours

    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: None)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
    _kill_process_group_posix(_fake_proc())
    assert (67890, signal.SIGTERM) in killpg_calls

    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
    assert _leader_is_ours(67890, RECORDED) is False


# --- stdio MCP orphan reaper ------------------------------------------------

def _reset_mcp_ledgers():
    from tools.mcp_tool_lifecycle import (
        _orphan_stdio_pid_servers, _orphan_stdio_pids, _stdio_pgids, _stdio_pids, _stdio_starttimes)
    from tools.mcp_tool import _lock
    with _lock:
        _stdio_pids.clear()
        _orphan_stdio_pids.clear()
        _orphan_stdio_pid_servers.clear()
        _stdio_pgids.clear()
        _stdio_starttimes.clear()


def _stage_orphan(fake_pid: int):
    from tools.mcp_tool_lifecycle import _orphan_stdio_pids, _stdio_pgids, _stdio_starttimes
    from tools.mcp_tool import _lock
    _reset_mcp_ledgers()
    with _lock:
        _orphan_stdio_pids.add(fake_pid)
        _stdio_pgids[fake_pid] = fake_pid
        _stdio_starttimes[fake_pid] = RECORDED


def test_mcp_orphan_reaper_survives_start_time_drift(monkeypatch):
    """A live orphaned MCP stdio server whose start-time reading drifted ~1 s since
    spawn (macOS boottime adjustment) must still be signalled, not skipped as
    recycled — the group otherwise leaks past the reaper."""
    from tools.mcp_tool_lifecycle import _kill_orphaned_mcp_children

    fake_pid = 484848
    _stage_orphan(fake_pid)
    killpg_calls = []
    monkeypatch.setattr("tools.mcp_tool_lifecycle._leader_start_time",
                        lambda pid: RECORDED + 100)
    monkeypatch.setattr("tools.mcp_tool_lifecycle.os.killpg",
                        lambda pgid, sig: killpg_calls.append((pgid, sig)))
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)  # no SIGKILL pass
    monkeypatch.setattr("tools.mcp_tool_lifecycle.time.sleep", lambda *_: None)

    _kill_orphaned_mcp_children()

    assert (fake_pid, signal.SIGTERM) in killpg_calls


def test_mcp_orphan_reaper_unreadable_live_leader_is_best_effort(monkeypatch):
    """An unreadable leader start time on a still-live orphan keeps the legacy
    best-effort signalling rather than skipping the kill."""
    from tools.mcp_tool_lifecycle import _kill_orphaned_mcp_children

    fake_pid = 494949
    _stage_orphan(fake_pid)
    killpg_calls = []
    monkeypatch.setattr("tools.mcp_tool_lifecycle._leader_start_time", lambda pid: None)
    monkeypatch.setattr("tools.mcp_tool_lifecycle.os.killpg",
                        lambda pgid, sig: killpg_calls.append((pgid, sig)))
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
    monkeypatch.setattr("tools.mcp_tool_lifecycle.time.sleep", lambda *_: None)

    _kill_orphaned_mcp_children()

    assert (fake_pid, signal.SIGTERM) in killpg_calls


def test_mcp_orphan_reaper_still_refuses_recycled_pid(monkeypatch):
    """A start time far outside the tolerance is a recycled PID: still skipped (#43044)."""
    from tools.mcp_tool_lifecycle import _kill_orphaned_mcp_children

    fake_pid = 505050
    _stage_orphan(fake_pid)
    killpg_calls = []
    monkeypatch.setattr("tools.mcp_tool_lifecycle._leader_start_time",
                        lambda pid: RECORDED + 500000)
    monkeypatch.setattr("tools.mcp_tool_lifecycle.os.killpg",
                        lambda pgid, sig: killpg_calls.append((pgid, sig)))
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
    monkeypatch.setattr("tools.mcp_tool_lifecycle.time.sleep", lambda *_: None)

    _kill_orphaned_mcp_children()

    assert killpg_calls == []

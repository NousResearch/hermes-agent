"""Regression for the MCP lifecycle task starving the whole process on a dead stdio child.

A stdio server with no ``keepalive_interval`` proves its session by idling one full default
interval with the child still alive (#62212). When the child exits instead of idling, that proof
can never arrive — but the wake that re-checks it re-armed a ZERO timeout, so the lifecycle task
spun at full speed holding the GIL: one core pinned in every process hosting the server, and the
host's event loop stalled (gateway turns and Desktop/TUI ``session.resume`` RPCs timing out for
tens of seconds) for as long as the child stayed dead.

A dead child is a dead transport. The loop must request the reconnect whose respawn the
rapid-drop budget can park, and a LIVE child must still prove the session exactly as before.
"""

import asyncio
import subprocess
import sys

import pytest

pytest.importorskip("mcp")
pytest.importorskip("psutil")

from tools import mcp_tool as _mcp_tool  # noqa: E402  (setattr seam: _core reads it at call time)
from tools.mcp_tool import MCPServerTask  # noqa: E402


def _reaped_pid() -> int:
    """A PID that is provably gone: a child we spawned and reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.mark.asyncio
async def test_unproven_stdio_session_with_dead_child_requests_reconnect(monkeypatch):
    """Dead child + unproven session: ask for a reconnect instead of spinning on a zero timeout."""
    monkeypatch.setattr(_mcp_tool, "_DEFAULT_KEEPALIVE_INTERVAL", 0.05)
    task = MCPServerTask("test")
    task._stdio_child_pids = {_reaped_pid()}
    assert task._stdio_children_dead() is True, "premise: the liveness probe must see the death"

    # Without the fix this never returns: the proof deadline is in the past, the child cannot
    # prove it, and every wake re-arms timeout=0.
    reason = await asyncio.wait_for(task._wait_for_lifecycle_event(), timeout=5)

    assert reason == "reconnect"


@pytest.mark.asyncio
async def test_unproven_stdio_session_with_live_child_is_still_proven(monkeypatch):
    """A live child still proves the session once the full interval elapses — no reconnect."""
    monkeypatch.setattr(_mcp_tool, "_DEFAULT_KEEPALIVE_INTERVAL", 0.05)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        task = MCPServerTask("test")
        task._stdio_child_pids = {child.pid}
        proven = asyncio.Event()
        mark_proven = task._mark_session_proven

        def _record() -> None:
            mark_proven()
            proven.set()

        task._mark_session_proven = _record
        lifecycle = asyncio.create_task(task._wait_for_lifecycle_event())
        try:
            await asyncio.wait_for(proven.wait(), timeout=5)
        finally:
            lifecycle.cancel()
            with pytest.raises(asyncio.CancelledError):
                await lifecycle

        assert task._session_proven is True
        assert not task._reconnect_event.is_set()
    finally:
        child.kill()
        child.wait()

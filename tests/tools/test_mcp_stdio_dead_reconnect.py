"""Regression test for #115483: dead stdio child past the proof deadline.

Premise: in ``_wait_for_lifecycle_event`` the stdio-no-keepalive branch computes
``timeout = max(0.0, proof_at - now)``, which sticks at 0 once the proof deadline passes, and the
expired-proof check no-ops on a dead child then ``continue``s. With the child dead and no events ever
firing, the supervisor spun on zero-timeout ``asyncio.wait`` wakes forever instead of reconnecting.

Fix contract: at expired ``proof_at`` with dead stdio children, return ``"reconnect"`` (failing
stale in-flight calls) instead of ``continue``; a live child still proves the session.
"""
import asyncio

import pytest

from tools.mcp_tool import MCPServerTask
import tools.mcp_tool_server_run as run_mod


async def _drive_one_expired_proof_wake(task, monkeypatch, *, on_first_wake=None) -> tuple[str, int]:
    """Run ``_wait_for_lifecycle_event`` with the proof deadline already expired. The first
    ``asyncio.wait`` returns nothing (a plain timeout wake); a second wake means the loop did NOT exit
    on the first one, so shut it down to terminate. Returns ``(reason, wake_count)``."""
    monkeypatch.setattr("tools.mcp_tool._DEFAULT_KEEPALIVE_INTERVAL", 0.0)
    real_wait = asyncio.wait
    wakes = {"n": 0}

    async def fake_wait(waiters, timeout=None, return_when=None):
        wakes["n"] += 1
        if wakes["n"] == 1:
            if on_first_wake is not None:
                on_first_wake(timeout)
            return set(), set(waiters)
        task._shutdown_event.set()
        return await real_wait(waiters, timeout=1.0, return_when=return_when or asyncio.FIRST_COMPLETED)

    monkeypatch.setattr(run_mod.asyncio, "wait", fake_wait)
    reason = await asyncio.wait_for(task._wait_for_lifecycle_event(), timeout=10)
    return reason, wakes["n"]


def _stdio_task(name: str, *, children_dead: bool) -> MCPServerTask:
    task = MCPServerTask(name)
    task._config = {"command": "true"}  # stdio: no URL, no keepalive_interval
    task._session_proven = False
    task._stdio_children_dead = lambda: children_dead
    return task


@pytest.mark.asyncio
async def test_expired_proof_with_dead_stdio_children_reconnects(monkeypatch):
    task = _stdio_task("test-stdio-dead", children_dead=True)
    failed = []
    task._fail_inflight_calls = lambda reason: failed.append(reason)

    def past_the_deadline_wakes_immediately(timeout):
        assert timeout is not None and timeout <= 1.0, timeout

    reason, wakes = await _drive_one_expired_proof_wake(
        task, monkeypatch, on_first_wake=past_the_deadline_wakes_immediately)
    assert reason == "reconnect", reason
    assert failed == ["reconnect"], failed
    assert wakes == 1, "must reconnect on the first expired-proof wake, not spin"


@pytest.mark.asyncio
async def test_expired_proof_with_live_stdio_children_still_proves(monkeypatch):
    """Guard: a live child at the expired proof deadline marks the session proven and keeps serving."""
    task = _stdio_task("test-stdio-live", children_dead=False)
    reason, _ = await _drive_one_expired_proof_wake(task, monkeypatch)
    assert reason == "shutdown", reason
    assert task._session_proven is True, "live child past proof deadline proves the session"

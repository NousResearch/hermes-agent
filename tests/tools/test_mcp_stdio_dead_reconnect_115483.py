"""Regression test for #115483: dead stdio child past the proof deadline.

Premise: in ``_wait_for_lifecycle_event`` the stdio-no-keepalive branch
computes ``timeout = max(0.0, proof_at - now)``, which sticks at 0 once the
proof deadline passes, and the expired-proof check no-ops on a dead child
(``... and not self._stdio_children_dead()``) then ``continue``s. With the
child dead and no events ever firing, the supervisor spins on zero-timeout
``asyncio.wait`` wakes forever instead of reconnecting.

Fix contract (narrow): at expired ``proof_at`` with dead stdio children,
return ``"reconnect"`` (failing stale in-flight calls) instead of
``continue``.
"""
import asyncio

import pytest

from tools.mcp_tool import MCPServerTask
import tools.mcp_tool_server_run as run_mod


@pytest.mark.asyncio
async def test_expired_proof_with_dead_stdio_children_reconnects(monkeypatch):
    task = MCPServerTask("test-stdio-dead")
    # Stdio config: no URL, no keepalive_interval.
    task._config = {"command": "true"}
    task._session_proven = False
    task.session = object()
    task._stdio_children_dead = lambda: True
    failed = []
    task._fail_inflight_calls = lambda reason: failed.append(reason)
    # Expire the proof deadline immediately.
    monkeypatch.setattr("tools.mcp_tool._DEFAULT_KEEPALIVE_INTERVAL", 0.0)

    real_wait = asyncio.wait
    calls = {"n": 0}

    async def fake_wait(waiters, timeout=None, return_when=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # Past the deadline the supervisor must wake immediately...
            assert timeout is not None and timeout <= 1.0, timeout
            # ...and reconnect instead of spinning: nothing completes here.
            return set(), set(waiters)
        # Pre-fix fallback: the loop spun, so shut it down to terminate.
        task._shutdown_event.set()
        return await real_wait(
            waiters, timeout=1.0,
            return_when=return_when or asyncio.FIRST_COMPLETED,
        )

    monkeypatch.setattr(run_mod.asyncio, "wait", fake_wait)
    reason = await asyncio.wait_for(task._wait_for_lifecycle_event(), timeout=10)
    assert reason == "reconnect", reason
    assert failed == ["reconnect"], failed
    assert calls["n"] == 1, "must reconnect on the first expired-proof wake, not spin"


@pytest.mark.asyncio
async def test_expired_proof_with_live_stdio_children_still_proves(monkeypatch):
    """Guard: the fix must not break the healthy path — a live child at the
    expired proof deadline marks the session proven and keeps serving."""
    task = MCPServerTask("test-stdio-live")
    task._config = {"command": "true"}
    task._session_proven = False
    task.session = object()
    task._stdio_children_dead = lambda: False
    monkeypatch.setattr("tools.mcp_tool._DEFAULT_KEEPALIVE_INTERVAL", 0.0)

    real_wait = asyncio.wait
    calls = {"n": 0}

    async def fake_wait(waiters, timeout=None, return_when=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return set(), set(waiters)
        task._shutdown_event.set()
        return await real_wait(
            waiters, timeout=1.0,
            return_when=return_when or asyncio.FIRST_COMPLETED,
        )

    monkeypatch.setattr(run_mod.asyncio, "wait", fake_wait)
    reason = await asyncio.wait_for(task._wait_for_lifecycle_event(), timeout=10)
    assert reason == "shutdown", reason
    assert task._session_proven is True, "live child past proof deadline proves the session"

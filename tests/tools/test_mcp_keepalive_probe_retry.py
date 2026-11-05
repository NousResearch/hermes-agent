"""A single timed-out keepalive probe must not tear down a healthy session.

A keepalive probe is one round trip over whatever network the host happens to have. A Wi-Fi
roam, sleep/wake or NAT rebind times one out while the server is perfectly healthy. Tearing the
session down on that single probe reported a working remote MCP server as ``failed`` to every
status reader, which is what paged the user with an "MCP server unreachable" toast for a server
that had answered 0.1s later. Two probes, seconds apart, still fail on a real outage.
"""

import asyncio

import pytest

from tools.mcp_tool import MCPServerTask
import tools.mcp_tool_server_run as run_mod


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    monkeypatch.setattr(run_mod, "_KEEPALIVE_PROBE_RETRY_DELAY", 0)


def _task(session=None) -> MCPServerTask:
    task = MCPServerTask("evy")
    task._config = {"url": "https://example.test/mcp"}
    task.session = session if session is not None else object()
    return task


# ── The probe helper itself ──────────────────────────────────────────────────

@pytest.mark.asyncio
class TestProbeKeepaliveWithRetry:
    async def test_one_flaky_probe_is_tolerated(self, monkeypatch):
        task = _task()
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError()

        monkeypatch.setattr(task, "_keepalive_probe", flaky)

        assert await task._probe_keepalive_with_retry() is None
        assert calls["n"] == 2
        assert task._suspect_reason is None

    async def test_two_failed_probes_report_the_failure(self, monkeypatch):
        task = _task()

        async def dead():
            raise TimeoutError()

        monkeypatch.setattr(task, "_keepalive_probe", dead)

        root = await task._probe_keepalive_with_retry()

        assert isinstance(root, TimeoutError)

    async def test_healthy_probe_is_unchanged(self, monkeypatch):
        task = _task()
        calls = {"n": 0}

        async def fine():
            calls["n"] += 1

        monkeypatch.setattr(task, "_keepalive_probe", fine)

        assert await task._probe_keepalive_with_retry() is None
        assert calls["n"] == 1

    async def test_reconnect_request_outranks_the_retry(self, monkeypatch):
        """A reconnect already requested elsewhere wins: no second probe, no failure reported."""
        task = _task()
        task._reconnect_event.set()
        calls = {"n": 0}

        async def dead():
            calls["n"] += 1
            raise TimeoutError()

        monkeypatch.setattr(task, "_keepalive_probe", dead)

        assert await task._probe_keepalive_with_retry() is None
        assert calls["n"] == 1

    async def test_busy_session_counts_as_alive(self, monkeypatch):
        """An RPC that arrives during the retry window proves the session is serving."""
        task = _task()
        calls = {"n": 0}

        async def dead():
            calls["n"] += 1
            raise TimeoutError()

        async def busy_session(_delay, *a, **kw):
            # The retry delay elapses with a tool call in flight.
            task._inflight_tasks.add(asyncio.get_running_loop().create_future())

        monkeypatch.setattr(task, "_keepalive_probe", dead)
        monkeypatch.setattr(run_mod.asyncio, "sleep", busy_session)

        assert await task._probe_keepalive_with_retry() is None
        assert calls["n"] == 1


# ── Through the real lifecycle loop ─────────────────────────────────────────

async def _drive_one_keepalive_cycle(task: MCPServerTask) -> str:
    """Run _wait_for_lifecycle_event through exactly one keepalive cycle, then fire shutdown."""
    real_wait = asyncio.wait
    cycles = {"n": 0}

    async def fake_wait(tasks, timeout=None, return_when=None):
        cycles["n"] += 1
        if cycles["n"] == 1:
            return set(), set(tasks)  # keepalive interval elapsed, nothing else happened
        task._shutdown_event.set()
        return await real_wait(
            tasks, timeout=0.5, return_when=return_when or asyncio.FIRST_COMPLETED
        )

    import tools.mcp_tool as mcp_mod
    orig = mcp_mod.asyncio.wait
    mcp_mod.asyncio.wait = fake_wait
    try:
        return await task._wait_for_lifecycle_event()
    finally:
        mcp_mod.asyncio.wait = orig


@pytest.mark.asyncio
class TestLifecycleLoop:
    async def test_flaky_probe_keeps_the_session(self, monkeypatch):
        task = _task()
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError()

        monkeypatch.setattr(task, "_keepalive_probe", flaky)

        reason = await _drive_one_keepalive_cycle(task)

        assert reason == "shutdown"
        assert task._suspect_reason is None
        assert not task._reconnect_event.is_set()
        assert task._session_proven is True

    async def test_dead_server_still_reconnects(self, monkeypatch):
        task = _task()

        async def dead():
            raise ConnectionError("transport closed")

        monkeypatch.setattr(task, "_keepalive_probe", dead)

        reason = await _drive_one_keepalive_cycle(task)

        assert reason == "reconnect"
        assert "keepalive failed" in (task._suspect_reason or "")

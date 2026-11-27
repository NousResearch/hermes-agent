"""A parked MCP server that keeps failing must not respawn its child on a fixed beat.

Each revival of a never-connected server costs ONE spawn (not the whole initial-connect ladder),
the wait before the next self-probe doubles up to a cap, and routine discovery
(``register_mcp_servers`` on every new session / cron tick) wakes a parked server only once per
park streak. Re-running the ladder on every wake respawned a doomed ``npx`` 4x every 300 s forever
(hermes-official-plugins#401). A ``single_instance`` server parks with no timed self-probe at all.
"""

import asyncio

import pytest

from tests.waiting import async_wait_until
from tools.mcp_tool import MCPServerTask


def _fast_backoff(monkeypatch):
    from tools import mcp_tool

    real_sleep = asyncio.sleep

    async def _no_wait(_delay, *a, **kw):
        await real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _no_wait)


def _failing_task(name, spawns):
    class _Task(MCPServerTask):
        def _is_http(self):
            return False

        def _deregister_tools(self):
            self._registered_tool_names = []

        async def _run_stdio(self, config):
            spawns[-1] += 1
            raise RuntimeError("npm error code EALLOWREMOTE")

    return _Task(name)


def _record_park_waits(monkeypatch, spawns, timeouts):
    real_wait = MCPServerTask._wait_for_reconnect_or_shutdown

    async def _recording_wait(self, timeout=None):
        timeouts.append(timeout)
        spawns.append(0)
        return await real_wait(self, timeout=timeout)

    monkeypatch.setattr(MCPServerTask, "_wait_for_reconnect_or_shutdown", _recording_wait)


def _rediscover(monkeypatch, name, task, config):
    """Run routine discovery against the cached parked task; True when it woke the task."""
    from tools import mcp_tool
    from tools import mcp_tool_discovery as discovery

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setitem(mcp_tool._servers, name, task)
    discovery.register_mcp_servers({name: config})
    woke = task._reconnect_event.is_set()
    mcp_tool._servers.pop(name, None)
    return woke


async def _stop(task, run_task):
    task._shutdown_event.set()
    task._reconnect_event.set()
    await asyncio.wait_for(run_task, timeout=15)


@pytest.mark.no_isolate
def test_single_instance_conflict_parks_quietly_and_never_self_probes(monkeypatch, tmp_path):
    """A single-instance server that never connects parks with no timed self-probe, routine
    discovery does not wake it, and an explicit reconnect still revives it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 1)
    _fast_backoff(monkeypatch)
    spawns, timeouts = [0], []
    _record_park_waits(monkeypatch, spawns, timeouts)
    config = {"command": "gbrain", "args": ["serve"], "single_instance": True}

    async def _scenario():
        task = _failing_task("gbrain", spawns)
        run_task = asyncio.ensure_future(task.run(config))
        await async_wait_until(lambda: len(timeouts) == 1, what="first park")
        assert timeouts == [None], "a resource-conflict park must wait without a timed self-probe"
        assert spawns == [2, 0]  # initial attempt + 1 retry, then parked

        assert not _rediscover(monkeypatch, "gbrain", task, config), \
            "routine discovery woke a quiet-parked single_instance server"

        task._reconnect_event.set()
        await async_wait_until(lambda: len(timeouts) == 2, what="explicit reconnect revival")
        assert spawns[1] == 1
        await _stop(task, run_task)

    asyncio.run(_scenario())


@pytest.mark.no_isolate
def test_parked_initial_failure_spawns_once_per_wake_with_backoff(monkeypatch, tmp_path):
    """A server that never connects runs the initial ladder ONCE; every later revival spawns a
    single child, the self-probe wait doubles up to the cap, and routine discovery wakes the
    parked server only for the first revival of the streak."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 100)
    monkeypatch.setattr(mcp_tool, "_PARKED_PROBE_BACKOFF_MAX", 400)
    _fast_backoff(monkeypatch)
    spawns, timeouts = [0], []
    _record_park_waits(monkeypatch, spawns, timeouts)
    config = {"command": "npx", "args": ["-y", "pkg@1.0.0"]}

    async def _scenario():
        task = _failing_task("npx-srv", spawns)
        run_task = asyncio.ensure_future(task.run(config))
        await async_wait_until(lambda: len(timeouts) == 1, what="first park")

        # First revival of the streak: a new session's discovery nudge is honoured.
        assert _rediscover(monkeypatch, "npx-srv", task, config)
        await async_wait_until(lambda: len(timeouts) == 2, what="discovery revival")
        # The streak has a failed revival now: further discovery passes leave it to the backoff.
        for _ in range(5):
            assert not _rediscover(monkeypatch, "npx-srv", task, config)

        for parks in (3, 4, 5):
            task._reconnect_event.set()
            await async_wait_until(lambda: len(timeouts) == parks, what=f"park {parks}")
        await _stop(task, run_task)

    asyncio.run(_scenario())

    assert spawns[0] == mcp_tool._MAX_INITIAL_CONNECT_RETRIES + 1
    assert spawns[1:5] == [1, 1, 1, 1], spawns
    assert timeouts[:5] == [100, 200, 400, 400, 400], timeouts

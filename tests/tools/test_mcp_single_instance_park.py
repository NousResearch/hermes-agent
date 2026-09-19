"""Single-instance MCP stdio servers must not hammer a timed self-probe when a duplicate
process already holds their single-writer resource (e.g. a PGLite datastore lock).

A stdio server marked ``single_instance: true`` has exactly one live child system-wide. When a
second Hermes context discovers it, its own spawn dies before the MCP handshake because the
resource is already taken; after the initial-connect ladder that is provably unwinnable, so the
server must park QUIETLY (no timed self-probe) instead of re-spawning a doomed child every
``_PARKED_RETRY_INTERVAL``. Explicit reconnect still revives it once the owner exits.
"""

import asyncio

import pytest

from tools.mcp_tool import MCPServerTask


@pytest.mark.no_isolate
def test_single_instance_conflict_parks_quietly_and_never_self_probes(monkeypatch, tmp_path):
    """A single-instance server that dies before the handshake parks without a timed self-probe,
    and is revived only by an explicit reconnect."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 1)
    # Tiny so a regression (a self-probe firing) would be caught fast.
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.02)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "deregistered": 0, "park_timeout": "unset"}

    _real_wait = MCPServerTask._wait_for_reconnect_or_shutdown

    async def _recording_wait(self, timeout=None):
        state["park_timeout"] = timeout
        return await _real_wait(self, timeout=timeout)

    monkeypatch.setattr(MCPServerTask, "_wait_for_reconnect_or_shutdown", _recording_wait)

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["deregistered"] += 1
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                raise RuntimeError("child exited: duplicate holds the datastore lock")

        task = _Task("gbrain")
        run_task = asyncio.ensure_future(
            task.run({"command": "gbrain", "args": ["serve"], "single_instance": True})
        )

        for _ in range(1000):
            await _real_sleep(0)
            if state["deregistered"] >= 1:
                break

        assert state["deregistered"] >= 1, "single-instance server never parked"
        assert state["park_timeout"] is None, (
            "resource-conflict park must wait without a timed self-probe"
        )
        calls_at_park = state["transport_calls"]
        assert calls_at_park == 2, (
            f"expected initial attempt + 1 retry before parking, got {calls_at_park}"
        )

        # A quiet park must not re-spawn: give any stray self-probe ample time to fire.
        for _ in range(200):
            await _real_sleep(0.01)
        assert state["transport_calls"] == calls_at_park, (
            "single-instance server self-probed after a resource conflict — the storm is back"
        )

        # An explicit reconnect (`hermes mcp` refresh) must still revive it.
        task._reconnect_event.set()
        for _ in range(1000):
            await _real_sleep(0)
            if state["transport_calls"] > calls_at_park:
                break
        assert state["transport_calls"] > calls_at_park, (
            "explicit reconnect did not revive the quiet-parked server"
        )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())


def test_park_probe_backoff_doubles_and_caps(monkeypatch):
    """The parked self-probe wait doubles per consecutive still-failed probe and caps, then
    resets once a session proves healthy — so a non-single-instance server also stops hammering."""
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 300)
    monkeypatch.setattr(mcp_tool, "_PARKED_PROBE_BACKOFF_MAX", 1000)

    srv = MCPServerTask("srv")
    assert srv._next_park_probe_timeout() == 300  # first probe at base cadence
    srv._parked_probe_failures = 1
    assert srv._next_park_probe_timeout() == 600
    srv._parked_probe_failures = 2
    assert srv._next_park_probe_timeout() == 1000  # 1200 capped
    srv._parked_probe_failures = 20
    assert srv._next_park_probe_timeout() == 1000

    srv._mark_session_proven()
    assert srv._parked_probe_failures == 0
    assert srv._next_park_probe_timeout() == 300

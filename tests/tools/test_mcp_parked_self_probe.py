"""Tests for the parked-server self-probe revival path (#57129).

Parking deregisters a server's tools, so no tool call can reach the
circuit-breaker half-open probe or ``_signal_reconnect`` — the only
things that set ``_reconnect_event``. The parked wait must therefore be
timed: the run task wakes on ``_PARKED_RETRY_INTERVAL`` and attempts one
revival probe on its own.
"""

import asyncio

import pytest


@pytest.mark.no_isolate
def test_parked_server_self_probes_and_revives(monkeypatch, tmp_path):
    """A parked server must revive on its own once the backend recovers,
    without any explicit _reconnect_event.set()."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 1)
    # Keep the self-probe cadence tiny so the test is fast.
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.05)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {
        "transport_calls": 0,
        "deregistered": 0,
        "backend_up": False,
        "revived_registration": 0,
    }

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["deregistered"] += 1
                self._registered_tool_names = []

            def _register_discovered_tools_if_needed(self):
                if self._ready.is_set() and not self._registered_tool_names:
                    state["revived_registration"] += 1
                    self._registered_tool_names = ["srv__tool"]

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                if state["transport_calls"] == 1:
                    # First connect succeeds (sets _ready), then dies.
                    self.session = object()
                    self._ready.set()
                    self.session = None
                    raise RuntimeError("backend outage begins")
                if not state["backend_up"]:
                    raise RuntimeError("backend still down")
                # Backend recovered: establish a session and park in the
                # lifecycle wait like the real transport does.
                self.session = object()
                self._register_discovered_tools_if_needed()
                await self._wait_for_lifecycle_event()

        task = _Task("srv")
        task._registered_tool_names = ["srv__tool"]

        run_task = asyncio.ensure_future(task.run({"command": "x"}))

        # Let it exhaust the budget (1 retry) and park.
        for _ in range(2000):
            await _real_sleep(0)
            if state["deregistered"] >= 1:
                break
        assert state["deregistered"] >= 1, "server never parked"
        assert not run_task.done(), "run task exited instead of parking"

        # The backend comes back. NOTHING sets _reconnect_event — revival
        # must come from the timed self-probe alone.
        state["backend_up"] = True
        for _ in range(200):
            await _real_sleep(0.01)
            if task.session is not None:
                break

        assert task.session is not None, (
            "parked server never self-probed back to life "
            f"(transport_calls={state['transport_calls']})"
        )
        assert state["revived_registration"] >= 1, (
            "revived server did not re-register its tools"
        )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

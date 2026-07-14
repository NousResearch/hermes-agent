"""Tests for MCP reconnect retry counter reset (#57604).

Verifies that the reconnect retry counter resets after each successful
reconnection, so transient blips do not accumulate toward permanent parking.
"""

import asyncio

import pytest


@pytest.mark.no_isolate
def test_reconnect_counter_resets_after_successful_session(monkeypatch, tmp_path):
    """Transient disconnections must not accumulate toward permanent parking.

    Before the fix, ``retries`` was a local variable in ``run()`` that only
    reset on clean transport return (line 2367) or park-wake (line 2468).
    Each exception from ``_run_stdio`` incremented it without reset, so 5
    transient blips over a long-uptime gateway would permanently park the
    server.

    After the fix, ``_reconnect_retries`` is an instance variable that resets
    to 0 whenever a session is successfully established (``_reset_server_error``
    call sites in ``_run_stdio`` / ``_run_http``).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    # Shrink budget so the test can exhaust it quickly if the counter
    # does NOT reset (the bug scenario).
    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 3)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {
        "transport_calls": 0,
        "parked": False,
        "max_retries_seen": 0,
    }

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] = True
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                call = state["transport_calls"]

                if call == 1:
                    # First connect: succeed (sets _ready), then fail.
                    self.session = object()
                    self._ready.set()
                    self.session = None
                    raise RuntimeError("blip 1")

                # Subsequent calls: succeed (session established, which
                # triggers _reset_server_error and should reset retries),
                # then immediately fail again — simulating a new transient
                # blip.  If retries accumulate, call 4 would exceed the
                # budget of 3 and park.  If retries reset correctly,
                # this loop can continue indefinitely.
                if call <= 8:
                    self.session = object()
                    # _run_stdio calls _reset_server_error and sets
                    # _reconnect_retries = 0 after session establishment.
                    # We simulate that by calling the real method.
                    mcp_tool._reset_server_error(self.name)
                    self._reconnect_retries = 0
                    self.session = None
                    raise RuntimeError(f"blip {call}")

                # If we reach here without parking, the fix works.
                self.session = object()
                mcp_tool._reset_server_error(self.name)
                self._reconnect_retries = 0
                await self._wait_for_lifecycle_event()
                return

        task = _Task("srv")
        task._registered_tool_names = ["srv__tool"]

        run_task = asyncio.ensure_future(task.run({"command": "x"}))

        # Let the scenario run.
        for _ in range(2000):
            await _real_sleep(0)
            if state["transport_calls"] >= 8 or state["parked"]:
                break

        # The fix: the server should NOT have parked despite 8 transient
        # disconnections, because each successful reconnection reset the
        # retry counter.
        assert not state["parked"], (
            f"server parked after {state['transport_calls']} transport calls "
            f"— retry counter accumulated instead of resetting"
        )
        assert state["transport_calls"] >= 8, (
            f"only {state['transport_calls']} transport calls reached "
            f"(expected >= 8)"
        )

        # Verify the counter is an instance variable, not a local.
        assert hasattr(task, "_reconnect_retries"), (
            "_reconnect_retries should be an instance variable"
        )

        # Clean shutdown.
        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=2)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())


@pytest.mark.no_isolate
def test_reconnect_counter_still_parks_on_consecutive_failures(monkeypatch, tmp_path):
    """The server must still park when failures are genuinely consecutive
    (no successful reconnection in between).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 2)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "parked": False}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] = True
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                call = state["transport_calls"]

                if call == 1:
                    self.session = object()
                    self._ready.set()
                    self.session = None
                    raise RuntimeError("first failure")

                # All subsequent calls fail WITHOUT establishing a session
                # (no _reset_server_error, no retry reset). This simulates
                # genuinely consecutive failures.
                raise RuntimeError(f"failure {call}")

        task = _Task("srv")
        task._registered_tool_names = ["srv__tool"]

        run_task = asyncio.ensure_future(task.run({"command": "x"}))

        for _ in range(500):
            await _real_sleep(0)
            if state["parked"] or run_task.done():
                break

        assert state["parked"], (
            "server should park on consecutive failures without successful reconnect"
        )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=2)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

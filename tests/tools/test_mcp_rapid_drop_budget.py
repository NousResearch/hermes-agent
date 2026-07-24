"""Tests for the MCP rapid-drop reconnect budget (#62212).

A flapping transport that completes the MCP handshake but drops moments
later used to reset ``_reconnect_retries`` on every (re)connect — the park
budget was never consumed, so the run loop respawned the transport forever
(observed: 6212 spawns in 63 hours). A session must now PROVE health
(survive a full keepalive interval, or serve a successful tool call) via
``_mark_session_proven()`` before the budget is cleared.

Also covers the fatal-signal guard in ``_reconnect_or_reraise_group``:
KeyboardInterrupt / SystemExit leaves must re-raise, never be converted
into an immediate reconnect.
"""

import asyncio

import pytest

from tools.mcp_tool import MCPServerTask


def _group(*excs) -> BaseExceptionGroup:
    return BaseExceptionGroup("transport drop", list(excs))


# ── Fatal signals must never be masked as a reconnect ────────────────────────

class TestReconnectGroupFatalSignals:
    def test_keyboard_interrupt_leaf_reraises(self):
        task = MCPServerTask("t")
        task._ready.set()
        with pytest.raises(BaseExceptionGroup):
            task._reconnect_or_reraise_group(
                _group(ConnectionError("drop"), KeyboardInterrupt())
            )

    def test_system_exit_leaf_reraises(self):
        task = MCPServerTask("t")
        task._ready.set()
        with pytest.raises(BaseExceptionGroup):
            task._reconnect_or_reraise_group(_group(SystemExit(1)))

    def test_nested_keyboard_interrupt_reraises(self):
        task = MCPServerTask("t")
        task._ready.set()
        nested = _group(_group(KeyboardInterrupt()))
        with pytest.raises(BaseExceptionGroup):
            task._reconnect_or_reraise_group(nested)

    def test_plain_transient_drop_still_reconnects(self):
        task = MCPServerTask("t")
        task._ready.set()
        assert task._reconnect_or_reraise_group(
            _group(ConnectionError("sse stream dropped"))
        ) == "reconnect"


# ── _mark_session_proven semantics ───────────────────────────────────────────

class TestMarkSessionProven:
    def test_proven_clears_budget(self):
        task = MCPServerTask("t")
        task._reconnect_retries = 4
        assert task._session_proven is False
        task._mark_session_proven()
        assert task._session_proven is True
        assert task._reconnect_retries == 0

    def test_unproven_after_fresh_connect(self):
        # __init__ starts unproven; transports reset the flag on handshake.
        task = MCPServerTask("t")
        assert task._session_proven is False


# ── Integration: flapping transport must reach the park ─────────────────────

@pytest.mark.no_isolate
def test_flapping_transport_reaches_park_within_budget(monkeypatch, tmp_path):
    """A transport that handshakes fine but immediately asks to reconnect
    (post-ready TaskGroup drop / keepalive failure) must park within a
    bounded number of spawns instead of respawning forever (#62212)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 3)

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
                # Handshake succeeds every time (the flapping pattern):
                # session established, _ready set, budget NOT cleared
                # because the session never proves itself…
                self.session = object()
                self._ready.set()
                self._session_proven = False
                # …then the transport immediately asks for a rebuild
                # (clean return — what a keepalive failure or a post-ready
                # TaskGroup drop produces).
                self.session = None
                return "reconnect"

        task = _Task("flappy")
        task._registered_tool_names = ["flappy__tool"]

        run_task = asyncio.ensure_future(task.run({"command": "x"}))

        for _ in range(2000):
            await _real_sleep(0)
            if state["parked"]:
                break

        assert state["parked"], (
            f"flapping transport never parked after "
            f"{state['transport_calls']} spawns — rapid-drop budget not charged"
        )
        # Budget must bound the spawn count: 1 initial + _MAX_RECONNECT_RETRIES
        # charged reconnects (+1 for the park-triggering attempt).
        assert state["transport_calls"] <= 3 + 2, (
            f"park took {state['transport_calls']} spawns — budget leaking"
        )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())


@pytest.mark.no_isolate
def test_proven_session_does_not_burn_budget(monkeypatch, tmp_path):
    """A session that proves healthy (keepalive success / tool call) before
    each drop must NOT accumulate toward the park — long-lived healthy
    servers with occasional blips keep reconnecting forever (#57604)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 3)

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
                if state["transport_calls"] > 10:
                    # Stop the scenario: hold the session open.
                    self.session = object()
                    self._ready.set()
                    await self._shutdown_event.wait()
                    return "shutdown"
                self.session = object()
                self._ready.set()
                self._session_proven = False
                # The session proves healthy (keepalive interval survived /
                # successful tool call) before the drop.
                self._mark_session_proven()
                self.session = None
                return "reconnect"

        task = _Task("healthy")
        task._registered_tool_names = ["healthy__tool"]

        run_task = asyncio.ensure_future(task.run({"command": "x"}))

        for _ in range(2000):
            await _real_sleep(0)
            if state["transport_calls"] > 10 or state["parked"]:
                break

        assert not state["parked"], (
            "healthy-but-blipping server parked — proven sessions must "
            "clear the budget"
        )
        assert state["transport_calls"] > 10

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

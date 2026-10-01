"""A parked server's timed self-probe must honour ``enabled: false`` / a deleted config
entry: the probe rebuilds the transport from the config captured at start, so without the
gate a disabled server re-runs its OAuth setup every ``_PARKED_RETRY_INTERVAL`` for the
life of the process — the two-warning flood behind the errors.log cap being hit."""

import asyncio

import pytest


@pytest.mark.no_isolate
def test_parked_self_probe_skips_disabled_or_deleted_entry(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools import mcp_tool_config as _config
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 1)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.05)

    configured = {"srv": {"command": "x"}}

    def _load():
        return {name: dict(cfg) for name, cfg in configured.items()}

    monkeypatch.setattr(_config, "_load_mcp_config", _load)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "deregistered": 0}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["deregistered"] += 1
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                if state["transport_calls"] == 1:
                    self.session = object()
                    self._ready.set()
                    self._ever_connected = True
                    self.session = None
                    raise RuntimeError("backend outage begins")
                raise RuntimeError("backend still down")

        task = _Task("srv")
        task._registered_tool_names = ["srv__tool"]

        run_task = asyncio.ensure_future(task.run({"command": "x"}))

        for _ in range(2000):
            await _real_sleep(0)
            if state["deregistered"] >= 1:
                break
        assert state["deregistered"] >= 1, "server never parked"
        assert not run_task.done(), "run task exited instead of parking"

        baseline = state["transport_calls"]
        for _ in range(100):
            await _real_sleep(0.01)
        assert state["transport_calls"] > baseline, (
            "self-probe never attempted while enabled"
        )

        # The user disables the entry. Across several probe intervals the parked task
        # must not rebuild the transport again.
        configured["srv"]["enabled"] = False
        disabled_at = state["transport_calls"]
        for _ in range(100):
            await _real_sleep(0.01)
        assert state["transport_calls"] == disabled_at, (
            "parked self-probe re-attempted a disabled server "
            f"(transport_calls={state['transport_calls']})"
        )
        assert not run_task.done(), "disabled gate must park, not exit the run task"

        # Same expectation for a deleted entry.
        del configured["srv"]
        deleted_at = state["transport_calls"]
        for _ in range(100):
            await _real_sleep(0.01)
        assert state["transport_calls"] == deleted_at, (
            "parked self-probe re-attempted a deleted server "
            f"(transport_calls={state['transport_calls']})"
        )

        # An explicit reconnect (manual refresh, `hermes mcp login`) still revives
        # regardless of the config gate.
        task._reconnect_event.set()
        revived_at = state["transport_calls"]
        for _ in range(200):
            await _real_sleep(0.01)
            if state["transport_calls"] > revived_at:
                break
        assert state["transport_calls"] > revived_at, (
            "explicit reconnect was gated by config"
        )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

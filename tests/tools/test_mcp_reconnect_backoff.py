"""Tests for the per-server ``reconnect_backoff`` config option in
``tools/mcp_tool.py``.

Hosted MCP endpoints can fail in multi-minute bursts (e.g. 503 storms from
a shared SaaS backend). With the default 1s ladder base every retry lands
inside the burst, the ladder exhausts in seconds, and the server parks
until the next self-probe. ``reconnect_backoff`` raises the ladder base per
server so the same number of attempts spans the burst.
"""

import asyncio
import logging
import math

import pytest

from tools.mcp_tool import (
    _DEFAULT_RECONNECT_BACKOFF,
    _MAX_BACKOFF_SECONDS,
    _resolve_reconnect_backoff,
)


class TestResolveReconnectBackoff:
    def test_default_when_absent(self):
        assert _resolve_reconnect_backoff("srv", {}) == _DEFAULT_RECONNECT_BACKOFF

    def test_configured_value(self):
        assert _resolve_reconnect_backoff("srv", {"reconnect_backoff": 45}) == 45.0

    def test_float_value(self):
        assert _resolve_reconnect_backoff("srv", {"reconnect_backoff": 2.5}) == 2.5

    def test_string_number_accepted(self):
        # YAML normally types this, but a quoted value must still work.
        assert _resolve_reconnect_backoff("srv", {"reconnect_backoff": "30"}) == 30.0

    def test_clamped_to_max_backoff(self):
        assert _resolve_reconnect_backoff(
            "srv", {"reconnect_backoff": 9999}
        ) == float(_MAX_BACKOFF_SECONDS)

    def test_clamped_to_floor(self):
        assert _resolve_reconnect_backoff(
            "srv", {"reconnect_backoff": 0}
        ) == _DEFAULT_RECONNECT_BACKOFF
        assert _resolve_reconnect_backoff(
            "srv", {"reconnect_backoff": -5}
        ) == _DEFAULT_RECONNECT_BACKOFF

    def test_invalid_value_falls_back_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _resolve_reconnect_backoff(
                "srv", {"reconnect_backoff": "fast"}
            )
        assert result == _DEFAULT_RECONNECT_BACKOFF
        assert "invalid reconnect_backoff" in caplog.text
        assert "srv" in caplog.text

    def test_none_falls_back_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _resolve_reconnect_backoff(
                "srv", {"reconnect_backoff": None}
            )
        assert result == _DEFAULT_RECONNECT_BACKOFF
        assert "invalid reconnect_backoff" in caplog.text

    @pytest.mark.parametrize("raw", [float("nan"), float("inf"), float("-inf"), "nan", "inf"])
    def test_non_finite_falls_back_with_warning(self, raw, caplog):
        # ``float()`` accepts these, and without an explicit guard nan/inf only land on a
        # sane value by accident of IEEE comparison rules inside min()/max().
        with caplog.at_level(logging.WARNING):
            result = _resolve_reconnect_backoff("srv", {"reconnect_backoff": raw})
        assert result == _DEFAULT_RECONNECT_BACKOFF
        assert math.isfinite(result)
        assert "invalid reconnect_backoff" in caplog.text


# ── Wiring: a configured value must reach the ladder's actual sleeps ─────────

@pytest.mark.no_isolate
def test_configured_backoff_reaches_ladder_delays(monkeypatch, tmp_path):
    """``reconnect_backoff: 30`` must produce a ~30s first reconnect wait and a
    ~60s second one (doubling, capped), not the default 1s/2s ladder. Closes the
    seam between ``_resolve_reconnect_backoff`` and ``run()``'s retry budget."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool, mcp_tool_server_run
    from tools.mcp_tool import MCPServerTask

    # Park after 2 reconnect attempts so the scenario terminates.
    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 2)
    # Deterministic delays: strip the +/-20% jitter.
    monkeypatch.setattr(mcp_tool_server_run, "_jittered", lambda s: s)

    _real_sleep = asyncio.sleep
    state = {"transport_calls": 0, "parked": False, "sleeps": []}

    async def _recording_sleep(delay, *a, **kw):
        # The harness's own 0s yields are noise; only ladder waits are >= 1s.
        if delay >= 1.0:
            state["sleeps"].append(delay)
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _recording_sleep)

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] = True
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                if state["transport_calls"] == 1:
                    # First connect succeeds so later failures take the RECONNECT
                    # ladder (gated on _ever_connected), then drops.
                    self.session = object()
                    self._ready.set()
                    self._ever_connected = True
                    self.session = None
                raise RuntimeError(f"drop {state['transport_calls']}")

        task = _Task("srv")
        task._registered_tool_names = ["srv__tool"]

        run_task = asyncio.ensure_future(task.run({"command": "x", "reconnect_backoff": 30}))
        for _ in range(2000):
            await _real_sleep(0)
            if state["parked"] or run_task.done():
                break

        assert state["parked"], "scenario never reached the park"
        assert state["sleeps"][:2] == [30.0, 60.0], (
            f"ladder slept {state['sleeps']} — configured reconnect_backoff did not reach the budget"
        )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

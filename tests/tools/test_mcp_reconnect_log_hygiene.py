"""Tests for MCP reconnect log hygiene and backoff jitter (#65673, #66092).

The retry/park machinery used to emit one WARNING per retry attempt — a
flapping server produced thousands of identical lines (#62212: 6212 spawns
in 63h). Now:

- per-attempt retry logs are DEBUG;
- state transitions carry exactly one WARNING each
  (connected→degraded, degraded→parked, parked→revived);
- backoff sleeps get ±20% jitter so herds of servers don't retry in
  lockstep.
"""

import asyncio
import logging

import pytest

from tools.mcp_tool import MCPServerTask, _jittered


# ── Jitter ───────────────────────────────────────────────────────────────────

class TestJitter:
    def test_jitter_within_20_percent(self):
        for _ in range(200):
            v = _jittered(10.0)
            assert 8.0 <= v <= 12.0

    def test_jitter_zero_is_zero(self):
        assert _jittered(0.0) == 0.0

    def test_jitter_never_negative(self):
        assert _jittered(0.001) >= 0.0

    def test_jitter_varies(self):
        values = {_jittered(10.0) for _ in range(50)}
        assert len(values) > 1, "jitter produced constant values"


# ── Log levels: retry chatter DEBUG, transitions WARNING ─────────────────────

@pytest.mark.no_isolate
def test_retry_attempts_log_debug_transitions_warn(monkeypatch, tmp_path, caplog):
    """Consecutive transient failures: each retry logs at DEBUG, and the
    degraded→parked transition logs exactly one WARNING."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

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
                if state["transport_calls"] == 1:
                    self.session = object()
                    self._ready.set()
                    self.session = None
                raise ConnectionError("backend down")

        task = _Task("noisy")
        task._registered_tool_names = ["noisy__tool"]

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            for _ in range(1000):
                await _real_sleep(0)
                if state["parked"]:
                    break

        assert state["parked"]

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    retry_records = [
        r for r in caplog.records if "connection lost (attempt" in r.getMessage()
    ]
    assert retry_records, "no per-attempt retry logs at all"
    assert all(r.levelno == logging.DEBUG for r in retry_records), (
        "per-attempt retry logs must be DEBUG, got: "
        + str({r.levelname for r in retry_records})
    )

    park_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "parking" in r.getMessage()
    ]
    assert len(park_warnings) == 1, (
        f"expected exactly 1 degraded→parked WARNING, got {len(park_warnings)}"
    )
    assert "degraded → parked" in park_warnings[0].getMessage()


@pytest.mark.no_isolate
def test_keepalive_failure_warns_connected_to_degraded(monkeypatch, tmp_path, caplog):
    """The connected→degraded transition (keepalive failure) is a WARNING."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    class _Task(MCPServerTask):
        async def _keepalive_probe(self):
            raise ConnectionError("session expired")

    task = _Task("kap")
    task._config = {"keepalive_interval": 0.01}
    task.session = object()

    monkeypatch.setattr(mcp_tool, "_MIN_KEEPALIVE_INTERVAL", 0.01)

    async def _scenario():
        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            reason = await task._wait_for_lifecycle_event()
        assert reason == "reconnect"

    asyncio.run(_scenario())

    degraded = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "connected → degraded" in r.getMessage()
    ]
    assert len(degraded) == 1


@pytest.mark.no_isolate
def test_parked_to_revived_warns_once_on_proven_health(monkeypatch, tmp_path, caplog):
    """After a park, the first PROVEN-healthy session logs one
    parked→connected revival WARNING (via _mark_session_proven)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    task = MCPServerTask("reviver")
    task._was_parked = True
    task._reconnect_retries = 5

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        task._mark_session_proven()
        # Second proof must not re-log.
        task._mark_session_proven()

    revived = [
        r for r in caplog.records
        if "revived" in r.getMessage() and "parked → connected" in r.getMessage()
    ]
    assert len(revived) == 1
    assert task._reconnect_retries == 0
    assert task._was_parked is False


@pytest.mark.no_isolate
def test_initial_retry_attempts_log_debug(monkeypatch, tmp_path, caplog):
    """Initial-connect per-attempt retries are DEBUG; only the final park
    (connecting→parked) is a WARNING."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"parked": False}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] = True
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                raise ConnectionError("dns blip")

        task = _Task("startup")

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            for _ in range(1000):
                await _real_sleep(0)
                if state["parked"]:
                    break

        assert state["parked"]

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    attempt_records = [
        r for r in caplog.records
        if "initial connection failed (attempt" in r.getMessage()
    ]
    assert attempt_records
    assert all(r.levelno == logging.DEBUG for r in attempt_records)

    park_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "connecting → parked" in r.getMessage()
    ]
    assert len(park_warnings) == 1

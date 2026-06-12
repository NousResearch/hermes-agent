"""Tests for the MCPServerTask reconnect signal.

When the OAuth layer cannot recover in-place (e.g., external refresh of a
single-use refresh_token made the SDK's in-memory refresh fail), the tool
handler signals MCPServerTask to tear down the current MCP session and
reconnect with fresh credentials. This file exercises the signal plumbing
in isolation from the full stdio/http transport machinery.
"""
import asyncio

import pytest


@pytest.mark.asyncio
async def test_reconnect_event_attribute_exists():
    """MCPServerTask has a _reconnect_event alongside _shutdown_event."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")
    assert hasattr(task, "_reconnect_event")
    assert isinstance(task._reconnect_event, asyncio.Event)
    assert not task._reconnect_event.is_set()


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_returns_reconnect():
    """When _reconnect_event fires, helper returns 'reconnect' and clears it."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._reconnect_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "reconnect"
    # Should have cleared so the next cycle starts fresh
    assert not task._reconnect_event.is_set()


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_returns_shutdown():
    """When _shutdown_event fires, helper returns 'shutdown'."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._shutdown_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "shutdown"


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_shutdown_wins_when_both_set():
    """If both events are set simultaneously, shutdown takes precedence."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._shutdown_event.set()
    task._reconnect_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "shutdown"


def test_sse_reconnect_interval_defaults_and_overrides():
    """SSE defaults to a 30s reconnect floor, but remains configurable."""
    from tools.mcp_tool import MCPServerTask, _DEFAULT_SSE_RECONNECT_INTERVAL

    task = MCPServerTask("test")
    assert task._resolve_reconnect_interval({
        "url": "https://example.com/mcp/sse",
        "transport": "sse",
    }) == _DEFAULT_SSE_RECONNECT_INTERVAL
    assert task._resolve_reconnect_interval({
        "url": "https://example.com/mcp/sse",
        "transport": "sse",
        "reconnect_interval": 45,
    }) == 45.0
    assert task._resolve_reconnect_interval({
        "url": "https://example.com/mcp",
    }) == 0.0


@pytest.mark.asyncio
async def test_sse_keepalive_reconnect_waits_before_reentry(monkeypatch):
    """Keepalive-triggered SSE reconnects wait before recreating the stream."""
    from tools.mcp_tool import MCPServerTask, _DEFAULT_SSE_RECONNECT_INTERVAL

    sleeps = []
    calls = 0

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def fake_run_http(self, config):
        nonlocal calls
        calls += 1
        self._ready.set()
        if calls == 1:
            self._last_reconnect_reason = "keepalive"
            return
        self._shutdown_event.set()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(MCPServerTask, "_run_http", fake_run_http)

    task = MCPServerTask("sse-test")
    await task.run({
        "url": "https://example.com/mcp/sse",
        "transport": "sse",
    })

    assert calls == 2
    assert sleeps == [_DEFAULT_SSE_RECONNECT_INTERVAL]


@pytest.mark.asyncio
async def test_sse_signal_reconnect_stays_immediate(monkeypatch):
    """Auth/manual reconnect signals must not inherit the keepalive delay."""
    from tools.mcp_tool import MCPServerTask

    calls = 0

    async def fail_sleep(delay):  # pragma: no cover - failure path assertion
        raise AssertionError(f"unexpected reconnect sleep: {delay}")

    async def fake_run_http(self, config):
        nonlocal calls
        calls += 1
        self._ready.set()
        if calls == 1:
            self._last_reconnect_reason = "signal"
            return
        self._shutdown_event.set()

    monkeypatch.setattr(asyncio, "sleep", fail_sleep)
    monkeypatch.setattr(MCPServerTask, "_run_http", fake_run_http)

    task = MCPServerTask("sse-test")
    await task.run({
        "url": "https://example.com/mcp/sse",
        "transport": "sse",
    })

    assert calls == 2


@pytest.mark.asyncio
async def test_sse_connection_loss_uses_minimum_reconnect_interval(monkeypatch):
    """Unexpected SSE drops should not retry with 1s/2s storm delays."""
    from tools.mcp_tool import MCPServerTask, _DEFAULT_SSE_RECONNECT_INTERVAL

    sleeps = []
    calls = 0

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def fake_run_http(self, config):
        nonlocal calls
        calls += 1
        self._ready.set()
        if calls == 1:
            raise ConnectionError("connection dropped")
        self._shutdown_event.set()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(MCPServerTask, "_run_http", fake_run_http)

    task = MCPServerTask("sse-test")
    await task.run({
        "url": "https://example.com/mcp/sse",
        "transport": "sse",
    })

    assert calls == 2
    assert sleeps == [_DEFAULT_SSE_RECONNECT_INTERVAL]


@pytest.mark.asyncio
async def test_sse_initial_failure_uses_minimum_reconnect_interval(monkeypatch):
    """SSE startup retries should not hammer a flaky remote endpoint."""
    from tools.mcp_tool import MCPServerTask, _DEFAULT_SSE_RECONNECT_INTERVAL

    sleeps = []
    calls = 0

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def fake_run_http(self, config):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("startup connection dropped")
        self._ready.set()
        self._shutdown_event.set()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(MCPServerTask, "_run_http", fake_run_http)

    task = MCPServerTask("sse-test")
    await task.run({
        "url": "https://example.com/mcp/sse",
        "transport": "sse",
    })

    assert calls == 2
    assert sleeps == [_DEFAULT_SSE_RECONNECT_INTERVAL]

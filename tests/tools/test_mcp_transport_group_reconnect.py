"""Regression tests for issue #66092.

Streamable-HTTP / SSE MCP transports run their stream pump inside an anyio
TaskGroup. A transient stream drop (idle timeout, brief backend blip) surfaces
as a ``BaseExceptionGroup`` escaping the transport context manager. Before the
fix that group reached ``run()``'s error path, which applied exponential
backoff and eventually parked the server for 300s and deregistered its tools —
a multi-minute tool outage for a sub-second glitch.

``_run_http`` now converts such a transport TaskGroup failure into a clean
``"reconnect"`` (immediate rebuild, no backoff/park), while still propagating
real shutdown/cancellation and genuine connect/handshake failures.
"""

from __future__ import annotations

import asyncio

import pytest

from tools.mcp_tool import MCPServerTask


def _group(*excs) -> BaseExceptionGroup:
    return BaseExceptionGroup("transport drop", list(excs))


# ── Unit coverage for the decision helper ────────────────────────────────────

class TestReconnectOrReraiseGroup:
    def test_live_session_transient_drop_reconnects(self):
        task = MCPServerTask("t")
        task._ready.set()  # a live session was established this attempt
        assert task._reconnect_or_reraise_group(
            _group(ConnectionError("sse stream dropped"))
        ) == "reconnect"

    def test_shutdown_in_progress_reraises(self):
        task = MCPServerTask("t")
        task._ready.set()
        task._shutdown_event.set()
        with pytest.raises(BaseExceptionGroup):
            task._reconnect_or_reraise_group(_group(ConnectionError("x")))

    def test_group_carrying_cancellation_reraises(self):
        task = MCPServerTask("t")
        task._ready.set()
        with pytest.raises(BaseException) as ei:
            task._reconnect_or_reraise_group(_group(asyncio.CancelledError()))
        # The cancellation must not be masked as a reconnect.
        assert ei.value.split(asyncio.CancelledError)[0] is not None

    def test_no_live_session_reraises_for_backoff(self):
        task = MCPServerTask("t")
        # _ready NOT set: a connect/handshake failure, not a transient drop —
        # let run()'s backoff handle it instead of hot-looping.
        with pytest.raises(BaseExceptionGroup):
            task._reconnect_or_reraise_group(_group(ConnectionError("handshake")))


# ── Integration coverage: real _run_http new-HTTP branch ─────────────────────

class _RaisingTransportCM:
    """Fake streamable-HTTP transport whose __aexit__ raises a group, mimicking
    the SDK's anyio TaskGroup tearing down on a stream drop."""

    def __init__(self, exc):
        self._exc = exc

    async def __aenter__(self):
        return (object(), object(), lambda: "session-id")

    async def __aexit__(self, *exc_info):
        raise self._exc


class _FakeSession:
    async def initialize(self):
        return object()


class _FakeSessionCM:
    async def __aenter__(self):
        return _FakeSession()

    async def __aexit__(self, *exc_info):
        return False


def _make_http_task(monkeypatch, transport_exc):
    task = MCPServerTask("http-drop")

    monkeypatch.setattr("tools.mcp_tool._MCP_HTTP_AVAILABLE", True, raising=False)
    monkeypatch.setattr("tools.mcp_tool._MCP_NEW_HTTP", True, raising=False)
    monkeypatch.setattr(
        "tools.mcp_tool.streamable_http_client",
        lambda url, http_client=None: _RaisingTransportCM(transport_exc),
        raising=False,
    )
    monkeypatch.setattr(
        "tools.mcp_tool.ClientSession",
        lambda *a, **k: _FakeSessionCM(),
        raising=False,
    )

    async def _no_discover(self):
        return None

    async def _reconnect_lifecycle(self):
        return "reconnect"

    monkeypatch.setattr(MCPServerTask, "_discover_tools", _no_discover)
    monkeypatch.setattr(MCPServerTask, "_wait_for_lifecycle_event", _reconnect_lifecycle)
    return task


def test_run_http_transport_group_returns_reconnect(monkeypatch):
    task = _make_http_task(monkeypatch, _group(ConnectionError("stream dropped")))
    reason = asyncio.run(task._run_http({"url": "http://127.0.0.1:9/mcp"}))
    assert reason == "reconnect"
    # We had a live session; readiness stays set for run() to clear on re-entry.
    assert task._ready.is_set()


def test_run_http_transport_group_reraises_on_shutdown(monkeypatch):
    task = _make_http_task(monkeypatch, _group(ConnectionError("stream dropped")))
    task._shutdown_event.set()
    with pytest.raises(BaseExceptionGroup):
        asyncio.run(task._run_http({"url": "http://127.0.0.1:9/mcp"}))

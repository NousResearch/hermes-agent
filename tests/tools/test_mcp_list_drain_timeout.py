"""Regression tests for the unbounded ``tools/list`` drain wedge.

An SSE MCP server behind a gateway answered ``initialize`` on a
keepalive-triggered reconnect and then never delivered a response to
``tools/list``. ``_discover_tools`` awaited that response forever while
holding the server's ``_rpc_lock``, and because discovery runs *after*
``self.session = session`` but *before* ``_ready.set()`` and
``_wait_for_lifecycle_event()``, the wedge was terminal: no keepalive
watchdog was running yet, the dead-session handler path never fired
(``session`` was non-None), and the circuit breaker's half-open probe just
re-blocked on the same lock. Every tool call on the server hung until the
300s caller-side timeout — for 40 hours, until the process was restarted.

The fix bounds both drains (``_discover_tools`` and the notification-driven
``_refresh_tools``) with ``asyncio.wait_for`` and treats a tool call that
never came back as a transport-liveness failure worth a rebuild.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mcp")


class _FakeAsyncCM:
    """Transport stand-in: yields streams, spawns nothing."""

    async def __aenter__(self):
        return (object(), object())

    async def __aexit__(self, *_exc):
        return False


def _capturing_session(captured: dict):
    """``ClientSession`` stand-in that records its construction kwargs."""

    class _FakeSession:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            session = MagicMock()

            async def _initialize():
                return MagicMock()

            session.initialize = _initialize
            return session

        async def __aexit__(self, *_exc):
            return False

    return _FakeSession


class _NeverAnswers:
    """Stand-in session whose ``tools/list`` response never arrives."""

    def __init__(self):
        self.calls = 0

    async def list_tools(self, cursor=None):
        self.calls += 1
        await asyncio.sleep(3600)


class TestDiscoveryDrainBound:
    def test_hanging_tools_list_raises_instead_of_wedging(self):
        """``_discover_tools`` must fail within the connect budget so
        ``run()``'s ``except Exception`` can tear the transport down."""
        from tools import mcp_tool

        server = mcp_tool.MCPServerTask("wedge-guard")
        server._config = {"connect_timeout": 0.2}
        server.session = _NeverAnswers()

        async def drive():
            start = time.monotonic()
            # The outer guard exists ONLY so a regression can't hang the
            # suite; the elapsed assertion below is what distinguishes
            # "bounded" (fixed) from "hung" (regressed).
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(server._discover_tools(), timeout=5.0)
            return time.monotonic() - start

        elapsed = asyncio.run(drive())
        assert elapsed < 2.0, (
            f"_discover_tools blocked {elapsed:.1f}s on an unanswered "
            f"tools/list — the connect_timeout bound was not applied."
        )
        # The lock must be free again: a still-held _rpc_lock is exactly
        # what blocked every tool call on the wedged server.
        assert not server._rpc_lock.locked()


class TestRefreshDrainBound:
    def test_hanging_refresh_signals_reconnect(self):
        """A ``tools/list_changed`` refresh that never gets an answer must
        give up and ask for a transport rebuild rather than hold the lock."""
        from tools import mcp_tool

        server = mcp_tool.MCPServerTask("refresh-guard")
        server._config = {"connect_timeout": 0.2}
        server.session = _NeverAnswers()
        server._registered_tool_names = ["mcp__refresh-guard__old_tool"]

        async def drive():
            start = time.monotonic()
            # No exception: the refresh task is fire-and-forget, so it
            # converts the timeout into a reconnect request.
            await asyncio.wait_for(server._refresh_tools(), timeout=5.0)
            return time.monotonic() - start

        elapsed = asyncio.run(drive())
        assert elapsed < 2.0, (
            f"_refresh_tools blocked {elapsed:.1f}s on an unanswered "
            f"tools/list — the drain bound was not applied."
        )
        assert server._reconnect_event.is_set(), (
            "an unanswered tools/list means response delivery is dead; the "
            "refresh must trigger a reconnect"
        )
        assert not server._rpc_lock.locked()
        # Registry state must be left alone — we never saw a fresh list.
        assert server._registered_tool_names == ["mcp__refresh-guard__old_tool"]


class TestTransportTimeoutTriggersReconnect:
    def test_reconnect_signalled_at_breaker_threshold(self, monkeypatch, tmp_path):
        """Tool calls that never come back must eventually rebuild the
        transport — the breaker alone only gates calls, it never revives."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools import mcp_tool

        async def _call_tool_hangs(*_a, **_kw):
            await asyncio.sleep(3600)

        signals = {"n": 0}
        server = MagicMock()
        server.name = "hung"
        server.session = SimpleNamespace(call_tool=_call_tool_hangs)
        server._rpc_lock = asyncio.Lock()
        server._is_recycled_stdio.return_value = False
        server._ready = SimpleNamespace(is_set=lambda: True)

        mcp_tool._servers["hung"] = server
        mcp_tool._server_error_counts.pop("hung", None)
        mcp_tool._server_breaker_opened_at.pop("hung", None)
        monkeypatch.setattr(
            mcp_tool, "_signal_reconnect",
            lambda _srv: signals.__setitem__("n", signals["n"] + 1) or True,
        )
        mcp_tool._ensure_mcp_loop()

        try:
            handler = mcp_tool._make_tool_handler("hung", "tool1", 0.2)
            for i in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD):
                parsed = json.loads(handler({}))
                assert "error" in parsed, parsed
                # Only the threshold-crossing failure asks for a rebuild.
                expected = 1 if i == mcp_tool._CIRCUIT_BREAKER_THRESHOLD - 1 else 0
                assert signals["n"] == expected, (
                    f"after {i + 1} timeouts: {signals['n']} reconnect signals"
                )
        finally:
            mcp_tool._servers.pop("hung", None)
            mcp_tool._server_error_counts.pop("hung", None)
            mcp_tool._server_breaker_opened_at.pop("hung", None)

    def test_server_side_errors_do_not_signal_reconnect(self, monkeypatch, tmp_path):
        """A server that answers with an error is alive — bumping the
        breaker is enough; rebuilding its transport would be pointless."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools import mcp_tool

        async def _call_tool_raises(*_a, **_kw):
            raise RuntimeError("tool blew up")

        signals = {"n": 0}
        server = MagicMock()
        server.name = "erroring"
        server.session = SimpleNamespace(call_tool=_call_tool_raises)
        server._rpc_lock = asyncio.Lock()
        server._is_recycled_stdio.return_value = False
        server._ready = SimpleNamespace(is_set=lambda: True)

        mcp_tool._servers["erroring"] = server
        mcp_tool._server_error_counts.pop("erroring", None)
        mcp_tool._server_breaker_opened_at.pop("erroring", None)
        monkeypatch.setattr(
            mcp_tool, "_signal_reconnect",
            lambda _srv: signals.__setitem__("n", signals["n"] + 1) or True,
        )
        mcp_tool._ensure_mcp_loop()

        try:
            handler = mcp_tool._make_tool_handler("erroring", "tool1", 5.0)
            for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD):
                assert "error" in json.loads(handler({}))
            assert signals["n"] == 0
        finally:
            mcp_tool._servers.pop("erroring", None)
            mcp_tool._server_error_counts.pop("erroring", None)
            mcp_tool._server_breaker_opened_at.pop("erroring", None)


class TestSessionReadTimeout:
    def test_sdk_still_accepts_the_kwarg(self):
        import inspect

        from mcp import ClientSession

        assert "read_timeout_seconds" in inspect.signature(
            ClientSession.__init__
        ).parameters, "pinned mcp SDK no longer accepts read_timeout_seconds"

    def test_http_session_gets_a_read_timeout(self):
        """``ClientSession`` must carry ``read_timeout_seconds``: without it
        the SDK awaits every response on ``anyio.fail_after(None)``."""
        from tools import mcp_tool

        captured: dict = {}
        server = mcp_tool.MCPServerTask("read-timeout-http")
        config = {
            "url": "https://example.invalid/sse",
            "transport": "sse",
            "timeout": 42,
        }

        async def drive():
            with patch.object(mcp_tool, "sse_client", lambda **_k: _FakeAsyncCM()), \
                 patch.object(mcp_tool, "ClientSession",
                              _capturing_session(captured)), \
                 patch.object(mcp_tool.MCPServerTask, "_discover_tools",
                              new=lambda _self: asyncio.sleep(0)), \
                 patch.object(mcp_tool.MCPServerTask, "_wait_for_lifecycle_event",
                              new=lambda _self: asyncio.sleep(0)):
                await server._run_http(config)

        asyncio.run(drive())
        assert captured.get("read_timeout_seconds") == timedelta(seconds=42), (
            f"HTTP session built without a read timeout: {captured!r}"
        )

    def test_stdio_session_gets_a_read_timeout(self):
        """Same backstop on stdio: a child that reads the request and never
        writes a response is the same unbounded await."""
        from tools import mcp_tool

        captured: dict = {}
        server = mcp_tool.MCPServerTask("read-timeout-stdio")
        config = {"command": "fake-mcp", "args": [], "timeout": 42}

        async def drive():
            with patch.object(mcp_tool, "stdio_client", lambda *_a, **_k: _FakeAsyncCM()), \
                 patch.object(mcp_tool, "ClientSession",
                              _capturing_session(captured)), \
                 patch.object(mcp_tool, "_resolve_stdio_command", lambda c, e: (c, e)), \
                 patch.object(mcp_tool, "_write_stderr_log_header", lambda *_a, **_k: None), \
                 patch.object(mcp_tool, "_get_mcp_stderr_log", lambda: None), \
                 patch("tools.osv_check.check_package_for_malware",
                       lambda *_a, **_k: None), \
                 patch.object(mcp_tool.MCPServerTask, "_discover_tools",
                              new=lambda _self: asyncio.sleep(0)), \
                 patch.object(mcp_tool.MCPServerTask, "_wait_for_lifecycle_event",
                              new=lambda _self: asyncio.sleep(0)):
                await server._run_stdio(config)

        asyncio.run(drive())
        assert captured.get("read_timeout_seconds") == timedelta(seconds=42), (
            f"stdio session built without a read timeout: {captured!r}"
        )

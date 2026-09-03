"""The streamable-HTTP transport must accept both SDK generations' arity.

``streamable_http_client`` yields ``(read, write, get_session_id)`` on mcp 1.x
and ``(read, write)`` on mcp 2.x. ``_run_http`` unpacked a fixed 3-tuple, which
is 1.x's shape, so on 2.x every HTTP and SSE server failed its handshake with
``ValueError: not enough values to unpack (expected 3, got 2)`` and parked after
exhausting its retry ladder.

It survived review because the existing coverage
(``test_mcp_client_cert.py``) fakes the transport with a 3-tuple — encoding the
old shape into the test — and because the common server configs are stdio,
which is a different code path entirely. So the assertion that matters is not
"does the happy path work" but "does it work for *each* arity the supported SDK
range actually yields".
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest


def _patch_sdk_async_client(dummy):
    from tools.mcp_tool import sdk_httpx

    return patch.object(sdk_httpx(), "AsyncClient", dummy)


class _DummyAsyncClient:
    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _DummySession:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def initialize(self):
        return None


def _transport_yielding(*values):
    class _Ctx:
        async def __aenter__(self):
            return values

        async def __aexit__(self, *a):
            return False

    return _Ctx()


@pytest.mark.parametrize("sdk,streams", [
    ("mcp 2.x", (MagicMock(), MagicMock())),
    ("mcp 1.x", (MagicMock(), MagicMock(), (lambda: None))),
])
def test_run_http_accepts_the_arity_each_sdk_generation_yields(sdk, streams):
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("remote")
    seen: dict = {}

    async def _discover_tools(self):
        seen["connected"] = True
        self._shutdown_event.set()

    async def _drive():
        with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
             patch("tools.mcp_tool._MCP_NEW_HTTP", True), \
             _patch_sdk_async_client(_DummyAsyncClient), \
             patch("tools.mcp_tool.streamable_http_client",
                   return_value=_transport_yielding(*streams)), \
             patch("tools.mcp_tool.ClientSession", _DummySession), \
             patch.object(MCPServerTask, "_discover_tools", _discover_tools):
            await server._run_http({"url": "https://example.com/mcp"})

    asyncio.run(_drive())

    assert seen.get("connected") is True, f"handshake never completed on {sdk}"
    assert server._error is None, f"{sdk}: {server._error!r}"


def test_the_session_streams_are_the_first_two_yielded():
    """Positional, not named: 1.x's third element is not a stream."""
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("remote")
    read, write = MagicMock(), MagicMock()
    passed: dict = {}

    class _CapturingSession(_DummySession):
        def __init__(self, *args, **kwargs):
            passed["args"] = args
            super().__init__(*args, **kwargs)

    async def _discover_tools(self):
        self._shutdown_event.set()

    async def _drive():
        with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
             patch("tools.mcp_tool._MCP_NEW_HTTP", True), \
             _patch_sdk_async_client(_DummyAsyncClient), \
             patch("tools.mcp_tool.streamable_http_client",
                   return_value=_transport_yielding(read, write, (lambda: None))), \
             patch("tools.mcp_tool.ClientSession", _CapturingSession), \
             patch.object(MCPServerTask, "_discover_tools", _discover_tools):
            await server._run_http({"url": "https://example.com/mcp"})

    asyncio.run(_drive())

    assert passed["args"][:2] == (read, write)


def test_the_seeded_protocol_header_matches_the_handshake_the_client_sends():
    """Header and body must agree about which revision this connection speaks.

    `ClientSession.initialize()` sends `LATEST_HANDSHAKE_VERSION`; from
    2026-07-28 onward `LATEST_PROTOCOL_VERSION` names a revision that replaced
    the handshake with a per-request envelope. Seeding the header from the
    latter advertised a revision the body does not speak, and a conforming
    server answered `params._meta is missing the required envelope key(s)` --
    observed against a live MCP endpoint, not hypothesised.
    """
    from tools import mcp_tool

    try:
        from mcp.client.session import LATEST_HANDSHAKE_VERSION as sdk_handshake
    except ImportError:
        pytest.skip("SDK predates the handshake/protocol version split")

    assert mcp_tool.LATEST_HANDSHAKE_VERSION == sdk_handshake


def test_the_seeded_header_is_the_handshake_version_on_the_wire():
    """Asserted through the header dict `_run_http` actually builds."""
    from unittest.mock import patch as _patch

    from tools.mcp_tool import MCPServerTask, LATEST_HANDSHAKE_VERSION

    server = MCPServerTask("remote")
    seen: dict = {}

    class _CapturingAsyncClient(_DummyAsyncClient):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    async def _discover_tools(self):
        self._shutdown_event.set()

    async def _drive():
        with _patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
             _patch("tools.mcp_tool._MCP_NEW_HTTP", True), \
             _patch_sdk_async_client(_CapturingAsyncClient), \
             _patch("tools.mcp_tool.streamable_http_client",
                    return_value=_transport_yielding(MagicMock(), MagicMock())), \
             _patch("tools.mcp_tool.ClientSession", _DummySession), \
             _patch.object(MCPServerTask, "_discover_tools", _discover_tools):
            await server._run_http({"url": "https://example.com/mcp"})

    asyncio.run(_drive())

    headers = {k.lower(): v for k, v in (seen.get("headers") or {}).items()}
    assert headers.get("mcp-protocol-version") == LATEST_HANDSHAKE_VERSION


def test_an_explicit_protocol_header_still_wins():
    """The override exists so a server needing a specific revision can have it."""
    from unittest.mock import patch as _patch

    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("remote")
    seen: dict = {}

    class _CapturingAsyncClient(_DummyAsyncClient):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    async def _discover_tools(self):
        self._shutdown_event.set()

    async def _drive():
        with _patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
             _patch("tools.mcp_tool._MCP_NEW_HTTP", True), \
             _patch_sdk_async_client(_CapturingAsyncClient), \
             _patch("tools.mcp_tool.streamable_http_client",
                    return_value=_transport_yielding(MagicMock(), MagicMock())), \
             _patch("tools.mcp_tool.ClientSession", _DummySession), \
             _patch.object(MCPServerTask, "_discover_tools", _discover_tools):
            await server._run_http({
                "url": "https://example.com/mcp",
                "headers": {"MCP-Protocol-Version": "2025-06-18"},
            })

    asyncio.run(_drive())

    headers = {k.lower(): v for k, v in (seen.get("headers") or {}).items()}
    assert headers.get("mcp-protocol-version") == "2025-06-18"

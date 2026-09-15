"""Proxy support for MCP HTTP/SSE transports.

Behaviour contract: httpx auto-detects environment/OS proxies only when ``transport is None``
(``allow_env_proxies = trust_env and transport is None``). Both MCP HTTP transports pass a custom
transport — the wire-body cap — so HTTP_PROXY / HTTPS_PROXY and the OS proxy were silently ignored
for every HTTP/SSE MCP server, and a network that reaches the MCP host only through a proxy failed
every connect (``All connection attempts failed``) with the server parked.

These tests pin the contract: a proxy that applies to the server URL reaches the SDK client as
explicit ``mounts``, a ``NO_PROXY`` host stays direct, and the transport wiring (body cap, TLS
kwargs, headers/auth passthrough) is otherwise unchanged.
"""

from __future__ import annotations

import asyncio
import os
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

URL = "https://mcp.example.com/mcp"
PROXY = "http://127.0.0.1:10808"


@pytest.fixture
def env_only_proxy(monkeypatch):
    """Environment-only proxy discovery, so the host machine's OS/registry proxy can't leak in."""
    def _from_env() -> dict:
        found = {}
        for scheme in ("http", "https", "all"):
            value = os.environ.get(f"{scheme}_proxy") or os.environ.get(f"{scheme.upper()}_PROXY")
            if value:
                found[scheme] = value
        return found

    monkeypatch.setattr(urllib.request, "getproxies", _from_env)
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    yield


class TestProxyMounts:
    def test_no_proxy_leaves_the_connect_direct(self, env_only_proxy):
        from tools.mcp_tool import sdk_httpx
        from tools.mcp_tool_transport import _mcp_proxy_mounts

        assert _mcp_proxy_mounts(sdk_httpx(), URL, True, None) is None

    def test_https_proxy_becomes_a_mount(self, env_only_proxy, monkeypatch):
        from tools.mcp_tool import sdk_httpx
        from tools.mcp_tool_transport import _mcp_proxy_mounts

        monkeypatch.setenv("HTTPS_PROXY", PROXY)
        mounts = _mcp_proxy_mounts(sdk_httpx(), URL, True, None)

        assert set(mounts) == {"https://"}
        assert mounts["https://"] is not None

    def test_all_proxy_covers_both_schemes(self, env_only_proxy, monkeypatch):
        from tools.mcp_tool import sdk_httpx
        from tools.mcp_tool_transport import _mcp_proxy_mounts

        monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:10808")  # Clash/WSL alias form
        mounts = _mcp_proxy_mounts(sdk_httpx(), URL, True, None)

        assert set(mounts) == {"http://", "https://"}

    def test_no_proxy_host_is_not_mounted(self, env_only_proxy, monkeypatch):
        from tools.mcp_tool import sdk_httpx
        from tools.mcp_tool_transport import _mcp_proxy_mounts

        monkeypatch.setenv("HTTPS_PROXY", PROXY)
        monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: host == "mcp.example.com")

        assert _mcp_proxy_mounts(sdk_httpx(), URL, True, None) is None


class _RecordingClient:
    """Stands in for the SDK's AsyncClient and records the kwargs it was built with."""

    captured: dict = {}

    def __init__(self, **kwargs):
        type(self).captured = dict(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _async_cm(value):
    class _CM:
        async def __aenter__(self):
            return value

        async def __aexit__(self, *args):
            return False

    return _CM()


class TestTransportWiring:
    def test_streamable_http_client_carries_mounts_and_body_cap(self, env_only_proxy, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", PROXY)
        from tools.mcp_tool import MCPServerTask, sdk_httpx

        server = MCPServerTask("remote")
        streams = server._streamable_http_transport(URL, {}, 5.0, True, None, None, False, set())
        _RecordingClient.captured = {}

        with patch.object(sdk_httpx(), "AsyncClient", _RecordingClient), \
             patch("tools.mcp_tool.streamable_http_client", MagicMock(return_value=_async_cm((MagicMock(), MagicMock())))):
            asyncio.run(_drive(streams))

        captured = _RecordingClient.captured
        assert captured["mounts"]["https://"] is not None  # proxy restored
        assert captured["transport"] is not None  # wire-body cap still the default transport

    def test_sse_client_factory_carries_mounts(self, env_only_proxy, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", PROXY)
        from tools.mcp_tool import MCPServerTask, sdk_httpx

        server = MCPServerTask("remote")
        _RecordingClient.captured = {}

        with patch("tools.mcp_tool.sse_client", MagicMock(return_value=_async_cm((MagicMock(), MagicMock())))) as sse, \
             patch.object(sdk_httpx(), "AsyncClient", _RecordingClient):
            server._sse_transport(URL, {}, 5.0, True, None, None, False)
            factory = sse.call_args.kwargs["httpx_client_factory"]
            client = factory(headers={"X-Test": "1"}, timeout=None, auth=None)

        assert client is not None
        assert _RecordingClient.captured["mounts"]["https://"] is not None
        assert _RecordingClient.captured["headers"] == {"X-Test": "1"}  # passthrough intact
        assert _RecordingClient.captured["transport"] is not None


async def _drive(streams):
    async with streams:
        pass

"""Proxy support for MCP HTTP/SSE transports (#111794).

httpx auto-detects environment/OS proxies only when ``transport is None``; both MCP HTTP
transports pass a custom transport (the wire-body cap), so HTTP_PROXY / HTTPS_PROXY were silently
ignored for every HTTP/SSE MCP server. The fix hands the SDK client explicit ``mounts``.

Invariants: (1) a proxy that applies to the server URL becomes a mount and a NO_PROXY host
(including the CIDR form only ``agent.proxy_bypass`` understands) stays direct; (2) both client
builders carry the mounts next to the body-cap transport, with headers/auth passthrough intact.
"""

from __future__ import annotations

import asyncio
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

URL = "https://mcp.example.com/mcp"
PROXY = "http://127.0.0.1:10808"
_PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy")


@pytest.fixture
def env_only_proxy(monkeypatch):
    """Environment-only proxy discovery so the host's OS/registry proxy can't leak in."""
    for key in _PROXY_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(urllib.request, "getproxies", urllib.request.getproxies_environment)
    monkeypatch.setattr(urllib.request, "proxy_bypass", urllib.request.proxy_bypass_environment)
    from tools.mcp_tool import _ensure_mcp_sdk, sdk_httpx

    if not _ensure_mcp_sdk() or sdk_httpx() is None:
        pytest.skip("mcp SDK not installed")


def test_proxy_env_becomes_a_mount_and_no_proxy_stays_direct(env_only_proxy, monkeypatch):
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_transport import _mcp_proxy_mounts

    httpx = sdk_httpx()
    assert _mcp_proxy_mounts(httpx, URL, True, None) is None  # no proxy configured: direct

    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    mounts = _mcp_proxy_mounts(httpx, URL, True, None)
    assert set(mounts) == {"https://"}
    # The mount wins over transport= for matching URLs, so both of its legs carry the wire-body cap.
    assert {type(leg).__name__ for leg in _legs(mounts["https://"])} == {"_BodyCapTransport"}
    assert _routed(mounts["https://"], URL) == "proxied"

    monkeypatch.setenv("HTTP_PROXY", PROXY)  # loopback is never dialed through a proxy, NO_PROXY or not
    mounts = _mcp_proxy_mounts(httpx, "http://127.0.0.1:5000/mcp", True, None)
    assert _routed(mounts["http://"], "http://127.0.0.1:5000/mcp") == "direct"

    monkeypatch.setenv("NO_PROXY", "mcp.example.com")
    assert _routed(_mcp_proxy_mounts(httpx, URL, True, None)["https://"], URL) == "direct"
    monkeypatch.setenv("NO_PROXY", "10.0.0.0/8")  # CIDR: only the repo matcher understands it
    router = _mcp_proxy_mounts(httpx, "https://10.1.2.3/mcp", True, None)["https://"]
    assert _routed(router, "https://10.1.2.3/mcp") == "direct"
    assert _routed(router, "https://11.1.2.3/mcp") == "proxied"


def test_each_request_gets_its_own_proxy_decision(env_only_proxy, monkeypatch):
    """The client that reaches the MCP server also dials its OAuth authorization server (#160462):
    a NO_PROXY MCP host must not send the AS metadata and /token calls direct, and a proxied MCP
    host must not send a NO_PROXY AS through the proxy."""
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_transport import _mcp_proxy_mounts

    httpx = sdk_httpx()
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    monkeypatch.setenv("NO_PROXY", ".internal.example")
    router = _mcp_proxy_mounts(httpx, "https://mcp.internal.example/mcp", True, None)["https://"]
    assert _routed(router, "https://mcp.internal.example/mcp") == "direct"
    assert _routed(router, "https://auth.vendor.example/token") == "proxied"

    router = _mcp_proxy_mounts(httpx, "https://mcp.vendor.example/mcp", True, None)["https://"]
    assert _routed(router, "https://mcp.vendor.example/mcp") == "proxied"
    assert _routed(router, "https://sso.internal.example/token") == "direct"


def _legs(router):
    return [router.proxied, router.direct]


def _routed(router, url: str) -> str:
    """Which leg ``router`` sends a request for ``url`` to, without dialing anything."""
    import httpx as plain_httpx

    class _Leg:
        def __init__(self, name):
            self.name = name

        async def handle_async_request(self, request):
            return self.name

    router.proxied, router.direct = _Leg("proxied"), _Leg("direct")
    return asyncio.run(router.handle_async_request(plain_httpx.Request("GET", url)))


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


async def _drive(streams):
    async with streams:
        pass


def test_both_client_builders_carry_proxy_mounts_next_to_the_body_cap(env_only_proxy, monkeypatch):
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

    _RecordingClient.captured = {}
    with patch("tools.mcp_tool.sse_client", MagicMock(return_value=_async_cm((MagicMock(), MagicMock())))) as sse, \
         patch.object(sdk_httpx(), "AsyncClient", _RecordingClient):
        server._sse_transport(URL, {}, 5.0, True, None, None, False)
        sse.call_args.kwargs["httpx_client_factory"](headers={"X-Test": "1"}, timeout=None, auth=None)
    captured = _RecordingClient.captured
    assert captured["mounts"]["https://"] is not None
    assert captured["headers"] == {"X-Test": "1"}  # SDK passthrough intact
    assert captured["transport"] is not None


def test_preflight_probe_uses_the_same_proxy_mounts_as_the_connect_client(env_only_proxy, monkeypatch):
    """The content-type preflight must reach the server the way the SDK client will: explicit mounts
    (repo NO_PROXY/loopback rules), not httpx's own env auto-detection."""
    import httpx

    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    from tools.mcp_tool import MCPServerTask

    class _Probe(_RecordingClient):
        async def head(self, *a, **k):
            raise httpx.ConnectError("stub")

    with patch.object(httpx, "AsyncClient", _Probe):
        asyncio.run(MCPServerTask("remote")._preflight_content_type(URL, timeout=1.0))
    captured = _Probe.captured
    assert {type(leg).__name__ for leg in _legs(captured["mounts"]["https://"])} == {"_BodyCapTransport"}
    assert captured["transport"] is not None  # explicit transport: httpx env proxy auto-detection is off

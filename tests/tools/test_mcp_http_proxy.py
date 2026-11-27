"""Proxy support for MCP HTTP/SSE transports (#111794).

httpx auto-detects environment/OS proxies only when ``transport is None``; both MCP HTTP
transports pass a custom transport (the wire-body cap), so HTTP_PROXY / HTTPS_PROXY were silently
ignored for every HTTP/SSE MCP server. The fix hands the SDK client explicit ``mounts``.

Invariants: (1) every request follows the proxy settings by its own host: a proxy becomes a mount,
and a NO_PROXY host (including the CIDR form only ``agent.proxy_bypass`` understands) stays direct,
also when it is the MCP host and the same client's OAuth requests go to hosts that need the proxy;
(2) both client builders carry the mounts next to the body-cap transport, with headers/auth
passthrough intact.
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
    # The mount wins over transport= for matching URLs, so both of its legs carry the wire-body cap.
    assert set(mounts) == {"https://"}
    assert type(mounts["https://"].proxy_leg()).__name__ == "_BodyCapTransport"
    assert type(mounts["https://"]._direct).__name__ == "_BodyCapTransport"
    assert _routed(httpx, mounts, URL) == "proxy"

    monkeypatch.setenv("HTTP_PROXY", PROXY)  # loopback is never dialed through a proxy, NO_PROXY or not
    loopback = "http://127.0.0.1:5000/mcp"
    assert _routed(httpx, _mcp_proxy_mounts(httpx, loopback, True, None), loopback) == "direct"

    monkeypatch.setenv("NO_PROXY", "mcp.example.com")
    assert _routed(httpx, _mcp_proxy_mounts(httpx, URL, True, None), URL) == "direct"
    monkeypatch.setenv("NO_PROXY", "10.0.0.0/8")  # CIDR: only the repo matcher understands it
    inside, outside = "https://10.1.2.3/mcp", "https://11.1.2.3/mcp"
    assert _routed(httpx, _mcp_proxy_mounts(httpx, inside, True, None), inside) == "direct"
    assert _routed(httpx, _mcp_proxy_mounts(httpx, outside, True, None), outside) == "proxy"


def _Recorder(httpx, name, taken):
    """A transport that records *name* in *taken* instead of sending the request."""

    class _Recorder(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            taken.append(name)
            return httpx.Response(204)

    return _Recorder()


def _routed(httpx, mounts, url):
    """The leg, ``"proxy"`` or ``"direct"``, that the client's mount sends a request for *url* on."""
    router = mounts[url.split("://", 1)[0] + "://"]
    taken = []
    router._proxied, router._direct = _Recorder(httpx, "proxy", taken), _Recorder(httpx, "direct", taken)
    asyncio.run(router.handle_async_request(httpx.Request("GET", url)))
    return taken[0]


def test_no_proxy_mcp_host_keeps_the_proxy_for_its_oauth_server(env_only_proxy, monkeypatch):
    """The MCP client also sends the OAuth requests of its auth flow (authorization server metadata,
    token, refresh) to other hosts. An MCP host in NO_PROXY stays direct without taking the proxy
    away from an authorization server that is only reachable through it."""
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_transport import _mcp_proxy_mounts

    httpx = sdk_httpx()
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    monkeypatch.setenv("NO_PROXY", ".example.com")
    mounts = _mcp_proxy_mounts(httpx, URL, True, None)
    assert mounts is not None
    assert _routed(httpx, mounts, URL) == "direct"
    assert _routed(httpx, mounts, "https://auth.example.org/oauth/token") == "proxy"


def test_unbuildable_proxy_fails_only_requests_that_need_it(env_only_proxy, monkeypatch):
    """A proxy httpx cannot build must not break a server that never uses it (loopback, NO_PROXY)."""
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_transport import _mcp_proxy_mounts

    httpx = sdk_httpx()
    monkeypatch.setenv("HTTP_PROXY", "ftp://proxy.example:21")  # a scheme httpx rejects
    loopback = "http://127.0.0.1:5000/mcp"
    router = _mcp_proxy_mounts(httpx, loopback, True, None)["http://"]
    router._direct = _Recorder(httpx, "direct", [])
    asyncio.run(router.handle_async_request(httpx.Request("GET", loopback)))
    with pytest.raises(ValueError):
        asyncio.run(router.handle_async_request(httpx.Request("GET", "http://auth.example.org/token")))


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
    assert type(captured["mounts"]["https://"]).__name__ == "_ProxyRouterTransport"
    assert type(captured["mounts"]["https://"].proxy_leg()).__name__ == "_BodyCapTransport"
    assert captured["transport"] is not None  # explicit transport: httpx env proxy auto-detection is off

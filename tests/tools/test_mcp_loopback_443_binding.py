"""IPv6-loopback and implicit-443 binding coverage for MCP proxy mounts.

``_mcp_proxy_mounts`` must keep a test server bound to ``::1`` direct (loopback is
never dialed through a proxy) and must treat an implicit-443 ``https://`` URL
identically to an explicit-``:443`` one — the preflight probe and the connect
client resolve mounts independently, so any divergence double-dials or
proxy-leaks one leg.
"""

from __future__ import annotations

import urllib.request

import pytest

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


def test_ipv6_loopback_test_server_never_proxied(env_only_proxy, monkeypatch):
    """A ``::1``-bound test server stays direct even with a proxy configured."""
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_transport import _mcp_proxy_mounts

    monkeypatch.setenv("HTTP_PROXY", PROXY)
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    httpx = sdk_httpx()
    assert _mcp_proxy_mounts(httpx, "http://[::1]:5000/mcp", True, None) is None
    assert _mcp_proxy_mounts(httpx, "https://[::1]:8443/mcp", True, None) is None


def test_implicit_and_explicit_443_behave_identically(env_only_proxy, monkeypatch):
    """``https://host/mcp`` and ``https://host:443/mcp`` resolve the same mounts."""
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_transport import _mcp_proxy_mounts

    httpx = sdk_httpx()
    implicit = "https://mcp.example.com/mcp"
    explicit = "https://mcp.example.com:443/mcp"

    assert _mcp_proxy_mounts(httpx, implicit, True, None) is None
    assert _mcp_proxy_mounts(httpx, explicit, True, None) is None

    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    implicit_mounts = _mcp_proxy_mounts(httpx, implicit, True, None)
    explicit_mounts = _mcp_proxy_mounts(httpx, explicit, True, None)
    assert implicit_mounts is not None and explicit_mounts is not None
    assert set(implicit_mounts) == set(explicit_mounts) == {"https://"}

    monkeypatch.setenv("NO_PROXY", "mcp.example.com")
    assert _mcp_proxy_mounts(httpx, implicit, True, None) is None
    assert _mcp_proxy_mounts(httpx, explicit, True, None) is None

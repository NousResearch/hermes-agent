"""MCP HTTP transports must resolve ``ssl_verify`` through ``agent.ssl_verify`` (#146924).

A long-lived runtime (``hermes serve`` / ``dashboard`` / ``gateway``) outlives a ``hermes pm
install`` environment rotation: ``certifi.where()`` stays frozen on the deleted environment
path, and httpx builds its default TLS context from it at client construction —
``FileNotFoundError`` inside transport bring-up, every MCP connect timing out until restart.
The LLM transport already routes through ``resolve_httpx_verify`` for exactly this failure
mode; the MCP paths handed the raw config value straight to httpx instead. These tests pin
the two config reads that must resolve first — ``_run_http``'s transport tuple and
``_prepare_run``'s preflight call: a dead bundle downgrades to the platform context, a
healthy one keeps ``True`` (zero behaviour change for the normal case), and an explicit
``ssl_verify: false`` stays off.
"""

from __future__ import annotations

import asyncio
import shutil
import ssl
from pathlib import Path
from unittest.mock import MagicMock, patch

import certifi
import pytest

from agent.ssl_verify import _shared_context
from tools.mcp_tool import MCPServerTask


@pytest.fixture
def tls_env(monkeypatch):
    """No ambient CA/proxy state: the resolution must depend on the config value alone."""
    for var in ("SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(var, raising=False)
    import urllib.request
    monkeypatch.setattr(urllib.request, "getproxies", dict)  # macOS reads system proxies here


@pytest.fixture
def dead_certifi_bundle(monkeypatch):
    """A bundle that died in an install-environment rotation and cannot be re-derived —
    ``certifi_bundle_path() -> None`` is ssl_verify's documented unrecoverable outcome."""
    from agent import ssl_verify
    monkeypatch.setattr(ssl_verify, "certifi_bundle_path", lambda: None)


class _SpyAsyncClient:
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


class _SpyHTTPTransport:
    """Record the kwargs (``verify`` among them) the transport bring-up forwards to httpx."""

    calls: list = []

    def __init__(self, **kwargs):
        _SpyHTTPTransport.calls.append(kwargs)


def _transport_yielding(*values):
    class _Ctx:
        async def __aenter__(self):
            return values

        async def __aexit__(self, *a):
            return False

    return _Ctx()


def _run_http_transport_calls(config: dict) -> list:
    from tools.mcp_tool import sdk_httpx

    async def _discover_tools(self):
        self._shutdown_event.set()

    async def _drive():
        _SpyHTTPTransport.calls = []
        with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
             patch("tools.mcp_tool._MCP_NEW_HTTP", True), \
             patch.object(sdk_httpx(), "AsyncClient", _SpyAsyncClient), \
             patch.object(sdk_httpx(), "AsyncHTTPTransport", _SpyHTTPTransport), \
             patch("tools.mcp_tool.streamable_http_client",
                   return_value=_transport_yielding(MagicMock(), MagicMock())), \
             patch("tools.mcp_tool.ClientSession", _DummySession), \
             patch.object(MCPServerTask, "_discover_tools", _discover_tools):
            await MCPServerTask("remote")._run_http(dict(config))

    asyncio.run(_drive())
    return _SpyHTTPTransport.calls


def test_run_http_downgrades_to_the_platform_context_when_the_bundle_is_dead(tls_env, dead_certifi_bundle):
    calls = _run_http_transport_calls({"url": "https://mcp.example.test/mcp"})
    assert calls, "transport bring-up never reached httpx"
    # True would make httpx build its context from the dead certifi path (FileNotFoundError).
    assert isinstance(calls[0]["verify"], ssl.SSLContext)


def test_run_http_keeps_true_for_a_healthy_bundle(tls_env):
    calls = _run_http_transport_calls({"url": "https://mcp.example.test/mcp"})
    assert calls and calls[0]["verify"] is True


def test_run_http_keeps_an_explicit_disable(tls_env):
    calls = _run_http_transport_calls({"url": "https://mcp.example.test/mcp", "ssl_verify": False})
    assert calls and calls[0]["verify"] is False


def test_run_http_resolves_a_custom_ca_bundle_path_to_its_context(tls_env, tmp_path):
    """ssl_verify doubles as a custom CA bundle when set to a path: it reaches the TLS layer as
    a context loading that bundle, not a bare string (provider config splits the same two roles
    into ssl_verify / ssl_ca_cert)."""
    ca_bundle = tmp_path / "ca.pem"
    shutil.copyfile(certifi.where(), ca_bundle)  # a real PEM: the resolved context loads it
    expected = _shared_context(str(Path(str(ca_bundle)).expanduser().resolve()))
    calls = _run_http_transport_calls({"url": "https://mcp.example.test/mcp", "ssl_verify": str(ca_bundle)})
    assert calls and calls[0]["verify"] is expected


def test_prepare_run_preflight_receives_the_resolved_verify(tls_env, dead_certifi_bundle, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    task = MCPServerTask("probe_srv")
    captured: dict = {}

    async def _fake_preflight(url, *, headers=None, ssl_verify=True, client_cert=None,
                              timeout=5.0, strict_redirect_headers=False):
        captured["verify"] = ssl_verify

    monkeypatch.setattr(task, "_preflight_content_type", _fake_preflight)
    asyncio.run(task._prepare_run({"url": "https://mcp.example.test/mcp",
                                   "sampling": {"enabled": False}, "elicitation": {"enabled": False}}))
    assert isinstance(captured["verify"], ssl.SSLContext)

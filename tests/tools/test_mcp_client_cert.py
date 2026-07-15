"""Tests for mTLS client certificate config on MCP HTTP/SSE transports.

Covers:

1. ``_resolve_client_cert`` helper — string, tuple, encrypted-key, validation
   errors, missing-file errors.

2. HTTP (new SDK ``streamable_http_client``) path forwards ``cert=`` into the
   user-owned ``httpx.AsyncClient``.

3. SSE path forwards ``cert`` and ``ssl_verify`` via an ``httpx_client_factory``
   without breaking the OAuth/headers/timeout passthrough.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _resolve_client_cert helper
# ---------------------------------------------------------------------------


class TestResolveClientCert:
    def test_returns_none_when_unset(self):
        from tools.mcp_tool import _resolve_client_cert

        assert _resolve_client_cert("srv", {}) is None
        assert _resolve_client_cert("srv", {"url": "https://x"}) is None

    def test_string_form_single_pem(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        pem = tmp_path / "combined.pem"
        pem.write_text("dummy")

        result = _resolve_client_cert("srv", {"client_cert": str(pem)})
        assert result == str(pem)

    def test_string_cert_with_separate_key(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        cert = tmp_path / "client.crt"
        key = tmp_path / "client.key"
        cert.write_text("cert")
        key.write_text("key")

        result = _resolve_client_cert("srv", {
            "client_cert": str(cert),
            "client_key": str(key),
        })
        assert result == (str(cert), str(key))

    def test_list_form_two_elements(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        cert = tmp_path / "client.crt"
        key = tmp_path / "client.key"
        cert.write_text("cert")
        key.write_text("key")

        result = _resolve_client_cert("srv", {
            "client_cert": [str(cert), str(key)],
        })
        assert result == (str(cert), str(key))

    def test_list_form_with_passphrase(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        cert = tmp_path / "client.crt"
        key = tmp_path / "client.key"
        cert.write_text("cert")
        key.write_text("key")

        result = _resolve_client_cert("srv", {
            "client_cert": [str(cert), str(key), "passphrase"],
        })
        assert result == (str(cert), str(key), "passphrase")

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        from tools.mcp_tool import _resolve_client_cert

        monkeypatch.setenv("HOME", str(tmp_path))
        pem = tmp_path / "client.pem"
        pem.write_text("dummy")

        result = _resolve_client_cert("srv", {"client_cert": "~/client.pem"})
        assert result == str(pem)

    def test_missing_file_raises(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        with pytest.raises(FileNotFoundError, match=r"srv.*client_cert.*not found"):
            _resolve_client_cert("srv", {
                "client_cert": str(tmp_path / "nope.pem"),
            })

    def test_missing_key_file_raises(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        cert = tmp_path / "client.crt"
        cert.write_text("cert")

        with pytest.raises(FileNotFoundError, match=r"srv.*client_key.*not found"):
            _resolve_client_cert("srv", {
                "client_cert": str(cert),
                "client_key": str(tmp_path / "missing.key"),
            })

    def test_list_with_bad_length_raises(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        with pytest.raises(ValueError, match=r"list form must have 2 or 3"):
            _resolve_client_cert("srv", {"client_cert": [str(tmp_path / "x")]})

    def test_list_plus_client_key_rejected(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        cert = tmp_path / "client.crt"
        key = tmp_path / "client.key"
        cert.write_text("cert")
        key.write_text("key")

        with pytest.raises(ValueError, match=r"either client_cert as a list"):
            _resolve_client_cert("srv", {
                "client_cert": [str(cert), str(key)],
                "client_key": str(key),
            })

    def test_non_string_path_rejected(self):
        from tools.mcp_tool import _resolve_client_cert

        with pytest.raises(ValueError, match=r"client_cert must be a non-empty string"):
            _resolve_client_cert("srv", {"client_cert": 123})

    def test_password_must_be_string(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        cert = tmp_path / "client.crt"
        key = tmp_path / "client.key"
        cert.write_text("cert")
        key.write_text("key")

        with pytest.raises(ValueError, match=r"key passphrase.*must be a string"):
            _resolve_client_cert("srv", {
                "client_cert": [str(cert), str(key), 42],
            })


# ---------------------------------------------------------------------------
# HTTP transport — cert forwarded into httpx.AsyncClient
# ---------------------------------------------------------------------------


class TestHTTPClientCert:
    def test_cert_forwarded_to_async_client(self, tmp_path):
        """When client_cert is set, the new-SDK HTTP path passes ``cert=``
        into ``httpx.AsyncClient``."""
        from tools.mcp_tool import MCPServerTask

        cert = tmp_path / "client.pem"
        cert.write_text("dummy")

        server = MCPServerTask("remote")
        captured: dict = {}

        class DummyAsyncClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class DummyTransportCtx:
            async def __aenter__(self):
                return MagicMock(), MagicMock(), (lambda: None)

            async def __aexit__(self, *a):
                return False

        class DummySession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def initialize(self):
                return None

        async def _discover_tools(self):
            self._shutdown_event.set()

        async def _drive():
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
                 patch("tools.mcp_tool._MCP_NEW_HTTP", True), \
                 patch("httpx.AsyncClient", DummyAsyncClient), \
                 patch("tools.mcp_tool.streamable_http_client",
                       return_value=DummyTransportCtx()), \
                 patch("tools.mcp_tool.ClientSession", DummySession), \
                 patch.object(MCPServerTask, "_discover_tools", _discover_tools):
                await server._run_http({
                    "url": "https://example.com/mcp",
                    "client_cert": str(cert),
                })

        asyncio.run(_drive())
        assert captured.get("cert") == str(cert)

    def test_cert_tuple_forwarded(self, tmp_path):
        """List/tuple form resolves to a tuple in ``cert=``."""
        from tools.mcp_tool import MCPServerTask

        cert = tmp_path / "client.crt"
        key = tmp_path / "client.key"
        cert.write_text("cert")
        key.write_text("key")

        server = MCPServerTask("remote")
        captured: dict = {}

        class DummyAsyncClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class DummyTransportCtx:
            async def __aenter__(self):
                return MagicMock(), MagicMock(), (lambda: None)

            async def __aexit__(self, *a):
                return False

        class DummySession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def initialize(self):
                return None

        async def _discover_tools(self):
            self._shutdown_event.set()

        async def _drive():
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
                 patch("tools.mcp_tool._MCP_NEW_HTTP", True), \
                 patch("httpx.AsyncClient", DummyAsyncClient), \
                 patch("tools.mcp_tool.streamable_http_client",
                       return_value=DummyTransportCtx()), \
                 patch("tools.mcp_tool.ClientSession", DummySession), \
                 patch.object(MCPServerTask, "_discover_tools", _discover_tools):
                await server._run_http({
                    "url": "https://example.com/mcp",
                    "client_cert": [str(cert), str(key)],
                })

        asyncio.run(_drive())
        assert captured.get("cert") == (str(cert), str(key))

    def test_no_cert_means_no_cert_kwarg(self):
        """When client_cert is unset, ``cert`` is not passed to ``httpx.AsyncClient``
        (matches SDK defaults)."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")
        captured: dict = {}

        class DummyAsyncClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class DummyTransportCtx:
            async def __aenter__(self):
                return MagicMock(), MagicMock(), (lambda: None)

            async def __aexit__(self, *a):
                return False

        class DummySession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def initialize(self):
                return None

        async def _discover_tools(self):
            self._shutdown_event.set()

        async def _drive():
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
                 patch("tools.mcp_tool._MCP_NEW_HTTP", True), \
                 patch("httpx.AsyncClient", DummyAsyncClient), \
                 patch("tools.mcp_tool.streamable_http_client",
                       return_value=DummyTransportCtx()), \
                 patch("tools.mcp_tool.ClientSession", DummySession), \
                 patch.object(MCPServerTask, "_discover_tools", _discover_tools):
                await server._run_http({"url": "https://example.com/mcp"})

        asyncio.run(_drive())
        assert "cert" not in captured

    def test_missing_cert_file_surfaces_clear_error(self, tmp_path):
        """A missing cert file fails fast with a server-scoped error message."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")

        async def _drive():
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
                 patch("tools.mcp_tool._MCP_NEW_HTTP", True):
                await server._run_http({
                    "url": "https://example.com/mcp",
                    "client_cert": str(tmp_path / "nope.pem"),
                })

        with pytest.raises(FileNotFoundError, match=r"remote.*client_cert.*not found"):
            asyncio.run(_drive())


# ---------------------------------------------------------------------------
# SSE transport — cert + verify routed via httpx_client_factory
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_sse_client():
    """Replace ``sse_client`` with a MagicMock that records its kwargs.

    Returns the captured kwargs dict so tests can assert how ``_run_http``
    called it.
    """
    captured_kwargs: dict = {}

    class _FakeStream:
        def __init__(self):
            self._read = AsyncMock()
            self._write = AsyncMock()

        async def __aenter__(self):
            return (self._read, self._write)

        async def __aexit__(self, *a):
            return False

    def fake_sse_client(**kwargs):
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        return _FakeStream()

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            mock_session = MagicMock()
            mock_session.initialize = AsyncMock()
            return mock_session

        async def __aexit__(self, *a):
            return False

    with patch("tools.mcp_tool.sse_client", new=fake_sse_client), \
         patch("tools.mcp_tool.ClientSession", new=_FakeSession):
        yield captured_kwargs


class TestSSEClientCert:
    def test_no_factory_when_defaults(self, patch_sse_client):
        """With no cert and ssl_verify=True (default), the SDK's own factory is
        used — we don't inject one."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("sse-test")
        server._auth_type = ""
        server._sampling = None

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event",
                              new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()):
                try:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp/sse",
                            "transport": "sse",
                        }),
                        timeout=2.0,
                    )
                except (asyncio.TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())
        assert "httpx_client_factory" not in patch_sse_client

    def test_factory_injected_when_secret_headers_set(self, patch_sse_client):
        """With user-configured headers (potential secrets) and default TLS, a
        factory IS injected so the redirect scrubber can strip them on a
        cross-origin redirect. The auto-seeded mcp-protocol-version header
        alone must NOT trigger this (covered by test_no_factory_when_defaults),
        and the scrubber must only strip the configured keys."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("sse-test")
        server._auth_type = ""
        server._sampling = None

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event",
                              new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()):
                try:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp/sse",
                            "transport": "sse",
                            "headers": {"X-API-Key": "sekrit"},
                        }),
                        timeout=2.0,
                    )
                except (asyncio.TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())
        factory = patch_sse_client.get("httpx_client_factory")
        assert factory is not None, (
            "expected httpx_client_factory when secret headers are configured"
        )

        # The factory's client must carry a *request* hook that scrubs secret
        # headers. It has to be a request hook, not a response hook: under
        # follow_redirects=True httpx never populates response.next_request, so
        # a response hook can't touch the request that is actually sent. The
        # end-to-end scrubbing behaviour is exercised against httpx's real
        # redirect machinery in TestRedirectSecretScrubber below.
        client = factory(headers=None, timeout=None, auth=None)
        request_hooks = client.event_hooks.get("request") or []
        assert request_hooks, "expected a redirect-scrubbing request hook"
        assert not (client.event_hooks.get("response") or []), (
            "scrubber must not be installed as a response hook — it would be a "
            "no-op on followed redirects"
        )

    def test_factory_injected_when_cert_set(self, patch_sse_client, tmp_path):
        """With client_cert set, an httpx_client_factory is injected that
        applies the cert (and follow_redirects=True to match the SDK)."""
        from tools.mcp_tool import MCPServerTask

        cert = tmp_path / "client.pem"
        cert.write_text("dummy")

        server = MCPServerTask("sse-test")
        server._auth_type = ""
        server._sampling = None

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event",
                              new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()):
                try:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp/sse",
                            "transport": "sse",
                            "client_cert": str(cert),
                        }),
                        timeout=2.0,
                    )
                except (asyncio.TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())

        factory = patch_sse_client.get("httpx_client_factory")
        assert factory is not None, "expected httpx_client_factory to be injected"

        # Invoke the factory the way the SDK would; capture the resulting
        # httpx.AsyncClient kwargs.
        captured_client_kwargs: dict = {}

        class DummyAsyncClient:
            def __init__(self, **kwargs):
                captured_client_kwargs.update(kwargs)

        import httpx
        with patch.object(httpx, "AsyncClient", DummyAsyncClient):
            factory(headers={"x": "y"}, timeout=httpx.Timeout(30.0), auth=None)

        assert captured_client_kwargs["cert"] == str(cert)
        assert captured_client_kwargs["verify"] is True
        assert captured_client_kwargs["follow_redirects"] is True
        assert captured_client_kwargs["headers"] == {"x": "y"}

    def test_factory_forwards_custom_ca_bundle(self, patch_sse_client, tmp_path):
        """ssl_verify as a path is forwarded to the factory's httpx client."""
        from tools.mcp_tool import MCPServerTask

        ca_bundle = tmp_path / "ca.pem"
        ca_bundle.write_text("dummy")

        server = MCPServerTask("sse-test")
        server._auth_type = ""
        server._sampling = None

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event",
                              new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()):
                try:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp/sse",
                            "transport": "sse",
                            "ssl_verify": str(ca_bundle),
                        }),
                        timeout=2.0,
                    )
                except (asyncio.TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())

        factory = patch_sse_client.get("httpx_client_factory")
        assert factory is not None

        captured_client_kwargs: dict = {}

        class DummyAsyncClient:
            def __init__(self, **kwargs):
                captured_client_kwargs.update(kwargs)

        import httpx
        with patch.object(httpx, "AsyncClient", DummyAsyncClient):
            factory(headers=None, timeout=None, auth=None)

        assert captured_client_kwargs["verify"] == str(ca_bundle)
        assert "cert" not in captured_client_kwargs


# ---------------------------------------------------------------------------
# Cross-origin secret-header scrubber — real httpx redirect lifecycle
# ---------------------------------------------------------------------------


class TestRedirectSecretScrubber:
    """End-to-end coverage for ``_make_redirect_secret_scrubber``.

    These drive httpx's real redirect-following machinery through a
    ``MockTransport`` and observe the headers that are *actually sent* at each
    hop, instead of hand-calling the hook with a synthetic ``next_request``.
    Because httpx invokes ``response`` hooks before it builds the followed
    redirect request (and only sets ``response.next_request`` in the
    non-following branch), a response-hook implementation is a silent no-op
    here; the scrubber must be a request hook to strip the followed hop.
    """

    def _drive(self, original_url, secret_keys, headers, redirect_to):
        import httpx

        from tools.mcp_tool import _make_redirect_secret_scrubber

        sent = []  # headers actually seen by the transport, per hop

        def handler(request: "httpx.Request") -> "httpx.Response":
            sent.append((str(request.url), dict(request.headers)))
            if str(request.url) == original_url:
                return httpx.Response(302, headers={"location": redirect_to})
            return httpx.Response(200, text="landed")

        scrubber = _make_redirect_secret_scrubber(original_url, secret_keys)

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                follow_redirects=True,
                event_hooks={"request": [scrubber]},
                headers=headers,
            ) as client:
                resp = await client.get(original_url)
                assert resp.status_code == 200

        asyncio.run(run())
        return sent

    def test_strips_secret_on_followed_cross_origin_redirect(self):
        sent = self._drive(
            "https://example.com/mcp",
            {"x-api-key"},
            {"X-API-Key": "sekrit"},
            "https://evil.example.net/landing",
        )
        # Two real hops: original request, then the followed redirect.
        assert len(sent) == 2
        first_url, first_headers = sent[0]
        assert first_url == "https://example.com/mcp"
        # Same origin -> secret preserved on the initial request.
        assert first_headers.get("x-api-key") == "sekrit"
        second_url, second_headers = sent[1]
        assert second_url == "https://evil.example.net/landing"
        # Foreign origin reached via a followed redirect -> secret stripped.
        assert "x-api-key" not in second_headers

    def test_strips_secret_on_same_host_different_port(self):
        sent = self._drive(
            "https://example.com/mcp",
            {"x-api-key"},
            {"X-API-Key": "sekrit"},
            "https://example.com:8443/landing",
        )
        assert len(sent) == 2
        second_url, second_headers = sent[1]
        assert second_url == "https://example.com:8443/landing"
        assert "x-api-key" not in second_headers

    def test_preserves_secret_on_same_origin_redirect(self):
        sent = self._drive(
            "https://example.com/mcp",
            {"x-api-key"},
            {"X-API-Key": "sekrit"},
            "https://example.com/mcp/v2",
        )
        assert len(sent) == 2
        second_url, second_headers = sent[1]
        assert second_url == "https://example.com/mcp/v2"
        # Same origin (scheme, host, default port) -> secret must survive.
        assert second_headers.get("x-api-key") == "sekrit"

    def test_only_configured_keys_are_stripped(self):
        sent = self._drive(
            "https://example.com/mcp",
            {"x-api-key"},
            {"X-API-Key": "sekrit", "mcp-protocol-version": "2025-11-25"},
            "https://evil.example.net/landing",
        )
        _, second_headers = sent[1]
        assert "x-api-key" not in second_headers
        # A non-secret protocol header that wasn't in the configured key set is
        # left untouched.
        assert second_headers.get("mcp-protocol-version") == "2025-11-25"

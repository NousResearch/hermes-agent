"""Pre-flight OAuth discovery must honour the RFC 9728 WWW-Authenticate challenge.

``build_protected_resource_metadata_discovery_urls(www_auth_url, server_url)`` documents
its priority order as challenge → path-based well-known → root-based well-known. The
pre-flight path passed a hardcoded ``None``, permanently skipping priority 1.

That is invisible against servers whose metadata sits at the domain root, and fatal
against servers that host it under the MCP path only: Interactive Brokers advertises
``resource_metadata="https://api.ibkr.com/v1/api/mcp-public/.well-known/oauth-protected-resource"``
(200, ``authorization_servers: ["https://api.ibkr.com"]``) while both well-known
fallbacks answer 404, so pre-flight could never learn the authorization server and every
derived URL (ASM, registration, authorize) pointed somewhere the server does not serve.

The advertised URL is a *preference*, never a replacement. Most of these tests pin that
second half: an advertised URL that is wrong, malformed, hostile, or on a streaming
endpoint must degrade to today's behaviour — never to a hang or a foreign origin.
"""
from __future__ import annotations

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


SERVER_URL = "https://mcp.example.com/v1/mcp"
ADVERTISED = "https://mcp.example.com/v1/mcp/.well-known/oauth-protected-resource"
PATH_WELLKNOWN = "https://mcp.example.com/.well-known/oauth-protected-resource/v1/mcp"
PRM_BODY = {"resource": SERVER_URL, "authorization_servers": ["https://auth.example.com"]}


def _provider(tmp_path, server_url=SERVER_URL):
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None
    reset_manager_for_tests()
    return _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url=server_url,
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=object(),  # non-HermesTokenStorage → _hermes_storage() is None
    ), httpx


def _resp(httpx, url, status, body=None, headers=None):
    return httpx.Response(status, json=body if body is not None else {},
                          headers=headers or {}, request=httpx.Request("GET", url))


def _install(httpx, monkeypatch, handler):
    """Swap AsyncClient for one whose send() defers to *handler(request, **kwargs)*."""
    class _Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def send(self, request, **kwargs):
            return await handler(request, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


def _install_real_client(httpx, monkeypatch, handler):
    """Keep the real AsyncClient, routing it through *handler(request)* via MockTransport.

    Needed where the behaviour under test lives in httpx itself — whether ``send`` reads
    the response body — which a stubbed client cannot reproduce.
    """
    _real = httpx.AsyncClient

    class _Mocked(_real):
        def __init__(self, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Mocked)


@pytest.mark.asyncio
async def test_prefetch_adopts_the_advertised_authorization_server(tmp_path, monkeypatch):
    """The bug itself: the advertised PRM's authorization_servers must reach the context.

    Asserts on *state*, not on the request log — a fetch that is then discarded would pass
    a log-based assertion while discovery still failed.
    """
    provider, httpx = _provider(tmp_path)
    seen: list[str] = []

    async def handler(request, **kwargs):
        url = str(request.url)
        seen.append(url)
        if url == SERVER_URL:
            return _resp(httpx, url, 401, headers={
                "WWW-Authenticate": f'Bearer resource_metadata="{ADVERTISED}"'})
        if url == ADVERTISED:
            return _resp(httpx, url, 200, PRM_BODY)
        return _resp(httpx, url, 404)

    _install(httpx, monkeypatch, handler)
    await provider._prefetch_oauth_metadata()

    assert seen[0] == SERVER_URL, "the challenge probe must come first"
    assert ADVERTISED in seen, "the advertised metadata URL was never fetched"
    assert provider.context.protected_resource_metadata is not None
    assert provider.context.auth_server_url == "https://auth.example.com"


@pytest.mark.asyncio
async def test_prefetch_without_challenge_keeps_wellknown_behaviour(tmp_path, monkeypatch):
    """A server that needs no auth probes 200 → no challenge → the fallbacks resolve it."""
    provider, httpx = _provider(tmp_path)
    seen: list[str] = []

    async def handler(request, **kwargs):
        url = str(request.url)
        seen.append(url)
        if url == SERVER_URL:
            return _resp(httpx, url, 200)
        if url in (ADVERTISED, PATH_WELLKNOWN):
            return _resp(httpx, url, 200, PRM_BODY)
        return _resp(httpx, url, 404)

    _install(httpx, monkeypatch, handler)
    await provider._prefetch_oauth_metadata()

    assert seen[0] == SERVER_URL
    assert provider.context.auth_server_url == "https://auth.example.com"


@pytest.mark.asyncio
async def test_advertised_url_that_404s_falls_through_to_fallbacks(tmp_path, monkeypatch):
    """A stale advertised URL must not become a new single point of failure."""
    provider, httpx = _provider(tmp_path)
    seen: list[str] = []

    async def handler(request, **kwargs):
        url = str(request.url)
        seen.append(url)
        if url == SERVER_URL:
            return _resp(httpx, url, 401, headers={
                "WWW-Authenticate": f'Bearer resource_metadata="{ADVERTISED}"'})
        if url == PATH_WELLKNOWN:
            return _resp(httpx, url, 200, PRM_BODY)
        return _resp(httpx, url, 404)

    _install(httpx, monkeypatch, handler)
    await provider._prefetch_oauth_metadata()

    assert ADVERTISED in seen
    assert PATH_WELLKNOWN in seen, "the well-known fallback was never tried"
    assert provider.context.auth_server_url == "https://auth.example.com"


@pytest.mark.asyncio
async def test_malformed_advertised_url_does_not_abort_discovery(tmp_path, monkeypatch):
    """A server advertising garbage must not cost us the well-known fallbacks.

    ``httpx2.InvalidURL`` is a plain ``Exception``, not an ``HTTPError``. Without a broad
    catch in ``_send`` this one candidate kills the whole prefetch — reproducing exactly
    the symptom the issue reports (no metadata → refresh 404 → browser prompt).
    """
    from mcp.client.auth.utils import create_oauth_metadata_request

    provider, httpx = _provider(tmp_path)
    seen: list[str] = []

    async def handler(request, **kwargs):
        url = str(request.url)
        seen.append(url)
        if url == SERVER_URL:
            return _resp(httpx, url, 401, headers={
                "WWW-Authenticate": 'Bearer resource_metadata="https://h:notaport/x"'})
        if url == PATH_WELLKNOWN:
            return _resp(httpx, url, 200, PRM_BODY)
        return _resp(httpx, url, 404)

    _install(httpx, monkeypatch, handler)

    # Pin that the real builder really does reject that advertised value, so this test
    # exercises the production failure rather than a stub's imagination.
    with pytest.raises(Exception) as excinfo:
        create_oauth_metadata_request("https://h:notaport/x")
    assert not isinstance(excinfo.value, httpx.HTTPError)

    await provider._prefetch_oauth_metadata()

    assert len(seen) > 2, "discovery must continue past the malformed candidate"
    assert provider.context.auth_server_url == "https://auth.example.com"


@pytest.mark.asyncio
async def test_cross_origin_prm_is_rejected(tmp_path, monkeypatch):
    """A PRM naming someone else's resource must not steer the authorization server.

    Before honouring an advertised URL, every candidate was derived from ``server_url``,
    so a cross-origin PRM was unreachable. The 401 carrying it is unauthenticated, so the
    SDK's RFC 8707 check — used by its own 401 branch and by the device flow — has to
    apply to pre-flight too.
    """
    hostile = {"resource": "https://attacker.invalid/other",
               "authorization_servers": ["https://attacker.invalid"]}
    provider, httpx = _provider(tmp_path)

    async def handler(request, **kwargs):
        url = str(request.url)
        if url == SERVER_URL:
            return _resp(httpx, url, 401, headers={
                "WWW-Authenticate": f'Bearer resource_metadata="{ADVERTISED}"'})
        if url == ADVERTISED:
            return _resp(httpx, url, 200, hostile)
        return _resp(httpx, url, 404)

    _install(httpx, monkeypatch, handler)
    await provider._prefetch_oauth_metadata()

    assert provider.context.protected_resource_metadata is None
    assert provider.context.auth_server_url != "https://attacker.invalid"


@pytest.mark.asyncio
async def test_probe_does_not_hang_on_a_streaming_endpoint(tmp_path, monkeypatch):
    """GET on a streamable-HTTP server answers ``text/event-stream`` and never ends.

    A normal ``send`` awaits a body that never terminates, and httpx's read timeout is
    *between chunks*, so the keepalives reset it — the client timeout never fires. The
    probe therefore must not read the body, or it hangs provider start-up, which runs
    under ``context.lock``.

    Uses a real ``httpx.AsyncClient`` over a MockTransport: a stubbed client would never
    read the body at all, so it could not tell ``stream=True`` from ``stream=False``.
    """
    import asyncio

    provider, httpx = _provider(tmp_path)

    async def handler(request):
        if str(request.url) == SERVER_URL:
            async def endless():
                while True:
                    await asyncio.sleep(0.01)
                    yield b"event: ping\ndata: {}\n\n"

            return httpx.Response(200, content=endless(),
                                  headers={"content-type": "text/event-stream"},
                                  request=request)
        return httpx.Response(404, json={}, request=request)

    _install_real_client(httpx, monkeypatch, handler)

    # Reaching this line at all is the assertion: with a body-reading probe this never
    # returns and wait_for raises TimeoutError.
    await asyncio.wait_for(provider._prefetch_oauth_metadata(), timeout=10.0)
    assert provider.context.protected_resource_metadata is None


@pytest.mark.asyncio
async def test_probe_failure_leaves_discovery_intact(tmp_path, monkeypatch):
    """A transport error on the probe degrades to today's behaviour, silently."""
    provider, httpx = _provider(tmp_path)
    seen: list[str] = []

    async def handler(request, **kwargs):
        url = str(request.url)
        seen.append(url)
        if url == SERVER_URL:
            raise httpx.ConnectError("probe failed")
        if url in (ADVERTISED, PATH_WELLKNOWN):
            return _resp(httpx, url, 200, PRM_BODY)
        return _resp(httpx, url, 404)

    _install(httpx, monkeypatch, handler)
    await provider._prefetch_oauth_metadata()

    assert len(seen) > 1, "the probe failure must not stop discovery"
    assert provider.context.auth_server_url == "https://auth.example.com"


def test_challenge_outranks_wellknown_but_does_not_replace_them():
    """The SDK contract this fix relies on — the fallbacks stay behind the advertised URL."""
    from mcp.client.auth.utils import build_protected_resource_metadata_discovery_urls

    with_challenge = build_protected_resource_metadata_discovery_urls(ADVERTISED, SERVER_URL)
    assert with_challenge[0] == ADVERTISED
    assert len(with_challenge) > 1
    assert any("/.well-known/oauth-protected-resource" in u for u in with_challenge[1:])

    assert build_protected_resource_metadata_discovery_urls(None, SERVER_URL) == [
        PATH_WELLKNOWN,
        "https://mcp.example.com/.well-known/oauth-protected-resource",
    ]
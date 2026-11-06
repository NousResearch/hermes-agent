"""Regression tests for the two OAuth auth-flow defects behind hung MCP discovery.

Streamable HTTP keeps one session-long ``GET /mcp`` SSE stream open for the life of
the connection while ``tools/list`` POSTs run alongside it.  Both defects wedge that
shape, and both leave the same fingerprint: ``initialize`` answers in milliseconds,
the ``GET`` goes out, and the next POST never returns.

1. ``HermesProviderMixin.async_auth_flow`` sniffed EVERY 200 response for
   origin-issued authorization-server metadata, and that sniff buffers the whole
   body (``response.aread()``).  Run against the MCP resource response it reads the
   SSE stream that never ends — blocking inside the auth flow with
   ``context.lock`` held, so every later request queues behind it forever.
2. ``HermesMCPOAuthProvider.async_auth_flow`` releases ``context.lock`` (a binary
   semaphore) around the resource request and re-acquires it afterwards.  If the
   flow is abandoned while suspended (cancellation at a timeout, transport
   teardown, HTTPX closing the flow), the SDK generator is never resumed, its
   ``async with self.context.lock`` never exits, and the permit is stranded.

``tests/tools/test_mcp_oauth_bidirectional.py`` drives these generators by hand and
never closes or cancels a pending flow, so it saw neither.  These tests drive the
real installed SDK/httpx2 ``AsyncClient`` path instead, over a mock transport: no
network, no credentials, every wait bounded by ``asyncio.wait_for``.
"""
from __future__ import annotations

import asyncio

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

SERVER_URL = "https://example.com/mcp"


async def _build_provider(tmp_path, monkeypatch):
    """A HermesMCPOAuthProvider holding valid cached tokens (no flow is needed)."""
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None, "SDK OAuth types must be available"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(OAuthToken(access_token="access-token", token_type="Bearer",
                                        expires_in=3600, refresh_token="refresh-token"))
    await storage.set_client_info(OAuthClientInformationFull(
        client_id="test-client",
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    ))
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url=SERVER_URL,
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )
    # Cached tokens with no cached metadata would make ``_initialize`` run pre-flight
    # discovery against the real well-known endpoints; these tests stay offline.
    monkeypatch.setattr(type(provider), "_prefetch_oauth_metadata", _no_discovery)
    return provider


async def _no_discovery(self) -> None:
    """Stand-in for pre-flight metadata discovery (no network in tests)."""


def _sse_transport(httpx, opened: asyncio.Event, release: asyncio.Event):
    """Mock transport: ``GET`` is a session-long SSE stream, ``POST`` answers at once."""

    class _PendingSSE(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b": open\n\n"
            opened.set()
            await release.wait()  # the stream stays open, as a real MCP session's does

        async def aclose(self) -> None:
            release.set()

    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=_PendingSSE())
        return httpx.Response(202, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_session_long_sse_get_does_not_wedge_later_requests(tmp_path, monkeypatch):
    """A never-ending SSE ``GET`` must not block the auth flow or the next POST."""
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()

    provider = await _build_provider(tmp_path, monkeypatch)
    opened, release = asyncio.Event(), asyncio.Event()

    async with httpx.AsyncClient(transport=_sse_transport(httpx, opened, release),
                                 auth=provider, timeout=httpx.Timeout(5.0)) as client:
        # Streamable HTTP's session-long GET: the auth flow must hand the response back
        # on its headers, never buffering a body that only ends with the session.
        get_response = await asyncio.wait_for(
            client.send(client.build_request("GET", SERVER_URL), stream=True), timeout=5.0)
        assert get_response.status_code == 200
        assert get_response.request.headers["authorization"] == "Bearer access-token"

        try:
            # Before the fix this blocked forever behind the GET's body read, which
            # held ``context.lock`` — the symptom behind the discovery timeout.
            post_response = await asyncio.wait_for(
                client.post(SERVER_URL, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
                timeout=5.0)
        except TimeoutError:  # pragma: no cover — the pre-fix failure mode
            pytest.fail("POST behind the session-long SSE GET blocked: the auth flow is "
                        "holding the OAuth state lock across the GET's open stream")
        assert post_response.status_code == 202
        assert post_response.request.headers["authorization"] == "Bearer access-token"
        # The permit is back: OAuth transitions can still serialize afterwards.
        assert provider.context.lock.value == 1

        release.set()
        await get_response.aclose()


@pytest.mark.asyncio
async def test_cancelled_resource_request_releases_the_state_lock(tmp_path, monkeypatch):
    """Cancelling mid-request must not strand the lock either (cleanup stays balanced)."""
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()

    provider = await _build_provider(tmp_path, monkeypatch)
    started, never = asyncio.Event(), asyncio.Event()

    async def handler(request):
        started.set()
        await never.wait()  # cancelled below; set only at teardown
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=provider,
                                 timeout=httpx.Timeout(30.0)) as client:
        task = asyncio.create_task(client.post(SERVER_URL, json={}))
        await asyncio.wait_for(started.wait(), timeout=5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.context.lock.value == 1

        # ... and the connection is still usable: a follow-up POST authenticates.
        async def ok_handler(request):
            return httpx.Response(200, json={})

        client._transport = httpx.MockTransport(ok_handler)
        response = await asyncio.wait_for(client.post(SERVER_URL, json={}), timeout=5.0)
        assert response.status_code == 200
    never.set()


async def _noop_redirect(_url: str) -> None:
    """Redirect handler that does nothing (no browser flow runs in these tests)."""


async def _noop_callback() -> tuple[str, str | None]:
    """Callback handler that must never be invoked here."""
    raise AssertionError("callback handler should not be invoked with valid cached tokens")

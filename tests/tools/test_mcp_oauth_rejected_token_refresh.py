"""Rejected resource tokens refresh silently before requiring browser consent."""
from __future__ import annotations

import asyncio
import contextlib
import sys
from urllib.parse import parse_qs

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

RESOURCE = "https://resource.example.com/mcp"
ISSUER = "https://auth.example.com"
TOKEN_ENDPOINT = f"{ISSUER}/token"


@pytest.fixture
def oauth_provider(tmp_path, monkeypatch):
    from mcp.shared.auth import (
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthMetadata,
        OAuthToken,
    )

    from pydantic import AnyHttpUrl, AnyUrl

    from tools.mcp_oauth import HermesTokenStorage, OAuthNonInteractiveError
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider, reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    async def build():
        storage = HermesTokenStorage("rejected-token")
        storage.bind_issuer(ISSUER)
        await storage.set_tokens(OAuthToken(
            access_token="access-one", refresh_token="refresh-one",
            token_type="Bearer", expires_in=3600, scope="read",
        ))
        await storage.set_client_info(OAuthClientInformationFull(
            client_id="synthetic-client", issuer=ISSUER,
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"], token_endpoint_auth_method="none",
        ))
        storage.save_oauth_metadata(OAuthMetadata(
            issuer=AnyHttpUrl(ISSUER), authorization_endpoint=AnyHttpUrl(f"{ISSUER}/authorize"),
            token_endpoint=AnyHttpUrl(TOKEN_ENDPOINT), registration_endpoint=AnyHttpUrl(f"{ISSUER}/register"),
            response_types_supported=["code"],
        ))
        redirects = []

        async def redirect(url):
            redirects.append(url)
            raise OAuthNonInteractiveError("synthetic background authorization")

        async def callback():
            raise AssertionError("background refresh must not need a browser callback")

        provider = HermesMCPOAuthProvider(
            server_name="rejected-token", server_url=RESOURCE,
            client_metadata=OAuthClientMetadata(
                redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")], scope="read",
            ), storage=storage, redirect_handler=redirect, callback_handler=callback,
        )
        await provider._initialize()
        return provider, storage, redirects

    return build


def _httpx():
    from tools.mcp_tool import sdk_httpx
    module = sdk_httpx()
    assert module is not None
    return module


def _response(request, status=401, **kwargs):
    return _httpx().Response(status, request=request, **kwargs)


async def _start(provider):
    request = _httpx().Request("POST", RESOURCE)
    flow = provider.async_auth_flow(request)
    outbound = await flow.__anext__()
    assert outbound is request
    assert outbound.headers["authorization"] == "Bearer access-one"
    return flow, outbound


async def _refresh(flow, outbound, *, rotate=True):
    refresh = await flow.asend(_response(outbound))
    assert refresh.method == "POST"
    assert str(refresh.url) == TOKEN_ENDPOINT
    form = parse_qs(refresh.content.decode())
    assert form["grant_type"] == ["refresh_token"]
    assert form["refresh_token"] == ["refresh-one"]
    body = {"access_token": "access-two", "token_type": "Bearer", "expires_in": 3600}
    if rotate:
        body["refresh_token"] = "refresh-two"
    retry = await flow.asend(_response(refresh, 200, json=body))
    assert retry is outbound
    assert retry.headers["authorization"] == "Bearer access-two"
    return retry


@contextlib.asynccontextmanager
async def _nothing_left_to_a_finalizer():
    """Hold every async generator started in the block. The event loop finalizes an abandoned generator
    within a turn or two, which frees what it holds and hides a missing ``aclose``. Held, a lock that is
    free inside the block was freed by the code under test."""
    started, hooks = [], sys.get_asyncgen_hooks()

    def hold(agen):
        started.append(agen)
        if hooks.firstiter is not None:
            hooks.firstiter(agen)

    sys.set_asyncgen_hooks(firstiter=hold, finalizer=hooks.finalizer)
    try:
        yield started
    finally:
        sys.set_asyncgen_hooks(*hooks)
        for agen in reversed(started):
            await agen.aclose()


async def _free_now(provider, storage):
    """Whether the SDK's context lock and the refresh fence can be taken, the lock without any wait."""
    import anyio

    from tools.mcp_oauth import RefreshFenceTimeout, acquire_refresh_fence, release_refresh_fence

    lock = provider.context.lock
    try:
        lock.acquire_nowait()
    except anyio.WouldBlock:
        lock_free = False
    else:
        lock.release()
        lock_free = True
    try:
        release_refresh_fence(await acquire_refresh_fence(storage._tokens_path(), timeout=2.0))
    except RefreshFenceTimeout:
        return {"context lock": lock_free, "refresh fence": False}
    return {"context lock": lock_free, "refresh fence": True}


async def _suspended(provider, at):
    """A flow stopped at one of its requests: the resource request, the refresh that answers its 401, the
    resource request sent again, or the first discovery request after that one was rejected as well."""
    flow, outbound = await _start(provider)
    if at == "refresh":
        refresh = await flow.asend(_response(outbound))
        assert str(refresh.url) == TOKEN_ENDPOINT
    elif at in ("retry", "discovery"):
        retry = await _refresh(flow, outbound)
        if at == "discovery":
            discovery = await flow.asend(_response(retry))
            assert "/.well-known/" in str(discovery.url)
    return flow


# The access token of the request that follows a flow left at each of those points. A flow left during
# its refresh had already marked the rejected token expired, so the next request refreshes first.
_TOKEN_AFTER = {"resource": "access-one", "refresh": "access-three", "retry": "access-two", "discovery": "access-two"}


async def _accepted_token(provider):
    """The access token of one more request, sent to a server that accepts whatever it gets."""
    request = _httpx().Request("POST", RESOURCE)
    flow = provider.async_auth_flow(request)
    try:
        outbound = await flow.__anext__()
        if str(outbound.url) == TOKEN_ENDPOINT:
            outbound = await flow.asend(_response(outbound, 200, json={
                "access_token": "access-three", "refresh_token": "refresh-three",
                "token_type": "Bearer", "expires_in": 3600,
            }))
        assert outbound is request
        with pytest.raises(StopAsyncIteration):
            await flow.asend(_response(outbound, 200))
        return outbound.headers["authorization"].removeprefix("Bearer ")
    finally:
        await flow.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("rotate", [True, False])
async def test_early_resource_401_refreshes_without_browser(oauth_provider, rotate):
    provider, storage, redirects = await oauth_provider()
    assert provider.context.is_token_valid()
    flow, outbound = await _start(provider)
    try:
        retry = await _refresh(flow, outbound, rotate=rotate)
        with pytest.raises(StopAsyncIteration):
            await flow.asend(_response(retry, 200))
        saved = await storage.get_tokens()
        assert saved.access_token == "access-two"
        assert saved.refresh_token == ("refresh-two" if rotate else "refresh-one")
        assert saved.scope == "read"
        assert redirects == []
    finally:
        await flow.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["refresh", "client", "metadata", "issuer"])
async def test_ineligible_rejected_token_keeps_existing_authorization_path(oauth_provider, missing):
    provider, storage, redirects = await oauth_provider()
    if missing == "refresh":
        provider.context.current_tokens.refresh_token = None
    elif missing == "client":
        provider.context.client_info = None
    elif missing == "metadata":
        provider.context.oauth_metadata = None
    else:
        storage.loaded_issuer = "https://different-issuer.example.com"
    flow, outbound = await _start(provider)
    try:
        discovery = await flow.asend(_response(outbound))
        assert discovery.method == "GET"
        assert "/.well-known/" in str(discovery.url)
        assert redirects == []
    finally:
        await flow.aclose()


@pytest.mark.asyncio
async def test_plain_403_never_spends_refresh_token(oauth_provider):
    provider, storage, redirects = await oauth_provider()
    flow, outbound = await _start(provider)
    try:
        with pytest.raises(StopAsyncIteration):
            await flow.asend(_response(outbound, 403))
        assert (await storage.get_tokens()).refresh_token == "refresh-one"
        assert redirects == []
    finally:
        await flow.aclose()


@pytest.mark.asyncio
async def test_insufficient_scope_keeps_consent_and_scope_union(oauth_provider):
    from tools.mcp_oauth import OAuthNonInteractiveError

    provider, storage, redirects = await oauth_provider()
    flow, outbound = await _start(provider)
    try:
        with pytest.raises(OAuthNonInteractiveError):
            await flow.asend(_response(outbound, 403, headers={
                "www-authenticate": 'Bearer error="insufficient_scope", scope="write"',
            }))
        assert len(redirects) == 1
        assert set(provider.context.client_metadata.scope.split()) == {"read", "write"}
        assert (await storage.get_tokens()).refresh_token == "refresh-one"
    finally:
        await flow.aclose()


@pytest.mark.asyncio
async def test_second_resource_401_does_not_loop_refresh(oauth_provider):
    provider, storage, redirects = await oauth_provider()
    flow, outbound = await _start(provider)
    try:
        retry = await _refresh(flow, outbound)
        discovery = await flow.asend(_response(retry))
        assert discovery.method == "GET"
        assert "/.well-known/" in str(discovery.url)
        assert (await storage.get_tokens()).refresh_token == "refresh-two"
        assert redirects == []
    finally:
        await flow.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("leave", ["close", RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("at", list(_TOKEN_AFTER))
async def test_leaving_the_flow_frees_both_locks_at_once(oauth_provider, at, leave):
    """HTTPX closes the flow after a transport error and when its task is cancelled; an exception thrown
    into the flow is covered too. Either way the SDK flow is closed by then, not later by a finalizer."""
    provider, storage, redirects = await oauth_provider()
    async with _nothing_left_to_a_finalizer() as started:
        async with asyncio.timeout(10):  # a pass that left the SDK flow open stalls the next one
            flow = await _suspended(provider, at)
        if leave == "close":
            await flow.aclose()
        else:
            with pytest.raises(leave, match="synthetic interruption"):
                await flow.athrow(leave("synthetic interruption"))
        assert await _free_now(provider, storage) == {"context lock": True, "refresh fence": True}
        assert [agen for agen in started if agen.ag_frame is not None] == []
        assert await _accepted_token(provider) == _TOKEN_AFTER[at]
    assert redirects == []


@pytest.mark.asyncio
async def test_concurrent_stale_401_reuses_single_refresh(oauth_provider):
    provider, storage, redirects = await oauth_provider()
    first, first_request = await _start(provider)
    second, second_request = await _start(provider)
    try:
        first_retry = await _refresh(first, first_request)
        second_retry = await second.asend(_response(second_request))
        assert second_retry is second_request
        assert second_retry.headers["authorization"] == "Bearer access-two"
        # The second flow sends its request again without the lock, or the first one could not finish.
        assert await _free_now(provider, storage) == {"context lock": True, "refresh fence": True}
        for flow, retry in ((first, first_retry), (second, second_retry)):
            with pytest.raises(StopAsyncIteration):
                await flow.asend(_response(retry, 200))
        assert (await storage.get_tokens()).refresh_token == "refresh-two"
        assert redirects == []
    finally:
        # Together: a flow stopped in an OAuth step holds the lock that closing the other one waits for.
        await asyncio.gather(first.aclose(), second.aclose())


@pytest.mark.asyncio
@pytest.mark.parametrize("body,status", [({"error": "invalid_grant"}, 400), ({}, 200)])
async def test_failed_refresh_falls_back_without_retry_loop(oauth_provider, body, status):
    provider, _, _ = await oauth_provider()
    flow, outbound = await _start(provider)
    try:
        refresh = await flow.asend(_response(outbound))
        assert str(refresh.url) == TOKEN_ENDPOINT
        retry = await flow.asend(_response(refresh, status, json=body))
        assert retry is outbound
        discovery = await flow.asend(_response(retry))
        assert discovery.method == "GET"
        assert "/.well-known/" in str(discovery.url)
    finally:
        await flow.aclose()


@pytest.mark.asyncio
async def test_token_endpoint_invalid_client_is_not_resource_recovery(oauth_provider):
    provider, storage, _ = await oauth_provider()
    flow, outbound = await _start(provider)
    try:
        refresh = await flow.asend(_response(outbound))
        assert str(refresh.url) == TOKEN_ENDPOINT
        retry = await flow.asend(_response(refresh, 401, json={"error": "invalid_client"}))
        assert retry is outbound
        assert provider.context.client_info is None
        assert not storage._client_info_path().exists()
        assert storage._client_info_path().with_name("rejected-token.client.json.bak").exists()
    finally:
        await flow.aclose()


@pytest.mark.asyncio
async def test_httpx_client_drives_silent_resource_recovery(oauth_provider):
    provider, storage, redirects = await oauth_provider()
    httpx = _httpx()
    seen = []

    async def respond(request):
        seen.append((request.method, str(request.url)))
        if str(request.url) == TOKEN_ENDPOINT:
            assert parse_qs(request.content.decode())["refresh_token"] == ["refresh-one"]
            return _response(request, 200, json={
                "access_token": "access-two", "refresh_token": "refresh-two",
                "token_type": "Bearer", "expires_in": 3600,
            })
        assert str(request.url) == RESOURCE
        if request.headers["authorization"] == "Bearer access-one":
            return _response(request)
        assert request.headers["authorization"] == "Bearer access-two"
        return _response(request, 200, json={"ok": True})

    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        response = await client.post(RESOURCE, json={"synthetic": True})
    assert response.json() == {"ok": True}
    assert seen == [("POST", RESOURCE), ("POST", TOKEN_ENDPOINT), ("POST", RESOURCE)]
    assert (await storage.get_tokens()).refresh_token == "refresh-two"
    assert redirects == []


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_status", [200, 400])
async def test_token_refreshed_in_the_same_pass_is_not_refreshed_again(oauth_provider, refresh_status):
    """A 401 after the SDK's own refresh attempt (the stored token had expired by our clock) means the
    refreshed token, or the request sent without one, was rejected: no second refresh, the SDK decides."""
    import time

    provider, _, redirects = await oauth_provider()
    provider.context.token_expiry_time = time.time() - 1
    request = _httpx().Request("POST", RESOURCE)
    flow = provider.async_auth_flow(request)
    try:
        refresh = await flow.__anext__()
        assert str(refresh.url) == TOKEN_ENDPOINT
        body = ({"access_token": "access-two", "refresh_token": "refresh-two", "token_type": "Bearer",
                 "expires_in": 3600} if refresh_status == 200 else {"error": "invalid_grant"})
        outbound = await flow.asend(_response(refresh, refresh_status, json=body))
        assert outbound is request
        discovery = await flow.asend(_response(outbound))
        assert discovery.method == "GET"
        assert "/.well-known/" in str(discovery.url)
        assert redirects == []
    finally:
        await flow.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("at", ["resource", "refresh", "retry"])
async def test_request_cancelled_inside_httpx_frees_both_locks_at_once(oauth_provider, at):
    """The real driver: HTTPX closes the flow from the cancelled task, where every wait that is not
    shielded is cancelled again."""
    import anyio

    provider, storage, redirects = await oauth_provider()
    httpx = _httpx()
    held, seen = asyncio.Event(), []

    async def respond(request):
        seen.append(str(request.url))
        if len(seen) == ["resource", "refresh", "retry"].index(at) + 1:
            held.set()
            await asyncio.Event().wait()  # until the request is cancelled
        if str(request.url) == TOKEN_ENDPOINT:
            return _response(request, 200, json={
                "access_token": "access-two", "refresh_token": "refresh-two",
                "token_type": "Bearer", "expires_in": 3600,
            })
        return _response(request)

    async with _nothing_left_to_a_finalizer() as started:
        async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
            async with anyio.create_task_group() as requests:
                requests.start_soon(client.post, RESOURCE)
                async with asyncio.timeout(10):  # as above: an SDK flow left open stalls the next pass
                    await held.wait()
                requests.cancel_scope.cancel()
        assert await _free_now(provider, storage) == {"context lock": True, "refresh fence": True}
        assert [agen for agen in started if agen.ag_frame is not None] == []
        assert await _accepted_token(provider) == _TOKEN_AFTER[at]
    assert redirects == []


@pytest.mark.asyncio
async def test_token_reloaded_for_the_second_pass_and_rejected_too_goes_to_authorization(oauth_provider, monkeypatch):
    """Another process may store a newer token between the passes, so the second pass sends that one without
    refreshing. If the server rejects it as well, the SDK's 401 branch runs: one silent refresh per request."""
    import time

    from mcp.shared.auth import OAuthToken

    from tools.mcp_oauth_manager import MCPOAuthManager

    provider, _, redirects = await oauth_provider()
    passes = []

    async def reload_before_second_pass(self, server_name, *, hermes_home=None):
        passes.append(server_name)
        if len(passes) == 2:
            provider.context.current_tokens = OAuthToken(
                access_token="access-other", refresh_token="refresh-other", token_type="Bearer",
                expires_in=3600, scope="read",
            )
            provider.context.token_expiry_time = time.time() + 3600
        return False

    monkeypatch.setattr(MCPOAuthManager, "invalidate_if_disk_changed", reload_before_second_pass)
    flow, outbound = await _start(provider)
    try:
        retry = await flow.asend(_response(outbound))
        assert retry is outbound
        assert retry.headers["authorization"] == "Bearer access-other"
        discovery = await flow.asend(_response(retry))
        assert discovery.method == "GET"
        assert "/.well-known/" in str(discovery.url)
        assert redirects == []
    finally:
        await flow.aclose()

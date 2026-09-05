"""Regression test for the ``HermesMCPOAuthProvider.async_auth_flow`` bidirectional
generator bridge.

PR #11383 introduced a subclass method that wrapped the SDK's ``auth_flow`` with::

    async for item in super().async_auth_flow(request):
        yield item

``httpx``'s auth_flow contract is a **bidirectional** async generator — the
driving code (``httpx._client._send_handling_auth``) does::

    next_request = await auth_flow.asend(response)

to feed HTTP responses back into the generator. The naive ``async for ...``
wrapper discards those ``.asend(response)`` values and resumes the inner
generator with ``None``, so the SDK's ``response = yield request`` branch in
``mcp/client/auth/oauth2.py`` sees ``response = None`` and crashes at
``if response.status_code == 401`` with
``AttributeError: 'NoneType' object has no attribute 'status_code'``.

This broke every OAuth MCP server on the first HTTP response regardless of
status code. The reason nothing caught it in CI: zero existing tests drive
the full ``.asend()`` round-trip — the integration tests in
``test_mcp_oauth_integration.py`` stop at ``_initialize()`` and disk-watching.

These tests drive the wrapper through a manual ``.asend()`` sequence to prove
the bridge forwards responses correctly into the inner SDK generator.
"""
from __future__ import annotations

import asyncio

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


@pytest.mark.asyncio
async def test_hermes_provider_forwards_asend_values(tmp_path, monkeypatch):
    """The wrapper MUST forward ``.asend(response)`` into the inner generator.

    This is the primary regression test. With the broken wrapper, the inner
    SDK generator sees ``response = None`` and raises ``AttributeError`` at
    ``oauth2.py:505``. With the correct bridge, a 200 response finishes the
    flow cleanly (``StopAsyncIteration``).
    """
    # The SDK's httpx flavour (httpx2 on mcp >= 2.0): the provider is an
    # Auth subclass from that module and its auth_flow only accepts its own
    # Request/Response types.
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    from mcp.shared.auth import OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None, "SDK OAuth types must be available"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    # Seed a valid-looking token so the SDK's _initialize loads something and
    # can_refresh_token() is True (though we don't exercise refresh here — we
    # go straight through the 200 path).
    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="old_access",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="old_refresh",
        )
    )
    # Also seed client_info so the SDK doesn't attempt registration.
    from mcp.shared.auth import OAuthClientInformationFull

    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)

    # First anext() drives the wrapper + inner generator until the inner
    # yields the outbound request (at oauth2.py:503 ``response = yield request``).
    outbound = await flow.__anext__()
    assert outbound is not None, "wrapper must yield the outbound request"
    assert outbound.url.host == "example.com"

    # Simulate httpx returning a 200 response.
    fake_response = httpx.Response(200, request=outbound)

    # The broken wrapper would crash here with AttributeError: 'NoneType'
    # object has no attribute 'status_code', because the SDK's inner generator
    # resumes with response=None and dereferences .status_code at line 505.
    #
    # The correct wrapper forwards the response, the SDK takes the non-401
    # non-403 exit, and the generator ends cleanly (StopAsyncIteration).
    with pytest.raises(StopAsyncIteration):
        await flow.asend(fake_response)


@pytest.mark.asyncio
async def test_hermes_provider_forwards_401_triggers_refresh(tmp_path, monkeypatch):
    """A 401 response MUST flow into the inner generator and trigger the
    SDK's 401 recovery branch.

    With the broken wrapper, the inner generator sees ``response = None``
    and the 401 check short-circuits into AttributeError. With the correct
    bridge, the 401 is routed into the SDK's ``response.status_code == 401``
    branch which begins discovery (yielding a metadata-discovery request).
    """
    # The SDK's httpx flavour (httpx2 on mcp >= 2.0): the provider is an
    # Auth subclass from that module and its auth_flow only accepts its own
    # Request/Response types.
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="old_access",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="old_refresh",
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        client_name="Hermes Agent",
    )
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)

    # Drive to the first yield (outbound MCP request).
    outbound = await flow.__anext__()

    # Reply with a 401 including a minimal WWW-Authenticate so the SDK's
    # 401 branch can parse resource metadata from it. We just need something
    # the SDK accepts before it tries to yield the metadata-discovery request.
    fake_401 = httpx.Response(
        401,
        request=outbound,
        headers={"www-authenticate": 'Bearer resource_metadata="https://example.com/.well-known/oauth-protected-resource"'},
    )

    # The correct bridge forwards the 401 into the SDK; the SDK then yields
    # its NEXT request (a metadata-discovery GET). We assert we get a request
    # back — any request. The broken bridge would have crashed with
    # AttributeError before we ever reach this point.
    next_request = await flow.asend(fake_401)
    assert isinstance(next_request, httpx.Request), (
        "wrapper must forward .asend() so the SDK's 401 branch can yield the "
        "next request in the discovery flow"
    )

    # Clean up the generator — we don't need to complete the full dance.
    await flow.aclose()


@pytest.mark.asyncio
async def test_long_lived_resource_request_does_not_block_concurrent_post(
    tmp_path, monkeypatch
):
    """A session-long MCP GET must not hold the provider state lock.

    MCP SDK 2.0.0 wraps its entire auth-flow generator in one lock.  Leaving
    the GET response pending then prevents a concurrent POST from even
    acquiring its Bearer token.  Hermes narrows that lock around resource I/O
    while retaining it for OAuth state transitions.
    """
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(
            access_token="access-token",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="refresh-token",
        )
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    get_request = httpx.Request("GET", "https://example.com/mcp")
    get_flow = provider.async_auth_flow(get_request)
    get_outbound = await get_flow.__anext__()

    # Keep the GET open, as streamable HTTP does for the session lifetime.
    # The POST must still authenticate and reach HTTPX without waiting for it.
    post_request = httpx.Request("POST", "https://example.com/mcp")
    post_flow = provider.async_auth_flow(post_request)
    post_outbound = await asyncio.wait_for(post_flow.__anext__(), timeout=2.0)

    assert post_outbound is post_request
    assert post_outbound.headers["authorization"] == "Bearer access-token"

    with pytest.raises(StopAsyncIteration):
        await post_flow.asend(httpx.Response(200, request=post_outbound))

    # A completed concurrent refresh must make the pending GET retry with the
    # new token rather than start a second OAuth transition from its stale 401.
    provider.context.current_tokens = OAuthToken(
        access_token="new-access-token",
        token_type="Bearer",
        expires_in=3600,
        refresh_token="refresh-token",
    )
    provider.context.update_token_expiry(provider.context.current_tokens)
    get_retry = await get_flow.asend(
        httpx.Response(
            401,
            request=get_outbound,
            headers={
                "www-authenticate": (
                    'Bearer resource_metadata="https://example.com/'
                    '.well-known/oauth-protected-resource"'
                )
            },
        )
    )
    assert get_retry is get_request
    assert get_retry.headers["authorization"] == "Bearer new-access-token"

    with pytest.raises(StopAsyncIteration):
        await get_flow.asend(httpx.Response(200, request=get_retry))


@pytest.mark.asyncio
async def test_tokenless_long_lived_request_does_not_block_concurrent_post(
    tmp_path, monkeypatch
):
    """A token-less session-long GET must not hold the provider state lock.

    Regression for the login double-attempt "fix" attempt: gating the lock release on
    ``sent_access_token is not None`` held ``context.lock`` across a token-less request's
    entire network round-trip. On an OAuth server that serves its SSE GET without demanding
    auth (e.g. Google Drive — acknowledged in hermes_cli/mcp_config.py), the GET never 401s,
    the lock is never returned, and every concurrent tool-call POST hangs for the session.

    The lock MUST be released for every resource request, authenticated or not. OAuth
    transitions (discovery/registration/authorization) are still serialized: the bridge
    re-acquires the lock before feeding the response back to the SDK, and the stale-401
    retry guard (mcp_oauth_manager.py) makes a second token-less flow reuse the first flow's
    freshly minted token instead of re-entering authorization.
    """
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    # No token, no client_info: both flows start token-less.

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    get_request = httpx.Request("GET", "https://example.com/mcp")
    get_flow = provider.async_auth_flow(get_request)
    get_outbound = await get_flow.__anext__()
    assert get_outbound is get_request
    assert "authorization" not in get_outbound.headers  # token-less: no Bearer yet

    # Keep the token-less GET pending, as streamable HTTP does for the session lifetime.
    # The concurrent POST must still reach its first yield (acquire the lock) without
    # waiting for the GET to 401/authorize/complete.
    post_request = httpx.Request("POST", "https://example.com/mcp")
    post_flow = provider.async_auth_flow(post_request)
    post_outbound = await asyncio.wait_for(post_flow.__anext__(), timeout=2.0)

    assert post_outbound is post_request
    assert "authorization" not in post_outbound.headers

    # Complete both flows cleanly (a 200 short-circuits the SDK's 401/403 branches).
    with pytest.raises(StopAsyncIteration):
        await post_flow.asend(httpx.Response(200, request=post_outbound))
    with pytest.raises(StopAsyncIteration):
        await get_flow.asend(httpx.Response(200, request=get_outbound))


async def _noop_redirect(_url: str) -> None:
    """Redirect handler that does nothing (won't be invoked in these tests)."""
    return None


@pytest.mark.asyncio
async def test_concurrent_authorization_across_providers_triggers_single_authorization(
    tmp_path, monkeypatch
):
    """Two concurrent token-less flows across separate providers must authorize ONCE.

    The login double-attempt bug: `hermes mcp login asana` printed TWO
    "MCP OAuth: authorization required" blocks (distinct ``state`` / ``code_challenge`` each) and
    the second flow died with ``OSError: [Errno 98] Address already in use``.

    Two flows on ONE provider already serialize to a single authorization via the SDK's
    ``context.lock`` + the stale-401 retry guard (covered by the neighbouring test). The live
    double-trigger requires SEPARATE contexts/locks — a duplicate or rebuilt provider for the same
    endpoint — so each flow carries its own ``context.lock`` and the SDK cannot serialize them. Both
    then reach the authorization step, print two URLs, and collide on the resolved callback port.

    The fix is a per-server authorization mutex (``MCPOAuthManager.authorization_lock``) held across
    the redirect + callback + token exchange: the loser blocks, then adopts the winner's token from
    shared storage and retries instead of re-authorizing.
    """
    from urllib.parse import parse_qs, urlparse

    from mcp.shared.auth import AuthorizationCodeResult, OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()

    assert _HERMES_PROVIDER_CLS is not None

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    # Both providers point at the SAME server and the SAME shared token storage, exactly as a
    # duplicate/rebuilt provider for one endpoint would: separate contexts/locks, one token file.
    storage = HermesTokenStorage("srv")
    redirect_calls: list[str] = []
    states: dict = {}

    def make_handlers(tag: str):
        async def redirect_handler(authorization_url: str) -> None:
            redirect_calls.append(authorization_url)
            states[tag] = parse_qs(urlparse(authorization_url).query).get("state", [None])[0]

        async def callback_handler():
            await asyncio.sleep(0.05)
            return AuthorizationCodeResult(
                code="FAKE_AUTH_CODE", state=states.get(tag), iss=None
            )

        return redirect_handler, callback_handler

    def make_provider(tag: str):
        redirect_handler, callback_handler = make_handlers(tag)
        return _HERMES_PROVIDER_CLS(
            server_name="srv",
            server_url="https://example.com/mcp",
            client_metadata=OAuthClientMetadata(
                redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
                client_name="Hermes Agent",
            ),
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

    provider_a = make_provider("a")
    provider_b = make_provider("b")

    WWW = 'Bearer resource_metadata="https://example.com/.well-known/oauth-protected-resource"'
    PRM = {"resource": "https://example.com/mcp", "authorization_servers": ["https://idp.example.com"]}
    ASM = {
        "issuer": "https://idp.example.com",
        "authorization_endpoint": "https://idp.example.com/authorize",
        "token_endpoint": "https://idp.example.com/token",
        "registration_endpoint": "https://idp.example.com/register",
        "response_types_supported": ["code"],
    }
    DCR = {
        "client_id": "client-1",
        "redirect_uris": ["http://127.0.0.1:12345/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    TOKEN = {"access_token": "tok", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "ref"}

    async def drive_one_flow(provider) -> None:
        req = httpx.Request("POST", "https://example.com/mcp")
        flow = provider.async_auth_flow(req)
        outbound = await flow.__anext__()

        # A flow that already carries a token completes cleanly and never re-authorizes.
        if "authorization" in outbound.headers:
            with pytest.raises(StopAsyncIteration):
                await flow.asend(httpx.Response(200, request=outbound))
            return

        # Token-less: 401 -> discovery -> registration -> (authorization OR adopted-token retry).
        nxt = await flow.asend(httpx.Response(401, request=outbound, headers={"www-authenticate": WWW}))
        nxt = await flow.asend(httpx.Response(200, request=nxt, json=PRM))
        nxt = await flow.asend(httpx.Response(200, request=nxt, json=ASM))
        nxt = await flow.asend(httpx.Response(200, request=nxt, json=DCR))

        # The winner yields the token request; the loser, blocked on the authorization mutex, is
        # diverted to the adopted-token retry (a request that already carries an Authorization
        # header) instead of a second token exchange.
        if "authorization" in nxt.headers:
            with pytest.raises(StopAsyncIteration):
                await flow.asend(httpx.Response(200, request=nxt))
            return

        # Winner: token exchange, then the SDK retries the resource request with the new token.
        nxt = await flow.asend(httpx.Response(200, request=nxt, json=TOKEN))
        assert "authorization" in nxt.headers
        with pytest.raises(StopAsyncIteration):
            await flow.asend(httpx.Response(200, request=nxt))

    await asyncio.gather(drive_one_flow(provider_a), drive_one_flow(provider_b))

    assert len(redirect_calls) == 1, (
        f"expected exactly one authorization flow (one browser URL), got {len(redirect_calls)}"
    )


@pytest.mark.asyncio
async def test_401_rejected_token_still_triggers_authorization(tmp_path, monkeypatch):
    """A 401 with an unexpired-but-rejected token must re-authorize, not re-adopt the same token.

    Regression for the adoption guard in ``_hermes_adopt_external_authorization``: it must adopt
    only a token that DIFFERS from the one this flow already holds. If it adopted the rejected
    token itself, a 401 (or 403 step-up) would retry the SAME bad ``Bearer`` forever and never
    reach the authorization step (zero redirects) — the login would silently spin instead of
    re-prompting the browser.
    """
    from urllib.parse import parse_qs, urlparse

    from mcp.shared.auth import (
        AuthorizationCodeResult,
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    assert _HERMES_PROVIDER_CLS is not None

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(access_token="STALE", token_type="Bearer", expires_in=3600, refresh_token="ref")
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    redirect_calls: list[str] = []
    captured_state: dict = {}

    async def redirect_handler(authorization_url: str) -> None:
        redirect_calls.append(authorization_url)
        captured_state["state"] = parse_qs(urlparse(authorization_url).query).get("state", [None])[0]

    async def callback_handler():
        await asyncio.sleep(0.05)
        return AuthorizationCodeResult(code="FAKE_AUTH_CODE", state=captured_state.get("state"), iss=None)

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )

    WWW = 'Bearer resource_metadata="https://example.com/.well-known/oauth-protected-resource"'
    PRM = {"resource": "https://example.com/mcp", "authorization_servers": ["https://idp.example.com"]}
    ASM = {
        "issuer": "https://idp.example.com",
        "authorization_endpoint": "https://idp.example.com/authorize",
        "token_endpoint": "https://idp.example.com/token",
        "response_types_supported": ["code"],
    }
    TOKEN = {"access_token": "tok", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "ref"}

    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)

    outbound = await flow.__anext__()
    assert "authorization" in outbound.headers  # sends the (rejected) STALE token

    # 401 -> PRM discovery -> ASM discovery -> (client already registered) -> authorization.
    nxt = await flow.asend(httpx.Response(401, request=outbound, headers={"www-authenticate": WWW}))
    nxt = await flow.asend(httpx.Response(200, request=nxt, json=PRM))
    nxt = await flow.asend(httpx.Response(200, request=nxt, json=ASM))

    assert len(redirect_calls) == 1, (
        f"expected the rejected token to trigger exactly one re-authorization, got {len(redirect_calls)}"
    )

    nxt = await flow.asend(httpx.Response(200, request=nxt, json=TOKEN))
    assert "authorization" in nxt.headers
    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200, request=nxt))


@pytest.mark.asyncio
async def test_403_insufficient_scope_step_up_still_triggers_authorization(tmp_path, monkeypatch):
    """A 403 ``insufficient_scope`` step-up must re-authorize, not re-adopt the same token.

    Same adoption guard as the 401 case, exercised through the SDK's 403 step-up branch (the
    second ``_perform_authorization`` call site). A step-up must produce a fresh authorization
    (one redirect), never a silent retry of the token that just lacked scope.
    """
    from urllib.parse import parse_qs, urlparse

    from mcp.shared.auth import (
        AuthorizationCodeResult,
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    assert _HERMES_PROVIDER_CLS is not None

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(access_token="STALE", token_type="Bearer", expires_in=3600, refresh_token="ref")
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )

    redirect_calls: list[str] = []
    captured_state: dict = {}

    async def redirect_handler(authorization_url: str) -> None:
        redirect_calls.append(authorization_url)
        captured_state["state"] = parse_qs(urlparse(authorization_url).query).get("state", [None])[0]

    async def callback_handler():
        await asyncio.sleep(0.05)
        return AuthorizationCodeResult(code="FAKE_AUTH_CODE", state=captured_state.get("state"), iss=None)

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )

    TOKEN = {"access_token": "tok", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "ref"}

    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)

    outbound = await flow.__anext__()
    assert "authorization" in outbound.headers  # sends the (scope-lacking) STALE token

    # 403 insufficient_scope -> scope union -> authorization (no discovery/registration).
    nxt = await flow.asend(
        httpx.Response(403, request=outbound, headers={"www-authenticate": 'Bearer error="insufficient_scope"'})
    )

    assert len(redirect_calls) == 1, (
        f"expected the scope step-up to trigger exactly one re-authorization, got {len(redirect_calls)}"
    )

    nxt = await flow.asend(httpx.Response(200, request=nxt, json=TOKEN))
    assert "authorization" in nxt.headers
    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200, request=nxt))


async def _noop_callback() -> tuple[str, str | None]:
    """Callback handler that won't be invoked in these tests."""
    raise AssertionError(
        "callback handler should not be invoked in bidirectional-generator tests"
    )

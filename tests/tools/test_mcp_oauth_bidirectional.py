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


async def _noop_redirect(_url: str) -> None:
    """Redirect handler that does nothing (won't be invoked in these tests)."""
    return None


async def _noop_callback() -> tuple[str, str | None]:
    """Callback handler that won't be invoked in these tests."""
    raise AssertionError(
        "callback handler should not be invoked in bidirectional-generator tests"
    )


@pytest.mark.asyncio
async def test_fabricated_dcr_registration_rejected_before_network(tmp_path, monkeypatch):
    """The bridge must raise instead of sending the SDK's *fabricated* DCR POST.

    Google's hosted Gmail/Drive MCP servers advertise no RFC 7591
    ``registration_endpoint`` in their ASM, so the SDK falls back to guessing
    ``POST {server_origin}/register`` (``create_client_registration_request``,
    mcp/client/auth/utils.py) — a URL those servers answer with an opaque 404
    (GH#78190). The guard in ``HermesMCPOAuthProvider`` must intercept that
    guessed request and raise an actionable ``OAuthRegistrationError`` before
    any network call, instead of letting the gateway burn the reconnect
    ladder on an unrecoverable registration failure.

    Drive the full discovery dance (401 -> PRM -> ASM without
    registration_endpoint -> fabricated registration POST) through the
    generator manually, exactly like the other tests in this file.
    """
    import httpx
    from mcp.client.auth import OAuthRegistrationError
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None, "SDK OAuth types must be available"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    # Empty storage: no tokens, no client_info — the SDK's 401 branch must
    # take the registration path (this is the cold gateway state for a
    # provider whose cached client is gone).
    storage = HermesTokenStorage("srv")

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

    # 1. Initial outbound MCP request.
    outbound = await flow.__anext__()
    assert outbound is not None

    # 2. 401 with a resource_metadata pointer -> SDK starts PRM discovery.
    fake_401 = httpx.Response(
        401,
        request=outbound,
        headers={
            "www-authenticate": (
                'Bearer resource_metadata="https://example.com/.well-known/'
                'oauth-protected-resource"'
            )
        },
    )
    prm_request = await flow.asend(fake_401)
    assert isinstance(prm_request, httpx.Request)
    assert prm_request.method == "GET"

    # 3. Serve PRM pointing at a fake authorization server.
    prm_response = httpx.Response(
        200,
        request=prm_request,
        headers={"Content-Type": "application/json"},
        json={
            "authorization_servers": ["https://auth.example.com"],
            "resource": "https://example.com/mcp",
            "bearer_methods_supported": ["header"],
        },
    )
    asm_request = await flow.asend(prm_response)
    assert isinstance(asm_request, httpx.Request)
    assert asm_request.method == "GET"

    # 4. Serve ASM with NO registration_endpoint (the Google Gmail/Drive
    #    MCP class of provider).
    asm_response = httpx.Response(
        200,
        request=asm_request,
        headers={"Content-Type": "application/json"},
        json={
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
        },
    )

    # 5. The SDK now wants to POST the fabricated registration URL — the
    #    bridge must raise the actionable error instead of yielding it.
    with pytest.raises(OAuthRegistrationError) as excinfo:
        await flow.asend(asm_response)
    text = str(excinfo.value)
    assert "does not support automatic client registration" in text
    assert "https://example.com/register" in text
    assert "config.yaml" in text
    assert "hermes mcp login srv" in text
    await flow.aclose()


@pytest.mark.asyncio
async def test_advertised_registration_endpoint_passes_through(tmp_path, monkeypatch):
    """A server that *advertises* a registration endpoint must NOT be blocked.

    Real-world case: Linear's MCP ASM advertises
    ``registration_endpoint: https://mcp.linear.app/register`` — the same
    origin+path the SDK would fabricate. The guard's AND condition (exact
    fabricated URL **and** no advertised registration_endpoint) must let
    such servers through; otherwise every legitimate DCR flow breaks.
    """
    import httpx
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None, "SDK OAuth types must be available"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
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

    outbound = await flow.__anext__()
    fake_401 = httpx.Response(
        401,
        request=outbound,
        headers={
            "www-authenticate": (
                'Bearer resource_metadata="https://example.com/.well-known/'
                'oauth-protected-resource"'
            )
        },
    )
    prm_request = await flow.asend(fake_401)
    prm_response = httpx.Response(
        200,
        request=prm_request,
        headers={"Content-Type": "application/json"},
        json={
            "authorization_servers": ["https://auth.example.com"],
            "resource": "https://example.com/mcp",
            "bearer_methods_supported": ["header"],
        },
    )
    asm_request = await flow.asend(prm_response)

    # ASM DOES advertise a registration endpoint — at origin+/register,
    # exactly the URL the SDK would otherwise fabricate (Linear's ASM does
    # this: https://mcp.linear.app/register).
    asm_response = httpx.Response(
        200,
        request=asm_request,
        headers={"Content-Type": "application/json"},
        json={
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "registration_endpoint": "https://example.com/register",
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
        },
    )

    # The registration POST must be yielded (not blocked): the SDK uses the
    # advertised endpoint and the guard must pass it through.
    registration_request = await flow.asend(asm_response)
    assert isinstance(registration_request, httpx.Request)
    assert registration_request.method == "POST"
    assert str(registration_request.url) == "https://example.com/register"

    # No OAuthRegistrationError may have been raised along the way. (The
    # dance ends here: a 200 registration response would hand control to the
    # browser-redirect flow, which is outside this test's scope.)
    await flow.aclose()


@pytest.mark.asyncio
async def test_guard_ignores_non_fabricated_post_paths(tmp_path, monkeypatch):
    """A POST to a *different* path must never be blocked by the guard.

    The guard is exact-URL matched: only the SDK's fabricated
    ``{origin}/register`` POST is rejected when the ASM lacks a
    registration_endpoint. A server-side registration at ``/foo/register``
    or any other path passes through untouched.
    """
    import httpx
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None, "SDK OAuth types must be available"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=HermesTokenStorage("srv"),
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )

    # No ASM discovered yet (oauth_metadata None) — the guard must still
    # ignore anything that is not the fabricated origin+/register POST.
    for url in (
        "https://example.com/foo/register",
        "https://example.com/register/",
        "https://example.com/registers",
        "https://other.example.com/register",
    ):
        await provider._maybe_reject_fabricated_registration(
            httpx.Request("POST", url)
        )  # must not raise

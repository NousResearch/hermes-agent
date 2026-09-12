"""SDK-built OAuth discovery/registration requests must carry a User-Agent.

The MCP SDK constructs ``/.well-known/...`` discovery and dynamic client
registration requests as bare ``httpx.Request`` objects inside
``async_auth_flow``. The client's default headers never apply to them, so
they leave with NO ``User-Agent`` at all. Some WAFs reject header-less
requests outright: coda.io answers 403 on every discovery and registration
call, which Hermes then misreports as "only allows pre-approved OAuth
clients". (``oauth.user_agent`` does not help — it is stamped only on
token-endpoint requests.)

``HermesMCPOAuthProvider`` now gives such requests a default User-Agent.
These tests drive the bridge through the 401 branch so the SDK yields a
real discovery request, then assert on its headers.
"""
from __future__ import annotations

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


async def _noop_redirect(url: str) -> None:
    pass


async def _noop_callback():
    return ("code", None)


async def _make_flow(tmp_path, monkeypatch):
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
        OAuthToken(access_token="old_access", token_type="Bearer", expires_in=3600, refresh_token="old_refresh")
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
    req = httpx.Request("POST", "https://example.com/mcp")
    return httpx, req, provider.async_auth_flow(req)


@pytest.mark.asyncio
async def test_discovery_request_gets_default_user_agent(tmp_path, monkeypatch):
    from tools.mcp_oauth_manager import DEFAULT_AUTH_REQUEST_USER_AGENT

    httpx, req, flow = await _make_flow(tmp_path, monkeypatch)
    outbound = await flow.__anext__()

    fake_401 = httpx.Response(
        401,
        request=outbound,
        headers={"www-authenticate": 'Bearer resource_metadata="https://example.com/.well-known/oauth-protected-resource"'},
    )
    discovery = await flow.asend(fake_401)

    assert discovery is not req, "SDK must have yielded its own discovery request"
    assert ".well-known" in str(discovery.url)
    assert discovery.headers.get("user-agent") == DEFAULT_AUTH_REQUEST_USER_AGENT

    await flow.aclose()


@pytest.mark.asyncio
async def test_callers_mcp_request_is_left_untouched(tmp_path, monkeypatch):
    """The bridge must not decorate the caller's own MCP request — those
    headers belong to the transport client, not the OAuth layer."""
    _httpx, req, flow = await _make_flow(tmp_path, monkeypatch)
    outbound = await flow.__anext__()

    assert outbound is req
    assert "user-agent" not in outbound.headers

    await flow.aclose()

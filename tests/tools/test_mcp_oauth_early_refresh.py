"""The early-refresh margin for MCP OAuth access tokens (``HermesMCPOAuthProvider``).

The SDK refreshes an access token only once its expiry has passed. An authorization server that
issues short tokens (Keycloak's default is 300 s) then lets a tool call start a second before
expiry, and an MCP server that re-introspects the token mid-request finds it inactive and refuses
with an application error, not a 401, so nothing retries. The provider now marks a refreshable
token as expired when it has at most the margin left, and the SDK's pre-request refresh branch
runs first.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


async def _noop_redirect(url: str) -> None:
    return None


async def _noop_callback():
    return ("code", None)


async def _provider(tmp_path, monkeypatch, *, expires_in, remaining=None, refresh_token="r", margin=None):
    """A provider over a stored token with the issuer's full ``expires_in`` lifetime; ``remaining`` moves the
    in-memory expiry to simulate elapsed time (the cap uses the lifetime, the check uses the remaining time)."""
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    storage = HermesTokenStorage("srv")
    await storage.set_tokens(OAuthToken(access_token="a", token_type="Bearer", expires_in=expires_in,
                                        refresh_token=refresh_token))
    await storage.set_client_info(OAuthClientInformationFull(
        client_id="test-client", redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        grant_types=["authorization_code", "refresh_token"], response_types=["code"],
        token_endpoint_auth_method="none"))
    kwargs = {} if margin is None else {"refresh_margin_seconds": margin}
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv", server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
                                            client_name="Hermes Agent"),
        storage=storage, redirect_handler=_noop_redirect, callback_handler=_noop_callback, **kwargs)

    async def _no_discovery() -> None:  # keep the unit test off the network
        return None

    monkeypatch.setattr(provider, "_prefetch_oauth_metadata", _no_discovery)
    await provider._initialize()
    if remaining is not None:
        provider.context.token_expiry_time = time.time() + remaining
    return provider


@pytest.mark.asyncio
async def test_token_inside_the_margin_is_marked_for_refresh(tmp_path, monkeypatch):
    provider = await _provider(tmp_path, monkeypatch, expires_in=300, remaining=90)
    assert provider.context.is_token_valid() and provider.context.can_refresh_token()

    assert provider._hermes_force_refresh_if_near_expiry() is True

    assert not provider.context.is_token_valid(), "the SDK must take its refresh branch on the next request"
    assert provider.context.can_refresh_token(), "the refresh grant stays in place"


@pytest.mark.asyncio
async def test_fresh_token_is_left_alone(tmp_path, monkeypatch):
    provider = await _provider(tmp_path, monkeypatch, expires_in=3600)
    expiry = provider.context.token_expiry_time

    assert provider._hermes_force_refresh_if_near_expiry() is False

    assert provider.context.is_token_valid()
    assert provider.context.token_expiry_time == expiry


@pytest.mark.asyncio
async def test_token_without_a_refresh_grant_is_never_forced(tmp_path, monkeypatch):
    provider = await _provider(tmp_path, monkeypatch, expires_in=300, remaining=90, refresh_token=None)
    assert not provider.context.can_refresh_token()

    assert provider._hermes_force_refresh_if_near_expiry() is False

    assert provider.context.is_token_valid(), "without a refresh grant an early expiry would only force a re-login"


@pytest.mark.asyncio
async def test_margin_is_capped_at_half_the_known_lifetime(tmp_path, monkeypatch):
    # A 100 s token with a 120 s margin would refresh on every request; the cap makes the margin 50 s.
    provider = await _provider(tmp_path, monkeypatch, expires_in=100, margin=120)
    assert provider._hermes_refresh_margin == 120.0

    assert provider._hermes_force_refresh_if_near_expiry() is False
    assert provider.context.is_token_valid()

    provider.context.token_expiry_time = time.time() + 40  # now inside the capped 50 s margin
    assert provider._hermes_force_refresh_if_near_expiry() is True


@pytest.mark.asyncio
async def test_cold_loaded_token_uses_its_remaining_time_as_the_lifetime(tmp_path, monkeypatch):
    # After a restart the storage rebases expires_in to the seconds left (90 here), so the cap is 45 s: the
    # token is not refreshed at 90 s left but is once inside 45 s. Weaker than after a refresh, never worse than before.
    provider = await _provider(tmp_path, monkeypatch, expires_in=90)
    assert provider._hermes_force_refresh_if_near_expiry() is False
    provider.context.token_expiry_time = time.time() + 30
    assert provider._hermes_force_refresh_if_near_expiry() is True


@pytest.mark.asyncio
async def test_zero_margin_disables_the_early_refresh(tmp_path, monkeypatch):
    provider = await _provider(tmp_path, monkeypatch, expires_in=300, remaining=90, margin=0)
    assert provider._hermes_refresh_margin == 0.0

    assert provider._hermes_force_refresh_if_near_expiry() is False
    assert provider.context.is_token_valid()


@pytest.mark.asyncio
async def test_margin_is_read_as_seconds_and_a_bad_value_is_refused(tmp_path, monkeypatch):
    """Fail loud where the config is read: the refusal names the key, nothing falls back silently."""
    provider = await _provider(tmp_path, monkeypatch, expires_in=3600, margin="45")
    assert provider._hermes_refresh_margin == 45.0
    for bad in ("soon", -5, True):
        with pytest.raises(ValueError, match="oauth.refresh_margin_seconds"):
            await _provider(tmp_path, monkeypatch, expires_in=3600, margin=bad)


@pytest.mark.asyncio
async def test_auth_flow_refreshes_before_sending_the_request(tmp_path, monkeypatch):
    """Driven through ``async_auth_flow``: a 300 s token with 90 s left makes the first outbound request the
    refresh POST, not the MCP request; with an hour left the MCP request goes out first with the current token."""
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    provider = await _provider(tmp_path, monkeypatch, expires_in=300, remaining=90)
    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)
    outbound = await flow.__anext__()
    try:
        assert outbound is not req
        assert outbound.method == "POST" and outbound.url.path.endswith("/token")
        assert b"grant_type=refresh_token" in outbound.content
    finally:
        await flow.aclose()

    provider = await _provider(tmp_path, monkeypatch, expires_in=3600)
    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)
    outbound = await flow.__anext__()
    try:
        assert outbound is req
        assert outbound.headers.get("Authorization") == "Bearer a"
    finally:
        await flow.aclose()

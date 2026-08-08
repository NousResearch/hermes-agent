"""Tests for pinning MCP OAuth endpoints per server (skip discovery).

GH#77684: allow overriding the OAuth endpoints for an MCP server in config
(``oauth.authorization_endpoint`` / ``token_endpoint`` /
``registration_endpoint`` / ``issuer``) instead of relying on RFC 8414
well-known discovery. This is the escape hatch for providers whose
discovery endpoints are rate-limited or broken (e.g. IBKR's hosted MCP 403s
the well-known endpoints under load, which previously left
``oauth_metadata=None`` and fell back to wrong ``{origin}/authorize`` /
``{origin}/token`` URLs).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


# ---------------------------------------------------------------------------
# build_pinned_oauth_metadata
# ---------------------------------------------------------------------------


def test_pinned_metadata_none_when_not_pinned():
    """No pin keys -> None (normal discovery path)."""
    from tools.mcp_oauth import build_pinned_oauth_metadata

    assert (
        build_pinned_oauth_metadata({}, server_url="https://mcp.example/mcp")
        is None
    )
    assert (
        build_pinned_oauth_metadata(
            {"client_id": "abc", "redirect_port": 0},
            server_url="https://mcp.example/mcp",
        )
        is None
    )


def test_pinned_metadata_builds_full_object():
    """Both required endpoints + optional registration/issuer are honored."""
    from tools.mcp_oauth import build_pinned_oauth_metadata

    meta = build_pinned_oauth_metadata(
        {
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "registration_endpoint": "https://auth.example.com/register",
        },
        server_url="https://mcp.example/mcp",
    )
    assert meta is not None
    # AnyHttpUrl normalizes to a trailing slash for bare origins.
    assert str(meta.issuer).rstrip("/") == "https://auth.example.com"
    assert str(meta.authorization_endpoint) == "https://auth.example.com/authorize"
    assert str(meta.token_endpoint) == "https://auth.example.com/token"
    assert str(meta.registration_endpoint) == "https://auth.example.com/register"


def test_pinned_metadata_issuer_defaults_to_server_origin():
    """issuer is optional and defaults to the server URL origin."""
    from tools.mcp_oauth import build_pinned_oauth_metadata

    meta = build_pinned_oauth_metadata(
        {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        },
        server_url="https://mcp.example.com/mcp",
    )
    assert meta is not None
    assert str(meta.issuer).rstrip("/") == "https://mcp.example.com"
    assert meta.registration_endpoint is None


@pytest.mark.parametrize(
    "cfg",
    [
        {"authorization_endpoint": "https://auth.example.com/authorize"},
        {"token_endpoint": "https://auth.example.com/token"},
        {"issuer": "https://auth.example.com"},
    ],
)
def test_pinned_metadata_partial_pin_raises(cfg):
    """Pinning without BOTH authorization_endpoint and token_endpoint fails fast."""
    from tools.mcp_oauth import build_pinned_oauth_metadata

    with pytest.raises(ValueError, match="authorization_endpoint"):
        build_pinned_oauth_metadata(cfg, server_url="https://mcp.example/mcp")


def test_pinned_metadata_requires_issuer_source():
    """No issuer and no server_url -> clear error."""
    from tools.mcp_oauth import build_pinned_oauth_metadata

    with pytest.raises(ValueError, match="issuer"):
        build_pinned_oauth_metadata(
            {
                "authorization_endpoint": "https://auth.example.com/authorize",
                "token_endpoint": "https://auth.example.com/token",
            },
            server_url=None,
        )


# ---------------------------------------------------------------------------
# Provider wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_initializes_with_pinned_metadata(tmp_path, monkeypatch):
    """get_or_build_provider + _initialize pre-populates context.oauth_metadata
    from the pinned config and skips pre-flight discovery."""
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    reset_manager_for_tests()

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider(
        "srv",
        "https://mcp.example.com/mcp",
        {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        },
    )
    assert provider is not None
    assert provider._hermes_pinned_metadata is not None

    await provider._initialize()

    assert provider.context.oauth_metadata is not None
    assert (
        str(provider.context.oauth_metadata.token_endpoint)
        == "https://auth.example.com/token"
    )
    assert (
        str(provider.context.oauth_metadata.authorization_endpoint)
        == "https://auth.example.com/authorize"
    )


@pytest.mark.asyncio
async def test_manager_partial_pin_raises(tmp_path, monkeypatch):
    """A partial pin surfaces as a clear ValueError through the manager."""
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    reset_manager_for_tests()

    mgr = MCPOAuthManager()
    with pytest.raises(ValueError, match="authorization_endpoint"):
        mgr.get_or_build_provider(
            "srv",
            "https://mcp.example.com/mcp",
            {"token_endpoint": "https://auth.example.com/token"},
        )


# ---------------------------------------------------------------------------
# Discovery is skipped on the 401 branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pinned_flow_skips_well_known_discovery(tmp_path, monkeypatch):
    """On a 401, the flow must NOT yield well-known discovery requests to the
    caller — they are intercepted and answered with synthetic 404s, so the
    pinned metadata survives and the flow proceeds straight to the token
    exchange against the pinned token_endpoint."""
    import httpx
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
    # Seed client_info so the SDK skips dynamic client registration and goes
    # straight from discovery to authorization.
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

    from tools.mcp_oauth import build_pinned_oauth_metadata

    pinned = build_pinned_oauth_metadata(
        {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        },
        server_url="https://mcp.example.com/mcp",
    )
    assert pinned is not None

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://mcp.example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
        oauth_metadata=pinned,
    )

    # The authorization step would build a browser URL and wait for a
    # callback; replace it so the flow yields the token-exchange request
    # we can observe.
    async def fake_perform_authorization() -> httpx.Request:
        return httpx.Request("POST", "https://auth.example.com/token")

    provider._perform_authorization = fake_perform_authorization  # type: ignore[method-assign]

    req = httpx.Request("POST", "https://mcp.example.com/mcp")
    flow = provider.async_auth_flow(req)

    # First yield: the outbound MCP request itself.
    outbound = await flow.__anext__()
    assert outbound.url.host == "mcp.example.com"

    # Reply with a 401 (no WWW-Authenticate resource_metadata, so the SDK
    # falls back to the well-known discovery URLs).
    fake_401 = httpx.Response(401, request=outbound)

    # The next yielded request must be the token exchange against the
    # PINNED token endpoint — never a well-known discovery GET.
    next_request = await flow.asend(fake_401)
    assert isinstance(next_request, httpx.Request)
    assert str(next_request.url) == "https://auth.example.com/token", (
        "flow should skip well-known discovery and go straight to the pinned "
        f"token endpoint, got {next_request.url}"
    )
    assert ".well-known" not in str(next_request.url)

    # Pinned metadata survived the 401 branch untouched.
    assert provider.context.oauth_metadata is not None
    assert (
        str(provider.context.oauth_metadata.token_endpoint)
        == "https://auth.example.com/token"
    )

    await flow.aclose()


@pytest.mark.asyncio
async def test_unpinned_flow_still_yields_discovery(tmp_path, monkeypatch):
    """Without a pin, the 401 branch must still yield the well-known discovery
    request to the caller — pinning must not change default behaviour."""
    import httpx
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
        server_url="https://mcp.example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )
    assert provider._hermes_pinned_metadata is None

    req = httpx.Request("POST", "https://mcp.example.com/mcp")
    flow = provider.async_auth_flow(req)

    outbound = await flow.__anext__()
    fake_401 = httpx.Response(401, request=outbound)

    # Without a pin, the SDK's 401 branch yields the PRM well-known GET.
    next_request = await flow.asend(fake_401)
    assert isinstance(next_request, httpx.Request)
    assert ".well-known/oauth-protected-resource" in str(next_request.url)

    await flow.aclose()


async def _noop_redirect(_url: str) -> None:
    """Redirect handler that does nothing (won't be invoked in these tests)."""
    return None


async def _noop_callback() -> tuple[str, str | None]:
    """Callback handler that won't be invoked in these tests."""
    raise AssertionError(
        "callback handler should not be invoked in bidirectional-generator tests"
    )

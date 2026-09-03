"""Servers that answer initialize/tools-list without auth (Google Workspace MCP)
never return 401, so the SDK's reactive OAuth flow never starts. In an
interactive login the provider must run the authorization flow anyway when no
token exists on disk.
"""
from __future__ import annotations

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


async def _noop_redirect(_url: str) -> None:
    return None


async def _noop_callback() -> tuple[str, str | None]:
    raise AssertionError("callback handler should not be invoked in this test")


async def _make_provider(tmp_path, monkeypatch):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
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
    return _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
    )


@pytest.mark.asyncio
async def test_interactive_no_token_200_starts_authorization(tmp_path, monkeypatch):
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_oauth import force_interactive_oauth

    httpx = sdk_httpx()
    provider = await _make_provider(tmp_path, monkeypatch)

    with force_interactive_oauth():
        flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
        outbound = await flow.__anext__()
        assert "authorization" not in outbound.headers

        next_request = await flow.asend(httpx.Response(200, request=outbound))
        assert isinstance(next_request, httpx.Request)
        assert "oauth-protected-resource" in str(next_request.url)
        await flow.aclose()


@pytest.mark.asyncio
async def test_non_interactive_no_token_200_passes_through(tmp_path, monkeypatch):
    from tools.mcp_tool import sdk_httpx
    from tools import mcp_oauth

    httpx = sdk_httpx()
    provider = await _make_provider(tmp_path, monkeypatch)
    monkeypatch.setattr(mcp_oauth, "_is_interactive", lambda: False)

    flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    outbound = await flow.__anext__()
    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx.Response(200, request=outbound))


class _AuthorizeReached(Exception):
    pass


async def _stop_callback() -> tuple[str, str | None]:
    raise _AuthorizeReached()


@pytest.mark.asyncio
async def test_trailing_slash_authorization_server_matches_issuer(tmp_path, monkeypatch):
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_oauth import force_interactive_oauth

    httpx = sdk_httpx()
    provider = await _make_provider(tmp_path, monkeypatch)
    provider.context.callback_handler = _stop_callback

    prm = {
        "resource": "https://example.com/mcp",
        "authorization_servers": ["https://accounts.google.com/"],
    }
    asm = {
        "issuer": "https://accounts.google.com",
        "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "response_types_supported": ["code"],
        "code_challenge_methods_supported": ["S256"],
    }
    with force_interactive_oauth():
        flow = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
        outbound = await flow.__anext__()
        prm_req = await flow.asend(httpx.Response(200, request=outbound))
        asm_req = await flow.asend(httpx.Response(200, request=prm_req, json=prm))
        assert "oauth-authorization-server" in str(asm_req.url)
        with pytest.raises(_AuthorizeReached):
            await flow.asend(httpx.Response(200, request=asm_req, json=asm))
    assert provider.context.auth_server_url == "https://accounts.google.com"

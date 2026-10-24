"""A refresh triggered by a new session's ``initialize`` must name the RFC 8707 ``resource``.

The SDK adds ``resource`` to a refresh only when protected-resource metadata is known or the
triggering request carries ``MCP-Protocol-Version``. ``initialize`` never carries that header,
and a cold-loaded provider restores authorization-server metadata from disk without the
protected-resource document. That refresh went out without ``resource``, and an authorization
server that binds grants to the resource (Cloudflare) answered 400 for a token that refreshed
fine from any other request.

The tests drive the real ``async_auth_flow`` of the manager's provider and inspect the refresh
request the SDK would send.
"""
from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import parse_qs

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required")

SERVER_URL = "https://mcp.example.com/mcp"


def _cold_provider(tmp_path, monkeypatch):
    """A provider as a restarted process builds it: expired token, client and metadata on disk."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tokens = tmp_path / "mcp-tokens"
    tokens.mkdir(parents=True)
    (tokens / "srv.json").write_text(json.dumps({
        "access_token": "at", "token_type": "Bearer", "expires_in": 3600,
        "refresh_token": "rt", "expires_at": time.time() - 5}), encoding="utf-8")
    (tokens / "srv.client.json").write_text(json.dumps({
        "client_id": "client", "redirect_uris": ["http://127.0.0.1:33333/callback"]}), encoding="utf-8")
    from mcp.shared.auth import OAuthMetadata
    from tools.mcp_oauth import HermesTokenStorage
    HermesTokenStorage("srv").save_oauth_metadata(OAuthMetadata.model_validate({
        "issuer": "https://idp.example.com",
        "authorization_endpoint": "https://idp.example.com/authorize",
        "token_endpoint": "https://idp.example.com/token",
        "response_types_supported": ["code"]}))
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    reset_manager_for_tests()
    return get_manager().get_or_build_provider("srv", SERVER_URL, None)


async def _refresh_form(provider, headers):
    """The form body of the refresh POST the auth flow yields for an MCP request with *headers*."""
    from tools.mcp_tool import sdk_httpx
    flow = provider.async_auth_flow(sdk_httpx().Request("POST", SERVER_URL, headers=headers))
    try:
        refresh = await flow.__anext__()
    finally:
        await flow.aclose()
    assert str(refresh.url) == "https://idp.example.com/token"
    form = parse_qs(refresh.content.decode())
    assert form["grant_type"] == ["refresh_token"]
    return form


@pytest.mark.parametrize("headers", [
    pytest.param({}, id="initialize-without-version"),
    pytest.param({"MCP-Protocol-Version": "2025-06-18"}, id="request-with-version"),
])
def test_refresh_names_the_resource_whichever_request_triggers_it(tmp_path, monkeypatch, headers):
    provider = _cold_provider(tmp_path, monkeypatch)
    form = asyncio.run(_refresh_form(provider, headers))
    assert form.get("resource") == [SERVER_URL]


def test_initialize_refresh_follows_the_last_negotiated_version(tmp_path, monkeypatch):
    """A server that negotiated a version before RFC 8707 support keeps getting refreshes without it."""
    provider = _cold_provider(tmp_path, monkeypatch)

    async def _both():
        negotiated = await _refresh_form(provider, {"MCP-Protocol-Version": "2025-03-26"})
        return negotiated, await _refresh_form(provider, {})

    negotiated, initialize = asyncio.run(_both())
    assert "resource" not in negotiated
    assert "resource" not in initialize

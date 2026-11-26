"""A configured ``oauth.scope`` is authoritative over the SDK's scope selection (#142477).

The MCP SDK's 401 branch reassigns ``client_metadata.scope`` from server metadata (WWW-Authenticate
hint, else PRM / AS ``scopes_supported``, plus ``offline_access``) before dynamic client registration
and the authorization URL read it. A server whose metadata advertises more scopes than its
registration accepts then refuses every login, and ``oauth.scope`` reads as broken. The flow below
runs the real provider through its 401 branch against an in-process stand-in server.
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required for OAuth support")

RESOURCE = "https://res.example/mcp"
AS = "https://as.example"
SELECTED = "mcp:read mcp:write admin offline_access"  # what SDK discovery assigns here


def _stand_in(httpx, seen):
    def handle(request):
        url = str(request.url)
        j = lambda status, body, **h: httpx.Response(status, json=body, headers=h, request=request)
        if url == RESOURCE:
            if request.headers.get("Authorization") == "Bearer AT-1":
                return j(200, {"ok": True})
            return j(401, {}, **{"WWW-Authenticate": 'Bearer resource_metadata="https://res.example/.well-known/oauth-protected-resource"'})
        if url == "https://res.example/.well-known/oauth-protected-resource":
            return j(200, {"resource": RESOURCE, "authorization_servers": [AS],
                           "scopes_supported": ["mcp:read", "mcp:write", "admin"]})
        if url == f"{AS}/.well-known/oauth-authorization-server":
            return j(200, {"issuer": AS, "authorization_endpoint": f"{AS}/authorize", "token_endpoint": f"{AS}/token",
                           "registration_endpoint": f"{AS}/register", "response_types_supported": ["code"],
                           "code_challenge_methods_supported": ["S256"],
                           "scopes_supported": ["mcp:read", "mcp:write", "admin", "offline_access"]})
        if url == f"{AS}/register":
            body = json.loads(request.content)
            seen["register"] = body.get("scope")
            return j(201, {**body, "client_id": "dcr-1"})
        if url == f"{AS}/token":
            return j(200, {"access_token": "AT-1", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "RT-1"})
        return j(404, {})
    return handle


async def _sign_in(tmp_path, monkeypatch, oauth_cfg):
    from tools.mcp_oauth import HermesTokenStorage, _authorization_code_result, _build_client_metadata
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    seen: dict = {}

    async def redirect(url):
        seen["authorize"] = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}

    async def callback():
        return _authorization_code_result("code-1", seen["authorize"]["state"], iss=AS)

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv", server_url=RESOURCE, storage=HermesTokenStorage("srv"),
        client_metadata=_build_client_metadata({"_resolved_port": 43123, **oauth_cfg}),
        redirect_handler=redirect, callback_handler=callback)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(_stand_in(httpx, seen))) as client:
        response = await client.get(RESOURCE)
    assert response.status_code == 200
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize(("oauth_cfg", "sent", "prompt"), [
    pytest.param({"scope": "mcp:read offline_access"}, "mcp:read offline_access", "consent", id="configured+offline"),
    pytest.param({"scope": "mcp:read"}, "mcp:read", None, id="configured-no-forced-consent"),
    pytest.param({}, SELECTED, "consent", id="unset-sdk-selection"),
])
async def test_sign_in_sends_the_configured_scope_at_every_step(tmp_path, monkeypatch, oauth_cfg, sent, prompt):
    """Invariant: registration and the authorization URL carry the configured scope (and no forced
    consent without ``offline_access``); an unset scope keeps the SDK's selection."""
    seen = await _sign_in(tmp_path, monkeypatch, oauth_cfg)
    assert seen["register"] == sent
    assert seen["authorize"]["scope"] == sent
    assert seen["authorize"].get("prompt") == prompt

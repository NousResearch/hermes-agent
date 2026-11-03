"""Configured ``oauth.scope`` wins over discovery, Google's root-slash issuer, and forced login.

#93719 / #101467: the SDK replaces ``client_metadata.scope`` with the 401 challenge scope or the server's
whole ``scopes_supported`` before every authorization (and unions 403 step-up challenges into it), so a
read-only ``oauth.scope`` still asked Google's Gmail MCP for ``https://mail.google.com/`` and ``gmail.send``.
#137289: Google's protected-resource metadata advertises ``https://accounts.google.com/`` while its metadata
document names ``https://accounts.google.com``; and its MCP servers answer ``initialize``/``tools/list``
anonymously, so ``hermes mcp login`` never started the browser flow (#89412).
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

G = "https://www.googleapis.com/auth/"
PIN = f"{G}gmail.readonly {G}gmail.compose"
BROAD = ["https://mail.google.com/", f"{G}gmail.modify", f"{G}gmail.send", f"{G}gmail.readonly", f"{G}gmail.compose"]
RESOURCE = "https://gmailmcp.example/mcp/v1"
PRM = "https://gmailmcp.example/.well-known/oauth-protected-resource/mcp/v1"
GOOGLE = "https://accounts.google.com"


def _google(httpx, log, *, challenge_scope=None, stepup_scope=None):
    granted = {}

    def handler(request):
        url = str(request.url).split("?")[0]
        j = lambda status, body, **h: httpx.Response(status, json=body, headers=h, request=request)
        if url == RESOURCE:
            if request.headers.get("Authorization") == "Bearer AT-1":
                if stepup_scope:
                    return j(403, {}, **{"WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{stepup_scope}"'})
                return j(200, {"ok": True})
            extra = f', scope="{challenge_scope}"' if challenge_scope else ""
            return j(401, {}, **{"WWW-Authenticate": f'Bearer resource_metadata="{PRM}"{extra}'})
        if url == PRM:
            return j(200, {"resource": RESOURCE, "authorization_servers": [f"{GOOGLE}/"], "scopes_supported": BROAD})
        if url.startswith(GOOGLE) and "/.well-known/" in url:
            return j(200, {"issuer": GOOGLE, "authorization_endpoint": f"{GOOGLE}/o/oauth2/v2/auth",
                           "token_endpoint": f"{GOOGLE}/token", "response_types_supported": ["code"],
                           "code_challenge_methods_supported": ["S256"], "scopes_supported": BROAD,
                           "grant_types_supported": ["authorization_code", "refresh_token"]})
        if url == f"{GOOGLE}/token":
            log.append("token")
            return j(200, {"access_token": "AT-1", "token_type": "Bearer", "expires_in": 3600,
                           "refresh_token": "RT-1", "scope": granted.get("scope", "")})
        return j(404, {})
    return handler, granted


async def _flow(tmp_path, monkeypatch, *, scope=PIN, requests=1, **server):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage, _authorization_code_result
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    log, seen = [], []
    handler, granted = _google(httpx, log, **server)
    storage = HermesTokenStorage("srv")
    await storage.set_client_info(OAuthClientInformationFull(
        client_id="pre-1", client_secret="s", redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")],
        token_endpoint_auth_method="client_secret_post"))

    async def redirect(url):
        q = dict(parse_qsl(urlsplit(url).query))
        seen.append(q)
        granted["scope"] = q.get("scope", "")

    async def callback():
        return _authorization_code_result("code-1", seen[-1]["state"])

    metadata = OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")], client_name="Hermes Agent",
                                   token_endpoint_auth_method="client_secret_post")
    if scope is not None:
        metadata.scope = scope
    provider = _HERMES_PROVIDER_CLS(server_name="srv", server_url=RESOURCE, storage=storage, client_metadata=metadata,
                                    redirect_handler=redirect, callback_handler=callback, preregistered=True)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(handler)) as client:
        for _ in range(requests):
            response = await client.get(RESOURCE)
    return response, seen, log


@pytest.mark.asyncio
async def test_configured_scope_wins_over_scopes_supported_and_root_slash_issuer(tmp_path, monkeypatch):
    response, seen, _ = await _flow(tmp_path, monkeypatch)
    assert response.status_code == 200
    assert seen[0]["scope"] == PIN
    assert seen[0]["access_type"] == "offline"


@pytest.mark.asyncio
async def test_unconfigured_scope_keeps_discovery(tmp_path, monkeypatch):
    response, seen, _ = await _flow(tmp_path, monkeypatch, scope=None)
    assert response.status_code == 200
    assert set(seen[0]["scope"].split()) >= set(BROAD)


@pytest.mark.asyncio
async def test_challenge_demanding_more_than_configured_scope_fails_closed(tmp_path, monkeypatch):
    from mcp.client.auth.exceptions import OAuthFlowError
    with pytest.raises(OAuthFlowError, match="outside the configured oauth.scope"):
        await _flow(tmp_path, monkeypatch, challenge_scope=f"{G}gmail.send")


@pytest.mark.asyncio
async def test_step_up_beyond_configured_scope_fails_closed(tmp_path, monkeypatch):
    from mcp.client.auth.exceptions import OAuthFlowError
    with pytest.raises(OAuthFlowError, match="outside the configured oauth.scope"):
        await _flow(tmp_path, monkeypatch, stepup_scope=f"{G}gmail.send", requests=2)


def test_root_slash_equivalence_is_narrow():
    from tools.mcp_oauth_provider import issuer_differs_only_by_root_slash as eq
    assert eq("https://accounts.google.com", "https://accounts.google.com/")
    assert not eq("https://accounts.google.com", "https://accounts.google.com")          # equal: SDK path
    assert not eq("https://accounts.google.com", "https://accounts.google.com.evil/")
    assert not eq("https://as.example/tenant", "https://as.example/tenant/")             # path: exact check
    assert not eq("http://accounts.google.com", "http://accounts.google.com/")           # https only


def test_login_forces_authorization_after_anonymous_probe(monkeypatch, tmp_path):
    """`hermes mcp login` with a pre-registered client drives the grant when tools/list was public."""
    import contextlib
    import io
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.mcp_config as mc

    provider = MagicMock()
    provider.force_authorization = AsyncMock()
    manager = MagicMock()
    manager.get_or_build_provider.return_value = provider
    tokens = iter((False, True))
    monkeypatch.setattr(mc, "_probe_single_server", lambda *args, **kwargs: [])
    monkeypatch.setattr(mc, "_oauth_tokens_present", lambda name: next(tokens))
    monkeypatch.setattr("tools.mcp_oauth_manager.get_manager", lambda: manager)
    with contextlib.redirect_stdout(io.StringIO()):
        assert mc._reauth_oauth_server("gmail", {"url": RESOURCE, "auth": "oauth",
                                                 "oauth": {"client_id": "c", "client_secret": "s", "scope": PIN}})
    provider.force_authorization.assert_awaited_once()

"""The Gmail MCP resource must not be granted more scope than the client needs.

``gmailmcp.googleapis.com`` advertises ``gmail.modify`` and the full ``https://mail.google.com/``
scope in its protected-resource metadata. The MCP SDK's scope-selection rule
(WWW-Authenticate -> PRM -> AS) asks for that advertised union, so a Hermes profile configured for
search/read/draft — with the label/trash tools excluded via ``tools.include`` — would still be
granted the power to label, trash and permanently delete any message in the mailbox. The tool
filter does not protect against this: scope is granted at consent time, not per tool.

Hermes now pins the Gmail MCP authorize request to read + compose, re-asserting it at the
authorization-code-grant entry point because the SDK recomputes ``client_metadata.scope`` from the
discovered resource metadata just before building the authorize URL. An explicit ``oauth.scope``
still wins, and other hosts are untouched.

Scope only. These tests advertise an issuer that matches its own metadata document, so the flow
reaches scope selection; the resource's separate trailing-slash issuer drift is a different defect
handled elsewhere (see the Google-issuer PRs) and is deliberately not exercised here.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

GMAIL_RESOURCE = "https://gmailmcp.googleapis.com/mcp/v1"
GOOGLE_AS = "https://accounts.google.com"

READ_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
COMPOSE_SCOPE = "https://www.googleapis.com/auth/gmail.compose"
MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"
FULL_MAIL_SCOPE = "https://mail.google.com/"

# What the Gmail MCP resource advertises: the union across its whole tool surface.
ADVERTISED_SCOPES = [READ_SCOPE, COMPOSE_SCOPE, MODIFY_SCOPE, FULL_MAIL_SCOPE]


def _asm(issuer):
    return {"issuer": issuer, "authorization_endpoint": f"{issuer}/o/oauth2/v2/auth",
            "token_endpoint": f"{issuer}/token", "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"]}


class _StandIn:
    """MCP resource + authorization server behind one httpx MockTransport."""

    def __init__(self, httpx, *, resource, auth_server, prm_scopes):
        self.httpx, self.resource, self.auth_server = httpx, resource, auth_server
        self.prm_scopes, self.hits = prm_scopes, []

    def __call__(self, request):
        url = str(request.url)
        path = urlsplit(url).path
        self.hits.append((request.method, url))
        j = lambda status, body, **h: self.httpx.Response(status, json=body, headers=h, request=request)  # noqa: E731
        if url == self.resource:
            if request.headers.get("Authorization") == "Bearer AT-1":
                return j(200, {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
            origin = self.auth_server.rstrip("/")
            return j(401, {}, **{"WWW-Authenticate":
                                 f'Bearer resource_metadata="{origin}/.well-known/oauth-protected-resource/mcp"'})
        if path == "/.well-known/oauth-protected-resource/mcp":
            return j(200, {"resource": self.resource, "authorization_servers": [self.auth_server],
                           "scopes_supported": self.prm_scopes})
        if path in ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"):
            return j(200, _asm(self.auth_server.rstrip("/")))
        if url == f"{self.auth_server.rstrip('/')}/token":
            return j(200, {"access_token": "AT-1", "token_type": "Bearer",
                           "expires_in": 3600, "refresh_token": "RT-1"})
        return j(404, {})


async def _run_flow(tmp_path, monkeypatch, *, resource=GMAIL_RESOURCE, auth_server=GOOGLE_AS,
                    prm_scopes=None, oauth_scope=None, server_name="gmail"):
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage, _authorization_code_result, _maybe_preregister_client
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    seen = {}

    async def redirect(url):
        seen["authorize_url"] = url
        seen["state"] = parse_qs(urlsplit(url).query)["state"][0]

    async def callback():
        return _authorization_code_result("code-1", seen["state"], iss=auth_server.rstrip("/"))

    # Google advertises no registration endpoint, so seed the pre-registered client.
    storage = HermesTokenStorage(server_name)
    client_metadata = OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")],
                                          client_name="Hermes Agent", scope=oauth_scope)
    cfg = {"client_id": "gcid.apps.googleusercontent.com", "client_secret": "cs", "_resolved_port": 45123}
    if oauth_scope:
        cfg["scope"] = oauth_scope
    _maybe_preregister_client(storage, cfg, client_metadata)

    provider = _HERMES_PROVIDER_CLS(
        server_name=server_name, server_url=resource, storage=storage,
        client_metadata=client_metadata, redirect_handler=redirect, callback_handler=callback)
    standin = _StandIn(httpx, resource=resource, auth_server=auth_server, prm_scopes=prm_scopes)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(standin)) as client:
        response = await client.get(resource)
    return response, standin, seen


def _requested_scopes(seen):
    return set(parse_qs(urlsplit(seen["authorize_url"]).query)["scope"][0].split())


@pytest.mark.asyncio
async def test_gmail_mcp_requests_only_read_and_compose(tmp_path, monkeypatch):
    """Against a PRM advertising modify + full mail, the authorize URL must stay narrow."""
    response, _, seen = await _run_flow(tmp_path, monkeypatch, prm_scopes=ADVERTISED_SCOPES)

    assert response.status_code == 200
    requested = _requested_scopes(seen)
    assert requested == {READ_SCOPE, COMPOSE_SCOPE}
    assert MODIFY_SCOPE not in requested
    assert FULL_MAIL_SCOPE not in requested


@pytest.mark.asyncio
async def test_pin_survives_the_sdk_recomputing_scope_from_prm(tmp_path, monkeypatch):
    """The URL's scope is the pin, not the advertised union the SDK derives from PRM."""
    _, _, seen = await _run_flow(tmp_path, monkeypatch, prm_scopes=ADVERTISED_SCOPES)

    query = parse_qs(urlsplit(seen["authorize_url"]).query)
    assert set(query["scope"][0].split()) == {READ_SCOPE, COMPOSE_SCOPE}
    # Google needs both for a usable refresh token.
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]


@pytest.mark.asyncio
async def test_narrow_scope_when_prm_does_not_advertise_scopes_at_all(tmp_path, monkeypatch):
    """No scopes_supported to fall back on: the pin still produces a usable, narrow request."""
    _, _, seen = await _run_flow(tmp_path, monkeypatch, prm_scopes=[])

    assert _requested_scopes(seen) == {READ_SCOPE, COMPOSE_SCOPE}


@pytest.mark.asyncio
async def test_configured_scope_wins_over_the_default_pin(tmp_path, monkeypatch):
    """A user who explicitly asks for modify (to use the label tools) still gets it."""
    configured = f"{READ_SCOPE} {MODIFY_SCOPE}"
    _, _, seen = await _run_flow(tmp_path, monkeypatch, prm_scopes=ADVERTISED_SCOPES,
                                 oauth_scope=configured)

    assert _requested_scopes(seen) == set(configured.split())


@pytest.mark.asyncio
async def test_other_hosts_keep_the_advertised_scopes(tmp_path, monkeypatch):
    """The pin is host-pinned: an unrelated server still asks for its own advertised union."""
    _, _, seen = await _run_flow(tmp_path, monkeypatch, resource="https://mcp.example.com/v1",
                                 auth_server="https://as.example", server_name="other",
                                 prm_scopes=["scope:read", "scope:write"])

    assert _requested_scopes(seen) == {"scope:read", "scope:write"}

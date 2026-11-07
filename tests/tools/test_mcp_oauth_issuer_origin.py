"""Metadata documents whose issuer the SDK's exact-string check (RFC 8414 §3.3) refuses but Hermes accepts.

Strava's MCP connector advertises ``authorization_servers: ["https://www.strava.com/mcp-issuer"]`` and
serves ``/.well-known/oauth-authorization-server/mcp-issuer`` with ``issuer: "https://www.strava.com"``
(#116233): accepted when the document comes from the well-known URL derived from the advertised identifier
and names that identifier's origin. Otter.ai advertises ``https://otter.ai/`` while its document names
``https://otter.ai`` (#132233): a root issuer differing only by its trailing slash is accepted, the SDK's
own ``issuers_match`` rule. Every other mismatch is still rejected, and an issuer equal to the advertised
identifier is left to the SDK.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

RESOURCE = "https://res.example/mcp"
AS_ORIGIN = "https://as.example"
ADVERTISED = f"{AS_ORIGIN}/mcp-issuer"
PATH_DOC = "/.well-known/oauth-authorization-server/mcp-issuer"


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    pytest.param(RESOURCE, id="mcp-resource"),
    pytest.param(f"{AS_ORIGIN}/proxy/.well-known/oauth-authorization-server", id="unrelated-prefix"),
    pytest.param(f"{AS_ORIGIN}/.well-known/oauth-authorization-server-extra", id="suffix-lookalike"),
    pytest.param(f"{RESOURCE}?next=/.well-known/oauth-authorization-server", id="query"),
    pytest.param(f"{RESOURCE}#/.well-known/openid-configuration", id="fragment"),
])
async def test_origin_issued_metadata_shim_does_not_read_non_discovery_response(url):
    from tools.mcp_oauth_provider import HermesProviderMixin

    class ResourceResponse:
        status_code = 200
        request = SimpleNamespace(url=url)

        async def aread(self):
            raise AssertionError("normal MCP resource response must not be consumed as OAuth metadata")

    provider = HermesProviderMixin.__new__(HermesProviderMixin)
    response = ResourceResponse()

    assert await provider._hermes_accept_origin_issued_metadata(response) is response


def _asm(issuer):
    return {"issuer": issuer, "authorization_endpoint": f"{AS_ORIGIN}/authorize", "token_endpoint": f"{AS_ORIGIN}/token",
            "registration_endpoint": f"{AS_ORIGIN}/register", "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"], "authorization_response_iss_parameter_supported": True}


class _StandIn:
    """Resource + authorization server behind one httpx MockTransport; ``issuer_doc`` maps ASM path -> document."""

    def __init__(self, httpx, issuer_doc, advertised=ADVERTISED):
        self.httpx, self.issuer_doc, self.advertised, self.hits = httpx, issuer_doc, advertised, []

    def __call__(self, request):
        url = str(request.url)
        path = urlsplit(url).path
        self.hits.append((request.method, url))
        j = lambda status, body, **h: self.httpx.Response(status, json=body, headers=h, request=request)
        if url == RESOURCE:
            if request.headers.get("Authorization") == "Bearer AT-1":
                return j(200, {"ok": True})
            return j(401, {}, **{"WWW-Authenticate": 'Bearer resource_metadata="https://res.example/.well-known/oauth-protected-resource"'})
        if url == "https://res.example/.well-known/oauth-protected-resource":
            return j(200, {"resource": RESOURCE, "authorization_servers": [self.advertised]})
        if url.startswith(AS_ORIGIN) and path in self.issuer_doc:
            return j(200, self.issuer_doc[path])
        if url == f"{AS_ORIGIN}/register":
            return j(201, {"client_id": "dcr-1", "redirect_uris": ["http://127.0.0.1:1/cb"], "token_endpoint_auth_method": "none",
                           "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"]})
        if url == f"{AS_ORIGIN}/token":
            return j(200, {"access_token": "AT-1", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "RT-1"})
        return j(404, {})


async def _run_flow(tmp_path, monkeypatch, issuer_doc, advertised=ADVERTISED, iss=AS_ORIGIN):
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage, _authorization_code_result
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
        return _authorization_code_result("code-1", seen["state"], iss=iss)

    storage = HermesTokenStorage("srv")
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv", server_url=RESOURCE, storage=storage,
        client_metadata=OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")], client_name="Hermes Agent"),
        redirect_handler=redirect, callback_handler=callback)
    standin = _StandIn(httpx, issuer_doc, advertised)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(standin)) as client:
        response = await client.get(RESOURCE)
    return response, standin, seen, provider


@pytest.mark.asyncio
async def test_origin_issued_document_of_path_scoped_server_completes_the_flow(tmp_path, monkeypatch):
    response, standin, seen, provider = await _run_flow(tmp_path, monkeypatch, {PATH_DOC: _asm(AS_ORIGIN)})

    assert response.status_code == 200
    assert seen["authorize_url"].startswith(f"{AS_ORIGIN}/authorize?")
    assert ("POST", f"{AS_ORIGIN}/register") in standin.hits and ("POST", f"{AS_ORIGIN}/token") in standin.hits
    assert str(provider.context.oauth_metadata.issuer).rstrip("/") == AS_ORIGIN
    # SEP-2352 binding stays on the advertised identifier, so the next 401 reuses this client instead of re-registering.
    assert json.loads((tmp_path / "mcp-tokens" / "srv.client.json").read_text())["issuer"] == ADVERTISED
    assert json.loads((tmp_path / "mcp-tokens" / "srv.json").read_text())["access_token"] == "AT-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("issuer_doc", [
    pytest.param({PATH_DOC: _asm("https://other.example")}, id="different-origin"),
    pytest.param({"/.well-known/oauth-authorization-server": _asm(AS_ORIGIN)}, id="root-document-only"),
    pytest.param({PATH_DOC: _asm(f"{AS_ORIGIN}/other-path")}, id="different-path"),
])
async def test_other_issuer_shapes_are_still_rejected(tmp_path, monkeypatch, issuer_doc):
    from mcp.client.auth.exceptions import OAuthFlowError, OAuthRegistrationError

    with pytest.raises((OAuthFlowError, OAuthRegistrationError)):
        await _run_flow(tmp_path, monkeypatch, issuer_doc)
    assert not (tmp_path / "mcp-tokens" / "srv.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(("advertised", "issuer"), [
    pytest.param(f"{AS_ORIGIN}/", AS_ORIGIN, id="advertised-with-slash"),  # Otter.ai, #132233
    pytest.param(AS_ORIGIN, f"{AS_ORIGIN}/", id="document-with-slash"),
])
async def test_root_issuer_differing_only_by_its_trailing_slash_completes_the_flow(
        tmp_path, monkeypatch, advertised, issuer):
    root_doc = {"/.well-known/oauth-authorization-server": _asm(issuer)}
    response, _, _, _ = await _run_flow(tmp_path, monkeypatch, root_doc, advertised=advertised, iss=issuer)

    assert response.status_code == 200
    assert json.loads((tmp_path / "mcp-tokens" / "srv.client.json").read_text())["issuer"] == advertised


@pytest.mark.asyncio
async def test_path_scoped_issuer_differing_only_by_a_trailing_slash_is_still_rejected(tmp_path, monkeypatch):
    from mcp.client.auth.exceptions import OAuthFlowError

    with pytest.raises(OAuthFlowError, match="issuer mismatch"):
        await _run_flow(tmp_path, monkeypatch, {PATH_DOC: _asm(f"{ADVERTISED}/")}, iss=f"{ADVERTISED}/")
    assert not (tmp_path / "mcp-tokens" / "srv.json").exists()


@pytest.mark.asyncio
async def test_issuer_equal_to_the_advertised_identifier_is_left_to_the_sdk():
    from tools.mcp_oauth_provider import HermesProviderMixin
    from tools.mcp_tool import sdk_httpx

    class _Provider(HermesProviderMixin):
        context: SimpleNamespace

    httpx = sdk_httpx()
    provider = _Provider.__new__(_Provider)
    provider.context = SimpleNamespace(auth_server_url=f"{AS_ORIGIN}/", oauth_metadata=None)
    request = httpx.Request("GET", f"{AS_ORIGIN}/.well-known/oauth-authorization-server")
    response = httpx.Response(200, json=_asm(f"{AS_ORIGIN}/"), request=request)

    assert await provider._hermes_accept_origin_issued_metadata(response) is response
    assert provider.context.oauth_metadata is None

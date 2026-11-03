"""A path-scoped authorization server whose RFC 8414 metadata names its origin (#116233).

Strava's MCP connector advertises ``authorization_servers: ["https://www.strava.com/mcp-issuer"]`` and
serves ``/.well-known/oauth-authorization-server/mcp-issuer`` with ``issuer: "https://www.strava.com"``.
The SDK's exact-string issuer check (RFC 8414 §3.3) rejected that document and discovery never completed.
Hermes accepts exactly this shape — the document fetched from the well-known URL derived from the advertised
identifier, naming that identifier's origin — through the real provider flow; every other mismatch is still
rejected.
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
ENTRA_ORIGIN = "https://login.microsoftonline.com"
ENTRA_ADVERTISED = f"{ENTRA_ORIGIN}/organizations/v2.0"
ENTRA_TENANT_TEMPLATE = f"{ENTRA_ORIGIN}/{{tenantid}}/v2.0"
ENTRA_TENANT_TEMPLATE_ENCODED = f"{ENTRA_ORIGIN}/%7Btenantid%7D/v2.0"
ENTRA_OIDC_DOC = f"{ENTRA_ADVERTISED}/.well-known/openid-configuration"


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


def _asm(issuer, as_origin=AS_ORIGIN):
    return {"issuer": issuer, "authorization_endpoint": f"{as_origin}/authorize", "token_endpoint": f"{as_origin}/token",
            "registration_endpoint": f"{as_origin}/register", "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"], "authorization_response_iss_parameter_supported": True}


class _StandIn:
    """Resource + authorization server behind one httpx MockTransport; ``issuer_doc`` maps ASM path -> document."""

    def __init__(self, httpx, issuer_doc, auth_server=ADVERTISED, as_origin=AS_ORIGIN):
        self.httpx, self.issuer_doc, self.auth_server, self.as_origin, self.hits = httpx, issuer_doc, auth_server, as_origin, []

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
            return j(200, {"resource": RESOURCE, "authorization_servers": [self.auth_server]})
        if url.startswith(self.as_origin) and path in self.issuer_doc:
            return j(200, self.issuer_doc[path])
        if url == f"{self.as_origin}/register":
            return j(201, {"client_id": "dcr-1", "redirect_uris": ["http://127.0.0.1:1/cb"], "token_endpoint_auth_method": "none",
                           "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"]})
        if url == f"{self.as_origin}/token":
            return j(200, {"access_token": "AT-1", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "RT-1"})
        return j(404, {})


async def _run_flow(tmp_path, monkeypatch, issuer_doc, auth_server=ADVERTISED, as_origin=AS_ORIGIN, callback_issuer=AS_ORIGIN):
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
        return _authorization_code_result("code-1", seen["state"], iss=callback_issuer)

    storage = HermesTokenStorage("srv")
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv", server_url=RESOURCE, storage=storage,
        client_metadata=OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")], client_name="Hermes Agent"),
        redirect_handler=redirect, callback_handler=callback)
    standin = _StandIn(httpx, issuer_doc, auth_server, as_origin)
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
async def test_entra_organizations_metadata_with_tenant_template_completes_the_flow(tmp_path, monkeypatch):
    response, standin, seen, provider = await _run_flow(
        tmp_path, monkeypatch,
        {urlsplit(ENTRA_OIDC_DOC).path: _asm(ENTRA_TENANT_TEMPLATE, ENTRA_ORIGIN)},
        auth_server=ENTRA_ADVERTISED, as_origin=ENTRA_ORIGIN, callback_issuer=ENTRA_TENANT_TEMPLATE_ENCODED)

    assert response.status_code == 200
    assert seen["authorize_url"].startswith(f"{ENTRA_ORIGIN}/authorize?")
    assert ("POST", f"{ENTRA_ORIGIN}/register") in standin.hits
    assert str(provider.context.oauth_metadata.issuer).replace("%7B", "{").replace("%7D", "}").rstrip("/") == ENTRA_TENANT_TEMPLATE


def test_entra_tenant_template_does_not_relax_other_aliases_or_documents():
    from tools.mcp_oauth_provider import metadata_issued_by_entra_tenant_template

    metadata = SimpleNamespace(issuer=ENTRA_TENANT_TEMPLATE)
    document = SimpleNamespace(status_code=200, url=ENTRA_OIDC_DOC)

    assert metadata_issued_by_entra_tenant_template(metadata, ENTRA_ADVERTISED, document)
    assert not metadata_issued_by_entra_tenant_template(
        metadata, f"{ENTRA_ORIGIN}/common/v2.0", document)
    assert not metadata_issued_by_entra_tenant_template(
        metadata, ENTRA_ADVERTISED,
        SimpleNamespace(status_code=200, url=f"{ENTRA_ORIGIN}/other/.well-known/openid-configuration"))


@pytest.mark.asyncio
async def test_device_login_accepts_entra_organizations_tenant_template():
    from tools.mcp_oauth_device import DeviceOAuthMetadata, _device_metadata
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()

    def handler(request):
        document = _asm(ENTRA_TENANT_TEMPLATE, ENTRA_ORIGIN)
        document.update({
            "device_authorization_endpoint": f"{ENTRA_ORIGIN}/devicecode",
            "grant_types_supported": ["urn:ietf:params:oauth:grant-type:device_code"],
        })
        return httpx.Response(200, json=document, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert isinstance(await _device_metadata(client, RESOURCE, ENTRA_ADVERTISED), DeviceOAuthMetadata)


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

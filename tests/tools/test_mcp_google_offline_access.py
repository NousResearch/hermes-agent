"""Google-hosted MCP OAuth: offline access and the device-flow issuer mismatch (#117510).

Google issues refresh tokens only for authorization requests carrying ``access_type=offline`` — its
idiom where OIDC servers use the ``offline_access`` scope that MCP discovery would advertise — so a
Google-hosted server (Gmail/Calendar MCP) authorized in a browser dies with the short-lived access
token: every later reconnect (a gateway process, cron) finds no refresh token and fails back to an
interactive login it cannot perform. The device flow could not even start: the advertised host-only
authorization server reaches issuer validation as ``str(AnyHttpUrl)`` with a trailing "/", which the
exact-string check rejects against Google's slash-less document issuer.
"""
from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qsl, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

RESOURCE = "https://gmailmcp.example/mcp"
PRM_PATH = "/.well-known/oauth-protected-resource"
ASM_PATH = "/.well-known/oauth-authorization-server"
GOOGLE = "https://accounts.google.com"  # advertised & document issuer, no trailing slash
OTHER_AS = "https://as.example"


def _asm_doc(issuer, **extra):
    doc = {"issuer": issuer, "authorization_endpoint": f"{issuer}/o/oauth2/v2/auth",
           "token_endpoint": f"{issuer}/token", "registration_endpoint": f"{issuer}/register",
           "response_types_supported": ["code"], "code_challenge_methods_supported": ["S256"],
           "grant_types_supported": ["authorization_code", "refresh_token"]}
    doc.update(extra)
    return doc


def _standin(httpx, *, issuer, authorization_servers):
    """Resource + authorization server behind one httpx MockTransport."""

    def handler(request):
        url = str(request.url)
        path = urlsplit(url).path
        j = lambda status, body, **h: httpx.Response(status, json=body, headers=h, request=request)  # noqa: E731
        if url == RESOURCE:
            if request.headers.get("Authorization") == "Bearer AT-1":
                return j(200, {"ok": True})
            return j(401, {}, **{"WWW-Authenticate": f'Bearer resource_metadata="https://gmailmcp.example{PRM_PATH}"'})
        if url == f"https://gmailmcp.example{PRM_PATH}":
            return j(200, {"resource": RESOURCE, "authorization_servers": authorization_servers})
        base = urlsplit(issuer).scheme + "://" + urlsplit(issuer).netloc
        if url.startswith(base) and path == ASM_PATH:
            return j(200, _asm_doc(issuer))
        if url == f"{base}/register":
            return j(201, {"client_id": "dcr-1", "redirect_uris": ["http://127.0.0.1:1/cb"],
                           "token_endpoint_auth_method": "none", "grant_types": ["authorization_code", "refresh_token"],
                           "response_types": ["code"]})
        if url == f"{base}/token":
            return j(200, {"access_token": "AT-1", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "RT-1"})
        return j(404, {})

    return handler


async def _run_browser_flow(tmp_path, monkeypatch, *, issuer, authorization_servers, scope=None):
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
        seen["state"] = dict(parse_qsl(urlsplit(url).query))["state"]

    async def callback():
        return _authorization_code_result("code-1", seen["state"])

    metadata = OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")], client_name="Hermes Agent")
    if scope is not None:
        metadata.scope = scope
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv", server_url=RESOURCE, storage=HermesTokenStorage("srv"), client_metadata=metadata,
        redirect_handler=redirect, callback_handler=callback)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(_standin(
            httpx, issuer=issuer, authorization_servers=authorization_servers))) as client:
        response = await client.get(RESOURCE)
    seen["provider"] = provider
    return response, seen


@pytest.mark.asyncio
async def test_google_authorization_url_asks_for_offline_access(tmp_path, monkeypatch):
    response, seen = await _run_browser_flow(
        tmp_path, monkeypatch, issuer=GOOGLE, authorization_servers=[GOOGLE], scope="email offline_access")

    assert response.status_code == 200
    pairs = parse_qsl(urlsplit(seen["authorize_url"]).query)
    # Exactly once each, even with an offline_access scope already requested.
    assert [value for key, value in pairs if key == "access_type"] == ["offline"]
    assert [value for key, value in pairs if key == "prompt"] == ["consent"]
    assert (tmp_path / "mcp-tokens" / "srv.json").exists()
    # Wraps once: a re-authorization must not nest another wrapper around the redirect handler.
    provider = seen["provider"]
    wrapped = provider.context.redirect_handler
    provider._request_google_offline_access()
    assert provider.context.redirect_handler is wrapped

    # Control: any other issuer keeps the SDK-built parameters untouched, and only the Google
    # authorization server itself (not a path or suffix lookalike) gets the parameters.
    response, seen = await _run_browser_flow(
        tmp_path / "other", monkeypatch, issuer=OTHER_AS, authorization_servers=[OTHER_AS])
    assert response.status_code == 200
    assert not any(key in ("access_type", "prompt") for key, _ in parse_qsl(urlsplit(seen["authorize_url"]).query))
    from tools.mcp_oauth_provider import google_offline_access_params

    def ctx(issuer):
        return SimpleNamespace(oauth_metadata=SimpleNamespace(issuer=issuer) if issuer is not None else None)

    for lookalike in (None, OTHER_AS, "https://evil.example/accounts.google.com", "https://accounts.google.com.evil.example"):
        assert google_offline_access_params(ctx(lookalike)) == {}, lookalike
    assert google_offline_access_params(ctx(GOOGLE)) == {"access_type": "offline"}


@pytest.mark.asyncio
async def test_device_flow_normalizes_issuer_and_asks_google_for_offline_access(capsys):
    from mcp.client.auth.exceptions import OAuthFlowError
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_device import DeviceOAuthMetadata, _authorize, _device_metadata
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()

    # Issuer check: the advertised host-only server arrives as ``str(AnyHttpUrl)`` with a trailing
    # "/"; both slash forms match Google's slash-less document issuer, a different issuer still fails.
    for document_issuer, advertised, accepted in (
            (GOOGLE, f"{GOOGLE}/", True), (f"{GOOGLE}/", f"{GOOGLE}/", True), ("https://evil.example", f"{GOOGLE}/", False)):
        def handler(request, document_issuer=document_issuer):
            doc = _asm_doc(document_issuer, device_authorization_endpoint=f"{GOOGLE}/device/code",
                           grant_types_supported=["authorization_code", "refresh_token",
                                                  "urn:ietf:params:oauth:grant-type:device_code"])
            return httpx.Response(200, json=doc, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            if accepted:
                assert isinstance(await _device_metadata(client, RESOURCE, advertised), DeviceOAuthMetadata)
            else:
                with pytest.raises(OAuthFlowError):
                    await _device_metadata(client, RESOURCE, advertised)

    # Device authorization request: Google's carries access_type=offline, any other issuer's does not.
    for issuer, expects_offline in ((GOOGLE, True), (OTHER_AS, False)):
        seen = {}

        def device_handler(request, issuer=issuer):
            if urlsplit(str(request.url)).path == "/device/code":
                body = request.content.decode() if request.content else ""
                seen["device_form"] = dict(pair.split("=", 1) for pair in body.split("&")) if body else {}
                return httpx.Response(200, json={
                    "device_code": "DC-1", "user_code": "UC-1", "verification_uri": f"{issuer}/activate",
                    "interval": 0.05, "expires_in": 60}, request=request)
            return httpx.Response(200, json={"access_token": "AT-1", "token_type": "Bearer",
                                             "expires_in": 3600}, request=request)

        provider = _HERMES_PROVIDER_CLS(
            server_name="srv", server_url=RESOURCE, storage=HermesTokenStorage("srv"),
            client_metadata=OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")], client_name="Hermes Agent"))
        provider.context.oauth_metadata = DeviceOAuthMetadata.model_validate(
            _asm_doc(issuer, device_authorization_endpoint=f"{issuer}/device/code"))
        provider.context.client_info = OAuthClientInformationFull.model_validate(
            {"client_id": "cid-1", "redirect_uris": ["http://127.0.0.1:1/cb"], "token_endpoint_auth_method": "none"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(device_handler)) as client:
            tokens = await _authorize(client, provider, {"timeout": 30})
        capsys.readouterr()
        assert tokens.access_token == "AT-1"
        assert ("access_type" in seen["device_form"]) is expects_offline, issuer

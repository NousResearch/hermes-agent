"""``oauth.auth_server_url`` (alias ``oauth.issuer``) pins the authorization server.

Microsoft Entra-fronted MCP servers (Dynamics 365 Business Central among them) advertise the
tenant-agnostic ``https://login.microsoftonline.com/common/v2.0`` in their protected-resource
metadata, and that authority's metadata names the TEMPLATED issuer
``https://login.microsoftonline.com/{tenantid}/v2.0``, so the RFC 8414 §3.3 issuer check rejects every
login. A tenant authority's metadata names itself, so pinning it satisfies the unmodified check.
"""
from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

RESOURCE = "https://res.example/mcp"
LOGIN = "https://login.microsoftonline.com"
TENANT_ID = "00000000-1111-2222-3333-444444444444"
TENANT = f"{LOGIN}/{TENANT_ID}/v2.0"
OAUTH = f"{LOGIN}/{TENANT_ID}/oauth2/v2.0"


def _metadata(segment, issuer):
    base = f"{LOGIN}/{segment}/oauth2/v2.0"
    return {"issuer": issuer, "authorization_endpoint": f"{base}/authorize", "token_endpoint": f"{base}/token",
            "device_authorization_endpoint": f"{base}/devicecode", "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token", "urn:ietf:params:oauth:grant-type:device_code"],
            "code_challenge_methods_supported": ["S256"]}


class _Entra:
    """A protected resource advertising Entra's ``common`` authority, plus Entra's real issuer shapes."""

    def __init__(self, httpx):
        self.httpx, self.hits = httpx, []

    def __call__(self, request):
        url = str(request.url)
        self.hits.append(url)
        j = lambda status, body, **h: self.httpx.Response(status, json=body, headers=h, request=request)
        tokens = {"access_token": "AT-1", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "RT-1"}
        routes = {
            "https://res.example/.well-known/oauth-protected-resource":
                (200, {"resource": RESOURCE, "authorization_servers": [f"{LOGIN}/common/v2.0"]}),
            f"{LOGIN}/common/v2.0/.well-known/openid-configuration": (200, _metadata("common", f"{LOGIN}/{{tenantid}}/v2.0")),
            f"{TENANT}/.well-known/openid-configuration": (200, _metadata(TENANT_ID, TENANT)),
            f"{OAUTH}/devicecode": (200, {"device_code": "DC-1", "user_code": "ABCD-EFGH", "expires_in": 60,
                                          "verification_uri": "https://microsoft.com/devicelogin", "interval": 0.01}),
            f"{OAUTH}/token": (200, tokens),
        }
        if url == RESOURCE:
            if request.headers.get("Authorization") == "Bearer AT-1":
                return j(200, {"ok": True})
            return j(401, {}, **{"WWW-Authenticate": 'Bearer resource_metadata="https://res.example/.well-known/oauth-protected-resource"'})
        status, body = routes.get(url, (404, {}))
        return j(status, body)


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["auth_server_url", "issuer"])
async def test_configured_tenant_authority_completes_browser_login(tmp_path, monkeypatch, key):
    from tools.mcp_oauth import _authorization_code_result
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_oauth_provider import build_provider_kwargs, prepare_oauth_config
    from tools.mcp_tool import sdk_httpx

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    # Entra has no RFC 7591 registration endpoint: the app is pre-registered with a fixed loopback port.
    oauth = {"client_id": "entra-app", "redirect_host": "localhost", "redirect_port": 1, key: TENANT}
    cfg, storage = prepare_oauth_config("bc", RESOURCE, oauth)
    kwargs = build_provider_kwargs(cfg, storage, ssh_proxy_hint=False)
    seen = {}

    async def redirect(url):
        seen["authorize_url"] = url
        seen["state"] = parse_qs(urlsplit(url).query)["state"][0]

    async def callback():
        return _authorization_code_result("code-1", seen["state"])

    kwargs.update(redirect_handler=redirect, callback_handler=callback)
    provider = _HERMES_PROVIDER_CLS(server_name="bc", preregistered=True, server_url=RESOURCE, **kwargs)
    httpx = sdk_httpx()
    entra = _Entra(httpx)
    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(entra)) as client:
        response = await client.get(RESOURCE)

    assert response.status_code == 200
    assert seen["authorize_url"].startswith(f"{OAUTH}/authorize?")
    assert str(provider.context.oauth_metadata.issuer) == TENANT
    assert not any("/common/" in url for url in entra.hits)


@pytest.mark.asyncio
async def test_configured_tenant_authority_completes_device_login(tmp_path, monkeypatch):
    import tools.mcp_tool
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_device import login_device

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    real = tools.mcp_tool.sdk_httpx()
    entra = _Entra(real)

    class _Client(real.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "transport": real.MockTransport(entra)})

    shim = SimpleNamespace(AsyncClient=_Client, Request=real.Request, HTTPError=real.HTTPError,
                           TimeoutException=real.TimeoutException)
    monkeypatch.setattr(tools.mcp_tool, "sdk_httpx", lambda: shim)

    await login_device("bc", RESOURCE, {"client_id": "entra-app", "auth_server_url": TENANT})

    assert HermesTokenStorage("bc").has_cached_tokens()
    assert f"{OAUTH}/devicecode" in entra.hits
    assert not any("/common/" in url for url in entra.hits)

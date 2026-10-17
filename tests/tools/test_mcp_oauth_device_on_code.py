"""Device-flow ``on_code``: a chat surface gets the link and code instead of the host terminal.

With ``on_code`` the code goes to the callback (the complete URI when the server sends one, RFC
8628 §3.3.1, so a phone opens it in one tap) and nothing is printed; without it the terminal prompt
is unchanged.
"""
from __future__ import annotations

from urllib.parse import urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

RESOURCE = "https://memory.example/mcp"
ISSUER = "https://as.example"


def _provider():
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_device import DeviceOAuthMetadata
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv", server_url=RESOURCE, storage=HermesTokenStorage("srv"),
        client_metadata=OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")], client_name="Hermes Agent"))
    provider.context.oauth_metadata = DeviceOAuthMetadata.model_validate({
        "issuer": ISSUER, "token_endpoint": f"{ISSUER}/token", "response_types_supported": ["code"],
        "device_authorization_endpoint": f"{ISSUER}/device_authorization"})
    provider.context.client_info = OAuthClientInformationFull.model_validate(
        {"client_id": "cid-1", "redirect_uris": ["http://127.0.0.1:1/cb"], "token_endpoint_auth_method": "none"})
    return provider


def _handler(httpx, *, complete):
    def handler(request):
        if urlsplit(str(request.url)).path == "/device_authorization":
            body = {"device_code": "DC-1", "user_code": "ABCD-EFGH", "verification_uri": f"{ISSUER}/device",
                    "interval": 0.05, "expires_in": 600}
            if complete:
                body["verification_uri_complete"] = f"{ISSUER}/device?user_code=ABCD-EFGH"
            return httpx.Response(200, json=body, request=request)
        return httpx.Response(200, json={"access_token": "AT-1", "token_type": "Bearer", "expires_in": 3600},
                              request=request)
    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [True, False])
async def test_on_code_receives_the_link_and_nothing_is_printed(capsys, complete):
    from tools.mcp_oauth_device import _authorize
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    seen = []

    async def on_code(link, code, minutes):
        seen.append((link, code, minutes))

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(httpx, complete=complete))) as client:
        tokens = await _authorize(client, _provider(), {"timeout": 300}, on_code)

    assert tokens.access_token == "AT-1"
    link = f"{ISSUER}/device?user_code=ABCD-EFGH" if complete else f"{ISSUER}/device"
    assert seen == [(link, "ABCD-EFGH", 5)]  # oauth.timeout (300 s) caps the code's 600 s
    assert "ABCD-EFGH" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_without_on_code_the_terminal_prompt_is_unchanged(capsys):
    from tools.mcp_oauth_device import _authorize
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(httpx, complete=True))) as client:
        await _authorize(client, _provider(), {"timeout": 300})
    err = capsys.readouterr().err
    assert f"open {ISSUER}/device on any device" in err and "Code: ABCD-EFGH" in err

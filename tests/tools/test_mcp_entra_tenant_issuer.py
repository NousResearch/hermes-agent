"""Entra-only MCP servers and the multi-tenant ``{tenantid}`` issuer template (#136978).

Fabric and the other Entra-only MCP endpoints advertise an authorization server like
``https://login.microsoftonline.com/organizations/v2.0``, but their authorization-server document's
``issuer`` keeps Entra's multi-tenant template with a literal ``{tenantid}`` placeholder (reaching
the check percent-encoded as ``%7Btenantid%7D`` via ``str(AnyHttpUrl)``). The device flow's
RFC 8414 §3.3 exact-string check rejected that pair before sign-in. The template shape — same
origin, every path segment except the placeholder identical — is accepted; every other mismatch
still fails.
"""
from __future__ import annotations

from urllib.parse import urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

RESOURCE = "https://api.fabric.microsoft.com/v1/mcp/core"
ENTRA_ADVERTISED = "https://login.microsoftonline.com/organizations/v2.0"
ENTRA_TEMPLATE = "https://login.microsoftonline.com/{tenantid}/v2.0"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


def _asm_doc(issuer):
    return {"issuer": issuer, "authorization_endpoint": f"{issuer}/oauth2/v2.0/authorize",
            "token_endpoint": f"{issuer}/oauth2/v2.0/token",
            "device_authorization_endpoint": f"{issuer}/oauth2/v2.0/devicecode",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token", DEVICE_GRANT]}


@pytest.mark.asyncio
@pytest.mark.parametrize("document_issuer,accepted", (
    # The shape Entra publishes: template issuer against the concrete advertised identifier.
    (ENTRA_TEMPLATE, True),
    # str(AnyHttpUrl) percent-encodes the placeholder; same document, still accepted.
    ("https://login.microsoftonline.com/%7Btenantid%7D/v2.0", True),
    # A concrete tenant GUID issuer is a real mismatch against "organizations".
    ("https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47/v2.0", False),
    # Different origin, different non-placeholder segment, fewer segments, placeholder on the
    # advertised side only: all still rejected.
    ("https://evil.example/{tenantid}/v2.0", False),
    ("https://login.microsoftonline.com/{tenantid}/v1.0", False),
    ("https://login.microsoftonline.com/{tenantid}", False),
    ("https://login.microsoftonline.com/organizations/v2.0", False),  # expected side is the template below
))
async def test_device_flow_accepts_entra_tenant_issuer_template(document_issuer, accepted):
    from mcp.client.auth.exceptions import OAuthFlowError

    from tools.mcp_oauth_device import DeviceOAuthMetadata, _device_metadata
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    # Case: the advertised side itself carries the placeholder and the document is concrete — the
    # wildcard never applies to the advertised identifier.
    advertised = ENTRA_TEMPLATE if document_issuer == "https://login.microsoftonline.com/organizations/v2.0" else ENTRA_ADVERTISED

    def handler(request):
        path = urlsplit(str(request.url)).path
        assert "/.well-known/" in path, str(request.url)
        return httpx.Response(200, json=_asm_doc(document_issuer), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if accepted:
            assert isinstance(await _device_metadata(client, RESOURCE, advertised), DeviceOAuthMetadata)
        else:
            with pytest.raises(OAuthFlowError):
                await _device_metadata(client, RESOURCE, advertised)

"""Entra multi-tenant OAuth endpoints publish a ``{tenantid}`` template as their issuer (#132730).

Microsoft's ``/common`` and ``/organizations`` front doors — the authorization servers Sentinel and
other Microsoft-hosted MCP servers advertise — return
``https://login.microsoftonline.com/{tenantid}/v2.0`` as the ``issuer`` of their authorization-server
metadata; the placeholder is only substituted on the tenant-specific endpoint. The SDK's exact-string
comparison (RFC 8414 §3.3) rejected that document, so login failed in both flows. Hermes accepts
exactly this pair — the advertised server is one of the two Entra multi-tenant endpoints and the
document issuer is the template on the same host — and rewrites an RFC 9207 redirect ``iss`` naming a
concrete Entra tenant endpoint back to the template before the SDK compares it. Every other issuer
shape is still rejected.
"""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

RESOURCE = "https://sentinel.example/mcp/triage"
ENTRA = "https://login.microsoftonline.com"
ORGANIZATIONS = f"{ENTRA}/organizations/v2.0"
COMMON = f"{ENTRA}/common/v2.0"
TEMPLATE_ISSUER = (
    f"{ENTRA}/{{tenantid}}/v2.0"  # raw braces, as Entra serves the document
)
TENANT_GUID = "9188f0b0-1e3a-4d8b-a2b0-3f2f55f2f6da"
ASM_PATH = "/.well-known/oauth-authorization-server/organizations/v2.0"


def _asm_doc(issuer, **extra):
    doc = {
        "issuer": issuer,
        "authorization_endpoint": f"{ENTRA}/authorize",
        "token_endpoint": f"{ENTRA}/token",
        "registration_endpoint": f"{ENTRA}/register",
        "response_types_supported": ["code"],
        "code_challenge_methods_supported": ["S256"],
        "authorization_response_iss_parameter_supported": True,
    }
    doc.update(extra)
    return doc


def test_template_matches_only_the_two_entra_multi_tenant_endpoints():
    from tools.mcp_oauth_device import DeviceOAuthMetadata
    from tools.mcp_oauth_provider import entra_issuer_template_matches

    def meta(issuer):
        # Model-validated so pydantic percent-encodes the braces exactly as on the real wire
        # (str(AnyHttpUrl) spells the template ".../%7Btenantid%7D/v2.0").
        return DeviceOAuthMetadata.model_validate(
            _asm_doc(
                issuer,
                device_authorization_endpoint=f"{ENTRA}/devicecode",
                grant_types_supported=[
                    "authorization_code",
                    "refresh_token",
                    "urn:ietf:params:oauth:grant-type:device_code",
                ],
            )
        )

    for advertised in (ORGANIZATIONS, COMMON, f"{ORGANIZATIONS}/"):
        assert entra_issuer_template_matches(meta(TEMPLATE_ISSUER), advertised), (
            advertised
        )
    # The percent-encoded spelling pydantic produces matches too.
    assert entra_issuer_template_matches(
        meta(f"{ENTRA}/%7Btenantid%7D/v2.0"), ORGANIZATIONS
    )

    for advertised, issuer in (
        (
            f"{ENTRA}/{TENANT_GUID}/v2.0",
            TEMPLATE_ISSUER,
        ),  # tenant-specific endpoint is not the template
        (ORGANIZATIONS, f"{ENTRA}/{{tenantid2}}/v2.0"),  # a different placeholder
        (ORGANIZATIONS, f"{ENTRA}/{{tenantid}}/v1"),  # a different path shape
        (ORGANIZATIONS, f"https://evil.example/{{tenantid}}/v2.0"),  # another host
        (
            f"https://{ENTRA.split('//')[1]}.evil.example/v2.0",
            TEMPLATE_ISSUER,
        ),  # lookalike advertised host
        (
            ORGANIZATIONS,
            f"https://login.microsoftonline.com:8443/{{tenantid}}/v2.0",
        ),  # another port
        (None, TEMPLATE_ISSUER),
        (ORGANIZATIONS, None),
    ):
        assert not entra_issuer_template_matches(
            meta(issuer) if issuer else SimpleNamespace(issuer=None), advertised
        ), (advertised, issuer)


def test_template_matches_tolerates_plain_metadata_objects():
    from tools.mcp_oauth_provider import entra_issuer_template_matches

    assert entra_issuer_template_matches(
        SimpleNamespace(issuer=TEMPLATE_ISSUER), ORGANIZATIONS
    )
    assert not entra_issuer_template_matches(
        SimpleNamespace(issuer=None), ORGANIZATIONS
    )


@pytest.mark.asyncio
async def test_device_flow_accepts_the_entra_template_document():
    from mcp.client.auth.exceptions import OAuthFlowError
    from tools.mcp_oauth_device import DeviceOAuthMetadata, _device_metadata
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()

    for advertised in (ORGANIZATIONS, COMMON):

        def handler(request, advertised=advertised):
            path = urlsplit(str(request.url)).path
            doc_path = (
                f"/.well-known/oauth-authorization-server{urlsplit(advertised).path}"
            )
            if path == doc_path:
                return httpx.Response(
                    200,
                    json=_asm_doc(
                        TEMPLATE_ISSUER,
                        device_authorization_endpoint=f"{ENTRA}/devicecode",
                        grant_types_supported=[
                            "authorization_code",
                            "refresh_token",
                            "urn:ietf:params:oauth:grant-type:device_code",
                        ],
                    ),
                    request=request,
                )
            return httpx.Response(404, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert isinstance(
                await _device_metadata(client, RESOURCE, advertised),
                DeviceOAuthMetadata,
            )

    # The boundary: another host claiming the template shape, and a template issuer on an
    # advertised server that is not one of the two multi-tenant endpoints, still fail.
    for advertised, issuer in (
        (ORGANIZATIONS, "https://evil.example/{tenantid}/v2.0"),
        (f"{ENTRA}/{TENANT_GUID}/v2.0", TEMPLATE_ISSUER),
    ):

        def handler(request, issuer=issuer, advertised=advertised):
            path = urlsplit(str(request.url)).path
            doc_path = (
                f"/.well-known/oauth-authorization-server{urlsplit(advertised).path}"
            )
            if path == doc_path:
                return httpx.Response(
                    200,
                    json=_asm_doc(
                        issuer, device_authorization_endpoint=f"{ENTRA}/devicecode"
                    ),
                    request=request,
                )
            return httpx.Response(404, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(OAuthFlowError):
                await _device_metadata(client, RESOURCE, advertised)


class _StandIn:
    """Resource + Entra authorization server behind one httpx MockTransport."""

    def __init__(self, httpx, asm_doc):
        self.httpx, self.asm_doc, self.hits = httpx, asm_doc, []

    def __call__(self, request):
        url = str(request.url)
        path = urlsplit(url).path
        self.hits.append((request.method, url))
        j = lambda status, body, **h: self.httpx.Response(
            status, json=body, headers=h, request=request
        )  # noqa: E731
        if url == RESOURCE:
            if request.headers.get("Authorization") == "Bearer AT-1":
                return j(200, {"ok": True})
            return j(
                401,
                {},
                **{
                    "WWW-Authenticate": 'Bearer resource_metadata="https://sentinel.example/.well-known/oauth-protected-resource/mcp/triage"'
                },
            )
        if (
            url
            == "https://sentinel.example/.well-known/oauth-protected-resource/mcp/triage"
        ):
            return j(
                200, {"resource": RESOURCE, "authorization_servers": [ORGANIZATIONS]}
            )
        if url.startswith(ENTRA) and path == ASM_PATH:
            return j(200, self.asm_doc)
        if url == f"{ENTRA}/register":
            return j(
                201,
                {
                    "client_id": "dcr-1",
                    "redirect_uris": ["http://127.0.0.1:1/cb"],
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                },
            )
        if url == f"{ENTRA}/token":
            return j(
                200,
                {
                    "access_token": "AT-1",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": "RT-1",
                },
            )
        return j(404, {})


async def _run_browser_flow(tmp_path, monkeypatch, *, asm_issuer, redirect_iss):
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
        return _authorization_code_result("code-1", seen["state"], iss=redirect_iss)

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url=RESOURCE,
        storage=HermesTokenStorage("srv"),
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")], client_name="Hermes Agent"
        ),
        redirect_handler=redirect,
        callback_handler=callback,
    )
    standin = _StandIn(httpx, _asm_doc(asm_issuer))
    async with httpx.AsyncClient(
        auth=provider, transport=httpx.MockTransport(standin)
    ) as client:
        response = await client.get(RESOURCE)
    return response, standin, seen, provider


@pytest.mark.asyncio
async def test_entra_template_document_completes_the_browser_flow(
    tmp_path, monkeypatch
):
    import json

    # Entra's redirect carries the concrete tenant endpoint as iss; the discovered issuer is the template.
    response, standin, seen, provider = await _run_browser_flow(
        tmp_path,
        monkeypatch,
        asm_issuer=TEMPLATE_ISSUER,
        redirect_iss=f"{ENTRA}/{TENANT_GUID}/v2.0",
    )

    assert response.status_code == 200
    assert seen["authorize_url"].startswith(f"{ENTRA}/authorize?")
    assert ("POST", f"{ENTRA}/register") in standin.hits and (
        "POST",
        f"{ENTRA}/token",
    ) in standin.hits
    # The installed document keeps Entra's template issuer (percent-encoded by pydantic).
    assert str(provider.context.oauth_metadata.issuer) == f"{ENTRA}/%7Btenantid%7D/v2.0"
    # SEP-2352 binding stays on the advertised identifier, so the next 401 reuses this client.
    assert (
        json.loads((tmp_path / "mcp-tokens" / "srv.client.json").read_text())["issuer"]
        == ORGANIZATIONS
    )
    assert (
        json.loads((tmp_path / "mcp-tokens" / "srv.json").read_text())["access_token"]
        == "AT-1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "asm_issuer,redirect_iss",
    [
        pytest.param(
            "https://evil.example/{tenantid}/v2.0",
            f"{ENTRA}/{TENANT_GUID}/v2.0",
            id="another-host-template",
        ),
        pytest.param(
            f"{ENTRA}/{{tenantid}}/v1",
            f"{ENTRA}/{TENANT_GUID}/v2.0",
            id="other-path-shape",
        ),
        pytest.param(
            TEMPLATE_ISSUER,
            "https://evil.example/v2.0",
            id="redirect-iss-names-another-host",
        ),
    ],
)
async def test_other_entra_like_shapes_are_still_rejected(
    tmp_path, monkeypatch, asm_issuer, redirect_iss
):
    from mcp.client.auth.exceptions import OAuthFlowError, OAuthRegistrationError

    with pytest.raises((OAuthFlowError, OAuthRegistrationError)):
        await _run_browser_flow(
            tmp_path, monkeypatch, asm_issuer=asm_issuer, redirect_iss=redirect_iss
        )
    assert not (tmp_path / "mcp-tokens" / "srv.json").exists()

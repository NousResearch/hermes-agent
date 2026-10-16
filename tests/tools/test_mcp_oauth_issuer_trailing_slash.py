"""A host-only authorization server whose RFC 8414 issuer spelling lacks the trailing slash (#132233).

Otter.ai advertises ``authorization_servers: ["https://otter.ai"]`` and serves
``/.well-known/oauth-authorization-server`` with ``issuer: "https://otter.ai"``. The SDK compares issuers
as exact strings (RFC 8414 §3.3) against the advertised identifier, which reaches the comparison as
``str(AnyHttpUrl)`` — normalized to the root path, ``"https://otter.ai/"`` — so the document's slash-less
spelling was rejected and OAuth discovery never completed. Hermes compares both sides root-slash-normalized
(``issuer_matches_modulo_trailing_slash``), the same convention the device flow, ``_metadata_issuer`` and
the refresh-token issuer binding already use; any other difference still fails the SDK's exact-string check.
"""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

RESOURCE = "https://res.example/mcp"
AS_ORIGIN = "https://otter.example"
ADVERTISED = AS_ORIGIN  # host-only identifier, no path
PATH_DOC = "/.well-known/oauth-authorization-server"


def _asm(issuer):
    return {
        "issuer": issuer,
        "authorization_endpoint": f"{AS_ORIGIN}/authorize",
        "token_endpoint": f"{AS_ORIGIN}/token",
        "registration_endpoint": f"{AS_ORIGIN}/register",
        "response_types_supported": ["code"],
        "code_challenge_methods_supported": ["S256"],
        "authorization_response_iss_parameter_supported": True,
    }


def test_matcher_accepts_only_trailing_slash_differences():
    from tools.mcp_oauth_provider import issuer_matches_modulo_trailing_slash

    def meta(issuer):
        return SimpleNamespace(issuer=issuer)

    for advertised, issued in [
        (
            "https://otter.example/",
            "https://otter.example",
        ),  # the #132233 shape: pydantic root-path vs document
        ("https://otter.example/", "https://otter.example/"),
        ("https://otter.example", "https://otter.example"),
        (
            "https://otter.example/as/",
            "https://otter.example/as",
        ),  # path-scoped identifier, slash-less document
    ]:
        assert issuer_matches_modulo_trailing_slash(meta(issued), advertised), (
            advertised,
            issued,
        )
    for advertised, issued in [
        (
            "https://otter.example/",
            "https://evil.example",
        ),  # another origin: the RFC 8414 §3.3 boundary
        ("https://otter.example/", "http://otter.example"),  # scheme downgrade
        ("https://otter.example/", "https://otter.example:8443"),  # another port
        (
            "https://otter.example/",
            "https://otter.example/oauth",
        ),  # issuer names a different path
        ("", "https://otter.example"),
        (None, "https://otter.example"),
        ("https://otter.example/", None),
    ]:
        assert not issuer_matches_modulo_trailing_slash(meta(issued), advertised), (
            advertised,
            issued,
        )


class _StandIn:
    """Resource + authorization server behind one httpx MockTransport; ``issuer_doc`` maps ASM path -> document."""

    def __init__(self, httpx, issuer_doc):
        self.httpx, self.issuer_doc, self.hits = httpx, issuer_doc, []

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
                    "WWW-Authenticate": 'Bearer resource_metadata="https://res.example/.well-known/oauth-protected-resource"'
                },
            )
        if url == "https://res.example/.well-known/oauth-protected-resource":
            return j(200, {"resource": RESOURCE, "authorization_servers": [ADVERTISED]})
        if url.startswith(AS_ORIGIN) and path in self.issuer_doc:
            return j(200, self.issuer_doc[path])
        if url == f"{AS_ORIGIN}/register":
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
        if url == f"{AS_ORIGIN}/token":
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


async def _run_flow(tmp_path, monkeypatch, issuer_doc):
    import json

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

    # A real authorization server echoes its own document's issuer spelling in the RFC 9207
    # ``iss`` parameter, so the stand-in keeps the two consistent per document.
    async def callback():
        return _authorization_code_result(
            "code-1", seen["state"], iss=seen["document_issuer"]
        )

    storage = HermesTokenStorage("srv")
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url=RESOURCE,
        storage=storage,
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:1/cb")], client_name="Hermes Agent"
        ),
        redirect_handler=redirect,
        callback_handler=callback,
    )
    seen["document_issuer"] = str(next(iter(issuer_doc.values()))["issuer"])
    standin = _StandIn(httpx, issuer_doc)
    async with httpx.AsyncClient(
        auth=provider, transport=httpx.MockTransport(standin)
    ) as client:
        response = await client.get(RESOURCE)
    return response, standin, seen, provider, json


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document_issuer",
    [
        pytest.param(f"{AS_ORIGIN}", id="document-without-slash"),
        pytest.param(f"{AS_ORIGIN}/", id="document-with-slash"),
    ],
)
async def test_host_only_issuer_completes_the_flow(
    tmp_path, monkeypatch, document_issuer
):
    """The #132233 shape: the advertised identifier and the document keep their own spellings and differ
    only by the trailing slash; discovery must complete for either spelling."""
    response, standin, seen, provider, json = await _run_flow(
        tmp_path, monkeypatch, {PATH_DOC: _asm(document_issuer)}
    )

    assert response.status_code == 200
    assert seen["authorize_url"].startswith(f"{AS_ORIGIN}/authorize?")
    assert ("POST", f"{AS_ORIGIN}/register") in standin.hits and (
        "POST",
        f"{AS_ORIGIN}/token",
    ) in standin.hits
    assert str(provider.context.oauth_metadata.issuer).rstrip("/") == AS_ORIGIN
    # SEP-2352 binding stays on the advertised identifier, so the next 401 reuses this client instead of re-registering.
    assert (
        json.loads((tmp_path / "mcp-tokens" / "srv.client.json").read_text())["issuer"]
        == AS_ORIGIN
    )
    assert (
        json.loads((tmp_path / "mcp-tokens" / "srv.json").read_text())["access_token"]
        == "AT-1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document_issuer",
    [
        pytest.param("https://evil.example", id="different-origin"),
        pytest.param("http://otter.example", id="scheme-downgrade"),
        pytest.param("https://otter.example:8443", id="different-port"),
        pytest.param(f"{AS_ORIGIN}/oauth", id="issuer-adds-a-path"),
    ],
)
async def test_other_issuer_differences_are_still_rejected(
    tmp_path, monkeypatch, document_issuer
):
    from mcp.client.auth.exceptions import OAuthFlowError, OAuthRegistrationError

    with pytest.raises((OAuthFlowError, OAuthRegistrationError)):
        await _run_flow(tmp_path, monkeypatch, {PATH_DOC: _asm(document_issuer)})
    assert not (tmp_path / "mcp-tokens" / "srv.json").exists()

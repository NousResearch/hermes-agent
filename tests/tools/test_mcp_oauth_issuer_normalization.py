"""Regression coverage for authorization-server issuer URL normalization.

Some providers publish the root issuer as ``https://host/`` in one discovery
response and ``https://host`` in another. Hermes may normalize that one root
path representation, but must keep path-based issuers strict.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("mcp.client.auth.oauth2")

from mcp.client.auth.utils import handle_auth_metadata_response, validate_metadata_issuer  # noqa: E402
from mcp.client.auth.exceptions import OAuthFlowError  # noqa: E402

from tools.mcp_oauth_provider import HermesProviderMixin, _canonicalize_issuer  # noqa: E402


class _Provider(HermesProviderMixin):
    pass


def _provider(expected_issuer: str) -> _Provider:
    provider = _Provider.__new__(_Provider)
    provider.context = SimpleNamespace(auth_server_url=expected_issuer)
    return provider


def _response(request_url: str, issuer: str) -> tuple[httpx.Request, httpx.Response]:
    request = httpx.Request("GET", request_url)
    response = httpx.Response(
        200,
        request=request,
        json={
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "response_types_supported": ["code"],
        },
    )
    return request, response


def test_root_issuer_slash_variants_pass_sdk_validation():
    provider = _provider("https://login.semgrep.dev/")
    request, response = _response(
        "https://login.semgrep.dev/.well-known/oauth-authorization-server",
        "https://login.semgrep.dev",
    )

    normalized = asyncio.run(provider._normalize_metadata_response(request, response))
    ok, metadata = asyncio.run(handle_auth_metadata_response(normalized))

    assert ok and metadata is not None
    assert provider.context.auth_server_url == "https://login.semgrep.dev"
    validate_metadata_issuer(metadata, provider.context.auth_server_url)


def test_different_issuer_remains_rejected():
    provider = _provider("https://login.semgrep.dev/")
    request, response = _response(
        "https://login.semgrep.dev/.well-known/oauth-authorization-server",
        "https://evil.example",
    )

    normalized = asyncio.run(provider._normalize_metadata_response(request, response))
    _, metadata = asyncio.run(handle_auth_metadata_response(normalized))

    assert metadata is not None
    with pytest.raises(OAuthFlowError):
        validate_metadata_issuer(metadata, provider.context.auth_server_url)


def test_path_based_issuer_slash_remains_strict():
    assert _canonicalize_issuer("https://auth.example.com/tenant/") == "https://auth.example.com/tenant/"

    provider = _provider("https://auth.example.com/tenant")
    request, response = _response(
        "https://auth.example.com/.well-known/oauth-authorization-server/tenant",
        "https://auth.example.com/tenant/",
    )
    normalized = asyncio.run(provider._normalize_metadata_response(request, response))
    _, metadata = asyncio.run(handle_auth_metadata_response(normalized))

    assert metadata is not None
    with pytest.raises(OAuthFlowError):
        validate_metadata_issuer(metadata, provider.context.auth_server_url)


def test_protected_resource_metadata_response_is_not_rewritten():
    provider = _provider("https://login.semgrep.dev/")
    request = httpx.Request("GET", "https://mcp.example/.well-known/oauth-protected-resource")
    response = httpx.Response(
        200,
        request=request,
        content=json.dumps({"authorization_servers": ["https://login.semgrep.dev/"]}).encode(),
    )

    result = asyncio.run(provider._normalize_metadata_response(request, response))

    assert result is response
    assert asyncio.run(result.aread()) == response.content

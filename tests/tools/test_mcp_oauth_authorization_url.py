"""Regression tests for query-bearing MCP OAuth authorization endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

import pytest

from tools.mcp_oauth_manager import (
    _HERMES_PROVIDER_CLS,
    _merge_authorization_endpoint_query,
)


RAILWAY_ENDPOINT = (
    "https://backboard.railway.com/oauth/auth?"
    "resource=https%3A%2F%2Fbackboard.railway.com"
)


def test_merges_sdk_parameters_with_existing_endpoint_query() -> None:
    malformed = (
        f"{RAILWAY_ENDPOINT}?response_type=code&client_id=client-1&"
        "redirect_uri=http%3A%2F%2F127.0.0.1%3A1234%2Fcallback"
    )

    repaired = _merge_authorization_endpoint_query(malformed, RAILWAY_ENDPOINT)
    parsed = urlsplit(repaired)
    params = parse_qs(parsed.query)

    assert repaired.count("?") == 1
    assert params["resource"] == ["https://backboard.railway.com"]
    assert params["response_type"] == ["code"]
    assert params["client_id"] == ["client-1"]


@pytest.mark.parametrize(
    ("generated", "endpoint"),
    [
        (
            "https://idp.example/authorize?response_type=code",
            "https://idp.example/authorize",
        ),
        (
            "https://other.example/authorize?response_type=code",
            "https://idp.example/authorize?audience=reports",
        ),
        ("https://idp.example/authorize?response_type=code", None),
    ],
)
def test_leaves_non_matching_urls_unchanged(generated: str, endpoint: str | None) -> None:
    assert _merge_authorization_endpoint_query(generated, endpoint) == generated


@pytest.mark.asyncio
async def test_provider_repairs_url_before_desktop_redirect_handler() -> None:
    if _HERMES_PROVIDER_CLS is None:
        pytest.skip("MCP SDK OAuth support is unavailable")

    captured: list[str] = []

    async def original_redirect(url: str) -> None:
        captured.append(url)

    provider = object.__new__(_HERMES_PROVIDER_CLS)
    provider.context = SimpleNamespace(
        oauth_metadata=SimpleNamespace(authorization_endpoint=RAILWAY_ENDPOINT),
        redirect_handler=original_redirect,
    )

    async def fake_sdk_grant(_self):
        await _self.context.redirect_handler(
            f"{RAILWAY_ENDPOINT}?response_type=code&client_id=client-1"
        )
        return "authorization-code", "pkce-verifier"

    from mcp.client.auth.oauth2 import OAuthClientProvider

    with patch.object(
        OAuthClientProvider,
        "_perform_authorization_code_grant",
        fake_sdk_grant,
    ):
        result = await provider._perform_authorization_code_grant()

    assert result == ("authorization-code", "pkce-verifier")
    assert parse_qs(urlsplit(captured[0]).query)["response_type"] == ["code"]
    assert provider.context.redirect_handler is original_redirect

"""Tests for tools/microsoft_graph_auth.py."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from tools.microsoft_graph_auth import (
    CachedAccessToken,
    DEFAULT_GRAPH_SCOPE,
    GraphCredentials,
    MicrosoftGraphConfigError,
    MicrosoftGraphTokenError,
    MicrosoftGraphTokenProvider,
)


class TestGraphCredentials:
    def test_from_env_raises_for_missing_required_values(self):
        with pytest.raises(MicrosoftGraphConfigError) as exc:
            GraphCredentials.from_env({})
        assert "MSGRAPH_TENANT_ID" in str(exc.value)
        assert "MSGRAPH_CLIENT_ID" in str(exc.value)
        assert "MSGRAPH_CLIENT_SECRET" in str(exc.value)

    def test_from_env_optional_returns_none_when_not_configured(self):
        assert GraphCredentials.from_env({}, required=False) is None

    def test_from_env_builds_normalized_credentials(self):
        creds = GraphCredentials.from_env(
            {
                "MSGRAPH_TENANT_ID": "tenant-123",
                "MSGRAPH_CLIENT_ID": "client-456",
                "MSGRAPH_CLIENT_SECRET": "secret-789",
            }
        )
        assert creds is not None
        assert creds.scope == DEFAULT_GRAPH_SCOPE
        assert creds.token_url.endswith("/tenant-123/oauth2/v2.0/token")


@pytest.mark.anyio
class TestMicrosoftGraphTokenProvider:
    async def test_reuses_cached_token_until_expiry(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(
                200,
                json={
                    "access_token": f"token-{len(calls)}",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
        )

        first = await provider.get_access_token()
        second = await provider.get_access_token()

        assert first == "token-1"
        assert second == "token-1"
        assert len(calls) == 1

    async def test_concurrent_calls_share_one_token_fetch(self):
        calls: list[int] = []

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
        )

        async def _fake_fetch():
            calls.append(1)
            await asyncio.sleep(0)
            return CachedAccessToken(
                access_token="token-1",
                token_type="Bearer",
                expires_at=9_999_999_999,
            )

        provider._fetch_access_token = _fake_fetch  # type: ignore[method-assign]

        first, second = await asyncio.gather(
            provider.get_access_token(),
            provider.get_access_token(),
        )

        assert first == "token-1"
        assert second == "token-1"
        assert len(calls) == 1


    async def test_http_error_includes_server_message(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"error": "invalid_client", "error_description": "bad secret"},
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
        )

        with pytest.raises(MicrosoftGraphTokenError) as exc:
            await provider.get_access_token()
        assert "bad secret" in str(exc.value)

    async def test_oversized_token_response_body_is_rejected(self):
        """#54974: a hostile / proxy-interposed token endpoint must not be allowed to
        buffer an unbounded response body.  ``_fetch_access_token`` enforces a
        ``_MSGRAPH_TOKEN_RESPONSE_MAX_BYTES`` cap; an over-cap body raises
        ``MicrosoftGraphTokenError`` instead of consuming the unbounded bytes.
        """
        from tools.microsoft_graph_auth import _MSGRAPH_TOKEN_RESPONSE_MAX_BYTES

        cap = _MSGRAPH_TOKEN_RESPONSE_MAX_BYTES
        oversized = b"x" * (cap + 1)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=oversized)

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError) as exc:
            await provider.get_access_token()
        assert "exceeds cap" in str(exc.value)

    async def test_oversized_error_response_body_is_capped_in_diagnostic(self):
        """#54974: same body cap applies to error-path parsing so a hostile error body
        can't smuggle an unbounded response past the helper either.  ``_extract_error_detail``
        returns a cap-notice string instead of attempting to JSON-decode the oversize body."""
        from tools.microsoft_graph_auth import _MSGRAPH_TOKEN_RESPONSE_MAX_BYTES

        cap = _MSGRAPH_TOKEN_RESPONSE_MAX_BYTES
        oversized = b"x" * (cap + 1)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, content=oversized)

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError) as exc:
            await provider.get_access_token()
        # ``_extract_error_detail`` returns a cap-notice, and ``_fetch_access_token`` wraps it
        # with the HTTP status code prefix.
        assert f"{cap}-byte cap" in str(exc.value)

    async def test_small_token_response_body_is_accepted_unchanged(self):
        """Sanity: small (under-cap) payloads are unaffected by the cap."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "access_token": "real-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
        )
        token = await provider.get_access_token()
        assert token == "real-token"

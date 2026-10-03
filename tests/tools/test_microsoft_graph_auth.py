"""Tests for tools/microsoft_graph_auth.py."""

from __future__ import annotations

import asyncio
import gzip

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


class _CountingByteStream(httpx.AsyncByteStream):
    """Offers ``chunks`` copies of ``chunk`` and counts the bytes the reader actually pulls,
    so a test can prove an over-cap body is abandoned mid-stream instead of buffered."""

    def __init__(self, chunk: bytes, chunks: int, counter: list[int]) -> None:
        self._chunk, self._chunks, self._counter = chunk, chunks, counter

    async def __aiter__(self):
        for _ in range(self._chunks):
            self._counter[0] += len(self._chunk)
            yield self._chunk

    async def aclose(self) -> None:
        return None


class _UnreadableByteStream(httpx.AsyncByteStream):
    """Fails the read: a response whose advertised length is already over the cap must never
    have its body read at all."""

    async def __aiter__(self):
        raise AssertionError("body was read despite an advertised Content-Length over the cap")
        yield b""  # pragma: no cover - unreachable, keeps this an async generator

    async def aclose(self) -> None:
        return None


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
        ``_MSGRAPH_TOKEN_RESPONSE_MAX_BYTES`` cap that gates the read itself; an over-cap body
        raises ``MicrosoftGraphTokenError`` instead of pulling the bytes in and measuring them.
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

    async def test_oversized_streaming_body_is_not_buffered_in_full(self):
        """#54974: the cap has to bound memory *during* the read.  A 4 MiB stream (64x the cap)
        must be abandoned as soon as it passes the cap, not materialised and then measured."""
        from tools.microsoft_graph_auth import _MSGRAPH_TOKEN_RESPONSE_MAX_BYTES

        cap = _MSGRAPH_TOKEN_RESPONSE_MAX_BYTES
        chunk, chunks, pulled = b"x" * 4096, 1024, [0]  # 4 MiB offered by the endpoint

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, stream=_CountingByteStream(chunk, chunks, pulled), request=request
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError) as exc:
            await provider.get_access_token()
        assert "exceeds cap" in str(exc.value)
        assert pulled[0] <= cap + len(chunk), (
            f"read {pulled[0]} bytes of a {len(chunk) * chunks}-byte body; cap is {cap}"
        )

    async def test_advertised_oversized_content_length_is_rejected_without_reading(self):
        """#54974: an honest ``Content-Length`` over the cap is refused before a body byte is
        read (the stream in this test fails if anyone iterates it)."""
        from tools.microsoft_graph_auth import _MSGRAPH_TOKEN_RESPONSE_MAX_BYTES

        cap = _MSGRAPH_TOKEN_RESPONSE_MAX_BYTES

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-length": str(cap + 1)},
                stream=_UnreadableByteStream(),
                request=request,
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError) as exc:
            await provider.get_access_token()
        assert "exceeds cap" in str(exc.value)

    async def test_cap_applies_after_decompression(self):
        """#54974: the cap counts decoded bytes.  A gzip body that is tiny on the wire but
        inflates past the cap is rejected at its inflated size, not waved through."""
        from tools.microsoft_graph_auth import _MSGRAPH_TOKEN_RESPONSE_MAX_BYTES

        cap = _MSGRAPH_TOKEN_RESPONSE_MAX_BYTES
        inflated = b"x" * (cap * 4)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=gzip.compress(inflated),
                headers={"content-encoding": "gzip"},
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError) as exc:
            await provider.get_access_token()
        assert "exceeds cap" in str(exc.value)

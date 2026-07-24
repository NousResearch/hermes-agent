"""Tests for tools/microsoft_graph_auth.py."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from tools.microsoft_graph_auth import (
    CachedAccessToken,
    DEFAULT_GRAPH_SCOPE,
    DEFAULT_GRAPH_AUTHORITY_URL,
    DEFAULT_TOKEN_SKEW_SECONDS,
    GraphCredentials,
    MicrosoftGraphAuthError,
    MicrosoftGraphConfigError,
    MicrosoftGraphTokenError,
    MicrosoftGraphTokenProvider,
    _extract_error_detail,
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
        creds = GraphCredentials.from_env({
            "MSGRAPH_TENANT_ID": "tenant-123",
            "MSGRAPH_CLIENT_ID": "client-456",
            "MSGRAPH_CLIENT_SECRET": "secret-789",
        })
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

    async def test_refreshes_when_cached_token_is_expired(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            expires_in = 0 if len(calls) == 1 else 3600
            return httpx.Response(
                200,
                json={
                    "access_token": f"token-{len(calls)}",
                    "expires_in": expires_in,
                    "token_type": "Bearer",
                },
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
            skew_seconds=0,
        )

        first = await provider.get_access_token()
        second = await provider.get_access_token()

        assert first == "token-1"
        assert second == "token-2"
        assert len(calls) == 2

    async def test_force_refresh_bypasses_cache(self):
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(
                200,
                json={
                    "access_token": f"token-{len(calls)}",
                    "expires_in": 3600,
                },
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
        )

        first = await provider.get_access_token()
        second = await provider.get_access_token(force_refresh=True)

        assert first == "token-1"
        assert second == "token-2"
        assert len(calls) == 2

    async def test_invalid_token_response_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"expires_in": 3600})

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("tenant", "client", "secret"),
            transport=httpx.MockTransport(handler),
        )

        with pytest.raises(MicrosoftGraphTokenError) as exc:
            await provider.get_access_token()
        assert "access_token" in str(exc.value)

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


# ---------------------------------------------------------------------------
# GraphCredentials — additional edge cases
# ---------------------------------------------------------------------------


class TestGraphCredentialsEdgeCases:
    def test_from_env_with_custom_scope_and_authority(self):
        creds = GraphCredentials.from_env({
            "MSGRAPH_TENANT_ID": "t",
            "MSGRAPH_CLIENT_ID": "c",
            "MSGRAPH_CLIENT_SECRET": "s",
            "MSGRAPH_SCOPE": "custom-scope",
            "MSGRAPH_AUTHORITY_URL": "https://login.example.com/",
        })
        assert creds is not None
        assert creds.scope == "custom-scope"
        assert creds.authority_url == "https://login.example.com/"
        assert creds.token_url == "https://login.example.com/t/oauth2/v2.0/token"

    def test_from_env_strips_whitespace(self):
        creds = GraphCredentials.from_env({
            "MSGRAPH_TENANT_ID": "  t  ",
            "MSGRAPH_CLIENT_ID": "  c  ",
            "MSGRAPH_CLIENT_SECRET": "  s  ",
            "MSGRAPH_SCOPE": "  scope  ",
            "MSGRAPH_AUTHORITY_URL": "  https://login.example.com/  ",
        })
        assert creds is not None
        assert creds.tenant_id == "t"
        assert creds.client_id == "c"
        assert creds.client_secret == "s"
        assert creds.scope == "scope"
        assert creds.authority_url == "https://login.example.com/"

    def test_from_env_partial_missing_lists_only_missing(self):
        with pytest.raises(MicrosoftGraphConfigError) as exc:
            GraphCredentials.from_env({
                "MSGRAPH_TENANT_ID": "t",
                "MSGRAPH_CLIENT_ID": "c",
            })
        msg = str(exc.value)
        assert "MSGRAPH_CLIENT_SECRET" in msg
        assert "MSGRAPH_TENANT_ID" not in msg
        assert "MSGRAPH_CLIENT_ID" not in msg

    def test_from_env_uses_os_environ_by_default(self, monkeypatch):
        monkeypatch.setenv("MSGRAPH_TENANT_ID", "env-t")
        monkeypatch.setenv("MSGRAPH_CLIENT_ID", "env-c")
        monkeypatch.setenv("MSGRAPH_CLIENT_SECRET", "env-s")
        creds = GraphCredentials.from_env()
        assert creds is not None
        assert creds.tenant_id == "env-t"

    def test_token_url_strips_trailing_slash_from_authority(self):
        creds = GraphCredentials(
            tenant_id="t",
            client_id="c",
            client_secret="s",
            authority_url="https://login.example.com///",
        )
        assert creds.token_url == "https://login.example.com/t/oauth2/v2.0/token"

    def test_token_url_strips_slashes_from_tenant(self):
        creds = GraphCredentials(
            tenant_id="  /t/  ",
            client_id="c",
            client_secret="s",
        )
        assert "/t/oauth2/v2.0/token" in creds.token_url

    def test_is_frozen(self):
        creds = GraphCredentials("t", "c", "s")
        with pytest.raises((AttributeError, TypeError)):
            creds.tenant_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CachedAccessToken
# ---------------------------------------------------------------------------


class TestCachedAccessToken:
    def test_is_expired_true_when_past_expiry(self):
        import time

        token = CachedAccessToken(access_token="tok", expires_at=time.time() - 100)
        assert token.is_expired() is True

    def test_is_expired_false_when_well_before_expiry(self):
        import time

        token = CachedAccessToken(access_token="tok", expires_at=time.time() + 3600)
        assert token.is_expired() is False

    def test_is_expired_true_when_within_skew(self):
        import time

        token = CachedAccessToken(access_token="tok", expires_at=time.time() + 60)
        assert token.is_expired(skew_seconds=120) is True

    def test_is_expired_with_zero_skew(self):
        import time

        token = CachedAccessToken(access_token="tok", expires_at=time.time() + 60)
        assert token.is_expired(skew_seconds=0) is False

    def test_is_expired_negative_skew_clamped(self):
        import time

        token = CachedAccessToken(access_token="tok", expires_at=time.time() + 60)
        assert token.is_expired(skew_seconds=-100) is False

    def test_expires_in_seconds_positive(self):
        import time

        token = CachedAccessToken(access_token="tok", expires_at=time.time() + 3600)
        assert token.expires_in_seconds > 3500
        assert token.expires_in_seconds <= 3600

    def test_expires_in_seconds_zero_when_expired(self):
        import time

        token = CachedAccessToken(access_token="tok", expires_at=time.time() - 100)
        assert token.expires_in_seconds == 0

    def test_default_token_type_is_bearer(self):
        import time

        token = CachedAccessToken(access_token="tok", expires_at=time.time() + 100)
        assert token.token_type == "Bearer"


# ---------------------------------------------------------------------------
# MicrosoftGraphTokenProvider — additional methods
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestMicrosoftGraphTokenProviderMethods:
    async def test_from_env_classmethod(self):
        provider = MicrosoftGraphTokenProvider.from_env(
            {
                "MSGRAPH_TENANT_ID": "t",
                "MSGRAPH_CLIENT_ID": "c",
                "MSGRAPH_CLIENT_SECRET": "s",
            },
            timeout=10.0,
        )
        assert provider.credentials.tenant_id == "t"
        assert provider.timeout == 10.0

    async def test_clear_cache(self):
        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
        )
        provider._cached_token = CachedAccessToken(
            access_token="tok",
            expires_at=9999999999,
        )
        assert provider._cached_token is not None
        provider.clear_cache()
        assert provider._cached_token is None

    async def test_inspect_token_health_without_cache(self):
        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
        )
        health = provider.inspect_token_health()
        assert health["configured"] is True
        assert health["tenant_id"] == "t"
        assert health["client_id"] == "c"
        assert health["cached"] is False
        assert health["expires_in_seconds"] is None
        assert health["is_expired"] is None
        assert health["refresh_skew_seconds"] == DEFAULT_TOKEN_SKEW_SECONDS

    async def test_inspect_token_health_with_cache(self):
        import time

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
        )
        provider._cached_token = CachedAccessToken(
            access_token="tok",
            expires_at=time.time() + 3600,
        )
        health = provider.inspect_token_health()
        assert health["cached"] is True
        assert health["expires_in_seconds"] > 3500
        assert health["is_expired"] is False

    async def test_inspect_token_health_with_expired_cache(self):
        import time

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
        )
        provider._cached_token = CachedAccessToken(
            access_token="tok",
            expires_at=time.time() - 100,
        )
        health = provider.inspect_token_health()
        assert health["cached"] is True
        assert health["is_expired"] is True

    async def test_negative_skew_clamped_to_zero(self):
        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            skew_seconds=-100,
        )
        assert provider.skew_seconds == 0


# ---------------------------------------------------------------------------
# MicrosoftGraphTokenProvider — token fetch error paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestTokenFetchErrorPaths:
    async def test_invalid_json_response_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=b"not json", headers={"content-type": "application/json"}
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError, match="not valid JSON"):
            await provider.get_access_token()

    async def test_missing_expires_in_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access_token": "tok"})

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError, match="valid expires_in"):
            await provider.get_access_token()

    async def test_non_integer_expires_in_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"access_token": "tok", "expires_in": "not-a-number"}
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError, match="valid expires_in"):
            await provider.get_access_token()

    async def test_none_expires_in_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access_token": "tok", "expires_in": None})

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError, match="valid expires_in"):
            await provider.get_access_token()

    async def test_empty_access_token_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access_token": "", "expires_in": 3600})

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError, match="access_token"):
            await provider.get_access_token()

    async def test_whitespace_access_token_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access_token": "   ", "expires_in": 3600})

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError, match="access_token"):
            await provider.get_access_token()

    async def test_custom_token_type_preserved(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "access_token": "tok",
                    "expires_in": 3600,
                    "token_type": "Custom",
                },
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        token = await provider.get_access_token()
        assert token == "tok"
        assert provider._cached_token.token_type == "Custom"  # type: ignore[union-attr]

    async def test_empty_token_type_defaults_to_bearer(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"access_token": "tok", "expires_in": 3600, "token_type": ""}
            )

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        await provider.get_access_token()
        assert provider._cached_token.token_type == "Bearer"  # type: ignore[union-attr]

    async def test_negative_expires_in_clamped_to_zero(self):
        import time

        before = time.time()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access_token": "tok", "expires_in": -100})

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        await provider.get_access_token()
        assert provider._cached_token.expires_at <= time.time() + 1  # type: ignore[union-attr]

    async def test_http_500_error_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError, match="HTTP 500"):
            await provider.get_access_token()

    async def test_http_400_with_json_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "invalid_request"})

        provider = MicrosoftGraphTokenProvider(
            GraphCredentials("t", "c", "s"),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphTokenError, match="invalid_request"):
            await provider.get_access_token()


def import_time():
    import time

    return time.time()


# ---------------------------------------------------------------------------
# _extract_error_detail
# ---------------------------------------------------------------------------


class TestExtractErrorDetail:
    def test_error_description_string(self):
        resp = httpx.Response(
            401, json={"error": "invalid_client", "error_description": "bad secret"}
        )
        assert _extract_error_detail(resp) == "bad secret"

    def test_error_dict_with_message_and_code(self):
        resp = httpx.Response(
            401, json={"error": {"code": "ErrorCode", "message": "error msg"}}
        )
        assert _extract_error_detail(resp) == "ErrorCode: error msg"

    def test_error_dict_with_message_only(self):
        resp = httpx.Response(401, json={"error": {"message": "just message"}})
        assert _extract_error_detail(resp) == "just message"

    def test_error_dict_with_code_only(self):
        resp = httpx.Response(401, json={"error": {"code": "CodeOnly"}})
        assert _extract_error_detail(resp) == "CodeOnly"

    def test_error_string(self):
        resp = httpx.Response(401, json={"error": "invalid_client"})
        assert _extract_error_detail(resp) == "invalid_client"

    def test_invalid_json_falls_back_to_text(self):
        resp = httpx.Response(
            401, content=b"plain text error", headers={"content-type": "text/plain"}
        )
        assert _extract_error_detail(resp) == "plain text error"

    def test_invalid_json_empty_text_returns_unknown(self):
        resp = httpx.Response(
            401, content=b"   ", headers={"content-type": "text/plain"}
        )
        assert _extract_error_detail(resp) == "unknown error"

    def test_non_dict_payload(self):
        resp = httpx.Response(401, json=["not", "a", "dict"])
        assert "not" in _extract_error_detail(resp)

    def test_dict_without_error_key(self):
        resp = httpx.Response(401, json={"other_key": "value"})
        result = _extract_error_detail(resp)
        assert "other_key" in result or "value" in result


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_config_error_is_auth_error(self):
        assert issubclass(MicrosoftGraphConfigError, MicrosoftGraphAuthError)

    def test_token_error_is_auth_error(self):
        assert issubclass(MicrosoftGraphTokenError, MicrosoftGraphAuthError)

    def test_auth_error_is_runtime_error(self):
        assert issubclass(MicrosoftGraphAuthError, RuntimeError)

    def test_config_error_and_token_error_distinct(self):
        assert MicrosoftGraphConfigError is not MicrosoftGraphTokenError

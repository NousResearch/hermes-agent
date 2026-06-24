"""Tests for tools/microsoft_graph_client.py."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tools.microsoft_graph_auth import GraphCredentials, MicrosoftGraphTokenProvider
from tools.microsoft_graph_client import (
    MicrosoftGraphAPIError,
    MicrosoftGraphClient,
    MicrosoftGraphClientError,
)


def _make_provider() -> MicrosoftGraphTokenProvider:
    provider = MicrosoftGraphTokenProvider(
        GraphCredentials("tenant", "client", "secret")
    )
    provider._cached_token = type(  # type: ignore[attr-defined]
        "Token",
        (),
        {
            "access_token": "cached-token",
            "is_expired": lambda self, skew_seconds=0: False,
            "expires_in_seconds": 3600,
        },
    )()
    return provider


@pytest.mark.anyio
class TestMicrosoftGraphClient:
    async def test_attaches_bearer_token_header(self):
        captured_auth: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_auth.append(request.headers["Authorization"])
            return httpx.Response(200, json={"ok": True})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        payload = await client.get_json("/me")
        assert payload == {"ok": True}
        assert captured_auth == ["Bearer cached-token"]

    async def test_retries_on_rate_limit_and_uses_retry_after(self):
        calls: list[int] = []
        sleeps: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(
                    429,
                    json={"error": {"code": "TooManyRequests", "message": "slow down"}},
                    headers={"Retry-After": "3"},
                )
            return httpx.Response(200, json={"ok": True})

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
            max_retries=2,
        )

        payload = await client.get_json("/me")

        assert payload == {"ok": True}
        assert len(calls) == 2
        assert sleeps == [3.0]

    async def test_raises_api_error_after_retry_budget_exhausted(self):
        sleeps: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
            max_retries=1,
        )

        with pytest.raises(MicrosoftGraphAPIError) as exc:
            await client.get_json("/me")
        assert exc.value.status_code == 503
        assert sleeps == [0.5]

    async def test_collect_paginated_flattens_value_arrays(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/items"):
                return httpx.Response(
                    200,
                    json={
                        "value": [{"id": "1"}],
                        "@odata.nextLink": "https://graph.microsoft.com/v1.0/items?page=2",
                    },
                )
            return httpx.Response(200, json={"value": [{"id": "2"}]})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        items = await client.collect_paginated("/items")
        assert items == [{"id": "1"}, {"id": "2"}]

    async def test_download_to_file_writes_binary_content(self, tmp_path: Path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"meeting-recording",
                headers={"content-type": "video/mp4"},
            )

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        destination = tmp_path / "recording.mp4"
        result = await client.download_to_file("/drive/item/content", destination)

        assert destination.read_bytes() == b"meeting-recording"
        assert result["content_type"] == "video/mp4"
        assert result["size_bytes"] == len(b"meeting-recording")

    async def test_download_to_file_streams_large_payload_in_chunks(
        self, tmp_path: Path, monkeypatch
    ):
        """Recordings can be hundreds of MB; verify the body is streamed.

        Uses a payload larger than the chunk size and counts how many
        ``aiter_bytes`` iterations the download loop performs. If the
        response were buffered in memory before the loop ran, only one
        non-empty chunk would be yielded.
        """
        payload = b"x" * (512 * 1024)  # 512 KiB

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=payload,
                headers={"content-type": "video/mp4"},
            )

        chunk_calls: list[int] = []
        original_aiter_bytes = httpx.Response.aiter_bytes

        async def counting_aiter_bytes(self, chunk_size: int | None = None):
            async for chunk in original_aiter_bytes(self, chunk_size):
                chunk_calls.append(len(chunk))
                yield chunk

        monkeypatch.setattr(httpx.Response, "aiter_bytes", counting_aiter_bytes)

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        destination = tmp_path / "big-recording.mp4"
        result = await client.download_to_file(
            "/drive/item/content", destination, chunk_size=65536
        )

        assert destination.read_bytes() == payload
        assert result["size_bytes"] == len(payload)
        assert len(chunk_calls) >= 2, (
            "Expected multiple chunks; got a single chunk "
            f"which suggests the body was buffered: {chunk_calls}"
        )
        assert not (tmp_path / "big-recording.mp4.part").exists()

    async def test_download_to_file_retries_on_transient_server_error(
        self, tmp_path: Path
    ):
        calls: list[int] = []
        sleeps: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(503, json={"error": {"message": "unavailable"}})
            return httpx.Response(
                200,
                content=b"payload",
                headers={"content-type": "application/octet-stream"},
            )

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
            max_retries=2,
        )
        destination = tmp_path / "artifact.bin"
        result = await client.download_to_file("/drive/item/content", destination)

        assert destination.read_bytes() == b"payload"
        assert result["size_bytes"] == len(b"payload")
        assert len(calls) == 2
        assert sleeps == [0.5]
        assert not (tmp_path / "artifact.bin.part").exists()

    async def test_download_to_file_cleans_partial_file_on_exhausted_retries(
        self, tmp_path: Path
    ):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})

        async def fake_sleep(delay: float) -> None:
            return None

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
            max_retries=1,
        )
        destination = tmp_path / "artifact.bin"

        with pytest.raises(MicrosoftGraphAPIError):
            await client.download_to_file("/drive/item/content", destination)

        assert not destination.exists()
        assert not (tmp_path / "artifact.bin.part").exists()

    async def test_invalid_json_response_raises_client_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"not-json",
                headers={"content-type": "application/json"},
            )

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )

        with pytest.raises(MicrosoftGraphClientError):
            await client.get_json("/me")


# ---------------------------------------------------------------------------
# from_env classmethod
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_from_env_raises_without_credentials(self, monkeypatch):
        monkeypatch.delenv("MSGRAPH_TENANT_ID", raising=False)
        monkeypatch.delenv("MSGRAPH_CLIENT_ID", raising=False)
        monkeypatch.delenv("MSGRAPH_CLIENT_SECRET", raising=False)
        from tools.microsoft_graph_auth import MicrosoftGraphConfigError

        with pytest.raises(MicrosoftGraphConfigError):
            MicrosoftGraphClient.from_env()

    def test_from_env_creates_client_with_credentials(self, monkeypatch):
        monkeypatch.setenv("MSGRAPH_TENANT_ID", "t")
        monkeypatch.setenv("MSGRAPH_CLIENT_ID", "c")
        monkeypatch.setenv("MSGRAPH_CLIENT_SECRET", "s")
        client = MicrosoftGraphClient.from_env()
        assert client.token_provider.credentials.tenant_id == "t"
        assert client.token_provider.credentials.client_id == "c"


# ---------------------------------------------------------------------------
# post_json / patch_json / delete
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestPostPatchDelete:
    async def test_post_json_sends_body(self):
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append({"method": request.method, "json": None})
            import json

            try:
                captured[-1]["json"] = json.loads(request.content)
            except Exception:
                pass
            return httpx.Response(201, json={"created": True})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        result = await client.post_json("/users", json_body={"name": "test"})
        assert result == {"created": True}
        assert captured[0]["method"] == "POST"
        assert captured[0]["json"] == {"name": "test"}

    async def test_post_json_no_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        result = await client.post_json("/endpoint")
        assert result == {"ok": True}

    async def test_patch_json_returns_decoded_json(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "PATCH"
            return httpx.Response(200, json={"updated": True})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        result = await client.patch_json("/users/1", json_body={"name": "new"})
        assert result == {"updated": True}

    async def test_patch_json_204_returns_empty_dict(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(204)

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        result = await client.patch_json("/users/1", json_body={"name": "new"})
        assert result == {}

    async def test_patch_json_empty_content_returns_empty_dict(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        result = await client.patch_json("/users/1", json_body={"name": "new"})
        assert result == {}

    async def test_delete_returns_dict_on_204(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "DELETE"
            return httpx.Response(204)

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        result = await client.delete("/users/1")
        assert result == {"deleted": True, "status_code": 204}

    async def test_delete_returns_json_on_200(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"deleted": True, "id": "123"})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        result = await client.delete("/users/1")
        assert result == {"deleted": True, "id": "123"}

    async def test_delete_empty_content_returns_dict(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        result = await client.delete("/users/1")
        assert result == {"deleted": True, "status_code": 200}


# ---------------------------------------------------------------------------
# iterate_pages — edge cases
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestIteratePages:
    async def test_iterate_pages_non_dict_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=["not", "a", "dict"])

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MicrosoftGraphClientError, match="Expected paginated"):
            async for _ in client.iterate_pages("/items"):
                pass

    async def test_iterate_pages_no_next_link_single_page(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"value": [{"id": "1"}]})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        pages = []
        async for page in client.iterate_pages("/items"):
            pages.append(page)
        assert len(pages) == 1
        assert pages[0]["value"] == [{"id": "1"}]

    async def test_iterate_pages_with_params(self):
        captured_params: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_params.append(dict(request.url.params))
            return httpx.Response(200, json={"value": []})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        async for _ in client.iterate_pages("/items", params={"$top": "10"}):
            pass
        assert captured_params[0]["$top"] == "10"

    async def test_collect_paginated_no_value_key(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"other": "data"})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        items = await client.collect_paginated("/items")
        assert items == []

    async def test_collect_paginated_value_not_list(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"value": "not-a-list"})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        items = await client.collect_paginated("/items")
        assert items == []


# ---------------------------------------------------------------------------
# _request — retry and error paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestRequestRetryPaths:
    async def test_retries_on_401_then_succeeds(self):
        calls: list[int] = []
        provider = _make_provider()

        async def mock_get_token(*, force_refresh=False):
            return "cached-token"

        provider.get_access_token = mock_get_token  # type: ignore[method-assign]

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(
                    401,
                    json={
                        "error": {"code": "Unauthorized", "message": "token expired"}
                    },
                )
            return httpx.Response(200, json={"ok": True})

        async def fake_sleep(delay: float) -> None:
            pass

        client = MicrosoftGraphClient(
            provider,
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
            max_retries=2,
        )
        result = await client.get_json("/me")
        assert result == {"ok": True}
        assert len(calls) == 2

    async def test_retries_on_500_then_raises_after_exhausted(self):
        sleeps: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": {"message": "server error"}})

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
            max_retries=1,
        )
        with pytest.raises(MicrosoftGraphAPIError) as exc:
            await client.get_json("/me")
        assert exc.value.status_code == 500
        assert len(sleeps) == 1

    async def test_http_error_retries_then_raises_client_error(self):
        sleeps: list[float] = []

        call_count: list[int] = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            call_count[0] += 1
            raise httpx.ConnectError("connection refused")

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
            max_retries=1,
        )
        with pytest.raises(MicrosoftGraphClientError, match="request failed"):
            await client.get_json("/me")
        assert len(sleeps) == 1

    async def test_content_type_header_set_on_post(self):
        captured: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request.headers.get("content-type", ""))
            return httpx.Response(200, json={"ok": True})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        await client.post_json("/me", json_body={"key": "val"})
        assert "application/json" in captured[0]

    async def test_custom_headers_merged(self):
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(dict(request.headers))
            return httpx.Response(200, json={"ok": True})

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        await client.get_json("/me", headers={"X-Custom": "value"})
        assert captured[0]["x-custom"] == "value"
        assert captured[0]["authorization"] == "Bearer cached-token"


# ---------------------------------------------------------------------------
# download_to_file — additional edge cases
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestDownloadToFileEdgeCases:
    async def test_download_retries_on_401(self, tmp_path: Path):
        calls: list[int] = []
        provider = _make_provider()

        async def mock_get_token(*, force_refresh=False):
            return "cached-token"

        provider.get_access_token = mock_get_token  # type: ignore[method-assign]

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(401, json={"error": {"message": "expired"}})
            return httpx.Response(
                200, content=b"data", headers={"content-type": "text/plain"}
            )

        async def fake_sleep(delay: float) -> None:
            pass

        client = MicrosoftGraphClient(
            provider,
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
            max_retries=2,
        )
        result = await client.download_to_file("/item", tmp_path / "out.txt")
        assert result["content_type"] == "text/plain"
        assert (tmp_path / "out.txt").read_bytes() == b"data"
        assert len(calls) == 2

    async def test_download_http_error_retries_then_raises(self, tmp_path: Path):
        call_count: list[int] = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            call_count[0] += 1
            raise httpx.ConnectError("connection refused")

        async def fake_sleep(delay: float) -> None:
            pass

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
            sleep=fake_sleep,
            max_retries=1,
        )
        with pytest.raises(MicrosoftGraphClientError, match="download failed"):
            await client.download_to_file("/item", tmp_path / "out.txt")

    async def test_download_creates_parent_dirs(self, tmp_path: Path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=b"data", headers={"content-type": "text/plain"}
            )

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        dest = tmp_path / "sub" / "dir" / "out.txt"
        result = await client.download_to_file("/item", dest)
        assert dest.exists()
        assert dest.read_bytes() == b"data"

    async def test_download_with_custom_headers(self, tmp_path: Path):
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(dict(request.headers))
            return httpx.Response(
                200, content=b"data", headers={"content-type": "text/plain"}
            )

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        await client.download_to_file(
            "/item", tmp_path / "out.txt", headers={"X-Custom": "val"}
        )
        assert captured[0]["x-custom"] == "val"


# ---------------------------------------------------------------------------
# _resolve_url
# ---------------------------------------------------------------------------


class TestResolveUrl:
    def test_absolute_url_passed_through(self):
        client = MicrosoftGraphClient(_make_provider())
        assert (
            client._resolve_url("https://example.com/api") == "https://example.com/api"
        )

    def test_relative_path_gets_base_prefix(self):
        client = MicrosoftGraphClient(_make_provider())
        assert client._resolve_url("/me") == "https://graph.microsoft.com/v1.0/me"

    def test_path_without_leading_slash_gets_slash(self):
        client = MicrosoftGraphClient(_make_provider())
        assert client._resolve_url("me") == "https://graph.microsoft.com/v1.0/me"

    def test_custom_base_url_strips_trailing_slash(self):
        client = MicrosoftGraphClient(
            _make_provider(), base_url="https://custom.example.com/api/"
        )
        assert client.base_url == "https://custom.example.com/api"
        assert client._resolve_url("/me") == "https://custom.example.com/api/me"


# ---------------------------------------------------------------------------
# _should_retry / _should_refresh_token / _retry_delay
# ---------------------------------------------------------------------------


class TestShouldRetry:
    def test_none_response_returns_true(self):
        assert MicrosoftGraphClient._should_retry(None) is True

    def test_429_returns_true(self):
        resp = httpx.Response(429)
        assert MicrosoftGraphClient._should_retry(resp) is True

    def test_500_returns_true(self):
        resp = httpx.Response(500)
        assert MicrosoftGraphClient._should_retry(resp) is True

    def test_503_returns_true(self):
        resp = httpx.Response(503)
        assert MicrosoftGraphClient._should_retry(resp) is True

    def test_200_returns_false(self):
        resp = httpx.Response(200)
        assert MicrosoftGraphClient._should_retry(resp) is False

    def test_404_returns_false(self):
        resp = httpx.Response(404)
        assert MicrosoftGraphClient._should_retry(resp) is False

    def test_400_returns_false(self):
        resp = httpx.Response(400)
        assert MicrosoftGraphClient._should_retry(resp) is False


class TestShouldRefreshToken:
    def test_none_error_returns_false(self):
        assert MicrosoftGraphClient._should_refresh_token(None) is False

    def test_401_api_error_returns_true(self):
        err = MicrosoftGraphAPIError(401, "GET", "/me", "unauthorized")
        assert MicrosoftGraphClient._should_refresh_token(err) is True

    def test_500_api_error_returns_false(self):
        err = MicrosoftGraphAPIError(500, "GET", "/me", "server error")
        assert MicrosoftGraphClient._should_refresh_token(err) is False

    def test_non_api_error_returns_false(self):
        err = RuntimeError("not an API error")
        assert MicrosoftGraphClient._should_refresh_token(err) is False


class TestRetryDelay:
    def test_none_response_uses_exponential_backoff(self):
        assert MicrosoftGraphClient._retry_delay(None, 0) == 0.5
        assert MicrosoftGraphClient._retry_delay(None, 1) == 1.0
        assert MicrosoftGraphClient._retry_delay(None, 2) == 2.0
        assert MicrosoftGraphClient._retry_delay(None, 3) == 4.0
        assert MicrosoftGraphClient._retry_delay(None, 4) == 8.0
        assert MicrosoftGraphClient._retry_delay(None, 5) == 8.0  # capped at 8.0

    def test_uses_retry_after_header(self):
        resp = httpx.Response(429, headers={"Retry-After": "5"})
        assert MicrosoftGraphClient._retry_delay(resp, 0) == 5.0

    def test_retry_after_invalid_falls_back_to_backoff(self):
        resp = httpx.Response(429, headers={"Retry-After": "not-a-number"})
        assert MicrosoftGraphClient._retry_delay(resp, 0) == 0.5

    def test_no_retry_after_header_uses_backoff(self):
        resp = httpx.Response(503)
        assert MicrosoftGraphClient._retry_delay(resp, 0) == 0.5


# ---------------------------------------------------------------------------
# _build_api_error
# ---------------------------------------------------------------------------


class TestBuildApiError:
    def test_error_dict_with_code_and_message(self):
        resp = httpx.Response(
            401, json={"error": {"code": "AuthError", "message": "bad token"}}
        )
        err = MicrosoftGraphClient._build_api_error("GET", "/me", resp)
        assert err.status_code == 401
        assert err.method == "GET"
        assert err.url == "/me"
        assert "AuthError" in str(err)
        assert "bad token" in str(err)

    def test_error_dict_message_only(self):
        resp = httpx.Response(401, json={"error": {"message": "just message"}})
        err = MicrosoftGraphClient._build_api_error("GET", "/me", resp)
        assert "just message" in str(err)

    def test_error_string(self):
        resp = httpx.Response(401, json={"error": "invalid_token"})
        err = MicrosoftGraphClient._build_api_error("GET", "/me", resp)
        assert "invalid_token" in str(err)

    def test_non_json_response_uses_text(self):
        resp = httpx.Response(
            401, content=b"plain text error", headers={"content-type": "text/plain"}
        )
        err = MicrosoftGraphClient._build_api_error("GET", "/me", resp)
        assert "plain text error" in str(err)

    def test_empty_response_text(self):
        resp = httpx.Response(401, content=b"")
        err = MicrosoftGraphClient._build_api_error("GET", "/me", resp)
        assert "unknown error" in str(err)

    def test_retry_after_header_parsed(self):
        resp = httpx.Response(
            429, json={"error": "rate limited"}, headers={"Retry-After": "10"}
        )
        err = MicrosoftGraphClient._build_api_error("GET", "/me", resp)
        assert err.retry_after_seconds == 10.0

    def test_retry_after_invalid_header(self):
        resp = httpx.Response(
            429, json={"error": "rate limited"}, headers={"Retry-After": "bad"}
        )
        err = MicrosoftGraphClient._build_api_error("GET", "/me", resp)
        assert err.retry_after_seconds is None

    def test_no_retry_after_header(self):
        resp = httpx.Response(401, json={"error": "unauthorized"})
        err = MicrosoftGraphClient._build_api_error("GET", "/me", resp)
        assert err.retry_after_seconds is None

    def test_payload_preserved(self):
        resp = httpx.Response(401, json={"error": "bad", "extra": "data"})
        err = MicrosoftGraphClient._build_api_error("GET", "/me", resp)
        assert err.payload is not None
        assert err.payload["error"] == "bad"

    def test_payload_none_for_non_json(self):
        resp = httpx.Response(
            401, content=b"not json", headers={"content-type": "text/plain"}
        )
        err = MicrosoftGraphClient._build_api_error("GET", "/me", resp)
        assert err.payload is None


# ---------------------------------------------------------------------------
# MicrosoftGraphClientError / MicrosoftGraphAPIError hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_api_error_is_client_error(self):
        assert issubclass(MicrosoftGraphAPIError, MicrosoftGraphClientError)

    def test_client_error_is_runtime_error(self):
        assert issubclass(MicrosoftGraphClientError, RuntimeError)

    def test_api_error_attributes(self):
        err = MicrosoftGraphAPIError(
            404,
            "DELETE",
            "/users/1",
            "not found",
            retry_after_seconds=5.0,
            payload={"key": "val"},
        )
        assert err.status_code == 404
        assert err.method == "DELETE"
        assert err.url == "/users/1"
        assert err.retry_after_seconds == 5.0
        assert err.payload == {"key": "val"}
        assert "404" in str(err)
        assert "DELETE" in str(err)
        assert "/users/1" in str(err)


# ---------------------------------------------------------------------------
# Constructor edge cases
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_max_retries_clamped_to_zero(self):
        client = MicrosoftGraphClient(_make_provider(), max_retries=-5)
        assert client.max_retries == 0

    def test_base_url_strips_trailing_slash(self):
        client = MicrosoftGraphClient(
            _make_provider(), base_url="https://example.com/api///"
        )
        assert client.base_url == "https://example.com/api"

    def test_default_user_agent(self):
        client = MicrosoftGraphClient(_make_provider())
        assert client.user_agent == "Hermes-Agent/graph-client"

    def test_custom_user_agent(self):
        client = MicrosoftGraphClient(_make_provider(), user_agent="Custom/1.0")
        assert client.user_agent == "Custom/1.0"

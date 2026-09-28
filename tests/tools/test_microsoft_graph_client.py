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
    provider = MicrosoftGraphTokenProvider(GraphCredentials("tenant", "client", "secret"))
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

    async def test_download_accepts_stream_content_by_default(self, tmp_path: Path):
        captured_accept: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_accept.append(request.headers["Accept"])
            return httpx.Response(
                200,
                content=b"recording-bytes",
                headers={"content-type": "video/mp4"},
            )

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        destination = tmp_path / "recording.mp4"

        result = await client.download_to_file(
            "/recordings/recording-1/content", destination
        )

        assert captured_accept == ["*/*"]
        assert destination.read_bytes() == b"recording-bytes"
        assert result["content_type"] == "video/mp4"

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

    async def test_put_bytes_sends_octet_stream_and_query_params(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["content_type"] = request.headers.get("Content-Type")
            captured["accept"] = request.headers.get("Accept")
            captured["authorization"] = request.headers.get("Authorization")
            captured["body"] = request.content
            return httpx.Response(
                201,
                json={"id": "item-1", "webUrl": "https://sp.example/file.bin"},
            )

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        payload = await client.put_bytes(
            "/drives/drive-1/items/folder-1:/report.bin:/content",
            content=b"%PDF-bytes",
            params={"@microsoft.graph.conflictBehavior": "rename"},
        )

        assert payload["id"] == "item-1"
        assert captured["method"] == "PUT"
        assert captured["content_type"] == "application/octet-stream"
        assert captured["accept"] == "application/json"
        assert captured["authorization"] == "Bearer cached-token"
        assert captured["body"] == b"%PDF-bytes"
        assert "microsoft.graph.conflictBehavior=rename" in str(captured["url"])

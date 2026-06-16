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
                return httpx.Response(
                    503, json={"error": {"message": "unavailable"}}
                )
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

"""Tests for tools/microsoft_graph_client.py."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tools import microsoft_graph_client as graph_client_mod
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

    async def test_get_json_rejects_oversized_success_response(self, monkeypatch):
        monkeypatch.setattr(graph_client_mod, "GRAPH_JSON_RESPONSE_MAX_BYTES", 8)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b'{"payload":"too large"}',
                headers={"content-type": "application/json"},
            )

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )

        with pytest.raises(MicrosoftGraphClientError) as exc:
            await client.get_json("/me")

        assert "exceeded 8 bytes" in str(exc.value)

    async def test_api_error_body_is_bounded(self, monkeypatch):
        monkeypatch.setattr(graph_client_mod, "GRAPH_JSON_RESPONSE_MAX_BYTES", 8)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                content=b"x" * 32,
                headers={"content-type": "text/plain"},
            )

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )

        with pytest.raises(MicrosoftGraphAPIError) as exc:
            await client.get_json("/me")

        assert exc.value.status_code == 400
        assert exc.value.payload is None
        assert "response body exceeded 8 bytes" in str(exc.value)
        assert "xxxxxxxxxxxxxxxx" not in str(exc.value)

    async def test_download_error_body_is_bounded(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(graph_client_mod, "GRAPH_JSON_RESPONSE_MAX_BYTES", 8)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                content=b"y" * 32,
                headers={"content-type": "text/plain"},
            )

        client = MicrosoftGraphClient(
            _make_provider(),
            transport=httpx.MockTransport(handler),
        )
        destination = tmp_path / "recording.mp4"

        with pytest.raises(MicrosoftGraphAPIError) as exc:
            await client.download_to_file("/drive/item/content", destination)

        assert "response body exceeded 8 bytes" in str(exc.value)
        assert "yyyyyyyyyyyyyyyy" not in str(exc.value)
        assert not destination.exists()
        assert not (tmp_path / "recording.mp4.part").exists()

"""Tests for Zulip attachment download support."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.zulip.attachments import download_zulip_upload, extract_upload_paths


async def _chunks(*chunks):
    for chunk in chunks:
        yield chunk


def test_extract_upload_paths_rejects_path_traversal():
    assert extract_upload_paths(
        "[bad](/user_uploads/2/%2e%2e/api/v1/messages) "
        "[ok](/user_uploads/2/a/token/report.pdf)"
    ) == ["/user_uploads/2/a/token/report.pdf"]


@pytest.mark.asyncio
async def test_download_zulip_upload_caches_document_from_signed_url():
    """A normal document follows Zulip's signed-URL flow into the media cache."""
    signed_response = MagicMock()
    signed_response.raise_for_status = MagicMock()
    signed_response.json.return_value = {"url": "https://storage.example/report.pdf?sig=abc"}

    file_response = MagicMock()
    file_response.raise_for_status = MagicMock()
    file_response.headers = {"content-type": "application/pdf; charset=binary"}
    file_response.aiter_bytes = MagicMock(return_value=_chunks(b"%PDF-", b"1.7"))
    file_context = MagicMock()
    file_context.__aenter__ = AsyncMock(return_value=file_response)
    file_context.__aexit__ = AsyncMock(return_value=False)

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=signed_response)
    client.stream = MagicMock(return_value=file_context)
    cached = SimpleNamespace(path="/tmp/cache/doc_report.pdf", media_type="application/pdf")

    with patch("httpx.AsyncClient", return_value=client), \
         patch("gateway.platforms.base.cache_media_bytes", return_value=cached) as cache:
        result = await download_zulip_upload(
            site_url="https://chat.example.com",
            bot_email="bot@example.com",
            api_key="api-key",
            path="/user_uploads/2/ab/token/report.pdf",
        )

    assert result.path == "/tmp/cache/doc_report.pdf"
    assert result.filename == "report.pdf"
    assert result.mime_type == "application/pdf"
    assert result.size_bytes == len(b"%PDF-1.7")
    assert client.get.await_args_list[0].args[0] == (
        "https://chat.example.com/api/v1/user_uploads/2/ab/token/report.pdf"
    )
    assert client.get.await_args_list[0].kwargs["auth"] == ("bot@example.com", "api-key")
    client.stream.assert_called_once_with("GET", "https://storage.example/report.pdf?sig=abc")
    cache.assert_called_once_with(
        b"%PDF-1.7", filename="report.pdf", mime_type="application/pdf"
    )


@pytest.mark.asyncio
async def test_download_zulip_upload_rejects_oversized_file_before_caching():
    signed_response = MagicMock()
    signed_response.raise_for_status = MagicMock()
    signed_response.json.return_value = {"url": "/temporary/upload"}

    file_response = MagicMock()
    file_response.raise_for_status = MagicMock()
    file_response.headers = {}
    file_response.aiter_bytes = MagicMock(return_value=_chunks(b"12345"))
    file_context = MagicMock()
    file_context.__aenter__ = AsyncMock(return_value=file_response)
    file_context.__aexit__ = AsyncMock(return_value=False)

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=signed_response)
    client.stream = MagicMock(return_value=file_context)

    with patch("httpx.AsyncClient", return_value=client), \
         patch("gateway.platforms.base.cache_media_bytes") as cache:
        with pytest.raises(ValueError, match="download limit"):
            await download_zulip_upload(
                site_url="https://chat.example.com",
                bot_email="bot@example.com",
                api_key="api-key",
                path="/user_uploads/2/ab/token/report.pdf",
                max_bytes=4,
            )

    cache.assert_not_called()

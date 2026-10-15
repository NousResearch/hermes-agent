"""Webex media follows the CLI/cron descriptor contract through real HTTP uploads."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.webex import adapter as webex
from tools import send_message_tool as delivery


@asynccontextmanager
async def _webex_transport(monkeypatch):
    requests = []

    async def upload(request):
        parts = []
        async for part in await request.multipart():
            parts.append((part.name, part.filename, bytes(await part.read())))
        requests.append((request.headers.get("Authorization"), parts))
        return web.json_response({"id": "uploaded-message"})

    app = web.Application()
    app.router.add_post("/v1/messages", upload)
    async with TestServer(app) as server:
        monkeypatch.setattr(webex, "API_BASE", str(server.make_url("/v1")))
        yield requests


@pytest.fixture
def media(tmp_path):
    path = tmp_path / "report.txt"
    path.write_bytes(b"delivered file contents")
    descriptors, message = BasePlatformAdapter.extract_media(f"MEDIA:{path}")
    assert message.strip() == ""
    return path, descriptors


@pytest.fixture
def webex_entry(monkeypatch):
    entries = []
    webex.register(SimpleNamespace(register_platform=lambda **kwargs: entries.append(PlatformEntry(**kwargs))))
    get_entry = platform_registry.get
    monkeypatch.setattr(platform_registry, "get", lambda name: entries[0] if name == "webex" else get_entry(name))
    return entries[0]


def _assert_upload(requests, path, *, caption):
    assert requests == [("Bearer webex-token", [
        ("roomId", None, b"room-id"),
        ("markdown", None, caption.encode()),
        ("parentId", None, b"thread-id"),
        ("files", path.name, b"delivered file contents"),
    ])]


@pytest.mark.asyncio
async def test_standalone_webex_uploads_extracted_media_descriptors(monkeypatch, media, webex_entry):
    path, descriptors = media
    monkeypatch.setattr(delivery, "_live_adapter", lambda platform: (None, None))
    async with _webex_transport(monkeypatch) as requests:
        result = await delivery._send_via_adapter(
            Platform("webex"), PlatformConfig(token="webex-token"), "room-id", "Report",
            thread_id="thread-id", media_files=descriptors)

    assert result.get("success"), result
    _assert_upload(requests, path, caption="Report")


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True])
async def test_attachment_only_webex_uses_registered_delivery(monkeypatch, media, webex_entry, live):
    path, descriptors = media
    config = PlatformConfig(token="webex-token")
    adapter = webex.WebexAdapter(config) if live else None
    monkeypatch.setattr(delivery, "_live_adapter", lambda platform: (None, adapter))
    async with _webex_transport(monkeypatch) as requests, aiohttp.ClientSession() as session:
        if adapter is not None:
            adapter._session = session
        result = await delivery._send_to_platform(
            Platform("webex"), config, "room-id", "", thread_id="thread-id", media_files=descriptors)

    assert result.get("success"), result
    assert "warnings" not in result
    if live:
        # Adapter uploads omit an empty caption rather than emitting an empty form field.
        assert requests == [("Bearer webex-token", [
            ("roomId", None, b"room-id"),
            ("parentId", None, b"thread-id"),
            ("files", path.name, b"delivered file contents"),
        ])]
    else:
        _assert_upload(requests, path, caption="")


@pytest.mark.asyncio
async def test_text_only_route_still_rejects_attachment_only_delivery(media):
    _path, descriptors = media
    result = await delivery._send_to_platform(
        Platform.EMAIL, PlatformConfig(), "recipient@example.com", "", media_files=descriptors)
    assert "only media attachments" in result["error"]

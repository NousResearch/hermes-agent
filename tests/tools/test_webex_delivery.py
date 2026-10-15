"""Webex media follows the CLI/cron descriptor contract through real HTTP uploads."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
async def _webex_transport(monkeypatch, *, fail_at=None):
    requests = []

    async def upload(request):
        parts = []
        if request.content_type == "multipart/form-data":
            async for part in await request.multipart():
                parts.append((part.name, part.filename, bytes(await part.read())))
        else:
            parts = [(key, None, value.encode()) for key, value in (await request.json()).items()]
        requests.append((request.headers.get("Authorization"), parts))
        if sum(name == "files" for name, _filename, _body in parts) > 1:
            return web.json_response({"message": "Only one file per message"}, status=400)
        if len(requests) == fail_at:
            return web.json_response({"message": "Upload failed"}, status=503)
        return web.json_response({"id": f"uploaded-message-{len(requests)}"})

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
        *([("markdown", None, caption.encode())] if caption else []),
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


@pytest.mark.asyncio
async def test_webex_sends_one_file_per_message_with_first_caption(monkeypatch, media, webex_entry, tmp_path):
    path, descriptors = media
    second = tmp_path / "second.txt"
    second.write_bytes(b"second attachment")
    monkeypatch.setattr(delivery, "_live_adapter", lambda platform: (None, None))
    async with _webex_transport(monkeypatch) as requests:
        result = await delivery._send_to_platform(
            Platform("webex"), PlatformConfig(token="webex-token"), "room-id", "Reports",
            thread_id="thread-id", media_files=[*descriptors, (str(second), False)])

    assert result == {"success": True, "message_id": "uploaded-message-2", "media_delivered": True}
    assert len(requests) == 2
    for index, (_auth, parts) in enumerate(requests):
        fields = {name: data for name, _filename, data in parts}
        assert fields["roomId"] == b"room-id"
        assert fields["parentId"] == b"thread-id"
        assert fields.get("markdown", b"") == (b"Reports" if index == 0 else b"")
        files = [(filename, data) for name, filename, data in parts if name == "files"]
        expected = path if index == 0 else second
        assert files == [(expected.name, expected.read_bytes())]


@pytest.mark.asyncio
async def test_webex_partial_media_failure_reports_delivered_count(monkeypatch, media, webex_entry):
    _path, descriptors = media
    monkeypatch.setattr(delivery, "_live_adapter", lambda platform: (None, None))
    async with _webex_transport(monkeypatch, fail_at=2) as requests:
        result = await delivery._send_to_platform(
            Platform("webex"), PlatformConfig(token="webex-token"), "room-id", "Reports",
            media_files=descriptors * 2)

    assert len(requests) == 2
    assert result["partial_success"] is True
    assert result["media_delivered"] is False
    assert "1/2" in result["error"]
    assert "503" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["", "Report"])
async def test_text_only_registry_sender_does_not_silently_drop_media(monkeypatch, media, message):
    _path, descriptors = media
    sender = AsyncMock(return_value={"success": True, "message_id": "text-message"})
    entry = PlatformEntry(name="text-only", label="Text only", adapter_factory=lambda cfg: None,
                          check_fn=lambda: True, standalone_sender_fn=sender)
    get_entry = platform_registry.get
    monkeypatch.setattr(platform_registry, "get", lambda name: entry if name == entry.name else get_entry(name))
    monkeypatch.setattr(delivery, "_live_adapter", lambda platform: (None, None))
    result = await delivery._send_to_platform(
        SimpleNamespace(value=entry.name), PlatformConfig(), "target", message, media_files=descriptors)

    if not message:
        assert "only media attachments" in result["error"]
        sender.assert_not_called()
    else:
        assert result["success"] is True
        assert "omitted" in result["warnings"][0]
        assert sender.await_args.kwargs["media_files"] == descriptors


@pytest.mark.asyncio
async def test_legacy_registered_media_sender_keeps_captioned_descriptor_contract(monkeypatch, media):
    _path, descriptors = media
    sender = AsyncMock(return_value={"success": True, "message_id": "media-id", "media_delivered": True})
    entry = PlatformEntry(name="legacy-media", label="Legacy media", adapter_factory=lambda cfg: None,
                          check_fn=lambda: True, standalone_sender_fn=sender)
    get_entry = platform_registry.get
    monkeypatch.setattr(platform_registry, "get", lambda name: entry if name == entry.name else get_entry(name))
    monkeypatch.setattr(delivery, "_live_adapter", lambda _platform: (None, None))

    result = await delivery._send_to_platform(
        SimpleNamespace(value=entry.name), PlatformConfig(), "target", "Report", media_files=descriptors)

    assert result["success"] is True
    assert sender.await_args.kwargs["media_files"] == descriptors
    assert "warnings" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("with_file", [False, True])
async def test_standalone_webex_bounds_multibyte_text_and_captions(monkeypatch, media, webex_entry, with_file):
    _path, descriptors = media
    monkeypatch.setattr(delivery, "_live_adapter", lambda platform: (None, None))
    message = "🙂" * 3000
    async with _webex_transport(monkeypatch) as requests:
        result = await delivery._send_via_adapter(
            Platform("webex"), PlatformConfig(token="webex-token"), "room-id", message,
            thread_id="thread-id", media_files=descriptors if with_file else [])

    assert result["success"] is True
    assert len(requests) > 1
    text_parts = []
    for _auth, parts in requests:
        fields = {name: data for name, _filename, data in parts}
        assert fields["roomId"] == b"room-id"
        assert fields["parentId"] == b"thread-id"
        text = fields.get("markdown", b"")
        assert len(text) <= 7439
        text_parts.append(text.decode())
    assert sum(text.count("🙂") for text in text_parts) == 3000
    assert sum(name == "files" for _auth, parts in requests for name, _filename, _data in parts) == int(with_file)
    if with_file:
        assert result["media_delivered"] is True
        assert any(name == "files" for name, _filename, _data in requests[-1][1])

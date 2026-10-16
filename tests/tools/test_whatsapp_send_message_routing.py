"""WhatsApp send_message routing regressions (#37906)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from tools.send_message_tool import _send_to_platform


class _RecordingWhatsAppAdapter:
    def __init__(self):
        self.text_calls = []
        self.image_calls = []

    async def send(self, *, chat_id, content, metadata=None):
        self.text_calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id="live-text")

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, **kwargs):
        self.image_calls.append({
            "chat_id": chat_id,
            "image_path": image_path,
            "caption": caption,
            "reply_to": reply_to,
            "metadata": kwargs.get("metadata"),
        })
        return SendResult(success=True, message_id="live-image")


def _pconfig():
    return SimpleNamespace(enabled=True, token=None, extra={})


def _whatsapp_entry_sender():
    from hermes_cli.plugins import discover_plugins
    from gateway.platform_registry import platform_registry

    discover_plugins()
    entry = platform_registry.get("whatsapp")
    assert entry is not None
    return entry


@pytest.mark.parametrize("chat_id", ["130631430344750@lid", "919900123456"])
def test_whatsapp_text_prefers_live_adapter_over_standalone(chat_id):
    """When the gateway owns a WhatsApp adapter, send_message must reuse it.

    The live adapter is where WhatsApp-specific JID normalization, chunking,
    reply prefixing, and bridge error handling live. Falling through to the
    standalone bridge sender bypasses that running adapter and reproduced #37906.
    """
    adapter = _RecordingWhatsAppAdapter()
    runner = SimpleNamespace(adapters={Platform.WHATSAPP: adapter})
    entry = _whatsapp_entry_sender()
    original = entry.standalone_sender_fn
    standalone_sender = AsyncMock(return_value={"error": "standalone should not be used"})
    entry.standalone_sender_fn = standalone_sender
    try:
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(
                _send_to_platform(
                    Platform.WHATSAPP,
                    _pconfig(),
                    chat_id,
                    "hello from gateway",
                )
            )
    finally:
        entry.standalone_sender_fn = original

    assert result == {"success": True, "message_id": "live-text"}
    assert adapter.text_calls == [{"chat_id": chat_id, "content": "hello from gateway", "metadata": None}]
    standalone_sender.assert_not_awaited()


def test_whatsapp_media_prefers_live_adapter_when_gateway_is_running(tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(b"png")
    adapter = _RecordingWhatsAppAdapter()
    runner = SimpleNamespace(adapters={Platform.WHATSAPP: adapter})
    entry = _whatsapp_entry_sender()
    original = entry.standalone_sender_fn
    standalone_sender = AsyncMock(return_value={"error": "standalone should not be used"})
    entry.standalone_sender_fn = standalone_sender
    try:
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(
                _send_to_platform(
                    Platform.WHATSAPP,
                    _pconfig(),
                    "15551234567@s.whatsapp.net",
                    "caption text",
                    media_files=[(str(image), False)],
                )
            )
    finally:
        entry.standalone_sender_fn = original

    assert result == {"success": True, "message_id": "live-image", "media_delivered": True}
    assert adapter.text_calls == []
    assert adapter.image_calls == [{
        "chat_id": "15551234567@s.whatsapp.net",
        "image_path": str(image),
        "caption": "caption text",
        "reply_to": None,
        "metadata": None,
    }]
    standalone_sender.assert_not_awaited()


def test_whatsapp_media_fallback_preserves_standalone_caption(tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(b"png")
    entry = _whatsapp_entry_sender()
    original = entry.standalone_sender_fn
    sender = AsyncMock(return_value={"success": True, "platform": "whatsapp", "message_id": "standalone-media"})
    entry.standalone_sender_fn = sender
    runner = SimpleNamespace(adapters={})
    pconfig = _pconfig()
    try:
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(
                _send_to_platform(
                    Platform.WHATSAPP,
                    pconfig,
                    "15551234567@s.whatsapp.net",
                    "caption text",
                    media_files=[(str(image), False)],
                )
            )
    finally:
        entry.standalone_sender_fn = original

    assert result == {"success": True, "platform": "whatsapp", "message_id": "standalone-media"}
    sender.assert_awaited_once_with(
        pconfig,
        "15551234567@s.whatsapp.net",
        "",
        thread_id=None,
        media_files=[(str(image), False)],
        caption="caption text",
        force_document=False,
    )


def test_whatsapp_falls_back_to_standalone_without_live_adapter():
    entry = _whatsapp_entry_sender()
    original = entry.standalone_sender_fn
    sender = AsyncMock(return_value={"success": True, "platform": "whatsapp", "message_id": "standalone"})
    entry.standalone_sender_fn = sender
    runner = SimpleNamespace(adapters={})
    pconfig = _pconfig()
    try:
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(
                _send_to_platform(
                    Platform.WHATSAPP,
                    pconfig,
                    "15551234567@s.whatsapp.net",
                    "cron fallback",
                )
            )
    finally:
        entry.standalone_sender_fn = original

    assert result == {"success": True, "platform": "whatsapp", "message_id": "standalone"}
    sender.assert_awaited_once_with(
        pconfig,
        "15551234567@s.whatsapp.net",
        "cron fallback",
        thread_id=None,
        media_files=[],
        force_document=False,
    )


def test_whatsapp_mentions_keep_the_standalone_bridge_payload():
    """Native WhatsApp mentions are a bridge-only payload, so keep that route."""
    adapter = _RecordingWhatsAppAdapter()
    runner = SimpleNamespace(adapters={Platform.WHATSAPP: adapter})
    entry = _whatsapp_entry_sender()
    original = entry.standalone_sender_fn
    sender = AsyncMock(return_value={"success": True, "platform": "whatsapp", "message_id": "mention"})
    entry.standalone_sender_fn = sender
    pconfig = _pconfig()
    try:
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = asyncio.run(
                _send_to_platform(
                    Platform.WHATSAPP,
                    pconfig,
                    "15551234567@s.whatsapp.net",
                    "hello @Alice",
                    mentions=["130631430344750@lid"],
                )
            )
    finally:
        entry.standalone_sender_fn = original

    assert result == {"success": True, "platform": "whatsapp", "message_id": "mention"}
    assert adapter.text_calls == []
    sender.assert_awaited_once_with(
        pconfig,
        "15551234567@s.whatsapp.net",
        "hello @Alice",
        thread_id=None,
        media_files=[],
        force_document=False,
        mentions=["130631430344750@lid"],
    )

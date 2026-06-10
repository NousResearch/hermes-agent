"""Regression tests for plugin media delivery in send_message_tool."""

import asyncio
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from tools.send_message_tool import _send_via_adapter


PHOTON = Platform("photon")


class _LiveAdapter:
    async def send(self, *, chat_id, content, metadata=None):  # pragma: no cover
        raise AssertionError("media delivery must not use generic adapter.send")


class _Runner:
    def __init__(self) -> None:
        self.adapters = {PHOTON: _LiveAdapter()}


def test_plugin_media_prefers_standalone_sender(monkeypatch):
    """Plugin MEDIA sends should use the plugin media-capable sender.

    The live adapter's generic ``send`` method only carries text and metadata;
    for Photon it can omit attachments or fail on event-loop-owned state.
    """
    calls = []

    async def standalone_sender(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
        calls.append(
            {
                "pconfig": pconfig,
                "chat_id": chat_id,
                "message": message,
                "thread_id": thread_id,
                "media_files": media_files,
                "force_document": force_document,
            }
        )
        return {"success": True, "message_id": "media-1"}

    entry = PlatformEntry(
        name="photon",
        label="Photon",
        adapter_factory=lambda cfg: _LiveAdapter(),
        check_fn=lambda: True,
        standalone_sender_fn=standalone_sender,
    )
    monkeypatch.setattr(platform_registry, "get", lambda name: entry if name == "photon" else None)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: _Runner())

    pconfig = SimpleNamespace(extra={})
    result = asyncio.run(
        _send_via_adapter(
            PHOTON,
            pconfig,
            "+15555550123",
            "caption",
            thread_id="thread-1",
            media_files=[("/tmp/test.png", False)],
            force_document=True,
        )
    )

    assert result == {"success": True, "message_id": "media-1"}
    assert calls == [
        {
            "pconfig": pconfig,
            "chat_id": "+15555550123",
            "message": "caption",
            "thread_id": "thread-1",
            "media_files": [("/tmp/test.png", False)],
            "force_document": True,
        }
    ]


def test_plugin_text_without_media_still_uses_live_adapter(monkeypatch):
    """Do not change the existing fast path for plugin text-only sends."""

    class _TextAdapter:
        async def send(self, *, chat_id, content, metadata=None):
            return SimpleNamespace(success=True, message_id="text-1", error=None)

    class _TextRunner:
        adapters = {PHOTON: _TextAdapter()}

    async def standalone_sender(*args, **kwargs):  # pragma: no cover
        raise AssertionError("text-only send should prefer live adapter")

    entry = PlatformEntry(
        name="photon",
        label="Photon",
        adapter_factory=lambda cfg: _TextAdapter(),
        check_fn=lambda: True,
        standalone_sender_fn=standalone_sender,
    )
    monkeypatch.setattr(platform_registry, "get", lambda name: entry if name == "photon" else None)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: _TextRunner())

    result = asyncio.run(
        _send_via_adapter(
            PHOTON,
            SimpleNamespace(extra={}),
            "+15555550123",
            "hello",
        )
    )

    assert result == {"success": True, "message_id": "text-1"}

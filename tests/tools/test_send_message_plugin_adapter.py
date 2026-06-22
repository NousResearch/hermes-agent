"""Plugin-platform send_message adapter routing tests."""

import asyncio
import threading
from types import SimpleNamespace

import pytest


class _FakePlatform:
    """Small stand-in for gateway.config.Platform with just a value."""

    def __init__(self, value):
        self.value = value


@pytest.mark.asyncio
async def test_live_adapter_send_runs_on_gateway_loop(monkeypatch):
    """A live plugin adapter is owned by the gateway loop, not the tool loop."""
    from tools.send_message_tool import _send_via_adapter

    gateway_loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(gateway_loop)
        ready.set()
        gateway_loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    assert ready.wait(timeout=2)

    class LoopRecordingAdapter:
        def __init__(self):
            self.loop = None

        async def send(self, **kwargs):
            self.loop = asyncio.get_running_loop()
            return SimpleNamespace(success=True, message_id="live-1", error=None)

    platform = _FakePlatform("fakeplatform")
    adapter = LoopRecordingAdapter()
    runner = SimpleNamespace(
        adapters={platform: adapter},
        _gateway_loop=gateway_loop,
    )

    try:
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)

        result = await _send_via_adapter(
            platform,
            SimpleNamespace(extra={}),
            "chat-1",
            "hi",
        )
    finally:
        gateway_loop.call_soon_threadsafe(gateway_loop.stop)
        thread.join(timeout=2)
        gateway_loop.close()

    assert result == {"success": True, "message_id": "live-1"}
    assert adapter.loop is gateway_loop


@pytest.mark.asyncio
async def test_live_adapter_media_uses_native_methods(monkeypatch, tmp_path):
    """MEDIA paths on plugin platforms should not be dropped on the live path."""
    from tools.send_message_tool import _send_via_adapter

    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(b"png")

    class MediaAdapter:
        def __init__(self):
            self.calls = []

        async def send(self, **kwargs):
            raise AssertionError("plain send should not handle media files")

        async def send_image_file(self, **kwargs):
            self.calls.append(("image", kwargs))
            return SimpleNamespace(success=True, message_id="img-1", error=None)

    platform = _FakePlatform("fakeplatform")
    adapter = MediaAdapter()
    runner = SimpleNamespace(adapters={platform: adapter}, _gateway_loop=None)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)

    result = await _send_via_adapter(
        platform,
        SimpleNamespace(extra={}),
        "chat-1",
        "caption",
        thread_id="thread-1",
        media_files=[(str(image_path), False)],
    )

    assert result == {"success": True, "message_id": "img-1"}
    assert len(adapter.calls) == 1
    kind, kwargs = adapter.calls[0]
    assert kind == "image"
    assert kwargs["chat_id"] == "chat-1"
    assert kwargs["image_path"] == str(image_path)
    assert kwargs["caption"] == "caption"
    assert kwargs["metadata"] == {"thread_id": "thread-1"}

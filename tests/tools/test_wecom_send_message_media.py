"""WeCom standalone MEDIA delivery (#153045).

``_standalone_send`` receives ``media_files`` / ``force_document`` from the generic send path
(cron and tool sends without a live adapter); it used to call only ``adapter.send`` and report
success while the attachment was dropped.
"""

from __future__ import annotations

import asyncio

from gateway.platforms.base import SendResult
from plugins.platforms.wecom import adapter as wecom


class _FakeAdapter:
    _ws_url = "wss://wecom.invalid"

    def __init__(self, fail_method=None):
        self.calls = []
        self.fail_method = fail_method

    async def send(self, chat_id, content, **kwargs):
        return self._record("send", chat_id, content)

    async def send_image_file(self, chat_id, path, **kwargs):
        return self._record("send_image_file", chat_id, path)

    async def send_document(self, chat_id, path, **kwargs):
        return self._record("send_document", chat_id, path)

    async def send_voice(self, chat_id, path, **kwargs):
        return self._record("send_voice", chat_id, path)

    async def send_video(self, chat_id, path, **kwargs):
        return self._record("send_video", chat_id, path)

    def _record(self, method, chat_id, value):
        self.calls.append((method, chat_id, value))
        if method == self.fail_method:
            return SendResult(success=False, error="upload rejected")
        return SendResult(success=True, message_id=f"m-{method}")


def _use_live_adapter(monkeypatch, fake):
    from tools import send_message_senders

    monkeypatch.setattr(send_message_senders, "_live_adapter", lambda platform: (None, fake))


def test_image_rides_its_native_method(tmp_path, monkeypatch):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    fake = _FakeAdapter()
    _use_live_adapter(monkeypatch, fake)

    result = asyncio.run(wecom._standalone_send(
        None, "chat-1", "hello", media_files=[(str(image), False)]))

    assert result["success"] is True
    assert fake.calls == [("send", "chat-1", "hello"), ("send_image_file", "chat-1", str(image))]


def test_force_document_sends_the_same_file_as_a_document(tmp_path, monkeypatch):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    fake = _FakeAdapter()
    _use_live_adapter(monkeypatch, fake)

    result = asyncio.run(wecom._standalone_send(
        None, "chat-1", "hello", media_files=[(str(image), False)], force_document=True))

    assert result["success"] is True
    assert [call[0] for call in fake.calls] == ["send", "send_document"]


def test_voice_note_is_sent_without_a_text_bubble(tmp_path, monkeypatch):
    voice = tmp_path / "note.ogg"
    voice.write_bytes(b"OggS")
    fake = _FakeAdapter()
    _use_live_adapter(monkeypatch, fake)

    result = asyncio.run(wecom._standalone_send(
        None, "chat-1", "", media_files=[(str(voice), True)]))

    assert result["success"] is True
    assert fake.calls == [("send_voice", "chat-1", str(voice))]


def test_failed_media_upload_is_reported_instead_of_dropped(tmp_path, monkeypatch):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    fake = _FakeAdapter(fail_method="send_image_file")
    _use_live_adapter(monkeypatch, fake)

    result = asyncio.run(wecom._standalone_send(
        None, "chat-1", "hello", media_files=[(str(image), False)]))

    assert "success" not in result
    assert "media send failed" in result["error"]
    assert "upload rejected" in result["error"]

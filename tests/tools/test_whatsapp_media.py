"""Tests for WhatsApp media delivery in send_message_tool.py.

The local Baileys bridge exposes POST /send (text) and POST /send-media
(native image/video/audio/document). These tests verify _send_whatsapp routes
text-only sends to /send and attachment sends to /send-media with the correct
payload (filePath, caption on the first attachment, mediaType=audio for voice).
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from tools.send_message_tool import _send_whatsapp


def _make_aiohttp_resp(status, json_data=None, text_data=None):
    """Minimal async-context-manager mock for an aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text_data or "")
    return resp


def _make_aiohttp_session(resp):
    """Wrap a response mock in a session mock that supports async-with for post."""
    request_ctx = MagicMock()
    request_ctx.__aenter__ = AsyncMock(return_value=resp)
    request_ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=request_ctx)

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx, session


class TestSendWhatsAppText:
    def test_text_only_uses_send_endpoint(self):
        """Backward compatibility: text-only goes to POST /send."""
        resp = _make_aiohttp_resp(200, json_data={"messageId": "wamid.TEXT"})
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(_send_whatsapp({"bridge_port": 3111}, "1@s.whatsapp.net", "hello"))

        assert result == {
            "success": True,
            "platform": "whatsapp",
            "chat_id": "1@s.whatsapp.net",
            "message_id": "wamid.TEXT",
        }
        session.post.assert_called_once()
        url, kwargs = session.post.call_args[0][0], session.post.call_args[1]
        assert url == "http://localhost:3111/send"
        assert kwargs["json"] == {"chatId": "1@s.whatsapp.net", "message": "hello"}

    def test_text_http_error_is_returned(self):
        resp = _make_aiohttp_resp(503, text_data="Not connected to WhatsApp")
        session_ctx, _ = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(_send_whatsapp({}, "1@s.whatsapp.net", "hi"))

        assert "error" in result
        assert "503" in result["error"]
        assert "Not connected" in result["error"]


class TestSendWhatsAppMedia:
    def test_image_uses_send_media_with_caption(self, tmp_path):
        """An image attachment goes to /send-media; text rides as the caption."""
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG\r\n")
        resp = _make_aiohttp_resp(200, json_data={"messageId": "wamid.IMG"})
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(_send_whatsapp(
                {"bridge_port": 3111},
                "1@s.whatsapp.net",
                "here is the screenshot",
                media_files=[(str(img), False)],
            ))

        assert result["success"] is True
        assert result["message_id"] == "wamid.IMG"
        # Only /send-media is hit (no separate text /send when an attachment carries the caption).
        session.post.assert_called_once()
        url, kwargs = session.post.call_args[0][0], session.post.call_args[1]
        assert url == "http://localhost:3111/send-media"
        assert kwargs["json"]["chatId"] == "1@s.whatsapp.net"
        assert kwargs["json"]["filePath"] == str(img)
        assert kwargs["json"]["caption"] == "here is the screenshot"
        assert "mediaType" not in kwargs["json"]  # image is inferred by the bridge

    def test_voice_sets_audio_media_type(self, tmp_path):
        """A voice attachment is sent as mediaType=audio so the bridge renders ptt."""
        voice = tmp_path / "note.ogg"
        voice.write_bytes(b"OggS")
        resp = _make_aiohttp_resp(200, json_data={"messageId": "wamid.PTT"})
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(_send_whatsapp(
                {"bridge_port": 3111},
                "1@s.whatsapp.net",
                "",
                media_files=[(str(voice), True)],
            ))

        assert result["success"] is True
        kwargs = session.post.call_args[1]
        assert kwargs["json"]["mediaType"] == "audio"
        assert "caption" not in kwargs["json"]  # no text → no caption

    def test_caption_only_on_first_attachment(self, tmp_path):
        """With multiple attachments, only the first carries the caption."""
        a = tmp_path / "a.png"; a.write_bytes(b"\x89PNG")
        b = tmp_path / "b.png"; b.write_bytes(b"\x89PNG")
        resp = _make_aiohttp_resp(200, json_data={"messageId": "wamid.X"})
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(_send_whatsapp(
                {}, "1@s.whatsapp.net", "caption text",
                media_files=[(str(a), False), (str(b), False)],
            ))

        assert result["success"] is True
        assert session.post.call_count == 2
        first_json = session.post.call_args_list[0][1]["json"]
        second_json = session.post.call_args_list[1][1]["json"]
        assert first_json["caption"] == "caption text"
        assert "caption" not in second_json

    def test_missing_media_file_errors(self):
        resp = _make_aiohttp_resp(200, json_data={"messageId": "x"})
        session_ctx, session = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(_send_whatsapp(
                {}, "1@s.whatsapp.net", "x",
                media_files=[("/does/not/exist.png", False)],
            ))

        assert "error" in result
        assert "not found" in result["error"]
        session.post.assert_not_called()

    def test_media_bridge_http_error_is_returned(self, tmp_path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG")
        resp = _make_aiohttp_resp(500, text_data="boom")
        session_ctx, _ = _make_aiohttp_session(resp)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(_send_whatsapp(
                {}, "1@s.whatsapp.net", "x", media_files=[(str(img), False)],
            ))

        assert "error" in result
        assert "500" in result["error"]
        assert "boom" in result["error"]

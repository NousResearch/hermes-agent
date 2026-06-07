"""Tests for WhatsApp media delivery in send_message_tool.py."""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform

from tools.send_message_tool import (
    _send_to_platform,
    _send_whatsapp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aiohttp_resp(status, json_data=None, text_data=None):
    """Build a minimal async-context-manager mock for an aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text_data or "")
    return resp


def _make_aiohttp_session_multi(resps):
    """Wrap response mocks in a session that returns them sequentially per post() call.

    *resps* is a list of response mocks. Each call to ``session.post()`` pops
    the next response from the list.
    """
    call_idx = [0]

    def _next_resp(*args, **kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        resp = resps[idx] if idx < len(resps) else resps[-1]
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    # Use a real MagicMock with side_effect to get proper call tracking
    session = MagicMock()
    session.post = MagicMock(side_effect=_next_resp)

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx, session


def _pconfig(extra=None):
    return SimpleNamespace(
        token="",
        extra=extra or {"bridge_port": 3000},
    )


# ---------------------------------------------------------------------------
# _send_whatsapp unit tests
# ---------------------------------------------------------------------------


class TestSendWhatsappTextOnly:
    def test_text_only_success(self):
        resp = _make_aiohttp_resp(200, json_data={"messageId": "msg-1"})
        session_ctx, session = _make_aiohttp_session_multi([resp])

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(
                _send_whatsapp({"bridge_port": 3000}, "chat1@s.whatsapp.net", "hello")
            )

        assert result == {
            "success": True,
            "platform": "whatsapp",
            "chat_id": "chat1@s.whatsapp.net",
            "message_id": "msg-1",
        }

    def test_text_only_error(self):
        resp = _make_aiohttp_resp(500, text_data="Internal Server Error")
        session_ctx, _ = _make_aiohttp_session_multi([resp])

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(
                _send_whatsapp({"bridge_port": 3000}, "chat1", "hello")
            )

        assert "error" in result
        assert "500" in result["error"]

    def test_default_bridge_port(self):
        resp = _make_aiohttp_resp(200, json_data={"messageId": "m1"})
        urls_called = []

        def _track_post(url, **kwargs):
            urls_called.append(url)
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        session = MagicMock()
        session.post = _track_post
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(
                _send_whatsapp({}, "chat1", "hi")
            )

        assert result["success"] is True
        assert any("localhost:3000" in u for u in urls_called)


class TestSendWhatsappWithMedia:
    def test_single_media_file(self):
        """Text + one media file: two POST calls (send + send-media)."""
        text_resp = _make_aiohttp_resp(200, json_data={"messageId": "msg-text"})
        media_resp = _make_aiohttp_resp(200, json_data={"messageId": "msg-media"})
        session_ctx, session = _make_aiohttp_session_multi([text_resp, media_resp])

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            media_path = f.name
            f.write(b"\xff\xd8\xff\xe0")

        try:
            with patch("aiohttp.ClientSession", return_value=session_ctx):
                result = asyncio.run(
                    _send_whatsapp(
                        {"bridge_port": 3000},
                        "chat1@s.whatsapp.net",
                        "check this",
                        media_files=[(media_path, False)],
                    )
                )

            assert result["success"] is True
            assert result["message_id"] == "msg-text"
            assert result["media_ids"] == ["msg-media"]

            # Verify two POST calls: /send and /send-media
            calls = session.post.call_args_list
            assert len(calls) == 2
            assert "/send" in calls[0][0][0]
            assert "/send-media" in calls[1][0][0]
            # First media should include caption
            media_payload = calls[1][1]["json"]
            assert media_payload["caption"] == "check this"
        finally:
            os.unlink(media_path)

    def test_multiple_media_files(self):
        """Text + 3 media files: 4 POST calls total."""
        text_resp = _make_aiohttp_resp(200, json_data={"messageId": "msg-text"})
        media_resps = [
            _make_aiohttp_resp(200, json_data={"messageId": f"media-{i}"})
            for i in range(3)
        ]
        session_ctx, session = _make_aiohttp_session_multi(
            [text_resp] + media_resps
        )

        media_paths = []
        try:
            for ext in [".jpg", ".mp4", ".pdf"]:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                    media_paths.append(f.name)
                    f.write(b"\x00" * 16)

            with patch("aiohttp.ClientSession", return_value=session_ctx):
                result = asyncio.run(
                    _send_whatsapp(
                        {"bridge_port": 3000},
                        "chat1@s.whatsapp.net",
                        "multi media",
                        media_files=[(p, False) for p in media_paths],
                    )
                )

            assert result["success"] is True
            assert result["media_ids"] == ["media-0", "media-1", "media-2"]
            # Only first media has caption
            calls = session.post.call_args_list
            first_media_payload = calls[1][1]["json"]
            assert "caption" in first_media_payload
            second_media_payload = calls[2][1]["json"]
            assert "caption" not in second_media_payload
        finally:
            for p in media_paths:
                os.unlink(p)

    def test_missing_media_file_skipped(self):
        """Non-existent media files are skipped with a warning."""
        text_resp = _make_aiohttp_resp(200, json_data={"messageId": "msg-text"})
        session_ctx, session = _make_aiohttp_session_multi([text_resp])

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(
                _send_whatsapp(
                    {"bridge_port": 3000},
                    "chat1",
                    "text",
                    media_files=[("/nonexistent/file.jpg", False)],
                )
            )

        assert result["success"] is True
        assert result.get("media_ids", []) == []
        # Only one POST call (text), no media POST
        assert session.post.call_count == 1

    def test_media_upload_failure_logged(self):
        """Media upload failure is logged but doesn't fail the overall send."""
        text_resp = _make_aiohttp_resp(200, json_data={"messageId": "msg-text"})
        media_resp = _make_aiohttp_resp(500, text_data="Server Error")
        session_ctx, session = _make_aiohttp_session_multi([text_resp, media_resp])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            media_path = f.name
            f.write(b"\x89PNG")

        try:
            with patch("aiohttp.ClientSession", return_value=session_ctx):
                result = asyncio.run(
                    _send_whatsapp(
                        {"bridge_port": 3000},
                        "chat1",
                        "text",
                        media_files=[(media_path, False)],
                    )
                )

            assert result["success"] is True
            assert result.get("media_ids", []) == []
        finally:
            os.unlink(media_path)

    def test_no_media_files_kwarg(self):
        """media_files=None should behave like text-only."""
        resp = _make_aiohttp_resp(200, json_data={"messageId": "msg-1"})
        session_ctx, session = _make_aiohttp_session_multi([resp])

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = asyncio.run(
                _send_whatsapp(
                    {"bridge_port": 3000}, "chat1", "hello", media_files=None
                )
            )

        assert result["success"] is True
        assert "media_ids" not in result


# ---------------------------------------------------------------------------
# _send_to_platform integration: WhatsApp media early-return path
# ---------------------------------------------------------------------------


class TestWhatsappPlatformMediaDispatch:
    def test_media_files_trigger_whatsapp_early_return(self):
        """When media_files is provided for WhatsApp, the early-return block fires."""
        mock_result = {
            "success": True,
            "platform": "whatsapp",
            "chat_id": "chat1@s.whatsapp.net",
            "message_id": "msg-1",
            "media_ids": ["media-1"],
        }

        with patch(
            "tools.send_message_tool._send_whatsapp",
            new=AsyncMock(return_value=mock_result),
        ):
            result = asyncio.run(
                _send_to_platform(
                    pconfig=_pconfig(),
                    platform=Platform.WHATSAPP,
                    chat_id="chat1@s.whatsapp.net",
                    message="see this image",
                    media_files=[("/tmp/image.jpg", False)],
                )
            )

        assert result["success"] is True
        assert result["media_ids"] == ["media-1"]

    def test_text_only_falls_through_to_generic_dispatch(self):
        """Without media_files, WhatsApp should use the generic elif dispatch."""
        mock_result = {"success": True, "platform": "whatsapp", "chat_id": "c1"}

        with patch(
            "tools.send_message_tool._send_whatsapp",
            new=AsyncMock(return_value=mock_result),
        ):
            result = asyncio.run(
                _send_to_platform(
                    pconfig=_pconfig(),
                    platform=Platform.WHATSAPP,
                    chat_id="c1",
                    message="just text",
                    media_files=None,
                )
            )

        assert result["success"] is True

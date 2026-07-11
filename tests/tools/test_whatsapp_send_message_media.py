"""WhatsApp media delivery for send_message (#19105).

Covers two layers:

* ``_bridge_media_type`` — extension/voice/force_document -> bridge mediaType.
* ``_standalone_send`` — text-first then per-file ``/send-media`` uploads,
  media-only (skip ``/send``), and missing-file errors. The bridge HTTP calls
  are mocked at the ``aiohttp.ClientSession`` boundary.
"""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.whatsapp.adapter import _bridge_media_type, _standalone_send


# ---------------------------------------------------------------------------
# _bridge_media_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,is_voice,force_document,expected",
    [
        ("a.png", False, False, "image"),
        ("a.JPG", False, False, "image"),
        ("a.jpeg", False, False, "image"),
        ("a.webp", False, False, "image"),
        ("a.gif", False, False, "image"),
        ("a.mp4", False, False, "video"),
        ("a.mov", False, False, "video"),
        ("a.webm", False, False, "video"),
        ("a.ogg", True, False, "audio"),
        ("a.opus", False, False, "audio"),
        ("a.mp3", False, False, "audio"),
        ("a.wav", False, False, "audio"),
        ("a.pdf", False, False, "document"),
        ("a.zip", False, False, "document"),
        # force_document overrides everything
        ("a.png", False, True, "document"),
        ("a.mp4", False, True, "document"),
        # is_voice wins over a video extension
        ("a.mp4", True, False, "audio"),
    ],
)
def test_bridge_media_type(path, is_voice, force_document, expected):
    assert _bridge_media_type(path, is_voice, force_document) == expected


# ---------------------------------------------------------------------------
# _standalone_send — bridge HTTP mocked
# ---------------------------------------------------------------------------


def _resp(status, json_data=None, text_data=None):
    r = AsyncMock()
    r.status = status
    r.json = AsyncMock(return_value=json_data or {})
    r.text = AsyncMock(return_value=text_data or "")
    return r


def _session_with(responses):
    """Build a mocked aiohttp.ClientSession that returns *responses* in order
    and records every POST (url, json_payload)."""
    calls = []
    idx = [0]

    def _post(url, **kwargs):
        calls.append((url, kwargs.get("json")))
        r = responses[idx[0]] if idx[0] < len(responses) else responses[-1]
        idx[0] += 1
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=r)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = MagicMock()
    session.post = MagicMock(side_effect=_post)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx, calls


def _pconfig():
    return SimpleNamespace(token="", extra={"bridge_port": 3000})


def _tmpfile(suffix):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"x")
    f.close()
    return f.name


def test_text_plus_mixed_media_routes_native_types():
    img = _tmpfile(".png")
    vid = _tmpfile(".mp4")
    voice = _tmpfile(".ogg")
    try:
        session_ctx, calls = _session_with(
            [
                _resp(200, {"messageId": "t1"}),
                _resp(200, {"messageId": "m1"}),
                _resp(200, {"messageId": "m2"}),
                _resp(200, {"messageId": "m3"}),
            ]
        )
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "12345",
                    "hello",
                    media_files=[(img, False), (vid, False), (voice, True)],
                )
            )
        assert res["success"] is True
        # text first, then three media uploads in order
        assert calls[0][0].endswith("/send")
        assert calls[0][1]["message"] == "hello"
        media_types = [c[1]["mediaType"] for c in calls if c[0].endswith("/send-media")]
        assert media_types == ["image", "video", "audio"]
        # chat id normalized to a WhatsApp JID
        assert "@" in calls[0][1]["chatId"]
    finally:
        for p in (img, vid, voice):
            os.unlink(p)


def test_media_only_skips_text_send():
    img = _tmpfile(".jpg")
    try:
        session_ctx, calls = _session_with([_resp(200, {"messageId": "m1"})])
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(_pconfig(), "12345", "", media_files=[(img, False)])
            )
        assert res["success"] is True
        assert all(c[0].endswith("/send-media") for c in calls)
    finally:
        os.unlink(img)


def test_force_document_sends_image_as_document():
    img = _tmpfile(".png")
    try:
        session_ctx, calls = _session_with(
            [_resp(200, {"messageId": "t1"}), _resp(200, {"messageId": "m1"})]
        )
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "12345",
                    "doc",
                    media_files=[(img, False)],
                    force_document=True,
                )
            )
        assert res["success"] is True
        media_call = [c for c in calls if c[0].endswith("/send-media")][0]
        assert media_call[1]["mediaType"] == "document"
        assert media_call[1]["fileName"] == os.path.basename(img)
    finally:
        os.unlink(img)


def test_missing_media_file_errors():
    session_ctx, _ = _session_with([_resp(200, {"messageId": "t1"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(
            _standalone_send(
                _pconfig(),
                "12345",
                "hi",
                media_files=[("/no/such/file.png", False)],
            )
        )
    assert "error" in res
    assert "not found" in res["error"]


def test_media_upload_error_propagates():
    img = _tmpfile(".png")
    try:
        session_ctx, _ = _session_with(
            [
                _resp(200, {"messageId": "t1"}),
                _resp(500, text_data="boom"),
            ]
        )
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(), "12345", "hi", media_files=[(img, False)]
                )
            )
        assert "error" in res
        assert "500" in res["error"]
    finally:
        os.unlink(img)


def test_text_only_unchanged_behavior():
    session_ctx, calls = _session_with([_resp(200, {"messageId": "t1"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(_standalone_send(_pconfig(), "12345", "just text"))
    assert res == {
        "success": True,
        "platform": "whatsapp",
        "chat_id": calls[0][1]["chatId"],
        "message_id": "t1",
    }
    assert len(calls) == 1 and calls[0][0].endswith("/send")


def test_caption_rides_media_no_separate_text_send():
    """MEDIA:<path> caption -> single /send-media with caption, no /send."""
    img = _tmpfile(".png")
    try:
        session_ctx, calls = _session_with([_resp(200, {"messageId": "m1"})])
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "12345",
                    "",
                    media_files=[(img, False)],
                    caption="2-bedroom floor plan",
                )
            )
        assert res["success"] is True
        # No separate /send — exactly one /send-media carrying the caption.
        assert len(calls) == 1
        assert calls[0][0].endswith("/send-media")
        assert calls[0][1]["caption"] == "2-bedroom floor plan"
        assert calls[0][1]["mediaType"] == "image"
    finally:
        os.unlink(img)


def test_caption_ignored_for_multi_file_send():
    """A caption never rides a multi-file send (association is ambiguous)."""
    img = _tmpfile(".png")
    img2 = _tmpfile(".jpg")
    try:
        session_ctx, calls = _session_with(
            [_resp(200, {"messageId": "m1"}), _resp(200, {"messageId": "m2"})]
        )
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "12345",
                    "",
                    media_files=[(img, False), (img2, False)],
                    caption="should be ignored",
                )
            )
        assert res["success"] is True
        media_calls = [c for c in calls if c[0].endswith("/send-media")]
        assert len(media_calls) == 2
        assert all("caption" not in c[1] for c in media_calls)
    finally:
        os.unlink(img)
        os.unlink(img2)


def test_missing_captioned_file_falls_back_to_text():
    """If the single captioned file is missing, the caption is delivered as a
    plain /send message rather than being silently lost (W1)."""
    session_ctx, calls = _session_with([_resp(200, {"messageId": "t1"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(
            _standalone_send(
                _pconfig(),
                "12345",
                "",
                media_files=[("/no/such/file.png", False)],
                caption="floor plan",
            )
        )
    # The send still surfaces the missing-file error...
    assert "error" in res
    assert "not found" in res["error"]
    # ...but the caption text was delivered on its own first.
    assert len(calls) == 1
    assert calls[0][0].endswith("/send")
    assert calls[0][1]["message"] == "floor plan"

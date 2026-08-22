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
        ("a.png", False, True, "document"),
        ("a.mp4", False, True, "document"),
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
        assert calls[0][0].endswith("/send")
        assert calls[0][1]["message"] == "hello"
        media_types = [c[1]["mediaType"] for c in calls if c[0].endswith("/send-media")]
        assert media_types == ["image", "video", "audio"]
        assert "@" in calls[0][1]["chatId"]
    finally:
        for p in (img, vid, voice):
            os.unlink(p)


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
    assert "error" in res
    assert "not found" in res["error"]
    assert len(calls) == 1
    assert calls[0][0].endswith("/send")
    assert calls[0][1]["message"] == "floor plan"


def test_standalone_send_forwards_mentions_on_text_send():
    session_ctx, calls = _session_with([_resp(200, {"messageId": "t1"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(
            _standalone_send(
                _pconfig(),
                "12345",
                "hey team",
                mentions=["67890", "13579@s.whatsapp.net"],
            )
        )

    assert res["success"] is True
    assert len(calls) == 1 and calls[0][0].endswith("/send")
    assert calls[0][1]["mentions"] == [
        "67890@s.whatsapp.net",
        "13579@s.whatsapp.net",
    ]


def test_standalone_send_without_mentions_omits_field():
    session_ctx, calls = _session_with([_resp(200, {"messageId": "t1"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(_standalone_send(_pconfig(), "12345", "plain"))

    assert res["success"] is True
    assert "mentions" not in calls[0][1]


def test_text_mentions_are_normalized_deduplicated_and_invalid_values_dropped():
    session_ctx, calls = _session_with([_resp(200, {"messageId": "t1"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(
            _standalone_send(
                _pconfig(),
                "120363000000000000@g.us",
                "hello team",
                mentions=[
                    "+1 (555) 000-0001",
                    "15550000001@s.whatsapp.net",
                    "15550000002@lid",
                    "not-a-phone",
                    "",
                    None,
                ],
            )
        )

    assert res["success"] is True
    assert calls[0][0].endswith("/send")
    assert calls[0][1]["mentions"] == [
        "15550000001@s.whatsapp.net",
        "15550000002@lid",
    ]


def test_captioned_media_preserves_mentions_on_send_media():
    image = _tmpfile(".png")
    try:
        session_ctx, calls = _session_with([_resp(200, {"messageId": "m1"})])
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "120363000000000000@g.us",
                    "",
                    media_files=[(image, False)],
                    caption="hello team",
                    mentions=["15550000001"],
                )
            )

        assert res["success"] is True
        assert len(calls) == 1
        assert calls[0][0].endswith("/send-media")
        assert calls[0][1]["caption"] == "hello team"
        assert calls[0][1]["mentions"] == ["15550000001@s.whatsapp.net"]
    finally:
        os.unlink(image)

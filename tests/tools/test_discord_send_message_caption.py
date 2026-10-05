"""Discord standalone MEDIA:<path> caption delivery.

When `hermes send --to discord "MEDIA:/x.png This Caption"` targets a normal
(non-forum) channel, the caption must ride on the media message content rather
than being posted as a separate message before the attachment. The Discord REST
calls are mocked at the aiohttp.ClientSession boundary.
"""

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from plugins.platforms.discord.adapter import _remember_channel_is_forum, _standalone_send


def _resp(status, json_data=None, text_data=None):
    r = AsyncMock()
    r.status = status
    body = json.dumps(json_data or {}).encode() if json_data is not None else (text_data or "").encode()
    r.json = AsyncMock(return_value=json_data or {})
    r.text = AsyncMock(return_value=text_data or "")
    # Discord's _standalone_read_*_limited helpers stream resp.content.read();
    # return the body once then EOF so the bounded reader terminates. AsyncMock
    # with a list side_effect yields each element on successive awaits.
    r.content = MagicMock()
    r.content.read = AsyncMock(side_effect=[body, b"", b""])
    # _standalone_response_encoding calls resp.get_encoding() expecting a str;
    # a bare AsyncMock would return a coroutine. Give it a plain callable.
    r.get_encoding = MagicMock(return_value="utf-8")
    return r


def _session_with(responses):
    """Mocked aiohttp.ClientSession recording every POST (url, json, data)."""
    calls = []
    idx = [0]

    def _post(url, **kwargs):
        calls.append((url, kwargs.get("json"), kwargs.get("data")))
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
    return SimpleNamespace(token="bot-token", extra={})


def _tmpfile(suffix):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"x")
    f.close()
    return f.name


def _payload_json_content(form_data):
    """Extract the 'content' from a FormData's payload_json field, if any."""
    for field in getattr(form_data, "_fields", []):
        # aiohttp FormData stores (type_options_dict, headers, value)
        try:
            type_opts = field[0]
            value = field[2]
        except (IndexError, TypeError):
            continue
        if type_opts.get("name") == "payload_json":
            return json.loads(value).get("content")
    return None


def test_caption_rides_media_non_forum():
    chat_id = "999000111"
    _remember_channel_is_forum(chat_id, False)  # avoid the live GET probe
    img = _tmpfile(".png")
    try:
        session_ctx, calls = _session_with([_resp(200, {"id": "m1"})])
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    chat_id,
                    "",
                    media_files=[(img, False)],
                    caption="2-bedroom floor plan",
                )
            )
        assert res["success"] is True
        # Exactly one POST (the media upload) — no separate text message.
        assert len(calls) == 1
        url, _json, data = calls[0]
        assert url.endswith("/messages")
        assert _payload_json_content(data) == "2-bedroom floor plan"
    finally:
        os.unlink(img)


def test_reply_to_writes_exact_message_reference():
    """A standalone Discord send can reply to a specific message in a thread."""
    chat_id = "999000333"
    thread_id = "999000444"
    source_message_id = "999000555"
    _remember_channel_is_forum(thread_id, False)
    session_ctx, calls = _session_with([_resp(200, {"id": "reply1"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(
            _standalone_send(
                _pconfig(),
                chat_id,
                "sanitized terminal receipt",
                thread_id=thread_id,
                reply_to=source_message_id,
            )
        )
    assert res["success"] is True
    assert res["message_id"] == "reply1"
    assert len(calls) == 1
    url, payload, _data = calls[0]
    assert url.endswith(f"/channels/{thread_id}/messages")
    assert payload == {
        "content": "sanitized terminal receipt",
        "message_reference": {
            "message_id": source_message_id,
            "channel_id": thread_id,
            "fail_if_not_exists": False,
        },
    }


def test_invalid_reply_anchor_fails_closed_without_posting():
    """An explicit but malformed anchor must not degrade into a top-level send."""
    chat_id = "999000666"
    thread_id = "999000777"
    _remember_channel_is_forum(thread_id, False)
    session_ctx, calls = _session_with([_resp(200, {"id": "unexpected"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(
            _standalone_send(
                _pconfig(),
                chat_id,
                "sanitized terminal receipt",
                thread_id=thread_id,
                reply_to="not-a-snowflake",
            )
        )
    assert res.get("success") is not True
    assert "numeric snowflakes" in res["error"]
    assert calls == []


def test_whitespace_reply_anchor_is_not_normalized():
    """Malformed identifiers must fail closed instead of being repaired by trimming."""
    _remember_channel_is_forum("999000888", False)
    session_ctx, calls = _session_with([_resp(200, {"id": "unexpected"})])
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        res = asyncio.run(
            _standalone_send(
                _pconfig(),
                "999000888",
                "sanitized terminal receipt",
                thread_id="999000888",
                reply_to=" 999000999",
            )
        )
    assert res.get("success") is not True
    assert "numeric snowflakes" in res["error"]
    assert calls == []


def test_no_caption_non_forum_keeps_separate_text():
    """Without a caption, text + media are two separate POSTs (unchanged)."""
    chat_id = "999000222"
    _remember_channel_is_forum(chat_id, False)
    img = _tmpfile(".png")
    try:
        session_ctx, calls = _session_with(
            [_resp(200, {"id": "t1"}), _resp(200, {"id": "m1"})]
        )
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    chat_id,
                    "hello",
                    media_files=[(img, False)],
                )
            )
        assert res["success"] is True
        # Two POSTs: the text content message, then the media upload.
        assert len(calls) == 2
        assert calls[0][1] == {"content": "hello"}
        assert calls[1][0].endswith("/messages")
    finally:
        os.unlink(img)

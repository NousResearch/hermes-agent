"""Discord standalone MP3 + body text must be ONE message.

Field report (2026-11-07): `hermes send -t discord:<channel> "MEDIA:/day.mp3
<day text>"` split into a text message plus a second captionless attachment
message, because `_media_caption_split` excluded audio from `_CAPTIONABLE_EXTS`
so the standalone Discord sender never received a caption. The Discord REST
multipart path does accept ``payload_json.content`` on an audio upload (a test
upload succeeded that night); only the shared caption chokepoint blocked it.
The four standalone senders that reach the Discord/Slack/WhatsApp/Telegram
media paths all carry a native caption on any single file, so the chokepoint —
not the senders — is the fix, pinned red on base by
``tests/tools/test_media_caption_split.py::test_single_audio_file_short_text_becomes_caption``.
"""

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from plugins.platforms.discord.adapter import _remember_channel_is_forum, _standalone_send


def _resp(status, json_data=None):
    r = AsyncMock()
    r.status = status
    body = json.dumps(json_data or {}).encode()
    r.json = AsyncMock(return_value=json_data or {})
    r.text = AsyncMock(return_value="")
    r.content = MagicMock()
    r.content.read = AsyncMock(side_effect=[body, b"", b""])
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


def _payload_json(form_data):
    for field in getattr(form_data, "_fields", []):
        try:
            type_opts, value = field[0], field[2]
        except (IndexError, TypeError):
            continue
        if type_opts.get("name") == "payload_json":
            return json.loads(value)
    return None


def _mp3():
    f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    f.write(b"ID3" + b"\x00" * 32)
    f.close()
    return f.name


def test_mp3_with_body_is_one_message_with_caption():
    chat_id = "1566982719139876926"
    _remember_channel_is_forum(chat_id, False)  # avoid the live GET probe
    mp3 = _mp3()
    try:
        session_ctx, calls = _session_with([_resp(200, {"id": "m1"})])
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(_standalone_send(
                SimpleNamespace(token="bot-token", extra={}), chat_id, "",
                media_files=[(mp3, False)], caption="2026-11-02 — Stops and Currents"))
        assert res["success"] is True
        # Exactly one POST — the multipart media upload carrying the body as content.
        assert len(calls) == 1
        url, _json, data = calls[0]
        assert url.endswith("/messages")
        assert _payload_json(data) == {"content": "2026-11-02 — Stops and Currents"}
    finally:
        os.unlink(mp3)


def test_mp3_without_body_never_invents_a_text_post():
    """Caption mode suppresses the separate text send; without a caption the
    media-only send stays a single attachment POST too."""
    chat_id = "1566982719139876926"
    _remember_channel_is_forum(chat_id, False)
    mp3 = _mp3()
    try:
        session_ctx, calls = _session_with([_resp(200, {"id": "m1"})])
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(_standalone_send(
                SimpleNamespace(token="bot-token", extra={}), chat_id, "",
                media_files=[(mp3, False)], caption="body rides on the upload"))
        assert res["success"] is True
        assert len(calls) == 1
        assert calls[0][0].endswith("/messages")
    finally:
        os.unlink(mp3)

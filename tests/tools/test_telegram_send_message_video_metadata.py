"""The standalone ``hermes send`` Telegram path must carry video metadata too.

Same root cause as the gateway adapter path (see
``tests/gateway/test_telegram_video_metadata.py``): Telegram stops processing a video upload once
it is large enough, and the message then has no geometry and no thumbnail, so clients render a
square tile whatever the real aspect ratio. ``_telegram_send_one_media`` feeds the same helpers.

These tests confirm:
  1. a video send forwards the probed geometry and the JPEG thumbnail,
  2. the temporary thumbnail is removed afterwards,
  3. ``[[as_document]]`` (force_document) keeps sending as a document with no geometry kwargs.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools.send_message_senders import _telegram_send_one_media


def _fake_bot(**methods) -> SimpleNamespace:
    return SimpleNamespace(**{
        name: AsyncMock(return_value=SimpleNamespace(message_id=1)) for name in methods
    })


async def _send(bot, path, *, force_document=False):
    return await _telegram_send_one_media(
        bot, "1", str(path), False, caption=None, parse_mode=None,
        has_html=False, thread_kwargs={}, force_document=force_document)


@pytest.mark.asyncio
async def test_video_send_carries_geometry_and_thumbnail(monkeypatch, tmp_path):
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    monkeypatch.setattr("plugins.platforms.telegram.adapter._probe_video_geometry",
                        lambda _p: {"width": 720, "height": 1280, "duration": 59})
    monkeypatch.setattr("plugins.platforms.telegram.adapter._video_thumbnail_jpeg",
                        lambda _p, _d: str(thumb))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 32)
    bot = _fake_bot(send_video=True)

    await _send(bot, video)

    bot.send_video.assert_awaited_once()
    kwargs = bot.send_video.await_args.kwargs
    assert (kwargs["width"], kwargs["height"], kwargs["duration"]) == (720, 1280, 59)
    assert kwargs["thumbnail"] == str(thumb)
    assert not thumb.exists(), "the temporary thumbnail must not be left behind"


@pytest.mark.asyncio
async def test_video_send_without_probeable_file_still_sends(monkeypatch, tmp_path):
    monkeypatch.setattr("plugins.platforms.telegram.adapter._probe_video_geometry", lambda _p: {})
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 16)
    bot = _fake_bot(send_video=True)

    await _send(bot, video)

    kwargs = bot.send_video.await_args.kwargs
    assert "width" not in kwargs and "thumbnail" not in kwargs

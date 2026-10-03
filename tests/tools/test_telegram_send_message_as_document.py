"""``[[as_document]]`` must deliver video files as real file attachments everywhere.

Two routing pieces previously turned an explicit "send as file" request into a
video message, and a third reclassified it server-side:

1. ``_telegram_send_media`` routed every video extension to ``sendVideo``
   regardless of ``force_document`` (which only forced *images* to documents),
   and ``_telegram_send_one_media`` ran the video geometry/thumbnail probe on
   every video path — pointless for a document send.
2. With a self-hosted Bot API server, ``sendDocument`` uploads were
   reclassified as video messages by server-side content-type detection, so
   even a correctly-routed document arrived as an inline video.

These tests pin the behaviour on the standalone ``hermes send`` path: with
``force_document`` a video goes out via ``sendDocument`` (no geometry/thumbnail,
``disable_content_type_detection`` set), and without it, video keeps sending as
a video message with its geometry and thumbnail.
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
async def test_as_document_video_goes_out_as_document(monkeypatch, tmp_path):
    # If the video probe is reached despite force_document, geometry/thumbnail
    # kwargs appear and the assertions below fail — i.e. this doubles as a
    # guard against regressing the routing.
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    monkeypatch.setattr("plugins.platforms.telegram.adapter._probe_video_geometry",
                        lambda _p: {"width": 720, "height": 1280, "duration": 59})
    monkeypatch.setattr("plugins.platforms.telegram.adapter._video_thumbnail_jpeg",
                        lambda _p, _d: str(thumb))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 32)
    bot = _fake_bot(send_document=True, send_video=True)

    await _send(bot, video, force_document=True)

    bot.send_document.assert_awaited_once()
    bot.send_video.assert_not_awaited()
    kwargs = bot.send_document.await_args.kwargs
    assert kwargs["disable_content_type_detection"] is True
    assert "width" not in kwargs and "thumbnail" not in kwargs


@pytest.mark.asyncio
async def test_video_without_as_document_still_sends_as_video(monkeypatch, tmp_path):
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    monkeypatch.setattr("plugins.platforms.telegram.adapter._probe_video_geometry",
                        lambda _p: {"width": 720, "height": 1280, "duration": 59})
    monkeypatch.setattr("plugins.platforms.telegram.adapter._video_thumbnail_jpeg",
                        lambda _p, _d: str(thumb))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 32)
    bot = _fake_bot(send_document=True, send_video=True)

    await _send(bot, video)

    bot.send_video.assert_awaited_once()
    bot.send_document.assert_not_awaited()
    kwargs = bot.send_video.await_args.kwargs
    assert kwargs["thumbnail"] == str(thumb)


@pytest.mark.asyncio
async def test_plain_document_send_disables_content_type_detection(tmp_path):
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(b"PK\x03\x04")
    bot = _fake_bot(send_document=True)

    await _send(bot, archive)

    bot.send_document.assert_awaited_once()
    kwargs = bot.send_document.await_args.kwargs
    assert kwargs["disable_content_type_detection"] is True

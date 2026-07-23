"""Guard test for the MEDIA:<path> caption chokepoint (_media_caption_split).

`hermes send` strips the MEDIA: tag and leaves the remaining prose as the
accompanying text. Historically every standalone sender posted that text as a
*separate* message before an uncaptioned media bubble, splitting
``hermes send --to whatsapp "MEDIA:/x.png This Caption"`` into two parts.

`_media_caption_split` is the single enforced decision point that all standalone
senders (WhatsApp, Telegram, Discord) consult to decide whether the text should
ride on the media bubble as a native caption. This test pins that contract so
the platforms can't diverge.
"""

from tools.send_message_tool import (
    _DEFAULT_CAPTION_LIMIT,
    _TELEGRAM_CAPTION_LIMIT,
    _media_caption_split,
)


def test_single_image_short_text_becomes_caption():
    caption, body = _media_caption_split(
        "This Caption", [("/tmp/F22.png", False)], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption == "This Caption"
    assert body == ""


def test_single_video_short_text_becomes_caption():
    caption, body = _media_caption_split(
        "Model unit tour", [("/tmp/tour.mp4", False)], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption == "Model unit tour"
    assert body == ""


def test_single_document_short_text_becomes_caption():
    caption, body = _media_caption_split(
        "Q3 report", [("/tmp/report.pdf", False)], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption == "Q3 report"
    assert body == ""


def test_multi_file_keeps_separate_body():
    text = "two photos"
    caption, body = _media_caption_split(
        text,
        [("/tmp/a.png", False), ("/tmp/b.png", False)],
        max_caption_len=_DEFAULT_CAPTION_LIMIT,
    )
    assert caption is None
    assert body == text


def test_voice_note_keeps_separate_body():
    text = "listen to this"
    caption, body = _media_caption_split(
        text, [("/tmp/note.ogg", True)], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption is None
    assert body == text


def test_empty_text_no_caption():
    caption, body = _media_caption_split(
        "   ", [("/tmp/a.png", False)], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption is None
    # body is returned unchanged (still whitespace) — sender's own guards drop it
    assert body == "   "


def test_no_media_no_caption():
    caption, body = _media_caption_split(
        "hello", [], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption is None
    assert body == "hello"


def test_text_over_limit_stays_separate_body():
    long_text = "x" * (_TELEGRAM_CAPTION_LIMIT + 1)
    caption, body = _media_caption_split(
        long_text, [("/tmp/a.png", False)], max_caption_len=_TELEGRAM_CAPTION_LIMIT
    )
    assert caption is None
    assert body == long_text


def test_text_at_limit_still_captions():
    text = "y" * _TELEGRAM_CAPTION_LIMIT
    caption, body = _media_caption_split(
        text, [("/tmp/a.png", False)], max_caption_len=_TELEGRAM_CAPTION_LIMIT
    )
    assert caption == text
    assert body == ""


def test_caption_is_stripped():
    caption, body = _media_caption_split(
        "  padded caption  ", [("/tmp/a.png", False)], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption == "padded caption"
    assert body == ""


def test_unknown_extension_keeps_separate_body():
    # A non-captionable kind (e.g. an audio note that isn't flagged voice)
    text = "some audio"
    caption, body = _media_caption_split(
        text, [("/tmp/song.mp3", False)], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption is None
    assert body == text

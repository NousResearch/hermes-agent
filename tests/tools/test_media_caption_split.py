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

from tools.send_message_tool import _DEFAULT_CAPTION_LIMIT, _media_caption_split
from tools.send_message_senders import _TELEGRAM_CAPTION_LIMIT


def test_single_image_short_text_becomes_caption():
    caption, body = _media_caption_split(
        "This Caption", [("/tmp/F22.png", False)], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption == "This Caption"
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


def test_unknown_extension_keeps_separate_body():
    # A non-captionable kind (e.g. a file type no sender captions)
    text = "some archive"
    caption, body = _media_caption_split(
        text, [("/tmp/blob.xyz", False)], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption is None
    assert body == text


def test_single_audio_file_short_text_becomes_caption():
    # A plain audio attachment (not voice-flagged) is captionable: the standalones put
    # the text on the media message itself (Discord multipart payload_json, Telegram
    # sendAudio caption, Slack initial_comment, WhatsApp bridge caption) instead of
    # splitting one captionless attachment message off from its body text.
    caption, body = _media_caption_split(
        "2026-11-02 — Stops and Currents", [("/tmp/day.mp3", False)],
        max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption == "2026-11-02 — Stops and Currents"
    assert body == ""


def test_voice_flagged_audio_keeps_separate_body():
    # A voice-flagged note still keeps its text separate: a caption on a voice bubble
    # reads as a separate label, and the client-side caption never renders on it.
    text = "listen to this"
    caption, body = _media_caption_split(
        text, [("/tmp/note.ogg", True)], max_caption_len=_DEFAULT_CAPTION_LIMIT
    )
    assert caption is None
    assert body == text

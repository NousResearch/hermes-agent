"""Non-Telegram MEDIA-tag caption delivery for send_message.

``MEDIA:<path> | <caption>`` captions only reached Telegram: the plugin-standalone and
live-adapter media paths rebuilt their caption from the *cleaned* body, and
``extract_media`` had already deleted the tag caption from it — the photo went out
uncaptioned and the words were gone. These tests pin the three delivery rules:

* a caption-capable media path (discord/slack/whatsapp standalone, live adapter, matrix)
  puts the tag caption on the attachment bubble;
* a platform with no caption field (feishu, signal, yuanbao, weixin) still delivers the
  caption text, as its own message;
* body text is never suppressed into a bubble a tag caption already owns.
"""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from tools.send_message_senders import _fold_captions_into_text
from tools.send_message_tool import _send_live_adapter_media, _send_plugin_standalone, _send_to_platform


def _tmpfile(suffix):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"x")
    f.close()
    return f.name


def _pconfig(**extra):
    return SimpleNamespace(enabled=True, token="tok", extra=extra)


class _RecordingSender:
    """Records every standalone-sender call (kwargs included)."""

    def __init__(self):
        self.calls = []

    async def __call__(self, pconfig, chat_id, message, **kwargs):
        # thread_id / force_document are plumbing, not the subject of these tests.
        recorded = {k: v for k, v in kwargs.items() if k not in ("thread_id", "force_document")}
        self.calls.append({"message": message, **recorded})
        return {"success": True, "message_id": "m1"}


# ---------------------------------------------------------------------------
# _fold_captions_into_text
# ---------------------------------------------------------------------------


def test_fold_appends_captions_after_body():
    assert _fold_captions_into_text("body", [("/a.png", False)], {"/a.png": "Cap"}) == "body\n\nCap"


def test_fold_keeps_caption_when_body_is_empty():
    assert _fold_captions_into_text("", [("/a.png", False)], {"/a.png": "Cap"}) == "Cap"


def test_fold_skips_excluded_and_uncaptioned_paths():
    media = [("/a.png", False), ("/b.png", False)]
    assert _fold_captions_into_text("body", media, {"/a.png": "A", "/b.png": "B"}, exclude="/b.png") == "body\n\nA"
    assert _fold_captions_into_text("body", media, {}) == "body"


# ---------------------------------------------------------------------------
# _send_plugin_standalone — caption-capable platforms (discord/slack/whatsapp)
# ---------------------------------------------------------------------------


def _patch_standalone_sender(monkeypatch, platform_name, sender):
    import tools.send_message_tool as smt
    monkeypatch.setattr(
        smt, "_plugin_standalone_sender",
        lambda name, *, label=None, discover=True: (sender, None) if name == platform_name else (None, {"error": "no"}))


def test_discord_standalone_tag_caption_rides_the_bubble(monkeypatch):
    img = _tmpfile(".png")
    try:
        sender = _RecordingSender()
        _patch_standalone_sender(monkeypatch, "discord", sender)
        result = asyncio.run(_send_plugin_standalone(
            "discord", _pconfig(), "ch", "", [""], [(img, False)],
            thread_id=None, max_len=None, force_document=False, media_captions={img: "2-bedroom plan"}))
        assert result["success"] is True
        assert sender.calls == [{"message": "", "media_files": [(img, False)], "caption": "2-bedroom plan"}]
    finally:
        os.unlink(img)


def test_discord_standalone_tag_caption_keeps_body_as_text(monkeypatch):
    """A tag caption owns the bubble: the body must go out as its own message, not vanish."""
    img = _tmpfile(".png")
    try:
        sender = _RecordingSender()
        _patch_standalone_sender(monkeypatch, "discord", sender)
        asyncio.run(_send_plugin_standalone(
            "discord", _pconfig(), "ch", "Билет 1", ["Билет 1"], [(img, False)],
            thread_id=None, max_len=None, force_document=False, media_captions={img: "Подпись"}))
        assert sender.calls == [
            {"message": "Билет 1", "media_files": []},
            {"message": "", "media_files": [(img, False)], "caption": "Подпись"},
        ]
    finally:
        os.unlink(img)


def test_whatsapp_standalone_tag_caption_used_verbatim(monkeypatch):
    """WhatsApp drops the body when a caption is set — the sender must get caption=, message=''."""
    img = _tmpfile(".png")
    try:
        sender = _RecordingSender()
        _patch_standalone_sender(monkeypatch, "whatsapp", sender)
        asyncio.run(_send_plugin_standalone(
            "whatsapp", _pconfig(), "12345", "", [""], [(img, False)],
            thread_id=None, max_len=4096, force_document=False, media_captions={img: "Подпись"}))
        assert sender.calls == [{"message": "", "media_files": [(img, False)], "caption": "Подпись"}]
    finally:
        os.unlink(img)


def test_captionable_without_tag_caption_is_unchanged(monkeypatch):
    """Backward compatibility: a short body still becomes the bubble caption of one file."""
    img = _tmpfile(".png")
    try:
        sender = _RecordingSender()
        _patch_standalone_sender(monkeypatch, "discord", sender)
        asyncio.run(_send_plugin_standalone(
            "discord", _pconfig(), "ch", "Short body", ["Short body"], [(img, False)],
            thread_id=None, max_len=None, force_document=False, media_captions=None))
        assert sender.calls == [{"message": "", "media_files": [(img, False)], "caption": "Short body"}]
    finally:
        os.unlink(img)


# ---------------------------------------------------------------------------
# _send_plugin_standalone — platforms without a caption field (feishu)
# ---------------------------------------------------------------------------


def test_feishu_folds_tag_caption_into_text(monkeypatch):
    img = _tmpfile(".png")
    try:
        sender = _RecordingSender()
        _patch_standalone_sender(monkeypatch, "feishu", sender)
        result = asyncio.run(_send_plugin_standalone(
            "feishu", _pconfig(), "ch", "", [""], [(img, False)],
            thread_id=None, max_len=8000, force_document=False, media_captions={img: "Подпись"}))
        assert result["success"] is True
        assert len(sender.calls) == 1
        assert sender.calls[0]["message"] == "Подпись"
        assert sender.calls[0]["media_files"] == [(img, False)]
    finally:
        os.unlink(img)


def test_caption_rides_bubble_even_when_it_belongs_to_a_later_file(monkeypatch):
    """A one-caption sender carries the first available tag caption; the other file stays bare."""
    a, b = _tmpfile(".png"), _tmpfile(".png")
    try:
        sender = _RecordingSender()
        _patch_standalone_sender(monkeypatch, "discord", sender)
        asyncio.run(_send_plugin_standalone(
            "discord", _pconfig(), "ch", "", [""], [(a, False), (b, False)],
            thread_id=None, max_len=None, force_document=False, media_captions={b: "Второй"}))
        assert sender.calls == [
            {"message": "", "media_files": [(a, False), (b, False)], "caption": "Второй"},
        ]
    finally:
        os.unlink(a), os.unlink(b)


# ---------------------------------------------------------------------------
# _send_live_adapter_media — live gateway adapter path
# ---------------------------------------------------------------------------


class _LiveAdapter:
    """Minimal adapter with its own media methods (not the BasePlatformAdapter stubs)."""

    def __init__(self):
        self.calls = []

    async def send(self, *, chat_id, content, metadata=None):
        self.calls.append(("send", content))
        return SimpleNamespace(success=True, message_id="t1")

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, metadata=None):
        self.calls.append(("image", image_path, caption))
        return SimpleNamespace(success=True, message_id="m1")

    async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None, metadata=None):
        self.calls.append(("document", file_path, caption))
        return SimpleNamespace(success=True, message_id="m1")


def test_live_adapter_tag_caption_rides_the_bubble():
    img = _tmpfile(".png")
    try:
        adapter = _LiveAdapter()
        result = asyncio.run(_send_live_adapter_media(
            adapter, "ch", "", [(img, False)], media_captions={img: "Подпись"}))
        assert result.get("success") is True
        assert adapter.calls == [("image", img, "Подпись")]
    finally:
        os.unlink(img)


def test_live_adapter_tag_caption_keeps_body_as_text():
    img = _tmpfile(".png")
    try:
        adapter = _LiveAdapter()
        asyncio.run(_send_live_adapter_media(
            adapter, "ch", "Билет 1", [(img, False)], media_captions={img: "Подпись"}))
        assert adapter.calls == [("send", "Билет 1"), ("image", img, "Подпись")]
    finally:
        os.unlink(img)


def test_live_adapter_caption_per_file_for_multi_media():
    a, b = _tmpfile(".png"), _tmpfile(".png")
    try:
        adapter = _LiveAdapter()
        asyncio.run(_send_live_adapter_media(
            adapter, "ch", "", [(a, False), (b, False)], media_captions={b: "Второй"}))
        assert adapter.calls == [("image", a, None), ("image", b, "Второй")]
    finally:
        os.unlink(a), os.unlink(b)


def test_live_adapter_without_tag_caption_is_unchanged():
    """Backward compatibility: short body + one captionable file becomes that file's caption."""
    img = _tmpfile(".png")
    try:
        adapter = _LiveAdapter()
        asyncio.run(_send_live_adapter_media(adapter, "ch", "Short body", [(img, False)]))
        assert adapter.calls == [("image", img, "Short body")]
    finally:
        os.unlink(img)


# ---------------------------------------------------------------------------
# _send_to_platform — routing: bubble caption vs folded text
# ---------------------------------------------------------------------------


def _patch_discord_registry(monkeypatch, sender):
    from hermes_cli.plugins import discover_plugins
    from gateway.platform_registry import platform_registry
    discover_plugins()
    entry = platform_registry.get("discord")
    original = entry.standalone_sender_fn
    monkeypatch.setattr(entry, "standalone_sender_fn", sender)
    return original


def test_send_to_platform_discord_passes_tag_caption(monkeypatch):
    img = _tmpfile(".png")
    try:
        sender = _RecordingSender()
        _patch_discord_registry(monkeypatch, sender)
        result = asyncio.run(_send_to_platform(
            Platform.DISCORD, _pconfig(), "ch", "", media_files=[(img, False)],
            media_captions={img: "Подпись"}))
        assert result["success"] is True
        assert sender.calls == [{"message": "", "media_files": [(img, False)], "caption": "Подпись"}]
    finally:
        os.unlink(img)


def test_send_to_platform_signal_folds_caption_into_text(monkeypatch):
    """Signal's JSON-RPC send has no caption field: the caption must arrive as text."""
    import tools.send_message_tool as smt
    img = _tmpfile(".png")
    sent = []

    async def fake_send_signal(extra, chat_id, chunk, media_files=None):
        sent.append((chunk, media_files))
        return {"success": True, "platform": "signal", "chat_id": chat_id}

    monkeypatch.setattr(smt, "_send_signal", fake_send_signal)
    try:
        result = asyncio.run(_send_to_platform(
            Platform.SIGNAL,
            SimpleNamespace(enabled=True, token=None,
                            extra={"http_url": "http://localhost:8080", "account": "+155****4567"}),
            "+155****4321", "", media_files=[(img, False)], media_captions={img: "Подпись"}))
        assert result["success"] is True
        assert sent[-1][0] == "Подпись"
        assert sent[-1][1] == [(img, False)]
    finally:
        os.unlink(img)


def test_captionless_text_only_route_keeps_caption_in_body(monkeypatch):
    """A text-only platform (no native media) drops the attachment — the caption text survives."""
    import tools.send_message_tool as smt
    img = _tmpfile(".png")
    sent = []

    async def fake_registry_send(platform_name, pconfig, message, thread_id=None):
        sent.append((platform_name, message))
        return {"success": True}

    monkeypatch.setitem(smt._TEXT_SENDERS, "email", lambda pc, cid, chunk, tid: fake_registry_send("email", pc, chunk, tid))
    try:
        result = asyncio.run(_send_to_platform(
            Platform.EMAIL, _pconfig(), "a@b.c", "", media_files=[(img, False)],
            media_captions={img: "Подпись"}))
        assert result["success"] is True
        assert sent == [("email", "Подпись")]
    finally:
        os.unlink(img)


# ---------------------------------------------------------------------------
# End-to-end: hermes send body -> extract_media_captions -> routing -> wire
# ---------------------------------------------------------------------------


def _payload_json_content(form_data):
    """Extract the 'content' from an aiohttp FormData's payload_json field, if any."""
    import json
    for field in getattr(form_data, "_fields", []):
        try:
            type_opts, value = field[0], field[2]
        except (IndexError, TypeError):
            continue
        if type_opts.get("name") == "payload_json":
            return json.loads(value).get("content")
    return None


def test_handle_send_reaches_discord_wire_with_tag_caption(monkeypatch):
    """Full chain with the real Discord sender; only the HTTP transport is stubbed.

    A non-Telegram platform is not configured on this host, so this is the closest a
    live E2E can get: the caption must land in the multipart ``payload_json`` content
    (one POST, no separate uncaptioned upload).
    """
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    import tools.send_message_tool as smt
    from plugins.platforms.discord.adapter import _remember_channel_is_forum

    img = _tmpfile(".png")
    chat_id = "999000444"
    _remember_channel_is_forum(chat_id, False)
    posts = []

    def _resp():
        r = AsyncMock()
        r.status = 200
        r.json = AsyncMock(return_value={"id": "m1"})
        r.text = AsyncMock(return_value="")
        r.content = MagicMock()
        r.content.read = AsyncMock(side_effect=[json.dumps({"id": "m1"}).encode(), b"", b""])
        r.get_encoding = MagicMock(return_value="utf-8")
        return r

    def _post(url, **kwargs):
        posts.append((url, kwargs.get("json"), kwargs.get("data")))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_resp())
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = MagicMock()
    session.post = MagicMock(side_effect=_post)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    cfg = SimpleNamespace(platforms={Platform.DISCORD: SimpleNamespace(enabled=True, token="tok", extra={})},
                          get_home_channel=lambda _p: None)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)
    from hermes_cli.plugins import discover_plugins
    discover_plugins()

    with patch("aiohttp.ClientSession", return_value=session_ctx):
        out = json.loads(smt.send_message_tool({
            "action": "send", "target": f"discord:{chat_id}",
            "message": f"MEDIA:{img} | Подпись на баббле"}))
    try:
        assert out.get("success") is True, out
        assert len(posts) == 1, posts
        assert _payload_json_content(posts[0][2]) == "Подпись на баббле"
    finally:
        os.unlink(img)


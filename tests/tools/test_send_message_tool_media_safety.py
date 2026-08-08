import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from tools.send_message_tool import _send_telegram


def _install_fake_telegram(monkeypatch, bot):
    telegram_mod = types.ModuleType("telegram")

    class FakeBot:
        def __init__(self, token, **_kwargs):
            self._bot = bot

        def __getattr__(self, name):
            return getattr(self._bot, name)

    telegram_mod.Bot = FakeBot

    constants_mod = types.ModuleType("telegram.constants")
    constants_mod.ParseMode = SimpleNamespace(HTML="HTML", MARKDOWN_V2="MarkdownV2")

    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def test_send_telegram_skips_unsafe_media_path(monkeypatch, tmp_path):
    test_file = tmp_path / "doc.txt"
    test_file.write_text("secret")

    bot = MagicMock()
    bot.send_message = AsyncMock()
    bot.send_photo = AsyncMock()
    bot.send_video = AsyncMock()
    bot.send_voice = AsyncMock()
    bot.send_audio = AsyncMock()
    bot.send_document = AsyncMock()
    _install_fake_telegram(monkeypatch, bot)
    monkeypatch.setattr(
        "gateway.platforms.base.BasePlatformAdapter.validate_media_delivery_path",
        lambda _path: None,
    )

    result = asyncio.run(
        _send_telegram(
            "token",
            "12345",
            "",
            media_files=[(str(test_file), False)],
        )
    )

    assert "error" in result
    assert "No deliverable text or media remained" in result["error"]
    assert result["warnings"] == [
        f"Skipping unsafe media path outside allowed roots: {test_file}"
    ]
    bot.send_document.assert_not_awaited()

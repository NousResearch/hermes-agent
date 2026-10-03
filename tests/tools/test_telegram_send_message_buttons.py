"""Standalone Telegram inline-keyboard buttons via `hermes send --buttons ...`.

Covers `_send_telegram`'s reply_markup attachment (tools/send_message_senders.py),
`_validate_buttons`'s platform gate (tools/send_message_tool.py), and the CLI's
`--buttons` spec parser (hermes_cli/send_cmd.py). No real Telegram network access —
the `telegram` package is stubbed exactly like test_telegram_send_message_caption.py.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_telegram_mock(monkeypatch: pytest.MonkeyPatch, bot_factory: MagicMock) -> None:
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)

    class _FakeButton:
        def __init__(self, label, callback_data=None):
            self.label = label
            self.callback_data = callback_data

    class _FakeMarkup:
        def __init__(self, rows):
            self.rows = rows

    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        MessageEntity=lambda **kw: SimpleNamespace(**kw),
        InlineKeyboardButton=_FakeButton,
        InlineKeyboardMarkup=_FakeMarkup,
        constants=constants_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def _make_bot() -> MagicMock:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    return bot


def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY",
                "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)


class TestSendTelegramButtons:
    def test_single_row_attaches_reply_markup_to_last_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_tool import _send_telegram

        _no_proxy(monkeypatch)
        bot = _make_bot()
        _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
        res = asyncio.run(_send_telegram(
            "tok", "123", "Consent request", buttons=[("YES", "cr:approve:CR-1"), ("NO", "cr:deny:CR-1")]))
        assert res["success"] is True
        bot.send_message.assert_awaited_once()
        markup = bot.send_message.await_args.kwargs.get("reply_markup")
        assert markup is not None
        assert len(markup.rows) == 1
        assert [b.callback_data for b in markup.rows[0]] == ["cr:approve:CR-1", "cr:deny:CR-1"]
        assert [b.label for b in markup.rows[0]] == ["YES", "NO"]

    def test_no_buttons_no_reply_markup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools.send_message_tool import _send_telegram

        _no_proxy(monkeypatch)
        bot = _make_bot()
        _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
        res = asyncio.run(_send_telegram("tok", "123", "plain text"))
        assert res["success"] is True
        assert "reply_markup" not in bot.send_message.await_args.kwargs

    def test_buttons_dropped_with_media(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os
        import tempfile

        from tools.send_message_tool import _send_telegram

        _no_proxy(monkeypatch)
        bot = _make_bot()
        bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=2))
        _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.write(b"x")
        f.close()
        try:
            res = asyncio.run(_send_telegram(
                "tok", "123", "caption", media_files=[(f.name, False)], buttons=[("YES", "cr:approve:CR-1")]))
            assert res["success"] is True
            assert any("dropped" in w for w in res.get("warnings", []))
            assert "reply_markup" not in bot.send_photo.await_args.kwargs
        finally:
            os.unlink(f.name)


class TestValidateButtons:
    def test_none_is_noop(self):
        from tools.send_message_tool import _validate_buttons

        assert _validate_buttons(None, "telegram") == (None, None)

    def test_non_telegram_platform_rejected(self):
        from tools.send_message_tool import _validate_buttons

        buttons, err = _validate_buttons([["YES", "x"]], "discord")
        assert buttons is None
        assert "telegram" in err

    def test_oversized_callback_data_rejected(self):
        from tools.send_message_tool import _validate_buttons

        buttons, err = _validate_buttons([["YES", "x" * 65]], "telegram")
        assert buttons is None
        assert "64-byte" in err

    def test_multi_row_normalized(self):
        from tools.send_message_tool import _validate_buttons

        # Use non-reserved prefixes for the normalisation shape test.
        buttons, err = _validate_buttons([[["YES", "app:approve:1"]], [["NO", "app:deny:1"]]], "telegram")
        assert err is None
        assert buttons == [[("YES", "app:approve:1")], [("NO", "app:deny:1")]]

    def test_reserved_prefix_rejected(self):
        from tools.send_message_tool import _validate_buttons

        # cr:, ea:, sc: etc. are reserved for the adapter's own button handlers.
        # Generic callers must not be able to forge them via send_message.
        buttons, err = _validate_buttons([["YES", "cr:yes:CR-1"]], "telegram")
        assert buttons is None
        assert "reserved adapter prefix" in err

        buttons2, err2 = _validate_buttons([["APPROVE", "ea:approve:99"]], "telegram")
        assert buttons2 is None
        assert "reserved adapter prefix" in err2


class TestParseButtonsArg:
    def test_single_row(self):
        from hermes_cli.send_cmd import _parse_buttons_arg

        assert _parse_buttons_arg("YES=ok:approve:1|NO=ok:deny:1") == [
            ["YES", "ok:approve:1"], ["NO", "ok:deny:1"]]

    def test_multi_row(self):
        from hermes_cli.send_cmd import _parse_buttons_arg

        assert _parse_buttons_arg("A=1;B=2") == [[["A", "1"]], [["B", "2"]]]

    def test_none_input(self):
        from hermes_cli.send_cmd import _parse_buttons_arg

        assert _parse_buttons_arg(None) is None
        assert _parse_buttons_arg("") is None

    def test_malformed_raises(self):
        from hermes_cli.send_cmd import _parse_buttons_arg

        with pytest.raises(ValueError):
            _parse_buttons_arg("YES-no-equals-sign")

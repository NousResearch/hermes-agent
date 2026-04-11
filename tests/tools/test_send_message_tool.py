"""Tests for tools/send_message_tool.py."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from tools.send_message_tool import (
    _parse_target_ref,
    _send_discord,
    _send_telegram,
    _send_to_platform,
    send_message_tool,
)


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_config():
    telegram_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    return SimpleNamespace(
        platforms={Platform.TELEGRAM: telegram_cfg},
        get_home_channel=lambda _platform: None,
    ), telegram_cfg


def _install_telegram_mock(monkeypatch, bot):
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    telegram_mod = SimpleNamespace(Bot=lambda token: bot, constants=constants_mod)
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def _ensure_slack_mock(monkeypatch):
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


class TestSendMessageTool:
    def test_cron_duplicate_target_is_skipped_and_explained(self):
        home = SimpleNamespace(chat_id="-1001")
        config, _telegram_cfg = _make_config()
        config.get_home_channel = lambda _platform: home

        with patch.dict(
            os.environ,
            {
                "HERMES_CRON_AUTO_DELIVER_PLATFORM": "telegram",
                "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "-1001",
            },
            clear=False,
        ), \
             patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        assert result["skipped"] is True
        assert result["reason"] == "cron_auto_delivery_duplicate_target"
        assert "final response" in result["note"]
        send_mock.assert_not_awaited()
        mirror_mock.assert_not_called()

    def test_cron_different_target_still_sends(self):
        config, telegram_cfg = _make_config()

        with patch.dict(
            os.environ,
            {
                "HERMES_CRON_AUTO_DELIVER_PLATFORM": "telegram",
                "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "-1001",
            },
            clear=False,
        ), \
             patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:-1002",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        assert result.get("skipped") is not True
        send_mock.assert_awaited_once_with(
            Platform.TELEGRAM,
            telegram_cfg,
            "-1002",
            "hello",
            thread_id=None,
            media_files=[],
        )
        mirror_mock.assert_called_once_with("telegram", "-1002", "hello", source_label="cli", thread_id=None)

    def test_cron_same_chat_different_thread_still_sends(self):
        config, telegram_cfg = _make_config()

        with patch.dict(
            os.environ,
            {
                "HERMES_CRON_AUTO_DELIVER_PLATFORM": "telegram",
                "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "-1001",
                "HERMES_CRON_AUTO_DELIVER_THREAD_ID": "17585",
            },
            clear=False,
        ), \
             patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:-1001:99999",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        assert result.get("skipped") is not True
        send_mock.assert_awaited_once_with(
            Platform.TELEGRAM,
            telegram_cfg,
            "-1001",
            "hello",
            thread_id="99999",
            media_files=[],
        )
        mirror_mock.assert_called_once_with("telegram", "-1001", "hello", source_label="cli", thread_id="99999")

    def test_sends_to_explicit_telegram_topic_target(self):
        config, telegram_cfg = _make_config()

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:-1001:17585",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.TELEGRAM,
            telegram_cfg,
            "-1001",
            "hello",
            thread_id="17585",
            media_files=[],
        )
        mirror_mock.assert_called_once_with("telegram", "-1001", "hello", source_label="cli", thread_id="17585")

    def test_resolved_telegram_topic_name_preserves_thread_id(self):
        config, telegram_cfg = _make_config()

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("gateway.channel_directory.resolve_channel_name", return_value="-1001:17585"), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:Coaching Chat / topic 17585",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.TELEGRAM,
            telegram_cfg,
            "-1001",
            "hello",
            thread_id="17585",
            media_files=[],
        )

    def test_display_label_target_resolves_via_channel_directory(self, tmp_path):
        config, telegram_cfg = _make_config()
        cache_file = tmp_path / "channel_directory.json"
        cache_file.write_text(json.dumps({
            "updated_at": "2026-01-01T00:00:00",
            "platforms": {
                "telegram": [
                    {"id": "-1001:17585", "name": "Coaching Chat / topic 17585", "type": "group"}
                ]
            },
        }))

        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file), \
             patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:Coaching Chat / topic 17585 (group)",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.TELEGRAM,
            telegram_cfg,
            "-1001",
            "hello",
            thread_id="17585",
            media_files=[],
        )

    def test_media_only_message_uses_placeholder_for_mirroring(self):
        config, telegram_cfg = _make_config()

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:-1001",
                        "message": "MEDIA:/tmp/example.ogg",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.TELEGRAM,
            telegram_cfg,
            "-1001",
            "",
            thread_id=None,
            media_files=[("/tmp/example.ogg", False)],
        )
        mirror_mock.assert_called_once_with(
            "telegram",
            "-1001",
            "[Sent audio attachment]",
            source_label="cli",
            thread_id=None,
        )

    def test_top_level_send_failure_redacts_query_token(self):
        config, _telegram_cfg = _make_config()
        leaked = "very-secret-query-token-123456"

        def _raise_and_close(coro):
            coro.close()
            raise RuntimeError(
                f"transport error: https://api.example.com/send?access_token={leaked}"
            )

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_raise_and_close):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:-1001",
                        "message": "hello",
                    }
                )
            )

        assert "error" in result
        assert leaked not in result["error"]
        assert "access_token=***" in result["error"]


class TestSendTelegramMediaDelivery:
    def test_sends_text_then_photo_for_media_tag(self, tmp_path, monkeypatch):
        image_path = tmp_path / "photo.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
        bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=2))
        bot.send_video = AsyncMock()
        bot.send_voice = AsyncMock()
        bot.send_audio = AsyncMock()
        bot.send_document = AsyncMock()
        _install_telegram_mock(monkeypatch, bot)

        result = asyncio.run(
            _send_telegram(
                "token",
                "12345",
                "Hello there",
                media_files=[(str(image_path), False)],
            )
        )

        assert result["success"] is True
        assert result["message_id"] == "2"
        bot.send_message.assert_awaited_once()
        bot.send_photo.assert_awaited_once()
        sent_text = bot.send_message.await_args.kwargs["text"]
        assert "MEDIA:" not in sent_text
        assert sent_text == "Hello there"

    def test_sends_voice_for_ogg_with_voice_directive(self, tmp_path, monkeypatch):
        voice_path = tmp_path / "voice.ogg"
        voice_path.write_bytes(b"OggS" + b"\x00" * 32)

        bot = MagicMock()
        bot.send_message = AsyncMock()
        bot.send_photo = AsyncMock()
        bot.send_video = AsyncMock()
        bot.send_voice = AsyncMock(return_value=SimpleNamespace(message_id=7))
        bot.send_audio = AsyncMock()
        bot.send_document = AsyncMock()
        _install_telegram_mock(monkeypatch, bot)

        result = asyncio.run(
            _send_telegram(
                "token",
                "12345",
                "",
                media_files=[(str(voice_path), True)],
            )
        )

        assert result["success"] is True
        bot.send_voice.assert_awaited_once()
        bot.send_audio.assert_not_awaited()
        bot.send_message.assert_not_awaited()

    def test_sends_audio_for_mp3(self, tmp_path, monkeypatch):
        audio_path = tmp_path / "clip.mp3"
        audio_path.write_bytes(b"ID3" + b"\x00" * 32)

        bot = MagicMock()
        bot.send_message = AsyncMock()
        bot.send_photo = AsyncMock()
        bot.send_video = AsyncMock()
        bot.send_voice = AsyncMock()
        bot.send_audio = AsyncMock(return_value=SimpleNamespace(message_id=8))
        bot.send_document = AsyncMock()
        _install_telegram_mock(monkeypatch, bot)

        result = asyncio.run(
            _send_telegram(
                "token",
                "12345",
                "",
                media_files=[(str(audio_path), False)],
            )
        )

        assert result["success"] is True
        bot.send_audio.assert_awaited_once()
        bot.send_voice.assert_not_awaited()

    def test_missing_media_returns_error_without_leaking_raw_tag(self, monkeypatch):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        bot.send_photo = AsyncMock()
        bot.send_video = AsyncMock()
        bot.send_voice = AsyncMock()
        bot.send_audio = AsyncMock()
        bot.send_document = AsyncMock()
        _install_telegram_mock(monkeypatch, bot)

        result = asyncio.run(
            _send_telegram(
                "token",
                "12345",
                "",
                media_files=[("/tmp/does-not-exist.png", False)],
            )
        )

        assert "error" in result
        assert "No deliverable text or media remained" in result["error"]
        bot.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# Regression: long messages are chunked before platform dispatch
# ---------------------------------------------------------------------------


class TestSendToPlatformChunking:
    def test_long_message_is_chunked(self):
        """Messages exceeding the platform limit are split into multiple sends."""
        send = AsyncMock(return_value={"success": True, "message_id": "1"})
        long_msg = "word " * 1000  # ~5000 chars, well over Discord's 2000 limit
        with patch("tools.send_message_tool._send_discord", send):
            result = asyncio.run(
                _send_to_platform(
                    Platform.DISCORD,
                    SimpleNamespace(enabled=True, token="***", extra={}),
                    "ch", long_msg,
                )
            )
        assert result["success"] is True
        assert send.await_count >= 3
        for call in send.await_args_list:
            assert len(call.args[2]) <= 2020  # each chunk fits the limit

    def test_slack_messages_are_formatted_before_send(self, monkeypatch):
        _ensure_slack_mock(monkeypatch)

        import gateway.platforms.slack as slack_mod

        monkeypatch.setattr(slack_mod, "SLACK_AVAILABLE", True)
        send = AsyncMock(return_value={"success": True, "message_id": "1"})

        with patch("tools.send_message_tool._send_slack", send):
            result = asyncio.run(
                _send_to_platform(
                    Platform.SLACK,
                    SimpleNamespace(enabled=True, token="***", extra={}),
                    "C123",
                    "**hello** from [Hermes](<https://example.com>)",
                )
            )

        assert result["success"] is True
        send.assert_awaited_once_with(
            "***",
            "C123",
            "*hello* from <https://example.com|Hermes>",
        )

    def test_slack_bold_italic_formatted_before_send(self, monkeypatch):
        """Bold+italic ***text*** survives tool-layer formatting."""
        _ensure_slack_mock(monkeypatch)
        import gateway.platforms.slack as slack_mod

        monkeypatch.setattr(slack_mod, "SLACK_AVAILABLE", True)
        send = AsyncMock(return_value={"success": True, "message_id": "1"})
        with patch("tools.send_message_tool._send_slack", send):
            result = asyncio.run(
                _send_to_platform(
                    Platform.SLACK,
                    SimpleNamespace(enabled=True, token="***", extra={}),
                    "C123",
                    "***important*** update",
                )
            )
        assert result["success"] is True
        sent_text = send.await_args.args[2]
        assert "*_important_*" in sent_text

    def test_slack_blockquote_formatted_before_send(self, monkeypatch):
        """Blockquote '>' markers must survive formatting (not escaped to '&gt;')."""
        _ensure_slack_mock(monkeypatch)
        import gateway.platforms.slack as slack_mod

        monkeypatch.setattr(slack_mod, "SLACK_AVAILABLE", True)
        send = AsyncMock(return_value={"success": True, "message_id": "1"})
        with patch("tools.send_message_tool._send_slack", send):
            result = asyncio.run(
                _send_to_platform(
                    Platform.SLACK,
                    SimpleNamespace(enabled=True, token="***", extra={}),
                    "C123",
                    "> important quote\n\nnormal text & stuff",
                )
            )
        assert result["success"] is True
        sent_text = send.await_args.args[2]
        assert sent_text.startswith("> important quote")
        assert "&amp;" in sent_text  # & is escaped
        assert "&gt;" not in sent_text.split("\n")[0]  # > in blockquote is NOT escaped

    def test_slack_pre_escaped_entities_not_double_escaped(self, monkeypatch):
        """Pre-escaped HTML entities survive tool-layer formatting without double-escaping."""
        _ensure_slack_mock(monkeypatch)
        import gateway.platforms.slack as slack_mod
        monkeypatch.setattr(slack_mod, "SLACK_AVAILABLE", True)
        send = AsyncMock(return_value={"success": True, "message_id": "1"})
        with patch("tools.send_message_tool._send_slack", send):
            result = asyncio.run(
                _send_to_platform(
                    Platform.SLACK,
                    SimpleNamespace(enabled=True, token="***", extra={}),
                    "C123",
                    "AT&amp;T &lt;tag&gt; test",
                )
            )
        assert result["success"] is True
        sent_text = send.await_args.args[2]
        assert "&amp;amp;" not in sent_text
        assert "&amp;lt;" not in sent_text
        assert "AT&amp;T" in sent_text

    def test_slack_url_with_parens_formatted_before_send(self, monkeypatch):
        """Wikipedia-style URL with parens survives tool-layer formatting."""
        _ensure_slack_mock(monkeypatch)
        import gateway.platforms.slack as slack_mod
        monkeypatch.setattr(slack_mod, "SLACK_AVAILABLE", True)
        send = AsyncMock(return_value={"success": True, "message_id": "1"})
        with patch("tools.send_message_tool._send_slack", send):
            result = asyncio.run(
                _send_to_platform(
                    Platform.SLACK,
                    SimpleNamespace(enabled=True, token="***", extra={}),
                    "C123",
                    "See [Foo](https://en.wikipedia.org/wiki/Foo_(bar))",
                )
            )
        assert result["success"] is True
        sent_text = send.await_args.args[2]
        assert "<https://en.wikipedia.org/wiki/Foo_(bar)|Foo>" in sent_text

    def test_telegram_media_attaches_to_last_chunk(self):

        sent_calls = []

        async def fake_send(token, chat_id, message, media_files=None, thread_id=None):
            sent_calls.append(media_files or [])
            return {"success": True, "platform": "telegram", "chat_id": chat_id, "message_id": str(len(sent_calls))}

        long_msg = "word " * 2000  # ~10000 chars, well over 4096
        media = [("/tmp/photo.png", False)]
        with patch("tools.send_message_tool._send_telegram", fake_send):
            asyncio.run(
                _send_to_platform(
                    Platform.TELEGRAM,
                    SimpleNamespace(enabled=True, token="tok", extra={}),
                    "123", long_msg, media_files=media,
                )
            )
        assert len(sent_calls) >= 3
        assert all(call == [] for call in sent_calls[:-1])
        assert sent_calls[-1] == media


# ---------------------------------------------------------------------------
# HTML auto-detection in Telegram send
# ---------------------------------------------------------------------------


class TestSendToPlatformWhatsapp:
    def test_whatsapp_routes_via_local_bridge_sender(self):
        chat_id = "test-user@lid"
        async_mock = AsyncMock(return_value={"success": True, "platform": "whatsapp", "chat_id": chat_id, "message_id": "abc123"})

        with patch("tools.send_message_tool._send_whatsapp", async_mock):
            result = asyncio.run(
                _send_to_platform(
                    Platform.WHATSAPP,
                    SimpleNamespace(enabled=True, token=None, extra={"bridge_port": 3000}),
                    chat_id,
                    "hello from hermes",
                )
            )

        assert result["success"] is True
        async_mock.assert_awaited_once_with({"bridge_port": 3000}, chat_id, "hello from hermes")


class TestSendTelegramHtmlDetection:
    """Verify that messages containing HTML tags are sent with parse_mode=HTML
    and that plain / markdown messages use MarkdownV2."""

    def _make_bot(self):
        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
        bot.send_photo = AsyncMock()
        bot.send_video = AsyncMock()
        bot.send_voice = AsyncMock()
        bot.send_audio = AsyncMock()
        bot.send_document = AsyncMock()
        return bot

    def test_html_message_uses_html_parse_mode(self, monkeypatch):
        bot = self._make_bot()
        _install_telegram_mock(monkeypatch, bot)

        asyncio.run(
            _send_telegram("tok", "123", "<b>Hello</b> world")
        )

        bot.send_message.assert_awaited_once()
        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == "HTML"
        assert kwargs["text"] == "<b>Hello</b> world"

    def test_plain_text_uses_markdown_v2(self, monkeypatch):
        bot = self._make_bot()
        _install_telegram_mock(monkeypatch, bot)

        asyncio.run(
            _send_telegram("tok", "123", "Just plain text, no tags")
        )

        bot.send_message.assert_awaited_once()
        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == "MarkdownV2"

    def test_html_with_code_and_pre_tags(self, monkeypatch):
        bot = self._make_bot()
        _install_telegram_mock(monkeypatch, bot)

        html = "<pre>code block</pre> and <code>inline</code>"
        asyncio.run(_send_telegram("tok", "123", html))

        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == "HTML"

    def test_closing_tag_detected(self, monkeypatch):
        bot = self._make_bot()
        _install_telegram_mock(monkeypatch, bot)

        asyncio.run(_send_telegram("tok", "123", "text </div> more"))

        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == "HTML"

    def test_angle_brackets_in_math_not_detected(self, monkeypatch):
        """Expressions like 'x < 5' or '3 > 2' should not trigger HTML mode."""
        bot = self._make_bot()
        _install_telegram_mock(monkeypatch, bot)

        asyncio.run(_send_telegram("tok", "123", "if x < 5 then y > 2"))

        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == "MarkdownV2"

    def test_html_parse_failure_falls_back_to_plain(self, monkeypatch):
        """If Telegram rejects the HTML, fall back to plain text."""
        bot = self._make_bot()
        bot.send_message = AsyncMock(
            side_effect=[
                Exception("Bad Request: can't parse entities: unsupported html tag"),
                SimpleNamespace(message_id=2),  # plain fallback succeeds
            ]
        )
        _install_telegram_mock(monkeypatch, bot)

        result = asyncio.run(
            _send_telegram("tok", "123", "<invalid>broken html</invalid>")
        )

        assert result["success"] is True
        assert bot.send_message.await_count == 2
        second_call = bot.send_message.await_args_list[1].kwargs
        assert second_call["parse_mode"] is None


# ---------------------------------------------------------------------------
# Tests for Discord thread_id support
# ---------------------------------------------------------------------------


class TestParseTargetRefDiscord:
    """_parse_target_ref correctly extracts chat_id and thread_id for Discord."""

    def test_discord_chat_id_with_thread_id(self):
        """discord:chat_id:thread_id returns both values."""
        chat_id, thread_id, is_explicit = _parse_target_ref("discord", "-1001234567890:17585")
        assert chat_id == "-1001234567890"
        assert thread_id == "17585"
        assert is_explicit is True

    def test_discord_chat_id_without_thread_id(self):
        """discord:chat_id returns None for thread_id."""
        chat_id, thread_id, is_explicit = _parse_target_ref("discord", "9876543210")
        assert chat_id == "9876543210"
        assert thread_id is None
        assert is_explicit is True

    def test_discord_large_snowflake_without_thread(self):
        """Large Discord snowflake IDs work without thread."""
        chat_id, thread_id, is_explicit = _parse_target_ref("discord", "1003724596514")
        assert chat_id == "1003724596514"
        assert thread_id is None
        assert is_explicit is True

    def test_discord_channel_with_thread(self):
        """Full Discord format: channel:thread."""
        chat_id, thread_id, is_explicit = _parse_target_ref("discord", "1003724596514:99999")
        assert chat_id == "1003724596514"
        assert thread_id == "99999"
        assert is_explicit is True

    def test_discord_whitespace_is_stripped(self):
        """Whitespace around Discord targets is stripped."""
        chat_id, thread_id, is_explicit = _parse_target_ref("discord", "  123456:789  ")
        assert chat_id == "123456"
        assert thread_id == "789"
        assert is_explicit is True


class TestSendDiscordThreadId:
    """_send_discord uses thread_id when provided."""

    @staticmethod
    def _build_mock(response_status, response_data=None, response_text="error body"):
        """Build a properly-structured aiohttp mock chain.

        session.post() returns a context manager yielding mock_resp.
        """
        mock_resp = MagicMock()
        mock_resp.status = response_status
        mock_resp.json = AsyncMock(return_value=response_data or {"id": "msg123"})
        mock_resp.text = AsyncMock(return_value=response_text)

        # mock_resp as async context manager (for "async with session.post(...) as resp")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_resp)

        return mock_session, mock_resp

    def _run(self, token, chat_id, message, thread_id=None):
        return asyncio.run(_send_discord(token, chat_id, message, thread_id=thread_id))

    def test_without_thread_id_uses_chat_id_endpoint(self):
        """When no thread_id, sends to /channels/{chat_id}/messages."""
        mock_session, _ = self._build_mock(200)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            self._run("tok", "111222333", "hello world")
        call_url = mock_session.post.call_args.args[0]
        assert call_url == "https://discord.com/api/v10/channels/111222333/messages"

    def test_with_thread_id_uses_thread_endpoint(self):
        """When thread_id is provided, sends to /channels/{thread_id}/messages."""
        mock_session, _ = self._build_mock(200)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            self._run("tok", "999888777", "hello from thread", thread_id="555444333")
        call_url = mock_session.post.call_args.args[0]
        assert call_url == "https://discord.com/api/v10/channels/555444333/messages"

    def test_success_returns_message_id(self):
        """Successful send returns the Discord message ID."""
        mock_session, _ = self._build_mock(200, response_data={"id": "9876543210"})
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = self._run("tok", "111", "hi", thread_id="999")
        assert result["success"] is True
        assert result["message_id"] == "9876543210"
        assert result["chat_id"] == "111"

    def test_error_status_returns_error_dict(self):
        """Non-200/201 responses return an error dict."""
        mock_session, _ = self._build_mock(403, response_data={"message": "Forbidden"})
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = self._run("tok", "111", "hi")
        assert "error" in result
        assert "403" in result["error"]


class TestSendToPlatformDiscordThread:
    """_send_to_platform passes thread_id through to _send_discord."""

    def test_discord_thread_id_passed_to_send_discord(self):
        """Discord platform with thread_id passes it to _send_discord."""
        send_mock = AsyncMock(return_value={"success": True, "message_id": "1"})

        with patch("tools.send_message_tool._send_discord", send_mock):
            result = asyncio.run(
                _send_to_platform(
                    Platform.DISCORD,
                    SimpleNamespace(enabled=True, token="tok", extra={}),
                    "-1001234567890",
                    "hello thread",
                    thread_id="17585",
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once()
        _, call_kwargs = send_mock.await_args
        assert call_kwargs["thread_id"] == "17585"

    def test_discord_no_thread_id_when_not_provided(self):
        """Discord platform without thread_id passes None."""
        send_mock = AsyncMock(return_value={"success": True, "message_id": "1"})

        with patch("tools.send_message_tool._send_discord", send_mock):
            result = asyncio.run(
                _send_to_platform(
                    Platform.DISCORD,
                    SimpleNamespace(enabled=True, token="tok", extra={}),
                    "9876543210",
                    "hello channel",
                )
            )

        send_mock.assert_awaited_once()
        _, call_kwargs = send_mock.await_args
        assert call_kwargs["thread_id"] is None

"""Tests for tools/send_message_tool.py."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from tools.send_message_tool import _send_telegram, send_message_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_config():
    telegram_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    return SimpleNamespace(
        platforms={Platform.TELEGRAM: telegram_cfg},
        get_home_channel=lambda _platform: None,
    ), telegram_cfg


def _install_telegram_mock(monkeypatch, bot):
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    telegram_mod = SimpleNamespace(Bot=lambda token: bot, constants=constants_mod)
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


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

"""Tests for tools/send_message_tool.py."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from gateway.platforms.base import SendResult
from tools.send_message_tool import (
    _send_discord,
    _send_slack,
    _send_telegram,
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
    telegram_mod = SimpleNamespace(Bot=lambda token: bot)
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)


def _successful_telegram_text_send(message_id="1"):
    return SendResult(
        success=True,
        message_id=message_id,
        raw_response={"message_ids": [message_id]},
    )


class TestSendMessageTool:
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
        mirror_mock.assert_called_once_with(
            "telegram",
            "-1001",
            "hello",
            source_label="cli",
            thread_id="17585",
        )

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
        bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=2))
        bot.send_video = AsyncMock()
        bot.send_voice = AsyncMock()
        bot.send_audio = AsyncMock()
        bot.send_document = AsyncMock()
        _install_telegram_mock(monkeypatch, bot)

        with patch(
            "tools.send_message_tool._shared_send_telegram_text",
            new=AsyncMock(return_value=_successful_telegram_text_send()),
        ) as send_text_mock:
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
        send_text_mock.assert_awaited_once()
        assert send_text_mock.await_args.args[2] == "Hello there"
        bot.send_photo.assert_awaited_once()

    def test_sends_voice_for_ogg_with_voice_directive(self, tmp_path, monkeypatch):
        voice_path = tmp_path / "voice.ogg"
        voice_path.write_bytes(b"OggS" + b"\x00" * 32)

        bot = MagicMock()
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

    def test_sends_audio_for_mp3(self, tmp_path, monkeypatch):
        audio_path = tmp_path / "clip.mp3"
        audio_path.write_bytes(b"ID3" + b"\x00" * 32)

        bot = MagicMock()
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


class TestSharedOutboundBehavior:
    @staticmethod
    def _mock_aiohttp_json_session(payloads, response_factory):
        mock_session = MagicMock()

        def _post(*args, **kwargs):
            payloads.append(kwargs["json"])
            response = response_factory()
            response.__aenter__ = AsyncMock(return_value=response)
            response.__aexit__ = AsyncMock(return_value=False)
            return response

        mock_session.post = MagicMock(side_effect=_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return mock_session

    def test_send_slack_uses_adapter_formatting(self):
        payloads = []

        def _response():
            resp = MagicMock()
            resp.json = AsyncMock(return_value={"ok": True, "ts": "123.456"})
            return resp

        mock_session = self._mock_aiohttp_json_session(payloads, _response)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(_send_slack("xoxb-token", "C123", "**hello**"))

        assert result["success"] is True
        assert payloads[0]["text"] == "*hello*"

    def test_send_discord_uses_code_fence_aware_chunking(self):
        payloads = []
        message_ids = iter(["m1", "m2", "m3", "m4"])

        def _response():
            resp = MagicMock()
            resp.status = 200
            resp.text = AsyncMock(return_value="OK")
            resp.json = AsyncMock(return_value={"id": next(message_ids, "mx")})
            return resp

        mock_session = self._mock_aiohttp_json_session(payloads, _response)
        long_code = "Before\n```python\n" + ("x = 1\n" * 500) + "```\nAfter"

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(_send_discord("bot-token", "123", long_code))

        assert result["success"] is True
        assert len(payloads) > 1
        assert all(payload["content"].count("```") % 2 == 0 for payload in payloads)

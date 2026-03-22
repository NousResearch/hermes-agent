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
    _send_kasia,
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

    def test_kasia_send_mirrors_to_resolved_chat_id(self):
        kasia_cfg = SimpleNamespace(enabled=True, token=None, extra={"bridge_port": 3010})
        config = SimpleNamespace(
            platforms={Platform.KASIA: kasia_cfg},
            get_home_channel=lambda _platform: None,
        )

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch(
                 "tools.send_message_tool._send_to_platform",
                 new=AsyncMock(
                     return_value={
                         "success": True,
                         "platform": "kasia",
                         "chat_id": "kaspa:qresolvedfriend",
                         "message_id": "job-1",
                     }
                 ),
             ) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "kasia:friend.kas",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.KASIA,
            kasia_cfg,
            "friend.kas",
            "hello",
            thread_id=None,
            media_files=[],
        )
        mirror_mock.assert_called_once_with(
            "kasia",
            "kaspa:qresolvedfriend",
            "hello",
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
                    SimpleNamespace(enabled=True, token="tok", extra={}),
                    "ch", long_msg,
                )
            )
        assert result["success"] is True
        assert send.await_count >= 3
        for call in send.await_args_list:
            assert len(call.args[2]) <= 2020  # each chunk fits the limit

    def test_telegram_media_attaches_to_last_chunk(self):
        """When chunked, media files are sent only with the last chunk."""
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


class TestSendToPlatformKasia:
    def test_kasia_target_parser_treats_prefixed_address_as_explicit(self):
        chat_id, thread_id, is_explicit = _parse_target_ref(
            "kasia",
            "kaspa:qpeeraddress",
        )

        assert chat_id == "kaspa:qpeeraddress"
        assert thread_id is None
        assert is_explicit is True

    def test_kasia_target_parser_treats_kns_name_as_explicit(self):
        chat_id, thread_id, is_explicit = _parse_target_ref(
            "kasia",
            "friend.kas",
        )

        assert chat_id == "friend.kas"
        assert thread_id is None
        assert is_explicit is True

    def test_kasia_target_parser_treats_broadcast_target_as_explicit(self):
        chat_id, thread_id, is_explicit = _parse_target_ref(
            "kasia",
            "broadcast:news",
        )

        assert chat_id == "broadcast:news"
        assert thread_id is None
        assert is_explicit is True

    def test_kasia_routes_via_local_bridge_sender(self):
        chat_id = "kaspa:qpeeraddress"
        async_mock = AsyncMock(
            return_value={
                "success": True,
                "platform": "kasia",
                "chat_id": chat_id,
                "message_id": "job-1",
                "job_id": "job-1",
                "status": "submitted",
                "status_message": "Submitted to the Kaspa node. Waiting for indexer visibility.",
            }
        )

        with patch("tools.send_message_tool._send_kasia", async_mock):
            result = asyncio.run(
                _send_to_platform(
                    Platform.KASIA,
                    SimpleNamespace(enabled=True, token=None, extra={"bridge_port": 3010}),
                    chat_id,
                    "hello from hermes",
                )
            )

        assert result["success"] is True
        assert result["status"] == "submitted"
        assert "Waiting for indexer visibility" in result["status_message"]
        async_mock.assert_awaited_once_with({"bridge_port": 3010}, chat_id, "hello from hermes")

    def test_kasia_initiate_routes_via_local_bridge_sender(self):
        chat_id = "friend.kas"
        async_mock = AsyncMock(
            return_value={
                "success": True,
                "platform": "kasia",
                "chat_id": "kaspa:qresolved",
                "message_id": "tx-init",
                "status": "sent",
            }
        )

        with patch("tools.send_message_tool._send_kasia", async_mock):
            result = asyncio.run(
                _send_to_platform(
                    Platform.KASIA,
                    SimpleNamespace(enabled=True, token=None, extra={"bridge_port": 3010}),
                    chat_id,
                    "",
                    action="initiate",
                    display_name="Peer",
                )
            )

        assert result["success"] is True
        assert result["status"] == "sent"
        async_mock.assert_awaited_once_with(
            {"bridge_port": 3010},
            chat_id,
            "",
            action="initiate",
            display_name="Peer",
            retry=False,
        )

    def test_kasia_initiate_resolves_kns_before_authorization(self, monkeypatch):
        calls = []

        class FakeResponse:
            def __init__(self, status, payload):
                self.status = status
                self._payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self):
                return self._payload

            async def text(self):
                return json.dumps(self._payload)

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def get(self, url, **kwargs):
                calls.append(("get", url))
                assert url.endswith("/resolve-target/friend.kas")
                return FakeResponse(
                    200,
                    {
                        "kind": "dm",
                        "chatId": "kaspa:qresolvedfriend",
                        "peerAddress": "kaspa:qresolvedfriend",
                        "knsName": "friend.kas",
                    },
                )

            def post(self, url, **kwargs):
                calls.append(("post", url, kwargs.get("json")))
                assert url.endswith("/handshakes/initiate")
                return FakeResponse(
                    200,
                    {
                        "status": "sent",
                        "txId": "tx-init",
                        "chatId": "kaspa:qresolvedfriend",
                    },
                )

        fake_aiohttp = SimpleNamespace(
            ClientSession=lambda: FakeSession(),
            ClientTimeout=lambda total: SimpleNamespace(total=total),
        )
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setenv("KASIA_ALLOWED_USERS", "kaspa:qresolvedfriend")
        monkeypatch.delenv("KASIA_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)

        result = asyncio.run(
            _send_kasia(
                {"bridge_port": 3010},
                "friend.kas",
                "",
                action="initiate",
                display_name="Friend",
            )
        )

        assert result["success"] is True
        assert result["chat_id"] == "kaspa:qresolvedfriend"
        assert calls[0][0] == "get"
        assert calls[1][0] == "post"
        assert calls[1][2]["chatId"] == "kaspa:qresolvedfriend"

    def test_kasia_send_resolves_kns_before_delivery(self, monkeypatch):
        calls = []

        class FakeResponse:
            def __init__(self, status, payload):
                self.status = status
                self._payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self):
                return self._payload

            async def text(self):
                return json.dumps(self._payload)

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def get(self, url, **kwargs):
                calls.append(("get", url))
                assert url.endswith("/resolve-target/friend.kas")
                return FakeResponse(
                    200,
                    {
                        "kind": "dm",
                        "chatId": "kaspa:qresolvedfriend",
                        "peerAddress": "kaspa:qresolvedfriend",
                        "knsName": "friend.kas",
                    },
                )

            def post(self, url, **kwargs):
                calls.append(("post", url, kwargs.get("json")))
                assert url.endswith("/send")
                return FakeResponse(
                    200,
                    {
                        "status": "submitted",
                        "jobId": "job-1",
                        "chatId": "kaspa:qresolvedfriend",
                    },
                )

        fake_aiohttp = SimpleNamespace(
            ClientSession=lambda: FakeSession(),
            ClientTimeout=lambda total: SimpleNamespace(total=total),
        )
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

        result = asyncio.run(
            _send_kasia(
                {"bridge_port": 3010},
                "friend.kas",
                "hello",
            )
        )

        assert result["success"] is True
        assert result["chat_id"] == "kaspa:qresolvedfriend"
        assert calls[0][0] == "get"
        assert calls[1][0] == "post"
        assert calls[1][2]["chatId"] == "kaspa:qresolvedfriend"


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

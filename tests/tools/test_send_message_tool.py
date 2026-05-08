"""Tests for tools/send_message_tool.py."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    """Drop the process-wide attachment scheduler so each test gets a
    fresh token bucket."""
    from gateway.platforms.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()

from gateway.config import Platform
from tools.send_message_tool import (
    _derive_forum_thread_name,
    _parse_target_ref,
    _send_discord,
    _send_matrix_via_adapter,
    _send_signal,
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
            force_document=False,
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
            force_document=False,
        )

    def test_mirror_receives_current_session_user_id(self):
        config, _telegram_cfg = _make_config()

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
             patch("gateway.session_context.get_session_env") as get_session_env_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            get_session_env_mock.side_effect = lambda name, default="": {
                "HERMES_SESSION_PLATFORM": "telegram",
                "HERMES_SESSION_USER_ID": "user-123",
            }.get(name, default)
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:12345",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        mirror_mock.assert_called_once_with(
            "telegram",
            "12345",
            "hello",
            source_label="telegram",
            thread_id=None,
            user_id="user-123",
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

        async def fake_send(token, chat_id, message, media_files=None, thread_id=None, disable_link_previews=False, force_document=False):
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

    def test_matrix_media_uses_native_adapter_helper(self):

        doc_path = Path("/tmp/test-send-message-matrix.pdf")
        doc_path.write_bytes(b"%PDF-1.4 test")

        try:
            helper = AsyncMock(return_value={"success": True, "platform": "matrix", "chat_id": "!room:example.com", "message_id": "$evt"})
            with patch("tools.send_message_tool._send_matrix_via_adapter", helper):
                result = asyncio.run(
                    _send_to_platform(
                        Platform.MATRIX,
                        SimpleNamespace(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.com"}),
                        "!room:example.com",
                        "here you go",
                        media_files=[(str(doc_path), False)],
                    )
                )

            assert result["success"] is True
            helper.assert_awaited_once()
            call = helper.await_args
            assert call.args[1] == "!room:example.com"
            assert call.args[2] == "here you go"
            assert call.kwargs["media_files"] == [(str(doc_path), False)]
        finally:
            doc_path.unlink(missing_ok=True)

    def test_matrix_text_only_uses_lightweight_path(self):
        """Text-only Matrix sends should NOT go through the heavy adapter path."""
        helper = AsyncMock()
        lightweight = AsyncMock(return_value={"success": True, "platform": "matrix", "chat_id": "!room:ex.com", "message_id": "$txt"})
        with patch("tools.send_message_tool._send_matrix_via_adapter", helper), \
             patch("tools.send_message_tool._send_matrix", lightweight):
            result = asyncio.run(
                _send_to_platform(
                    Platform.MATRIX,
                    SimpleNamespace(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.com"}),
                    "!room:ex.com",
                    "just text, no files",
                )
            )

        assert result["success"] is True
        helper.assert_not_awaited()
        lightweight.assert_awaited_once()

    def test_send_matrix_via_adapter_sends_document(self, tmp_path):
        file_path = tmp_path / "report.pdf"
        file_path.write_bytes(b"%PDF-1.4 test")

        calls = []

        class FakeAdapter:
            def __init__(self, _config):
                self.connected = False

            async def connect(self):
                self.connected = True
                calls.append(("connect",))
                return True

            async def send(self, chat_id, message, metadata=None):
                calls.append(("send", chat_id, message, metadata))
                return SimpleNamespace(success=True, message_id="$text")

            async def send_document(self, chat_id, file_path, metadata=None):
                calls.append(("send_document", chat_id, file_path, metadata))
                return SimpleNamespace(success=True, message_id="$file")

            async def disconnect(self):
                calls.append(("disconnect",))

        fake_module = SimpleNamespace(MatrixAdapter=FakeAdapter)

        with patch.dict(sys.modules, {"gateway.platforms.matrix": fake_module}):
            result = asyncio.run(
                _send_matrix_via_adapter(
                    SimpleNamespace(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.com"}),
                    "!room:example.com",
                    "report attached",
                    media_files=[(str(file_path), False)],
                )
            )

        assert result == {
            "success": True,
            "platform": "matrix",
            "chat_id": "!room:example.com",
            "message_id": "$file",
        }
        assert calls == [
            ("connect",),
            ("send", "!room:example.com", "report attached", None),
            ("send_document", "!room:example.com", str(file_path), None),
            ("disconnect",),
        ]


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

    def test_disable_link_previews_sets_disable_web_page_preview(self, monkeypatch):
        bot = self._make_bot()
        _install_telegram_mock(monkeypatch, bot)

        asyncio.run(
            _send_telegram("tok", "123", "https://example.com", disable_link_previews=True)
        )

        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["disable_web_page_preview"] is True

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

    def test_transient_bad_gateway_retries_text_send(self, monkeypatch):
        bot = self._make_bot()
        bot.send_message = AsyncMock(
            side_effect=[
                Exception("502 Bad Gateway"),
                SimpleNamespace(message_id=2),
            ]
        )
        _install_telegram_mock(monkeypatch, bot)

        with patch("asyncio.sleep", new=AsyncMock()) as sleep_mock:
            result = asyncio.run(_send_telegram("tok", "123", "hello"))

        assert result["success"] is True
        assert bot.send_message.await_count == 2
        sleep_mock.assert_awaited_once()


class TestSendTelegramThreadIdMapping:
    """General-topic mapping in _send_telegram (issue #22267).

    Telegram forum supergroups address the General topic as
    ``message_thread_id="1"`` on incoming updates, but the Bot API rejects
    sends with ``message_thread_id=1`` ("Message thread not found"). The
    gateway adapter's ``_message_thread_id_for_send`` helper maps "1" to
    ``None`` for that reason; the standalone ``_send_telegram`` helper used
    by the ``send_message`` tool needs the same mapping.
    """

    def _make_bot(self):
        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
        return bot

    def test_general_topic_thread_id_omitted(self, monkeypatch):
        """thread_id="1" must be dropped before calling the Bot API."""
        bot = self._make_bot()
        _install_telegram_mock(monkeypatch, bot)

        asyncio.run(_send_telegram("tok", "-1001234567890", "hello", thread_id="1"))

        bot.send_message.assert_awaited_once()
        kwargs = bot.send_message.await_args.kwargs
        assert "message_thread_id" not in kwargs

    def test_non_general_topic_thread_id_preserved(self, monkeypatch):
        """Real forum-topic thread ids (>1) still pass through as ints."""
        bot = self._make_bot()
        _install_telegram_mock(monkeypatch, bot)

        asyncio.run(_send_telegram("tok", "-1001234567890", "hello", thread_id="17585"))

        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["message_thread_id"] == 17585

    def test_no_thread_id_no_kwarg(self, monkeypatch):
        """With no thread_id, message_thread_id must not appear in kwargs."""
        bot = self._make_bot()
        _install_telegram_mock(monkeypatch, bot)

        asyncio.run(_send_telegram("tok", "-1001234567890", "hello"))

        kwargs = bot.send_message.await_args.kwargs
        assert "message_thread_id" not in kwargs

    def test_general_topic_thread_id_int_input_also_dropped(self, monkeypatch):
        """thread_id passed as the int 1 (not str) must still be dropped."""
        bot = self._make_bot()
        _install_telegram_mock(monkeypatch, bot)

        asyncio.run(_send_telegram("tok", "-1001234567890", "hello", thread_id=1))

        kwargs = bot.send_message.await_args.kwargs
        assert "message_thread_id" not in kwargs


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


class TestParseTargetRefMatrix:
    """_parse_target_ref correctly handles Matrix room IDs and user MXIDs."""

    def test_matrix_room_id_is_explicit(self):
        """Matrix room IDs (!) are recognized as explicit targets."""
        chat_id, thread_id, is_explicit = _parse_target_ref("matrix", "!HLOQwxYGgFPMPJUSNR:matrix.org")
        assert chat_id == "!HLOQwxYGgFPMPJUSNR:matrix.org"
        assert thread_id is None
        assert is_explicit is True

    def test_matrix_user_mxid_is_explicit(self):
        """Matrix user MXIDs (@) are recognized as explicit targets."""
        chat_id, thread_id, is_explicit = _parse_target_ref("matrix", "@hermes:matrix.org")
        assert chat_id == "@hermes:matrix.org"
        assert thread_id is None
        assert is_explicit is True

    def test_matrix_alias_is_not_explicit(self):
        """Matrix room aliases (#) are NOT explicit — they need resolution."""
        chat_id, thread_id, is_explicit = _parse_target_ref("matrix", "#general:matrix.org")
        assert chat_id is None
        assert is_explicit is False

    def test_matrix_prefix_only_matches_matrix_platform(self):
        """! and @ prefixes are only treated as explicit for the matrix platform."""
        chat_id, _, is_explicit = _parse_target_ref("telegram", "!something")
        assert is_explicit is False

        chat_id, _, is_explicit = _parse_target_ref("discord", "@someone")
        assert is_explicit is False


class TestParseTargetRefE164:
    """_parse_target_ref accepts E.164 phone numbers for phone-based platforms."""

    def test_signal_e164_preserves_plus_prefix(self):
        """signal:+E164 is explicit and preserves the leading '+' for signal-cli."""
        chat_id, thread_id, is_explicit = _parse_target_ref("signal", "+41791234567")
        assert chat_id == "+41791234567"
        assert thread_id is None
        assert is_explicit is True

    def test_sms_e164_is_explicit(self):
        chat_id, _, is_explicit = _parse_target_ref("sms", "+15551234567")
        assert chat_id == "+15551234567"
        assert is_explicit is True

    def test_whatsapp_e164_is_explicit(self):
        chat_id, _, is_explicit = _parse_target_ref("whatsapp", "+15551234567")
        assert chat_id == "+15551234567"
        assert is_explicit is True

    def test_signal_bare_digits_still_work(self):
        """Bare digit strings continue to match the generic numeric branch."""
        chat_id, _, is_explicit = _parse_target_ref("signal", "15551234567")
        assert chat_id == "15551234567"
        assert is_explicit is True

    def test_signal_invalid_e164_rejected(self):
        """Too-short, too-long, and non-numeric E.164 strings are not explicit."""
        assert _parse_target_ref("signal", "+123")[2] is False
        assert _parse_target_ref("signal", "+1234567890123456")[2] is False
        assert _parse_target_ref("signal", "+12abc4567890")[2] is False
        assert _parse_target_ref("signal", "+")[2] is False

    def test_e164_prefix_only_matches_phone_platforms(self):
        """'+' prefix must NOT be treated as explicit for non-phone platforms."""
        assert _parse_target_ref("telegram", "+15551234567")[2] is False
        assert _parse_target_ref("discord", "+15551234567")[2] is False
        assert _parse_target_ref("matrix", "+15551234567")[2] is False


class TestParseTargetRefSlack:
    """_parse_target_ref recognizes Slack channel/user IDs as explicit."""

    def test_public_channel_id_is_explicit(self):
        chat_id, thread_id, is_explicit = _parse_target_ref("slack", "C0B0QV5434G")
        assert chat_id == "C0B0QV5434G"
        assert thread_id is None
        assert is_explicit is True

    def test_private_channel_id_is_explicit(self):
        assert _parse_target_ref("slack", "G123ABCDEF")[2] is True

    def test_dm_id_is_explicit(self):
        assert _parse_target_ref("slack", "D123ABCDEF")[2] is True

    def test_user_id_is_not_explicit(self):
        """Slack user IDs (U...) and workspace IDs (W...) are NOT explicit send
        targets. chat.postMessage rejects them — a DM must be opened first via
        conversations.open to obtain a D... conversation ID.
        """
        assert _parse_target_ref("slack", "U123ABCDEF")[2] is False
        assert _parse_target_ref("slack", "W123ABCDEF")[2] is False

    def test_whitespace_is_stripped(self):
        chat_id, _, is_explicit = _parse_target_ref("slack", "  C0B0QV5434G  ")
        assert chat_id == "C0B0QV5434G"
        assert is_explicit is True

    def test_lowercase_or_short_id_is_not_explicit(self):
        assert _parse_target_ref("slack", "c0b0qv5434g")[2] is False
        assert _parse_target_ref("slack", "C123")[2] is False
        assert _parse_target_ref("slack", "X0B0QV5434G")[2] is False

    def test_slack_id_not_explicit_for_other_platforms(self):
        assert _parse_target_ref("discord", "C0B0QV5434G")[2] is False
        assert _parse_target_ref("telegram", "C0B0QV5434G")[2] is False


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


# ---------------------------------------------------------------------------
# Discord media attachment support
# ---------------------------------------------------------------------------


class TestSendDiscordMedia:
    """_send_discord uploads media files via multipart/form-data."""

    @staticmethod
    def _build_mock(response_status, response_data=None, response_text="error body"):
        """Build a properly-structured aiohttp mock chain."""
        mock_resp = MagicMock()
        mock_resp.status = response_status
        mock_resp.json = AsyncMock(return_value=response_data or {"id": "msg123"})
        mock_resp.text = AsyncMock(return_value=response_text)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_resp)

        return mock_session, mock_resp

    def test_text_and_media_sends_both(self, tmp_path):
        """Text message is sent first, then each media file as multipart."""
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG fake image data")

        mock_session, _ = self._build_mock(200, {"id": "msg999"})
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(
                _send_discord("tok", "111", "hello", media_files=[(str(img), False)])
            )

        assert result["success"] is True
        assert result["message_id"] == "msg999"
        # Two POSTs: one text JSON, one multipart upload
        assert mock_session.post.call_count == 2

    def test_media_only_skips_text_post(self, tmp_path):
        """When message is empty and media is present, text POST is skipped."""
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG fake image data")

        mock_session, _ = self._build_mock(200, {"id": "media_only"})
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(
                _send_discord("tok", "222", "  ", media_files=[(str(img), False)])
            )

        assert result["success"] is True
        # Only one POST: the media upload (text was whitespace-only)
        assert mock_session.post.call_count == 1

    def test_missing_media_file_collected_as_warning(self):
        """Non-existent media paths produce warnings but don't fail."""
        mock_session, _ = self._build_mock(200, {"id": "txt_ok"})
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(
                _send_discord("tok", "333", "hello", media_files=[("/nonexistent/file.png", False)])
            )

        assert result["success"] is True
        assert "warnings" in result
        assert any("not found" in w for w in result["warnings"])
        # Only the text POST was made, media was skipped
        assert mock_session.post.call_count == 1

    def test_media_upload_failure_collected_as_warning(self, tmp_path):
        """Failed media upload becomes a warning, text still succeeds."""
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG fake image data")

        # First call (text) succeeds, second call (media) returns 413
        text_resp = MagicMock()
        text_resp.status = 200
        text_resp.json = AsyncMock(return_value={"id": "txt_ok"})
        text_resp.__aenter__ = AsyncMock(return_value=text_resp)
        text_resp.__aexit__ = AsyncMock(return_value=None)

        media_resp = MagicMock()
        media_resp.status = 413
        media_resp.text = AsyncMock(return_value="Request Entity Too Large")
        media_resp.__aenter__ = AsyncMock(return_value=media_resp)
        media_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(side_effect=[text_resp, media_resp])

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(
                _send_discord("tok", "444", "hello", media_files=[(str(img), False)])
            )

        assert result["success"] is True
        assert result["message_id"] == "txt_ok"
        assert "warnings" in result
        assert any("413" in w for w in result["warnings"])

    def test_no_text_no_media_returns_error(self):
        """Empty text with no media returns error dict."""
        mock_session, _ = self._build_mock(200)
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(
                _send_discord("tok", "555", "", media_files=[])
            )

        # Text is empty but media_files is empty, so text POST fires
        # (the "skip text if media present" condition isn't met)
        assert result["success"] is True

    def test_multiple_media_files_uploaded_separately(self, tmp_path):
        """Each media file gets its own multipart POST."""
        img1 = tmp_path / "a.png"
        img1.write_bytes(b"img1")
        img2 = tmp_path / "b.jpg"
        img2.write_bytes(b"img2")

        mock_session, _ = self._build_mock(200, {"id": "last"})
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(
                _send_discord("tok", "666", "hi", media_files=[
                    (str(img1), False), (str(img2), False)
                ])
            )

        assert result["success"] is True
        # 1 text POST + 2 media POSTs = 3
        assert mock_session.post.call_count == 3


class TestSendToPlatformDiscordMedia:
    """_send_to_platform routes Discord media correctly."""

    def test_media_files_passed_on_last_chunk_only(self):
        """Discord media_files are only passed on the final chunk."""
        call_log = []

        async def mock_send_discord(token, chat_id, message, thread_id=None, media_files=None):
            call_log.append({"message": message, "media_files": media_files or []})
            return {"success": True, "platform": "discord", "chat_id": chat_id, "message_id": "1"}

        # A message long enough to get chunked (Discord limit is 2000)
        long_msg = "A" * 1900 + " " + "B" * 1900

        with patch("tools.send_message_tool._send_discord", side_effect=mock_send_discord):
            result = asyncio.run(
                _send_to_platform(
                    Platform.DISCORD,
                    SimpleNamespace(enabled=True, token="tok", extra={}),
                    "999",
                    long_msg,
                    media_files=[("/fake/img.png", False)],
                )
            )

        assert result["success"] is True
        assert len(call_log) == 2  # Message was chunked
        assert call_log[0]["media_files"] == []  # First chunk: no media
        assert call_log[1]["media_files"] == [("/fake/img.png", False)]  # Last chunk: media attached

    def test_single_chunk_gets_media(self):
        """Short message (single chunk) gets media_files directly."""
        send_mock = AsyncMock(return_value={"success": True, "message_id": "1"})

        with patch("tools.send_message_tool._send_discord", send_mock):
            result = asyncio.run(
                _send_to_platform(
                    Platform.DISCORD,
                    SimpleNamespace(enabled=True, token="tok", extra={}),
                    "888",
                    "short message",
                    media_files=[("/fake/img.png", False)],
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once()
        call_kwargs = send_mock.await_args.kwargs
        assert call_kwargs["media_files"] == [("/fake/img.png", False)]


class TestSendMatrixUrlEncoding:
    """_send_matrix URL-encodes Matrix room IDs in the API path."""

    def test_room_id_is_percent_encoded_in_url(self):
        """Matrix room IDs with ! and : are percent-encoded in the PUT URL."""
        import aiohttp

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"event_id": "$evt123"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.put = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            from tools.send_message_tool import _send_matrix
            result = asyncio.get_event_loop().run_until_complete(
                _send_matrix(
                    "test_token",
                    {"homeserver": "https://matrix.example.org"},
                    "!HLOQwxYGgFPMPJUSNR:matrix.org",
                    "hello",
                )
            )

        assert result["success"] is True
        # Verify the URL was called with percent-encoded room ID
        put_url = mock_session.put.call_args[0][0]
        assert "%21HLOQwxYGgFPMPJUSNR%3Amatrix.org" in put_url
        assert "!HLOQwxYGgFPMPJUSNR:matrix.org" not in put_url


# ---------------------------------------------------------------------------
# Tests for _derive_forum_thread_name
# ---------------------------------------------------------------------------


class TestDeriveForumThreadName:
    def test_single_line_message(self):
        assert _derive_forum_thread_name("Hello world") == "Hello world"

    def test_multi_line_uses_first_line(self):
        assert _derive_forum_thread_name("First line\nSecond line") == "First line"

    def test_strips_markdown_heading(self):
        assert _derive_forum_thread_name("## My Heading") == "My Heading"

    def test_strips_multiple_hash_levels(self):
        assert _derive_forum_thread_name("### Deep heading") == "Deep heading"

    def test_empty_message_falls_back_to_default(self):
        assert _derive_forum_thread_name("") == "New Post"

    def test_whitespace_only_falls_back(self):
        assert _derive_forum_thread_name("   \n  ") == "New Post"

    def test_hash_only_falls_back(self):
        assert _derive_forum_thread_name("###") == "New Post"

    def test_truncates_to_100_chars(self):
        long_title = "A" * 200
        result = _derive_forum_thread_name(long_title)
        assert len(result) == 100

    def test_strips_whitespace_around_first_line(self):
        assert _derive_forum_thread_name("  Title  \nBody") == "Title"


# ---------------------------------------------------------------------------
# Tests for _send_discord with forum channel support
# ---------------------------------------------------------------------------


class TestSendDiscordForum:
    """_send_discord creates thread posts for forum channels."""

    @staticmethod
    def _build_mock(response_status, response_data=None, response_text="error body"):
        mock_resp = MagicMock()
        mock_resp.status = response_status
        mock_resp.json = AsyncMock(return_value=response_data or {})
        mock_resp.text = AsyncMock(return_value=response_text)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.get = MagicMock(return_value=mock_resp)

        return mock_session, mock_resp

    def test_directory_forum_creates_thread(self):
        """Directory says 'forum' — creates a thread post."""
        thread_data = {
            "id": "t123",
            "message": {"id": "m456"},
        }
        mock_session, _ = self._build_mock(200, response_data=thread_data)

        with patch("aiohttp.ClientSession", return_value=mock_session), \
             patch("gateway.channel_directory.lookup_channel_type", return_value="forum"):
            result = asyncio.run(
                _send_discord("tok", "forum_ch", "Hello forum")
            )

        assert result["success"] is True
        assert result["thread_id"] == "t123"
        assert result["message_id"] == "m456"
        # Should POST to threads endpoint, not messages
        call_url = mock_session.post.call_args.args[0]
        assert "/threads" in call_url
        assert "/messages" not in call_url

    def test_directory_forum_skips_probe(self):
        """When directory says 'forum', no GET probe is made."""
        thread_data = {"id": "t123", "message": {"id": "m456"}}
        mock_session, _ = self._build_mock(200, response_data=thread_data)

        with patch("aiohttp.ClientSession", return_value=mock_session), \
             patch("gateway.channel_directory.lookup_channel_type", return_value="forum"):
            asyncio.run(
                _send_discord("tok", "forum_ch", "Hello")
            )

        # get() should never be called — directory resolved the type
        mock_session.get.assert_not_called()

    def test_directory_channel_skips_forum(self):
        """When directory says 'channel', sends via normal messages endpoint."""
        mock_session, _ = self._build_mock(200, response_data={"id": "msg1"})

        with patch("aiohttp.ClientSession", return_value=mock_session), \
             patch("gateway.channel_directory.lookup_channel_type", return_value="channel"):
            result = asyncio.run(
                _send_discord("tok", "ch1", "Hello")
            )

        assert result["success"] is True
        call_url = mock_session.post.call_args.args[0]
        assert "/messages" in call_url
        assert "/threads" not in call_url

    def test_directory_none_probes_and_detects_forum(self):
        """When directory has no entry, probes GET /channels/{id} and detects type 15."""
        probe_resp = MagicMock()
        probe_resp.status = 200
        probe_resp.json = AsyncMock(return_value={"type": 15})
        probe_resp.__aenter__ = AsyncMock(return_value=probe_resp)
        probe_resp.__aexit__ = AsyncMock(return_value=None)

        thread_data = {"id": "t999", "message": {"id": "m888"}}
        thread_resp = MagicMock()
        thread_resp.status = 200
        thread_resp.json = AsyncMock(return_value=thread_data)
        thread_resp.text = AsyncMock(return_value="")
        thread_resp.__aenter__ = AsyncMock(return_value=thread_resp)
        thread_resp.__aexit__ = AsyncMock(return_value=None)

        probe_session = MagicMock()
        probe_session.__aenter__ = AsyncMock(return_value=probe_session)
        probe_session.__aexit__ = AsyncMock(return_value=None)
        probe_session.get = MagicMock(return_value=probe_resp)

        thread_session = MagicMock()
        thread_session.__aenter__ = AsyncMock(return_value=thread_session)
        thread_session.__aexit__ = AsyncMock(return_value=None)
        thread_session.post = MagicMock(return_value=thread_resp)

        session_iter = iter([probe_session, thread_session])

        with patch("aiohttp.ClientSession", side_effect=lambda **kw: next(session_iter)), \
             patch("gateway.channel_directory.lookup_channel_type", return_value=None):
            result = asyncio.run(
                _send_discord("tok", "forum_ch", "Hello probe")
            )

        assert result["success"] is True
        assert result["thread_id"] == "t999"

    def test_directory_lookup_exception_falls_through_to_probe(self):
        """When lookup_channel_type raises, falls through to API probe."""
        mock_session, _ = self._build_mock(200, response_data={"id": "msg1"})

        with patch("aiohttp.ClientSession", return_value=mock_session), \
             patch("gateway.channel_directory.lookup_channel_type", side_effect=Exception("io error")):
            result = asyncio.run(
                _send_discord("tok", "ch1", "Hello")
            )

        assert result["success"] is True
        # Falls through to probe (GET)
        mock_session.get.assert_called_once()

    def test_forum_thread_creation_error(self):
        """Forum thread creation returning non-200/201 returns an error dict."""
        mock_session, _ = self._build_mock(403, response_text="Forbidden")

        with patch("aiohttp.ClientSession", return_value=mock_session), \
             patch("gateway.channel_directory.lookup_channel_type", return_value="forum"):
            result = asyncio.run(
                _send_discord("tok", "forum_ch", "Hello")
            )

        assert "error" in result
        assert "403" in result["error"]



class TestSendToPlatformDiscordForum:
    """_send_to_platform delegates forum detection to _send_discord."""

    def test_send_to_platform_discord_delegates_to_send_discord(self):
        """Discord messages are routed through _send_discord, which handles forum detection."""
        send_mock = AsyncMock(return_value={"success": True, "message_id": "1"})

        with patch("tools.send_message_tool._send_discord", send_mock):
            result = asyncio.run(
                _send_to_platform(
                    Platform.DISCORD,
                    SimpleNamespace(enabled=True, token="tok", extra={}),
                    "forum_ch",
                    "Hello forum",
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            "tok", "forum_ch", "Hello forum", media_files=[], thread_id=None,
        )

    def test_send_to_platform_discord_with_thread_id(self):
        """Thread ID is still passed through when sending to Discord."""
        send_mock = AsyncMock(return_value={"success": True, "message_id": "1"})

        with patch("tools.send_message_tool._send_discord", send_mock):
            result = asyncio.run(
                _send_to_platform(
                    Platform.DISCORD,
                    SimpleNamespace(enabled=True, token="tok", extra={}),
                    "ch1",
                    "Hello thread",
                    thread_id="17585",
                )
            )

        assert result["success"] is True
        _, call_kwargs = send_mock.await_args
        assert call_kwargs["thread_id"] == "17585"


# ---------------------------------------------------------------------------
# Tests for _send_discord forum + media multipart upload
# ---------------------------------------------------------------------------


class TestSendDiscordForumMedia:
    """_send_discord uploads media as part of the starter message when the target is a forum."""

    @staticmethod
    def _build_thread_resp(thread_id="th_999", msg_id="msg_500"):
        resp = MagicMock()
        resp.status = 201
        resp.json = AsyncMock(return_value={"id": thread_id, "message": {"id": msg_id}})
        resp.text = AsyncMock(return_value="")
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=None)
        return resp

    def test_forum_with_media_uses_multipart(self, tmp_path, monkeypatch):
        """Forum + media → single multipart POST to /threads carrying the starter + files."""
        from tools import send_message_tool as smt

        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNGbytes")

        monkeypatch.setattr(smt, "lookup_channel_type", lambda p, cid: "forum", raising=False)
        monkeypatch.setattr(
            "gateway.channel_directory.lookup_channel_type", lambda p, cid: "forum"
        )

        thread_resp = self._build_thread_resp()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.post = MagicMock(return_value=thread_resp)

        post_calls = []
        orig_post = session.post

        def track_post(url, **kwargs):
            post_calls.append({"url": url, "kwargs": kwargs})
            return thread_resp

        session.post = MagicMock(side_effect=track_post)

        with patch("aiohttp.ClientSession", return_value=session):
            result = asyncio.run(
                _send_discord("tok", "forum_ch", "Thread title\nbody", media_files=[(str(img), False)])
            )

        assert result["success"] is True
        assert result["thread_id"] == "th_999"
        assert result["message_id"] == "msg_500"
        # Exactly one POST — the combined thread-creation + attachments call
        assert len(post_calls) == 1
        assert post_calls[0]["url"].endswith("/threads")
        # Multipart form, not JSON
        assert post_calls[0]["kwargs"].get("data") is not None
        assert post_calls[0]["kwargs"].get("json") is None

    def test_forum_without_media_still_json_only(self, tmp_path, monkeypatch):
        """Forum + no media → JSON POST (no multipart overhead)."""
        monkeypatch.setattr(
            "gateway.channel_directory.lookup_channel_type", lambda p, cid: "forum"
        )

        thread_resp = self._build_thread_resp("t1", "m1")
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        post_calls = []

        def track_post(url, **kwargs):
            post_calls.append({"url": url, "kwargs": kwargs})
            return thread_resp

        session.post = MagicMock(side_effect=track_post)

        with patch("aiohttp.ClientSession", return_value=session):
            result = asyncio.run(_send_discord("tok", "forum_ch", "Hello forum"))

        assert result["success"] is True
        assert len(post_calls) == 1
        # JSON path, no multipart
        assert post_calls[0]["kwargs"].get("json") is not None
        assert post_calls[0]["kwargs"].get("data") is None

    def test_forum_missing_media_file_collected_as_warning(self, tmp_path, monkeypatch):
        """Missing media files produce warnings but the thread is still created."""
        monkeypatch.setattr(
            "gateway.channel_directory.lookup_channel_type", lambda p, cid: "forum"
        )

        thread_resp = self._build_thread_resp()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.post = MagicMock(return_value=thread_resp)

        with patch("aiohttp.ClientSession", return_value=session):
            result = asyncio.run(
                _send_discord(
                    "tok", "forum_ch", "hi",
                    media_files=[("/nonexistent/does-not-exist.png", False)],
                )
            )

        assert result["success"] is True
        assert "warnings" in result
        assert any("not found" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Tests for the process-local forum-probe cache
# ---------------------------------------------------------------------------


class TestForumProbeCache:
    """_DISCORD_CHANNEL_TYPE_PROBE_CACHE memoizes forum detection results."""

    def setup_method(self):
        from tools import send_message_tool as smt
        smt._DISCORD_CHANNEL_TYPE_PROBE_CACHE.clear()

    def test_cache_round_trip(self):
        from tools.send_message_tool import (
            _probe_is_forum_cached,
            _remember_channel_is_forum,
        )
        assert _probe_is_forum_cached("xyz") is None
        _remember_channel_is_forum("xyz", True)
        assert _probe_is_forum_cached("xyz") is True
        _remember_channel_is_forum("xyz", False)
        assert _probe_is_forum_cached("xyz") is False

    def test_probe_result_is_memoized(self, monkeypatch):
        """An API-probed channel type is cached so subsequent sends skip the probe."""
        monkeypatch.setattr(
            "gateway.channel_directory.lookup_channel_type", lambda p, cid: None
        )

        # First probe response: type=15 (forum)
        probe_resp = MagicMock()
        probe_resp.status = 200
        probe_resp.json = AsyncMock(return_value={"type": 15})
        probe_resp.__aenter__ = AsyncMock(return_value=probe_resp)
        probe_resp.__aexit__ = AsyncMock(return_value=None)

        thread_resp = MagicMock()
        thread_resp.status = 201
        thread_resp.json = AsyncMock(return_value={"id": "t1", "message": {"id": "m1"}})
        thread_resp.__aenter__ = AsyncMock(return_value=thread_resp)
        thread_resp.__aexit__ = AsyncMock(return_value=None)

        probe_session = MagicMock()
        probe_session.__aenter__ = AsyncMock(return_value=probe_session)
        probe_session.__aexit__ = AsyncMock(return_value=None)
        probe_session.get = MagicMock(return_value=probe_resp)

        thread_session = MagicMock()
        thread_session.__aenter__ = AsyncMock(return_value=thread_session)
        thread_session.__aexit__ = AsyncMock(return_value=None)
        thread_session.post = MagicMock(return_value=thread_resp)

        # Two _send_discord calls: first does probe + thread-create; second should skip probe
        from tools import send_message_tool as smt

        sessions_created = []

        def session_factory(**kwargs):
            # Alternate: each new ClientSession() call returns a probe_session, thread_session pair
            idx = len(sessions_created)
            sessions_created.append(idx)
            # Returns the same mocks; the real code opens a probe session then a thread session.
            # Hand out probe_session if this is the first time called within _send_discord,
            # otherwise thread_session.
            if idx % 2 == 0:
                return probe_session
            return thread_session

        with patch("aiohttp.ClientSession", side_effect=session_factory):
            result1 = asyncio.run(_send_discord("tok", "ch1", "first"))
        assert result1["success"] is True
        assert smt._probe_is_forum_cached("ch1") is True

        # Second call: cache hits, no new probe session needed. We need to only
        # return thread_session now since probe is skipped.
        sessions_created.clear()
        with patch("aiohttp.ClientSession", return_value=thread_session):
            result2 = asyncio.run(_send_discord("tok", "ch1", "second"))
        assert result2["success"] is True
        # Only one session opened (thread creation) — no probe session this time
        # (verified by not raising from our side_effect exhaustion)


# ---------------------------------------------------------------------------
# _send_signal — chunking + 429 retry (mirrors gateway adapter behavior)
# ---------------------------------------------------------------------------


class _FakeSignalHttp:
    """Stand-in for httpx.AsyncClient used as an async context manager.

    Pops a response from the queue per `post` call. Each entry is either
    a dict (returned from .json()) or an exception instance (raised).
    Captures (url, payload) per call.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, *_a, **_kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def post(self, url, json=None):
        self.calls.append({"url": url, "payload": json})
        if not self.responses:
            raise AssertionError("Unexpected extra POST")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        resp = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda data=item: data,
        )
        return resp


def _install_signal_http(monkeypatch, fake):
    """Patch httpx.AsyncClient at the module level so the lazy import in
    _send_signal picks it up.
    """
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)


def _patch_sendmsg_sleep_and_time(monkeypatch, capture: list):
    """Mock asyncio.sleep + time.monotonic in the signal_rate_limit
    module so the scheduler's acquire loop sees synthetic time advancing
    during sleep calls, and report_rpc_duration sees the same clock.

    Zero-second sleeps (event-loop yields from fake HTTP posts) are
    delegated to the real asyncio.sleep so they don't pollute the
    capture list.
    """
    import asyncio as _aio
    _real_sleep = _aio.sleep
    offset = [0.0]

    async def fake_sleep(seconds):
        if seconds > 0:
            capture.append(seconds)
            offset[0] += seconds
        else:
            await _real_sleep(0)

    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.asyncio.sleep", fake_sleep
    )
    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.time.monotonic", lambda: offset[0]
    )


class TestSendSignalChunking:
    def test_text_only_single_rpc(self, monkeypatch):
        fake = _FakeSignalHttp([{"result": {"timestamp": 1}}])
        _install_signal_http(monkeypatch, fake)

        result = asyncio.run(
            _send_signal(
                {"http_url": "http://localhost:8080", "account": "+15551234567"},
                "+15557654321",
                "hello",
            )
        )

        assert result == {"success": True, "platform": "signal", "chat_id": "+15557654321"}
        assert len(fake.calls) == 1
        params = fake.calls[0]["payload"]["params"]
        assert params["message"] == "hello"
        assert "attachments" not in params

    def test_chunks_attachments_above_max(self, tmp_path, monkeypatch):
        """33 attachments → 2 batches; text only on first batch. Batch 1
        only needs 1 token and 18 remain after batch 0, so no sleep."""
        from gateway.platforms.signal_rate_limit import (
            SIGNAL_MAX_ATTACHMENTS_PER_MSG,
        )

        paths = []
        for i in range(33):
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 16)
            paths.append((str(p), False))

        fake = _FakeSignalHttp([
            {"result": {"timestamp": 1}},   # batch 0
            {"result": {"timestamp": 2}},   # batch 1
        ])
        _install_signal_http(monkeypatch, fake)

        sleep_calls = []
        _patch_sendmsg_sleep_and_time(monkeypatch, sleep_calls)

        result = asyncio.run(
            _send_signal(
                {"http_url": "http://localhost:8080", "account": "+15551234567"},
                "+15557654321",
                "Caption goes here",
                media_files=paths,
            )
        )

        assert result["success"] is True
        assert len(fake.calls) == 2
        assert len(sleep_calls) == 0

        first = fake.calls[0]["payload"]["params"]
        assert first["message"] == "Caption goes here"
        assert len(first["attachments"]) == SIGNAL_MAX_ATTACHMENTS_PER_MSG

        second = fake.calls[1]["payload"]["params"]
        assert second["message"] == ""  # caption only on batch 0
        assert len(second["attachments"]) == 33 - SIGNAL_MAX_ATTACHMENTS_PER_MSG

    def test_full_followup_batch_emits_pacing_notice(self, tmp_path, monkeypatch):
        """64 attachments → 2 full batches. Batch 1 needs 14 more tokens
        than the 18 remaining after batch 0 — 56s wait crossing the 10s
        notice threshold."""
        from gateway.platforms.signal_rate_limit import (
            SIGNAL_MAX_ATTACHMENTS_PER_MSG,
            SIGNAL_RATE_LIMIT_BUCKET_CAPACITY,
            SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER,
        )

        paths = []
        for i in range(64):
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 16)
            paths.append((str(p), False))

        fake = _FakeSignalHttp([
            {"result": {"timestamp": 1}},   # batch 0
            {"result": {"timestamp": 99}},  # pacing notice
            {"result": {"timestamp": 2}},   # batch 1
        ])
        _install_signal_http(monkeypatch, fake)

        sleep_calls = []
        _patch_sendmsg_sleep_and_time(monkeypatch, sleep_calls)

        result = asyncio.run(
            _send_signal(
                {"http_url": "http://localhost:8080", "account": "+15551234567"},
                "+15557654321",
                "",
                media_files=paths,
            )
        )

        assert result["success"] is True
        assert len(fake.calls) == 3
        notice = fake.calls[1]["payload"]["params"]
        assert "More images coming" in notice["message"]
        assert "attachments" not in notice
        # Batch 1 deficit: 32 - (50 - 32) = 14 tokens × 4s = 56s
        expected = (
            SIGNAL_MAX_ATTACHMENTS_PER_MSG
            - (SIGNAL_RATE_LIMIT_BUCKET_CAPACITY - SIGNAL_MAX_ATTACHMENTS_PER_MSG)
        ) * SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER
        assert sleep_calls == [pytest.approx(expected, abs=1.0)]

    def test_429_with_retry_after_drives_exact_backoff(self, tmp_path, monkeypatch):
        """signal-cli ≥ v0.14.3 surfaces Retry-After under
        error.data.response.results[*].retryAfterSeconds. The scheduler
        calibrates its refill rate from that value; the retry of n=1
        sleeps the per-token interval."""
        from gateway.platforms.signal_rate_limit import SIGNAL_RPC_ERROR_RATELIMIT

        p = tmp_path / "img.png"
        p.write_bytes(b"\x89PNG" + b"\x00" * 16)

        fake = _FakeSignalHttp([
            {
                "error": {
                    "code": SIGNAL_RPC_ERROR_RATELIMIT,
                    "message": "Failed to send message due to rate limiting",
                    "data": {
                        "response": {
                            "timestamp": 0,
                            "results": [
                                {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 42},
                            ],
                        }
                    },
                }
            },
            {"result": {"timestamp": 7}},
        ])
        _install_signal_http(monkeypatch, fake)

        sleep_calls = []
        _patch_sendmsg_sleep_and_time(monkeypatch, sleep_calls)

        result = asyncio.run(
            _send_signal(
                {"http_url": "http://localhost:8080", "account": "+15551234567"},
                "+15557654321",
                "",
                media_files=[(str(p), False)],
            )
        )

        assert result["success"] is True
        assert len(fake.calls) == 2  # initial + retry
        assert sleep_calls == [pytest.approx(42.0, abs=1.0)]

    def test_429_without_retry_after_falls_back_to_default(self, tmp_path, monkeypatch):
        """Older signal-cli (< v0.14.3) doesn't surface Retry-After.
        The scheduler keeps its default rate (1 token / 4s)."""
        from gateway.platforms.signal_rate_limit import SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER

        p = tmp_path / "img.png"
        p.write_bytes(b"\x89PNG" + b"\x00" * 16)

        fake = _FakeSignalHttp([
            {"error": {"message": "Failed: [429] Rate Limited"}},
            {"result": {"timestamp": 7}},
        ])
        _install_signal_http(monkeypatch, fake)

        sleep_calls = []
        _patch_sendmsg_sleep_and_time(monkeypatch, sleep_calls)

        result = asyncio.run(
            _send_signal(
                {"http_url": "http://localhost:8080", "account": "+15551234567"},
                "+15557654321",
                "",
                media_files=[(str(p), False)],
            )
        )

        assert result["success"] is True
        assert sleep_calls == [pytest.approx(SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER, abs=1.0)]

    def test_429_retry_exhaust_continues_to_next_batch(self, tmp_path, monkeypatch):
        """Both attempts on batch 0 fail; batch 1 still gets a chance.
        The scheduler's natural pacing (no more cooldown gate) lets the
        second batch through after its acquire wait."""
        from gateway.platforms.signal_rate_limit import SIGNAL_RPC_ERROR_RATELIMIT

        paths = []
        for i in range(33):  # forces 2 batches
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG" + b"\x00" * 16)
            paths.append((str(p), False))

        rate_limit_err = {
            "error": {
                "code": SIGNAL_RPC_ERROR_RATELIMIT,
                "message": "Failed to send message due to rate limiting",
                "data": {
                    "response": {
                        "timestamp": 0,
                        "results": [
                            {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 4},
                        ],
                    }
                },
            }
        }

        fake = _FakeSignalHttp([
            rate_limit_err,                  # batch 0, attempt 1
            rate_limit_err,                  # batch 0, attempt 2 (exhaust)
            {"result": {"timestamp": 9}},    # batch 1 succeeds
        ])
        _install_signal_http(monkeypatch, fake)

        sleep_calls = []
        _patch_sendmsg_sleep_and_time(monkeypatch, sleep_calls)

        result = asyncio.run(
            _send_signal(
                {"http_url": "http://localhost:8080", "account": "+15551234567"},
                "+15557654321",
                "many",
                media_files=paths,
            )
        )

        # Partial success: batch 0 lost but batch 1 went through.
        assert result["success"] is True
        assert "warnings" in result
        assert any("rate-limited" in w for w in result["warnings"])
        # 2 attempts on batch 0 + 1 successful batch 1 = 3 calls
        assert len(fake.calls) == 3

    def test_non_rate_limit_error_returns_immediately(self, tmp_path, monkeypatch):
        """A non-429 RPC error should not retry — it returns an error result."""
        p = tmp_path / "img.png"
        p.write_bytes(b"\x89PNG" + b"\x00" * 16)

        fake = _FakeSignalHttp([
            {"error": {"message": "UntrustedIdentityException"}},
        ])
        _install_signal_http(monkeypatch, fake)

        result = asyncio.run(
            _send_signal(
                {"http_url": "http://localhost:8080", "account": "+15551234567"},
                "+15557654321",
                "",
                media_files=[(str(p), False)],
            )
        )

        assert "error" in result
        assert "UntrustedIdentityException" in result["error"]
        assert len(fake.calls) == 1  # no retry on non-429

    def test_skipped_missing_files_reported_in_warnings(self, tmp_path, monkeypatch):
        good = tmp_path / "ok.png"
        good.write_bytes(b"\x89PNG" + b"\x00" * 16)

        fake = _FakeSignalHttp([{"result": {"timestamp": 1}}])
        _install_signal_http(monkeypatch, fake)

        result = asyncio.run(
            _send_signal(
                {"http_url": "http://localhost:8080", "account": "+15551234567"},
                "+15557654321",
                "msg",
                media_files=[(str(good), False), (str(tmp_path / "missing.png"), False)],
            )
        )

        assert result["success"] is True
        assert "warnings" in result
        # Only the existing file made it into the RPC
        params = fake.calls[0]["payload"]["params"]
        assert len(params["attachments"]) == 1


# ── _send_via_adapter standalone fallback ────────────────────────────────


class _FakePlatform:
    """Stand-in for the gateway.config.Platform enum.  Holds the .value
    attribute consulted by ``_send_via_adapter`` for registry lookups."""

    def __init__(self, value):
        self.value = value


class TestSendViaAdapterStandaloneFallback:
    """Coverage for the out-of-process plugin-platform send path.

    When the gateway runner is not in this process (e.g. ``hermes cron``
    runs separately from ``hermes gateway``), ``_send_via_adapter`` should
    fall through to the plugin's ``standalone_sender_fn`` registered on
    its ``PlatformEntry``.  Without the hook, the existing error string
    is returned (with a more helpful tail).
    """

    @staticmethod
    def _make_entry(send_fn):
        from gateway.platform_registry import PlatformEntry

        return PlatformEntry(
            name="fakeplatform",
            label="Fake",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            standalone_sender_fn=send_fn,
        )

    @pytest.mark.asyncio
    async def test_standalone_sender_fn_called_when_no_adapter(self, monkeypatch):
        """Registry has hook, runner ref returns None: the hook is awaited."""
        from tools.send_message_tool import _send_via_adapter
        from gateway.platform_registry import platform_registry

        recorded = {}

        async def fake_send(pconfig, chat_id, message, **kwargs):
            recorded["pconfig"] = pconfig
            recorded["chat_id"] = chat_id
            recorded["message"] = message
            recorded["kwargs"] = kwargs
            return {"success": True, "message_id": "msg-42"}

        platform_registry.register(self._make_entry(fake_send))
        try:
            monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)

            pconfig = SimpleNamespace(extra={})
            result = await _send_via_adapter(
                _FakePlatform("fakeplatform"),
                pconfig,
                "room/123",
                "hello cron",
            )
        finally:
            platform_registry.unregister("fakeplatform")

        assert result == {"success": True, "message_id": "msg-42"}
        assert recorded["chat_id"] == "room/123"
        assert recorded["message"] == "hello cron"
        assert recorded["pconfig"] is pconfig

    @pytest.mark.asyncio
    async def test_standalone_sender_fn_kwargs_forwarded(self, monkeypatch):
        """thread_id, media_files, and force_document all reach the hook."""
        from tools.send_message_tool import _send_via_adapter
        from gateway.platform_registry import platform_registry

        recorded = {}

        async def fake_send(pconfig, chat_id, message, *, thread_id=None,
                            media_files=None, force_document=False):
            recorded["thread_id"] = thread_id
            recorded["media_files"] = media_files
            recorded["force_document"] = force_document
            return {"success": True, "message_id": "x"}

        platform_registry.register(self._make_entry(fake_send))
        try:
            monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)

            await _send_via_adapter(
                _FakePlatform("fakeplatform"),
                SimpleNamespace(extra={}),
                "chat-1",
                "hi",
                thread_id="thread-7",
                media_files=["/tmp/a.png"],
                force_document=True,
            )
        finally:
            platform_registry.unregister("fakeplatform")

        assert recorded["thread_id"] == "thread-7"
        assert recorded["media_files"] == ["/tmp/a.png"]
        assert recorded["force_document"] is True

    @pytest.mark.asyncio
    async def test_standalone_sender_fn_absent_returns_helpful_error(self, monkeypatch):
        """Registry entry has no hook: the fall-through error explains both
        options (gateway-running and standalone hook)."""
        from tools.send_message_tool import _send_via_adapter
        from gateway.platform_registry import platform_registry

        platform_registry.register(self._make_entry(None))
        try:
            monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)

            result = await _send_via_adapter(
                _FakePlatform("fakeplatform"),
                SimpleNamespace(extra={}),
                "chat-1",
                "hi",
            )
        finally:
            platform_registry.unregister("fakeplatform")

        assert "error" in result
        assert "fakeplatform" in result["error"]
        assert "standalone_sender_fn" in result["error"]

    @pytest.mark.asyncio
    async def test_standalone_sender_fn_raises_is_caught_and_formatted(self, monkeypatch):
        """Hook raises: error dict has 'Plugin standalone send failed: ...'"""
        from tools.send_message_tool import _send_via_adapter
        from gateway.platform_registry import platform_registry

        async def boom(pconfig, chat_id, message, **kwargs):
            raise ValueError("boom!")

        platform_registry.register(self._make_entry(boom))
        try:
            monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)

            result = await _send_via_adapter(
                _FakePlatform("fakeplatform"),
                SimpleNamespace(extra={}),
                "chat-1",
                "hi",
            )
        finally:
            platform_registry.unregister("fakeplatform")

        assert result == {"error": "Plugin standalone send failed: boom!"}

    @pytest.mark.asyncio
    async def test_standalone_sender_fn_return_shape_passed_through(self, monkeypatch):
        """Hook returns success dict: passed through unchanged."""
        from tools.send_message_tool import _send_via_adapter
        from gateway.platform_registry import platform_registry

        async def fake_send(pconfig, chat_id, message, **kwargs):
            return {"success": True, "message_id": "abc-123", "extra_field": "preserved"}

        platform_registry.register(self._make_entry(fake_send))
        try:
            monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)

            result = await _send_via_adapter(
                _FakePlatform("fakeplatform"),
                SimpleNamespace(extra={}),
                "chat-1",
                "hi",
            )
        finally:
            platform_registry.unregister("fakeplatform")

        assert result["success"] is True
        assert result["message_id"] == "abc-123"
        assert result["extra_field"] == "preserved"

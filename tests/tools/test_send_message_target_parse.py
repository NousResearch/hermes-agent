"""Parser-only and lightweight routing tests for send_message targets.

These stay separate from ``test_send_message_tool.py`` because that module
skips wholesale when optional Telegram dependencies are not installed.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.send_message_tool import _parse_target_ref, _send_to_platform, send_message_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def test_buzz_uuid_target_is_explicit() -> None:
    channel_id = "31b543d5-80d4-4df5-8a5c-cefca1a58fdd"

    assert _parse_target_ref("buzz", channel_id) == (channel_id, None, True)


def test_live_buzz_media_delivers_every_file_with_reply_metadata(tmp_path) -> None:
    from gateway.platforms.base import SendResult

    platform = Platform("buzz")
    first = tmp_path / "first.txt"
    second = tmp_path / "second.pdf"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    calls = []

    class Adapter:
        async def send(self, *, chat_id, content, metadata=None):
            calls.append(("text", content, metadata))
            return SendResult(success=True, message_id="evt-text")

        async def send_document(self, chat_id, file_path, **kwargs):
            calls.append(("document", file_path, kwargs))
            return SendResult(success=True, message_id=f"evt-{len(calls)}")

    runner = SimpleNamespace(adapters={platform: Adapter()})
    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = asyncio.run(
            _send_to_platform(
                platform,
                SimpleNamespace(enabled=True, token=None, extra={}),
                "31b543d5-80d4-4df5-8a5c-cefca1a58fdd",
                "attached files",
                thread_id="reply-root",
                media_files=[(str(first), False), (str(second), False)],
            )
        )

    assert result["success"] is True
    assert result["media_delivered"] is True
    assert calls == [
        ("text", "attached files", {"thread_id": "reply-root"}),
        ("document", str(first), {"caption": None, "reply_to": "reply-root", "metadata": {"thread_id": "reply-root"}}),
        ("document", str(second), {"caption": None, "reply_to": "reply-root", "metadata": {"thread_id": "reply-root"}}),
    ]


def test_live_buzz_single_image_uses_caption_without_duplicate_text(tmp_path) -> None:
    from gateway.platforms.base import SendResult

    platform = Platform("buzz")
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    calls = []

    class Adapter:
        async def send(self, **kwargs):
            calls.append(("text", kwargs))
            return SendResult(success=True, message_id="unexpected")

        async def send_image_file(self, chat_id, image_path, **kwargs):
            calls.append(("image", image_path, kwargs))
            return SendResult(success=True, message_id="evt-image")

    runner = SimpleNamespace(adapters={platform: Adapter()})
    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = asyncio.run(
            _send_to_platform(
                platform,
                SimpleNamespace(enabled=True, token=None, extra={}),
                "31b543d5-80d4-4df5-8a5c-cefca1a58fdd",
                "screenshot caption",
                thread_id="reply-root",
                media_files=[(str(image), False)],
            )
        )

    assert result == {
        "success": True,
        "message_id": "evt-image",
        "media_delivered": True,
    }
    assert calls == [
        ("image", str(image), {"caption": "screenshot caption", "reply_to": "reply-root", "metadata": {"thread_id": "reply-root"}})
    ]


def test_live_buzz_media_failure_is_explicit_not_omitted(tmp_path) -> None:
    from gateway.platforms.base import SendResult

    platform = Platform("buzz")
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    media_calls = []

    class Adapter:
        async def send(self, **kwargs):
            return SendResult(success=True, message_id="evt-text")

        async def send_document(self, chat_id, file_path, **kwargs):
            media_calls.append(file_path)
            if file_path == str(second):
                return SendResult(success=False, error="relay rejected upload")
            return SendResult(success=True, message_id="evt-first")

    runner = SimpleNamespace(adapters={platform: Adapter()})
    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = asyncio.run(
            _send_to_platform(
                platform,
                SimpleNamespace(enabled=True, token=None, extra={}),
                "31b543d5-80d4-4df5-8a5c-cefca1a58fdd",
                "attached files",
                media_files=[(str(first), False), (str(second), False)],
            )
        )

    assert media_calls == [str(first), str(second)]
    assert "relay rejected upload" in result["error"]
    assert "media_delivered" not in result


def test_live_buzz_media_only_send_reaches_adapter(tmp_path) -> None:
    from gateway.platforms.base import SendResult

    platform = Platform("buzz")
    document = tmp_path / "report.txt"
    document.write_text("report", encoding="utf-8")
    media_calls = []

    class Adapter:
        async def send(self, **kwargs):
            raise AssertionError("media-only send must not emit an empty text message")

        async def send_document(self, chat_id, file_path, **kwargs):
            media_calls.append((chat_id, file_path, kwargs))
            return SendResult(success=True, message_id="evt-document")

    runner = SimpleNamespace(adapters={platform: Adapter()})
    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = asyncio.run(
            _send_to_platform(
                platform,
                SimpleNamespace(enabled=True, token=None, extra={}),
                "31b543d5-80d4-4df5-8a5c-cefca1a58fdd",
                "",
                media_files=[(str(document), False)],
            )
        )

    assert result == {
        "success": True,
        "message_id": "evt-document",
        "media_delivered": True,
    }
    assert len(media_calls) == 1
    assert media_calls[0][1] == str(document)


def test_live_buzz_media_exception_reports_partial_delivery(tmp_path) -> None:
    from gateway.platforms.base import SendResult

    platform = Platform("buzz")
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    class Adapter:
        async def send(self, **kwargs):
            return SendResult(success=True, message_id="evt-text")

        async def send_document(self, chat_id, file_path, **kwargs):
            if file_path == str(second):
                raise RuntimeError("relay transport failed")
            return SendResult(success=True, message_id="evt-first")

    runner = SimpleNamespace(adapters={platform: Adapter()})
    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = asyncio.run(
            _send_to_platform(
                platform,
                SimpleNamespace(enabled=True, token=None, extra={}),
                "31b543d5-80d4-4df5-8a5c-cefca1a58fdd",
                "attached files",
                media_files=[(str(first), False), (str(second), False)],
            )
        )

    assert "after 1/2 files" in result["error"]
    assert "relay transport failed" in result["error"]
    assert "media_delivered" not in result


def test_live_adapter_inherited_media_fallback_is_not_claimed_as_delivery(tmp_path) -> None:
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    platform = Platform("buzz")
    document = tmp_path / "report.txt"
    document.write_text("report", encoding="utf-8")

    class Adapter:
        name = "Fallback-only"
        send_document = BasePlatformAdapter.send_document

        async def send(self, **kwargs):
            return SendResult(success=True, message_id="warning-text-only")

    runner = SimpleNamespace(adapters={platform: Adapter()})
    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = asyncio.run(
            _send_to_platform(
                platform,
                SimpleNamespace(enabled=True, token=None, extra={}),
                "31b543d5-80d4-4df5-8a5c-cefca1a58fdd",
                "attached",
                media_files=[(str(document), False)],
            )
        )

    assert "does not implement native document delivery" in result["error"]
    assert "media_delivered" not in result


def test_live_buzz_adapter_exception_is_bounded() -> None:
    platform = Platform("buzz")

    class Adapter:
        async def send(self, **kwargs):
            raise RuntimeError("x" * 100_000)

    runner = SimpleNamespace(adapters={platform: Adapter()})
    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = asyncio.run(
            _send_to_platform(
                platform,
                SimpleNamespace(enabled=True, token=None, extra={}),
                "31b543d5-80d4-4df5-8a5c-cefca1a58fdd",
                "hello",
            )
        )

    assert result["error"].startswith("Plugin platform send failed: ")
    assert len(result["error"]) <= 1024


def test_send_message_routes_buzz_uuid_without_home_fallback() -> None:
    buzz_platform = Platform("buzz")
    buzz_cfg = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={buzz_platform: buzz_cfg},
        get_home_channel=lambda _platform: SimpleNamespace(chat_id="home-channel"),
    )
    channel_id = "31b543d5-80d4-4df5-8a5c-cefca1a58fdd"

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", side_effect=AssertionError("raw UUID should not resolve via directory")), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": f"buzz:{channel_id}",
                    "message": "hello group",
                }
            )
        )

    assert result["success"] is True
    assert "note" not in result
    send_mock.assert_awaited_once_with(
        buzz_platform,
        buzz_cfg,
        channel_id,
        "hello group",
        thread_id=None,
        media_files=[],
        force_document=False,
    )


def test_photon_e164_target_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref("photon", "+15551234567")

    assert chat_id == "+15551234567"
    assert thread_id is None
    assert is_explicit is True


def test_e164_target_still_requires_phone_platform() -> None:
    assert _parse_target_ref("matrix", "+15551234567")[2] is False


def test_send_message_routes_whatsapp_group_jid_without_home_fallback() -> None:
    whatsapp_cfg = SimpleNamespace(enabled=True, token=None, extra={"api_url": "http://bridge"})
    config = SimpleNamespace(
        platforms={Platform.WHATSAPP: whatsapp_cfg},
        get_home_channel=lambda _platform: SimpleNamespace(chat_id="15551234567@s.whatsapp.net"),
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", side_effect=AssertionError("raw JID should not resolve via directory")), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "whatsapp:120363408391911677@g.us",
                    "message": "hello group",
                }
            )
        )

    assert result["success"] is True
    assert "note" not in result
    send_mock.assert_awaited_once_with(
        Platform.WHATSAPP,
        whatsapp_cfg,
        "120363408391911677@g.us",
        "hello group",
        thread_id=None,
        media_files=[],
        force_document=False,
    )


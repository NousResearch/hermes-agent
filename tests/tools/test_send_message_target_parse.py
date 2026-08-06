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


def test_buzz_uuid_target_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "buzz", "e74ae728-e68a-4f8b-b285-3f19ad2df1f8"
    )

    assert chat_id == "e74ae728-e68a-4f8b-b285-3f19ad2df1f8"
    assert thread_id is None
    assert is_explicit is True


def test_send_message_routes_buzz_uuid_without_home_fallback() -> None:
    buzz = Platform("buzz")
    buzz_cfg = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={buzz: buzz_cfg},
        get_home_channel=lambda _platform: SimpleNamespace(
            chat_id="93b1a8c9-7093-4117-9db1-7199bacabd48"
        ),
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", side_effect=AssertionError("raw Buzz UUID should not resolve via directory")), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "buzz:e74ae728-e68a-4f8b-b285-3f19ad2df1f8",
                    "message": "exact DM",
                }
            )
        )

    assert result["success"] is True
    assert "note" not in result
    send_mock.assert_awaited_once_with(
        buzz,
        buzz_cfg,
        "e74ae728-e68a-4f8b-b285-3f19ad2df1f8",
        "exact DM",
        thread_id=None,
        media_files=[],
        force_document=False,
    )


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


def test_resolved_opaque_plugin_target_uses_directory_id() -> None:
    from gateway.platform_registry import PlatformEntry, platform_registry

    platform_name = "opaque-resolved-test"
    entry = PlatformEntry(
        name=platform_name,
        label="Opaque resolved test",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
    )
    platform_registry.register(entry)
    platform = Platform(platform_name)
    pconfig = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={platform: pconfig},
        get_home_channel=lambda _platform: None,
    )
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch(
                 "gateway.channel_directory.resolve_channel_name",
                 return_value="opaque:directory-id",
             ), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch(
                 "tools.send_message_tool._send_to_platform",
                 new=AsyncMock(return_value={"success": True}),
             ) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": f"{platform_name}:Friendly name",
                        "message": "hello",
                    }
                )
            )
    finally:
        platform_registry.unregister(platform_name)

    assert result["success"] is True
    send_mock.assert_awaited_once_with(
        platform,
        pconfig,
        "opaque:directory-id",
        "hello",
        thread_id=None,
        media_files=[],
        force_document=False,
    )


def test_unresolved_plugin_target_requires_explicit_parser() -> None:
    from gateway.platform_registry import PlatformEntry, platform_registry

    platform_name = "opaque-verbatim-test"
    entry = PlatformEntry(
        name=platform_name,
        label="Opaque verbatim test",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
    )
    platform_registry.register(entry)
    platform = Platform(platform_name)
    # Simulate a fresh `hermes send` process: the dynamic Platform member
    # is known from config, but plugin discovery has not registered its
    # adapter entry yet.
    platform_registry.unregister(platform_name)
    pconfig = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={platform: pconfig},
        get_home_channel=lambda _platform: None,
    )
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch(
                 "gateway.channel_directory.resolve_channel_name",
                 return_value=None,
             ), \
             patch(
                 "hermes_cli.plugins.discover_plugins",
                 side_effect=lambda: platform_registry.register(entry),
             ) as discover_mock, \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch(
                 "tools.send_message_tool._send_to_platform",
                 new=AsyncMock(return_value={"success": True}),
             ) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": f"{platform_name}:dm:panyaozhen",
                        "message": "hello",
                    }
                )
            )
    finally:
        platform_registry.unregister(platform_name)

    assert result == {
        "error": f"Could not resolve 'dm:panyaozhen' on {platform_name}. "
        "The plugin parser did not recognize it and no channel-directory entry matched."
    }
    discover_mock.assert_called_once_with()
    send_mock.assert_not_awaited()


def test_unresolved_builtin_target_keeps_directory_error() -> None:
    telegram_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    config = SimpleNamespace(
        platforms={Platform.TELEGRAM: telegram_cfg},
        get_home_channel=lambda _platform: None,
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("gateway.channel_directory.resolve_channel_name", return_value=None), \
         patch(
             "tools.send_message_tool._send_to_platform",
             new=AsyncMock(return_value={"success": True}),
         ) as send_mock:
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "telegram:missing-room",
                    "message": "hello",
                }
            )
        )

    assert result == {
        "error": "Could not resolve 'missing-room' on telegram. "
        "Use send_message(action='list') to see available targets."
    }
    send_mock.assert_not_awaited()



def test_unresolved_builtin_target_passes_through_when_requested() -> None:
    """Cron and react keep the old pass-through behavior for unresolved
    built-in targets: with no model in the loop to react to an error, the
    raw id must reach the adapter, as it did before resolve_send_target
    took over these callers."""
    from tools.send_message_tool import resolve_send_target

    with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
        chat_id, thread_id, error = resolve_send_target(
            "telegram", "ops-room", pass_unresolved_references=True
        )

    assert error is None
    assert chat_id == "ops-room"
    assert thread_id is None


def test_unresolved_builtin_target_still_errors_for_the_model_tool() -> None:
    """The model-facing default stays strict: unresolved targets error with a hint."""
    from tools.send_message_tool import resolve_send_target

    with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
        chat_id, _thread_id, error = resolve_send_target("telegram", "ops-room")

    assert chat_id is None
    assert error is not None


def test_photon_group_guid_passes_through_when_requested() -> None:
    """The reported regression case: a photon group GUID matches no parser
    pattern (only DM GUIDs have an explicit rule) and no directory entry.
    Photon registers as a parser-less plugin platform, so the pass-through
    applies once platforms are prepared."""
    from tools.send_message_tool import (
        prepare_send_message_platforms,
        resolve_send_target,
    )

    prepare_send_message_platforms()
    with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
        chat_id, thread_id, error = resolve_send_target(
            "photon", "iMessage;+;chat527148912345", pass_unresolved_references=True
        )

    assert error is None
    assert chat_id == "iMessage;+;chat527148912345"
    assert thread_id is None


def test_parserless_plugin_target_passes_through_when_requested() -> None:
    """A plugin platform that declares no parser has no explicit syntax at
    all, so passing the raw id through is the only way cron can target it."""
    from gateway.platform_registry import PlatformEntry, platform_registry
    from tools.send_message_tool import resolve_send_target

    platform_name = "opaque-cron-fallback-test"
    entry = PlatformEntry(
        name=platform_name,
        label="Opaque cron fallback test",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
    )
    platform_registry.register(entry)
    try:
        with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
            chat_id, thread_id, error = resolve_send_target(
                platform_name, "dm:panyaozhen", pass_unresolved_references=True
            )
    finally:
        platform_registry.unregister(platform_name)

    assert error is None
    assert chat_id == "dm:panyaozhen"
    assert thread_id is None


def test_plugin_parser_stays_authoritative_despite_fallback() -> None:
    """A plugin that DOES declare a parser stays strict for every caller:
    its parser is the authority on native syntax, so an unrecognized
    target errors even with pass_unresolved_references."""
    from gateway.platform_registry import PlatformEntry, platform_registry
    from tools.send_message_tool import resolve_send_target

    platform_name = "opaque-parser-strict-test"
    entry = PlatformEntry(
        name=platform_name,
        label="Opaque parser strict test",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        parse_target_ref_fn=lambda ref: None,
    )
    platform_registry.register(entry)
    try:
        with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
            chat_id, _thread_id, error = resolve_send_target(
                platform_name, "dm:panyaozhen", pass_unresolved_references=True
            )
    finally:
        platform_registry.unregister(platform_name)

    assert chat_id is None
    assert error is not None

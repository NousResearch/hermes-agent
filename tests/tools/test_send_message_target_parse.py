"""Parser-only and lightweight routing tests for send_message targets.

These stay separate from ``test_send_message_tool.py`` because that module
skips wholesale when optional Telegram dependencies are not installed.
"""

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from tools.send_message_tool import _parse_target_ref, _send_to_platform, send_message_tool


def _register_twitter_platform():
    from gateway.platform_registry import PlatformEntry, platform_registry
    from plugins.platforms.twitter import register

    previous = platform_registry.get("twitter")

    class Context:
        @staticmethod
        def register_platform(**kwargs):
            platform_registry.register(PlatformEntry(source="plugin", **kwargs))

        @staticmethod
        def register_tool(**_kwargs):
            return None

    register(Context())
    return previous


def _restore_twitter_platform(previous):
    from gateway.platform_registry import platform_registry

    if previous is None:
        platform_registry.unregister("twitter")
    else:
        platform_registry.register(previous)


def _run_async_immediately(coro):
    return asyncio.run(coro)


def test_photon_e164_target_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref("photon", "+15551234567")

    assert chat_id == "+15551234567"
    assert thread_id is None
    assert is_explicit is True


def test_e164_target_still_requires_phone_platform() -> None:
    assert _parse_target_ref("matrix", "+15551234567")[2] is False


def test_whatsapp_group_jid_target_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "whatsapp", "120363408391911677@g.us"
    )

    assert chat_id == "120363408391911677@g.us"
    assert thread_id is None
    assert is_explicit is True


def test_whatsapp_native_jids_are_explicit() -> None:
    assert _parse_target_ref("whatsapp", "19255551234@s.whatsapp.net")[2] is True
    assert _parse_target_ref("whatsapp", "149606612619433@lid")[2] is True
    assert _parse_target_ref("whatsapp", "status@broadcast")[2] is True
    assert _parse_target_ref("whatsapp", "120363000000000000@newsletter")[2] is True


def test_whatsapp_jid_suffix_only_matches_whatsapp() -> None:
    assert _parse_target_ref("telegram", "120363408391911677@g.us")[2] is False
    assert _parse_target_ref("signal", "149606612619433@lid")[2] is False


def test_whatsapp_friendly_name_still_uses_directory_resolution() -> None:
    assert _parse_target_ref("whatsapp", "general")[2] is False


@pytest.mark.parametrize(
    ("target", "chat_id", "interaction_id"),
    [
        ("tweet:100:101:102", "tweet:100:101", "102"),
        ("dm:42-7:501", "dm:42-7", "501"),
    ],
)
def test_twitter_plugin_parser_preserves_colon_route_and_interaction_id(
    target, chat_id, interaction_id
) -> None:
    previous = _register_twitter_platform()
    try:
        assert _parse_target_ref("twitter", target) == (
            chat_id,
            interaction_id,
            True,
        )
    finally:
        _restore_twitter_platform(previous)


def test_send_message_routes_twitter_reply_with_separate_interaction_id() -> None:
    previous = _register_twitter_platform()
    twitter = Platform("twitter")
    twitter_cfg = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={twitter: twitter_cfg},
        get_home_channel=lambda _platform: None,
    )

    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("gateway.channel_directory.resolve_channel_name", side_effect=AssertionError("explicit Twitter target must not use directory resolution")), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "twitter:tweet:100:101:102",
                        "message": "scheduled reply",
                    }
                )
            )
    finally:
        _restore_twitter_platform(previous)

    assert result["success"] is True
    send_mock.assert_awaited_once_with(
        twitter,
        twitter_cfg,
        "tweet:100:101",
        "scheduled reply",
        thread_id="102",
        media_files=[],
        force_document=False,
    )


@pytest.mark.asyncio
async def test_twitter_live_adapter_receives_media_without_omission_warning(
    monkeypatch,
) -> None:
    previous = _register_twitter_platform()
    twitter = Platform("twitter")
    media_files = [("/tmp/image.png", False)]
    recorded = {}

    class Adapter:
        async def send(self, *, chat_id, content, metadata=None):
            recorded["metadata"] = metadata
            return SimpleNamespace(success=True, message_id="tweet-1")

    runner = SimpleNamespace(adapters={twitter: Adapter()})
    fake_gateway_run = ModuleType("gateway.run")
    fake_gateway_run._gateway_runner_ref = lambda: runner
    monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

    try:
        result = await _send_to_platform(
            twitter,
            SimpleNamespace(enabled=True, token=None, extra={}),
            "timeline",
            "hello with image",
            media_files=media_files,
        )
    finally:
        _restore_twitter_platform(previous)

    assert result == {"success": True, "message_id": "tweet-1"}
    assert recorded["metadata"] == {"media_files": media_files}


@pytest.mark.asyncio
async def test_twitter_standalone_sender_accepts_media_only(monkeypatch) -> None:
    previous = _register_twitter_platform()
    twitter = Platform("twitter")
    media_files = [("/tmp/image.png", False)]

    from gateway.platform_registry import platform_registry

    entry = platform_registry.get("twitter")
    original_sender = entry.standalone_sender_fn
    standalone_send = AsyncMock(
        return_value={"success": True, "message_id": "tweet-1"}
    )
    entry.standalone_sender_fn = standalone_send
    fake_gateway_run = ModuleType("gateway.run")
    fake_gateway_run._gateway_runner_ref = lambda: None
    monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

    try:
        result = await _send_to_platform(
            twitter,
            SimpleNamespace(enabled=True, token=None, extra={}),
            "timeline",
            "",
            media_files=media_files,
        )
    finally:
        entry.standalone_sender_fn = original_sender
        _restore_twitter_platform(previous)

    assert result == {"success": True, "message_id": "tweet-1"}
    standalone_send.assert_awaited_once_with(
        SimpleNamespace(enabled=True, token=None, extra={}),
        "timeline",
        "",
        thread_id=None,
        media_files=media_files,
        force_document=False,
    )


@pytest.mark.asyncio
async def test_plugin_without_media_capability_still_rejects_media_only() -> None:
    from gateway.platform_registry import PlatformEntry, platform_registry

    standalone_send = AsyncMock(return_value={"success": True})
    platform_registry.register(
        PlatformEntry(
            name="textonly_test",
            label="Text only test",
            adapter_factory=lambda _config: None,
            check_fn=lambda: True,
            standalone_sender_fn=standalone_send,
        )
    )
    platform = Platform("textonly_test")

    try:
        result = await _send_to_platform(
            platform,
            SimpleNamespace(enabled=True, token=None, extra={}),
            "channel",
            "",
            media_files=[("/tmp/image.png", False)],
        )
    finally:
        platform_registry.unregister("textonly_test")

    assert "only media attachments" in result["error"]
    standalone_send.assert_not_awaited()


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

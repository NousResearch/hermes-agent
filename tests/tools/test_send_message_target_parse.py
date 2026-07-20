"""Parser-only and lightweight routing tests for send_message targets.

These stay separate from ``test_send_message_tool.py`` because that module
skips wholesale when optional Telegram dependencies are not installed.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.send_message_tool import _parse_target_ref, send_message_tool


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


def test_teams_discovered_thread_target_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "teams",
        "19:channel@thread.tacv2;messageid=1780267076971",
    )

    assert chat_id == "19:channel@thread.tacv2"
    assert thread_id == "1780267076971"
    assert is_explicit is True


def test_teams_friendly_name_still_uses_directory_resolution() -> None:
    assert _parse_target_ref("teams", "general")[2] is False


def test_teams_plain_native_targets_are_explicit() -> None:
    assert _parse_target_ref("teams", "19:channel@thread.tacv2") == (
        "19:channel@thread.tacv2",
        None,
        True,
    )
    assert _parse_target_ref("teams", "a:personal-conversation") == (
        "a:personal-conversation",
        None,
        True,
    )


def test_teams_malformed_thread_targets_are_not_explicit() -> None:
    assert _parse_target_ref("teams", "19:channel@thread.tacv2;messageid=")[2] is False
    assert (
        _parse_target_ref(
            "teams",
            "19:channel@thread.tacv2;messageid=../activities",
        )[2]
        is False
    )
    assert (
        _parse_target_ref(
            "teams",
            "19:channel@thread.tacv2;messageid=123;messageid=456",
        )[2]
        is False
    )


def test_send_message_routes_discovered_teams_thread_target() -> None:
    teams_platform = Platform("teams")
    teams_cfg = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={teams_platform: teams_cfg},
        get_home_channel=lambda _platform: None,
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", side_effect=AssertionError("native Teams target should not use directory resolution")), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "teams:19:channel@thread.tacv2;messageid=1780267076971",
                    "message": "hello thread",
                }
            )
        )

    assert result["success"] is True
    send_mock.assert_awaited_once_with(
        teams_platform,
        teams_cfg,
        "19:channel@thread.tacv2",
        "hello thread",
        thread_id="1780267076971",
        media_files=[],
        force_document=False,
    )


def test_send_message_routes_directory_resolved_teams_thread_target() -> None:
    teams_platform = Platform("teams")
    teams_cfg = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={teams_platform: teams_cfg},
        get_home_channel=lambda _platform: None,
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", return_value="19:channel@thread.tacv2;messageid=1780267076971"), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "teams:general",
                    "message": "hello thread",
                }
            )
        )

    assert result["success"] is True
    send_mock.assert_awaited_once_with(
        teams_platform,
        teams_cfg,
        "19:channel@thread.tacv2",
        "hello thread",
        thread_id="1780267076971",
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

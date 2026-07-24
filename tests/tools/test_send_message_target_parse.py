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


_SIGNAL_GROUP_ID = "w3W0kYAdHINWdTXDFl18LiH9O1xVwIadYMhVOD9tFaE="


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


def test_signal_prefixed_group_id_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "signal", f"group:{_SIGNAL_GROUP_ID}"
    )

    assert chat_id == f"group:{_SIGNAL_GROUP_ID}"
    assert thread_id is None
    assert is_explicit is True


def test_signal_prefixed_group_id_strips_inner_whitespace() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "signal", f"group:  {_SIGNAL_GROUP_ID}  "
    )

    assert chat_id == f"group:{_SIGNAL_GROUP_ID}"
    assert thread_id is None
    assert is_explicit is True


def test_signal_group_prefix_accepts_future_id_forms() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "signal", "group:abc-def_123"
    )

    assert chat_id == "group:abc-def_123"
    assert thread_id is None
    assert is_explicit is True


def test_signal_bare_gv2_group_id_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref("signal", _SIGNAL_GROUP_ID)

    assert chat_id == f"group:{_SIGNAL_GROUP_ID}"
    assert thread_id is None
    assert is_explicit is True


def test_signal_non_group_shapes_are_not_reclassified() -> None:
    non_groups = (
        f"group.{_SIGNAL_GROUP_ID}",
        "aGVsbG8=",
        "0123456789abcdef0123456789abcdef01234567",
        f"{'A' * 41}=",
        "123e4567-e89b-12d3-a456-426614174000",
    )

    for target in non_groups:
        chat_id, thread_id, is_explicit = _parse_target_ref("signal", target)
        assert chat_id is None
        assert thread_id is None
        assert is_explicit is False


def test_signal_long_digits_keep_numeric_recipient_behavior() -> None:
    digits = "1234567890123456789012345678901234567890"
    chat_id, thread_id, is_explicit = _parse_target_ref("signal", digits)

    assert chat_id == digits
    assert thread_id is None
    assert is_explicit is True


def test_send_message_routes_bare_signal_group_id_without_directory_lookup() -> None:
    signal_cfg = SimpleNamespace(
        enabled=True,
        token=None,
        extra={"http_url": "http://signal", "account": "+15551234567"},
    )
    config = SimpleNamespace(
        platforms={Platform.SIGNAL: signal_cfg},
        get_home_channel=lambda _platform: SimpleNamespace(chat_id="+15551234567"),
    )

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch("tools.interrupt.is_interrupted", return_value=False),
        patch(
            "gateway.channel_directory.resolve_channel_name",
            side_effect=AssertionError("raw Signal group ID should bypass directory lookup"),
        ),
        patch("model_tools._run_async", side_effect=_run_async_immediately),
        patch(
            "tools.send_message_tool._send_to_platform",
            new=AsyncMock(return_value={"success": True}),
        ) as send_mock,
        patch("gateway.mirror.mirror_to_session", return_value=True),
    ):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": f"signal:{_SIGNAL_GROUP_ID}",
                    "message": "hello group",
                }
            )
        )

    assert result["success"] is True
    assert "note" not in result
    send_mock.assert_awaited_once_with(
        Platform.SIGNAL,
        signal_cfg,
        f"group:{_SIGNAL_GROUP_ID}",
        "hello group",
        thread_id=None,
        media_files=[],
        force_document=False,
    )

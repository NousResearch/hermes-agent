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


# ---------------------------------------------------------------------------
# WeCom native ID routing — regression tests for #wecom-group-route (2026-07)
#
# Before the fix _parse_target_ref had no wecom branch, so 'wecom:wr<groupid>'
# fell through to resolve_channel_name; the resolver's step-0 exact-id lookup
# would return the same id, which the caller then re-parsed and still saw as
# non-explicit, ultimately delivering the message to the home DM channel
# instead of the target group.
# ---------------------------------------------------------------------------


def test_wecom_group_id_is_explicit() -> None:
    """WeCom open_chatid (group) starts with 'wr' — must route as explicit."""
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "wecom", "wrFQbCAAA0_qNz5FZGaKcXauJTU7U3Q"
    )
    assert chat_id == "wrFQbCAAA0_qNz5FZGaKcXauJTU7U3Q"
    assert thread_id is None
    assert is_explicit is True


def test_wecom_user_id_is_explicit() -> None:
    """WeCom open_userid (DM) starts with 'wo' — also explicit."""
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "wecom", "wofFQbCAAAW20pGpj4fHwyCCz2WyUYCA"
    )
    assert chat_id == "wofFQbCAAAW20pGpj4fHwyCCz2WyUYCA"
    assert thread_id is None
    assert is_explicit is True


def test_wecom_wm_prefix_is_explicit() -> None:
    """'wm' covers ancillary WeCom IDs (media/message tokens, misc)."""
    assert _parse_target_ref("wecom", "wmXYZ12345")[2] is True


def test_wecom_leading_whitespace_ignored() -> None:
    """Callers occasionally splat 'wecom:  wr...' after string concat."""
    chat_id, _, is_explicit = _parse_target_ref("wecom", "  wrABCDEF  ")
    assert is_explicit is True
    assert chat_id == "wrABCDEF"


def test_wecom_friendly_name_still_uses_directory_resolution() -> None:
    """Non-native names (aliases, group titles) must NOT be treated as
    explicit so the channel-directory resolver still runs."""
    assert _parse_target_ref("wecom", "工作群")[2] is False
    assert _parse_target_ref("wecom", "engineering")[2] is False


def test_wecom_prefix_only_matches_wecom_platform() -> None:
    """The wr/wo/wm heuristic must not leak into other platforms."""
    assert _parse_target_ref("telegram", "wrFQbCAAA0_qNz5FZGa")[2] is False
    assert _parse_target_ref("weixin", "wofFQbCAAAW20pGpj4fH")[2] is False


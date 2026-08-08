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


def test_unresolved_explicit_target_errors_while_bare_platform_uses_home(
    tmp_path,
    monkeypatch,
) -> None:
    from gateway import channel_directory

    target_id = "19:private-dm@thread.v2"
    directory_path = tmp_path / "channel_directory.json"
    directory_path.write_text(
        json.dumps({
            "platforms": {
                "teams": [{"id": target_id, "name": "Private DM", "type": "dm"}]
            }
        })
    )
    monkeypatch.setattr(channel_directory, "DIRECTORY_PATH", directory_path)
    monkeypatch.setattr(
        channel_directory,
        "CHANNEL_ALIASES_PATH",
        tmp_path / "channel_aliases.json",
    )

    teams_platform = Platform("teams")
    teams_cfg = SimpleNamespace(enabled=True, token=None, extra={})
    config = SimpleNamespace(
        platforms={teams_platform: teams_cfg},
        get_home_channel=lambda _platform: SimpleNamespace(chat_id="home-chat"),
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.send_message_tool._send_to_platform",
             new=AsyncMock(return_value={"success": True}),
         ) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        explicit_result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": f"teams:{target_id}",
                    "message": "confidential",
                }
            )
        )
        bare_result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "teams",
                    "message": "status update",
                }
            )
        )

    assert "Could not resolve" in explicit_result["error"]
    assert bare_result["success"] is True
    assert bare_result["note"] == "Sent to teams home channel (chat_id: home-chat)"
    send_mock.assert_awaited_once_with(
        teams_platform,
        teams_cfg,
        "home-chat",
        "status update",
        thread_id=None,
        media_files=[],
        force_document=False,
    )

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


def test_line_user_id_target_is_explicit() -> None:
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "line", "U0123456789abcdef0123456789abcdef"
    )

    assert chat_id == "U0123456789abcdef0123456789abcdef"
    assert thread_id is None
    assert is_explicit is True


def test_line_group_and_room_targets_are_not_direct_send_targets() -> None:
    for native_id in ("C0123456789abcdef0123456789abcdef", "R0123456789abcdef0123456789abcdef"):
        chat_id, thread_id, is_explicit = _parse_target_ref("line", native_id)

        assert chat_id is None
        assert thread_id is None
        assert is_explicit is False


def test_send_message_uses_line_home_channel_from_env(monkeypatch) -> None:
    line_platform = Platform("line")
    config = SimpleNamespace(
        platforms={},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "dummy-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "dummy-secret")
    monkeypatch.setenv("LINE_HOME_CHANNEL", "Uhomechannel0123456789abcdef")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name", side_effect=AssertionError("bare line should use home env, not directory")), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "line",
                    "message": "hello line",
                }
            )
        )

    assert result["success"] is True
    assert result["note"] == "Sent to line home channel (chat_id: Uhomechannel0123456789abcdef)"
    send_mock.assert_awaited_once()
    assert send_mock.await_args is not None
    args = send_mock.await_args.args
    assert args[0] == line_platform
    assert args[1].enabled is True
    assert args[1].extra["channel_access_token"] == "dummy-token"
    assert args[1].extra["channel_secret"] == "dummy-secret"
    assert args[2:] == ("Uhomechannel0123456789abcdef", "hello line")


def test_line_media_routes_images_and_documents_via_live_adapter(tmp_path) -> None:
    from tools.send_message_tool import _send_to_platform
    from gateway.platforms.base import SendResult

    line_platform = Platform("line")
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"png")
    doc_path = tmp_path / "report.pdf"
    doc_path.write_bytes(b"pdf")

    adapter = AsyncMock()
    adapter.send.return_value = SendResult(success=True, message_id="text")
    adapter.send_image_file.return_value = SendResult(success=True, message_id="img")
    adapter.send_document.return_value = SendResult(success=True, message_id="doc")
    runner = SimpleNamespace(adapters={line_platform: adapter})
    pconfig = SimpleNamespace(enabled=True, token=None, extra={})

    with patch("gateway.run._gateway_runner_ref", return_value=runner):
        result = asyncio.run(
            _send_to_platform(
                line_platform,
                pconfig,
                "Utarget",
                "attached",
                media_files=[(str(image_path), False), (str(doc_path), False)],
            )
        )

    assert result["success"] is True
    adapter.send.assert_awaited_once()
    adapter.send_image_file.assert_awaited_once_with(
        chat_id="Utarget",
        image_path=str(image_path),
        caption=None,
        metadata=None,
    )
    adapter.send_document.assert_awaited_once_with(
        chat_id="Utarget",
        file_path=str(doc_path),
        caption=None,
        file_name="report.pdf",
        metadata=None,
    )


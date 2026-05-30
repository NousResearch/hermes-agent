from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def reload_send_message_tool(monkeypatch, enabled: bool):
    if enabled:
        monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")
    else:
        monkeypatch.delenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", raising=False)
    import tools.send_message_tool as smt
    return importlib.reload(smt)


def test_legacy_bot_tools_not_registered_by_default(monkeypatch):
    smt = reload_send_message_tool(monkeypatch, enabled=False)
    from tools.registry import registry

    assert registry.get_entry("send_bot_message") is None
    assert registry.get_entry("send_bot_approval_decision") is None
    assert "control-plane DB" in smt.send_bot_message_tool({"target": "discord:1", "recipient_bot_id": "2", "kind": "status", "reply_expected": False, "body": "x"})


def test_legacy_bot_tools_register_only_when_explicitly_enabled(monkeypatch):
    reload_send_message_tool(monkeypatch, enabled=True)
    from tools.registry import registry

    assert registry.get_entry("send_bot_message") is not None
    assert registry.get_entry("send_bot_approval_decision") is not None


def test_discord_adapter_rejects_inbound_bot_msg_by_default(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "12345")
    monkeypatch.delenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", raising=False)

    from gateway.config import PlatformConfig
    from gateway.platforms.discord import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999))
    bot = SimpleNamespace(bot=True, id=12345)
    msg = SimpleNamespace(
        author=bot,
        content="\n".join(["<@99999>", "BOT_MSG v1", "reply_expected: true", "kind: status", "---", "body"]),
        channel=SimpleNamespace(id=222),
        id=1,
    )

    assert adapter._should_accept_bot_message(msg, "mentions") is False
    assert adapter._should_react_malformed_bot_message(msg) is False


def test_lower_level_legacy_bot_send_handlers_are_disabled_by_default(monkeypatch):
    smt = reload_send_message_tool(monkeypatch, enabled=False)
    payload = {
        "target": "discord:1",
        "recipient_bot_id": "2",
        "kind": "status",
        "reply_expected": False,
        "body": "x",
    }
    assert json.loads(smt._handle_send_bot_message(payload))["error"] == "legacy_discord_bot_to_bot_disabled"
    assert json.loads(smt._handle_send_bot_approval_decision({
        "target": "discord:1",
        "recipient_bot_id": "2",
        "approval_id": "a1",
        "decision": "approve",
    }))["error"] == "legacy_discord_bot_to_bot_disabled"


def test_raw_allowed_bot_mentions_blocked_via_send_message_even_when_legacy_disabled(monkeypatch):
    smt = reload_send_message_tool(monkeypatch, enabled=False)
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "222")
    from gateway.config import Platform

    config = SimpleNamespace(
        platforms={Platform.DISCORD: SimpleNamespace(enabled=True, token="fake-token", extra={})},
        get_home_channel=lambda _platform: None,
    )
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        result = json.loads(smt.send_message_tool({"target": "discord:111", "message": "<@222>\nBOT_MSG v1\nkind: status\n---\nbody"}))
    assert "error" in result
    assert "control-plane DB" in result["error"]


@pytest.mark.asyncio
async def test_discord_standalone_bot_send_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", raising=False)
    from plugins.platforms.discord.adapter import _standalone_send_bot_message

    result = await _standalone_send_bot_message(
        SimpleNamespace(token="fake-token"),
        "1",
        recipient_bot_id="2",
        body="x",
        reply_expected=False,
        kind="status",
        correlation_id="c1",
    )
    assert result["error"] == "legacy_discord_bot_to_bot_disabled"


@pytest.mark.asyncio
async def test_discord_routing_guard_blocks_operational_final_response_at_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", raising=False)

    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
    from gateway.session import SessionSource

    class Adapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
            self.calls = []

        async def connect(self):
            return True

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.calls.append(content)
            return SendResult(success=True, message_id="m1")

        async def get_chat_info(self, chat_id):
            return {}

    adapter = Adapter()
    event = MessageEvent(
        text="trigger",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="555",
            chat_type="thread",
            thread_id="999",
            user_id="888",
            user_name="StatuePM",
            is_bot=True,
        ),
        message_id="111",
    )
    original = "ACTION_REQUIRED for Galt/default: route through the DB now"
    result = await adapter._send_text_response_with_routing_guard(event=event, text_content=original)

    assert result.success is True
    assert adapter.calls != [original]
    assert (tmp_path / "logs" / "routing_guard").exists()
    assert "ROUTING_GUARD" in adapter.calls[0]

"""Behavior contracts for SoLo's Telegram notification identity split."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.delivery import DeliveryTransport
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter, _standalone_send
from tools.send_message_tool import _send_to_platform


@pytest.mark.asyncio
async def test_standalone_delivery_prefers_solo_hermes_token(monkeypatch):
    captured = []

    async def fake_send(token, chat_id, message, **kwargs):
        captured.append((token, chat_id, message, kwargs))
        return {"success": True, "message_id": "11"}

    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "notification-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "conversation-token")
    monkeypatch.setattr("agent.secret_scope.get_secret", lambda name, default="": "notification-token" if name == "SOLO_HERMES_BOT_TOKEN" else default)
    monkeypatch.setattr("tools.send_message_tool._send_telegram", fake_send)

    result = await _standalone_send(
        SimpleNamespace(token="configured-conversation-token", extra={}),
        "8148316720",
        "cron result",
        thread_id="inherited-topic",
    )

    assert result["success"] is True
    assert captured[0][0] == "notification-token"
    assert captured[0][3]["thread_id"] is None


@pytest.mark.asyncio
async def test_standalone_delivery_fails_closed_without_dedicated_token(monkeypatch):
    captured = []

    async def fake_send(token, chat_id, message, **kwargs):
        captured.append((token, kwargs.get("thread_id")))
        return {"success": True, "message_id": "12"}

    monkeypatch.delenv("SOLO_HERMES_BOT_TOKEN", raising=False)
    monkeypatch.setattr("agent.secret_scope.get_secret", lambda *_args: "")
    monkeypatch.setattr("tools.send_message_tool._send_telegram", fake_send)

    result = await _standalone_send(
        SimpleNamespace(token="configured-conversation-token", extra={}),
        "8148316720",
        "cron result",
        thread_id="77",
    )

    assert result["success"] is False
    assert "SOLO_HERMES_BOT_TOKEN" in result["error"]
    assert captured == []


@pytest.mark.asyncio
async def test_standalone_delivery_rejects_noncanonical_target(monkeypatch):
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "notification-token")
    result = await _standalone_send(
        SimpleNamespace(token="conversation-token", extra={}),
        "agent-chat",
        "cron result",
    )
    assert result["success"] is False
    assert "canonical" in result["error"]


@pytest.mark.asyncio
async def test_adapter_notification_uses_solo_hermes_token(monkeypatch):
    captured = []

    async def fake_send(token, chat_id, message, **kwargs):
        captured.append((token, kwargs.get("thread_id")))
        return {
            "success": True,
            "message_id": 13,
            "notification_proof": {
                "sender_username": "@solo_hermes_bot",
                "target_chat_id": "8148316720",
                "job_id": None,
                "profile": None,
                "thread_id": None,
            },
        }

    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "notification-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "conversation-token")
    monkeypatch.setattr("agent.secret_scope.get_secret", lambda name, default="": "notification-token" if name == "SOLO_HERMES_BOT_TOKEN" else default)
    monkeypatch.setattr("tools.send_message_tool._send_telegram", fake_send)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="conversation-token"))
    result = await adapter.send_notification(
        "8148316720",
        "gateway online",
        metadata={"thread_id": "77"},
    )

    assert result.success is True
    assert result.message_id == "13"
    assert result.notification_proof["sender_username"] == "@solo_hermes_bot"
    assert captured == [("notification-token", None)]


@pytest.mark.asyncio
async def test_operational_send_routes_through_standalone_sender(monkeypatch):
    captured = []

    async def fake_standalone(_config, chat_id, message, **kwargs):
        captured.append((chat_id, message, kwargs))
        return {"success": True, "message_id": "flat"}

    entry = SimpleNamespace(standalone_sender_fn=fake_standalone)
    monkeypatch.setattr("gateway.platform_registry.platform_registry.get", lambda _name: entry)

    result = await _send_to_platform(
        Platform.TELEGRAM,
        SimpleNamespace(token="conversation-token", extra={}),
        "8148316720",
        "cron result",
        thread_id="inherited-topic",
        operational=True,
    )

    assert result["success"] is True
    assert captured == [
        (
            "8148316720",
            "cron result",
            {
                "thread_id": None,
                "media_files": [],
            },
        )
    ]


@pytest.mark.asyncio
async def test_regular_send_message_keeps_conversation_token(monkeypatch):
    captured = []

    async def fake_send(token, chat_id, message, **kwargs):
        captured.append((token, chat_id, kwargs.get("thread_id")))
        return {"success": True, "message_id": "regular"}

    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "notification-token")
    monkeypatch.setattr("tools.send_message_tool._send_telegram", fake_send)

    result = await _send_to_platform(
        Platform.TELEGRAM,
        SimpleNamespace(token="conversation-token", extra={}),
        "8148316720",
        "conversation message",
        thread_id="conversation-topic",
    )

    assert result["success"] is True
    assert captured == [("conversation-token", "8148316720", "conversation-topic")]


@pytest.mark.asyncio
async def test_adapter_notification_fails_closed_without_dedicated_token(monkeypatch):
    async def forbidden_standalone_send(*_args, **_kwargs):
        raise AssertionError("fallback notification must use the connected adapter bot")

    monkeypatch.delenv("SOLO_HERMES_BOT_TOKEN", raising=False)
    monkeypatch.setattr("agent.secret_scope.get_secret", lambda *_args: "")
    monkeypatch.setattr("tools.send_message_tool._send_telegram", forbidden_standalone_send)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="conversation-token"))
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=14))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    adapter._rich_messages_enabled = False

    result = await adapter.send_notification(
        "8148316720",
        "gateway online",
        metadata={"notify": True},
    )

    assert result.success is False
    assert "SOLO_HERMES_BOT_TOKEN" in result.error

@pytest.mark.asyncio
async def test_regular_adapter_send_stays_on_connected_conversation_bot(monkeypatch):
    async def forbidden_standalone_send(*_args, **_kwargs):
        raise AssertionError("regular adapter send must not use notification REST sender")

    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "notification-token")
    monkeypatch.setattr("tools.send_message_tool._send_telegram", forbidden_standalone_send)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="conversation-token"))
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=14))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    adapter._rich_messages_enabled = False

    result = await adapter.send("8148316720", "conversation reply", metadata={"notify": True})

    assert result.success is True
    bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_delivery_transport_uses_notification_lane_for_native_adapter():
    adapter = MagicMock()
    adapter.send_notification = AsyncMock(
        return_value=SendResult(success=True, message_id="15")
    )
    adapter.send = AsyncMock()
    transport = DeliveryTransport(
        adapter=adapter,
        config=None,
        transport_platform=Platform.TELEGRAM,
    )

    result = await transport.send_notification(
        Platform.TELEGRAM,
        "8148316720",
        "gateway online",
        metadata={"thread_id": "77"},
    )

    assert result.success is True
    adapter.send_notification.assert_awaited_once_with(
        "8148316720",
        "gateway online",
        metadata={},
    )
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_delivery_transport_preserves_relay_notification_routing():
    relay = MagicMock()
    relay.send_for_platform = AsyncMock(
        return_value=SendResult(success=True, message_id="16")
    )
    transport = DeliveryTransport(
        adapter=relay,
        config=None,
        transport_platform=Platform.RELAY,
    )

    await transport.send_notification(
        Platform.TELEGRAM,
        "8148316720",
        "gateway online",
        metadata={"scope_id": "scope"},
    )

    relay.send_for_platform.assert_awaited_once_with(
        Platform.TELEGRAM,
        "8148316720",
        "gateway online",
        metadata={"scope_id": "scope"},
    )

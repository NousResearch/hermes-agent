import json

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.support_ops_routing import (
    ALEX_MENTION,
    EMIL_OWNER_MENTION,
    SKYVISION_BACKEND_CHANNEL_ID,
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
)
from tools import send_message_tool


def _discord_config():
    return GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="fake-token"),
        }
    )


def test_send_message_blocks_backend_resolver_mention_in_control_tower(monkeypatch):
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: _discord_config())

    async def _unexpected_send(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("send path should be blocked before platform delivery")

    monkeypatch.setattr(send_message_tool, "_send_to_platform", _unexpected_send)

    result = json.loads(
        send_message_tool._handle_send(
            {
                "target": f"discord:{SKYVISION_CONTROL_TOWER_CHANNEL_ID}:1521047924069371954",
                "message": f"{ALEX_MENTION} моля за действие по клиентския бонус.",
            }
        )
    )

    assert "error" in result
    assert "blocked_backend_resolver_mention_wrong_discord_lane" in result["error"]
    assert "1504852408227069993" in result["error"]


def test_send_message_allows_backend_resolver_mention_in_backend_lane(monkeypatch):
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: _discord_config())
    sent = {}

    async def _fake_send(platform, pconfig, chat_id, message, **kwargs):
        sent.update({"platform": platform, "chat_id": chat_id, "message": message, **kwargs})
        return {"success": True, "message_id": "sent-1"}

    monkeypatch.setattr(send_message_tool, "_send_to_platform", _fake_send)

    result = json.loads(
        send_message_tool._handle_send(
            {
                "target": f"discord:{SKYVISION_BACKEND_CHANNEL_ID}:1521049963428053125",
                "message": f"{ALEX_MENTION} моля за действие по клиентския бонус.",
            }
        )
    )

    assert result["success"] is True
    assert result["message_id"] == "sent-1"
    assert sent["chat_id"] == SKYVISION_BACKEND_CHANNEL_ID
    assert sent["thread_id"] == "1521049963428053125"


def test_send_message_blocks_owner_route_back_mention_outside_control_tower(monkeypatch):
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: _discord_config())

    async def _unexpected_send(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("send path should be blocked before platform delivery")

    monkeypatch.setattr(send_message_tool, "_send_to_platform", _unexpected_send)

    result = json.loads(
        send_message_tool._handle_send(
            {
                "target": "discord:1504852553031221391:1521247233130106901",
                "message": f"{EMIL_OWNER_MENTION} Емо, Пламенка върна корекцията за SkyAI.",
            }
        )
    )

    assert "error" in result
    assert "blocked_owner_route_back_mention_wrong_discord_lane" in result["error"]
    assert SKYVISION_CONTROL_TOWER_CHANNEL_ID in result["error"]


def test_send_message_allows_owner_route_back_mention_in_control_tower(monkeypatch):
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: _discord_config())
    sent = {}

    async def _fake_send(platform, pconfig, chat_id, message, **kwargs):
        sent.update({"platform": platform, "chat_id": chat_id, "message": message, **kwargs})
        return {"success": True, "message_id": "sent-owner"}

    monkeypatch.setattr(send_message_tool, "_send_to_platform", _fake_send)

    result = json.loads(
        send_message_tool._handle_send(
            {
                "target": f"discord:{SKYVISION_CONTROL_TOWER_CHANNEL_ID}:1507026708702826617",
                "message": f"{EMIL_OWNER_MENTION} Емо, Пламенка върна корекцията за SkyAI.",
            }
        )
    )

    assert result["success"] is True
    assert result["message_id"] == "sent-owner"
    assert sent["chat_id"] == SKYVISION_CONTROL_TOWER_CHANNEL_ID
    assert sent["thread_id"] == "1507026708702826617"

import json
from gateway.config import GatewayConfig, Platform, PlatformConfig
from tools import send_message_tool


def _discord_config():
    return GatewayConfig(platforms={
        Platform.DISCORD: PlatformConfig(enabled=True, token="fake"),
    })


def test_authored_names_and_mentions_do_not_route_or_block(monkeypatch):
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: _discord_config())
    monkeypatch.setattr("gateway.channel_directory.is_discord_public_target", lambda _target: True)
    sent = {}

    async def _fake_send(platform, pconfig, chat_id, message, **kwargs):
        sent.update({"chat_id": chat_id, "message": message, **kwargs})
        return {"success": True, "message_id": "sent-1"}

    monkeypatch.setattr(send_message_tool, "_send_to_platform", _fake_send)
    message = "<@1282940511962791959> Алекс, моля провери казуса."
    result = json.loads(send_message_tool._handle_send({
        "target": "discord:1504852355588423801:1521047924069371954",
        "message": message,
    }))
    assert result["success"] is True
    assert sent["message"] == message


def test_unknown_or_dm_discord_target_fails_closed(monkeypatch):
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: _discord_config())
    monkeypatch.setattr("gateway.channel_directory.is_discord_public_target", lambda _target: False)
    result = json.loads(send_message_tool._handle_send({
        "target": "discord:123456789012345678",
        "message": "hello",
    }))
    assert "public guild channels/threads" in result["error"]

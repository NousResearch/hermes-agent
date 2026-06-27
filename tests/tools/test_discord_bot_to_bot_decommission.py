from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def reload_send_message_tool(monkeypatch, *, legacy_env: bool = False):
    if legacy_env:
        monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")
    else:
        monkeypatch.delenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", raising=False)
    import tools.send_message_tool as smt
    return importlib.reload(smt)


def test_legacy_bot_tools_never_registered_even_when_env_enabled(monkeypatch):
    reload_send_message_tool(monkeypatch, legacy_env=True)
    from tools.registry import registry

    assert registry.get_entry("send_bot_message") is None
    assert registry.get_entry("send_bot_approval_decision") is None


def test_legacy_bot_tool_shims_removed_from_send_message_module(monkeypatch):
    smt = reload_send_message_tool(monkeypatch, legacy_env=True)

    assert not hasattr(smt, "send_bot_message_tool")
    assert not hasattr(smt, "send_bot_approval_decision_tool")


def test_send_message_allows_raw_discord_bot_mention_via_normal_send(monkeypatch):
    smt = reload_send_message_tool(monkeypatch, legacy_env=True)
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "222")

    from gateway.config import Platform

    sent = []

    async def fake_send_to_platform(platform, pconfig, chat_id, message, **kwargs):
        sent.append({
            "platform": platform,
            "chat_id": chat_id,
            "message": message,
            "kwargs": kwargs,
        })
        return {"success": True, "message_id": "m1"}

    config = SimpleNamespace(
        platforms={Platform.DISCORD: SimpleNamespace(enabled=True, token="fake-token", extra={})},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr(smt, "_send_to_platform", fake_send_to_platform)
    monkeypatch.setattr("gateway.mirror.mirror_to_session", lambda *a, **kw: True)

    body = "hello <@222>\nBOT_MSG v1\nkind: status\n---\nbody"
    result = json.loads(smt.send_message_tool({"target": "discord:111", "message": body}))

    assert result["success"] is True
    assert sent == [{
        "platform": Platform.DISCORD,
        "chat_id": "111",
        "message": body,
        "kwargs": {"thread_id": None, "media_files": [], "force_document": False},
    }]


def test_runtime_files_do_not_contain_discord_routing_guard_strings():
    runtime_files = [
        Path("gateway/platforms/base.py"),
        Path("gateway/platforms/discord.py"),
        Path("plugins/platforms/discord/adapter.py"),
        Path("tools/send_message_tool.py"),
    ]
    needles = [
        "BOT_ROUTING_GUARD",
        "[ROUTING_GUARD]",
        "_send_text_response_with_routing_guard",
        "_should_guard_discord_bot_final_response",
    ]

    hits = {
        str(path): [needle for needle in needles if needle in path.read_text(encoding="utf-8")]
        for path in runtime_files
    }
    assert {path: file_hits for path, file_hits in hits.items() if file_hits} == {}


@pytest.mark.asyncio
async def test_discord_adapter_has_no_bot_to_bot_admission_runtime(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "12345")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")

    from gateway.config import PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))

    assert not hasattr(adapter, "_should_accept_bot_message")
    assert not hasattr(adapter, "_handle_bot_approval_decision")
    assert not hasattr(adapter, "_should_react_malformed_bot_message")


def test_discord_standalone_bot_send_helper_removed(monkeypatch):
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")
    import plugins.platforms.discord.adapter as adapter

    assert not hasattr(adapter, "_standalone_send_bot_message")

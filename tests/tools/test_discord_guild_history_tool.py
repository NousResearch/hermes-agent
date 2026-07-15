from __future__ import annotations

import json

from gateway.production_capability_prerequisites import FIRST_WAVE_TOOLSETS
from toolsets import _HERMES_CORE_TOOLS, resolve_toolset
from tools import discord_guild_history_tool as tool
from tools.registry import registry


class _Client:
    def read(self, **kwargs):
        return {
            "target": {
                "target_type": "guild_channel",
                "guild_id": "100",
                "channel_id": kwargs["channel_id"],
            },
            "messages": [],
            "query": {
                "limit": kwargs["limit"],
                "before_message_id": kwargs["before_message_id"],
                "after_message_id": kwargs["after_message_id"],
            },
            "has_more": False,
            "order": "oldest_to_newest",
        }


def test_tool_is_narrow_service_gated_and_passes_only_model_selected_query(
    monkeypatch,
) -> None:
    entry = registry.get_entry("discord_guild_history")
    assert entry is not None
    assert entry.toolset == "discord_guild_read"
    assert entry.check_fn is tool.discord_guild_history_configured
    assert "discord_guild_read" in FIRST_WAVE_TOOLSETS
    assert "discord_guild_history" in resolve_toolset("discord_guild_read")
    assert "discord_guild_history" not in _HERMES_CORE_TOOLS
    parameters = tool.DISCORD_GUILD_HISTORY_SCHEMA["parameters"]
    assert set(parameters["properties"]) == {
        "channel_id",
        "limit",
        "before_message_id",
        "after_message_id",
    }
    assert parameters["additionalProperties"] is False
    assert "requester" not in json.dumps(parameters).casefold()
    assert "authority" not in json.dumps(parameters).casefold()
    monkeypatch.setattr(
        tool,
        "privileged_discord_guild_history_client",
        lambda: _Client(),
    )

    result = json.loads(
        tool._read_guild_history({
            "channel_id": "200",
            "limit": 7,
            "after_message_id": "300",
        })
    )
    assert result["query"] == {
        "limit": 7,
        "before_message_id": None,
        "after_message_id": "300",
    }


def test_tool_reports_connector_boundary_block_without_fallback_dispatch(
    monkeypatch,
) -> None:
    class _Blocked:
        def read(self, **_kwargs):
            raise tool.DiscordGuildHistoryClientError("connector_history_blocked")

    monkeypatch.setattr(
        tool,
        "privileged_discord_guild_history_client",
        lambda: _Blocked(),
    )
    outer = json.loads(tool._read_guild_history({"channel_id": "200", "limit": 1}))
    assert outer["error"] == "connector_history_blocked"
    assert "DMs" in outer["instruction"]
    assert "authenticated requester's live" in outer["instruction"]


def test_model_schema_supports_acl_private_lanes_without_model_authorship() -> None:
    description = tool.DISCORD_GUILD_HISTORY_SCHEMA["description"]
    assert "ACL-authorized private guild lanes" in description
    assert "you cannot supply or broaden authority" in description

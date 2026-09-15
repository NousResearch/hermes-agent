"""Cross-layer tests for request-owned Discord permission-overwrite actions."""

import json
from unittest.mock import Mock, call

import pytest

from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools import discord_action_extensions_tool as _discord_extensions  # noqa: F401
from tools import discord_tool as discord
from tools.discord_api.permissions_action import DELETE_ACTION, SET_ACTION
from tools.registry import registry

GUILD = "123456789012345678"
OTHER_GUILD = "999999999999999999"
CHANNEL = "987654321098765432"
OVERWRITE = "777777777777777777"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    reset_session_vars()
    monkeypatch.setattr(discord, "_get_bot_token", lambda: "active-profile-token")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"discord": {"server_actions": ""}},
    )
    yield
    reset_session_vars()


def _bind(*, guild=GUILD, user="42"):
    return set_session_vars(
        platform="discord",
        user_id=user,
        scope_id=guild,
        profile="worker",
        session_key=f"agent:worker:discord:channel:{guild}:123",
    )


def _error(payload):
    return str(json.loads(payload)["error"])


def test_actions_are_owned_by_canonical_discord_admin_only():
    admin = registry._tools["discord_admin"]
    actions = admin.schema["parameters"]["properties"]["action"]["enum"]
    assert SET_ACTION in actions
    assert DELETE_ACTION in actions
    assert "discord_permissions" not in registry._tools


def test_set_verifies_channel_guild_before_mutating(monkeypatch):
    request = Mock(side_effect=[{"guild_id": GUILD}, None])
    monkeypatch.setattr(discord, "_discord_request", request)
    tokens = _bind()
    try:
        result = json.loads(
            discord.discord_admin_handler(
                action=SET_ACTION,
                channel_id=f"000{CHANNEL}",
                overwrite_id=f"000{OVERWRITE}",
                target_type="role",
                allow=1024,
                deny="8",
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["success"] is True
    assert result["target_type"] == "role"
    assert result["allow"] == "1024"
    assert request.call_args_list == [
        call("GET", f"/channels/{CHANNEL}", "active-profile-token"),
        call(
            "PUT",
            f"/channels/{CHANNEL}/permissions/{OVERWRITE}",
            "active-profile-token",
            body={"allow": "1024", "deny": "8", "type": 0},
        ),
    ]


def test_delete_verifies_channel_then_deletes(monkeypatch):
    request = Mock(side_effect=[{"guild_id": GUILD}, None])
    monkeypatch.setattr(discord, "_discord_request", request)
    tokens = _bind()
    try:
        result = json.loads(
            discord.discord_admin_handler(
                action=DELETE_ACTION,
                channel_id=CHANNEL,
                overwrite_id=OVERWRITE,
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["operation"] == "delete"
    assert request.call_args_list == [
        call("GET", f"/channels/{CHANNEL}", "active-profile-token"),
        call(
            "DELETE",
            f"/channels/{CHANNEL}/permissions/{OVERWRITE}",
            "active-profile-token",
        ),
    ]


def test_cross_guild_channel_fails_closed_before_mutation(monkeypatch):
    request = Mock(return_value={"guild_id": OTHER_GUILD})
    monkeypatch.setattr(discord, "_discord_request", request)
    tokens = _bind()
    try:
        result = discord.discord_admin_handler(
            action=SET_ACTION,
            channel_id=CHANNEL,
            overwrite_id=OVERWRITE,
            target_type="member",
        )
    finally:
        clear_session_vars(tokens)

    assert "not owned by the active Discord request guild" in _error(result)
    request.assert_called_once_with(
        "GET", f"/channels/{CHANNEL}", "active-profile-token"
    )


@pytest.mark.parametrize(
    ("platform", "user", "guild", "message"),
    [
        ("slack", "42", GUILD, "active Discord request context"),
        ("discord", "", GUILD, "authenticated requester"),
        ("discord", "42", "", "active guild context"),
    ],
)
def test_request_owner_is_required(monkeypatch, platform, user, guild, message):
    request = Mock()
    monkeypatch.setattr(discord, "_discord_request", request)
    tokens = set_session_vars(platform=platform, user_id=user, scope_id=guild)
    try:
        result = discord.discord_admin_handler(
            action=DELETE_ACTION,
            channel_id=CHANNEL,
            overwrite_id=OVERWRITE,
        )
    finally:
        clear_session_vars(tokens)
    assert message in _error(result)
    request.assert_not_called()


def test_invalid_wire_contract_rejected_before_channel_lookup(monkeypatch):
    request = Mock()
    monkeypatch.setattr(discord, "_discord_request", request)
    tokens = _bind()
    try:
        result = discord.discord_admin_handler(
            action=SET_ACTION,
            channel_id="0",
            overwrite_id=OVERWRITE,
            target_type="role",
        )
    finally:
        clear_session_vars(tokens)
    assert "within [1" in _error(result)
    request.assert_not_called()


def test_real_server_action_allowlist_can_enable_permission_action(monkeypatch):
    request = Mock(side_effect=[{"guild_id": GUILD}, None])
    monkeypatch.setattr(discord, "_discord_request", request)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"discord": {"server_actions": SET_ACTION}},
    )
    tokens = _bind()
    try:
        result = json.loads(
            discord.discord_admin_handler(
                action=SET_ACTION,
                channel_id=CHANNEL,
                overwrite_id=OVERWRITE,
                target_type="member",
            )
        )
    finally:
        clear_session_vars(tokens)
    assert result["success"] is True


def test_canonical_403_enrichment_uses_manage_roles_hint(monkeypatch):
    request = Mock(side_effect=discord.DiscordAPIError(403, "missing permissions"))
    monkeypatch.setattr(discord, "_discord_request", request)
    tokens = _bind()
    try:
        result = discord.discord_admin_handler(
            action=SET_ACTION,
            channel_id=CHANNEL,
            overwrite_id=OVERWRITE,
            target_type="role",
        )
    finally:
        clear_session_vars(tokens)
    error = _error(result)
    assert "MANAGE_ROLES" in error
    assert "missing permissions" in error

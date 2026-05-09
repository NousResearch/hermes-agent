"""Tests for the Discord server introspection and management tool."""

import json
import os
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from tools.discord_tool import (
    DiscordAPIError,
    _ACTIONS,
    _ADMIN_ACTIONS,
    _CORE_ACTIONS,
    _available_actions,
    _build_schema,
    _channel_type_name,
    _detect_capabilities,
    _discord_request,
    _enrich_403,
    _get_bot_token,
    _load_allowed_actions_config,
    _reset_capability_cache,
    check_discord_tool_requirements,
    discord_admin_handler,
    discord_core,
    get_dynamic_schema,
    get_dynamic_schema_admin,
    get_dynamic_schema_core,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_urlopen(response_data, status=200):
    """Create a mock for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Token / check_fn
# ---------------------------------------------------------------------------

class TestCheckRequirements:
    def test_no_token(self, monkeypatch):
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        assert check_discord_tool_requirements() is False

    def test_empty_token(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "")
        assert check_discord_tool_requirements() is False

    def test_valid_token(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token-123")
        assert check_discord_tool_requirements() is True

    def test_get_bot_token(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "  my-token  ")
        assert _get_bot_token() == "my-token"

    def test_get_bot_token_missing(self, monkeypatch):
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        assert _get_bot_token() is None


# ---------------------------------------------------------------------------
# Channel type names
# ---------------------------------------------------------------------------

class TestChannelTypeNames:
    def test_known_types(self):
        assert _channel_type_name(0) == "text"
        assert _channel_type_name(2) == "voice"
        assert _channel_type_name(4) == "category"
        assert _channel_type_name(5) == "announcement"
        assert _channel_type_name(13) == "stage"
        assert _channel_type_name(15) == "forum"

    def test_unknown_type(self):
        assert _channel_type_name(99) == "unknown(99)"


# ---------------------------------------------------------------------------
# Discord API request helper
# ---------------------------------------------------------------------------

class TestDiscordRequest:
    @patch("tools.discord_tool.urllib.request.urlopen")
    def test_get_request(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen({"ok": True})
        result = _discord_request("GET", "/test", "token123")
        assert result == {"ok": True}

        # Verify the request was constructed correctly
        call_args = mock_urlopen_fn.call_args
        req = call_args[0][0]
        assert "https://discord.com/api/v10/test" in req.full_url
        assert req.get_header("Authorization") == "Bot token123"
        assert req.get_method() == "GET"

    @patch("tools.discord_tool.urllib.request.urlopen")
    def test_get_with_params(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen({"ok": True})
        _discord_request("GET", "/test", "tok", params={"foo": "bar"})
        req = mock_urlopen_fn.call_args[0][0]
        assert "foo=bar" in req.full_url

    @patch("tools.discord_tool.urllib.request.urlopen")
    def test_post_with_body(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen({"id": "123"})
        result = _discord_request("POST", "/channels", "tok", body={"name": "test"})
        assert result == {"id": "123"}
        req = mock_urlopen_fn.call_args[0][0]
        assert req.data == json.dumps({"name": "test"}).encode("utf-8")

    @patch("tools.discord_tool.urllib.request.urlopen")
    def test_204_returns_none(self, mock_urlopen_fn):
        mock_resp = _mock_urlopen({}, status=204)
        mock_urlopen_fn.return_value = mock_resp
        result = _discord_request("PUT", "/pins/1", "tok")
        assert result is None

    @patch("tools.discord_tool.urllib.request.urlopen")
    def test_http_error(self, mock_urlopen_fn):
        error_body = json.dumps({"message": "Missing Access"}).encode()
        http_error = urllib.error.HTTPError(
            url="https://discord.com/api/v10/test",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=BytesIO(error_body),
        )
        mock_urlopen_fn.side_effect = http_error
        with pytest.raises(DiscordAPIError) as exc_info:
            _discord_request("GET", "/test", "tok")
        assert exc_info.value.status == 403
        assert "Missing Access" in exc_info.value.body


# ---------------------------------------------------------------------------
# Main handler: validation
# ---------------------------------------------------------------------------

class TestDiscordServerValidation:
    def test_no_token(self, monkeypatch):
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert "error" in result
        assert "DISCORD_BOT_TOKEN" in result["error"]

    def test_unknown_action(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_core(action="bad_action"))
        assert "error" in result
        assert "Unknown action" in result["error"]
        assert "available_actions" in result

    def test_missing_required_guild_id(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_admin_handler(action="list_channels"))
        assert "error" in result
        assert "guild_id" in result["error"]

    def test_missing_required_channel_id(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_core(action="fetch_messages"))
        assert "error" in result
        assert "channel_id" in result["error"]

    def test_missing_required_message_id_for_delete(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_admin_handler(action="delete_message", channel_id="11"))
        assert "error" in result
        assert "message_id" in result["error"]

    def test_missing_multiple_params(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_admin_handler(action="add_role"))
        assert "error" in result
        assert "guild_id" in result["error"]
        assert "user_id" in result["error"]
        assert "role_id" in result["error"]


# ---------------------------------------------------------------------------
# Action: list_guilds
# ---------------------------------------------------------------------------

class TestListGuilds:
    @patch("tools.discord_tool._discord_request")
    def test_list_guilds(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"id": "111", "name": "Test Server", "icon": "abc", "owner": True, "permissions": "123"},
            {"id": "222", "name": "Other Server", "icon": None, "owner": False, "permissions": "456"},
        ]
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert result["count"] == 2
        assert result["guilds"][0]["name"] == "Test Server"
        assert result["guilds"][1]["id"] == "222"
        mock_req.assert_called_once_with("GET", "/users/@me/guilds", "test-token")


# ---------------------------------------------------------------------------
# Action: server_info
# ---------------------------------------------------------------------------

class TestServerInfo:
    @patch("tools.discord_tool._discord_request")
    def test_server_info(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {
            "id": "111",
            "name": "My Server",
            "description": "A cool server",
            "icon": "icon_hash",
            "owner_id": "999",
            "approximate_member_count": 42,
            "approximate_presence_count": 10,
            "features": ["COMMUNITY"],
            "premium_tier": 2,
            "premium_subscription_count": 5,
            "verification_level": 1,
        }
        result = json.loads(discord_admin_handler(action="server_info", guild_id="111"))
        assert result["name"] == "My Server"
        assert result["member_count"] == 42
        assert result["online_count"] == 10
        mock_req.assert_called_once_with(
            "GET", "/guilds/111", "test-token", params={"with_counts": "true"}
        )


# ---------------------------------------------------------------------------
# Action: list_channels
# ---------------------------------------------------------------------------

class TestListChannels:
    @patch("tools.discord_tool._discord_request")
    def test_list_channels_organized(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"id": "10", "name": "General", "type": 4, "position": 0, "parent_id": None},
            {"id": "11", "name": "chat", "type": 0, "position": 0, "parent_id": "10", "topic": "Main chat", "nsfw": False},
            {"id": "12", "name": "voice", "type": 2, "position": 1, "parent_id": "10", "topic": None, "nsfw": False},
            {"id": "13", "name": "no-category", "type": 0, "position": 0, "parent_id": None, "topic": None, "nsfw": False},
        ]
        result = json.loads(discord_admin_handler(action="list_channels", guild_id="111"))
        assert result["total_channels"] == 3  # excludes the category itself
        groups = result["channel_groups"]
        # Uncategorized first
        assert groups[0]["category"] is None
        assert len(groups[0]["channels"]) == 1
        assert groups[0]["channels"][0]["name"] == "no-category"
        # Then the category
        assert groups[1]["category"]["name"] == "General"
        assert len(groups[1]["channels"]) == 2

    @patch("tools.discord_tool._discord_request")
    def test_empty_guild(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = []
        result = json.loads(discord_admin_handler(action="list_channels", guild_id="111"))
        assert result["total_channels"] == 0


# ---------------------------------------------------------------------------
# Action: channel_info
# ---------------------------------------------------------------------------

class TestChannelInfo:
    @patch("tools.discord_tool._discord_request")
    def test_channel_info(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {
            "id": "11", "name": "general", "type": 0, "guild_id": "111",
            "topic": "Welcome!", "nsfw": False, "position": 0,
            "parent_id": "10", "rate_limit_per_user": 0, "last_message_id": "999",
        }
        result = json.loads(discord_admin_handler(action="channel_info", channel_id="11"))
        assert result["name"] == "general"
        assert result["type"] == "text"
        assert result["guild_id"] == "111"


# ---------------------------------------------------------------------------
# Action: list_roles
# ---------------------------------------------------------------------------

class TestListRoles:
    @patch("tools.discord_tool._discord_request")
    def test_list_roles_sorted(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"id": "1", "name": "@everyone", "position": 0, "color": 0, "mentionable": False, "managed": False, "hoist": False},
            {"id": "2", "name": "Admin", "position": 2, "color": 16711680, "mentionable": True, "managed": False, "hoist": True},
            {"id": "3", "name": "Mod", "position": 1, "color": 255, "mentionable": True, "managed": False, "hoist": True},
        ]
        result = json.loads(discord_admin_handler(action="list_roles", guild_id="111"))
        assert result["count"] == 3
        # Should be sorted by position descending
        assert result["roles"][0]["name"] == "Admin"
        assert result["roles"][0]["color"] == "#ff0000"
        assert result["roles"][1]["name"] == "Mod"
        assert result["roles"][2]["name"] == "@everyone"


# ---------------------------------------------------------------------------
# Action: member_info
# ---------------------------------------------------------------------------

class TestMemberInfo:
    @patch("tools.discord_tool._discord_request")
    def test_member_info(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {
            "user": {"id": "42", "username": "testuser", "global_name": "Test User", "avatar": "abc", "bot": False},
            "nick": "Testy",
            "roles": ["2", "3"],
            "joined_at": "2024-01-01T00:00:00Z",
            "premium_since": None,
        }
        result = json.loads(discord_admin_handler(action="member_info", guild_id="111", user_id="42"))
        assert result["username"] == "testuser"
        assert result["nickname"] == "Testy"
        assert result["roles"] == ["2", "3"]


# ---------------------------------------------------------------------------
# Action: search_members
# ---------------------------------------------------------------------------

class TestSearchMembers:
    @patch("tools.discord_tool._discord_request")
    def test_search_members(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"user": {"id": "42", "username": "testuser", "global_name": "Test", "bot": False}, "nick": None, "roles": []},
        ]
        result = json.loads(discord_core(action="search_members", guild_id="111", query="test"))
        assert result["count"] == 1
        assert result["members"][0]["username"] == "testuser"
        mock_req.assert_called_once_with(
            "GET", "/guilds/111/members/search", "test-token",
            params={"query": "test", "limit": "50"},
        )

    @patch("tools.discord_tool._discord_request")
    def test_search_members_limit_capped(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = []
        discord_core(action="search_members", guild_id="111", query="x", limit=200)
        call_params = mock_req.call_args[1]["params"]
        assert call_params["limit"] == "100"  # Capped at 100


# ---------------------------------------------------------------------------
# Action: fetch_messages
# ---------------------------------------------------------------------------

class TestFetchMessages:
    @patch("tools.discord_tool._discord_request")
    def test_fetch_messages(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {
                "id": "1001",
                "content": "Hello world",
                "author": {"id": "42", "username": "user1", "global_name": "User One", "bot": False},
                "timestamp": "2024-01-01T12:00:00Z",
                "edited_timestamp": None,
                "attachments": [],
                "pinned": False,
            },
        ]
        result = json.loads(discord_core(action="fetch_messages", channel_id="11"))
        assert result["count"] == 1
        assert result["messages"][0]["content"] == "Hello world"
        assert result["messages"][0]["author"]["username"] == "user1"

    @patch("tools.discord_tool._discord_request")
    def test_fetch_messages_with_pagination(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = []
        discord_core(action="fetch_messages", channel_id="11", before="999", limit=10)
        call_params = mock_req.call_args[1]["params"]
        assert call_params["before"] == "999"
        assert call_params["limit"] == "10"


# ---------------------------------------------------------------------------
# Action: list_pins
# ---------------------------------------------------------------------------

class TestListPins:
    @patch("tools.discord_tool._discord_request")
    def test_list_pins(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"id": "500", "content": "Important announcement", "author": {"username": "admin"}, "timestamp": "2024-01-01T00:00:00Z"},
        ]
        result = json.loads(discord_admin_handler(action="list_pins", channel_id="11"))
        assert result["count"] == 1
        assert result["pinned_messages"][0]["content"] == "Important announcement"


# ---------------------------------------------------------------------------
# Actions: pin_message / unpin_message / delete_message
# ---------------------------------------------------------------------------

class TestPinUnpinDelete:
    @patch("tools.discord_tool._discord_request")
    def test_pin_message(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = None  # 204
        result = json.loads(discord_admin_handler(action="pin_message", channel_id="11", message_id="500"))
        assert result["success"] is True
        mock_req.assert_called_once_with("PUT", "/channels/11/pins/500", "test-token")

    @patch("tools.discord_tool._discord_request")
    def test_unpin_message(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = None
        result = json.loads(discord_admin_handler(action="unpin_message", channel_id="11", message_id="500"))
        assert result["success"] is True
        mock_req.assert_called_once_with("DELETE", "/channels/11/pins/500", "test-token")

    @patch("tools.discord_tool._discord_request")
    def test_delete_message(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = None
        result = json.loads(discord_admin_handler(action="delete_message", channel_id="11", message_id="500"))
        assert result["success"] is True
        assert "deleted" in result["message"]
        mock_req.assert_called_once_with("DELETE", "/channels/11/messages/500", "test-token")


# ---------------------------------------------------------------------------
# Action: create_thread
# ---------------------------------------------------------------------------

class TestCreateThread:
    @patch("tools.discord_tool._discord_request")
    def test_create_standalone_thread(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {"id": "800", "name": "New Thread"}
        result = json.loads(discord_core(action="create_thread", channel_id="11", name="New Thread"))
        assert result["success"] is True
        assert result["thread_id"] == "800"
        # Verify the API call
        mock_req.assert_called_once_with(
            "POST", "/channels/11/threads", "test-token",
            body={"name": "New Thread", "auto_archive_duration": 1440, "type": 11},
        )

    @patch("tools.discord_tool._discord_request")
    def test_create_thread_from_message(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {"id": "801", "name": "Discussion"}
        result = json.loads(discord_core(
            action="create_thread", channel_id="11", name="Discussion", message_id="1001",
        ))
        assert result["success"] is True
        mock_req.assert_called_once_with(
            "POST", "/channels/11/messages/1001/threads", "test-token",
            body={"name": "Discussion", "auto_archive_duration": 1440},
        )


# ---------------------------------------------------------------------------
# Actions: add_role / remove_role
# ---------------------------------------------------------------------------

class TestRoleManagement:
    @patch("tools.discord_tool._discord_request")
    def test_add_role(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = None
        result = json.loads(discord_admin_handler(
            action="add_role", guild_id="111", user_id="42", role_id="2",
        ))
        assert result["success"] is True
        mock_req.assert_called_once_with(
            "PUT", "/guilds/111/members/42/roles/2", "test-token",
        )

    @patch("tools.discord_tool._discord_request")
    def test_remove_role(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = None
        result = json.loads(discord_admin_handler(
            action="remove_role", guild_id="111", user_id="42", role_id="2",
        ))
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @patch("tools.discord_tool._discord_request")
    def test_api_error_handled(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = DiscordAPIError(403, '{"message": "Missing Access"}')
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert "error" in result
        assert "403" in result["error"]

    @patch("tools.discord_tool._discord_request")
    def test_unexpected_error_handled_admin(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = RuntimeError("something broke")
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert "error" in result
        assert "something broke" in result["error"]

    @patch("tools.discord_tool._discord_request")
    def test_unexpected_error_handled_core(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = RuntimeError("something broke")
        result = json.loads(discord_core(action="fetch_messages", channel_id="11"))
        assert "error" in result
        assert "something broke" in result["error"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_core_tool_registered(self):
        from tools.registry import registry
        entry = registry._tools.get("discord")
        assert entry is not None
        assert entry.schema["name"] == "discord"
        assert entry.toolset == "discord"
        assert entry.check_fn is not None
        assert entry.requires_env == ["DISCORD_BOT_TOKEN"]

    def test_admin_tool_registered(self):
        from tools.registry import registry
        entry = registry._tools.get("discord_admin")
        assert entry is not None
        assert entry.schema["name"] == "discord_admin"
        assert entry.toolset == "discord_admin"
        assert entry.check_fn is not None
        assert entry.requires_env == ["DISCORD_BOT_TOKEN"]

    def test_core_schema_actions(self):
        """Core static schema should list only core actions."""
        from tools.registry import registry
        entry = registry._tools["discord"]
        actions = set(entry.schema["parameters"]["properties"]["action"]["enum"])
        assert actions == {"fetch_messages", "search_members", "create_thread"}

    def test_admin_schema_actions(self):
        """Admin static schema should list only admin actions."""
        from tools.registry import registry
        entry = registry._tools["discord_admin"]
        actions = set(entry.schema["parameters"]["properties"]["action"]["enum"])
        expected_admin = set(_ACTIONS.keys()) - {"fetch_messages", "search_members", "create_thread"}
        assert actions == expected_admin

    def test_all_actions_covered(self):
        """Core + admin actions should cover all known actions."""
        assert set(_CORE_ACTIONS.keys()) | set(_ADMIN_ACTIONS.keys()) == set(_ACTIONS.keys())
        assert set(_CORE_ACTIONS.keys()) & set(_ADMIN_ACTIONS.keys()) == set()

    def test_schema_parameter_bounds(self):
        from tools.registry import registry
        entry = registry._tools["discord"]
        props = entry.schema["parameters"]["properties"]
        assert props["limit"]["minimum"] == 1
        assert props["limit"]["maximum"] == 100
        assert props["auto_archive_duration"]["enum"] == [60, 1440, 4320, 10080]

    def test_core_schema_description(self):
        """Core schema description should mention core actions."""
        from tools.registry import registry
        entry = registry._tools["discord"]
        desc = entry.schema["description"]
        assert "fetch_messages(channel_id)" in desc
        assert "search_members(guild_id, query)" in desc
        assert "create_thread(channel_id, name)" in desc
        # Admin actions should NOT be in core description
        assert "list_guilds()" not in desc
        assert "add_role(" not in desc

    def test_admin_schema_description(self):
        """Admin schema description should mention admin actions."""
        from tools.registry import registry
        entry = registry._tools["discord_admin"]
        desc = entry.schema["description"]
        assert "list_guilds()" in desc
        assert "add_role(guild_id, user_id, role_id)" in desc
        assert "delete_message(channel_id, message_id)" in desc
        # Core actions should NOT be in admin description
        assert "fetch_messages(" not in desc
        assert "create_thread(" not in desc

    def test_handler_callable(self):
        from tools.registry import registry
        entry = registry._tools["discord"]
        assert callable(entry.handler)
        entry_admin = registry._tools["discord_admin"]
        assert callable(entry_admin.handler)


# ---------------------------------------------------------------------------
# Toolset: discord / discord_admin only in hermes-discord
# ---------------------------------------------------------------------------

class TestToolsetInclusion:
    def test_discord_tools_in_hermes_discord_toolset(self):
        from toolsets import TOOLSETS
        assert "discord" in TOOLSETS["hermes-discord"]["tools"]
        assert "discord_admin" in TOOLSETS["hermes-discord"]["tools"]

    def test_discord_tools_not_in_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS
        assert "discord" not in _HERMES_CORE_TOOLS
        assert "discord_admin" not in _HERMES_CORE_TOOLS

    def test_discord_tools_not_in_other_toolsets(self):
        from toolsets import TOOLSETS
        for name, ts in TOOLSETS.items():
            if name in ("hermes-discord", "hermes-gateway", "discord", "discord_admin"):
                continue
            tools = ts.get("tools", [])
            assert "discord" not in tools or name == "discord", (
                f"discord tool should not be in toolset '{name}'"
            )
            assert "discord_admin" not in tools or name == "discord_admin", (
                f"discord_admin tool should not be in toolset '{name}'"
            )


# ---------------------------------------------------------------------------
# Capability detection (privileged intents)
# ---------------------------------------------------------------------------

class TestCapabilityDetection:
    def setup_method(self):
        _reset_capability_cache()

    def teardown_method(self):
        _reset_capability_cache()

    @patch("tools.discord_tool._discord_request")
    def test_both_intents_enabled(self, mock_req):
        # flags: GUILD_MEMBERS (1<<14) + MESSAGE_CONTENT (1<<18) = 278528
        mock_req.return_value = {"flags": (1 << 14) | (1 << 18)}
        caps = _detect_capabilities("tok")
        assert caps["has_members_intent"] is True
        assert caps["has_message_content"] is True
        assert caps["detected"] is True

    @patch("tools.discord_tool._discord_request")
    def test_no_intents(self, mock_req):
        mock_req.return_value = {"flags": 0}
        caps = _detect_capabilities("tok")
        assert caps["has_members_intent"] is False
        assert caps["has_message_content"] is False
        assert caps["detected"] is True

    @patch("tools.discord_tool._discord_request")
    def test_limited_intent_variants_counted(self, mock_req):
        # GUILD_MEMBERS_LIMITED (1<<15), MESSAGE_CONTENT_LIMITED (1<<19)
        mock_req.return_value = {"flags": (1 << 15) | (1 << 19)}
        caps = _detect_capabilities("tok")
        assert caps["has_members_intent"] is True
        assert caps["has_message_content"] is True

    @patch("tools.discord_tool._discord_request")
    def test_only_members_intent(self, mock_req):
        mock_req.return_value = {"flags": 1 << 14}
        caps = _detect_capabilities("tok")
        assert caps["has_members_intent"] is True
        assert caps["has_message_content"] is False

    @patch("tools.discord_tool._discord_request")
    def test_detection_failure_is_permissive(self, mock_req):
        """If detection fails (network/401/revoked token), expose everything
        and let runtime errors surface. Silent failure should never hide
        actions the bot actually has."""
        mock_req.side_effect = DiscordAPIError(401, "unauthorized")
        caps = _detect_capabilities("tok")
        assert caps["detected"] is False
        assert caps["has_members_intent"] is True
        assert caps["has_message_content"] is True

    @patch("tools.discord_tool._discord_request")
    def test_detection_is_cached(self, mock_req):
        mock_req.return_value = {"flags": 0}
        _detect_capabilities("tok")
        _detect_capabilities("tok")
        _detect_capabilities("tok")
        assert mock_req.call_count == 1

    @patch("tools.discord_tool._discord_request")
    def test_force_refresh(self, mock_req):
        mock_req.return_value = {"flags": 0}
        _detect_capabilities("tok")
        _detect_capabilities("tok", force=True)
        assert mock_req.call_count == 2

    @patch("tools.discord_tool._discord_request")
    def test_cache_is_keyed_by_token(self, mock_req):
        """Regression: token A's capabilities must not leak to token B.

        Before the fix, the cache was a single module-global dict. The first
        call populated it and every subsequent call — regardless of token —
        returned the same cached value, producing wrong schema gating for
        rotated or multi-token deployments.
        """
        def _per_token_flags(method, path, token, **_kwargs):
            # token A: both intents; token B: neither.
            if token == "tok_a":
                return {"flags": (1 << 14) | (1 << 18)}
            return {"flags": 0}

        mock_req.side_effect = _per_token_flags

        caps_a = _detect_capabilities("tok_a")
        caps_b = _detect_capabilities("tok_b")

        assert caps_a["has_members_intent"] is True
        assert caps_a["has_message_content"] is True
        assert caps_b["has_members_intent"] is False
        assert caps_b["has_message_content"] is False
        # Each token should hit the endpoint exactly once.
        assert mock_req.call_count == 2

        # Re-requesting either token serves from its own cache entry.
        _detect_capabilities("tok_a")
        _detect_capabilities("tok_b")
        assert mock_req.call_count == 2


# ---------------------------------------------------------------------------
# Config allowlist
# ---------------------------------------------------------------------------

class TestConfigAllowlist:
    @pytest.fixture(autouse=True)
    def _reset_tools_logger(self):
        """Restore the ``tools`` logger level after cross-test pollution.

        ``AIAgent(quiet_mode=True)`` globally sets ``tools`` and
        ``tools.*`` children to ``ERROR`` (see run_agent.py quiet_mode
        block).  xdist workers are persistent, so a streaming test on the
        same worker will silence WARNING-level logs from
        ``tools.discord_tool`` for every test that follows.  Reset here so
        ``caplog`` can capture warnings regardless of worker history.
        """
        import logging as _logging
        _prev_tools = _logging.getLogger("tools").level
        _prev_dt = _logging.getLogger("tools.discord_tool").level
        _logging.getLogger("tools").setLevel(_logging.NOTSET)
        _logging.getLogger("tools.discord_tool").setLevel(_logging.NOTSET)
        try:
            yield
        finally:
            _logging.getLogger("tools").setLevel(_prev_tools)
            _logging.getLogger("tools.discord_tool").setLevel(_prev_dt)

    def test_empty_string_returns_none(self, monkeypatch):
        """Empty config means no allowlist — all actions visible."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        assert _load_allowed_actions_config() is None

    def test_missing_key_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {}},
        )
        assert _load_allowed_actions_config() is None

    def test_comma_separated_string(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds,list_channels,fetch_messages"}},
        )
        result = _load_allowed_actions_config()
        assert result == ["list_guilds", "list_channels", "fetch_messages"]

    def test_yaml_list(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ["list_guilds", "server_info"]}},
        )
        result = _load_allowed_actions_config()
        assert result == ["list_guilds", "server_info"]

    def test_unknown_names_dropped(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds,bogus_action,fetch_messages"}},
        )
        with caplog.at_level("WARNING"):
            result = _load_allowed_actions_config()
        assert result == ["list_guilds", "fetch_messages"]
        assert "bogus_action" in caplog.text

    def test_config_load_failure_is_permissive(self, monkeypatch):
        """If config can't be loaded at all, fall back to None (all allowed)."""
        def bad_load():
            raise RuntimeError("disk gone")
        monkeypatch.setattr("hermes_cli.config.load_config", bad_load)
        assert _load_allowed_actions_config() is None

    def test_unexpected_type_ignored(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": {"unexpected": "dict"}}},
        )
        with caplog.at_level("WARNING"):
            result = _load_allowed_actions_config()
        assert result is None
        assert "unexpected type" in caplog.text


# ---------------------------------------------------------------------------
# Action filtering combines intents + allowlist
# ---------------------------------------------------------------------------

class TestAvailableActions:
    def test_all_available_when_unrestricted(self):
        caps = {"detected": True, "has_members_intent": True, "has_message_content": True}
        assert _available_actions(caps, None) == list(_ACTIONS.keys())

    def test_no_members_intent_hides_member_actions(self):
        caps = {"detected": True, "has_members_intent": False, "has_message_content": True}
        actions = _available_actions(caps, None)
        assert "search_members" not in actions
        assert "member_info" not in actions
        # fetch_messages stays — MESSAGE_CONTENT affects content field but action works
        assert "fetch_messages" in actions

    def test_no_message_content_keeps_fetch_messages(self):
        """MESSAGE_CONTENT affects the content field, not the action.
        Hiding fetch_messages would lose author/timestamp/attachments access."""
        caps = {"detected": True, "has_members_intent": True, "has_message_content": False}
        actions = _available_actions(caps, None)
        assert "fetch_messages" in actions
        assert "list_pins" in actions

    def test_allowlist_intersects_with_intents(self):
        """Allowlist can only narrow — not re-enable intent-gated actions."""
        caps = {"detected": True, "has_members_intent": False, "has_message_content": True}
        allowlist = ["list_guilds", "search_members", "fetch_messages"]
        actions = _available_actions(caps, allowlist)
        # search_members gated by intent → stripped even though allowlisted
        assert actions == ["list_guilds", "fetch_messages"]

    def test_empty_allowlist_yields_empty(self):
        caps = {"detected": True, "has_members_intent": True, "has_message_content": True}
        assert _available_actions(caps, []) == []

    def test_allowlist_preserves_canonical_order(self):
        caps = {"detected": True, "has_members_intent": True, "has_message_content": True}
        # Pass allowlist out of canonical order
        allowlist = ["fetch_messages", "list_guilds", "server_info"]
        assert _available_actions(caps, allowlist) == ["list_guilds", "server_info", "fetch_messages"]


# ---------------------------------------------------------------------------
# Dynamic schema build (integration of intents + config)
# ---------------------------------------------------------------------------

class TestDynamicSchema:
    def setup_method(self):
        _reset_capability_cache()

    def teardown_method(self):
        _reset_capability_cache()

    @patch("tools.discord_tool._discord_request")
    def test_no_token_returns_none(self, mock_req, monkeypatch):
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        assert get_dynamic_schema_core() is None
        assert get_dynamic_schema_admin() is None
        mock_req.assert_not_called()

    @patch("tools.discord_tool._discord_request")
    def test_full_intents_core_schema(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.return_value = {"flags": (1 << 14) | (1 << 18)}
        schema = get_dynamic_schema_core()
        actions = set(schema["parameters"]["properties"]["action"]["enum"])
        assert actions == set(_CORE_ACTIONS.keys())
        assert schema["name"] == "discord"

    @patch("tools.discord_tool._discord_request")
    def test_full_intents_admin_schema(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.return_value = {"flags": (1 << 14) | (1 << 18)}
        schema = get_dynamic_schema_admin()
        actions = set(schema["parameters"]["properties"]["action"]["enum"])
        assert actions == set(_ADMIN_ACTIONS.keys())
        assert schema["name"] == "discord_admin"
        # No content warning when MESSAGE_CONTENT is enabled
        assert "MESSAGE_CONTENT" not in schema["description"]

    @patch("tools.discord_tool._discord_request")
    def test_no_members_intent_removes_member_actions_from_admin_schema(
        self, mock_req, monkeypatch,
    ):
        """member_info is an admin action; it should be hidden when
        GUILD_MEMBERS intent is missing."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.return_value = {"flags": 1 << 18}  # only MESSAGE_CONTENT
        schema = get_dynamic_schema_admin()
        actions = schema["parameters"]["properties"]["action"]["enum"]
        assert "member_info" not in actions
        assert "member_info" not in schema["description"]

    @patch("tools.discord_tool._discord_request")
    def test_no_members_intent_hides_search_members_from_core(
        self, mock_req, monkeypatch,
    ):
        """search_members is a core action gated by GUILD_MEMBERS intent."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.return_value = {"flags": 1 << 18}  # only MESSAGE_CONTENT
        schema = get_dynamic_schema_core()
        actions = schema["parameters"]["properties"]["action"]["enum"]
        assert "search_members" not in actions

    @patch("tools.discord_tool._discord_request")
    def test_no_message_content_adds_warning_note(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.return_value = {"flags": 1 << 14}  # only GUILD_MEMBERS
        schema = get_dynamic_schema_core()
        assert "MESSAGE_CONTENT" in schema["description"]
        # But fetch_messages is still available
        actions = schema["parameters"]["properties"]["action"]["enum"]
        assert "fetch_messages" in actions

    @patch("tools.discord_tool._discord_request")
    def test_config_allowlist_narrows_admin_schema(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds,list_channels"}},
        )
        mock_req.return_value = {"flags": (1 << 14) | (1 << 18)}
        schema = get_dynamic_schema_admin()
        actions = schema["parameters"]["properties"]["action"]["enum"]
        assert actions == ["list_guilds", "list_channels"]
        assert "list_guilds()" in schema["description"]
        assert "add_role(" not in schema["description"]

    @patch("tools.discord_tool._discord_request")
    def test_empty_allowlist_with_valid_values_hides_tools(self, mock_req, monkeypatch):
        """If the allowlist resolves to zero valid actions (e.g. all names
        were typos), get_dynamic_schema returns None so the tool is dropped."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "typo_one,typo_two"}},
        )
        mock_req.return_value = {"flags": (1 << 14) | (1 << 18)}
        assert get_dynamic_schema_core() is None
        assert get_dynamic_schema_admin() is None

    @patch("tools.discord_tool._discord_request")
    def test_backward_compat_wrapper(self, mock_req, monkeypatch):
        """get_dynamic_schema() should delegate to get_dynamic_schema_core()."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.return_value = {"flags": (1 << 14) | (1 << 18)}
        schema = get_dynamic_schema()
        assert schema is not None
        assert schema["name"] == "discord"
        actions = set(schema["parameters"]["properties"]["action"]["enum"])
        assert actions == set(_CORE_ACTIONS.keys())


# ---------------------------------------------------------------------------
# Runtime allowlist enforcement (defense in depth — schema already filtered)
# ---------------------------------------------------------------------------

class TestRuntimeAllowlistEnforcement:
    @patch("tools.discord_tool._discord_request")
    def test_denied_action_blocked_at_runtime(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds"}},
        )
        result = json.loads(discord_admin_handler(action="add_role", guild_id="1", user_id="2", role_id="3"))
        assert "error" in result
        assert "disabled by config" in result["error"]
        mock_req.assert_not_called()

    @patch("tools.discord_tool._discord_request")
    def test_allowed_action_proceeds(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds"}},
        )
        mock_req.return_value = []
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert "guilds" in result


# ---------------------------------------------------------------------------
# 403 enrichment
# ---------------------------------------------------------------------------

class Test403Enrichment:
    def test_enrich_known_action(self):
        msg = _enrich_403("add_role", '{"message":"Missing Permissions"}')
        assert "MANAGE_ROLES" in msg
        assert "Missing Permissions" in msg  # Raw body preserved

    def test_enrich_unknown_action_includes_body(self):
        msg = _enrich_403("some_new_action", '{"message":"weird"}')
        assert "some_new_action" in msg
        assert "weird" in msg

    @patch("tools.discord_tool._discord_request")
    def test_403_in_runtime_is_enriched(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.side_effect = DiscordAPIError(403, '{"message":"Missing Permissions"}')
        result = json.loads(discord_admin_handler(
            action="add_role", guild_id="1", user_id="2", role_id="3",
        ))
        assert "error" in result
        assert "MANAGE_ROLES" in result["error"]

    @patch("tools.discord_tool._discord_request")
    def test_non_403_errors_are_not_enriched(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.side_effect = DiscordAPIError(500, "server error")
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert "500" in result["error"]
        assert "MANAGE_ROLES" not in result["error"]


# ---------------------------------------------------------------------------
# model_tools integration — dynamic schema replaces static
# ---------------------------------------------------------------------------

class TestModelToolsIntegration:
    def setup_method(self):
        _reset_capability_cache()

    def teardown_method(self):
        _reset_capability_cache()

    @patch("tools.discord_tool._discord_request")
    def test_discord_admin_schema_rebuilt_by_get_tool_definitions(
        self, mock_req, monkeypatch,
    ):
        """When model_tools.get_tool_definitions runs with discord_admin
        available, it should replace the static schema with the dynamic one."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds,server_info"}},
        )
        # Bot without GUILD_MEMBERS intent
        mock_req.return_value = {"flags": 0}

        from model_tools import get_tool_definitions
        tools = get_tool_definitions(enabled_toolsets=["hermes-discord"], quiet_mode=True)
        discord_admin_tool = next(
            (t for t in tools if t.get("function", {}).get("name") == "discord_admin"),
            None,
        )
        assert discord_admin_tool is not None, "discord_admin should be in the schema"
        actions = discord_admin_tool["function"]["parameters"]["properties"]["action"]["enum"]
        assert actions == ["list_guilds", "server_info"]

    @patch("tools.discord_tool._discord_request")
    def test_discord_tools_dropped_when_allowlist_empties_them(
        self, mock_req, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "all_bogus_names"}},
        )
        mock_req.return_value = {"flags": 0}

        from model_tools import get_tool_definitions
        tools = get_tool_definitions(enabled_toolsets=["hermes-discord"], quiet_mode=True)
        names = [t.get("function", {}).get("name") for t in tools]
        assert "discord" not in names
        assert "discord_admin" not in names
        assert "discord_server" not in names

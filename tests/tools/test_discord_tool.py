"""Tests for the Discord server introspection and management tool."""

import json
import os
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from tools.discord_tool import (
    DiscordAPIError,
    _channel_type_name,
    _discord_request,
    _get_bot_token,
    check_discord_tool_requirements,
    discord_server,
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
        result = json.loads(discord_server(action="list_guilds"))
        assert "error" in result
        assert "DISCORD_BOT_TOKEN" in result["error"]

    def test_unknown_action(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_server(action="bad_action"))
        assert "error" in result
        assert "Unknown action" in result["error"]
        assert "available_actions" in result

    def test_missing_required_guild_id(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_server(action="list_channels"))
        assert "error" in result
        assert "guild_id" in result["error"]

    def test_missing_required_channel_id(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_server(action="fetch_messages"))
        assert "error" in result
        assert "channel_id" in result["error"]

    def test_missing_multiple_params(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_server(action="add_role"))
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
        result = json.loads(discord_server(action="list_guilds"))
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
        result = json.loads(discord_server(action="server_info", guild_id="111"))
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
        result = json.loads(discord_server(action="list_channels", guild_id="111"))
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
        result = json.loads(discord_server(action="list_channels", guild_id="111"))
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
        result = json.loads(discord_server(action="channel_info", channel_id="11"))
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
        result = json.loads(discord_server(action="list_roles", guild_id="111"))
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
        result = json.loads(discord_server(action="member_info", guild_id="111", user_id="42"))
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
        result = json.loads(discord_server(action="search_members", guild_id="111", query="test"))
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
        discord_server(action="search_members", guild_id="111", query="x", limit=200)
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
        result = json.loads(discord_server(action="fetch_messages", channel_id="11"))
        assert result["count"] == 1
        assert result["messages"][0]["content"] == "Hello world"
        assert result["messages"][0]["author"]["username"] == "user1"

    @patch("tools.discord_tool._discord_request")
    def test_fetch_messages_with_pagination(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = []
        discord_server(action="fetch_messages", channel_id="11", before="999", limit=10)
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
        result = json.loads(discord_server(action="list_pins", channel_id="11"))
        assert result["count"] == 1
        assert result["pinned_messages"][0]["content"] == "Important announcement"


# ---------------------------------------------------------------------------
# Actions: pin_message / unpin_message
# ---------------------------------------------------------------------------

class TestPinUnpin:
    @patch("tools.discord_tool._discord_request")
    def test_pin_message(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = None  # 204
        result = json.loads(discord_server(action="pin_message", channel_id="11", message_id="500"))
        assert result["success"] is True
        mock_req.assert_called_once_with("PUT", "/channels/11/pins/500", "test-token")

    @patch("tools.discord_tool._discord_request")
    def test_unpin_message(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = None
        result = json.loads(discord_server(action="unpin_message", channel_id="11", message_id="500"))
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Action: create_thread
# ---------------------------------------------------------------------------

class TestCreateThread:
    @patch("tools.discord_tool._discord_request")
    def test_create_standalone_thread(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {"id": "800", "name": "New Thread"}
        result = json.loads(discord_server(action="create_thread", channel_id="11", name="New Thread"))
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
        result = json.loads(discord_server(
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
        result = json.loads(discord_server(
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
        result = json.loads(discord_server(
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
        result = json.loads(discord_server(action="list_guilds"))
        assert "error" in result
        assert "403" in result["error"]

    @patch("tools.discord_tool._discord_request")
    def test_unexpected_error_handled(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = RuntimeError("something broke")
        result = json.loads(discord_server(action="list_guilds"))
        assert "error" in result
        assert "something broke" in result["error"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_tool_registered(self):
        from tools.registry import registry
        entry = registry._tools.get("discord_server")
        assert entry is not None
        assert entry.schema["name"] == "discord_server"
        assert entry.toolset == "discord"
        assert entry.check_fn is not None
        assert entry.requires_env == ["DISCORD_BOT_TOKEN"]

    def test_schema_actions(self):
        from tools.registry import registry
        entry = registry._tools["discord_server"]
        actions = entry.schema["parameters"]["properties"]["action"]["enum"]
        expected = [
            "list_guilds", "server_info", "list_channels", "channel_info",
            "list_roles", "member_info", "search_members", "fetch_messages",
            "list_pins", "pin_message", "unpin_message", "create_thread",
            "add_role", "remove_role",
        ]
        assert actions == expected

    def test_handler_callable(self):
        from tools.registry import registry
        entry = registry._tools["discord_server"]
        assert callable(entry.handler)


# ---------------------------------------------------------------------------
# Toolset: discord_server only in hermes-discord
# ---------------------------------------------------------------------------

class TestToolsetInclusion:
    def test_discord_server_in_hermes_discord_toolset(self):
        from toolsets import TOOLSETS
        assert "discord_server" in TOOLSETS["hermes-discord"]["tools"]

    def test_discord_server_not_in_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS
        assert "discord_server" not in _HERMES_CORE_TOOLS

    def test_discord_server_not_in_other_toolsets(self):
        from toolsets import TOOLSETS
        for name, ts in TOOLSETS.items():
            if name == "hermes-discord":
                continue
            # The gateway toolset might include it if it unions all platform tools
            if name == "hermes-gateway":
                continue
            assert "discord_server" not in ts.get("tools", []), (
                f"discord_server should not be in toolset '{name}'"
            )

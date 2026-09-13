"""Tests for the Discord server introspection and management tool."""

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, call, patch

import pytest

from tools.discord_tool import (
    DiscordAPIError,
    _ACTIONS,
    _ADMIN_ACTIONS,
    _CORE_ACTIONS,
    _available_actions,
    _channel_type_name,
    _detect_capabilities,
    _discord_request,
    _get_bot_token,
    _load_leave_guild_ids,
    _load_allowed_actions_config,
    _reset_capability_cache,
    check_discord_tool_requirements,
    discord_admin_handler,
    discord_core,
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
    @pytest.mark.parametrize("token, expected", [(None, False), ("test-token-123", True)])
    def test_requirements_follow_token(self, monkeypatch, token, expected):
        if token is None:
            monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        else:
            monkeypatch.setenv("DISCORD_BOT_TOKEN", token)
        assert check_discord_tool_requirements() is expected

    def test_get_bot_token(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "  my-token  ")
        assert _get_bot_token() == "my-token"

    def test_multiplex_scope_token_wins_over_process_environment(self, monkeypatch):
        from agent import secret_scope

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "another-profile-token")
        secret_scope.set_multiplex_active(True)
        scope_token = secret_scope.set_secret_scope(
            {"DISCORD_BOT_TOKEN": "  active-profile-token  "}
        )
        try:
            assert _get_bot_token() == "active-profile-token"
            assert check_discord_tool_requirements() is True
        finally:
            secret_scope.reset_secret_scope(scope_token)
            secret_scope.set_multiplex_active(False)

    def test_multiplex_scope_missing_token_fails_closed(self, monkeypatch):
        from agent import secret_scope
        from tools.registry import invalidate_check_fn_cache, registry

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "another-profile-token")
        secret_scope.set_multiplex_active(True)
        scope_token = secret_scope.set_secret_scope({"UNRELATED_SECRET": "value"})
        invalidate_check_fn_cache()
        try:
            assert _get_bot_token() is None
            assert check_discord_tool_requirements() is False
            assert registry.get_definitions({"discord", "discord_admin"}) == []
        finally:
            invalidate_check_fn_cache()
            secret_scope.reset_secret_scope(scope_token)
            secret_scope.set_multiplex_active(False)

    def test_non_multiplex_scope_miss_keeps_environment_compatibility(self, monkeypatch):
        from agent import secret_scope

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "process-token")
        secret_scope.set_multiplex_active(False)
        scope_token = secret_scope.set_secret_scope({"UNRELATED_SECRET": "value"})
        try:
            assert _get_bot_token() == "process-token"
            assert check_discord_tool_requirements() is True
        finally:
            secret_scope.reset_secret_scope(scope_token)

    def test_gateway_tool_prompt_gate_uses_active_profile_token(self, monkeypatch):
        from agent import secret_scope
        from gateway.session import _discord_tools_loaded

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "another-profile-token")
        secret_scope.set_multiplex_active(True)
        scope_token = secret_scope.set_secret_scope({"UNRELATED_SECRET": "value"})
        try:
            with patch("hermes_cli.config.load_config") as load_config:
                assert _discord_tools_loaded() is False
                load_config.assert_not_called()
        finally:
            secret_scope.reset_secret_scope(scope_token)
            secret_scope.set_multiplex_active(False)

# ---------------------------------------------------------------------------
# Channel type names
# ---------------------------------------------------------------------------

class TestChannelTypeNames:
    def test_type_names(self):
        assert _channel_type_name(0) == "text"
        assert _channel_type_name(2) == "voice"
        assert _channel_type_name(4) == "category"
        assert _channel_type_name(15) == "forum"
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
        assert req.get_header("Content-Type") is None
        assert req.data is None
        assert req.get_method() == "GET"

    @patch("tools.discord_tool.urllib.request.urlopen")
    def test_json_body_sets_content_type(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen({"ok": True})
        result = _discord_request("POST", "/test", "token123", body={"name": "Beta Guild"})
        assert result == {"ok": True}

        req = mock_urlopen_fn.call_args.args[0]
        assert req.data == b'{"name": "Beta Guild"}'
        assert req.get_header("Content-type") == "application/json"
        assert req.get_header("Authorization") == "Bot token123"
        assert req.get_header("User-agent") == "Hermes-Agent (https://github.com/NousResearch/hermes-agent)"


    @patch("tools.discord_tool.urllib.request.urlopen")
    def test_response_body_size_limit(self, mock_urlopen_fn, monkeypatch):
        monkeypatch.setattr("tools.discord_tool._DISCORD_RESPONSE_BODY_MAX_BYTES", 8)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"x" * 9
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen_fn.return_value = mock_resp

        with pytest.raises(DiscordAPIError) as exc_info:
            _discord_request("GET", "/test", "tok")

        assert exc_info.value.status == 502
        assert "response body exceeded 8 bytes" in exc_info.value.body
        mock_resp.read.assert_called_once_with(9)


# ---------------------------------------------------------------------------
# Main handler: validation
# ---------------------------------------------------------------------------

class TestDiscordServerValidation:
    def test_no_token(self, monkeypatch):
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert "error" in result
        assert "DISCORD_BOT_TOKEN" in result["error"]


    def test_missing_multiple_params(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_admin_handler(action="add_role"))
        assert "error" in result
        assert "guild_id" in result["error"]
        assert "user_id" in result["error"]
        assert "role_id" in result["error"]


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
# Action: search_members
# ---------------------------------------------------------------------------

class TestSearchMembers:
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


class TestCreateGuildChannels:
    @pytest.mark.parametrize(
        "action, channel_type",
        [("create_category", 4), ("create_text_channel", 0), ("create_forum", 15)],
    )
    @patch("tools.discord_tool._discord_request")
    def test_creation_uses_the_action_channel_type(
        self, mock_req, monkeypatch, action, channel_type,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = [[], {"id": "800", "name": "project"}]

        result = json.loads(discord_admin_handler(
            action=action, guild_id="111", name="project",
        ))

        assert result == {
            "guild_id": "111", "channel_id": "800", "name": "project",
            "type": _channel_type_name(channel_type), "created": True,
        }
        assert mock_req.call_args_list[0].args == (
            "GET", "/guilds/111/channels", "test-token",
        )
        assert mock_req.call_args_list[1].args == (
            "POST", "/guilds/111/channels", "test-token",
        )
        assert mock_req.call_args_list[1].kwargs["body"]["type"] == channel_type

    @patch("tools.discord_tool._discord_request")
    def test_exact_existing_channel_is_returned_without_post(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"id": "700", "name": "Projects", "type": 4, "parent_id": None},
            {"id": "800", "name": "project", "type": 15, "parent_id": "700"},
        ]

        result = json.loads(discord_admin_handler(
            action="create_forum", guild_id="111", name="project", category_id="700",
        ))

        assert result["channel_id"] == "800"
        assert result["created"] is False
        mock_req.assert_called_once_with("GET", "/guilds/111/channels", "test-token")

    @pytest.mark.parametrize(
        "channels",
        [
            [],
            [{"id": "700", "name": "not-a-category", "type": 0, "parent_id": None}],
        ],
    )
    @patch("tools.discord_tool._discord_request")
    def test_creation_rejects_invalid_category_before_post(
        self, mock_req, monkeypatch, channels,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = channels

        result = json.loads(discord_admin_handler(
            action="create_text_channel", guild_id="111", name="project", category_id="700",
        ))

        assert "error" in result
        assert "category_id" in result["error"]
        mock_req.assert_called_once_with("GET", "/guilds/111/channels", "test-token")

    @patch("tools.discord_tool._discord_request")
    def test_same_name_incompatible_channel_fails_without_post(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"id": "800", "name": "project", "type": 0, "parent_id": None},
        ]
        result = json.loads(discord_admin_handler(
            action="create_forum", guild_id="111", name="project",
        ))
        assert "error" in result
        assert "different type or category" in result["error"]
        mock_req.assert_called_once_with("GET", "/guilds/111/channels", "test-token")

    @patch("tools.discord_tool._discord_request")
    def test_same_name_channel_with_requested_field_mismatch_is_not_compatible(
        self, mock_req, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {
                "id": "800", "name": "project", "type": 15, "parent_id": None,
                "topic": "old",
            },
        ]
        result = json.loads(discord_admin_handler(
            action="create_forum", guild_id="111", name="project", topic="new",
        ))
        assert "error" in result
        assert "requested settings" in result["error"]
        mock_req.assert_called_once_with("GET", "/guilds/111/channels", "test-token")

    @patch("tools.discord_tool._discord_request")
    def test_optional_fields_and_zero_overwrite_values_are_preserved(
        self, mock_req, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = [
            [{"id": "700", "name": "Projects", "type": 4, "parent_id": None}],
            {"id": "800", "name": "project"},
        ]
        overwrites = [{"id": "222", "type": 0, "allow": 0, "deny": 0}]

        result = json.loads(discord_admin_handler(
            action="create_forum", guild_id="111", name="project", category_id="700",
            topic="Work", position=0, nsfw=False, rate_limit_per_user=0,
            permission_overwrites=overwrites,
        ))

        assert result["created"] is True
        assert mock_req.call_args_list[1].kwargs["body"] == {
            "name": "project", "type": 15, "parent_id": "700", "topic": "Work",
            "position": 0, "nsfw": False, "rate_limit_per_user": 0,
            "permission_overwrites": [{"id": "222", "type": 0, "allow": "0", "deny": "0"}],
        }

    @pytest.mark.parametrize(
        "overwrites",
        [
            "not-an-array",
            [{"id": "222", "type": 2, "allow": 0, "deny": 0}],
            [{"id": "bad", "type": 0, "allow": 0, "deny": 0}],
            [{"id": "222", "type": 0, "allow": -1, "deny": 0}],
            [{"id": "222", "type": 0, "allow": None, "deny": 0}],
            [
                {"id": "222", "type": 0, "allow": 0, "deny": 0},
                {"id": "222", "type": 0, "allow": 0, "deny": 0},
            ],
            [{"id": str(index + 1), "type": 0, "allow": 0, "deny": 0} for index in range(101)],
        ],
    )
    @patch("tools.discord_tool._discord_request")
    def test_malformed_overwrites_fail_before_http(
        self, mock_req, monkeypatch, overwrites,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_admin_handler(
            action="create_text_channel", guild_id="111", name="project",
            permission_overwrites=overwrites,
        ))
        assert "error" in result
        assert "permission_overwrites" in result["error"]
        mock_req.assert_not_called()

    @patch("tools.discord_tool._discord_request")
    def test_grant_main_user_adds_only_view_and_manage_overwrites(
        self, mock_req, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        monkeypatch.setenv("DISCORD_MAIN_USER_ID", "222")
        mock_req.side_effect = [[], {"id": "333"}, {"id": "800", "name": "project"}]

        result = json.loads(discord_admin_handler(
            action="create_text_channel", guild_id="111", name="project",
            grant_main_user_access=True,
        ))

        assert result["created"] is True
        assert mock_req.call_args_list[1].args == ("GET", "/users/@me", "test-token")
        assert mock_req.call_args_list[2].kwargs["body"]["permission_overwrites"] == [
            {"id": "222", "type": 1, "allow": "1040", "deny": "0"},
            {"id": "333", "type": 1, "allow": "1040", "deny": "0"},
        ]

    @pytest.mark.parametrize(
        "main_id, allowed",
        [(None, ""), (None, "*"), (None, "222,333"), ("not-numeric", "222")],
    )
    @patch("tools.discord_tool._discord_request")
    def test_grant_main_user_fails_closed_before_http(
        self, mock_req, monkeypatch, main_id, allowed,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        if main_id is None:
            monkeypatch.delenv("DISCORD_MAIN_USER_ID", raising=False)
        else:
            monkeypatch.setenv("DISCORD_MAIN_USER_ID", main_id)
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", allowed)

        result = json.loads(discord_admin_handler(
            action="create_text_channel", guild_id="111", name="project",
            grant_main_user_access=True,
        ))

        assert "error" in result
        assert "DISCORD_MAIN_USER_ID" in result["error"]
        mock_req.assert_not_called()

    @pytest.mark.parametrize(
        "action, params",
        [
            ("create_text_channel", {"guild_id": "bad", "name": "x"}),
            ("create_text_channel", {"guild_id": "111", "name": ""}),
            ("create_text_channel", {"guild_id": "111", "name": "x", "topic": "x" * 4097}),
            ("create_text_channel", {"guild_id": "111", "name": "x", "topic": "x" * 1025}),
            ("create_text_channel", {"guild_id": "111", "name": "x", "position": -1}),
            ("create_text_channel", {"guild_id": "111", "name": "x", "nsfw": 1}),
            ("create_text_channel", {"guild_id": "111", "name": "x", "rate_limit_per_user": 21601}),
            ("create_category", {"guild_id": "111", "name": "x", "topic": "unsupported"}),
            ("create_category", {"guild_id": "111", "name": "x", "grant_main_user_access": True}),
            ("create_role", {"guild_id": "111", "name": "x", "color": 0x1000000}),
            ("create_role", {"guild_id": "111", "name": "x", "hoist": "false"}),
        ],
    )
    @patch("tools.discord_tool._discord_request")
    def test_creation_boundaries_fail_before_http(self, mock_req, monkeypatch, action, params):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_admin_handler(action=action, **params))
        assert "error" in result
        mock_req.assert_not_called()


class TestEditGuildChannels:
    @pytest.mark.parametrize(
        "action, params, body",
        [
            ("rename_channel", {"name": "renamed"}, {"name": "renamed"}),
            ("set_channel_topic", {"topic": "New topic"}, {"topic": "New topic"}),
            (
                "move_channel_to_category", {"category_id": "700", "position": 0},
                {"parent_id": "700", "position": 0},
            ),
        ],
    )
    @patch("tools.discord_tool._discord_request")
    def test_channel_edits_patch_only_requested_fields(
        self, mock_req, monkeypatch, action, params, body,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        if action == "set_channel_topic":
            mock_req.side_effect = [{"id": "800", "type": 0}, {"id": "800"}]
        elif action == "move_channel_to_category":
            mock_req.side_effect = [
                {"id": "800", "type": 0, "guild_id": "111"},
                {"id": "700", "type": 4, "guild_id": "111"},
                {"id": "800"},
            ]
        else:
            mock_req.return_value = {"id": "800"}

        result = json.loads(discord_admin_handler(
            action=action, channel_id="800", **params,
        ))

        assert result == {"channel_id": "800", "updated": True}
        expected_calls = [call("PATCH", "/channels/800", "test-token", body=body)]
        if action == "set_channel_topic":
            expected_calls.insert(0, call("GET", "/channels/800", "test-token"))
        elif action == "move_channel_to_category":
            expected_calls[:0] = [
                call("GET", "/channels/800", "test-token"),
                call("GET", "/channels/700", "test-token"),
            ]
        assert mock_req.call_args_list == expected_calls

    @pytest.mark.parametrize(
        "destination",
        [
            {"id": "700", "type": 0, "guild_id": "111"},
            {"id": "700", "type": 4, "guild_id": "999"},
        ],
    )
    @patch("tools.discord_tool._discord_request")
    def test_move_rejects_invalid_category_before_patch(
        self, mock_req, monkeypatch, destination,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = [
            {"id": "800", "type": 0, "guild_id": "111"},
            destination,
        ]

        result = json.loads(discord_admin_handler(
            action="move_channel_to_category", channel_id="800", category_id="700",
        ))

        assert "error" in result
        assert "category_id" in result["error"]
        assert mock_req.call_args_list == [
            call("GET", "/channels/800", "test-token"),
            call("GET", "/channels/700", "test-token"),
        ]

    @patch("tools.discord_tool._discord_request")
    def test_move_rejects_unsupported_source_type_before_patch(
        self, mock_req, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {"id": "800", "type": 4, "guild_id": "111"}

        result = json.loads(discord_admin_handler(
            action="move_channel_to_category", channel_id="800", category_id="700",
        ))

        assert "error" in result
        assert "channel_id" in result["error"]
        mock_req.assert_called_once_with("GET", "/channels/800", "test-token")

    @pytest.mark.parametrize(
        "action, expected_allow, expected_deny, locked",
        [("lock_channel", "16", "2049", True), ("unlock_channel", "2064", "1", False)],
    )
    @patch("tools.discord_tool._discord_request")
    def test_locking_changes_only_everyone_send_messages_bit(
        self, mock_req, monkeypatch, action, expected_allow, expected_deny, locked,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = [
            {
                "id": "800", "guild_id": "111", "type": 0,
                "permission_overwrites": [
                    {"id": "111", "type": 0, "allow": "2064", "deny": "1"},
                    {"id": "222", "type": 1, "allow": "7", "deny": "8"},
                ],
            },
            {"id": "800"},
        ]

        result = json.loads(discord_admin_handler(
            action=action, channel_id="800", guild_id="111",
        ))

        assert result == {"channel_id": "800", "guild_id": "111", "locked": locked}
        assert mock_req.call_args_list[1].args == (
            "PUT", "/channels/800/permissions/111", "test-token",
        )
        assert mock_req.call_args_list[1].kwargs["body"] == {
            "type": 0, "allow": expected_allow, "deny": expected_deny,
        }

    @patch("tools.discord_tool._discord_request")
    def test_lock_refuses_channel_from_another_guild(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {"id": "800", "guild_id": "999", "permission_overwrites": []}

        result = json.loads(discord_admin_handler(
            action="lock_channel", channel_id="800", guild_id="111",
        ))

        assert "error" in result
        assert "guild_id" in result["error"]
        mock_req.assert_called_once_with("GET", "/channels/800", "test-token")

    @pytest.mark.parametrize("channel_type", [2, 4, 13])
    @patch("tools.discord_tool._discord_request")
    def test_lock_rejects_unsupported_channel_type_before_put(
        self, mock_req, monkeypatch, channel_type,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {
            "id": "800", "guild_id": "111", "type": channel_type,
            "permission_overwrites": [],
        }

        result = json.loads(discord_admin_handler(
            action="lock_channel", channel_id="800", guild_id="111",
        ))

        assert "error" in result
        assert "channel_id" in result["error"]
        mock_req.assert_called_once_with("GET", "/channels/800", "test-token")

    @patch("tools.discord_tool._discord_request")
    def test_set_topic_rejects_non_forum_limit_before_http(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {"id": "800", "type": 0}
        result = json.loads(discord_admin_handler(
            action="set_channel_topic", channel_id="800", topic="x" * 1025,
        ))
        assert "error" in result
        assert "1024" in result["error"]
        mock_req.assert_called_once_with("GET", "/channels/800", "test-token")

    @patch("tools.discord_tool._discord_request")
    def test_set_topic_allows_forum_limit(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = [{"id": "800", "type": 15}, {"id": "800"}]
        result = json.loads(discord_admin_handler(
            action="set_channel_topic", channel_id="800", topic="x" * 4096,
        ))
        assert result == {"channel_id": "800", "updated": True}
        assert mock_req.call_args_list == [
            call("GET", "/channels/800", "test-token"),
            call("PATCH", "/channels/800", "test-token", body={"topic": "x" * 4096}),
        ]


class TestManageRoles:
    @patch("tools.discord_tool._discord_request")
    def test_create_role_is_idempotent_and_preserves_zero_color(
        self, mock_req, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = [[], {"id": "900", "name": "Editors"}]

        result = json.loads(discord_admin_handler(
            action="create_role", guild_id="111", name="Editors", color=0,
            hoist=False, mentionable=True,
        ))

        assert result == {
            "guild_id": "111", "role_id": "900", "name": "Editors", "created": True,
        }
        assert mock_req.call_args_list[0].args == ("GET", "/guilds/111/roles", "test-token")
        assert mock_req.call_args_list[1].args == ("POST", "/guilds/111/roles", "test-token")
        assert mock_req.call_args_list[1].kwargs["body"] == {
            "name": "Editors", "color": 0, "hoist": False, "mentionable": True,
        }

    @patch("tools.discord_tool._discord_request")
    def test_existing_unmanaged_role_is_returned_without_post(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"id": "900", "name": "Editors", "managed": False, "permissions": "0"},
        ]
        result = json.loads(discord_admin_handler(
            action="create_role", guild_id="111", name="Editors",
        ))
        assert result["created"] is False
        assert result["role_id"] == "900"
        mock_req.assert_called_once_with("GET", "/guilds/111/roles", "test-token")

    @patch("tools.discord_tool._discord_request")
    def test_existing_privileged_role_is_an_incompatible_collision(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"id": "900", "name": "Editors", "managed": False, "permissions": "8"},
        ]
        result = json.loads(discord_admin_handler(
            action="create_role", guild_id="111", name="Editors",
        ))
        assert "error" in result
        assert "permissions" in result["error"]
        mock_req.assert_called_once_with("GET", "/guilds/111/roles", "test-token")

    @patch("tools.discord_tool._discord_request")
    def test_update_role_rejects_empty_payload_before_http(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_admin_handler(
            action="update_role", guild_id="111", role_id="900",
        ))
        assert "error" in result
        assert "at least one" in result["error"]
        mock_req.assert_not_called()

    @patch("tools.discord_tool._discord_request")
    def test_update_role_patches_only_supplied_fields(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {"id": "900", "name": "Editors"}
        result = json.loads(discord_admin_handler(
            action="update_role", guild_id="111", role_id="900", color=0,
        ))
        assert result == {"guild_id": "111", "role_id": "900", "updated": True}
        mock_req.assert_called_once_with(
            "PATCH", "/guilds/111/roles/900", "test-token", body={"color": 0},
        )

    def test_role_schema_does_not_expose_permissions_or_administrator(self):
        schema = get_dynamic_schema_admin
        properties = __import__("tools.discord_tool", fromlist=["_SCHEMA_PROPERTIES"])._SCHEMA_PROPERTIES
        assert "permissions" not in properties
        assert "administrator" not in properties
        assert callable(schema)


class TestLeaveGuild:
    @pytest.fixture(autouse=True)
    def _leave_config(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"leave_guild_ids": ["900000000000000001", "900000000000000002"]}},
        )

    @patch("tools.discord_tool._discord_request")
    def test_leave_guild_requires_approved_target_and_exact_name(
        self, mock_req, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [{"id": "900000000000000001", "name": "Alpha Guild"}]

        result = json.loads(discord_admin_handler(
            action="leave_guild", guild_id="900000000000000001",
            expected_guild_name="Wrong",
        ))

        assert "error" in result
        assert "Alpha Guild" not in result["error"]
        assert "expected_guild_name" in result["error"]
        mock_req.assert_called_once_with(
            "GET", "/users/@me/guilds", "test-token", params={"limit": "200"},
        )

    @patch("tools.discord_tool._discord_request")
    def test_leave_guild_is_idempotent_when_approved_target_is_absent(
        self, mock_req, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = []
        result = json.loads(discord_admin_handler(
            action="leave_guild", guild_id="900000000000000002",
            expected_guild_name="Beta Guild",
        ))
        assert result == {
            "guild_id": "900000000000000002", "name": "Beta Guild", "left": False,
        }
        mock_req.assert_called_once_with(
            "GET", "/users/@me/guilds", "test-token", params={"limit": "200"},
        )

    @patch("tools.discord_tool._discord_request")
    def test_leave_guild_deletes_only_after_exact_identity_match(
        self, mock_req, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = [
            [{"id": "900000000000000002", "name": "Beta Guild"}], None,
        ]
        result = json.loads(discord_admin_handler(
            action="leave_guild", guild_id="900000000000000002",
            expected_guild_name="Beta Guild",
        ))
        assert result == {
            "guild_id": "900000000000000002", "name": "Beta Guild", "left": True,
        }
        assert mock_req.call_args_list[1].args == (
            "DELETE", "/users/@me/guilds/900000000000000002", "test-token",
        )

    @patch("tools.discord_tool.time.sleep")
    @patch("tools.discord_tool._discord_request")
    def test_leave_guild_retries_delete_rate_limit_without_reinventory(
        self, mock_req, mock_sleep, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = [
            [{"id": "900000000000000002", "name": "Beta Guild"}],
            DiscordAPIError(429, '{"message":"You are being rate limited.","retry_after":0.5}'),
            None,
        ]

        result = json.loads(discord_admin_handler(
            action="leave_guild", guild_id="900000000000000002",
            expected_guild_name="Beta Guild",
        ))

        assert result == {
            "guild_id": "900000000000000002", "name": "Beta Guild", "left": True,
        }
        assert [item.args[0] for item in mock_req.call_args_list] == ["GET", "DELETE", "DELETE"]
        mock_sleep.assert_called_once_with(0.5)

    @pytest.mark.parametrize("body", ["not-json", '{"retry_after": "later"}', '{"retry_after": 5.1}'])
    @patch("tools.discord_tool.time.sleep")
    @patch("tools.discord_tool._discord_request")
    def test_leave_guild_rejects_invalid_or_excessive_rate_limit_data(
        self, mock_req, mock_sleep, monkeypatch, body,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = [
            [{"id": "900000000000000002", "name": "Beta Guild"}],
            DiscordAPIError(429, body),
        ]

        result = json.loads(discord_admin_handler(
            action="leave_guild", guild_id="900000000000000002",
            expected_guild_name="Beta Guild",
        ))

        assert "error" in result
        assert "retry" in result["error"].lower()
        assert body not in result["error"]
        mock_sleep.assert_not_called()
        assert mock_req.call_count == 2

    @patch("tools.discord_tool.time.sleep")
    @patch("tools.discord_tool._discord_request")
    def test_leave_guild_stops_after_one_rate_limit_retry(
        self, mock_req, mock_sleep, monkeypatch,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        rate_limit = DiscordAPIError(429, '{"retry_after":0}')
        mock_req.side_effect = [
            [{"id": "900000000000000002", "name": "Beta Guild"}], rate_limit, rate_limit,
        ]

        result = json.loads(discord_admin_handler(
            action="leave_guild", guild_id="900000000000000002",
            expected_guild_name="Beta Guild",
        ))

        assert "error" in result
        assert "429" in result["error"]
        assert mock_req.call_count == 3
        mock_sleep.assert_called_once_with(0.0)

    @pytest.mark.parametrize(
        "guild_id,name",
        [
            ("900000000000000003", "Gamma Guild"),
            ("999", "Other"),
            ("*", "Beta Guild"),
        ],
    )
    @patch("tools.discord_tool._discord_request")
    def test_leave_guild_rejects_every_other_target_before_http(
        self, mock_req, monkeypatch, guild_id, name,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_admin_handler(
            action="leave_guild", guild_id=guild_id, expected_guild_name=name,
        ))
        assert "error" in result
        mock_req.assert_not_called()

    def test_new_surface_has_no_other_destructive_administration(self):
        import tools.discord_tool as discord_tool

        assert discord_tool._NEW_DESTRUCTIVE_ACTIONS == frozenset({"leave_guild"})
        assert not ({"delete_channel", "delete_role", "ban", "kick", "purge", "webhook"} & _ACTIONS.keys())


class TestExpandedAdminContract:
    def test_original_actions_and_new_actions_are_disjoint_and_complete(self):
        import tools.discord_tool as discord_tool

        original = {
            "list_guilds", "server_info", "list_channels", "channel_info", "list_roles",
            "member_info", "search_members", "fetch_messages", "list_pins", "pin_message",
            "unpin_message", "delete_message", "create_thread", "add_role", "remove_role",
        }
        expected_new = {
            "create_category", "create_text_channel", "create_forum", "rename_channel",
            "set_channel_topic", "move_channel_to_category", "lock_channel", "unlock_channel",
            "create_role", "update_role", "leave_guild",
        }
        assert expected_new <= discord_tool._NEW_ACTION_NAMES
        assert original | expected_new <= set(_ACTIONS)
        assert original.isdisjoint(expected_new)

    def test_manifest_declares_required_parameters_and_schema_fields(self):
        import tools.discord_tool as discord_tool

        expected = {
            "create_category": ["guild_id", "name"],
            "create_text_channel": ["guild_id", "name"],
            "create_forum": ["guild_id", "name"],
            "rename_channel": ["channel_id", "name"],
            "set_channel_topic": ["channel_id", "topic"],
            "move_channel_to_category": ["channel_id", "category_id"],
            "lock_channel": ["channel_id", "guild_id"],
            "unlock_channel": ["channel_id", "guild_id"],
            "create_role": ["guild_id", "name"],
            "update_role": ["guild_id", "role_id"],
            "leave_guild": ["guild_id", "expected_guild_name"],
        }
        assert {name: discord_tool._REQUIRED_PARAMS[name] for name in expected} == expected
        schema = discord_tool._build_schema(list(discord_tool._ADMIN_ACTIONS), tool_name="discord_admin")
        assert schema is not None
        assert expected.keys() <= set(schema["parameters"]["properties"]["action"]["enum"])
        for field in {field for fields in expected.values() for field in fields}:
            assert field in schema["parameters"]["properties"]

    def test_admin_only_parameters_do_not_expand_the_core_discord_schema(self):
        import tools.discord_tool as discord_tool

        core = discord_tool._build_schema(list(_CORE_ACTIONS), tool_name="discord")
        admin = discord_tool._build_schema(list(_ADMIN_ACTIONS), tool_name="discord_admin")
        assert core is not None and admin is not None
        admin_only = {
            "category_id", "topic", "position", "nsfw", "rate_limit_per_user",
            "permission_overwrites", "grant_main_user_access", "color", "hoist",
            "mentionable", "expected_guild_name",
        }
        assert not (admin_only & core["parameters"]["properties"].keys())
        assert admin_only <= admin["parameters"]["properties"].keys()

    @patch("tools.discord_tool._discord_request")
    def test_new_action_api_error_is_actionable_and_redacts_token(
        self, mock_req, monkeypatch, caplog,
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "secret-token")
        mock_req.side_effect = DiscordAPIError(500, "failed secret-token traceback")

        result = json.loads(discord_admin_handler(
            action="create_text_channel", guild_id="111", name="project",
        ))

        assert "error" in result
        assert "retry" in result["error"].lower()
        assert "secret-token" not in result["error"]
        assert "secret-token" not in caplog.text


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

    def test_all_actions_covered(self):
        """Core + admin actions should cover all known actions."""
        assert set(_CORE_ACTIONS.keys()) | set(_ADMIN_ACTIONS.keys()) == set(_ACTIONS.keys())
        assert set(_CORE_ACTIONS.keys()) & set(_ADMIN_ACTIONS.keys()) == set()


# ---------------------------------------------------------------------------
# Toolset: discord / discord_admin only in hermes-discord
# ---------------------------------------------------------------------------

class TestToolsetInclusion:
    def test_discord_tools_only_in_hermes_discord_toolset(self):
        from toolsets import TOOLSETS, _HERMES_CORE_TOOLS
        assert "discord" in TOOLSETS["hermes-discord"]["tools"]
        assert "discord_admin" in TOOLSETS["hermes-discord"]["tools"]
        assert "discord" not in _HERMES_CORE_TOOLS
        assert "discord_admin" not in _HERMES_CORE_TOOLS

    def test_discord_tools_not_in_other_toolsets(self):
        from toolsets import TOOLSETS
        for name, ts in TOOLSETS.items():
            if name in {"hermes-discord", "hermes-gateway", "discord", "discord_admin"}:
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
    def test_detection_failure_is_permissive(self, mock_req):
        """If detection fails (network/401/revoked token), expose everything
        and let runtime errors surface. Silent failure should never hide
        actions the bot actually has."""
        mock_req.side_effect = DiscordAPIError(401, "unauthorized")
        caps = _detect_capabilities("tok")
        assert caps["detected"] is False
        assert caps["has_members_intent"] is True
        assert caps["has_message_content"] is True


class TestNonBlockingCapabilityDetection:
    """The schema-build path must never block on a discord.com HTTP call.

    ``_detect_capabilities_nonblocking`` resolves memory cache → disk cache →
    permissive default (+ background detect), keeping the ~2s blocking
    detection off the first-token critical path (TTFT fix, July 2026).
    """

    def setup_method(self):
        _reset_capability_cache()

    def teardown_method(self):
        _reset_capability_cache()

    def test_memory_cache_hit_no_network(self):
        from tools.discord_tool import _capability_cache, _detect_capabilities_nonblocking
        caps_in = {"has_members_intent": False, "has_message_content": True, "detected": True}
        _capability_cache["tok"] = caps_in
        with patch("tools.discord_tool._discord_request") as mock_req:
            caps = _detect_capabilities_nonblocking("tok")
        assert caps == caps_in
        mock_req.assert_not_called()


    def test_disk_cache_expires(self, tmp_path, monkeypatch):
        import time as _time

        import tools.discord_tool as dt
        monkeypatch.setattr(
            dt, "_capability_disk_cache_path",
            lambda: tmp_path / "discord_capabilities.json",
        )
        caps_in = {"has_members_intent": True, "has_message_content": True, "detected": True}
        dt._save_caps_to_disk("tok", caps_in)
        # Rewrite timestamp to be stale
        import json as _json
        p = tmp_path / "discord_capabilities.json"
        data = _json.loads(p.read_text())
        for entry in data.values():
            entry["ts"] = _time.time() - dt._CAPABILITY_DISK_TTL_SECONDS - 10
        p.write_text(_json.dumps(data))
        assert dt._load_caps_from_disk("tok") is None

    def test_schema_build_uses_nonblocking_path(self, monkeypatch):
        """get_dynamic_schema_core must not call the blocking detection."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        with patch("tools.discord_tool._load_caps_from_disk", return_value=None), \
             patch("tools.discord_tool.threading.Thread") as mock_thread, \
             patch("tools.discord_tool._discord_request") as mock_req:
            schema = get_dynamic_schema_core()
        # No blocking HTTP call happened on the schema-build path
        mock_req.assert_not_called()
        # Background detection was scheduled exactly once
        assert mock_thread.call_count == 1
        assert schema is not None
        actions = set(schema["parameters"]["properties"]["action"]["enum"])
        assert actions == set(_CORE_ACTIONS.keys())  # permissive default

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

    def test_comma_separated_string(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds,list_channels,fetch_messages"}},
        )
        result = _load_allowed_actions_config()
        assert result == ["list_guilds", "list_channels", "fetch_messages"]

    def test_leave_guild_ids_missing_defaults_to_empty(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"discord": {}})
        assert _load_leave_guild_ids() == frozenset()

    @pytest.mark.parametrize("raw", [None, "900000000000000001", ("900000000000000001",), [1]])
    def test_leave_guild_ids_rejects_malformed_values(self, monkeypatch, raw):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"leave_guild_ids": raw}},
        )
        with pytest.raises(ValueError, match="discord.leave_guild_ids"):
            _load_leave_guild_ids()

    def test_leave_guild_ids_rejects_excessive_list(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"leave_guild_ids": [str(i + 1) for i in range(101)]}},
        )
        with pytest.raises(ValueError, match="at most"):
            _load_leave_guild_ids()

    def test_leave_guild_ids_accepts_numeric_string_list(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"leave_guild_ids": ["900000000000000001"]}},
        )
        assert _load_leave_guild_ids() == frozenset({"900000000000000001"})


    def test_config_load_failure_is_permissive(self, monkeypatch):
        """If config can't be loaded at all, fall back to None (all allowed)."""
        def bad_load():
            raise RuntimeError("disk gone")
        monkeypatch.setattr("hermes_cli.config.load_config", bad_load)
        assert _load_allowed_actions_config() is None


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

    def test_allowlist_intersects_with_intents(self):
        """Allowlist can only narrow — not re-enable intent-gated actions."""
        caps = {"detected": True, "has_members_intent": False, "has_message_content": True}
        allowlist = ["list_guilds", "search_members", "fetch_messages"]
        actions = _available_actions(caps, allowlist)
        # search_members gated by intent → stripped even though allowlisted;
        # result stays in canonical order regardless of allowlist order
        assert actions == ["list_guilds", "fetch_messages"]


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
        # Warm the capability cache — schema builds are non-blocking and use
        # the permissive default until detection has completed (background
        # thread + disk cache); filtering applies once caps are known.
        _detect_capabilities("tok")
        schema = get_dynamic_schema_core()
        actions = schema["parameters"]["properties"]["action"]["enum"]
        assert "search_members" not in actions

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
        assert "Missing Permissions" in result["error"]  # Raw body preserved

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
        from model_tools import _clear_tool_defs_cache
        from tools.registry import invalidate_check_fn_cache
        _clear_tool_defs_cache()
        invalidate_check_fn_cache()

    def teardown_method(self):
        _reset_capability_cache()
        from model_tools import _clear_tool_defs_cache
        from tools.registry import invalidate_check_fn_cache
        _clear_tool_defs_cache()
        invalidate_check_fn_cache()

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
        # skip_tool_search_assembly: this test exercises the dynamic schema
        # rebuild; under tiered disclosure the discord tools defer behind the
        # bridge, but the rebuilt schema is what tool_describe serves.
        tools = get_tool_definitions(enabled_toolsets=["hermes-discord"], quiet_mode=True,
                                     skip_tool_search_assembly=True)
        discord_admin_tool = next(
            (t for t in tools if t.get("function", {}).get("name") == "discord_admin"),
            None,
        )
        assert discord_admin_tool is not None, "discord_admin should be in the schema"
        actions = discord_admin_tool["function"]["parameters"]["properties"]["action"]["enum"]
        assert actions == ["list_guilds", "server_info"]

"""Tests for tools/blooio_reaction_tool.py."""

import json
import os
from unittest.mock import MagicMock, patch

from model_tools import get_tool_definitions
from tools.blooio_reaction_tool import _blooio_reaction_available, react_to_message_tool


class TestBlooioReactionAvailability:
    def test_available_only_in_blooio_session(self):
        with patch.dict(
            os.environ,
            {
                "HERMES_SESSION_PLATFORM": "blooio",
                "HERMES_SESSION_CHAT_ID": "+15551234567",
                "BLOOIO_API_KEY": "test-key",
            },
            clear=False,
        ):
            assert _blooio_reaction_available() is True

        with patch.dict(
            os.environ,
            {
                "HERMES_SESSION_PLATFORM": "telegram",
                "HERMES_SESSION_CHAT_ID": "+15551234567",
                "BLOOIO_API_KEY": "test-key",
            },
            clear=False,
        ):
            assert _blooio_reaction_available() is False

    def test_tool_definition_hidden_outside_blooio(self):
        with patch.dict(os.environ, {"HERMES_SESSION_PLATFORM": "telegram"}, clear=False):
            defs = get_tool_definitions(enabled_toolsets=["messaging"], quiet_mode=True)
        tool_names = {item["function"]["name"] for item in defs}
        assert "react_to_message" not in tool_names

    def test_tool_definition_visible_in_blooio(self):
        with patch.dict(
            os.environ,
            {
                "HERMES_SESSION_PLATFORM": "blooio",
                "HERMES_SESSION_CHAT_ID": "+15551234567",
                "BLOOIO_API_KEY": "test-key",
            },
            clear=False,
        ):
            defs = get_tool_definitions(enabled_toolsets=["messaging"], quiet_mode=True)
        tool_names = {item["function"]["name"] for item in defs}
        assert "react_to_message" in tool_names


class TestBlooioReactionTool:
    def test_rejects_invalid_reaction(self):
        with patch.dict(
            os.environ,
            {
                "HERMES_SESSION_PLATFORM": "blooio",
                "HERMES_SESSION_CHAT_ID": "+15551234567",
                "BLOOIO_API_KEY": "test-key",
            },
            clear=False,
        ):
            result = json.loads(react_to_message_tool({"reaction": "love"}))
        assert "error" in result
        assert "must start with '+'" in result["error"]

    def test_uses_current_message_id_when_present(self):
        response = MagicMock(status_code=200, text='{"success": true, "message_id": "msg_123", "reaction": "love", "action": "add"}')
        response.json.return_value = {"success": True, "message_id": "msg_123", "reaction": "love", "action": "add"}

        with patch.dict(
            os.environ,
            {
                "HERMES_SESSION_PLATFORM": "blooio",
                "HERMES_SESSION_CHAT_ID": "+15551234567",
                "HERMES_SESSION_MESSAGE_ID": "msg_abc123",
                "BLOOIO_API_KEY": "test-key",
            },
            clear=False,
        ), patch("tools.blooio_reaction_tool.requests.post", return_value=response) as post_mock:
            result = json.loads(react_to_message_tool({"reaction": "+love"}))

        assert result["success"] is True
        assert result["message_id"] == "msg_123"
        called_url = post_mock.call_args.args[0]
        assert "/chats/%2B15551234567/messages/msg_abc123/reactions" in called_url
        assert post_mock.call_args.kwargs["json"] == {"reaction": "+love"}

    def test_falls_back_to_latest_inbound_message(self):
        response = MagicMock(status_code=200, text='{"success": true, "message_id": "msg_last", "reaction": "like", "action": "add"}')
        response.json.return_value = {"success": True, "message_id": "msg_last", "reaction": "like", "action": "add"}

        with patch.dict(
            os.environ,
            {
                "HERMES_SESSION_PLATFORM": "blooio",
                "HERMES_SESSION_CHAT_ID": "+15551234567",
                "BLOOIO_API_KEY": "test-key",
            },
            clear=False,
        ), patch("tools.blooio_reaction_tool.requests.post", return_value=response) as post_mock:
            result = json.loads(react_to_message_tool({"reaction": "+like"}))

        assert result["success"] is True
        called_url = post_mock.call_args.args[0]
        assert "/messages/-1/reactions" in called_url
        assert post_mock.call_args.kwargs["json"] == {"reaction": "+like", "direction": "inbound"}


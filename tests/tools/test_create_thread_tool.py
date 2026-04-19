"""Tests for tools/create_thread_tool.py"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestCreateThreadSchema:
    """Test the tool schema is well-formed."""

    def test_schema_has_required_fields(self):
        from tools.create_thread_tool import CREATE_THREAD_SCHEMA
        assert CREATE_THREAD_SCHEMA["name"] == "create_thread"
        props = CREATE_THREAD_SCHEMA["parameters"]["properties"]
        assert "channel_id" in props
        assert "name" in props
        assert "message" in props
        assert "message_id" in props
        assert "auto_archive_duration" in props
        required = CREATE_THREAD_SCHEMA["parameters"]["required"]
        assert "channel_id" in required
        assert "name" in required

    def test_valid_auto_archive_values(self):
        from tools.create_thread_tool import _VALID_AUTO_ARCHIVE
        assert _VALID_AUTO_ARCHIVE == {60, 1440, 4320, 10080}


class TestCreateThreadValidation:
    """Test input validation in create_thread_tool."""

    def test_missing_channel_id(self):
        from tools.create_thread_tool import create_thread_tool
        result = json.loads(create_thread_tool({"name": "test"}))
        assert "error" in result
        assert "channel_id" in result["error"]

    def test_missing_name(self):
        from tools.create_thread_tool import create_thread_tool
        result = json.loads(create_thread_tool({"channel_id": "123"}))
        assert "error" in result
        assert "name" in result["error"]

    def test_invalid_auto_archive(self):
        from tools.create_thread_tool import create_thread_tool
        result = json.loads(create_thread_tool({
            "channel_id": "123",
            "name": "test",
            "auto_archive_duration": 999,
        }))
        assert "error" in result
        assert "auto_archive_duration" in result["error"]

    @patch("tools.create_thread_tool._get_discord_token", return_value=None)
    def test_no_token(self, mock_token):
        from tools.create_thread_tool import create_thread_tool
        result = json.loads(create_thread_tool({
            "channel_id": "123",
            "name": "test",
        }))
        assert "error" in result
        assert "token" in result["error"].lower()


class TestCreateThreadToolHandler:
    """Test the sync tool handler with _run_async mocked."""

    @patch("tools.create_thread_tool._get_discord_token", return_value="fake-token")
    @patch("model_tools._run_async")
    def test_successful_thread_creation_with_message(self, mock_run_async, mock_token):
        from tools.create_thread_tool import create_thread_tool

        # Single call to _create_discord_thread (which internally sends seed + creates thread)
        mock_run_async.return_value = {
            "success": True,
            "thread_id": "999888777",
            "thread_name": "test-thread",
            "seed_message_id": "111222333",
        }

        result = json.loads(create_thread_tool({
            "channel_id": "123456",
            "name": "test-thread",
            "message": "Hello from the tool!",
        }))

        assert result["success"] is True
        assert result["thread_id"] == "999888777"
        assert result["thread_name"] == "test-thread"
        assert result["seed_message_id"] == "111222333"
        assert mock_run_async.call_count == 1

    @patch("tools.create_thread_tool._get_discord_token", return_value="fake-token")
    @patch("model_tools._run_async")
    def test_thread_without_message_uses_default_seed(self, mock_run_async, mock_token):
        from tools.create_thread_tool import create_thread_tool

        mock_run_async.return_value = {
            "success": True,
            "thread_id": "999",
            "thread_name": "no-msg",
            "seed_message_id": "888",
        }

        result = json.loads(create_thread_tool({
            "channel_id": "123456",
            "name": "no-msg",
        }))

        assert result["success"] is True
        assert result["thread_id"] == "999"
        assert result["seed_message_id"] == "888"
        assert mock_run_async.call_count == 1

    @patch("tools.create_thread_tool._get_discord_token", return_value="fake-token")
    @patch("model_tools._run_async")
    def test_thread_creation_api_error(self, mock_run_async, mock_token):
        from tools.create_thread_tool import create_thread_tool

        mock_run_async.return_value = {
            "error": "Discord API error (403): Missing Access"
        }

        result = json.loads(create_thread_tool({
            "channel_id": "123456",
            "name": "fail-thread",
        }))

        assert "error" in result
        assert "403" in result["error"]

    @patch("tools.create_thread_tool._get_discord_token", return_value="fake-token")
    @patch("model_tools._run_async")
    def test_thread_from_existing_message(self, mock_run_async, mock_token):
        from tools.create_thread_tool import create_thread_tool

        mock_run_async.return_value = {
            "success": True,
            "thread_id": "555666",
            "thread_name": "from-message",
        }

        result = json.loads(create_thread_tool({
            "channel_id": "123456",
            "name": "from-message",
            "message_id": "444333",
        }))

        assert result["success"] is True
        assert result["thread_id"] == "555666"

    @patch("tools.create_thread_tool._get_discord_token", return_value="fake-token")
    @patch("model_tools._run_async", side_effect=Exception("async blew up"))
    def test_unexpected_exception(self, mock_run_async, mock_token):
        from tools.create_thread_tool import create_thread_tool

        result = json.loads(create_thread_tool({
            "channel_id": "123456",
            "name": "boom",
        }))

        assert "error" in result
        assert "Thread creation failed" in result["error"]


class TestRegistry:
    """Test that the tool is properly registered."""

    def test_registered_in_registry(self):
        from tools.registry import registry
        entry = registry.get_entry("create_thread")
        assert entry is not None
        assert entry.toolset == "messaging"
        assert entry.emoji == "🧵"

    def test_in_messaging_toolset(self):
        from toolsets import TOOLSETS
        assert "create_thread" in TOOLSETS["messaging"]["tools"]

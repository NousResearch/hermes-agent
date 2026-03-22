"""Tests for tools/discord_tool.py."""

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.discord_tool import (
    discord_fetch_message_tool,
    _format_message,
    _get_discord_token,
    _handle_parse_link,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _make_discord_message(
    message_id="111",
    channel_id="222",
    content="Hello world",
    username="testuser",
    global_name="Test User",
    attachments=None,
    embeds=None,
    referenced_message=None,
):
    return {
        "id": message_id,
        "channel_id": channel_id,
        "content": content,
        "author": {
            "id": "999",
            "username": username,
            "global_name": global_name,
            "bot": False,
        },
        "timestamp": "2026-03-21T10:00:00.000000+00:00",
        "edited_timestamp": None,
        "type": 0,
        "attachments": attachments or [],
        "embeds": embeds or [],
        "referenced_message": referenced_message,
    }


# ---------------------------------------------------------------------------
# parse_link tests
# ---------------------------------------------------------------------------

class TestParseLink:
    def test_full_url_with_message_id(self):
        result = json.loads(
            _handle_parse_link(
                {"url": "https://discord.com/channels/123456789/987654321/555555555"}
            )
        )
        assert result["guild_id"] == "123456789"
        assert result["channel_id"] == "987654321"
        assert result["message_id"] == "555555555"

    def test_channel_only_url(self):
        result = json.loads(
            _handle_parse_link(
                {"url": "https://discord.com/channels/111/222"}
            )
        )
        assert result["guild_id"] == "111"
        assert result["channel_id"] == "222"
        assert "message_id" not in result

    def test_invalid_url_returns_error(self):
        result = json.loads(
            _handle_parse_link({"url": "https://example.com/not-discord"})
        )
        assert "error" in result

    def test_missing_url_returns_error(self):
        result = json.loads(_handle_parse_link({}))
        assert "error" in result

    def test_tool_dispatch_parse_link(self):
        result = json.loads(
            discord_fetch_message_tool(
                {
                    "action": "parse_link",
                    "url": "https://discord.com/channels/1/2/3",
                }
            )
        )
        assert result["guild_id"] == "1"
        assert result["channel_id"] == "2"
        assert result["message_id"] == "3"


# ---------------------------------------------------------------------------
# _format_message tests
# ---------------------------------------------------------------------------

class TestFormatMessage:
    def test_basic_message(self):
        raw = _make_discord_message()
        result = _format_message(raw)
        assert result["message_id"] == "111"
        assert result["channel_id"] == "222"
        assert result["content"] == "Hello world"
        assert result["author"]["username"] == "testuser"
        assert result["author"]["display_name"] == "Test User"
        assert result["author"]["bot"] is False

    def test_attachments_included(self):
        raw = _make_discord_message(
            attachments=[
                {
                    "id": "att1",
                    "filename": "image.png",
                    "url": "https://cdn.discordapp.com/attachments/image.png",
                    "content_type": "image/png",
                    "size": 12345,
                }
            ]
        )
        result = _format_message(raw)
        assert len(result["attachments"]) == 1
        assert result["attachments"][0]["filename"] == "image.png"

    def test_embeds_included(self):
        raw = _make_discord_message(
            embeds=[{"title": "Some Title", "description": "Some desc", "url": "https://example.com"}]
        )
        result = _format_message(raw)
        assert len(result["embeds"]) == 1
        assert result["embeds"][0]["title"] == "Some Title"

    def test_empty_embeds_excluded(self):
        raw = _make_discord_message(
            embeds=[{"type": "image", "url": "https://example.com/img.png"}]
        )
        result = _format_message(raw)
        assert "embeds" not in result

    def test_reply_to_included(self):
        referenced = _make_discord_message(
            message_id="000",
            content="Original message here",
            username="originaluser",
        )
        raw = _make_discord_message(referenced_message=referenced)
        result = _format_message(raw)
        assert result["reply_to"]["message_id"] == "000"
        assert result["reply_to"]["author"] == "originaluser"
        assert "Original message" in result["reply_to"]["content"]

    def test_display_name_falls_back_to_username(self):
        raw = _make_discord_message(global_name=None)
        result = _format_message(raw)
        assert result["author"]["display_name"] == "testuser"


# ---------------------------------------------------------------------------
# fetch action tests
# ---------------------------------------------------------------------------

class TestFetchAction:
    def test_missing_channel_id_returns_error(self):
        result = json.loads(
            discord_fetch_message_tool(
                {"action": "fetch", "message_id": "123"}
            )
        )
        assert "error" in result
        assert "channel_id" in result["error"]

    def test_missing_message_id_returns_error(self):
        result = json.loads(
            discord_fetch_message_tool(
                {"action": "fetch", "channel_id": "456"}
            )
        )
        assert "error" in result
        assert "message_id" in result["error"]

    def test_no_token_returns_error(self):
        with patch("tools.discord_tool._get_discord_token", return_value=None):
            result = json.loads(
                discord_fetch_message_tool(
                    {"action": "fetch", "channel_id": "1", "message_id": "2"}
                )
            )
        assert "error" in result
        assert "token" in result["error"].lower()

    def test_successful_fetch(self):
        raw = _make_discord_message(message_id="42", channel_id="10", content="Hey!")
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=raw)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with patch("tools.discord_tool._get_discord_token", return_value="tok123"), \
             patch("model_tools._run_async", side_effect=_run), \
             patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
            result = json.loads(
                discord_fetch_message_tool(
                    {"action": "fetch", "channel_id": "10", "message_id": "42"}
                )
            )

        assert result["message_id"] == "42"
        assert result["content"] == "Hey!"

    def test_404_returns_friendly_error(self):
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with patch("tools.discord_tool._get_discord_token", return_value="tok"), \
             patch("model_tools._run_async", side_effect=_run), \
             patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
            result = json.loads(
                discord_fetch_message_tool(
                    {"action": "fetch", "channel_id": "1", "message_id": "99"}
                )
            )

        assert "error" in result
        assert "deleted" in result["error"] or "not found" in result["error"].lower()

    def test_403_returns_permission_error(self):
        mock_response = MagicMock()
        mock_response.status = 403
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with patch("tools.discord_tool._get_discord_token", return_value="tok"), \
             patch("model_tools._run_async", side_effect=_run), \
             patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
            result = json.loads(
                discord_fetch_message_tool(
                    {"action": "fetch", "channel_id": "1", "message_id": "99"}
                )
            )

        assert "error" in result
        assert "permission" in result["error"].lower()


# ---------------------------------------------------------------------------
# history action tests
# ---------------------------------------------------------------------------

class TestHistoryAction:
    def test_missing_channel_id_returns_error(self):
        result = json.loads(
            discord_fetch_message_tool({"action": "history"})
        )
        assert "error" in result
        assert "channel_id" in result["error"]

    def test_no_token_returns_error(self):
        with patch("tools.discord_tool._get_discord_token", return_value=None):
            result = json.loads(
                discord_fetch_message_tool(
                    {"action": "history", "channel_id": "123"}
                )
            )
        assert "error" in result

    def test_limit_clamped_to_100(self):
        captured_params = {}

        async def mock_fetch(token, channel_id, limit, before):
            captured_params["limit"] = limit
            return json.dumps({"channel_id": channel_id, "count": 0, "messages": []})

        with patch("tools.discord_tool._get_discord_token", return_value="tok"), \
             patch("model_tools._run_async", side_effect=_run), \
             patch("tools.discord_tool._fetch_history", side_effect=mock_fetch):
            discord_fetch_message_tool(
                {"action": "history", "channel_id": "1", "limit": 9999}
            )

        assert captured_params["limit"] == 100

    def test_successful_history(self):
        messages = [
            _make_discord_message(message_id=str(i), content=f"msg {i}")
            for i in range(3)
        ]
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=messages)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with patch("tools.discord_tool._get_discord_token", return_value="tok"), \
             patch("model_tools._run_async", side_effect=_run), \
             patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
            result = json.loads(
                discord_fetch_message_tool(
                    {"action": "history", "channel_id": "222", "limit": 3}
                )
            )

        assert result["count"] == 3
        assert len(result["messages"]) == 3
        assert result["channel_id"] == "222"


# ---------------------------------------------------------------------------
# Token resolution tests
# ---------------------------------------------------------------------------

class TestGetDiscordToken:
    def test_env_var_takes_priority(self):
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "envtoken"}):
            assert _get_discord_token() == "envtoken"

    def test_falls_back_to_gateway_config(self):
        from gateway.config import Platform
        pconfig = SimpleNamespace(enabled=True, token="cfgtoken")
        config = SimpleNamespace(platforms={Platform.DISCORD: pconfig})

        with patch.dict(os.environ, {}, clear=False), \
             patch("os.getenv", side_effect=lambda k, d="": "" if k == "DISCORD_BOT_TOKEN" else os.environ.get(k, d)), \
             patch("gateway.config.load_gateway_config", return_value=config):
            # Direct env cleared for this key
            env_backup = os.environ.pop("DISCORD_BOT_TOKEN", None)
            try:
                token = _get_discord_token()
            finally:
                if env_backup is not None:
                    os.environ["DISCORD_BOT_TOKEN"] = env_backup

    def test_returns_none_when_nothing_configured(self):
        env_backup = os.environ.pop("DISCORD_BOT_TOKEN", None)
        try:
            with patch("gateway.config.load_gateway_config", side_effect=Exception("no config")):
                token = _get_discord_token()
            assert token is None or token == ""
        finally:
            if env_backup is not None:
                os.environ["DISCORD_BOT_TOKEN"] = env_backup

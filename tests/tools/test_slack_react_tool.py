"""Tests for the slack_react tool."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure the tool module is imported fresh and registered."""
    import importlib
    import tools.slack_react_tool as m  # noqa: F401 — triggers registration

    yield


def _call(args, **kw):
    from tools.slack_react_tool import slack_react_tool

    return json.loads(slack_react_tool(args, **kw))


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------

class TestTokenResolution:
    def test_missing_token_returns_error(self):
        with patch("tools.slack_react_tool._get_slack_token", return_value=None):
            result = _call({"emoji": "thumbsup", "channel": "C123", "timestamp": "123.456"})
        assert result["error"].startswith("Slack gateway not configured")

    def test_missing_channel_returns_error(self):
        with (
            patch("tools.slack_react_tool._get_slack_token", return_value="xoxb-fake"),
            patch("tools.slack_react_tool._get_current_context", return_value=(None, None)),
        ):
            result = _call({"emoji": "thumbsup", "timestamp": "123.456"})
        assert "channel" in result["error"]

    def test_missing_timestamp_returns_error(self):
        with (
            patch("tools.slack_react_tool._get_slack_token", return_value="xoxb-fake"),
            patch("tools.slack_react_tool._get_current_context", return_value=("C123", None)),
        ):
            result = _call({"emoji": "thumbsup", "channel": "C123"})
        assert "timestamp" in result["error"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def _mock_aiohttp_ok(self):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"ok": True})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return mock_session

    def test_successful_reaction(self):
        mock_session = self._mock_aiohttp_ok()
        with (
            patch("tools.slack_react_tool._get_slack_token", return_value="xoxb-fake"),
            patch("tools.slack_react_tool._get_current_context", return_value=(None, None)),
            patch("aiohttp.ClientSession", return_value=mock_session),
        ):
            result = _call(
                {"emoji": "white_check_mark", "channel": "C0B3XC4TZ1S", "timestamp": "1716201336.943429"}
            )
        assert result["success"] is True
        assert result["emoji"] == "white_check_mark"

    def test_context_fallback(self):
        """channel/ts resolved from session context when not passed explicitly."""
        mock_session = self._mock_aiohttp_ok()
        with (
            patch("tools.slack_react_tool._get_slack_token", return_value="xoxb-fake"),
            patch(
                "tools.slack_react_tool._get_current_context",
                return_value=("C_CTX", "111.222"),
            ),
            patch("aiohttp.ClientSession", return_value=mock_session),
        ):
            result = _call({"emoji": "eyes"})
        assert result["success"] is True
        assert result["channel"] == "C_CTX"

    def test_already_reacted_is_success(self):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"ok": False, "error": "already_reacted"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("tools.slack_react_tool._get_slack_token", return_value="xoxb-fake"),
            patch("tools.slack_react_tool._get_current_context", return_value=(None, None)),
            patch("aiohttp.ClientSession", return_value=mock_session),
        ):
            result = _call({"emoji": "thumbsup", "channel": "C1", "timestamp": "1.2"})
        assert result["success"] is True
        assert result.get("note") == "already reacted"

    def test_emoji_colons_stripped(self):
        mock_session = self._mock_aiohttp_ok()
        with (
            patch("tools.slack_react_tool._get_slack_token", return_value="xoxb-fake"),
            patch("tools.slack_react_tool._get_current_context", return_value=(None, None)),
            patch("aiohttp.ClientSession", return_value=mock_session),
        ):
            result = _call(
                {"emoji": ":thumbsup:", "channel": "C1", "timestamp": "1.2"}
            )
        assert result["success"] is True
        assert result["emoji"] == "thumbsup"


# ---------------------------------------------------------------------------
# API errors
# ---------------------------------------------------------------------------

class TestApiErrors:
    def test_slack_api_error_propagated(self):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"ok": False, "error": "invalid_name"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("tools.slack_react_tool._get_slack_token", return_value="xoxb-fake"),
            patch("tools.slack_react_tool._get_current_context", return_value=(None, None)),
            patch("aiohttp.ClientSession", return_value=mock_session),
        ):
            result = _call({"emoji": "???", "channel": "C1", "timestamp": "1.2"})
        assert "error" in result
        assert "invalid_name" in result["error"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_tool_is_registered(self):
        from tools.registry import registry

        names = [t["schema"]["name"] for t in registry.get_tools()]
        assert "slack_react" in names

    def test_toolset_is_messaging(self):
        from tools.registry import registry

        tool = next(t for t in registry.get_tools() if t["schema"]["name"] == "slack_react")
        assert tool["toolset"] == "messaging"

    def test_emoji_parameter_required(self):
        from tools.registry import registry

        tool = next(t for t in registry.get_tools() if t["schema"]["name"] == "slack_react")
        assert "emoji" in tool["schema"]["parameters"]["required"]

"""Tests for social relay tools."""

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from tools.social_tools import (
    social_tool,
    check_social_requirements,
    _load_social_config,
    _check_permission,
    _check_rate_limit,
    _rate_counters,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module state between tests."""
    import tools.social_tools as mod
    mod._config_cache = None
    _rate_counters.clear()
    yield
    mod._config_cache = None
    _rate_counters.clear()


@pytest.fixture
def social_config(tmp_path):
    """Create a config.yaml with social enabled."""
    config_content = """
social:
  enabled: true
  relay: "http://localhost:8787"
  permissions:
    post: true
    reply: true
    like: true
    repost: true
    follow: true
    delete: true
  limits:
    max_posts_per_hour: 5
    max_replies_per_hour: 10
    max_likes_per_hour: 20
  payments:
    enabled: true
    method: "tempo"
    max_spend_per_hour: 0.01
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


@pytest.fixture
def social_disabled(tmp_path):
    """Create a config.yaml with social disabled."""
    config_content = """
social:
  enabled: false
  relay: "http://localhost:8787"
"""
    (tmp_path / "config.yaml").write_text(config_content)
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


@pytest.fixture
def no_config(tmp_path):
    """No config.yaml at all."""
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


class TestCheckSocialRequirements:
    def test_returns_true_when_enabled(self, social_config):
        assert check_social_requirements() is True

    def test_returns_false_when_disabled(self, social_disabled):
        assert check_social_requirements() is False

    def test_returns_false_when_no_config(self, no_config):
        assert check_social_requirements() is False


class TestCheckPermission:
    def test_returns_none_when_allowed(self, social_config):
        assert _check_permission("post") is None

    def test_returns_error_when_disabled(self, social_disabled):
        error = _check_permission("post")
        assert error is not None
        assert "not enabled" in error

    def test_returns_error_when_permission_denied(self, tmp_path):
        config_content = """
social:
  enabled: true
  relay: "http://localhost:8787"
  permissions:
    post: false
"""
        (tmp_path / "config.yaml").write_text(config_content)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            import tools.social_tools as mod
            mod._config_cache = None
            error = _check_permission("post")
            assert error is not None
            assert "Permission denied" in error


class TestCheckRateLimit:
    def test_allows_within_limit(self, social_config):
        assert _check_rate_limit("post") is None

    def test_blocks_when_limit_exceeded(self, social_config):
        # Config says max 5 posts/hour
        for _ in range(5):
            _check_rate_limit("post")
        error = _check_rate_limit("post")
        assert error is not None
        assert "Rate limit" in error


class TestSocialToolReadActions:
    @patch("tools.social_tools._relay_get")
    def test_feed_returns_posts(self, mock_get, social_config):
        mock_get.return_value = {
            "ok": True,
            "data": [
                {
                    "id": "aa" * 32,
                    "pubkey": "bb" * 32,
                    "created_at": 1000,
                    "kind": 1,
                    "tags": [["t", "test"]],
                    "content": "Hello!",
                    "sig": "cc" * 64,
                }
            ],
        }
        result = json.loads(social_tool(action="feed", limit=10))
        assert result["count"] == 1
        assert "Hello!" in result["posts"][0]["content"]

    @patch("tools.social_tools._relay_get")
    def test_search_returns_results(self, mock_get, social_config):
        mock_get.return_value = {"ok": True, "data": []}
        result = json.loads(social_tool(action="search", query="hermes"))
        assert result["count"] == 0
        assert result["query"] == "hermes"

    def test_search_requires_query(self, social_config):
        result = json.loads(social_tool(action="search"))
        assert "error" in result

    @patch("tools.social_tools._relay_get")
    def test_view_agent(self, mock_get, social_config):
        mock_get.return_value = {
            "ok": True,
            "data": {"pubkey": "aa" * 32, "display_name": "Test"},
        }
        result = json.loads(social_tool(action="view_agent", target="aa" * 32))
        assert result["agent"]["display_name"] == "Test"

    def test_view_agent_requires_target(self, social_config):
        result = json.loads(social_tool(action="view_agent"))
        assert "error" in result


class TestSocialToolWriteActions:
    def test_post_requires_identity(self, social_config):
        result = json.loads(social_tool(action="post", content="hello"))
        assert "error" in result
        assert "identity" in result["error"].lower()

    @patch("tools.social_tools.identity_exists", return_value=True)
    @patch("tools.social_tools.get_identity")
    @patch("tools.social_tools._relay_post")
    @patch("tools.social_tools.create_post_event")
    def test_post_success(self, mock_create, mock_post, mock_ident, mock_exists, social_config):
        mock_ident.return_value = MagicMock(pubkey_hex="aa" * 32)
        mock_create.return_value = {"id": "bb" * 32, "pubkey": "aa" * 32}
        mock_post.return_value = {"ok": True}

        result = json.loads(social_tool(action="post", content="Hello AgentNet!"))
        assert result["posted"] is True

    def test_post_blocked_when_disabled(self, social_disabled):
        result = json.loads(social_tool(action="post", content="test"))
        assert "error" in result

    @patch("tools.social_tools.identity_exists", return_value=True)
    @patch("tools.social_tools.get_identity")
    def test_unknown_action(self, mock_ident, mock_exists, social_config):
        mock_ident.return_value = MagicMock(pubkey_hex="aa" * 32)
        result = json.loads(social_tool(action="nonexistent"))
        assert "error" in result
        assert "Unknown action" in result["error"]

    @patch("tools.social_tools.identity_exists", return_value=True)
    @patch("tools.social_tools.get_identity")
    @patch("tools.social_tools._relay_post")
    @patch("tools.social_tools._relay_get")
    @patch("tools.social_tools._resolve_tempo_address", return_value="0x" + "a" * 40)
    @patch("tools.social_tools._send_usdc")
    def test_like_should_not_post_if_tip_fails(
        self, mock_send, mock_resolve, mock_get, mock_post, mock_ident, mock_exists, social_config
    ):
        """If tip transfer fails, like event should NOT be posted to relay."""
        mock_ident.return_value = MagicMock(pubkey_hex="bb" * 32)
        mock_get.return_value = {"ok": True, "data": {"pubkey": "cc" * 32}}
        mock_post.return_value = {"ok": True}
        mock_send.return_value = {"sent": False, "reason": "insufficient balance"}

        result = json.loads(social_tool(action="like", target="dd" * 32))

        # Expected: like should NOT go through if tip fails
        assert result.get("liked") is not True, "Like should not post if micro-tip fails"

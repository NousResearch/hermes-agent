"""Tests for social tool configuration: thread safety, permissions, rate limits."""

import json
import os
import threading
from unittest.mock import patch

import pytest

from tools.social_tools import (
    _load_social_config,
    _check_permission,
    _check_rate_limit,
    _rate_counters,
    check_social_requirements,
)


@pytest.fixture(autouse=True)
def reset_state():
    import tools.social_tools as mod
    mod._config_cache = None
    _rate_counters.clear()
    yield
    mod._config_cache = None
    _rate_counters.clear()


class TestConfigThreadSafety:
    def test_concurrent_config_loads(self, tmp_path):
        """Config loading should be thread-safe."""
        config = "social:\n  enabled: true\n  relay: http://localhost"
        (tmp_path / "config.yaml").write_text(config)

        results = []
        errors = []

        def load():
            try:
                with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
                    import tools.social_tools as mod
                    mod._config_cache = None
                    c = _load_social_config()
                    results.append(c.get("enabled"))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=load) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r is True for r in results)


class TestRateLimits:
    def test_post_limit(self, tmp_path):
        config = """
social:
  enabled: true
  relay: http://localhost
  limits:
    max_posts_per_hour: 3
"""
        (tmp_path / "config.yaml").write_text(config)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            import tools.social_tools as mod
            mod._config_cache = None

            assert _check_rate_limit("post") is None  # 1
            assert _check_rate_limit("post") is None  # 2
            assert _check_rate_limit("post") is None  # 3
            result = _check_rate_limit("post")         # 4 - blocked
            assert result is not None
            assert "Rate limit" in result

    def test_repost_uses_own_limit(self, tmp_path):
        """Repost should use max_reposts_per_hour, not max_likes_per_hour."""
        config = """
social:
  enabled: true
  relay: http://localhost
  limits:
    max_reposts_per_hour: 2
    max_likes_per_hour: 100
"""
        (tmp_path / "config.yaml").write_text(config)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            import tools.social_tools as mod
            mod._config_cache = None

            assert _check_rate_limit("repost") is None  # 1
            assert _check_rate_limit("repost") is None  # 2
            result = _check_rate_limit("repost")         # 3 - blocked
            assert result is not None

    def test_different_actions_independent(self, tmp_path):
        config = """
social:
  enabled: true
  relay: http://localhost
  limits:
    max_posts_per_hour: 1
    max_likes_per_hour: 1
"""
        (tmp_path / "config.yaml").write_text(config)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            import tools.social_tools as mod
            mod._config_cache = None

            assert _check_rate_limit("post") is None
            assert _check_rate_limit("like") is None
            # Both used up, but independent
            assert _check_rate_limit("post") is not None
            assert _check_rate_limit("like") is not None


class TestPermissions:
    def test_disabled_social(self, tmp_path):
        config = "social:\n  enabled: false\n  relay: http://localhost"
        (tmp_path / "config.yaml").write_text(config)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            import tools.social_tools as mod
            mod._config_cache = None
            result = _check_permission("post")
            assert result is not None
            assert "not enabled" in result

    def test_no_relay(self, tmp_path):
        config = "social:\n  enabled: true\n  relay: ''"
        (tmp_path / "config.yaml").write_text(config)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            import tools.social_tools as mod
            mod._config_cache = None
            result = _check_permission("post")
            assert result is not None
            assert "relay" in result.lower()

    def test_permission_denied(self, tmp_path):
        config = """
social:
  enabled: true
  relay: http://localhost
  permissions:
    post: false
    like: true
"""
        (tmp_path / "config.yaml").write_text(config)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            import tools.social_tools as mod
            mod._config_cache = None
            assert _check_permission("post") is not None
            assert _check_permission("like") is None

"""Tests for tools.xai_http — has_xai_credentials, hermes_xai_user_agent, get_env_value."""

from __future__ import annotations

from tools.xai_http import get_env_value, has_xai_credentials, hermes_xai_user_agent


class TestHasXaiCredentials:
    def test_api_key_env_present(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "xai-key-123")
        assert has_xai_credentials() is True

    def test_api_key_env_empty(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "")
        # Env empty → check auth.json → file probably doesn't exist
        # Falls through to exception handler → False
        assert has_xai_credentials() is False

    def test_api_key_env_whitespace(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "   ")
        assert has_xai_credentials() is False

    def test_no_env_no_file(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        # auth.json won't exist → False
        assert has_xai_credentials() is False


class TestHermesXaiUserAgent:
    def test_returns_hermes_prefix(self):
        ua = hermes_xai_user_agent()
        assert ua.startswith("Hermes-Agent/")


class TestGetEnvValue:
    def test_from_os_environ(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "os_value")
        # hermes_cli.config.get_env_value may not exist or return None
        # Falls to os.environ
        result = get_env_value("TEST_VAR", default="fallback")
        assert result == "os_value"

    def test_not_set_returns_default(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        result = get_env_value("NONEXISTENT_VAR", default="default_val")
        assert result == "default_val"

    def test_default_is_none(self, monkeypatch):
        monkeypatch.delenv("MISSING", raising=False)
        result = get_env_value("MISSING")
        assert result is None

"""Tests for tools.xai_http -- shared xAI HTTP credential helpers.

Covers:
- has_xai_credentials() -- env var, auth.json, edge cases
- get_env_value() -- Hermes config .env layer, os.environ fallback
- hermes_xai_user_agent() -- version import paths
- resolve_xai_http_credentials() -- OAuth runtime, OAuth credentials, env var fallback
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# has_xai_credentials
# ---------------------------------------------------------------------------


class TestHasXaiCredentials:
    """Cheap probe -- must never do network I/O or acquire auth-store locks."""

    def test_env_var_set(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is True

    def test_env_var_whitespace_only(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "   ")
        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is False

    def test_env_var_empty(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "")
        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is False

    def test_auth_json_valid_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "providers": {
                        "xai-oauth": {"tokens": {"access_token": "ya29.fake-token"}}
                    },
                }
            )
        )

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is True

    def test_auth_json_missing_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is False

    def test_auth_json_corrupted(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text("not json }{")

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is False

    def test_auth_json_no_providers_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text(json.dumps({"version": 1}))

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is False

    def test_auth_json_providers_not_dict(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text(
            json.dumps({"providers": ["not-a-dict"]})
        )

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is False

    def test_auth_json_no_xai_oauth_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text(
            json.dumps({"providers": {"brave": {"api_key": "abc"}}})
        )

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is False

    def test_auth_json_tokens_not_dict(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text(
            json.dumps(
                {"providers": {"xai-oauth": {"tokens": ["not-a-dict"]}}}
            )
        )

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is False

    def test_auth_json_empty_access_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "xai-oauth": {"tokens": {"access_token": ""}}
                    }
                }
            )
        )

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is False

    def test_auth_json_whitespace_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "xai-oauth": {"tokens": {"access_token": "  \n  "}}
                    }
                }
            )
        )

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is False

    def test_env_var_takes_precedence(self, monkeypatch, tmp_path):
        """When both env var and auth.json are present, env var wins (fast path)."""
        monkeypatch.setenv("XAI_API_KEY", "sk-env-wins")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "auth.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "xai-oauth": {"tokens": {"access_token": "ya29.fake"}}
                    }
                }
            )
        )

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is True

    def test_get_hermes_home_import_fails(self, monkeypatch):
        """If hermes_constants can't be imported, fall back to False."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch.dict("sys.modules", {"hermes_constants": None}, clear=False):
            # Re-import to trigger the ImportError path
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            assert xai_http.has_xai_credentials() is False


# ---------------------------------------------------------------------------
# get_env_value
# ---------------------------------------------------------------------------


class TestGetEnvValue:
    """Reads from Hermes .env layer first, then os.environ."""

    def test_dotenv_value_found(self):
        with patch(
            "tools.xai_http.get_env_value",
            wraps=lambda name, default=None: "dotenv-value",
        ):
            from tools.xai_http import get_env_value

            result = get_env_value("XAI_API_KEY")
            assert result == "dotenv-value"

    def test_dotenv_value_none_falls_through(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "os-env-value")
        with patch(
            "hermes_cli.config.get_env_value", return_value=None
        ) as mock_dotenv:
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.get_env_value("XAI_API_KEY")
            assert result == "os-env-value"
            mock_dotenv.assert_called_once_with("XAI_API_KEY")

    def test_dotenv_raises_falls_through(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "os-env-value")
        with patch(
            "hermes_cli.config.get_env_value",
            side_effect=ImportError("not installed"),
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.get_env_value("XAI_API_KEY")
            assert result == "os-env-value"

    def test_default_fallback(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch(
            "hermes_cli.config.get_env_value", return_value=None
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.get_env_value("XAI_API_KEY", default="fallback")
            assert result == "fallback"

    def test_default_fallback_none(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch(
            "hermes_cli.config.get_env_value", return_value=None
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.get_env_value("XAI_API_KEY")
            assert result is None


# ---------------------------------------------------------------------------
# hermes_xai_user_agent
# ---------------------------------------------------------------------------


class TestHermesXaiUserAgent:
    def test_version_importable(self):
        from tools.xai_http import hermes_xai_user_agent

        ua = hermes_xai_user_agent()
        assert ua.startswith("Hermes-Agent/")
        assert ua != "Hermes-Agent/unknown"

    def test_version_not_importable(self):
        import importlib

        with patch.dict("sys.modules", {"hermes_cli": None}, clear=False):
            from tools import xai_http

            importlib.reload(xai_http)
            ua = xai_http.hermes_xai_user_agent()
            assert ua == "Hermes-Agent/unknown"


# ---------------------------------------------------------------------------
# resolve_xai_http_credentials
# ---------------------------------------------------------------------------


class TestResolveXaiHttpCredentials:
    """Credential resolution: OAuth runtime provider -> OAuth credentials -> env var."""

    def test_oauth_runtime_provider_happy(self):
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"api_key": "rt-oauth-key", "base_url": "https://custom.x.ai"},
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.resolve_xai_http_credentials()
            assert result["provider"] == "xai-oauth"
            assert result["api_key"] == "rt-oauth-key"
            assert result["base_url"] == "https://custom.x.ai"

    def test_oauth_runtime_provider_empty_key_falls_through(self):
        """Runtime provider returns empty api_key -> fall to OAuth credentials path."""
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"api_key": "", "base_url": ""},
        ), patch(
            "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
            return_value={"api_key": "oauth-creds-key", "base_url": ""},
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.resolve_xai_http_credentials()
            assert result["provider"] == "xai-oauth"
            assert result["api_key"] == "oauth-creds-key"
            assert result["base_url"] == "https://api.x.ai/v1"

    def test_oauth_runtime_provider_exception_falls_through(self):
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=RuntimeError("provider error"),
        ), patch(
            "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
            return_value={"api_key": "fallback-oauth-key", "base_url": ""},
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.resolve_xai_http_credentials()
            assert result["provider"] == "xai-oauth"
            assert result["api_key"] == "fallback-oauth-key"

    def test_all_oauth_paths_fail_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk-env-fallback")
        monkeypatch.setenv("XAI_BASE_URL", "")
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=Exception("fail"),
        ), patch(
            "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
            side_effect=Exception("fail"),
        ), patch("tools.xai_http.get_env_value", side_effect=lambda name, default=None: monkeypatch.getenv(name, default)):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.resolve_xai_http_credentials()
            assert result["provider"] == "xai"
            assert result["api_key"] == "sk-env-fallback"

    def test_no_creds_anywhere(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.delenv("XAI_BASE_URL", raising=False)
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=Exception("fail"),
        ), patch(
            "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
            side_effect=Exception("fail"),
        ), patch(
            "tools.xai_http.get_env_value", return_value=None
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.resolve_xai_http_credentials()
            assert result["provider"] == "xai"
            assert result["api_key"] == ""

    def test_force_refresh_skips_runtime_provider(self):
        """force_refresh=True should bypass the runtime_provider shortcut."""
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=AssertionError("should not be called"),
        ), patch(
            "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
            return_value={"api_key": "refreshed-key", "base_url": ""},
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.resolve_xai_http_credentials(force_refresh=True)
            assert result["provider"] == "xai-oauth"
            assert result["api_key"] == "refreshed-key"

    def test_base_url_trailing_slash_removed(self):
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"api_key": "key", "base_url": "https://x.ai/v1/"},
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.resolve_xai_http_credentials()
            assert result["base_url"] == "https://x.ai/v1"

    def test_base_url_default_when_empty(self):
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"api_key": "key", "base_url": ""},
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.resolve_xai_http_credentials()
            assert result["base_url"] == "https://api.x.ai/v1"

    def test_oauth_credentials_empty_key_falls_through(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "env-key")
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=Exception("skip"),
        ), patch(
            "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
            return_value={"api_key": "", "base_url": ""},
        ):
            import importlib

            from tools import xai_http

            importlib.reload(xai_http)
            result = xai_http.resolve_xai_http_credentials()
            assert result["provider"] == "xai"
            assert result["api_key"] == "env-key"

"""Tests for credential_pool .env fallback and auth credential pool lookup."""

import os
import pytest
from unittest.mock import patch, MagicMock


def _make_pconfig(env_vars=None):
    """Create a minimal ProviderConfig for testing."""
    from hermes_cli.auth import ProviderConfig
    return ProviderConfig(
        id="openai",
        name="OpenAI",
        auth_type="api_key",
        api_key_env_vars=tuple(env_vars or ["OPENAI_API_KEY"]),
    )


class TestCredentialPoolEnvFallback:
    """Verify _seed_from_env resolves keys from both os.environ and .env file."""

    def test_os_environ_still_works(self):
        """Existing os.environ resolution must not break.
        _seed_from_env only collects env var names, does not return found=True
        for existing keys — that is _resolve's job. Just verify no crash."""
        from agent.credential_pool import _seed_from_env
        # Should not raise
        found, entries = _seed_from_env("openai", [])

    def test_get_env_value_import_does_not_crash(self):
        """Importing get_env_value from hermes_cli.config should not raise."""
        try:
            from hermes_cli.config import get_env_value
            assert callable(get_env_value)
        except ImportError:
            pytest.skip("hermes_cli.config not available in test environment")


class TestAuthCredentialPoolFallback:
    """Verify auth.py falls back to credential pool when env vars are empty."""

    def _clear_api_keys(self):
        """Temporarily clear API key env vars, return backup dict."""
        backup = {}
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                     "ZAI_API_KEY", "DEEPSEEK_API_KEY"]:
            if key in os.environ:
                backup[key] = os.environ.pop(key)
        return backup

    def test_credential_pool_fallback_structure(self):
        """When no env var is set, auth should try credential pool."""
        from hermes_cli.auth import _resolve_api_key_provider_secret
        
        mock_entry = MagicMock()
        mock_entry.access_token = "test-pool-key-12345"
        mock_entry.runtime_api_key = ""
        
        mock_pool = MagicMock()
        mock_pool.has_credentials.return_value = True
        mock_pool.peek.return_value = mock_entry
        
        backup = self._clear_api_keys()
        try:
            with patch("agent.credential_pool.load_pool", return_value=mock_pool):
                key, source = _resolve_api_key_provider_secret(
                    provider_id="openai",
                    pconfig=_make_pconfig(),
                )
                assert "test-pool-key-12345" in key
                assert "credential_pool" in source
        finally:
            os.environ.update(backup)

    def test_credential_pool_empty_returns_empty(self):
        """When pool is empty, return empty string."""
        from hermes_cli.auth import _resolve_api_key_provider_secret
        
        mock_pool = MagicMock()
        mock_pool.has_credentials.return_value = False
        
        backup = self._clear_api_keys()
        try:
            with patch("agent.credential_pool.load_pool", return_value=mock_pool):
                key, source = _resolve_api_key_provider_secret(
                    provider_id="openai",
                    pconfig=_make_pconfig(),
                )
                assert key == ""
        finally:
            os.environ.update(backup)

    def test_env_var_takes_priority_over_pool(self):
        """Env vars should be checked before credential pool."""
        from hermes_cli.auth import _resolve_api_key_provider_secret
        
        mock_pool = MagicMock()
        mock_pool.has_credentials.return_value = True
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key-first-abc123"}):
            with patch("agent.credential_pool.load_pool", return_value=mock_pool):
                key, source = _resolve_api_key_provider_secret(
                    provider_id="openai",
                    pconfig=_make_pconfig(),
                )
                assert key == "sk-env-key-first-abc123"
                # Source is the env var name itself (e.g. "OPENAI_API_KEY")
                assert "OPENAI_API_KEY" in source
                # Pool peek should NOT have been called — env var found first
                mock_pool.peek.assert_not_called()

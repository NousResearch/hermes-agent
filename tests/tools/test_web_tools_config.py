"""Tests for web backend client configuration and singleton behavior.

Coverage:
  _get_firecrawl_client() — configuration matrix, singleton caching,
  constructor failure recovery, return value verification, edge cases.
  _get_backend() — backend selection logic with env var combinations.
  _get_parallel_client() — Parallel client configuration, singleton caching.
  check_web_api_key() — unified availability check.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestFirecrawlClientConfig:
    """Test suite for Firecrawl client initialization."""

    def setup_method(self):
        """Reset client and env vars before each test."""
        import tools.web_tools
        tools.web_tools._firecrawl_client = None
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"):
            os.environ.pop(key, None)

    def teardown_method(self):
        """Reset client after each test."""
        import tools.web_tools
        tools.web_tools._firecrawl_client = None
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"):
            os.environ.pop(key, None)

    # ── Configuration matrix ─────────────────────────────────────────

    def test_cloud_mode_key_only(self):
        """API key without URL → cloud Firecrawl."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                result = _get_firecrawl_client()
                mock_fc.assert_called_once_with(api_key="fc-test")
                assert result is mock_fc.return_value

    def test_self_hosted_with_key(self):
        """Both key + URL → self-hosted with auth."""
        with patch.dict(os.environ, {
            "FIRECRAWL_API_KEY": "fc-test",
            "FIRECRAWL_API_URL": "http://localhost:3002",
        }):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                result = _get_firecrawl_client()
                mock_fc.assert_called_once_with(
                    api_key="fc-test", api_url="http://localhost:3002"
                )
                assert result is mock_fc.return_value

    def test_self_hosted_no_key(self):
        """URL only, no key → self-hosted without auth."""
        with patch.dict(os.environ, {"FIRECRAWL_API_URL": "http://localhost:3002"}):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                result = _get_firecrawl_client()
                mock_fc.assert_called_once_with(api_url="http://localhost:3002")
                assert result is mock_fc.return_value

    def test_no_config_raises_with_helpful_message(self):
        """Neither key nor URL → ValueError with guidance."""
        with patch("tools.web_tools.Firecrawl"):
            from tools.web_tools import _get_firecrawl_client
            with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
                _get_firecrawl_client()

    # ── Singleton caching ────────────────────────────────────────────

    def test_singleton_returns_same_instance(self):
        """Second call returns cached client without re-constructing."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                client1 = _get_firecrawl_client()
                client2 = _get_firecrawl_client()
                assert client1 is client2
                mock_fc.assert_called_once()  # constructed only once

    def test_constructor_failure_allows_retry(self):
        """If Firecrawl() raises, next call should retry (not return None)."""
        import tools.web_tools
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                mock_fc.side_effect = [RuntimeError("init failed"), MagicMock()]
                from tools.web_tools import _get_firecrawl_client

                with pytest.raises(RuntimeError):
                    _get_firecrawl_client()

                # Client stayed None, so retry should work
                assert tools.web_tools._firecrawl_client is None
                result = _get_firecrawl_client()
                assert result is not None

    # ── Edge cases ───────────────────────────────────────────────────

    def test_empty_string_key_treated_as_absent(self):
        """FIRECRAWL_API_KEY='' should not be passed as api_key."""
        with patch.dict(os.environ, {
            "FIRECRAWL_API_KEY": "",
            "FIRECRAWL_API_URL": "http://localhost:3002",
        }):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                _get_firecrawl_client()
                # Empty string is falsy, so only api_url should be passed
                mock_fc.assert_called_once_with(api_url="http://localhost:3002")

    def test_empty_string_key_no_url_raises(self):
        """FIRECRAWL_API_KEY='' with no URL → should raise."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": ""}):
            with patch("tools.web_tools.Firecrawl"):
                from tools.web_tools import _get_firecrawl_client
                with pytest.raises(ValueError):
                    _get_firecrawl_client()


class TestBackendSelection:
    """Test suite for _get_backend() backend selection logic.

    The backend is configured via config.yaml (web.backend), set by
    ``hermes tools``.  Falls back to key-based detection for legacy/manual
    setups.
    """

    _ENV_KEYS = ("PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL")

    def setup_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def teardown_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    # ── Config-based selection (web.backend in config.yaml) ───────────

    def test_config_parallel(self):
        """web.backend=parallel in config → 'parallel' regardless of keys."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "parallel"}):
            assert _get_backend() == "parallel"

    def test_config_firecrawl(self):
        """web.backend=firecrawl in config → 'firecrawl' even if Parallel key set."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "firecrawl"

    def test_config_case_insensitive(self):
        """web.backend=Parallel (mixed case) → 'parallel'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "Parallel"}):
            assert _get_backend() == "parallel"

    # ── Fallback (no web.backend in config) ───────────────────────────

    def test_fallback_parallel_only_key(self):
        """Only PARALLEL_API_KEY set → 'parallel'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "parallel"

    def test_fallback_both_keys_defaults_to_firecrawl(self):
        """Both keys set, no config → 'firecrawl' (backward compat)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key", "FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "firecrawl"

    def test_fallback_firecrawl_only_key(self):
        """Only FIRECRAWL_API_KEY set → 'firecrawl'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "firecrawl"

    def test_fallback_no_keys_defaults_to_firecrawl(self):
        """No keys, no config → 'firecrawl' (will fail at client init)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}):
            assert _get_backend() == "firecrawl"

    def test_invalid_config_falls_through_to_fallback(self):
        """web.backend=invalid → ignored, uses key-based fallback."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "tavily"}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "parallel"


class TestParallelClientConfig:
    """Test suite for Parallel client initialization."""

    def setup_method(self):
        import tools.web_tools
        tools.web_tools._parallel_client = None
        os.environ.pop("PARALLEL_API_KEY", None)

    def teardown_method(self):
        import tools.web_tools
        tools.web_tools._parallel_client = None
        os.environ.pop("PARALLEL_API_KEY", None)

    def test_creates_client_with_key(self):
        """PARALLEL_API_KEY set → creates Parallel client."""
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            from tools.web_tools import _get_parallel_client
            from parallel import Parallel
            client = _get_parallel_client()
            assert client is not None
            assert isinstance(client, Parallel)

    def test_no_key_raises_with_helpful_message(self):
        """No PARALLEL_API_KEY → ValueError with guidance."""
        from tools.web_tools import _get_parallel_client
        with pytest.raises(ValueError, match="PARALLEL_API_KEY"):
            _get_parallel_client()

    def test_singleton_returns_same_instance(self):
        """Second call returns cached client."""
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            from tools.web_tools import _get_parallel_client
            client1 = _get_parallel_client()
            client2 = _get_parallel_client()
            assert client1 is client2


class TestCheckWebApiKey:
    """Test suite for check_web_api_key() unified availability check."""

    _ENV_KEYS = ("PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL")

    def setup_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def teardown_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def test_parallel_key_only(self):
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_firecrawl_key_only(self):
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_firecrawl_url_only(self):
        with patch.dict(os.environ, {"FIRECRAWL_API_URL": "http://localhost:3002"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_no_keys_returns_false(self):
        from tools.web_tools import check_web_api_key
        assert check_web_api_key() is False

    def test_both_keys_returns_true(self):
        with patch.dict(os.environ, {
            "PARALLEL_API_KEY": "test-key",
            "FIRECRAWL_API_KEY": "fc-test",
        }):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

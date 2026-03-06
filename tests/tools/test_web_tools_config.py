"""Tests for Firecrawl client configuration and singleton behavior.

Coverage:
  _get_firecrawl_client() — configuration matrix, singleton caching,
  constructor failure recovery, return value verification, edge cases.
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

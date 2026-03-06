"""Tests for Firecrawl client configuration."""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestFirecrawlClientConfig:
    """Test suite for Firecrawl client initialization with API URL support."""

    def teardown_method(self):
        """Reset client between tests."""
        import tools.web_tools

        tools.web_tools._firecrawl_client = None

    def _clear_firecrawl_env(self):
        """Remove Firecrawl env vars so tests start clean."""
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"):
            os.environ.pop(key, None)

    def test_client_with_api_key_only(self):
        """Test client initialization with only API key (cloud mode)."""
        self._clear_firecrawl_env()
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"}, clear=False):
            with patch("tools.web_tools.Firecrawl") as mock_firecrawl:
                from tools.web_tools import _get_firecrawl_client

                _get_firecrawl_client()
                mock_firecrawl.assert_called_once_with(api_key="test-key")

    def test_client_with_api_key_and_url(self):
        """Test client initialization with API key and custom URL."""
        self._clear_firecrawl_env()
        with patch.dict(
            os.environ,
            {
                "FIRECRAWL_API_KEY": "test-key",
                "FIRECRAWL_API_URL": "http://localhost:3002",
            },
            clear=False,
        ):
            with patch("tools.web_tools.Firecrawl") as mock_firecrawl:
                from tools.web_tools import _get_firecrawl_client

                _get_firecrawl_client()
                mock_firecrawl.assert_called_once_with(
                    api_key="test-key", api_url="http://localhost:3002"
                )

    def test_client_with_url_only_no_key(self):
        """Self-hosted mode: URL without API key should work."""
        self._clear_firecrawl_env()
        with patch.dict(
            os.environ,
            {"FIRECRAWL_API_URL": "http://localhost:3002"},
            clear=False,
        ):
            with patch("tools.web_tools.Firecrawl") as mock_firecrawl:
                from tools.web_tools import _get_firecrawl_client

                _get_firecrawl_client()
                mock_firecrawl.assert_called_once_with(
                    api_url="http://localhost:3002"
                )

    def test_no_key_no_url_raises(self):
        """Neither key nor URL set should raise a clear error."""
        self._clear_firecrawl_env()
        with patch("tools.web_tools.Firecrawl"):
            from tools.web_tools import _get_firecrawl_client

            with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
                _get_firecrawl_client()

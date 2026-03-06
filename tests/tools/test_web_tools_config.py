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

    def test_client_with_api_key_only(self):
        """Test client initialization with only API key (no custom URL)."""
        env_vars = {"FIRECRAWL_API_KEY": "test-key"}
        env_vars.pop("FIRECRAWL_API_URL", None)

        with patch.dict(os.environ, env_vars, clear=False):
            # Remove FIRECRAWL_API_URL from env if it exists
            if "FIRECRAWL_API_URL" in os.environ:
                del os.environ["FIRECRAWL_API_URL"]

            with patch("tools.web_tools.Firecrawl") as mock_firecrawl:
                from tools.web_tools import _get_firecrawl_client

                _get_firecrawl_client()
                mock_firecrawl.assert_called_once_with(api_key="test-key")

    def test_client_with_api_key_and_url(self):
        """Test client initialization with API key and custom URL."""
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

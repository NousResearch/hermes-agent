"""Tests for SearXNG web backend integration."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestSearxngSearch:
    def test_search_dispatches_to_searxng(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "url": "https://example.com/docs",
                    "title": "Example Docs",
                    "content": "Useful documentation",
                }
            ]
        }

        with patch("tools.web_tools._get_backend", return_value="searxng"), \
             patch("tools.web_tools._searxng_request", return_value=mock_response.json.return_value), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool

            result = json.loads(web_search_tool("example query", limit=3))

        assert result["success"] is True
        assert result["data"]["web"][0] == {
            "title": "Example Docs",
            "url": "https://example.com/docs",
            "description": "Useful documentation",
            "position": 1,
        }


class TestSearxngExtract:
    def test_extract_fetches_pages_directly(self):
        with patch("tools.web_tools._get_backend", return_value="searxng"), \
             patch("tools.web_tools._searxng_extract", new=AsyncMock(return_value=[
                 {
                     "url": "https://example.com",
                     "title": "Example",
                     "content": "Extracted body text",
                     "raw_content": "Extracted body text",
                     "metadata": {"sourceURL": "https://example.com", "title": "Example"},
                 }
             ])), \
             patch("tools.web_tools.process_content_with_llm", return_value=None):
            from tools.web_tools import web_extract_tool

            result = json.loads(asyncio.get_event_loop().run_until_complete(
                web_extract_tool(["https://example.com"], use_llm_processing=False)
            ))

        assert result["results"][0]["url"] == "https://example.com"
        assert result["results"][0]["title"] == "Example"
        assert result["results"][0]["content"] == "Extracted body text"


class TestSearxngCrawl:
    @pytest.mark.asyncio
    async def test_crawl_returns_unsupported_message(self):
        with patch("tools.web_tools._get_backend", return_value="searxng"):
            from tools.web_tools import web_crawl_tool

            result = json.loads(await web_crawl_tool("https://example.com", use_llm_processing=False))

        assert result["success"] is False
        assert "does not support crawling" in result["error"]
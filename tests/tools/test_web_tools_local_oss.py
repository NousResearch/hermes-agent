"""Tests for native SearXNG + Crawl4AI backend dispatch."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.usefixtures("web_registry_populated")


def test_web_search_dispatches_to_searxng():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Hermes Agent",
                "url": "https://example.com/hermes",
                "content": "Local OSS backend docs",
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("tools.web_tools._load_web_config", return_value={"backend": "local_oss"}), \
         patch("tools.web_tools._ensure_web_plugins_loaded", return_value=None), \
         patch("plugins.web.local_oss.provider.httpx.get", return_value=mock_response) as mock_get, \
         patch("plugins.web.local_oss.provider._get_searxng_api_url", return_value="https://search.example.test"), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        from tools.web_tools import web_search_tool

        result = json.loads(web_search_tool("hermes", limit=3))

    assert result["success"] is True
    assert result["data"]["web"][0]["title"] == "Hermes Agent"
    assert result["data"]["web"][0]["url"] == "https://example.com/hermes"
    mock_get.assert_called_once()


def test_normalize_crawl4ai_document_uses_nested_markdown_content():
    from tools.web_tools import _normalize_crawl4ai_document

    result = _normalize_crawl4ai_document(
        {
            "url": "https://example.com/page",
            "metadata": {"title": "Page"},
            "markdown": {
                "fit_markdown": "Fit markdown",
                "raw_markdown": "Raw markdown",
            },
            "success": True,
        }
    )

    assert result["title"] == "Page"
    assert result["content"] == "Fit markdown"


def test_web_extract_dispatches_to_crawl4ai():
    raw_response = {
        "results": [
            {
                "url": "https://example.com/page",
                "title": "Page",
                "markdown": {
                    "raw_markdown": "Raw markdown",
                },
                "success": True,
            }
        ]
    }
    md_response = {
        "url": "https://example.com/page",
        "markdown": "Extracted markdown",
        "success": True,
    }

    with patch("tools.web_tools._load_web_config", return_value={"extract_backend": "crawl4ai"}), \
         patch("tools.web_tools._ensure_web_plugins_loaded", return_value=None), \
         patch.dict("os.environ", {"CRAWL4AI_API_URL": "https://crawl.example.test"}), \
         patch(
             "plugins.web.local_oss.provider._crawl4ai_post",
             new=AsyncMock(side_effect=[raw_response, md_response]),
         ) as mock_post:
        from tools.web_tools import web_extract_tool

        result = json.loads(
            asyncio.run(
                web_extract_tool(
                    ["https://example.com/page"],
                    use_llm_processing=False,
                )
            )
        )

    assert result["results"][0]["url"] == "https://example.com/page"
    assert result["results"][0]["title"] == "Page"
    assert result["results"][0]["content"] == "Extracted markdown"
    crawl_payload = mock_post.await_args_list[0].args[1]
    md_payload = mock_post.await_args_list[1].args[1]
    assert mock_post.await_args_list[0].args[0] == "/crawl"
    assert crawl_payload["urls"] == ["https://example.com/page"]
    assert crawl_payload["crawler_config"]["cache_mode"] == "bypass"
    assert mock_post.await_args_list[1].args[0] == "/md"
    assert md_payload == {"url": "https://example.com/page", "f": "fit"}


def test_web_crawl_dispatches_to_crawl4ai():
    raw_response = {
        "results": [
            {
                "url": "https://example.com/docs/start",
                "title": "Start",
                "markdown": {
                    "raw_markdown": "Raw doc page",
                },
                "success": True,
            }
        ]
    }
    md_response = {
        "url": "https://example.com/docs/start",
        "markdown": "Doc page",
        "success": True,
    }

    with patch("tools.web_tools._load_web_config", return_value={"crawl_backend": "crawl4ai"}), \
         patch.dict("os.environ", {"CRAWL4AI_API_URL": "https://crawl.example.test"}), \
         patch(
             "plugins.web.local_oss.provider._crawl4ai_post",
             new=AsyncMock(side_effect=[raw_response, md_response]),
         ) as mock_post, \
         patch("tools.web_tools.check_website_access", return_value=None), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        from tools.web_tools import web_crawl_tool

        result = json.loads(
            asyncio.run(
                web_crawl_tool(
                    "https://example.com/docs",
                    depth="advanced",
                    use_llm_processing=False,
                )
            )
        )

    assert result["results"][0]["url"] == "https://example.com/docs/start"
    assert result["results"][0]["title"] == "Start"
    assert result["results"][0]["content"] == "Doc page"
    payload = mock_post.await_args_list[0].args[1]
    strategy = payload["crawler_config"]["deep_crawl_strategy"]
    assert strategy["type"] == "BFSDeepCrawlStrategy"
    assert strategy["params"]["max_depth"] == 2
    assert strategy["params"]["max_pages"] == 20
    assert mock_post.await_args_list[1].args[0] == "/md"

import json
import os
from unittest.mock import MagicMock, patch

import pytest


class TestSearXNGBackendSelection:
    _ENV_KEYS = (
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_GATEWAY_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_SCHEME",
        "TOOL_GATEWAY_USER_TOKEN",
        "TAVILY_API_KEY",
        "SEARXNG_BASE_URL",
    )

    def setup_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def teardown_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def test_search_backend_prefers_explicit_searxng(self):
        from tools.web_tools import _get_search_backend

        with patch("tools.web_tools._load_web_config", return_value={"search_backend": "searxng"}):
            assert _get_search_backend() == "searxng"

    def test_search_backend_chain_honors_explicit_fallback_order(self):
        from tools.web_tools import _get_search_backend_chain

        with patch(
            "tools.web_tools._load_web_config",
            return_value={"search_backend": "searxng", "search_fallback_backends": ["tavily", "exa", "tavily"]},
        ):
            assert _get_search_backend_chain() == ["searxng", "tavily", "exa"]

    def test_search_backend_chain_parses_string_fallback_order(self):
        from tools.web_tools import _get_search_backend_chain

        with patch(
            "tools.web_tools._load_web_config",
            return_value={"search_backend": "searxng", "search_fallback_backends": "tavily, exa, tavily"},
        ):
            assert _get_search_backend_chain() == ["searxng", "tavily", "exa"]

    def test_extract_backend_prefers_explicit_extract_backend(self):
        from tools.web_tools import _get_extract_backend

        with patch("tools.web_tools._load_web_config", return_value={"backend": "searxng", "extract_backend": "exa"}):
            assert _get_extract_backend() == "exa"

    def test_extract_backend_falls_back_from_legacy_searxng_to_tavily(self):
        from tools.web_tools import _get_extract_backend

        with patch("tools.web_tools._load_web_config", return_value={"backend": "searxng"}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
            assert _get_extract_backend() == "tavily"

    def test_crawl_backend_prefers_explicit_crawl_backend(self):
        from tools.web_tools import _get_crawl_backend

        with patch("tools.web_tools._load_web_config", return_value={"backend": "searxng", "crawl_backend": "tavily"}):
            assert _get_crawl_backend() == "tavily"

    def test_check_web_search_api_key_accepts_configured_searxng(self):
        from tools.web_tools import check_web_search_api_key

        with patch("tools.web_tools._load_web_config", return_value={"backend": "searxng"}), \
             patch("tools.web_tools._check_searxng_backend", return_value=True):
            assert check_web_search_api_key() is True

    def test_check_web_search_api_key_accepts_available_fallback_when_primary_is_down(self):
        from tools.web_tools import check_web_search_api_key

        with patch("tools.web_tools._get_search_backend_chain", return_value=["searxng", "tavily", "exa"]), \
             patch("tools.web_tools._is_backend_available", side_effect=lambda backend: backend == "tavily"):
            assert check_web_search_api_key() is True

    def test_check_web_extract_api_key_rejects_search_only_searxng_without_fallback(self):
        from tools.web_tools import check_web_extract_api_key

        with patch("tools.web_tools._load_web_config", return_value={"backend": "searxng"}):
            assert check_web_extract_api_key() is False

    def test_check_web_crawl_api_key_rejects_search_only_searxng_without_fallback(self):
        from tools.web_tools import check_web_crawl_api_key

        with patch("tools.web_tools._load_web_config", return_value={"backend": "searxng"}):
            assert check_web_crawl_api_key() is False


class TestSearXNGSearch:
    def test_searxng_search_normalizes_results(self):
        from tools.web_tools import _searxng_search

        response = MagicMock()
        response.json.return_value = {
            "results": [
                {"title": "One", "url": "https://one.example", "content": "First result"},
                {"title": "Two", "url": "https://two.example", "content": "Second result"},
            ]
        }
        response.raise_for_status = MagicMock()

        with patch("tools.web_tools._load_web_config", return_value={"searxng": {"base_url": "http://localhost:8888", "timeout": 12}}), \
             patch("tools.web_tools.httpx.get", return_value=response) as mock_get:
            result = _searxng_search("OpenAI", limit=2)

        assert result["success"] is True
        assert result["data"]["web"] == [
            {
                "title": "One",
                "url": "https://one.example",
                "description": "First result",
                "position": 1,
            },
            {
                "title": "Two",
                "url": "https://two.example",
                "description": "Second result",
                "position": 2,
            },
        ]
        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["params"]["q"] == "OpenAI"
        assert mock_get.call_args.kwargs["params"]["format"] == "json"
        assert mock_get.call_args.kwargs["timeout"] == 12

    def test_web_search_tool_dispatches_to_searxng(self):
        with patch("tools.web_tools._get_search_backend_chain", return_value=["searxng"]), \
             patch("tools.web_tools._is_backend_available", return_value=True), \
             patch("tools.web_tools._search_with_backend", return_value={
                 "success": True,
                 "data": {
                     "web": [
                         {
                             "title": "Result",
                             "url": "https://example.com",
                             "description": "desc",
                             "position": 1,
                         }
                     ]
                 },
             }), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool

            result = json.loads(web_search_tool("test query", limit=3))

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Result"

    def test_web_search_tool_falls_back_to_tavily_when_searxng_request_fails(self):
        backend_search = MagicMock(side_effect=[
            RuntimeError("SearXNG unavailable"),
            {
                "success": True,
                "data": {
                    "web": [
                        {
                            "title": "Fallback Result",
                            "url": "https://fallback.example.com",
                            "description": "desc",
                            "position": 1,
                        }
                    ]
                },
            },
        ])

        with patch("tools.web_tools._get_search_backend_chain", return_value=["searxng", "tavily"]), \
             patch("tools.web_tools._is_backend_available", return_value=True), \
             patch("tools.web_tools._search_with_backend", backend_search), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool

            result = json.loads(web_search_tool("test query", limit=3))

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Fallback Result"
        assert backend_search.call_args_list[0].args == ("searxng", "test query", 3)
        assert backend_search.call_args_list[1].args == ("tavily", "test query", 3)

    def test_web_search_tool_returns_helpful_error_when_all_search_backends_fail(self):
        backend_search = MagicMock(side_effect=RuntimeError("Tavily failed"))

        with patch("tools.web_tools._get_search_backend_chain", return_value=["searxng", "tavily"]), \
             patch("tools.web_tools._is_backend_available", side_effect=lambda backend: backend == "tavily"), \
             patch("tools.web_tools._search_with_backend", backend_search), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool

            result = json.loads(web_search_tool("test query", limit=3))

        assert "searxng" in result["error"]
        assert "tavily" in result["error"]

    @pytest.mark.asyncio
    async def test_web_extract_tool_returns_helpful_error_when_no_extract_backend_exists(self):
        with patch("tools.web_tools._get_extract_backend", return_value=""), \
             patch("tools.web_tools._load_web_config", return_value={"backend": "searxng"}):
            from tools.web_tools import web_extract_tool

            result = json.loads(await web_extract_tool(["https://example.com"], use_llm_processing=False))

        assert result["success"] is False
        assert "SearXNG supports search only" in result["error"]

    @pytest.mark.asyncio
    async def test_web_crawl_tool_returns_helpful_error_when_no_crawl_backend_exists(self):
        with patch("tools.web_tools._get_crawl_backend", return_value=""), \
             patch("tools.web_tools._get_backend", return_value="searxng"):
            from tools.web_tools import web_crawl_tool

            result = json.loads(await web_crawl_tool("https://example.com", use_llm_processing=False))

        assert result["success"] is False
        assert "SearXNG supports search only" in result["error"]

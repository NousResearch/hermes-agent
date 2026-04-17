"""Regression tests for runtime web backend failover.

Covers:
- retryable search failures fail over to the next backend
- non-retryable search failures do not silently hop providers
- retryable extract failures fail over to the next backend
- backend error classification
"""

import json
from unittest.mock import AsyncMock, patch

import pytest


class TestRetryableErrorClassification:
    def test_retryable_credit_error(self):
        from tools.web_tools import _is_retryable_backend_error

        assert _is_retryable_backend_error(RuntimeError("402 Insufficient credits")) is True

    def test_non_retryable_bad_request_error(self):
        from tools.web_tools import _is_retryable_backend_error

        assert _is_retryable_backend_error(RuntimeError("400 bad request: invalid query")) is False


class TestWebSearchFailover:
    def test_explicit_backend_disables_cross_provider_failover(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}), \
             patch("tools.web_tools._is_backend_available", return_value=True):
            from tools.web_tools import _get_backend_fallback_chain

            assert _get_backend_fallback_chain("firecrawl") == ["firecrawl"]

    def test_search_fails_over_on_retryable_backend_error(self):
        with patch("tools.web_tools._get_backend", return_value="firecrawl"), \
             patch("tools.web_tools._get_backend_fallback_chain", return_value=["firecrawl", "exa"]), \
             patch(
                 "tools.web_tools._dispatch_web_search_backend",
                 side_effect=[
                     RuntimeError("402 Insufficient credits"),
                     {
                         "success": True,
                         "data": {"web": [{"title": "Recovered", "url": "https://example.com", "description": "ok"}]},
                     },
                 ],
             ) as mock_dispatch, \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool

            result = json.loads(web_search_tool("fallback query", limit=3))

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Recovered"
        assert mock_dispatch.call_count == 2
        assert mock_dispatch.call_args_list[0].args[0] == "firecrawl"
        assert mock_dispatch.call_args_list[1].args[0] == "exa"

    def test_search_does_not_failover_on_non_retryable_error(self):
        with patch("tools.web_tools._get_backend", return_value="firecrawl"), \
             patch("tools.web_tools._get_backend_fallback_chain", return_value=["firecrawl", "exa"]), \
             patch(
                 "tools.web_tools._dispatch_web_search_backend",
                 side_effect=RuntimeError("400 bad request: invalid query"),
             ) as mock_dispatch, \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool

            result = json.loads(web_search_tool("bad query", limit=3))

        assert "error" in result
        assert "invalid query" in result["error"].lower()
        assert mock_dispatch.call_count == 1


class TestWebExtractFailover:
    def test_all_error_extract_results_trigger_failover(self):
        from tools.web_tools import _extract_results_should_failover

        assert _extract_results_should_failover([
            {"url": "https://example.com", "error": "402 Insufficient credits", "content": ""},
            {"url": "https://example.org", "error": "429 Too Many Requests", "content": ""},
        ]) is True

    def test_partial_extract_results_do_not_trigger_failover(self):
        from tools.web_tools import _extract_results_should_failover

        assert _extract_results_should_failover([
            {"url": "https://example.com", "error": "402 Insufficient credits", "content": ""},
            {"url": "https://example.org", "error": None, "content": "Recovered body"},
        ]) is False

    @pytest.mark.asyncio
    async def test_policy_blocked_url_never_reaches_backend_dispatch(self):
        with patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value={"host": "blocked.example", "rule": "deny", "source": "test", "message": "Blocked by policy"}), \
             patch("tools.web_tools._dispatch_web_extract_backend", new=AsyncMock()) as mock_dispatch:
            from tools.web_tools import web_extract_tool

            result = json.loads(await web_extract_tool(["https://blocked.example"], use_llm_processing=False))

        assert result["results"][0]["error"] == "Blocked by policy"
        mock_dispatch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_extract_fails_over_on_retryable_backend_error(self):
        with patch("tools.web_tools._get_backend", return_value="firecrawl"), \
             patch("tools.web_tools._get_backend_fallback_chain", return_value=["firecrawl", "exa"]), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch(
                 "tools.web_tools._dispatch_web_extract_backend",
                 new=AsyncMock(side_effect=[
                     RuntimeError("402 Insufficient credits"),
                     [{"url": "https://example.com", "title": "Recovered", "content": "Extracted body"}],
                 ]),
             ) as mock_dispatch:
            from tools.web_tools import web_extract_tool

            result = json.loads(await web_extract_tool(["https://example.com"], use_llm_processing=False))

        assert result["results"][0]["title"] == "Recovered"
        assert result["results"][0]["content"] == "Extracted body"
        assert mock_dispatch.await_count == 2
        assert mock_dispatch.await_args_list[0].args[0] == "firecrawl"
        assert mock_dispatch.await_args_list[1].args[0] == "exa"

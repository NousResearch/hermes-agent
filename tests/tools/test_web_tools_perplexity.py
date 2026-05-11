"""Tests for the Perplexity web search backend.

Coverage:
  tools.web_providers.perplexity._get_perplexity_api_key  — env name handling.
  tools.web_providers.perplexity._perplexity_search_request — headers, payload, errors.
  tools.web_providers.perplexity._normalize_perplexity_results — shape adapter.
  tools.web_providers.perplexity.perplexity_search — limit clamping + interrupt path.
  tools.web_providers.perplexity.PerplexitySearchProvider — plugin provider contract.
  tools.web_tools._get_backend / _is_backend_available — Perplexity is the
    new shipped default and a recognised configured backend.
  tools.web_tools.web_search_tool — dispatches to the registered Perplexity backend.
"""

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest


# ─── _get_perplexity_api_key ───────────────────────────────────────────────────

class TestGetPerplexityApiKey:
    """Auth env resolution."""

    def test_raises_when_neither_env_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PERPLEXITY_API_KEY", None)
            os.environ.pop("PPLX_API_KEY", None)
            from tools.web_providers.perplexity import _get_perplexity_api_key
            with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
                _get_perplexity_api_key()

    def test_prefers_perplexity_api_key(self):
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-primary", "PPLX_API_KEY": "pplx-alias"}):
            from tools.web_providers.perplexity import _get_perplexity_api_key
            assert _get_perplexity_api_key() == "pplx-primary"

    def test_falls_back_to_pplx_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PERPLEXITY_API_KEY", None)
            os.environ["PPLX_API_KEY"] = "pplx-alias"
            from tools.web_providers.perplexity import _get_perplexity_api_key
            assert _get_perplexity_api_key() == "pplx-alias"

    def test_strips_whitespace(self):
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "  pplx-test  "}):
            from tools.web_providers.perplexity import _get_perplexity_api_key
            assert _get_perplexity_api_key() == "pplx-test"


# ─── _perplexity_search_request ────────────────────────────────────────────────

class TestPerplexitySearchRequest:
    """HTTP request shape: URL, auth header, attribution header, error propagation."""

    def test_posts_with_bearer_auth_and_integration_header(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-test"}):
            with patch("tools.web_providers.perplexity.httpx.post", return_value=mock_response) as mock_post:
                from tools.web_providers.perplexity import _perplexity_search_request
                result = _perplexity_search_request({"query": "hello", "max_results": 5})

                assert result == {"results": []}
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                # URL
                assert args[0] == "https://api.perplexity.ai/search"
                # Headers
                headers = kwargs.get("headers", {})
                assert headers.get("Authorization") == "Bearer pplx-test"
                assert headers.get("Content-Type") == "application/json"
                assert headers.get("X-Pplx-Integration", "").startswith("hermes-agent/")
                # Payload
                assert kwargs.get("json") == {"query": "hello", "max_results": 5}

    def test_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=mock_response
        )

        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-bad"}):
            with patch("tools.web_providers.perplexity.httpx.post", return_value=mock_response):
                from tools.web_providers.perplexity import _perplexity_search_request
                with pytest.raises(httpx.HTTPStatusError):
                    _perplexity_search_request({"query": "x"})


# ─── _normalize_perplexity_results ─────────────────────────────────────────────

class TestNormalizePerplexityResults:
    """Response shape adapter — mirror what web_search consumers expect."""

    def test_basic_normalization(self):
        from tools.web_providers.perplexity import _normalize_perplexity_results
        raw = {
            "results": [
                {"title": "Python Docs", "url": "https://docs.python.org", "snippet": "Official docs"},
                {"title": "Tutorial", "url": "https://example.com", "snippet": "A tutorial"},
            ]
        }
        out = _normalize_perplexity_results(raw)
        assert out["success"] is True
        web = out["data"]["web"]
        assert len(web) == 2
        assert web[0] == {
            "url": "https://docs.python.org",
            "title": "Python Docs",
            "description": "Official docs",
            "position": 1,
        }
        assert web[1]["position"] == 2

    def test_handles_content_field_when_snippet_missing(self):
        from tools.web_providers.perplexity import _normalize_perplexity_results
        raw = {"results": [{"title": "T", "url": "https://e.com", "content": "long body"}]}
        out = _normalize_perplexity_results(raw)
        assert out["data"]["web"][0]["description"] == "long body"

    def test_handles_empty_results(self):
        from tools.web_providers.perplexity import _normalize_perplexity_results
        out = _normalize_perplexity_results({"results": []})
        assert out == {"success": True, "data": {"web": []}}

    def test_missing_fields_default_to_empty(self):
        from tools.web_providers.perplexity import _normalize_perplexity_results
        out = _normalize_perplexity_results({"results": [{}]})
        item = out["data"]["web"][0]
        assert item == {"url": "", "title": "", "description": "", "position": 1}


# ─── perplexity_search (top-level entrypoint) ──────────────────────────────────

class TestPerplexitySearch:
    """End-to-end perplexity_search behaviour."""

    def test_clamps_limit_to_max_20(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-test"}):
            with patch("tools.web_providers.perplexity.httpx.post", return_value=mock_response) as mock_post:
                from tools.web_providers.perplexity import perplexity_search
                perplexity_search("hi", limit=500)
                payload = mock_post.call_args.kwargs["json"]
                assert payload["max_results"] == 20

    def test_clamps_limit_to_min_1(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-test"}):
            with patch("tools.web_providers.perplexity.httpx.post", return_value=mock_response) as mock_post:
                from tools.web_providers.perplexity import perplexity_search
                perplexity_search("hi", limit=0)
                payload = mock_post.call_args.kwargs["json"]
                assert payload["max_results"] == 1

    def test_returns_normalized_results(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"title": "T", "url": "https://e.com", "snippet": "S"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-test"}):
            with patch("tools.web_providers.perplexity.httpx.post", return_value=mock_response):
                from tools.web_providers.perplexity import perplexity_search
                out = perplexity_search("hi", limit=5)
                assert out["success"] is True
                assert out["data"]["web"][0]["title"] == "T"
                assert out["data"]["web"][0]["position"] == 1

    def test_returns_interrupted_when_interrupted(self):
        with patch("tools.interrupt.is_interrupted", return_value=True):
            from tools.web_providers.perplexity import perplexity_search
            out = perplexity_search("hi", limit=5)
            assert out == {"error": "Interrupted", "success": False}


# ─── PerplexitySearchProvider ────────────────────────────────────────────────

class TestPerplexitySearchProvider:
    """Plugin-facing provider contract."""

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from tools.web_providers.perplexity import PerplexitySearchProvider

        provider = PerplexitySearchProvider()
        assert isinstance(provider, WebSearchProvider)
        assert provider.name == "perplexity"
        assert provider.display_name == "Perplexity"
        assert provider.supports_search() is True
        assert provider.supports_extract() is False
        assert provider.supports_crawl() is False

    def test_setup_schema_prompts_for_perplexity_api_key(self):
        from tools.web_providers.perplexity import PerplexitySearchProvider

        schema = PerplexitySearchProvider().get_setup_schema()
        assert schema["name"] == "Perplexity"
        assert schema["env_vars"][0]["key"] == "PERPLEXITY_API_KEY"

    def test_returns_error_dict_when_unconfigured(self):
        from tools.web_providers.perplexity import PerplexitySearchProvider

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PERPLEXITY_API_KEY", None)
            os.environ.pop("PPLX_API_KEY", None)
            result = PerplexitySearchProvider().search("hi", limit=5)

        assert result["success"] is False
        assert "PERPLEXITY_API_KEY" in result["error"]


# ─── web_tools.py backend integration ──────────────────────────────────────────

class TestWebToolsPerplexityBackendSelection:
    """Verify _get_backend / _is_backend_available recognise Perplexity."""

    def _clean_env(self) -> dict:
        """Return env with all web backend vars cleared so we can isolate Perplexity."""
        cleared = {
            "PERPLEXITY_API_KEY": "",
            "PPLX_API_KEY": "",
            "FIRECRAWL_API_KEY": "",
            "FIRECRAWL_API_URL": "",
            "PARALLEL_API_KEY": "",
            "TAVILY_API_KEY": "",
            "EXA_API_KEY": "",
            "SEARXNG_URL": "",
            "BRAVE_SEARCH_API_KEY": "",
        }
        return cleared

    def test_configured_perplexity_is_honored(self):
        env = self._clean_env()
        env["PERPLEXITY_API_KEY"] = "pplx-test"
        with patch.dict(os.environ, env, clear=False):
            with patch("tools.web_tools._load_web_config", return_value={"backend": "perplexity"}):
                from tools.web_tools import _get_backend
                assert _get_backend() == "perplexity"

    def test_auto_detected_when_only_perplexity_key_present(self):
        env = self._clean_env()
        env["PERPLEXITY_API_KEY"] = "pplx-test"
        with patch.dict(os.environ, env, clear=False):
            with patch("tools.web_tools._load_web_config", return_value={}):
                with patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
                    with patch("tools.web_tools._ddgs_package_importable", return_value=False):
                        from tools.web_tools import _get_backend
                        assert _get_backend() == "perplexity"

    def test_pplx_api_key_alias_is_accepted(self):
        env = self._clean_env()
        env["PPLX_API_KEY"] = "pplx-test"
        with patch.dict(os.environ, env, clear=False):
            with patch("tools.web_tools._load_web_config", return_value={}):
                with patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
                    with patch("tools.web_tools._ddgs_package_importable", return_value=False):
                        from tools.web_tools import _get_backend
                        assert _get_backend() == "perplexity"

    def test_perplexity_is_default_when_no_keys_present(self):
        env = self._clean_env()
        with patch.dict(os.environ, env, clear=False):
            with patch("tools.web_tools._load_web_config", return_value={}):
                with patch("tools.web_tools._is_tool_gateway_ready", return_value=False):
                    with patch("tools.web_tools._ddgs_package_importable", return_value=False):
                        from tools.web_tools import _get_backend
                        # No keys at all → fallback default is now perplexity.
                        assert _get_backend() == "perplexity"

    def test_is_backend_available_for_perplexity(self):
        from tools.web_tools import _is_backend_available
        env = self._clean_env()
        env["PERPLEXITY_API_KEY"] = "pplx-test"
        with patch.dict(os.environ, env, clear=False):
            assert _is_backend_available("perplexity") is True

    def test_is_backend_unavailable_when_no_key(self):
        from tools.web_tools import _is_backend_available
        env = self._clean_env()
        with patch.dict(os.environ, env, clear=False):
            assert _is_backend_available("perplexity") is False


class TestWebSearchToolDispatch:
    """web_search_tool routes to the registered Perplexity provider."""

    def test_web_search_tool_calls_perplexity_provider(self):
        # Patch the provider class so we don't make a real HTTP call.
        from tools.web_providers import perplexity as pplx_mod

        captured = {}

        def fake_search(self, query, limit=10):
            captured["query"] = query
            captured["limit"] = limit
            return {"success": True, "data": {"web": [{"url": "https://x.test", "title": "X",
                                                       "description": "y", "position": 1}]}}

        with patch.object(pplx_mod.PerplexitySearchProvider, "search", fake_search):
            from hermes_cli.plugins import _ensure_plugins_discovered

            _ensure_plugins_discovered()
            with patch("tools.web_tools._get_search_backend", return_value="perplexity"):
                from tools.web_tools import web_search_tool
                # Call the underlying function — web_search_tool is registered via @tool.
                result_raw = web_search_tool.fn(query="hermes agent", limit=7) \
                    if hasattr(web_search_tool, "fn") else web_search_tool("hermes agent", 7)
                result = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
                assert result["success"] is True
                assert result["data"]["web"][0]["url"] == "https://x.test"
                assert captured["query"] == "hermes agent"
                assert captured["limit"] == 7

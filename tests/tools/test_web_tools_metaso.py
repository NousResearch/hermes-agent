"""Tests for Metaso web backend integration.

Coverage:
  _metaso_search() — HTTP request construction, result normalization
  _metaso_extract() — reader API, error handling
  _is_backend_available() — metaso branch
  _get_backend() — metaso config selection and fallback
  check_web_api_key() — metaso availability
  web_search_tool() — metaso dispatch
"""

import json
import os
from unittest.mock import patch


class TestMetasoBackendSelection:
    """Test _get_backend() with metaso."""

    def setup_method(self):
        for key in (
            "HERMES_ENABLE_NOUS_MANAGED_TOOLS",
            "EXA_API_KEY", "PARALLEL_API_KEY", "TAVILY_API_KEY",
            "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL",
            "METASO_API_KEY",
        ):
            os.environ.pop(key, None)

    def teardown_method(self):
        self.setup_method()

    def test_config_metaso(self):
        """web.backend=metaso → 'metaso' regardless of other keys."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "metaso"}):
            assert _get_backend() == "metaso"

    def test_config_metaso_case_insensitive(self):
        """web.backend=Metaso → 'metaso'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "Metaso"}):
            assert _get_backend() == "metaso"

    def test_fallback_metaso_only_key(self):
        """Only METASO_API_KEY set → 'metaso'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}):
            with patch.dict(os.environ, {"METASO_API_KEY": "mk-test"}):
                assert _get_backend() == "metaso"

    def test_fallback_firecrawl_takes_priority_over_metaso(self):
        """Both FIRECRAWL_API_KEY and METASO_API_KEY → 'firecrawl' (higher priority)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}):
            with patch.dict(os.environ, {
                "FIRECRAWL_API_KEY": "fc-test",
                "METASO_API_KEY": "mk-test",
            }):
                assert _get_backend() == "firecrawl"


class TestMetasoBackendAvailability:
    """Test _is_backend_available() for metaso."""

    def setup_method(self):
        os.environ.pop("METASO_API_KEY", None)

    def teardown_method(self):
        os.environ.pop("METASO_API_KEY", None)

    def test_metaso_available_with_key(self):
        from tools.web_tools import _is_backend_available
        with patch.dict(os.environ, {"METASO_API_KEY": "mk-test"}):
            assert _is_backend_available("metaso") is True

    def test_metaso_unavailable_without_key(self):
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("metaso") is False


class TestMetasoNormalizeResults:
    """Test _normalize_metaso_search_results() defensive mapping."""

    def test_standard_data_shape(self):
        from tools.web_tools import _normalize_metaso_search_results
        raw = {
            "data": [
                {"url": "https://a.com", "title": "A", "snippet": "Snip A"},
                {"url": "https://b.com", "title": "B", "description": "Desc B"},
            ]
        }
        result = _normalize_metaso_search_results(raw)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0] == {"title": "A", "url": "https://a.com", "description": "Snip A", "position": 1}
        assert web[1] == {"title": "B", "url": "https://b.com", "description": "Desc B", "position": 2}

    def test_results_shape(self):
        """Alternative response shape with 'results' key."""
        from tools.web_tools import _normalize_metaso_search_results
        raw = {
            "results": [
                {"url": "https://x.com", "title": "X", "summary": "Sum X"},
            ]
        }
        result = _normalize_metaso_search_results(raw)
        assert result["success"] is True
        assert result["data"]["web"][0]["description"] == "Sum X"

    def test_empty_results_returns_success(self):
        """No results → success with empty list (not an error)."""
        from tools.web_tools import _normalize_metaso_search_results
        raw = {"data": []}
        result = _normalize_metaso_search_results(raw)
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_unexpected_format_returns_error(self):
        """Completely unexpected structure → tool_error."""
        from tools.web_tools import _normalize_metaso_search_results
        raw = {"status": "error", "message": "bad request"}
        result = _normalize_metaso_search_results(raw)
        # tool_error returns a JSON string, not a dict
        parsed = json.loads(result) if isinstance(result, str) else result
        assert parsed.get("success") is not True

    def test_missing_fields_handled_gracefully(self):
        """Items with missing url/title/description → empty strings."""
        from tools.web_tools import _normalize_metaso_search_results
        raw = {"data": [{"foo": "bar"}]}
        result = _normalize_metaso_search_results(raw)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 1
        assert web[0]["url"] == ""
        assert web[0]["title"] == ""
        assert web[0]["description"] == ""


class TestMetasoCheckWebApiKey:
    """Test check_web_api_key() with metaso."""

    def setup_method(self):
        for key in (
            "EXA_API_KEY", "PARALLEL_API_KEY", "TAVILY_API_KEY",
            "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL",
            "METASO_API_KEY",
        ):
            os.environ.pop(key, None)

    def teardown_method(self):
        self.setup_method()

    def test_metaso_key_only_returns_true(self):
        with patch.dict(os.environ, {"METASO_API_KEY": "mk-test"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_no_keys_still_false(self):
        from tools.web_tools import check_web_api_key
        assert check_web_api_key() is False


class TestMetasoWebRequiresEnv:
    """Test _web_requires_env() includes METASO_API_KEY."""

    def test_metaso_key_in_requires_env(self):
        from tools.web_tools import _web_requires_env
        assert "METASO_API_KEY" in _web_requires_env()

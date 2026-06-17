"""Tests for the iFlow Search web provider."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.conftest import register_all_web_providers


def _clear_iflow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IFLOW_API_KEY", "")
    monkeypatch.setenv("IFLOW_BASE_URL", "")


class TestIFlowProviderAvailability:
    def test_provider_name_and_capabilities(self):
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        provider = IFlowWebSearchProvider()
        assert provider.name == "iflow"
        assert provider.display_name == "iFlow Search"
        assert provider.supports_search() is True
        assert provider.supports_extract() is True

    def test_available_when_key_set(self, monkeypatch):
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        assert IFlowWebSearchProvider().is_available() is True

    def test_unavailable_when_key_missing(self, monkeypatch):
        _clear_iflow_env(monkeypatch)
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        assert IFlowWebSearchProvider().is_available() is False

    def test_default_base_url(self, monkeypatch):
        _clear_iflow_env(monkeypatch)
        from plugins.web.iflow.provider import DEFAULT_IFLOW_BASE_URL, _iflow_base_url

        assert _iflow_base_url() == DEFAULT_IFLOW_BASE_URL

    def test_base_url_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("IFLOW_BASE_URL", "https://example.test///")
        from plugins.web.iflow.provider import _iflow_base_url

        assert _iflow_base_url() == "https://example.test"


class TestIFlowSearch:
    @staticmethod
    def _mock_resp(json_data, status_code=200):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = json_data
        response.raise_for_status = MagicMock()
        return response

    def test_missing_key_returns_error(self, monkeypatch):
        _clear_iflow_env(monkeypatch)
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        result = IFlowWebSearchProvider().search("test", limit=5)
        assert result["success"] is False
        assert "IFLOW_API_KEY is not set" in result["error"]
        assert "~/.hermes" not in result["error"]

    def test_web_search_success_normalizes_results(self, monkeypatch):
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        monkeypatch.setenv("IFLOW_BASE_URL", "https://platform.iflow.cn/")
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        sample = {
            "success": True,
            "code": "200",
            "message": "ok",
            "data": {
                "organic": [
                    {
                        "title": "A",
                        "link": "https://a.example",
                        "snippet": "desc A",
                        "position": 1,
                        "date": "2026-01-01",
                    },
                    {
                        "title": "B",
                        "url": "https://b.example",
                        "content": "desc B",
                    },
                ]
            },
        }
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs["json"]
            captured["headers"] = kwargs["headers"]
            return self._mock_resp(sample)

        with patch("httpx.post", side_effect=fake_post):
            result = IFlowWebSearchProvider().search("hello", limit=2)

        assert captured["url"] == "https://platform.iflow.cn/api/search/webSearch"
        assert captured["json"] == {"keywords": "hello", "num": 2}
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://a.example"
        assert result["data"]["web"][0]["description"] == "desc A"
        assert result["data"]["web"][0]["metadata"]["published_time"] == "2026-01-01"
        assert result["data"]["web"][1]["position"] == 2

    def test_limit_is_clamped(self, monkeypatch):
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs["json"]
            return self._mock_resp({"success": True, "code": "200", "data": {"organic": []}})

        with patch("httpx.post", side_effect=fake_post):
            IFlowWebSearchProvider().search("hello", limit=1000)

        assert captured["json"]["num"] == 100

    def test_http_error_is_sanitized(self, monkeypatch):
        import httpx

        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        response = self._mock_resp({}, status_code=401)
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "unauthorized",
            request=MagicMock(),
            response=response,
        )

        with patch("httpx.post", return_value=response):
            result = IFlowWebSearchProvider().search("hello")

        assert result["success"] is False
        assert "invalid or unauthorized" in result["error"]
        assert "test-key" not in result["error"]

    def test_bad_json_returns_invalid_response(self, monkeypatch):
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        response = self._mock_resp({})
        response.json.side_effect = ValueError("not json")

        with patch("httpx.post", return_value=response):
            result = IFlowWebSearchProvider().search("hello")

        assert result["success"] is False
        assert "invalid JSON response" in result["error"]

    def test_business_error(self, monkeypatch):
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        response = self._mock_resp(
            {"success": False, "code": "40101", "message": "bad credentials"}
        )

        with patch("httpx.post", return_value=response):
            result = IFlowWebSearchProvider().search("hello")

        assert result["success"] is False
        assert "40101" in result["error"]
        assert "bad credentials" in result["error"]


class TestIFlowExtract:
    @staticmethod
    def _mock_resp(json_data, status_code=200):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = json_data
        response.raise_for_status = MagicMock()
        return response

    def test_web_fetch_success_normalizes_document(self, monkeypatch):
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        sample = {
            "success": True,
            "code": "200",
            "data": {
                "title": "Example",
                "content": "Page content",
                "url": "https://example.com/",
                "fromCache": True,
            },
        }

        with patch("httpx.post", return_value=self._mock_resp(sample)):
            result = IFlowWebSearchProvider().extract(["https://example.com"])

        assert result == [
            {
                "url": "https://example.com/",
                "title": "Example",
                "content": "Page content",
                "raw_content": "Page content",
                "metadata": {
                    "sourceURL": "https://example.com/",
                    "title": "Example",
                    "fromCache": True,
                },
            }
        ]

    def test_web_fetch_error_returns_per_url_error(self, monkeypatch):
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        from plugins.web.iflow.provider import IFlowWebSearchProvider

        with patch(
            "httpx.post",
            return_value=self._mock_resp(
                {"success": False, "code": "50001", "message": "fetch failed"}
            ),
        ):
            result = IFlowWebSearchProvider().extract(["https://example.com"])

        assert result[0]["url"] == "https://example.com"
        assert result[0]["content"] == ""
        assert "fetch failed" in result[0]["error"]


class TestIFlowBackendWiring:
    def test_backend_available_true_when_key_set(self, monkeypatch):
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        from tools.web_tools import _is_backend_available

        assert _is_backend_available("iflow") is True

    def test_backend_available_false_when_key_missing(self, monkeypatch):
        _clear_iflow_env(monkeypatch)
        from tools.web_tools import _is_backend_available

        assert _is_backend_available("iflow") is False

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "iflow"})
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        assert web_tools._get_backend() == "iflow"

    def test_auto_detect_picks_iflow_when_only_key_set(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in (
            "FIRECRAWL_API_KEY",
            "FIRECRAWL_API_URL",
            "PARALLEL_API_KEY",
            "TAVILY_API_KEY",
            "EXA_API_KEY",
            "SEARXNG_URL",
            "BRAVE_SEARCH_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "iflow"

    def test_check_web_api_key_true_when_iflow_configured(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "iflow"})
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        assert web_tools.check_web_api_key() is True

    def test_web_search_tool_dispatches_iflow(self, monkeypatch):
        from tools import web_tools

        register_all_web_providers()
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "iflow"})
        monkeypatch.setenv("IFLOW_API_KEY", "test-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        with patch(
            "httpx.post",
            return_value=TestIFlowSearch._mock_resp(
                {"success": True, "code": "200", "data": {"organic": []}}
            ),
        ):
            result = json.loads(web_tools.web_search_tool("hello", limit=1))

        assert result == {"success": True, "data": {"web": []}}

    def test_tools_config_provider_row_is_visible(self, monkeypatch):
        from hermes_cli import tools_config

        monkeypatch.setenv("IFLOW_API_KEY", "")
        rows = tools_config._plugin_web_search_providers()
        row = next((r for r in rows if r.get("web_backend") == "iflow"), None)

        assert row is not None
        assert row["name"] == "iFlow Search"
        assert row["badge"] == "paid"
        assert row["tag"] == "Search + extract in one provider."
        assert [e["key"] for e in row["env_vars"]] == ["IFLOW_API_KEY"]


@pytest.mark.integration
class TestIFlowLiveIntegration:
    def test_live_web_search_when_key_present(self):
        if not os.getenv("IFLOW_API_KEY"):
            pytest.skip("IFLOW_API_KEY is not set")

        from plugins.web.iflow.provider import IFlowWebSearchProvider

        result = IFlowWebSearchProvider().search("Hermes Agent", limit=1)
        assert result["success"] is True
        assert "web" in result["data"]

    def test_live_web_fetch_when_key_present(self):
        if not os.getenv("IFLOW_API_KEY"):
            pytest.skip("IFLOW_API_KEY is not set")

        from plugins.web.iflow.provider import IFlowWebSearchProvider

        result = IFlowWebSearchProvider().extract(["https://example.com"])
        assert len(result) == 1
        assert result[0]["url"]
        assert "content" in result[0]

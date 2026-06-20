from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestYouProvider:
    def test_is_available_accepts_you_or_ydc_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.you.provider import YouWebSearchProvider

        p = YouWebSearchProvider()
        assert p.is_available() is False

        monkeypatch.setenv("YOU_API_KEY", "real")
        assert p.is_available() is True

        monkeypatch.delenv("YOU_API_KEY", raising=False)
        monkeypatch.setenv("YDC_API_KEY", "real")
        assert p.is_available() is True

    def test_search_normalizes_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.you.provider import YouWebSearchProvider

        monkeypatch.setenv("YDC_API_KEY", "ydc-test")
        provider = YouWebSearchProvider()

        response = MagicMock()
        response.json.return_value = {
            "results": {
                "web": [
                    {
                        "title": "Hermes Agent",
                        "url": "https://hermes-agent.nousresearch.com",
                        "description": "Hermes docs",
                    },
                    {
                        "title": "Nous Research",
                        "url": "https://nousresearch.com",
                        "description": "Nous homepage",
                    },
                ]
            }
        }
        response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=response) as mocked_get:
            result = provider.search("hermes agent", limit=2)

        assert result["success"] is True
        assert result["data"]["web"] == [
            {
                "title": "Hermes Agent",
                "url": "https://hermes-agent.nousresearch.com",
                "description": "Hermes docs",
                "position": 1,
            },
            {
                "title": "Nous Research",
                "url": "https://nousresearch.com",
                "description": "Nous homepage",
                "position": 2,
            },
        ]
        mocked_get.assert_called_once()
        assert mocked_get.call_args.kwargs["headers"]["X-API-Key"] == "ydc-test"
        assert mocked_get.call_args.kwargs["params"]["query"] == "hermes agent"
        assert mocked_get.call_args.kwargs["params"]["count"] == 2

    def test_search_returns_typed_error_when_unconfigured(self) -> None:
        from plugins.web.you.provider import YouWebSearchProvider

        result = YouWebSearchProvider().search("test", limit=5)
        assert result["success"] is False
        assert "error" in result


class TestYouProviderIntegration:
    def test_plugin_discovery_registers_you_provider(self) -> None:
        from agent.web_search_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        provider = get_provider("you")

        assert provider is not None
        assert provider.name == "you"
        assert provider.supports_search() is True
        assert provider.supports_extract() is False

    def test_check_web_api_key_accepts_you_as_configured_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools import web_tools

        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("YDC_API_KEY", "ydc-test")
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "you"})

        assert web_tools.check_web_api_key() is True

    def test_check_web_api_key_accepts_you_in_autodetect_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools import web_tools

        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("YDC_API_KEY", "ydc-test")
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})

        assert web_tools.check_web_api_key() is True

    def test_backend_autodetect_prefers_you_over_ddgs_when_you_key_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools import web_tools

        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("YDC_API_KEY", "ydc-test")
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})

        assert web_tools._get_search_backend() == "you"

    def test_backend_autodetect_keeps_tavily_ahead_of_you(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools import web_tools

        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("YDC_API_KEY", "ydc-test")
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})

        assert web_tools._get_search_backend() == "tavily"

    def test_backend_autodetect_keeps_firecrawl_ahead_of_you(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools import web_tools

        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        monkeypatch.setenv("YDC_API_KEY", "ydc-test")
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})

        assert web_tools._get_search_backend() == "firecrawl"

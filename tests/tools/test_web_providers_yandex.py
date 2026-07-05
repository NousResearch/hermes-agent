"""Tests for the Yandex Search API web search provider.

Covers:
- YandexWebSearchProvider.is_available() env var gating (API key + folder ID)
- YandexWebSearchProvider.search() — happy path (base64+XML decode/parse),
  HTTP error, request error, malformed response, Yandex-reported <error>
- Result normalization (title, url, description, position) incl. <hlword>
  emphasis-tag stripping and <headline>/<passage> description fallback
- Limit truncation via groupsOnPage
- _is_backend_available("yandex") integration
- _get_backend() recognizes "yandex" as a valid configured backend
- web_extract returns a search-only error when yandex is active
"""
from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.conftest import register_all_web_providers


def _b64_xml(xml_body: str) -> str:
    return base64.b64encode(xml_body.encode("utf-8")).decode("ascii")


_SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<yandexsearch version="1.0">
<response>
<results>
<grouping>
<group>
<doc>
<url>https://a.example.com/</url>
<domain>a.example.com</domain>
<title>Title A</title>
<headline>Headline for A</headline>
<passages><passage>Passage A</passage></passages>
</doc>
</group>
<group>
<doc>
<url>https://b.example.com/</url>
<title>Title <hlword>B</hlword></title>
<passages><passage>Passage B</passage></passages>
</doc>
</group>
<group>
<doc>
<url>https://c.example.com/</url>
<title>Title C</title>
</doc>
</group>
</grouping>
</results>
</response>
</yandexsearch>"""

_ERROR_XML = """<?xml version="1.0" encoding="utf-8"?>
<yandexsearch version="1.0">
<response>
<error code="15">Backend error</error>
</response>
</yandexsearch>"""


def _set_creds(monkeypatch):
    monkeypatch.setenv("YANDEX_SEARCH_API_KEY", "yc-key-123")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "b1gexample")


# ---------------------------------------------------------------------------
# YandexWebSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestYandexProviderIsConfigured:
    def test_configured_when_key_and_folder_set(self, monkeypatch):
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider
        assert YandexWebSearchProvider().is_available() is True

    def test_not_configured_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("YANDEX_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("YANDEX_FOLDER_ID", "b1gexample")
        from plugins.web.yandex.provider import YandexWebSearchProvider
        assert YandexWebSearchProvider().is_available() is False

    def test_not_configured_when_folder_missing(self, monkeypatch):
        monkeypatch.setenv("YANDEX_SEARCH_API_KEY", "yc-key-123")
        monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
        from plugins.web.yandex.provider import YandexWebSearchProvider
        assert YandexWebSearchProvider().is_available() is False

    def test_not_configured_when_whitespace(self, monkeypatch):
        monkeypatch.setenv("YANDEX_SEARCH_API_KEY", "   ")
        monkeypatch.setenv("YANDEX_FOLDER_ID", "   ")
        from plugins.web.yandex.provider import YandexWebSearchProvider
        assert YandexWebSearchProvider().is_available() is False

    def test_provider_name(self):
        from plugins.web.yandex.provider import YandexWebSearchProvider
        assert YandexWebSearchProvider().name == "yandex"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.yandex.provider import YandexWebSearchProvider
        assert issubclass(YandexWebSearchProvider, WebSearchProvider)

    def test_search_only(self):
        from plugins.web.yandex.provider import YandexWebSearchProvider
        p = YandexWebSearchProvider()
        assert p.supports_search() is True
        assert p.supports_extract() is False


class TestYandexProviderSearch:
    """search() is operation-based: POST submits, GET polls until done.

    ``httpx.post`` is mocked as the submit call (returns ``{"id": ...}``);
    ``httpx.get`` is mocked as the operation-status poll (returns
    ``{"done": True, "response": {"rawData": ...}}`` immediately, since the
    provider's poll loop just re-calls GET until ``done`` is true).
    """

    @staticmethod
    def _mock_resp(json_data, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.return_value = json_data
        m.raise_for_status = MagicMock()
        return m

    def _mock_submit(self, operation_id="op-123"):
        return self._mock_resp({"id": operation_id})

    def _mock_done(self, raw_data=None, error=None):
        response = {"done": True}
        if error is not None:
            response["error"] = error
        else:
            response["response"] = {"rawData": raw_data}
        return self._mock_resp(response)

    def test_happy_path_normalizes_results(self, monkeypatch):
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        with patch("httpx.post", return_value=self._mock_submit()), \
             patch("httpx.get", return_value=self._mock_done(_b64_xml(_SAMPLE_XML))):
            result = YandexWebSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {
            "title": "Title A",
            "url": "https://a.example.com/",
            "description": "Headline for A",
            "position": 1,
        }
        # <hlword> emphasis tag stripped from title text.
        assert web[1]["title"] == "Title B"
        # No <headline> -> falls back to the first <passage>.
        assert web[1]["description"] == "Passage B"
        # No <headline> and no <passages> -> empty description, not a crash.
        assert web[2]["description"] == ""
        assert web[2]["position"] == 3

    def test_sends_api_key_header_and_folder_id(self, monkeypatch):
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json", {})
            return self._mock_submit()

        with patch("httpx.post", side_effect=fake_post), \
             patch("httpx.get", return_value=self._mock_done(_b64_xml(_SAMPLE_XML))):
            YandexWebSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://searchapi.api.cloud.yandex.net/v2/web/searchAsync"
        assert captured["headers"].get("Authorization") == "Api-Key yc-key-123"
        assert captured["json"]["folderId"] == "b1gexample"
        assert captured["json"]["query"]["queryText"] == "q"
        assert captured["json"]["responseFormat"] == "FORMAT_XML"

    def test_polls_operation_endpoint_with_id(self, monkeypatch):
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            return self._mock_done(_b64_xml(_SAMPLE_XML))

        with patch("httpx.post", return_value=self._mock_submit("op-abc-999")), \
             patch("httpx.get", side_effect=fake_get):
            YandexWebSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://operation.api.cloud.yandex.net/operations/op-abc-999"
        assert captured["headers"].get("Authorization") == "Api-Key yc-key-123"

    def test_polls_until_done_true(self, monkeypatch):
        """First poll reports not-done; second reports done — must poll again, not give up."""
        _set_creds(monkeypatch)
        monkeypatch.setattr("plugins.web.yandex.provider._POLL_INTERVAL_SECONDS", 0.01)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        responses = [
            self._mock_resp({"done": False}),
            self._mock_done(_b64_xml(_SAMPLE_XML)),
        ]

        with patch("httpx.post", return_value=self._mock_submit()), \
             patch("httpx.get", side_effect=responses):
            result = YandexWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 3

    def test_limit_maps_to_groups_on_page(self, monkeypatch):
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_submit()

        with patch("httpx.post", side_effect=fake_post), \
             patch("httpx.get", return_value=self._mock_done(_b64_xml(_SAMPLE_XML))):
            YandexWebSearchProvider().search("q", limit=7)

        assert captured["json"]["groupSpec"]["groupsOnPage"] == "7"

    def test_limit_is_capped_at_100(self, monkeypatch):
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_submit()

        with patch("httpx.post", side_effect=fake_post), \
             patch("httpx.get", return_value=self._mock_done(_b64_xml(_SAMPLE_XML))):
            YandexWebSearchProvider().search("q", limit=500)

        assert captured["json"]["groupSpec"]["groupsOnPage"] == "100"

    def test_optional_region_and_lang_passthrough(self, monkeypatch):
        _set_creds(monkeypatch)
        monkeypatch.setenv("YANDEX_SEARCH_REGION", "225")
        monkeypatch.setenv("YANDEX_SEARCH_LANG", "LOCALIZATION_EN")
        from plugins.web.yandex.provider import YandexWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_submit()

        with patch("httpx.post", side_effect=fake_post), \
             patch("httpx.get", return_value=self._mock_done(_b64_xml(_SAMPLE_XML))):
            YandexWebSearchProvider().search("q", limit=5)

        assert captured["json"]["region"] == "225"
        assert captured["json"]["l10n"] == "LOCALIZATION_EN"

    def test_region_and_lang_omitted_by_default(self, monkeypatch):
        _set_creds(monkeypatch)
        monkeypatch.delenv("YANDEX_SEARCH_REGION", raising=False)
        monkeypatch.delenv("YANDEX_SEARCH_LANG", raising=False)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return self._mock_submit()

        with patch("httpx.post", side_effect=fake_post), \
             patch("httpx.get", return_value=self._mock_done(_b64_xml(_SAMPLE_XML))):
            YandexWebSearchProvider().search("q", limit=5)

        assert "region" not in captured["json"]
        assert "l10n" not in captured["json"]

    def test_yandex_reported_error_returns_failure(self, monkeypatch):
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        with patch("httpx.post", return_value=self._mock_submit()), \
             patch("httpx.get", return_value=self._mock_done(_b64_xml(_ERROR_XML))):
            result = YandexWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "Backend error" in result["error"]
        assert "15" in result["error"]

    def test_operation_error_field_returns_failure(self, monkeypatch):
        """The operation itself can report failure via its own `error` field
        (distinct from a successful response whose XML body contains
        <error>) — e.g. quota exceeded, invalid folderId, etc."""
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        with patch("httpx.post", return_value=self._mock_submit()), \
             patch("httpx.get", return_value=self._mock_done(error={"code": 7, "message": "quota exceeded"})):
            result = YandexWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "quota exceeded" in result["error"]

    def test_operation_timeout_returns_failure(self, monkeypatch):
        """Operation never reports done=true within the poll budget -> typed error, not a hang."""
        _set_creds(monkeypatch)
        monkeypatch.setattr("plugins.web.yandex.provider._POLL_INTERVAL_SECONDS", 0.01)
        monkeypatch.setattr("plugins.web.yandex.provider._POLL_TIMEOUT_SECONDS", 0.03)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        with patch("httpx.post", return_value=self._mock_submit()), \
             patch("httpx.get", return_value=self._mock_resp({"done": False})):
            result = YandexWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "did not complete" in result["error"].lower()

    def test_http_error_on_submit_returns_failure(self, monkeypatch):
        import httpx
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        bad = MagicMock()
        bad.status_code = 401
        bad.json.return_value = {"message": "Invalid API key"}
        err = httpx.HTTPStatusError("401", request=MagicMock(), response=bad)

        with patch("httpx.post", side_effect=err):
            result = YandexWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "401" in result["error"]
        assert "Invalid API key" in result["error"]

    def test_http_error_on_poll_returns_failure(self, monkeypatch):
        import httpx
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        bad = MagicMock()
        bad.status_code = 404
        err = httpx.HTTPStatusError("404", request=MagicMock(), response=bad)

        with patch("httpx.post", return_value=self._mock_submit()), \
             patch("httpx.get", side_effect=err):
            result = YandexWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "404" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        with patch("httpx.post", side_effect=httpx.RequestError("boom")):
            result = YandexWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "boom" in result["error"] or "Yandex" in result["error"]

    def test_malformed_base64_returns_failure(self, monkeypatch):
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        with patch("httpx.post", return_value=self._mock_submit()), \
             patch("httpx.get", return_value=self._mock_done("not-valid-base64!!!")):
            result = YandexWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "error" in result

    def test_malformed_xml_returns_failure(self, monkeypatch):
        _set_creds(monkeypatch)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        broken_xml = _b64_xml("<yandexsearch><response><results>")  # unterminated
        with patch("httpx.post", return_value=self._mock_submit()), \
             patch("httpx.get", return_value=self._mock_done(broken_xml)):
            result = YandexWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "error" in result

    def test_missing_api_key_returns_failure(self, monkeypatch):
        monkeypatch.delenv("YANDEX_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("YANDEX_FOLDER_ID", "b1gexample")
        from plugins.web.yandex.provider import YandexWebSearchProvider

        result = YandexWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "YANDEX_SEARCH_API_KEY" in result["error"]

    def test_missing_folder_id_returns_failure(self, monkeypatch):
        monkeypatch.setenv("YANDEX_SEARCH_API_KEY", "yc-key-123")
        monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
        from plugins.web.yandex.provider import YandexWebSearchProvider

        result = YandexWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "YANDEX_FOLDER_ID" in result["error"]


# ---------------------------------------------------------------------------
# Integration: _is_backend_available / _get_backend / check_web_api_key
# ---------------------------------------------------------------------------


class TestYandexBackendWiring:
    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        # "yandex" isn't in _LEGACY_WEB_BACKENDS, so availability resolution
        # goes through the registry — needs the plugin registered first.
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_is_backend_available_true_when_configured(self, monkeypatch):
        _set_creds(monkeypatch)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("yandex") is True

    def test_is_backend_available_false_when_missing(self, monkeypatch):
        monkeypatch.delenv("YANDEX_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("yandex") is False

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "yandex"})
        _set_creds(monkeypatch)
        assert web_tools._get_backend() == "yandex"

    def test_check_web_api_key_true_when_yandex_configured(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "yandex"})
        _set_creds(monkeypatch)
        assert web_tools.check_web_api_key() is True


# ---------------------------------------------------------------------------
# yandex is search-only: web_extract returns a clear error
# ---------------------------------------------------------------------------


class TestYandexSearchOnlyErrors:
    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_web_extract_returns_search_only_error(self, monkeypatch):
        import asyncio
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "yandex"})
        _set_creds(monkeypatch)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)

        async def _allow_ssrf(_url: str) -> bool:
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_ssrf)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        )
        result = json.loads(result_str)
        assert result["success"] is False
        assert "search-only" in result["error"].lower()
        assert "yandex" in result["error"].lower()

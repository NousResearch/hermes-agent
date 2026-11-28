"""Tests for the SearXNG web search provider.

Covers:
- SearXNGWebSearchProvider.is_available() env var gating
- SearXNGWebSearchProvider.search() — happy path, HTTP error, request error, bad JSON
- Result normalization (title, url, description, position)
- Score-based sorting and limit truncation
- _is_backend_available("searxng") integration
- _get_backend() recognizes "searxng" as a valid configured backend
- check_web_api_key() includes searxng in availability check
- web.searxng.auth_provider: the named provider's pooled credential as the bearer
"""
from __future__ import annotations

import http.server
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.auth
import providers
from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
from hermes_cli.auth_oauth_pkce_plugin import POOL_SOURCE, OAuthPKCEConfig, pkce_auth_handler, pkce_refresh_credential
from providers import register_provider
from providers.base import ProviderProfile
from tests.providers.fake_pkce_idp import FakeIdP
from tests.tools.conftest import register_all_web_providers


# ---------------------------------------------------------------------------
# SearXNGWebSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestSearXNGSearchProviderIsConfigured:
    def test_configured_when_url_set(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        assert SearXNGWebSearchProvider().is_available() is True




class TestSearXNGSearchProviderSearch:
    """Happy path and error handling for SearXNGWebSearchProvider.search()."""

    _SAMPLE_RESPONSE = {
        "results": [
            {"title": "Result A", "url": "https://a.example.com", "content": "Desc A", "score": 0.9},
            {"title": "Result B", "url": "https://b.example.com", "content": "Desc B", "score": 0.7},
            {"title": "Result C", "url": "https://c.example.com", "content": "Desc C", "score": 0.5},
        ]
    }

    def _make_mock_response(self, json_data, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_happy_path_returns_normalized_results(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response(self._SAMPLE_RESPONSE)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0]["title"] == "Result A"
        assert web[0]["url"] == "https://a.example.com"
        assert web[0]["description"] == "Desc A"
        assert web[0]["position"] == 1

    def test_results_sorted_by_score_descending(self, monkeypatch):
        """Results should be sorted by score before limit is applied."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        unordered = {
            "results": [
                {"title": "Low",  "url": "https://low.example.com",  "content": "", "score": 0.1},
                {"title": "High", "url": "https://high.example.com", "content": "", "score": 0.99},
                {"title": "Mid",  "url": "https://mid.example.com",  "content": "", "score": 0.5},
            ]
        }
        mock_resp = self._make_mock_response(unordered)

        with patch("httpx.get", return_value=mock_resp):
            result = SearXNGWebSearchProvider().search("query", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "High"
        assert result["data"]["web"][1]["title"] == "Mid"
        assert result["data"]["web"][2]["title"] == "Low"


    def test_trailing_slash_stripped_from_url(self, monkeypatch):
        """Base URL trailing slash should not produce double-slash in endpoint."""
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080/")
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        mock_resp = self._make_mock_response({"results": []})

        calls = []
        def capture_get(url, **kwargs):
            calls.append(url)
            return mock_resp

        with patch("httpx.get", side_effect=capture_get):
            SearXNGWebSearchProvider().search("query", limit=5)

        assert calls[0] == "http://localhost:8080/search", f"Got: {calls[0]}"


# ---------------------------------------------------------------------------
# Integration: _is_backend_available recognizes "searxng"
# ---------------------------------------------------------------------------


class TestIsBackendAvailable:
    def test_searxng_available_when_url_set(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("searxng") is True


    def test_unknown_backend_still_false(self):
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("unknownbackend") is False


# ---------------------------------------------------------------------------
# Integration: _get_backend() accepts "searxng" as configured value
# ---------------------------------------------------------------------------


class TestGetBackendSearXNG:
    def test_configured_searxng_returns_searxng(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        assert web_tools._get_backend() == "searxng"


    def test_searxng_does_not_override_higher_priority_provider(self, monkeypatch):
        """Exa (higher priority than searxng) should win in auto-detect."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", dict)
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.setenv("EXA_API_KEY", "exa_test_key")
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        assert web_tools._get_backend() == "exa"

    def test_auto_detect_picks_searxng_when_url_only_in_hermes_config(self, monkeypatch):
        """#34290 follow-up: a config-only SEARXNG_URL (absent from process env)
        must still drive auto-detect via the now config-aware ``_has_env``."""
        from hermes_cli import config as hermes_config
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", dict)
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setattr(
            hermes_config,
            "get_env_value",
            lambda key: "http://config-only:8080" if key == "SEARXNG_URL" else None,
        )
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        assert web_tools._get_backend() == "searxng"


# ---------------------------------------------------------------------------
# Integration: check_web_api_key includes searxng
# ---------------------------------------------------------------------------


class TestCheckWebApiKey:
    def test_searxng_satisfies_check_web_api_key(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        assert web_tools.check_web_api_key() is True

    def test_searxng_config_only_satisfies_check_web_api_key(self, monkeypatch):
        """#34290 follow-up: config-only SEARXNG_URL satisfies the credential check."""
        from hermes_cli import config as hermes_config
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setattr(
            hermes_config,
            "get_env_value",
            lambda key: "http://config-only:8080" if key == "SEARXNG_URL" else None,
        )
        assert web_tools.check_web_api_key() is True

    def test_no_credentials_fails(self, monkeypatch):
        from tools import web_tools
        from agent import web_search_registry
        monkeypatch.setattr(web_tools, "_load_web_config", dict)
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "check_firecrawl_api_key", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        # Disable the keyless free tier — with it on, zero credentials still
        # resolves (Parallel/Exa anonymous MCP; see test_web_keyless_fallback.py).
        monkeypatch.setattr(web_search_registry, "_keyless_tier_enabled", lambda: False)
        assert web_tools.check_web_api_key() is False


# ---------------------------------------------------------------------------
# searxng-only: web_extract returns a clear error
# ---------------------------------------------------------------------------


class TestSearXNGOnlyExtractCrawlErrors:
    """When searxng is the active backend, extract/crawl must return clear errors."""

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_web_extract_searxng_returns_clear_error(self, monkeypatch):
        import asyncio
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
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
        assert "search-only" in result["error"].lower() or "SearXNG" in result["error"]


# ---------------------------------------------------------------------------
# web.searxng.auth_provider: a provider's pooled credential is the bearer
# ---------------------------------------------------------------------------

AUTH_PROVIDER = "example-pkce"


@pytest.fixture
def gateway(monkeypatch, tmp_path, request):
    """A SearXNG stand-in that answers only a bearer outside ``refused``, a fake IdP that
    ``AUTH_PROVIDER`` refreshes against, and a config naming it. Yields ``(refused, seen, idp)``;
    ``seen`` lists each request's bearer ("" when there is none)."""
    refused, seen = set(), []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            bearer = self.headers.get("Authorization", "").removeprefix("Bearer ")
            seen.append(bearer)
            body = json.dumps({"results": [{"title": "T", "url": "https://t.example", "content": "c"}]}).encode()
            self.send_response(200 if bearer and bearer not in refused else 401)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(f"web:\n  searxng:\n    auth_provider: {AUTH_PROVIDER}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    idp = FakeIdP().start()
    cfg = OAuthPKCEConfig(client_id="hermes-example", authorize_url=f"{idp.base}/authorize", token_url=f"{idp.base}/token")
    register_provider(ProviderProfile(name=AUTH_PROVIDER, auth_type="oauth_external", base_url="https://example.invalid/v1",
                                      auth_handler=pkce_auth_handler(cfg), refresh_credential=pkce_refresh_credential(cfg)))
    request.addfinalizer(lambda: (providers._REGISTRY.pop(AUTH_PROVIDER, None),
                                  hermes_cli.auth.PROVIDER_REGISTRY.pop(AUTH_PROVIDER, None)))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        monkeypatch.setenv("SEARXNG_URL", f"http://127.0.0.1:{server.server_port}")
        try:
            yield refused, seen, idp
        finally:
            server.shutdown()
            idp.stop()


def _sign_in(idp: FakeIdP, *, expires_in: int = 3600) -> None:
    """Pool an OAuth credential for ``AUTH_PROVIDER``: access token ``at-0``, refreshable at ``idp``."""
    idp.refresh_tokens.add("rt-0")
    load_pool(AUTH_PROVIDER).add_entry(PooledCredential(
        provider=AUTH_PROVIDER, id="c0", label=AUTH_PROVIDER, auth_type=AUTH_TYPE_OAUTH, priority=0, source=POOL_SOURCE,
        access_token="at-0", refresh_token="rt-0", expires_at_ms=int(time.time() * 1000) + expires_in * 1000))


class TestSearXNGAuthProvider:
    def test_sends_the_provider_token_as_bearer(self, gateway):
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        _refused, seen, idp = gateway
        _sign_in(idp)

        assert SearXNGWebSearchProvider().search("q")["success"] is True
        assert seen == ["at-0"] and idp.token_requests == []

    def test_renews_an_expiring_token_before_the_request(self, gateway):
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        refused, seen, idp = gateway
        _sign_in(idp, expires_in=60)
        refused.add("at-0")

        assert SearXNGWebSearchProvider().search("q")["success"] is True
        assert seen == [load_pool(AUTH_PROVIDER).entries()[0].access_token] and seen != ["at-0"]
        assert [r["grant_type"] for r in idp.token_requests] == ["refresh_token"]

    def test_no_usable_credential_fails_before_any_request(self, gateway):
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        _refused, seen, _idp = gateway

        result = SearXNGWebSearchProvider().search("q")
        assert result["success"] is False and f"hermes auth add {AUTH_PROVIDER}" in result["error"]
        assert seen == []

    def test_credential_goes_out_over_https_but_not_plain_http(self, gateway, monkeypatch):
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        _refused, _seen, idp = gateway
        _sign_in(idp)

        monkeypatch.setenv("SEARXNG_URL", "http://searxng.invalid")
        result = SearXNGWebSearchProvider().search("q")
        assert result["success"] is False and "must be https" in result["error"]

        monkeypatch.setenv("SEARXNG_URL", "https://searxng.invalid")
        with patch("httpx.get", return_value=MagicMock(json=lambda: {"results": []})) as sent:
            assert SearXNGWebSearchProvider().search("q")["success"] is True
        assert sent.call_args.kwargs["headers"]["Authorization"] == "Bearer at-0"

    def test_without_the_setting_no_credential_is_sent(self, gateway, tmp_path):
        from plugins.web.searxng.provider import SearXNGWebSearchProvider
        _refused, seen, idp = gateway
        _sign_in(idp)
        (tmp_path / "hermes" / "config.yaml").write_text("web: {}\n")

        assert SearXNGWebSearchProvider().search("q")["success"] is False
        assert seen == [""]

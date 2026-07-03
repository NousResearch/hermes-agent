"""Tests for web backend client configuration and singleton behavior.

Coverage:
  _get_firecrawl_client() — configuration matrix, singleton caching,
  constructor failure recovery, return value verification, edge cases.
  _get_backend() — backend selection logic with env var combinations.
  _get_parallel_client() — Parallel client configuration, singleton caching.
  check_web_api_key() — unified availability check across all web backends.
"""

import importlib
import json
import os
import sys
import types
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestFirecrawlClientConfig:
    """Test suite for Firecrawl client initialization."""

    def setup_method(self):
        """Reset client and env vars before each test."""
        import tools.web_tools
        tools.web_tools._firecrawl_client = None
        tools.web_tools._firecrawl_client_config = None
        for key in (
            "FIRECRAWL_API_KEY",
            "FIRECRAWL_API_URL",
            "FIRECRAWL_GATEWAY_URL",
            "TOOL_GATEWAY_DOMAIN",
            "TOOL_GATEWAY_SCHEME",
            "TOOL_GATEWAY_USER_TOKEN",
        ):
            os.environ.pop(key, None)
        # Enable managed tools by default for these tests — patch both the
        # local web_tools import and the managed_tool_gateway import so the
        # full firecrawl client init path sees True.
        self._managed_patchers = [
            patch("tools.web_tools.managed_nous_tools_enabled", return_value=True),
            patch("tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True),
        ]
        for p in self._managed_patchers:
            p.start()

    def teardown_method(self):
        """Reset client after each test."""
        import tools.web_tools
        tools.web_tools._firecrawl_client = None
        tools.web_tools._firecrawl_client_config = None
        for key in (
            "FIRECRAWL_API_KEY",
            "FIRECRAWL_API_URL",
            "FIRECRAWL_GATEWAY_URL",
            "TOOL_GATEWAY_DOMAIN",
            "TOOL_GATEWAY_SCHEME",
            "TOOL_GATEWAY_USER_TOKEN",
        ):
            os.environ.pop(key, None)
        for p in self._managed_patchers:
            p.stop()

    # ── Configuration matrix ─────────────────────────────────────────

    def test_no_config_raises_with_helpful_message(self):
        """Neither key nor URL → ValueError with guidance."""
        with patch("tools.web_tools.Firecrawl"):
            with patch("tools.web_tools._read_nous_access_token", return_value=None):
                from tools.web_tools import _get_firecrawl_client
                with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
                    _get_firecrawl_client()

    def test_tool_gateway_domain_builds_firecrawl_gateway_origin(self):
        """Shared gateway domain should derive the Firecrawl vendor hostname."""
        with patch.dict(os.environ, {"TOOL_GATEWAY_DOMAIN": "nousresearch.com"}):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch("tools.web_tools.Firecrawl") as mock_fc:
                    from tools.web_tools import _get_firecrawl_client
                    result = _get_firecrawl_client()
                    mock_fc.assert_called_once_with(
                        api_key="nous-token",
                        api_url="https://firecrawl-gateway.nousresearch.com",
                    )
                    assert result is mock_fc.return_value

    def test_tool_gateway_scheme_can_switch_derived_gateway_origin_to_http(self):
        """Shared gateway scheme should allow local plain-http vendor hosts."""
        with patch.dict(os.environ, {
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
            "TOOL_GATEWAY_SCHEME": "http",
        }):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch("tools.web_tools.Firecrawl") as mock_fc:
                    from tools.web_tools import _get_firecrawl_client
                    result = _get_firecrawl_client()
                    mock_fc.assert_called_once_with(
                        api_key="nous-token",
                        api_url="http://firecrawl-gateway.nousresearch.com",
                    )
                    assert result is mock_fc.return_value

    def test_invalid_tool_gateway_scheme_raises(self):
        """Unexpected shared gateway schemes should fail fast."""
        with patch.dict(os.environ, {
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
            "TOOL_GATEWAY_SCHEME": "ftp",
        }):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                from tools.web_tools import _get_firecrawl_client
                with pytest.raises(ValueError, match="TOOL_GATEWAY_SCHEME"):
                    _get_firecrawl_client()

    def test_explicit_firecrawl_gateway_url_takes_precedence(self):
        """An explicit Firecrawl gateway origin should override the shared domain."""
        with patch.dict(os.environ, {
            "FIRECRAWL_GATEWAY_URL": "https://firecrawl-gateway.localhost:3009/",
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
        }):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch("tools.web_tools.Firecrawl") as mock_fc:
                    from tools.web_tools import _get_firecrawl_client
                    _get_firecrawl_client()
                    mock_fc.assert_called_once_with(
                        api_key="nous-token",
                        api_url="https://firecrawl-gateway.localhost:3009",
                    )

    def test_default_gateway_domain_targets_nous_production_origin(self):
        """Default gateway origin should point at the Firecrawl vendor hostname."""
        with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                _get_firecrawl_client()
                mock_fc.assert_called_once_with(
                    api_key="nous-token",
                    api_url="https://firecrawl-gateway.nousresearch.com",
                )

    def test_nous_auth_token_respects_hermes_home_override(self, tmp_path):
        """Auth lookup should read from HERMES_HOME/auth.json, not ~/.hermes/auth.json."""
        real_home = tmp_path / "real-home"
        (real_home / ".hermes").mkdir(parents=True)

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        (hermes_home / "auth.json").write_text(json.dumps({
            "providers": {
                "nous": {
                    "access_token": "nous-token",
                }
            }
        }))

        with patch.dict(os.environ, {
            "HOME": str(real_home),
            "HERMES_HOME": str(hermes_home),
        }, clear=False):
            import tools.web_tools
            importlib.reload(tools.web_tools)
            assert tools.web_tools._read_nous_access_token() == "nous-token"

    # ── Singleton caching ────────────────────────────────────────────

    def test_singleton_returns_same_instance(self):
        """Second call returns cached client without re-constructing."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                client1 = _get_firecrawl_client()
                client2 = _get_firecrawl_client()
                assert client1 is client2
                mock_fc.assert_called_once()  # constructed only once

    def test_constructor_failure_allows_retry(self):
        """If Firecrawl() raises, next call should retry (not return None)."""
        import tools.web_tools
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                mock_fc.side_effect = [RuntimeError("init failed"), MagicMock()]
                from tools.web_tools import _get_firecrawl_client

                with pytest.raises(RuntimeError):
                    _get_firecrawl_client()

                # Client stayed None, so retry should work
                assert tools.web_tools._firecrawl_client is None
                result = _get_firecrawl_client()
                assert result is not None

    # ── Edge cases ───────────────────────────────────────────────────

    def test_empty_string_key_no_url_raises(self):
        """FIRECRAWL_API_KEY='' with no URL → should raise."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": ""}):
            with patch("tools.web_tools.Firecrawl"):
                with patch("tools.web_tools._read_nous_access_token", return_value=None):
                    from tools.web_tools import _get_firecrawl_client
                    with pytest.raises(ValueError):
                        _get_firecrawl_client()


class TestBackendSelection:
    """Test suite for _get_backend() backend selection logic.

    The backend is configured via config.yaml (web.backend), set by
    ``hermes tools``.  Falls back to key-based detection for legacy/manual
    setups.
    """

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
    )

    def setup_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)
        self._managed_patchers = [
            patch("tools.web_tools.managed_nous_tools_enabled", return_value=True),
            patch("tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True),
        ]
        for p in self._managed_patchers:
            p.start()

    def teardown_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)
        for p in self._managed_patchers:
            p.stop()

    # ── Config-based selection (web.backend in config.yaml) ───────────

    def test_config_parallel(self):
        """web.backend=parallel in config → 'parallel' regardless of keys."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "parallel"}):
            assert _get_backend() == "parallel"

    def test_config_exa(self):
        """web.backend=exa in config → 'exa' regardless of other keys."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "exa"}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "exa"

    def test_config_firecrawl(self):
        """web.backend=firecrawl in config → 'firecrawl' even if Parallel key set."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "firecrawl"

    def test_config_tavily(self):
        """web.backend=tavily in config → 'tavily' regardless of other keys."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "tavily"}):
            assert _get_backend() == "tavily"

    def test_config_tavily_overrides_env_keys(self):
        """web.backend=tavily in config → 'tavily' even if Firecrawl key set."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "tavily"}), \
             patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "tavily"

    def test_config_case_insensitive(self):
        """web.backend=Parallel (mixed case) → 'parallel'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "Parallel"}):
            assert _get_backend() == "parallel"

    def test_config_tavily_case_insensitive(self):
        """web.backend=Tavily (mixed case) → 'tavily'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "Tavily"}):
            assert _get_backend() == "tavily"

    # ── Fallback (no web.backend in config) ───────────────────────────

    def test_fallback_parallel_only_key(self):
        """Only PARALLEL_API_KEY set → 'parallel'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "parallel"

    def test_fallback_exa_only_key(self):
        """Only EXA_API_KEY set → 'exa'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"EXA_API_KEY": "exa-test"}):
            assert _get_backend() == "exa"

    def test_fallback_exa_takes_priority_over_parallel(self):
        """Direct-credential backends are tried in the order tavily > exa > parallel
        so an explicit Exa key wins when both Exa and Parallel are configured."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"EXA_API_KEY": "exa-test", "PARALLEL_API_KEY": "par-test"}):
            assert _get_backend() == "exa"

    def test_fallback_tavily_only_key(self):
        """Only TAVILY_API_KEY set → 'tavily'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
            assert _get_backend() == "tavily"

    def test_fallback_tavily_beats_firecrawl_direct(self):
        """Tavily ranks above firecrawl in the explicit-credential block."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test", "FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "tavily"

    def test_fallback_tavily_beats_parallel(self):
        """Tavily is first in the explicit-credential block so it wins over parallel."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test", "PARALLEL_API_KEY": "par-test"}):
            assert _get_backend() == "tavily"

    def test_fallback_parallel_beats_firecrawl_direct(self):
        """Parallel + Firecrawl-direct → parallel (parallel is the higher-priority
        explicit-credential backend; firecrawl-direct ranks below it)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key", "FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "parallel"

    def test_fallback_firecrawl_only_key(self):
        """Only FIRECRAWL_API_KEY set → 'firecrawl'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "firecrawl"

    def test_fallback_no_keys_defaults_to_firecrawl(self):
        """No keys, no config → 'firecrawl' (will fail at client init)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._ddgs_package_importable", return_value=False):
            assert _get_backend() == "firecrawl"

    def test_invalid_config_falls_through_to_fallback(self):
        """web.backend=invalid → ignored, uses key-based fallback."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "nonexistent"}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "parallel"

    def test_managed_gateway_does_not_preempt_explicit_tavily(self):
        """Regression: a Nous OAuth token (managed gateway "ready") must NOT
        beat an explicitly configured TAVILY_API_KEY in the fallback path.
        Free Nous tiers don't include web search, so the user's deliberate
        Tavily setup would fail at runtime with "no subscription" if the
        gateway pre-empted it."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=True), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
            assert _get_backend() == "tavily"

    def test_managed_gateway_only_falls_through_to_firecrawl(self):
        """When no explicit-credential backend is configured, a Nous-managed
        gateway token still selects firecrawl — the convenience path is
        preserved, just no longer pre-empts."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=True):
            assert _get_backend() == "firecrawl"


class TestParallelClientConfig:
    """Test suite for Parallel client initialization."""

    def setup_method(self):
        import tools.web_tools
        tools.web_tools._parallel_client = None
        os.environ.pop("PARALLEL_API_KEY", None)
        fake_parallel = types.ModuleType("parallel")

        class Parallel:
            def __init__(self, api_key):
                self.api_key = api_key

        class AsyncParallel:
            def __init__(self, api_key):
                self.api_key = api_key

        fake_parallel.Parallel = Parallel
        fake_parallel.AsyncParallel = AsyncParallel
        sys.modules["parallel"] = fake_parallel

    def teardown_method(self):
        import tools.web_tools
        tools.web_tools._parallel_client = None
        os.environ.pop("PARALLEL_API_KEY", None)
        sys.modules.pop("parallel", None)

    def test_creates_client_with_key(self):
        """PARALLEL_API_KEY set → creates Parallel client."""
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            from tools.web_tools import _get_parallel_client
            from parallel import Parallel
            client = _get_parallel_client()
            assert client is not None
            assert isinstance(client, Parallel)

    def test_no_key_raises_with_helpful_message(self):
        """No PARALLEL_API_KEY → ValueError with guidance."""
        from tools.web_tools import _get_parallel_client
        with pytest.raises(ValueError, match="PARALLEL_API_KEY"):
            _get_parallel_client()

    def test_singleton_returns_same_instance(self):
        """Second call returns cached client."""
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            from tools.web_tools import _get_parallel_client
            client1 = _get_parallel_client()
            client2 = _get_parallel_client()
            assert client1 is client2


class TestWebSearchSchema:
    """Test suite for web_search tool schema and handler wiring."""

    def test_schema_exposes_optional_limit(self):
        import tools.web_tools

        limit_schema = tools.web_tools.WEB_SEARCH_SCHEMA["parameters"]["properties"]["limit"]

        assert limit_schema["type"] == "integer"
        assert limit_schema["minimum"] == 1
        assert limit_schema["maximum"] == 100
        assert limit_schema["default"] == 5
        assert "limit" not in tools.web_tools.WEB_SEARCH_SCHEMA["parameters"]["required"]

    def test_registered_handler_passes_limit(self):
        import tools.web_tools

        entry = tools.web_tools.registry.get_entry("web_search")
        with patch("tools.web_tools.web_search_tool", return_value='{"success": true}') as mock_search:
            result = entry.handler({"query": "site:example.com docs", "limit": 12})

        assert result == '{"success": true}'
        mock_search.assert_called_once_with("site:example.com docs", limit=12)

    def test_registered_handler_defaults_limit_to_five(self):
        import tools.web_tools

        entry = tools.web_tools.registry.get_entry("web_search")
        with patch("tools.web_tools.web_search_tool", return_value='{"success": true}') as mock_search:
            result = entry.handler({"query": "docs"})

        assert result == '{"success": true}'
        mock_search.assert_called_once_with("docs", limit=5)

    def test_web_search_matrix_reports_provider_presence_without_values(self, monkeypatch):
        import tools.web_tools

        class FakeProvider:
            name = "firecrawl"
            display_name = "Firecrawl"

            def is_available(self):
                return True

            def supports_search(self):
                return True

            def supports_extract(self):
                return True

            def supports_crawl(self):
                return True

        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-secret-value")
        with patch("hermes_cli.plugins.discover_plugins"), \
             patch("agent.web_search_registry.list_providers", return_value=[FakeProvider()]), \
             patch("agent.web_search_registry.get_active_search_provider", return_value=FakeProvider()), \
             patch("agent.web_search_registry.get_active_extract_provider", return_value=FakeProvider()), \
             patch("agent.web_search_registry.get_active_crawl_provider", return_value=FakeProvider()), \
             patch("hermes_cli.config.load_config", return_value={"browser": {"cloud_provider": "firecrawl"}}):
            result = json.loads(
                tools.web_tools.web_search_matrix(
                    require_capabilities=["search", "extract"],
                    require_providers=["firecrawl"],
                )
            )

        assert result["success"] is True
        assert result["status"] == "ok"
        assert result["active"]["search"]["name"] == "firecrawl"
        firecrawl = result["providers"][0]
        assert firecrawl["name"] == "firecrawl"
        assert firecrawl["available"] is True
        assert firecrawl["capabilities"] == {
            "search": True,
            "extract": True,
            "crawl": True,
        }
        assert result["firecrawl_surfaces"]["search"] is True
        assert result["firecrawl_surfaces"]["scrape"] is True
        assert result["firecrawl_surfaces"]["extract"] is True
        assert result["firecrawl_surfaces"]["interact"] is True
        assert "FIRECRAWL_API_KEY" in firecrawl["config"]["present"]
        assert "fc-secret-value" not in json.dumps(result)

    def test_web_search_matrix_dependency_blocked_when_firecrawl_unavailable(self, monkeypatch):
        import tools.web_tools

        class FakeProvider:
            name = "firecrawl"
            display_name = "Firecrawl"

            def is_available(self):
                return False

            def supports_search(self):
                return True

            def supports_extract(self):
                return True

            def supports_crawl(self):
                return True

        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        with patch("hermes_cli.plugins.discover_plugins"), \
             patch("agent.web_search_registry.list_providers", return_value=[FakeProvider()]), \
             patch("agent.web_search_registry.get_active_search_provider", return_value=None), \
             patch("agent.web_search_registry.get_active_extract_provider", return_value=None), \
             patch("agent.web_search_registry.get_active_crawl_provider", return_value=None), \
             patch("hermes_cli.config.load_config", return_value={"browser": {"cloud_provider": "firecrawl"}}):
            result = json.loads(
                tools.web_tools.web_search_matrix(
                    require_capabilities=["search", "extract"],
                    require_providers=["firecrawl"],
                )
            )

        assert result["success"] is False
        assert result["status"] == "dependency_blocked"
        assert result["firecrawl_surfaces"]["interact"] is False
        assert any("firecrawl" in reason.lower() for reason in result["blocked_reasons"])

    def test_web_search_matrix_query_fuses_provider_results(self, monkeypatch):
        import tools.web_tools

        class FakeProvider:
            def __init__(self, name, url):
                self.name = name
                self.display_name = name.title()
                self.url = url

            def is_available(self):
                return True

            def supports_search(self):
                return True

            def supports_extract(self):
                return self.name == "firecrawl"

            def supports_crawl(self):
                return self.name == "firecrawl"

            def search(self, query, limit=5):
                return {
                    "success": True,
                    "data": {
                        "web": [
                            {
                                "title": f"{self.name} result",
                                "url": self.url,
                                "description": query,
                                "position": 1,
                            }
                        ]
                    },
                }

        firecrawl = FakeProvider("firecrawl", "https://example.com/page/")
        exa = FakeProvider("exa", "https://example.com/page")
        providers = {"firecrawl": firecrawl, "exa": exa}

        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-secret-value")
        monkeypatch.setenv("EXA_API_KEY", "exa-secret-value")
        with patch("hermes_cli.plugins.discover_plugins"), \
             patch("agent.web_search_registry.list_providers", return_value=[firecrawl, exa]), \
             patch("agent.web_search_registry.get_provider", side_effect=providers.get), \
             patch("agent.web_search_registry.get_active_search_provider", return_value=firecrawl), \
             patch("agent.web_search_registry.get_active_extract_provider", return_value=firecrawl), \
             patch("agent.web_search_registry.get_active_crawl_provider", return_value=firecrawl), \
             patch("hermes_cli.config.load_config", return_value={"browser": {"cloud_provider": "firecrawl"}}):
            result = json.loads(
                tools.web_tools.web_search_matrix(
                    query="ontology evidence",
                    providers=["firecrawl", "exa"],
                    require_providers=["firecrawl"],
                )
            )

        assert result["success"] is True
        assert result["query"] == "ontology evidence"
        assert result["providers_used"] == ["firecrawl", "exa"]
        assert result["data"]["web"][0]["provider_hits"] == 2
        assert result["data"]["web"][0]["providers"] == ["exa", "firecrawl"]
        assert result["provider_status"]["available_providers"] == ["exa", "firecrawl"]
        assert "fc-secret-value" not in json.dumps(result)
        assert "exa-secret-value" not in json.dumps(result)

    def test_web_search_clamps_limit_before_backend_call(self):
        import tools.web_tools

        # After the web-provider plugin migration, _parallel_search lives in
        # plugins.web.parallel.provider.ParallelWebSearchProvider.search; the
        # tool dispatcher resolves a provider from the registry and calls
        # provider.search(query, limit). Mock the provider lookup so we can
        # assert the limit is clamped before reaching the backend.
        fake_search = MagicMock(return_value={"success": True, "data": {"web": []}})
        fake_provider = MagicMock(
            name="ParallelWebSearchProvider",
            supports_search=MagicMock(return_value=True),
        )
        fake_provider.search = fake_search
        fake_provider.name = "parallel"

        with patch("tools.web_tools._get_search_backend", return_value="parallel"), \
             patch("agent.web_search_registry.get_provider", return_value=fake_provider), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(tools.web_tools._debug, "log_call"), \
             patch.object(tools.web_tools._debug, "save"):
            result = json.loads(tools.web_tools.web_search_tool("docs", limit=500))

        assert result == {"success": True, "data": {"web": []}}
        fake_search.assert_called_once_with("docs", 100)

    def test_web_search_falls_back_when_configured_provider_hits_quota(self):
        import tools.web_tools

        primary = MagicMock(
            name="FirecrawlWebSearchProvider",
            supports_search=MagicMock(return_value=True),
        )
        primary.name = "firecrawl"
        primary.search.return_value = {
            "success": False,
            "error": "Payment Required: Insufficient credits",
        }
        fallback = MagicMock(
            name="ParallelWebSearchProvider",
            supports_search=MagicMock(return_value=True),
        )
        fallback.name = "parallel"
        fallback.is_available.return_value = True
        fallback.search.return_value = {
            "success": True,
            "data": {"web": [{"url": "https://example.com", "title": "ok"}]},
        }

        def provider_for(name):
            return {"firecrawl": primary, "parallel": fallback}.get(name)

        with patch("tools.web_tools._get_search_backend", return_value="firecrawl"), \
             patch("agent.web_search_registry.get_provider", side_effect=provider_for), \
             patch("agent.web_search_registry.list_providers", return_value=[primary, fallback]), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(tools.web_tools._debug, "log_call"), \
             patch.object(tools.web_tools._debug, "save"):
            result = json.loads(tools.web_tools.web_search_tool("docs", limit=3))

        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://example.com"
        assert result["meta"]["primary_provider"] == "firecrawl"
        assert result["meta"]["provider"] == "parallel"
        assert result["meta"]["fallback_from"] == "firecrawl"
        assert "Insufficient credits" in result["meta"]["fallback_reason"]
        assert result["meta"]["providers_attempted"] == ["firecrawl", "parallel"]
        primary.search.assert_called_once_with("docs", 3)
        fallback.search.assert_called_once_with("docs", 3)

    @pytest.mark.asyncio
    async def test_web_extract_falls_back_to_direct_http_on_firecrawl_credit_exhaustion(self):
        import tools.web_tools

        class FakeFirecrawlProvider:
            name = "firecrawl"
            display_name = "Firecrawl"

            def supports_extract(self):
                return True

            async def extract(self, urls, **kwargs):
                return [
                    {
                        "url": urls[0],
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": "Payment Required: Insufficient credits",
                    }
                ]

        direct_result = {
            "url": "https://www.medicaid.gov/example",
            "title": "Official source",
            "content": "official Medicaid source text",
            "raw_content": "official Medicaid source text",
            "metadata": {
                "source": "direct_http",
                "sha256": "abc123",
            },
        }

        with patch("tools.web_tools._get_extract_backend", return_value="firecrawl"), \
             patch("agent.web_search_registry.get_provider", return_value=FakeFirecrawlProvider()), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_auxiliary_model", return_value=False), \
             patch("tools.web_tools._direct_http_extract_one", new=AsyncMock(return_value=direct_result)), \
             patch.object(tools.web_tools._debug, "log_call"), \
             patch.object(tools.web_tools._debug, "save"):
            result = json.loads(
                await tools.web_tools.web_extract_tool(
                    ["https://www.medicaid.gov/example"],
                    use_llm_processing=False,
                )
            )

        entry = result["results"][0]
        assert entry["content"] == "official Medicaid source text"
        assert entry["error"] is None
        assert entry["degradation"]["category"] == "provider_credit_exhaustion"
        assert entry["degradation"]["primary_provider"] == "firecrawl"
        assert entry["degradation"]["fallback_provider"] == "direct_http"
        assert entry["degradation"]["fallback_status"] == "succeeded"
        assert result["meta"]["degradations"][0]["category"] == "provider_credit_exhaustion"

    def test_web_search_falls_back_when_configured_provider_raises_retryable_error(self):
        import tools.web_tools

        primary = MagicMock(
            name="FirecrawlWebSearchProvider",
            supports_search=MagicMock(return_value=True),
        )
        primary.name = "firecrawl"
        primary.search.side_effect = RuntimeError("rate limited by provider")
        fallback = MagicMock(
            name="ParallelWebSearchProvider",
            supports_search=MagicMock(return_value=True),
        )
        fallback.name = "parallel"
        fallback.is_available.return_value = True
        fallback.search.return_value = {
            "success": True,
            "data": {"web": [{"url": "https://example.com", "title": "ok"}]},
        }

        def provider_for(name):
            return {"firecrawl": primary, "parallel": fallback}.get(name)

        with patch("tools.web_tools._get_search_backend", return_value="firecrawl"), \
             patch("agent.web_search_registry.get_provider", side_effect=provider_for), \
             patch("agent.web_search_registry.list_providers", return_value=[primary, fallback]), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(tools.web_tools._debug, "log_call"), \
             patch.object(tools.web_tools._debug, "save"):
            result = json.loads(tools.web_tools.web_search_tool("docs", limit=3))

        assert result["success"] is True
        assert result["meta"]["primary_provider"] == "firecrawl"
        assert result["meta"]["provider"] == "parallel"
        assert result["meta"]["fallback_reason"] == "rate limited by provider"
        primary.search.assert_called_once_with("docs", 3)
        fallback.search.assert_called_once_with("docs", 3)


class TestWebSearchMatrix:
    """Test suite for web_search_matrix provider coverage reporting."""

    class FakeProvider:
        def __init__(self, name, *, available=True, response=None):
            self.name = name
            self.display_name = name.title()
            self._available = available
            self._response = response or {
                "success": True,
                "data": {
                    "web": [
                        {
                            "title": f"{name} result",
                            "url": f"https://{name}.example/result",
                            "description": "result",
                            "position": 1,
                        }
                    ]
                },
            }

        def supports_search(self):
            return True

        def is_available(self):
            return self._available

        def search(self, query, limit):
            return self._response

    def test_matrix_reports_degraded_coverage_for_unavailable_provider(self):
        import tools.web_tools

        exa = self.FakeProvider("exa")
        firecrawl = self.FakeProvider("firecrawl", available=False)
        providers = {"exa": exa, "firecrawl": firecrawl}

        with patch("agent.web_search_registry.list_providers", return_value=[exa, firecrawl]), \
             patch("agent.web_search_registry.get_provider", side_effect=lambda name: providers.get(name)), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            result = json.loads(tools.web_tools.web_search_matrix_tool("ontology source registry", limit=2))

        assert result["success"] is True
        assert result["coverage_status"] == "degraded"
        assert result["degraded_coverage"] is True
        assert result["providers_succeeded"] == 1
        by_provider = {item["provider"]: item for item in result["providers"]}
        assert by_provider["exa"]["status"] == "ok"
        assert by_provider["firecrawl"]["status"] == "unavailable"

    def test_required_provider_failure_makes_matrix_unsuccessful(self):
        import tools.web_tools

        exa = self.FakeProvider("exa")
        firecrawl = self.FakeProvider("firecrawl", available=False)
        providers = {"exa": exa, "firecrawl": firecrawl}

        with patch("agent.web_search_registry.list_providers", return_value=[exa, firecrawl]), \
             patch("agent.web_search_registry.get_provider", side_effect=lambda name: providers.get(name)), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            result = json.loads(
                tools.web_tools.web_search_matrix_tool(
                    "ontology source registry",
                    providers=["exa", "firecrawl"],
                    required_providers=["firecrawl"],
                )
            )

        assert result["success"] is False
        assert result["coverage_status"] == "degraded"
        assert "firecrawl" in result["error"]


class TestWebSearchErrorHandling:
    """Test suite for web_search_tool() error responses."""

    def test_search_error_response_does_not_expose_diagnostics(self):
        import tools.web_tools

        # After the web-provider plugin migration, the firecrawl client lives
        # at plugins.web.firecrawl.provider._get_firecrawl_client. We mock the
        # registry's get_provider to return a fake provider whose .search()
        # raises so we can verify error sanitization.
        fake_provider = MagicMock(
            name="FirecrawlWebSearchProvider",
            supports_search=MagicMock(return_value=True),
        )
        fake_provider.search.side_effect = RuntimeError("boom")
        fake_provider.name = "firecrawl"

        with patch("tools.web_tools._get_search_backend", return_value="firecrawl"), \
             patch("agent.web_search_registry.get_provider", return_value=fake_provider), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(tools.web_tools._debug, "log_call") as mock_log_call, \
             patch.object(tools.web_tools._debug, "save"):
            result = json.loads(tools.web_tools.web_search_tool("test query", limit=3))

        assert result == {"error": "Error searching web: boom"}

        debug_payload = mock_log_call.call_args.args[1]
        assert debug_payload["error"] == "Error searching web: boom"
        assert "traceback" not in debug_payload["error"]
        assert "exception_type" not in debug_payload["error"]
        assert "config" not in result
        assert "exception_type" not in result
        assert "exception_chain" not in result
        assert "traceback" not in result


class TestWebExtractProviderStatus:
    """Provider degradation metadata survives web_extract result trimming."""

    @pytest.mark.asyncio
    async def test_trimmed_extract_output_keeps_provider_status(self):
        import tools.web_tools

        provider_status = {
            "provider": "firecrawl",
            "status": "degraded",
            "reason": "credit_exhausted",
            "operator_action_required": False,
            "policy": "Treat Firecrawl as optional degraded coverage.",
            "fallback_path": "Use Parallel search plus direct capture.",
        }

        class FakeProvider:
            name = "firecrawl"
            display_name = "Firecrawl"

            def supports_extract(self):
                return True

            async def extract(self, urls, **_kwargs):
                return [
                    {
                        "url": urls[0],
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": "Firecrawl degraded: optional degraded coverage.",
                        "provider_status": provider_status,
                    }
                ]

        with patch("tools.web_tools._get_extract_backend", return_value="firecrawl"), \
             patch("agent.web_search_registry.get_provider", return_value=FakeProvider()), \
             patch.object(tools.web_tools._debug, "log_call"), \
             patch.object(tools.web_tools._debug, "save"):
            result = json.loads(
                await tools.web_tools.web_extract_tool(
                    ["https://example.com/source.pdf"],
                    use_llm_processing=False,
                )
            )

        assert result["provider_status"] == provider_status
        assert result["results"][0]["provider_status"] == provider_status


class TestCheckWebApiKey:
    """Test suite for check_web_api_key() unified availability check."""

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
    )

    def setup_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)
        self._managed_patchers = [
            patch("tools.web_tools.managed_nous_tools_enabled", return_value=True),
            patch("tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True),
        ]
        for p in self._managed_patchers:
            p.start()

    def teardown_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)
        for p in self._managed_patchers:
            p.stop()

    def test_parallel_key_only(self):
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_exa_key_only(self):
        with patch.dict(os.environ, {"EXA_API_KEY": "exa-test"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_firecrawl_key_only(self):
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_firecrawl_url_only(self):
        with patch.dict(os.environ, {"FIRECRAWL_API_URL": "http://localhost:3002"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_tavily_key_only(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_no_keys_returns_false(self):
        from tools.web_tools import check_web_api_key
        with patch("tools.web_tools._ddgs_package_importable", return_value=False):
            assert check_web_api_key() is False

    def test_both_keys_returns_true(self):
        with patch.dict(os.environ, {
            "PARALLEL_API_KEY": "test-key",
            "FIRECRAWL_API_KEY": "fc-test",
        }):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_all_three_keys_returns_true(self):
        with patch.dict(os.environ, {
            "PARALLEL_API_KEY": "test-key",
            "FIRECRAWL_API_KEY": "fc-test",
            "TAVILY_API_KEY": "tvly-test",
        }):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_tool_gateway_returns_true(self):
        with patch("tools.web_tools._peek_nous_access_token", return_value="nous-token"):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_tool_gateway_availability_skips_refresh_for_expired_cached_token(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        expired_at = "2000-01-01T00:00:00+00:00"
        (tmp_path / "auth.json").write_text(json.dumps({
            "providers": {
                "nous": {
                    "access_token": "expired-token",
                    "refresh_token": "refresh-token",
                    "expires_at": expired_at,
                }
            }
        }))
        refresh_calls = []

        def _record_refresh(*, refresh_skew_seconds=120, **_kwargs):
            refresh_calls.append(refresh_skew_seconds)
            return "fresh-token"

        monkeypatch.setattr(
            "hermes_cli.auth.resolve_nous_access_token",
            _record_refresh,
        )

        with patch.dict(
            os.environ,
            {"FIRECRAWL_GATEWAY_URL": "http://127.0.0.1:3002"},
            clear=False,
        ):
            from tools.web_tools import check_web_api_key

            assert check_web_api_key() is True

        assert refresh_calls == []

    def test_configured_backend_must_match_available_provider(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "parallel"}):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch.dict(os.environ, {"FIRECRAWL_GATEWAY_URL": "http://127.0.0.1:3002"}, clear=False):
                    from tools.web_tools import check_web_api_key
                    assert check_web_api_key() is False

    def test_configured_firecrawl_backend_accepts_managed_gateway(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}):
            with patch("tools.web_tools._peek_nous_access_token", return_value="nous-token"):
                with patch.dict(os.environ, {"FIRECRAWL_GATEWAY_URL": "http://127.0.0.1:3002"}, clear=False):
                    from tools.web_tools import check_web_api_key
                    assert check_web_api_key() is True


def test_web_requires_env_includes_exa_key():
    from tools.web_tools import _web_requires_env

    assert "EXA_API_KEY" in _web_requires_env()


class TestNonBuiltinProviderAvailability:
    """Regression: a plugin-registered WebSearchProvider with no built-in
    provider credentials must still light up web_search / web_extract tools.

    The web_tools availability gate delegates non-legacy backend names to the
    web_search_registry's provider ``is_available()``.  This class verifies
    that a custom (non-built-in) provider discovered via the registry is
    sufficient to make check_web_api_key() return True, _get_backend() return
    the custom name, the per-capability selection honor it (issue #32698), and
    the tool registry entries remain active.

    Original tests contributed by @m0n5t3r (PR #28652 / issue #28651).
    """

    # All env vars that could make a built-in provider available.
    _WEB_ENV_KEYS = (
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_GATEWAY_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_SCHEME",
        "TOOL_GATEWAY_USER_TOKEN",
        "TAVILY_API_KEY",
        "SEARXNG_URL",
        "BRAVE_SEARCH_API_KEY",
        "XAI_API_KEY",
    )

    @staticmethod
    def _create_fake_provider(*, search=True, extract=True):
        """Dynamically create a WebSearchProvider subclass.

        Uses a local class definition (not a nested class) to avoid
        Python 3.13 __bases__ deallocator issue with nested class
        reassignment.
        """
        from agent.web_search_provider import WebSearchProvider

        class FakePluginProvider(WebSearchProvider):
            @property
            def name(self):
                return "fake-plugin-prov"

            def is_available(self):
                return True

            def supports_search(self):
                return search

            def supports_extract(self):
                return extract

        return FakePluginProvider()

    def setup_method(self):
        """Strip all built-in web provider env vars and reset the registry."""
        for key in self._WEB_ENV_KEYS:
            os.environ.pop(key, None)
        from agent.web_search_registry import _reset_for_tests, register_provider
        _reset_for_tests()
        register_provider(self._create_fake_provider())

    def teardown_method(self):
        """Reset the registry and restore env after each test."""
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()
        for key in self._WEB_ENV_KEYS:
            os.environ.pop(key, None)

    def test_check_web_api_key_returns_true_for_custom_provider(self):
        """With only a custom provider registered (no built-in creds),
        check_web_api_key() must return True."""
        with patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.web_tools._peek_nous_access_token", return_value=None):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_get_backend_discovers_custom_provider(self):
        """_get_backend() must return the custom provider name when it's
        the only available provider."""
        with patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.web_tools._peek_nous_access_token", return_value=None):
            from tools.web_tools import _get_backend
            assert _get_backend() == "fake-plugin-prov"

    def test_is_backend_available_delegates_to_registry(self):
        """_is_backend_available() must consult the registry for a
        non-legacy backend name."""
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("fake-plugin-prov") is True
        # Unknown, unregistered name -> False (no legacy probe matches).
        assert _is_backend_available("totally-unknown-backend") is False

    def test_capability_backend_honors_custom_extract_provider(self):
        """Per-capability selection (_get_extract_backend) must resolve the
        custom provider when configured, instead of dead-ending — issue #32698."""
        with patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.web_tools._peek_nous_access_token", return_value=None), \
             patch("tools.web_tools._load_web_config",
                   return_value={"extract_backend": "fake-plugin-prov"}):
            from tools.web_tools import _get_extract_backend
            assert _get_extract_backend() == "fake-plugin-prov"

    def test_tool_registry_entries_not_filtered_out(self):
        """web_search and web_extract tool entries must remain in the
        registry when only a custom provider is available."""
        with patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.web_tools._peek_nous_access_token", return_value=None):
            import tools.web_tools
            web_search_entry = tools.web_tools.registry.get_entry("web_search")
            web_extract_entry = tools.web_tools.registry.get_entry("web_extract")
            assert web_search_entry is not None, \
                "web_search tool was filtered out despite custom provider being available"
            assert web_extract_entry is not None, \
                "web_extract tool was filtered out despite custom provider being available"

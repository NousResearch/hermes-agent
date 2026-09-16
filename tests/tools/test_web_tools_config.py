"""Tests for web backend client configuration and singleton behavior.

Coverage:
  _get_firecrawl_client() — configuration matrix, singleton caching,
  constructor failure recovery, return value verification, edge cases.
  _get_backend() — backend selection logic with env var combinations.
  plugins.web.parallel.provider._get_sync_client() — Parallel client configuration, singleton caching.
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
        # tool_backend_helpers definition and the managed_tool_gateway import so the
        # full firecrawl client init path sees True.
        self._managed_patchers = [
            patch("tools.tool_backend_helpers.managed_nous_tools_enabled", return_value=True),
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
        with patch("plugins.web.firecrawl.provider.Firecrawl"):
            with patch("tools.managed_tool_gateway.read_nous_access_token", return_value=None):
                from plugins.web.firecrawl.provider import _get_firecrawl_client
                with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
                    _get_firecrawl_client()

    def test_tool_gateway_domain_builds_firecrawl_gateway_origin(self):
        """Shared gateway domain should derive the Firecrawl vendor hostname."""
        with patch.dict(os.environ, {"TOOL_GATEWAY_DOMAIN": "nousresearch.com"}):
            with patch("tools.managed_tool_gateway.read_nous_access_token", return_value="nous-token"):
                with patch("plugins.web.firecrawl.provider.Firecrawl") as mock_fc:
                    from plugins.web.firecrawl.provider import _get_firecrawl_client
                    result = _get_firecrawl_client()
                    mock_fc.assert_called_once_with(
                        api_key="nous-token",
                        api_url="https://firecrawl-gateway.nousresearch.com",
                    )
                    assert result is mock_fc.return_value


    # ── Singleton caching ────────────────────────────────────────────


    def test_constructor_failure_allows_retry(self):
        """If Firecrawl() raises, next call should retry (not return None)."""
        import tools.web_tools
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            with patch("plugins.web.firecrawl.provider.Firecrawl") as mock_fc:
                mock_fc.side_effect = [RuntimeError("init failed"), MagicMock()]
                from plugins.web.firecrawl.provider import _get_firecrawl_client

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
            with patch("plugins.web.firecrawl.provider.Firecrawl"):
                with patch("tools.managed_tool_gateway.read_nous_access_token", return_value=None):
                    from plugins.web.firecrawl.provider import _get_firecrawl_client
                    with pytest.raises(ValueError):
                        _get_firecrawl_client()

    def test_explicit_firecrawl_config_without_creds_uses_keyless_client(self):
        """Explicit Firecrawl config should build the keyless cloud client."""
        from plugins.web.firecrawl import provider as firecrawl_provider

        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}):
            with patch("tools.managed_tool_gateway.read_nous_access_token", return_value=None):
                with patch("plugins.web.firecrawl.provider.Firecrawl", side_effect=AssertionError("SDK path should not run")):
                    from plugins.web.firecrawl.provider import _get_firecrawl_client

                    result = _get_firecrawl_client()

        assert isinstance(result, firecrawl_provider._KeylessFirecrawlClient)
        assert result.api_url == "https://api.firecrawl.dev"

    def test_keyless_firecrawl_search_omits_authorization_header(self, monkeypatch):
        """Keyless Firecrawl search must not send a bearer header."""
        from plugins.web.firecrawl import provider as firecrawl_provider

        captured = {}

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"success": True, "data": {"web": []}}

        def _fake_post(url, *, json, headers, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return _Response()

        monkeypatch.setattr(firecrawl_provider.httpx, "post", _fake_post)

        client = firecrawl_provider._KeylessFirecrawlClient()
        result = client.search(query="firecrawl", limit=1)

        assert result["success"] is True
        assert captured["url"] == "https://api.firecrawl.dev/v2/search"
        assert captured["json"] == {"query": "firecrawl", "limit": 1}
        assert captured["headers"] == {"Content-Type": "application/json"}
        assert "Authorization" not in captured["headers"]

    def test_keyless_firecrawl_scrape_omits_authorization_header(self, monkeypatch):
        """Keyless Firecrawl scrape must not send a bearer header."""
        from plugins.web.firecrawl import provider as firecrawl_provider

        captured = {}

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"success": True, "data": {"markdown": "# ok"}}

        def _fake_post(url, *, json, headers, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return _Response()

        monkeypatch.setattr(firecrawl_provider.httpx, "post", _fake_post)

        client = firecrawl_provider._KeylessFirecrawlClient()
        result = client.scrape(url="https://example.com", formats=["markdown"])

        assert result["success"] is True
        assert captured["url"] == "https://api.firecrawl.dev/v2/scrape"
        assert captured["json"] == {"url": "https://example.com", "formats": ["markdown"]}
        assert captured["headers"] == {"Content-Type": "application/json"}
        assert "Authorization" not in captured["headers"]


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
        "KEENABLE_API_KEY",
        "TAVILY_API_KEY",
    )

    def setup_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)
        self._managed_patchers = [
            patch("tools.tool_backend_helpers.managed_nous_tools_enabled", return_value=True),
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
        """Direct-credential backends are tried in the order tavily > exa > parallel > keenable
        so an explicit Exa key wins when both Exa and Parallel are configured."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"EXA_API_KEY": "exa-test", "PARALLEL_API_KEY": "par-test"}):
            assert _get_backend() == "exa"

    def test_fallback_keenable_only_key(self):
        """Only KEENABLE_API_KEY set → 'keenable'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"KEENABLE_API_KEY": "kn-test"}):
            assert _get_backend() == "keenable"

    def test_fallback_exa_beats_firecrawl_direct(self):
        """Exa ranks above firecrawl in the explicit-credential block."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"EXA_API_KEY": "exa-test", "FIRECRAWL_API_KEY": "fc-test"}):
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

    def test_fallback_tavily_beats_exa(self):
        """Tavily ranks above Exa in the explicit-credential block."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test", "EXA_API_KEY": "exa-test"}):
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
        """No keys, no config, keyless tier off → 'firecrawl' sentinel.

        With the keyless tier on (default), zero credentials resolves to
        the Parallel/Exa free tier instead — covered in
        test_web_keyless_fallback.py.
        """
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False), \
             patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.web_tools._list_registered_web_providers", return_value=[]), \
             patch("agent.web_search_registry._keyless_tier_enabled", return_value=False):
            assert _get_backend() == "firecrawl"

    def test_invalid_config_is_returned_verbatim(self):
        """Strict selection: web.backend=nonexistent is returned as-is so the
        dispatch path raises the honest selection-naming error — never
        silently rerouted through the credential ladder."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "nonexistent"}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "nonexistent"

    def test_stored_backend_wins_over_other_credentials(self):
        """Strict selection: a stored web.backend beats env keys for other
        vendors — no availability probe, no credential override."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}), \
             patch.dict(os.environ, {"EXA_API_KEY": "exa-test"}):
            assert _get_backend() == "firecrawl"

    def test_nous_backend_maps_to_firecrawl(self):
        """The managed 'nous' selection is serviced by the firecrawl
        provider (whose client resolver routes managed)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "nous"}):
            assert _get_backend() == "firecrawl"

    def test_managed_gateway_does_not_preempt_explicit_exa(self):
        """Regression: a Nous OAuth token (managed gateway "ready") must NOT
        beat an explicitly configured EXA_API_KEY in the fallback path.
        Free Nous tiers don't include web search, so the user's deliberate
        Exa setup would fail at runtime with "no subscription" if the
        gateway pre-empted it."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=True), \
             patch.dict(os.environ, {"EXA_API_KEY": "exa-test"}):
            assert _get_backend() == "exa"

    def test_managed_gateway_does_not_preempt_explicit_tavily(self):
        """A Nous OAuth token must not beat an explicit TAVILY_API_KEY."""
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
            from plugins.web.parallel.provider import _get_sync_client as _get_parallel_client
            from parallel import Parallel
            client = _get_parallel_client()
            assert client is not None
            assert isinstance(client, Parallel)

    def test_no_key_raises_with_helpful_message(self):
        """No PARALLEL_API_KEY → ValueError with guidance."""
        from plugins.web.parallel.provider import _get_sync_client as _get_parallel_client
        with pytest.raises(ValueError, match="PARALLEL_API_KEY"):
            _get_parallel_client()

    def test_singleton_returns_same_instance(self):
        """Second call returns cached client."""
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            from plugins.web.parallel.provider import _get_sync_client as _get_parallel_client
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
        "KEENABLE_API_KEY",
        "TAVILY_API_KEY",
    )

    def setup_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)
        self._managed_patchers = [
            patch("tools.tool_backend_helpers.managed_nous_tools_enabled", return_value=True),
            patch("tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True),
            # ddgs availability is package-presence driven and the plugin
            # registry can hold an available ddgs provider. Neutralize both
            # fallback surfaces so this class only exercises env-key/gateway
            # resolution — otherwise these tests flip on machines where the
            # optional ``ddgs`` package is installed (dev venvs) vs CI.
            patch("tools.web_tools._ddgs_package_importable", return_value=False),
            patch("agent.web_search_registry.get_active_search_provider", return_value=None),
            patch("agent.web_search_registry.get_active_extract_provider", return_value=None),
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

    def test_null_backend_value_does_not_crash(self):
        # config.yaml with ``web:\n  backend:`` yields backend=None. The gate
        # must not raise AttributeError on None.lower() — mirrors _get_backend.
        with patch("tools.web_tools._load_web_config", return_value={"backend": None}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is False


    def test_configured_firecrawl_backend_accepts_managed_gateway(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}):
            with patch("tools.managed_tool_gateway.peek_nous_access_token", return_value="nous-token"):
                with patch.dict(os.environ, {"FIRECRAWL_GATEWAY_URL": "http://127.0.0.1:3002"}, clear=False):
                    from tools.web_tools import check_web_api_key
                    assert check_web_api_key() is True

    def test_explicit_unavailable_active_provider_is_not_ready(self):
        """#78412: get_active_* may return a configured backend whose
        is_available() is False. check_web_api_key must still report False so
        doctor does not paint a green check for a backend that cannot run.
        """
        class _UnavailableProvider:
            name = "firecrawl"

            def is_available(self):
                return False

        unavailable = _UnavailableProvider()
        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}), \
             patch("tools.web_tools._is_backend_available", return_value=False), \
             patch(
                 "agent.web_search_registry.get_active_search_provider",
                 return_value=unavailable,
             ), \
             patch(
                 "agent.web_search_registry.get_active_extract_provider",
                 return_value=unavailable,
             ):
            from tools.web_tools import check_web_api_key, _provider_is_ready
            assert _provider_is_ready(unavailable) is False
            assert check_web_api_key() is False

    def test_explicit_available_active_provider_is_ready(self):
        """Registry-selected available provider still lights the gate."""
        class _AvailableProvider:
            name = "custom-ok"

            def is_available(self):
                return True

        available = _AvailableProvider()
        with patch("tools.web_tools._load_web_config", return_value={"backend": "custom-ok"}), \
             patch("tools.web_tools._is_backend_available", return_value=False), \
             patch(
                 "agent.web_search_registry.get_active_search_provider",
                 return_value=available,
             ), \
             patch(
                 "agent.web_search_registry.get_active_extract_provider",
                 return_value=None,
             ):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True


def test_web_requires_env_includes_exa_key():
    from tools.web_tools import _web_requires_env

    env = _web_requires_env()
    assert "EXA_API_KEY" in env
    assert "TAVILY_API_KEY" in env


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
        "KEENABLE_API_KEY",
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
             patch("tools.managed_tool_gateway.peek_nous_access_token", return_value=None):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_get_backend_discovers_custom_provider(self):
        """_get_backend() must return the custom provider name when it's
        the only available provider."""
        with patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.managed_tool_gateway.peek_nous_access_token", return_value=None):
            from tools.web_tools import _get_backend
            assert _get_backend() == "fake-plugin-prov"


    def test_capability_backend_honors_custom_extract_provider(self):
        """Per-capability selection (_get_extract_backend) must resolve the
        custom provider when configured, instead of dead-ending — issue #32698."""
        with patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.managed_tool_gateway.peek_nous_access_token", return_value=None), \
             patch("tools.web_tools._load_web_config",
                   return_value={"extract_backend": "fake-plugin-prov"}):
            from tools.web_tools import _get_extract_backend
            assert _get_extract_backend() == "fake-plugin-prov"

    def test_tool_registry_entries_not_filtered_out(self):
        """web_search and web_extract tool entries must remain in the
        registry when only a custom provider is available."""
        with patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.managed_tool_gateway.peek_nous_access_token", return_value=None):
            import tools.web_tools
            web_search_entry = tools.web_tools.registry.get_entry("web_search")
            web_extract_entry = tools.web_tools.registry.get_entry("web_extract")
            assert web_search_entry is not None, \
                "web_search tool was filtered out despite custom provider being available"
            assert web_extract_entry is not None, \
                "web_extract tool was filtered out despite custom provider being available"


class TestFirecrawlEnvResolution:
    """Verify Firecrawl reads env values from hermes_cli.config.get_env_value,
    not just os.getenv.  This catches the regression reported in #40190 where
    values stored in ~/.hermes/.env were invisible to the provider."""

    def test_direct_config_reads_via_get_env_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_get_direct_firecrawl_config() must use get_env_value, not os.getenv."""
        # Ensure os.environ does NOT carry the key
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)

        fake_key = "fc-test-key-from-dotenv"
        with patch(
            "hermes_cli.config.get_env_value",
            side_effect=lambda k: fake_key if k == "FIRECRAWL_API_KEY" else None,
        ):
            from plugins.web.firecrawl.provider import _get_direct_firecrawl_config

            result = _get_direct_firecrawl_config()
            assert result is not None, "get_env_value fallback should find the key"
            mode, kwargs, _cache_key = result
            assert mode == "sdk"
            assert kwargs["api_key"] == fake_key

    def test_direct_config_reads_url_via_get_env_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Self-hosted URL from .env must be picked up."""
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)

        fake_url = "https://firecrawl.internal.example.com"
        with patch(
            "hermes_cli.config.get_env_value",
            side_effect=lambda k: fake_url if k == "FIRECRAWL_API_URL" else None,
        ):
            from plugins.web.firecrawl.provider import _get_direct_firecrawl_config

            result = _get_direct_firecrawl_config()
            assert result is not None
            mode, kwargs, _cache_key = result
            assert mode == "sdk"
            assert kwargs["api_url"] == fake_url.rstrip("/")


class TestSiblingProvidersEnvResolution:
    """The same #40190 bug class widened: every keyed web provider must
    resolve its credential through the config-aware lookup (os.environ OR
    ~/.hermes/.env), not bare os.getenv. Parametrized over the four
    providers that previously read only the process environment."""

    _CASES = [
        ("plugins.web.exa.provider", "ExaWebSearchProvider", "EXA_API_KEY"),
        ("plugins.web.parallel.provider", "ParallelWebSearchProvider", "PARALLEL_API_KEY"),
        ("plugins.web.keenable.provider", "KeenableWebSearchProvider", "KEENABLE_API_KEY"),
        ("plugins.web.tavily.provider", "TavilyWebSearchProvider", "TAVILY_API_KEY"),
        ("plugins.web.brave_free.provider", "BraveFreeWebSearchProvider", "BRAVE_SEARCH_API_KEY"),
    ]

    @pytest.mark.parametrize("module_path,cls_name,env_key", _CASES)
    def test_is_available_reads_via_get_env_value(
        self, monkeypatch, module_path, cls_name, env_key
    ):
        """is_available() must see a key that lives only in the .env layer."""
        monkeypatch.delenv(env_key, raising=False)

        import importlib
        module = importlib.import_module(module_path)
        provider = getattr(module, cls_name)()

        assert provider.is_available() is False

        with patch(
            "hermes_cli.config.get_env_value",
            side_effect=lambda k: "test-key-from-dotenv" if k == env_key else None,
        ):
            assert provider.is_available() is True, (
                f"{cls_name}.is_available() ignored {env_key} from the "
                "config-aware env layer (get_env_value)"
            )

    def test_keenable_search_reads_key_via_get_env_value(self, monkeypatch):
        """Keyed Keenable must Bearer-auth with a key that lives only in .env."""
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_response.text = "{}"

        with patch(
            "hermes_cli.config.get_env_value",
            side_effect=lambda k: "kn-from-dotenv" if k == "KEENABLE_API_KEY" else None,
        ), patch(
            "requests.post", return_value=mock_response
        ) as mock_post:
            from plugins.web.keenable.provider import KeenableWebSearchProvider

            KeenableWebSearchProvider().search("q", limit=2)
            headers = mock_post.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer kn-from-dotenv"
            assert headers["X-Keenable-Title"] == "hermes-agent"

    def test_tavily_request_reads_key_via_get_env_value(self, monkeypatch):
        """Keyed Tavily must Bearer-auth with a key that lives only in .env."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_response.text = "{}"

        with patch(
            "hermes_cli.config.get_env_value",
            side_effect=lambda k: "tvly-from-dotenv" if k == "TAVILY_API_KEY" else None,
        ), patch(
            "plugins.web.tavily.provider.httpx.post", return_value=mock_response
        ) as mock_post:
            from plugins.web.tavily.provider import _tavily_request

            _tavily_request("search", {"query": "q"})
            headers = mock_post.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer tvly-from-dotenv"
            assert headers["X-Client-Name"] == "hermes-agent"
            assert "X-Tavily-Access-Mode" not in headers


    def test_get_provider_env_unset_returns_empty(self, monkeypatch):
        monkeypatch.delenv("WSP_TEST_UNSET_KEY", raising=False)
        with patch("hermes_cli.config.get_env_value", return_value=None):
            from agent.web_search_provider import get_provider_env

            assert get_provider_env("WSP_TEST_UNSET_KEY") == ""

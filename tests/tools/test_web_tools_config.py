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


class _AsyncClientStub:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url):
        return self._response


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

    def test_check_auxiliary_model_re_resolves_backend_each_call(self):
        """Availability checks should not be pinned to module import state."""
        import tools.web_tools

        # Simulate the pre-fix import-time cache slot for regression coverage.
        tools.web_tools.__dict__["_aux_async_client"] = None

        with patch(
            "tools.web_tools.get_async_text_auxiliary_client",
            side_effect=[(None, None), (MagicMock(base_url="https://api.openrouter.ai/v1"), "test-model")],
        ):
            assert tools.web_tools.check_auxiliary_model() is False
            assert tools.web_tools.check_auxiliary_model() is True

    @pytest.mark.asyncio
    async def test_summarizer_re_resolves_backend_after_initial_unavailable_state(self):
        """Summarization should pick up a backend that becomes available later in-process."""
        import tools.web_tools

        tools.web_tools.__dict__["_aux_async_client"] = None

        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="summary text"))]

        with patch(
            "tools.web_tools._resolve_web_extract_auxiliary",
            side_effect=[(None, None, {}), (MagicMock(base_url="https://api.openrouter.ai/v1"), "test-model", {})],
        ), patch(
            "tools.web_tools.async_call_llm",
            new=AsyncMock(return_value=response),
        ) as mock_async_call:
            assert tools.web_tools.check_auxiliary_model() is False
            result = await tools.web_tools._call_summarizer_llm(
                "Some content worth summarizing",
                "Source: https://example.com\n\n",
                None,
            )

        assert result == "summary text"
        mock_async_call.assert_awaited_once()

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

    def test_fallback_parallel_takes_priority_over_exa(self):
        """Exa should only win the fallback path when it is the only configured backend."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"EXA_API_KEY": "exa-test", "PARALLEL_API_KEY": "par-test"}):
            assert _get_backend() == "parallel"

    def test_fallback_tavily_only_key(self):
        """Only TAVILY_API_KEY set → 'tavily'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
            assert _get_backend() == "tavily"

    def test_fallback_tavily_with_firecrawl_prefers_firecrawl(self):
        """Tavily + Firecrawl keys, no config → 'firecrawl' (backward compat)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test", "FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "firecrawl"

    def test_fallback_tavily_with_parallel_prefers_parallel(self):
        """Tavily + Parallel keys, no config → 'parallel' (Parallel takes priority over Tavily)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test", "PARALLEL_API_KEY": "par-test"}):
            # Parallel + no Firecrawl → parallel
            assert _get_backend() == "parallel"

    def test_fallback_both_keys_defaults_to_firecrawl(self):
        """Both keys set, no config → 'firecrawl' (backward compat)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key", "FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "firecrawl"

    def test_fallback_firecrawl_only_key(self):
        """Only FIRECRAWL_API_KEY set → 'firecrawl'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "firecrawl"

    def test_fallback_no_keys_defaults_to_duckduckgo(self):
        """No keys, no config → 'duckduckgo' fallback backend."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}):
            assert _get_backend() == "duckduckgo"

    def test_invalid_config_falls_through_to_fallback(self):
        """web.backend=invalid → ignored, uses key-based fallback."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "nonexistent"}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "parallel"


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


class TestWebSearchErrorHandling:
    """Test suite for web_search_tool() error responses."""

    def test_search_error_response_does_not_expose_diagnostics(self):
        import tools.web_tools

        firecrawl_client = MagicMock()
        firecrawl_client.search.side_effect = RuntimeError("boom")

        with patch("tools.web_tools._get_backend", return_value="firecrawl"), \
             patch("tools.web_tools._get_firecrawl_client", return_value=firecrawl_client), \
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


class TestDuckDuckGoFallbackDispatch:
    """Fallback dispatch tests when no provider backend is configured."""

    def test_search_dispatches_to_duckduckgo(self):
        import tools.web_tools

        expected = {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Duck Result",
                        "url": "https://example.com",
                        "description": "fallback result",
                        "position": 1,
                    }
                ]
            },
        }

        with patch("tools.web_tools._get_backend", return_value="duckduckgo"), \
             patch("tools.web_tools._duckduckgo_search", return_value=expected) as mock_search, \
             patch("tools.interrupt.is_interrupted", return_value=False):
            result = json.loads(tools.web_tools.web_search_tool("fallback query", limit=2))

        assert result == expected
        mock_search.assert_called_once_with("fallback query", 2)

    @pytest.mark.asyncio
    async def test_extract_dispatches_to_direct_http_fallback(self):
        import tools.web_tools

        expected = [
            {
                "url": "https://example.com",
                "title": "Example",
                "content": "hello",
                "raw_content": "hello",
                "metadata": {"sourceURL": "https://example.com"},
            }
        ]

        with patch("tools.web_tools._get_backend", return_value="duckduckgo"), \
             patch("tools.web_tools._direct_extract_urls", new=AsyncMock(return_value=expected)) as mock_extract, \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None):
            result = json.loads(await tools.web_tools.web_extract_tool(["https://example.com"], use_llm_processing=False))

        assert result["results"][0]["url"] == "https://example.com"
        assert result["results"][0]["content"] == "hello"
        mock_extract.assert_awaited_once_with(["https://example.com"], None)

    @pytest.mark.asyncio
    async def test_direct_extract_prefers_trafilatura_when_available(self):
        import tools.web_tools

        html = """
        <html>
          <head><title>Example Article</title></head>
          <body>
            <article><h1>Headline</h1><p>Alpha beta gamma.</p></article>
          </body>
        </html>
        """
        response = MagicMock()
        response.text = html
        response.headers = {"content-type": "text/html; charset=utf-8"}
        response.url = "https://example.com/article"
        response.raise_for_status = MagicMock()

        fake_trafilatura = types.SimpleNamespace(
            extract=MagicMock(return_value="# Headline\n\nAlpha beta gamma.")
        )

        with patch.dict(sys.modules, {"trafilatura": fake_trafilatura}), \
             patch("tools.web_tools.httpx.AsyncClient", return_value=_AsyncClientStub(response)), \
             patch("tools.web_tools.check_website_access", return_value=None):
            result = await tools.web_tools._direct_extract_urls(["https://example.com/article"], format="markdown")

        assert result[0]["title"] == "Example Article"
        assert result[0]["content"] == "# Headline\n\nAlpha beta gamma."
        assert result[0]["raw_content"] == "# Headline\n\nAlpha beta gamma."
        assert result[0]["excerpt"] == "Headline Alpha beta gamma."
        assert result[0]["metadata"]["sourceURL"] == "https://example.com/article"
        assert result[0]["metadata"]["contentType"] == "text/html; charset=utf-8"
        assert result[0]["metadata"]["extractor"] == "trafilatura"
        assert result[0]["metadata"]["fallbackUsed"] is True
        assert result[0]["metadata"]["contentLength"] == len("# Headline\n\nAlpha beta gamma.")
        assert result[0]["metadata"]["qualityStatus"] == "ok"
        assert result[0]["metadata"]["qualityFlags"] == []
        fake_trafilatura.extract.assert_called_once_with(
            html,
            url="https://example.com/article",
            output_format="markdown",
        )

    @pytest.mark.asyncio
    async def test_direct_extract_falls_back_to_html_cleanup_when_trafilatura_returns_none(self):
        import tools.web_tools

        html = """
        <html>
          <head><title>Example Article</title></head>
          <body>
            <article><h1>Headline</h1><p>Alpha beta gamma.</p></article>
          </body>
        </html>
        """
        response = MagicMock()
        response.text = html
        response.headers = {"content-type": "text/html; charset=utf-8"}
        response.url = "https://example.com/article"
        response.raise_for_status = MagicMock()

        fake_trafilatura = types.SimpleNamespace(extract=MagicMock(return_value=None))

        with patch.dict(sys.modules, {"trafilatura": fake_trafilatura}), \
             patch("tools.web_tools.httpx.AsyncClient", return_value=_AsyncClientStub(response)), \
             patch("tools.web_tools.check_website_access", return_value=None):
            result = await tools.web_tools._direct_extract_urls(["https://example.com/article"], format="markdown")

        assert "Headline" in result[0]["content"]
        assert "Alpha beta gamma." in result[0]["content"]
        assert result[0]["raw_content"] == result[0]["content"]
        assert result[0]["metadata"]["extractor"] == "html_cleanup"
        assert result[0]["metadata"]["qualityStatus"] == "ok"

    @pytest.mark.asyncio
    async def test_direct_extract_marks_thin_content(self):
        import tools.web_tools

        response = MagicMock()
        response.text = "ok"
        response.headers = {"content-type": "text/plain; charset=utf-8"}
        response.url = "https://example.com/tiny"
        response.raise_for_status = MagicMock()

        with patch("tools.web_tools.httpx.AsyncClient", return_value=_AsyncClientStub(response)), \
             patch("tools.web_tools.check_website_access", return_value=None):
            result = await tools.web_tools._direct_extract_urls(["https://example.com/tiny"], format="markdown")

        assert result[0]["content"] == "ok"
        assert result[0]["excerpt"] == "ok"
        assert result[0]["metadata"]["extractor"] == "plain_text"
        assert result[0]["metadata"]["qualityStatus"] == "thin"
        assert result[0]["metadata"]["qualityFlags"] == ["thin_content"]

    @pytest.mark.asyncio
    async def test_web_extract_output_keeps_excerpt_and_metadata(self):
        import tools.web_tools

        expected = [
            {
                "url": "https://example.com/article",
                "title": "Example",
                "content": "Body text",
                "raw_content": "Body text",
                "excerpt": "Body text",
                "metadata": {
                    "sourceURL": "https://example.com/article",
                    "extractor": "trafilatura",
                    "qualityStatus": "ok",
                    "qualityFlags": [],
                },
            }
        ]

        with patch("tools.web_tools._get_backend", return_value="duckduckgo"), \
             patch("tools.web_tools._direct_extract_urls", new=AsyncMock(return_value=expected)), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None):
            result = json.loads(await tools.web_tools.web_extract_tool(["https://example.com/article"], use_llm_processing=False))

        assert result["results"][0]["excerpt"] == "Body text"
        assert result["results"][0]["metadata"]["extractor"] == "trafilatura"
        assert result["results"][0]["metadata"]["qualityStatus"] == "ok"

    def test_web_extract_schema_exposes_dynamic_mode(self):
        from tools.web_tools import WEB_EXTRACT_SCHEMA

        props = WEB_EXTRACT_SCHEMA["parameters"]["properties"]
        assert "mode" in props
        assert props["mode"]["enum"] == ["static", "dynamic", "attached"]
        assert props["mode"]["default"] == "static"

    @pytest.mark.asyncio
    async def test_web_extract_dynamic_mode_uses_browser_rendering(self):
        import tools.web_tools

        with patch("tools.web_tools._get_backend", return_value="duckduckgo"), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.browser_tool.browser_navigate", return_value=json.dumps({
                 "success": True,
                 "url": "https://example.com/app",
                 "title": "Rendered App",
             })) as mock_nav, \
             patch("tools.browser_tool._browser_eval", return_value=json.dumps({
                 "success": True,
                 "result": {
                     "title": "Rendered App",
                     "content": "Loaded content from hydrated page",
                 },
             })) as mock_eval, \
             patch("tools.web_tools._direct_extract_urls", new=AsyncMock(return_value=[])) as mock_static:
            result = json.loads(await tools.web_tools.web_extract_tool(
                ["https://example.com/app"],
                mode="dynamic",
                use_llm_processing=False,
            ))

        first = result["results"][0]
        assert first["title"] == "Rendered App"
        assert first["content"] == "Loaded content from hydrated page"
        assert first["metadata"]["extractor"] == "browser_rendered"
        assert first["metadata"]["requestedMode"] == "dynamic"
        assert first["metadata"]["extractionMode"] == "dynamic"
        assert mock_nav.call_args.args[0] == "https://example.com/app"
        assert "document.body" in mock_eval.call_args.args[0]
        mock_static.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_web_extract_dynamic_mode_retries_thin_rendered_content_before_accepting_result(self):
        import tools.web_tools

        with patch("tools.web_tools._get_backend", return_value="duckduckgo"), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.browser_tool.browser_navigate", return_value=json.dumps({
                 "success": True,
                 "url": "https://example.com/app",
                 "title": "Rendered App",
             })), \
             patch("tools.browser_tool._browser_eval", side_effect=[
                 json.dumps({
                     "success": True,
                     "result": {
                         "title": "Rendered App",
                         "content": "Loading...",
                         "readyState": "interactive",
                         "contentScope": "body",
                         "observedContentLength": 10,
                     },
                 }),
                 json.dumps({
                     "success": True,
                     "result": {
                         "title": "Rendered App",
                         "content": "Loaded content from hydrated page after retry",
                         "readyState": "complete",
                         "contentScope": "main",
                         "observedContentLength": 44,
                     },
                 }),
             ]) as mock_eval, \
             patch("tools.web_tools.asyncio.sleep", new=AsyncMock()) as mock_sleep, \
             patch("tools.web_tools._direct_extract_urls", new=AsyncMock(return_value=[])) as mock_static:
            result = json.loads(await tools.web_tools.web_extract_tool(
                ["https://example.com/app"],
                mode="dynamic",
                use_llm_processing=False,
            ))

        first = result["results"][0]
        assert first["title"] == "Rendered App"
        assert first["content"] == "Loaded content from hydrated page after retry"
        assert first["metadata"]["extractor"] == "browser_rendered"
        assert first["metadata"]["requestedMode"] == "dynamic"
        assert first["metadata"]["extractionMode"] == "dynamic"
        assert first["metadata"]["renderAttemptCount"] == 2
        assert first["metadata"]["renderWaitStrategy"] == "retry_on_empty_or_thin_content"
        assert first["metadata"]["renderReadyState"] == "complete"
        assert first["metadata"]["renderContentScope"] == "main"
        assert first["metadata"]["renderObservedContentLength"] == 44
        assert mock_eval.call_count == 2
        mock_sleep.assert_awaited_once()
        mock_static.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_web_extract_dynamic_mode_falls_back_to_static_when_browser_eval_fails(self):
        import tools.web_tools

        static_expected = [
            {
                "url": "https://example.com/app",
                "title": "Static Example",
                "content": "Static fallback body",
                "raw_content": "Static fallback body",
                "excerpt": "Static fallback body",
                "metadata": {
                    "sourceURL": "https://example.com/app",
                    "extractor": "html_cleanup",
                    "qualityStatus": "ok",
                    "qualityFlags": [],
                },
            }
        ]

        with patch("tools.web_tools._get_backend", return_value="duckduckgo"), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.browser_tool.browser_navigate", return_value=json.dumps({
                 "success": True,
                 "url": "https://example.com/app",
                 "title": "Rendered App",
             })), \
             patch("tools.browser_tool._browser_eval", return_value=json.dumps({
                 "success": False,
                 "error": "JavaScript evaluation is not supported by this browser backend.",
             })), \
             patch("tools.web_tools._direct_extract_urls", new=AsyncMock(return_value=static_expected)) as mock_static:
            result = json.loads(await tools.web_tools.web_extract_tool(
                ["https://example.com/app"],
                mode="dynamic",
                use_llm_processing=False,
            ))

        first = result["results"][0]
        assert first["title"] == "Static Example"
        assert first["content"] == "Static fallback body"
        assert first["metadata"]["extractor"] == "html_cleanup"
        assert first["metadata"]["requestedMode"] == "dynamic"
        assert first["metadata"]["extractionMode"] == "static"
        assert first["metadata"]["dynamicFallbackUsed"] is True
        assert first["metadata"]["dynamicFallbackReason"] == "browser_eval_failed"
        mock_static.assert_awaited_once_with(["https://example.com/app"], "markdown")

    @pytest.mark.asyncio
    async def test_web_extract_attached_mode_marks_profile_and_login_state_unverified_without_explicit_evidence(self):
        import tools.web_tools

        with patch.dict(os.environ, {"BROWSER_CDP_URL": "ws://127.0.0.1:9222/devtools/browser/test"}, clear=False), \
             patch("tools.web_tools._get_backend", return_value="duckduckgo"), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.browser_tool.browser_navigate", return_value=json.dumps({
                 "success": True,
                 "url": "https://example.com/dashboard",
                 "title": "Attached Dashboard",
             })) as mock_nav, \
             patch("tools.browser_tool._browser_eval", return_value=json.dumps({
                 "success": True,
                 "result": {
                     "url": "https://example.com/dashboard",
                     "title": "Attached Dashboard",
                     "content": "Logged-in rendered content from attached session",
                     "evidence": {
                         "visibleCookieCount": 0,
                         "localStorageCount": 0,
                         "sessionStorageCount": 0,
                     },
                 },
             })) as mock_eval, \
             patch("tools.web_tools._direct_extract_urls", new=AsyncMock(return_value=[])) as mock_static:
            result = json.loads(await tools.web_tools.web_extract_tool(
                ["https://example.com/dashboard"],
                mode="attached",
                use_llm_processing=False,
            ))

        first = result["results"][0]
        assert first["title"] == "Attached Dashboard"
        assert first["content"] == "Logged-in rendered content from attached session"
        assert first["metadata"]["extractor"] == "browser_attached"
        assert first["metadata"]["requestedMode"] == "attached"
        assert first["metadata"]["extractionMode"] == "attached"
        assert first["metadata"]["cdpAttached"] is True
        assert first["metadata"]["cdpUrl"] == "ws://127.0.0.1:9222/devtools/browser/test"
        assert first["metadata"]["browserSessionReused"] is True
        assert first["metadata"]["profileInherited"] is None
        assert first["metadata"]["loginStateInherited"] is None
        assert first["metadata"]["inheritanceVerification"] == "not_verified"
        assert first["metadata"]["attachedClientStateEvidence"] == {
            "visibleCookieCount": 0,
            "localStorageCount": 0,
            "sessionStorageCount": 0,
        }
        assert mock_nav.call_args.args[0] == "https://example.com/dashboard"
        assert "document.body" in mock_eval.call_args.args[0]
        mock_static.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_web_extract_attached_mode_marks_client_state_observed_when_browser_evidence_exists(self):
        import tools.web_tools

        with patch.dict(os.environ, {"BROWSER_CDP_URL": "ws://127.0.0.1:9222/devtools/browser/test"}, clear=False), \
             patch("tools.web_tools._get_backend", return_value="duckduckgo"), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.browser_tool.browser_navigate", return_value=json.dumps({
                 "success": True,
                 "url": "https://example.com/dashboard",
                 "title": "Attached Dashboard",
             })), \
             patch("tools.browser_tool._browser_eval", return_value=json.dumps({
                 "success": True,
                 "result": {
                     "url": "https://example.com/dashboard",
                     "title": "Attached Dashboard",
                     "content": "Rendered content with browser state evidence",
                     "evidence": {
                         "visibleCookieCount": 2,
                         "localStorageCount": 1,
                         "sessionStorageCount": 0,
                     },
                 },
             })), \
             patch("tools.web_tools._direct_extract_urls", new=AsyncMock(return_value=[])):
            result = json.loads(await tools.web_tools.web_extract_tool(
                ["https://example.com/dashboard"],
                mode="attached",
                use_llm_processing=False,
            ))

        first = result["results"][0]
        assert first["metadata"]["inheritanceVerification"] == "client_state_observed"
        assert first["metadata"]["attachedClientStateEvidence"] == {
            "visibleCookieCount": 2,
            "localStorageCount": 1,
            "sessionStorageCount": 0,
        }

    @pytest.mark.asyncio
    async def test_web_extract_attached_mode_falls_back_to_static_when_browser_navigate_fails(self):
        import tools.web_tools

        static_expected = [
            {
                "url": "https://example.com/dashboard",
                "title": "Static Dashboard",
                "content": "Static fallback body",
                "raw_content": "Static fallback body",
                "excerpt": "Static fallback body",
                "metadata": {
                    "sourceURL": "https://example.com/dashboard",
                    "extractor": "html_cleanup",
                    "qualityStatus": "ok",
                    "qualityFlags": [],
                },
            }
        ]

        with patch.dict(os.environ, {"BROWSER_CDP_URL": "ws://127.0.0.1:9222/devtools/browser/test"}, clear=False), \
             patch("tools.web_tools._get_backend", return_value="duckduckgo"), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.browser_tool.browser_navigate", return_value=json.dumps({
                 "success": False,
                 "error": "Failed to attach to CDP session",
             })), \
             patch("tools.web_tools._direct_extract_urls", new=AsyncMock(return_value=static_expected)) as mock_static:
            result = json.loads(await tools.web_tools.web_extract_tool(
                ["https://example.com/dashboard"],
                mode="attached",
                use_llm_processing=False,
            ))

        first = result["results"][0]
        assert first["title"] == "Static Dashboard"
        assert first["metadata"]["requestedMode"] == "attached"
        assert first["metadata"]["extractionMode"] == "static"
        assert first["metadata"]["dynamicFallbackUsed"] is True
        assert first["metadata"]["dynamicFallbackReason"] == "browser_navigate_failed"
        mock_static.assert_awaited_once_with(["https://example.com/dashboard"], "markdown")


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

    def test_no_keys_returns_true_via_duckduckgo_fallback(self):
        from tools.web_tools import check_web_api_key
        assert check_web_api_key() is True

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
        with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_configured_backend_must_match_available_provider(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "parallel"}):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch.dict(os.environ, {"FIRECRAWL_GATEWAY_URL": "http://127.0.0.1:3002"}, clear=False):
                    from tools.web_tools import check_web_api_key
                    assert check_web_api_key() is False

    def test_configured_parallel_without_key_returns_false(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "parallel"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is False

    def test_configured_exa_without_key_returns_false(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "exa"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is False

    def test_configured_firecrawl_backend_accepts_managed_gateway(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch.dict(os.environ, {"FIRECRAWL_GATEWAY_URL": "http://127.0.0.1:3002"}, clear=False):
                    from tools.web_tools import check_web_api_key
                    assert check_web_api_key() is True


def test_web_requires_env_includes_exa_key():
    from tools.web_tools import _web_requires_env

    assert "EXA_API_KEY" in _web_requires_env()

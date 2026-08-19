"""Keyless free-tier web search/extract fallback (Parallel + Exa MCP).

Covers:
- keyless_mcp response parsing (SSE + plain JSON, error shapes)
- provider keyless routing: no key -> keyless path; key present -> SDK path
- registry keyless walk: fires only when nothing is keyed; respects
  web.keyless_fallback: false
- _get_backend() keyless tier: strictly after every keyed candidate
- check_web_api_key() lights up on a zero-credential install
"""

import json
from unittest.mock import patch

import pytest

import tools.web_tools as web_tools
from agent import web_search_registry as registry
from plugins.web import keyless_mcp
from plugins.web.exa.provider import ExaWebSearchProvider
from plugins.web.parallel.provider import ParallelWebSearchProvider


@pytest.fixture(autouse=True)
def _no_web_env(monkeypatch):
    """Blank every web credential and neutralize config lookups."""
    for var in (
        "EXA_API_KEY", "PARALLEL_API_KEY", "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "BRAVE_SEARCH_API_KEY",
        "SEARXNG_URL", "TOOL_GATEWAY_USER_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env", lambda name: "", raising=True
    )
    monkeypatch.setattr(web_tools, "_env_value", lambda name: "", raising=True)
    monkeypatch.setattr(web_tools, "_load_web_config", dict, raising=True)
    monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False, raising=True)
    monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False, raising=True)
    yield


@pytest.fixture()
def fresh_registry():
    """Isolated registry snapshot with real exa/parallel providers."""
    with registry._lock:
        saved = dict(registry._providers)
        saved_scoped = {k: dict(v) for k, v in registry._scoped_providers.items()}
        registry._providers.clear()
        registry._scoped_providers.clear()
    registry.register_provider(ParallelWebSearchProvider())
    registry.register_provider(ExaWebSearchProvider())
    yield registry
    with registry._lock:
        registry._providers.clear()
        registry._providers.update(saved)
        registry._scoped_providers.clear()
        registry._scoped_providers.update(saved_scoped)


# ---------------------------------------------------------------------------
# keyless_mcp parsing
# ---------------------------------------------------------------------------


class TestParseMcpBody:
    def test_sse_body(self):
        payload = {"result": {"content": [{"type": "text", "text": "hello"}]}}
        body = f"event: message\ndata: {json.dumps(payload)}\n\n"
        assert keyless_mcp._parse_mcp_body(body) == "hello"

    def test_plain_json_body(self):
        payload = {"result": {"content": [{"type": "text", "text": "hi"}]}}
        assert keyless_mcp._parse_mcp_body(json.dumps(payload)) == "hi"

    def test_jsonrpc_error_raises(self):
        body = json.dumps({"error": {"code": -32000, "message": "rate limit"}})
        with pytest.raises(keyless_mcp.KeylessMCPError, match="rate limit"):
            keyless_mcp._parse_mcp_body(body)

    def test_is_error_result_raises(self):
        body = json.dumps(
            {"result": {"isError": True, "content": [{"type": "text", "text": "boom"}]}}
        )
        with pytest.raises(keyless_mcp.KeylessMCPError, match="boom"):
            keyless_mcp._parse_mcp_body(body)

    def test_garbage_raises(self):
        with pytest.raises(keyless_mcp.KeylessMCPError):
            keyless_mcp._parse_mcp_body("<html>nope</html>")


class TestExaTextParsing:
    def test_parses_blocks(self):
        text = (
            "Title: First\nURL: https://a.example\nPublished: N/A\n"
            "Highlights:\nsome highlight\nmore\n"
            "\n---\n"
            "Title: Second\nURL: https://b.example\nHighlights:\nother\n"
        )
        results = keyless_mcp._parse_exa_search_text(text, limit=5)
        assert [r["url"] for r in results] == ["https://a.example", "https://b.example"]
        assert results[0]["description"] == "some highlight more"
        assert results[0]["position"] == 1

    def test_limit_respected(self):
        text = "\n---\n".join(
            f"Title: T{i}\nURL: https://x{i}.example" for i in range(6)
        )
        assert len(keyless_mcp._parse_exa_search_text(text, limit=2)) == 2


class TestKeylessCalls:
    def test_parallel_search_shapes_results(self):
        payload = json.dumps(
            {
                "results": [
                    {"url": "https://a", "title": "A", "excerpts": ["x", "y"]},
                    {"url": "https://b", "title": "B", "excerpts": []},
                ]
            }
        )
        with patch.object(keyless_mcp, "mcp_call", return_value=payload) as call:
            out = keyless_mcp.parallel_search_keyless("query", limit=5)
        assert out["success"] is True
        assert out["data"]["web"][0] == {
            "url": "https://a", "title": "A", "description": "x y", "position": 1,
        }
        args = call.call_args[0]
        assert args[0] == keyless_mcp.PARALLEL_MCP_URL
        assert args[1] == "web_search"
        assert "model_name" not in args[2]  # analytics field deliberately omitted

    def test_parallel_search_failure_mentions_key_setup(self):
        with patch.object(
            keyless_mcp, "mcp_call", side_effect=keyless_mcp.KeylessMCPError("429")
        ):
            out = keyless_mcp.parallel_search_keyless("q")
        assert out["success"] is False
        assert "PARALLEL_API_KEY" in out["error"]

    def test_parallel_extract_covers_missing_urls(self):
        payload = json.dumps({"results": [{"url": "https://a", "title": "A", "excerpts": ["c"]}]})
        with patch.object(keyless_mcp, "mcp_call", return_value=payload):
            out = keyless_mcp.parallel_extract_keyless(["https://a", "https://gone"])
        assert out[0]["content"] == "c"
        assert out[1]["url"] == "https://gone"
        assert "error" in out[1]

    def test_exa_search_rate_limit_is_soft_error(self):
        with patch.object(
            keyless_mcp, "mcp_call",
            side_effect=keyless_mcp.KeylessMCPError("free MCP rate limit"),
        ):
            out = keyless_mcp.exa_search_keyless("q")
        assert out["success"] is False
        assert "EXA_API_KEY" in out["error"]

    def test_exa_extract_per_url(self):
        with patch.object(
            keyless_mcp, "mcp_call", return_value="# Page Title\nbody text"
        ) as call:
            out = keyless_mcp.exa_extract_keyless(["https://a", "https://b"])
        assert call.call_count == 2
        assert out[0]["title"] == "Page Title"
        assert out[0]["content"].startswith("# Page Title")


# ---------------------------------------------------------------------------
# Provider routing: keyless vs keyed
# ---------------------------------------------------------------------------


class TestProviderRouting:
    def test_parallel_keyless_path_when_no_key(self):
        provider = ParallelWebSearchProvider()
        with patch.object(
            keyless_mcp, "parallel_search_keyless",
            return_value={"success": True, "data": {"web": []}},
        ) as keyless:
            out = provider.search("q", limit=3)
        assert out["success"] is True
        keyless.assert_called_once_with("q", 3)

    def test_exa_keyless_path_when_no_key(self):
        provider = ExaWebSearchProvider()
        with patch.object(
            keyless_mcp, "exa_search_keyless",
            return_value={"success": True, "data": {"web": []}},
        ) as keyless:
            out = provider.search("q", limit=3)
        assert out["success"] is True
        keyless.assert_called_once_with("q", 3)

    def test_parallel_keyed_path_skips_keyless(self, monkeypatch):
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env",
            lambda name: "sk-real" if name == "PARALLEL_API_KEY" else "",
        )
        provider = ParallelWebSearchProvider()
        with patch.object(keyless_mcp, "parallel_search_keyless") as keyless, \
                patch("plugins.web.parallel.provider._get_sync_client") as client:
            client.return_value.beta.search.return_value.results = []
            out = provider.search("q")
        keyless.assert_not_called()
        assert out["success"] is True

    def test_keyless_disabled_falls_through_to_key_error(self, monkeypatch):
        monkeypatch.setattr(registry, "_keyless_tier_enabled", lambda: False)
        provider = ParallelWebSearchProvider()
        out = provider.search("q")
        assert out["success"] is False
        assert "PARALLEL_API_KEY" in out["error"]

    def test_is_available_stays_false_keyless(self):
        # Keyless tier must NOT leak into is_available() (legacy walk order).
        assert ParallelWebSearchProvider().is_available() is False
        assert ExaWebSearchProvider().is_available() is False
        assert ParallelWebSearchProvider().is_keyless_available() is True
        assert ExaWebSearchProvider().is_keyless_available() is True

    def test_tier_free_forces_keyless_even_with_key(self, monkeypatch):
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env",
            lambda name: "sk-real" if name == "PARALLEL_API_KEY" else "",
        )
        monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "free")
        provider = ParallelWebSearchProvider()
        with patch.object(
            keyless_mcp, "parallel_search_keyless",
            return_value={"success": True, "data": {"web": []}},
        ) as keyless:
            out = provider.search("q")
        keyless.assert_called_once()
        assert out["success"] is True

    def test_tier_paid_forces_keyed_without_key(self, monkeypatch):
        monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "paid")
        provider = ParallelWebSearchProvider()
        with patch.object(keyless_mcp, "parallel_search_keyless") as keyless:
            out = provider.search("q")
        keyless.assert_not_called()
        assert out["success"] is False
        assert "PARALLEL_API_KEY" in out["error"]

    def test_tier_paid_disables_keyless_availability(self, monkeypatch):
        monkeypatch.setattr(keyless_mcp, "provider_tier", lambda name: "paid")
        assert ParallelWebSearchProvider().is_keyless_available() is False
        assert ExaWebSearchProvider().is_keyless_available() is False

    def test_provider_tier_reads_config(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"web": {"provider_tier": {"exa": "FREE", "parallel": "bogus"}}},
        )
        assert keyless_mcp.provider_tier("exa") == "free"
        assert keyless_mcp.provider_tier("parallel") == "auto"  # invalid → auto
        assert keyless_mcp.provider_tier("tavily") == "auto"    # unset → auto

    @pytest.mark.asyncio
    async def test_parallel_keyless_extract(self):
        provider = ParallelWebSearchProvider()
        with patch.object(
            keyless_mcp, "parallel_extract_keyless",
            return_value=[{"url": "https://a", "title": "", "content": "c"}],
        ) as keyless:
            out = await provider.extract(["https://a"])
        assert out[0]["content"] == "c"
        keyless.assert_called_once_with(["https://a"])


# ---------------------------------------------------------------------------
# Registry + _get_backend resolution order
# ---------------------------------------------------------------------------


class TestResolutionOrder:
    def test_registry_falls_back_to_keyless_parallel(self, fresh_registry, monkeypatch):
        monkeypatch.setattr(registry, "_read_config_key", lambda *p: None)
        provider = registry.get_active_search_provider()
        assert provider is not None
        assert provider.name == "parallel"  # _KEYLESS_PREFERENCE order

    def test_registry_keyless_disabled_returns_none(self, fresh_registry, monkeypatch):
        monkeypatch.setattr(registry, "_read_config_key", lambda *p: None)
        monkeypatch.setattr(registry, "_keyless_tier_enabled", lambda: False)
        assert registry.get_active_search_provider() is None

    def test_keyed_provider_beats_keyless(self, fresh_registry, monkeypatch):
        # Exa keyed, Parallel keyless: legacy walk must pick exa (keyed)
        # even though parallel precedes exa in _KEYLESS_PREFERENCE.
        monkeypatch.setattr(registry, "_read_config_key", lambda *p: None)
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env",
            lambda name: "sk-real" if name == "EXA_API_KEY" else "",
        )
        provider = registry.get_active_search_provider()
        assert provider is not None and provider.name == "exa"

    def test_get_backend_keyless_last(self, monkeypatch):
        # No creds at all -> keyless parallel.
        monkeypatch.setattr(
            web_tools, "_registered_web_provider",
            lambda name: {"parallel": ParallelWebSearchProvider(),
                          "exa": ExaWebSearchProvider()}.get(name),
        )
        monkeypatch.setattr(web_tools, "_list_registered_web_providers", list)
        assert web_tools._get_backend() == "parallel"

    def test_get_backend_key_beats_keyless(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_env_value",
            lambda name: "sk-x" if name == "TAVILY_API_KEY" else "",
        )
        assert web_tools._get_backend() == "tavily"

    def test_get_backend_keyless_disabled(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_registered_web_provider",
            lambda name: {"parallel": ParallelWebSearchProvider(),
                          "exa": ExaWebSearchProvider()}.get(name),
        )
        monkeypatch.setattr(web_tools, "_list_registered_web_providers", list)
        monkeypatch.setattr(registry, "_keyless_tier_enabled", lambda: False)
        assert web_tools._get_backend() == "firecrawl"  # legacy sentinel

    def test_check_web_api_key_true_on_keyless_install(self, fresh_registry, monkeypatch):
        monkeypatch.setattr(registry, "_read_config_key", lambda *p: None)
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(web_tools, "check_firecrawl_api_key", lambda: False)
        assert web_tools.check_web_api_key() is True

    def test_check_web_api_key_false_when_disabled(self, fresh_registry, monkeypatch):
        monkeypatch.setattr(registry, "_read_config_key", lambda *p: None)
        monkeypatch.setattr(registry, "_keyless_tier_enabled", lambda: False)
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(web_tools, "check_firecrawl_api_key", lambda: False)
        assert web_tools.check_web_api_key() is False


# ---------------------------------------------------------------------------
# hermes tools picker: tier variant rows
# ---------------------------------------------------------------------------


class TestPickerTierRows:
    def test_variant_schemas_flatten_to_tier_rows(self, fresh_registry, monkeypatch):
        from hermes_cli import tools_config

        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered", lambda: None
        )
        rows = tools_config._plugin_web_search_providers()
        by_backend_tier = {
            (r["web_backend"], r.get("web_tier")): r["name"] for r in rows
        }
        assert ("parallel", "free") in by_backend_tier
        assert ("parallel", "paid") in by_backend_tier
        assert ("exa", "free") in by_backend_tier
        assert ("exa", "paid") in by_backend_tier
        # Free rows must not prompt for a key; paid rows must.
        for r in rows:
            if r.get("web_tier") == "free":
                assert r["env_vars"] == []
            if r.get("web_tier") == "paid":
                assert r["env_vars"], r

    def test_selection_persists_tier(self):
        from hermes_cli.tools_config import _write_provider_config

        config: dict = {}
        _write_provider_config(
            {"web_backend": "exa", "web_tier": "free", "env_vars": []},
            config,
            managed_feature=None,
        )
        assert config["web"]["backend"] == "exa"
        assert config["web"]["provider_tier"]["exa"] == "free"
        # Re-selecting a tier-agnostic row clears the stale tier.
        _write_provider_config(
            {"web_backend": "exa", "env_vars": []}, config, managed_feature=None
        )
        assert "exa" not in config["web"]["provider_tier"]

    def test_tier_match_highlights_correct_row(self):
        from hermes_cli.tools_config import _web_tier_matches

        free_row = {"web_backend": "parallel", "web_tier": "free"}
        paid_row = {"web_backend": "parallel", "web_tier": "paid"}
        cfg_free = {"web": {"backend": "parallel", "provider_tier": {"parallel": "free"}}}
        cfg_paid = {"web": {"backend": "parallel", "provider_tier": {"parallel": "paid"}}}
        assert _web_tier_matches(free_row, cfg_free) is True
        assert _web_tier_matches(paid_row, cfg_free) is False
        assert _web_tier_matches(paid_row, cfg_paid) is True
        assert _web_tier_matches(free_row, cfg_paid) is False
        # Auto (unset tier, no key in the hermetic env): free row highlights.
        cfg_auto = {"web": {"backend": "parallel"}}
        assert _web_tier_matches(free_row, cfg_auto) is True
        assert _web_tier_matches(paid_row, cfg_auto) is False

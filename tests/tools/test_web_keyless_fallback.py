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
    def test_parallel_keyless_path_when_no_key(self, monkeypatch):
        # Pin parallel so the ring deterministically starts there.
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "parallel")
        provider = ParallelWebSearchProvider()
        with patch.dict(
            keyless_mcp._KEYLESS_SEARCHERS,
            {"parallel": lambda q, l: {"success": True, "data": {"web": []}}},
        ):
            out = provider.search("q", limit=3)
        assert out["success"] is True

    def test_exa_keyless_path_when_no_key(self, monkeypatch):
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "exa")
        provider = ExaWebSearchProvider()
        with patch.dict(
            keyless_mcp._KEYLESS_SEARCHERS,
            {"exa": lambda q, l: {"success": True, "data": {"web": []}}},
        ):
            out = provider.search("q", limit=3)
        assert out["success"] is True

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
    async def test_parallel_keyless_extract(self, monkeypatch):
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "parallel")
        provider = ParallelWebSearchProvider()
        with patch.dict(
            keyless_mcp._KEYLESS_EXTRACTORS,
            {"parallel": lambda urls: [{"url": "https://a", "title": "", "content": "c"}]},
        ):
            out = await provider.extract(["https://a"])
        assert out[0]["content"] == "c"


# ---------------------------------------------------------------------------
# Registry + _get_backend resolution order
# ---------------------------------------------------------------------------


class TestResolutionOrder:
    def test_registry_falls_back_to_keyless(self, fresh_registry, monkeypatch):
        monkeypatch.setattr(registry, "_read_config_key", lambda *p: None)
        provider = registry.get_active_search_provider()
        assert provider is not None
        # Ring: resolution picks the first REGISTERED vendor in ring order
        # (only exa/parallel are registered in this fixture).
        expected = next(
            v for v in registry._keyless_preference() if v in ("exa", "parallel")
        )
        assert provider.name == expected

    def test_keyless_ring_rotates_and_covers_all_vendors(self, fresh_registry, monkeypatch):
        monkeypatch.setattr(registry, "_read_config_key", lambda *p: None)
        # The ring order always contains all five vendors, starting at the
        # current cursor and wrapping.
        order = registry._keyless_preference()
        assert sorted(order) == sorted(keyless_mcp._KEYLESS_RING)
        # Unpinned dispatch rotates: consecutive _ring_order calls start at
        # successive vendors (round-robin cursor advances per request).
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda name: False)
        starts = [keyless_mcp._ring_order("exa")[0] for _ in range(len(keyless_mcp._KEYLESS_RING))]
        assert sorted(starts) == sorted(keyless_mcp._KEYLESS_RING)  # full cycle
        # Pinned dispatch starts at the pinned vendor every time.
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda name: name == "tavily")
        assert keyless_mcp._ring_order("tavily")[0] == "tavily"
        assert keyless_mcp._ring_order("tavily")[0] == "tavily"

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
        # No creds at all -> a keyless vendor per the process-stable split.
        monkeypatch.setattr(
            web_tools, "_registered_web_provider",
            lambda name: {"parallel": ParallelWebSearchProvider(),
                          "exa": ExaWebSearchProvider()}.get(name),
        )
        monkeypatch.setattr(web_tools, "_list_registered_web_providers", list)
        from agent.web_search_registry import _keyless_preference
        expected = next(
            v for v in _keyless_preference() if v in ("exa", "parallel")
        )
        assert web_tools._get_backend() == expected

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


# ---------------------------------------------------------------------------
# Cross-vendor keyless failover
# ---------------------------------------------------------------------------


class TestKeylessFailover:
    def _ok(self, vendor):
        return {"success": True, "data": {"web": [{"url": f"https://{vendor}.example"}]}}

    def _throttled(self, vendor):
        return {"success": False, "error": f"Keyless {vendor} search failed: free MCP rate limit."}

    def _pin(self, monkeypatch, name):
        """Pin *name* so the ring starts there deterministically."""
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == name)

    def test_search_fails_over_on_rate_limit(self, monkeypatch):
        self._pin(monkeypatch, "exa")
        monkeypatch.setitem(keyless_mcp._KEYLESS_SEARCHERS, "exa", lambda q, l: self._throttled("Exa"))
        monkeypatch.setitem(keyless_mcp._KEYLESS_SEARCHERS, "parallel", lambda q, l: self._ok("parallel"))
        out = keyless_mcp.search_with_failover("exa", "q", 3)
        assert out["success"] is True
        assert out["data"]["served_by"] == "parallel"

    def test_search_no_failover_on_non_throttle_error(self, monkeypatch):
        self._pin(monkeypatch, "exa")
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_SEARCHERS, "exa",
            lambda q, l: {"success": False, "error": "Unrecognized MCP response shape"},
        )
        called = []
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_SEARCHERS, "parallel",
            lambda q, l: called.append(1) or self._ok("parallel"),
        )
        out = keyless_mcp.search_with_failover("exa", "q")
        assert out["success"] is False
        assert not called  # peer never tried

    def test_search_all_throttled_reports_ring(self, monkeypatch):
        self._pin(monkeypatch, "exa")
        for vendor in keyless_mcp._KEYLESS_RING:
            monkeypatch.setitem(
                keyless_mcp._KEYLESS_SEARCHERS, vendor,
                lambda q, l, v=vendor: self._throttled(v),
            )
        out = keyless_mcp.search_with_failover("exa", "q")
        assert out["success"] is False
        assert "all keyless vendors throttled" in out["error"]

    def test_search_walks_ring_past_multiple_throttles(self, monkeypatch):
        # exa -> parallel -> tavily all throttled; firecrawl serves.
        self._pin(monkeypatch, "exa")
        for vendor in ("exa", "parallel", "tavily"):
            monkeypatch.setitem(
                keyless_mcp._KEYLESS_SEARCHERS, vendor,
                lambda q, l, v=vendor: self._throttled(v),
            )
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_SEARCHERS, "firecrawl",
            lambda q, l: self._ok("firecrawl"),
        )
        out = keyless_mcp.search_with_failover("exa", "q")
        assert out["success"] is True
        assert out["data"]["served_by"] == "firecrawl"

    def test_failover_respects_peer_paid_pin(self, monkeypatch):
        # Every vendor except exa throttles; exa is pinned paid so its free
        # endpoint must never be used.
        monkeypatch.setattr(
            keyless_mcp, "provider_tier",
            lambda name: "paid" if name == "exa" else "auto",
        )
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "parallel")
        called = []
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_SEARCHERS, "exa",
            lambda q, l: called.append(1) or self._ok("exa"),
        )
        for vendor in ("parallel", "tavily", "firecrawl", "keenable"):
            monkeypatch.setitem(
                keyless_mcp._KEYLESS_SEARCHERS, vendor,
                lambda q, l, v=vendor: self._throttled(v),
            )
        out = keyless_mcp.search_with_failover("parallel", "q")
        assert out["success"] is False
        assert not called  # exa pinned paid: its free tier is opted out

    def test_extract_fails_over_when_all_urls_throttled(self, monkeypatch):
        self._pin(monkeypatch, "exa")
        throttled = [
            {"url": "https://a", "title": "", "content": "", "error": "rate limit hit"},
            {"url": "https://b", "title": "", "content": "", "error": "429 too many requests"},
        ]
        good = [
            {"url": "https://a", "title": "A", "content": "x"},
            {"url": "https://b", "title": "B", "content": "y"},
        ]
        monkeypatch.setitem(keyless_mcp._KEYLESS_EXTRACTORS, "exa", lambda urls: throttled)
        monkeypatch.setitem(keyless_mcp._KEYLESS_EXTRACTORS, "parallel", lambda urls: good)
        out = keyless_mcp.extract_with_failover("exa", ["https://a", "https://b"])
        assert out == good

    def test_extract_partial_failure_stays_on_primary(self, monkeypatch):
        self._pin(monkeypatch, "exa")
        partial = [
            {"url": "https://a", "title": "A", "content": "x"},
            {"url": "https://b", "title": "", "content": "", "error": "rate limit"},
        ]
        called = []
        monkeypatch.setitem(keyless_mcp._KEYLESS_EXTRACTORS, "exa", lambda urls: partial)
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_EXTRACTORS, "parallel",
            lambda urls: called.append(1) or [],
        )
        out = keyless_mcp.extract_with_failover("exa", ["https://a", "https://b"])
        assert out == partial
        assert not called

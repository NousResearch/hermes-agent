"""One-shot keyless rescue: keyed/configured backend fails → THIS call rides
the keyless ring; the NEXT call attempts the chosen backend again.

Covers:
- eligibility: keyed ring vendors and non-ring backends are eligible;
  keyless-mode ring calls are not (they already walked the ring); config
  gates (keyless_rescue / keyless_fallback) turn it off
- search dispatcher: failure-result and raised-exception paths both rescue,
  result annotated with rescued_from + backend_error
- statelessness: the very next dispatch calls the chosen backend again
- extract dispatcher: whole-batch failure rescues; partial failure passes
  through untouched unless ``web.rescue_partial_failures`` is on, in which
  case only genuine failures of a mixed batch ride the ring (origin HTTP
  answers and policy blocks are never re-fetched) and rescued entries are
  served but never cached
- rescue failure: original backend error survives, rescue note appended
"""

import json
from unittest.mock import patch

import pytest

from tools import web_tools
from tools import web_tools_extract
from tools import web_tools_rescue
from plugins.web import keyless_mcp
from plugins.web.keenable.provider import KeenableWebSearchProvider

# Captured before the autouse fixture swaps in a stub: lets one test exercise the real
# config.yaml read path instead of the patched reader every other test uses.
_REAL_LOAD_WEB_CONFIG = web_tools._load_web_config


class _KeyedBoomProvider:
    """Minimal keyed provider double that always fails."""

    name = "keenable"
    display_name = "Keenable"

    def supports_search(self):
        return True

    def supports_extract(self):
        return True

    def is_available(self):
        return True

    def search(self, query, limit=5):
        return {"success": False, "error": "HTTP 500 upstream exploded"}

    def extract(self, urls, **kwargs):
        return [
            {"url": u, "title": "", "content": "", "error": "HTTP 500 upstream exploded"}
            for u in urls
        ]


class _RaisingProvider(_KeyedBoomProvider):
    def search(self, query, limit=5):
        raise RuntimeError("connection reset by peer")


class _GatewayFirecrawlBoomProvider(_KeyedBoomProvider):
    """Managed-gateway Firecrawl double with no direct provider key."""

    name = "firecrawl"
    display_name = "Firecrawl"


@pytest.fixture(autouse=True)
def _keyed_keenable_env(monkeypatch):
    """Simulate a keyed Keenable setup with rescue enabled."""
    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        lambda name: "kn-real" if name == "KEENABLE_API_KEY" else "",
    )
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "keenable"})
    monkeypatch.setattr(
        "agent.web_search_registry._keyless_tier_enabled", lambda: True
    )
    yield


def _ring_ok(vendor="exa"):
    return {"success": True, "data": {"web": [{"url": f"https://{vendor}.example"}]}}


async def _run_extract_tool(monkeypatch, provider, urls):
    """Drive ``web_extract_tool`` end-to-end with *provider* resolved by the registry."""
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        "agent.web_search_registry.get_provider", lambda name: provider
    )

    async def _allow_all(url, **kwargs):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
    raw = await web_tools.web_extract_tool(list(urls))
    data = json.loads(raw)
    return data["results"] if isinstance(data, dict) and "results" in data else data


class TestEligibility:
    def test_keyed_ring_vendor_is_eligible(self):
        assert web_tools_rescue._rescue_eligible(_KeyedBoomProvider()) is True

    def test_keyless_mode_ring_vendor_not_eligible(self, monkeypatch):
        # No key: the keenable call already rode the ring; no double-walk.
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env", lambda name: ""
        )
        assert web_tools_rescue._rescue_eligible(KeenableWebSearchProvider()) is False

    def test_gateway_selected_ring_vendor_is_eligible_without_direct_key(self, monkeypatch):
        # The persisted Nous route uses its subscriber token, not the keyless ring — eligible.
        # The same keyless Firecrawl selected directly DID walk the ring — not eligible.
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env", lambda name: ""
        )
        monkeypatch.setattr("plugins.web.firecrawl.provider._env", lambda name: "")
        monkeypatch.setattr("plugins.web.firecrawl.provider._is_tool_gateway_ready", lambda: True)
        monkeypatch.setattr(
            "tools.tool_backend_helpers.read_selection", lambda kind: "nous"
        )
        assert web_tools_rescue._rescue_eligible(_GatewayFirecrawlBoomProvider()) is True
        monkeypatch.setattr(
            "tools.tool_backend_helpers.read_selection", lambda kind: "firecrawl"
        )
        monkeypatch.setattr("plugins.web.keyless_mcp._web_config_selects", lambda name: name == "firecrawl")
        assert web_tools_rescue._rescue_eligible(_GatewayFirecrawlBoomProvider()) is False

    def test_non_ring_backend_is_eligible(self):
        class _SearxProvider(_KeyedBoomProvider):
            name = "searxng"

        assert web_tools_rescue._rescue_eligible(_SearxProvider()) is True

    def test_config_gate_disables(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"backend": "keenable", "keyless_rescue": False},
        )
        assert web_tools_rescue._rescue_eligible(_KeyedBoomProvider()) is False

    def test_keyless_fallback_off_disables(self, monkeypatch):
        monkeypatch.setattr(
            "agent.web_search_registry._keyless_tier_enabled", lambda: False
        )
        assert web_tools_rescue._rescue_eligible(_KeyedBoomProvider()) is False


class TestSearchRescue:
    def _dispatch(self, monkeypatch, provider):
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(
            "agent.web_search_registry.get_provider", lambda name: provider
        )
        return json.loads(web_tools.web_search_tool("q", limit=2))

    def test_failure_result_rescued_and_annotated(self, monkeypatch):
        with patch.object(
            keyless_mcp, "search_with_failover", return_value=_ring_ok()
        ) as ring:
            out = self._dispatch(monkeypatch, _KeyedBoomProvider())
        assert out["success"] is True
        assert out["data"]["rescued_from"] == "keenable"
        assert "HTTP 500" in out["data"]["backend_error"]
        assert "next call" in out["data"]["backend_error"].lower()
        ring.assert_called_once()

    def test_raised_exception_rescued(self, monkeypatch):
        with patch.object(
            keyless_mcp, "search_with_failover", return_value=_ring_ok()
        ):
            out = self._dispatch(monkeypatch, _RaisingProvider())
        assert out["success"] is True
        assert "connection reset" in out["data"]["backend_error"]

    def test_gateway_selected_firecrawl_failure_is_rescued(self, monkeypatch):
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "firecrawl"})
        monkeypatch.setattr(
            "agent.web_search_provider.get_provider_env", lambda name: ""
        )
        monkeypatch.setattr(
            "tools.tool_backend_helpers.read_selection", lambda kind: "nous"
        )
        with patch.object(
            keyless_mcp, "search_with_failover", return_value=_ring_ok()
        ) as ring:
            out = self._dispatch(monkeypatch, _GatewayFirecrawlBoomProvider())
        assert out["success"] is True
        assert out["data"]["rescued_from"] == "firecrawl"
        ring.assert_called_once()

    def test_stateless_next_call_uses_chosen_backend(self, monkeypatch):
        calls = {"backend": 0}

        class _Counting(_KeyedBoomProvider):
            def search(self, query, limit=5):
                calls["backend"] += 1
                return {"success": False, "error": "HTTP 500 upstream exploded"}

        provider = _Counting()
        with patch.object(
            keyless_mcp, "search_with_failover", return_value=_ring_ok()
        ) as ring:
            self._dispatch(monkeypatch, provider)
            self._dispatch(monkeypatch, provider)
        # The chosen backend was attempted on BOTH calls (no sticky failover),
        # and each failure triggered its own one-shot rescue.
        assert calls["backend"] == 2
        assert ring.call_count == 2

    def test_rescue_failure_keeps_original_error(self, monkeypatch):
        with patch.object(
            keyless_mcp, "search_with_failover",
            return_value={"success": False, "error": "all throttled"},
        ):
            out = self._dispatch(monkeypatch, _KeyedBoomProvider())
        assert out["success"] is False
        assert "HTTP 500 upstream exploded" in out["error"]
        assert "keyless rescue also failed" in out["error"]

    def test_no_rescue_when_disabled(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"backend": "keenable", "keyless_rescue": False},
        )
        with patch.object(keyless_mcp, "search_with_failover") as ring:
            out = self._dispatch(monkeypatch, _KeyedBoomProvider())
        assert out["success"] is False
        ring.assert_not_called()


class TestExtractRescue:
    async def _dispatch(self, monkeypatch, provider, urls):
        return await _run_extract_tool(monkeypatch, provider, urls)

    @pytest.mark.asyncio
    async def test_whole_batch_failure_rescued(self, monkeypatch):
        good = [
            {"url": "https://a", "title": "A", "content": "x" * 50,
             "raw_content": "x" * 50, "metadata": {"sourceURL": "https://a"}},
            {"url": "https://b", "title": "B", "content": "y" * 50,
             "raw_content": "y" * 50, "metadata": {"sourceURL": "https://b"}},
        ]
        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=good
        ) as ring:
            results = await self._dispatch(
                monkeypatch, _KeyedBoomProvider(), ["https://a", "https://b"]
            )
        assert all(not r.get("error") for r in results)
        assert results[0]["content"].startswith("x")
        ring.assert_called_once()

    def test_rescue_extract_annotates_results(self, monkeypatch):
        good = [
            {"url": "https://a", "title": "A", "content": "x",
             "raw_content": "x", "metadata": {"sourceURL": "https://a"}},
        ]
        failed = [{"url": "https://a", "title": "", "content": "", "error": "HTTP 500"}]
        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=good
        ):
            out = web_tools_rescue._rescue_extract("keenable", ["https://a"], failed)
        assert out[0]["metadata"]["rescued_from"] == "keenable"
        assert "HTTP 500" in out[0]["metadata"]["backend_error"]

    @pytest.mark.asyncio
    async def test_partial_failure_not_rescued(self, monkeypatch):
        class _Partial(_KeyedBoomProvider):
            def extract(self, urls, **kwargs):
                return [
                    {"url": urls[0], "title": "A", "content": "fine",
                     "raw_content": "fine", "metadata": {}},
                    {"url": urls[1], "title": "", "content": "", "error": "404"},
                ]

        with patch.object(keyless_mcp, "extract_with_failover") as ring:
            results = await self._dispatch(
                monkeypatch, _Partial(), ["https://a", "https://b"]
            )
        assert results[1].get("error")
        ring.assert_not_called()

    @pytest.mark.asyncio
    async def test_rescue_failure_keeps_original_errors(self, monkeypatch):
        still_bad = [
            {"url": "https://a", "title": "", "content": "", "error": "ring dead"},
            {"url": "https://b", "title": "", "content": "", "error": "ring dead"},
        ]
        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=still_bad
        ):
            results = await self._dispatch(
                monkeypatch, _KeyedBoomProvider(), ["https://a", "https://b"]
            )
        assert all("HTTP 500" in r.get("error", "") for r in results)


class TestExtractPartialRescue:
    """``web.rescue_partial_failures``: a mixed batch rescues only its genuinely failed URLs.

    Default off because a partially failed batch can double-fetch against a paid backend. When on,
    the whole-batch branch still gets exactly one rescue, policy blocks and origin HTTP answers are
    never re-fetched, and a rescued entry is served but never cached.
    """

    @staticmethod
    def _mixed_provider(error: str = "HTTP 500 upstream exploded"):
        class _Mixed(_KeyedBoomProvider):
            def extract(self, urls, **kwargs):
                return [
                    {"url": urls[0], "title": "A", "content": "a" * 40,
                     "raw_content": "a" * 40, "metadata": {}},
                    {"url": urls[1], "title": "", "content": "", "error": error},
                ]

        return _Mixed()

    @staticmethod
    def _ring_page(url: str = "https://b"):
        return [{"url": url, "title": "B", "content": "b" * 40, "raw_content": "b" * 40,
                 "metadata": {"sourceURL": url}}]

    @pytest.fixture
    def _partial_on(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"backend": "keenable", "rescue_partial_failures": True},
        )

    @pytest.mark.asyncio
    async def test_default_off_mixed_batch_passes_through(self):
        provider = self._mixed_provider()
        expected = provider.extract(["https://a", "https://b"])
        with patch.object(keyless_mcp, "extract_with_failover") as ring:
            results = await web_tools_extract._dispatch_extract(
                provider, ["https://a", "https://b"], None
            )
        ring.assert_not_called()
        assert results == expected

    @pytest.mark.asyncio
    async def test_enabled_rescues_only_the_failed_url_in_request_order(
        self, monkeypatch, _partial_on
    ):
        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=self._ring_page()
        ) as ring:
            results = await web_tools_extract._dispatch_extract(
                self._mixed_provider(), ["https://a", "https://b"], None
            )
        ring.assert_called_once_with("keenable", ["https://b"])
        assert [r.get("url") for r in results] == ["https://a", "https://b"]
        assert not results[0].get("error") and results[0]["content"] == "a" * 40
        assert not results[1].get("error") and results[1]["content"] == "b" * 40
        assert results[1]["metadata"]["rescued_from"] == "keenable"

    @pytest.mark.asyncio
    async def test_enabled_mixed_batch_is_served_end_to_end(self, monkeypatch, _partial_on):
        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=self._ring_page()
        ) as ring:
            results = await _run_extract_tool(
                monkeypatch, self._mixed_provider(), ["https://a", "https://b"]
            )
        ring.assert_called_once_with("keenable", ["https://b"])
        assert [r["url"] for r in results] == ["https://a", "https://b"]
        assert all(not r.get("error") for r in results)
        assert results[1]["content"] == "b" * 40

    @pytest.mark.asyncio
    async def test_all_failed_batch_gets_exactly_one_rescue(self, monkeypatch, _partial_on):
        pages = [self._ring_page("https://a")[0], self._ring_page("https://b")[0]]
        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=pages
        ) as ring:
            results = await web_tools_extract._dispatch_extract(
                _KeyedBoomProvider(), ["https://a", "https://b"], None
            )
        assert ring.call_count == 1  # whole-batch branch fires; partial mode adds nothing
        assert all(not r.get("error") for r in results)
        assert [r.get("url") for r in results] == ["https://a", "https://b"]

    @pytest.mark.asyncio
    async def test_policy_blocked_entry_is_preserved_and_never_rescued(
        self, monkeypatch, _partial_on
    ):
        blocked = {
            "url": "https://b", "title": "", "content": "",
            "error": "Blocked by website policy: example.com",
            "blocked_by_policy": {"host": "example.com", "rule": "blocklist", "source": "config"},
        }

        class _PolicyBlocked(_KeyedBoomProvider):
            def extract(self, urls, **kwargs):
                return [
                    {"url": urls[0], "title": "A", "content": "a" * 40,
                     "raw_content": "a" * 40, "metadata": {}},
                    dict(blocked),
                ]

        with patch.object(keyless_mcp, "extract_with_failover") as ring:
            results = await web_tools_extract._dispatch_extract(
                _PolicyBlocked(), ["https://a", "https://b"], None
            )
        ring.assert_not_called()
        assert results[1] == blocked

    @pytest.mark.asyncio
    async def test_origin_http_answer_is_not_rescued(self, monkeypatch, _partial_on):
        provider = self._mixed_provider("HTTP 404 from origin: Not Found")
        with patch.object(keyless_mcp, "extract_with_failover") as ring:
            results = await web_tools_extract._dispatch_extract(
                provider, ["https://a", "https://b"], None
            )
        ring.assert_not_called()
        assert results[1]["error"] == "HTTP 404 from origin: Not Found"

    @pytest.mark.asyncio
    async def test_backend_style_http_error_is_rescued(self, monkeypatch, _partial_on):
        # DeepSeek's per-URL text is the backend's own failure, not the origin's answer.
        provider = self._mixed_provider("DeepSeek fetch failed: HTTP 404")
        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=self._ring_page()
        ) as ring:
            results = await web_tools_extract._dispatch_extract(
                provider, ["https://a", "https://b"], None
            )
        ring.assert_called_once_with("keenable", ["https://b"])
        assert results[1]["metadata"]["rescued_from"] == "keenable"

    @pytest.mark.asyncio
    async def test_original_success_caches_but_rescued_entry_never_does(
        self, monkeypatch, _partial_on
    ):
        cached = []
        monkeypatch.setattr(
            "tools.web_result_cache.extract_cache_put",
            lambda url, content, title, **kwargs: cached.append(url) or True,
        )
        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=self._ring_page()
        ):
            results = await web_tools_extract._dispatch_extract(
                self._mixed_provider(), ["https://a", "https://b"], None
            )
        assert cached == ["https://a"]
        assert results[1]["metadata"]["rescued_from"] == "keenable"

    @pytest.mark.asyncio
    async def test_rescue_failure_keeps_original_results(self, monkeypatch, _partial_on):
        still_bad = [{"url": "https://b", "title": "", "content": "", "error": "ring dead"}]
        with patch.object(keyless_mcp, "extract_with_failover", return_value=still_bad):
            results = await web_tools_extract._dispatch_extract(
                self._mixed_provider(), ["https://a", "https://b"], None
            )
        assert results[0]["content"] == "a" * 40
        assert results[1]["error"] == "HTTP 500 upstream exploded"
        assert "rescued_from" not in (results[1].get("metadata") or {})

    @pytest.mark.asyncio
    async def test_explicit_off_is_byte_identical(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"backend": "keenable", "rescue_partial_failures": False},
        )
        provider = self._mixed_provider()
        expected = provider.extract(["https://a", "https://b"])
        with patch.object(keyless_mcp, "extract_with_failover") as ring:
            results = await web_tools_extract._dispatch_extract(
                provider, ["https://a", "https://b"], None
            )
        ring.assert_not_called()
        assert json.dumps(results, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)

    def test_setting_reads_through_the_config_loader(self, tmp_path, monkeypatch):
        """The knob reaches the gate from a real config.yaml, not only through the patched reader."""
        from hermes_cli.config import atomic_config_write

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        atomic_config_write(home / "config.yaml", {"web": {"rescue_partial_failures": True}})
        monkeypatch.setattr(web_tools, "_load_web_config", _REAL_LOAD_WEB_CONFIG)
        assert web_tools_rescue._rescue_partial_enabled() is True

    def test_parity_mismatch_whole_batch_keeps_ring_list_fallback(self):
        results = [{"url": "https://a", "title": "", "content": "", "error": "HTTP 500 abort"}]
        pages = [self._ring_page("https://a")[0], self._ring_page("https://b")[0]]
        with patch.object(
            keyless_mcp, "extract_with_failover", return_value=pages
        ) as ring:
            merged = web_tools_rescue._rescue_extract(
                "keenable", ["https://a", "https://b"], results
            )
        ring.assert_called_once_with("keenable", ["https://a", "https://b"])
        assert merged == pages  # existing whole-batch fallback: the ring list replaces the batch

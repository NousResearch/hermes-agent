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
  through untouched
- rescue failure: original backend error survives, rescue note appended
"""

import json
from unittest.mock import patch

import pytest

from tools import web_tools
from tools import web_tools_rescue
from plugins.web import keyless_mcp
from plugins.web.keenable.provider import KeenableWebSearchProvider


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


class _EmptyProvider(_KeyedBoomProvider):
    """Backend that reports success with zero hits (e.g. a scraping-based upstream silently blocked the query)."""

    def __init__(self):
        self.calls = 0

    def search(self, query, limit=5):
        self.calls += 1
        return {"success": True, "data": {"web": []}}


class TestEmptySearchRescue:
    _dispatch = TestSearchRescue._dispatch

    def test_empty_success_rescued_and_annotated(self, monkeypatch):
        with patch.object(keyless_mcp, "search_with_failover", return_value=_ring_ok()) as ring:
            out = self._dispatch(monkeypatch, _EmptyProvider())
        assert out["success"] is True
        assert out["data"]["web"] == _ring_ok()["data"]["web"]
        assert out["data"]["rescued_from"] == "keenable"
        assert "0 results" in out["data"]["backend_error"]
        ring.assert_called_once()

    def test_empty_rescue_not_cached_next_call_hits_backend(self, monkeypatch):
        provider = _EmptyProvider()
        with patch.object(keyless_mcp, "search_with_failover", return_value=_ring_ok()) as ring:
            self._dispatch(monkeypatch, provider)
            self._dispatch(monkeypatch, provider)
        assert provider.calls == 2
        assert ring.call_count == 2

    def _assert_empty_kept(self, out, again, provider):
        # An empty success stays a success (it may be genuine) but is marked, never an error.
        assert out["success"] is True and out["data"]["web"] == []
        assert "0 results from 'keenable'" in out["data"]["note"]
        assert "rescued_from" not in out["data"]
        assert again == out and provider.calls == 2  # empty results are never cached

    def test_ring_also_empty_keeps_empty_success_with_note(self, monkeypatch):
        provider = _EmptyProvider()
        ring_resp = {"success": True, "data": {"web": []}}
        with patch.object(keyless_mcp, "search_with_failover", return_value=ring_resp) as ring:
            out = self._dispatch(monkeypatch, provider)
            again = self._dispatch(monkeypatch, provider)
        self._assert_empty_kept(out, again, provider)
        assert "found nothing either" in out["data"]["note"]
        assert ring.call_count == 2

    def test_ring_failing_keeps_empty_success_without_claiming_nothing_found(self, monkeypatch):
        provider = _EmptyProvider()
        ring_resp = {"success": False, "error": "all throttled"}
        with patch.object(keyless_mcp, "search_with_failover", return_value=ring_resp) as ring:
            out = self._dispatch(monkeypatch, provider)
            again = self._dispatch(monkeypatch, provider)
        self._assert_empty_kept(out, again, provider)
        assert "rescue was attempted but failed" in out["data"]["note"]
        assert "found nothing either" not in out["data"]["note"]
        assert ring.call_count == 2

    def test_ring_raising_keeps_empty_success_not_cached_and_retries_primary(self, monkeypatch):
        """A rescue that RAISES (e.g. a ring parser choking on a malformed payload) is best-effort:
        the original empty success is preserved, not turned into a tool error, and never cached."""
        provider = _EmptyProvider()
        boom = AttributeError("'NoneType' object has no attribute 'get'")
        with patch.object(keyless_mcp, "search_with_failover", side_effect=boom) as ring:
            out = self._dispatch(monkeypatch, provider)
            again = self._dispatch(monkeypatch, provider)
        self._assert_empty_kept(out, again, provider)
        assert "error" not in out
        assert "rescue was attempted but failed" in out["data"]["note"]
        assert "found nothing either" not in out["data"]["note"]
        assert ring.call_count == 2  # primary retried, rescue retried: nothing sticky

    def test_ring_parser_crash_on_malformed_payload_keeps_empty_success(self, monkeypatch):
        """Real ring + parser path: only ring ordering and transport are mocked. A Parallel payload of
        ``{"results":[null]}`` makes the parser raise; the empty success must survive."""
        provider = _EmptyProvider()
        with patch.object(keyless_mcp, "_ring_order", return_value=["parallel"]), \
                patch.object(keyless_mcp, "mcp_call", return_value='{"results":[null]}'):
            out = self._dispatch(monkeypatch, provider)
        assert out["success"] is True and out["data"]["web"] == []
        assert "0 results from 'keenable'" in out["data"]["note"]

    def test_rescue_does_not_swallow_keyboard_interrupt(self, monkeypatch):
        provider = _EmptyProvider()
        with patch.object(keyless_mcp, "search_with_failover", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                web_tools._memoized_search(provider, "interrupt-me", 5)

    def test_ineligible_empty_gets_note_without_ring_call(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_load_web_config", lambda: {"backend": "keenable", "keyless_rescue": False},
        )
        provider = _EmptyProvider()
        with patch.object(keyless_mcp, "search_with_failover") as ring:
            out = self._dispatch(monkeypatch, provider)
            self._dispatch(monkeypatch, provider)
        ring.assert_not_called()
        assert out["success"] is True and out["data"]["web"] == []
        assert "0 results" in out["data"]["note"] and "either" not in out["data"]["note"]
        assert provider.calls == 2

    def test_nonempty_success_still_cached_without_ring(self, monkeypatch):
        class _Hit(_EmptyProvider):
            def search(self, query, limit=5):
                self.calls += 1
                return {"success": True, "data": {"web": [{"url": "https://hit.example"}]}}

        provider = _Hit()
        with patch.object(keyless_mcp, "search_with_failover") as ring:
            out = self._dispatch(monkeypatch, provider)
            self._dispatch(monkeypatch, provider)
        ring.assert_not_called()
        assert "note" not in out["data"]
        assert provider.calls == 1


class TestExtractRescue:
    async def _dispatch(self, monkeypatch, provider, urls):
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

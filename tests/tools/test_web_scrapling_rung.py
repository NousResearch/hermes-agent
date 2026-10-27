"""Scrapling as the 5th keyless ring rung (local, no API key, no rate limit).

The rung wraps the Scrapling sidecar (``~/.hermes/venvs/scrapling``) through a
subprocess adapter — the gateway runtime is a different Python than the
sidecar, so the import boundary must stay a process boundary.

Covers:
- ring membership: Scrapling is a peer of exa/parallel/firecrawl/keenable
- dual contract: the rung serves BOTH search and extract
- local search source: DDG Lite (robots.txt ``Allow: /``), title/description/url
  mapping including the ``uddg=`` redirect wrapper
- extract: title + readable text, thin-body JS escalation
- failover: a local rung that fails must not break the chain, and must not be
  treated as a vendor refusal that silently swallows the query
"""

import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from agent import web_search_registry as registry
from plugins.web import keyless_mcp
from plugins.web.scrapling import provider as scrapling_provider


@pytest.fixture(autouse=True)
def _no_web_env(monkeypatch):
    """Blank every web credential and neutralize config lookups."""
    for var in (
        "EXA_API_KEY", "PARALLEL_API_KEY", "KEENABLE_API_KEY", "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "BRAVE_SEARCH_API_KEY",
        "SEARXNG_URL", "TOOL_GATEWAY_USER_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env", lambda name: "", raising=True
    )
    monkeypatch.setattr("tools.web_tools._env_value", lambda name: "", raising=True)
    monkeypatch.setattr("tools.web_tools._load_web_config", dict, raising=True)
    monkeypatch.setattr("tools.web_tools._is_tool_gateway_ready", lambda: False, raising=True)
    yield


# --- ring membership ------------------------------------------------------

class TestRingMembership:
    def test_scrapling_is_a_ring_member(self):
        assert "scrapling" in keyless_mcp._KEYLESS_RING

    def test_ring_has_five_members(self):
        assert len(keyless_mcp._KEYLESS_RING) == 5

    def test_scrapling_is_first_in_rotation_order(self):
        """The local rung is listed first so a fresh install can serve locally."""
        assert keyless_mcp._KEYLESS_RING[0] == "scrapling"

    def test_cloud_rungs_keep_their_order(self):
        """Appending a rung must not reshuffle the existing vendor sequence."""
        assert keyless_mcp._KEYLESS_RING[1:] == ("exa", "parallel", "firecrawl", "keenable")

    def test_scrapling_has_both_adapters(self):
        """A ring entry needs a searcher AND an extractor to satisfy both contracts."""
        assert "scrapling" in keyless_mcp._KEYLESS_SEARCHERS
        assert "scrapling" in keyless_mcp._KEYLESS_EXTRACTORS

    def test_registry_preference_includes_scrapling(self):
        assert "scrapling" in registry._KEYLESS_PREFERENCE

    def test_registry_preference_matches_ring_set(self):
        assert set(registry._KEYLESS_PREFERENCE) == set(keyless_mcp._KEYLESS_RING)

    def test_tavily_is_still_not_a_ring_member(self):
        assert "tavily" not in keyless_mcp._KEYLESS_RING
        assert "tavily" not in keyless_mcp._KEYLESS_SEARCHERS
        assert "tavily" not in keyless_mcp._KEYLESS_EXTRACTORS
        assert "tavily" not in registry._KEYLESS_PREFERENCE


# --- the local search source ---------------------------------------------

class TestLocalSearchParsing:
    """DDG Lite result rows -> Hermes ``web_hit`` shape."""

    HTML = (
        "<html><body><table>"
        "<tr><td valign='top'>1.&nbsp;</td><td>"
        "<a rel='nofollow' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=abc'"
        " class='result-link'>Example Title</a></td></tr>"
        "<tr><td>&nbsp;&nbsp;&nbsp;</td><td class='result-snippet'>An example snippet</td></tr>"
        "<tr><td>&nbsp;&nbsp;&nbsp;</td><td><span class='link-text'>example.com</span></td></tr>"
        "<tr><td valign='top'>2.&nbsp;</td><td>"
        "<a rel='nofollow' href='https://direct.example.org/page' class='result-link'>Second &amp; Title</a></td></tr>"
        "<tr><td>&nbsp;&nbsp;&nbsp;</td><td class='result-snippet'>Second snippet</td></tr>"
        "</table></body></html>"
    )

    def test_parses_title_url_and_description(self):
        rows = scrapling_provider.parse_ddg_lite_results(self.HTML)
        assert len(rows) == 2
        assert rows[0]["title"] == "Example Title"
        assert rows[0]["url"] == "https://example.com/a?x=1"
        assert rows[0]["description"] == "An example snippet"
        assert rows[0]["position"] == 1

    def test_decodes_uddg_redirect_wrapper(self):
        rows = scrapling_provider.parse_ddg_lite_results(self.HTML)
        assert rows[0]["url"] == "https://example.com/a?x=1"

    def test_keeps_direct_href_when_not_wrapped(self):
        rows = scrapling_provider.parse_ddg_lite_results(self.HTML)
        assert rows[1]["url"] == "https://direct.example.org/page"

    def test_unescapes_html_entities_in_title(self):
        rows = scrapling_provider.parse_ddg_lite_results(self.HTML)
        assert rows[1]["title"] == "Second & Title"

    def test_positions_are_one_based(self):
        rows = scrapling_provider.parse_ddg_lite_results(self.HTML)
        assert [r["position"] for r in rows] == [1, 2]

    def test_empty_html_yields_no_rows(self):
        assert scrapling_provider.parse_ddg_lite_results("<html><body></body></html>") == []

    def test_handles_cjk_titles(self):
        html = (
            "<a rel='nofollow' href='https://hk.finance.yahoo.com/quote/0388.HK/' "
            "class='result-link'>香港交易所 (0388.HK) 股價</a>"
            "<td class='result-snippet'>香港交易所 總覽</td>"
        )
        rows = scrapling_provider.parse_ddg_lite_results(html)
        assert rows[0]["title"] == "香港交易所 (0388.HK) 股價"
        assert rows[0]["description"] == "香港交易所 總覽"

    def test_skips_result_without_description(self):
        html = "<a rel='nofollow' href='https://x.example/' class='result-link'>Only title</a>"
        assert scrapling_provider.parse_ddg_lite_results(html) == []


# --- the sidecar adapter -------------------------------------------------

class TestSidecarAdapter:
    def test_extract_uses_document_shape(self):
        with patch.object(scrapling_provider, "_run_sidecar") as run:
            run.return_value = {
                "ok": True,
                "results": [{"url": "https://example.com/", "title": "Example", "text": "body text"}],
            }
            out = scrapling_provider.scrapling_extract_keyless(["https://example.com/"])
        assert out[0]["url"] == "https://example.com/"
        assert out[0]["title"] == "Example"
        assert out[0]["content"] == "body text"
        assert out[0]["raw_content"] == "body text"
        assert out[0]["metadata"]["sourceURL"] == "https://example.com/"

    def test_extract_error_entry_has_error_key(self):
        with patch.object(scrapling_provider, "_run_sidecar") as run:
            run.return_value = {
                "ok": True,
                "results": [{"url": "https://example.com/", "error": "HTTP 500: boom"}],
            }
            out = scrapling_provider.scrapling_extract_keyless(["https://example.com/"])
        assert out[0]["error"] == "HTTP 500: boom"

    def test_search_uses_web_hit_shape(self):
        with patch.object(scrapling_provider, "_run_sidecar") as run:
            run.return_value = {
                "ok": True,
                "results": [
                    {"url": "https://a.example/", "title": "A", "description": "da", "position": 1},
                ],
            }
            out = scrapling_provider.scrapling_search_keyless("q")
        assert out["success"] is True
        assert out["data"]["web"][0] == {
            "url": "https://a.example/", "title": "A", "description": "da", "position": 1,
        }

    def test_search_failure_is_soft(self):
        """A local rung failure must be shaped so the ring ADVANCES to the next
        vendor (a structured 5xx), not stop the walk with a hard error."""
        with patch.object(scrapling_provider, "_run_sidecar") as run:
            run.return_value = {"ok": False, "error": "sidecar missing"}
            out = scrapling_provider.scrapling_search_keyless("q")
        assert out["success"] is False
        assert "sidecar missing" in out["error"]

    def test_sidecar_timeout_is_reported(self):
        with patch.object(scrapling_provider, "_run_sidecar") as run:
            run.return_value = {"ok": False, "error": "sidecar timed out after 60s"}
            out = scrapling_provider.scrapling_extract_keyless(["https://example.com/"])
        assert out[0]["error"] == (
            "Keyless Scrapling extract failed: HTTP 503: local rung unavailable: "
            "sidecar timed out after 60s. Use another web backend via `hermes tools`."
        )

    def test_empty_search_result_hands_over(self):
        """An empty parse means the endpoint refused us — not a real 'no
        results' answer — so the rung must fail over instead."""
        with patch.object(scrapling_provider, "_run_sidecar") as run:
            run.return_value = {"ok": True, "results": []}
            out = scrapling_provider.scrapling_search_keyless("q")
        assert out["success"] is False
        assert "no results parsed" in out["error"]

    def test_worker_is_a_separate_interpreter(self):
        """The gateway runtime is not the sidecar runtime: the adapter must call
        the sidecar venv's python, never import scrapling in-process."""
        assert scrapling_provider.sidecar_python() != sys.executable
        assert "scrapling" not in sys.modules

    def test_sidecar_path_follows_hermes_home(self, monkeypatch, tmp_path):
        """The sidecar path must resolve against the ACTIVE Hermes home, so a
        test run against an isolated home never touches the real one."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv(scrapling_provider.SIDECAR_VENV_ENV, raising=False)
        assert scrapling_provider.sidecar_venv() == str(tmp_path / "venvs" / "scrapling")
        assert scrapling_provider.sidecar_python() == str(
            tmp_path / "venvs" / "scrapling" / "bin" / "python"
        )

    def test_sidecar_venv_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(scrapling_provider.SIDECAR_VENV_ENV, str(tmp_path / "custom"))
        assert scrapling_provider.sidecar_venv() == str(tmp_path / "custom")

    def test_install_hint_uses_uv_not_stdlib_venv(self):
        """``python3 -m venv`` needs the ``python3-venv`` system package, which
        is absent on Debian/Ubuntu hosts without it — the documented command
        must use ``uv`` (which Hermes already requires) with ``--seed`` so the
        venv has pip."""
        hint = scrapling_provider.SIDECAR_INSTALL_HINT
        assert "uv venv --seed" in hint
        assert "python3 -m venv" not in hint
        assert "scrapling[fetchers]" in hint

    def test_missing_sidecar_error_names_the_fix(self):
        """The error a user sees must say how to install the sidecar, not just
        where it looked."""
        with patch.object(scrapling_provider, "sidecar_available", return_value=False):
            envelope = scrapling_provider._run_sidecar({"op": "search", "query": "q"})
        assert envelope["ok"] is False
        assert "install it with" in envelope["error"]
        assert "uv venv --seed" in envelope["error"]


# --- failover semantics ---------------------------------------------------

class TestFailover:
    def test_local_failure_fails_over_to_next_vendor(self, monkeypatch):
        """The local rung failing must advance to the next cloud vendor."""
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "scrapling")
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_SEARCHERS, "scrapling",
            lambda q, l: {"success": False, "error": "Keyless Scrapling search failed: HTTP 503: local rung unavailable"},
        )
        monkeypatch.setitem(
            keyless_mcp._KEYLESS_SEARCHERS, "exa",
            lambda q, l: {"success": True, "data": {"web": []}},
        )
        out = keyless_mcp.search_with_failover("scrapling", "q")
        assert out["success"] is True
        assert out["data"]["served_by"] == "exa"

    def test_local_outage_is_failover_eligible(self):
        """The ring's refusal regex must recognise the local rung's 5xx, or the
        walk stops and the whole query fails."""
        from plugins.web.keyless_mcp import _is_search_failover_eligible

        assert _is_search_failover_eligible(
            "Keyless Scrapling search failed: HTTP 503: local rung unavailable: sidecar missing"
        )

    def test_real_local_outage_fails_over(self, monkeypatch):
        """End-to-end through the real provider error shape: a missing sidecar
        must hand the query to the next vendor, not fail it."""
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "scrapling")
        with patch.object(scrapling_provider, "_run_sidecar") as run:
            run.return_value = {"ok": False, "error": "Scrapling sidecar not installed"}
            monkeypatch.setitem(
                keyless_mcp._KEYLESS_SEARCHERS, "exa",
                lambda q, l: {"success": True, "data": {"web": []}},
            )
            out = keyless_mcp.search_with_failover("scrapling", "q")
        assert out["success"] is True
        assert out["data"]["served_by"] == "exa"

    def test_exhausted_ring_reports_all_vendors(self, monkeypatch):
        monkeypatch.setattr(keyless_mcp, "_vendor_pinned", lambda n: n == "scrapling")
        for vendor in keyless_mcp._KEYLESS_RING:
            monkeypatch.setitem(
                keyless_mcp._KEYLESS_SEARCHERS, vendor,
                lambda q, l, v=vendor: {"success": False, "error": f"{v} rate limit"},
            )
        out = keyless_mcp.search_with_failover("scrapling", "q")
        assert out["success"] is False
        assert "all keyless vendors unavailable" in out["error"]
        assert "scrapling" in out["error"]

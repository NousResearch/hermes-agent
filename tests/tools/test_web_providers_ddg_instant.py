"""Standalone unittest for the DuckDuckGo Instant Answer API web search provider.

Uses only the stdlib (``unittest``, ``urllib``) so it runs with plain
``python3 -m unittest tests/tools/test_web_providers_ddg_instant.py`` —
no pytest, no httpx, no hermes-agent test runner required.

The full pytest suite in this directory exercises the plugin against the
real Hermes web-search registry via tests/tools/conftest.py; this
unittest is the lightweight, dep-free sanity check that ships with the
plugin so a fresh checkout can validate behavior before the test runner
is wired up.
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
from pathlib import Path

# Make ``plugins/`` importable when this file is run as a standalone script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from plugins.web.ddg_instant.provider import (  # noqa: E402
    DDGInstantWebSearchProvider,
    _coerce_topic,
    _flatten_topics,
    _http_get_json,
)


# ---------------------------------------------------------------------------
# Identity / availability
# ---------------------------------------------------------------------------


class TestDDGInstantIdentity(unittest.TestCase):
    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        self.assertTrue(issubclass(DDGInstantWebSearchProvider, WebSearchProvider))

    def test_name_and_display_name(self):
        self.assertEqual(DDGInstantWebSearchProvider.NAME, "ddg-instant")
        self.assertIn("DuckDuckGo", DDGInstantWebSearchProvider.DISPLAY_NAME)

    def test_is_available_is_true_no_deps(self):
        """No API key, no extra package — always available."""
        self.assertTrue(DDGInstantWebSearchProvider().is_available())


# ---------------------------------------------------------------------------
# search() — shape + error paths
# ---------------------------------------------------------------------------


class TestDDGInstantSearch(unittest.TestCase):
    def _abstract_payload(self):
        return {
            "Abstract": "Linux is a family of free open-source OSes.",
            "AbstractText": "Linux is a family of free open-source operating systems based on the Linux kernel.",
            "AbstractURL": "https://en.wikipedia.org/wiki/Linux",
            "AbstractSource": "Wikipedia",
            "RelatedTopics": [
                {"Text": "Criticism of Linux — some critical viewpoints", "FirstURL": "https://duckduckgo.com/Criticism_of_Linux"},
                {"Text": "Linux distribution — variants of Linux", "FirstURL": "https://duckduckgo.com/Linux_distribution"},
            ],
            "Results": [],
        }

    def test_happy_path_abstract_plus_related_topics(self):
        provider = DDGInstantWebSearchProvider()
        original = _http_get_json

        def fake_http(url, params, timeout):
            return self._abstract_payload(), None

        try:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = fake_http
            out = provider.search("linux", limit=5)
        finally:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = original

        self.assertTrue(out["success"])
        rows = out["data"]["web"]
        self.assertEqual(len(rows), 3)  # 1 abstract + 2 related topics
        # First hit is the abstract
        self.assertEqual(rows[0]["title"], "Wikipedia")
        self.assertEqual(rows[0]["url"], "https://en.wikipedia.org/wiki/Linux")
        self.assertIn("free open-source", rows[0]["description"])
        self.assertEqual(rows[0]["position"], 1)
        # Subsequent hits are related topics with split titles
        self.assertEqual(rows[1]["title"], "Criticism of Linux")
        self.assertEqual(rows[1]["url"], "https://duckduckgo.com/Criticism_of_Linux")
        self.assertEqual(rows[1]["position"], 2)

    def test_limit_truncates_results(self):
        provider = DDGInstantWebSearchProvider()
        payload = self._abstract_payload()
        payload["RelatedTopics"] = [
            {"Text": f"Topic {i} — body {i}", "FirstURL": f"https://duckduckgo.com/Topic_{i}"}
            for i in range(10)
        ]

        def fake_http(url, params, timeout):
            return payload, None

        original = _http_get_json
        try:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = fake_http
            out = provider.search("linux", limit=2)
        finally:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = original

        self.assertTrue(out["success"])
        self.assertEqual(len(out["data"]["web"]), 2)

    def test_empty_query_fails_gracefully(self):
        out = DDGInstantWebSearchProvider().search("", limit=5)
        self.assertFalse(out["success"])
        self.assertIn("Empty query", out["error"])

    def test_network_error_fails_gracefully(self):
        def fake_http(url, params, timeout):
            return None, "DuckDuckGo Instant Answer network error: DNS lookup failed"

        original = _http_get_json
        try:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = fake_http
            out = DDGInstantWebSearchProvider().search("python", limit=5)
        finally:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = original

        self.assertFalse(out["success"])
        self.assertIn("network error", out["error"])

    def test_http_error_fails_gracefully(self):
        def fake_http(url, params, timeout):
            return None, "DuckDuckGo Instant Answer HTTP error: HTTP 503"

        original = _http_get_json
        try:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = fake_http
            out = DDGInstantWebSearchProvider().search("python", limit=5)
        finally:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = original

        self.assertFalse(out["success"])
        self.assertIn("HTTP error", out["error"])

    def test_no_abstract_no_related_topics_returns_instructive_failure(self):
        """Sparse response (how-to/tutorial queries) must NOT silently return
        []. Must fail with an error pointing the caller at a full-text backend."""
        def fake_http(url, params, timeout):
            return {"Abstract": "", "AbstractText": "", "AbstractURL": "",
                    "RelatedTopics": [], "Results": []}, None

        original = _http_get_json
        try:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = fake_http
            out = DDGInstantWebSearchProvider().search("python async tutorial", limit=5)
        finally:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = original

        self.assertFalse(out["success"])
        self.assertIn("no results", out["error"].lower())
        self.assertTrue("ddgs" in out["error"] or "full-text" in out["error"].lower())

    def test_skip_top_level_categories(self):
        provider = DDGInstantWebSearchProvider()
        payload = {
            "Abstract": "", "AbstractText": "", "AbstractURL": "",
            "RelatedTopics": [
                {"Name": "Linux", "TopLevelLink": True},  # nav, skip
                {"Text": "Linux kernel — core component", "FirstURL": "https://duckduckgo.com/Linux_kernel"},
            ],
            "Results": [],
        }

        def fake_http(url, params, timeout):
            return payload, None

        original = _http_get_json
        try:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = fake_http
            out = provider.search("linux", limit=5)
        finally:
            import plugins.web.ddg_instant.provider as prov_mod
            prov_mod._http_get_json = original

        self.assertTrue(out["success"])
        self.assertEqual(len(out["data"]["web"]), 1)
        self.assertEqual(out["data"]["web"][0]["title"], "Linux kernel")

    def test_flatten_nested_topics(self):
        raw = [
            {"Name": "Linux", "Topics": [
                {"Text": "Linux kernel — core", "FirstURL": "https://duckduckgo.com/Linux_kernel"},
                {"Text": "Linus Torvald — creator", "FirstURL": "https://duckduckgo.com/Linus_Torvald"},
            ]},
            {"Text": "Direct entry — standalone", "FirstURL": "https://duckduckgo.com/Direct"},
        ]
        out = _flatten_topics(raw)
        titles = [h["title"] for h in out]
        self.assertEqual(titles, ["Linux kernel", "Linus Torvald", "Direct entry"])


# ---------------------------------------------------------------------------
# Live network smoke test (gated by env var so unit-test runs stay offline)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    sys.platform == "linux" and Path("/opt/data/work/wt-ddg-websearch").exists(),
    "live DDG network test only runs in the wt-ddg-websearch worktree",
)
class TestDDGInstantLiveNetwork(unittest.TestCase):
    def test_real_ddg_instant_query_returns_results(self):
        """Hit the real DDG Instant Answer API for 'linux' and confirm shape."""
        out = DDGInstantWebSearchProvider().search("linux", limit=3)
        self.assertTrue(out["success"], msg=f"DDG live query failed: {out.get('error')}")
        self.assertTrue(out["data"]["web"])
        for r in out["data"]["web"]:
            self.assertIn("url", r)
            self.assertIn("title", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
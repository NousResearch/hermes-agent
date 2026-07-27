#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from tools.skills_hub import ClawHubSource, SkillMeta


class _MockResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json_data


class TestClawHubSource(unittest.TestCase):
    def setUp(self):
        self.src = ClawHubSource()
        self._safe_patcher = patch("tools.skills_hub.is_safe_url", return_value=True)
        self._policy_patcher = patch("tools.skills_hub.check_website_access", return_value=None)
        self._safe_patcher.start()
        self._policy_patcher.start()

    def tearDown(self):
        self._policy_patcher.stop()
        self._safe_patcher.stop()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch.object(ClawHubSource, "_load_catalog_index", return_value=[])
    @patch("tools.skills_hub.httpx.get")
    def test_search_uses_listing_endpoint_as_fallback(
        self, mock_get, _mock_load_catalog, _mock_read_cache, _mock_write_cache
    ):
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "items": [
                            {
                                "slug": "caldav-calendar",
                                "displayName": "CalDAV Calendar",
                                "summary": "Calendar integration",
                                "tags": ["calendar", "productivity"],
                            }
                        ]
                    },
                )
            if url.endswith("/skills/caldav"):
                return _MockResponse(status_code=404, json_data={})
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        results = self.src.search("caldav", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "caldav-calendar")
        self.assertEqual(results[0].name, "CalDAV Calendar")
        self.assertEqual(results[0].description, "Calendar integration")

        self.assertGreaterEqual(mock_get.call_count, 2)
        args, kwargs = mock_get.call_args_list[0]
        self.assertTrue(args[0].endswith("/skills"))
        self.assertEqual(kwargs["params"], {"search": "caldav", "limit": 5})

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch.object(
        ClawHubSource,
        "_load_catalog_index",
        return_value=[],
    )
    @patch("tools.skills_hub.httpx.get")
    def test_search_falls_back_to_exact_slug_when_search_results_are_irrelevant(
        self, mock_get, _mock_load_catalog, _mock_read_cache, _mock_write_cache
    ):
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "items": [
                            {
                                "slug": "apple-music-dj",
                                "displayName": "Apple Music DJ",
                                "summary": "Unrelated result",
                            }
                        ]
                    },
                )
            if url.endswith("/skills/self-improving-agent"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "skill": {
                            "slug": "self-improving-agent",
                            "displayName": "self-improving-agent",
                            "summary": "Captures learnings and errors for continuous improvement.",
                            "tags": {"latest": "3.0.2", "automation": "3.0.2"},
                        },
                        "latestVersion": {"version": "3.0.2"},
                    },
                )
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        results = self.src.search("self-improving-agent", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "self-improving-agent")
        self.assertEqual(results[0].name, "self-improving-agent")
        self.assertIn("continuous improvement", results[0].description)

    @patch("tools.skills_hub.httpx.get")
    def test_search_repairs_poisoned_cache_with_exact_slug_lookup(self, mock_get):
        mock_get.return_value = _MockResponse(
            status_code=200,
            json_data={
                "skill": {
                    "slug": "self-improving-agent",
                    "displayName": "self-improving-agent",
                    "summary": "Captures learnings and errors for continuous improvement.",
                    "tags": {"latest": "3.0.2", "automation": "3.0.2"},
                },
                "latestVersion": {"version": "3.0.2"},
            },
        )

        poisoned = [
            SkillMeta(
                name="Apple Music DJ",
                description="Unrelated cached result",
                source="clawhub",
                identifier="apple-music-dj",
                trust_level="community",
                tags=[],
            )
        ]
        results = self.src._finalize_search_results("self-improving-agent", poisoned, 5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "self-improving-agent")
        mock_get.assert_called_once()
        self.assertTrue(mock_get.call_args.args[0].endswith("/skills/self-improving-agent"))

    @patch.object(
        ClawHubSource,
        "_exact_slug_meta",
        return_value=SkillMeta(
            name="self-improving-agent",
            description="Captures learnings and errors for continuous improvement.",
            source="clawhub",
            identifier="self-improving-agent",
            trust_level="community",
            tags=["automation"],
        ),
    )
    def test_search_matches_space_separated_query_to_hyphenated_slug(
        self, _mock_exact_slug
    ):
        results = self.src.search("self improving", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "self-improving-agent")

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_maps_display_name_and_summary(self, mock_get):
        mock_get.return_value = _MockResponse(
            status_code=200,
            json_data={
                "slug": "caldav-calendar",
                "displayName": "CalDAV Calendar",
                "summary": "Calendar integration",
                "tags": ["calendar"],
            },
        )

        meta = self.src.inspect("caldav-calendar")

        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "CalDAV Calendar")
        self.assertEqual(meta.description, "Calendar integration")
        self.assertEqual(meta.identifier, "caldav-calendar")

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_handles_nested_skill_payload(self, mock_get):
        mock_get.return_value = _MockResponse(
            status_code=200,
            json_data={
                "skill": {
                    "slug": "self-improving-agent",
                    "displayName": "self-improving-agent",
                    "summary": "Captures learnings and errors for continuous improvement.",
                    "tags": {"latest": "3.0.2", "automation": "3.0.2"},
                },
                "latestVersion": {"version": "3.0.2"},
            },
        )

        meta = self.src.inspect("self-improving-agent")

        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "self-improving-agent")
        self.assertIn("continuous improvement", meta.description)
        self.assertEqual(meta.identifier, "self-improving-agent")
        self.assertEqual(meta.tags, ["automation"])

    @patch("tools.skills_hub._ssrf_safe_http_get")
    @patch("tools.skills_hub.httpx.get")
    def test_fetch_resolves_latest_version_and_downloads_raw_files(self, mock_get, mock_safe_get):
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills/caldav-calendar"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "slug": "caldav-calendar",
                        "latestVersion": {"version": "1.0.1"},
                    },
                )
            if url.endswith("/skills/caldav-calendar/versions/1.0.1"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "files": [
                            {"path": "SKILL.md", "rawUrl": "https://files.example/skill-md"},
                            {"path": "README.md", "content": "hello"},
                        ]
                    },
                )
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect
        mock_safe_get.return_value = _MockResponse(status_code=200, text="# Skill")

        bundle = self.src.fetch("caldav-calendar")

        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.name, "caldav-calendar")
        self.assertIn("SKILL.md", bundle.files)
        self.assertEqual(bundle.files["SKILL.md"], "# Skill")
        self.assertEqual(bundle.files["README.md"], "hello")
        mock_safe_get.assert_called_once_with("https://files.example/skill-md", timeout=20)

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_falls_back_to_versions_list(self, mock_get):
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills/caldav-calendar"):
                return _MockResponse(status_code=200, json_data={"slug": "caldav-calendar"})
            if url.endswith("/skills/caldav-calendar/versions"):
                return _MockResponse(status_code=200, json_data=[{"version": "2.0.0"}])
            if url.endswith("/skills/caldav-calendar/versions/2.0.0"):
                return _MockResponse(status_code=200, json_data={"files": {"SKILL.md": "# Skill"}})
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        bundle = self.src.fetch("caldav-calendar")
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.files["SKILL.md"], "# Skill")

    @patch("tools.skills_hub.check_website_access", return_value=None)
    @patch("tools.skills_hub.is_safe_url")
    @patch("tools.skills_hub.httpx.get")
    @patch("tools.skills_hub._ssrf_safe_http_get")
    def test_fetch_blocks_private_raw_url(self, mock_safe_get, mock_get, mock_safe, _mock_policy):
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills/caldav-calendar"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "slug": "caldav-calendar",
                        "latestVersion": {"version": "1.0.1"},
                    },
                )
            if url.endswith("/download"):
                return _MockResponse(status_code=404)
            if url.endswith("/skills/caldav-calendar/versions/1.0.1"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "files": [
                            {"path": "SKILL.md", "rawUrl": "http://127.0.0.1/private-skill"},
                        ]
                    },
                )
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect
        mock_safe.side_effect = lambda url: not url.startswith("http://127.0.0.1/")

        bundle = self.src.fetch("caldav-calendar")

        self.assertIsNone(bundle)
        self.assertEqual(mock_get.call_count, 3)
        mock_safe_get.assert_not_called()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_search_empty_query_paginates_full_catalog(
        self, mock_get, _mock_read_cache, _mock_write_cache
    ):
        """Empty query must walk the cursor-paginated catalog.

        Regression for the silent 200-skill truncation: ClawHub's listing
        endpoint caps any single page at 200 items + returns a `nextCursor`.
        The build_skills_index.py crawler calls `search("", limit=N)` with a
        large N to dump the full catalog. Before the fix, that hit a single
        unpaginated request and silently dropped 99% of the catalog.
        """
        # Three pages: 200 + 200 + 50 items, then no cursor → stop.
        page_calls = {"n": 0}
        pages = [
            {
                "items": [{"slug": f"a-skill-{i}", "displayName": f"A {i}"} for i in range(200)],
                "nextCursor": "cursor-page-2",
            },
            {
                "items": [{"slug": f"b-skill-{i}", "displayName": f"B {i}"} for i in range(200)],
                "nextCursor": "cursor-page-3",
            },
            {
                "items": [{"slug": f"c-skill-{i}", "displayName": f"C {i}"} for i in range(50)],
                "nextCursor": None,
            },
        ]

        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills"):
                idx = page_calls["n"]
                page_calls["n"] += 1
                if idx < len(pages):
                    return _MockResponse(status_code=200, json_data=pages[idx])
                return _MockResponse(status_code=200, json_data={"items": []})
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        results = self.src.search("", limit=10_000)

        # 200 + 200 + 50 = 450 unique skills, all retrieved via cursor pagination.
        self.assertEqual(len(results), 450)
        self.assertEqual(page_calls["n"], 3, "expected exactly 3 cursor-paginated pages")
        identifiers = {meta.identifier for meta in results}
        self.assertIn("a-skill-0", identifiers)
        self.assertIn("b-skill-199", identifiers)
        self.assertIn("c-skill-49", identifiers)

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_catalog_walk_aborts_on_budget_and_does_not_poison_cache(
        self, mock_get, _mock_read_cache, mock_write_cache
    ):
        """A walk truncated by the wall-clock budget must stop early and must
        NOT write the (partial) result to the cache. Before the budget guard
        the walk ran up to 750 pages and cached unconditionally — a truncated
        walk poisoned the cache with incomplete catalog data."""
        page_calls = {"n": 0}

        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills"):
                idx = page_calls["n"]
                page_calls["n"] += 1
                # Always advertise another page so the walk would never stop
                # on its own — only the budget can break it.
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "items": [
                            {"slug": f"skill-{idx}", "displayName": f"Skill {idx}"}
                        ],
                        "nextCursor": f"cursor-{idx + 1}",
                    },
                )
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        # Force the deadline to be in the past immediately. Budget only applies
        # to bounded browse walks (max_items > 0), not the index builder path.
        with patch.object(ClawHubSource, "CATALOG_WALK_BUDGET_SECONDS", -1):
            results = self.src._load_catalog_index(max_items=10)

        # Walk broke well before the 750-page cap.
        self.assertLess(page_calls["n"], 750)
        # Truncated walk must not poison the cache.
        mock_write_cache.assert_not_called()
        # Whatever was gathered is still returned to the caller.
        self.assertIsInstance(results, list)

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_catalog_walk_caches_when_terminating_naturally_within_budget(
        self, mock_get, _mock_read_cache, mock_write_cache
    ):
        """Happy path: a walk that exhausts the cursor within the budget DOES
        write the cache."""

        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "items": [
                            {"slug": "only-skill", "displayName": "Only Skill"}
                        ],
                        # No nextCursor -> natural termination.
                    },
                )
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        results = self.src._load_catalog_index()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "only-skill")
        mock_write_cache.assert_called_once()


class TestClawHubCatalogWalkBounded(unittest.TestCase):
    """max_items bounds the walk so browse's cold-start fallback renders one
    page without walking the entire 50k+ catalog. The offline index builder
    keeps max_items=0 (unbounded) and walks to exhaustion."""

    def setUp(self):
        self.src = ClawHubSource()
        self._safe_patcher = patch("tools.skills_hub.is_safe_url", return_value=True)
        self._policy_patcher = patch("tools.skills_hub.check_website_access", return_value=None)
        self._safe_patcher.start()
        self._policy_patcher.start()

    def tearDown(self):
        self._policy_patcher.stop()
        self._safe_patcher.stop()

    def _infinite_pages(self, page_calls):
        """A side_effect that always advertises another cursor — the walk would
        never stop on its own, so only max_items / budget can break it."""

        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills"):
                idx = page_calls["n"]
                page_calls["n"] += 1
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "items": [
                            {"slug": f"skill-{idx}", "displayName": f"Skill {idx}"}
                        ],
                        "nextCursor": f"cursor-{idx + 1}",
                    },
                )
            return _MockResponse(status_code=404, json_data={})

        return side_effect

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_max_items_stops_walk_early_and_does_not_cache(
        self, mock_get, _mock_read_cache, mock_write_cache
    ):
        """A bounded walk stops as soon as it has >= max_items skills and must
        NOT poison the shared full-catalog cache with the partial slice."""
        page_calls = {"n": 0}
        mock_get.side_effect = self._infinite_pages(page_calls)

        results = self.src._load_catalog_index(max_items=5)

        # Each mocked page yields exactly 1 item, so ~5 pages cover the bound.
        self.assertGreaterEqual(len(results), 5)
        self.assertLess(page_calls["n"], 750, "bounded walk should stop well before the cap")
        self.assertLess(page_calls["n"], 20, "should stop within a few pages of the bound")
        # Partial (bounded) walk must not be cached.
        mock_write_cache.assert_not_called()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_max_items_zero_ignores_wall_clock_budget(
        self, mock_get, _mock_read_cache, _mock_write_cache
    ):
        """Index builder path (max_items=0) must not truncate on the browse budget."""
        page_calls = {"n": 0}
        mock_get.side_effect = self._infinite_pages(page_calls)

        with patch.object(ClawHubSource, "CATALOG_WALK_BUDGET_SECONDS", -1):
            results = self.src._load_catalog_index(max_items=0)

        # No budget -> walks until the 750-page safety cap, not ~14 pages in 12s.
        self.assertEqual(page_calls["n"], 750)
        self.assertEqual(len(results), 750)

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_max_items_zero_is_unbounded_and_caches(
        self, mock_get, _mock_read_cache, mock_write_cache
    ):
        """max_items=0 (the index builder's path) walks to natural termination
        and DOES cache the complete catalog."""

        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills"):
                return _MockResponse(
                    status_code=200,
                    json_data={
                        "items": [
                            {"slug": "a", "displayName": "A"},
                            {"slug": "b", "displayName": "B"},
                            {"slug": "c", "displayName": "C"},
                        ],
                        # No nextCursor -> natural termination.
                    },
                )
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        results = self.src._load_catalog_index(max_items=0)

        self.assertEqual(len(results), 3)
        mock_write_cache.assert_called_once()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_empty_query_browse_bounds_walk_to_limit(
        self, mock_get, _mock_read_cache, _mock_write_cache
    ):
        """search("", limit=N) is the browse cold-start path — it must bound the
        catalog walk to N rather than walking the whole 50k+ catalog."""
        page_calls = {"n": 0}
        mock_get.side_effect = self._infinite_pages(page_calls)

        results = self.src.search("", limit=10)

        self.assertEqual(len(results), 10, "browse page should be exactly `limit` items")
        # Walk stopped near the bound, not at the 750-page cap.
        self.assertLess(page_calls["n"], 30)


if __name__ == "__main__":
    unittest.main()

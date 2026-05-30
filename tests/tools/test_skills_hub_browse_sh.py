#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from tools.skills_hub import BrowseShSource, SkillMeta, SkillBundle


# Catalog shape mirrors the real ``GET https://browse.sh/api/skills`` response:
# ``slug`` is ``<hostname>/<task-id>`` and ``name`` is the task name.
SAMPLE_CATALOG = [
    {
        "slug": "airbnb.com/search-listings-ddgioa",
        "name": "search-listings",
        "title": "Airbnb Search Listings",
        "description": "Search and browse Airbnb listings by location and dates.",
        "hostname": "airbnb.com",
        "category": "travel",
        "tags": ["travel", "accommodation"],
        "sourceUrl": "https://github.com/browserbase/browse.sh/blob/main/skills/airbnb.com/search-listings-ddgioa/SKILL.md",
        "recommendedMethod": "stagehand",
        "proxies": False,
        "installCount": 42,
    },
    {
        "slug": "amazon.com/search-products-xyz",
        "name": "search-products",
        "title": "Amazon Product Search",
        "description": "Search for products on Amazon.",
        "hostname": "amazon.com",
        "category": "shopping",
        "tags": ["shopping", "ecommerce"],
        "sourceUrl": "https://github.com/browserbase/browse.sh/blob/main/skills/amazon.com/search-products-xyz/SKILL.md",
        "recommendedMethod": "stagehand",
        "proxies": False,
        "installCount": 99,
    },
]


class _MockResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json_data


class TestBrowseShSource(unittest.TestCase):
    def setUp(self):
        self.src = BrowseShSource()

    def test_source_id(self):
        self.assertEqual(self.src.source_id(), "browse-sh")

    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_search_returns_results(self, _mock_catalog):
        results = self.src.search("airbnb", limit=10)
        self.assertGreaterEqual(len(results), 1)
        meta = results[0]
        self.assertIsInstance(meta, SkillMeta)
        self.assertEqual(meta.name, "search-listings")
        self.assertEqual(meta.source, "browse-sh")
        self.assertEqual(meta.trust_level, "community")
        self.assertEqual(meta.identifier, "browse-sh/airbnb.com/search-listings-ddgioa")
        self.assertIn("travel", meta.tags)

    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_search_filters_by_query(self, _mock_catalog):
        results = self.src.search("amazon", limit=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].extra["hostname"], "amazon.com")

        results_all = self.src.search("", limit=10)
        self.assertEqual(len(results_all), 2)

    @patch("tools.skills_hub.httpx.get")
    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_fetch_returns_bundle(self, _mock_catalog, mock_get):
        # First call: GET /api/skills/{slug} returns the detail object with skillMdUrl.
        # Second call: GET the CDN blob URL returns the SKILL.md text.
        blob_url = (
            "https://gh0lfhlmyzhg6tww.public.blob.vercel-storage.com"
            "/skills/airbnb.com/search-listings-ddgioa/SKILL.md"
        )
        mock_get.side_effect = [
            _MockResponse(status_code=200, json_data={"skillMdUrl": blob_url}),
            _MockResponse(status_code=200, text="# Airbnb Skill\n\nSearch and book Airbnb listings."),
        ]
        bundle = self.src.fetch("browse-sh/airbnb.com/search-listings-ddgioa")
        self.assertIsNotNone(bundle)
        self.assertIsInstance(bundle, SkillBundle)
        self.assertEqual(bundle.name, "search-listings")
        self.assertIn("SKILL.md", bundle.files)
        self.assertIn("Airbnb", bundle.files["SKILL.md"])
        self.assertEqual(bundle.source, "browse-sh")
        self.assertEqual(bundle.trust_level, "community")
        self.assertEqual(bundle.identifier, "browse-sh/airbnb.com/search-listings-ddgioa")
        self.assertEqual(bundle.metadata["skill_md_url"], blob_url)
        # Two HTTP calls: detail endpoint + blob.
        self.assertEqual(mock_get.call_count, 2)
        first_url = mock_get.call_args_list[0].args[0]
        second_url = mock_get.call_args_list[1].args[0]
        self.assertIn("/api/skills/airbnb.com/search-listings-ddgioa", first_url)
        self.assertEqual(second_url, blob_url)

    @patch("tools.skills_hub.httpx.get")
    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_fetch_falls_back_to_raw_github_url(self, _mock_catalog, mock_get):
        # Detail endpoint fails → fall back to a raw.githubusercontent.com sourceUrl.
        raw_catalog = [dict(SAMPLE_CATALOG[0])]
        raw_catalog[0]["sourceUrl"] = (
            "https://raw.githubusercontent.com/example/repo/main/skills/"
            "airbnb.com/search-listings-ddgioa/SKILL.md"
        )
        with patch.object(BrowseShSource, "_fetch_catalog", return_value=raw_catalog):
            mock_get.side_effect = [
                _MockResponse(status_code=500, json_data=None),  # detail endpoint fails
                _MockResponse(status_code=200, text="# Fallback content"),
            ]
            bundle = self.src.fetch("browse-sh/airbnb.com/search-listings-ddgioa")
            self.assertIsNotNone(bundle)
            self.assertEqual(bundle.files["SKILL.md"], "# Fallback content")

    @patch("tools.skills_hub.is_safe_url", side_effect=[True, False])
    @patch("tools.skills_hub.httpx.get")
    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_fetch_blocks_unsafe_skill_md_url(self, _mock_catalog, mock_get, _mock_safe):
        blob_url = "http://169.254.169.254/latest/meta-data"
        mock_get.return_value = _MockResponse(status_code=200, json_data={"skillMdUrl": blob_url})

        bundle = self.src.fetch("browse-sh/airbnb.com/search-listings-ddgioa")

        self.assertIsNone(bundle)
        # Detail endpoint is fetched, but the unsafe skillMdUrl is blocked before fetch.
        self.assertEqual(mock_get.call_count, 1)

    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_fetch_missing_slug_returns_none(self, _mock_catalog):
        result = self.src.fetch("browse-sh/nonexistent.com/no-such-skill")
        self.assertIsNone(result)

    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_inspect_returns_meta(self, _mock_catalog):
        meta = self.src.inspect("browse-sh/airbnb.com/search-listings-ddgioa")
        self.assertIsNotNone(meta)
        self.assertIsInstance(meta, SkillMeta)
        self.assertEqual(meta.name, "search-listings")
        self.assertEqual(meta.identifier, "browse-sh/airbnb.com/search-listings-ddgioa")
        self.assertEqual(meta.extra["hostname"], "airbnb.com")
        self.assertEqual(meta.extra["category"], "travel")
        self.assertEqual(meta.extra["install_count"], 42)


if __name__ == "__main__":
    unittest.main()

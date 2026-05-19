#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from tools.skills_hub import BrowseShSource, SkillMeta, SkillBundle


SAMPLE_CATALOG = [
    {
        "slug": "airbnb.com/search-listings-ddgioa",
        "name": "airbnb.com",
        "title": "Airbnb Search Listings",
        "description": "Search and browse Airbnb listings by location and dates.",
        "hostname": "airbnb.com",
        "category": "travel",
        "tags": ["travel", "accommodation"],
        "sourceUrl": "https://github.com/browserbase/browse-sh/blob/main/skills/airbnb.com/SKILL.md",
        "recommendedMethod": "stagehand",
        "proxies": False,
        "installCount": 42,
    },
    {
        "slug": "amazon.com/search-products-xyz",
        "name": "amazon.com",
        "title": "Amazon Product Search",
        "description": "Search for products on Amazon.",
        "hostname": "amazon.com",
        "category": "shopping",
        "tags": ["shopping", "ecommerce"],
        "sourceUrl": "https://raw.githubusercontent.com/browserbase/browse-sh/main/skills/amazon.com/SKILL.md",
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
        self.assertEqual(meta.name, "airbnb.com")
        self.assertEqual(meta.source, "browse-sh")
        self.assertEqual(meta.trust_level, "community")
        self.assertEqual(meta.identifier, "browse-sh/airbnb.com/search-listings-ddgioa")
        self.assertIn("travel", meta.tags)

    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_search_filters_by_query(self, _mock_catalog):
        results = self.src.search("amazon", limit=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "amazon.com")

        results_all = self.src.search("", limit=10)
        self.assertEqual(len(results_all), 2)

    @patch("tools.skills_hub.httpx.get")
    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_fetch_returns_bundle(self, _mock_catalog, mock_get):
        mock_get.return_value = _MockResponse(
            status_code=200,
            text="# Airbnb Skill\n\nSearch and book Airbnb listings.",
        )
        bundle = self.src.fetch("browse-sh/airbnb.com/search-listings-ddgioa")
        self.assertIsNotNone(bundle)
        self.assertIsInstance(bundle, SkillBundle)
        self.assertEqual(bundle.name, "airbnb.com")
        self.assertIn("SKILL.md", bundle.files)
        self.assertIn("Airbnb", bundle.files["SKILL.md"])
        self.assertEqual(bundle.source, "browse-sh")
        self.assertEqual(bundle.trust_level, "community")
        self.assertEqual(bundle.identifier, "browse-sh/airbnb.com/search-listings-ddgioa")
        mock_get.assert_called_once()
        call_url = mock_get.call_args.args[0]
        self.assertIn("raw.githubusercontent.com", call_url)

    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_fetch_missing_slug_returns_none(self, _mock_catalog):
        result = self.src.fetch("browse-sh/nonexistent.com/no-such-skill")
        self.assertIsNone(result)

    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    def test_inspect_returns_meta(self, _mock_catalog):
        meta = self.src.inspect("browse-sh/airbnb.com/search-listings-ddgioa")
        self.assertIsNotNone(meta)
        self.assertIsInstance(meta, SkillMeta)
        self.assertEqual(meta.name, "airbnb.com")
        self.assertEqual(meta.identifier, "browse-sh/airbnb.com/search-listings-ddgioa")
        self.assertEqual(meta.extra["hostname"], "airbnb.com")
        self.assertEqual(meta.extra["category"], "travel")
        self.assertEqual(meta.extra["install_count"], 42)

    def test_to_raw_url_conversion(self):
        # GitHub HTML URL should be converted
        html_url = "https://github.com/browserbase/browse-sh/blob/main/skills/airbnb.com/SKILL.md"
        raw_url = self.src._to_raw_url(html_url)
        self.assertEqual(
            raw_url,
            "https://raw.githubusercontent.com/browserbase/browse-sh/main/skills/airbnb.com/SKILL.md",
        )

        # Already a raw URL — should be returned unchanged
        already_raw = "https://raw.githubusercontent.com/browserbase/browse-sh/main/skills/amazon.com/SKILL.md"
        self.assertEqual(self.src._to_raw_url(already_raw), already_raw)

        # Unrecognised URL — should return None
        self.assertIsNone(self.src._to_raw_url("https://example.com/something"))


if __name__ == "__main__":
    unittest.main()

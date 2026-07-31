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

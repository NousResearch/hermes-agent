#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from tools.skills_hub_models import SkillMeta
from tools.skills_hub_sources import BrowseShSource


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


# A catalog entry whose detail endpoint has been dropped: /api/skills/{slug} 404s (None here),
# while the skill itself lives on in the repo its sourceUrl points at.
MOVED_ENTRY = {
    "slug": "browserbase.com/cookie-sync",
    "name": "cookie-sync",
    "title": "Cookie Sync",
    "description": "Sync cookies from local Chrome to a Browserbase context.",
    "hostname": "browserbase.com",
    "category": "browser",
    "tags": [],
    "sourceUrl": "https://github.com/browserbase/skills/blob/main/skills/cookie-sync/SKILL.md",
    "externalUrl": "https://github.com/browserbase/skills/blob/main/skills/cookie-sync/SKILL.md",
    "installCount": 0,
}
MOVED_RAW_URL = "https://raw.githubusercontent.com/browserbase/skills/main/skills/cookie-sync/SKILL.md"
MOVED_SKILL_MD = "---\nname: cookie-sync\ndescription: Sync cookies.\n---\n\n# Cookie Sync\n"


class TestBrowseShDroppedDetailEntry(unittest.TestCase):
    """The catalog outlives the detail endpoint; the entry's own GitHub link must still install."""

    def setUp(self):
        self.src = BrowseShSource()

    @patch.object(BrowseShSource, "_fetch_catalog", return_value=[MOVED_ENTRY])
    @patch("tools.skills_hub_sources._get_text")
    @patch("tools.skills_hub_sources._get_json", return_value={"error": "no skill found"})
    def test_fetch_falls_back_to_source_url(self, _mock_json, mock_text, _mock_catalog):
        mock_text.return_value = MOVED_SKILL_MD
        bundle = self.src.fetch("browse-sh/browserbase.com/cookie-sync")
        assert bundle is not None
        self.assertEqual(bundle.files["SKILL.md"], MOVED_SKILL_MD)
        self.assertEqual(bundle.name, "cookie-sync")
        self.assertEqual(bundle.metadata["skill_md_url"], MOVED_RAW_URL)

    @patch.object(BrowseShSource, "_fetch_catalog", return_value=[MOVED_ENTRY])
    @patch("tools.skills_hub_sources._get_text", return_value=None)
    @patch("tools.skills_hub_sources._get_json", return_value=None)
    def test_fetch_none_when_fallback_url_is_unfetchable(self, _mock_json, _mock_text, _mock_catalog):
        self.assertIsNone(self.src.fetch("browse-sh/browserbase.com/cookie-sync"))

    @patch.object(BrowseShSource, "_fetch_catalog", return_value=SAMPLE_CATALOG)
    @patch("tools.skills_hub_sources._get_text")
    @patch("tools.skills_hub_sources._get_json",
           return_value={"skillMdUrl": "https://cdn.example/skill.md"})
    def test_skill_md_url_prefers_detail_endpoint(self, _mock_json, mock_text, _mock_catalog):
        mock_text.return_value = MOVED_SKILL_MD
        bundle = self.src.fetch("browse-sh/airbnb.com/search-listings-ddgioa")
        assert bundle is not None
        self.assertEqual(bundle.metadata["skill_md_url"], "https://cdn.example/skill.md")

    def test_github_raw_url_conversions(self):
        from tools.skills_hub_sources import _github_raw_url
        self.assertEqual(_github_raw_url(MOVED_ENTRY["sourceUrl"]), MOVED_RAW_URL)
        self.assertEqual(_github_raw_url(MOVED_RAW_URL), MOVED_RAW_URL)
        self.assertIsNone(_github_raw_url("https://example.com/browserbase/skills/blob/main/x/SKILL.md"))
        self.assertIsNone(_github_raw_url(""))


if __name__ == "__main__":
    unittest.main()

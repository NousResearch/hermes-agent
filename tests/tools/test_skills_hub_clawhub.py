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

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_resolves_latest_version_and_downloads_raw_files(self, mock_get):
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
            if url == "https://files.example/skill-md":
                return _MockResponse(status_code=200, text="# Skill")
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        bundle = self.src.fetch("caldav-calendar")

        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.name, "caldav-calendar")
        self.assertIn("SKILL.md", bundle.files)
        self.assertEqual(bundle.files["SKILL.md"], "# Skill")
        self.assertEqual(bundle.files["README.md"], "hello")

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
    def test_fetch_blocks_private_raw_url(self, mock_get, mock_safe, _mock_policy):
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


if __name__ == "__main__":
    unittest.main()

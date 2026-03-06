#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from tools.skills_hub import ClawHubSource


class _MockResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class TestClawHubSource(unittest.TestCase):
    def setUp(self):
        self.src = ClawHubSource()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_search_uses_new_endpoint_and_parses_items(self, mock_get, _mock_read_cache, _mock_write_cache):
        mock_get.return_value = _MockResponse(
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

        results = self.src.search("caldav", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "caldav-calendar")
        self.assertEqual(results[0].name, "CalDAV Calendar")
        self.assertEqual(results[0].description, "Calendar integration")

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertTrue(args[0].endswith("/skills"))
        self.assertEqual(kwargs["params"], {"search": "caldav", "limit": 5})

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


if __name__ == "__main__":
    unittest.main()

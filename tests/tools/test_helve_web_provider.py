"""Unit tests for the bundled Helve web search + extract plugin."""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _ensure_httpx_stub():
    if "httpx" not in sys.modules:
        mod = types.ModuleType("httpx")
        mod.post = MagicMock()
        mod.get = MagicMock()
        mod.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        mod.RequestError = type("RequestError", (Exception,), {})
        sys.modules["httpx"] = mod


class TestHelveWebSearchProvider(unittest.TestCase):
    def setUp(self):
        _ensure_httpx_stub()
        from plugins.web.helve.provider import HelveWebSearchProvider

        self.provider = HelveWebSearchProvider()

    def test_metadata_and_setup_schema(self):
        self.assertEqual(self.provider.name, "helve")
        self.assertEqual(self.provider.display_name, "Helve")
        self.assertTrue(self.provider.supports_search())
        self.assertTrue(self.provider.supports_extract())
        schema = self.provider.get_setup_schema()
        self.assertEqual(schema["name"], "Helve")
        self.assertEqual(schema["env_vars"][0]["key"], "HELVE_API_KEY")

    def test_search_normalizes_results(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "id": "srch_1",
            "provider": "fusion",
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": "Alpha",
                    "snippets": ["First snippet", "Second snippet"],
                },
                {
                    "url": "https://example.com/b",
                    "title": "Beta",
                    "summary": "Summary text",
                },
            ],
        }
        env = {"HELVE_API_KEY": "sk_live_test", "HELVE_SEARCH_PROVIDER": "fusion"}
        with patch("plugins.web.helve.provider.provider_env", side_effect=lambda k: env.get(k, "")), \
             patch("plugins.web._common._interrupted", return_value=False), \
             patch("plugins.web.helve.provider.httpx.post", return_value=resp) as mock_post:
            out = self.provider.search("agent tools", limit=5)

        self.assertTrue(out["success"])
        web = out["data"]["web"]
        self.assertEqual(len(web), 2)
        self.assertEqual(web[0]["url"], "https://example.com/a")
        self.assertEqual(web[0]["description"], "First snippet\nSecond snippet")
        self.assertEqual(web[1]["description"], "Summary text")
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["provider"], "fusion")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk_live_test")

    def test_extract_normalizes_and_marks_missing_urls(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "results": [
                {
                    "url": "https://example.com/ok",
                    "final_url": "https://example.com/ok-final",
                    "title": "OK Page",
                    "content": "# Markdown body",
                }
            ]
        }
        env = {"HELVE_API_KEY": "sk_live_test"}
        with patch("plugins.web.helve.provider.provider_env", side_effect=lambda k: env.get(k, "")), \
             patch("plugins.web._common._interrupted", return_value=False), \
             patch("plugins.web.helve.provider.httpx.post", return_value=resp):
            docs = self.provider.extract(["https://example.com/ok", "https://example.com/missing"])

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["content"], "# Markdown body")
        self.assertEqual(docs[0]["metadata"]["sourceURL"], "https://example.com/ok-final")
        self.assertIn("error", docs[1])

    def test_missing_key_returns_clear_error(self):
        with patch("plugins.web.helve.provider.provider_env", return_value=""), \
             patch("plugins.web._common._interrupted", return_value=False):
            out = self.provider.search("test")
        self.assertFalse(out["success"])
        self.assertIn("HELVE_API_KEY", out["error"])

    def test_http_402_propagates_detail(self):
        resp = MagicMock()
        resp.status_code = 402
        resp.text = '{"error":{"type":"spend_limit_exceeded","message":"Key spend limit reached"}}'
        with patch("plugins.web.helve.provider.provider_env", return_value="sk_live_test"), \
             patch("plugins.web._common._interrupted", return_value=False), \
             patch("plugins.web.helve.provider.httpx.post", return_value=resp):
            out = self.provider.search("test")
        self.assertFalse(out["success"])
        self.assertIn("spend_limit_exceeded", out["error"])


if __name__ == "__main__":
    unittest.main()

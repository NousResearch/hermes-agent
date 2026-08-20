import json
import os
import tempfile
import unittest
from pathlib import Path


class InventoryExtractorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["HERMES_HOME"] = str(Path(self.tmp.name) / "companyintel")
        os.environ["HERMES_PROFILE"] = "companyintel"

    def tearDown(self):
        self.tmp.cleanup()

    def test_extracts_bounded_normalized_inventory_from_site_sources(self):
        from tools.companyintel_inventory import InventoryLimits, extract_inventory

        pages = {
            "https://example.com/": ("text/html", b'''<html><head>
              <title>Example Company</title>
              <meta name="description" content="Official company site">
              <link rel="canonical" href="https://example.com/">
              <link rel="alternate" type="application/rss+xml" href="/feed.xml">
              <script type="application/ld+json">{"@type":"Organization","name":"Example Company","url":"https://example.com/"}</script>
            </head><body>
              <a href="/about">About</a><a href="https://outside.example/news">Outside</a>
              <a href="/report.pdf">Report</a><img src="/logo.svg"><script src="/assets/app.js"></script>
              <script>gtag('config', 'G-ABC123');</script>
            </body></html>'''),
            "https://example.com/robots.txt": ("text/plain", b"User-agent: *\nSitemap: https://example.com/sitemap.xml\n"),
            "https://example.com/sitemap.xml": ("application/xml", b"<urlset><url><loc>https://example.com/about</loc></url></urlset>"),
            "https://example.com/feed.xml": ("application/rss+xml", b"<rss><channel><item><link>https://example.com/news</link><title>News</title></item></channel></rss>"),
            "https://example.com/llms.txt": ("text/plain", b"# Example\n- About: https://example.com/about\n"),
        }

        def fetch(url, limits):
            return pages.get(url, ("", b""))

        result = extract_inventory("https://example.com", limits=InventoryLimits(max_urls=12), fetcher=fetch)

        self.assertEqual(result["schema_version"], "companyintel-inventory/v1")
        self.assertEqual(result["target"]["url"], "https://example.com")
        self.assertEqual(result["metadata"]["title"], "Example Company")
        self.assertIn("https://example.com/about", result["urls"])
        self.assertIn("https://example.com/report.pdf", result["documents"])
        self.assertIn("https://example.com/logo.svg", result["images"])
        self.assertIn("G-ABC123", result["identifiers"])
        self.assertIn("https://example.com/sitemap.xml", result["discovered_sources"])
        self.assertIn("outside.example", result["external_domains"])
        self.assertLessEqual(result["stats"]["fetched_urls"], 12)
        self.assertLessEqual(len(json.dumps(result)), 20000)

    def test_rejects_private_literal_targets_before_fetch(self):
        from tools.companyintel_inventory import extract_inventory

        with self.assertRaises(ValueError):
            extract_inventory("http://127.0.0.1:8000")

    def test_graph_inventory_action_persists_evidence_and_nodes(self):
        from unittest.mock import patch
        from tools.companyintel_graph_tool import companyintel_graph

        def fetch(url, limits):
            if url == "https://example.com/":
                return "text/html", b'<title>Example</title><a href="/report.pdf">Report</a>'
            return "", b""

        with patch("tools.companyintel_inventory._fetch_url", side_effect=fetch):
            result = json.loads(companyintel_graph({
                "action": "inventory",
                "run_id": "run_inventory_001",
                "target_url": "https://example.com",
            }))
        self.assertTrue(result["ok"], result)
        self.assertGreaterEqual(result["findings"], 1)
        summary = json.loads(companyintel_graph({"action": "summary", "run_id": "run_inventory_001"}))
        self.assertGreaterEqual(summary["evidence"], 1)
        self.assertGreaterEqual(summary["nodes"], 2)
        graph = json.loads((Path(self.tmp.name) / "companyintel" / "companyintel" / "runs" / "run_inventory_001" / "graph.json").read_text())
        self.assertEqual(graph["inventory"]["schema_version"], "companyintel-inventory/v1")


if __name__ == "__main__":
    unittest.main()

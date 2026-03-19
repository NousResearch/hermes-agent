#!/usr/bin/env python3
"""Tests for the HeuristSource adapter in tools/skills_hub.py."""

import unittest
from unittest.mock import patch, MagicMock

from tools.skills_hub import HeuristSource, SkillMeta, SkillBundle


class _MockResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


# Sample API fixtures ---------------------------------------------------

SKILL_LIST_RESPONSE = {
    "skills": [
        {
            "id": "09284204",
            "slug": "heurist-mesh",
            "name": "heurist-mesh-skill",
            "description": "Access real-time crypto data via Heurist Mesh agents.",
            "category": "Crypto",
            "labels": ["defi", "research"],
            "risk_tier": "low",
            "verification_status": "verified",
            "author": {"display_name": "Heurist Network"},
            "file_url": "https://gateway.autonomys.xyz/file/abc123",
            "external_api_dependencies": [],
            "reference_urls": [],
            "download_count": 40,
            "star_count": 5,
            "capabilities": {
                "requires_secrets": True,
                "requires_private_keys": False,
                "requires_exchange_api_keys": False,
                "can_sign_transactions": False,
                "uses_leverage": False,
                "accesses_user_portfolio": False,
            },
            "homepage": None,
        },
        {
            "id": "252751da",
            "slug": "pay-for-service",
            "name": "pay-for-service",
            "description": "Make a paid API request to an x402 endpoint.",
            "category": "Crypto",
            "labels": ["execution", "payments"],
            "risk_tier": "high",
            "verification_status": "verified",
            "author": {"display_name": "Coinbase"},
            "file_url": "https://gateway.autonomys.xyz/file/def456",
            "external_api_dependencies": [],
            "reference_urls": [],
            "download_count": 1,
            "star_count": 0,
            "capabilities": {
                "requires_secrets": False,
                "requires_private_keys": False,
                "requires_exchange_api_keys": False,
                "can_sign_transactions": True,
                "uses_leverage": False,
                "accesses_user_portfolio": True,
            },
            "homepage": None,
        },
    ],
    "total": 2,
}

SKILL_DETAIL_SINGLE = {
    "id": "252751da",
    "slug": "pay-for-service",
    "name": "pay-for-service",
    "description": "Make a paid API request to an x402 endpoint.",
    "category": "Crypto",
    "labels": ["execution", "payments"],
    "risk_tier": "high",
    "verification_status": "verified",
    "author": {"display_name": "Coinbase"},
    "file_url": "https://gateway.autonomys.xyz/file/def456",
    "external_api_dependencies": [],
    "capabilities": {
        "requires_secrets": False,
        "requires_private_keys": False,
        "requires_exchange_api_keys": False,
        "can_sign_transactions": True,
        "uses_leverage": False,
        "accesses_user_portfolio": True,
    },
    "approved_sha256": "abc123hash",
    "source_url": "https://github.com/coinbase/skills",
    "is_folder": False,
    "folder_manifest": None,
    "approved_at": "2026-03-01T00:00:00+00:00",
}

SKILL_DETAIL_FOLDER = {
    "id": "09284204",
    "slug": "heurist-mesh",
    "name": "heurist-mesh-skill",
    "description": "Access real-time crypto data via Heurist Mesh agents.",
    "category": "Crypto",
    "labels": ["defi", "research"],
    "risk_tier": "low",
    "verification_status": "verified",
    "author": {"display_name": "Heurist Network"},
    "file_url": "https://gateway.autonomys.xyz/file/abc123",
    "external_api_dependencies": [],
    "capabilities": {
        "requires_secrets": True,
        "requires_private_keys": False,
        "requires_exchange_api_keys": False,
        "can_sign_transactions": False,
        "uses_leverage": False,
        "accesses_user_portfolio": False,
    },
    "approved_sha256": "508a77bb",
    "source_url": "https://github.com/heurist-network/heurist-mesh-skill",
    "is_folder": True,
    "folder_manifest": {
        "SKILL.md": "cid1",
        "README.md": "cid2",
    },
    "approved_at": "2026-03-09T00:00:00+00:00",
}

FILES_RESPONSE = {
    "slug": "heurist-mesh",
    "is_folder": True,
    "file_count": 2,
    "files": [
        {
            "path": "SKILL.md",
            "cid": "cid1",
            "gateway_url": "https://gateway.autonomys.xyz/file/cid1",
        },
        {
            "path": "README.md",
            "cid": "cid2",
            "gateway_url": "https://gateway.autonomys.xyz/file/cid2",
        },
    ],
}


class TestHeuristSourceId(unittest.TestCase):
    def test_source_id(self):
        src = HeuristSource()
        self.assertEqual(src.source_id(), "heurist")

    def test_trust_level(self):
        src = HeuristSource()
        self.assertEqual(src.trust_level_for("heurist:anything"), "community")


class TestHeuristStripPrefix(unittest.TestCase):
    def test_strip_prefix(self):
        self.assertEqual(HeuristSource._strip_prefix("heurist:my-skill"), "my-skill")

    def test_no_prefix(self):
        self.assertEqual(HeuristSource._strip_prefix("my-skill"), "my-skill")


class TestHeuristSearch(unittest.TestCase):
    def setUp(self):
        self.src = HeuristSource()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_search_returns_results(self, mock_get, _mock_read, _mock_write):
        mock_get.return_value = _MockResponse(200, SKILL_LIST_RESPONSE)

        results = self.src.search("crypto", limit=10)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].name, "heurist-mesh-skill")
        self.assertEqual(results[0].identifier, "heurist:heurist-mesh")
        self.assertEqual(results[0].source, "heurist")
        self.assertEqual(results[0].trust_level, "community")
        self.assertEqual(results[0].tags, ["defi", "research"])
        self.assertEqual(results[0].extra["risk_tier"], "low")
        self.assertEqual(results[0].extra["author"], "Heurist Network")

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_search_passes_query_params(self, mock_get, _mock_read, _mock_write):
        mock_get.return_value = _MockResponse(200, {"skills": []})

        self.src.search("defi", limit=5)

        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["search"], "defi")
        self.assertEqual(kwargs["params"]["limit"], 5)
        self.assertEqual(kwargs["params"]["verification_status"], "verified")

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_search_empty_query(self, mock_get, _mock_read, _mock_write):
        mock_get.return_value = _MockResponse(200, {"skills": []})

        self.src.search("", limit=10)

        _, kwargs = mock_get.call_args
        self.assertNotIn("search", kwargs["params"])

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_search_api_error(self, mock_get, _mock_read, _mock_write):
        mock_get.return_value = _MockResponse(500)

        results = self.src.search("test")
        self.assertEqual(results, [])

    @patch("tools.skills_hub._read_index_cache")
    def test_search_uses_cache(self, mock_read):
        cached = [
            {
                "name": "cached-skill",
                "description": "From cache",
                "source": "heurist",
                "identifier": "heurist:cached-skill",
                "trust_level": "community",
                "tags": [],
                "extra": {},
                "repo": None,
                "path": None,
            }
        ]
        mock_read.return_value = cached

        results = self.src.search("test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "cached-skill")


class TestHeuristInspect(unittest.TestCase):
    def setUp(self):
        self.src = HeuristSource()

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_returns_meta(self, mock_get):
        mock_get.return_value = _MockResponse(200, SKILL_DETAIL_SINGLE)

        meta = self.src.inspect("heurist:pay-for-service")

        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "pay-for-service")
        self.assertEqual(meta.identifier, "heurist:pay-for-service")
        self.assertEqual(meta.extra["approved_sha256"], "abc123hash")
        self.assertEqual(meta.extra["source_url"], "https://github.com/coinbase/skills")
        self.assertFalse(meta.extra["is_folder"])

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_strips_prefix(self, mock_get):
        mock_get.return_value = _MockResponse(200, SKILL_DETAIL_SINGLE)

        self.src.inspect("heurist:pay-for-service")

        args, _ = mock_get.call_args
        self.assertTrue(args[0].endswith("/skills/pay-for-service"))

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_not_found(self, mock_get):
        mock_get.return_value = _MockResponse(404)

        meta = self.src.inspect("heurist:nonexistent")
        self.assertIsNone(meta)


class TestHeuristFetchSingleFile(unittest.TestCase):
    def setUp(self):
        self.src = HeuristSource()

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_single_file_skill(self, mock_get):
        def side_effect(url, **kwargs):
            if url.endswith("/skills/pay-for-service"):
                return _MockResponse(200, SKILL_DETAIL_SINGLE)
            if "gateway.autonomys.xyz" in url:
                return _MockResponse(200, text="---\nname: pay-for-service\n---\n# Instructions\n")
            return _MockResponse(404)

        mock_get.side_effect = side_effect

        bundle = self.src.fetch("heurist:pay-for-service")

        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.name, "pay-for-service")
        self.assertEqual(bundle.source, "heurist")
        self.assertEqual(bundle.identifier, "heurist:pay-for-service")
        self.assertIn("SKILL.md", bundle.files)
        self.assertEqual(bundle.metadata["approved_sha256"], "abc123hash")
        self.assertEqual(bundle.metadata["risk_tier"], "high")
        self.assertTrue(bundle.metadata["capabilities"]["can_sign_transactions"])

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_not_found(self, mock_get):
        mock_get.return_value = _MockResponse(404)

        bundle = self.src.fetch("heurist:nonexistent")
        self.assertIsNone(bundle)


class TestHeuristFetchFolder(unittest.TestCase):
    def setUp(self):
        self.src = HeuristSource()

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_folder_skill(self, mock_get):
        def side_effect(url, **kwargs):
            if url.endswith("/skills/heurist-mesh") and "/files" not in url:
                return _MockResponse(200, SKILL_DETAIL_FOLDER)
            if url.endswith("/skills/heurist-mesh/files"):
                return _MockResponse(200, FILES_RESPONSE)
            if "gateway.autonomys.xyz/file/cid1" in url:
                return _MockResponse(200, text="---\nname: heurist-mesh\n---\n# SKILL content\n")
            if "gateway.autonomys.xyz/file/cid2" in url:
                return _MockResponse(200, text="# README\nHeurist Mesh Skill\n")
            return _MockResponse(404)

        mock_get.side_effect = side_effect

        bundle = self.src.fetch("heurist:heurist-mesh")

        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.name, "heurist-mesh")
        self.assertIn("SKILL.md", bundle.files)
        self.assertIn("README.md", bundle.files)
        self.assertEqual(len(bundle.files), 2)
        self.assertEqual(bundle.metadata["approved_sha256"], "508a77bb")

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_folder_no_skill_md(self, mock_get):
        """If folder files endpoint returns no SKILL.md, fetch returns None."""
        files_no_skill = {
            "slug": "broken",
            "is_folder": True,
            "file_count": 1,
            "files": [
                {
                    "path": "README.md",
                    "cid": "cid2",
                    "gateway_url": "https://gateway.autonomys.xyz/file/cid2",
                },
            ],
        }

        def side_effect(url, **kwargs):
            if "/files" not in url and url.endswith("/skills/broken"):
                return _MockResponse(200, {**SKILL_DETAIL_FOLDER, "slug": "broken", "is_folder": True})
            if url.endswith("/skills/broken/files"):
                return _MockResponse(200, files_no_skill)
            if "gateway.autonomys.xyz" in url:
                return _MockResponse(200, text="# Just a readme\n")
            return _MockResponse(404)

        mock_get.side_effect = side_effect

        bundle = self.src.fetch("heurist:broken")
        self.assertIsNone(bundle)


class TestHeuristSecurityMetadata(unittest.TestCase):
    """Verify that security metadata (risk_tier, capabilities) is surfaced."""

    def setUp(self):
        self.src = HeuristSource()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_search_surfaces_capabilities(self, mock_get, _mock_read, _mock_write):
        mock_get.return_value = _MockResponse(200, SKILL_LIST_RESPONSE)

        results = self.src.search("pay", limit=10)

        pay_skill = next((r for r in results if r.identifier == "heurist:pay-for-service"), None)
        self.assertIsNotNone(pay_skill)
        self.assertEqual(pay_skill.extra["risk_tier"], "high")
        self.assertTrue(pay_skill.extra["capabilities"]["can_sign_transactions"])
        self.assertTrue(pay_skill.extra["capabilities"]["accesses_user_portfolio"])

    @patch("tools.skills_hub.httpx.get")
    def test_inspect_surfaces_security_fields(self, mock_get):
        mock_get.return_value = _MockResponse(200, SKILL_DETAIL_SINGLE)

        meta = self.src.inspect("heurist:pay-for-service")

        self.assertEqual(meta.extra["risk_tier"], "high")
        self.assertTrue(meta.extra["capabilities"]["can_sign_transactions"])
        self.assertEqual(meta.extra["approved_sha256"], "abc123hash")

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_includes_security_in_metadata(self, mock_get):
        def side_effect(url, **kwargs):
            if url.endswith("/skills/pay-for-service"):
                return _MockResponse(200, SKILL_DETAIL_SINGLE)
            if "gateway.autonomys.xyz" in url:
                return _MockResponse(200, text="---\nname: test\n---\n# Test\n")
            return _MockResponse(404)

        mock_get.side_effect = side_effect

        bundle = self.src.fetch("heurist:pay-for-service")
        self.assertEqual(bundle.metadata["risk_tier"], "high")
        self.assertTrue(bundle.metadata["capabilities"]["can_sign_transactions"])


class TestHeuristRouterRegistration(unittest.TestCase):
    """Verify HeuristSource is present in the source router."""

    @patch("tools.skills_hub.TapsManager")
    def test_heurist_in_router(self, _mock_taps):
        _mock_taps.return_value.list_taps.return_value = []
        from tools.skills_hub import create_source_router
        auth = MagicMock()
        sources = create_source_router(auth=auth)
        source_ids = [s.source_id() for s in sources]
        self.assertIn("heurist", source_ids)


if __name__ == "__main__":
    unittest.main()

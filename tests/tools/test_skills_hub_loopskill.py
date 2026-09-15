#!/usr/bin/env python3
"""Behaviour-contract tests for the LoopSkill Skills Hub adapter.

Written to fail on ``main`` (no ``tools/skills_hub_loopskill`` module) and pass
once ``LoopSkillSource`` lands. Mirrors ``tests/tools/test_skills_hub_browse_sh.py``
in shape: mock the HTTP layer, never hit the network.
"""

import unittest
from unittest.mock import patch

from tools.skills_hub_loopskill import LoopSkillSource
from tools.skills_hub_models import SkillBundle, SkillMeta, _dedupe_by_trust
from tools.skills_hub_search import create_source_router

# Shape mirrors the verified live response for
# GET https://app.loopskill.io/api/skills/search?q=<term>
SAMPLE_SEARCH_RESPONSE = {
    "results": [
        {
            "id": "fc39ba1b-d050-4f8e-ae37-db38a566f3c5",
            "slug": "clean-architecture",
            "title": "clean-architecture",
            "description": "Structure software around the Dependency Rule.",
            "category": "ops",
            "tier": "free",
            "is_public": True,
            "latest_version": "1.0.0",
        },
        {
            "id": "aaaaaaaa-0000-0000-0000-000000000000",
            "slug": "premium-thing",
            "title": "premium-thing",
            "description": "A paid skill.",
            "category": "ops",
            "tier": "pro",
            "is_public": True,
            "latest_version": "2.0.0",
        },
    ],
    "total": 2, "page": 1, "page_size": 50, "backend": "keyword",
}

# Shape mirrors the verified live response for GET /api/skills/{slug}.
SAMPLE_FREE_DETAIL = {
    "slug": "clean-architecture", "title": "clean-architecture",
    "description": "Structure software around the Dependency Rule.",
    "readme": "---\nname: clean-architecture\ndescription: Structure software.\n---\n\n# Clean Architecture\n",
    "tier": "free", "license": "MIT", "latest_version": "1.0.0",
}

SAMPLE_LOCKED_DETAIL = {
    "slug": "premium-thing", "title": "premium-thing", "description": "A paid skill.",
    "readme": "---\nname: premium-thing\n---\n\nSHOULD NEVER BE RETURNED\n",
    "tier": "pro", "license": "proprietary", "latest_version": "2.0.0",
}


class TestLoopSkillSource(unittest.TestCase):
    def setUp(self):
        self.src = LoopSkillSource()

    def test_source_id(self):
        self.assertEqual(self.src.source_id(), "loopskill")

    @patch("tools.skills_hub_loopskill._get_json", return_value=SAMPLE_SEARCH_RESPONSE)
    def test_search_maps_result_to_skill_meta(self, _mock_get):
        results = self.src.search("clean architecture", limit=10)
        self.assertGreaterEqual(len(results), 1)
        meta = next(m for m in results if m.extra.get("slug") == "clean-architecture")
        self.assertIsInstance(meta, SkillMeta)
        self.assertEqual(meta.source, "loopskill")
        self.assertEqual(meta.identifier, "loopskill/clean-architecture")
        self.assertEqual(meta.trust_level, "community")

    @patch("tools.skills_hub_loopskill._get_json", return_value=SAMPLE_SEARCH_RESPONSE)
    def test_search_flags_locked_tier_result(self, _mock_get):
        results = self.src.search("premium", limit=10)
        locked = next(m for m in results if m.extra.get("slug") == "premium-thing")
        self.assertTrue(locked.extra.get("locked"))
        self.assertEqual(locked.extra.get("tier"), "pro")

    def test_search_round_trips_into_install(self):
        """A search hit's identifier round-trips through fetch() into an
        installable SkillBundle for a free-tier skill — the core behaviour
        contract, not a value freeze."""
        with patch("tools.skills_hub_loopskill._get_json", return_value=SAMPLE_SEARCH_RESPONSE):
            results = self.src.search("clean architecture", limit=10)
        meta = next(m for m in results if m.extra.get("slug") == "clean-architecture")

        with patch("tools.skills_hub_loopskill._get_json", return_value=SAMPLE_FREE_DETAIL):
            bundle = self.src.fetch(meta.identifier)

        self.assertIsInstance(bundle, SkillBundle)
        self.assertEqual(bundle.source, "loopskill")
        self.assertEqual(bundle.identifier, meta.identifier)
        self.assertEqual(bundle.trust_level, "community")
        self.assertIn("SKILL.md", bundle.files)
        self.assertIn("Clean Architecture", bundle.files["SKILL.md"])

    def test_fetch_refuses_locked_tier(self):
        """fetch() must never return locked (non-free) content anonymously —
        the paid-skill invariant from the well-known adapter's paywall test."""
        with patch("tools.skills_hub_loopskill._get_json", return_value=SAMPLE_LOCKED_DETAIL):
            bundle = self.src.fetch("loopskill/premium-thing")
        self.assertIsNone(bundle)

    def test_inspect_flags_locked_tier(self):
        with patch("tools.skills_hub_loopskill._get_json", return_value=SAMPLE_LOCKED_DETAIL):
            meta = self.src.inspect("loopskill/premium-thing")
        self.assertIsNotNone(meta)
        self.assertTrue(meta.extra.get("locked"))

    def test_registered_in_source_router(self):
        """Structural contract: loopskill participates in the router, not a
        hardcoded source count."""
        sources = create_source_router()
        source_ids = {src.source_id() for src in sources}
        self.assertIn("loopskill", source_ids)

    def test_community_result_never_displaces_higher_trust_same_identifier(self):
        """Relationship test mirroring _dedupe_by_trust's existing coverage:
        a loopskill (community) result never wins over a builtin/trusted
        result sharing the same identifier."""
        loopskill_meta = SkillMeta(
            name="dup", description="loopskill copy", source="loopskill",
            identifier="shared/dup", trust_level="community",
        )
        builtin_meta = SkillMeta(
            name="dup", description="builtin copy", source="official",
            identifier="shared/dup", trust_level="builtin",
        )
        deduped = _dedupe_by_trust([loopskill_meta, builtin_meta])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].trust_level, "builtin")

        deduped_reversed = _dedupe_by_trust([builtin_meta, loopskill_meta])
        self.assertEqual(len(deduped_reversed), 1)
        self.assertEqual(deduped_reversed[0].trust_level, "builtin")


if __name__ == "__main__":
    unittest.main()

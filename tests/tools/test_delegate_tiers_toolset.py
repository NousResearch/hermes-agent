#!/usr/bin/env python3
"""
Additional edge case tests for toolset-aware routing and cache invalidation.

Run: python -m pytest tests/tools/test_delegate_tiers_toolset.py -v -o 'addopts='
"""

import unittest
from unittest.mock import patch

from tools.delegate_tool import (
    _infer_delegate_tier,
    _get_max_concurrent_children,
    _invalidate_max_concurrent_cache,
    _resolve_effective_tier,
)
import tools.delegate_tool as dt


class TestToolsetAwareRouting(unittest.TestCase):
    """Verify toolsets influence tier selection."""

    def test_web_only_with_research_keyword(self):
        result = _infer_delegate_tier("find the best database options", "", ["web"], {})
        self.assertEqual(result, "research")

    def test_web_only_default_research(self):
        result = _infer_delegate_tier("search for information about this topic", "", ["web"], {})
        self.assertEqual(result, "research")

    def test_file_only_with_review_keyword(self):
        result = _infer_delegate_tier("examine this code for issues", "", ["file"], {})
        self.assertEqual(result, "review")

    def test_terminal_file_with_build_keyword(self):
        result = _infer_delegate_tier("build the new feature module", "", ["terminal", "file"], {})
        self.assertEqual(result, "heavy")

    def test_keyword_overrides_toolset(self):
        """Explicit review keyword wins even with terminal+file toolsets."""
        result = _infer_delegate_tier("review the security of the auth module", "", ["terminal", "file"], {})
        self.assertEqual(result, "review")

    def test_light_keyword_wins_over_toolsets(self):
        """Explicit light keyword wins even with terminal+file toolsets."""
        result = _infer_delegate_tier("count the lines in the file", "", ["terminal", "file"], {})
        self.assertEqual(result, "light")

    def test_empty_toolsets_no_bias(self):
        """Empty toolsets don't add bias."""
        result = _infer_delegate_tier("some complex task that needs work", "", [], {})
        self.assertIsNone(result)  # no signal -> None (LLM fallback)

    def test_web_only_default_research(self):
        """Web-only toolset with non-keyword query defaults to research."""
        result = _infer_delegate_tier("get the current status of deployment", "", ["web"], {})
        self.assertEqual(result, "research")

    def test_mixed_toolsets_no_web_only_bias(self):
        """Mixed toolsets (web+file) don't trigger web-only bias; returns None when no keyword matches."""
        result = _infer_delegate_tier("verify the schema is correct", "", ["web", "file"], {})
        # "verify" is in REVIEW_SIGNALS, so review wins regardless of toolsets
        self.assertEqual(result, "review")


class TestCacheInvalidation(unittest.TestCase):

    def test_invalidate_clears_cache(self):
        dt._cached_max_concurrent = 42
        dt._cached_config_fingerprint = "abc"
        _invalidate_max_concurrent_cache()
        self.assertIsNone(dt._cached_max_concurrent)
        self.assertIsNone(dt._cached_config_fingerprint)

    def test_fingerprint_changes_with_config(self):
        with patch("tools.delegate_tool._load_config", return_value={"max_concurrent_children": 5}):
            fp1 = dt._config_fingerprint()
        with patch("tools.delegate_tool._load_config", return_value={"max_concurrent_children": 10}):
            fp2 = dt._config_fingerprint()
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_stable_same_config(self):
        with patch("tools.delegate_tool._load_config", return_value={"max_concurrent_children": 5}):
            fp1 = dt._config_fingerprint()
            fp2 = dt._config_fingerprint()
        self.assertEqual(fp1, fp2)


class TestAmbiguousGoalResolution(unittest.TestCase):

    def test_review_planning_review_wins(self):
        """Review has highest priority among ambiguous signals."""
        result = _infer_delegate_tier("review this architecture plan", "", [], {})
        self.assertEqual(result, "review")

    def test_research_planning_planning_wins(self):
        """Planning beats research when both match."""
        result = _infer_delegate_tier("plan the research methodology", "", [], {})
        self.assertEqual(result, "planning")

    def test_light_research_research_wins(self):
        """Research beats light when both match."""
        result = _infer_delegate_tier("list and research the available options", "", [], {})
        self.assertEqual(result, "research")


if __name__ == "__main__":
    unittest.main()

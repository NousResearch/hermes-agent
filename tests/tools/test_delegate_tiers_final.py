#!/usr/bin/env python3
"""
Final edge case tests — fills remaining gaps from re-review.

Covers:
- default_tier pointing to unknown tier
- per-task batch pool override
- malformed tier config with own pool
- cache behavior verification

Run: python -m pytest tests/tools/test_delegate_tiers_final.py -v -o 'addopts='
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _validate_pool_model,
    delegate_task,
    resolve_tier_config,
)
import tools.delegate_tool as dt


def _make_mock_parent(depth=0):
    import threading
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


class TestDefaultTierUnknownName(unittest.TestCase):
    """default_tier pointing to an unknown tier should fall back to flat config."""

    def test_default_tier_unknown_logs_warning(self):
        import logging
        with self.assertLogs("tools.delegate_tool", level="WARNING") as cm:
            result = resolve_tier_config(
                {
                    "model": "x",
                    "default_tier": "nonexistent",
                    "tiers": {"light": {"model": "y"}},
                },
            )
        self.assertTrue(any("nonexistent" in msg for msg in cm.output))
        self.assertEqual(result["model"], "x")
        self.assertNotIn("tiers", result)
        self.assertNotIn("default_tier", result)


class TestPerTaskPoolInBatch(unittest.TestCase):
    """Per-task tier pool should be used for pool validation in batch mode."""

    @patch("tools.delegate_tool._load_config")
    def test_batch_per_task_pool_override(self, mock_cfg):
        mock_cfg.return_value = {
            "model": "base-model",
            "pool": [{"model": "base-pool-model"}],
            "tiers": {
                "review": {
                    "model": "review-model",
                    "pool": [{"model": "review-pool-model"}],
                },
            },
        }
        parent = _make_mock_parent()
        captured = []

        def capture_child(**kwargs):
            captured.append(kwargs)
            child = MagicMock()
            child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "messages": [],
            }
            return child

        with patch("tools.delegate_tool._build_child_agent", side_effect=capture_child):
            delegate_task(
                tasks=[
                    {"goal": "base task"},
                    {"goal": "review task", "tier": "review"},
                ],
                parent_agent=parent,
            )
        # First task: default_tier -> uses top-level pool
        # (base-model not in base pool -> falls back to base-pool-model)
        self.assertEqual(captured[0]["model"], "base-pool-model")
        # Second task: review tier -> uses review pool
        # (review-model not in review pool -> falls back to review-pool-model)
        self.assertEqual(captured[1]["model"], "review-pool-model")


class TestMalformedTierWithPool(unittest.TestCase):
    """Malformed tier entry should not crash, even if it has a pool key."""

    def test_non_dict_tier_entry_ignored(self):
        cfg = {
            "model": "x",
            "tiers": {"review": "not-a-dict"},
        }
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["model"], "x")
        self.assertNotIn("tiers", result)

    def test_tier_with_none_pool_falls_back(self):
        cfg = {
            "model": "x",
            "tiers": {"review": {"model": "y", "pool": None}},
        }
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["model"], "y")


class TestCacheBehavior(unittest.TestCase):

    def test_cache_hit_avoids_config_read(self):
        """Cached value is returned without calling _load_config."""
        old = dt._cached_max_concurrent
        try:
            dt._cached_max_concurrent = 7
            with patch("tools.delegate_tool._load_config") as mock_cfg:
                result = dt._get_max_concurrent_children()
                mock_cfg.assert_not_called()
            self.assertEqual(result, 7)
        finally:
            dt._cached_max_concurrent = old

    def test_cache_miss_uses_config(self):
        """First call with no cache reads config."""
        old = dt._cached_max_concurrent
        try:
            dt._cached_max_concurrent = None
            with patch("tools.delegate_tool._load_config", return_value={"max_concurrent_children": 5}):
                result = dt._get_max_concurrent_children()
            self.assertEqual(result, 5)
        finally:
            dt._cached_max_concurrent = old


class TestPoolEdgeCases(unittest.TestCase):

    def test_pool_with_only_none_entries(self):
        """Pool of None entries should not crash."""
        self.assertEqual(_validate_pool_model("anything", [None, None]), "anything")

    def test_pool_with_dict_no_model_then_valid(self):
        pool = [{"strengths": "x"}, {"model": "good"}]
        self.assertEqual(_validate_pool_model("bad", pool), "good")

    def test_pool_validates_exact_match(self):
        pool = [{"model": "a"}, {"model": "b"}, {"model": "c"}]
        self.assertEqual(_validate_pool_model("b", pool), "b")


if __name__ == "__main__":
    unittest.main()

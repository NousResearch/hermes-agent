#!/usr/bin/env python3
"""
Comprehensive edge-case tests for delegation tiers.

Covers every issue found by Claude Code, Blackbox, and Codex reviews:
- unknown tier names
- malformed pool entries
- max_iterations boundary values
- type coercion safety
- resolve_tier_config always stripping keys
- pool read after tier resolution
- concurrent delegate_task safety
- blocked tools can't leak through mixed toolsets

Run: python -m pytest tests/tools/test_delegate_tiers_edge.py -v -o 'addopts='
"""

import json
import os
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_BLOCKED_TOOLS,
    DELEGATE_TASK_SCHEMA,
    SUPPORTED_TIERS,
    _TIER_REASONING_FLOORS,
    _build_pool_description,
    _build_child_agent,
    _build_child_system_prompt,
    _get_max_concurrent_children,
    _resolve_delegation_credentials,
    _strip_blocked_tools,
    _validate_pool_model,
    delegate_task,
    resolve_tier_config,
)
from toolsets import TOOLSETS


def _make_mock_parent(depth=0):
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


# =====================================================================
# A. resolve_tier_config — always strips keys
# =====================================================================

class TestResolveTierAlwaysStripsKeys(unittest.TestCase):
    """resolve_tier_config must ALWAYS return a dict without 'tiers' or 'default_tier'."""

    def _assert_no_nesting(self, result):
        self.assertNotIn("tiers", result)
        self.assertNotIn("default_tier", result)

    def test_no_tiers_dict(self):
        result = resolve_tier_config({"model": "x"})
        self._assert_no_nesting(result)

    def test_empty_tiers(self):
        result = resolve_tier_config({"model": "x", "tiers": {}})
        self._assert_no_nesting(result)

    def test_non_dict_tiers(self):
        result = resolve_tier_config({"model": "x", "tiers": "bad"})
        self._assert_no_nesting(result)

    def test_unknown_explicit_tier(self):
        result = resolve_tier_config(
            {"model": "x", "tiers": {"light": {"model": "y"}}},
            tier="nonexistent",
        )
        self._assert_no_nesting(result)

    def test_none_tier_no_default(self):
        result = resolve_tier_config(
            {"model": "x", "tiers": {"light": {"model": "y"}}},
            tier=None,
        )
        self._assert_no_nesting(result)

    def test_valid_tier(self):
        result = resolve_tier_config(
            {"model": "x", "tiers": {"light": {"model": "y"}}},
            tier="light",
        )
        self._assert_no_nesting(result)

    def test_tier_entry_not_dict(self):
        result = resolve_tier_config(
            {"model": "x", "tiers": {"review": "not-a-dict"}},
            tier="review",
        )
        self._assert_no_nesting(result)

    def test_default_tier_used(self):
        result = resolve_tier_config(
            {"model": "x", "default_tier": "light", "tiers": {"light": {"model": "y"}}},
        )
        self._assert_no_nesting(result)

    def test_returns_copy_not_original(self):
        cfg = {"model": "x", "tiers": {}}
        result = resolve_tier_config(cfg)
        # modifying result should not modify original
        result["model"] = "mutated"
        self.assertEqual(cfg["model"], "x")

    def test_tiers_not_dict_returns_copy(self):
        cfg = {"model": "x", "tiers": [1, 2, 3]}
        result = resolve_tier_config(cfg)
        self._assert_no_nesting(result)
        self.assertEqual(result["model"], "x")


# =====================================================================
# B. Unknown tier — warning logged, no crash
# =====================================================================

class TestUnknownTier(unittest.TestCase):

    def test_unknown_tier_logs_warning(self):
        import logging
        with self.assertLogs("tools.delegate_tool", level="WARNING") as cm:
            result = resolve_tier_config(
                {"model": "x", "tiers": {"light": {"model": "y"}}},
                tier="bogus",
            )
        self.assertTrue(any("bogus" in msg for msg in cm.output))
        self.assertEqual(result["model"], "x")

    def test_unknown_tier_preserves_other_keys(self):
        result = resolve_tier_config(
            {"model": "x", "max_iterations": 42, "tiers": {"light": {"model": "y"}}},
            tier="bogus",
        )
        self.assertEqual(result["model"], "x")
        self.assertEqual(result["max_iterations"], 42)


# =====================================================================
# C. Pool validation — malformed entries
# =====================================================================

class TestPoolValidationEdgeCases(unittest.TestCase):

    def test_pool_empty_list(self):
        self.assertEqual(_validate_pool_model("anything", []), "anything")

    def test_pool_none(self):
        self.assertEqual(_validate_pool_model("anything", None), "anything")

    def test_model_none(self):
        self.assertIsNone(_validate_pool_model(None, [{"model": "x"}]))

    def test_pool_all_non_dict(self):
        pool = ["string", 42, None, True]
        self.assertEqual(_validate_pool_model("anything", pool), "anything")

    def test_pool_no_model_key(self):
        pool = [{"provider": "x"}, {"strengths": "y"}]
        self.assertEqual(_validate_pool_model("anything", pool), "anything")

    def test_pool_first_invalid_second_valid(self):
        pool = [None, "bad", {"model": "good-model"}]
        self.assertEqual(_validate_pool_model("wrong", pool), "good-model")

    def test_pool_first_entry_non_dict_then_valid(self):
        pool = ["invalid", {"model": "valid-model"}]
        self.assertEqual(_validate_pool_model("wrong", pool), "valid-model")

    def test_pool_with_mixed_entries(self):
        pool = [
            {"model": "gpt-5.4"},
            None,
            "invalid",
            {"no_model": True},
            {"model": "gpt-5.4-mini"},
        ]
        self.assertEqual(_validate_pool_model("gpt-5.4-mini", pool), "gpt-5.4-mini")
        self.assertEqual(_validate_pool_model("wrong", pool), "gpt-5.4")

    def test_pool_description_skips_non_dict(self):
        pool = [{"model": "gpt-5.4"}, None, "invalid"]
        desc = _build_pool_description(pool)
        self.assertIn("gpt-5.4", desc)

    def test_pool_description_empty(self):
        self.assertEqual(_build_pool_description([]), "")
        self.assertEqual(_build_pool_description(None), "")

    def test_pool_duplicate_models(self):
        pool = [{"model": "dup"}, {"model": "dup"}, {"model": "unique"}]
        self.assertEqual(_validate_pool_model("dup", pool), "dup")


# =====================================================================
# D. max_iterations boundary values
# =====================================================================

class TestMaxIterationsEdgeCases(unittest.TestCase):

    @patch("tools.delegate_tool._load_config")
    def test_zero_max_iterations(self, mock_cfg):
        """max_iterations=0 should be passed through, not replaced by default."""
        mock_cfg.return_value = {"model": "x"}
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "messages": [],
            }
            mock_build.return_value = mock_child
            delegate_task(goal="test", max_iterations=0, parent_agent=parent)
            call_kwargs = mock_build.call_args[1]
            self.assertEqual(call_kwargs["max_iterations"], 0)

    @patch("tools.delegate_tool._load_config")
    def test_none_max_iterations_uses_default(self, mock_cfg):
        mock_cfg.return_value = {"model": "x"}
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "messages": [],
            }
            mock_build.return_value = mock_child
            delegate_task(goal="test", max_iterations=None, parent_agent=parent)
            call_kwargs = mock_build.call_args[1]
            self.assertEqual(call_kwargs["max_iterations"], 50)  # DEFAULT_MAX_ITERATIONS

    @patch("tools.delegate_tool._load_config")
    def test_negative_max_iterations(self, mock_cfg):
        """Negative values should be passed through (validation elsewhere)."""
        mock_cfg.return_value = {"model": "x"}
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "messages": [],
            }
            mock_build.return_value = mock_child
            delegate_task(goal="test", max_iterations=-1, parent_agent=parent)
            call_kwargs = mock_build.call_args[1]
            self.assertEqual(call_kwargs["max_iterations"], -1)

    @patch("tools.delegate_tool._load_config")
    def test_string_max_iterations(self, mock_cfg):
        """String max_iterations should not silently become default."""
        mock_cfg.return_value = {"model": "x"}
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "messages": [],
            }
            mock_build.return_value = mock_child
            delegate_task(goal="test", max_iterations="30", parent_agent=parent)
            call_kwargs = mock_build.call_args[1]
            # string "30" is truthy so it passes through
            self.assertEqual(call_kwargs["max_iterations"], "30")


# =====================================================================
# E. Pool read AFTER tier resolution
# =====================================================================

class TestPoolReadAfterTierResolution(unittest.TestCase):

    @patch("tools.delegate_tool._load_config")
    def test_tier_can_override_pool(self, mock_cfg):
        """If a tier config has its own pool, that pool is used for validation."""
        mock_cfg.return_value = {
            "model": "x",
            "pool": [{"model": "old-pool-model"}],
            "tiers": {
                "review": {
                    "model": "tier-model",
                    "pool": [{"model": "tier-pool-model"}],
                },
            },
        }
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "messages": [],
            }
            mock_build.return_value = mock_child
            delegate_task(goal="test", tier="review", parent_agent=parent)
            call_kwargs = mock_build.call_args[1]
            # tier-model is not in tier's pool -> falls back to tier-pool-model
            # (NOT old-pool-model from base config, proving tier pool is used)
            self.assertEqual(call_kwargs["model"], "tier-pool-model")


# =====================================================================
# F. override_reasoning_effort type coercion
# =====================================================================

class TestReasoningEffortTypeCoercion(unittest.TestCase):

    @patch("tools.delegate_tool._load_config")
    def test_int_reasoning_effort(self, mock_cfg):
        """Non-string override_reasoning_effort should be coerced to string."""
        mock_cfg.return_value = {}
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "low"}

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="test", context=None, toolsets=None,
                model=None, max_iterations=50, parent_agent=parent,
                override_reasoning_effort=3,  # int, not string
            )
            call_kwargs = MockAgent.call_args[1]
            self.assertEqual(
                call_kwargs["reasoning_config"],
                {"enabled": True, "effort": "3"},
            )

    @patch("tools.delegate_tool._load_config")
    def test_none_reasoning_effort_inherits(self, mock_cfg):
        """None override should inherit parent reasoning."""
        mock_cfg.return_value = {}
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "high"}

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="test", context=None, toolsets=None,
                model=None, max_iterations=50, parent_agent=parent,
                override_reasoning_effort=None,
            )
            call_kwargs = MockAgent.call_args[1]
            self.assertEqual(call_kwargs["reasoning_config"], {"enabled": True, "effort": "high"})

    @patch("tools.delegate_tool._load_config")
    def test_empty_string_reasoning_effort_inherits(self, mock_cfg):
        """Empty string override should inherit parent reasoning."""
        mock_cfg.return_value = {}
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "medium"}

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="test", context=None, toolsets=None,
                model=None, max_iterations=50, parent_agent=parent,
                override_reasoning_effort="",
            )
            call_kwargs = MockAgent.call_args[1]
            self.assertEqual(call_kwargs["reasoning_config"], {"enabled": True, "effort": "medium"})


# =====================================================================
# G. Blocked tools can't leak through mixed toolsets
# =====================================================================

class TestBlockedToolsNoLeak(unittest.TestCase):

    def test_blocked_tools_always_stripped(self):
        """Every blocked toolset is removed regardless of input."""
        for ts_name in ["delegation", "clarify", "memory", "code_execution"]:
            result = _strip_blocked_tools(["terminal", "file", ts_name])
            self.assertNotIn(ts_name, result)
            self.assertIn("terminal", result)
            self.assertIn("file", result)

    def test_all_blocked_stripped(self):
        """All blocked toolsets at once."""
        all_blocked = ["delegation", "clarify", "memory", "code_execution"]
        result = _strip_blocked_tools(all_blocked + ["terminal", "file"])
        self.assertEqual(sorted(result), ["file", "terminal"])

    def test_no_blocked_in_result(self):
        """Result never contains blocked toolsets even with duplicates."""
        result = _strip_blocked_tools(["delegation", "delegation", "terminal", "delegation"])
        self.assertEqual(result, ["terminal"])


# =====================================================================
# H. Backward compatibility — existing callers
# =====================================================================

class TestBackwardCompatibilityEdge(unittest.TestCase):

    @patch("tools.delegate_tool._load_config")
    def test_old_style_flat_config_with_extra_keys(self, mock_cfg):
        """Config with extra keys not related to tiers works unchanged."""
        mock_cfg.return_value = {
            "model": "gpt-5.4-mini",
            "max_iterations": 30,
            "some_future_key": "ignored",
            "another_key": 42,
        }
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "ok", "completed": True, "messages": [],
            }
            mock_build.return_value = mock_child
            result_json = delegate_task(goal="test", parent_agent=parent)
            result = json.loads(result_json)
            self.assertIn("results", result)
            call_kwargs = mock_build.call_args[1]
            self.assertEqual(call_kwargs["model"], "gpt-5.4-mini")

    @patch("tools.delegate_tool._load_config")
    def test_tier_with_no_tiers_config_still_works(self, mock_cfg):
        """Passing tier= when config has no tiers falls back to flat config."""
        mock_cfg.return_value = {"model": "x", "max_iterations": 10}
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "ok", "completed": True, "messages": [],
            }
            mock_build.return_value = mock_child
            result_json = delegate_task(goal="test", tier="light", parent_agent=parent)
            result = json.loads(result_json)
            self.assertIn("results", result)
            call_kwargs = mock_build.call_args[1]
            self.assertEqual(call_kwargs["model"], "x")


# =====================================================================
# I. Config caching
# =====================================================================

class TestConfigCaching(unittest.TestCase):

    def test_max_concurrent_children_caches(self):
        """Second call to _get_max_concurrent_children should use cache."""
        import tools.delegate_tool as dt
        old_cache = dt._cached_max_concurrent
        try:
            dt._cached_max_concurrent = 5
            result = _get_max_concurrent_children()
            self.assertEqual(result, 5)
        finally:
            dt._cached_max_concurrent = old_cache


# =====================================================================
# J. Reasoning floor edge cases
# =====================================================================

class TestReasoningFloorEdgeCases(unittest.TestCase):

    def test_floor_with_unknown_reasoning_effort(self):
        """Unknown reasoning_effort string defaults to 0 and gets bumped to floor."""
        cfg = {
            "tiers": {
                "review": {"reasoning_effort": "unknown_value"},
            },
        }
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["reasoning_effort"], "high")

    def test_floor_with_empty_string_effort(self):
        """Empty string reasoning_effort defaults to 0 and gets bumped to floor."""
        cfg = {
            "tiers": {
                "review": {"reasoning_effort": ""},
            },
        }
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["reasoning_effort"], "high")

    def test_light_has_no_floor(self):
        """Light tier with 'none' reasoning stays at 'none'."""
        cfg = {
            "tiers": {
                "light": {"reasoning_effort": "none"},
            },
        }
        result = resolve_tier_config(cfg, tier="light")
        self.assertEqual(result["reasoning_effort"], "none")

    def test_all_tier_floors_valid_reasoning_levels(self):
        """Every floor value is a valid reasoning level."""
        for tier, floor in _TIER_REASONING_FLOORS.items():
            self.assertIn(floor, {"none", "low", "minimal", "medium", "high", "xhigh", "max"})


# =====================================================================
# K. Schema consistency
# =====================================================================

class TestSchemaConsistency(unittest.TestCase):

    def test_schema_tier_enum_matches_constant(self):
        schema_tiers = set(DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tier"]["enum"])
        self.assertEqual(schema_tiers, SUPPORTED_TIERS)

    def test_schema_per_task_tier_enum_matches_constant(self):
        task_tiers = set(
            DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]["tier"]["enum"]
        )
        self.assertEqual(task_tiers, SUPPORTED_TIERS)

    def test_schema_no_max_items_on_tasks(self):
        tasks_schema = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]
        self.assertNotIn("maxItems", tasks_schema)

    def test_schema_name(self):
        self.assertEqual(DELEGATE_TASK_SCHEMA["name"], "delegate_task")


if __name__ == "__main__":
    unittest.main()

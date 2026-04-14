#!/usr/bin/env python3
"""
Extensive tests for delegation tier profiles, pool validation,
reasoning effort override, and comparative tier behavior.

Run with:  python -m pytest tests/tools/test_delegate_tiers.py -v
"""

import json
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from tools.delegate_tool import (
    DELEGATE_BLOCKED_TOOLS,
    DELEGATE_TASK_SCHEMA,
    SUPPORTED_TIERS,
    _TIER_REASONING_FLOORS,
    _REASONING_ORDER,
    _build_child_agent,
    _build_child_system_prompt,
    _build_pool_description,
    _get_max_concurrent_children,
    _resolve_delegation_credentials,
    _run_single_child,
    _strip_blocked_tools,
    _validate_pool_model,
    check_delegate_requirements,
    delegate_task,
    resolve_tier_config,
)


def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields delegate_task expects."""
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


# =========================================================================
# 1. TIER RESOLUTION TESTS
# =========================================================================

class TestTierResolution(unittest.TestCase):
    """Test resolve_tier_config() with various config shapes."""

    def test_flat_config_no_tiers(self):
        """Config without tiers dict returns unchanged (except stripped tier keys)."""
        cfg = {"model": "gpt-5.4-mini", "reasoning_effort": "low"}
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["model"], "gpt-5.4-mini")
        self.assertEqual(result["reasoning_effort"], "low")
        self.assertNotIn("tiers", result)
        self.assertNotIn("default_tier", result)

    def test_flat_config_empty_tiers(self):
        """Config with empty tiers dict returns flat (tiers stripped)."""
        cfg = {"model": "gpt-5.4-mini", "tiers": {}}
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["model"], "gpt-5.4-mini")
        self.assertNotIn("tiers", result)
        self.assertNotIn("default_tier", result)

    def test_explicit_tier_overrides(self):
        """Explicit tier overrides model, reasoning, max_iterations."""
        cfg = {
            "model": "gpt-5.4-mini",
            "reasoning_effort": "low",
            "max_iterations": 25,
            "tiers": {
                "heavy": {
                    "model": "gpt-5.4",
                    "reasoning_effort": "medium",
                    "max_iterations": 50,
                }
            },
        }
        result = resolve_tier_config(cfg, tier="heavy")
        self.assertEqual(result["model"], "gpt-5.4")
        self.assertEqual(result["reasoning_effort"], "medium")
        self.assertEqual(result["max_iterations"], 50)
        self.assertNotIn("tiers", result)
        self.assertNotIn("default_tier", result)

    def test_default_tier_used_when_no_explicit(self):
        """default_tier is used when no explicit tier arg."""
        cfg = {
            "model": "gpt-5.4-mini",
            "reasoning_effort": "low",
            "default_tier": "review",
            "tiers": {
                "review": {
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                }
            },
        }
        result = resolve_tier_config(cfg)
        self.assertEqual(result["model"], "gpt-5.4")
        self.assertEqual(result["reasoning_effort"], "high")

    def test_explicit_tier_overrides_default_tier(self):
        """Explicit tier takes priority over default_tier."""
        cfg = {
            "model": "gpt-5.4-mini",
            "default_tier": "light",
            "tiers": {
                "light": {"model": "gpt-5.4-mini", "reasoning_effort": "low"},
                "review": {"model": "gpt-5.4", "reasoning_effort": "high"},
            },
        }
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["model"], "gpt-5.4")
        self.assertEqual(result["reasoning_effort"], "high")

    def test_unknown_tier_returns_flat(self):
        """Unknown tier name falls back to flat config."""
        cfg = {
            "model": "gpt-5.4-mini",
            "tiers": {"light": {"model": "gpt-5.4-mini"}},
        }
        result = resolve_tier_config(cfg, tier="nonexistent")
        self.assertEqual(result["model"], "gpt-5.4-mini")

    def test_none_tier_returns_flat(self):
        """None tier with no default_tier returns flat (tiers stripped)."""
        cfg = {
            "model": "gpt-5.4-mini",
            "tiers": {"light": {"model": "x"}},
        }
        result = resolve_tier_config(cfg, tier=None)
        self.assertEqual(result["model"], "gpt-5.4-mini")
        self.assertNotIn("tiers", result)
        self.assertNotIn("default_tier", result)

    def test_strips_tier_keys_from_result(self):
        """Result never contains 'tiers' or 'default_tier' keys."""
        cfg = {
            "model": "gpt-5.4-mini",
            "default_tier": "light",
            "tiers": {
                "light": {"model": "gpt-5.4-mini"},
                "heavy": {"model": "gpt-5.4"},
            },
        }
        result = resolve_tier_config(cfg, tier="heavy")
        self.assertNotIn("tiers", result)
        self.assertNotIn("default_tier", result)

    def test_tier_preserves_base_keys(self):
        """Tier merge preserves base config keys not overridden."""
        cfg = {
            "model": "gpt-5.4-mini",
            "max_iterations": 50,
            "reasoning_effort": "low",
            "tiers": {
                "heavy": {"model": "gpt-5.4"},  # heavy has no reasoning_effort override
            },
        }
        result = resolve_tier_config(cfg, tier="heavy")
        self.assertEqual(result["model"], "gpt-5.4")
        self.assertEqual(result["max_iterations"], 50)  # preserved
        # heavy has floor "medium" which bumps "low" -> "medium"
        self.assertEqual(result["reasoning_effort"], "medium")

    def test_all_five_tiers_resolve(self):
        """All 5 supported tiers resolve correctly from a full config."""
        cfg = {
            "model": "gpt-5.4-mini",
            "reasoning_effort": "low",
            "max_iterations": 25,
            "default_tier": "heavy",
            "tiers": {
                "light": {"model": "gpt-5.4-mini", "reasoning_effort": "low", "max_iterations": 25},
                "heavy": {"model": "gpt-5.4", "reasoning_effort": "medium", "max_iterations": 50},
                "review": {"model": "gpt-5.4", "reasoning_effort": "xhigh", "max_iterations": 60},
                "planning": {"model": "xiaomi/mimo-v2-pro", "reasoning_effort": "high", "max_iterations": 60},
                "research": {"model": "gpt-5.4", "reasoning_effort": "high", "max_iterations": 60},
            },
        }
        for tier_name in SUPPORTED_TIERS:
            result = resolve_tier_config(cfg, tier=tier_name)
            self.assertIn("model", result)
            self.assertNotIn("tiers", result)

    def test_provider_override_in_tier(self):
        """Tier can override provider."""
        cfg = {
            "model": "gpt-5.4-mini",
            "provider": "openai-codex",
            "tiers": {
                "planning": {
                    "model": "xiaomi/mimo-v2-pro",
                    "provider": "nous",
                }
            },
        }
        result = resolve_tier_config(cfg, tier="planning")
        self.assertEqual(result["provider"], "nous")
        self.assertEqual(result["model"], "xiaomi/mimo-v2-pro")


# =========================================================================
# 2. REASONING FLOOR GUARDRAILS
# =========================================================================

class TestReasoningFloors(unittest.TestCase):
    """Test that tier reasoning floors prevent degradation."""

    def test_review_floor_enforced(self):
        """Review tier with low effort gets bumped to 'high'."""
        cfg = {
            "model": "gpt-5.4",
            "tiers": {
                "review": {"model": "gpt-5.4", "reasoning_effort": "low"},
            },
        }
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["reasoning_effort"], "high")

    def test_planning_floor_enforced(self):
        """Planning tier with medium effort gets bumped to 'high'."""
        cfg = {
            "tiers": {
                "planning": {"reasoning_effort": "medium"},
            },
        }
        result = resolve_tier_config(cfg, tier="planning")
        self.assertEqual(result["reasoning_effort"], "high")

    def test_heavy_floor_enforced(self):
        """Heavy tier with low effort gets bumped to 'medium'."""
        cfg = {
            "tiers": {
                "heavy": {"reasoning_effort": "low"},
            },
        }
        result = resolve_tier_config(cfg, tier="heavy")
        self.assertEqual(result["reasoning_effort"], "medium")

    def test_research_floor_enforced(self):
        """Research tier with low effort gets bumped to 'medium'."""
        cfg = {
            "tiers": {
                "research": {"reasoning_effort": "low"},
            },
        }
        result = resolve_tier_config(cfg, tier="research")
        self.assertEqual(result["reasoning_effort"], "medium")

    def test_floor_not_lowered(self):
        """If tier already above floor, floor doesn't lower it."""
        cfg = {
            "tiers": {
                "review": {"reasoning_effort": "xhigh"},
            },
        }
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["reasoning_effort"], "xhigh")

    def test_light_no_floor(self):
        """Light tier has no floor -- stays at whatever is set."""
        cfg = {
            "tiers": {
                "light": {"reasoning_effort": "none"},
            },
        }
        result = resolve_tier_config(cfg, tier="light")
        self.assertEqual(result["reasoning_effort"], "none")

    def test_no_reasoning_set_floor_applied(self):
        """If no reasoning_effort in tier, floor still applies."""
        cfg = {
            "tiers": {
                "review": {"model": "gpt-5.4"},
            },
        }
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["reasoning_effort"], "high")

    def test_reasoning_order_completeness(self):
        """Verify REASONING_ORDER covers all expected levels."""
        expected = {"none", "low", "minimal", "medium", "high", "xhigh", "max"}
        self.assertEqual(set(_REASONING_ORDER.keys()), expected)
        # Verify ordering
        self.assertLess(_REASONING_ORDER["low"], _REASONING_ORDER["medium"])
        self.assertLess(_REASONING_ORDER["medium"], _REASONING_ORDER["high"])
        self.assertLess(_REASONING_ORDER["high"], _REASONING_ORDER["xhigh"])


# =========================================================================
# 3. POOL VALIDATION TESTS
# =========================================================================

class TestPoolValidation(unittest.TestCase):
    """Test _validate_pool_model() behavior."""

    def test_valid_model_passes(self):
        """Model in pool passes through unchanged."""
        pool = [{"model": "gpt-5.4"}, {"model": "gpt-5.4-mini"}]
        self.assertEqual(_validate_pool_model("gpt-5.4", pool), "gpt-5.4")

    def test_invalid_model_falls_back(self):
        """Model not in pool falls back to first pool entry."""
        pool = [{"model": "gpt-5.4"}, {"model": "gpt-5.4-mini"}]
        result = _validate_pool_model("nonexistent", pool)
        self.assertEqual(result, "gpt-5.4")

    def test_empty_pool_passes_through(self):
        """Empty pool means no validation -- model passes unchanged."""
        self.assertEqual(_validate_pool_model("anything", []), "anything")

    def test_none_model_passes(self):
        """None model passes through regardless of pool."""
        pool = [{"model": "gpt-5.4"}]
        self.assertIsNone(_validate_pool_model(None, pool))

    def test_pool_with_provider(self):
        """Pool entries with provider are still matched by model name."""
        pool = [
            {"model": "gpt-5.4", "provider": "openai-codex"},
            {"model": "xiaomi/mimo-v2-pro", "provider": "nous"},
        ]
        self.assertEqual(
            _validate_pool_model("xiaomi/mimo-v2-pro", pool),
            "xiaomi/mimo-v2-pro",
        )

    def test_pool_description_builder(self):
        """_build_pool_description creates readable output."""
        pool = [
            {"model": "gpt-5.4-mini", "strengths": "quick lookups"},
            {"model": "gpt-5.4", "provider": "openai-codex", "strengths": "coding, debugging"},
        ]
        desc = _build_pool_description(pool)
        self.assertIn("gpt-5.4-mini", desc)
        self.assertIn("quick lookups", desc)
        self.assertIn("openai-codex", desc)

    def test_pool_description_empty(self):
        """Empty pool returns empty description."""
        self.assertEqual(_build_pool_description([]), "")
        self.assertEqual(_build_pool_description(None), "")

    def test_pool_description_skips_invalid_entries(self):
        """Non-dict entries in pool are skipped."""
        pool = [{"model": "gpt-5.4"}, "invalid", {"model": "gpt-5.4-mini"}]
        desc = _build_pool_description(pool)
        self.assertIn("gpt-5.4", desc)
        self.assertIn("gpt-5.4-mini", desc)


# =========================================================================
# 4. REASONING EFFORT OVERRIDE IN _build_child_agent
# =========================================================================

class TestReasoningEffortOverride(unittest.TestCase):
    """Test override_reasoning_effort in _build_child_agent."""

    @patch("tools.delegate_tool._load_config")
    def test_override_sets_child_reasoning(self, mock_cfg):
        """override_reasoning_effort sets child reasoning_config."""
        mock_cfg.return_value = {}
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "low"}

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            child = _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=50,
                parent_agent=parent,
                override_reasoning_effort="high",
            )
            call_kwargs = MockAgent.call_args[1]
            self.assertEqual(
                call_kwargs["reasoning_config"],
                {"enabled": True, "effort": "high"},
            )

    @patch("tools.delegate_tool._load_config")
    def test_override_none_disables_reasoning(self, mock_cfg):
        """override_reasoning_effort='none' disables reasoning."""
        mock_cfg.return_value = {}
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "high"}

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            child = _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=50,
                parent_agent=parent,
                override_reasoning_effort="none",
            )
            call_kwargs = MockAgent.call_args[1]
            self.assertEqual(
                call_kwargs["reasoning_config"],
                {"enabled": False, "effort": "none"},
            )

    @patch("tools.delegate_tool._load_config")
    def test_no_override_inherits_parent(self, mock_cfg):
        """Without override, child inherits parent reasoning_config."""
        mock_cfg.return_value = {}
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "medium"}

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            child = _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=50,
                parent_agent=parent,
            )
            call_kwargs = MockAgent.call_args[1]
            self.assertEqual(call_kwargs["reasoning_config"], {"enabled": True, "effort": "medium"})

    @patch("tools.delegate_tool._load_config")
    def test_override_bypasses_config_effort(self, mock_cfg):
        """override_reasoning_effort takes priority over delegation config."""
        mock_cfg.return_value = {"reasoning_effort": "low"}
        parent = _make_mock_parent()
        parent.reasoning_config = None

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            child = _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=50,
                parent_agent=parent,
                override_reasoning_effort="xhigh",
            )
            call_kwargs = MockAgent.call_args[1]
            self.assertEqual(
                call_kwargs["reasoning_config"],
                {"enabled": True, "effort": "xhigh"},
            )


# =========================================================================
# 5. SCHEMA VALIDATION
# =========================================================================

class TestSchemaValidation(unittest.TestCase):
    """Test DELEGATE_TASK_SCHEMA includes tier fields."""

    def test_schema_has_tier_top_level(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("tier", props)
        self.assertEqual(props["tier"]["enum"], sorted(SUPPORTED_TIERS))

    def test_schema_has_tier_per_task(self):
        task_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        self.assertIn("tier", task_props)
        self.assertEqual(task_props["tier"]["enum"], sorted(SUPPORTED_TIERS))

    def test_schema_has_goal_context_toolsets(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("goal", props)
        self.assertIn("context", props)
        self.assertIn("toolsets", props)
        self.assertIn("tasks", props)
        self.assertIn("max_iterations", props)
        self.assertIn("acp_command", props)
        self.assertIn("acp_args", props)

    def test_schema_name(self):
        self.assertEqual(DELEGATE_TASK_SCHEMA["name"], "delegate_task")

    def test_supported_tiers_matches_schema(self):
        """SUPPORTED_TIERS matches the enum in the schema."""
        schema_tiers = set(DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tier"]["enum"])
        self.assertEqual(schema_tiers, SUPPORTED_TIERS)


# =========================================================================
# 6. INTEGRATION: delegate_task WITH TIERS (MOCKED)
# =========================================================================

class TestDelegateTaskTierIntegration(unittest.TestCase):
    """Test delegate_task() end-to-end with tier resolution (mocked LLM)."""

    @patch("tools.delegate_tool._load_config")
    def test_single_task_with_tier(self, mock_cfg):
        """Single task with tier uses resolved config."""
        mock_cfg.return_value = {
            "model": "gpt-5.4-mini",
            "reasoning_effort": "low",
            "tiers": {
                "review": {
                    "model": "gpt-5.4",
                    "reasoning_effort": "xhigh",
                    "max_iterations": 60,
                }
            },
        }
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "Review complete",
                "completed": True,
                "messages": [],
            }
            mock_build.return_value = mock_child

            result_json = delegate_task(
                goal="Review the auth module",
                tier="review",
                parent_agent=parent,
            )
            result = json.loads(result_json)

            # Verify child was built with review tier config
            call_kwargs = mock_build.call_args[1]
            self.assertEqual(call_kwargs["model"], "gpt-5.4")
            self.assertEqual(call_kwargs["override_reasoning_effort"], "xhigh")
            self.assertEqual(call_kwargs["max_iterations"], 60)

    @patch("tools.delegate_tool._load_config")
    def test_batch_per_task_tiers(self, mock_cfg):
        """Batch mode: each task can have its own tier."""
        mock_cfg.return_value = {
            "model": "gpt-5.4-mini",
            "reasoning_effort": "low",
            "tiers": {
                "light": {"model": "gpt-5.4-mini", "reasoning_effort": "low", "max_iterations": 25},
                "review": {"model": "gpt-5.4", "reasoning_effort": "xhigh", "max_iterations": 60},
            },
        }
        parent = _make_mock_parent()
        children_configs = []

        def capture_child(**kwargs):
            children_configs.append(kwargs)
            child = MagicMock()
            child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "messages": [],
            }
            return child

        with patch("tools.delegate_tool._build_child_agent", side_effect=capture_child):
            result_json = delegate_task(
                tasks=[
                    {"goal": "Quick lookup", "tier": "light"},
                    {"goal": "Deep review", "tier": "review"},
                ],
                parent_agent=parent,
            )
            result = json.loads(result_json)
            self.assertEqual(len(result["results"]), 2)

            # First child: light tier
            self.assertEqual(children_configs[0]["model"], "gpt-5.4-mini")
            self.assertEqual(children_configs[0]["override_reasoning_effort"], "low")
            self.assertEqual(children_configs[0]["max_iterations"], 25)

            # Second child: review tier (xhigh already above floor, stays xhigh)
            self.assertEqual(children_configs[1]["model"], "gpt-5.4")
            self.assertEqual(children_configs[1]["override_reasoning_effort"], "xhigh")
            self.assertEqual(children_configs[1]["max_iterations"], 60)

    @patch("tools.delegate_tool._load_config")
    def test_top_level_tier_applied_to_all(self, mock_cfg):
        """Top-level tier applies to all tasks without per-task tier."""
        mock_cfg.return_value = {
            "model": "gpt-5.4-mini",
            "tiers": {
                "heavy": {"model": "gpt-5.4", "reasoning_effort": "medium", "max_iterations": 50},
            },
        }
        parent = _make_mock_parent()
        children_configs = []

        def capture_child(**kwargs):
            children_configs.append(kwargs)
            child = MagicMock()
            child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "messages": [],
            }
            return child

        with patch("tools.delegate_tool._build_child_agent", side_effect=capture_child):
            delegate_task(
                tasks=[
                    {"goal": "Task A"},
                    {"goal": "Task B"},
                ],
                tier="heavy",
                parent_agent=parent,
            )
            for cfg in children_configs:
                self.assertEqual(cfg["model"], "gpt-5.4")

    @patch("tools.delegate_tool._load_config")
    def test_pool_validation_applied(self, mock_cfg):
        """Pool validation kicks in when pool is configured."""
        mock_cfg.return_value = {
            "model": "fake-model",
            "pool": [
                {"model": "gpt-5.4", "strengths": "coding"},
                {"model": "gpt-5.4-mini", "strengths": "quick"},
            ],
        }
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "messages": [],
            }
            mock_build.return_value = mock_child

            result_json = delegate_task(goal="test", parent_agent=parent)
            # fake-model not in pool, should fall back to gpt-5.4
            call_kwargs = mock_build.call_args[1]
            self.assertEqual(call_kwargs["model"], "gpt-5.4")

    @patch("tools.delegate_tool._load_config")
    def test_no_tiers_uses_flat_config(self, mock_cfg):
        """Without tiers, delegate_task uses flat config as before."""
        mock_cfg.return_value = {
            "model": "gpt-5.4-mini",
            "reasoning_effort": "low",
            "max_iterations": 25,
        }
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done", "completed": True, "messages": [],
            }
            mock_build.return_value = mock_child

            result_json = delegate_task(goal="test", parent_agent=parent)
            call_kwargs = mock_build.call_args[1]
            self.assertEqual(call_kwargs["model"], "gpt-5.4-mini")
            self.assertEqual(call_kwargs["max_iterations"], 25)


# =========================================================================
# 7. COMPARATIVE: LIGHT VS HEAVY VS REVIEW
# =========================================================================

class TestTierComparativeBehavior(unittest.TestCase):
    """Compare tier configs to verify expected differentiation."""

    def _make_full_config(self):
        return {
            "model": "gpt-5.4-mini",
            "provider": "openai-codex",
            "reasoning_effort": "low",
            "max_iterations": 25,
            "default_tier": "heavy",
            "tiers": {
                "light": {"model": "gpt-5.4-mini", "reasoning_effort": "low", "max_iterations": 25},
                "heavy": {"model": "gpt-5.4", "reasoning_effort": "medium", "max_iterations": 50},
                "review": {"model": "gpt-5.4", "reasoning_effort": "xhigh", "max_iterations": 60},
                "planning": {"model": "xiaomi/mimo-v2-pro", "provider": "nous", "reasoning_effort": "high", "max_iterations": 60},
                "research": {"model": "gpt-5.4", "reasoning_effort": "high", "max_iterations": 60},
            },
        }

    def test_light_is_cheapest(self):
        """Light tier should have lowest model, lowest reasoning, fewest iterations."""
        cfg = self._make_full_config()
        light = resolve_tier_config(cfg, tier="light")
        heavy = resolve_tier_config(cfg, tier="heavy")
        self.assertEqual(light["model"], "gpt-5.4-mini")
        self.assertLessEqual(
            _REASONING_ORDER.get(light["reasoning_effort"], 0),
            _REASONING_ORDER.get(heavy["reasoning_effort"], 0),
        )
        self.assertLess(light["max_iterations"], heavy["max_iterations"])

    def test_review_has_highest_reasoning(self):
        """Review tier should have highest reasoning effort."""
        cfg = self._make_full_config()
        review = resolve_tier_config(cfg, tier="review")
        self.assertEqual(_REASONING_ORDER.get(review["reasoning_effort"], 0), 4)  # xhigh

    def test_planning_uses_different_provider(self):
        """Planning tier can route to a completely different provider."""
        cfg = self._make_full_config()
        planning = resolve_tier_config(cfg, tier="planning")
        self.assertEqual(planning["model"], "xiaomi/mimo-v2-pro")
        self.assertEqual(planning["provider"], "nous")

    def test_heavy_is_default(self):
        """Without explicit tier, default_tier='heavy' is used."""
        cfg = self._make_full_config()
        result = resolve_tier_config(cfg)
        self.assertEqual(result["model"], "gpt-5.4")
        self.assertEqual(result["reasoning_effort"], "medium")

    def test_cost_order_light_lt_heavy_lt_review(self):
        """Iterate cost proxy: light < heavy < review."""
        cfg = self._make_full_config()
        tiers = {}
        for t in SUPPORTED_TIERS:
            resolved = resolve_tier_config(cfg, tier=t)
            # Cost proxy: reasoning_order * max_iterations
            cost = _REASONING_ORDER.get(resolved.get("reasoning_effort", "none"), 0) * resolved.get("max_iterations", 0)
            tiers[t] = cost
        self.assertLess(tiers["light"], tiers["heavy"])
        self.assertLess(tiers["heavy"], tiers["review"])


# =========================================================================
# 8. BACKWARD COMPATIBILITY
# =========================================================================

class TestBackwardCompatibility(unittest.TestCase):
    """Ensure existing configs without tiers still work."""

    @patch("tools.delegate_tool._load_config")
    def test_empty_config_delegates(self, mock_cfg):
        """Empty config still produces a valid delegation."""
        mock_cfg.return_value = {}
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

    @patch("tools.delegate_tool._load_config")
    def test_flat_config_no_tiers(self, mock_cfg):
        """Flat config without tiers dict works exactly as before."""
        mock_cfg.return_value = {
            "model": "gpt-5.4-mini",
            "max_iterations": 30,
        }
        parent = _make_mock_parent()

        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "ok", "completed": True, "messages": [],
            }
            mock_build.return_value = mock_child

            result_json = delegate_task(goal="test", parent_agent=parent)
            call_kwargs = mock_build.call_args[1]
            self.assertEqual(call_kwargs["model"], "gpt-5.4-mini")
            self.assertEqual(call_kwargs["max_iterations"], 30)

    def test_strip_blocked_tools_unchanged(self):
        """_strip_blocked_tools behavior unchanged."""
        result = _strip_blocked_tools(["terminal", "file", "delegation", "clarify", "memory", "code_execution"])
        self.assertEqual(sorted(result), ["file", "terminal"])

    def test_build_child_prompt_unchanged(self):
        """System prompt builder still works as before."""
        prompt = _build_child_system_prompt("Fix the tests", "Error in test_foo.py")
        self.assertIn("Fix the tests", prompt)
        self.assertIn("Error in test_foo.py", prompt)


# =========================================================================
# 9. EDGE CASES
# =========================================================================

class TestTierEdgeCases(unittest.TestCase):
    """Edge cases in tier/pool handling."""

    def test_tier_case_insensitive(self):
        """Tier names are case-insensitive."""
        cfg = {
            "model": "x",
            "tiers": {"review": {"model": "review-model"}},
        }
        result = resolve_tier_config(cfg, tier="REVIEW")
        self.assertEqual(result["model"], "review-model")

    def test_tier_with_whitespace(self):
        """Tier names with whitespace are trimmed."""
        cfg = {
            "model": "x",
            "tiers": {"light": {"model": "light-model"}},
        }
        result = resolve_tier_config(cfg, tier="  light  ")
        self.assertEqual(result["model"], "light-model")

    def test_empty_string_tier_treated_as_none(self):
        """Empty string tier falls back to default_tier or flat."""
        cfg = {
            "model": "x",
            "default_tier": "light",
            "tiers": {"light": {"model": "light-model"}},
        }
        result = resolve_tier_config(cfg, tier="")
        self.assertEqual(result["model"], "light-model")

    def test_tier_entry_not_dict(self):
        """Non-dict tier entry falls back to flat (tiers stripped)."""
        cfg = {
            "model": "x",
            "tiers": {"review": "not-a-dict"},
        }
        result = resolve_tier_config(cfg, tier="review")
        self.assertEqual(result["model"], "x")
        self.assertNotIn("tiers", result)
        self.assertNotIn("default_tier", result)

    def test_pool_with_malformed_entries(self):
        """Pool with malformed entries doesn't crash validation."""
        pool = [{"model": "gpt-5.4"}, None, "invalid", {"no_model": True}]
        # Should not raise
        result = _validate_pool_model("gpt-5.4", pool)
        self.assertEqual(result, "gpt-5.4")

    def test_all_supported_tiers_have_floors_defined(self):
        """Verify tier floor definitions are valid reasoning levels."""
        for tier, floor in _TIER_REASONING_FLOORS.items():
            self.assertIn(tier, SUPPORTED_TIERS)
            self.assertIn(floor, _REASONING_ORDER)


if __name__ == "__main__":
    unittest.main()

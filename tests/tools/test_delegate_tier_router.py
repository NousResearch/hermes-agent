import json
import unittest
from unittest.mock import MagicMock, patch

import tools.delegate_tool as dt
from tools.delegate_tool import (
    SUPPORTED_TIERS,
    _infer_delegate_tier,
    _infer_delegate_tier_llm,
    _resolve_effective_tier,
    DELEGATE_TASK_SCHEMA,
    delegate_task,
)


class TestDelegateTierRouter(unittest.TestCase):
    def test_heuristic_review_signals(self):
        self.assertEqual(_infer_delegate_tier("please review this code", "", [], {}), "review")
        self.assertEqual(_infer_delegate_tier("audit the patch", "", [], {}), "review")
        self.assertEqual(_infer_delegate_tier("find bugs in this diff", "", [], {}), "review")

    def test_heuristic_planning_signals(self):
        self.assertEqual(_infer_delegate_tier("plan the architecture", "", [], {}), "planning")
        self.assertEqual(_infer_delegate_tier("how should we approach this?", "", [], {}), "planning")

    def test_heuristic_research_signals(self):
        self.assertEqual(_infer_delegate_tier("research this topic", "", [], {}), "research")
        self.assertEqual(_infer_delegate_tier("compare options for a database", "", [], {}), "research")

    def test_heuristic_light_signals(self):
        self.assertEqual(_infer_delegate_tier("count the lines", "", [], {}), "light")
        self.assertEqual(_infer_delegate_tier("list the files", "", [], {}), "light")
        self.assertEqual(_infer_delegate_tier("how many items are there", "", [], {}), "light")

    def test_heuristic_no_signal_returns_none(self):
        """When no keyword signals match, heuristic returns None (for LLM fallback or default_tier)."""
        self.assertIsNone(_infer_delegate_tier("implement the feature", "", [], {}))

    def test_heuristic_priority_review_wins(self):
        self.assertEqual(_infer_delegate_tier("review this plan", "", [], {}), "review")

    def test_heuristic_empty_goal(self):
        self.assertIsNone(_infer_delegate_tier("", "", [], {}))
        self.assertIsNone(_infer_delegate_tier("short", "", [], {}))

    def test_heuristic_case_insensitive(self):
        self.assertEqual(_infer_delegate_tier("REVIEW this code", "", [], {}), "review")

    def test_resolve_effective_tier_explicit_wins(self):
        with patch.object(dt, "_infer_delegate_tier", return_value="review") as heur, patch.object(dt, "_infer_delegate_tier_llm", return_value="planning") as llm:
            self.assertEqual(_resolve_effective_tier("heavy", "goal", "", [], {"auto_tier_selection": True}), "heavy")
            heur.assert_not_called()
            llm.assert_not_called()

    def test_resolve_effective_tier_auto_triggers_router(self):
        with patch.object(dt, "_infer_delegate_tier", return_value="review") as heur:
            self.assertEqual(_resolve_effective_tier("auto", "goal", "", [], {"auto_tier_selection": True}), "review")
            heur.assert_called_once()

    def test_resolve_effective_tier_none_with_flag(self):
        with patch.object(dt, "_infer_delegate_tier", return_value="planning") as heur:
            self.assertEqual(_resolve_effective_tier(None, "goal", "", [], {"auto_tier_selection": True}), "planning")
            heur.assert_called_once()

    def test_resolve_effective_tier_none_without_flag(self):
        with patch.object(dt, "_infer_delegate_tier") as heur:
            self.assertIsNone(_resolve_effective_tier(None, "goal", "", [], {"auto_tier_selection": False}))
            heur.assert_not_called()

    def test_resolve_effective_tier_fallback_chain(self):
        with patch.object(dt, "_infer_delegate_tier", return_value=None), patch.object(dt, "_infer_delegate_tier_llm", return_value=None):
            self.assertIsNone(_resolve_effective_tier("auto", "goal", "", [], {"auto_tier_selection": True, "auto_tier_strategy": "hybrid"}))

    def test_batch_per_task_routing(self):
        parent = MagicMock()
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = MagicMock()
        parent.platform = "cli"
        parent.provider = "openrouter"
        parent.api_mode = "chat_completions"
        parent.model = "anthropic/claude-sonnet-4"
        parent.base_url = "https://openrouter.ai/api/v1"
        parent.api_key = "x"
        parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
        parent._session_db = None
        parent.tool_progress_callback = None
        parent.thinking_callback = None

        raw_cfg = {"auto_tier_selection": True, "tiers": {"review": {"max_iterations": 1}}}
        with patch.object(dt, "_load_config", return_value=raw_cfg), \
             patch.object(dt, "_resolve_effective_tier", side_effect=[None, "review", "heavy", None, None]) as resolver, \
             patch.object(dt, "resolve_tier_config", side_effect=lambda cfg, tier=None: {"tier": tier} if tier else {}) as rtc, \
             patch.object(dt, "_resolve_delegation_credentials", return_value={"model": "m", "provider": "p", "base_url": None, "api_key": None, "api_mode": None}), \
             patch.object(dt, "_build_child_agent", side_effect=[MagicMock(), MagicMock(), MagicMock()]), \
             patch.object(dt, "_run_single_child", return_value={"task_index": 0, "status": "completed", "summary": "ok", "api_calls": 0, "duration_seconds": 0}), \
             patch.object(dt, "_get_max_concurrent_children", return_value=3):
            out = json.loads(delegate_task(tasks=[{"goal": "a"}, {"goal": "b"}, {"goal": "c"}], parent_agent=parent))
            self.assertEqual(len(out["results"]), 3)
            self.assertEqual(resolver.call_count, 5)
            self.assertEqual(rtc.call_count, 4)

    def test_auto_not_passed_to_resolve_tier_config(self):
        raw_cfg = {"auto_tier_selection": True}
        with patch.object(dt, "_load_config", return_value=raw_cfg), \
             patch.object(dt, "_resolve_effective_tier", return_value="review") as rte, \
             patch.object(dt, "resolve_tier_config") as rtc, \
             patch.object(dt, "_resolve_delegation_credentials", return_value={"model": "m", "provider": "p", "base_url": None, "api_key": None, "api_mode": None}), \
             patch.object(dt, "_build_child_agent", return_value=MagicMock()), \
             patch.object(dt, "_run_single_child", return_value={"task_index": 0, "status": "completed", "summary": "ok", "api_calls": 0, "duration_seconds": 0}), \
             patch.object(dt, "_get_max_concurrent_children", return_value=3):
            delegate_task(goal="g", parent_agent=MagicMock(_delegate_depth=0, _active_children=[], _active_children_lock=MagicMock(), platform="cli", provider="openrouter", api_mode="chat_completions", model="m", base_url="u", api_key="k", providers_allowed=None, providers_ignored=None, providers_order=None, provider_sort=None, _session_db=None, tool_progress_callback=None, thinking_callback=None))
            self.assertGreaterEqual(rtc.call_count, 1)
            for call in rtc.call_args_list:
                self.assertNotEqual(call.kwargs.get("tier"), "auto")
            self.assertEqual(rte.call_count, 2)

    def test_supported_tiers_contains_auto(self):
        self.assertIn("auto", SUPPORTED_TIERS)

    def test_schema_enum_contains_auto(self):
        enum_values = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tier"]["enum"]
        self.assertIn("auto", enum_values)

    def test_llm_router_valid_response(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"tier":"review","confidence":0.92,"rationale":"x"}'))])
        with patch("tools.delegate_tool.OpenAI", return_value=mock_client):
            self.assertEqual(_infer_delegate_tier_llm("goal", "", [], {"model": "m", "base_url": "u"}), "review")

    def test_llm_router_low_confidence(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"tier":"review","confidence":0.5,"rationale":"x"}'))])
        with patch("tools.delegate_tool.OpenAI", return_value=mock_client):
            self.assertIsNone(_infer_delegate_tier_llm("goal", "", [], {"model": "m", "base_url": "u"}))

    def test_llm_router_invalid_json(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='not json'))])
        with patch("tools.delegate_tool.OpenAI", return_value=mock_client):
            self.assertIsNone(_infer_delegate_tier_llm("goal", "", [], {"model": "m", "base_url": "u"}))

    def test_llm_router_timeout(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("timeout")
        with patch("tools.delegate_tool.OpenAI", return_value=mock_client):
            self.assertIsNone(_infer_delegate_tier_llm("goal", "", [], {"model": "m", "base_url": "u"}))

    def test_llm_router_invalid_tier(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"tier":"bogus","confidence":0.99,"rationale":"x"}'))])
        with patch("tools.delegate_tool.OpenAI", return_value=mock_client):
            self.assertIsNone(_infer_delegate_tier_llm("goal", "", [], {"model": "m", "base_url": "u"}))

    def test_config_disabled_skips_router(self):
        with patch.object(dt, "_infer_delegate_tier") as heur, patch.object(dt, "_infer_delegate_tier_llm") as llm:
            self.assertIsNone(_resolve_effective_tier(None, "goal", "", [], {"auto_tier_selection": False}))
            heur.assert_not_called()
            llm.assert_not_called()

    def test_config_strategy_heuristic_only(self):
        with patch.object(dt, "_infer_delegate_tier", return_value="review") as heur, patch.object(dt, "_infer_delegate_tier_llm") as llm:
            self.assertEqual(_resolve_effective_tier("auto", "goal", "", [], {"auto_tier_selection": True, "auto_tier_strategy": "heuristic"}), "review")
            llm.assert_not_called()

    def test_config_strategy_llm_only(self):
        with patch.object(dt, "_infer_delegate_tier") as heur, patch.object(dt, "_infer_delegate_tier_llm", return_value="planning") as llm:
            self.assertEqual(_resolve_effective_tier("auto", "goal", "", [], {"auto_tier_selection": True, "auto_tier_strategy": "llm"}), "planning")
            heur.assert_not_called()


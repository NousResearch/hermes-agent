#!/usr/bin/env python3
"""Per-delegation cost in the serialized result entry.

Each entry in the results array returned to the parent model should
carry the child's spend (cost_usd + cost_status) so the model can see
what each delegation cost — previously the cost was only folded into
the parent session total and stripped from the entry.

The parent rollup (session_estimated_cost_usd fold via _child_cost_usd)
must stay unchanged, and the internal _child_cost_usd field must still
be stripped before serialization.

Inspired by: Perplexity Agent API result shape (idea-level)
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import delegate_task


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
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
    parent.session_estimated_cost_usd = 0.0
    parent.session_cost_source = "none"
    parent.session_cost_status = "unknown"
    return parent


def _make_mock_child(cost=0.1234567, cost_status="estimated"):
    child = MagicMock()
    child.run_conversation.return_value = {
        "final_response": "done",
        "completed": True,
        "api_calls": 2,
        "messages": [],
    }
    child.session_prompt_tokens = 100
    child.session_completion_tokens = 50
    child.session_estimated_cost_usd = cost
    child.session_cost_status = cost_status
    child.model = "anthropic/claude-sonnet-4"
    child.session_id = "child-session"
    return child


class TestCostInResultEntry(unittest.TestCase):
    def _run(self, child):
        parent = _make_mock_parent()
        with patch("run_agent.AIAgent", return_value=child):
            result = json.loads(
                delegate_task(goal="Test per-delegation cost", parent_agent=parent)
            )
        return parent, result

    def test_entry_carries_cost_usd_and_status(self):
        child = _make_mock_child(cost=0.1234567, cost_status="estimated")
        _, result = self._run(child)
        entry = result["results"][0]
        self.assertIn("cost_usd", entry)
        self.assertAlmostEqual(entry["cost_usd"], 0.123457, places=6)
        self.assertEqual(entry["cost_status"], "estimated")

    def test_reported_status_passes_through(self):
        child = _make_mock_child(cost=0.5, cost_status="reported")
        _, result = self._run(child)
        entry = result["results"][0]
        self.assertEqual(entry["cost_status"], "reported")

    def test_zero_cost_child_has_zero_cost_entry(self):
        child = _make_mock_child(cost=0.0, cost_status="unknown")
        _, result = self._run(child)
        entry = result["results"][0]
        self.assertEqual(entry["cost_usd"], 0.0)

    def test_internal_child_cost_field_still_stripped(self):
        child = _make_mock_child(cost=0.25)
        _, result = self._run(child)
        for entry in result["results"]:
            self.assertNotIn("_child_cost_usd", entry)
            self.assertNotIn("_child_role", entry)

    def test_parent_rollup_unchanged(self):
        child = _make_mock_child(cost=0.25)
        parent, _ = self._run(child)
        self.assertAlmostEqual(parent.session_estimated_cost_usd, 0.25, places=6)
        self.assertEqual(parent.session_cost_source, "subagent")
        self.assertEqual(parent.session_cost_status, "estimated")

    def test_non_numeric_child_cost_degrades_to_zero(self):
        child = _make_mock_child(cost=0.0)
        child.session_estimated_cost_usd = "not-a-number"
        _, result = self._run(child)
        entry = result["results"][0]
        self.assertEqual(entry["cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()

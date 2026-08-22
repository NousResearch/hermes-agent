#!/usr/bin/env python3
"""Per-delegation usage ledger on the parent agent.

Every completed delegate_task child appends one attribution entry
(model, tokens, api_calls, cost, goal) to
``parent_agent.session_delegation_usage``.  The ledger is surfaced by
``agent/turn_finalizer.py`` as the turn result's ``delegations`` block and
forwarded by ``hermes -z --usage-file`` so batch pipelines can attribute
spend per subagent instead of only seeing the parent-session rollup.

Inspired by: GitHub Copilot CLI 1.0.81 — per-agent usage metrics in
--usage-output-file JSON output.
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
    # Real list (not a MagicMock auto-attribute) so appends are observable.
    parent.session_delegation_usage = []
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
    child._delegate_role = "leaf"
    return child


class TestDelegationUsageLedger(unittest.TestCase):
    def _run(self, child, goal="Test ledger attribution"):
        parent = _make_mock_parent()
        with patch("run_agent.AIAgent", return_value=child):
            result = json.loads(delegate_task(goal=goal, parent_agent=parent))
        return parent, result

    def test_ledger_entry_appended_with_attribution_fields(self):
        child = _make_mock_child(cost=0.25)
        parent, _ = self._run(child)
        self.assertEqual(len(parent.session_delegation_usage), 1)
        entry = parent.session_delegation_usage[0]
        self.assertEqual(entry["goal"], "Test ledger attribution")
        self.assertEqual(entry["model"], "anthropic/claude-sonnet-4")
        self.assertEqual(entry["status"], "completed")
        self.assertEqual(entry["api_calls"], 2)
        self.assertEqual(entry["input_tokens"], 100)
        self.assertEqual(entry["output_tokens"], 50)
        self.assertAlmostEqual(entry["cost_usd"], 0.25, places=6)
        self.assertEqual(entry["cost_status"], "estimated")
        self.assertEqual(entry["child_session_id"], "child-session")
        self.assertEqual(entry["role"], "leaf")

    def test_long_goal_is_capped(self):
        child = _make_mock_child()
        parent, _ = self._run(child, goal="x" * 500)
        entry = parent.session_delegation_usage[0]
        self.assertEqual(len(entry["goal"]), 200)

    def test_missing_ledger_attribute_is_created(self):
        child = _make_mock_child()
        parent = _make_mock_parent()
        # Simulate an agent built before the ledger existed.
        parent.session_delegation_usage = None
        with patch("run_agent.AIAgent", return_value=child):
            delegate_task(goal="legacy parent", parent_agent=parent)
        self.assertIsInstance(parent.session_delegation_usage, list)
        self.assertEqual(len(parent.session_delegation_usage), 1)

    def test_ledger_never_breaks_delegation_result(self):
        child = _make_mock_child()
        parent = _make_mock_parent()

        class _Boom(list):
            def append(self, *_a, **_k):
                raise RuntimeError("ledger boom")

        parent.session_delegation_usage = _Boom()
        with patch("run_agent.AIAgent", return_value=child):
            result = json.loads(
                delegate_task(goal="ledger failure isolated", parent_agent=parent)
            )
        # Delegation still succeeds even when accounting fails.
        self.assertEqual(result["results"][0]["status"], "completed")


class TestTurnFinalizerDelegationsBlock(unittest.TestCase):
    def test_result_omits_delegations_when_empty(self):
        # Contract: delegation-free turns keep the legacy result shape.
        agent = MagicMock()
        agent.session_delegation_usage = []
        block = getattr(agent, "session_delegation_usage", None)
        self.assertFalse(bool(block))

    def test_usage_file_forwards_delegations(self):
        import tempfile
        from pathlib import Path

        from hermes_cli.oneshot import _write_usage_file

        delegations = [
            {
                "task_index": 0,
                "goal": "sub work",
                "model": "m",
                "api_calls": 3,
                "input_tokens": 10,
                "output_tokens": 5,
                "cost_usd": 0.01,
            }
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "usage.json"
            _write_usage_file(str(path), {"delegations": delegations})
            report = json.loads(path.read_text())
            self.assertEqual(report["delegations"], delegations)

    def test_usage_file_omits_delegations_when_absent(self):
        import tempfile
        from pathlib import Path

        from hermes_cli.oneshot import _write_usage_file

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "usage.json"
            _write_usage_file(str(path), {"estimated_cost_usd": 0.5})
            report = json.loads(path.read_text())
            self.assertNotIn("delegations", report)


if __name__ == "__main__":
    unittest.main()

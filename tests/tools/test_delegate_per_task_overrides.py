#!/usr/bin/env python3
"""
Tests for per-task model/provider/reasoning_effort overrides in
delegate_task batch mode.

Behavior contract:
  - A batch task carrying 'model' / 'provider' / 'reasoning_effort' builds
    its child with those values, without affecting sibling tasks.
  - Tasks without overrides keep exactly today's inheritance defaults.
  - Empty or invalid override strings are ignored gracefully (no error,
    defaults preserved).

Run with:  scripts/run_tests.sh tests/tools/test_delegate_per_task_overrides.py
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task


def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key-parent"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.reasoning_config = None
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


def _batch_results(n):
    return [
        {
            "task_index": i,
            "status": "completed",
            "summary": f"Result {i}",
            "api_calls": 1,
            "duration_seconds": 1.0,
        }
        for i in range(n)
    ]


class TestPerTaskOverrideSchema(unittest.TestCase):
    def test_task_items_expose_override_keys(self):
        item_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"][
            "items"
        ]["properties"]
        for key in ("model", "provider", "reasoning_effort"):
            self.assertIn(key, item_props)
            self.assertEqual(item_props[key]["type"], "string")
        # The keys are optional — 'goal' remains the only required field.
        self.assertEqual(
            DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"][
                "required"
            ],
            ["goal"],
        )


class TestPerTaskOverrides(unittest.TestCase):
    @patch("tools.delegate_tool._run_single_child")
    def test_per_task_model_and_effort_override(self, mock_run):
        """Task 0 carries overrides; task 1 keeps inherited defaults."""
        mock_run.side_effect = _batch_results(2)
        parent = _make_mock_parent()
        tasks = [
            {
                "goal": "Task with overrides",
                "model": "openai/gpt-5-mini",
                "reasoning_effort": "low",
            },
            {"goal": "Task without overrides"},
        ]

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))

        self.assertIn("results", result)
        self.assertEqual(len(MockAgent.call_args_list), 2)
        _, kwargs0 = MockAgent.call_args_list[0]
        _, kwargs1 = MockAgent.call_args_list[1]
        self.assertEqual(kwargs0["model"], "openai/gpt-5-mini")
        self.assertEqual(
            kwargs0["reasoning_config"], {"enabled": True, "effort": "low"}
        )
        # Sibling task is untouched — parent inheritance as before.
        self.assertEqual(kwargs1["model"], parent.model)
        self.assertIsNone(kwargs1["reasoning_config"])
        self.assertEqual(kwargs1["provider"], parent.provider)

    @patch("tools.delegate_tool._run_single_child")
    def test_per_task_provider_override(self, mock_run):
        mock_run.side_effect = _batch_results(1)
        parent = _make_mock_parent()
        tasks = [{"goal": "Run on zai", "provider": "zai", "model": "glm-4.7"}]

        runtime_bundle = {
            "provider": "zai",
            "base_url": "https://api.z.ai/api/coding/paas/v4",
            "api_key": "test-key-zai",
            "api_mode": "anthropic_messages",
            "model": "glm-4.7",
        }
        with patch("run_agent.AIAgent") as MockAgent, patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=runtime_bundle,
        ) as mock_resolve:
            MockAgent.return_value = MagicMock()
            result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))

        self.assertIn("results", result)
        mock_resolve.assert_called_once_with(requested="zai", target_model="glm-4.7")
        _, kwargs = MockAgent.call_args
        self.assertEqual(kwargs["provider"], "zai")
        self.assertEqual(kwargs["base_url"], runtime_bundle["base_url"])
        self.assertEqual(kwargs["api_key"], runtime_bundle["api_key"])
        self.assertEqual(kwargs["model"], "glm-4.7")

    @patch("tools.delegate_tool._run_single_child")
    def test_batch_without_overrides_unchanged(self, mock_run):
        """No override keys → byte-identical defaults to today's behavior."""
        mock_run.side_effect = _batch_results(2)
        parent = _make_mock_parent()
        tasks = [{"goal": "Plain task A"}, {"goal": "Plain task B"}]

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))

        self.assertIn("results", result)
        for _, kwargs in MockAgent.call_args_list:
            self.assertEqual(kwargs["model"], parent.model)
            self.assertEqual(kwargs["provider"], parent.provider)
            self.assertEqual(kwargs["base_url"], parent.base_url)
            self.assertEqual(kwargs["api_key"], parent.api_key)
            self.assertIsNone(kwargs["reasoning_config"])

    @patch("tools.delegate_tool._run_single_child")
    def test_invalid_and_empty_overrides_ignored(self, mock_run):
        """Empty strings and unknown values degrade to defaults, no error."""
        mock_run.side_effect = _batch_results(1)
        parent = _make_mock_parent()
        tasks = [
            {
                "goal": "Bad overrides",
                "model": "   ",
                "provider": "",
                "reasoning_effort": "bananas",
            }
        ]

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))

        self.assertNotIn("error", result)
        self.assertIn("results", result)
        _, kwargs = MockAgent.call_args
        self.assertEqual(kwargs["model"], parent.model)
        self.assertEqual(kwargs["provider"], parent.provider)
        self.assertIsNone(kwargs["reasoning_config"])

    @patch("tools.delegate_tool._run_single_child")
    def test_unresolvable_provider_falls_back_gracefully(self, mock_run):
        """A provider that fails resolution warns and keeps defaults."""
        mock_run.side_effect = _batch_results(1)
        parent = _make_mock_parent()
        tasks = [{"goal": "Bad provider", "provider": "no-such-provider"}]

        with patch("run_agent.AIAgent") as MockAgent, patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=RuntimeError("unknown provider"),
        ):
            MockAgent.return_value = MagicMock()
            result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))

        self.assertNotIn("error", result)
        self.assertIn("results", result)
        _, kwargs = MockAgent.call_args
        self.assertEqual(kwargs["model"], parent.model)
        self.assertEqual(kwargs["provider"], parent.provider)
        self.assertEqual(kwargs["api_key"], parent.api_key)


if __name__ == "__main__":
    unittest.main()
